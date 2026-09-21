"""Pure scientific campaign and RQ-view contract for RQ1 through RQ4.

This module describes which unique campaigns must eventually be executed and
how each research question views those campaigns.  It contains no scheduler,
runner, evaluator, mutation generation, result storage, or budget accounting.
The valid-execution boundary remains defined by the materialization and
resolved-retrieval contracts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite, sqrt
from types import MappingProxyType

from ufuzz.retrieval_feedback import MutationRelation


class Benchmark(StrEnum):
    LOCOMO = "locomo"
    LONGMEMEVAL_S = "longmemeval-s"


class Backend(StrEnum):
    MEM0 = "mem0"
    AMEM = "a-mem"
    GRAPHITI = "graphiti"
    MEMOS = "memos"


class MutationSpace(StrEnum):
    FULL = "full"
    QUERY_ONLY = "query_only"
    MEMORY_ONLY = "memory_only"


class SelectionFamily(StrEnum):
    RANDOM_MUTATION = "random_mutation"
    UNGUIDED_LLM = "unguided_llm"
    LLM_AS_JUDGE = "llm_as_judge"
    COVERAGE_GUIDED = "coverage_guided"
    U_FUZZ = "u_fuzz"


class FairnessGroup(StrEnum):
    FULL_SPACE_PRIMARY = "full_space_primary"
    RESTRICTED_SPACE = "restricted_space"
    FULL_SPACE_ABLATION = "full_space_ablation"


class OnlineSearchSignal(StrEnum):
    NONE = "none"
    LLM_EXPLORATION_NO_RETRIEVAL_FEEDBACK = (
        "llm_exploration_no_retrieval_feedback"
    )
    ONLINE_LLM_JUDGE = "online_llm_judge"
    COVERAGE_GAIN_ONLY = "coverage_gain_only"
    U_FUZZ_RETRIEVAL_FEEDBACK = "u_fuzz_retrieval_feedback"


class FeedbackCombinationRule(StrEnum):
    """Disabled U-Fuzz components are zeroed while full weights remain."""

    ZERO_DISABLED_WITHOUT_RENORMALIZATION = (
        "zero_disabled_without_renormalization"
    )


class ResearchQuestion(StrEnum):
    RQ1 = "rq1"
    RQ2 = "rq2"
    RQ3 = "rq3"
    RQ4 = "rq4"


class MetricName(StrEnum):
    UF_AT_B = "UF@B"
    COV_AT_B = "Cov@B"
    DELTA_UF_AT_B = "DeltaUF@B"
    DELTA_COV_AT_B = "DeltaCov@B"


class ReportingPlacement(StrEnum):
    MAIN_PAPER = "main_paper"
    APPENDIX = "appendix"


RQ4_METRICS = (MetricName.UF_AT_B, MetricName.COV_AT_B)
RQ4_INTERPRETATION_EVIDENCE = frozenset(
    {"e0_cardinality", "retrievable_entry_granularity_profile_identity"}
)


class NativeMemoryLLMProvider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


@dataclass(frozen=True, slots=True)
class NativeMemoryLLMCondition:
    """Planning identity only; exact production model manifests remain unbound."""

    provider: NativeMemoryLLMProvider

    def __post_init__(self) -> None:
        if not isinstance(self.provider, NativeMemoryLLMProvider):
            raise TypeError("provider must be a NativeMemoryLLMProvider")


NATIVE_MEMORY_LLM_CONDITIONS = tuple(
    NativeMemoryLLMCondition(provider) for provider in NativeMemoryLLMProvider
)


QUERY_MUTATION_RELATIONS = frozenset(
    {
        MutationRelation.MEANING_PRESERVING_QUERY,
        MutationRelation.TARGET_CHANGING_QUERY,
        MutationRelation.UNSUPPORTED_QUERY,
    }
)
MEMORY_MUTATION_RELATIONS = frozenset(
    {
        MutationRelation.UPDATE,
        MutationRelation.DELETION,
        MutationRelation.UNRELATED_CHANGE,
    }
)
ALL_MUTATION_RELATIONS = QUERY_MUTATION_RELATIONS | MEMORY_MUTATION_RELATIONS


def mutation_relations_for(space: MutationSpace) -> frozenset[MutationRelation]:
    if not isinstance(space, MutationSpace):
        raise TypeError("space must be a MutationSpace")
    if space is MutationSpace.QUERY_ONLY:
        return QUERY_MUTATION_RELATIONS
    if space is MutationSpace.MEMORY_ONLY:
        return MEMORY_MUTATION_RELATIONS
    return ALL_MUTATION_RELATIONS


@dataclass(frozen=True, slots=True)
class MutationRelationConfiguration:
    """Exact causal mutation-relation mask for a frozen method."""

    broad_space: MutationSpace
    enabled_relations: frozenset[MutationRelation]

    def __post_init__(self) -> None:
        if not isinstance(self.broad_space, MutationSpace):
            raise TypeError("broad_space must be a MutationSpace")
        relations = frozenset(self.enabled_relations)
        if any(not isinstance(item, MutationRelation) for item in relations):
            raise TypeError("enabled_relations must contain MutationRelation values")
        object.__setattr__(self, "enabled_relations", relations)
        canonical = mutation_relations_for(self.broad_space)
        if self.broad_space is MutationSpace.FULL:
            if relations != canonical and not (
                len(relations) == 5 and relations < ALL_MUTATION_RELATIONS
            ):
                raise ValueError(
                    "FULL permits all six relations or one leave-one-out mask"
                )
        elif relations != canonical:
            raise ValueError("restricted spaces require their exact canonical mask")


FULL_RELATION_CONFIGURATION = MutationRelationConfiguration(
    MutationSpace.FULL, ALL_MUTATION_RELATIONS
)
QUERY_ONLY_RELATION_CONFIGURATION = MutationRelationConfiguration(
    MutationSpace.QUERY_ONLY, QUERY_MUTATION_RELATIONS
)
MEMORY_ONLY_RELATION_CONFIGURATION = MutationRelationConfiguration(
    MutationSpace.MEMORY_ONLY, MEMORY_MUTATION_RELATIONS
)


def without_relation(relation: MutationRelation) -> MutationRelationConfiguration:
    if not isinstance(relation, MutationRelation):
        raise TypeError("relation must be a MutationRelation")
    return MutationRelationConfiguration(
        MutationSpace.FULL, ALL_MUTATION_RELATIONS - {relation}
    )


@dataclass(frozen=True, slots=True)
class FeedbackComponentMask:
    coverage: bool
    novelty: bool
    parent_divergence: bool

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, bool)
            for value in (self.coverage, self.novelty, self.parent_divergence)
        ):
            raise TypeError("feedback component flags must be bools")


FULL_UFUZZ_FEEDBACK = FeedbackComponentMask(True, True, True)
WITHOUT_COVERAGE = FeedbackComponentMask(False, True, True)
WITHOUT_NOVELTY = FeedbackComponentMask(True, False, True)
WITHOUT_PARENT_DIVERGENCE = FeedbackComponentMask(True, True, False)


@dataclass(frozen=True, slots=True)
class MethodScientificKey:
    """Causal method semantics used by scientific campaign identity.

    Presentation labels, fairness annotations, and RQ membership deliberately
    do not enter this key. The frozen MethodSpec contract below admits only
    zeroing without renormalization.
    """

    method_id: str
    selection_family: SelectionFamily
    relation_configuration: MutationRelationConfiguration
    online_search_signal: OnlineSearchSignal
    feedback_components: FeedbackComponentMask | None
    feedback_combination_rule: FeedbackCombinationRule | None

    def __post_init__(self) -> None:
        if not isinstance(self.method_id, str) or not self.method_id:
            raise ValueError("method_id must be a non-empty string")
        for name, value, expected in (
            ("selection_family", self.selection_family, SelectionFamily),
            (
                "relation_configuration",
                self.relation_configuration,
                MutationRelationConfiguration,
            ),
            ("online_search_signal", self.online_search_signal, OnlineSearchSignal),
        ):
            if not isinstance(value, expected):
                raise TypeError(f"{name} must be a {expected.__name__}")
        if self.feedback_components is not None and not isinstance(
            self.feedback_components, FeedbackComponentMask
        ):
            raise TypeError("feedback_components must be a FeedbackComponentMask")
        if self.feedback_combination_rule is not None and not isinstance(
            self.feedback_combination_rule, FeedbackCombinationRule
        ):
            raise TypeError(
                "feedback_combination_rule must be a FeedbackCombinationRule"
            )


@dataclass(frozen=True, slots=True)
class MethodSpec:
    method_id: str
    display_name: str
    selection_family: SelectionFamily
    relation_configuration: MutationRelationConfiguration
    fairness_group: FairnessGroup
    online_search_signal: OnlineSearchSignal
    feedback_components: FeedbackComponentMask | None = None
    feedback_combination_rule: FeedbackCombinationRule | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.method_id, str) or not self.method_id:
            raise ValueError("method_id must be a non-empty string")
        if not isinstance(self.display_name, str) or not self.display_name:
            raise ValueError("display_name must be a non-empty string")
        for name, value, expected in (
            ("selection_family", self.selection_family, SelectionFamily),
            (
                "relation_configuration",
                self.relation_configuration,
                MutationRelationConfiguration,
            ),
            ("fairness_group", self.fairness_group, FairnessGroup),
            ("online_search_signal", self.online_search_signal, OnlineSearchSignal),
        ):
            if not isinstance(value, expected):
                raise TypeError(f"{name} must be a {expected.__name__}")
        if self.selection_family is SelectionFamily.U_FUZZ:
            if not isinstance(self.feedback_components, FeedbackComponentMask):
                raise ValueError("U-Fuzz methods require a feedback component mask")
            if self.feedback_combination_rule is not (
                FeedbackCombinationRule.ZERO_DISABLED_WITHOUT_RENORMALIZATION
            ):
                raise ValueError("U-Fuzz methods must retain the full score weights")
            if self.online_search_signal is not OnlineSearchSignal.U_FUZZ_RETRIEVAL_FEEDBACK:
                raise ValueError("U-Fuzz methods require frozen retrieval feedback")
        elif (
            self.feedback_components is not None
            or self.feedback_combination_rule is not None
        ):
            raise ValueError("non-U-Fuzz methods cannot carry a U-Fuzz component mask")
        if self.fairness_group is FairnessGroup.RESTRICTED_SPACE:
            if self.mutation_space is MutationSpace.FULL:
                raise ValueError("restricted methods cannot use the full mutation space")
        elif self.mutation_space is not MutationSpace.FULL:
            raise ValueError("full-space methods must use the full mutation space")
        if len(self.enabled_relations) == 5:
            if self.selection_family is not SelectionFamily.U_FUZZ or (
                self.fairness_group is not FairnessGroup.FULL_SPACE_ABLATION
            ):
                raise ValueError("leave-one-out masks are U-Fuzz ablations")
            if self.feedback_components != FULL_UFUZZ_FEEDBACK:
                raise ValueError("operator ablations retain full U-Fuzz feedback")

    @property
    def mutation_space(self) -> MutationSpace:
        return self.relation_configuration.broad_space

    @property
    def enabled_relations(self) -> frozenset[MutationRelation]:
        return self.relation_configuration.enabled_relations

    @property
    def scientific_key(self) -> MethodScientificKey:
        return MethodScientificKey(
            self.method_id,
            self.selection_family,
            self.relation_configuration,
            self.online_search_signal,
            self.feedback_components,
            self.feedback_combination_rule,
        )


RANDOM_MUTATION = MethodSpec(
    "random-mutation",
    "Random Mutation",
    SelectionFamily.RANDOM_MUTATION,
    FULL_RELATION_CONFIGURATION,
    FairnessGroup.FULL_SPACE_PRIMARY,
    OnlineSearchSignal.NONE,
)
UNGUIDED_LLM = MethodSpec(
    "unguided-llm",
    "Unguided LLM",
    SelectionFamily.UNGUIDED_LLM,
    FULL_RELATION_CONFIGURATION,
    FairnessGroup.FULL_SPACE_PRIMARY,
    OnlineSearchSignal.LLM_EXPLORATION_NO_RETRIEVAL_FEEDBACK,
)
LLM_AS_JUDGE = MethodSpec(
    "llm-as-judge",
    "LLM-as-Judge",
    SelectionFamily.LLM_AS_JUDGE,
    FULL_RELATION_CONFIGURATION,
    FairnessGroup.FULL_SPACE_PRIMARY,
    OnlineSearchSignal.ONLINE_LLM_JUDGE,
)
COVERAGE_GUIDED = MethodSpec(
    "coverage-guided",
    "Coverage-Guided",
    SelectionFamily.COVERAGE_GUIDED,
    FULL_RELATION_CONFIGURATION,
    FairnessGroup.FULL_SPACE_PRIMARY,
    OnlineSearchSignal.COVERAGE_GAIN_ONLY,
)
UFUZZ_Q = MethodSpec(
    "ufuzz-q",
    "U-Fuzz-Q",
    SelectionFamily.U_FUZZ,
    QUERY_ONLY_RELATION_CONFIGURATION,
    FairnessGroup.RESTRICTED_SPACE,
    OnlineSearchSignal.U_FUZZ_RETRIEVAL_FEEDBACK,
    FULL_UFUZZ_FEEDBACK,
    FeedbackCombinationRule.ZERO_DISABLED_WITHOUT_RENORMALIZATION,
)
UFUZZ_M = MethodSpec(
    "ufuzz-m",
    "U-Fuzz-M",
    SelectionFamily.U_FUZZ,
    MEMORY_ONLY_RELATION_CONFIGURATION,
    FairnessGroup.RESTRICTED_SPACE,
    OnlineSearchSignal.U_FUZZ_RETRIEVAL_FEEDBACK,
    FULL_UFUZZ_FEEDBACK,
    FeedbackCombinationRule.ZERO_DISABLED_WITHOUT_RENORMALIZATION,
)
UFUZZ = MethodSpec(
    "ufuzz",
    "U-Fuzz",
    SelectionFamily.U_FUZZ,
    FULL_RELATION_CONFIGURATION,
    FairnessGroup.FULL_SPACE_PRIMARY,
    OnlineSearchSignal.U_FUZZ_RETRIEVAL_FEEDBACK,
    FULL_UFUZZ_FEEDBACK,
    FeedbackCombinationRule.ZERO_DISABLED_WITHOUT_RENORMALIZATION,
)


def _operator_ablation(
    method_id: str, display_name: str, removed: MutationRelation
) -> MethodSpec:
    return MethodSpec(
        method_id,
        display_name,
        SelectionFamily.U_FUZZ,
        without_relation(removed),
        FairnessGroup.FULL_SPACE_ABLATION,
        OnlineSearchSignal.U_FUZZ_RETRIEVAL_FEEDBACK,
        FULL_UFUZZ_FEEDBACK,
        FeedbackCombinationRule.ZERO_DISABLED_WITHOUT_RENORMALIZATION,
    )


UFUZZ_WITHOUT_MEANING_PRESERVING_QUERY = _operator_ablation(
    "ufuzz-without-meaning-preserving-query",
    "U-Fuzz w/o Meaning-Preserving Query",
    MutationRelation.MEANING_PRESERVING_QUERY,
)
UFUZZ_WITHOUT_TARGET_CHANGING_QUERY = _operator_ablation(
    "ufuzz-without-target-changing-query",
    "U-Fuzz w/o Target-Changing Query",
    MutationRelation.TARGET_CHANGING_QUERY,
)
UFUZZ_WITHOUT_UNSUPPORTED_QUERY = _operator_ablation(
    "ufuzz-without-unsupported-query",
    "U-Fuzz w/o Unsupported Query",
    MutationRelation.UNSUPPORTED_QUERY,
)
UFUZZ_WITHOUT_UPDATE = _operator_ablation(
    "ufuzz-without-update", "U-Fuzz w/o Update", MutationRelation.UPDATE
)
UFUZZ_WITHOUT_DELETION = _operator_ablation(
    "ufuzz-without-deletion", "U-Fuzz w/o Deletion", MutationRelation.DELETION
)
UFUZZ_WITHOUT_UNRELATED_CHANGE = _operator_ablation(
    "ufuzz-without-unrelated-change",
    "U-Fuzz w/o Unrelated Change",
    MutationRelation.UNRELATED_CHANGE,
)
UFUZZ_WITHOUT_COVERAGE = MethodSpec(
    "ufuzz-without-coverage",
    "U-Fuzz w/o Coverage",
    SelectionFamily.U_FUZZ,
    FULL_RELATION_CONFIGURATION,
    FairnessGroup.FULL_SPACE_ABLATION,
    OnlineSearchSignal.U_FUZZ_RETRIEVAL_FEEDBACK,
    WITHOUT_COVERAGE,
    FeedbackCombinationRule.ZERO_DISABLED_WITHOUT_RENORMALIZATION,
)
UFUZZ_WITHOUT_NOVELTY = MethodSpec(
    "ufuzz-without-novelty",
    "U-Fuzz w/o Novelty",
    SelectionFamily.U_FUZZ,
    FULL_RELATION_CONFIGURATION,
    FairnessGroup.FULL_SPACE_ABLATION,
    OnlineSearchSignal.U_FUZZ_RETRIEVAL_FEEDBACK,
    WITHOUT_NOVELTY,
    FeedbackCombinationRule.ZERO_DISABLED_WITHOUT_RENORMALIZATION,
)
UFUZZ_WITHOUT_PARENT_DIVERGENCE = MethodSpec(
    "ufuzz-without-parent-divergence",
    "U-Fuzz w/o Parent Divergence",
    SelectionFamily.U_FUZZ,
    FULL_RELATION_CONFIGURATION,
    FairnessGroup.FULL_SPACE_ABLATION,
    OnlineSearchSignal.U_FUZZ_RETRIEVAL_FEEDBACK,
    WITHOUT_PARENT_DIVERGENCE,
    FeedbackCombinationRule.ZERO_DISABLED_WITHOUT_RENORMALIZATION,
)

RQ1_METHODS = (
    RANDOM_MUTATION,
    UNGUIDED_LLM,
    LLM_AS_JUDGE,
    COVERAGE_GUIDED,
    UFUZZ_Q,
    UFUZZ_M,
    UFUZZ,
)
RQ2_METHODS = (UFUZZ_Q, UFUZZ_M, UFUZZ)
RQ3_REFERENCE_METHODS = (UFUZZ,)
# Compatibility name for code that distinguishes reused view rows from new raw
# campaigns. RQ3 now reuses only the Full U-Fuzz reference.
RQ3_REUSED_METHODS = RQ3_REFERENCE_METHODS
RQ3_OPERATOR_ABLATIONS = (
    UFUZZ_WITHOUT_MEANING_PRESERVING_QUERY,
    UFUZZ_WITHOUT_TARGET_CHANGING_QUERY,
    UFUZZ_WITHOUT_UNSUPPORTED_QUERY,
    UFUZZ_WITHOUT_UPDATE,
    UFUZZ_WITHOUT_DELETION,
    UFUZZ_WITHOUT_UNRELATED_CHANGE,
)
RQ3_FEEDBACK_ABLATIONS = (
    UFUZZ_WITHOUT_COVERAGE,
    UFUZZ_WITHOUT_NOVELTY,
    UFUZZ_WITHOUT_PARENT_DIVERGENCE,
)
NEW_RQ3_METHODS = RQ3_OPERATOR_ABLATIONS + RQ3_FEEDBACK_ABLATIONS
RQ3_METHODS = RQ3_REFERENCE_METHODS + NEW_RQ3_METHODS
RQ4_METHODS = (RANDOM_MUTATION, COVERAGE_GUIDED, UFUZZ)
ALL_METHODS = RQ1_METHODS + NEW_RQ3_METHODS
METHODS_BY_ID = MappingProxyType({method.method_id: method for method in ALL_METHODS})
RQ1_METHOD_SCIENTIFIC_KEYS = frozenset(
    method.scientific_key for method in RQ1_METHODS
)
NEW_RQ3_METHOD_SCIENTIFIC_KEYS = frozenset(
    method.scientific_key for method in NEW_RQ3_METHODS
)
RQ4_METHOD_SCIENTIFIC_KEYS = frozenset(
    method.scientific_key for method in RQ4_METHODS
)

BACKENDS = (Backend.MEM0, Backend.AMEM, Backend.GRAPHITI, Backend.MEMOS)
RQ4_BACKENDS = (Backend.MEM0, Backend.GRAPHITI)
REPETITION_INDICES = (0, 1, 2)
RQ2_CHECKPOINTS = tuple(range(1000, 8001, 1000))
BENCHMARK_MAX_BUDGET = MappingProxyType(
    {Benchmark.LOCOMO: 8000, Benchmark.LONGMEMEVAL_S: 4000}
)
RQ3_BUDGET = 4000
RQ4_BUDGET = 2000
RQ3_REPORTING_PLACEMENT = MappingProxyType(
    {
        Backend.MEM0: ReportingPlacement.MAIN_PAPER,
        Backend.AMEM: ReportingPlacement.APPENDIX,
        Backend.GRAPHITI: ReportingPlacement.APPENDIX,
        Backend.MEMOS: ReportingPlacement.APPENDIX,
    }
)


@dataclass(frozen=True, slots=True, order=True)
class ConfigurationBinding:
    dimension: str
    configuration_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.dimension, str) or not self.dimension:
            raise ValueError("configuration dimension must be non-empty")
        if not isinstance(self.configuration_id, str) or not self.configuration_id:
            raise ValueError("configuration ID must be non-empty")


@dataclass(frozen=True, slots=True)
class CampaignKey:
    benchmark: Benchmark
    backend: Backend
    method: MethodScientificKey
    repetition_index: int
    max_budget: int
    native_memory_llm_condition: NativeMemoryLLMCondition | None = None
    configuration_bindings: tuple[ConfigurationBinding, ...] = ()


@dataclass(frozen=True, slots=True)
class CampaignSpec:
    """One unique scientific campaign, independent of any RQ view."""

    benchmark: Benchmark
    backend: Backend
    method: MethodSpec
    repetition_index: int
    max_budget: int
    native_memory_llm_condition: NativeMemoryLLMCondition | None = None
    configuration_bindings: tuple[ConfigurationBinding, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.benchmark, Benchmark):
            raise TypeError("benchmark must be a Benchmark")
        if not isinstance(self.backend, Backend):
            raise TypeError("backend must be a Backend")
        if not isinstance(self.method, MethodSpec):
            raise TypeError("method must be a MethodSpec")
        if self.repetition_index not in REPETITION_INDICES:
            raise ValueError("repetition index must be one of 0, 1, 2")
        if isinstance(self.max_budget, bool) or not isinstance(self.max_budget, int):
            raise TypeError("max budget must be an integer")
        condition = self.native_memory_llm_condition
        if condition is None:
            if self.benchmark is Benchmark.LONGMEMEVAL_S:
                allowed = (
                    self.method.scientific_key in RQ1_METHOD_SCIENTIFIC_KEYS
                    and self.max_budget == 4000
                )
            elif self.method.scientific_key in RQ1_METHOD_SCIENTIFIC_KEYS:
                allowed = self.max_budget == 8000
            else:
                allowed = (
                    self.method.scientific_key in NEW_RQ3_METHOD_SCIENTIFIC_KEYS
                    and self.max_budget == RQ3_BUDGET
                )
        else:
            if not isinstance(condition, NativeMemoryLLMCondition):
                raise TypeError(
                    "native_memory_llm_condition must be a NativeMemoryLLMCondition"
                )
            allowed = (
                self.benchmark is Benchmark.LOCOMO
                and self.backend in RQ4_BACKENDS
                and self.method.scientific_key in RQ4_METHOD_SCIENTIFIC_KEYS
                and self.max_budget == RQ4_BUDGET
            )
        if not allowed:
            raise ValueError("campaign fields do not match a frozen experiment role")
        bindings = tuple(sorted(tuple(self.configuration_bindings)))
        if any(not isinstance(item, ConfigurationBinding) for item in bindings):
            raise TypeError("configuration bindings must be ConfigurationBinding values")
        dimensions = [item.dimension for item in bindings]
        if len(set(dimensions)) != len(dimensions):
            raise ValueError("configuration dimensions must be unique")
        object.__setattr__(self, "configuration_bindings", bindings)

    @property
    def scientific_key(self) -> CampaignKey:
        return CampaignKey(
            self.benchmark,
            self.backend,
            self.method.scientific_key,
            self.repetition_index,
            self.max_budget,
            self.native_memory_llm_condition,
            self.configuration_bindings,
        )

    @property
    def has_configuration_bindings(self) -> bool:
        """Describe binding presence; this alone is not production readiness."""

        return bool(self.configuration_bindings)


@dataclass(frozen=True, slots=True)
class RQViewEntry:
    """One observational view onto a canonical raw campaign."""

    research_question: ResearchQuestion
    display_method_id: str
    checkpoint: int
    campaign: CampaignSpec

    def __post_init__(self) -> None:
        if not isinstance(self.research_question, ResearchQuestion):
            raise TypeError("research_question must be a ResearchQuestion")
        if not isinstance(self.display_method_id, str) or not self.display_method_id:
            raise ValueError("display method ID must be non-empty")
        if isinstance(self.checkpoint, bool) or not isinstance(self.checkpoint, int):
            raise TypeError("checkpoint must be an integer")
        if not 0 < self.checkpoint <= self.campaign.max_budget:
            raise ValueError("checkpoint lies outside the campaign trajectory")


@dataclass(frozen=True, slots=True)
class EvaluationPlan:
    campaigns: tuple[CampaignSpec, ...]
    rq1: tuple[RQViewEntry, ...]
    rq2: tuple[RQViewEntry, ...]
    rq3: tuple[RQViewEntry, ...]
    rq4: tuple[RQViewEntry, ...]

    def __post_init__(self) -> None:
        campaigns = tuple(self.campaigns)
        keys = [campaign.scientific_key for campaign in campaigns]
        if len(set(keys)) != len(keys):
            raise ValueError("evaluation plan contains duplicate scientific campaigns")
        canonical = {campaign.scientific_key: campaign for campaign in campaigns}
        for entry in self.rq1 + self.rq2 + self.rq3 + self.rq4:
            if canonical.get(entry.campaign.scientific_key) is not entry.campaign:
                raise ValueError("RQ views must reference canonical campaign objects")
        object.__setattr__(self, "campaigns", campaigns)
        object.__setattr__(self, "rq1", tuple(self.rq1))
        object.__setattr__(self, "rq2", tuple(self.rq2))
        object.__setattr__(self, "rq3", tuple(self.rq3))
        object.__setattr__(self, "rq4", tuple(self.rq4))

    @property
    def total_planned_valid_executions(self) -> int:
        return sum(campaign.max_budget for campaign in self.campaigns)

    @property
    def rq1_rq3_campaigns(self) -> tuple[CampaignSpec, ...]:
        return tuple(
            campaign for campaign in self.campaigns
            if campaign.native_memory_llm_condition is None
        )

    @property
    def rq4_campaigns(self) -> tuple[CampaignSpec, ...]:
        return tuple(
            campaign for campaign in self.campaigns
            if campaign.native_memory_llm_condition is not None
        )


def build_evaluation_plan() -> EvaluationPlan:
    """Build the deduplicated 330-campaign plan and its four RQ views."""

    campaigns: dict[CampaignKey, CampaignSpec] = {}

    def add(
        benchmark: Benchmark,
        backend: Backend,
        method: MethodSpec,
        repetition: int,
        budget: int,
        condition: NativeMemoryLLMCondition | None = None,
    ) -> CampaignSpec:
        spec = CampaignSpec(
            benchmark, backend, method, repetition, budget, condition
        )
        return campaigns.setdefault(spec.scientific_key, spec)

    for benchmark in Benchmark:
        for backend in BACKENDS:
            for method in RQ1_METHODS:
                for repetition in REPETITION_INDICES:
                    add(
                        benchmark,
                        backend,
                        method,
                        repetition,
                        BENCHMARK_MAX_BUDGET[benchmark],
                    )
    for backend in BACKENDS:
        for method in NEW_RQ3_METHODS:
            for repetition in REPETITION_INDICES:
                add(Benchmark.LOCOMO, backend, method, repetition, RQ3_BUDGET)

    for backend in RQ4_BACKENDS:
        for condition in NATIVE_MEMORY_LLM_CONDITIONS:
            for method in RQ4_METHODS:
                for repetition in REPETITION_INDICES:
                    add(
                        Benchmark.LOCOMO,
                        backend,
                        method,
                        repetition,
                        RQ4_BUDGET,
                        condition,
                    )

    def get(
        benchmark: Benchmark,
        backend: Backend,
        method: MethodSpec,
        repetition: int,
        budget: int,
        condition: NativeMemoryLLMCondition | None = None,
    ) -> CampaignSpec:
        key = CampaignSpec(
            benchmark, backend, method, repetition, budget, condition
        ).scientific_key
        return campaigns[key]

    rq1 = tuple(
        RQViewEntry(
            ResearchQuestion.RQ1,
            method.method_id,
            BENCHMARK_MAX_BUDGET[benchmark],
            get(
                benchmark,
                backend,
                method,
                repetition,
                BENCHMARK_MAX_BUDGET[benchmark],
            ),
        )
        for benchmark in Benchmark
        for backend in BACKENDS
        for method in RQ1_METHODS
        for repetition in REPETITION_INDICES
    )
    rq2 = tuple(
        RQViewEntry(
            ResearchQuestion.RQ2,
            method.method_id,
            checkpoint,
            get(Benchmark.LOCOMO, backend, method, repetition, 8000),
        )
        for backend in BACKENDS
        for method in RQ2_METHODS
        for repetition in REPETITION_INDICES
        for checkpoint in RQ2_CHECKPOINTS
    )
    rq3 = tuple(
        RQViewEntry(
            ResearchQuestion.RQ3,
            method.method_id,
            RQ3_BUDGET,
            get(
                Benchmark.LOCOMO,
                backend,
                method,
                repetition,
                8000 if method in RQ3_REFERENCE_METHODS else RQ3_BUDGET,
            ),
        )
        for backend in BACKENDS
        for method in RQ3_METHODS
        for repetition in REPETITION_INDICES
    )
    rq4 = tuple(
        RQViewEntry(
            ResearchQuestion.RQ4,
            method.method_id,
            RQ4_BUDGET,
            get(
                Benchmark.LOCOMO,
                backend,
                method,
                repetition,
                RQ4_BUDGET,
                condition,
            ),
        )
        for backend in RQ4_BACKENDS
        for condition in NATIVE_MEMORY_LLM_CONDITIONS
        for method in RQ4_METHODS
        for repetition in REPETITION_INDICES
    )
    ordered_campaigns = tuple(
        sorted(
            campaigns.values(),
            key=lambda item: (
                item.benchmark.value,
                item.backend.value,
                item.method.method_id,
                item.native_memory_llm_condition.provider.value
                if item.native_memory_llm_condition
                else "",
                item.max_budget,
                item.repetition_index,
            ),
        )
    )
    return EvaluationPlan(ordered_campaigns, rq1, rq2, rq3, rq4)


@dataclass(frozen=True, slots=True)
class CheckpointMetrics:
    checkpoint: int
    distinct_faults: int
    coverage_fraction: float

    def __post_init__(self) -> None:
        if isinstance(self.checkpoint, bool) or not isinstance(self.checkpoint, int):
            raise TypeError("checkpoint must be an integer")
        if isinstance(self.distinct_faults, bool) or not isinstance(
            self.distinct_faults, int
        ):
            raise TypeError("distinct fault count must be an integer")
        if self.distinct_faults < 0:
            raise ValueError("distinct fault count cannot be negative")
        if isinstance(self.coverage_fraction, bool) or not isinstance(
            self.coverage_fraction, (int, float)
        ):
            raise TypeError("coverage fraction must be numeric")
        if not 0.0 <= float(self.coverage_fraction) <= 1.0:
            raise ValueError("coverage fraction must lie in [0, 1]")


def validate_rq2_prefix_series(series: Sequence[CheckpointMetrics]) -> None:
    """Require exact checkpoints and monotone cumulative UF and coverage."""

    values = tuple(series)
    if tuple(item.checkpoint for item in values) != RQ2_CHECKPOINTS:
        raise ValueError("RQ2 series must contain the eight frozen checkpoints")
    for previous, current in zip(values, values[1:]):
        if current.distinct_faults < previous.distinct_faults:
            raise ValueError("cumulative UF cannot decrease")
        if current.coverage_fraction < previous.coverage_fraction:
            raise ValueError("cumulative coverage cannot decrease")


@dataclass(frozen=True, slots=True)
class AggregatedValue:
    arithmetic_mean: float
    sample_standard_deviation: float


def aggregate_three_repetitions(values: Mapping[int, float]) -> AggregatedValue:
    """Return arithmetic mean and sample standard deviation with ddof=1."""

    if set(values) != set(REPETITION_INDICES):
        raise ValueError("aggregation requires repetition indices 0, 1, and 2")
    ordered = tuple(float(values[index]) for index in REPETITION_INDICES)
    mean = sum(ordered) / 3.0
    variance = sum((value - mean) ** 2 for value in ordered) / 2.0
    return AggregatedValue(mean, sqrt(variance))


@dataclass(frozen=True, slots=True)
class RQ3MetricObservation:
    """One raw UF@4K or Cov@4K value bound to an exact RQ3 view entry."""

    view_entry: RQViewEntry
    metric: MetricName
    value: float

    def __post_init__(self) -> None:
        if not isinstance(self.view_entry, RQViewEntry):
            raise TypeError("view_entry must be an RQViewEntry")
        if self.view_entry.research_question is not ResearchQuestion.RQ3:
            raise ValueError("RQ3 reporting requires an RQ3 view entry")
        if self.view_entry.display_method_id != self.view_entry.campaign.method.method_id:
            raise ValueError("display method must match the campaign method")
        if self.metric not in (MetricName.UF_AT_B, MetricName.COV_AT_B):
            raise ValueError("RQ3 observations must contain raw UF@B or Cov@B")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise TypeError("metric value must be numeric")
        value = float(self.value)
        if not isfinite(value):
            raise ValueError("metric value must be finite")
        if self.metric is MetricName.UF_AT_B and value < 0:
            raise ValueError("UF@B cannot be negative")
        if self.metric is MetricName.COV_AT_B and not 0.0 <= value <= 1.0:
            raise ValueError("Cov@B must lie in [0, 1]")
        object.__setattr__(self, "value", value)


@dataclass(frozen=True, slots=True)
class PairedAblationDelta:
    """One matched-repetition RQ3 ablation-minus-Full delta."""

    backend: Backend
    ablation: MethodSpec
    repetition_index: int
    checkpoint: int
    raw_metric: MetricName
    full_value: float
    ablation_value: float

    def __post_init__(self) -> None:
        if not isinstance(self.backend, Backend):
            raise TypeError("backend must be a Backend")
        if self.ablation not in NEW_RQ3_METHODS:
            raise ValueError("ablation must be one of the nine frozen RQ3 ablations")
        if self.repetition_index not in REPETITION_INDICES:
            raise ValueError("repetition index must be one of 0, 1, and 2")
        if self.checkpoint != RQ3_BUDGET:
            raise ValueError("RQ3 paired deltas require B=4000")
        if self.raw_metric not in (MetricName.UF_AT_B, MetricName.COV_AT_B):
            raise ValueError("paired deltas require raw UF@B or Cov@B")
        for name, value in (
            ("full_value", self.full_value),
            ("ablation_value", self.ablation_value),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, float(value))
        if self.raw_metric is MetricName.UF_AT_B and (
            self.full_value < 0 or self.ablation_value < 0
        ):
            raise ValueError("UF@B inputs cannot be negative")
        if self.raw_metric is MetricName.COV_AT_B and not (
            0.0 <= self.full_value <= 1.0
            and 0.0 <= self.ablation_value <= 1.0
        ):
            raise ValueError("Cov@B inputs must lie in [0, 1]")

    @property
    def delta(self) -> float:
        """Ablation minus Full; negative means removal reduced the metric."""

        return self.ablation_value - self.full_value

    @property
    def derived_metric(self) -> MetricName:
        return (
            MetricName.DELTA_UF_AT_B
            if self.raw_metric is MetricName.UF_AT_B
            else MetricName.DELTA_COV_AT_B
        )


@dataclass(frozen=True, slots=True)
class RQ3AblationSummary:
    """Mean and sample standard deviation of three matched-repetition deltas."""

    paired_deltas: tuple[PairedAblationDelta, ...]

    def __post_init__(self) -> None:
        values = tuple(self.paired_deltas)
        if any(not isinstance(item, PairedAblationDelta) for item in values):
            raise TypeError("paired_deltas must contain PairedAblationDelta values")
        if {item.repetition_index for item in values} != set(REPETITION_INDICES):
            raise ValueError("summary requires exactly repetitions 0, 1, and 2")
        if len(values) != len(REPETITION_INDICES):
            raise ValueError("summary cannot contain duplicate repetitions")
        dimensions = {
            (item.backend, item.ablation, item.checkpoint, item.raw_metric)
            for item in values
        }
        if len(dimensions) != 1:
            raise ValueError("paired deltas must share backend, ablation, B, and metric")
        object.__setattr__(
            self,
            "paired_deltas",
            tuple(sorted(values, key=lambda item: item.repetition_index)),
        )

    @property
    def backend(self) -> Backend:
        return self.paired_deltas[0].backend

    @property
    def ablation(self) -> MethodSpec:
        return self.paired_deltas[0].ablation

    @property
    def derived_metric(self) -> MetricName:
        return self.paired_deltas[0].derived_metric

    @property
    def arithmetic_mean(self) -> float:
        return aggregate_three_repetitions(
            {item.repetition_index: item.delta for item in self.paired_deltas}
        ).arithmetic_mean

    @property
    def sample_standard_deviation(self) -> float:
        return aggregate_three_repetitions(
            {item.repetition_index: item.delta for item in self.paired_deltas}
        ).sample_standard_deviation


def summarize_rq3_paired_delta(
    full_observations: Sequence[RQ3MetricObservation],
    ablation_observations: Sequence[RQ3MetricObservation],
) -> RQ3AblationSummary:
    """Pair exact RQ3 repetitions and aggregate ablation-minus-Full deltas."""

    def by_repetition(
        observations: Sequence[RQ3MetricObservation], label: str
    ) -> dict[int, RQ3MetricObservation]:
        indexed: dict[int, RQ3MetricObservation] = {}
        for observation in observations:
            if not isinstance(observation, RQ3MetricObservation):
                raise TypeError(f"{label} observations must be RQ3MetricObservation values")
            repetition = observation.view_entry.campaign.repetition_index
            if repetition in indexed:
                raise ValueError(f"duplicate {label} repetition {repetition}")
            indexed[repetition] = observation
        if set(indexed) != set(REPETITION_INDICES):
            raise ValueError(f"{label} requires repetitions 0, 1, and 2")
        return indexed

    full = by_repetition(full_observations, "Full")
    ablation = by_repetition(ablation_observations, "ablation")
    pairs: list[PairedAblationDelta] = []
    for repetition in REPETITION_INDICES:
        full_value = full[repetition]
        ablation_value = ablation[repetition]
        full_entry = full_value.view_entry
        ablation_entry = ablation_value.view_entry
        if full_entry.campaign.method is not UFUZZ:
            raise ValueError("RQ3 Full reference must be U-Fuzz")
        if ablation_entry.campaign.method not in NEW_RQ3_METHODS:
            raise ValueError("RQ3 comparison method must be an exact frozen ablation")
        if full_value.metric is not ablation_value.metric:
            raise ValueError("paired observations must use the same raw metric")
        if full_entry.campaign.benchmark is not Benchmark.LOCOMO or (
            ablation_entry.campaign.benchmark is not Benchmark.LOCOMO
        ):
            raise ValueError("RQ3 paired observations must use LoCoMo")
        if full_entry.campaign.backend is not ablation_entry.campaign.backend:
            raise ValueError("paired observations must use the same backend")
        if full_entry.checkpoint != ablation_entry.checkpoint:
            raise ValueError("paired observations must use the same B")
        if full_entry.checkpoint != RQ3_BUDGET:
            raise ValueError("RQ3 paired observations require B=4000")
        if full_entry.campaign.max_budget != 8000:
            raise ValueError("Full reference must reuse the RQ1 Bmax=8000 campaign")
        if ablation_entry.campaign.max_budget != RQ3_BUDGET:
            raise ValueError("ablation must use its raw Bmax=4000 campaign")
        if full_entry.campaign.native_memory_llm_condition is not None or (
            ablation_entry.campaign.native_memory_llm_condition is not None
        ):
            raise ValueError("RQ3 cannot use an RQ4 native-memory-LLM condition")
        pairs.append(
            PairedAblationDelta(
                full_entry.campaign.backend,
                ablation_entry.campaign.method,
                repetition,
                full_entry.checkpoint,
                full_value.metric,
                full_value.value,
                ablation_value.value,
            )
        )
    return RQ3AblationSummary(tuple(pairs))


@dataclass(frozen=True, slots=True)
class EngineeringTarget:
    gpu_count: int
    wall_clock_seconds: int
    planned_valid_executions: int

    @property
    def minimum_average_valid_executions_per_second(self) -> float:
        return self.planned_valid_executions / self.wall_clock_seconds


ENGINEERING_TARGET = EngineeringTarget(6, 24 * 60 * 60, 1_440_000)


EVALUATION_PLAN = build_evaluation_plan()
