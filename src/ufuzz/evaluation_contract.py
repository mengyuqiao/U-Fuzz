"""Pure scientific campaign and RQ-view contract for RQ1, RQ2, and RQ3.

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
from math import sqrt
from types import MappingProxyType

from ufuzz.retrieval_feedback import MutationRelation


class Benchmark(StrEnum):
    LOCOMO = "locomo"
    LONGMEMEVAL_S = "longmemeval-s"


class Backend(StrEnum):
    MEM0 = "mem0"
    AMEM = "a-mem"
    GRAPHITI = "graphiti"


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


class MetricName(StrEnum):
    UF_AT_B = "UF@B"
    COV_AT_B = "Cov@B"


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


def mutation_relations_for(space: MutationSpace) -> frozenset[MutationRelation]:
    if not isinstance(space, MutationSpace):
        raise TypeError("space must be a MutationSpace")
    if space is MutationSpace.QUERY_ONLY:
        return QUERY_MUTATION_RELATIONS
    if space is MutationSpace.MEMORY_ONLY:
        return MEMORY_MUTATION_RELATIONS
    return QUERY_MUTATION_RELATIONS | MEMORY_MUTATION_RELATIONS


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
    mutation_space: MutationSpace
    online_search_signal: OnlineSearchSignal
    feedback_components: FeedbackComponentMask | None
    feedback_combination_rule: FeedbackCombinationRule | None

    def __post_init__(self) -> None:
        if not isinstance(self.method_id, str) or not self.method_id:
            raise ValueError("method_id must be a non-empty string")
        for name, value, expected in (
            ("selection_family", self.selection_family, SelectionFamily),
            ("mutation_space", self.mutation_space, MutationSpace),
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
    mutation_space: MutationSpace
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
            ("mutation_space", self.mutation_space, MutationSpace),
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

    @property
    def scientific_key(self) -> MethodScientificKey:
        return MethodScientificKey(
            self.method_id,
            self.selection_family,
            self.mutation_space,
            self.online_search_signal,
            self.feedback_components,
            self.feedback_combination_rule,
        )


RANDOM_MUTATION = MethodSpec(
    "random-mutation",
    "Random Mutation",
    SelectionFamily.RANDOM_MUTATION,
    MutationSpace.FULL,
    FairnessGroup.FULL_SPACE_PRIMARY,
    OnlineSearchSignal.NONE,
)
UNGUIDED_LLM = MethodSpec(
    "unguided-llm",
    "Unguided LLM",
    SelectionFamily.UNGUIDED_LLM,
    MutationSpace.FULL,
    FairnessGroup.FULL_SPACE_PRIMARY,
    OnlineSearchSignal.LLM_EXPLORATION_NO_RETRIEVAL_FEEDBACK,
)
LLM_AS_JUDGE = MethodSpec(
    "llm-as-judge",
    "LLM-as-Judge",
    SelectionFamily.LLM_AS_JUDGE,
    MutationSpace.FULL,
    FairnessGroup.FULL_SPACE_PRIMARY,
    OnlineSearchSignal.ONLINE_LLM_JUDGE,
)
COVERAGE_GUIDED = MethodSpec(
    "coverage-guided",
    "Coverage-Guided",
    SelectionFamily.COVERAGE_GUIDED,
    MutationSpace.FULL,
    FairnessGroup.FULL_SPACE_PRIMARY,
    OnlineSearchSignal.COVERAGE_GAIN_ONLY,
)
UFUZZ_Q = MethodSpec(
    "ufuzz-q",
    "U-Fuzz-Q",
    SelectionFamily.U_FUZZ,
    MutationSpace.QUERY_ONLY,
    FairnessGroup.RESTRICTED_SPACE,
    OnlineSearchSignal.U_FUZZ_RETRIEVAL_FEEDBACK,
    FULL_UFUZZ_FEEDBACK,
    FeedbackCombinationRule.ZERO_DISABLED_WITHOUT_RENORMALIZATION,
)
UFUZZ_M = MethodSpec(
    "ufuzz-m",
    "U-Fuzz-M",
    SelectionFamily.U_FUZZ,
    MutationSpace.MEMORY_ONLY,
    FairnessGroup.RESTRICTED_SPACE,
    OnlineSearchSignal.U_FUZZ_RETRIEVAL_FEEDBACK,
    FULL_UFUZZ_FEEDBACK,
    FeedbackCombinationRule.ZERO_DISABLED_WITHOUT_RENORMALIZATION,
)
UFUZZ = MethodSpec(
    "ufuzz",
    "U-Fuzz",
    SelectionFamily.U_FUZZ,
    MutationSpace.FULL,
    FairnessGroup.FULL_SPACE_PRIMARY,
    OnlineSearchSignal.U_FUZZ_RETRIEVAL_FEEDBACK,
    FULL_UFUZZ_FEEDBACK,
    FeedbackCombinationRule.ZERO_DISABLED_WITHOUT_RENORMALIZATION,
)
UFUZZ_WITHOUT_COVERAGE = MethodSpec(
    "ufuzz-without-coverage",
    "U-Fuzz w/o Coverage",
    SelectionFamily.U_FUZZ,
    MutationSpace.FULL,
    FairnessGroup.FULL_SPACE_ABLATION,
    OnlineSearchSignal.U_FUZZ_RETRIEVAL_FEEDBACK,
    WITHOUT_COVERAGE,
    FeedbackCombinationRule.ZERO_DISABLED_WITHOUT_RENORMALIZATION,
)
UFUZZ_WITHOUT_NOVELTY = MethodSpec(
    "ufuzz-without-novelty",
    "U-Fuzz w/o Novelty",
    SelectionFamily.U_FUZZ,
    MutationSpace.FULL,
    FairnessGroup.FULL_SPACE_ABLATION,
    OnlineSearchSignal.U_FUZZ_RETRIEVAL_FEEDBACK,
    WITHOUT_NOVELTY,
    FeedbackCombinationRule.ZERO_DISABLED_WITHOUT_RENORMALIZATION,
)
UFUZZ_WITHOUT_PARENT_DIVERGENCE = MethodSpec(
    "ufuzz-without-parent-divergence",
    "U-Fuzz w/o Parent Divergence",
    SelectionFamily.U_FUZZ,
    MutationSpace.FULL,
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
RQ3_METHODS = (
    COVERAGE_GUIDED,
    UFUZZ_WITHOUT_COVERAGE,
    UFUZZ_WITHOUT_NOVELTY,
    UFUZZ_WITHOUT_PARENT_DIVERGENCE,
    UFUZZ,
)
NEW_RQ3_METHODS = (
    UFUZZ_WITHOUT_COVERAGE,
    UFUZZ_WITHOUT_NOVELTY,
    UFUZZ_WITHOUT_PARENT_DIVERGENCE,
)
ALL_METHODS = RQ1_METHODS + NEW_RQ3_METHODS
METHODS_BY_ID = MappingProxyType({method.method_id: method for method in ALL_METHODS})

BACKENDS = (Backend.MEM0, Backend.AMEM, Backend.GRAPHITI)
REPETITION_INDICES = (0, 1, 2)
RQ2_CHECKPOINTS = tuple(range(1000, 8001, 1000))
BENCHMARK_MAX_BUDGET = MappingProxyType(
    {Benchmark.LOCOMO: 8000, Benchmark.LONGMEMEVAL_S: 4000}
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
    configuration_bindings: tuple[ConfigurationBinding, ...] = ()


@dataclass(frozen=True, slots=True)
class CampaignSpec:
    """One unique scientific campaign, independent of any RQ view."""

    benchmark: Benchmark
    backend: Backend
    method: MethodSpec
    repetition_index: int
    max_budget: int
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
        expected_budget = BENCHMARK_MAX_BUDGET[self.benchmark]
        if self.max_budget != expected_budget:
            raise ValueError("campaign max budget differs from the benchmark contract")
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

    def __post_init__(self) -> None:
        campaigns = tuple(self.campaigns)
        keys = [campaign.scientific_key for campaign in campaigns]
        if len(set(keys)) != len(keys):
            raise ValueError("evaluation plan contains duplicate scientific campaigns")
        canonical = {campaign.scientific_key: campaign for campaign in campaigns}
        for entry in self.rq1 + self.rq2 + self.rq3:
            if canonical.get(entry.campaign.scientific_key) is not entry.campaign:
                raise ValueError("RQ views must reference canonical campaign objects")
        object.__setattr__(self, "campaigns", campaigns)
        object.__setattr__(self, "rq1", tuple(self.rq1))
        object.__setattr__(self, "rq2", tuple(self.rq2))
        object.__setattr__(self, "rq3", tuple(self.rq3))

    @property
    def total_planned_valid_executions(self) -> int:
        return sum(campaign.max_budget for campaign in self.campaigns)


def build_evaluation_plan() -> EvaluationPlan:
    """Build the deduplicated 153-campaign plan and its three RQ views."""

    campaigns: dict[CampaignKey, CampaignSpec] = {}

    def add(benchmark: Benchmark, backend: Backend, method: MethodSpec, rep: int) -> None:
        spec = CampaignSpec(
            benchmark,
            backend,
            method,
            rep,
            BENCHMARK_MAX_BUDGET[benchmark],
        )
        campaigns.setdefault(spec.scientific_key, spec)

    for benchmark in Benchmark:
        for backend in BACKENDS:
            for method in RQ1_METHODS:
                for repetition in REPETITION_INDICES:
                    add(benchmark, backend, method, repetition)
    for backend in BACKENDS:
        for method in NEW_RQ3_METHODS:
            for repetition in REPETITION_INDICES:
                add(Benchmark.LOCOMO, backend, method, repetition)

    def get(
        benchmark: Benchmark, backend: Backend, method: MethodSpec, repetition: int
    ) -> CampaignSpec:
        key = CampaignSpec(
            benchmark,
            backend,
            method,
            repetition,
            BENCHMARK_MAX_BUDGET[benchmark],
        ).scientific_key
        return campaigns[key]

    rq1 = tuple(
        RQViewEntry(
            ResearchQuestion.RQ1,
            method.method_id,
            BENCHMARK_MAX_BUDGET[benchmark],
            get(benchmark, backend, method, repetition),
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
            get(Benchmark.LOCOMO, backend, method, repetition),
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
            8000,
            get(Benchmark.LOCOMO, backend, method, repetition),
        )
        for backend in BACKENDS
        for method in RQ3_METHODS
        for repetition in REPETITION_INDICES
    )
    ordered_campaigns = tuple(
        sorted(
            campaigns.values(),
            key=lambda item: (
                item.benchmark.value,
                item.backend.value,
                item.method.method_id,
                item.repetition_index,
            ),
        )
    )
    return EvaluationPlan(ordered_campaigns, rq1, rq2, rq3)


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
class EngineeringTarget:
    gpu_count: int
    wall_clock_seconds: int
    planned_valid_executions: int

    @property
    def minimum_average_valid_executions_per_second(self) -> float:
        return self.planned_valid_executions / self.wall_clock_seconds


ENGINEERING_TARGET = EngineeringTarget(6, 24 * 60 * 60, 972000)


EVALUATION_PLAN = build_evaluation_plan()
