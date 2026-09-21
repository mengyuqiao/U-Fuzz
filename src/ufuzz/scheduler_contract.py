"""Pure scientific scheduler semantics for the frozen RQ1--RQ4 plan.

This module describes trajectory-producing choices without executing a
scheduler.  It has no queue implementation, backend access, model calls,
budget counter, evaluator, result storage, or runtime worker policy.  Existing
state, materialization, retrieval, coverage, feedback, and evaluation
contracts remain authoritative for their respective boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from types import MappingProxyType

from ufuzz.evaluation_contract import (
    COVERAGE_GUIDED,
    LLM_AS_JUDGE,
    METHODS_BY_ID,
    RANDOM_MUTATION,
    RQ2_CHECKPOINTS,
    UFUZZ,
    UFUZZ_M,
    UFUZZ_Q,
    UFUZZ_WITHOUT_COVERAGE,
    UFUZZ_WITHOUT_NOVELTY,
    UFUZZ_WITHOUT_PARENT_DIVERGENCE,
    RQ3_OPERATOR_ABLATIONS,
    UNGUIDED_LLM,
    MethodSpec,
    MutationRelationConfiguration,
)
from ufuzz.state_contract import (
    LogicalSeed,
    MutationOpportunity,
    validate_opportunity_parent,
)


class SchedulerPolicyVersion(StrEnum):
    ROOT_CYCLIC_RETAIN_ALL_RELATION_MASK_V2 = (
        "root-cyclic-retain-all-relation-mask-v2"
    )


class RootSchedulingRule(StrEnum):
    """Method-neutral allocation of valid executions across roots."""

    CYCLIC_PERSISTENT_REPETITION_PERMUTATION = (
        "cyclic_persistent_repetition_permutation"
    )


class FrontierSelectionRule(StrEnum):
    UNIFORM_ELIGIBLE_FRONTIER_ITEMS = "uniform_eligible_frontier_items"
    LLM_CONTROLLER_SELECTS_ONE = "llm_controller_selects_one"
    MAXIMAL_PARENT_PRIORITY = "maximal_parent_priority"


class PrioritySource(StrEnum):
    NONE = "none"
    COVERAGE_GAIN = "coverage_gain"
    FROZEN_UFUZZ_METHOD_SCORE = "frozen_ufuzz_method_score"
    ONLINE_JUDGE_SCORE = "online_judge_score"


BOOTSTRAP_PRIORITY = 0.0


class OpportunityDisposition(StrEnum):
    ELIGIBLE = "eligible"
    TERMINAL_SUCCESS = "terminal_success"
    TERMINAL_GENERATION_EXHAUSTED = "terminal_generation_exhausted"

    @property
    def terminal(self) -> bool:
        return self is not OpportunityDisposition.ELIGIBLE


class RootVisitDisposition(StrEnum):
    """The only scientific outcomes that finish one root visit."""

    TERMINAL_SUCCESS = "terminal_success"
    TERMINAL_GENERATION_EXHAUSTED = "terminal_generation_exhausted"


class OpportunityEnumerationStatus(StrEnum):
    CERTIFIED_COMPLETE = "certified_complete"
    BLOCKED_CAPABILITY_FAILURE = "blocked_capability_failure"
    BLOCKED_TRANSIENT_INFRASTRUCTURE_FAILURE = (
        "blocked_transient_infrastructure_failure"
    )


class SeedAdmissionKind(StrEnum):
    INITIAL = "initial"
    VALID_EXECUTED_CHILD = "valid_executed_child"


class OpportunityEnumerationPolicyVersion(StrEnum):
    COMPLETE_AT_SEED_ADMISSION_RELATION_MASK_V2 = (
        "complete-at-seed-admission-relation-mask-v2"
    )


class OpportunityExistenceInput(StrEnum):
    FROZEN_LOGICAL_SEED_STATE = "frozen_logical_seed_state"
    STRUCTURAL_INDEX_CERTIFICATION = "structural_index_certification"
    EXACT_MUTATION_RELATION_MASK = "exact_mutation_relation_mask"
    FROZEN_SCIENTIFIC_CONFIGURATION = "frozen_scientific_configuration"


class OpportunityExistenceForbiddenFactor(StrEnum):
    RETRIEVAL_FEEDBACK = "retrieval_feedback"
    PARENT_PRIORITY = "parent_priority"
    EVALUATOR_OUTPUT = "evaluator_output"
    QUEUE_PRESSURE = "queue_pressure"
    WORKER_ORDER = "worker_order"
    GPU_BATCH_ORDER = "gpu_batch_order"
    WALL_CLOCK = "wall_clock"


class SeedRetentionRule(StrEnum):
    RETAIN_ALL_VALID_REMATERIALIZABLE_CHILDREN = (
        "retain_all_valid_rematerializable_children"
    )


class ParentAvailabilityRule(StrEnum):
    RETAIN_UNTIL_COMPLETE_OPPORTUNITY_SET_TERMINAL = (
        "retain_until_complete_opportunity_set_terminal"
    )


class ChildEligibilityRule(StrEnum):
    NEXT_ROUND = "next_round"


class DepthRule(StrEnum):
    NO_SCIENTIFIC_MAXIMUM = "no_scientific_maximum"


class RelationBalancingRule(StrEnum):
    NONE = "none"


class RealizationYieldRule(StrEnum):
    ONE_EXACT_ARTIFACT_PER_ATTEMPT = "one_exact_artifact_per_attempt"


class PriorityTiming(StrEnum):
    NOT_APPLICABLE = "not_applicable"
    POST_VALID_EXECUTION_FUTURE_DECISIONS_ONLY = (
        "post_valid_execution_future_decisions_only"
    )


class MethodInformationProfile(StrEnum):
    STANDARD_SEARCH_SAFE = "standard_search_safe"
    UNGUIDED_CONTROLLER_RESTRICTED = "unguided_controller_restricted"
    ONLINE_JUDGE_POST_EXECUTION_RESTRICTED = (
        "online_judge_post_execution_restricted"
    )


class RetryClass(StrEnum):
    MUTATION_REALIZATION_OUTPUT_RESAMPLING = (
        "mutation_realization_output_resampling"
    )
    UNGUIDED_CONTROLLER_OUTPUT_RESAMPLING = (
        "unguided_controller_output_resampling"
    )
    ONLINE_JUDGE_OUTPUT_RESAMPLING = "online_judge_output_resampling"
    TRANSIENT_TRANSPORT_OR_SERVICE = "transient_transport_or_service"


class RetryMode(StrEnum):
    REPLAY_SAME_SCIENTIFIC_EVENT = "replay_same_scientific_event"
    RESAMPLE_WITH_INCREMENTED_ROLE_ATTEMPT = (
        "resample_with_incremented_role_attempt"
    )


class ExhaustionKind(StrEnum):
    ROOT_SCIENTIFIC = "root_scientific"
    CAMPAIGN_SCIENTIFIC_SHORTFALL = "campaign_scientific_shortfall"
    INFRASTRUCTURE_BLOCKED_RESUMABLE = "infrastructure_blocked_resumable"
    MODEL_SERVICE_BLOCKED_RESUMABLE = "model_service_blocked_resumable"


class RandomStreamName(StrEnum):
    ROOT_ORDER = "root_order"
    RANDOM_BASELINE_CHOICE = "random_baseline_choice"
    SCHEDULER_TIE_BREAK = "scheduler_tie_break"
    MUTATION_REALIZATION = "mutation_realization"
    UNGUIDED_CONTROLLER_SAMPLING = "unguided_controller_sampling"
    ONLINE_JUDGE_SAMPLING = "online_judge_sampling"
    SEMANTIC_VALIDATION_SAMPLING = "semantic_validation_sampling"
    RESPONSE_GENERATION_SAMPLING = "response_generation_sampling"
    FINAL_EVALUATOR_CFS_SAMPLING = "final_evaluator_cfs_sampling"


class UnguidedContextField(StrEnum):
    ROOT_QUERY_SEMANTIC_CONTEXT = "root_query_semantic_context"
    PERSISTENT_LOGICAL_PARENT_RECIPE = "persistent_logical_parent_recipe"
    MUTATION_RELATION_OPPORTUNITY = "mutation_relation_opportunity"
    PRIOR_MUTATION_PATH_STRUCTURE = "prior_mutation_path_structure"


class ForbiddenSearchInput(StrEnum):
    COVERAGE_STATE = "coverage_state"
    RAW_OR_NORMALIZED_COVERAGE_GAIN = "raw_or_normalized_coverage_gain"
    RETRIEVAL_SIGNATURE = "retrieval_signature"
    NOVELTY = "novelty"
    PARENT_DIVERGENCE = "parent_divergence"
    UFUZZ_SCORE = "ufuzz_score"
    FINAL_EVALUATOR_VERDICT = "final_evaluator_verdict"
    FAILURE_SURFACE = "failure_surface"
    CANONICAL_FAULT_SIGNATURE = "canonical_fault_signature"
    UF_AT_B_RESULT = "uf_at_b_result"
    GOLD_ANSWER = "gold_answer"
    GOLD_ANSWERS = "gold_answers"
    GOLD_OR_REFERENCE_VAULT = "gold_or_reference_vault"
    REFERENCE_VAULT_EVALUATOR_DATA = "reference_vault_evaluator_data"
    FINAL_CORRECTNESS = "final_correctness"
    ANSWER_CORRECTNESS = "answer_correctness"


class OnlineJudgeEvidenceField(StrEnum):
    EXACT_QUERY_ARTIFACT = "exact_query_artifact"
    EXACT_REALIZED_MUTATION_ARTIFACT = "exact_realized_mutation_artifact"
    RANKED_PUBLIC_MEMORY_CONTENT_PROJECTIONS = (
        "ranked_public_memory_content_projections"
    )
    RESOLVED_RETRIEVAL_STRUCTURE_RANKS = "resolved_retrieval_structure_ranks"
    NON_EVALUATOR_CAMPAIGN_CONTEXT = "non_evaluator_campaign_context"


class RetrievalDepthUse(StrEnum):
    BACKEND_TOP_K = "backend_top_k"
    INITIALIZATION_SIGNATURE = "initialization_signature"
    PARENT_CHILD_SIGNATURES = "parent_child_signatures"
    RBO_PERSISTENCE = "rbo_persistence"
    NORMALIZED_COVERAGE_GAIN = "normalized_coverage_gain"


class ProductionBindingRequirement(StrEnum):
    BENCHMARK_VERSION_PREPROCESSING = "benchmark_version_preprocessing"
    BACKEND_RUNTIME_PROFILE = "backend_runtime_profile"
    SCHEDULER_POLICY_VERSION = "scheduler_policy_version"
    PRNG_DERIVATION_AND_REPETITION_SEED = (
        "prng_derivation_and_repetition_seed"
    )
    RETRIEVAL_DEPTH = "retrieval_depth"
    MUTATION_REALIZATION = "mutation_realization"
    MUTATION_INDEX_OPPORTUNITY_ENUMERATION = (
        "mutation_index_opportunity_enumeration"
    )
    MUTATION_GENERATION_ATTEMPT_CAP = "mutation_generation_attempt_cap"
    TRANSIENT_INFRASTRUCTURE_REPLAY_CAP = (
        "transient_infrastructure_replay_cap"
    )
    SEMANTIC_VALIDATION_STRUCTURAL_EXTRACTION = (
        "semantic_validation_structural_extraction"
    )
    RESPONSE_GENERATION = "response_generation"
    FINAL_EVALUATOR_CFS = "final_evaluator_cfs"
    NATIVE_MEMORY_LLM_CONFIGURATION = "native_memory_llm_configuration"
    UNGUIDED_CONTROLLER = "unguided_controller"
    UNGUIDED_CONTROLLER_OUTPUT_RESAMPLING_CAP = (
        "unguided_controller_output_resampling_cap"
    )
    ONLINE_JUDGE = "online_judge"
    ONLINE_JUDGE_OUTPUT_RESAMPLING_CAP = (
        "online_judge_output_resampling_cap"
    )


class EngineeringOnlySetting(StrEnum):
    WORKER_COUNT = "worker_count"
    GPU_MODEL = "gpu_model"
    BATCH_SIZE = "batch_size"
    SERVING_QUEUE_DEPTH = "serving_queue_depth"
    REPLICA_COUNT = "replica_count"


class ValidExecutionStage(StrEnum):
    CONSTRUCT_RESOLVED_OBSERVATION = "construct_resolved_observation"
    CROSS_VALID_EXECUTION_BOUNDARY = "cross_valid_execution_boundary"
    INCREMENT_VALID_EXECUTION_INDEX = "increment_valid_execution_index"
    ADOPT_COVERAGE_UPDATE_STATE = "adopt_coverage_update_state"
    COMPUTE_POST_EXECUTION_FEEDBACK = "compute_post_execution_feedback"
    INSERT_ROOT_HISTORY_AFTER_SCORING = "insert_root_history_after_scoring"
    RETAIN_VALID_CHILD = "retain_valid_child"
    ASSIGN_FUTURE_PRIORITY = "assign_future_priority"
    PERSIST_ORDERED_WITNESS = "persist_ordered_witness"
    POST_B_RESPONSE_AND_FINAL_EVALUATION = (
        "post_b_response_and_final_evaluation"
    )


ATOMIC_VALID_EXECUTION_ORDER = tuple(ValidExecutionStage)


@dataclass(frozen=True, slots=True)
class RootOrderEvent:
    """Method-neutral identity for the campaign-start root permutation."""

    benchmark_corpus_id: str
    repetition_index: int
    root_corpus_id: str
    stream: RandomStreamName = RandomStreamName.ROOT_ORDER

    def __post_init__(self) -> None:
        for name in ("benchmark_corpus_id", "root_corpus_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if isinstance(self.repetition_index, bool) or not isinstance(
            self.repetition_index, int
        ):
            raise TypeError("repetition_index must be an integer")
        if self.repetition_index < 0:
            raise ValueError("repetition_index cannot be negative")
        if self.stream is not RandomStreamName.ROOT_ORDER:
            raise ValueError("root order must use the named root_order stream")


@dataclass(frozen=True, slots=True)
class MutationRealizationEvent:
    """Arrival-order-independent identity for one realization attempt."""

    benchmark_corpus_id: str
    root_checkpoint_id: str
    logical_parent_identity: str
    logical_opportunity_identity: str
    repetition_index: int
    realization_attempt_index: int
    stream: RandomStreamName = RandomStreamName.MUTATION_REALIZATION

    def __post_init__(self) -> None:
        for name in (
            "benchmark_corpus_id",
            "root_checkpoint_id",
            "logical_parent_identity",
            "logical_opportunity_identity",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("repetition_index", "realization_attempt_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.stream is not RandomStreamName.MUTATION_REALIZATION:
            raise ValueError("realization events require the mutation_realization stream")


@dataclass(frozen=True, slots=True)
class ScientificRandomEvent:
    """Runtime-order-free identity for any other stochastic scientific role."""

    stream: RandomStreamName
    stable_scientific_event_id: str
    role_attempt_index: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.stream, RandomStreamName):
            raise TypeError("stream must be a RandomStreamName")
        if not isinstance(self.stable_scientific_event_id, str) or not (
            self.stable_scientific_event_id
        ):
            raise ValueError("stable_scientific_event_id must be non-empty")
        if isinstance(self.role_attempt_index, bool) or not isinstance(
            self.role_attempt_index, int
        ):
            raise TypeError("role_attempt_index must be an integer")
        if self.role_attempt_index < 0:
            raise ValueError("role_attempt_index cannot be negative")


@dataclass(frozen=True, slots=True)
class RootRoundContract:
    scheduling_rule: RootSchedulingRule
    order_stream: RandomStreamName
    max_valid_executions_per_active_root: int
    stop_immediately_at_budget: bool
    skip_temporarily_ineligible_roots: bool
    remove_scientifically_exhausted_roots: bool
    method_neutral: bool
    frontier_decisions_per_root_visit: int
    second_frontier_selection_in_same_visit: bool

    def __post_init__(self) -> None:
        if self.scheduling_rule is not (
            RootSchedulingRule.CYCLIC_PERSISTENT_REPETITION_PERMUTATION
        ):
            raise ValueError("the frozen outer scheduler is cyclic")
        if self.order_stream is not RandomStreamName.ROOT_ORDER:
            raise ValueError("the root permutation requires the root_order stream")
        if self.max_valid_executions_per_active_root != 1:
            raise ValueError("a root may contribute at most one valid execution per round")
        if self.frontier_decisions_per_root_visit != 1:
            raise ValueError("one root visit is exactly one frontier decision")
        if self.second_frontier_selection_in_same_visit:
            raise ValueError("a root visit cannot select a second frontier item")
        if not all(
            (
                self.stop_immediately_at_budget,
                self.skip_temporarily_ineligible_roots,
                self.remove_scientifically_exhausted_roots,
                self.method_neutral,
            )
        ):
            raise ValueError("root-round control flags are fixed to true")


@dataclass(frozen=True, slots=True)
class FrontierItem:
    """One persistent parent and one method-independent opportunity."""

    parent: LogicalSeed
    opportunity: MutationOpportunity

    def __post_init__(self) -> None:
        validate_opportunity_parent(self.opportunity, self.parent)

    @property
    def scientific_identity(self) -> tuple[str, str, str, str]:
        return (
            self.parent.campaign_id,
            self.parent.root_checkpoint_id,
            self.parent.seed_id,
            self.opportunity.opportunity_id,
        )


@dataclass(frozen=True, slots=True)
class RootVisitResult:
    """Terminal scientific result of exactly one selected frontier decision."""

    selected_item: FrontierItem
    disposition: RootVisitDisposition
    frontier_decisions: int
    valid_executions: int
    visit_complete: bool
    may_select_second_frontier_in_round: bool
    advance_to_next_root: bool

    def __post_init__(self) -> None:
        if not isinstance(self.selected_item, FrontierItem):
            raise TypeError("selected_item must be a FrontierItem")
        if not isinstance(self.disposition, RootVisitDisposition):
            raise TypeError("disposition must be a RootVisitDisposition")
        if self.frontier_decisions != 1:
            raise ValueError("one root visit must contain exactly one frontier decision")
        expected_valid = (
            1 if self.disposition is RootVisitDisposition.TERMINAL_SUCCESS else 0
        )
        if self.valid_executions != expected_valid:
            raise ValueError("root-visit valid execution count contradicts disposition")
        if not self.visit_complete or not self.advance_to_next_root:
            raise ValueError("a terminal frontier disposition ends the root visit")
        if self.may_select_second_frontier_in_round:
            raise ValueError("a terminal root visit cannot select another frontier item")

    @classmethod
    def terminal(
        cls,
        selected_item: FrontierItem,
        disposition: OpportunityDisposition,
    ) -> RootVisitResult:
        if disposition is OpportunityDisposition.TERMINAL_SUCCESS:
            visit_disposition = RootVisitDisposition.TERMINAL_SUCCESS
        elif disposition is OpportunityDisposition.TERMINAL_GENERATION_EXHAUSTED:
            visit_disposition = RootVisitDisposition.TERMINAL_GENERATION_EXHAUSTED
        else:
            raise ValueError("a terminal root-visit result requires terminal disposition")
        return cls(
            selected_item,
            visit_disposition,
            1,
            1 if disposition is OpportunityDisposition.TERMINAL_SUCCESS else 0,
            True,
            False,
            True,
        )


@dataclass(frozen=True, slots=True)
class RootVisitTransientRetry:
    """Same-event transport replay that remains inside one root visit."""

    selected_item: FrontierItem
    frontier_decisions: int = 1
    same_realized_artifact: bool = True
    same_scientific_event: bool = True
    visit_complete: bool = False
    advance_to_next_root: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.selected_item, FrontierItem):
            raise TypeError("selected_item must be a FrontierItem")
        if self.frontier_decisions != 1:
            raise ValueError("transport retry cannot create a frontier decision")
        if not self.same_realized_artifact or not self.same_scientific_event:
            raise ValueError("transport retry must replay the same scientific event")
        if self.visit_complete or self.advance_to_next_root:
            raise ValueError("transport retry remains inside the current root visit")


@dataclass(frozen=True, slots=True)
class SeedOpportunityEnumeration:
    """Complete immutable scheduler-visible opportunities fixed at admission."""

    seed: LogicalSeed
    relation_configuration: MutationRelationConfiguration
    admission_kind: SeedAdmissionKind
    admission_round: int
    status: OpportunityEnumerationStatus
    opportunities: frozenset[MutationOpportunity]
    policy_version: OpportunityEnumerationPolicyVersion = (
        OpportunityEnumerationPolicyVersion.COMPLETE_AT_SEED_ADMISSION_RELATION_MASK_V2
    )

    def __post_init__(self) -> None:
        if not isinstance(self.seed, LogicalSeed):
            raise TypeError("seed must be a LogicalSeed")
        if not isinstance(
            self.relation_configuration, MutationRelationConfiguration
        ):
            raise TypeError(
                "relation_configuration must be a MutationRelationConfiguration"
            )
        if not isinstance(self.admission_kind, SeedAdmissionKind):
            raise TypeError("admission_kind must be a SeedAdmissionKind")
        if isinstance(self.admission_round, bool) or not isinstance(
            self.admission_round, int
        ):
            raise TypeError("admission_round must be an integer")
        if self.admission_kind is SeedAdmissionKind.INITIAL:
            if self.admission_round != 0:
                raise ValueError("initial seed enumeration occurs before round one")
        elif self.admission_round < 1:
            raise ValueError("child admission_round must be the producing round")
        if not isinstance(self.status, OpportunityEnumerationStatus):
            raise TypeError("status must be an OpportunityEnumerationStatus")
        if self.policy_version is not (
            OpportunityEnumerationPolicyVersion.COMPLETE_AT_SEED_ADMISSION_RELATION_MASK_V2
        ):
            raise ValueError("enumeration must use the frozen policy version")
        opportunities = frozenset(self.opportunities)
        object.__setattr__(self, "opportunities", opportunities)
        if self.status is not OpportunityEnumerationStatus.CERTIFIED_COMPLETE:
            if opportunities:
                raise ValueError("an uncertified enumeration cannot expose a partial set")
            return
        allowed_relations = self.relation_configuration.enabled_relations
        opportunity_ids: set[str] = set()
        for opportunity in opportunities:
            validate_opportunity_parent(opportunity, self.seed)
            if opportunity.relation not in allowed_relations:
                raise ValueError(
                    "enumerated opportunity violates exact mutation-relation mask"
                )
            if opportunity.opportunity_id in opportunity_ids:
                raise ValueError("opportunity IDs must be unique within one seed")
            opportunity_ids.add(opportunity.opportunity_id)

    @property
    def certified_complete(self) -> bool:
        return self.status is OpportunityEnumerationStatus.CERTIFIED_COMPLETE

    @property
    def locally_exhausted(self) -> bool:
        return self.certified_complete and not self.opportunities

    @property
    def supports_scientific_exhaustion(self) -> bool:
        return self.certified_complete

    @property
    def first_eligible_round(self) -> int:
        return 1 if self.admission_kind is SeedAdmissionKind.INITIAL else (
            self.admission_round + 1
        )

    def eligible_in_round(self, round_index: int) -> bool:
        if isinstance(round_index, bool) or not isinstance(round_index, int):
            raise TypeError("round_index must be an integer")
        return self.certified_complete and round_index >= self.first_eligible_round


OPPORTUNITY_EXISTENCE_INPUTS = frozenset(OpportunityExistenceInput)
OPPORTUNITY_EXISTENCE_FORBIDDEN_FACTORS = frozenset(
    OpportunityExistenceForbiddenFactor
)


@dataclass(frozen=True, slots=True)
class OpportunityEnumerationContract:
    policy_version: OpportunityEnumerationPolicyVersion
    complete_at_seed_admission: bool
    initial_enumeration_before_round_one: bool
    child_enumeration_during_admission: bool
    immutable_after_admission: bool
    existence_inputs: frozenset[OpportunityExistenceInput]
    forbidden_factors: frozenset[OpportunityExistenceForbiddenFactor]
    implementation_binding: ProductionBindingRequirement

    def __post_init__(self) -> None:
        if self.policy_version is not (
            OpportunityEnumerationPolicyVersion.COMPLETE_AT_SEED_ADMISSION_RELATION_MASK_V2
        ):
            raise ValueError("unknown opportunity-enumeration policy")
        if not all(
            (
                self.complete_at_seed_admission,
                self.initial_enumeration_before_round_one,
                self.child_enumeration_during_admission,
                self.immutable_after_admission,
            )
        ):
            raise ValueError("complete immutable admission enumeration is required")
        object.__setattr__(self, "existence_inputs", frozenset(self.existence_inputs))
        object.__setattr__(self, "forbidden_factors", frozenset(self.forbidden_factors))
        if self.existence_inputs != OPPORTUNITY_EXISTENCE_INPUTS:
            raise ValueError("opportunity-existence inputs are frozen")
        if self.forbidden_factors != OPPORTUNITY_EXISTENCE_FORBIDDEN_FACTORS:
            raise ValueError("dynamic opportunity discovery is forbidden")
        if self.implementation_binding is not (
            ProductionBindingRequirement.MUTATION_INDEX_OPPORTUNITY_ENUMERATION
        ):
            raise ValueError("enumeration implementation needs exact provenance")


@dataclass(frozen=True, slots=True)
class FrontierEligibility:
    """Pure snapshot of whether one frontier item may be selected."""

    item: FrontierItem
    parent_retained: bool
    relation_configuration: MutationRelationConfiguration
    disposition: OpportunityDisposition = OpportunityDisposition.ELIGIBLE

    def __post_init__(self) -> None:
        if not isinstance(self.parent_retained, bool):
            raise TypeError("parent_retained must be a bool")
        if not isinstance(
            self.relation_configuration, MutationRelationConfiguration
        ):
            raise TypeError(
                "relation_configuration must be a MutationRelationConfiguration"
            )
        if not isinstance(self.disposition, OpportunityDisposition):
            raise TypeError("disposition must be an OpportunityDisposition")

    @property
    def relation_allowed(self) -> bool:
        return (
            self.item.opportunity.relation
            in self.relation_configuration.enabled_relations
        )

    @property
    def eligible(self) -> bool:
        return (
            self.parent_retained
            and self.relation_allowed
            and self.disposition is OpportunityDisposition.ELIGIBLE
        )


def validate_opportunity_disposition_transition(
    before: OpportunityDisposition,
    after: OpportunityDisposition,
) -> OpportunityDisposition:
    """Reject revisiting any terminal parent/opportunity lifecycle."""

    if not isinstance(before, OpportunityDisposition) or not isinstance(
        after, OpportunityDisposition
    ):
        raise TypeError("opportunity dispositions must use the frozen enum")
    if before.terminal:
        raise ValueError("a terminal parent/opportunity pair cannot be revisited")
    if after is OpportunityDisposition.ELIGIBLE:
        raise ValueError("a scientific attempt must end or remain external to transition")
    return after


@dataclass(frozen=True, slots=True)
class RealizationContract:
    yield_rule: RealizationYieldRule
    generic_preexecution_candidate_pool: bool
    shared_interface_and_validator: bool
    event_keyed_randomness: bool
    arrival_order_independent: bool
    generation_attempt_cap: None = None

    def __post_init__(self) -> None:
        if self.yield_rule is not RealizationYieldRule.ONE_EXACT_ARTIFACT_PER_ATTEMPT:
            raise ValueError("one realization attempt yields one exact artifact")
        if self.generic_preexecution_candidate_pool:
            raise ValueError("the frozen scheduler has no generic candidate pool")
        if not all(
            (
                self.shared_interface_and_validator,
                self.event_keyed_randomness,
                self.arrival_order_independent,
            )
        ):
            raise ValueError("realization reproducibility guarantees are required")
        if self.generation_attempt_cap is not None:
            raise ValueError("the numeric generation-attempt cap remains unbound")


@dataclass(frozen=True, slots=True)
class RetentionContract:
    seed_rule: SeedRetentionRule
    parent_rule: ParentAvailabilityRule
    child_eligibility: ChildEligibilityRule
    depth_rule: DepthRule
    logical_queue_capacity: None = None
    score_threshold: None = None
    eviction_rule: None = None
    aging_rule: None = None
    maximum_scientific_depth: None = None

    def __post_init__(self) -> None:
        if self.seed_rule is not (
            SeedRetentionRule.RETAIN_ALL_VALID_REMATERIALIZABLE_CHILDREN
        ):
            raise ValueError("all valid rematerializable children are retained")
        if self.parent_rule is not (
            ParentAvailabilityRule.RETAIN_UNTIL_COMPLETE_OPPORTUNITY_SET_TERMINAL
        ):
            raise ValueError("parents remain until their frontier is terminal")
        if self.child_eligibility is not ChildEligibilityRule.NEXT_ROUND:
            raise ValueError("new children first become eligible next round")
        if self.depth_rule is not DepthRule.NO_SCIENTIFIC_MAXIMUM:
            raise ValueError("the scheduler cannot impose a maximum depth")
        if any(
            value is not None
            for value in (
                self.logical_queue_capacity,
                self.score_threshold,
                self.eviction_rule,
                self.aging_rule,
                self.maximum_scientific_depth,
            )
        ):
            raise ValueError("capacity, thresholds, eviction, aging, and depth are absent")


@dataclass(frozen=True, slots=True)
class MethodSchedulingContract:
    method_id: str
    selection_rule: FrontierSelectionRule
    priority_source: PrioritySource
    priority_timing: PriorityTiming
    consumes_coverage_novelty_divergence_feedback: bool
    selection_stream: RandomStreamName
    information_profile: MethodInformationProfile
    initial_seed_priority: float = BOOTSTRAP_PRIORITY
    online_judge_is_post_execution: bool = False
    child_priority_overwrites_parent: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.method_id, str) or not self.method_id:
            raise ValueError("method_id must be a non-empty string")
        if self.priority_source is PrioritySource.NONE:
            if self.priority_timing is not PriorityTiming.NOT_APPLICABLE:
                raise ValueError("methods without priority require no priority timing")
        elif self.priority_timing is not (
            PriorityTiming.POST_VALID_EXECUTION_FUTURE_DECISIONS_ONLY
        ):
            raise ValueError("method priority must be post-execution and future-facing")
        if (
            isinstance(self.initial_seed_priority, bool)
            or not isinstance(self.initial_seed_priority, (int, float))
            or float(self.initial_seed_priority) != BOOTSTRAP_PRIORITY
        ):
            raise ValueError("initial seeds require numeric bootstrap priority 0.0")
        if self.child_priority_overwrites_parent:
            raise ValueError("a child's priority cannot overwrite its parent")
        if self.priority_source is PrioritySource.ONLINE_JUDGE_SCORE:
            if not self.online_judge_is_post_execution:
                raise ValueError("the online judge must be post-execution")
        elif self.online_judge_is_post_execution:
            raise ValueError("only LLM-as-Judge carries online judge timing")

    @property
    def priority_required_before_future_selection(self) -> bool:
        return self.priority_source is not PrioritySource.NONE


_METHOD_SCHEDULING_CONTRACTS = MappingProxyType(
    {
        RANDOM_MUTATION.method_id: MethodSchedulingContract(
            RANDOM_MUTATION.method_id,
            FrontierSelectionRule.UNIFORM_ELIGIBLE_FRONTIER_ITEMS,
            PrioritySource.NONE,
            PriorityTiming.NOT_APPLICABLE,
            False,
            RandomStreamName.RANDOM_BASELINE_CHOICE,
            MethodInformationProfile.STANDARD_SEARCH_SAFE,
        ),
        UNGUIDED_LLM.method_id: MethodSchedulingContract(
            UNGUIDED_LLM.method_id,
            FrontierSelectionRule.LLM_CONTROLLER_SELECTS_ONE,
            PrioritySource.NONE,
            PriorityTiming.NOT_APPLICABLE,
            False,
            RandomStreamName.UNGUIDED_CONTROLLER_SAMPLING,
            MethodInformationProfile.UNGUIDED_CONTROLLER_RESTRICTED,
        ),
        LLM_AS_JUDGE.method_id: MethodSchedulingContract(
            LLM_AS_JUDGE.method_id,
            FrontierSelectionRule.MAXIMAL_PARENT_PRIORITY,
            PrioritySource.ONLINE_JUDGE_SCORE,
            PriorityTiming.POST_VALID_EXECUTION_FUTURE_DECISIONS_ONLY,
            False,
            RandomStreamName.SCHEDULER_TIE_BREAK,
            MethodInformationProfile.ONLINE_JUDGE_POST_EXECUTION_RESTRICTED,
            online_judge_is_post_execution=True,
        ),
        COVERAGE_GUIDED.method_id: MethodSchedulingContract(
            COVERAGE_GUIDED.method_id,
            FrontierSelectionRule.MAXIMAL_PARENT_PRIORITY,
            PrioritySource.COVERAGE_GAIN,
            PriorityTiming.POST_VALID_EXECUTION_FUTURE_DECISIONS_ONLY,
            True,
            RandomStreamName.SCHEDULER_TIE_BREAK,
            MethodInformationProfile.STANDARD_SEARCH_SAFE,
        ),
        **{
            method.method_id: MethodSchedulingContract(
                method.method_id,
                FrontierSelectionRule.MAXIMAL_PARENT_PRIORITY,
                PrioritySource.FROZEN_UFUZZ_METHOD_SCORE,
                PriorityTiming.POST_VALID_EXECUTION_FUTURE_DECISIONS_ONLY,
                True,
                RandomStreamName.SCHEDULER_TIE_BREAK,
                MethodInformationProfile.STANDARD_SEARCH_SAFE,
            )
            for method in (
                UFUZZ_Q,
                UFUZZ_M,
                UFUZZ,
                UFUZZ_WITHOUT_COVERAGE,
                UFUZZ_WITHOUT_NOVELTY,
                UFUZZ_WITHOUT_PARENT_DIVERGENCE,
                *RQ3_OPERATOR_ABLATIONS,
            )
        },
    }
)


def _require_frozen_method(method: MethodSpec) -> MethodSpec:
    if not isinstance(method, MethodSpec):
        raise TypeError("method must be a MethodSpec")
    expected = METHODS_BY_ID.get(method.method_id)
    if expected is None or expected.scientific_key != method.scientific_key:
        raise ValueError("method is outside the frozen RQ1--RQ4 scientific plan")
    return method


def scheduling_contract_for(method: MethodSpec) -> MethodSchedulingContract:
    _require_frozen_method(method)
    try:
        return _METHOD_SCHEDULING_CONTRACTS[method.method_id]
    except KeyError as error:
        raise ValueError("method is outside the frozen RQ1--RQ4 plan") from error


def validate_coverage_priority(value: int) -> int:
    """Coverage-Guided priority is the exact nonnegative integer G_t."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("coverage priority must be an integer")
    if value < 0:
        raise ValueError("coverage priority cannot be negative")
    return value


def _validate_unit_interval_priority(value: float | int, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


def validate_ufuzz_priority(value: float | int) -> float:
    return _validate_unit_interval_priority(value, name="U-Fuzz priority")


def validate_online_judge_priority(value: float | int) -> float:
    """Reject invalid judge output; callers must use model-output resampling."""

    return _validate_unit_interval_priority(value, name="online judge priority")


@dataclass(frozen=True, slots=True)
class InformationBoundaryContract:
    unguided_allowed: frozenset[UnguidedContextField]
    unguided_forbidden: frozenset[ForbiddenSearchInput]
    online_judge_allowed: frozenset[OnlineJudgeEvidenceField]
    online_judge_forbidden: frozenset[ForbiddenSearchInput]
    online_judge_requires_final_response: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "unguided_allowed", frozenset(self.unguided_allowed))
        object.__setattr__(self, "unguided_forbidden", frozenset(self.unguided_forbidden))
        object.__setattr__(
            self, "online_judge_allowed", frozenset(self.online_judge_allowed)
        )
        object.__setattr__(
            self, "online_judge_forbidden", frozenset(self.online_judge_forbidden)
        )
        if self.unguided_allowed != frozenset(UnguidedContextField):
            raise ValueError("Unguided LLM allowed context is frozen")
        if self.unguided_forbidden != frozenset(ForbiddenSearchInput):
            raise ValueError("Unguided LLM forbidden context is frozen")
        if self.online_judge_allowed != frozenset(OnlineJudgeEvidenceField):
            raise ValueError("online judge evidence is frozen")
        if self.online_judge_forbidden != frozenset(ForbiddenSearchInput):
            raise ValueError("online judge forbidden context is frozen")
        if self.online_judge_requires_final_response:
            raise ValueError("the online judge cannot require final response y")


@dataclass(frozen=True, slots=True)
class DuplicateContract:
    successful_parent_opportunity_executes_once: bool
    same_artifact_transient_retry_is_same_attempt: bool
    suppress_equal_artifacts_across_distinct_opportunities: bool
    retrieval_history_uses_existing_contract: bool
    coverage_uses_existing_set_semantics: bool
    cfs_dedup_is_post_search: bool

    def __post_init__(self) -> None:
        if not all(
            (
                self.successful_parent_opportunity_executes_once,
                self.same_artifact_transient_retry_is_same_attempt,
                self.retrieval_history_uses_existing_contract,
                self.coverage_uses_existing_set_semantics,
                self.cfs_dedup_is_post_search,
            )
        ):
            raise ValueError("duplicate boundaries are mandatory")
        if self.suppress_equal_artifacts_across_distinct_opportunities:
            raise ValueError("distinct opportunities cannot be globally deduplicated")


@dataclass(frozen=True, slots=True)
class ExhaustionContract:
    root_requires_no_nonterminal_frontier: bool
    root_requires_complete_certified_enumerations: bool
    incomplete_enumeration_is_scientific_exhaustion: bool
    campaign_requires_all_roots_exhausted: bool
    stop_at_achieved_budget: bool
    record_explicit_shortfall: bool
    fabricate_replacement_probes: bool
    extrapolate_metrics_to_target: bool
    shortfall_target_table_value_complete: bool

    def __post_init__(self) -> None:
        if not all(
            (
                self.root_requires_no_nonterminal_frontier,
                self.root_requires_complete_certified_enumerations,
                self.campaign_requires_all_roots_exhausted,
                self.stop_at_achieved_budget,
                self.record_explicit_shortfall,
            )
        ):
            raise ValueError("scientific exhaustion evidence is required")
        if self.incomplete_enumeration_is_scientific_exhaustion:
            raise ValueError("incomplete enumeration is blocked, not exhausted")
        if any(
            (
                self.fabricate_replacement_probes,
                self.extrapolate_metrics_to_target,
                self.shortfall_target_table_value_complete,
            )
        ):
            raise ValueError("shortfall cannot be hidden or extrapolated")


@dataclass(frozen=True, slots=True)
class RetryContract:
    numeric_caps_are_unbound: bool
    transient_replays_same_artifact: bool
    transient_preserves_event_randomness: bool
    transient_creates_new_scientific_sample: bool
    transient_increments_role_attempt_index: bool
    transient_exhaustion_kind: ExhaustionKind
    model_output_resampling_increments_role_attempt_index: bool
    model_output_resampling_creates_new_scientific_sample: bool
    realization_cap_disposition: OpportunityDisposition
    controller_random_fallback: bool
    judge_default_score_fallback: bool
    judge_failure_refunds_budget: bool
    judge_cap_exhaustion_kind: ExhaustionKind

    def __post_init__(self) -> None:
        if not all(
            (
                self.numeric_caps_are_unbound,
                self.transient_replays_same_artifact,
                self.transient_preserves_event_randomness,
                self.model_output_resampling_increments_role_attempt_index,
                self.model_output_resampling_creates_new_scientific_sample,
            )
        ):
            raise ValueError("retry identity and unbound caps are required")
        if (
            self.transient_creates_new_scientific_sample
            or self.transient_increments_role_attempt_index
        ):
            raise ValueError("transport replay cannot resample a scientific event")
        if self.transient_exhaustion_kind is not (
            ExhaustionKind.INFRASTRUCTURE_BLOCKED_RESUMABLE
        ):
            raise ValueError("transient cap exhaustion is infrastructure-blocked")
        if self.realization_cap_disposition is not (
            OpportunityDisposition.TERMINAL_GENERATION_EXHAUSTED
        ):
            raise ValueError("generation cap exhaustion is a terminal disposition")
        if self.controller_random_fallback or self.judge_default_score_fallback:
            raise ValueError("controller and judge failures have no fallback")
        if self.judge_failure_refunds_budget:
            raise ValueError("online judge failure cannot refund B")
        if self.judge_cap_exhaustion_kind is not (
            ExhaustionKind.MODEL_SERVICE_BLOCKED_RESUMABLE
        ):
            raise ValueError("judge cap exhaustion is model-service-blocked")


@dataclass(frozen=True, slots=True)
class RetrySemantics:
    retry_class: RetryClass
    mode: RetryMode
    same_parent_scientific_event: bool
    replays_exact_event_identity: bool
    same_stochastic_sample: bool
    increments_role_attempt_index: bool

    def __post_init__(self) -> None:
        if not self.same_parent_scientific_event:
            raise ValueError("all retries remain attached to the same parent event")
        if self.mode is RetryMode.REPLAY_SAME_SCIENTIFIC_EVENT:
            if not self.replays_exact_event_identity or not self.same_stochastic_sample:
                raise ValueError("transport retry must replay the same event and sample")
            if self.increments_role_attempt_index:
                raise ValueError("transport retry cannot increment a scientific attempt")
        else:
            if (
                self.replays_exact_event_identity
                or self.same_stochastic_sample
                or not self.increments_role_attempt_index
            ):
                raise ValueError("scientific resampling requires a new indexed sample")


_RETRY_SEMANTICS = MappingProxyType(
    {
        RetryClass.TRANSIENT_TRANSPORT_OR_SERVICE: RetrySemantics(
            RetryClass.TRANSIENT_TRANSPORT_OR_SERVICE,
            RetryMode.REPLAY_SAME_SCIENTIFIC_EVENT,
            True,
            True,
            True,
            False,
        ),
        **{
            retry_class: RetrySemantics(
                retry_class,
                RetryMode.RESAMPLE_WITH_INCREMENTED_ROLE_ATTEMPT,
                True,
                False,
                False,
                True,
            )
            for retry_class in (
                RetryClass.MUTATION_REALIZATION_OUTPUT_RESAMPLING,
                RetryClass.UNGUIDED_CONTROLLER_OUTPUT_RESAMPLING,
                RetryClass.ONLINE_JUDGE_OUTPUT_RESAMPLING,
            )
        },
    }
)


def retry_semantics_for(retry_class: RetryClass) -> RetrySemantics:
    if not isinstance(retry_class, RetryClass):
        raise TypeError("retry_class must be a RetryClass")
    return _RETRY_SEMANTICS[retry_class]


@dataclass(frozen=True, slots=True)
class RandomnessContract:
    streams: frozenset[RandomStreamName]
    no_shared_mutable_global_rng: bool
    event_keyed: bool
    stable_event_identity_required: bool
    every_stochastic_scientific_role_covered: bool
    post_b_stochastic_roles_covered: bool
    excluded_runtime_factors: frozenset[str]
    derivation_version: None = None
    master_seed: None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "streams", frozenset(self.streams))
        object.__setattr__(
            self, "excluded_runtime_factors", frozenset(self.excluded_runtime_factors)
        )
        if self.streams != frozenset(RandomStreamName):
            raise ValueError("all named scientific random streams are required")
        if not all(
            (
                self.no_shared_mutable_global_rng,
                self.event_keyed,
                self.stable_event_identity_required,
                self.every_stochastic_scientific_role_covered,
                self.post_b_stochastic_roles_covered,
            )
        ):
            raise ValueError("randomness must be event-keyed without global RNG")
        required_exclusions = {
            "python_hash",
            "container_iteration_order",
            "current_list_insertion_timing",
            "worker_assignment",
            "worker_id",
            "worker_completion_order",
            "request_arrival_order",
            "gpu_batch_order",
            "batch_id",
            "gpu_replica_choice",
            "device_id",
            "wall_clock",
            "backend_uuid",
            "runtime_physical_uuid",
        }
        if not required_exclusions.issubset(self.excluded_runtime_factors):
            raise ValueError("runtime ordering factors must not affect randomness")
        if self.derivation_version is not None or self.master_seed is not None:
            raise ValueError("seed values and derivation remain unbound")


@dataclass(frozen=True, slots=True)
class PostBBoundaryContract:
    response_generation_is_post_b: bool
    response_is_search_inert: bool
    response_may_be_deferred: bool
    response_failure_refunds_budget: bool
    final_evaluator_is_post_response_and_post_b: bool
    final_evaluator_is_search_inert: bool
    final_evaluator_may_be_deferred: bool
    checkpoint_filters_by_execution_index: bool
    campaign_local_cfs_dedup_after_filtering: bool
    fixed_event_outputs_independent_of_batch_worker_device: bool
    non_equivalent_serving_becomes_scientific_or_ineligible: bool

    def __post_init__(self) -> None:
        if not all(
            (
                self.response_generation_is_post_b,
                self.response_is_search_inert,
                self.response_may_be_deferred,
                self.final_evaluator_is_post_response_and_post_b,
                self.final_evaluator_is_search_inert,
                self.final_evaluator_may_be_deferred,
                self.checkpoint_filters_by_execution_index,
                self.campaign_local_cfs_dedup_after_filtering,
                self.fixed_event_outputs_independent_of_batch_worker_device,
                self.non_equivalent_serving_becomes_scientific_or_ineligible,
            )
        ):
            raise ValueError("post-B response/evaluator boundary is fixed")
        if self.response_failure_refunds_budget:
            raise ValueError("response failure cannot refund B")


@dataclass(frozen=True, slots=True)
class CheckpointContract:
    checkpoints: tuple[int, ...]
    observational_only: bool
    mutable_search_state_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoints", tuple(self.checkpoints))
        if self.checkpoints != RQ2_CHECKPOINTS:
            raise ValueError("checkpoint positions must match the evaluation contract")
        if not self.observational_only:
            raise ValueError("checkpoints must be observational")
        if self.mutable_search_state_fields:
            raise ValueError("checkpoint recording cannot mutate search state")


@dataclass(frozen=True, slots=True)
class RetrievalDepthContract:
    minimum: int
    globally_shared_across_rq1_rq4: bool
    controls: frozenset[RetrievalDepthUse]
    production_binding: ProductionBindingRequirement
    configured_value: None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "controls", frozenset(self.controls))
        if self.minimum != 2:
            raise ValueError("primary retrieval depth minimum is two")
        if not self.globally_shared_across_rq1_rq4:
            raise ValueError("one k must be shared across RQ1--RQ4")
        if self.controls != frozenset(RetrievalDepthUse):
            raise ValueError("k must control every frozen retrieval-depth use")
        if self.production_binding is not ProductionBindingRequirement.RETRIEVAL_DEPTH:
            raise ValueError("k must enter production scientific identity")
        if self.configured_value is not None:
            raise ValueError("numeric k remains unbound in this phase")


def validate_configured_retrieval_depth(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("retrieval depth must be an integer")
    if value < 2:
        raise ValueError("retrieval depth must be at least two")
    return value


BASE_PRODUCTION_BINDINGS = frozenset(
    {
        ProductionBindingRequirement.BENCHMARK_VERSION_PREPROCESSING,
        ProductionBindingRequirement.BACKEND_RUNTIME_PROFILE,
        ProductionBindingRequirement.SCHEDULER_POLICY_VERSION,
        ProductionBindingRequirement.PRNG_DERIVATION_AND_REPETITION_SEED,
        ProductionBindingRequirement.RETRIEVAL_DEPTH,
        ProductionBindingRequirement.MUTATION_REALIZATION,
        ProductionBindingRequirement.MUTATION_INDEX_OPPORTUNITY_ENUMERATION,
        ProductionBindingRequirement.MUTATION_GENERATION_ATTEMPT_CAP,
        ProductionBindingRequirement.TRANSIENT_INFRASTRUCTURE_REPLAY_CAP,
        ProductionBindingRequirement.SEMANTIC_VALIDATION_STRUCTURAL_EXTRACTION,
        ProductionBindingRequirement.RESPONSE_GENERATION,
        ProductionBindingRequirement.FINAL_EVALUATOR_CFS,
    }
)


def required_production_bindings(
    method: MethodSpec,
) -> frozenset[ProductionBindingRequirement]:
    _require_frozen_method(method)
    result = set(BASE_PRODUCTION_BINDINGS)
    if method.method_id == UNGUIDED_LLM.method_id:
        result.add(ProductionBindingRequirement.UNGUIDED_CONTROLLER)
        result.add(
            ProductionBindingRequirement.UNGUIDED_CONTROLLER_OUTPUT_RESAMPLING_CAP
        )
    if method.method_id == LLM_AS_JUDGE.method_id:
        result.add(ProductionBindingRequirement.ONLINE_JUDGE)
        result.add(
            ProductionBindingRequirement.ONLINE_JUDGE_OUTPUT_RESAMPLING_CAP
        )
    return frozenset(result)


@dataclass(frozen=True, slots=True)
class SchedulerScientificPolicy:
    version: SchedulerPolicyVersion
    root_round: RootRoundContract
    opportunity_enumeration: OpportunityEnumerationContract
    realization: RealizationContract
    retention: RetentionContract
    relation_balancing: RelationBalancingRule
    duplicates: DuplicateContract
    exhaustion: ExhaustionContract
    retries: RetryContract
    randomness: RandomnessContract
    information_boundary: InformationBoundaryContract
    post_b: PostBBoundaryContract
    checkpoints: CheckpointContract
    retrieval_depth: RetrievalDepthContract

    def __post_init__(self) -> None:
        if self.version is not (
            SchedulerPolicyVersion.ROOT_CYCLIC_RETAIN_ALL_RELATION_MASK_V2
        ):
            raise ValueError("unknown scheduler scientific policy version")
        if self.relation_balancing is not RelationBalancingRule.NONE:
            raise ValueError("the frozen scheduler has no relation quota")


SCHEDULER_POLICY = SchedulerScientificPolicy(
    SchedulerPolicyVersion.ROOT_CYCLIC_RETAIN_ALL_RELATION_MASK_V2,
    RootRoundContract(
        RootSchedulingRule.CYCLIC_PERSISTENT_REPETITION_PERMUTATION,
        RandomStreamName.ROOT_ORDER,
        1,
        True,
        True,
        True,
        True,
        1,
        False,
    ),
    OpportunityEnumerationContract(
        OpportunityEnumerationPolicyVersion.COMPLETE_AT_SEED_ADMISSION_RELATION_MASK_V2,
        True,
        True,
        True,
        True,
        OPPORTUNITY_EXISTENCE_INPUTS,
        OPPORTUNITY_EXISTENCE_FORBIDDEN_FACTORS,
        ProductionBindingRequirement.MUTATION_INDEX_OPPORTUNITY_ENUMERATION,
    ),
    RealizationContract(
        RealizationYieldRule.ONE_EXACT_ARTIFACT_PER_ATTEMPT,
        False,
        True,
        True,
        True,
    ),
    RetentionContract(
        SeedRetentionRule.RETAIN_ALL_VALID_REMATERIALIZABLE_CHILDREN,
        ParentAvailabilityRule.RETAIN_UNTIL_COMPLETE_OPPORTUNITY_SET_TERMINAL,
        ChildEligibilityRule.NEXT_ROUND,
        DepthRule.NO_SCIENTIFIC_MAXIMUM,
    ),
    RelationBalancingRule.NONE,
    DuplicateContract(True, True, False, True, True, True),
    ExhaustionContract(True, True, False, True, True, True, False, False, False),
    RetryContract(
        True,
        True,
        True,
        False,
        False,
        ExhaustionKind.INFRASTRUCTURE_BLOCKED_RESUMABLE,
        True,
        True,
        OpportunityDisposition.TERMINAL_GENERATION_EXHAUSTED,
        False,
        False,
        False,
        ExhaustionKind.MODEL_SERVICE_BLOCKED_RESUMABLE,
    ),
    RandomnessContract(
        frozenset(RandomStreamName),
        True,
        True,
        True,
        True,
        True,
        frozenset(
            {
                "python_hash",
                "container_iteration_order",
                "current_list_insertion_timing",
                "worker_assignment",
                "worker_id",
                "worker_completion_order",
                "request_arrival_order",
                "gpu_batch_order",
                "batch_id",
                "gpu_replica_choice",
                "device_id",
                "wall_clock",
                "backend_uuid",
                "runtime_physical_uuid",
            }
        ),
    ),
    InformationBoundaryContract(
        frozenset(UnguidedContextField),
        frozenset(ForbiddenSearchInput),
        frozenset(OnlineJudgeEvidenceField),
        frozenset(ForbiddenSearchInput),
        False,
    ),
    PostBBoundaryContract(
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ),
    CheckpointContract(RQ2_CHECKPOINTS, True, ()),
    RetrievalDepthContract(
        2,
        True,
        frozenset(RetrievalDepthUse),
        ProductionBindingRequirement.RETRIEVAL_DEPTH,
    ),
)


ENGINEERING_ONLY_SETTINGS = frozenset(EngineeringOnlySetting)
