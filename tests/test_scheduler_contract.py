from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
import math
import unittest

from ufuzz.coverage import CoverageEntryId
from ufuzz.evaluation_contract import (
    ALL_METHODS,
    COVERAGE_GUIDED,
    LLM_AS_JUDGE,
    RANDOM_MUTATION,
    RQ2_CHECKPOINTS,
    UFUZZ,
    UFUZZ_M,
    UFUZZ_Q,
    UFUZZ_WITHOUT_COVERAGE,
    UFUZZ_WITHOUT_NOVELTY,
    UFUZZ_WITHOUT_PARENT_DIVERGENCE,
    UNGUIDED_LLM,
    MutationSpace,
)
from ufuzz.retrieval_feedback import MutationRelation
from ufuzz.scheduler_contract import (
    ATOMIC_VALID_EXECUTION_ORDER,
    BASE_PRODUCTION_BINDINGS,
    BOOTSTRAP_PRIORITY,
    ENGINEERING_ONLY_SETTINGS,
    SCHEDULER_POLICY,
    ChildEligibilityRule,
    DepthRule,
    EngineeringOnlySetting,
    ExhaustionKind,
    ForbiddenSearchInput,
    FrontierEligibility,
    FrontierItem,
    FrontierSelectionRule,
    MethodInformationProfile,
    MutationRealizationEvent,
    OnlineJudgeEvidenceField,
    OPPORTUNITY_EXISTENCE_FORBIDDEN_FACTORS,
    OPPORTUNITY_EXISTENCE_INPUTS,
    OpportunityEnumerationPolicyVersion,
    OpportunityEnumerationStatus,
    OpportunityDisposition,
    PrioritySource,
    ProductionBindingRequirement,
    RandomStreamName,
    RetryClass,
    RetryMode,
    RootOrderEvent,
    RootSchedulingRule,
    RootVisitResult,
    RootVisitTransientRetry,
    SchedulerScientificPolicy,
    ScientificRandomEvent,
    SeedAdmissionKind,
    SeedOpportunityEnumeration,
    SeedRetentionRule,
    UnguidedContextField,
    ValidExecutionStage,
    required_production_bindings,
    retry_semantics_for,
    scheduling_contract_for,
    validate_coverage_priority,
    validate_configured_retrieval_depth,
    validate_online_judge_priority,
    validate_opportunity_disposition_transition,
    validate_ufuzz_priority,
)
from ufuzz.state_contract import (
    DescendantStateCertificate,
    ExecutableQueryArtifact,
    LiveLineageState,
    LogicalSeed,
    MutationOpportunity,
)


class SchedulerContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.campaign = "campaign"
        self.root = "root"
        self.coverage_id = CoverageEntryId(self.campaign, self.root, "e1")
        descendant = DescendantStateCertificate(
            "descendant",
            "synthetic",
            "config",
            self.campaign,
            self.root,
            frozenset({self.coverage_id}),
            (LiveLineageState((self.coverage_id,), "observable", "provenance"),),
            frozenset(),
            "whole-observable",
            "transition-state",
            (),
        )
        self.parent = LogicalSeed(
            "seed-parent",
            self.campaign,
            "synthetic",
            self.root,
            "config",
            ExecutableQueryArtifact.create(
                artifact_id="query",
                executable_text="Exact query?",
                metadata={"query_id": "q1"},
            ),
            (),
            descendant,
            ((self.coverage_id,),),
        )

    def opportunity(self, opportunity_id: str = "opportunity") -> MutationOpportunity:
        return MutationOpportunity.create(
            opportunity_id=opportunity_id,
            campaign_id=self.campaign,
            root_checkpoint_id=self.root,
            parent_seed_id=self.parent.seed_id,
            relation=MutationRelation.MEANING_PRESERVING_QUERY,
            canonical_target={"slot": "wording"},
            applicability_evidence={"certified": True},
            generation_constraints={"preserve_intent": True},
        )

    def test_contract_is_immutable_and_hash_safe(self) -> None:
        hash(SCHEDULER_POLICY)
        with self.assertRaises(FrozenInstanceError):
            SCHEDULER_POLICY.relation_balancing = "changed"  # type: ignore[misc]

    def test_root_round_is_cyclic_method_neutral_and_one_valid_per_root(self) -> None:
        root = SCHEDULER_POLICY.root_round
        self.assertIs(
            root.scheduling_rule,
            RootSchedulingRule.CYCLIC_PERSISTENT_REPETITION_PERMUTATION,
        )
        self.assertEqual(root.max_valid_executions_per_active_root, 1)
        self.assertTrue(root.method_neutral)
        self.assertTrue(root.stop_immediately_at_budget)
        self.assertTrue(root.skip_temporarily_ineligible_roots)
        self.assertTrue(root.remove_scientifically_exhausted_roots)
        self.assertEqual(root.frontier_decisions_per_root_visit, 1)
        self.assertFalse(root.second_frontier_selection_in_same_visit)

    def test_one_root_visit_selects_one_item_and_terminal_success_ends_it(self) -> None:
        item = FrontierItem(self.parent, self.opportunity())
        result = RootVisitResult.terminal(
            item, OpportunityDisposition.TERMINAL_SUCCESS
        )
        self.assertEqual(result.frontier_decisions, 1)
        self.assertEqual(result.valid_executions, 1)
        self.assertTrue(result.visit_complete)
        self.assertTrue(result.advance_to_next_root)
        self.assertFalse(result.may_select_second_frontier_in_round)

    def test_generation_exhaustion_ends_visit_without_second_selection(self) -> None:
        item = FrontierItem(self.parent, self.opportunity())
        result = RootVisitResult.terminal(
            item, OpportunityDisposition.TERMINAL_GENERATION_EXHAUSTED
        )
        self.assertEqual(result.frontier_decisions, 1)
        self.assertEqual(result.valid_executions, 0)
        self.assertTrue(result.visit_complete)
        self.assertTrue(result.advance_to_next_root)
        self.assertFalse(result.may_select_second_frontier_in_round)

    def test_same_artifact_transient_retry_stays_inside_root_visit(self) -> None:
        item = FrontierItem(self.parent, self.opportunity())
        retry = RootVisitTransientRetry(item)
        self.assertEqual(retry.frontier_decisions, 1)
        self.assertTrue(retry.same_realized_artifact)
        self.assertTrue(retry.same_scientific_event)
        self.assertFalse(retry.visit_complete)
        self.assertFalse(retry.advance_to_next_root)

    def test_root_order_identity_excludes_method_and_runtime_order(self) -> None:
        event = RootOrderEvent("locomo-v1", 1, "eligible-roots-v1")
        self.assertEqual(event.stream, RandomStreamName.ROOT_ORDER)
        self.assertNotIn("method", {field.name for field in fields(event)})
        self.assertNotIn("worker", {field.name for field in fields(event)})

    def test_realization_event_is_stable_and_method_neutral(self) -> None:
        first = MutationRealizationEvent(
            "locomo-v1", self.root, self.parent.seed_id, "opportunity", 1, 2
        )
        second = MutationRealizationEvent(
            "locomo-v1", self.root, self.parent.seed_id, "opportunity", 1, 2
        )
        self.assertEqual(first, second)
        event_fields = {field.name for field in fields(first)}
        self.assertNotIn("method", event_fields)
        self.assertNotIn("worker", event_fields)
        self.assertNotIn("batch", event_fields)

    def test_frontier_uses_persistent_parent_and_opportunity(self) -> None:
        opportunity = self.opportunity()
        item = FrontierItem(self.parent, opportunity)
        self.assertIs(item.parent, self.parent)
        self.assertIs(item.opportunity, opportunity)
        self.assertEqual(
            item.scientific_identity,
            (self.campaign, self.root, self.parent.seed_id, opportunity.opportunity_id),
        )
        self.assertNotIn("backend_entry_id", {field.name for field in fields(item)})

    def test_frontier_eligibility_requires_retention_space_and_nonterminal_state(self) -> None:
        item = FrontierItem(self.parent, self.opportunity())
        self.assertTrue(FrontierEligibility(item, True, MutationSpace.FULL).eligible)
        self.assertFalse(FrontierEligibility(item, False, MutationSpace.FULL).eligible)
        self.assertFalse(
            FrontierEligibility(item, True, MutationSpace.MEMORY_ONLY).eligible
        )
        self.assertFalse(
            FrontierEligibility(
                item,
                True,
                MutationSpace.FULL,
                OpportunityDisposition.TERMINAL_SUCCESS,
            ).eligible
        )

    def test_initial_seed_opportunities_are_complete_immutable_and_round_one_eligible(self) -> None:
        enumeration = SeedOpportunityEnumeration(
            self.parent,
            MutationSpace.FULL,
            SeedAdmissionKind.INITIAL,
            0,
            OpportunityEnumerationStatus.CERTIFIED_COMPLETE,
            frozenset({self.opportunity()}),
        )
        self.assertTrue(enumeration.certified_complete)
        self.assertFalse(enumeration.locally_exhausted)
        self.assertEqual(enumeration.first_eligible_round, 1)
        self.assertTrue(enumeration.eligible_in_round(1))
        with self.assertRaises(FrozenInstanceError):
            enumeration.opportunities = frozenset()  # type: ignore[misc]
        with self.assertRaises(AttributeError):
            enumeration.opportunities.add(self.opportunity("later"))  # type: ignore[attr-defined]

    def test_child_is_enumerated_at_admission_but_waits_until_next_round(self) -> None:
        enumeration = SeedOpportunityEnumeration(
            self.parent,
            MutationSpace.FULL,
            SeedAdmissionKind.VALID_EXECUTED_CHILD,
            4,
            OpportunityEnumerationStatus.CERTIFIED_COMPLETE,
            frozenset({self.opportunity()}),
        )
        self.assertFalse(enumeration.eligible_in_round(4))
        self.assertTrue(enumeration.eligible_in_round(5))

    def test_empty_complete_enumeration_exhausts_locally_but_incomplete_does_not(self) -> None:
        complete = SeedOpportunityEnumeration(
            self.parent,
            MutationSpace.FULL,
            SeedAdmissionKind.INITIAL,
            0,
            OpportunityEnumerationStatus.CERTIFIED_COMPLETE,
            frozenset(),
        )
        blocked = SeedOpportunityEnumeration(
            self.parent,
            MutationSpace.FULL,
            SeedAdmissionKind.INITIAL,
            0,
            OpportunityEnumerationStatus.BLOCKED_CAPABILITY_FAILURE,
            frozenset(),
        )
        self.assertTrue(complete.locally_exhausted)
        self.assertTrue(complete.supports_scientific_exhaustion)
        self.assertFalse(blocked.locally_exhausted)
        self.assertFalse(blocked.supports_scientific_exhaustion)
        infrastructure_blocked = replace(
            blocked,
            status=OpportunityEnumerationStatus.BLOCKED_TRANSIENT_INFRASTRUCTURE_FAILURE,
        )
        self.assertFalse(infrastructure_blocked.supports_scientific_exhaustion)
        with self.assertRaisesRegex(ValueError, "partial set"):
            SeedOpportunityEnumeration(
                self.parent,
                MutationSpace.FULL,
                SeedAdmissionKind.INITIAL,
                0,
                OpportunityEnumerationStatus.BLOCKED_CAPABILITY_FAILURE,
                frozenset({self.opportunity()}),
            )

    def test_opportunity_existence_is_fixed_without_feedback_or_runtime_order(self) -> None:
        contract = SCHEDULER_POLICY.opportunity_enumeration
        self.assertIs(
            contract.policy_version,
            OpportunityEnumerationPolicyVersion.COMPLETE_AT_SEED_ADMISSION_V1,
        )
        self.assertEqual(contract.existence_inputs, OPPORTUNITY_EXISTENCE_INPUTS)
        self.assertEqual(
            contract.forbidden_factors,
            OPPORTUNITY_EXISTENCE_FORBIDDEN_FACTORS,
        )
        self.assertTrue(contract.immutable_after_admission)
        self.assertTrue(contract.complete_at_seed_admission)
        self.assertIs(
            contract.implementation_binding,
            ProductionBindingRequirement.MUTATION_INDEX_OPPORTUNITY_ENUMERATION,
        )

    def test_successful_parent_opportunity_cannot_execute_twice(self) -> None:
        self.assertIs(
            validate_opportunity_disposition_transition(
                OpportunityDisposition.ELIGIBLE,
                OpportunityDisposition.TERMINAL_SUCCESS,
            ),
            OpportunityDisposition.TERMINAL_SUCCESS,
        )
        with self.assertRaisesRegex(ValueError, "cannot be revisited"):
            validate_opportunity_disposition_transition(
                OpportunityDisposition.TERMINAL_SUCCESS,
                OpportunityDisposition.TERMINAL_SUCCESS,
            )

    def test_generation_exhaustion_is_terminal_and_attempt_cap_unbound(self) -> None:
        self.assertTrue(OpportunityDisposition.TERMINAL_GENERATION_EXHAUSTED.terminal)
        self.assertIsNone(SCHEDULER_POLICY.realization.generation_attempt_cap)
        self.assertIs(
            SCHEDULER_POLICY.retries.realization_cap_disposition,
            OpportunityDisposition.TERMINAL_GENERATION_EXHAUSTED,
        )

    def test_one_artifact_per_attempt_and_no_candidate_pool(self) -> None:
        realization = SCHEDULER_POLICY.realization
        self.assertEqual(
            realization.yield_rule.value,
            "one_exact_artifact_per_attempt",
        )
        self.assertFalse(realization.generic_preexecution_candidate_pool)
        self.assertTrue(realization.shared_interface_and_validator)
        self.assertTrue(realization.arrival_order_independent)

    def test_all_valid_children_are_retained_without_pruning(self) -> None:
        retention = SCHEDULER_POLICY.retention
        self.assertIs(
            retention.seed_rule,
            SeedRetentionRule.RETAIN_ALL_VALID_REMATERIALIZABLE_CHILDREN,
        )
        self.assertIsNone(retention.score_threshold)
        self.assertIsNone(retention.logical_queue_capacity)
        self.assertIsNone(retention.eviction_rule)
        self.assertIsNone(retention.aging_rule)

    def test_children_start_next_round_and_chaining_has_no_max_depth(self) -> None:
        retention = SCHEDULER_POLICY.retention
        self.assertIs(retention.child_eligibility, ChildEligibilityRule.NEXT_ROUND)
        self.assertIs(retention.depth_rule, DepthRule.NO_SCIENTIFIC_MAXIMUM)
        self.assertIsNone(retention.maximum_scientific_depth)

    def test_distinct_opportunities_are_not_artifact_deduplicated(self) -> None:
        first = FrontierItem(self.parent, self.opportunity("opportunity-1"))
        second = FrontierItem(self.parent, self.opportunity("opportunity-2"))
        self.assertNotEqual(first, second)
        self.assertFalse(
            SCHEDULER_POLICY.duplicates.suppress_equal_artifacts_across_distinct_opportunities
        )
        self.assertTrue(SCHEDULER_POLICY.duplicates.cfs_dedup_is_post_search)

    def test_random_is_uniform_over_frontier_items(self) -> None:
        contract = scheduling_contract_for(RANDOM_MUTATION)
        self.assertIs(
            contract.selection_rule,
            FrontierSelectionRule.UNIFORM_ELIGIBLE_FRONTIER_ITEMS,
        )
        self.assertIs(contract.priority_source, PrioritySource.NONE)
        self.assertIs(
            contract.selection_stream,
            RandomStreamName.RANDOM_BASELINE_CHOICE,
        )

    def test_guided_priorities_are_post_execution_parent_priorities(self) -> None:
        coverage = scheduling_contract_for(COVERAGE_GUIDED)
        ufuzz = scheduling_contract_for(UFUZZ)
        ablation = scheduling_contract_for(UFUZZ_WITHOUT_NOVELTY)
        self.assertIs(coverage.priority_source, PrioritySource.COVERAGE_GAIN)
        self.assertIs(ufuzz.priority_source, PrioritySource.FROZEN_UFUZZ_METHOD_SCORE)
        self.assertIs(
            ablation.priority_source,
            PrioritySource.FROZEN_UFUZZ_METHOD_SCORE,
        )
        self.assertTrue(coverage.consumes_coverage_novelty_divergence_feedback)
        self.assertTrue(ufuzz.consumes_coverage_novelty_divergence_feedback)
        self.assertIs(
            scheduling_contract_for(UFUZZ_Q).selection_rule,
            FrontierSelectionRule.MAXIMAL_PARENT_PRIORITY,
        )
        self.assertEqual(ufuzz.initial_seed_priority, BOOTSTRAP_PRIORITY)
        self.assertFalse(ufuzz.child_priority_overwrites_parent)

    def test_bootstrap_and_guided_priority_numeric_order_is_exact(self) -> None:
        self.assertEqual(BOOTSTRAP_PRIORITY, 0.0)
        self.assertEqual(validate_ufuzz_priority(0.0), BOOTSTRAP_PRIORITY)
        self.assertGreater(validate_ufuzz_priority(0.25), BOOTSTRAP_PRIORITY)
        self.assertEqual(validate_coverage_priority(0), BOOTSTRAP_PRIORITY)
        self.assertGreater(validate_coverage_priority(1), BOOTSTRAP_PRIORITY)
        self.assertEqual(validate_online_judge_priority(0), BOOTSTRAP_PRIORITY)
        self.assertEqual(validate_online_judge_priority(1), 1.0)
        for invalid in (-0.01, 1.01, math.nan, math.inf, -math.inf):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    validate_online_judge_priority(invalid)
        with self.assertRaises(TypeError):
            validate_online_judge_priority(True)
        self.assertNotIn(
            "sentinel",
            {field.name for field in fields(scheduling_contract_for(UFUZZ))},
        )

    def test_unguided_llm_has_exact_information_boundary(self) -> None:
        boundary = SCHEDULER_POLICY.information_boundary
        self.assertEqual(boundary.unguided_allowed, frozenset(UnguidedContextField))
        self.assertEqual(boundary.unguided_forbidden, frozenset(ForbiddenSearchInput))
        contract = scheduling_contract_for(UNGUIDED_LLM)
        self.assertIs(
            contract.selection_rule,
            FrontierSelectionRule.LLM_CONTROLLER_SELECTS_ONE,
        )
        self.assertFalse(contract.consumes_coverage_novelty_divergence_feedback)

    def test_online_judge_is_post_execution_and_excludes_gold_and_response(self) -> None:
        boundary = SCHEDULER_POLICY.information_boundary
        self.assertEqual(
            boundary.online_judge_allowed,
            frozenset(OnlineJudgeEvidenceField),
        )
        self.assertEqual(
            boundary.online_judge_forbidden,
            frozenset(ForbiddenSearchInput),
        )
        self.assertFalse(boundary.online_judge_requires_final_response)
        judge = scheduling_contract_for(LLM_AS_JUDGE)
        self.assertIs(judge.priority_source, PrioritySource.ONLINE_JUDGE_SCORE)
        self.assertTrue(judge.online_judge_is_post_execution)
        self.assertTrue(judge.priority_required_before_future_selection)
        self.assertFalse(judge.consumes_coverage_novelty_divergence_feedback)
        self.assertIs(
            judge.information_profile,
            MethodInformationProfile.ONLINE_JUDGE_POST_EXECUTION_RESTRICTED,
        )

    def test_no_relation_balancing_quota_exists(self) -> None:
        self.assertEqual(SCHEDULER_POLICY.relation_balancing.value, "none")

    def test_scientific_exhaustion_records_shortfall(self) -> None:
        exhaustion = SCHEDULER_POLICY.exhaustion
        self.assertTrue(exhaustion.root_requires_complete_certified_enumerations)
        self.assertFalse(exhaustion.incomplete_enumeration_is_scientific_exhaustion)
        self.assertTrue(exhaustion.record_explicit_shortfall)
        self.assertTrue(exhaustion.stop_at_achieved_budget)
        self.assertFalse(exhaustion.fabricate_replacement_probes)
        self.assertFalse(exhaustion.extrapolate_metrics_to_target)
        self.assertFalse(exhaustion.shortfall_target_table_value_complete)

    def test_transient_failure_does_not_create_scientific_exhaustion(self) -> None:
        retry = SCHEDULER_POLICY.retries
        self.assertTrue(retry.transient_replays_same_artifact)
        self.assertTrue(retry.transient_preserves_event_randomness)
        self.assertFalse(retry.transient_creates_new_scientific_sample)
        self.assertFalse(retry.transient_increments_role_attempt_index)
        self.assertIs(
            retry.transient_exhaustion_kind,
            ExhaustionKind.INFRASTRUCTURE_BLOCKED_RESUMABLE,
        )

    def test_transport_replay_and_scientific_resampling_are_distinct(self) -> None:
        transport = retry_semantics_for(
            RetryClass.TRANSIENT_TRANSPORT_OR_SERVICE
        )
        self.assertIs(transport.mode, RetryMode.REPLAY_SAME_SCIENTIFIC_EVENT)
        self.assertTrue(transport.replays_exact_event_identity)
        self.assertTrue(transport.same_stochastic_sample)
        self.assertFalse(transport.increments_role_attempt_index)
        for retry_class in (
            RetryClass.MUTATION_REALIZATION_OUTPUT_RESAMPLING,
            RetryClass.UNGUIDED_CONTROLLER_OUTPUT_RESAMPLING,
            RetryClass.ONLINE_JUDGE_OUTPUT_RESAMPLING,
        ):
            with self.subTest(retry_class=retry_class):
                resampling = retry_semantics_for(retry_class)
                self.assertIs(
                    resampling.mode,
                    RetryMode.RESAMPLE_WITH_INCREMENTED_ROLE_ATTEMPT,
                )
                self.assertFalse(resampling.replays_exact_event_identity)
                self.assertFalse(resampling.same_stochastic_sample)
                self.assertTrue(resampling.increments_role_attempt_index)

    def test_controller_and_judge_failures_have_no_fallback(self) -> None:
        retry = SCHEDULER_POLICY.retries
        self.assertFalse(retry.controller_random_fallback)
        self.assertFalse(retry.judge_default_score_fallback)
        self.assertFalse(retry.judge_failure_refunds_budget)
        self.assertIs(
            retry.judge_cap_exhaustion_kind,
            ExhaustionKind.MODEL_SERVICE_BLOCKED_RESUMABLE,
        )

    def test_named_event_keyed_randomness_excludes_runtime_order(self) -> None:
        randomness = SCHEDULER_POLICY.randomness
        self.assertEqual(randomness.streams, frozenset(RandomStreamName))
        self.assertTrue(randomness.no_shared_mutable_global_rng)
        self.assertTrue(randomness.event_keyed)
        self.assertTrue(randomness.every_stochastic_scientific_role_covered)
        self.assertTrue(randomness.post_b_stochastic_roles_covered)
        self.assertIsNone(randomness.master_seed)
        self.assertIsNone(randomness.derivation_version)
        for item in (
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
        ):
            self.assertIn(item, randomness.excluded_runtime_factors)

    def test_all_stochastic_roles_have_runtime_independent_event_identity(self) -> None:
        required = {
            RandomStreamName.ROOT_ORDER,
            RandomStreamName.RANDOM_BASELINE_CHOICE,
            RandomStreamName.SCHEDULER_TIE_BREAK,
            RandomStreamName.MUTATION_REALIZATION,
            RandomStreamName.UNGUIDED_CONTROLLER_SAMPLING,
            RandomStreamName.ONLINE_JUDGE_SAMPLING,
            RandomStreamName.SEMANTIC_VALIDATION_SAMPLING,
            RandomStreamName.RESPONSE_GENERATION_SAMPLING,
            RandomStreamName.FINAL_EVALUATOR_CFS_SAMPLING,
        }
        self.assertEqual(set(RandomStreamName), required)
        event = ScientificRandomEvent(
            RandomStreamName.FINAL_EVALUATOR_CFS_SAMPLING,
            "campaign:execution:17",
            2,
        )
        event_fields = {field.name for field in fields(event)}
        self.assertEqual(
            event_fields,
            {"stream", "stable_scientific_event_id", "role_attempt_index"},
        )
        self.assertTrue(
            event_fields.isdisjoint(
                {
                    "worker_id",
                    "batch_id",
                    "device_id",
                    "backend_uuid",
                    "runtime_physical_uuid",
                }
            )
        )

    def test_response_and_final_evaluator_are_post_b_and_search_inert(self) -> None:
        boundary = SCHEDULER_POLICY.post_b
        self.assertTrue(boundary.response_generation_is_post_b)
        self.assertTrue(boundary.response_is_search_inert)
        self.assertTrue(boundary.response_may_be_deferred)
        self.assertFalse(boundary.response_failure_refunds_budget)
        self.assertTrue(boundary.final_evaluator_is_post_response_and_post_b)
        self.assertTrue(boundary.final_evaluator_is_search_inert)
        self.assertTrue(boundary.final_evaluator_may_be_deferred)
        self.assertTrue(
            boundary.fixed_event_outputs_independent_of_batch_worker_device
        )
        self.assertTrue(
            boundary.non_equivalent_serving_becomes_scientific_or_ineligible
        )

    def test_atomic_valid_execution_order_is_exact(self) -> None:
        self.assertEqual(
            ATOMIC_VALID_EXECUTION_ORDER,
            (
                ValidExecutionStage.CONSTRUCT_RESOLVED_OBSERVATION,
                ValidExecutionStage.CROSS_VALID_EXECUTION_BOUNDARY,
                ValidExecutionStage.INCREMENT_VALID_EXECUTION_INDEX,
                ValidExecutionStage.ADOPT_COVERAGE_UPDATE_STATE,
                ValidExecutionStage.COMPUTE_POST_EXECUTION_FEEDBACK,
                ValidExecutionStage.INSERT_ROOT_HISTORY_AFTER_SCORING,
                ValidExecutionStage.RETAIN_VALID_CHILD,
                ValidExecutionStage.ASSIGN_FUTURE_PRIORITY,
                ValidExecutionStage.PERSIST_ORDERED_WITNESS,
                ValidExecutionStage.POST_B_RESPONSE_AND_FINAL_EVALUATION,
            ),
        )

    def test_checkpoints_are_observational_only(self) -> None:
        checkpoints = SCHEDULER_POLICY.checkpoints
        self.assertEqual(checkpoints.checkpoints, RQ2_CHECKPOINTS)
        self.assertTrue(checkpoints.observational_only)
        self.assertEqual(checkpoints.mutable_search_state_fields, ())

    def test_retrieval_depth_is_unbound_global_scientific_configuration(self) -> None:
        depth = SCHEDULER_POLICY.retrieval_depth
        self.assertEqual(depth.minimum, 2)
        self.assertTrue(depth.globally_shared_across_rq1_rq3)
        self.assertIsNone(depth.configured_value)
        self.assertIs(
            depth.production_binding,
            ProductionBindingRequirement.RETRIEVAL_DEPTH,
        )
        self.assertEqual(validate_configured_retrieval_depth(10), 10)
        with self.assertRaises(ValueError):
            validate_configured_retrieval_depth(1)

    def test_production_binding_dimensions_and_method_specific_additions(self) -> None:
        self.assertEqual(required_production_bindings(UFUZZ), BASE_PRODUCTION_BINDINGS)
        self.assertEqual(
            required_production_bindings(UNGUIDED_LLM),
            BASE_PRODUCTION_BINDINGS
            | {
                ProductionBindingRequirement.UNGUIDED_CONTROLLER,
                ProductionBindingRequirement.UNGUIDED_CONTROLLER_OUTPUT_RESAMPLING_CAP,
            },
        )
        self.assertEqual(
            required_production_bindings(LLM_AS_JUDGE),
            BASE_PRODUCTION_BINDINGS
            | {
                ProductionBindingRequirement.ONLINE_JUDGE,
                ProductionBindingRequirement.ONLINE_JUDGE_OUTPUT_RESAMPLING_CAP,
            },
        )
        self.assertTrue(
            {
                ProductionBindingRequirement.MUTATION_GENERATION_ATTEMPT_CAP,
                ProductionBindingRequirement.TRANSIENT_INFRASTRUCTURE_REPLAY_CAP,
                ProductionBindingRequirement.MUTATION_INDEX_OPPORTUNITY_ENUMERATION,
            }.issubset(BASE_PRODUCTION_BINDINGS)
        )

    def test_all_ten_frozen_methods_map_exactly_and_altered_specs_fail_closed(self) -> None:
        self.assertEqual(len(ALL_METHODS), 10)
        expected = {
            RANDOM_MUTATION.method_id: (
                MutationSpace.FULL,
                FrontierSelectionRule.UNIFORM_ELIGIBLE_FRONTIER_ITEMS,
                PrioritySource.NONE,
                MethodInformationProfile.STANDARD_SEARCH_SAFE,
            ),
            UNGUIDED_LLM.method_id: (
                MutationSpace.FULL,
                FrontierSelectionRule.LLM_CONTROLLER_SELECTS_ONE,
                PrioritySource.NONE,
                MethodInformationProfile.UNGUIDED_CONTROLLER_RESTRICTED,
            ),
            LLM_AS_JUDGE.method_id: (
                MutationSpace.FULL,
                FrontierSelectionRule.MAXIMAL_PARENT_PRIORITY,
                PrioritySource.ONLINE_JUDGE_SCORE,
                MethodInformationProfile.ONLINE_JUDGE_POST_EXECUTION_RESTRICTED,
            ),
            COVERAGE_GUIDED.method_id: (
                MutationSpace.FULL,
                FrontierSelectionRule.MAXIMAL_PARENT_PRIORITY,
                PrioritySource.COVERAGE_GAIN,
                MethodInformationProfile.STANDARD_SEARCH_SAFE,
            ),
            **{
                method.method_id: (
                    method.mutation_space,
                    FrontierSelectionRule.MAXIMAL_PARENT_PRIORITY,
                    PrioritySource.FROZEN_UFUZZ_METHOD_SCORE,
                    MethodInformationProfile.STANDARD_SEARCH_SAFE,
                )
                for method in (
                    UFUZZ_Q,
                    UFUZZ_M,
                    UFUZZ,
                    UFUZZ_WITHOUT_COVERAGE,
                    UFUZZ_WITHOUT_NOVELTY,
                    UFUZZ_WITHOUT_PARENT_DIVERGENCE,
                )
            },
        }
        for method in ALL_METHODS:
            with self.subTest(method=method.method_id):
                contract = scheduling_contract_for(method)
                self.assertEqual(
                    (
                        method.mutation_space,
                        contract.selection_rule,
                        contract.priority_source,
                        contract.information_profile,
                    ),
                    expected[method.method_id],
                )
                required_production_bindings(method)
        with self.assertRaisesRegex(ValueError, "outside the frozen"):
            scheduling_contract_for(
                replace(
                    UFUZZ,
                    feedback_components=UFUZZ_WITHOUT_NOVELTY.feedback_components,
                )
            )

    def test_scheduler_visible_structures_have_no_evaluator_payload_fields(self) -> None:
        forbidden = {
            "gold_answer",
            "gold_answers",
            "reference_vault",
            "final_correctness",
            "final_evaluator_verdict",
            "failure_surface",
            "canonical_cfs",
            "uf_at_b",
        }
        for contract_type in (
            FrontierItem,
            FrontierEligibility,
            SeedOpportunityEnumeration,
            RootVisitResult,
            RootVisitTransientRetry,
            ScientificRandomEvent,
        ):
            with self.subTest(contract_type=contract_type.__name__):
                self.assertTrue(
                    {field.name for field in fields(contract_type)}.isdisjoint(forbidden)
                )

    def test_engineering_settings_are_excluded_from_scientific_bindings(self) -> None:
        self.assertEqual(ENGINEERING_ONLY_SETTINGS, frozenset(EngineeringOnlySetting))
        binding_values = {item.value for item in ProductionBindingRequirement}
        self.assertTrue(
            binding_values.isdisjoint(item.value for item in ENGINEERING_ONLY_SETTINGS)
        )

    def test_policy_contains_contract_only_not_runtime_state(self) -> None:
        self.assertIsInstance(SCHEDULER_POLICY, SchedulerScientificPolicy)
        policy_fields = {field.name for field in fields(SCHEDULER_POLICY)}
        self.assertTrue(
            policy_fields.isdisjoint(
                {"queue", "runner", "backend", "worker", "model", "evaluator"}
            )
        )


if __name__ == "__main__":
    unittest.main()
