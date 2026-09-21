from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import unittest

from ufuzz.evaluation_contract import (
    ALL_METHODS,
    ALL_MUTATION_RELATIONS,
    BACKENDS,
    BENCHMARK_MAX_BUDGET,
    COVERAGE_GUIDED,
    ENGINEERING_TARGET,
    EVALUATION_PLAN,
    FULL_UFUZZ_FEEDBACK,
    MEMORY_MUTATION_RELATIONS,
    NATIVE_MEMORY_LLM_CONDITIONS,
    NEW_RQ3_METHODS,
    QUERY_MUTATION_RELATIONS,
    RANDOM_MUTATION,
    REPETITION_INDICES,
    RQ1_METHODS,
    RQ2_CHECKPOINTS,
    RQ2_METHODS,
    RQ3_BUDGET,
    RQ3_FEEDBACK_ABLATIONS,
    RQ3_METHODS,
    RQ3_OPERATOR_ABLATIONS,
    RQ3_REUSED_METHODS,
    RQ4_BACKENDS,
    RQ4_BUDGET,
    RQ4_METHODS,
    RQ4_METRICS,
    RQ4_INTERPRETATION_EVIDENCE,
    UFUZZ,
    UFUZZ_M,
    UFUZZ_Q,
    UFUZZ_WITHOUT_COVERAGE,
    UFUZZ_WITHOUT_NOVELTY,
    UFUZZ_WITHOUT_PARENT_DIVERGENCE,
    Backend,
    Benchmark,
    CampaignSpec,
    CheckpointMetrics,
    ConfigurationBinding,
    FairnessGroup,
    FeedbackCombinationRule,
    FeedbackComponentMask,
    MethodScientificKey,
    MutationRelationConfiguration,
    MutationSpace,
    NativeMemoryLLMCondition,
    NativeMemoryLLMProvider,
    OnlineSearchSignal,
    ResearchQuestion,
    SelectionFamily,
    aggregate_three_repetitions,
    validate_rq2_prefix_series,
)
from ufuzz.retrieval_feedback import MutationRelation


class EvaluationContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = EVALUATION_PLAN

    def entries(self, rq: ResearchQuestion):
        return getattr(self.plan, rq.value)

    def entry(self, rq, backend, method_id, repetition, checkpoint, condition=None):
        return next(
            item for item in self.entries(rq)
            if item.campaign.backend is backend
            and item.display_method_id == method_id
            and item.campaign.repetition_index == repetition
            and item.checkpoint == checkpoint
            and item.campaign.benchmark is Benchmark.LOCOMO
            and item.campaign.native_memory_llm_condition == condition
        )

    def test_frozen_dimensions(self) -> None:
        self.assertEqual(len(BACKENDS), 4)
        self.assertEqual(BACKENDS, (Backend.MEM0, Backend.AMEM, Backend.GRAPHITI, Backend.MEMOS))
        self.assertEqual(REPETITION_INDICES, (0, 1, 2))
        self.assertEqual(BENCHMARK_MAX_BUDGET[Benchmark.LOCOMO], 8000)
        self.assertEqual(BENCHMARK_MAX_BUDGET[Benchmark.LONGMEMEVAL_S], 4000)
        self.assertEqual(RQ2_CHECKPOINTS, tuple(range(1000, 8001, 1000)))

    def test_original_methods_and_exact_relation_masks(self) -> None:
        self.assertEqual(len(RQ1_METHODS), 7)
        self.assertEqual(len(ALL_METHODS), 16)
        for method in RQ1_METHODS[:4] + (UFUZZ,):
            self.assertEqual(method.enabled_relations, ALL_MUTATION_RELATIONS)
        self.assertEqual(UFUZZ_Q.enabled_relations, QUERY_MUTATION_RELATIONS)
        self.assertEqual(UFUZZ_M.enabled_relations, MEMORY_MUTATION_RELATIONS)
        self.assertIs(UFUZZ_Q.mutation_space, MutationSpace.QUERY_ONLY)
        self.assertIs(UFUZZ_M.mutation_space, MutationSpace.MEMORY_ONLY)

    def test_relation_configuration_rejects_inconsistent_masks(self) -> None:
        with self.assertRaises(ValueError):
            MutationRelationConfiguration(MutationSpace.QUERY_ONLY, MEMORY_MUTATION_RELATIONS)
        with self.assertRaises(ValueError):
            MutationRelationConfiguration(MutationSpace.FULL, QUERY_MUTATION_RELATIONS)
        with self.assertRaises(ValueError):
            MutationRelationConfiguration(MutationSpace.FULL, frozenset())

    def test_operator_ablations_remove_exactly_one_relation(self) -> None:
        expected = {
            "ufuzz-without-meaning-preserving-query": MutationRelation.MEANING_PRESERVING_QUERY,
            "ufuzz-without-target-changing-query": MutationRelation.TARGET_CHANGING_QUERY,
            "ufuzz-without-unsupported-query": MutationRelation.UNSUPPORTED_QUERY,
            "ufuzz-without-update": MutationRelation.UPDATE,
            "ufuzz-without-deletion": MutationRelation.DELETION,
            "ufuzz-without-unrelated-change": MutationRelation.UNRELATED_CHANGE,
        }
        self.assertEqual(len(RQ3_OPERATOR_ABLATIONS), 6)
        removed = set()
        for method in RQ3_OPERATOR_ABLATIONS:
            self.assertEqual(len(method.enabled_relations), 5)
            missing = ALL_MUTATION_RELATIONS - method.enabled_relations
            self.assertEqual(len(missing), 1)
            removed.update(missing)
            self.assertEqual(missing, {expected[method.method_id]})
            self.assertEqual(method.feedback_components, FULL_UFUZZ_FEEDBACK)
            self.assertIs(method.mutation_space, MutationSpace.FULL)
        self.assertEqual(removed, set(ALL_MUTATION_RELATIONS))

    def test_feedback_ablations_keep_six_relations_and_zero_without_renormalizing(self) -> None:
        expected = {
            UFUZZ_WITHOUT_COVERAGE: (False, True, True),
            UFUZZ_WITHOUT_NOVELTY: (True, False, True),
            UFUZZ_WITHOUT_PARENT_DIVERGENCE: (True, True, False),
        }
        self.assertEqual(len(RQ3_FEEDBACK_ABLATIONS), 3)
        for method, flags in expected.items():
            self.assertEqual(method.enabled_relations, ALL_MUTATION_RELATIONS)
            mask = method.feedback_components
            self.assertEqual((mask.coverage, mask.novelty, mask.parent_divergence), flags)
            self.assertIs(method.feedback_combination_rule, FeedbackCombinationRule.ZERO_DISABLED_WITHOUT_RENORMALIZATION)

    def test_scientific_identity_includes_exact_relation_mask(self) -> None:
        changed = replace(UFUZZ.scientific_key, relation_configuration=RQ3_OPERATOR_ABLATIONS[0].relation_configuration)
        self.assertNotEqual(changed, UFUZZ.scientific_key)
        original = CampaignSpec(Benchmark.LOCOMO, Backend.MEM0, UFUZZ, 0, 8000)
        altered = CampaignSpec(
            Benchmark.LOCOMO, Backend.MEM0, RQ3_OPERATOR_ABLATIONS[0], 0, 4000
        )
        self.assertNotEqual(original.scientific_key, altered.scientific_key)

    def test_method_identity_excludes_presentation_metadata(self) -> None:
        renamed = replace(UFUZZ, display_name="label")
        regrouped = replace(UFUZZ, fairness_group=FairnessGroup.FULL_SPACE_ABLATION)
        self.assertEqual(UFUZZ.scientific_key, renamed.scientific_key)
        self.assertEqual(UFUZZ.scientific_key, regrouped.scientific_key)
        original_campaign = CampaignSpec(
            Benchmark.LOCOMO, Backend.MEM0, UFUZZ, 0, 8000
        )
        self.assertEqual(
            original_campaign.scientific_key,
            CampaignSpec(
                Benchmark.LOCOMO, Backend.MEM0, renamed, 0, 8000
            ).scientific_key,
        )
        self.assertEqual(
            original_campaign.scientific_key,
            CampaignSpec(
                Benchmark.LOCOMO, Backend.MEM0, regrouped, 0, 8000
            ).scientific_key,
        )
        self.assertEqual(set(MethodScientificKey.__dataclass_fields__), {
            "method_id", "selection_family", "relation_configuration",
            "online_search_signal", "feedback_components", "feedback_combination_rule",
        })

    def test_other_causal_fields_change_method_identity(self) -> None:
        key = UFUZZ.scientific_key
        variants = (
            replace(key, method_id="ufuzz-causally-renamed"),
            replace(key, selection_family=SelectionFamily.COVERAGE_GUIDED),
            replace(key, online_search_signal=OnlineSearchSignal.COVERAGE_GAIN_ONLY),
            replace(key, feedback_components=FeedbackComponentMask(True, False, True)),
            replace(key, feedback_combination_rule=None),
        )
        self.assertTrue(all(value != key for value in variants))

    def test_plan_counts(self) -> None:
        rq1_locomo = {e.campaign.scientific_key for e in self.plan.rq1 if e.campaign.benchmark is Benchmark.LOCOMO}
        rq1_long = {e.campaign.scientific_key for e in self.plan.rq1 if e.campaign.benchmark is Benchmark.LONGMEMEVAL_S}
        rq1_keys = rq1_locomo | rq1_long
        rq3_new = {e.campaign.scientific_key for e in self.plan.rq3} - rq1_keys
        self.assertEqual(len(rq1_locomo), 84)
        self.assertEqual(len(rq1_long), 84)
        self.assertEqual(len(rq1_keys), 168)
        self.assertEqual(sum(c.max_budget for c in rq1_keys), 1_008_000)
        self.assertEqual(len(rq3_new), 108)
        self.assertEqual(sum(c.max_budget for c in rq3_new), 432_000)
        self.assertEqual(len(self.plan.rq1_rq3_campaigns), 276)
        self.assertEqual(sum(c.max_budget for c in self.plan.rq1_rq3_campaigns), 1_440_000)
        self.assertEqual(len(self.plan.rq4_campaigns), 54)
        self.assertEqual(sum(c.max_budget for c in self.plan.rq4_campaigns), 108_000)
        self.assertEqual(len(self.plan.campaigns), 330)
        self.assertEqual(self.plan.total_planned_valid_executions, 1_548_000)

    def test_rq_view_cardinalities(self) -> None:
        self.assertEqual(len(self.plan.rq1), 168)
        self.assertEqual(len(self.plan.rq2), 4 * 3 * 3 * 8)
        self.assertEqual(len(RQ3_METHODS), 14)
        self.assertEqual(len(RQ3_REUSED_METHODS), 5)
        self.assertEqual(len(NEW_RQ3_METHODS), 9)
        self.assertEqual(len(self.plan.rq3), 14 * 4 * 3)
        self.assertEqual(len(self.plan.rq4), 54)
        self.assertEqual(
            {entry.campaign.scientific_key for entry in self.plan.rq2}
            - {entry.campaign.scientific_key for entry in self.plan.rq1},
            set(),
        )

    def test_rq2_adds_zero_and_reuses_rq1_for_every_backend(self) -> None:
        rq1_keys = {e.campaign.scientific_key for e in self.plan.rq1}
        self.assertTrue({e.campaign.scientific_key for e in self.plan.rq2} <= rq1_keys)
        for backend in BACKENDS:
            for method in RQ2_METHODS:
                for repetition in REPETITION_INDICES:
                    rq1 = self.entry(ResearchQuestion.RQ1, backend, method.method_id, repetition, 8000)
                    rq2 = self.entry(ResearchQuestion.RQ2, backend, method.method_id, repetition, 8000)
                    self.assertIs(rq1.campaign, rq2.campaign)

    def test_rq3_reuses_five_rq1_prefixes(self) -> None:
        for backend in BACKENDS:
            for method in RQ3_REUSED_METHODS:
                for repetition in REPETITION_INDICES:
                    rq1 = self.entry(ResearchQuestion.RQ1, backend, method.method_id, repetition, 8000)
                    rq3 = self.entry(ResearchQuestion.RQ3, backend, method.method_id, repetition, RQ3_BUDGET)
                    self.assertIs(rq1.campaign, rq3.campaign)
                    self.assertEqual(rq3.campaign.max_budget, 8000)
        for entry in self.plan.rq3:
            if entry.campaign.method in NEW_RQ3_METHODS:
                self.assertEqual(entry.campaign.max_budget, 4000)

    def test_rq4_structure_and_provider_identity(self) -> None:
        self.assertEqual(RQ4_BACKENDS, (Backend.MEM0, Backend.GRAPHITI))
        self.assertEqual(RQ4_METHODS, (RANDOM_MUTATION, COVERAGE_GUIDED, UFUZZ))
        self.assertEqual(tuple(c.provider for c in NATIVE_MEMORY_LLM_CONDITIONS), tuple(NativeMemoryLLMProvider))
        keys = []
        for condition in NATIVE_MEMORY_LLM_CONDITIONS:
            entry = self.entry(ResearchQuestion.RQ4, Backend.MEM0, UFUZZ.method_id, 0, RQ4_BUDGET, condition)
            self.assertEqual(entry.campaign.max_budget, 2000)
            keys.append(entry.campaign.scientific_key)
        self.assertEqual(len(set(keys)), 3)
        self.assertFalse(hasattr(NativeMemoryLLMCondition, "api_key"))
        self.assertFalse(hasattr(NativeMemoryLLMCondition, "credential"))
        self.assertEqual(tuple(metric.value for metric in RQ4_METRICS), ("UF@B", "Cov@B"))
        self.assertEqual(
            RQ4_INTERPRETATION_EVIDENCE,
            frozenset(
                {"e0_cardinality", "retrievable_entry_granularity_profile_identity"}
            ),
        )

    def test_plan_specific_budget_validation(self) -> None:
        for backend in BACKENDS:
            for method in RQ1_METHODS:
                CampaignSpec(Benchmark.LOCOMO, backend, method, 0, 8000)
                CampaignSpec(Benchmark.LONGMEMEVAL_S, backend, method, 0, 4000)
            for method in NEW_RQ3_METHODS:
                CampaignSpec(Benchmark.LOCOMO, backend, method, 0, 4000)
        for backend in RQ4_BACKENDS:
            for method in RQ4_METHODS:
                for condition in NATIVE_MEMORY_LLM_CONDITIONS:
                    CampaignSpec(
                        Benchmark.LOCOMO, backend, method, 0, 2000, condition
                    )

    def test_raw_campaign_legality_rejects_every_prohibited_family(self) -> None:
        condition = NATIVE_MEMORY_LLM_CONDITIONS[0]
        prohibited = (
            (Benchmark.LOCOMO, Backend.MEM0, UFUZZ, 0, 4000, None),
            (Benchmark.LOCOMO, Backend.MEM0, UFUZZ, 0, 2000, None),
            (Benchmark.LOCOMO, Backend.MEM0, NEW_RQ3_METHODS[0], 0, 8000, None),
            (Benchmark.LONGMEMEVAL_S, Backend.MEM0, NEW_RQ3_METHODS[0], 0, 4000, None),
            (Benchmark.LONGMEMEVAL_S, Backend.MEM0, UFUZZ, 0, 4000, condition),
            (Benchmark.LOCOMO, Backend.AMEM, UFUZZ, 0, 2000, condition),
            (Benchmark.LOCOMO, Backend.MEMOS, UFUZZ, 0, 2000, condition),
            (Benchmark.LOCOMO, Backend.MEM0, UFUZZ, 0, 8000, condition),
            (Benchmark.LOCOMO, Backend.MEM0, UFUZZ, 0, 4000, condition),
            (Benchmark.LOCOMO, Backend.MEM0, RANDOM_MUTATION, 0, 2000, None),
        )
        for fields in prohibited:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                CampaignSpec(*fields)
        for method in tuple(
            method for method in ALL_METHODS if method not in RQ4_METHODS
        ):
            with self.subTest(method=method.method_id), self.assertRaises(ValueError):
                CampaignSpec(
                    Benchmark.LOCOMO,
                    Backend.MEM0,
                    method,
                    0,
                    2000,
                    condition,
                )

    def test_rq3_reference_cannot_be_an_independent_4k_raw_campaign(self) -> None:
        for method in RQ3_REUSED_METHODS:
            with self.subTest(method=method.method_id), self.assertRaises(ValueError):
                CampaignSpec(Benchmark.LOCOMO, Backend.MEM0, method, 0, 4000)
        for method in RQ3_REUSED_METHODS:
            entry = self.entry(
                ResearchQuestion.RQ3,
                Backend.MEM0,
                method.method_id,
                0,
                RQ3_BUDGET,
            )
            self.assertEqual(entry.campaign.max_budget, 8000)

    def test_configuration_bindings_normalize_and_affect_identity(self) -> None:
        a = ConfigurationBinding("backend", "v1")
        b = ConfigurationBinding("model", "v2")
        first = CampaignSpec(Benchmark.LOCOMO, Backend.MEM0, UFUZZ, 0, 8000, configuration_bindings=(a, b))
        second = CampaignSpec(Benchmark.LOCOMO, Backend.MEM0, UFUZZ, 0, 8000, configuration_bindings=(b, a))
        self.assertEqual(first.scientific_key, second.scientific_key)
        self.assertNotEqual(first.scientific_key, replace(second, configuration_bindings=(replace(a, configuration_id="v3"), b)).scientific_key)
        with self.assertRaises(ValueError):
            CampaignSpec(Benchmark.LOCOMO, Backend.MEM0, UFUZZ, 0, 8000, configuration_bindings=(a, replace(a, configuration_id="v2")))

    def test_campaigns_are_immutable_and_engineering_target_is_not_identity(self) -> None:
        spec = CampaignSpec(Benchmark.LOCOMO, Backend.MEM0, UFUZZ, 0, 8000)
        with self.assertRaises(FrozenInstanceError):
            spec.max_budget = 1
        self.assertNotIn("research_question", spec.scientific_key.__dataclass_fields__)
        self.assertNotIn("gpu_count", spec.scientific_key.__dataclass_fields__)
        self.assertEqual(len({spec, spec}), 1)
        self.assertEqual(hash(spec), hash(spec))
        self.assertEqual(ENGINEERING_TARGET.planned_valid_executions, 1_440_000)
        self.assertEqual(ENGINEERING_TARGET.minimum_average_valid_executions_per_second, 1_440_000 / 86400)

    def test_view_metadata_and_checkpoints_are_excluded_from_campaign_identity(self) -> None:
        key_fields = set(CampaignSpec(
            Benchmark.LOCOMO, Backend.MEM0, UFUZZ, 0, 8000
        ).scientific_key.__dataclass_fields__)
        self.assertFalse(
            key_fields
            & {
                "research_question",
                "display_method_id",
                "checkpoint",
                "gpu_count",
                "wall_clock_seconds",
                "worker_count",
                "batch_size",
            }
        )
        entries = [
            self.entry(ResearchQuestion.RQ2, Backend.MEM0, UFUZZ.method_id, 0, cp)
            for cp in RQ2_CHECKPOINTS
        ]
        self.assertTrue(all(item.campaign is entries[0].campaign for item in entries))

    def test_planning_topology_is_configuration_unbound(self) -> None:
        self.assertTrue(
            all(
                not campaign.has_configuration_bindings
                and campaign.configuration_bindings == ()
                for campaign in self.plan.campaigns
            )
        )

    def test_primary_fairness_groups_and_supported_feedback_rule_are_exact(self) -> None:
        self.assertEqual(
            {method.method_id for method in RQ1_METHODS if method.fairness_group is FairnessGroup.FULL_SPACE_PRIMARY},
            {"random-mutation", "unguided-llm", "llm-as-judge", "coverage-guided", "ufuzz"},
        )
        self.assertEqual(
            {method.method_id for method in RQ1_METHODS if method.fairness_group is FairnessGroup.RESTRICTED_SPACE},
            {"ufuzz-q", "ufuzz-m"},
        )
        self.assertEqual(
            tuple(FeedbackCombinationRule),
            (FeedbackCombinationRule.ZERO_DISABLED_WITHOUT_RENORMALIZATION,),
        )

    def test_prefix_validation(self) -> None:
        valid = [CheckpointMetrics(cp, i // 2, (i // 2) / 10) for i, cp in enumerate(RQ2_CHECKPOINTS)]
        validate_rq2_prefix_series(valid)
        bad_uf = list(valid); bad_uf[5] = CheckpointMetrics(6000, 0, valid[5].coverage_fraction)
        with self.assertRaisesRegex(ValueError, "UF"):
            validate_rq2_prefix_series(bad_uf)
        bad_cov = list(valid); bad_cov[5] = CheckpointMetrics(6000, valid[5].distinct_faults, 0.0)
        with self.assertRaisesRegex(ValueError, "coverage"):
            validate_rq2_prefix_series(bad_cov)

    def test_three_repetition_aggregation_uses_sample_standard_deviation(self) -> None:
        value = aggregate_three_repetitions({0: 1.0, 1: 2.0, 2: 3.0})
        self.assertEqual(value.arithmetic_mean, 2.0)
        self.assertEqual(value.sample_standard_deviation, 1.0)
        with self.assertRaises(ValueError):
            aggregate_three_repetitions({0: 1.0, 1: 2.0})


if __name__ == "__main__":
    unittest.main()
