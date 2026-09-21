from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import unittest

from ufuzz.evaluation_contract import (
    BACKENDS,
    BENCHMARK_MAX_BUDGET,
    COVERAGE_GUIDED,
    ENGINEERING_TARGET,
    EVALUATION_PLAN,
    FULL_UFUZZ_FEEDBACK,
    MEMORY_MUTATION_RELATIONS,
    NEW_RQ3_METHODS,
    QUERY_MUTATION_RELATIONS,
    REPETITION_INDICES,
    RQ1_METHODS,
    RQ2_CHECKPOINTS,
    RQ2_METHODS,
    RQ3_METHODS,
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
    EngineeringTarget,
    FairnessGroup,
    FeedbackCombinationRule,
    FeedbackComponentMask,
    MethodScientificKey,
    MutationSpace,
    OnlineSearchSignal,
    ResearchQuestion,
    SelectionFamily,
    aggregate_three_repetitions,
    mutation_relations_for,
    validate_rq2_prefix_series,
)


class EvaluationContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = EVALUATION_PLAN

    def _entry(
        self,
        rq: ResearchQuestion,
        backend: Backend,
        method_id: str,
        repetition: int,
        checkpoint: int,
    ):
        entries = {
            ResearchQuestion.RQ1: self.plan.rq1,
            ResearchQuestion.RQ2: self.plan.rq2,
            ResearchQuestion.RQ3: self.plan.rq3,
        }[rq]
        return next(
            entry
            for entry in entries
            if entry.campaign.backend is backend
            and entry.display_method_id == method_id
            and entry.campaign.repetition_index == repetition
            and entry.checkpoint == checkpoint
            and entry.campaign.benchmark is Benchmark.LOCOMO
        )

    def test_frozen_dimensions_and_displayed_methods(self) -> None:
        self.assertEqual(len(RQ1_METHODS), 7)
        self.assertEqual(
            tuple(method.method_id for method in RQ1_METHODS),
            (
                "random-mutation",
                "unguided-llm",
                "llm-as-judge",
                "coverage-guided",
                "ufuzz-q",
                "ufuzz-m",
                "ufuzz",
            ),
        )
        self.assertEqual(len(BACKENDS), 3)
        self.assertEqual(REPETITION_INDICES, (0, 1, 2))
        self.assertEqual(BENCHMARK_MAX_BUDGET[Benchmark.LOCOMO], 8000)
        self.assertEqual(BENCHMARK_MAX_BUDGET[Benchmark.LONGMEMEVAL_S], 4000)
        self.assertEqual(RQ2_CHECKPOINTS, tuple(range(1000, 8001, 1000)))

    def test_unique_method_and_campaign_counts(self) -> None:
        locomo = [
            campaign
            for campaign in self.plan.campaigns
            if campaign.benchmark is Benchmark.LOCOMO
        ]
        longmemeval = [
            campaign
            for campaign in self.plan.campaigns
            if campaign.benchmark is Benchmark.LONGMEMEVAL_S
        ]
        self.assertEqual(len({item.method.scientific_key for item in locomo}), 10)
        self.assertEqual(
            len({item.method.scientific_key for item in longmemeval}), 7
        )
        self.assertEqual(len(locomo), 90)
        self.assertEqual(len(longmemeval), 63)
        self.assertEqual(len(self.plan.campaigns), 153)
        self.assertEqual(
            len({campaign.scientific_key for campaign in self.plan.campaigns}),
            153,
        )

    def test_view_cardinalities_and_full_space_fairness_group(self) -> None:
        self.assertEqual(len(self.plan.rq1), 2 * 7 * 3 * 3)
        self.assertEqual(len(self.plan.rq2), 3 * 3 * 3 * 8)
        self.assertEqual(len(self.plan.rq3), 5 * 3 * 3)
        self.assertEqual(
            {
                method.method_id
                for method in RQ1_METHODS
                if method.fairness_group is FairnessGroup.FULL_SPACE_PRIMARY
            },
            {
                "random-mutation",
                "unguided-llm",
                "llm-as-judge",
                "coverage-guided",
                "ufuzz",
            },
        )

    def test_campaign_specs_are_immutable_and_hash_safe(self) -> None:
        spec = CampaignSpec(Benchmark.LOCOMO, Backend.MEM0, UFUZZ, 0, 8000)
        self.assertEqual({spec}, {spec})
        self.assertEqual({spec.scientific_key}, {spec.scientific_key})
        with self.assertRaises(FrozenInstanceError):
            spec.max_budget = 4000

    def test_method_scientific_identity_excludes_presentation_metadata(self) -> None:
        renamed = replace(UFUZZ, display_name="Presentation-only label")
        regrouped = replace(UFUZZ, fairness_group=FairnessGroup.FULL_SPACE_ABLATION)
        self.assertEqual(UFUZZ.scientific_key, renamed.scientific_key)
        self.assertEqual(UFUZZ.scientific_key, regrouped.scientific_key)
        original_campaign = CampaignSpec(
            Benchmark.LOCOMO, Backend.MEM0, UFUZZ, 0, 8000
        )
        renamed_campaign = CampaignSpec(
            Benchmark.LOCOMO, Backend.MEM0, renamed, 0, 8000
        )
        regrouped_campaign = CampaignSpec(
            Benchmark.LOCOMO, Backend.MEM0, regrouped, 0, 8000
        )
        self.assertEqual(
            original_campaign.scientific_key, renamed_campaign.scientific_key
        )
        self.assertEqual(
            original_campaign.scientific_key, regrouped_campaign.scientific_key
        )

    def test_each_causal_method_field_changes_scientific_identity(self) -> None:
        key = UFUZZ.scientific_key
        variants = (
            replace(key, selection_family=SelectionFamily.COVERAGE_GUIDED),
            replace(key, mutation_space=MutationSpace.QUERY_ONLY),
            replace(key, online_search_signal=OnlineSearchSignal.COVERAGE_GAIN_ONLY),
            replace(
                key,
                feedback_components=FeedbackComponentMask(True, False, True),
            ),
            replace(
                key,
                feedback_combination_rule=None,
            ),
        )
        self.assertTrue(all(candidate != key for candidate in variants))
        self.assertEqual(len(set(variants)), len(variants))

    def test_method_scientific_key_does_not_carry_view_metadata(self) -> None:
        self.assertEqual(
            set(MethodScientificKey.__dataclass_fields__),
            {
                "method_id",
                "selection_family",
                "mutation_space",
                "online_search_signal",
                "feedback_components",
                "feedback_combination_rule",
            },
        )

    def test_total_planned_valid_execution_budget(self) -> None:
        self.assertEqual(90 * 8000, 720000)
        self.assertEqual(63 * 4000, 252000)
        self.assertEqual(self.plan.total_planned_valid_executions, 972000)

    def test_rq2_adds_no_campaign_and_reuses_raw_trajectories(self) -> None:
        rq1_locomo_keys = {
            entry.campaign.scientific_key
            for entry in self.plan.rq1
            if entry.campaign.benchmark is Benchmark.LOCOMO
        }
        self.assertTrue(
            {
                entry.campaign.scientific_key for entry in self.plan.rq2
            }.issubset(rq1_locomo_keys)
        )
        self.assertEqual(
            len({entry.campaign.scientific_key for entry in self.plan.rq2}),
            3 * 3 * 3,
        )
        for backend in BACKENDS:
            for method in RQ2_METHODS:
                for repetition in REPETITION_INDICES:
                    endpoint = self._entry(
                        ResearchQuestion.RQ2,
                        backend,
                        method.method_id,
                        repetition,
                        8000,
                    )
                    rq1 = self._entry(
                        ResearchQuestion.RQ1,
                        backend,
                        method.method_id,
                        repetition,
                        8000,
                    )
                    self.assertIs(endpoint.campaign, rq1.campaign)

    def test_rq3_adds_only_three_ablation_families(self) -> None:
        rq1_keys = {entry.campaign.scientific_key for entry in self.plan.rq1}
        rq3_keys = {entry.campaign.scientific_key for entry in self.plan.rq3}
        self.assertEqual(len(rq3_keys.difference(rq1_keys)), 27)
        self.assertEqual(len(NEW_RQ3_METHODS), 3)
        self.assertEqual(len(RQ3_METHODS), 5)
        for backend in BACKENDS:
            for repetition in REPETITION_INDICES:
                for method in (COVERAGE_GUIDED, UFUZZ):
                    rq1 = self._entry(
                        ResearchQuestion.RQ1,
                        backend,
                        method.method_id,
                        repetition,
                        8000,
                    )
                    rq3 = self._entry(
                        ResearchQuestion.RQ3,
                        backend,
                        method.method_id,
                        repetition,
                        8000,
                    )
                    self.assertIs(rq1.campaign, rq3.campaign)

    def test_full_ufuzz_identity_is_shared_across_all_three_views(self) -> None:
        for backend in BACKENDS:
            for repetition in REPETITION_INDICES:
                rq1 = self._entry(
                    ResearchQuestion.RQ1, backend, UFUZZ.method_id, repetition, 8000
                )
                rq2 = self._entry(
                    ResearchQuestion.RQ2, backend, UFUZZ.method_id, repetition, 8000
                )
                rq3 = self._entry(
                    ResearchQuestion.RQ3, backend, UFUZZ.method_id, repetition, 8000
                )
                self.assertIs(rq1.campaign, rq2.campaign)
                self.assertIs(rq2.campaign, rq3.campaign)

    def test_q_and_m_mutation_spaces_are_structurally_restricted(self) -> None:
        self.assertIs(UFUZZ_Q.mutation_space, MutationSpace.QUERY_ONLY)
        self.assertIs(UFUZZ_M.mutation_space, MutationSpace.MEMORY_ONLY)
        self.assertIs(UFUZZ.mutation_space, MutationSpace.FULL)
        self.assertIs(UFUZZ_Q.fairness_group, FairnessGroup.RESTRICTED_SPACE)
        self.assertEqual(
            mutation_relations_for(UFUZZ_Q.mutation_space), QUERY_MUTATION_RELATIONS
        )
        self.assertEqual(
            mutation_relations_for(UFUZZ_M.mutation_space), MEMORY_MUTATION_RELATIONS
        )
        self.assertEqual(
            mutation_relations_for(UFUZZ.mutation_space),
            QUERY_MUTATION_RELATIONS | MEMORY_MUTATION_RELATIONS,
        )

    def test_rq3_component_masks_zero_without_renormalization(self) -> None:
        expected = {
            UFUZZ_WITHOUT_COVERAGE: (False, True, True),
            UFUZZ_WITHOUT_NOVELTY: (True, False, True),
            UFUZZ_WITHOUT_PARENT_DIVERGENCE: (True, True, False),
            UFUZZ: (True, True, True),
        }
        for method, flags in expected.items():
            mask = method.feedback_components
            self.assertEqual(
                (mask.coverage, mask.novelty, mask.parent_divergence), flags
            )
            self.assertIs(
                method.feedback_combination_rule,
                FeedbackCombinationRule.ZERO_DISABLED_WITHOUT_RENORMALIZATION,
            )
        self.assertEqual(UFUZZ.feedback_components, FULL_UFUZZ_FEEDBACK)
        self.assertIsNone(COVERAGE_GUIDED.feedback_components)

    def test_prefix_validator_rejects_decreasing_uf(self) -> None:
        values = [
            CheckpointMetrics(checkpoint, index + 1, 0.1 * index)
            for index, checkpoint in enumerate(RQ2_CHECKPOINTS)
        ]
        values[4] = CheckpointMetrics(5000, 1, 0.4)
        with self.assertRaisesRegex(ValueError, "UF"):
            validate_rq2_prefix_series(values)

    def test_prefix_validator_rejects_decreasing_coverage(self) -> None:
        values = [
            CheckpointMetrics(checkpoint, index, 0.1 * index)
            for index, checkpoint in enumerate(RQ2_CHECKPOINTS)
        ]
        values[5] = CheckpointMetrics(6000, 5, 0.1)
        with self.assertRaisesRegex(ValueError, "coverage"):
            validate_rq2_prefix_series(values)

    def test_prefix_validator_accepts_flat_and_increasing_values(self) -> None:
        validate_rq2_prefix_series(
            tuple(
                CheckpointMetrics(checkpoint, index // 2, (index // 2) / 10)
                for index, checkpoint in enumerate(RQ2_CHECKPOINTS)
            )
        )

    def test_campaign_key_excludes_rq_and_engineering_target(self) -> None:
        spec = CampaignSpec(Benchmark.LOCOMO, Backend.MEM0, UFUZZ, 0, 8000)
        same = CampaignSpec(Benchmark.LOCOMO, Backend.MEM0, UFUZZ, 0, 8000)
        other_target = replace(ENGINEERING_TARGET, gpu_count=12)
        self.assertEqual(spec, same)
        self.assertEqual(spec.scientific_key, same.scientific_key)
        self.assertNotEqual(ENGINEERING_TARGET, other_target)
        self.assertNotIn("research_question", CampaignSpec.__dataclass_fields__)
        self.assertNotIn("wall_clock_seconds", CampaignSpec.__dataclass_fields__)

    def test_configuration_bindings_are_normalized_causal_identity(self) -> None:
        from ufuzz.evaluation_contract import ConfigurationBinding

        backend = ConfigurationBinding("backend", "mem0-profile-a")
        generator = ConfigurationBinding("generator", "unfrozen-example")
        first = CampaignSpec(
            Benchmark.LOCOMO,
            Backend.MEM0,
            UFUZZ,
            0,
            8000,
            (backend, generator),
        )
        reordered = CampaignSpec(
            Benchmark.LOCOMO,
            Backend.MEM0,
            UFUZZ,
            0,
            8000,
            (generator, backend),
        )
        changed = CampaignSpec(
            Benchmark.LOCOMO,
            Backend.MEM0,
            UFUZZ,
            0,
            8000,
            (ConfigurationBinding("backend", "another-config"), generator),
        )
        self.assertEqual(first, reordered)
        self.assertEqual(first.scientific_key, reordered.scientific_key)
        self.assertNotEqual(first.scientific_key, changed.scientific_key)
        self.assertTrue(first.has_configuration_bindings)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            CampaignSpec(
                Benchmark.LOCOMO,
                Backend.MEM0,
                UFUZZ,
                0,
                8000,
                (
                    ConfigurationBinding("backend", "one"),
                    ConfigurationBinding("backend", "two"),
                ),
            )

    def test_planning_topology_is_explicitly_configuration_unbound(self) -> None:
        self.assertTrue(
            all(
                not campaign.has_configuration_bindings
                for campaign in self.plan.campaigns
            )
        )

    def test_rq2_checkpoints_are_views_of_one_max_budget_campaign(self) -> None:
        entries = [
            entry
            for entry in self.plan.rq2
            if entry.campaign.backend is Backend.MEM0
            and entry.campaign.method == UFUZZ
            and entry.campaign.repetition_index == 0
        ]
        self.assertEqual(
            tuple(entry.checkpoint for entry in entries), RQ2_CHECKPOINTS
        )
        self.assertEqual(len({id(entry.campaign) for entry in entries}), 1)
        self.assertEqual(entries[0].campaign.max_budget, 8000)

    def test_three_run_aggregation_uses_sample_standard_deviation(self) -> None:
        value = aggregate_three_repetitions({0: 1.0, 1: 2.0, 2: 3.0})
        self.assertEqual(value.arithmetic_mean, 2.0)
        self.assertEqual(value.sample_standard_deviation, 1.0)

    def test_24_hour_target_is_planning_only(self) -> None:
        self.assertEqual(ENGINEERING_TARGET.gpu_count, 6)
        self.assertEqual(ENGINEERING_TARGET.wall_clock_seconds, 86400)
        self.assertEqual(
            ENGINEERING_TARGET.minimum_average_valid_executions_per_second,
            11.25,
        )
        self.assertEqual(
            ENGINEERING_TARGET,
            EngineeringTarget(6, 86400, 972000),
        )


if __name__ == "__main__":
    unittest.main()
