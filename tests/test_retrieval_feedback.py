from __future__ import annotations

import unittest

from ufuzz.coverage import (
    CoverageEligibility,
    CoverageEntryId,
    CoverageIneligibleError,
    CoverageState,
    FrozenCheckpointInventory,
    InitialCoverageEntry,
    InitialEntryLineage,
    LineageRelation,
    LineageResolutionError,
    LineageStatus,
    PhysicalEntryRef,
)
from ufuzz.retrieval_feedback import (
    BehaviorHistory,
    MissingExactParentSignatureError,
    MutationRelation,
    behavior_score,
    bounded_coverage_gain,
    canonical_tuple,
    coverage_guided_score,
    parent_child_divergence,
    rbo_distance,
    rbo_ext,
    rbo_p_for_depth,
    retrieval_signature,
    score_observed_retrieval,
    ufuzz_score,
)


CAMPAIGN = "campaign-1"
ROOT = "checkpoint-1"


def coverage_id(name: str, root: str = ROOT) -> CoverageEntryId:
    return CoverageEntryId(CAMPAIGN, root, name)


def initial_entry(name: str, root: str = ROOT) -> InitialCoverageEntry:
    return InitialCoverageEntry(
        coverage_id=coverage_id(name, root),
        root_checkpoint_id=root,
        initial_backend_entry_id=f"backend-{name}",
        initial_observable_projection={"content": name, "rankable": True},
        provenance_ids=(f"source-{name}",),
    )


def physical_ref(name: str, state: str = "state-0", root: str = ROOT) -> PhysicalEntryRef:
    return PhysicalEntryRef(CAMPAIGN, root, state, name)


def reference_rbo_ext(first: tuple[str, ...], second: tuple[str, ...], p: float) -> float:
    """Independent direct transcription of Webber et al. Equation 32."""

    if not first and not second:
        return 1.0
    if not first or not second:
        return 0.0
    short, long = (first, second) if len(first) <= len(second) else (second, first)
    short_length = len(short)
    long_length = len(long)
    overlap = [0]
    for depth in range(1, long_length + 1):
        short_prefix = set(short[: min(depth, short_length)])
        long_prefix = set(long[:depth])
        overlap.append(len(short_prefix & long_prefix))

    first_sum = sum(
        overlap[depth] * (p**depth) / depth
        for depth in range(1, long_length + 1)
    )
    second_sum = sum(
        overlap[short_length]
        * (depth - short_length)
        * (p**depth)
        / (short_length * depth)
        for depth in range(short_length + 1, long_length + 1)
    )
    tail = (
        (overlap[long_length] - overlap[short_length]) / long_length
        + overlap[short_length] / short_length
    ) * (p**long_length)
    return ((1.0 - p) / p) * (first_sum + second_sum) + tail


class LineageAndCoverageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.entries = tuple(initial_entry(f"e{index}") for index in range(1, 5))
        inventory = FrozenCheckpointInventory(CAMPAIGN, ROOT, self.entries)
        self.lineage = InitialEntryLineage(CAMPAIGN, (inventory,))
        self.refs = tuple(physical_ref(f"m{index}") for index in range(1, 5))
        for entry, ref in zip(self.entries, self.refs, strict=True):
            self.lineage.certify_binding(
                ref,
                (entry.coverage_id,),
                LineageRelation.INITIAL,
            )

    def test_update_replacement_and_certified_merge_lineage(self) -> None:
        same = self.lineage.certify_same_id_update(self.refs[0])
        self.assertEqual(same.coverage_ids, frozenset({coverage_id("e1")}))

        replacement = physical_ref("replacement", state="state-1")
        replaced = self.lineage.certify_replacement(self.refs[0], replacement)
        self.assertEqual(replaced.relation, LineageRelation.REPLACEMENT_UPDATE)
        self.assertEqual(replaced.coverage_ids, same.coverage_ids)

        merged = physical_ref("merged", state="state-2")
        binding = self.lineage.certify_merge(self.refs[:2], merged)
        self.assertEqual(binding.relation, LineageRelation.CERTIFIED_MERGE)
        self.assertEqual(
            binding.coverage_ids,
            frozenset({coverage_id("e1"), coverage_id("e2")}),
        )

    def test_uncertifiable_lineage_is_explicit_and_never_invents_ids(self) -> None:
        unknown = physical_ref("ambiguous")
        result = self.lineage.mark_uncertifiable(
            unknown,
            status=LineageStatus.CAPABILITY_FAILURE,
            reason="backend merge cannot be certified",
        )
        self.assertEqual(result.coverage_ids, frozenset())
        with self.assertRaises(LineageResolutionError) as caught:
            self.lineage.resolve(unknown)
        self.assertEqual(caught.exception.binding.status, LineageStatus.CAPABILITY_FAILURE)

    def test_raw_gain_repeat_and_multi_entry_merge(self) -> None:
        state = CoverageState.from_entries(self.entries)
        first = state.observe(
            (
                {coverage_id("e1"), coverage_id("e2")},
                {coverage_id("e3"), coverage_id("e4")},
            )
        )
        self.assertEqual(first.raw_gain, 4)
        self.assertGreater(first.raw_gain, 2)
        self.assertEqual(first.state.reached_count, 4)
        self.assertEqual(first.state.cov_fraction, 1.0)

        repeated = first.state.observe(({coverage_id("e1"), coverage_id("e2")},))
        self.assertEqual(repeated.raw_gain, 0)
        self.assertEqual(repeated.state.reached_count, 4)

    def test_deletion_does_not_shrink_frozen_denominator(self) -> None:
        state = CoverageState.from_entries(self.entries)
        self.lineage.mark_deleted(self.refs[0])
        self.assertTrue(self.lineage.is_deleted(self.refs[0]))
        self.assertEqual(state.denominator_count, 4)
        self.assertIn(coverage_id("e1"), state.initial_ids)

    def test_out_of_inventory_id_is_rejected(self) -> None:
        state = CoverageState.from_entries(self.entries)
        with self.assertRaisesRegex(ValueError, "outside frozen E_0"):
            state.observe(({coverage_id("outside")},))

    def test_empty_inventory_is_explicitly_rq1_ineligible(self) -> None:
        state = CoverageState(frozenset())
        self.assertEqual(
            state.eligibility,
            CoverageEligibility.RQ1_INELIGIBLE_EMPTY_INVENTORY,
        )
        self.assertIsNone(state.cov_fraction)
        self.assertEqual(state.denominator_count, 0)
        with self.assertRaises(CoverageIneligibleError):
            state.observe(())

    def test_observable_projection_is_immutable_and_search_safe(self) -> None:
        projection = {"content": "safe", "nested": {"rank": 1}}
        entry = InitialCoverageEntry(
            coverage_id=coverage_id("safe"),
            root_checkpoint_id=ROOT,
            initial_backend_entry_id=None,
            initial_observable_projection=projection,
        )
        projection["nested"]["rank"] = 2
        self.assertEqual(entry.initial_observable_projection["nested"]["rank"], 1)
        with self.assertRaisesRegex(ValueError, "evaluator-only"):
            InitialCoverageEntry(
                coverage_id=coverage_id("unsafe"),
                root_checkpoint_id=ROOT,
                initial_backend_entry_id=None,
                initial_observable_projection={"gold_answer": "secret"},
            )


class SignatureAndRBOTests(unittest.TestCase):
    def setUp(self) -> None:
        self.e1 = coverage_id("e1")
        self.e2 = coverage_id("e2")
        self.e3 = coverage_id("e3")

    def test_canonical_tuple_order_independence_and_singleton(self) -> None:
        self.assertEqual(canonical_tuple({self.e2, self.e1}), (self.e1, self.e2))
        self.assertEqual(canonical_tuple({self.e1, self.e2}), (self.e1, self.e2))
        self.assertEqual(canonical_tuple({self.e1}), (self.e1,))
        with self.assertRaises(ValueError):
            canonical_tuple(set())

    def test_signature_first_occurrence_dedup_and_empty(self) -> None:
        signature = retrieval_signature(({self.e1}, {self.e2}, {self.e1}))
        self.assertEqual(signature, ((self.e1,), (self.e2,)))
        self.assertEqual(retrieval_signature(()), ())

    def test_multi_entry_tokens_remain_atomic(self) -> None:
        signature = retrieval_signature(({self.e2, self.e1}, {self.e3}))
        self.assertEqual(signature, ((self.e1, self.e2), (self.e3,)))
        self.assertNotEqual(signature, ((self.e1,), (self.e2,), (self.e3,)))

    def test_rbo_empty_identical_disjoint_and_bounds(self) -> None:
        p = rbo_p_for_depth(4)
        self.assertEqual(rbo_distance((), (), p), 0.0)
        self.assertEqual(rbo_distance((), ("a",), p), 1.0)
        self.assertEqual(rbo_distance(("a",), (), p), 1.0)
        self.assertAlmostEqual(rbo_distance(("a", "b"), ("a", "b"), p), 0.0)
        self.assertAlmostEqual(rbo_distance(("a", "b"), ("x", "y"), p), 1.0)
        for first, second in (
            (("a",), ("a",)),
            (("a", "b"), ("b", "a")),
            (("a", "b"), ("a", "b", "c")),
            (("a", "b", "c"), ("a", "x")),
        ):
            similarity = rbo_ext(first, second, p)
            distance = rbo_distance(first, second, p)
            self.assertGreaterEqual(similarity, 0.0)
            self.assertLessEqual(similarity, 1.0)
            self.assertGreaterEqual(distance, 0.0)
            self.assertLessEqual(distance, 1.0)
            self.assertAlmostEqual(
                distance,
                rbo_distance(second, first, p),
            )

    def test_rbo_is_rank_sensitive_and_top_weighted(self) -> None:
        p = 0.8
        base = ("a", "b", "c", "d", "e", "f")
        top_swap = ("b", "a", "c", "d", "e", "f")
        deep_swap = ("a", "b", "c", "d", "f", "e")
        self.assertGreater(rbo_distance(base, top_swap, p), 0.0)
        self.assertGreater(
            rbo_distance(base, top_swap, p),
            rbo_distance(base, deep_swap, p),
        )

    def test_rbo_treats_multi_entry_tuple_as_one_exact_token(self) -> None:
        p = rbo_p_for_depth(3)
        merged = ((self.e1, self.e2), (self.e3,))
        expanded = ((self.e1,), (self.e2,), (self.e3,))
        self.assertEqual(rbo_distance(merged, merged, p), 0.0)
        self.assertGreater(rbo_distance(merged, expanded, p), 0.0)

    def test_rbo_parameter_and_rank_validation(self) -> None:
        with self.assertRaises(ValueError):
            rbo_p_for_depth(1)
        with self.assertRaises(ValueError):
            rbo_ext(("a", "a"), ("a",), 0.5)
        with self.assertRaises(ValueError):
            rbo_ext(("a",), ("a",), 1.0)

    def test_rbo_agrees_with_independent_equation_32_reference(self) -> None:
        p = 0.8
        cases = (
            (("a", "b", "c"), ("a", "b", "c")),
            (("a", "b", "c", "d"), ("a", "x", "c", "y")),
            (("a", "b"), ("x", "a", "b", "z")),
            (("x", "a", "b", "z"), ("a", "b")),
            (("a", "b", "c"), ("x", "y", "z")),
        )
        for first, second in cases:
            with self.subTest(first=first, second=second):
                self.assertAlmostEqual(
                    rbo_ext(first, second, p),
                    reference_rbo_ext(first, second, p),
                    places=14,
                )


class HistoryAndScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.e1 = coverage_id("e1")
        self.e2 = coverage_id("e2")
        self.e3 = coverage_id("e3")
        self.first = ((self.e1,), (self.e2,))
        self.second = ((self.e2,), (self.e1,))
        self.third = ((self.e3,),)

    def coverage_update(self, raw_gain: int):
        initial_ids = frozenset({self.e1, self.e2, self.e3})
        if raw_gain == 1:
            return CoverageState(initial_ids).observe(({self.e3},))
        if raw_gain == 0:
            return CoverageState(initial_ids, frozenset({self.e3})).observe(
                ({self.e3},)
            )
        raise ValueError("test helper supports only gains 0 and 1")

    def test_initialization_history_is_checkpoint_local_without_coverage_credit(self) -> None:
        history = BehaviorHistory(CAMPAIGN, {ROOT: (self.first,)})
        state = CoverageState(frozenset({self.e1, self.e2, self.e3}))
        other_root_signature = ((coverage_id("e1", "checkpoint-2"),),)
        self.assertFalse(history.is_new_behavior(ROOT, self.first))
        self.assertTrue(
            history.is_new_behavior("checkpoint-2", other_root_signature)
        )
        self.assertEqual(state.reached_count, 0)
        self.assertEqual(state.cov_fraction, 0.0)

    def test_cross_checkpoint_signature_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot cross"):
            retrieval_signature(
                (
                    {self.e1},
                    {coverage_id("e1", "checkpoint-2")},
                )
            )

    def test_behavior_history_rejects_another_campaign(self) -> None:
        history = BehaviorHistory(CAMPAIGN, {ROOT: (self.first,)})
        other_campaign_signature = (
            (CoverageEntryId("campaign-2", ROOT, "e1"),),
        )
        with self.assertRaisesRegex(ValueError, "another campaign"):
            history.is_new_behavior(ROOT, other_campaign_signature)

    def test_new_behavior_scored_before_observe(self) -> None:
        history = BehaviorHistory(CAMPAIGN, {ROOT: (self.first,)})
        self.assertTrue(history.is_new_behavior(ROOT, self.second))
        history.observe(ROOT, self.second)
        self.assertFalse(history.is_new_behavior(ROOT, self.second))

    def test_novelty_uses_nearest_history_and_reports_empty_archive(self) -> None:
        p = rbo_p_for_depth(4)
        history = BehaviorHistory(CAMPAIGN, {ROOT: (self.first, self.second)})
        result = history.novelty(ROOT, self.second, p)
        self.assertEqual(result.score, 0.0)
        self.assertFalse(result.prior_history_empty)

        expected = min(
            rbo_distance(self.third, self.first, p),
            rbo_distance(self.third, self.second, p),
        )
        self.assertAlmostEqual(history.novelty(ROOT, self.third, p).score, expected)

        empty = history.novelty(
            "checkpoint-2",
            ((coverage_id("e1", "checkpoint-2"),),),
            p,
        )
        self.assertEqual(empty.score, 1.0)
        self.assertTrue(empty.prior_history_empty)

    def test_exact_parent_divergence_and_no_fallback(self) -> None:
        p = rbo_p_for_depth(4)
        expected = rbo_distance(self.first, self.second, p)
        self.assertAlmostEqual(
            parent_child_divergence(self.first, self.second, p),
            expected,
        )

        history = BehaviorHistory(CAMPAIGN, {ROOT: (self.first,)})
        with self.assertRaises(MissingExactParentSignatureError):
            score_observed_retrieval(
                root_checkpoint_id=ROOT,
                observed_signature=self.second,
                coverage_update=self.coverage_update(1),
                retrieval_depth=4,
                mutation_relation=MutationRelation.MEANING_PRESERVING_QUERY,
                history=history,
            )
        self.assertTrue(history.is_new_behavior(ROOT, self.second))

        with self.assertRaisesRegex(ValueError, "must not supply"):
            score_observed_retrieval(
                root_checkpoint_id=ROOT,
                observed_signature=self.second,
                coverage_update=self.coverage_update(1),
                retrieval_depth=4,
                mutation_relation=MutationRelation.UPDATE,
                history=history,
                exact_parent_signature=self.first,
            )

    def test_observed_feedback_inserts_after_scoring(self) -> None:
        history = BehaviorHistory(CAMPAIGN, {ROOT: (self.first,)})
        feedback = score_observed_retrieval(
            root_checkpoint_id=ROOT,
            observed_signature=self.second,
            coverage_update=self.coverage_update(1),
            retrieval_depth=4,
            mutation_relation=MutationRelation.MEANING_PRESERVING_QUERY,
            history=history,
            exact_parent_signature=self.first,
        )
        self.assertTrue(feedback.new_behavior)
        self.assertIsNotNone(feedback.divergence)
        self.assertFalse(history.is_new_behavior(ROOT, self.second))

        repeated = score_observed_retrieval(
            root_checkpoint_id=ROOT,
            observed_signature=self.second,
            coverage_update=self.coverage_update(0),
            retrieval_depth=4,
            mutation_relation=MutationRelation.MEANING_PRESERVING_QUERY,
            history=history,
            exact_parent_signature=self.first,
        )
        self.assertFalse(repeated.new_behavior)
        self.assertEqual(repeated.novelty, 0.0)

    def test_other_relations_never_receive_divergence(self) -> None:
        history = BehaviorHistory(CAMPAIGN, {ROOT: (self.first,)})
        feedback = score_observed_retrieval(
            root_checkpoint_id=ROOT,
            observed_signature=self.second,
            coverage_update=self.coverage_update(1),
            retrieval_depth=4,
            mutation_relation=MutationRelation.UPDATE,
            history=history,
        )
        self.assertIsNone(feedback.divergence)
        self.assertEqual(feedback.behavior_score, feedback.novelty)

    def test_composite_scorer_requires_coverage_update(self) -> None:
        history = BehaviorHistory(CAMPAIGN, {ROOT: (self.first,)})
        with self.assertRaisesRegex(TypeError, "CoverageUpdate"):
            score_observed_retrieval(
                root_checkpoint_id=ROOT,
                observed_signature=self.second,
                coverage_update=1,  # type: ignore[arg-type]
                retrieval_depth=4,
                mutation_relation=MutationRelation.UPDATE,
                history=history,
            )

    def test_merge_clipping_is_ufuzz_only(self) -> None:
        self.assertEqual(coverage_guided_score(4), 4)
        self.assertEqual(bounded_coverage_gain(4, 2), 1.0)
        self.assertEqual(bounded_coverage_gain(1, 2), 0.5)

    def test_behavior_and_final_score_formulas(self) -> None:
        meaning_preserving = behavior_score(
            0.8,
            MutationRelation.MEANING_PRESERVING_QUERY,
            0.4,
        )
        self.assertAlmostEqual(meaning_preserving, 0.6)
        self.assertAlmostEqual(ufuzz_score(0.5, meaning_preserving), 0.55)
        self.assertEqual(behavior_score(0.8, MutationRelation.UPDATE), 0.8)

    def test_score_domains_and_bounds(self) -> None:
        for gain in (0, 1, 4, 100):
            bounded = bounded_coverage_gain(gain, 2)
            self.assertGreaterEqual(bounded, 0.0)
            self.assertLessEqual(bounded, 1.0)
            for novelty in (0.0, 0.5, 1.0):
                behavior = behavior_score(novelty, MutationRelation.UPDATE)
                score = ufuzz_score(bounded, behavior)
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)

        with self.assertRaises(ValueError):
            coverage_guided_score(-1)
        with self.assertRaises(ValueError):
            bounded_coverage_gain(1, 1)
        with self.assertRaises(ValueError):
            behavior_score(1.1, MutationRelation.UPDATE)
        with self.assertRaises(ValueError):
            behavior_score(0.5, MutationRelation.DELETION, 0.5)
        with self.assertRaises(ValueError):
            ufuzz_score(-0.1, 0.5)


if __name__ == "__main__":
    unittest.main()
