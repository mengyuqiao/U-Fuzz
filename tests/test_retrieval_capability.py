from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import unittest

from ufuzz.coverage import (
    CoverageEligibility,
    CoverageIneligibleError,
    InitialEntryLineage,
    LineageRelation,
    LineageResolutionError,
    LineageStatus,
    PhysicalEntryRef,
)
from ufuzz.retrieval_capability import (
    CapabilityOperation,
    CapabilityResult,
    InitializationEquivalenceError,
    RankedPhysicalRetrieval,
    RetrievableEntryCapability,
    RetrievableEntryInventory,
    RetrievableEntryProjection,
    RetrievableInventoryEntry,
    RetrievalCapabilityBlocker,
    RetrievalCapabilityError,
    assemble_campaign_coverage,
    build_frozen_checkpoint_inventory,
    compare_initialization_inventories,
    resolve_fuzzing_retrieval,
    resolve_initialization_signature,
)
from ufuzz.retrieval_feedback import (
    BehaviorHistory,
    MutationRelation,
    score_observed_retrieval,
)


ROOT = "checkpoint-1"


@dataclass(frozen=True, slots=True)
class _SyntheticObject:
    backend_id: str
    content: str
    provenance_ids: tuple[str, ...] = ()


class _SyntheticRetrievableCapability:
    """Deterministic public-result capability used only for contract tests."""

    backend = "synthetic"
    selected_public_retrieval_api = "Synthetic.search"
    retrieval_object_class = "SyntheticMemory"

    def __init__(
        self,
        objects: tuple[_SyntheticObject, ...],
        *,
        inventory_order: tuple[str, ...] | None = None,
        blocked_inventory_reason: str | None = None,
    ) -> None:
        self.objects = {item.backend_id: item for item in objects}
        self.inventory_order = inventory_order or tuple(self.objects)
        self.blocked_inventory_reason = blocked_inventory_reason
        self.next_retrieval: tuple[str, ...] = ()

    def _physical(
        self,
        backend_id: str,
        *,
        campaign_id: str,
        root_checkpoint_id: str,
        state_id: str,
    ) -> PhysicalEntryRef:
        return PhysicalEntryRef(
            campaign_id,
            root_checkpoint_id,
            state_id,
            backend_id,
        )

    def _projection(self, item: _SyntheticObject) -> RetrievableEntryProjection:
        return RetrievableEntryProjection(
            self.backend,
            self.retrieval_object_class,
            {"content": item.content},
        )

    async def inventory(
        self,
        *,
        campaign_id: str,
        root_checkpoint_id: str,
        state_id: str,
    ) -> CapabilityResult[RetrievableEntryInventory]:
        if self.blocked_inventory_reason is not None:
            return CapabilityResult.failure(
                RetrievalCapabilityBlocker(
                    self.backend,
                    root_checkpoint_id,
                    CapabilityOperation.INVENTORY,
                    self.blocked_inventory_reason,
                )
            )
        entries = tuple(
            RetrievableInventoryEntry(
                self._physical(
                    backend_id,
                    campaign_id=campaign_id,
                    root_checkpoint_id=root_checkpoint_id,
                    state_id=state_id,
                ),
                self._projection(self.objects[backend_id]),
                self.objects[backend_id].provenance_ids,
            )
            for backend_id in self.inventory_order
        )
        return CapabilityResult.success(
            RetrievableEntryInventory(
                backend=self.backend,
                selected_public_retrieval_api=self.selected_public_retrieval_api,
                retrieval_object_class=self.retrieval_object_class,
                retrieval_returnability_basis=(
                    "inventory and search expose the same SyntheticMemory objects"
                ),
                campaign_id=campaign_id,
                root_checkpoint_id=root_checkpoint_id,
                state_id=state_id,
                entries=entries,
            )
        )

    async def retrieve(
        self,
        *,
        campaign_id: str,
        root_checkpoint_id: str,
        state_id: str,
        query: str,
        top_k: int,
    ) -> CapabilityResult[RankedPhysicalRetrieval]:
        del query
        return CapabilityResult.success(
            RankedPhysicalRetrieval(
                backend=self.backend,
                selected_public_retrieval_api=self.selected_public_retrieval_api,
                retrieval_object_class=self.retrieval_object_class,
                campaign_id=campaign_id,
                root_checkpoint_id=root_checkpoint_id,
                state_id=state_id,
                ranked_physical_entries=tuple(
                    self._physical(
                        backend_id,
                        campaign_id=campaign_id,
                        root_checkpoint_id=root_checkpoint_id,
                        state_id=state_id,
                    )
                    for backend_id in self.next_retrieval[:top_k]
                ),
            )
        )

    def replacement_ref(
        self,
        backend_id: str,
        *,
        campaign_id: str,
        root_checkpoint_id: str,
        state_id: str,
    ) -> PhysicalEntryRef:
        return self._physical(
            backend_id,
            campaign_id=campaign_id,
            root_checkpoint_id=root_checkpoint_id,
            state_id=state_id,
        )


class RetrievalCapabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.objects = (
            _SyntheticObject("physical-a", "Alice lives in Boston", ("source-a",)),
            _SyntheticObject("physical-b", "Bob enjoys tennis", ("source-b",)),
        )

    @staticmethod
    def inventory(
        capability: _SyntheticRetrievableCapability,
        *,
        campaign: str = "campaign-1",
        root: str = ROOT,
        state: str = "state-1",
    ) -> RetrievableEntryInventory:
        return asyncio.run(
            capability.inventory(
                campaign_id=campaign,
                root_checkpoint_id=root,
                state_id=state,
            )
        ).require()

    @staticmethod
    def retrieval(
        capability: _SyntheticRetrievableCapability,
        *,
        campaign: str = "campaign-1",
        root: str = ROOT,
        state: str = "state-1",
        top_k: int = 10,
    ) -> RankedPhysicalRetrieval:
        return asyncio.run(
            capability.retrieve(
                campaign_id=campaign,
                root_checkpoint_id=root,
                state_id=state,
                query="query",
                top_k=top_k,
            )
        ).require()

    def test_synthetic_double_satisfies_capability_protocol(self) -> None:
        capability = _SyntheticRetrievableCapability(self.objects)
        self.assertIsInstance(capability, RetrievableEntryCapability)

    def test_equivalence_ignores_enumeration_and_physical_ids(self) -> None:
        reference = self.inventory(_SyntheticRetrievableCapability(self.objects))
        replica_objects = (
            _SyntheticObject("replica-y", "Bob enjoys tennis"),
            _SyntheticObject("replica-x", "Alice lives in Boston"),
        )
        candidate = self.inventory(
            _SyntheticRetrievableCapability(
                replica_objects,
                inventory_order=("replica-y", "replica-x"),
            ),
            campaign="campaign-2",
            state="replica-state",
        )
        result = compare_initialization_inventories(reference, candidate)
        self.assertTrue(result.equivalent)
        self.assertIs(result.require_equivalent(), result)
        self.assertEqual(result.reference_count, 2)
        self.assertEqual(result.candidate_count, 2)
        self.assertEqual(result.missing_from_candidate, ())
        self.assertEqual(result.excess_in_candidate, ())

    def test_equivalence_preserves_projection_multiplicity(self) -> None:
        reference = self.inventory(
            _SyntheticRetrievableCapability(
                (
                    _SyntheticObject("a1", "P"),
                    _SyntheticObject("a2", "P"),
                    _SyntheticObject("a3", "Q"),
                )
            )
        )
        candidate = self.inventory(
            _SyntheticRetrievableCapability(
                (_SyntheticObject("b1", "P"), _SyntheticObject("b2", "Q"))
            ),
            campaign="campaign-2",
            state="state-2",
        )
        result = compare_initialization_inventories(reference, candidate)
        self.assertFalse(result.equivalent)
        with self.assertRaises(InitializationEquivalenceError):
            result.require_equivalent()
        self.assertEqual(result.missing_from_candidate[0].count, 1)
        self.assertEqual(result.excess_in_candidate, ())

    def test_unequal_projection_is_inequivalent(self) -> None:
        reference = self.inventory(_SyntheticRetrievableCapability(self.objects))
        candidate = self.inventory(
            _SyntheticRetrievableCapability(
                (
                    _SyntheticObject("x", "Alice lives in Boston"),
                    _SyntheticObject("y", "Bob enjoys golf"),
                )
            ),
            campaign="campaign-2",
            state="state-2",
        )
        result = compare_initialization_inventories(reference, candidate)
        self.assertFalse(result.equivalent)
        self.assertEqual(sum(v.count for v in result.missing_from_candidate), 1)
        self.assertEqual(sum(v.count for v in result.excess_in_candidate), 1)

    def test_explicit_inventory_capability_blocker(self) -> None:
        result = asyncio.run(
            _SyntheticRetrievableCapability(
                self.objects,
                blocked_inventory_reason="public result inventory cannot be enumerated",
            ).inventory(
                campaign_id="campaign-1",
                root_checkpoint_id=ROOT,
                state_id="state-1",
            )
        )
        self.assertFalse(result.supported)
        with self.assertRaises(RetrievalCapabilityError):
            result.require()

    def test_inventory_build_preserves_equal_content_as_distinct_entries(self) -> None:
        inventory = self.inventory(
            _SyntheticRetrievableCapability(
                (
                    _SyntheticObject("physical-1", "same"),
                    _SyntheticObject("physical-2", "same"),
                )
            )
        )
        built = build_frozen_checkpoint_inventory(inventory)
        self.assertEqual(len(built.inventory.entries), 2)
        self.assertEqual(len(built.inventory.coverage_ids), 2)
        self.assertEqual(len(built.initial_bindings), 2)
        self.assertTrue(
            all(
                binding.relation is LineageRelation.INITIAL
                and binding.status is LineageStatus.CERTIFIED
                and len(binding.coverage_ids) == 1
                for binding in built.initial_bindings
            )
        )
        self.assertEqual(built.checkpoint_coverage_state.reached_count, 0)

    def test_duplicate_physical_inventory_identity_is_rejected(self) -> None:
        physical = PhysicalEntryRef("campaign-1", ROOT, "state-1", "same-id")
        projection = RetrievableEntryProjection(
            "synthetic", "SyntheticMemory", {"content": "same"}
        )
        entry = RetrievableInventoryEntry(physical, projection)
        with self.assertRaisesRegex(ValueError, "duplicate physical identity"):
            RetrievableEntryInventory(
                "synthetic",
                "Synthetic.search",
                "SyntheticMemory",
                "same public result objects",
                "campaign-1",
                ROOT,
                "state-1",
                (entry, entry),
            )

    def test_ranked_resolution_and_post_execution_scoring_end_to_end(self) -> None:
        capability = _SyntheticRetrievableCapability(self.objects)
        inventory = self.inventory(capability)
        built = build_frozen_checkpoint_inventory(inventory)
        capability.next_retrieval = ("physical-b", "physical-a")
        retrieval = self.retrieval(capability)
        observation = resolve_fuzzing_retrieval(
            retrieval,
            built.lineage,
            built.checkpoint_coverage_state,
        )
        self.assertEqual(
            tuple(binding.physical_entry.backend_entry_id for binding in observation.resolved_lineages),
            ("physical-b", "physical-a"),
        )
        self.assertEqual(observation.coverage_update.raw_gain, 2)
        self.assertEqual(
            observation.coverage_update.retrieved_ids,
            frozenset(
                coverage_id
                for binding in observation.resolved_lineages
                for coverage_id in binding.coverage_ids
            ),
        )
        history = BehaviorHistory("campaign-1")
        feedback = score_observed_retrieval(
            root_checkpoint_id=ROOT,
            observed_signature=observation.signature,
            coverage_update=observation.coverage_update,
            retrieval_depth=2,
            mutation_relation=MutationRelation.UPDATE,
            history=history,
        )
        self.assertEqual(feedback.signature, observation.signature)
        self.assertEqual(feedback.raw_coverage_gain, 2)

    def test_merge_is_atomic_in_signature_and_expanded_for_coverage(self) -> None:
        capability = _SyntheticRetrievableCapability(self.objects)
        built = build_frozen_checkpoint_inventory(self.inventory(capability))
        sources = tuple(binding.physical_entry for binding in built.initial_bindings)
        merged = capability.replacement_ref(
            "physical-merge",
            campaign_id="campaign-1",
            root_checkpoint_id=ROOT,
            state_id="state-1",
        )
        merged_binding = built.lineage.certify_merge(sources, merged)
        retrieval = RankedPhysicalRetrieval(
            "synthetic",
            "Synthetic.search",
            "SyntheticMemory",
            "campaign-1",
            ROOT,
            "state-1",
            (merged,),
        )
        observation = resolve_fuzzing_retrieval(
            retrieval,
            built.lineage,
            built.checkpoint_coverage_state,
        )
        self.assertEqual(len(observation.signature), 1)
        self.assertEqual(observation.signature[0], tuple(sorted(merged_binding.coverage_ids)))
        self.assertEqual(len(observation.coverage_update.retrieved_ids), 2)
        self.assertEqual(observation.coverage_update.raw_gain, 2)

    def test_same_id_replacement_and_deletion_preserve_e0(self) -> None:
        capability = _SyntheticRetrievableCapability(self.objects)
        built = build_frozen_checkpoint_inventory(self.inventory(capability))
        original = built.initial_bindings[0].physical_entry
        initial_ids = built.lineage.resolve(original)
        same_id = built.lineage.certify_same_id_update(original)
        self.assertEqual(same_id.coverage_ids, initial_ids)
        replacement = capability.replacement_ref(
            "replacement-a",
            campaign_id="campaign-1",
            root_checkpoint_id=ROOT,
            state_id="state-1",
        )
        replacement_binding = built.lineage.certify_replacement(original, replacement)
        self.assertEqual(replacement_binding.coverage_ids, initial_ids)
        denominator = built.checkpoint_coverage_state.denominator_count
        built.lineage.mark_deleted(original)
        self.assertTrue(built.lineage.is_deleted(original))
        self.assertEqual(
            built.checkpoint_coverage_state.denominator_count,
            denominator,
        )

    def test_unresolved_and_uncertifiable_objects_reject_whole_observation(self) -> None:
        capability = _SyntheticRetrievableCapability(self.objects)
        built = build_frozen_checkpoint_inventory(self.inventory(capability))
        cases = (
            ("missing", None),
            ("inapplicable", LineageStatus.INAPPLICABLE),
            ("capability-failure", LineageStatus.CAPABILITY_FAILURE),
        )
        for backend_id, status in cases:
            with self.subTest(status=status):
                unknown = capability.replacement_ref(
                    backend_id,
                    campaign_id="campaign-1",
                    root_checkpoint_id=ROOT,
                    state_id="state-1",
                )
                if status is not None:
                    built.lineage.mark_uncertifiable(
                        unknown,
                        status=status,
                        reason="cannot certify physical-to-initial lineage",
                    )
                retrieval = RankedPhysicalRetrieval(
                    "synthetic",
                    "Synthetic.search",
                    "SyntheticMemory",
                    "campaign-1",
                    ROOT,
                    "state-1",
                    (built.initial_bindings[0].physical_entry, unknown),
                )
                with self.assertRaises(LineageResolutionError):
                    resolve_fuzzing_retrieval(
                        retrieval,
                        built.lineage,
                        built.checkpoint_coverage_state,
                    )

    def test_empty_retrieval_is_valid_but_empty_e0_is_ineligible(self) -> None:
        capability = _SyntheticRetrievableCapability(self.objects)
        built = build_frozen_checkpoint_inventory(self.inventory(capability))
        empty_retrieval = self.retrieval(capability)
        observation = resolve_fuzzing_retrieval(
            empty_retrieval,
            built.lineage,
            built.checkpoint_coverage_state,
        )
        self.assertEqual(observation.signature, ())
        self.assertEqual(observation.coverage_update.retrieved_ids, frozenset())
        self.assertEqual(observation.coverage_update.raw_gain, 0)

        another_campaign = build_frozen_checkpoint_inventory(
            self.inventory(
                _SyntheticRetrievableCapability(self.objects),
                campaign="campaign-2",
                state="state-2",
            )
        )
        with self.assertRaisesRegex(ValueError, "coverage state belongs"):
            resolve_fuzzing_retrieval(
                empty_retrieval,
                built.lineage,
                another_campaign.checkpoint_coverage_state,
            )

        empty_capability = _SyntheticRetrievableCapability(())
        empty_build = build_frozen_checkpoint_inventory(self.inventory(empty_capability))
        self.assertIs(
            empty_build.checkpoint_coverage_state.eligibility,
            CoverageEligibility.RQ1_INELIGIBLE_EMPTY_INVENTORY,
        )
        with self.assertRaises(CoverageIneligibleError):
            resolve_fuzzing_retrieval(
                self.retrieval(empty_capability),
                empty_build.lineage,
                empty_build.checkpoint_coverage_state,
            )

    def test_initialization_signature_never_mutates_coverage(self) -> None:
        capability = _SyntheticRetrievableCapability(self.objects)
        built = build_frozen_checkpoint_inventory(self.inventory(capability))
        capability.next_retrieval = ("physical-a",)
        initial = built.checkpoint_coverage_state
        resolved = resolve_initialization_signature(
            self.retrieval(capability),
            built.lineage,
        )
        self.assertNotEqual(resolved.signature, ())
        self.assertEqual(initial.reached_count, 0)
        self.assertEqual(built.checkpoint_coverage_state, initial)

    def test_cross_campaign_and_checkpoint_physical_retrievals_are_rejected(self) -> None:
        capability = _SyntheticRetrievableCapability(self.objects)
        built = build_frozen_checkpoint_inventory(self.inventory(capability))
        for physical in (
            PhysicalEntryRef("campaign-2", ROOT, "state-1", "physical-a"),
            PhysicalEntryRef("campaign-1", "checkpoint-2", "state-1", "physical-a"),
        ):
            retrieval = RankedPhysicalRetrieval(
                "synthetic",
                "Synthetic.search",
                "SyntheticMemory",
                physical.campaign_id,
                physical.root_checkpoint_id,
                "state-1",
                (physical,),
            )
            with self.assertRaises((ValueError, LineageResolutionError)):
                resolve_fuzzing_retrieval(
                    retrieval,
                    built.lineage,
                    built.checkpoint_coverage_state,
                )

    def test_resolved_observation_rejects_inconsistent_derived_outputs(self) -> None:
        capability = _SyntheticRetrievableCapability(self.objects)
        built = build_frozen_checkpoint_inventory(self.inventory(capability))
        capability.next_retrieval = ("physical-a",)
        observation = resolve_fuzzing_retrieval(
            self.retrieval(capability),
            built.lineage,
            built.checkpoint_coverage_state,
        )
        with self.assertRaisesRegex(ValueError, "signature is inconsistent"):
            replace(observation, signature=())
        mismatched_update = replace(
            observation.coverage_update,
            retrieved_ids=frozenset(),
        )
        with self.assertRaisesRegex(ValueError, "retrieved IDs differ"):
            replace(observation, coverage_update=mismatched_update)

        wrong_gain = replace(
            observation.coverage_update,
            raw_gain=observation.coverage_update.raw_gain + 1,
        )
        with self.assertRaisesRegex(ValueError, "raw gain"):
            replace(observation, coverage_update=wrong_gain)

        other_id = next(
            coverage_id
            for coverage_id in built.inventory.coverage_ids
            if coverage_id not in observation.coverage_update.retrieved_ids
        )
        wrong_newly_reached = replace(
            observation.coverage_update,
            newly_reached_ids=frozenset({other_id}),
            raw_gain=1,
        )
        with self.assertRaisesRegex(ValueError, "subset of retrieved"):
            replace(observation, coverage_update=wrong_newly_reached)

        previous_state = built.checkpoint_coverage_state
        unreached_update = replace(
            observation.coverage_update,
            state=previous_state,
        )
        with self.assertRaisesRegex(ValueError, "reached in the next"):
            replace(observation, coverage_update=unreached_update)

    def test_campaign_e0_is_checkpoint_disjoint_union_without_semantic_dedup(self) -> None:
        first = build_frozen_checkpoint_inventory(
            self.inventory(
                _SyntheticRetrievableCapability(
                    (_SyntheticObject("first-physical", "identical content"),)
                ),
                root="checkpoint-1",
                state="state-1",
            )
        )
        second = build_frozen_checkpoint_inventory(
            self.inventory(
                _SyntheticRetrievableCapability(
                    (_SyntheticObject("second-physical", "identical content"),)
                ),
                root="checkpoint-2",
                state="state-2",
            )
        )
        campaign = assemble_campaign_coverage("campaign-1", (first, second))
        self.assertEqual(campaign.coverage_state.denominator_count, 2)
        self.assertEqual(len(campaign.initial_ids), 2)
        self.assertEqual(
            {coverage_id.root_checkpoint_id for coverage_id in campaign.initial_ids},
            {"checkpoint-1", "checkpoint-2"},
        )
        self.assertIs(campaign.coverage_state.eligibility, CoverageEligibility.ELIGIBLE)

    def test_campaign_empty_inventory_rule_uses_union_over_checkpoints(self) -> None:
        empty_first = build_frozen_checkpoint_inventory(
            self.inventory(
                _SyntheticRetrievableCapability(()),
                root="checkpoint-1",
                state="state-1",
            )
        )
        nonempty_second = build_frozen_checkpoint_inventory(
            self.inventory(
                _SyntheticRetrievableCapability(
                    (
                        _SyntheticObject("second-a", "A"),
                        _SyntheticObject("second-b", "B"),
                    )
                ),
                root="checkpoint-2",
                state="state-2",
            )
        )
        mixed = assemble_campaign_coverage(
            "campaign-1",
            (empty_first, nonempty_second),
        )
        self.assertIs(mixed.coverage_state.eligibility, CoverageEligibility.ELIGIBLE)
        self.assertEqual(mixed.coverage_state.denominator_count, 2)

        empty_second = build_frozen_checkpoint_inventory(
            self.inventory(
                _SyntheticRetrievableCapability(()),
                root="checkpoint-2",
                state="state-2",
            )
        )
        all_empty = assemble_campaign_coverage(
            "campaign-1",
            (empty_first, empty_second),
        )
        self.assertIs(
            all_empty.coverage_state.eligibility,
            CoverageEligibility.RQ1_INELIGIBLE_EMPTY_INVENTORY,
        )
        self.assertIsNone(all_empty.coverage_state.cov_fraction)

    def test_campaign_coverage_accumulates_across_checkpoint_executions(self) -> None:
        first_capability = _SyntheticRetrievableCapability(
            (_SyntheticObject("first", "first checkpoint"),)
        )
        second_capability = _SyntheticRetrievableCapability(
            (_SyntheticObject("second", "second checkpoint"),)
        )
        first = build_frozen_checkpoint_inventory(
            self.inventory(
                first_capability,
                root="checkpoint-1",
                state="state-1",
            )
        )
        second = build_frozen_checkpoint_inventory(
            self.inventory(
                second_capability,
                root="checkpoint-2",
                state="state-2",
            )
        )
        campaign = assemble_campaign_coverage("campaign-1", (first, second))
        first_capability.next_retrieval = ("first",)
        first_observation = resolve_fuzzing_retrieval(
            self.retrieval(
                first_capability,
                root="checkpoint-1",
                state="state-1",
            ),
            campaign.lineage,
            campaign.coverage_state,
        )
        self.assertEqual(first_observation.coverage_update.state.reached_count, 1)
        self.assertEqual(first_observation.coverage_update.state.denominator_count, 2)

        second_capability.next_retrieval = ("second",)
        second_observation = resolve_fuzzing_retrieval(
            self.retrieval(
                second_capability,
                root="checkpoint-2",
                state="state-2",
            ),
            campaign.lineage,
            first_observation.coverage_update.state,
        )
        self.assertEqual(second_observation.coverage_update.raw_gain, 1)
        self.assertEqual(second_observation.coverage_update.state.reached_count, 2)
        self.assertEqual(second_observation.coverage_update.state.denominator_count, 2)
        self.assertEqual(second_observation.coverage_update.state.cov_fraction, 1.0)

    def test_campaign_lineage_cannot_cross_checkpoint_identity(self) -> None:
        first = build_frozen_checkpoint_inventory(
            self.inventory(
                _SyntheticRetrievableCapability(
                    (_SyntheticObject("first", "first"),)
                ),
                root="checkpoint-1",
                state="state-1",
            )
        )
        second = build_frozen_checkpoint_inventory(
            self.inventory(
                _SyntheticRetrievableCapability(
                    (_SyntheticObject("second", "second"),)
                ),
                root="checkpoint-2",
                state="state-2",
            )
        )
        campaign = assemble_campaign_coverage("campaign-1", (first, second))
        checkpoint_two_id = next(iter(second.inventory.coverage_ids))
        checkpoint_one_physical = PhysicalEntryRef(
            "campaign-1",
            "checkpoint-1",
            "state-1",
            "cross-checkpoint-attempt",
        )
        with self.assertRaisesRegex(ValueError, "cannot cross root checkpoints"):
            campaign.lineage.certify_binding(
                checkpoint_one_physical,
                (checkpoint_two_id,),
                LineageRelation.INITIAL,
            )

    def test_campaign_assembly_rejects_another_campaign(self) -> None:
        other = build_frozen_checkpoint_inventory(
            self.inventory(
                _SyntheticRetrievableCapability(self.objects),
                campaign="campaign-2",
                root="checkpoint-1",
                state="state-2",
            )
        )
        with self.assertRaisesRegex(ValueError, "another campaign"):
            assemble_campaign_coverage("campaign-1", (other,))


if __name__ == "__main__":
    unittest.main()
