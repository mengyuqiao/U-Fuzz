from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import unittest

from ufuzz.backends.base import OperationReceipt
from ufuzz.coverage import (
    CoverageEntryId,
    FrozenCheckpointInventory,
    InitialCoverageEntry,
    InitialEntryLineage,
    LineageRelation,
    LineageResolutionError,
    PhysicalEntryRef,
)
from ufuzz.retrieval_feedback import MutationRelation
from ufuzz.state_contract import (
    AffectedPhysicalTransition,
    AffectedTransitionKind,
    CertificationStatus,
    DescendantStateCertificate,
    LiveLineageState,
    LogicalSeed,
    MaterializationFailure,
    MaterializationFailureKind,
    MaterializationResult,
    MutationOpportunity,
    PhysicalLineageEndpoint,
    PhysicalTransitionCertificate,
    PhysicalTransitionOutcome,
    RealizedMutationArtifact,
    RebindingKind,
    ReplayBinding,
    ReplayRebindingCertificate,
    SeedAdmission,
    SemanticMutationCertificate,
    build_rebound_lineage,
    descendant_states_equivalent,
    validate_opportunity_parent,
    validate_memory_child_contract,
)


class StateContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.campaign = "campaign-1"
        self.root = "checkpoint-1"
        self.e1 = CoverageEntryId(self.campaign, self.root, "e1")
        self.e2 = CoverageEntryId(self.campaign, self.root, "e2")
        self.entries = (
            InitialCoverageEntry(
                self.e1,
                self.root,
                "initial-1",
                {"content": "P"},
                ("source-1",),
            ),
            InitialCoverageEntry(
                self.e2,
                self.root,
                "initial-2",
                {"content": "P"},
                ("source-2",),
            ),
        )
        self.inventory = FrozenCheckpointInventory(
            self.campaign, self.root, self.entries
        )
        self.p1 = self.physical("state-before", "p1")
        self.p2 = self.physical("state-before", "p2")
        self.s1 = self.physical("state-after", "s1")
        self.s2 = self.physical("state-after", "s2")

    def physical(self, state_id: str, backend_entry_id: str) -> PhysicalEntryRef:
        return PhysicalEntryRef(
            self.campaign, self.root, state_id, backend_entry_id
        )

    def endpoint(
        self, physical: PhysicalEntryRef, *ids: CoverageEntryId
    ) -> PhysicalLineageEndpoint:
        return PhysicalLineageEndpoint(physical, tuple(ids))

    def memory_opportunity(
        self,
        *,
        relation: MutationRelation = MutationRelation.UPDATE,
        possible: frozenset[PhysicalTransitionOutcome] | None = None,
        acceptable: frozenset[PhysicalTransitionOutcome] | None = None,
    ) -> MutationOpportunity:
        return MutationOpportunity.create(
            opportunity_id="op-1",
            campaign_id=self.campaign,
            root_checkpoint_id=self.root,
            parent_seed_id="seed-parent",
            relation=relation,
            canonical_target={"region": ["Ada", "employer"]},
            target_lineage=(self.e1,),
            applicability_evidence={"structural_certificate": "semantic-1"},
            generation_constraints={"exact_target": True},
            possible_transition_outcomes=possible
            or frozenset(
                {
                    PhysicalTransitionOutcome.SAME_ID,
                    PhysicalTransitionOutcome.REPLACED,
                    PhysicalTransitionOutcome.UNCHANGED,
                }
            ),
            acceptable_transition_outcomes=acceptable
            or frozenset(
                {
                    PhysicalTransitionOutcome.SAME_ID,
                    PhysicalTransitionOutcome.REPLACED,
                }
            ),
        )

    def semantic_certificate(
        self, opportunity: MutationOpportunity
    ) -> SemanticMutationCertificate:
        return SemanticMutationCertificate(
            "semantic-cert-1",
            opportunity.opportunity_id,
            opportunity.campaign_id,
            opportunity.root_checkpoint_id,
            opportunity.relation,
            True,
            {"obligation": "exact semantic target"},
            "semantic-proof-digest",
        )

    def memory_artifact(
        self, opportunity: MutationOpportunity | None = None
    ) -> RealizedMutationArtifact:
        opportunity = opportunity or self.memory_opportunity()
        return RealizedMutationArtifact(
            "artifact-1",
            opportunity,
            self.semantic_certificate(opportunity),
            "query-parent",
            None,
            "replace_value",
            {
                "target": "Ada/employer",
                "replacement": "Analytical Engines Ltd",
                "reference_time": "2025-01-01T00:00:00Z",
            },
        )

    def affected(
        self,
        kind: AffectedTransitionKind,
        predecessors: tuple[PhysicalLineageEndpoint, ...],
        successors: tuple[PhysicalLineageEndpoint, ...],
        *,
        transition_id: str = "target-transition",
    ) -> AffectedPhysicalTransition:
        return AffectedPhysicalTransition(
            transition_id,
            kind,
            predecessors,
            successors,
            backend_proof_digest=f"proof-{transition_id}",
        )

    def transition_certificate(
        self,
        affected: tuple[AffectedPhysicalTransition, ...],
        outcome: PhysicalTransitionOutcome,
        *,
        semantic_postcondition_satisfied: bool = True,
        certificate_id: str = "physical-cert-1",
        realized_artifact_id: str = "artifact-1",
        semantic_certificate_id: str = "semantic-cert-1",
    ) -> PhysicalTransitionCertificate:
        return PhysicalTransitionCertificate(
            certificate_id,
            realized_artifact_id,
            semantic_certificate_id,
            self.campaign,
            self.root,
            frozenset({self.e1, self.e2}),
            CertificationStatus.CERTIFIED,
            outcome,
            "target-transition",
            affected,
            frozenset(
                ref for item in affected for ref in item.predecessor_refs
            ),
            frozenset(ref for item in affected for ref in item.successor_refs),
            semantic_postcondition_satisfied,
            "complete-backend-proof",
        )

    def descendant(
        self,
        *,
        live: tuple[LiveLineageState, ...] | None = None,
        deleted: frozenset[CoverageEntryId] = frozenset(),
        transitions: tuple[str, ...] = ("physical-cert-1",),
    ) -> DescendantStateCertificate:
        return DescendantStateCertificate(
            "descendant-cert-1",
            "synthetic",
            "config-v1",
            self.campaign,
            self.root,
            frozenset({self.e1, self.e2}),
            live
            if live is not None
            else (
                LiveLineageState((self.e1,), "projection-e1", "provenance-e1"),
                LiveLineageState((self.e2,), "projection-e2", "provenance-e2"),
            ),
            deleted,
            "complete-observable-state",
            "transition-relevant-state",
            transitions,
        )

    def root_rebinding(
        self,
        bindings: tuple[ReplayBinding, ...] | None = None,
    ) -> ReplayRebindingCertificate:
        bindings = bindings or (
            ReplayBinding(self.s1, (self.e1,), "binding-e1"),
            ReplayBinding(self.s2, (self.e2,), "binding-e2"),
        )
        return ReplayRebindingCertificate(
            "rebind-root-1",
            RebindingKind.ROOT,
            "synthetic",
            "config-v1",
            self.campaign,
            self.root,
            frozenset({self.e1, self.e2}),
            bindings,
            tuple(binding.lineage for binding in bindings),
            frozenset(),
            "root-rebinding-proof",
        )

    def test_persistent_objects_are_immutable_and_hash_safe(self) -> None:
        opportunity = self.memory_opportunity()
        artifact = self.memory_artifact(opportunity)
        descendant = self.descendant()
        seed = LogicalSeed(
            "seed-parent",
            self.campaign,
            "synthetic",
            self.root,
            "config-v1",
            {"query_id": "q1", "text": "Where does Ada work?"},
            (artifact,),
            descendant,
            ((self.e1,),),
        )
        for value in (opportunity, artifact, descendant, seed, self.root_rebinding()):
            hash(value)
        with self.assertRaises(FrozenInstanceError):
            seed.backend = "changed"  # type: ignore[misc]

    def test_unresolved_is_not_a_physical_outcome(self) -> None:
        self.assertNotIn("unresolved", {item.value for item in PhysicalTransitionOutcome})
        with self.assertRaises(ValueError):
            PhysicalTransitionOutcome("unresolved")

    def test_certified_requires_observed_outcome(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires exactly one observed"):
            PhysicalTransitionCertificate(
                "cert",
                "artifact-1",
                "semantic-cert-1",
                self.campaign,
                self.root,
                frozenset({self.e1, self.e2}),
                CertificationStatus.CERTIFIED,
                None,
                None,
                (),
                frozenset(),
                frozenset(),
                True,
                "proof",
            )

    def test_unresolved_cannot_claim_observed_outcome(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot claim an observed outcome"):
            PhysicalTransitionCertificate(
                "cert",
                "artifact-1",
                "semantic-cert-1",
                self.campaign,
                self.root,
                frozenset({self.e1, self.e2}),
                CertificationStatus.UNRESOLVED,
                PhysicalTransitionOutcome.SAME_ID,
                None,
                (),
                frozenset(),
                frozenset(),
                None,
                "proof",
                "cannot establish lineage",
            )

    def test_acceptable_outcomes_must_be_subset_of_possible(self) -> None:
        with self.assertRaisesRegex(ValueError, "subset"):
            self.memory_opportunity(
                possible=frozenset({PhysicalTransitionOutcome.SAME_ID}),
                acceptable=frozenset({PhysicalTransitionOutcome.REPLACED}),
            )

    def test_query_opportunity_has_no_physical_transition_contract(self) -> None:
        opportunity = MutationOpportunity.create(
            opportunity_id="query-op",
            campaign_id=self.campaign,
            root_checkpoint_id=self.root,
            parent_seed_id="seed-parent",
            relation=MutationRelation.MEANING_PRESERVING_QUERY,
            canonical_target={"slot": "wording"},
            applicability_evidence={"meaning_preserved": True},
            generation_constraints={"language": "en"},
        )
        self.assertEqual(opportunity.possible_transition_outcomes, frozenset())
        self.assertEqual(opportunity.acceptable_transition_outcomes, frozenset())
        with self.assertRaisesRegex(ValueError, "no physical transition"):
            MutationOpportunity.create(
                opportunity_id="bad-query-op",
                campaign_id=self.campaign,
                root_checkpoint_id=self.root,
                parent_seed_id="seed-parent",
                relation=MutationRelation.MEANING_PRESERVING_QUERY,
                canonical_target={"slot": "wording"},
                applicability_evidence={},
                generation_constraints={},
                possible_transition_outcomes={PhysicalTransitionOutcome.UNCHANGED},
            )

    def test_realized_artifact_requires_successful_semantic_validation(self) -> None:
        opportunity = self.memory_opportunity()
        invalid = replace(self.semantic_certificate(opportunity), valid=False)
        with self.assertRaisesRegex(ValueError, "valid semantic certificate"):
            RealizedMutationArtifact(
                "invalid-artifact",
                opportunity,
                invalid,
                "query-parent",
                None,
                "replace_value",
                {"replacement": "Analytical Engines Ltd"},
            )

    def test_unchanged_cannot_be_acceptable_for_memory_relations(self) -> None:
        for relation in (
            MutationRelation.UPDATE,
            MutationRelation.DELETION,
            MutationRelation.UNRELATED_CHANGE,
        ):
            with self.subTest(relation=relation):
                with self.assertRaisesRegex(ValueError, "UNCHANGED cannot satisfy"):
                    self.memory_opportunity(
                        relation=relation,
                        possible=frozenset({PhysicalTransitionOutcome.UNCHANGED}),
                        acceptable=frozenset({PhysicalTransitionOutcome.UNCHANGED}),
                    )

    def test_operation_receipt_is_not_a_physical_certificate(self) -> None:
        receipt = OperationReceipt(
            "synthetic",
            "state",
            "update",
            True,
            "p1",
            ("p1",),
            (),
            {"id": "p1"},
            {"id": "p1"},
            {},
        )
        self.assertNotIsInstance(receipt, PhysicalTransitionCertificate)
        with self.assertRaises((TypeError, AttributeError)):
            self.transition_certificate(
                (receipt,),  # type: ignore[arg-type]
                PhysicalTransitionOutcome.SAME_ID,
            )

    def test_transition_rejects_new_coverage_id(self) -> None:
        new_id = CoverageEntryId(self.campaign, self.root, "new-id")
        with self.assertRaisesRegex(ValueError, "preserve inherited lineage"):
            self.affected(
                AffectedTransitionKind.REPLACED,
                (self.endpoint(self.p1, self.e1),),
                (self.endpoint(self.s1, new_id),),
            )

        item = self.affected(
            AffectedTransitionKind.DELETED,
            (self.endpoint(self.p1, new_id),),
            (),
        )
        with self.assertRaisesRegex(ValueError, "outside frozen E_0"):
            self.transition_certificate((item,), PhysicalTransitionOutcome.DELETED)

    def test_complete_affected_set_is_required(self) -> None:
        item = self.affected(
            AffectedTransitionKind.SAME_ID_CHANGED,
            (self.endpoint(self.p1, self.e1),),
            (self.endpoint(self.physical("state-after", "p1"), self.e1),),
        )
        with self.assertRaisesRegex(ValueError, "account for all successors"):
            PhysicalTransitionCertificate(
                "cert",
                "artifact-1",
                "semantic-cert-1",
                self.campaign,
                self.root,
                frozenset({self.e1, self.e2}),
                CertificationStatus.CERTIFIED,
                PhysicalTransitionOutcome.SAME_ID,
                "target-transition",
                (item,),
                item.predecessor_refs,
                frozenset(),
                True,
                "proof",
            )

    def test_transition_group_ids_and_physical_classification_are_unique(self) -> None:
        first = self.affected(
            AffectedTransitionKind.DELETED,
            (self.endpoint(self.p1, self.e1),),
            (),
        )
        duplicate_id = self.affected(
            AffectedTransitionKind.DELETED,
            (self.endpoint(self.p2, self.e2),),
            (),
        )
        with self.assertRaisesRegex(ValueError, "transition IDs must be unique"):
            self.transition_certificate(
                (first, duplicate_id),
                PhysicalTransitionOutcome.DELETED,
            )

        duplicate_predecessor = self.affected(
            AffectedTransitionKind.DELETED,
            (self.endpoint(self.p1, self.e1),),
            (),
            transition_id="collateral-transition",
        )
        with self.assertRaisesRegex(ValueError, "predecessor is classified"):
            self.transition_certificate(
                (first, duplicate_predecessor),
                PhysicalTransitionOutcome.DELETED,
            )

    def test_unresolved_object_cannot_also_be_resolved_in_certificate(self) -> None:
        resolved = self.affected(
            AffectedTransitionKind.REPLACED,
            (self.endpoint(self.p1, self.e1),),
            (self.endpoint(self.s1, self.e1),),
        )
        unresolved = AffectedPhysicalTransition(
            "unresolved-collateral",
            AffectedTransitionKind.OTHERWISE_UNRESOLVED,
            (),
            (),
            unresolved_predecessors=(self.s1,),
            backend_proof_digest="unresolved-proof",
        )
        with self.assertRaisesRegex(ValueError, "both resolved and unresolved"):
            PhysicalTransitionCertificate(
                "unresolved-cert",
                "artifact-1",
                "semantic-cert-1",
                self.campaign,
                self.root,
                frozenset({self.e1, self.e2}),
                CertificationStatus.UNRESOLVED,
                None,
                None,
                (resolved, unresolved),
                resolved.predecessor_refs | unresolved.predecessor_refs,
                resolved.successor_refs,
                None,
                "complete-proof",
                "unresolved collateral",
            )

    def test_root_rebinding_maps_every_e0_once(self) -> None:
        certificate = self.root_rebinding()
        lineage = build_rebound_lineage(certificate, self.inventory)
        self.assertEqual(lineage.resolve(self.s1), frozenset({self.e1}))
        self.assertEqual(lineage.resolve(self.s2), frozenset({self.e2}))
        with self.assertRaisesRegex(ValueError, "account for every"):
            ReplayRebindingCertificate(
                "incomplete",
                RebindingKind.ROOT,
                "synthetic",
                "config-v1",
                self.campaign,
                self.root,
                frozenset({self.e1, self.e2}),
                (ReplayBinding(self.s1, (self.e1,), "binding-e1"),),
                ((self.e1,),),
                frozenset(),
                "proof",
            )

    def test_root_rebinding_cannot_mark_ids_deleted(self) -> None:
        with self.assertRaisesRegex(ValueError, "root re-binding cannot"):
            ReplayRebindingCertificate(
                "bad-root",
                RebindingKind.ROOT,
                "synthetic",
                "config-v1",
                self.campaign,
                self.root,
                frozenset({self.e1, self.e2}),
                (ReplayBinding(self.s1, (self.e1,), "binding-e1"),),
                ((self.e1,),),
                frozenset({self.e2}),
                "proof",
            )

    def test_descendant_deletion_is_explicit(self) -> None:
        certificate = ReplayRebindingCertificate(
            "descendant-delete",
            RebindingKind.DESCENDANT,
            "synthetic",
            "config-v1",
            self.campaign,
            self.root,
            frozenset({self.e1, self.e2}),
            (ReplayBinding(self.s1, (self.e1,), "binding-e1"),),
            ((self.e1,),),
            frozenset({self.e2}),
            "proof",
        )
        self.assertEqual(certificate.deleted_ids, frozenset({self.e2}))
        lineage = build_rebound_lineage(certificate, self.inventory)
        self.assertEqual(lineage.coverage_ids, frozenset({self.e1, self.e2}))
        self.assertEqual(lineage.resolve(self.s1), frozenset({self.e1}))

    def test_descendant_rebinding_preserves_split_multiplicity(self) -> None:
        bindings = (
            ReplayBinding(self.s1, (self.e1,), "split-binding-1"),
            ReplayBinding(self.s2, (self.e1,), "split-binding-2"),
            ReplayBinding(
                self.physical("state-after", "s3"),
                (self.e2,),
                "binding-e2",
            ),
        )
        certificate = ReplayRebindingCertificate(
            "descendant-split",
            RebindingKind.DESCENDANT,
            "synthetic",
            "config-v1",
            self.campaign,
            self.root,
            frozenset({self.e1, self.e2}),
            bindings,
            ((self.e1,), (self.e1,), (self.e2,)),
            frozenset(),
            "proof",
        )
        self.assertEqual(
            certificate.expected_live_lineage_multiset.count((self.e1,)),
            2,
        )
        with self.assertRaisesRegex(ValueError, "lineage multiset"):
            replace(
                certificate,
                expected_live_lineage_multiset=((self.e1,), (self.e2,)),
            )

    def test_rebinding_rejects_cross_checkpoint_physical_object(self) -> None:
        other = PhysicalEntryRef(
            self.campaign,
            "other-checkpoint",
            "state-after",
            "s2",
        )
        with self.assertRaisesRegex(ValueError, "another root checkpoint"):
            self.root_rebinding(
                (
                    ReplayBinding(self.s1, (self.e1,), "binding-e1"),
                    ReplayBinding(other, (self.e2,), "binding-e2"),
                )
            )

    def test_certified_replacement_preserves_existing_e0(self) -> None:
        item = self.affected(
            AffectedTransitionKind.REPLACED,
            (self.endpoint(self.p1, self.e1),),
            (self.endpoint(self.s1, self.e1),),
        )
        certificate = self.transition_certificate(
            (item,), PhysicalTransitionOutcome.REPLACED
        )
        self.assertEqual(item.successors[0].lineage, (self.e1,))
        self.assertEqual(certificate.frozen_e0, frozenset({self.e1, self.e2}))

    def test_target_transition_kind_must_match_observed_outcome(self) -> None:
        replacement = self.affected(
            AffectedTransitionKind.REPLACED,
            (self.endpoint(self.p1, self.e1),),
            (self.endpoint(self.s1, self.e1),),
        )
        valid = self.transition_certificate(
            (replacement,), PhysicalTransitionOutcome.REPLACED
        )
        self.assertEqual(valid.observed_outcome, PhysicalTransitionOutcome.REPLACED)
        with self.assertRaisesRegex(ValueError, "disagrees with target transition"):
            self.transition_certificate(
                (replacement,), PhysicalTransitionOutcome.MERGED
            )

    def test_merge_multiple_initialized_lineages_to_one_successor(self) -> None:
        item = self.affected(
            AffectedTransitionKind.MERGE,
            (
                self.endpoint(self.p1, self.e1),
                self.endpoint(self.p2, self.e2),
            ),
            (self.endpoint(self.s1, self.e2, self.e1),),
        )
        certificate = self.transition_certificate(
            (item,), PhysicalTransitionOutcome.MERGED
        )
        self.assertEqual(certificate.observed_outcome, PhysicalTransitionOutcome.MERGED)
        self.assertEqual(item.successors[0].lineage, (self.e1, self.e2))

    def test_split_one_initialized_lineage_to_multiple_successors(self) -> None:
        item = self.affected(
            AffectedTransitionKind.SPLIT,
            (self.endpoint(self.p1, self.e1),),
            (
                self.endpoint(self.s1, self.e1),
                self.endpoint(self.s2, self.e1),
            ),
        )
        certificate = self.transition_certificate(
            (item,), PhysicalTransitionOutcome.SPLIT
        )
        self.assertEqual(certificate.observed_outcome, PhysicalTransitionOutcome.SPLIT)
        with self.assertRaisesRegex(ValueError, "one initialized predecessor"):
            self.affected(
                AffectedTransitionKind.SPLIT,
                (self.endpoint(self.p1, self.e1, self.e2),),
                (
                    self.endpoint(self.s1, self.e1, self.e2),
                    self.endpoint(self.s2, self.e1, self.e2),
                ),
            )

    def test_rebinding_rejects_new_coverage_id(self) -> None:
        new_id = CoverageEntryId(self.campaign, self.root, "new-id")
        with self.assertRaisesRegex(ValueError, "outside frozen E_0"):
            ReplayRebindingCertificate(
                "bad-new-id",
                RebindingKind.DESCENDANT,
                "synthetic",
                "config-v1",
                self.campaign,
                self.root,
                frozenset({self.e1, self.e2}),
                (
                    ReplayBinding(self.s1, (self.e1,), "binding-e1"),
                    ReplayBinding(self.s2, (self.e2, new_id), "binding-e2"),
                ),
                ((self.e1,), (self.e2, new_id)),
                frozenset(),
                "proof",
            )

    def test_duplicate_projection_is_not_binding_evidence(self) -> None:
        # The two initialized projections are deliberately equal.  A binding
        # still requires backend-certified evidence for each exact object.
        self.assertEqual(
            self.entries[0].initial_observable_projection,
            self.entries[1].initial_observable_projection,
        )
        with self.assertRaisesRegex(ValueError, "binding_proof_digest"):
            ReplayBinding(self.s1, (self.e1,), "")
        failure = MaterializationResult[ReplayRebindingCertificate].failed(
            MaterializationFailure(
                MaterializationFailureKind.E0_REBINDING_AMBIGUOUS,
                self.campaign,
                self.root,
                "equal projections lack a stable certified discriminator",
            )
        )
        self.assertIsNone(failure.value)
        self.assertEqual(
            failure.failure.kind,
            MaterializationFailureKind.E0_REBINDING_AMBIGUOUS,
        )

    def test_failed_rebinding_does_not_mutate_existing_lineage(self) -> None:
        existing = InitialEntryLineage(self.campaign, (self.inventory,))
        existing.certify_binding(self.p1, (self.e1,), LineageRelation.INITIAL)
        other_inventory = FrozenCheckpointInventory(
            self.campaign,
            self.root,
            (self.entries[0],),
        )
        with self.assertRaisesRegex(ValueError, "does not equal"):
            build_rebound_lineage(self.root_rebinding(), other_inventory)
        self.assertEqual(existing.resolve(self.p1), frozenset({self.e1}))
        with self.assertRaises(LineageResolutionError):
            existing.resolve(self.s1)

    def test_equal_projection_multiset_different_lineage_is_not_equivalent(self) -> None:
        expected = self.descendant(
            live=(
                LiveLineageState((self.e1,), "projection-P", "source-1"),
                LiveLineageState((self.e2,), "projection-Q", "source-2"),
            )
        )
        swapped = replace(
            expected,
            certificate_id="observed-cert",
            live_lineage_states=(
                LiveLineageState((self.e1,), "projection-Q", "source-2"),
                LiveLineageState((self.e2,), "projection-P", "source-1"),
            ),
        )
        self.assertCountEqual(
            [item.observable_state_digest for item in expected.live_lineage_states],
            [item.observable_state_digest for item in swapped.live_lineage_states],
        )
        self.assertFalse(descendant_states_equivalent(expected, swapped))

    def test_logical_seed_identity_has_no_replay_local_physical_ids(self) -> None:
        artifact = self.memory_artifact()
        descendant = self.descendant()
        seed_one = LogicalSeed(
            "seed-parent",
            self.campaign,
            "synthetic",
            self.root,
            "config-v1",
            {"query_id": "q1", "text": "Where does Ada work?"},
            (artifact,),
            descendant,
            ((self.e1,),),
        )
        # These materializations use different replay-local IDs, but neither
        # identity is part of the persistent logical seed.
        first = self.root_rebinding()
        second = self.root_rebinding(
            (
                ReplayBinding(
                    self.physical("other-state", "other-1"),
                    (self.e1,),
                    "other-proof-1",
                ),
                ReplayBinding(
                    self.physical("other-state", "other-2"),
                    (self.e2,),
                    "other-proof-2",
                ),
            )
        )
        seed_two = replace(seed_one)
        self.assertNotEqual(first, second)
        self.assertEqual(seed_one, seed_two)
        self.assertEqual(seed_one.seed_id, "seed-parent")
        self.assertEqual(seed_two.seed_id, "seed-parent")

    def test_opportunity_must_reference_exact_parent_seed_id(self) -> None:
        parent = LogicalSeed(
            "seed-parent",
            self.campaign,
            "synthetic",
            self.root,
            "config-v1",
            {"query_id": "q1", "text": "Where does Ada work?"},
            (self.memory_artifact(),),
            self.descendant(),
            ((self.e1,),),
        )
        opportunity = self.memory_opportunity()
        validate_opportunity_parent(opportunity, parent)
        with self.assertRaisesRegex(ValueError, "another parent seed"):
            validate_opportunity_parent(
                replace(opportunity, parent_seed_id="another-seed"),
                parent,
            )

    def test_authoritative_parent_signature_is_frozen(self) -> None:
        seed = LogicalSeed(
            "seed-parent",
            self.campaign,
            "synthetic",
            self.root,
            "config-v1",
            {"query_id": "q1", "text": "Where does Ada work?"},
            (self.memory_artifact(),),
            self.descendant(),
            ((self.e1,),),
        )
        with self.assertRaises(FrozenInstanceError):
            seed.authoritative_parent_signature = ((self.e2,),)  # type: ignore[misc]
        distinct_seed = replace(seed, authoritative_parent_signature=((self.e2,),))
        self.assertNotEqual(seed, distinct_seed)
        self.assertEqual(seed.authoritative_parent_signature, ((self.e1,),))

    def test_non_rematerializable_valid_leaf_is_not_retainable(self) -> None:
        admission = SeedAdmission.for_executed_child(
            execution_valid=True, rematerializable=False
        )
        self.assertTrue(admission.execution_valid)
        self.assertFalse(admission.rematerializable)
        self.assertFalse(admission.retainable_as_parent)
        with self.assertRaisesRegex(ValueError, "retainable parent"):
            SeedAdmission(True, False, True)

    def test_memory_child_composition_requires_acceptable_certified_outcome(self) -> None:
        opportunity = self.memory_opportunity()
        artifact = self.memory_artifact(opportunity)
        item = self.affected(
            AffectedTransitionKind.REPLACED,
            (self.endpoint(self.p1, self.e1),),
            (self.endpoint(self.s1, self.e1),),
        )
        transition = self.transition_certificate(
            (item,), PhysicalTransitionOutcome.REPLACED
        )
        descendant = self.descendant()
        validate_memory_child_contract(artifact, transition, descendant)

        drift = replace(transition, semantic_postcondition_satisfied=False)
        with self.assertRaisesRegex(ValueError, "semantic postcondition"):
            validate_memory_child_contract(artifact, drift, descendant)

        different_artifact = replace(
            artifact,
            artifact_id="artifact-2",
            operation_payload={"replacement": "Difference Engine LLC"},
        )
        with self.assertRaisesRegex(ValueError, "another realized artifact"):
            validate_memory_child_contract(
                different_artifact,
                transition,
                descendant,
            )

        different_semantic = replace(
            artifact,
            semantic_certificate=replace(
                artifact.semantic_certificate,
                certificate_id="semantic-cert-2",
            ),
        )
        with self.assertRaisesRegex(ValueError, "another semantic certificate"):
            validate_memory_child_contract(
                different_semantic,
                transition,
                descendant,
            )

        wrong_target_opportunity = replace(
            opportunity,
            target_lineage=(self.e2,),
        )
        wrong_target_artifact = RealizedMutationArtifact(
            "artifact-wrong-target",
            wrong_target_opportunity,
            self.semantic_certificate(wrong_target_opportunity),
            "query-parent",
            None,
            "replace_value",
            {"replacement": "Analytical Engines Ltd"},
        )
        with self.assertRaisesRegex(ValueError, "opportunity target lineage"):
            validate_memory_child_contract(
                wrong_target_artifact,
                replace(
                    transition,
                    realized_artifact_id=wrong_target_artifact.artifact_id,
                ),
                descendant,
            )

    def test_nested_evaluator_only_keys_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "evaluator-only"):
            MutationOpportunity.create(
                opportunity_id="unsafe",
                campaign_id=self.campaign,
                root_checkpoint_id=self.root,
                parent_seed_id="parent",
                relation=MutationRelation.UPDATE,
                canonical_target={"safe": {"gold_answer": "secret"}},
                target_lineage=(self.e1,),
                applicability_evidence={},
                generation_constraints={},
                possible_transition_outcomes={PhysicalTransitionOutcome.SAME_ID},
                acceptable_transition_outcomes={PhysicalTransitionOutcome.SAME_ID},
            )
        with self.assertRaisesRegex(ValueError, "evaluator-only"):
            RealizedMutationArtifact(
                "artifact",
                self.memory_opportunity(),
                self.semantic_certificate(self.memory_opportunity()),
                "query",
                None,
                None,
                {"nested": {"oracle_verdict": True}},
            )


if __name__ == "__main__":
    unittest.main()
