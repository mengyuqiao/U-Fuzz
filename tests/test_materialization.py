from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import unittest

from ufuzz.coverage import (
    CoverageEntryId,
    CoverageState,
    FrozenCheckpointInventory,
    InitialCoverageEntry,
    PhysicalEntryRef,
)
from ufuzz.materialization import (
    CertifiedMaterialization,
    EphemeralMaterializedState,
    MaterializationProtocol,
    RootMaterialization,
    RootMaterializationRequest,
    TransitionReplay,
    apply_certified_transition,
    materialize_and_observe,
    materialize_seed,
    observe_materialization,
)
from ufuzz.retrieval_capability import RankedPhysicalRetrieval
from ufuzz.retrieval_feedback import BehaviorHistory, MutationRelation
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
)


@dataclass(slots=True)
class _Record:
    lineage: tuple[CoverageEntryId, ...]
    value: str


@dataclass(slots=True)
class _SyntheticHandle:
    state_id: str
    records: dict[str, _Record]
    discarded: bool = False


class _SyntheticMaterializer(MaterializationProtocol[_SyntheticHandle]):
    backend = "synthetic"
    frozen_configuration_id = "synthetic-config-v1"

    def __init__(
        self,
        *,
        campaign_id: str,
        root_checkpoint_id: str,
        e1: CoverageEntryId,
        e2: CoverageEntryId,
        root_failure: MaterializationFailureKind | None = None,
        root_binding_state_mismatch: bool = False,
        root_inequivalent: bool = False,
        corrupt_certificate_index: int | None = None,
        semantic_drift_index: int | None = None,
        replay_failure_index: int | None = None,
        replay_failure_kind: MaterializationFailureKind = (
            MaterializationFailureKind.TRANSITION_REPLAY_DIVERGED
        ),
        fail_after_mutation: bool = False,
        new_state_transition_index: int | None = None,
        wrong_successor_state_index: int | None = None,
        final_inequivalent: bool = False,
        final_lineage_inconsistent: bool = False,
        extra_final_transition: bool = False,
        unknown_retrieval: bool = False,
        retrieval_scope_error: str | None = None,
        retrieval_failure: MaterializationFailureKind | None = None,
        retrieval_only_lineage: tuple[CoverageEntryId, ...] | None = None,
    ) -> None:
        self.campaign_id = campaign_id
        self.root_checkpoint_id = root_checkpoint_id
        self.e1 = e1
        self.e2 = e2
        self.root_failure = root_failure
        self.root_binding_state_mismatch = root_binding_state_mismatch
        self.root_inequivalent = root_inequivalent
        self.corrupt_certificate_index = corrupt_certificate_index
        self.semantic_drift_index = semantic_drift_index
        self.replay_failure_index = replay_failure_index
        self.replay_failure_kind = replay_failure_kind
        self.fail_after_mutation = fail_after_mutation
        self.new_state_transition_index = new_state_transition_index
        self.wrong_successor_state_index = wrong_successor_state_index
        self.final_inequivalent = final_inequivalent
        self.final_lineage_inconsistent = final_lineage_inconsistent
        self.extra_final_transition = extra_final_transition
        self.unknown_retrieval = unknown_retrieval
        self.retrieval_scope_error = retrieval_scope_error
        self.retrieval_failure = retrieval_failure
        self.retrieval_only_lineage = retrieval_only_lineage
        self.run_count = 0
        self.root_requests: list[RootMaterializationRequest] = []
        self.executed_artifact_ids: list[str] = []
        self.seen_query_artifacts: list[object] = []
        self.discarded_handles: list[_SyntheticHandle] = []
        self.retired_handles: list[_SyntheticHandle] = []
        self.last_handle: _SyntheticHandle | None = None

    def _failure(
        self, kind: MaterializationFailureKind, reason: str
    ) -> MaterializationResult:
        return MaterializationResult.failed(
            MaterializationFailure(
                kind,
                self.campaign_id,
                self.root_checkpoint_id,
                reason,
            )
        )

    def _physical(self, state_id: str, backend_entry_id: str) -> PhysicalEntryRef:
        return PhysicalEntryRef(
            self.campaign_id,
            self.root_checkpoint_id,
            state_id,
            backend_entry_id,
        )

    def _descendant(
        self,
        records: dict[str, _Record],
        transition_ids: tuple[str, ...],
    ) -> DescendantStateCertificate:
        live_ids = frozenset(
            coverage_id
            for record in records.values()
            for coverage_id in record.lineage
        )
        frozen_e0 = frozenset({self.e1, self.e2})
        live_states = tuple(
            LiveLineageState(
                record.lineage,
                f"value:{record.value}",
                "provenance:"
                + "+".join(item.opaque_id for item in record.lineage),
            )
            for record in records.values()
        )
        semantic_state = sorted(
            (
                tuple(item.opaque_id for item in record.lineage),
                record.value,
            )
            for record in records.values()
        )
        return DescendantStateCertificate(
            certificate_id="observed-descendant",
            backend=self.backend,
            frozen_configuration_id=self.frozen_configuration_id,
            campaign_id=self.campaign_id,
            root_checkpoint_id=self.root_checkpoint_id,
            frozen_e0=frozen_e0,
            live_lineage_states=live_states,
            deleted_ids=frozen_e0.difference(live_ids),
            observable_state_digest=repr(semantic_state),
            transition_state_digest=(
                "root" if not transition_ids else "path:" + ",".join(transition_ids)
            ),
            accepted_transition_certificate_ids=transition_ids,
        )

    async def materialize_root(
        self, request: RootMaterializationRequest
    ) -> MaterializationResult[RootMaterialization[_SyntheticHandle]]:
        if not isinstance(request, RootMaterializationRequest):
            raise TypeError("root materializer received descendant state")
        if any(
            hasattr(request, name)
            for name in (
                "current_query_artifact",
                "memory_transition_artifacts",
                "authoritative_parent_signature",
                "expected_descendant_state",
            )
        ):
            raise AssertionError("root request exposes descendant/search state")
        self.root_requests.append(request)
        if self.root_failure is not None:
            return self._failure(self.root_failure, "configured root failure")
        self.run_count += 1
        state_id = f"synthetic-state-{self.run_count}"
        handle = _SyntheticHandle(
            state_id,
            {
                f"run-{self.run_count}-p1": _Record((self.e1,), "initial-e1"),
                f"run-{self.run_count}-p2": _Record((self.e2,), "initial-e2"),
            },
        )
        self.last_handle = handle
        state = EphemeralMaterializedState(
            handle,
            self.backend,
            self.frozen_configuration_id,
            self.campaign_id,
            self.root_checkpoint_id,
            state_id,
        )
        bindings = tuple(
            ReplayBinding(
                self._physical(
                    "another-materialized-state"
                    if self.root_binding_state_mismatch
                    else state_id,
                    backend_id,
                ),
                record.lineage,
                f"root-proof:{backend_id}",
            )
            for backend_id, record in handle.records.items()
        )
        rebinding = ReplayRebindingCertificate(
            "synthetic-root-rebinding",
            RebindingKind.ROOT,
            self.backend,
            self.frozen_configuration_id,
            self.campaign_id,
            self.root_checkpoint_id,
            frozenset({self.e1, self.e2}),
            bindings,
            tuple(item.lineage for item in bindings),
            frozenset(),
            "complete-root-proof",
        )
        observed = self._descendant(handle.records, ())
        if self.root_inequivalent:
            observed = replace(
                observed,
                observable_state_digest="inequivalent-root-state",
            )
        return MaterializationResult.success(
            RootMaterialization(state, rebinding, observed)
        )

    def _find_exact_lineage(
        self,
        handle: _SyntheticHandle,
        lineage: tuple[CoverageEntryId, ...],
    ) -> tuple[str, _Record]:
        matches = [
            (backend_id, record)
            for backend_id, record in handle.records.items()
            if record.lineage == lineage
        ]
        if len(matches) != 1:
            raise RuntimeError(f"expected one live record for lineage {lineage!r}")
        return matches[0]

    async def replay_transition(
        self,
        state: EphemeralMaterializedState[_SyntheticHandle],
        artifact: RealizedMutationArtifact,
    ) -> MaterializationResult[TransitionReplay[_SyntheticHandle]]:
        index = len(self.executed_artifact_ids)
        self.executed_artifact_ids.append(artifact.artifact_id)
        if self.replay_failure_index == index and not self.fail_after_mutation:
            return self._failure(self.replay_failure_kind, "configured replay failure")

        source_handle = state.handle
        returns_new_state = self.new_state_transition_index == index
        if returns_new_state:
            successor_state_id = f"{state.state_id}-successor-{index}"
            handle = _SyntheticHandle(
                successor_state_id,
                dict(source_handle.records),
            )
            result_state = EphemeralMaterializedState(
                handle,
                self.backend,
                self.frozen_configuration_id,
                self.campaign_id,
                self.root_checkpoint_id,
                successor_state_id,
            )
        else:
            handle = source_handle
            result_state = state
        operation = artifact.operation_payload["operation"]
        target_id = artifact.operation_payload["target"]
        target = self.e1 if target_id == "e1" else self.e2
        predecessor_id, predecessor_record = self._find_exact_lineage(
            source_handle, (target,)
        )
        predecessor = PhysicalLineageEndpoint(
            self._physical(state.state_id, predecessor_id),
            predecessor_record.lineage,
        )
        predecessors = (predecessor,)
        consumed_ids = {predecessor_id}
        certified_successor_state_id = (
            "wrong-successor-state"
            if self.wrong_successor_state_index == index
            else result_state.state_id
        )
        successors: tuple[PhysicalLineageEndpoint, ...]
        if operation == "same_id":
            handle.records[predecessor_id] = _Record(
                predecessor_record.lineage,
                artifact.operation_payload["value"],
            )
            successors = (
                PhysicalLineageEndpoint(
                    self._physical(certified_successor_state_id, predecessor_id),
                    predecessor_record.lineage,
                ),
            )
            kind = AffectedTransitionKind.SAME_ID_CHANGED
            outcome = PhysicalTransitionOutcome.SAME_ID
        elif operation == "replace":
            del handle.records[predecessor_id]
            replacement_id = f"run-{self.run_count}-r9"
            handle.records[replacement_id] = _Record(
                predecessor_record.lineage,
                artifact.operation_payload["value"],
            )
            successors = (
                PhysicalLineageEndpoint(
                    self._physical(certified_successor_state_id, replacement_id),
                    predecessor_record.lineage,
                ),
            )
            kind = AffectedTransitionKind.REPLACED
            outcome = PhysicalTransitionOutcome.REPLACED
        elif operation == "delete":
            del handle.records[predecessor_id]
            successors = ()
            kind = AffectedTransitionKind.DELETED
            outcome = PhysicalTransitionOutcome.DELETED
        elif operation == "merge":
            other = self.e2 if target is self.e1 else self.e1
            other_id, other_record = self._find_exact_lineage(source_handle, (other,))
            other_predecessor = PhysicalLineageEndpoint(
                self._physical(state.state_id, other_id),
                other_record.lineage,
            )
            consumed_ids.add(other_id)
            del handle.records[predecessor_id]
            del handle.records[other_id]
            merged_id = f"run-{self.run_count}-merged"
            merged_lineage = tuple(sorted({target, other}))
            handle.records[merged_id] = _Record(
                merged_lineage,
                artifact.operation_payload["value"],
            )
            predecessors = (predecessor, other_predecessor)
            successors = (
                PhysicalLineageEndpoint(
                    self._physical(certified_successor_state_id, merged_id),
                    merged_lineage,
                ),
            )
            kind = AffectedTransitionKind.MERGE
            outcome = PhysicalTransitionOutcome.MERGED
        elif operation == "split":
            del handle.records[predecessor_id]
            first_id = f"run-{self.run_count}-split-a"
            second_id = f"run-{self.run_count}-split-b"
            handle.records[first_id] = _Record(
                predecessor_record.lineage,
                artifact.operation_payload["first_value"],
            )
            handle.records[second_id] = _Record(
                predecessor_record.lineage,
                artifact.operation_payload["second_value"],
            )
            successors = (
                PhysicalLineageEndpoint(
                    self._physical(certified_successor_state_id, first_id),
                    predecessor_record.lineage,
                ),
                PhysicalLineageEndpoint(
                    self._physical(certified_successor_state_id, second_id),
                    predecessor_record.lineage,
                ),
            )
            kind = AffectedTransitionKind.SPLIT
            outcome = PhysicalTransitionOutcome.SPLIT
        else:
            raise RuntimeError(f"unsupported synthetic operation {operation!r}")

        affected = AffectedPhysicalTransition(
            f"target:{artifact.artifact_id}",
            kind,
            predecessors,
            successors,
            backend_proof_digest=f"synthetic-proof:{artifact.artifact_id}",
        )
        affected_groups = [affected]
        if returns_new_state:
            for backend_id, record in source_handle.records.items():
                if backend_id in consumed_ids:
                    continue
                affected_groups.append(
                    AffectedPhysicalTransition(
                        f"unchanged:{artifact.artifact_id}:{backend_id}",
                        AffectedTransitionKind.UNCHANGED_LINEAGE,
                        (
                            PhysicalLineageEndpoint(
                                self._physical(state.state_id, backend_id),
                                record.lineage,
                            ),
                        ),
                        (
                            PhysicalLineageEndpoint(
                                self._physical(result_state.state_id, backend_id),
                                record.lineage,
                            ),
                        ),
                        backend_proof_digest=(
                            f"synthetic-unchanged:{artifact.artifact_id}:{backend_id}"
                        ),
                    )
                )
        all_predecessors = frozenset(
            physical
            for group in affected_groups
            for physical in group.predecessor_refs
        )
        all_successors = frozenset(
            physical
            for group in affected_groups
            for physical in group.successor_refs
        )
        certificate = PhysicalTransitionCertificate(
            certificate_id=f"cert-{artifact.artifact_id}",
            realized_artifact_id=artifact.artifact_id,
            semantic_certificate_id=artifact.semantic_certificate.certificate_id,
            campaign_id=self.campaign_id,
            root_checkpoint_id=self.root_checkpoint_id,
            frozen_e0=frozenset({self.e1, self.e2}),
            status=CertificationStatus.CERTIFIED,
            observed_outcome=outcome,
            target_transition_id=affected.transition_id,
            affected=tuple(affected_groups),
            observed_affected_predecessors=all_predecessors,
            observed_affected_successors=all_successors,
            semantic_postcondition_satisfied=(
                self.semantic_drift_index != index
            ),
            backend_proof_digest=f"complete:{artifact.artifact_id}",
        )
        if self.corrupt_certificate_index == index:
            certificate = replace(
                certificate,
                realized_artifact_id="another-artifact",
            )
        if self.replay_failure_index == index and self.fail_after_mutation:
            if returns_new_state:
                handle.discarded = True
                self.discarded_handles.append(handle)
            return self._failure(self.replay_failure_kind, "failure after mutation")
        if returns_new_state:
            source_handle.discarded = True
            self.retired_handles.append(source_handle)
            self.last_handle = handle
        return MaterializationResult.success(
            TransitionReplay(result_state, certificate)
        )

    async def certify_descendant(
        self,
        state: EphemeralMaterializedState[_SyntheticHandle],
        transition_certificates: tuple[PhysicalTransitionCertificate, ...],
    ) -> MaterializationResult[DescendantStateCertificate]:
        transition_ids = tuple(item.certificate_id for item in transition_certificates)
        observed = self._descendant(state.handle.records, transition_ids)
        if self.final_lineage_inconsistent:
            observed = self._descendant(
                {
                    "reported-e1-a": _Record((self.e1,), "reported-a"),
                    "reported-e1-b": _Record((self.e1,), "reported-b"),
                },
                transition_ids,
            )
        if self.extra_final_transition:
            observed = replace(
                observed,
                accepted_transition_certificate_ids=transition_ids + ("extra-cert",),
            )
        if self.final_inequivalent:
            observed = replace(
                observed,
                observable_state_digest="inequivalent-descendant-state",
            )
        return MaterializationResult.success(observed)

    async def retrieve(
        self,
        state: EphemeralMaterializedState[_SyntheticHandle],
        *,
        query_artifact,
        top_k: int,
    ) -> MaterializationResult[RankedPhysicalRetrieval]:
        self.seen_query_artifacts.append(query_artifact)
        if self.retrieval_failure is not None:
            return self._failure(self.retrieval_failure, "configured retrieval failure")
        records = list(state.handle.records.items())
        if self.retrieval_only_lineage is not None:
            records = [
                item
                for item in records
                if item[1].lineage == self.retrieval_only_lineage
            ]
        backend = self.backend
        campaign_id = self.campaign_id
        root_checkpoint_id = self.root_checkpoint_id
        state_id = state.state_id
        if self.retrieval_scope_error == "backend":
            backend = "another-backend"
        elif self.retrieval_scope_error == "campaign":
            campaign_id = "another-campaign"
        elif self.retrieval_scope_error == "root":
            root_checkpoint_id = "another-root"
        elif self.retrieval_scope_error == "state":
            state_id = "another-state"
        physical = [
            PhysicalEntryRef(campaign_id, root_checkpoint_id, state_id, backend_id)
            for backend_id, _ in records[:top_k]
        ]
        if self.unknown_retrieval:
            physical.append(self._physical(state.state_id, "unknown-physical"))
        return MaterializationResult.success(
            RankedPhysicalRetrieval(
                backend,
                "Synthetic.search",
                "SyntheticRecord",
                campaign_id,
                root_checkpoint_id,
                state_id,
                tuple(physical),
            )
        )

    async def discard(
        self, state: EphemeralMaterializedState[_SyntheticHandle]
    ) -> None:
        state.handle.discarded = True
        self.discarded_handles.append(state.handle)


class MaterializationIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.campaign = "campaign-1"
        self.root = "checkpoint-1"
        self.e1 = CoverageEntryId(self.campaign, self.root, "e1")
        self.e2 = CoverageEntryId(self.campaign, self.root, "e2")
        self.inventory = FrozenCheckpointInventory(
            self.campaign,
            self.root,
            (
                InitialCoverageEntry(
                    self.e1, self.root, "initial-p1", {"value": "initial-e1"}
                ),
                InitialCoverageEntry(
                    self.e2, self.root, "initial-p2", {"value": "initial-e2"}
                ),
            ),
        )
        self.coverage = CoverageState(self.inventory.coverage_ids)
        self.root_records = {
            "p1": _Record((self.e1,), "initial-e1"),
            "p2": _Record((self.e2,), "initial-e2"),
        }

    def protocol(self, **kwargs) -> _SyntheticMaterializer:
        return _SyntheticMaterializer(
            campaign_id=self.campaign,
            root_checkpoint_id=self.root,
            e1=self.e1,
            e2=self.e2,
            **kwargs,
        )

    def descendant(
        self,
        records: dict[str, _Record],
        transition_ids: tuple[str, ...],
    ) -> DescendantStateCertificate:
        return self.protocol()._descendant(records, transition_ids)

    def artifact(
        self,
        artifact_id: str,
        operation: str,
        *,
        target: str = "e1",
        value: str | None = None,
    ) -> RealizedMutationArtifact:
        outcome = {
            "same_id": PhysicalTransitionOutcome.SAME_ID,
            "replace": PhysicalTransitionOutcome.REPLACED,
            "delete": PhysicalTransitionOutcome.DELETED,
            "merge": PhysicalTransitionOutcome.MERGED,
            "split": PhysicalTransitionOutcome.SPLIT,
        }[operation]
        relation = (
            MutationRelation.DELETION
            if operation == "delete"
            else MutationRelation.UPDATE
        )
        target_lineage = (
            (self.e1, self.e2)
            if operation == "merge"
            else ((self.e1,) if target == "e1" else (self.e2,))
        )
        opportunity = MutationOpportunity.create(
            opportunity_id=f"op-{artifact_id}",
            campaign_id=self.campaign,
            root_checkpoint_id=self.root,
            parent_seed_id=f"parent-{artifact_id}",
            relation=relation,
            canonical_target={"target": target},
            target_lineage=target_lineage,
            applicability_evidence={"synthetic": True},
            generation_constraints={"operation": operation},
            possible_transition_outcomes={outcome},
            acceptable_transition_outcomes={outcome},
        )
        semantic = SemanticMutationCertificate(
            f"sem-{artifact_id}",
            opportunity.opportunity_id,
            self.campaign,
            self.root,
            relation,
            True,
            {"operation": operation},
            f"semantic-proof-{artifact_id}",
        )
        payload: dict[str, object] = {"operation": operation, "target": target}
        if operation == "split":
            payload.update({"first_value": "split-a", "second_value": "split-b"})
        elif operation != "delete":
            payload["value"] = value or {
                "same_id": "same-id-updated",
                "replace": "replacement-value",
                "merge": "merged-value",
            }[operation]
        return RealizedMutationArtifact(
            artifact_id,
            opportunity,
            semantic,
            "query-parent",
            None,
            operation,
            payload,
        )

    def seed(
        self,
        artifacts: tuple[RealizedMutationArtifact, ...] = (),
        *,
        expected_records: dict[str, _Record] | None = None,
        expected_transition_ids: tuple[str, ...] | None = None,
        seed_id: str = "seed-1",
    ) -> LogicalSeed:
        records = expected_records if expected_records is not None else self.root_records
        transition_ids = (
            expected_transition_ids
            if expected_transition_ids is not None
            else tuple(f"cert-{item.artifact_id}" for item in artifacts)
        )
        return LogicalSeed(
            seed_id,
            self.campaign,
            "synthetic",
            self.root,
            "synthetic-config-v1",
            {"query_id": "q1", "text": "Where does Ada work?"},
            artifacts,
            self.descendant(records, transition_ids),
            ((self.e2,),),
        )

    def run_materialization(
        self,
        seed: LogicalSeed,
        protocol: _SyntheticMaterializer,
        *,
        coverage: CoverageState | None = None,
        top_k: int = 10,
    ):
        return asyncio.run(
            materialize_and_observe(
                seed=seed,
                protocol=protocol,
                inventory=self.inventory,
                expected_root_state=self.descendant(self.root_records, ()),
                initialization_artifact_id="initialization-artifact-1",
                coverage_state=coverage or self.coverage,
                top_k=top_k,
            )
        )

    def materialize_only(
        self,
        seed: LogicalSeed,
        protocol: _SyntheticMaterializer,
    ):
        return asyncio.run(
            materialize_seed(
                seed=seed,
                protocol=protocol,
                inventory=self.inventory,
                expected_root_state=self.descendant(self.root_records, ()),
                initialization_artifact_id="initialization-artifact-1",
            )
        )

    def observe_only(
        self,
        seed: LogicalSeed,
        protocol: _SyntheticMaterializer,
        materialization: CertifiedMaterialization,
        *,
        coverage: CoverageState | None = None,
        top_k: int = 10,
    ):
        return asyncio.run(
            observe_materialization(
                seed_id=seed.seed_id,
                protocol=protocol,
                materialization=materialization,
                query_artifact=seed.current_query_artifact,
                coverage_state=coverage or self.coverage,
                top_k=top_k,
            )
        )

    def test_materialization_only_does_not_retrieve_or_mutate_campaign_state(self) -> None:
        seed = self.seed()
        protocol = self.protocol()
        before_coverage = self.coverage
        history = BehaviorHistory(
            self.campaign,
            {self.root: (seed.authoritative_parent_signature,)},
        )
        before_history = history.signatures(self.root)
        result = self.materialize_only(seed, protocol)
        self.assertIsNone(result.failure)
        materialization = result.value
        self.assertIsInstance(materialization, CertifiedMaterialization)
        self.assertEqual(materialization.seed_id, seed.seed_id)
        self.assertEqual(materialization.transition_certificates, ())
        self.assertEqual(
            materialization.lineage.coverage_ids,
            self.inventory.coverage_ids,
        )
        self.assertEqual(protocol.seen_query_artifacts, [])
        self.assertEqual(len(protocol.root_requests), 1)
        root_request = protocol.root_requests[0]
        self.assertEqual(
            root_request.initialization_artifact_id,
            "initialization-artifact-1",
        )
        self.assertFalse(
            any(
                hasattr(root_request, name)
                for name in (
                    "current_query_artifact",
                    "memory_transition_artifacts",
                    "authoritative_parent_signature",
                    "expected_descendant_state",
                )
            )
        )
        self.assertFalse(materialization.state.handle.discarded)
        self.assertEqual(self.coverage, before_coverage)
        self.assertEqual(self.coverage.reached_ids, frozenset())
        self.assertEqual(history.signatures(self.root), before_history)

    def test_query_only_observation_is_a_separate_generic_bridge(self) -> None:
        seed = self.seed()
        protocol = self.protocol()
        materialized = self.materialize_only(seed, protocol)
        self.assertIsNone(materialized.failure)
        observed = self.observe_only(seed, protocol, materialized.value)
        self.assertIsNone(observed.failure)
        self.assertEqual(observed.value.seed_id, seed.seed_id)
        self.assertEqual(observed.value.materialization, materialized.value)
        self.assertEqual(protocol.seen_query_artifacts, [seed.current_query_artifact])
        self.assertFalse(materialized.value.state.handle.discarded)

    def test_root_rebinding_ambiguity_and_initialization_inequivalence(self) -> None:
        seed = self.seed()
        ambiguous = self.run_materialization(
            seed,
            self.protocol(
                root_failure=MaterializationFailureKind.E0_REBINDING_AMBIGUOUS
            ),
        )
        self.assertEqual(
            ambiguous.failure.kind,
            MaterializationFailureKind.E0_REBINDING_AMBIGUOUS,
        )

        protocol = self.protocol(root_binding_state_mismatch=True)
        invalid_binding = self.run_materialization(seed, protocol)
        self.assertIsNone(invalid_binding.value)
        self.assertEqual(
            invalid_binding.failure.kind,
            MaterializationFailureKind.E0_REBINDING_AMBIGUOUS,
        )
        self.assertTrue(protocol.last_handle.discarded)
        self.assertEqual(self.inventory.coverage_ids, frozenset({self.e1, self.e2}))

        protocol = self.protocol(root_inequivalent=True)
        inequivalent = self.run_materialization(seed, protocol)
        self.assertEqual(
            inequivalent.failure.kind,
            MaterializationFailureKind.ROOT_INITIALIZATION_INEQUIVALENT,
        )
        self.assertTrue(protocol.last_handle.discarded)

    def test_same_id_and_replacement_preserve_initialized_lineage(self) -> None:
        cases = (
            (
                self.artifact("same", "same_id"),
                {"p1": _Record((self.e1,), "same-id-updated"),
                 "p2": _Record((self.e2,), "initial-e2")},
                PhysicalTransitionOutcome.SAME_ID,
            ),
            (
                self.artifact("replace", "replace"),
                {"r9": _Record((self.e1,), "replacement-value"),
                 "p2": _Record((self.e2,), "initial-e2")},
                PhysicalTransitionOutcome.REPLACED,
            ),
        )
        for artifact, records, outcome in cases:
            with self.subTest(outcome=outcome):
                seed = self.seed((artifact,), expected_records=records)
                result = self.run_materialization(seed, self.protocol())
                self.assertIsNone(result.failure)
                materialization = result.value.materialization
                self.assertEqual(
                    materialization.transition_certificates[0].observed_outcome,
                    outcome,
                )
                live_e1 = [
                    binding
                    for binding in materialization.current_rebinding.live_bindings
                    if binding.lineage == (self.e1,)
                ]
                self.assertEqual(len(live_e1), 1)
                self.assertEqual(materialization.lineage.coverage_ids, {self.e1, self.e2})

    def test_in_place_and_new_state_transition_identity_rules(self) -> None:
        artifact = self.artifact("same", "same_id")
        expected = {
            "p1": _Record((self.e1,), "same-id-updated"),
            "p2": _Record((self.e2,), "initial-e2"),
        }
        seed = self.seed((artifact,), expected_records=expected)

        in_place = self.materialize_only(seed, self.protocol())
        self.assertIsNone(in_place.failure)
        in_place_state = in_place.value.state.state_id
        self.assertEqual(
            {
                binding.physical_entry.state_id
                for binding in in_place.value.current_rebinding.live_bindings
            },
            {in_place_state},
        )

        protocol = self.protocol(new_state_transition_index=0)
        new_state = self.materialize_only(seed, protocol)
        self.assertIsNone(new_state.failure)
        self.assertEqual(len(protocol.retired_handles), 1)
        self.assertTrue(protocol.retired_handles[0].discarded)
        self.assertFalse(new_state.value.state.handle.discarded)
        self.assertEqual(
            {
                binding.physical_entry.state_id
                for binding in new_state.value.current_rebinding.live_bindings
            },
            {new_state.value.state.state_id},
        )

    def test_wrong_successor_or_current_binding_state_rejects_transition(self) -> None:
        artifact = self.artifact("same", "same_id")
        seed = self.seed(
            (artifact,),
            expected_records={
                "p1": _Record((self.e1,), "same-id-updated"),
                "p2": _Record((self.e2,), "initial-e2"),
            },
        )
        protocol = self.protocol(wrong_successor_state_index=0)
        wrong_successor = self.materialize_only(seed, protocol)
        self.assertEqual(
            wrong_successor.failure.kind,
            MaterializationFailureKind.TRANSITION_REPLAY_DIVERGED,
        )
        self.assertTrue(protocol.last_handle.discarded)

        root_protocol = self.protocol()
        root = self.materialize_only(self.seed(), root_protocol).value
        binding, other = root.current_rebinding.live_bindings
        foreign_binding = replace(
            binding,
            physical_entry=replace(
                binding.physical_entry,
                state_id="another-materialization",
            ),
        )
        foreign_current = replace(
            root.current_rebinding,
            live_bindings=(foreign_binding, other),
        )
        replay = asyncio.run(
            root_protocol.replay_transition(root.state, artifact)
        ).value
        with self.assertRaisesRegex(ValueError, "another live state"):
            apply_certified_transition(
                current=foreign_current,
                transition=replay.certificate,
                inventory=self.inventory,
                current_state_id=root.state.state_id,
                successor_state_id=replay.state.state_id,
            )

    def test_deletion_keeps_denominator_and_creates_no_deletion_credit(self) -> None:
        artifact = self.artifact("delete-e1", "delete")
        seed = self.seed(
            (artifact,),
            expected_records={"p2": _Record((self.e2,), "initial-e2")},
        )
        result = self.run_materialization(seed, self.protocol())
        self.assertIsNone(result.failure)
        execution = result.value
        self.assertEqual(
            execution.materialization.current_rebinding.deleted_ids,
            frozenset({self.e1}),
        )
        self.assertEqual(
            execution.observation.coverage_update.state.denominator_count,
            2,
        )
        self.assertEqual(
            execution.observation.coverage_update.retrieved_ids,
            frozenset({self.e2}),
        )
        self.assertNotIn(self.e1, execution.observation.coverage_update.retrieved_ids)

    def test_merge_stays_one_atomic_retrieval_token(self) -> None:
        artifact = self.artifact("merge", "merge")
        merged = {"merged": _Record((self.e1, self.e2), "merged-value")}
        seed = self.seed((artifact,), expected_records=merged)
        result = self.run_materialization(seed, self.protocol())
        self.assertIsNone(result.failure)
        observation = result.value.observation
        self.assertEqual(observation.signature, ((self.e1, self.e2),))
        self.assertEqual(len(observation.resolved_lineages), 1)
        self.assertEqual(observation.coverage_update.raw_gain, 2)

    def test_split_keeps_two_physical_bindings_and_set_valued_coverage(self) -> None:
        artifact = self.artifact("split", "split")
        records = {
            "p2": _Record((self.e2,), "initial-e2"),
            "split-a": _Record((self.e1,), "split-a"),
            "split-b": _Record((self.e1,), "split-b"),
        }
        seed = self.seed((artifact,), expected_records=records)
        protocol = self.protocol(retrieval_only_lineage=(self.e1,))
        result = self.run_materialization(seed, protocol)
        self.assertIsNone(result.failure)
        execution = result.value
        split_bindings = [
            item
            for item in execution.materialization.current_rebinding.live_bindings
            if item.lineage == (self.e1,)
        ]
        self.assertEqual(len(split_bindings), 2)
        self.assertEqual(len(execution.observation.resolved_lineages), 2)
        self.assertEqual(execution.observation.signature, ((self.e1,),))
        self.assertEqual(
            execution.observation.coverage_update.retrieved_ids,
            frozenset({self.e1}),
        )

    def test_ordered_two_step_replay_and_changed_order(self) -> None:
        update = self.artifact("update-e1", "same_id", value="updated-e1")
        delete = self.artifact("delete-e2", "delete", target="e2")
        final_records = {"p1": _Record((self.e1,), "updated-e1")}
        seed = self.seed((update, delete), expected_records=final_records)
        protocol = self.protocol()
        result = self.run_materialization(seed, protocol)
        self.assertIsNone(result.failure)
        self.assertEqual(
            protocol.executed_artifact_ids,
            ["update-e1", "delete-e2"],
        )
        self.assertEqual(
            result.value.materialization.observed_descendant_state.accepted_transition_certificate_ids,
            ("cert-update-e1", "cert-delete-e2"),
        )

        swapped = self.seed(
            (delete, update),
            expected_records=final_records,
            expected_transition_ids=("cert-update-e1", "cert-delete-e2"),
            seed_id="swapped-seed",
        )
        swapped_result = self.run_materialization(swapped, self.protocol())
        self.assertEqual(
            swapped_result.failure.kind,
            MaterializationFailureKind.DESCENDANT_STATE_INEQUIVALENT,
        )

    def test_omitted_and_extra_transition_fail(self) -> None:
        update = self.artifact("update-e1", "same_id", value="updated-e1")
        delete = self.artifact("delete-e2", "delete", target="e2")
        seed = self.seed(
            (update, delete),
            expected_records={"p1": _Record((self.e1,), "updated-e1")},
        )
        omitted = self.run_materialization(
            seed,
            self.protocol(replay_failure_index=1),
        )
        self.assertEqual(
            omitted.failure.kind,
            MaterializationFailureKind.TRANSITION_REPLAY_DIVERGED,
        )
        extra = self.run_materialization(
            seed,
            self.protocol(extra_final_transition=True),
        )
        self.assertEqual(
            extra.failure.kind,
            MaterializationFailureKind.DESCENDANT_STATE_INEQUIVALENT,
        )

    def test_artifact_mismatch_semantic_drift_and_replay_divergence(self) -> None:
        artifact = self.artifact("update", "same_id")
        seed = self.seed(
            (artifact,),
            expected_records={
                "p1": _Record((self.e1,), "same-id-updated"),
                "p2": _Record((self.e2,), "initial-e2"),
            },
        )
        mismatched = self.run_materialization(
            seed,
            self.protocol(corrupt_certificate_index=0),
        )
        self.assertEqual(
            mismatched.failure.kind,
            MaterializationFailureKind.TRANSITION_REPLAY_DIVERGED,
        )
        drift = self.run_materialization(
            seed,
            self.protocol(semantic_drift_index=0),
        )
        self.assertEqual(
            drift.failure.kind,
            MaterializationFailureKind.TRANSITION_SEMANTIC_DRIFT,
        )
        divergence = self.run_materialization(
            seed,
            self.protocol(replay_failure_index=0),
        )
        self.assertEqual(
            divergence.failure.kind,
            MaterializationFailureKind.TRANSITION_REPLAY_DIVERGED,
        )

    def test_final_state_inequivalence_and_protocol_failures_propagate(self) -> None:
        seed = self.seed()
        final_protocol = self.protocol(final_inequivalent=True)
        final = self.run_materialization(
            seed,
            final_protocol,
        )
        self.assertEqual(
            final.failure.kind,
            MaterializationFailureKind.DESCENDANT_STATE_INEQUIVALENT,
        )
        self.assertTrue(final_protocol.last_handle.discarded)
        for kind in (
            MaterializationFailureKind.TRANSIENT_BACKEND_FAILURE,
            MaterializationFailureKind.BACKEND_CAPABILITY_FAILURE,
        ):
            with self.subTest(kind=kind):
                result = self.run_materialization(
                    seed,
                    self.protocol(root_failure=kind),
                )
                self.assertEqual(result.failure.kind, kind)

    def test_observed_expected_match_cannot_override_current_lineage(self) -> None:
        reported = {
            "reported-e1-a": _Record((self.e1,), "reported-a"),
            "reported-e1-b": _Record((self.e1,), "reported-b"),
        }
        seed = self.seed(expected_records=reported)
        protocol = self.protocol(final_lineage_inconsistent=True)
        result = self.materialize_only(seed, protocol)
        self.assertEqual(
            result.failure.kind,
            MaterializationFailureKind.DESCENDANT_STATE_INEQUIVALENT,
        )
        self.assertTrue(protocol.last_handle.discarded)

    def test_unknown_retrieval_object_is_lineage_violation(self) -> None:
        protocol = self.protocol(unknown_retrieval=True)
        result = self.run_materialization(self.seed(), protocol)
        self.assertEqual(
            result.failure.kind,
            MaterializationFailureKind.POST_RETRIEVAL_LINEAGE_VIOLATION,
        )
        self.assertTrue(protocol.last_handle.discarded)

    def test_retrieval_must_match_certified_materialization_scope(self) -> None:
        seed = self.seed()
        materialization_protocol = self.protocol()
        materialized = self.materialize_only(seed, materialization_protocol)
        self.assertIsNone(materialized.failure)

        valid = self.observe_only(seed, materialization_protocol, materialized.value)
        self.assertIsNone(valid.failure)
        self.assertFalse(materialized.value.state.handle.discarded)

        for field in ("state", "campaign", "root", "backend"):
            with self.subTest(field=field):
                result = self.observe_only(
                    seed,
                    self.protocol(retrieval_scope_error=field),
                    materialized.value,
                )
                self.assertEqual(
                    result.failure.kind,
                    MaterializationFailureKind.POST_RETRIEVAL_LINEAGE_VIOLATION,
                )
                self.assertFalse(materialized.value.state.handle.discarded)

    def test_observation_failure_borrows_materialization_without_corruption(self) -> None:
        seed = self.seed()
        protocol = self.protocol()
        materialized = self.materialize_only(seed, protocol)
        before_hash = hash(seed)
        before_coverage = self.coverage
        failed = self.observe_only(
            seed,
            self.protocol(
                retrieval_failure=MaterializationFailureKind.TRANSIENT_BACKEND_FAILURE
            ),
            materialized.value,
        )
        self.assertEqual(
            failed.failure.kind,
            MaterializationFailureKind.TRANSIENT_BACKEND_FAILURE,
        )
        self.assertFalse(materialized.value.state.handle.discarded)
        self.assertEqual(hash(seed), before_hash)
        self.assertEqual(self.coverage, before_coverage)

    def test_authoritative_signature_is_not_replaced_by_observation(self) -> None:
        seed = self.seed()
        original = seed.authoritative_parent_signature
        result = self.run_materialization(seed, self.protocol())
        self.assertIsNone(result.failure)
        self.assertNotEqual(result.value.observation.signature, original)
        self.assertEqual(seed.authoritative_parent_signature, original)

    def test_failed_partial_replay_discards_ephemeral_state_only(self) -> None:
        update = self.artifact("update-e1", "same_id", value="updated-e1")
        delete = self.artifact("delete-e2", "delete", target="e2")
        seed = self.seed(
            (update, delete),
            expected_records={"p1": _Record((self.e1,), "updated-e1")},
        )
        seed_hash = hash(seed)
        signature = seed.authoritative_parent_signature
        coverage = self.coverage
        protocol = self.protocol(
            replay_failure_index=1,
            fail_after_mutation=True,
        )
        result = self.run_materialization(seed, protocol)
        self.assertIsNone(result.value)
        self.assertEqual(
            result.failure.kind,
            MaterializationFailureKind.TRANSITION_REPLAY_DIVERGED,
        )
        self.assertTrue(protocol.last_handle.discarded)
        self.assertEqual(hash(seed), seed_hash)
        self.assertEqual(seed.authoritative_parent_signature, signature)
        self.assertEqual(self.coverage, coverage)
        self.assertEqual(self.inventory.coverage_ids, frozenset({self.e1, self.e2}))

    def test_valid_leaf_can_later_fail_rematerialization_admission(self) -> None:
        seed = self.seed()
        first = self.run_materialization(seed, self.protocol())
        self.assertIsNone(first.failure)
        later = self.run_materialization(
            seed,
            self.protocol(
                root_failure=MaterializationFailureKind.E0_REBINDING_AMBIGUOUS
            ),
        )
        self.assertEqual(
            later.failure.kind,
            MaterializationFailureKind.E0_REBINDING_AMBIGUOUS,
        )
        admission = SeedAdmission.for_executed_child(
            execution_valid=True,
            rematerializable=False,
        )
        self.assertFalse(admission.retainable_as_parent)


if __name__ == "__main__":
    unittest.main()
