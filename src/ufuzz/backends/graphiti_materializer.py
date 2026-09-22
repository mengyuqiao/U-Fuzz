"""Certified materialization for the pinned Graphiti EntityEdge profile.

The upstream ingestion path remains native: root replay invokes ``add_episode``
and is published only when its complete public EntityEdge projection and
provenance multiset exactly match the campaign's frozen E0 inventory.  This
module therefore supplies a fail-closed materializer without claiming that an
unbound LLM/embedding configuration is reproducible.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Any

from ufuzz.backends.graphiti import (
    GRAPHITI_TAG_COMMIT,
    GRAPHITI_VERSION,
    GraphitiAdapter,
)
from ufuzz.backends.graphiti_capability import GraphitiRetrievableEntryCapability
from ufuzz.backends.base import InitializationArtifact, OperationReceipt, StateHandle
from ufuzz.coverage import CoverageEntryId, FrozenCheckpointInventory
from ufuzz.materialization import (
    EphemeralMaterializedState,
    MaterializationProtocol,
    RootMaterialization,
    RootMaterializationRequest,
    TransitionReplay,
)
from ufuzz.retrieval_capability import RetrievableInventoryEntry
from ufuzz.retrieval_feedback import MutationRelation, canonical_tuple
from ufuzz.state_contract import (
    AffectedPhysicalTransition,
    AffectedTransitionKind,
    CertificationStatus,
    DescendantStateCertificate,
    ExecutableQueryArtifact,
    LiveLineageState,
    MaterializationFailure,
    MaterializationFailureKind,
    MaterializationResult,
    PhysicalLineageEndpoint,
    PhysicalTransitionCertificate,
    PhysicalTransitionOutcome,
    RealizedMutationArtifact,
    RebindingKind,
    ReplayBinding,
    ReplayRebindingCertificate,
)
from ufuzz.structural import StructuralIndex


GRAPHITI_NATIVE_CONFIGURATION_ID = f"graphiti-native:{GRAPHITI_TAG_COMMIT}:entity-edge:neo4j:v1"


class _EvidenceError(ValueError):
    pass


class _AmbiguousRebinding(_EvidenceError):
    pass


@dataclass(slots=True)
class _Context:
    handle: StateHandle
    capability: GraphitiRetrievableEntryCapability
    inventory: FrozenCheckpointInventory
    root_rebinding: ReplayRebindingCertificate
    structural_index: StructuralIndex | None
    lineage_by_physical_id: dict[str, tuple[CoverageEntryId, ...]]


@dataclass(frozen=True, slots=True)
class _Operation:
    name: str
    new_text: str | None
    provenance_ids: tuple[str, ...]
    context: Mapping[str, Any] | None
    intended_fact_id: str | None = None


def _canon(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return ["float", value.hex()]
    if isinstance(value, CoverageEntryId):
        return [value.campaign_id, value.root_checkpoint_id, value.opaque_id]
    if isinstance(value, Mapping):
        return {key: _canon(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_canon(item) for item in value]
    if isinstance(value, (set, frozenset)):
        encoded = [_canon(item) for item in value]
        return sorted(encoded, key=lambda item: json.dumps(item, sort_keys=True))
    raise TypeError(f"unsupported evidence type: {type(value).__name__}")


def _digest(value: Any) -> str:
    encoded = json.dumps(_canon(value), sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_thaw(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(_thaw(item) for item in value)
    return value


class GraphitiMaterializationProtocol(MaterializationProtocol[StateHandle]):
    """Materialize and certify one dedicated-process Graphiti state."""

    backend = "graphiti"
    frozen_configuration_id = GRAPHITI_NATIVE_CONFIGURATION_ID

    def __init__(
        self,
        *,
        initialization_artifacts: Mapping[str, InitializationArtifact],
        frozen_inventories: Mapping[tuple[str, str], FrozenCheckpointInventory],
        structural_indexes: Mapping[tuple[str, str], StructuralIndex] | None = None,
        adapter: GraphitiAdapter | None = None,
    ) -> None:
        if not initialization_artifacts or not frozen_inventories:
            raise ValueError("Graphiti materializer requires artifact and inventory registries")
        for key, artifact in initialization_artifacts.items():
            if not key or InitializationArtifact.create(
                artifact.checkpoint_id, artifact.sources, artifact.frozen_config
            ).digest != artifact.digest:
                raise ValueError("initialization artifact is invalid or changed")
            if artifact.frozen_config.get("graphiti_profile_id") != self.frozen_configuration_id:
                raise ValueError("artifact does not freeze the selected Graphiti profile")
        for key, inventory in frozen_inventories.items():
            if key != (inventory.campaign_id, inventory.root_checkpoint_id):
                raise ValueError("frozen inventory registry key mismatch")
        for key, index in (structural_indexes or {}).items():
            if key[1] != index.checkpoint_id or key not in frozen_inventories:
                raise ValueError("structural-index registry scope mismatch")
        self._artifacts = MappingProxyType(dict(initialization_artifacts))
        self._inventories = MappingProxyType(dict(frozen_inventories))
        self._structural_indexes = MappingProxyType(dict(structural_indexes or {}))
        self._adapter = adapter or GraphitiAdapter()
        self._states: dict[str, _Context] = {}

    @staticmethod
    def _failure(
        campaign: str,
        checkpoint: str,
        kind: MaterializationFailureKind,
        reason: str,
    ) -> MaterializationResult[Any]:
        return MaterializationResult.failed(
            MaterializationFailure(kind, campaign, checkpoint, reason)
        )

    def _validate_root(self, request: RootMaterializationRequest):
        if (request.backend, request.frozen_configuration_id) != (
            self.backend,
            self.frozen_configuration_id,
        ):
            raise _EvidenceError("root request does not use the frozen Graphiti profile")
        artifact = self._artifacts.get(request.initialization_artifact_id)
        inventory = self._inventories.get(
            (request.campaign_id, request.root_checkpoint_id)
        )
        if artifact is None or inventory is None:
            raise _EvidenceError("unknown artifact or frozen inventory")
        if (
            artifact.checkpoint_id != request.root_checkpoint_id
            or inventory.coverage_ids != request.frozen_e0
        ):
            raise _EvidenceError("root scope or E0 differs from the frozen registry")
        return artifact, inventory

    @staticmethod
    def _key_frozen(entry: Any):
        return tuple(entry.provenance_ids), entry.initial_observable_projection

    @staticmethod
    def _key_live(entry: RetrievableInventoryEntry):
        return tuple(entry.provenance_ids), entry.projection.observable

    def _root_rebinding(self, request, frozen, fresh, artifact):
        old: dict[Any, list[Any]] = defaultdict(list)
        new: dict[Any, list[Any]] = defaultdict(list)
        for entry in frozen.entries:
            old[self._key_frozen(entry)].append(entry)
        for entry in fresh.entries:
            new[self._key_live(entry)].append(entry)
        if Counter(map(self._key_frozen, frozen.entries)) != Counter(
            map(self._key_live, fresh.entries)
        ):
            raise _EvidenceError("fresh projection/provenance multiset differs from E0")
        if any(len(values) != 1 for values in (*old.values(), *new.values())):
            raise _AmbiguousRebinding("duplicate exact Graphiti replay key is ambiguous")
        bindings = tuple(
            sorted(
                (
                    ReplayBinding(
                        new[key][0].physical_entry,
                        (old[key][0].coverage_id,),
                        _digest(
                            {
                                "profile": self.frozen_configuration_id,
                                "key": key,
                                "coverage": old[key][0].coverage_id,
                            }
                        ),
                    )
                    for key in old
                ),
                key=lambda binding: binding.lineage,
            )
        )
        return ReplayRebindingCertificate(
            f"graphiti:root:{artifact.digest}",
            RebindingKind.ROOT,
            self.backend,
            self.frozen_configuration_id,
            request.campaign_id,
            request.root_checkpoint_id,
            request.frozen_e0,
            bindings,
            tuple(binding.lineage for binding in bindings),
            frozenset(),
            _digest(
                {
                    "artifact": artifact.digest,
                    "bindings": tuple(
                        (binding.lineage, binding.binding_proof_digest)
                        for binding in bindings
                    ),
                }
            ),
        )

    def _context(self, state):
        value = self._states.get(state.state_id)
        if value is None or value.handle is not state.handle:
            raise _EvidenceError("unknown or retired Graphiti materialized state")
        if (
            state.backend,
            state.frozen_configuration_id,
            state.campaign_id,
            state.root_checkpoint_id,
        ) != (
            self.backend,
            self.frozen_configuration_id,
            value.inventory.campaign_id,
            value.inventory.root_checkpoint_id,
        ):
            raise _EvidenceError("materialized state scope mismatch")
        return value

    async def _inventory(self, state, context):
        result = await context.capability.inventory(
            campaign_id=state.campaign_id,
            root_checkpoint_id=state.root_checkpoint_id,
            state_id=state.state_id,
        )
        if result.blocker:
            raise _EvidenceError(result.blocker.reason)
        return result.value

    def _lineage_map(self, context, public):
        result = {}
        for entry in public.entries:
            identifier = entry.physical_entry.backend_entry_id
            if identifier not in context.lineage_by_physical_id:
                raise _EvidenceError("new or replacement physical ID has no certified lineage")
            result[identifier] = context.lineage_by_physical_id[identifier]
        return result

    def _descendant(self, state, context, public, transition_ids):
        mapping = self._lineage_map(context, public)
        rows, live, live_ids = [], [], set()
        for entry in public.entries:
            lineage = canonical_tuple(mapping[entry.physical_entry.backend_entry_id])
            live_ids.update(lineage)
            projection_digest = _digest(
                {
                    "profile": self.frozen_configuration_id,
                    "projection": entry.projection.observable,
                }
            )
            state_digest = _digest(
                {
                    "profile": self.frozen_configuration_id,
                    "provenance": entry.provenance_ids,
                }
            )
            live.append(LiveLineageState(lineage, projection_digest, state_digest))
            rows.append((lineage, entry.projection.observable, entry.provenance_ids))
        rows.sort(key=_digest)
        deleted = context.inventory.coverage_ids.difference(live_ids)
        observable = _digest(
            {
                "profile": self.frozen_configuration_id,
                "rows": tuple(rows),
                "deleted": deleted,
            }
        )
        transition = _digest(
            {
                "profile": self.frozen_configuration_id,
                "transitions": transition_ids,
                "auxiliary": "group-scoped EntityEdge, endpoint, episode, embedding, and temporal state",
            }
        )
        return DescendantStateCertificate(
            "graphiti:descendant:" + _digest((observable, transition)),
            self.backend,
            self.frozen_configuration_id,
            state.campaign_id,
            state.root_checkpoint_id,
            context.inventory.coverage_ids,
            tuple(live),
            deleted,
            observable,
            transition,
            transition_ids,
        )

    async def materialize_root(self, request):
        handle = None
        try:
            artifact, frozen = self._validate_root(request)
            handle = await self._adapter.replay_state(artifact)
            state = EphemeralMaterializedState(
                handle,
                self.backend,
                self.frozen_configuration_id,
                request.campaign_id,
                request.root_checkpoint_id,
                handle.state_id,
            )
            capability = GraphitiRetrievableEntryCapability(handle)
            result = await capability.inventory(
                campaign_id=request.campaign_id,
                root_checkpoint_id=request.root_checkpoint_id,
                state_id=handle.state_id,
            )
            if result.blocker:
                raise _EvidenceError(result.blocker.reason)
            rebinding = self._root_rebinding(request, frozen, result.value, artifact)
            context = _Context(
                handle,
                capability,
                frozen,
                rebinding,
                self._structural_indexes.get(
                    (request.campaign_id, request.root_checkpoint_id)
                ),
                {
                    binding.physical_entry.backend_entry_id: binding.lineage
                    for binding in rebinding.live_bindings
                },
            )
            observed = self._descendant(state, context, result.value, ())
            self._states[state.state_id] = context
            return MaterializationResult.success(
                RootMaterialization(state, rebinding, observed)
            )
        except _AmbiguousRebinding as error:
            if handle is not None:
                await self._adapter.teardown(handle)
            return self._failure(
                request.campaign_id,
                request.root_checkpoint_id,
                MaterializationFailureKind.E0_REBINDING_AMBIGUOUS,
                str(error),
            )
        except _EvidenceError as error:
            if handle is not None:
                await self._adapter.teardown(handle)
            return self._failure(
                request.campaign_id,
                request.root_checkpoint_id,
                MaterializationFailureKind.ROOT_INITIALIZATION_INEQUIVALENT,
                str(error),
            )
        except Exception as error:
            if handle is not None:
                await self._adapter.teardown(handle)
            return self._failure(
                request.campaign_id,
                request.root_checkpoint_id,
                MaterializationFailureKind.TRANSIENT_BACKEND_FAILURE,
                f"Graphiti root creation failed: {type(error).__name__}: {error}",
            )

    @staticmethod
    def _parse(artifact):
        relation, payload = artifact.opportunity.relation, artifact.operation_payload
        if relation is MutationRelation.DELETION:
            if set(payload) != {"operation", "intended_fact_id"}:
                raise _EvidenceError(
                    "Graphiti deletion payload requires operation and intended_fact_id"
                )
            fact_id = payload["intended_fact_id"]
            if payload["operation"] != "delete" or not isinstance(fact_id, str) or not fact_id:
                raise _EvidenceError("Graphiti deletion payload is invalid")
            return _Operation("delete", None, (), None, fact_id)
        name = (
            "update"
            if relation is MutationRelation.UPDATE
            else "unrelated_change"
            if relation is MutationRelation.UNRELATED_CHANGE
            else None
        )
        if name is None or payload.get("operation") != name:
            raise _EvidenceError("Graphiti materializer accepts only exact memory mutations")
        if not {"operation", "new_text", "provenance_ids", "context"}.issubset(
            payload
        ) or set(payload) != {"operation", "new_text", "provenance_ids", "context"}:
            raise _EvidenceError("update payload schema is invalid")
        text, provenance = payload["new_text"], payload["provenance_ids"]
        if (
            not isinstance(text, str)
            or not text
            or not isinstance(provenance, tuple)
            or not provenance
        ):
            raise _EvidenceError("new_text/provenance_ids are invalid")
        context = (
            _thaw(payload.get("context")) if payload.get("context") is not None else None
        )
        if not isinstance(context, Mapping) or set(context) != {
            "expected_successor_observable"
        }:
            raise _EvidenceError(
                "Graphiti update requires one exact expected successor projection"
            )
        if not isinstance(context["expected_successor_observable"], Mapping):
            raise _EvidenceError("expected successor projection is not a mapping")
        return _Operation(name, text, provenance, context)

    def _resolve_target(self, context, public, artifact):
        target = canonical_tuple(artifact.opportunity.target_lineage)
        mapping = self._lineage_map(context, public)
        matches = [
            entry
            for entry in public.entries
            if mapping[entry.physical_entry.backend_entry_id] == target
        ]
        if len(matches) != 1:
            raise _EvidenceError("target lineage does not resolve to exactly one live edge")
        return matches[0], target

    @staticmethod
    def _by_id(inventory):
        return {
            entry.physical_entry.backend_entry_id: entry for entry in inventory.entries
        }

    @staticmethod
    def _update_receipt(
        receipt: OperationReceipt, state_id: str, operation: str, target: str
    ):
        expected = (
            "unrelated_existing_region_change"
            if operation == "unrelated_change"
            else operation
        )
        return (
            receipt.backend == "graphiti"
            and receipt.state_id == state_id
            and receipt.operation == expected
            and receipt.success
            and receipt.target_local_id == target
            and len(receipt.affected_local_ids) >= 2
        )

    async def _deletion_episode(self, context: _Context, edge_id: str) -> str:
        values = await context.handle.backend_state.edges.entity.get_by_group_ids(
            [context.handle.metadata["group_id"]], limit=None, uuid_cursor=None
        )
        matches = [edge for edge in values if str(getattr(edge, "uuid", "")) == edge_id]
        if len(matches) != 1:
            raise _EvidenceError("deletion target does not resolve to one EntityEdge")
        episodes = tuple(str(value) for value in getattr(matches[0], "episodes", ()))
        if len(episodes) != 1:
            raise _EvidenceError("deletion target is not owned by exactly one episode")
        return episodes[0]

    async def replay_transition(self, state, artifact):
        try:
            context, operation = self._context(state), self._parse(artifact)
            pre = await self._inventory(state, context)
            target, lineage = self._resolve_target(context, pre, artifact)
            identifier = target.physical_entry.backend_entry_id
            if operation.name == "delete":
                if context.structural_index is None:
                    raise _EvidenceError("Graphiti deletion requires its frozen structural index")
                episode_id = await self._deletion_episode(context, identifier)
                deletion_evidence = await self._adapter.certify_episode_deletion(
                    context.handle,
                    episode_id,
                    operation.intended_fact_id or "",
                    context.structural_index,
                )
                receipt = await self._adapter.delete(
                    context.handle,
                    episode_id,
                    context={"deletion_certificate": deletion_evidence},
                )
            elif operation.name == "update":
                receipt = await self._adapter.update(
                    context.handle,
                    identifier,
                    operation.new_text or "",
                    provenance_ids=operation.provenance_ids,
                    context=operation.context,
                )
            else:
                receipt = await self._adapter.unrelated_existing_region_change(
                    context.handle,
                    identifier,
                    operation.new_text or "",
                    provenance_ids=operation.provenance_ids,
                    context=operation.context,
                )
            if operation.name == "delete":
                if (
                    receipt.backend != "graphiti"
                    or receipt.state_id != state.state_id
                    or receipt.operation != "delete"
                    or not receipt.success
                    or receipt.target_local_id != episode_id
                    or receipt.affected_local_ids != (episode_id,)
                ):
                    raise _EvidenceError("deletion receipt does not certify exact episode")
            elif not self._update_receipt(
                receipt, state.state_id, operation.name, identifier
            ):
                raise _EvidenceError("update receipt does not certify exact native episode")
            post = await self._inventory(state, context)
            before, after = self._by_id(pre), self._by_id(post)
            if operation.name == "delete":
                if set(before).difference(after) != {identifier} or set(after).difference(before):
                    raise _EvidenceError("delete did not remove exactly the target")
                if any(before[key] != after[key] for key in after):
                    raise _EvidenceError("delete changed an unrelated public EntityEdge")
                successors = ()
                kind = AffectedTransitionKind.DELETED
                outcome = PhysicalTransitionOutcome.DELETED
                context.lineage_by_physical_id.pop(identifier, None)
            else:
                if operation.provenance_ids != before[identifier].provenance_ids:
                    raise _EvidenceError("mutation payload does not preserve exact source provenance")
                added = set(after).difference(before)
                removed = set(before).difference(after)
                changed = {
                    key for key in set(before).intersection(after) if before[key] != after[key]
                }
                receipt_edges = set(receipt.affected_local_ids[1:])
                if len(added) != 1 or not added.issubset(receipt_edges):
                    raise _EvidenceError(
                        "native update did not expose exactly one receipt-bound successor edge"
                    )
                successor_id = next(iter(added))
                expected = operation.context["expected_successor_observable"]
                if after[successor_id].projection.observable != expected:
                    raise _EvidenceError(
                        "native successor differs from the exact certified projection"
                    )
                if after[successor_id].provenance_ids != before[identifier].provenance_ids:
                    raise _EvidenceError("native successor lost or changed source provenance")
                if removed == {identifier} and not changed:
                    successors = (
                        PhysicalLineageEndpoint(after[successor_id].physical_entry, lineage),
                    )
                    kind = AffectedTransitionKind.REPLACED
                    outcome = PhysicalTransitionOutcome.REPLACED
                    context.lineage_by_physical_id.pop(identifier, None)
                elif not removed and changed.issubset({identifier}):
                    successors = (
                        PhysicalLineageEndpoint(after[identifier].physical_entry, lineage),
                        PhysicalLineageEndpoint(after[successor_id].physical_entry, lineage),
                    )
                    kind = AffectedTransitionKind.SPLIT
                    outcome = PhysicalTransitionOutcome.SPLIT
                else:
                    raise _EvidenceError("native update changed collateral EntityEdge state")
                for successor in successors:
                    context.lineage_by_physical_id[
                        successor.physical_entry.backend_entry_id
                    ] = lineage
            if outcome not in artifact.opportunity.acceptable_transition_outcomes:
                raise _EvidenceError("observed Graphiti transition outcome is not acceptable")
            closure = await context.capability.retrieve(
                campaign_id=state.campaign_id,
                root_checkpoint_id=state.root_checkpoint_id,
                state_id=state.state_id,
                query=target.projection.observable["fact"],
                top_k=max(1, len(before)),
            )
            if closure.blocker is not None:
                raise _EvidenceError(
                    "post-transition retrieval closure failed: " + closure.blocker.reason
                )
            predecessor = PhysicalLineageEndpoint(target.physical_entry, lineage)
            transition_id = f"graphiti:target:{artifact.artifact_id}"
            proof = _digest(
                {
                    "artifact": artifact.artifact_id,
                    "operation": operation.name,
                    "pre": target.projection.observable,
                    "post": tuple(
                        after[item.physical_entry.backend_entry_id].projection.observable
                        for item in successors
                    ),
                }
            )
            affected = AffectedPhysicalTransition(
                transition_id,
                kind,
                (predecessor,),
                successors,
                backend_proof_digest=proof,
            )
            certificate = PhysicalTransitionCertificate(
                f"graphiti:transition:{artifact.artifact_id}",
                artifact.artifact_id,
                artifact.semantic_certificate.certificate_id,
                state.campaign_id,
                state.root_checkpoint_id,
                context.inventory.coverage_ids,
                CertificationStatus.CERTIFIED,
                outcome,
                transition_id,
                (affected,),
                frozenset({predecessor.physical_entry}),
                frozenset(item.physical_entry for item in successors),
                True,
                _digest(
                    {
                        "proof": proof,
                        "receipt": {
                            "operation": receipt.operation,
                            "target": receipt.target_local_id,
                            "affected": receipt.affected_local_ids,
                            "provenance": receipt.provenance_ids,
                            "success": receipt.success,
                        },
                    }
                ),
            )
            return MaterializationResult.success(TransitionReplay(state, certificate))
        except _EvidenceError as error:
            return self._failure(
                state.campaign_id,
                state.root_checkpoint_id,
                MaterializationFailureKind.TRANSITION_REPLAY_DIVERGED,
                str(error),
            )
        except Exception as error:
            return self._failure(
                state.campaign_id,
                state.root_checkpoint_id,
                MaterializationFailureKind.TRANSIENT_BACKEND_FAILURE,
                f"Graphiti transition failed: {type(error).__name__}: {error}",
            )

    async def certify_descendant(self, state, transition_certificates):
        try:
            context = self._context(state)
            public = await self._inventory(state, context)
            return MaterializationResult.success(
                self._descendant(
                    state,
                    context,
                    public,
                    tuple(value.certificate_id for value in transition_certificates),
                )
            )
        except _EvidenceError as error:
            return self._failure(
                state.campaign_id,
                state.root_checkpoint_id,
                MaterializationFailureKind.DESCENDANT_STATE_INEQUIVALENT,
                str(error),
            )

    async def retrieve(self, state, *, query_artifact: ExecutableQueryArtifact, top_k: int):
        try:
            context = self._context(state)
            result = await context.capability.retrieve(
                campaign_id=state.campaign_id,
                root_checkpoint_id=state.root_checkpoint_id,
                state_id=state.state_id,
                query=query_artifact.executable_text,
                top_k=top_k,
            )
            if result.blocker:
                raise _EvidenceError(result.blocker.reason)
            return MaterializationResult.success(result.value)
        except _EvidenceError as error:
            return self._failure(
                state.campaign_id,
                state.root_checkpoint_id,
                MaterializationFailureKind.BACKEND_CAPABILITY_FAILURE,
                str(error),
            )

    async def discard(self, state) -> None:
        context = self._states.pop(state.state_id, None)
        if context is not None:
            await self._adapter.teardown(context.handle)
