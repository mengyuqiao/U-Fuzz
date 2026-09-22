"""Certified materialization for the frozen MemOS GeneralTextMemory profile."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ufuzz.backends.base import InitializationArtifact, OperationReceipt, StateHandle
from ufuzz.backends.memos import MEMOS_PROFILE_ID, MemosAdapter, MemosProfileError
from ufuzz.backends.memos_capability import MemosRetrievableEntryCapability
from ufuzz.coverage import CoverageEntryId, FrozenCheckpointInventory, FrozenMapping
from ufuzz.materialization import (
    EphemeralMaterializedState, MaterializationProtocol, RootMaterialization,
    RootMaterializationRequest, TransitionReplay,
)
from ufuzz.retrieval_capability import RankedPhysicalRetrieval, RetrievableEntryInventory, RetrievableInventoryEntry
from ufuzz.retrieval_feedback import MutationRelation, canonical_tuple
from ufuzz.state_contract import (
    AffectedPhysicalTransition, AffectedTransitionKind, CertificationStatus,
    DescendantStateCertificate, ExecutableQueryArtifact, LiveLineageState,
    MaterializationFailure, MaterializationFailureKind, MaterializationResult,
    PhysicalLineageEndpoint, PhysicalTransitionCertificate,
    PhysicalTransitionOutcome, RealizedMutationArtifact, RebindingKind,
    ReplayBinding, ReplayRebindingCertificate,
)


MEMOS_GENERAL_TEXT_CONFIGURATION_ID = MEMOS_PROFILE_ID


class _EvidenceError(ValueError):
    pass


class _AmbiguousRebinding(_EvidenceError):
    pass


@dataclass(slots=True)
class _Context:
    handle: StateHandle
    capability: MemosRetrievableEntryCapability
    inventory: FrozenCheckpointInventory
    root_rebinding: ReplayRebindingCertificate


@dataclass(frozen=True, slots=True)
class _Operation:
    name: str
    new_text: str | None
    provenance_ids: tuple[str, ...]
    context: Mapping[str, Any] | None


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
    return sha256(json.dumps(_canon(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_thaw(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(_thaw(item) for item in value)
    return value


class MemosGeneralTextMaterializationProtocol(MaterializationProtocol[StateHandle]):
    backend = "memos"
    frozen_configuration_id = MEMOS_GENERAL_TEXT_CONFIGURATION_ID

    def __init__(self, *, initialization_artifacts: Mapping[str, InitializationArtifact],
                 frozen_inventories: Mapping[tuple[str, str], FrozenCheckpointInventory],
                 root_dir: str | Path | None = None,
                 adapter: MemosAdapter | None = None) -> None:
        if not initialization_artifacts or not frozen_inventories:
            raise ValueError("MemOS materializer requires artifact and inventory registries")
        for key, artifact in initialization_artifacts.items():
            if not key or InitializationArtifact.create(
                artifact.checkpoint_id, artifact.sources, artifact.frozen_config
            ).digest != artifact.digest:
                raise ValueError("initialization artifact is invalid or changed")
            if artifact.frozen_config.get("memos_profile_id") != MEMOS_PROFILE_ID:
                raise ValueError("artifact does not freeze the selected MemOS profile")
        for key, inventory in frozen_inventories.items():
            if key != (inventory.campaign_id, inventory.root_checkpoint_id):
                raise ValueError("frozen inventory registry key mismatch")
        self._artifacts = MappingProxyType(dict(initialization_artifacts))
        self._inventories = MappingProxyType(dict(frozen_inventories))
        self._adapter = adapter or MemosAdapter(root_dir=root_dir)
        self._states: dict[str, _Context] = {}

    @staticmethod
    def _failure(campaign: str, checkpoint: str, kind: MaterializationFailureKind, reason: str):
        return MaterializationResult.failed(MaterializationFailure(kind, campaign, checkpoint, reason))

    def _validate_root(self, request: RootMaterializationRequest):
        if (request.backend, request.frozen_configuration_id) != (self.backend, self.frozen_configuration_id):
            raise _EvidenceError("root request does not use the frozen MemOS profile")
        artifact = self._artifacts.get(request.initialization_artifact_id)
        inventory = self._inventories.get((request.campaign_id, request.root_checkpoint_id))
        if artifact is None or inventory is None:
            raise _EvidenceError("unknown artifact or frozen inventory")
        if artifact.checkpoint_id != request.root_checkpoint_id or inventory.coverage_ids != request.frozen_e0:
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
        if Counter(map(self._key_frozen, frozen.entries)) != Counter(map(self._key_live, fresh.entries)):
            raise _EvidenceError("fresh projection/provenance multiset differs from E0")
        if any(len(values) != 1 for values in (*old.values(), *new.values())):
            raise _AmbiguousRebinding("duplicate exact MemOS replay key is ambiguous")
        bindings = tuple(sorted((
            ReplayBinding(new[key][0].physical_entry, (old[key][0].coverage_id,),
                          _digest({"profile": self.frozen_configuration_id, "key": key,
                                   "coverage": old[key][0].coverage_id}))
            for key in old
        ), key=lambda binding: binding.lineage))
        return ReplayRebindingCertificate(
            f"memos-general-text:root:{artifact.digest}", RebindingKind.ROOT,
            self.backend, self.frozen_configuration_id, request.campaign_id,
            request.root_checkpoint_id, request.frozen_e0, bindings,
            tuple(binding.lineage for binding in bindings), frozenset(),
            _digest({"artifact": artifact.digest, "bindings": tuple(
                (binding.lineage, binding.binding_proof_digest) for binding in bindings)}),
        )

    def _context(self, state):
        value = self._states.get(state.state_id)
        if value is None or value.handle is not state.handle:
            raise _EvidenceError("unknown or retired MemOS materialized state")
        if (state.backend, state.frozen_configuration_id, state.campaign_id,
            state.root_checkpoint_id) != (self.backend, self.frozen_configuration_id,
            value.inventory.campaign_id, value.inventory.root_checkpoint_id):
            raise _EvidenceError("materialized state scope mismatch")
        return value

    async def _inventory(self, state, context):
        result = await context.capability.inventory(
            campaign_id=state.campaign_id, root_checkpoint_id=state.root_checkpoint_id,
            state_id=state.state_id)
        if result.blocker:
            raise _EvidenceError(result.blocker.reason)
        return result.value

    @staticmethod
    def _root_map(context):
        return {binding.physical_entry.backend_entry_id: binding.lineage
                for binding in context.root_rebinding.live_bindings}

    def _lineage_map(self, context, public):
        roots = self._root_map(context)
        result = {}
        for entry in public.entries:
            identifier = entry.physical_entry.backend_entry_id
            if identifier not in roots:
                raise _EvidenceError("new or replacement physical ID has no certified lineage")
            result[identifier] = roots[identifier]
        return result

    def _descendant(self, state, context, public, transition_ids):
        mapping = self._lineage_map(context, public)
        rows, live, live_ids = [], [], set()
        for entry in public.entries:
            lineage = canonical_tuple(mapping[entry.physical_entry.backend_entry_id])
            live_ids.update(lineage)
            pd = _digest({"profile": self.frozen_configuration_id, "projection": entry.projection.observable})
            sd = _digest({"profile": self.frozen_configuration_id, "provenance": entry.provenance_ids})
            live.append(LiveLineageState(lineage, pd, sd))
            rows.append((lineage, entry.projection.observable, entry.provenance_ids))
        rows.sort(key=_digest)
        deleted = context.inventory.coverage_ids.difference(live_ids)
        observable = _digest({"profile": self.frozen_configuration_id, "rows": tuple(rows), "deleted": deleted})
        transition = _digest({"profile": self.frozen_configuration_id, "transitions": transition_ids,
                              "auxiliary": "embedded vector is derived exactly from each public row"})
        return DescendantStateCertificate(
            "memos-general-text:descendant:" + _digest((observable, transition)),
            self.backend, self.frozen_configuration_id, state.campaign_id,
            state.root_checkpoint_id, context.inventory.coverage_ids, tuple(live),
            deleted, observable, transition, transition_ids,
        )

    async def materialize_root(self, request):
        try:
            artifact, frozen = self._validate_root(request)
            handle = await self._adapter.replay_state(artifact)
            state = EphemeralMaterializedState(handle, self.backend, self.frozen_configuration_id,
                                               request.campaign_id, request.root_checkpoint_id, handle.state_id)
            capability = MemosRetrievableEntryCapability(handle)
            result = await capability.inventory(campaign_id=request.campaign_id,
                                                root_checkpoint_id=request.root_checkpoint_id,
                                                state_id=handle.state_id)
            if result.blocker:
                raise _EvidenceError(result.blocker.reason)
            rebinding = self._root_rebinding(request, frozen, result.value, artifact)
            context = _Context(handle, capability, frozen, rebinding)
            observed = self._descendant(state, context, result.value, ())
            self._states[state.state_id] = context
            return MaterializationResult.success(RootMaterialization(state, rebinding, observed))
        except _AmbiguousRebinding as error:
            if 'handle' in locals():
                await self._adapter.teardown(handle)
            return self._failure(request.campaign_id, request.root_checkpoint_id,
                                 MaterializationFailureKind.E0_REBINDING_AMBIGUOUS, str(error))
        except MemosProfileError as error:
            if 'handle' in locals():
                await self._adapter.teardown(handle)
            return self._failure(request.campaign_id, request.root_checkpoint_id,
                                 MaterializationFailureKind.BACKEND_CAPABILITY_FAILURE, str(error))
        except (_EvidenceError, MemosProfileError) as error:
            if 'handle' in locals():
                await self._adapter.teardown(handle)
            return self._failure(request.campaign_id, request.root_checkpoint_id,
                                 MaterializationFailureKind.ROOT_INITIALIZATION_INEQUIVALENT, str(error))
        except Exception as error:
            if 'handle' in locals():
                await self._adapter.teardown(handle)
            return self._failure(request.campaign_id, request.root_checkpoint_id,
                                 MaterializationFailureKind.TRANSIENT_BACKEND_FAILURE,
                                 f"MemOS root creation failed: {type(error).__name__}: {error}")

    @staticmethod
    def _parse(artifact):
        relation, payload = artifact.opportunity.relation, artifact.operation_payload
        if relation is MutationRelation.DELETION:
            if set(payload) != {"operation"} or payload["operation"] != "delete":
                raise _EvidenceError("deletion payload must be exactly {'operation': 'delete'}")
            return _Operation("delete", None, (), None)
        name = "update" if relation is MutationRelation.UPDATE else (
            "unrelated_change" if relation is MutationRelation.UNRELATED_CHANGE else None)
        if name is None or payload.get("operation") != name:
            raise _EvidenceError("MemOS materializer accepts only exact memory-mutation payloads")
        if not {"operation", "new_text", "provenance_ids"}.issubset(payload) or not set(payload).issubset(
            {"operation", "new_text", "provenance_ids", "context"}):
            raise _EvidenceError("update payload schema is invalid")
        text, provenance = payload["new_text"], payload["provenance_ids"]
        if not isinstance(text, str) or not text or not isinstance(provenance, tuple) or not provenance:
            raise _EvidenceError("new_text/provenance_ids are invalid")
        context = _thaw(payload.get("context")) if payload.get("context") is not None else None
        return _Operation(name, text, provenance, context)

    def _resolve_target(self, context, public, artifact):
        target = canonical_tuple(artifact.opportunity.target_lineage)
        mapping = self._lineage_map(context, public)
        matches = [entry for entry in public.entries
                   if mapping[entry.physical_entry.backend_entry_id] == target]
        if len(matches) != 1:
            raise _EvidenceError("target lineage does not resolve to exactly one live item")
        return matches[0], target

    @staticmethod
    def _by_id(inventory):
        return {entry.physical_entry.backend_entry_id: entry for entry in inventory.entries}

    @staticmethod
    def _receipt(receipt: OperationReceipt, state_id: str, operation: str, target: str):
        expected = "unrelated_change" if operation == "unrelated_change" else operation
        return (receipt.backend == "memos" and receipt.state_id == state_id and
                receipt.operation == expected and receipt.success and receipt.target_local_id == target and
                receipt.affected_local_ids == (target,))

    async def replay_transition(self, state, artifact):
        try:
            context, operation = self._context(state), self._parse(artifact)
            pre = await self._inventory(state, context)
            target, lineage = self._resolve_target(context, pre, artifact)
            identifier = target.physical_entry.backend_entry_id
            if operation.name == "delete":
                receipt = await self._adapter.delete(context.handle, identifier)
            elif operation.name == "update":
                receipt = await self._adapter.update(context.handle, identifier, operation.new_text or "",
                                                     provenance_ids=operation.provenance_ids, context=operation.context)
            else:
                receipt = await self._adapter.unrelated_existing_region_change(
                    context.handle, identifier, operation.new_text or "",
                    provenance_ids=operation.provenance_ids, context=operation.context)
            if not self._receipt(receipt, state.state_id, operation.name, identifier):
                raise _EvidenceError("operation receipt does not certify exact target")
            post = await self._inventory(state, context)
            before, after = self._by_id(pre), self._by_id(post)
            if operation.name == "delete":
                if set(before).difference(after) != {identifier} or set(after).difference(before):
                    raise _EvidenceError("delete did not remove exactly the target")
                if any(before[key] != after[key] for key in after):
                    raise _EvidenceError("delete changed an unrelated public item")
                successors, kind, outcome = (), AffectedTransitionKind.DELETED, PhysicalTransitionOutcome.DELETED
            else:
                if set(before) != set(after):
                    raise _EvidenceError("update added/deleted a public item")
                changed = {key for key in before if before[key] != after[key]}
                if changed != {identifier}:
                    raise _EvidenceError("update did not change exactly the target")
                observable = after[identifier].projection.observable
                if observable["memory"] != operation.new_text or after[identifier].provenance_ids != before[identifier].provenance_ids:
                    raise _EvidenceError("updated text/provenance differs from exact payload")
                successors = (PhysicalLineageEndpoint(after[identifier].physical_entry, lineage),)
                kind, outcome = AffectedTransitionKind.SAME_ID_CHANGED, PhysicalTransitionOutcome.SAME_ID
            old_query = target.projection.observable["memory"]
            closure = await context.capability.retrieve(
                campaign_id=state.campaign_id,
                root_checkpoint_id=state.root_checkpoint_id,
                state_id=state.state_id,
                query=old_query,
                top_k=max(1, len(before)),
            )
            if closure.blocker is not None:
                raise _EvidenceError(
                    "post-transition retrieval closure failed: " + closure.blocker.reason
                )
            predecessor = PhysicalLineageEndpoint(target.physical_entry, lineage)
            transition_id = f"memos-general-text:target:{artifact.artifact_id}"
            proof = _digest({"artifact": artifact.artifact_id, "operation": operation.name,
                             "pre": target.projection.observable,
                             "post": None if not successors else after[identifier].projection.observable})
            affected = AffectedPhysicalTransition(transition_id, kind, (predecessor,), successors,
                                                  backend_proof_digest=proof)
            certificate = PhysicalTransitionCertificate(
                f"memos-general-text:transition:{artifact.artifact_id}", artifact.artifact_id,
                artifact.semantic_certificate.certificate_id, state.campaign_id,
                state.root_checkpoint_id, context.inventory.coverage_ids,
                CertificationStatus.CERTIFIED, outcome, transition_id, (affected,),
                frozenset({predecessor.physical_entry}),
                frozenset(item.physical_entry for item in successors), True,
                _digest({"proof": proof, "receipt": receipt.raw}),
            )
            return MaterializationResult.success(TransitionReplay(state, certificate))
        except (_EvidenceError, MemosProfileError) as error:
            return self._failure(state.campaign_id, state.root_checkpoint_id,
                                 MaterializationFailureKind.TRANSITION_REPLAY_DIVERGED, str(error))
        except Exception as error:
            return self._failure(state.campaign_id, state.root_checkpoint_id,
                                 MaterializationFailureKind.TRANSIENT_BACKEND_FAILURE,
                                 f"MemOS transition failed: {type(error).__name__}: {error}")

    async def certify_descendant(self, state, transition_certificates):
        try:
            context = self._context(state)
            public = await self._inventory(state, context)
            return MaterializationResult.success(self._descendant(
                state, context, public, tuple(value.certificate_id for value in transition_certificates)))
        except _EvidenceError as error:
            return self._failure(state.campaign_id, state.root_checkpoint_id,
                                 MaterializationFailureKind.DESCENDANT_STATE_INEQUIVALENT, str(error))

    async def retrieve(self, state, *, query_artifact: ExecutableQueryArtifact, top_k: int):
        try:
            context = self._context(state)
            result = await context.capability.retrieve(
                campaign_id=state.campaign_id, root_checkpoint_id=state.root_checkpoint_id,
                state_id=state.state_id, query=query_artifact.executable_text, top_k=top_k)
            if result.blocker:
                raise _EvidenceError(result.blocker.reason)
            return MaterializationResult.success(result.value)
        except _EvidenceError as error:
            return self._failure(state.campaign_id, state.root_checkpoint_id,
                                 MaterializationFailureKind.BACKEND_CAPABILITY_FAILURE, str(error))

    async def discard(self, state) -> None:
        context = self._states.pop(state.state_id, None)
        if context is not None:
            await self._adapter.teardown(context.handle)
