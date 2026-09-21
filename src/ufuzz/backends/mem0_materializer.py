"""Certified materialization for the deterministic Mem0 Profile A.

This module supports only the locally validated profile: mem0ai 2.0.12,
qdrant-client 1.18.0, infer=False, local isolated Qdrant, MockEmbeddings,
spaCy 3.8.14 with en-core-web-sm 3.8.0, a noncalling LLM, no reranker,
and no fastembed sparse vectors.  Entity linking and entity boosting remain
active after text-changing updates.  This is not validation of the final
infer=True RQ1 Mem0 configuration.

Public Memory records alone form E_0.  The private entity collection is read
only to certify transition-relevant backend state and never supplies coverage
entries or retrieval tokens.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from importlib import metadata, util
import json
import math
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Any, Iterator

from ufuzz.backends.base import InitializationArtifact, OperationReceipt, StateHandle
from ufuzz.backends.mem0 import Mem0Adapter
from ufuzz.backends.mem0_capability import Mem0RetrievableEntryCapability
from ufuzz.coverage import (
    CoverageEntryId,
    EVALUATOR_ONLY_KEYS,
    FrozenCheckpointInventory,
    FrozenMapping,
    PhysicalEntryRef,
)
from ufuzz.materialization import (
    EphemeralMaterializedState,
    MaterializationProtocol,
    RootMaterialization,
    RootMaterializationRequest,
    TransitionReplay,
)
from ufuzz.retrieval_capability import (
    RankedPhysicalRetrieval,
    RetrievableEntryInventory,
    RetrievableInventoryEntry,
)
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


MEM0_PROFILE_A_CONFIGURATION_ID = (
    "mem0-profile-a:mem0ai-2.0.12:qdrant-1.18.0:mock-embedding:"
    "spacy-3.8.14:en-core-web-sm-3.8.0:no-fastembed:infer-false:v1"
)
_EMBEDDER_PROVIDER = "openai"
_LLM_PROVIDER = "openai"
_PROFILE_LOCK = RLock()
_ENTITY_SCOPE_MARKER = "current-materialization-user-scope"
_ENTITY_PAYLOAD_KEYS = frozenset(
    {"data", "entity_type", "linked_memory_ids", "user_id"}
)
_RUNTIME_CONTEXT_KEYS = frozenset(
    {
        "id",
        "user_id",
        "agent_id",
        "run_id",
        "data",
        "hash",
        "text_lemmatized",
        "created_at",
        "updated_at",
        "score",
        "score_details",
        "rank",
        "actor_id",
    }
)


class _ProfileANonCallingLlm:
    """Constructor-compatible LLM that makes accidental model use fail closed."""

    def __init__(self, config: Any = None) -> None:
        self.config = config

    def generate_response(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Profile-A noncalling LLM must never be invoked")


@contextmanager
def _profile_a_runtime() -> Iterator[None]:
    """Disable every installed Mem0 telemetry path during a Profile-A call."""

    import mem0.memory.main as memory_main
    import mem0.memory.telemetry as memory_telemetry

    with _PROFILE_LOCK:
        prior_main = memory_main.MEM0_TELEMETRY
        prior_telemetry = memory_telemetry.MEM0_TELEMETRY
        memory_main.MEM0_TELEMETRY = False
        memory_telemetry.MEM0_TELEMETRY = False
        try:
            yield
        finally:
            memory_main.MEM0_TELEMETRY = prior_main
            memory_telemetry.MEM0_TELEMETRY = prior_telemetry


@contextmanager
def _profile_a_mem0_factories() -> Iterator[None]:
    """Bind Mem0 constructor factories to local deterministic implementations."""

    from mem0.configs.llms.base import BaseLlmConfig
    from mem0.utils.factory import EmbedderFactory, LlmFactory

    with _profile_a_runtime():
        prior_embedder = EmbedderFactory.provider_to_class[_EMBEDDER_PROVIDER]
        prior_llm = LlmFactory.provider_to_class[_LLM_PROVIDER]
        EmbedderFactory.provider_to_class[_EMBEDDER_PROVIDER] = (
            "mem0.embeddings.mock.MockEmbeddings"
        )
        LlmFactory.provider_to_class[_LLM_PROVIDER] = (
            "ufuzz.backends.mem0_materializer._ProfileANonCallingLlm",
            BaseLlmConfig,
        )
        try:
            yield
        finally:
            EmbedderFactory.provider_to_class[_EMBEDDER_PROVIDER] = prior_embedder
            LlmFactory.provider_to_class[_LLM_PROVIDER] = prior_llm


def _create_profile_a_memory(config: Mapping[str, Any]) -> Any:
    """Create one real Mem0 Memory without constructing an external model client."""

    from mem0 import Memory

    with _profile_a_mem0_factories():
        memory = Memory.from_config(dict(config))
    return memory


def _profile_a_config() -> dict[str, Any]:
    return {
        "vector_store": {
            "provider": "qdrant",
            "config": {"embedding_model_dims": 10},
        },
        "embedder": {"provider": _EMBEDDER_PROVIDER, "config": {}},
        "llm": {"provider": _LLM_PROVIDER, "config": {}},
        "reranker": None,
    }


def _assert_profile_a_environment() -> None:
    expected = {
        "mem0ai": "2.0.12",
        "qdrant-client": "1.18.0",
        "spacy": "3.8.14",
        "en-core-web-sm": "3.8.0",
    }
    for package, wanted in expected.items():
        actual = metadata.version(package)
        if actual != wanted:
            raise RuntimeError(
                f"Mem0 Profile A requires {package}=={wanted}; found {actual}"
            )
    if util.find_spec("fastembed") is not None:
        raise RuntimeError("Mem0 Profile A requires fastembed to be absent")

    from mem0.embeddings.base import EmbeddingBase
    from mem0.embeddings.mock import MockEmbeddings

    if MockEmbeddings.embed_batch is not EmbeddingBase.embed_batch:
        raise RuntimeError("Mem0 Profile A requires inherited EmbeddingBase.embed_batch")
    embedding = MockEmbeddings()
    expected_vector = [value / 10 for value in range(1, 11)]
    if embedding.embed("profile-a", "search") != expected_vector:
        raise RuntimeError("installed MockEmbeddings behavior differs from Profile A")
    if embedding.embed_batch(["a", "b"], "search") != [
        expected_vector,
        expected_vector,
    ]:
        raise RuntimeError("installed MockEmbeddings.embed_batch differs from Profile A")

    import spacy

    if not spacy.util.is_package("en_core_web_sm"):
        raise RuntimeError("Mem0 Profile A requires installed en_core_web_sm")


def _canonical_value(value: Any) -> Any:
    """Encode immutable evidence with explicit type tags for stable SHA-256."""

    if value is None:
        return ["none"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical evidence cannot contain non-finite floats")
        return ["float", value.hex()]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    if isinstance(value, CoverageEntryId):
        return [
            "coverage-id",
            value.campaign_id,
            value.root_checkpoint_id,
            value.opaque_id,
        ]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("canonical evidence mappings require string keys")
        return [
            "mapping",
            [
                [key, _canonical_value(item)]
                for key, item in sorted(value.items(), key=lambda pair: pair[0])
            ],
        ]
    if isinstance(value, (tuple, list)):
        return ["sequence", [_canonical_value(item) for item in value]]
    if isinstance(value, (set, frozenset)):
        encoded = [_canonical_value(item) for item in value]
        encoded.sort(key=lambda item: json.dumps(item, separators=(",", ":")))
        return ["set", encoded]
    raise TypeError(f"unsupported canonical evidence type: {type(value).__name__}")


def _stable_digest(value: Any) -> str:
    payload = json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _lineage_evidence(
    lineage: Sequence[CoverageEntryId],
) -> tuple[tuple[str, str, str], ...]:
    """Represent inherited E_0 identity with immutable JSON-like values."""

    return tuple(
        (item.campaign_id, item.root_checkpoint_id, item.opaque_id)
        for item in canonical_tuple(lineage)
    )


def _thaw(value: Any) -> Any:
    if isinstance(value, FrozenMapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    if isinstance(value, frozenset):
        return {_thaw(item) for item in value}
    return value


def _assert_search_safe(value: Any, path: str = "$") -> None:
    """Reject structured evaluator state at the concrete registry boundary."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string key")
            if key.casefold() in EVALUATOR_ONLY_KEYS:
                raise ValueError(f"{path} contains evaluator-only key {key!r}")
            _assert_search_safe(item, f"{path}.{key}")
    elif isinstance(value, (tuple, list, set, frozenset)):
        for index, item in enumerate(value):
            _assert_search_safe(item, f"{path}[{index}]")


@dataclass(frozen=True, slots=True)
class _AuxiliarySnapshot:
    canonical_rows: tuple[FrozenMapping, ...]
    digest: str


@dataclass(slots=True)
class _StateContext:
    handle: StateHandle
    capability: Mem0RetrievableEntryCapability
    inventory: FrozenCheckpointInventory
    root_rebinding: ReplayRebindingCertificate
    initialization_artifact_id: str


@dataclass(frozen=True, slots=True)
class _ParsedOperation:
    operation: str
    new_text: str | None
    provenance_ids: tuple[str, ...]
    context: Mapping[str, Any] | None


class _ProtocolEvidenceError(ValueError):
    pass


class Mem0ProfileAMaterializationProtocol(MaterializationProtocol[StateHandle]):
    """MaterializationProtocol for the exact deterministic Profile A only.

    Entity rows are certified as auxiliary transition state.  They never enter
    E_0.  Runtime entity UUIDs are excluded only while the enforced zero-or-one
    scoped-row invariant holds.
    """

    backend = "mem0"
    frozen_configuration_id = MEM0_PROFILE_A_CONFIGURATION_ID

    def __init__(
        self,
        *,
        initialization_artifacts: Mapping[str, InitializationArtifact],
        frozen_inventories: Mapping[
            tuple[str, str], FrozenCheckpointInventory
        ],
        root_dir: str | Path | None = None,
    ) -> None:
        _assert_profile_a_environment()
        artifacts = dict(initialization_artifacts)
        inventories = dict(frozen_inventories)
        if not artifacts:
            raise ValueError("Profile-A materializer requires initialization artifacts")
        if not inventories:
            raise ValueError("Profile-A materializer requires frozen inventories")
        for artifact_id, artifact in artifacts.items():
            if not isinstance(artifact_id, str) or not artifact_id:
                raise ValueError("initialization artifact IDs must be non-empty strings")
            if not isinstance(artifact, InitializationArtifact):
                raise TypeError("artifact registry values must be InitializationArtifact")
            recreated = InitializationArtifact.create(
                artifact.checkpoint_id,
                artifact.sources,
                artifact.frozen_config,
            )
            if recreated.digest != artifact.digest:
                raise ValueError(f"initialization artifact digest mismatch: {artifact_id}")
            if artifact.frozen_config.get("infer") is not False:
                raise ValueError("Profile-A initialization artifacts must freeze infer=False")
            _assert_search_safe(artifact.frozen_config, "initialization.frozen_config")
        for key, inventory in inventories.items():
            if not isinstance(key, tuple) or len(key) != 2:
                raise TypeError("inventory registry keys must be (campaign, checkpoint)")
            if key != (inventory.campaign_id, inventory.root_checkpoint_id):
                raise ValueError("frozen inventory registry key does not match inventory")
        self._artifacts = MappingProxyType(artifacts)
        self._inventories = MappingProxyType(inventories)
        self._adapter = Mem0Adapter(
            config=_profile_a_config(),
            root_dir=root_dir,
            memory_factory=_create_profile_a_memory,
            infer=False,
            allow_external_store=False,
        )
        self._states: dict[str, _StateContext] = {}

    def _failure(
        self,
        campaign_id: str,
        root_checkpoint_id: str,
        kind: MaterializationFailureKind,
        reason: str,
    ) -> MaterializationResult[Any]:
        return MaterializationResult.failed(
            MaterializationFailure(kind, campaign_id, root_checkpoint_id, reason)
        )

    def _validate_root_request(
        self, request: RootMaterializationRequest
    ) -> tuple[InitializationArtifact, FrozenCheckpointInventory]:
        if (
            request.backend,
            request.frozen_configuration_id,
        ) != (self.backend, self.frozen_configuration_id):
            raise _ProtocolEvidenceError("root request does not use Mem0 Profile A")
        artifact = self._artifacts.get(request.initialization_artifact_id)
        if artifact is None:
            raise _ProtocolEvidenceError("unknown initialization_artifact_id")
        inventory = self._inventories.get(
            (request.campaign_id, request.root_checkpoint_id)
        )
        if inventory is None:
            raise _ProtocolEvidenceError("no frozen inventory for campaign/checkpoint")
        if artifact.checkpoint_id != request.root_checkpoint_id:
            raise _ProtocolEvidenceError("initialization artifact uses another checkpoint")
        if inventory.coverage_ids != request.frozen_e0:
            raise _ProtocolEvidenceError("root request frozen E_0 differs from registry")
        if artifact.frozen_config.get("infer") is not False:
            raise _ProtocolEvidenceError("initialization artifact is not Profile A")
        recreated = InitializationArtifact.create(
            artifact.checkpoint_id, artifact.sources, artifact.frozen_config
        )
        if recreated.digest != artifact.digest:
            raise _ProtocolEvidenceError("initialization artifact digest changed")
        return artifact, inventory

    @staticmethod
    def _inventory_key_from_frozen(entry: Any) -> tuple[Any, Any]:
        return (tuple(entry.provenance_ids), entry.initial_observable_projection)

    @staticmethod
    def _inventory_key_from_fresh(
        entry: RetrievableInventoryEntry,
    ) -> tuple[Any, Any]:
        return (tuple(entry.provenance_ids), entry.projection.observable)

    def _build_root_rebinding(
        self,
        request: RootMaterializationRequest,
        frozen: FrozenCheckpointInventory,
        fresh: RetrievableEntryInventory,
        artifact: InitializationArtifact,
    ) -> ReplayRebindingCertificate:
        frozen_projection_counts = Counter(
            entry.initial_observable_projection for entry in frozen.entries
        )
        fresh_projection_counts = Counter(
            entry.projection.observable for entry in fresh.entries
        )
        if frozen_projection_counts != fresh_projection_counts:
            raise _ProtocolEvidenceError("observable initialization inventory differs")

        frozen_by_key: dict[tuple[Any, Any], list[Any]] = defaultdict(list)
        fresh_by_key: dict[tuple[Any, Any], list[RetrievableInventoryEntry]] = (
            defaultdict(list)
        )
        for entry in frozen.entries:
            frozen_by_key[self._inventory_key_from_frozen(entry)].append(entry)
        for entry in fresh.entries:
            fresh_by_key[self._inventory_key_from_fresh(entry)].append(entry)
        frozen_key_counts = {
            key: len(values) for key, values in frozen_by_key.items()
        }
        fresh_key_counts = {key: len(values) for key, values in fresh_by_key.items()}
        if frozen_key_counts != fresh_key_counts:
            raise _ProtocolEvidenceError(
                "stable provenance/projection replay-key multiset differs"
            )
        if any(len(values) != 1 for values in frozen_by_key.values()) or any(
            len(values) != 1 for values in fresh_by_key.values()
        ):
            raise RuntimeError("duplicate exact Mem0 replay key")

        bindings: list[ReplayBinding] = []
        for key, frozen_values in frozen_by_key.items():
            coverage_entry = frozen_values[0]
            fresh_entry = fresh_by_key[key][0]
            bindings.append(
                ReplayBinding(
                    fresh_entry.physical_entry,
                    (coverage_entry.coverage_id,),
                    _stable_digest(
                        {
                            "profile": self.frozen_configuration_id,
                            "coverage": coverage_entry.coverage_id,
                            "provenance": key[0],
                            "projection": key[1],
                        }
                    ),
                )
            )
        bindings.sort(key=lambda item: item.lineage)
        proof = _stable_digest(
            {
                "profile": self.frozen_configuration_id,
                "artifact_digest": artifact.digest,
                "bindings": tuple(
                    (binding.lineage, binding.binding_proof_digest)
                    for binding in bindings
                ),
            }
        )
        return ReplayRebindingCertificate(
            certificate_id=f"mem0-profile-a:root:{artifact.digest}",
            kind=RebindingKind.ROOT,
            backend=self.backend,
            frozen_configuration_id=self.frozen_configuration_id,
            campaign_id=request.campaign_id,
            root_checkpoint_id=request.root_checkpoint_id,
            frozen_e0=request.frozen_e0,
            live_bindings=tuple(bindings),
            expected_live_lineage_multiset=tuple(
                binding.lineage for binding in bindings
            ),
            deleted_ids=frozenset(),
            backend_proof_digest=proof,
        )

    def _context(
        self, state: EphemeralMaterializedState[StateHandle]
    ) -> _StateContext:
        context = self._states.get(state.state_id)
        if context is None or context.handle is not state.handle:
            raise _ProtocolEvidenceError("unknown or retired Profile-A state")
        if (
            state.backend,
            state.frozen_configuration_id,
            state.campaign_id,
            state.root_checkpoint_id,
            state.state_id,
        ) != (
            self.backend,
            self.frozen_configuration_id,
            context.inventory.campaign_id,
            context.inventory.root_checkpoint_id,
            context.handle.state_id,
        ):
            raise _ProtocolEvidenceError("ephemeral state scope differs from registry")
        return context

    async def _public_inventory(
        self,
        state: EphemeralMaterializedState[StateHandle],
        context: _StateContext,
    ) -> RetrievableEntryInventory:
        with _profile_a_runtime():
            result = await context.capability.inventory(
                campaign_id=state.campaign_id,
                root_checkpoint_id=state.root_checkpoint_id,
                state_id=state.state_id,
            )
        if result.blocker is not None:
            raise _ProtocolEvidenceError(result.blocker.reason)
        assert result.value is not None
        return result.value

    @staticmethod
    def _root_lineage_map(
        context: _StateContext,
    ) -> dict[str, tuple[CoverageEntryId, ...]]:
        return {
            binding.physical_entry.backend_entry_id: binding.lineage
            for binding in context.root_rebinding.live_bindings
        }

    def _current_lineage_map(
        self,
        context: _StateContext,
        public: RetrievableEntryInventory,
    ) -> dict[str, tuple[CoverageEntryId, ...]]:
        root = self._root_lineage_map(context)
        current: dict[str, tuple[CoverageEntryId, ...]] = {}
        for entry in public.entries:
            backend_id = entry.physical_entry.backend_entry_id
            lineage = root.get(backend_id)
            if lineage is None:
                raise _ProtocolEvidenceError(
                    "Profile A observed a new public memory ID without lineage"
                )
            current[backend_id] = lineage
        return current

    @staticmethod
    def _point_evidence(point: Any) -> tuple[Any, Any, Any]:
        point_id = str(getattr(point, "id", ""))
        payload = getattr(point, "payload", None)
        vector = getattr(point, "vector", None)
        if not point_id:
            raise _ProtocolEvidenceError("entity row has no physical ID")
        return point_id, payload, vector

    @staticmethod
    def _dense_vector(vector: Any) -> tuple[float, ...]:
        if isinstance(vector, Mapping):
            keys = set(vector)
            if keys != {""}:
                raise _ProtocolEvidenceError(
                    "Profile A entity row has an unexpected vector slot"
                )
            vector = vector[""]
        if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)):
            raise _ProtocolEvidenceError("entity dense vector is malformed")
        if len(vector) != 10:
            raise _ProtocolEvidenceError("Profile A entity vector dimension is not 10")
        values: list[float] = []
        for item in vector:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise _ProtocolEvidenceError("entity vector contains a non-number")
            value = float(item)
            if not math.isfinite(value):
                raise _ProtocolEvidenceError("entity vector contains a non-finite value")
            values.append(value)
        return tuple(values)

    @staticmethod
    def _enumerate_entity_points(memory: Any) -> tuple[Any, ...]:
        if memory._entity_store is None:
            return ()
        store = memory._entity_store

        def enumerate_once() -> tuple[Any, ...]:
            values: list[Any] = []
            offset = None
            while True:
                page, offset = store.client.scroll(
                    collection_name=store.collection_name,
                    limit=64,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True,
                )
                values.extend(page)
                if offset is None:
                    break
            exact_count = store.client.count(
                collection_name=store.collection_name,
                exact=True,
            ).count
            if exact_count != len(values):
                raise _ProtocolEvidenceError(
                    "entity scroll count differs from exact collection count"
                )
            values.sort(key=lambda point: str(point.id))
            return tuple(values)

        first = enumerate_once()
        second = enumerate_once()
        first_evidence = tuple(
            Mem0ProfileAMaterializationProtocol._point_evidence(point)
            for point in first
        )
        second_evidence = tuple(
            Mem0ProfileAMaterializationProtocol._point_evidence(point)
            for point in second
        )
        if first_evidence != second_evidence:
            raise _ProtocolEvidenceError("entity collection snapshot is unstable")
        return first

    def _auxiliary_snapshot(
        self,
        state: EphemeralMaterializedState[StateHandle],
        context: _StateContext,
        public: RetrievableEntryInventory,
    ) -> _AuxiliarySnapshot:
        memory = context.handle.backend_state
        entity_store = memory._entity_store
        if entity_store is not None:
            expected_collection = f"{memory.collection_name}_entities"
            if entity_store.collection_name != expected_collection:
                raise _ProtocolEvidenceError("entity collection ownership mismatch")
            if entity_store.client is not memory.vector_store.client:
                raise _ProtocolEvidenceError("entity collection does not share owned client")
        points = self._enumerate_entity_points(memory)
        if len(points) > 1:
            raise _ProtocolEvidenceError(
                "Profile A zero-or-one entity-row invariant was violated"
            )
        lineage_by_id = self._current_lineage_map(context, public)
        canonical_rows: list[FrozenMapping] = []
        for point in points:
            _, payload_value, vector_value = self._point_evidence(point)
            if not isinstance(payload_value, Mapping):
                raise _ProtocolEvidenceError("entity payload is not a mapping")
            payload = dict(payload_value)
            if set(payload) != _ENTITY_PAYLOAD_KEYS:
                raise _ProtocolEvidenceError(
                    "entity payload differs from exact Profile-A schema"
                )
            data = payload["data"]
            entity_type = payload["entity_type"]
            linked = payload["linked_memory_ids"]
            if not isinstance(data, str) or not data:
                raise _ProtocolEvidenceError("entity data must be a non-empty string")
            if not isinstance(entity_type, str) or not entity_type:
                raise _ProtocolEvidenceError("entity_type must be a non-empty string")
            if not isinstance(linked, list) or any(
                not isinstance(item, str) or not item for item in linked
            ):
                raise _ProtocolEvidenceError("linked_memory_ids is malformed")
            if payload["user_id"] != state.state_id:
                raise _ProtocolEvidenceError("entity row belongs to another user scope")
            linked_lineages: list[tuple[CoverageEntryId, ...]] = []
            for memory_id in linked:
                lineage = lineage_by_id.get(memory_id)
                if lineage is None:
                    raise _ProtocolEvidenceError(
                        "entity row contains an unresolved or stale memory link"
                    )
                linked_lineages.append(canonical_tuple(lineage))
            linked_lineages.sort()
            normalized_key = memory._normalize_entity_text(data)
            if not normalized_key:
                raise _ProtocolEvidenceError("entity normalized key is empty")
            canonical_rows.append(
                FrozenMapping(
                    {
                        "normalized_key": normalized_key,
                        "data": data,
                        "entity_type": entity_type,
                        "dense_vector": self._dense_vector(vector_value),
                        "sparse_vector_state": "absent",
                        "linked_lineages": tuple(
                            _lineage_evidence(lineage)
                            for lineage in linked_lineages
                        ),
                        "scope": _ENTITY_SCOPE_MARKER,
                        "multiplicity": 1,
                    }
                )
            )
        canonical = tuple(canonical_rows)
        return _AuxiliarySnapshot(
            canonical,
            _stable_digest(
                {
                    "profile": self.frozen_configuration_id,
                    "entity_rows": canonical,
                }
            ),
        )

    def _build_descendant(
        self,
        state: EphemeralMaterializedState[StateHandle],
        context: _StateContext,
        public: RetrievableEntryInventory,
        auxiliary: _AuxiliarySnapshot,
        transition_ids: tuple[str, ...],
    ) -> DescendantStateCertificate:
        lineage_by_id = self._current_lineage_map(context, public)
        live_states: list[LiveLineageState] = []
        public_rows: list[Any] = []
        live_ids: set[CoverageEntryId] = set()
        for entry in public.entries:
            backend_id = entry.physical_entry.backend_entry_id
            lineage = canonical_tuple(lineage_by_id[backend_id])
            live_ids.update(lineage)
            projection_digest = _stable_digest(
                {"profile": self.frozen_configuration_id, "projection": entry.projection.observable}
            )
            provenance_digest = _stable_digest(
                {"profile": self.frozen_configuration_id, "provenance": entry.provenance_ids}
            )
            live_states.append(
                LiveLineageState(lineage, projection_digest, provenance_digest)
            )
            public_rows.append(
                (lineage, entry.projection.observable, tuple(entry.provenance_ids))
            )
        public_rows.sort(key=lambda value: _stable_digest(value))
        live_states.sort(
            key=lambda item: _stable_digest(
                (
                    item.lineage,
                    item.observable_state_digest,
                    item.stable_provenance_digest,
                )
            )
        )
        deleted = context.inventory.coverage_ids.difference(live_ids)
        observable_digest = _stable_digest(
            {
                "profile": self.frozen_configuration_id,
                "public_rows": tuple(public_rows),
                "deleted": deleted,
                "transition_ids": transition_ids,
            }
        )
        transition_digest = _stable_digest(
            {
                "profile": self.frozen_configuration_id,
                "auxiliary_digest": auxiliary.digest,
                "auxiliary_rows": auxiliary.canonical_rows,
                "transition_ids": transition_ids,
            }
        )
        certificate_id = "mem0-profile-a:descendant:" + _stable_digest(
            {
                "observable": observable_digest,
                "transition": transition_digest,
                "live": tuple(
                    (
                        item.lineage,
                        item.observable_state_digest,
                        item.stable_provenance_digest,
                    )
                    for item in live_states
                ),
                "deleted": deleted,
                "transitions": transition_ids,
            }
        )
        return DescendantStateCertificate(
            certificate_id=certificate_id,
            backend=self.backend,
            frozen_configuration_id=self.frozen_configuration_id,
            campaign_id=state.campaign_id,
            root_checkpoint_id=state.root_checkpoint_id,
            frozen_e0=context.inventory.coverage_ids,
            live_lineage_states=tuple(live_states),
            deleted_ids=deleted,
            observable_state_digest=observable_digest,
            transition_state_digest=transition_digest,
            accepted_transition_certificate_ids=transition_ids,
        )

    def _validate_live_memory(self, state: StateHandle) -> None:
        from mem0.embeddings.base import EmbeddingBase
        from mem0.embeddings.mock import MockEmbeddings
        from mem0.vector_stores.qdrant import Qdrant

        memory = state.backend_state
        if type(memory.embedding_model) is not MockEmbeddings:
            raise _ProtocolEvidenceError("state does not use exact MockEmbeddings")
        if type(memory.embedding_model).embed_batch is not EmbeddingBase.embed_batch:
            raise _ProtocolEvidenceError("state does not use inherited embed_batch")
        if type(memory.llm) is not _ProfileANonCallingLlm:
            raise _ProtocolEvidenceError("state does not use the noncalling LLM")
        if memory.reranker is not None:
            raise _ProtocolEvidenceError("Profile A cannot use a reranker")
        if not isinstance(memory.vector_store, Qdrant) or not memory.vector_store.is_local:
            raise _ProtocolEvidenceError("Profile A requires isolated local Qdrant")
        if memory.vector_store.embedding_model_dims != 10:
            raise _ProtocolEvidenceError("Profile A Qdrant dimension must be 10")

    async def _retire_handle(self, handle: StateHandle) -> None:
        memory = handle.backend_state
        try:
            with _profile_a_runtime():
                await self._adapter.teardown(handle)
        finally:
            try:
                memory.close()
            except Exception:
                pass
            try:
                memory.vector_store.client.close()
            except Exception:
                pass

    async def materialize_root(
        self, request: RootMaterializationRequest
    ) -> MaterializationResult[RootMaterialization[StateHandle]]:
        try:
            artifact, inventory = self._validate_root_request(request)
        except (TypeError, ValueError, _ProtocolEvidenceError) as error:
            return self._failure(
                request.campaign_id,
                request.root_checkpoint_id,
                MaterializationFailureKind.BACKEND_CAPABILITY_FAILURE,
                str(error),
            )
        handle: StateHandle | None = None
        try:
            with _profile_a_runtime():
                handle = await self._adapter.replay_state(artifact)
            self._validate_live_memory(handle)
            capability = Mem0RetrievableEntryCapability(handle)
            state = EphemeralMaterializedState(
                handle=handle,
                backend=self.backend,
                frozen_configuration_id=self.frozen_configuration_id,
                campaign_id=request.campaign_id,
                root_checkpoint_id=request.root_checkpoint_id,
                state_id=handle.state_id,
            )
            with _profile_a_runtime():
                inventory_result = await capability.inventory(
                    campaign_id=request.campaign_id,
                    root_checkpoint_id=request.root_checkpoint_id,
                    state_id=handle.state_id,
                )
            if inventory_result.blocker is not None:
                raise _ProtocolEvidenceError(inventory_result.blocker.reason)
            assert inventory_result.value is not None
            fresh = inventory_result.value
            try:
                rebinding = self._build_root_rebinding(
                    request, inventory, fresh, artifact
                )
            except RuntimeError as error:
                await self._retire_handle(handle)
                return self._failure(
                    request.campaign_id,
                    request.root_checkpoint_id,
                    MaterializationFailureKind.E0_REBINDING_AMBIGUOUS,
                    str(error),
                )
            context = _StateContext(
                handle, capability, inventory, rebinding, request.initialization_artifact_id
            )
            auxiliary = self._auxiliary_snapshot(state, context, fresh)
            if auxiliary.canonical_rows:
                raise _ProtocolEvidenceError("Profile-A root auxiliary state is not empty")
            observed = self._build_descendant(state, context, fresh, auxiliary, ())
            self._states[state.state_id] = context
            return MaterializationResult.success(
                RootMaterialization(state, rebinding, observed)
            )
        except _ProtocolEvidenceError as error:
            if handle is not None:
                await self._retire_handle(handle)
            return self._failure(
                request.campaign_id,
                request.root_checkpoint_id,
                MaterializationFailureKind.ROOT_INITIALIZATION_INEQUIVALENT,
                str(error),
            )
        except Exception as error:
            if handle is not None:
                await self._retire_handle(handle)
            return self._failure(
                request.campaign_id,
                request.root_checkpoint_id,
                MaterializationFailureKind.TRANSIENT_BACKEND_FAILURE,
                f"Profile-A root creation failed: {type(error).__name__}: {error}",
            )

    @staticmethod
    def _parse_operation(artifact: RealizedMutationArtifact) -> _ParsedOperation:
        relation = artifact.opportunity.relation
        payload = artifact.operation_payload
        if relation is MutationRelation.DELETION:
            if set(payload) != {"operation"} or payload["operation"] != "delete":
                raise _ProtocolEvidenceError(
                    "Profile-A deletion payload must be exactly {'operation': 'delete'}"
                )
            return _ParsedOperation("delete", None, (), None)
        expected_operation = (
            "update"
            if relation is MutationRelation.UPDATE
            else "unrelated_change"
            if relation is MutationRelation.UNRELATED_CHANGE
            else None
        )
        if expected_operation is None:
            raise _ProtocolEvidenceError("Profile A replays memory mutations only")
        allowed = {"operation", "new_text", "provenance_ids", "context"}
        required = {"operation", "new_text", "provenance_ids"}
        if not required.issubset(payload) or not set(payload).issubset(allowed):
            raise _ProtocolEvidenceError("Profile-A update payload schema is invalid")
        if payload["operation"] != expected_operation:
            raise _ProtocolEvidenceError("operation payload disagrees with mutation relation")
        new_text = payload["new_text"]
        provenance = payload["provenance_ids"]
        if not isinstance(new_text, str) or not new_text:
            raise _ProtocolEvidenceError("new_text must be a non-empty exact string")
        if not isinstance(provenance, tuple) or any(
            not isinstance(item, str) or not item for item in provenance
        ):
            raise _ProtocolEvidenceError("provenance_ids must be an immutable string sequence")
        if len(set(provenance)) != len(provenance):
            raise _ProtocolEvidenceError("provenance_ids cannot repeat")
        context_value = payload.get("context")
        if context_value is not None and not isinstance(context_value, FrozenMapping):
            raise _ProtocolEvidenceError("context must be an immutable mapping")
        context = _thaw(context_value) if context_value is not None else None
        if context:
            invalid = sorted(
                key
                for key in context
                if key in _RUNTIME_CONTEXT_KEYS or key.startswith("ufuzz_")
            )
            if invalid:
                raise _ProtocolEvidenceError(
                    "Profile-A context contains runtime/reserved keys: "
                    + ", ".join(invalid)
                )
        return _ParsedOperation(expected_operation, new_text, provenance, context)

    @staticmethod
    def _public_by_id(
        inventory: RetrievableEntryInventory,
    ) -> dict[str, RetrievableInventoryEntry]:
        return {
            entry.physical_entry.backend_entry_id: entry
            for entry in inventory.entries
        }

    def _resolve_target(
        self,
        context: _StateContext,
        public: RetrievableEntryInventory,
        artifact: RealizedMutationArtifact,
    ) -> tuple[RetrievableInventoryEntry, tuple[CoverageEntryId, ...]]:
        target = canonical_tuple(artifact.opportunity.target_lineage)
        mapping = self._current_lineage_map(context, public)
        matches = [
            entry
            for entry in public.entries
            if mapping[entry.physical_entry.backend_entry_id] == target
        ]
        if len(matches) != 1:
            raise _ProtocolEvidenceError(
                "target lineage does not resolve to exactly one live Mem0 record"
            )
        return matches[0], target

    @staticmethod
    def _receipt_matches(
        receipt: OperationReceipt,
        state_id: str,
        operation: str,
        target_id: str,
    ) -> bool:
        return (
            receipt.backend == "mem0"
            and receipt.state_id == state_id
            and receipt.operation == operation
            and receipt.success
            and receipt.target_local_id == target_id
            and receipt.affected_local_ids == (target_id,)
        )

    def _validate_update_diff(
        self,
        pre: RetrievableEntryInventory,
        post: RetrievableEntryInventory,
        target_id: str,
        operation: _ParsedOperation,
    ) -> None:
        before = self._public_by_id(pre)
        after = self._public_by_id(post)
        if set(before) != set(after):
            raise _ProtocolEvidenceError(
                "Profile-A update added or deleted a public memory ID"
            )
        changed = {
            backend_id
            for backend_id in before
            if (
                before[backend_id].projection != after[backend_id].projection
                or before[backend_id].provenance_ids
                != after[backend_id].provenance_ids
            )
        }
        if changed != {target_id}:
            raise _ProtocolEvidenceError(
                "Profile-A update did not change exactly the selected public target"
            )
        target = after[target_id]
        observable = target.projection.observable
        if not isinstance(observable, Mapping) or observable["memory"] != operation.new_text:
            raise _ProtocolEvidenceError("updated target text differs from exact payload")
        expected_provenance = tuple(
            sorted(
                set(before[target_id].provenance_ids).union(
                    operation.provenance_ids
                )
            )
        )
        if target.provenance_ids != expected_provenance:
            raise _ProtocolEvidenceError("updated target provenance differs from payload")

    def _validate_delete_diff(
        self,
        pre: RetrievableEntryInventory,
        post: RetrievableEntryInventory,
        target_id: str,
    ) -> None:
        before = self._public_by_id(pre)
        after = self._public_by_id(post)
        if set(before).difference(after) != {target_id}:
            raise _ProtocolEvidenceError("delete did not remove exactly the target ID")
        if set(after).difference(before):
            raise _ProtocolEvidenceError("delete unexpectedly created a public memory")
        for backend_id in after:
            if before[backend_id] != after[backend_id]:
                raise _ProtocolEvidenceError("delete changed an unaffected public memory")

    async def replay_transition(
        self,
        state: EphemeralMaterializedState[StateHandle],
        artifact: RealizedMutationArtifact,
    ) -> MaterializationResult[TransitionReplay[StateHandle]]:
        try:
            context = self._context(state)
            operation = self._parse_operation(artifact)
            pre = await self._public_inventory(state, context)
            target_entry, target_lineage = self._resolve_target(context, pre, artifact)
            pre_auxiliary = self._auxiliary_snapshot(state, context, pre)
            target_id = target_entry.physical_entry.backend_entry_id
            if operation.operation == "delete":
                with _profile_a_runtime():
                    receipt = await self._adapter.delete(context.handle, target_id)
                receipt_operation = "delete"
            elif operation.operation == "update":
                with _profile_a_runtime():
                    receipt = await self._adapter.update(
                        context.handle,
                        target_id,
                        operation.new_text or "",
                        provenance_ids=operation.provenance_ids,
                        context=operation.context,
                    )
                receipt_operation = "update"
            else:
                with _profile_a_runtime():
                    receipt = await self._adapter.unrelated_existing_region_change(
                        context.handle,
                        target_id,
                        operation.new_text or "",
                        provenance_ids=operation.provenance_ids,
                        context=operation.context,
                    )
                receipt_operation = "unrelated_existing_region_change"
            if not self._receipt_matches(
                receipt, state.state_id, receipt_operation, target_id
            ):
                raise _ProtocolEvidenceError("Mem0 OperationReceipt did not match replay")
            post = await self._public_inventory(state, context)
            if operation.operation == "delete":
                self._validate_delete_diff(pre, post, target_id)
            else:
                self._validate_update_diff(pre, post, target_id, operation)
            post_auxiliary = self._auxiliary_snapshot(state, context, post)

            predecessor = PhysicalLineageEndpoint(
                target_entry.physical_entry, target_lineage
            )
            if operation.operation == "delete":
                successors: tuple[PhysicalLineageEndpoint, ...] = ()
                kind = AffectedTransitionKind.DELETED
                outcome = PhysicalTransitionOutcome.DELETED
            else:
                post_target = self._public_by_id(post)[target_id]
                successors = (
                    PhysicalLineageEndpoint(
                        post_target.physical_entry, target_lineage
                    ),
                )
                kind = AffectedTransitionKind.SAME_ID_CHANGED
                outcome = PhysicalTransitionOutcome.SAME_ID
            transition_id = f"mem0-profile-a:target:{artifact.artifact_id}"
            group_proof = _stable_digest(
                {
                    "profile": self.frozen_configuration_id,
                    "artifact": artifact.artifact_id,
                    "target_lineage": target_lineage,
                    "operation": operation.operation,
                    "pre_projection": target_entry.projection.observable,
                    "post_projection": (
                        None
                        if operation.operation == "delete"
                        else self._public_by_id(post)[target_id].projection.observable
                    ),
                    "pre_auxiliary": pre_auxiliary.digest,
                    "post_auxiliary": post_auxiliary.digest,
                }
            )
            affected = AffectedPhysicalTransition(
                transition_id=transition_id,
                kind=kind,
                predecessors=(predecessor,),
                successors=successors,
                backend_proof_digest=group_proof,
            )
            certificate_id = f"mem0-profile-a:transition:{artifact.artifact_id}"
            certificate = PhysicalTransitionCertificate(
                certificate_id=certificate_id,
                realized_artifact_id=artifact.artifact_id,
                semantic_certificate_id=artifact.semantic_certificate.certificate_id,
                campaign_id=state.campaign_id,
                root_checkpoint_id=state.root_checkpoint_id,
                frozen_e0=context.inventory.coverage_ids,
                status=CertificationStatus.CERTIFIED,
                observed_outcome=outcome,
                target_transition_id=transition_id,
                affected=(affected,),
                observed_affected_predecessors=frozenset({predecessor.physical_entry}),
                observed_affected_successors=frozenset(
                    item.physical_entry for item in successors
                ),
                semantic_postcondition_satisfied=True,
                backend_proof_digest=_stable_digest(
                    {
                        "profile": self.frozen_configuration_id,
                        "certificate": certificate_id,
                        "group_proof": group_proof,
                        "receipt_operation": receipt.operation,
                        "receipt_provenance": receipt.provenance_ids,
                    }
                ),
            )
            return MaterializationResult.success(TransitionReplay(state, certificate))
        except _ProtocolEvidenceError as error:
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
                f"Profile-A transition replay failed: {type(error).__name__}: {error}",
            )

    async def certify_descendant(
        self,
        state: EphemeralMaterializedState[StateHandle],
        transition_certificates: tuple[PhysicalTransitionCertificate, ...],
    ) -> MaterializationResult[DescendantStateCertificate]:
        try:
            context = self._context(state)
            public = await self._public_inventory(state, context)
            auxiliary = self._auxiliary_snapshot(state, context, public)
            transition_ids = tuple(
                certificate.certificate_id
                for certificate in transition_certificates
            )
            return MaterializationResult.success(
                self._build_descendant(
                    state, context, public, auxiliary, transition_ids
                )
            )
        except _ProtocolEvidenceError as error:
            return self._failure(
                state.campaign_id,
                state.root_checkpoint_id,
                MaterializationFailureKind.DESCENDANT_STATE_INEQUIVALENT,
                str(error),
            )
        except Exception as error:
            return self._failure(
                state.campaign_id,
                state.root_checkpoint_id,
                MaterializationFailureKind.TRANSIENT_BACKEND_FAILURE,
                f"Profile-A descendant certification failed: {type(error).__name__}: {error}",
            )

    async def retrieve(
        self,
        state: EphemeralMaterializedState[StateHandle],
        *,
        query_artifact: ExecutableQueryArtifact,
        top_k: int,
    ) -> MaterializationResult[RankedPhysicalRetrieval]:
        try:
            context = self._context(state)
            with _profile_a_runtime():
                result = await context.capability.retrieve(
                    campaign_id=state.campaign_id,
                    root_checkpoint_id=state.root_checkpoint_id,
                    state_id=state.state_id,
                    query=query_artifact.executable_text,
                    top_k=top_k,
                )
            if result.blocker is not None:
                raise _ProtocolEvidenceError(result.blocker.reason)
            assert result.value is not None
            return MaterializationResult.success(result.value)
        except _ProtocolEvidenceError as error:
            return self._failure(
                state.campaign_id,
                state.root_checkpoint_id,
                MaterializationFailureKind.BACKEND_CAPABILITY_FAILURE,
                str(error),
            )
        except Exception as error:
            return self._failure(
                state.campaign_id,
                state.root_checkpoint_id,
                MaterializationFailureKind.TRANSIENT_BACKEND_FAILURE,
                f"Profile-A retrieval failed: {type(error).__name__}: {error}",
            )

    async def discard(
        self, state: EphemeralMaterializedState[StateHandle]
    ) -> None:
        context = self._states.pop(state.state_id, None)
        if context is None:
            return
        await self._retire_handle(context.handle)
