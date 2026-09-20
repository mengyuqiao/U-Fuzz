"""Mem0 2.0.12 retrievable-entry capability.

The measured object class is the public memory-record dictionary returned by
``Memory.search()``.  Inventory uses the matching public ``Memory.get_all()``
format in the existing adapter's isolated local-Qdrant/user scope.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ufuzz.backends.base import OperationReceipt, StateHandle
from ufuzz.coverage import (
    InitialEntryLineage,
    LineageBinding,
    PhysicalEntryRef,
)
from ufuzz.retrieval_capability import (
    CapabilityOperation,
    CapabilityResult,
    RankedPhysicalRetrieval,
    RetrievableEntryInventory,
    RetrievableEntryProjection,
    RetrievableInventoryEntry,
    RetrievalCapabilityBlocker,
)


MEM0_PUBLIC_RETRIEVAL_API = "Memory.search"
MEM0_INVENTORY_API = "Memory.get_all"
MEM0_RETRIEVAL_OBJECT_CLASS = "Mem0MemoryRecord"
MEM0_INITIAL_INVENTORY_LIMIT = 64
UFUZZ_METADATA_PREFIX = "ufuzz_"
MEM0_REPLAY_OR_RUNTIME_METADATA_KEYS = frozenset(
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
    }
)


@dataclass(frozen=True, slots=True)
class Mem0Scope:
    """Adapter-compatible isolation boundary for one Mem0 checkpoint state."""

    user_id: str
    collection_name: str
    vector_store_path: str
    history_db_path: str

    @property
    def filters(self) -> dict[str, str]:
        return {"user_id": self.user_id}


class _Mem0CapabilityFailure(ValueError):
    def __init__(self, operation: CapabilityOperation, reason: str) -> None:
        self.operation = operation
        super().__init__(reason)


def _nonempty_record_id(record: Mapping[str, Any]) -> str:
    """Return the exact public ``record['id']`` or reject the record."""

    if "id" not in record:
        raise _Mem0CapabilityFailure(
            CapabilityOperation.PHYSICAL_IDENTITY,
            "Mem0 public memory record has no 'id' field",
        )
    value = record["id"]
    if not isinstance(value, str) or not value.strip():
        raise _Mem0CapabilityFailure(
            CapabilityOperation.PHYSICAL_IDENTITY,
            "Mem0 public memory record has an empty or non-string 'id'",
        )
    return value


def mem0_observable_projection(
    record: Mapping[str, Any],
) -> RetrievableEntryProjection:
    """Project replay-stable, query-independent public record state.

    The exact memory text is preserved.  The four promoted context fields are
    stable public record context that can affect how a record is interpreted
    or whether it remains retrieval-eligible. Missing optional fields are
    represented uniformly by ``None``.

    Physical ID, hash, rank, score/score details, replay-local session IDs,
    created/updated timestamps, and U-Fuzz-reserved metadata are excluded.
    Provenance remains on ``RetrievableInventoryEntry.provenance_ids``. Any
    other query-independent public metadata is preserved exactly under
    ``stable_metadata``; it is never normalized semantically.

    The current adapter's initialization path passes only ``source_metadata``
    keys in the reserved ``ufuzz_`` namespace, so its stable remainder is
    normally empty. Mem0's public formatter can nevertheless expose other
    payload metadata, which this projection retains rather than blanket
    dropping the public ``metadata`` mapping.
    """

    memory = record.get("memory")
    if not isinstance(memory, str):
        raise _Mem0CapabilityFailure(
            CapabilityOperation.PROJECTION,
            "Mem0 public memory record has no string 'memory' field",
        )
    metadata = record.get("metadata")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise _Mem0CapabilityFailure(
            CapabilityOperation.PROJECTION,
            "Mem0 public memory record metadata is not a mapping",
        )
    if any(not isinstance(key, str) for key in metadata):
        raise _Mem0CapabilityFailure(
            CapabilityOperation.PROJECTION,
            "Mem0 public memory record metadata has a non-string key",
        )
    stable_metadata = {
        key: value
        for key, value in metadata.items()
        if not key.startswith(UFUZZ_METADATA_PREFIX)
        and key not in MEM0_REPLAY_OR_RUNTIME_METADATA_KEYS
    }
    return RetrievableEntryProjection(
        backend="mem0",
        retrieval_object_class=MEM0_RETRIEVAL_OBJECT_CLASS,
        observable={
            "memory": memory,
            "role": record.get("role"),
            "actor_id": record.get("actor_id"),
            "attributed_to": record.get("attributed_to"),
            "expiration_date": record.get("expiration_date"),
            "stable_metadata": stable_metadata,
        },
    )


def _public_results(response: Any, api_name: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(response, Mapping) or "results" not in response:
        raise _Mem0CapabilityFailure(
            CapabilityOperation.INVENTORY
            if api_name == MEM0_INVENTORY_API
            else CapabilityOperation.RANKED_RETRIEVAL,
            f"{api_name} did not return the documented results wrapper",
        )
    values = response["results"]
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise _Mem0CapabilityFailure(
            CapabilityOperation.INVENTORY
            if api_name == MEM0_INVENTORY_API
            else CapabilityOperation.RANKED_RETRIEVAL,
            f"{api_name} results are not a sequence",
        )
    records: list[Mapping[str, Any]] = []
    for value in values:
        if not isinstance(value, Mapping):
            raise _Mem0CapabilityFailure(
                CapabilityOperation.INVENTORY
                if api_name == MEM0_INVENTORY_API
                else CapabilityOperation.RANKED_RETRIEVAL,
                f"{api_name} returned a non-mapping memory record",
            )
        records.append(value)
    return tuple(records)


def mem0_scope_from_state(state: StateHandle) -> Mem0Scope:
    """Validate and recover the existing Mem0Adapter isolation scope."""

    if state.backend != "mem0":
        raise ValueError("Mem0 capability requires a Mem0 state")
    config = state.metadata.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("Mem0 state has no adapter initialization config")
    vector_store = config.get("vector_store")
    if not isinstance(vector_store, Mapping) or vector_store.get("provider") != "qdrant":
        raise ValueError("Mem0 capability supports the adapter's local Qdrant scope only")
    vector_config = vector_store.get("config")
    if not isinstance(vector_config, Mapping):
        raise ValueError("Mem0 local Qdrant configuration is missing")
    if {"host", "url", "api_key", "client"}.intersection(vector_config):
        raise ValueError("Mem0 capability rejects external/shared Qdrant configuration")
    collection_name = vector_config.get("collection_name")
    expected_collection = f"ufuzz_{state.state_id.replace('-', '_')}"
    if collection_name != expected_collection:
        raise ValueError("Mem0 collection does not match the adapter state namespace")
    path = vector_config.get("path")
    history = config.get("history_db_path")
    if not isinstance(path, str) or not path:
        raise ValueError("Mem0 state has no isolated local vector-store path")
    if not isinstance(history, str) or not history:
        raise ValueError("Mem0 state has no isolated history database path")
    vector_path = Path(path)
    history_path = Path(history)
    if (
        vector_path.name != "qdrant"
        or history_path.name != "history.db"
        or vector_path.parent != history_path.parent
        or vector_path.parent.name != state.state_id
    ):
        raise ValueError("Mem0 storage paths do not match the adapter state directory")
    return Mem0Scope(
        user_id=state.state_id,
        collection_name=collection_name,
        vector_store_path=path,
        history_db_path=history,
    )


class Mem0RetrievableEntryCapability:
    """Concrete public-memory-record capability for one isolated Mem0 state."""

    backend = "mem0"
    selected_public_retrieval_api = MEM0_PUBLIC_RETRIEVAL_API
    retrieval_object_class = MEM0_RETRIEVAL_OBJECT_CLASS

    def __init__(
        self,
        state: StateHandle,
        *,
        initial_inventory_limit: int = MEM0_INITIAL_INVENTORY_LIMIT,
    ) -> None:
        if initial_inventory_limit <= 0:
            raise ValueError("initial inventory limit must be positive")
        self.state = state
        self.scope = mem0_scope_from_state(state)
        self.initial_inventory_limit = initial_inventory_limit

    def _blocker(
        self,
        root_checkpoint_id: str,
        operation: CapabilityOperation,
        reason: str,
    ) -> CapabilityResult[Any]:
        return CapabilityResult.failure(
            RetrievalCapabilityBlocker(
                backend="mem0",
                root_checkpoint_id=root_checkpoint_id,
                operation=operation,
                reason=reason,
            )
        )

    def _validate_request_scope(
        self,
        root_checkpoint_id: str,
        state_id: str,
    ) -> None:
        if root_checkpoint_id != self.state.checkpoint_id:
            raise ValueError("Mem0 capability request uses another root checkpoint")
        if state_id != self.state.state_id:
            raise ValueError("Mem0 capability request uses another backend state")

    def _complete_inventory_records(self) -> tuple[Mapping[str, Any], ...]:
        """Enumerate local Qdrant through bounded public get_all calls.

        Mem0 2.0.12 exposes no inventory cursor.  Its local-Qdrant ``list``
        implementation forwards ``top_k`` as the exact Qdrant scroll limit.
        Reissuing the public call with geometrically increasing limits until
        fewer records are returned proves exhaustion for this supported store.
        Prefix ID sets must be monotone so unstable/truncated providers fail
        rather than fabricate a complete E_0. Once exhaustion is observed,
        the same call is repeated and its ID-to-projection snapshot must match
        exactly, ignoring enumeration order. This is a stability check, not a
        transactional lock: no mutation may intentionally run while inventory
        and initialization equivalence are being established.
        """

        limit = self.initial_inventory_limit
        previous_ids: frozenset[str] = frozenset()
        while True:
            response = self.state.backend_state.get_all(
                filters=self.scope.filters,
                top_k=limit,
                show_expired=False,
            )
            records = _public_results(response, MEM0_INVENTORY_API)
            if len(records) > limit:
                raise _Mem0CapabilityFailure(
                    CapabilityOperation.INVENTORY,
                    "Memory.get_all returned more records than its requested top_k",
                )
            current_snapshot = _inventory_snapshot(records)
            current_ids = frozenset(current_snapshot)
            if not previous_ids.issubset(current_ids):
                raise _Mem0CapabilityFailure(
                    CapabilityOperation.INVENTORY,
                    "Memory.get_all results were not stable as top_k increased",
                )
            if len(records) < limit:
                repeated_response = self.state.backend_state.get_all(
                    filters=self.scope.filters,
                    top_k=limit,
                    show_expired=False,
                )
                repeated_records = _public_results(
                    repeated_response,
                    MEM0_INVENTORY_API,
                )
                if len(repeated_records) > limit:
                    raise _Mem0CapabilityFailure(
                        CapabilityOperation.INVENTORY,
                        "repeated Memory.get_all snapshot exceeded requested top_k",
                    )
                repeated_snapshot = _inventory_snapshot(repeated_records)
                if repeated_snapshot.keys() != current_snapshot.keys():
                    raise _Mem0CapabilityFailure(
                        CapabilityOperation.INVENTORY,
                        "stable initialization snapshot could not be established: "
                        "physical ID set changed after exhaustion",
                    )
                if repeated_snapshot != current_snapshot:
                    raise _Mem0CapabilityFailure(
                        CapabilityOperation.INVENTORY,
                        "stable initialization snapshot could not be established: "
                        "observable projection changed after exhaustion",
                    )
                return records
            previous_ids = current_ids
            limit *= 2

    async def inventory(
        self,
        *,
        campaign_id: str,
        root_checkpoint_id: str,
        state_id: str,
    ) -> CapabilityResult[RetrievableEntryInventory]:
        try:
            self._validate_request_scope(root_checkpoint_id, state_id)
            records = self._complete_inventory_records()
            entries = tuple(
                RetrievableInventoryEntry(
                    physical_entry=PhysicalEntryRef(
                        campaign_id=campaign_id,
                        root_checkpoint_id=root_checkpoint_id,
                        state_id=state_id,
                        backend_entry_id=_nonempty_record_id(record),
                    ),
                    projection=mem0_observable_projection(record),
                    provenance_ids=_provenance_ids(record),
                )
                for record in records
            )
            return CapabilityResult.success(
                RetrievableEntryInventory(
                    backend="mem0",
                    selected_public_retrieval_api=MEM0_PUBLIC_RETRIEVAL_API,
                    retrieval_object_class=MEM0_RETRIEVAL_OBJECT_CLASS,
                    retrieval_returnability_basis=(
                        "mem0ai 2.0.12 Memory.get_all and Memory.search use the "
                        "same public MemoryItem-derived memory-record format; "
                        "local Qdrant exhaustion was verified by increasing top_k"
                    ),
                    campaign_id=campaign_id,
                    root_checkpoint_id=root_checkpoint_id,
                    state_id=state_id,
                    entries=entries,
                )
            )
        except _Mem0CapabilityFailure as error:
            return self._blocker(
                root_checkpoint_id,
                error.operation,
                str(error),
            )
        except Exception as error:
            return self._blocker(
                root_checkpoint_id,
                CapabilityOperation.INVENTORY,
                f"scoped Mem0 inventory could not be established: {error}",
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
        try:
            self._validate_request_scope(root_checkpoint_id, state_id)
            response = self.state.backend_state.search(
                query,
                top_k=top_k,
                filters=self.scope.filters,
            )
            return CapabilityResult.success(
                self.ranked_retrieval_from_search_result(
                    response,
                    campaign_id=campaign_id,
                    root_checkpoint_id=root_checkpoint_id,
                    state_id=state_id,
                )
            )
        except _Mem0CapabilityFailure as error:
            return self._blocker(root_checkpoint_id, error.operation, str(error))
        except Exception as error:
            return self._blocker(
                root_checkpoint_id,
                CapabilityOperation.RANKED_RETRIEVAL,
                f"scoped Mem0 search could not be established: {error}",
            )

    def ranked_retrieval_from_search_result(
        self,
        response: Any,
        *,
        campaign_id: str,
        root_checkpoint_id: str,
        state_id: str,
    ) -> RankedPhysicalRetrieval:
        """Convert one actual search response without changing its rank order."""

        self._validate_request_scope(root_checkpoint_id, state_id)
        records = _public_results(response, MEM0_PUBLIC_RETRIEVAL_API)
        physical: list[PhysicalEntryRef] = []
        for record in records:
            mem0_observable_projection(record)
            physical.append(
                PhysicalEntryRef(
                    campaign_id=campaign_id,
                    root_checkpoint_id=root_checkpoint_id,
                    state_id=state_id,
                    backend_entry_id=_nonempty_record_id(record),
                )
            )
        return RankedPhysicalRetrieval(
            backend="mem0",
            selected_public_retrieval_api=MEM0_PUBLIC_RETRIEVAL_API,
            retrieval_object_class=MEM0_RETRIEVAL_OBJECT_CLASS,
            campaign_id=campaign_id,
            root_checkpoint_id=root_checkpoint_id,
            state_id=state_id,
            ranked_physical_entries=tuple(physical),
        )


def _provenance_ids(record: Mapping[str, Any]) -> tuple[str, ...]:
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        return ()
    values: list[str] = []
    direct = metadata.get("ufuzz_provenance_id")
    if isinstance(direct, str) and direct:
        values.append(direct)
    multiple = metadata.get("ufuzz_provenance_ids")
    if isinstance(multiple, Sequence) and not isinstance(multiple, (str, bytes)):
        values.extend(value for value in multiple if isinstance(value, str) and value)
    return tuple(dict.fromkeys(values))


def _inventory_snapshot(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, RetrievableEntryProjection]:
    snapshot: dict[str, RetrievableEntryProjection] = {}
    for record in records:
        record_id = _nonempty_record_id(record)
        if record_id in snapshot:
            raise _Mem0CapabilityFailure(
                CapabilityOperation.PHYSICAL_IDENTITY,
                "Memory.get_all returned duplicate physical record IDs",
            )
        snapshot[record_id] = mem0_observable_projection(record)
    return snapshot


def certify_mem0_same_id_update(
    lineage: InitialEntryLineage,
    before: PhysicalEntryRef,
    *,
    receipt: OperationReceipt,
) -> LineageBinding:
    """Certify Mem0's in-place update on the actual existing StateHandle."""

    if (
        receipt.backend != "mem0"
        or receipt.operation != "update"
        or not receipt.success
        or receipt.state_id != before.state_id
        or receipt.target_local_id != before.backend_entry_id
        or receipt.affected_local_ids != (before.backend_entry_id,)
        or not isinstance(receipt.after, Mapping)
        or receipt.after.get("id") != before.backend_entry_id
    ):
        raise ValueError("Mem0 update receipt does not certify a same-ID transition")
    return lineage.certify_same_id_update(before)


def mark_mem0_deleted(
    lineage: InitialEntryLineage,
    physical_entry: PhysicalEntryRef,
    *,
    receipt: OperationReceipt,
) -> None:
    """Mark a receipt-certified deletion while preserving frozen E_0."""

    if (
        receipt.backend != "mem0"
        or receipt.operation != "delete"
        or not receipt.success
        or receipt.state_id != physical_entry.state_id
        or receipt.target_local_id != physical_entry.backend_entry_id
        or receipt.affected_local_ids != (physical_entry.backend_entry_id,)
        or receipt.after is not None
    ):
        raise ValueError("Mem0 deletion receipt does not certify record removal")
    lineage.mark_deleted(physical_entry)
