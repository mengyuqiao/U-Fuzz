"""Retrievable-entry capability for the frozen MemOS GeneralText profile."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ufuzz.backends.base import StateHandle
from ufuzz.backends.memos import MEMOS_PROFILE_ID, PROVENANCE_KEY
from ufuzz.coverage import PhysicalEntryRef
from ufuzz.retrieval_capability import (
    CapabilityOperation,
    CapabilityResult,
    RankedPhysicalRetrieval,
    RetrievableEntryInventory,
    RetrievableEntryProjection,
    RetrievableInventoryEntry,
    RetrievalCapabilityBlocker,
)


MEMOS_PUBLIC_RETRIEVAL_API = "GeneralTextMemory.search"
MEMOS_INVENTORY_API = "GeneralTextMemory.get_all"
MEMOS_RETRIEVAL_OBJECT_CLASS = "TextualMemoryItem"


class _Failure(ValueError):
    def __init__(self, operation: CapabilityOperation, reason: str) -> None:
        self.operation = operation
        super().__init__(reason)


def _item_id(item: Any) -> str:
    value = getattr(item, "id", None)
    if not isinstance(value, str) or not value:
        raise _Failure(CapabilityOperation.PHYSICAL_IDENTITY, "item has no non-empty string ID")
    return value


def _metadata(item: Any) -> Mapping[str, Any]:
    value = getattr(item, "metadata", None)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", exclude_none=True)
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise _Failure(CapabilityOperation.PROJECTION, "item metadata is not a string-keyed mapping")
    return value


def memos_provenance_ids(item: Any) -> tuple[str, ...]:
    info = _metadata(item).get("info")
    if not isinstance(info, Mapping):
        raise _Failure(CapabilityOperation.PROJECTION, "item metadata.info is missing")
    provenance = info.get(PROVENANCE_KEY)
    if not isinstance(provenance, Mapping) or provenance.get("schema") != PROVENANCE_KEY:
        raise _Failure(CapabilityOperation.PROJECTION, "frozen provenance schema is missing")
    value = provenance.get("provenance_id")
    if not isinstance(value, str) or not value:
        raise _Failure(CapabilityOperation.PROJECTION, "frozen provenance ID is missing")
    return (value,)


def memos_observable_projection(item: Any) -> RetrievableEntryProjection:
    memory = getattr(item, "memory", None)
    if not isinstance(memory, str):
        raise _Failure(CapabilityOperation.PROJECTION, "item memory is not a string")
    # Preserve all public logical metadata. The physical UUID and embeddings
    # live outside this projection; GeneralText search supplies no rank/score
    # field on TextualMemoryItem.
    return RetrievableEntryProjection(
        backend="memos",
        retrieval_object_class=MEMOS_RETRIEVAL_OBJECT_CLASS,
        observable={"memory": memory, "metadata": dict(_metadata(item))},
    )


def memos_scope_from_state(state: StateHandle) -> tuple[Path, str]:
    if state.backend != "memos" or state.metadata.get("profile_id") != MEMOS_PROFILE_ID:
        raise ValueError("state is not the frozen MemOS GeneralText profile")
    state_dir = Path(state.metadata.get("state_dir", "")).resolve()
    qdrant = Path(state.metadata.get("qdrant_path", "")).resolve()
    collection = state.metadata.get("collection")
    config = state.metadata.get("config")
    memory_config = config.get("config") if isinstance(config, Mapping) else None
    vector = memory_config.get("vector_db") if isinstance(memory_config, Mapping) else None
    vector_config = vector.get("config") if isinstance(vector, Mapping) else None
    if (
        not state.metadata.get("adapter_owned")
        or state_dir.name != state.state_id
        or qdrant != state_dir / "qdrant"
        or not isinstance(collection, str)
        or collection != f"ufuzz_memos_{state.state_id.replace('-', '_')}"
        or not isinstance(vector, Mapping)
        or vector.get("backend") != "qdrant"
        or not isinstance(vector_config, Mapping)
        or Path(vector_config.get("path", "")).resolve() != qdrant
        or vector_config.get("collection_name") != collection
        or {"host", "port", "url", "client"}.intersection(vector_config)
    ):
        raise ValueError("MemOS embedded-Qdrant scope is not adapter-owned and isolated")
    return qdrant, collection


class MemosRetrievableEntryCapability:
    backend = "memos"
    selected_public_retrieval_api = MEMOS_PUBLIC_RETRIEVAL_API
    retrieval_object_class = MEMOS_RETRIEVAL_OBJECT_CLASS

    def __init__(self, state: StateHandle) -> None:
        self.state = state
        self.scope = memos_scope_from_state(state)

    def _blocker(self, checkpoint: str, operation: CapabilityOperation, reason: str):
        return CapabilityResult.failure(
            RetrievalCapabilityBlocker("memos", checkpoint, operation, reason)
        )

    def _scope(self, checkpoint: str, state_id: str) -> None:
        if checkpoint != self.state.checkpoint_id or state_id != self.state.state_id:
            raise ValueError("capability request uses another state/checkpoint")

    def _snapshot(self) -> tuple[Any, ...]:
        first = tuple(self.state.backend_state.get_all())
        second = tuple(self.state.backend_state.get_all())
        def keyed(values: Sequence[Any]) -> dict[str, RetrievableEntryProjection]:
            result: dict[str, RetrievableEntryProjection] = {}
            for item in values:
                identifier = _item_id(item)
                if identifier in result:
                    raise _Failure(CapabilityOperation.PHYSICAL_IDENTITY, "duplicate inventory physical ID")
                result[identifier] = memos_observable_projection(item)
                memos_provenance_ids(item)
            return result
        if keyed(first) != keyed(second):
            raise _Failure(CapabilityOperation.INVENTORY, "complete get_all snapshot is unstable")
        return first

    async def inventory(self, *, campaign_id: str, root_checkpoint_id: str,
                        state_id: str) -> CapabilityResult[RetrievableEntryInventory]:
        try:
            self._scope(root_checkpoint_id, state_id)
            entries = tuple(
                RetrievableInventoryEntry(
                    PhysicalEntryRef(campaign_id, root_checkpoint_id, state_id, _item_id(item)),
                    memos_observable_projection(item),
                    memos_provenance_ids(item),
                )
                for item in self._snapshot()
            )
            return CapabilityResult.success(RetrievableEntryInventory(
                "memos", MEMOS_PUBLIC_RETRIEVAL_API, MEMOS_RETRIEVAL_OBJECT_CLASS,
                "GeneralTextMemory.get_all enumerates the same persistent TextualMemoryItem class returned by search",
                campaign_id, root_checkpoint_id, state_id, entries,
            ))
        except _Failure as error:
            return self._blocker(root_checkpoint_id, error.operation, str(error))
        except Exception as error:
            return self._blocker(root_checkpoint_id, CapabilityOperation.INVENTORY, str(error))

    async def retrieve(self, *, campaign_id: str, root_checkpoint_id: str,
                       state_id: str, query: str, top_k: int) -> CapabilityResult[RankedPhysicalRetrieval]:
        try:
            self._scope(root_checkpoint_id, state_id)
            if top_k <= 0:
                raise _Failure(CapabilityOperation.RANKED_RETRIEVAL, "top_k must be positive")
            inventory = {_item_id(item): memos_observable_projection(item) for item in self._snapshot()}
            results = tuple(self.state.backend_state.search(query, top_k))
            if len(results) > top_k:
                raise _Failure(CapabilityOperation.RANKED_RETRIEVAL, "search exceeded top_k")
            ids: list[str] = []
            for item in results:
                identifier = _item_id(item)
                if identifier in ids:
                    raise _Failure(CapabilityOperation.RANKED_RETRIEVAL, "search repeated a physical ID")
                if identifier not in inventory:
                    raise _Failure(CapabilityOperation.RANKED_RETRIEVAL, "search returned an object absent from inventory")
                if memos_observable_projection(item) != inventory[identifier]:
                    raise _Failure(CapabilityOperation.RANKED_RETRIEVAL, "search projection disagrees with inventory")
                ids.append(identifier)
            return CapabilityResult.success(RankedPhysicalRetrieval(
                "memos", MEMOS_PUBLIC_RETRIEVAL_API, MEMOS_RETRIEVAL_OBJECT_CLASS,
                campaign_id, root_checkpoint_id, state_id,
                tuple(PhysicalEntryRef(campaign_id, root_checkpoint_id, state_id, value) for value in ids),
            ))
        except _Failure as error:
            return self._blocker(root_checkpoint_id, error.operation, str(error))
        except Exception as error:
            return self._blocker(root_checkpoint_id, CapabilityOperation.RANKED_RETRIEVAL, str(error))
