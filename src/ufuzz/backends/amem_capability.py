"""A-Mem retrievable-entry capability at the frozen upstream revision.

The measured objects are the memory-note dictionaries returned by
``AgenticMemorySystem.search()``.  Inventory uses ``self.memories`` only after
proving exact ID agreement with the Chroma collection that supplies search
candidates, then repeats the ID-to-public-projection snapshot for stability.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ufuzz.backends.amem import AMEM_COMMIT
from ufuzz.backends.base import OperationReceipt, StateHandle
from ufuzz.coverage import InitialEntryLineage, LineageBinding, PhysicalEntryRef
from ufuzz.retrieval_capability import (
    CapabilityOperation,
    CapabilityResult,
    RankedPhysicalRetrieval,
    RetrievableEntryInventory,
    RetrievableEntryProjection,
    RetrievableInventoryEntry,
    RetrievalCapabilityBlocker,
)


AMEM_PUBLIC_RETRIEVAL_API = "AgenticMemorySystem.search"
AMEM_INVENTORY_SOURCES = "AgenticMemorySystem.memories + Chroma collection.get"
AMEM_RETRIEVAL_OBJECT_CLASS = "AMemMemoryNoteResult"
AMEM_COLLECTION_NAME = "memories"
UFUZZ_PROVENANCE_TAG_PREFIX = "ufuzz-source:"


@dataclass(frozen=True, slots=True)
class AMemScope:
    """The existing adapter's one-StateHandle A-Mem isolation boundary."""

    state_id: str
    collection_name: str
    backend_object_identity: int
    retriever_object_identity: int


class _AMemCapabilityFailure(ValueError):
    def __init__(self, operation: CapabilityOperation, reason: str) -> None:
        self.operation = operation
        super().__init__(reason)


def _nonempty_memory_id(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _AMemCapabilityFailure(
            CapabilityOperation.PHYSICAL_IDENTITY,
            f"{where} has an empty or non-string memory ID",
        )
    return value


def _ordered_strings(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise _AMemCapabilityFailure(
            CapabilityOperation.PROJECTION,
            f"A-Mem public {field} is not a sequence",
        )
    if any(not isinstance(item, str) for item in value):
        raise _AMemCapabilityFailure(
            CapabilityOperation.PROJECTION,
            f"A-Mem public {field} contains a non-string value",
        )
    return tuple(value)


def amem_observable_projection(value: Any) -> RetrievableEntryProjection:
    """Project the exact query-independent fields exposed by public search.

    At commit ``ceffb860...``, ``search()`` returns ``id``, ``content``,
    ``context``, ``keywords``, and query-specific ``score``.  The projection
    therefore preserves content and context exactly. Keyword order is also
    preserved: the pinned extraction prompt defines it as most-to-least
    important, storage serializes that list intact, the evolution prompt
    consumes it in that order, and public search returns it intact. It
    excludes physical ID, score/rank, timestamps, links, tags, category,
    counters, and evolution history because those fields are either volatile,
    harness-owned, or not exposed by the selected public retrieval API.
    """

    if isinstance(value, Mapping):
        content = value.get("content")
        context = value.get("context")
        keywords = value.get("keywords")
    else:
        content = getattr(value, "content", None)
        context = getattr(value, "context", None)
        keywords = getattr(value, "keywords", None)
    if not isinstance(content, str):
        raise _AMemCapabilityFailure(
            CapabilityOperation.PROJECTION,
            "A-Mem public memory note has no string content",
        )
    if not isinstance(context, str):
        raise _AMemCapabilityFailure(
            CapabilityOperation.PROJECTION,
            "A-Mem public memory note has no string context",
        )
    return RetrievableEntryProjection(
        backend="a-mem",
        retrieval_object_class=AMEM_RETRIEVAL_OBJECT_CLASS,
        observable={
            "content": content,
            "context": context,
            "keywords": _ordered_strings(keywords, field="keywords"),
        },
    )


def _note_id(note: Any) -> str:
    return _nonempty_memory_id(getattr(note, "id", None), where="MemoryNote")


def _provenance_ids(note: Any) -> tuple[str, ...]:
    tags = getattr(note, "tags", ())
    if not isinstance(tags, Sequence) or isinstance(tags, (str, bytes)):
        return ()
    values = [
        tag.removeprefix(UFUZZ_PROVENANCE_TAG_PREFIX)
        for tag in tags
        if isinstance(tag, str)
        and tag.startswith(UFUZZ_PROVENANCE_TAG_PREFIX)
        and tag != UFUZZ_PROVENANCE_TAG_PREFIX
    ]
    return tuple(dict.fromkeys(values))


def _chroma_ids(collection: Any) -> tuple[str, ...]:
    response = collection.get()
    if not isinstance(response, Mapping):
        raise _AMemCapabilityFailure(
            CapabilityOperation.INVENTORY,
            "A-Mem Chroma collection.get() did not return a mapping",
        )
    raw_ids = response.get("ids")
    if not isinstance(raw_ids, Sequence) or isinstance(raw_ids, (str, bytes)):
        raise _AMemCapabilityFailure(
            CapabilityOperation.INVENTORY,
            "A-Mem Chroma collection.get() did not return an ID sequence",
        )
    ids = tuple(
        _nonempty_memory_id(value, where="Chroma collection") for value in raw_ids
    )
    if len(set(ids)) != len(ids):
        raise _AMemCapabilityFailure(
            CapabilityOperation.PHYSICAL_IDENTITY,
            "A-Mem Chroma collection contains duplicate physical IDs",
        )
    count = collection.count()
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise _AMemCapabilityFailure(
            CapabilityOperation.INVENTORY,
            "A-Mem Chroma collection.count() did not return a non-negative integer",
        )
    if len(ids) != count:
        raise _AMemCapabilityFailure(
            CapabilityOperation.INVENTORY,
            "A-Mem Chroma collection.get() inventory is incomplete: "
            f"returned {len(ids)} IDs but collection.count() reports {count}",
        )
    return ids


def amem_scope_from_state(state: StateHandle) -> AMemScope:
    """Validate the StateHandle-owned, fixed-collection adapter scope."""

    if state.backend != "a-mem":
        raise ValueError("A-Mem capability requires an A-Mem state")
    backend_state = state.backend_state
    memories = getattr(backend_state, "memories", None)
    retriever = getattr(backend_state, "retriever", None)
    collection = getattr(retriever, "collection", None)
    if not isinstance(memories, Mapping):
        raise ValueError("A-Mem state has no public memories mapping")
    if (
        collection is None
        or not callable(getattr(collection, "get", None))
        or not callable(getattr(collection, "count", None))
    ):
        raise ValueError("A-Mem state has no enumerable Chroma collection")
    collection_name = getattr(collection, "name", None)
    if collection_name != AMEM_COLLECTION_NAME:
        raise ValueError("A-Mem collection does not match the adapter's fixed scope")
    return AMemScope(
        state_id=state.state_id,
        collection_name=collection_name,
        backend_object_identity=id(backend_state),
        retriever_object_identity=id(retriever),
    )


class AMemRetrievableEntryCapability:
    """Concrete A-Mem public-memory-note capability for one StateHandle."""

    backend = "a-mem"
    selected_public_retrieval_api = AMEM_PUBLIC_RETRIEVAL_API
    retrieval_object_class = AMEM_RETRIEVAL_OBJECT_CLASS

    def __init__(self, state: StateHandle) -> None:
        self.state = state
        self.scope = amem_scope_from_state(state)

    def _blocker(
        self,
        root_checkpoint_id: str,
        operation: CapabilityOperation,
        reason: str,
    ) -> CapabilityResult[Any]:
        return CapabilityResult.failure(
            RetrievalCapabilityBlocker(
                backend="a-mem",
                root_checkpoint_id=root_checkpoint_id,
                operation=operation,
                reason=reason,
            )
        )

    def _validate_request_scope(self, root_checkpoint_id: str, state_id: str) -> None:
        if root_checkpoint_id != self.state.checkpoint_id:
            raise ValueError("A-Mem capability request uses another root checkpoint")
        if state_id != self.state.state_id:
            raise ValueError("A-Mem capability request uses another backend state")
        current = amem_scope_from_state(self.state)
        if current != self.scope:
            raise ValueError("A-Mem StateHandle no longer owns its original retriever scope")

    def _snapshot(self) -> dict[str, tuple[RetrievableEntryProjection, tuple[str, ...]]]:
        memories = self.state.backend_state.memories
        snapshot: dict[str, tuple[RetrievableEntryProjection, tuple[str, ...]]] = {}
        for mapping_id, note in memories.items():
            memory_id = _nonempty_memory_id(mapping_id, where="self.memories key")
            if memory_id in snapshot:
                raise _AMemCapabilityFailure(
                    CapabilityOperation.PHYSICAL_IDENTITY,
                    "A-Mem self.memories contains duplicate physical IDs",
                )
            if _note_id(note) != memory_id:
                raise _AMemCapabilityFailure(
                    CapabilityOperation.INVENTORY,
                    "A-Mem self.memories key and MemoryNote.id disagree",
                )
            snapshot[memory_id] = (
                amem_observable_projection(note),
                _provenance_ids(note),
            )
        retriever_ids = frozenset(_chroma_ids(self.state.backend_state.retriever.collection))
        memory_ids = frozenset(snapshot)
        if retriever_ids != memory_ids:
            chroma_only = sorted(retriever_ids - memory_ids)
            memory_only = sorted(memory_ids - retriever_ids)
            raise _AMemCapabilityFailure(
                CapabilityOperation.INVENTORY,
                "A-Mem self.memories/Chroma identity mismatch "
                f"(Chroma-only={chroma_only!r}, self.memories-only={memory_only!r})",
            )
        return snapshot

    def _stable_snapshot(
        self,
    ) -> dict[str, tuple[RetrievableEntryProjection, tuple[str, ...]]]:
        """Repeat exact enumeration; callers must not mutate during this check."""

        first = self._snapshot()
        second = self._snapshot()
        if first.keys() != second.keys():
            raise _AMemCapabilityFailure(
                CapabilityOperation.INVENTORY,
                "stable A-Mem initialization snapshot could not be established: ID set changed",
            )
        if first != second:
            raise _AMemCapabilityFailure(
                CapabilityOperation.INVENTORY,
                "stable A-Mem initialization snapshot could not be established: "
                "public projection or provenance changed",
            )
        return first

    async def inventory(
        self,
        *,
        campaign_id: str,
        root_checkpoint_id: str,
        state_id: str,
    ) -> CapabilityResult[RetrievableEntryInventory]:
        try:
            self._validate_request_scope(root_checkpoint_id, state_id)
            snapshot = self._stable_snapshot()
            entries = tuple(
                RetrievableInventoryEntry(
                    physical_entry=PhysicalEntryRef(
                        campaign_id,
                        root_checkpoint_id,
                        state_id,
                        memory_id,
                    ),
                    projection=projection,
                    provenance_ids=provenance_ids,
                )
                for memory_id, (projection, provenance_ids) in snapshot.items()
            )
            return CapabilityResult.success(
                RetrievableEntryInventory(
                    backend="a-mem",
                    selected_public_retrieval_api=AMEM_PUBLIC_RETRIEVAL_API,
                    retrieval_object_class=AMEM_RETRIEVAL_OBJECT_CLASS,
                    retrieval_returnability_basis=(
                        f"A-Mem {AMEM_COMMIT}: public search candidates are Chroma IDs "
                        "resolved through self.memories; unbounded collection.get() was "
                        "checked against collection.count(), exact two-sided ID agreement "
                        "and a repeated ID-to-public-projection snapshot were verified"
                    ),
                    campaign_id=campaign_id,
                    root_checkpoint_id=root_checkpoint_id,
                    state_id=state_id,
                    entries=entries,
                )
            )
        except _AMemCapabilityFailure as error:
            return self._blocker(root_checkpoint_id, error.operation, str(error))
        except Exception as error:
            return self._blocker(
                root_checkpoint_id,
                CapabilityOperation.INVENTORY,
                f"scoped A-Mem inventory could not be established: {error}",
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
            values = self.state.backend_state.search(query, k=top_k)
            return CapabilityResult.success(
                self.ranked_retrieval_from_search_result(
                    values,
                    campaign_id=campaign_id,
                    root_checkpoint_id=root_checkpoint_id,
                    state_id=state_id,
                )
            )
        except _AMemCapabilityFailure as error:
            return self._blocker(root_checkpoint_id, error.operation, str(error))
        except Exception as error:
            return self._blocker(
                root_checkpoint_id,
                CapabilityOperation.RANKED_RETRIEVAL,
                f"scoped A-Mem search could not be established: {error}",
            )

    def ranked_retrieval_from_search_result(
        self,
        values: Any,
        *,
        campaign_id: str,
        root_checkpoint_id: str,
        state_id: str,
    ) -> RankedPhysicalRetrieval:
        """Preserve the exact public search order and physical note IDs."""

        self._validate_request_scope(root_checkpoint_id, state_id)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise _AMemCapabilityFailure(
                CapabilityOperation.RANKED_RETRIEVAL,
                "AgenticMemorySystem.search() did not return a sequence",
            )
        physical: list[PhysicalEntryRef] = []
        for value in values:
            if not isinstance(value, Mapping):
                raise _AMemCapabilityFailure(
                    CapabilityOperation.RANKED_RETRIEVAL,
                    "AgenticMemorySystem.search() returned a non-mapping result",
                )
            amem_observable_projection(value)
            memory_id = _nonempty_memory_id(
                value.get("id"), where="AgenticMemorySystem.search() result"
            )
            physical.append(
                PhysicalEntryRef(
                    campaign_id,
                    root_checkpoint_id,
                    state_id,
                    memory_id,
                )
            )
        return RankedPhysicalRetrieval(
            backend="a-mem",
            selected_public_retrieval_api=AMEM_PUBLIC_RETRIEVAL_API,
            retrieval_object_class=AMEM_RETRIEVAL_OBJECT_CLASS,
            campaign_id=campaign_id,
            root_checkpoint_id=root_checkpoint_id,
            state_id=state_id,
            ranked_physical_entries=tuple(physical),
        )

    def certify_same_id_update(
        self,
        lineage: InitialEntryLineage,
        before: PhysicalEntryRef,
        *,
        receipt: OperationReceipt,
    ) -> LineageBinding:
        """Certify pinned A-Mem's delete/re-add update under the same ID."""

        self._validate_physical_scope(before)
        if (
            receipt.backend != "a-mem"
            or receipt.operation != "update"
            or not receipt.success
            or receipt.state_id != before.state_id
            or receipt.target_local_id != before.backend_entry_id
            or receipt.affected_local_ids != (before.backend_entry_id,)
            or not isinstance(receipt.after, Mapping)
            or receipt.after.get("id") != before.backend_entry_id
        ):
            raise ValueError("A-Mem update receipt does not certify a same-ID transition")
        snapshot = self._stable_snapshot()
        if before.backend_entry_id not in snapshot:
            raise ValueError("updated A-Mem ID is absent from consistent public state")
        return lineage.certify_same_id_update(before)

    def mark_deleted(
        self,
        lineage: InitialEntryLineage,
        physical_entry: PhysicalEntryRef,
        *,
        receipt: OperationReceipt,
    ) -> None:
        """Verify both stores removed one note, then retain its frozen E_0 ID."""

        self._validate_physical_scope(physical_entry)
        if (
            receipt.backend != "a-mem"
            or receipt.operation != "delete"
            or not receipt.success
            or receipt.state_id != physical_entry.state_id
            or receipt.target_local_id != physical_entry.backend_entry_id
            or receipt.affected_local_ids != (physical_entry.backend_entry_id,)
            or receipt.after is not None
        ):
            raise ValueError("A-Mem deletion receipt does not certify note removal")
        snapshot = self._stable_snapshot()
        if physical_entry.backend_entry_id in snapshot:
            raise ValueError("deleted A-Mem ID remains in consistent public state")
        lineage.mark_deleted(physical_entry)

    def _validate_physical_scope(self, physical_entry: PhysicalEntryRef) -> None:
        if physical_entry.root_checkpoint_id != self.state.checkpoint_id:
            raise ValueError("A-Mem physical entry uses another root checkpoint")
        if physical_entry.state_id != self.state.state_id:
            raise ValueError("A-Mem physical entry uses another backend state")
