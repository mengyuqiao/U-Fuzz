"""Adapter for A-Mem at the frozen upstream commit."""

from __future__ import annotations

from dataclasses import replace
from importlib import metadata, util
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

from ufuzz.backends.base import (
    BackendAdapter,
    BackendCapabilities,
    Capability,
    CapabilityStatus,
    InitializationArtifact,
    OperationReceipt,
    StateHandle,
    public_dataclass,
)
from ufuzz.domain import SourceUnit
from ufuzz.mapping import RetrievalRecord


AMEM_COMMIT = "ceffb860f0712bbae97b184d440df62bc910ca8d"
AMEM_VERSION = "0.0.1"


class AMemAdapter(BackendAdapter):
    _process_owner_state_id: str | None = None

    def __init__(
        self,
        *,
        model_name: str = "all-MiniLM-L6-v2",
        llm_backend: str = "openai",
        llm_model: str = "gpt-4o-mini",
        api_key: str | None = None,
        memory_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self.llm_backend = llm_backend
        self.llm_model = llm_model
        self.api_key = api_key
        self.memory_factory = memory_factory
        self._states: dict[str, StateHandle] = {}
        self._provenance: dict[str, dict[str, set[str]]] = {}

    def capabilities(self) -> BackendCapabilities:
        available = util.find_spec("agentic_memory") is not None
        try:
            version = metadata.version("agentic-memory")
        except metadata.PackageNotFoundError:
            version = "not-installed"
        return BackendCapabilities(
            backend="a-mem",
            package="agentic-memory",
            version_or_commit=(
                f"required={AMEM_VERSION}; installed={version}; commit={AMEM_COMMIT}"
            ),
            available=available,
            dependencies=(
                f"agentic-memory @ git+https://github.com/agiresearch/A-mem.git@{AMEM_COMMIT}",
                "sentence-transformers==5.6.0",
                "chromadb==1.5.9",
                "rank-bm25==0.2.2",
                "nltk==3.10.3",
                "litellm==1.101.0",
                "numpy==2.5.1",
                "scikit-learn==1.9.0",
                "openai==2.45.0",
            ),
            services=(),
            operations={
                "ranked_retrieval": Capability(CapabilityStatus.SUPPORTED, "public search API"),
                "provenance": Capability(
                    CapabilityStatus.CONDITIONAL,
                    "returned note IDs plus adapter sidecar; A-Mem has no native provenance field",
                ),
                "update": Capability(CapabilityStatus.SUPPORTED, "public update API"),
                "delete": Capability(CapabilityStatus.SUPPORTED, "public delete API"),
                "unrelated_existing_region_change": Capability(
                    CapabilityStatus.SUPPORTED, "public update API on an existing note"
                ),
                "exact_clone": Capability(CapabilityStatus.UNSUPPORTED, "no public clone API"),
                "replay": Capability(
                    CapabilityStatus.CONDITIONAL,
                    "fresh construction replays input but invokes LLM and embedding pipelines",
                ),
                "observable_state": Capability(
                    CapabilityStatus.CONDITIONAL,
                    "public memories mapping and search results are observable",
                ),
                "initialization_equivalence": Capability(
                    CapabilityStatus.UNRESOLVED,
                    "constructor resets a fixed Chroma collection and replay may be nondeterministic",
                ),
            },
            controllable_initialization=(
                "embedding model name",
                "LLM backend/model/API configuration",
                "ingestion order",
            ),
            notes=(
                "AgenticMemorySystem uses an ephemeral chromadb.Client, hard-codes "
                "the collection name 'memories', and resets the shared in-process client",
                "a dedicated process provides storage isolation; concurrent states "
                "within one process are unsafe",
            ),
        )

    def _factory(self) -> Any:
        if self.memory_factory is not None:
            return self.memory_factory()
        if util.find_spec("agentic_memory") is None:
            raise RuntimeError(f"A-Mem commit {AMEM_COMMIT} is not installed")
        from agentic_memory.memory_system import AgenticMemorySystem

        return AgenticMemorySystem(
            model_name=self.model_name,
            llm_backend=self.llm_backend,
            llm_model=self.llm_model,
            api_key=self.api_key,
        )

    async def create_isolated_state(
        self, artifact: InitializationArtifact
    ) -> StateHandle:
        if self._states or type(self)._process_owner_state_id is not None:
            raise RuntimeError(
                "A-Mem uses a fixed reset-on-construction Chroma collection; "
                "only one in-process state is safe"
            )
        state_id = uuid4().hex
        state = StateHandle(
            backend="a-mem",
            state_id=state_id,
            checkpoint_id=artifact.checkpoint_id,
            initialization_digest=artifact.digest,
            backend_state=self._factory(),
            metadata={"artifact": artifact},
        )
        self._states[state_id] = state
        self._provenance[state_id] = {}
        type(self)._process_owner_state_id = state_id
        return state

    async def ingest(
        self, state: StateHandle, sources: Sequence[SourceUnit]
    ) -> tuple[OperationReceipt, ...]:
        receipts: list[OperationReceipt] = []
        for source in sources:
            before = await self.observable_state(state)
            local_id = str(
                state.backend_state.add_note(
                    source.text,
                    time=source.timestamp,
                    tags=[f"ufuzz-source:{source.provenance_id}"],
                )
            )
            self._provenance[state.state_id][local_id] = {source.provenance_id}
            receipts.append(
                OperationReceipt(
                    backend="a-mem",
                    state_id=state.state_id,
                    operation="ingest",
                    success=True,
                    target_local_id=None,
                    affected_local_ids=(local_id,),
                    provenance_ids=(source.provenance_id,),
                    before=before,
                    after=await self.observable_state(state),
                    raw={"memory_id": local_id},
                )
            )
        return tuple(receipts)

    async def retrieve(
        self, state: StateHandle, query: str, top_k: int
    ) -> tuple[RetrievalRecord, ...]:
        values = state.backend_state.search(query, k=top_k)
        records: list[RetrievalRecord] = []
        for rank, value in enumerate(values, start=1):
            local_id = str(value.get("id") or f"rank-{rank}")
            score = value.get("score")
            records.append(
                RetrievalRecord(
                    backend="a-mem",
                    local_id=local_id,
                    text=str(value.get("content") or ""),
                    rank=rank,
                    score=float(score) if isinstance(score, (int, float)) else None,
                    provenance_ids=tuple(
                        sorted(self._provenance[state.state_id].get(local_id, set()))
                    ),
                    metadata={
                        key: item
                        for key, item in value.items()
                        if key not in {"id", "content", "score"}
                    },
                    raw=value,
                )
            )
        return tuple(records)

    async def update(
        self,
        state: StateHandle,
        target_local_id: str,
        new_text: str,
        *,
        provenance_ids: Sequence[str],
        context: Mapping[str, Any] | None = None,
    ) -> OperationReceipt:
        before = public_dataclass(state.backend_state.read(target_local_id))
        success = bool(state.backend_state.update(target_local_id, content=new_text))
        self._provenance[state.state_id].setdefault(target_local_id, set()).update(
            provenance_ids
        )
        return OperationReceipt(
            backend="a-mem",
            state_id=state.state_id,
            operation="update",
            success=success,
            target_local_id=target_local_id,
            affected_local_ids=(target_local_id,),
            provenance_ids=tuple(provenance_ids),
            before=before,
            after=public_dataclass(state.backend_state.read(target_local_id)),
            raw={"success": success},
        )

    async def delete(
        self,
        state: StateHandle,
        target_local_id: str,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> OperationReceipt:
        before = public_dataclass(state.backend_state.read(target_local_id))
        success = bool(state.backend_state.delete(target_local_id))
        after = public_dataclass(state.backend_state.read(target_local_id))
        provenance = tuple(sorted(self._provenance[state.state_id].pop(target_local_id, set())))
        return OperationReceipt(
            backend="a-mem",
            state_id=state.state_id,
            operation="delete",
            success=success and after is None,
            target_local_id=target_local_id,
            affected_local_ids=(target_local_id,),
            provenance_ids=provenance,
            before=before,
            after=after,
            raw={"success": success},
        )

    async def unrelated_existing_region_change(
        self,
        state: StateHandle,
        target_local_id: str,
        new_text: str,
        *,
        provenance_ids: Sequence[str],
        context: Mapping[str, Any] | None = None,
    ) -> OperationReceipt:
        receipt = await self.update(
            state,
            target_local_id,
            new_text,
            provenance_ids=provenance_ids,
            context=context,
        )
        return replace(receipt, operation="unrelated_existing_region_change")

    async def observable_state(self, state: StateHandle) -> Any:
        values = []
        for local_id, note in state.backend_state.memories.items():
            item = public_dataclass(note)
            if isinstance(item, dict):
                item = {key: value for key, value in item.items() if key != "id"}
            values.append(item)
        return sorted(values, key=repr)

    async def clone_state(self, state: StateHandle) -> StateHandle:
        raise NotImplementedError("A-Mem has no public exact-clone API")

    async def replay_state(self, artifact: InitializationArtifact) -> StateHandle:
        state = await self.create_isolated_state(artifact)
        await self.ingest(state, artifact.sources)
        return state

    async def reset(self, state: StateHandle) -> None:
        for local_id in tuple(state.backend_state.memories):
            state.backend_state.delete(local_id)
        self._provenance[state.state_id].clear()

    async def teardown(self, state: StateHandle) -> None:
        try:
            await self.reset(state)
        finally:
            self._states.pop(state.state_id, None)
            self._provenance.pop(state.state_id, None)
            if type(self)._process_owner_state_id == state.state_id:
                type(self)._process_owner_state_id = None
