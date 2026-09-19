"""Adapter for mem0ai 2.0.12 public APIs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from importlib import metadata, util
from pathlib import Path
import shutil
import tempfile
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
    source_metadata,
)
from ufuzz.domain import SourceUnit
from ufuzz.mapping import RetrievalRecord


class Mem0Adapter(BackendAdapter):
    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        root_dir: str | Path | None = None,
        memory_factory: Callable[[Mapping[str, Any]], Any] | None = None,
        infer: bool = True,
        allow_external_store: bool = False,
    ) -> None:
        self.config = deepcopy(dict(config or {}))
        self.root_dir = Path(root_dir or tempfile.mkdtemp(prefix="ufuzz-mem0-"))
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.memory_factory = memory_factory
        self.infer = infer
        self.allow_external_store = allow_external_store
        self._states: dict[str, StateHandle] = {}
        self._provenance: dict[str, dict[str, set[str]]] = {}

    def capabilities(self) -> BackendCapabilities:
        available = util.find_spec("mem0") is not None
        version = metadata.version("mem0ai") if available else "not-installed"
        return BackendCapabilities(
            backend="mem0",
            package="mem0ai",
            version_or_commit=version,
            available=available,
            dependencies=("mem0ai==2.0.12",),
            services=(),
            operations={
                "ranked_retrieval": Capability(CapabilityStatus.SUPPORTED, "public search API"),
                "provenance": Capability(CapabilityStatus.CONDITIONAL, "metadata plus adapter sidecar"),
                "update": Capability(CapabilityStatus.SUPPORTED, "public update API"),
                "delete": Capability(CapabilityStatus.SUPPORTED, "public delete API"),
                "unrelated_existing_region_change": Capability(
                    CapabilityStatus.SUPPORTED, "public update API on an existing entry"
                ),
                "exact_clone": Capability(CapabilityStatus.UNSUPPORTED, "no public clone API"),
                "replay": Capability(CapabilityStatus.SUPPORTED, "fresh isolated store and replay"),
                "observable_state": Capability(CapabilityStatus.SUPPORTED, "public get_all/get APIs"),
                "initialization_equivalence": Capability(
                    CapabilityStatus.UNRESOLVED,
                    "canonical projection is observable; final fingerprint is intentionally deferred",
                ),
            },
            controllable_initialization=(
                "LLM provider/model/config",
                "embedder provider/model/config",
                "vector-store path and collection",
                "history database path",
                "ingestion order",
                "infer flag",
            ),
            notes=(
                "OSS timestamp and reference_date options are not usable in mem0ai 2.0.12",
            ),
        )

    def _factory(self, config: Mapping[str, Any]) -> Any:
        if self.memory_factory is not None:
            return self.memory_factory(config)
        if util.find_spec("mem0") is None:
            raise RuntimeError("mem0ai==2.0.12 is not installed")
        from mem0 import Memory

        return Memory.from_config(dict(config))

    def _state_config(self, state_id: str) -> dict[str, Any]:
        config = deepcopy(self.config)
        state_dir = self.root_dir / state_id
        state_dir.mkdir(parents=True, exist_ok=False)
        vector_store = config.setdefault("vector_store", {})
        vector_store.setdefault("provider", "qdrant")
        vector_config = vector_store.setdefault("config", {})
        if not self.allow_external_store:
            if vector_store["provider"] != "qdrant":
                raise ValueError(
                    "isolated Mem0 state requires local Qdrant storage; "
                    "external providers require allow_external_store=True"
                )
            external_keys = {"host", "url", "api_key", "client"} & vector_config.keys()
            if external_keys:
                raise ValueError(
                    "isolated Mem0 state rejects external Qdrant settings: "
                    + ", ".join(sorted(external_keys))
                )
        vector_config["collection_name"] = f"ufuzz_{state_id.replace('-', '_')}"
        if vector_store["provider"] == "qdrant":
            vector_config["path"] = str(state_dir / "qdrant")
        config["history_db_path"] = str(state_dir / "history.db")
        return config

    async def create_isolated_state(
        self, artifact: InitializationArtifact
    ) -> StateHandle:
        state_id = uuid4().hex
        config = self._state_config(state_id)
        memory = self._factory(config)
        state = StateHandle(
            backend="mem0",
            state_id=state_id,
            checkpoint_id=artifact.checkpoint_id,
            initialization_digest=artifact.digest,
            backend_state=memory,
            metadata={"config": config, "artifact": artifact},
        )
        self._states[state_id] = state
        self._provenance[state_id] = {}
        return state

    async def ingest(
        self, state: StateHandle, sources: Sequence[SourceUnit]
    ) -> tuple[OperationReceipt, ...]:
        receipts: list[OperationReceipt] = []
        memory = state.backend_state
        for source in sources:
            before = await self.observable_state(state)
            response = memory.add(
                [{"role": source.role or "user", "content": source.text}],
                user_id=state.state_id,
                metadata=source_metadata(source),
                infer=self.infer,
            )
            ids = self._result_ids(response)
            for local_id in ids:
                self._provenance[state.state_id].setdefault(local_id, set()).add(
                    source.provenance_id
                )
            receipts.append(
                OperationReceipt(
                    backend="mem0",
                    state_id=state.state_id,
                    operation="ingest",
                    success=True,
                    target_local_id=None,
                    affected_local_ids=tuple(ids),
                    provenance_ids=(source.provenance_id,),
                    before=before,
                    after=await self.observable_state(state),
                    raw=response,
                )
            )
        return tuple(receipts)

    @staticmethod
    def _result_ids(response: Any) -> list[str]:
        if isinstance(response, dict):
            values = response.get("results", response.get("memories", []))
        else:
            values = response if isinstance(response, list) else []
        ids: list[str] = []
        for value in values or []:
            if isinstance(value, dict):
                local_id = value.get("id") or value.get("memory_id")
                if local_id is not None:
                    ids.append(str(local_id))
        return ids

    async def retrieve(
        self, state: StateHandle, query: str, top_k: int
    ) -> tuple[RetrievalRecord, ...]:
        response = state.backend_state.search(
            query,
            top_k=top_k,
            filters={"user_id": state.state_id},
        )
        values = response.get("results", []) if isinstance(response, dict) else response
        records: list[RetrievalRecord] = []
        for rank, value in enumerate(values or [], start=1):
            if not isinstance(value, dict):
                continue
            local_id = str(value.get("id") or value.get("memory_id") or f"rank-{rank}")
            metadata_value = value.get("metadata")
            entry_metadata = dict(metadata_value) if isinstance(metadata_value, dict) else {}
            provenance = set(self._provenance[state.state_id].get(local_id, set()))
            direct = entry_metadata.get("ufuzz_provenance_id")
            if direct:
                provenance.add(str(direct))
            score = value.get("score")
            records.append(
                RetrievalRecord(
                    backend="mem0",
                    local_id=local_id,
                    text=str(value.get("memory") or value.get("text") or ""),
                    rank=rank,
                    score=float(score) if isinstance(score, (int, float)) else None,
                    provenance_ids=tuple(sorted(provenance)),
                    metadata=entry_metadata,
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
        before = self._safe_get(state.backend_state, target_local_id)
        metadata_value = dict(context or {})
        if provenance_ids:
            metadata_value["ufuzz_provenance_ids"] = list(provenance_ids)
        raw = state.backend_state.update(
            target_local_id,
            text=new_text,
            metadata=metadata_value or None,
        )
        self._provenance[state.state_id].setdefault(target_local_id, set()).update(
            provenance_ids
        )
        return OperationReceipt(
            backend="mem0",
            state_id=state.state_id,
            operation="update",
            success=True,
            target_local_id=target_local_id,
            affected_local_ids=(target_local_id,),
            provenance_ids=tuple(provenance_ids),
            before=before,
            after=self._safe_get(state.backend_state, target_local_id),
            raw=raw,
        )

    async def delete(
        self,
        state: StateHandle,
        target_local_id: str,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> OperationReceipt:
        before = self._safe_get(state.backend_state, target_local_id)
        raw = state.backend_state.delete(target_local_id)
        after = self._safe_get(state.backend_state, target_local_id)
        provenance = tuple(sorted(self._provenance[state.state_id].pop(target_local_id, set())))
        return OperationReceipt(
            backend="mem0",
            state_id=state.state_id,
            operation="delete",
            success=after is None,
            target_local_id=target_local_id,
            affected_local_ids=(target_local_id,),
            provenance_ids=provenance,
            before=before,
            after=after,
            raw=raw,
            reason=None if after is None else "entry remains observable after public delete",
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

    @staticmethod
    def _safe_get(memory: Any, local_id: str) -> Any:
        try:
            return memory.get(local_id)
        except Exception:
            return None

    async def observable_state(self, state: StateHandle) -> Any:
        raw = state.backend_state.get_all(filters={"user_id": state.state_id}, top_k=10_000)
        values = raw.get("results", raw) if isinstance(raw, dict) else raw
        projection = []
        for value in values or []:
            if isinstance(value, dict):
                projection.append(
                    {
                        "memory": value.get("memory") or value.get("text"),
                        "metadata": value.get("metadata", {}),
                    }
                )
            else:
                projection.append(public_dataclass(value))
        return sorted(projection, key=lambda item: repr(item))

    async def clone_state(self, state: StateHandle) -> StateHandle:
        raise NotImplementedError("Mem0 has no public exact-clone API")

    async def replay_state(self, artifact: InitializationArtifact) -> StateHandle:
        state = await self.create_isolated_state(artifact)
        await self.ingest(state, artifact.sources)
        return state

    async def reset(self, state: StateHandle) -> None:
        state.backend_state.reset()
        self._provenance[state.state_id].clear()

    async def teardown(self, state: StateHandle) -> None:
        try:
            await self.reset(state)
        finally:
            self._states.pop(state.state_id, None)
            self._provenance.pop(state.state_id, None)
            shutil.rmtree(self.root_dir / state.state_id, ignore_errors=True)
