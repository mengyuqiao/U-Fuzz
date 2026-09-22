"""MemOS v2.0.33 GeneralTextMemory adapter for RQ1--RQ3.

The selected profile directly constructs textual items.  MemOS's extractor
configuration is constructor compatibility state only and is guarded by a
noncalling implementation.  No Tree, MemReader, feedback, scheduler, or chat
path is part of this adapter.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
from importlib import metadata as importlib_metadata
from importlib import util as importlib_util
import json
from pathlib import Path
import shutil
from threading import RLock
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import unquote, urlparse
from uuid import uuid4

from ufuzz.backends.base import (
    BackendAdapter,
    BackendCapabilities,
    Capability,
    CapabilityStatus,
    InitializationArtifact,
    OperationReceipt,
    StateHandle,
)
from ufuzz.domain import SourceUnit
from ufuzz.mapping import RetrievalRecord


MEMOS_VERSION = "2.0.33"
MEMOS_SOURCE_COMMIT = "78a372a4fc853a24d2a78efa3b4bbbd27ab9f7ad"
MEMOS_BACKEND = "general_text"
MEMOS_EMBEDDING_TRANSPORT_MODEL = "nomic-embed-text:latest"
MEMOS_EMBEDDING_MANIFEST_DIGEST = (
    "0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f"
)
MEMOS_EMBEDDING_DIMENSION = 768
MEMOS_DISTANCE = "cosine"
MEMOS_PROFILE_ID = (
    "memos-general-text:v2.0.33:78a372a4:embedded-qdrant-1.16.2:"
    "nomic-embed-text-0a109f42:dim-768:cosine:direct-items:v1"
)
PROVENANCE_KEY = "ufuzz_provenance_v1"
_FACTORY_LOCK = RLock()


class MemosProfileError(RuntimeError):
    """Fail-closed profile/configuration violation."""


class _NonCallingExtractor:
    def __init__(self, config: Any = None) -> None:
        self.config = config

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        raise MemosProfileError("native MemOS chat/extraction is disabled")

    def generate_response(self, *args: Any, **kwargs: Any) -> Any:
        raise MemosProfileError("native MemOS chat/extraction is disabled")


@contextmanager
def _noncalling_extractor_factory():
    """Bind only the constructor-required extractor to an inert object."""

    from memos.llms.factory import LLMFactory

    with _FACTORY_LOCK:
        original = LLMFactory.__dict__["from_config"]
        LLMFactory.from_config = staticmethod(lambda _config: _NonCallingExtractor(_config))
        try:
            yield
        finally:
            setattr(LLMFactory, "from_config", original)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")


def _provenance(source: SourceUnit) -> dict[str, Any]:
    return {
        "schema": PROVENANCE_KEY,
        "benchmark": source.benchmark,
        "checkpoint_id": source.checkpoint_id,
        "session_id": source.session_id,
        "source_id": source.source_id,
        "ordinal": source.ordinal,
        "timestamp": source.timestamp,
        "speaker": source.speaker,
        "role": source.role,
        "provenance_id": source.provenance_id,
        "root_text_sha256": sha256(source.text.encode("utf-8")).hexdigest(),
    }


def _metadata_dict(item: Any) -> dict[str, Any]:
    metadata = item.metadata
    if hasattr(metadata, "model_dump"):
        return metadata.model_dump(mode="json", exclude_none=True)
    if isinstance(metadata, Mapping):
        return deepcopy(dict(metadata))
    raise MemosProfileError("TextualMemoryItem metadata is not public/mappable")


def _item_id(item: Any) -> str:
    value = getattr(item, "id", None)
    if not isinstance(value, str) or not value:
        raise MemosProfileError("TextualMemoryItem has no stable physical ID")
    return value


def _item_provenance(item: Any) -> tuple[str, ...]:
    value = _metadata_dict(item).get("info", {}).get(PROVENANCE_KEY)
    if not isinstance(value, Mapping):
        raise MemosProfileError("TextualMemoryItem is missing frozen provenance")
    provenance_id = value.get("provenance_id")
    if not isinstance(provenance_id, str) or not provenance_id:
        raise MemosProfileError("TextualMemoryItem provenance ID is missing")
    return (provenance_id,)


class MemosAdapter(BackendAdapter):
    """Exact selected GeneralTextMemory adapter with per-state local Qdrant."""

    def __init__(
        self,
        *,
        root_dir: str | Path | None = None,
        memory_factory: Callable[[Mapping[str, Any]], Any] | None = None,
        item_factory: Callable[[str, Mapping[str, Any]], Any] | None = None,
        embedding_identity_resolver: Callable[[], str] | None = None,
        runtime_identity_verifier: Callable[[], None] | None = None,
    ) -> None:
        self.root_dir = Path(root_dir or "/tmp/ufuzz-memos-general-text").resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._memory_factory = memory_factory or self._create_memory
        self._item_factory = item_factory or self._create_item
        self._embedding_identity_resolver = (
            embedding_identity_resolver or self._resolve_embedding_digest
        )
        self._runtime_identity_verifier = runtime_identity_verifier or self._verify_runtime_identity
        self._live_state_ids: set[str] = set()

    def capabilities(self) -> BackendCapabilities:
        supported = lambda reason, observable=None: Capability(
            CapabilityStatus.SUPPORTED, reason, observable
        )
        return BackendCapabilities(
            backend="memos",
            package="MemoryOS",
            version_or_commit=f"v{MEMOS_VERSION}@{MEMOS_SOURCE_COMMIT}",
            available=importlib_util.find_spec("memos") is not None,
            dependencies=("MemoryOS==2.0.33", "qdrant-client==1.16.2"),
            services=("local nomic embedding service (immutable digest guarded)",),
            operations={
                "complete_inventory": supported("GeneralTextMemory.get_all", "TextualMemoryItem"),
                "ranked_retrieval": supported("GeneralTextMemory.search preserves order"),
                "update": supported("same-ID GeneralTextMemory.update"),
                "delete": supported("exact-ID GeneralTextMemory.delete"),
                "unrelated_change": supported("same-ID update of certified existing target"),
                "initialization_equivalence": supported("exact provenance/projection rebinding"),
                "provenance": supported("ufuzz_provenance_v1 in metadata.info"),
                "merge_split": Capability(CapabilityStatus.UNSUPPORTED, "inapplicable to this profile"),
                "native_chat": Capability(CapabilityStatus.UNSUPPORTED, "fail-closed and inert"),
            },
            controllable_initialization=(
                "direct TextualMemoryItem construction",
                "dedicated process",
                "adapter-owned Qdrant path and collection",
            ),
            notes=("RQ1--RQ3 profile; Tree and native extraction are excluded",),
        )

    @staticmethod
    def _verify_runtime_identity() -> None:
        if importlib_metadata.version("MemoryOS") != MEMOS_VERSION:
            raise MemosProfileError("installed MemoryOS version differs from 2.0.33")
        if importlib_metadata.version("qdrant-client") != "1.16.2":
            raise MemosProfileError("installed qdrant-client version differs from 1.16.2")
        distribution = importlib_metadata.distribution("MemoryOS")
        direct = distribution.read_text("direct_url.json")
        if not direct:
            raise MemosProfileError("MemoryOS installation lacks an exact source provenance record")
        record = json.loads(direct)
        commit = record.get("vcs_info", {}).get("commit_id")
        if commit is None and record.get("dir_info", {}).get("editable") is True:
            parsed = urlparse(record.get("url", ""))
            source = Path(unquote(parsed.path)).resolve() if parsed.scheme == "file" else None
            if source is not None:
                git = source / ".git"
                if git.is_file():
                    marker = git.read_text(encoding="utf-8").strip()
                    if marker.startswith("gitdir: "):
                        git = Path(marker[8:])
                        if not git.is_absolute():
                            git = (source / git).resolve()
                head = git / "HEAD"
                if head.is_file():
                    value = head.read_text(encoding="utf-8").strip()
                    if value.startswith("ref: "):
                        ref = git / value[5:]
                        if ref.is_file():
                            commit = ref.read_text(encoding="utf-8").strip()
                    else:
                        commit = value
        if commit != MEMOS_SOURCE_COMMIT:
            raise MemosProfileError("installed MemoryOS source commit is not the frozen commit")

    @staticmethod
    def _resolve_embedding_digest() -> str:
        """Resolve Ollama's immutable manifest digest, never the moving alias."""

        import ollama

        response = ollama.list()
        models = getattr(response, "models", response.get("models", ()) if isinstance(response, Mapping) else ())
        candidates = []
        for model in models:
            name = getattr(model, "model", None) or getattr(model, "name", None)
            digest = getattr(model, "digest", None)
            if isinstance(model, Mapping):
                name = name or model.get("model") or model.get("name")
                digest = digest or model.get("digest")
            if isinstance(name, str) and name == MEMOS_EMBEDDING_TRANSPORT_MODEL:
                candidates.append(digest)
        if candidates != [MEMOS_EMBEDDING_MANIFEST_DIGEST]:
            raise MemosProfileError("exact nomic embedding manifest digest is unavailable or ambiguous")
        return candidates[0]

    def _verify_embedding_identity(self) -> None:
        actual = self._embedding_identity_resolver()
        if actual != MEMOS_EMBEDDING_MANIFEST_DIGEST:
            raise MemosProfileError(
                f"embedding digest mismatch: expected {MEMOS_EMBEDDING_MANIFEST_DIGEST}, got {actual}"
            )

    @staticmethod
    def _create_item(memory: str, metadata: Mapping[str, Any]) -> Any:
        from memos.memories.textual.item import TextualMemoryItem, TextualMemoryMetadata

        return TextualMemoryItem(memory=memory, metadata=TextualMemoryMetadata(**dict(metadata)))

    @staticmethod
    def _create_memory(config: Mapping[str, Any]) -> Any:
        from memos.configs.memory import MemoryConfigFactory
        from memos.memories.factory import MemoryFactory

        with _noncalling_extractor_factory():
            return MemoryFactory.from_config(MemoryConfigFactory(**dict(config)))

    def _state_config(self, state_id: str, path: Path, collection: str) -> dict[str, Any]:
        return {
            "backend": MEMOS_BACKEND,
            "config": {
                "extractor_llm": {"backend": "ollama", "config": {"model_name_or_path": "disabled"}},
                "embedder": {
                    "backend": "ollama",
                    "config": {
                        "model_name_or_path": MEMOS_EMBEDDING_TRANSPORT_MODEL,
                        "embedding_dims": MEMOS_EMBEDDING_DIMENSION,
                    },
                },
                "vector_db": {
                    "backend": "qdrant",
                    "config": {
                        "collection_name": collection,
                        "path": str(path),
                        "distance_metric": MEMOS_DISTANCE,
                        "vector_dimension": MEMOS_EMBEDDING_DIMENSION,
                    },
                },
            },
        }

    async def create_isolated_state(self, artifact: InitializationArtifact) -> StateHandle:
        if self._live_state_ids:
            raise MemosProfileError(
                "the production profile permits one live state per dedicated adapter process"
            )
        self._runtime_identity_verifier()
        self._verify_embedding_identity()
        state_id = str(uuid4())
        state_dir = (self.root_dir / state_id).resolve()
        if state_dir.parent != self.root_dir:
            raise MemosProfileError("adapter-owned state path escaped root")
        qdrant_path = state_dir / "qdrant"
        qdrant_path.mkdir(parents=True, exist_ok=False)
        collection = f"ufuzz_memos_{state_id.replace('-', '_')}"
        config = self._state_config(state_id, qdrant_path, collection)
        try:
            memory = self._memory_factory(config)
        except Exception:
            shutil.rmtree(state_dir, ignore_errors=True)
            raise
        handle = StateHandle(
            backend="memos",
            state_id=state_id,
            checkpoint_id=artifact.checkpoint_id,
            initialization_digest=artifact.digest,
            backend_state=memory,
            metadata={
                "profile_id": MEMOS_PROFILE_ID,
                "state_dir": str(state_dir),
                "qdrant_path": str(qdrant_path),
                "collection": collection,
                "config": config,
                "embedding_digest": MEMOS_EMBEDDING_MANIFEST_DIGEST,
                "adapter_owned": True,
                "dedicated_process_required": True,
            },
        )
        self._live_state_ids.add(state_id)
        return handle

    def _new_item(self, source: SourceUnit) -> Any:
        provenance = _provenance(source)
        metadata = {
            "status": "activated",
            "version": 1,
            "type": "benchmark_source",
            "key": source.provenance_id,
            "source": "conversation",
            "tags": [source.benchmark, source.checkpoint_id],
            "visibility": "private",
            "updated_at": source.timestamp or "1970-01-01T00:00:00+00:00",
            "info": {PROVENANCE_KEY: provenance},
        }
        return self._item_factory(source.text, metadata)

    async def ingest(self, state: StateHandle, sources: Sequence[SourceUnit]) -> tuple[OperationReceipt, ...]:
        if any(source.checkpoint_id != state.checkpoint_id for source in sources):
            raise MemosProfileError("source belongs to another checkpoint")
        provenance_ids = tuple(source.provenance_id for source in sources)
        if len(set(provenance_ids)) != len(provenance_ids):
            raise MemosProfileError("root sources repeat a stable provenance identity")
        items = tuple(self._new_item(source) for source in sources)
        state.backend_state.add(list(items))
        return tuple(
            OperationReceipt(
                "memos", state.state_id, "add", True, None, (_item_id(item),),
                (source.provenance_id,), None, item, {"api": "GeneralTextMemory.add"},
            )
            for source, item in zip(sources, items, strict=True)
        )

    async def retrieve(self, state: StateHandle, query: str, top_k: int) -> tuple[RetrievalRecord, ...]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        items = tuple(state.backend_state.search(query, top_k))
        if len(items) > top_k:
            raise MemosProfileError("GeneralTextMemory.search exceeded top_k")
        return tuple(
            RetrievalRecord(
                backend="memos", local_id=_item_id(item), text=item.memory,
                rank=rank, score=None, provenance_ids=_item_provenance(item),
                metadata=_metadata_dict(item), raw=item,
            )
            for rank, item in enumerate(items, 1)
        )

    def _get(self, state: StateHandle, target_local_id: str) -> Any:
        item = state.backend_state.get(target_local_id)
        if item is None:
            raise MemosProfileError("exact target is absent")
        return item

    async def _update(self, state: StateHandle, target_local_id: str, new_text: str,
                      provenance_ids: Sequence[str], operation: str) -> OperationReceipt:
        before = self._get(state, target_local_id)
        if tuple(provenance_ids) != _item_provenance(before):
            raise MemosProfileError("requested provenance does not match exact target")
        replacement = self._item_factory(new_text, _metadata_dict(before))
        state.backend_state.update(target_local_id, replacement)
        after = self._get(state, target_local_id)
        if _item_id(after) != target_local_id or _item_provenance(after) != tuple(provenance_ids):
            raise MemosProfileError("native update did not preserve ID/provenance")
        return OperationReceipt(
            "memos", state.state_id, operation, True, target_local_id,
            (target_local_id,), tuple(provenance_ids), before, after,
            {"api": "GeneralTextMemory.update", "same_id": True},
        )

    async def update(self, state: StateHandle, target_local_id: str, new_text: str, *,
                     provenance_ids: Sequence[str], context: Mapping[str, Any] | None = None) -> OperationReceipt:
        return await self._update(state, target_local_id, new_text, provenance_ids, "update")

    async def unrelated_existing_region_change(self, state: StateHandle, target_local_id: str,
                     new_text: str, *, provenance_ids: Sequence[str],
                     context: Mapping[str, Any] | None = None) -> OperationReceipt:
        return await self._update(state, target_local_id, new_text, provenance_ids, "unrelated_change")

    async def delete(self, state: StateHandle, target_local_id: str, *,
                     context: Mapping[str, Any] | None = None) -> OperationReceipt:
        before = self._get(state, target_local_id)
        state.backend_state.delete([target_local_id])
        try:
            remaining = state.backend_state.get(target_local_id)
        except (KeyError, ValueError):
            remaining = None
        if remaining is not None:
            raise MemosProfileError("deleted target remains visible")
        return OperationReceipt(
            "memos", state.state_id, "delete", True, target_local_id,
            (target_local_id,), _item_provenance(before), before, None,
            {"api": "GeneralTextMemory.delete"},
        )

    async def observable_state(self, state: StateHandle) -> Any:
        rows = [({"memory": item.memory, "metadata": _metadata_dict(item)},
                 _item_provenance(item)) for item in state.backend_state.get_all()]
        return tuple(sorted(rows, key=lambda row: _canonical_json(row)))

    async def clone_state(self, state: StateHandle) -> StateHandle:
        raise NotImplementedError("fresh artifact replay is authoritative")

    async def replay_state(self, artifact: InitializationArtifact) -> StateHandle:
        state = await self.create_isolated_state(artifact)
        try:
            await self.ingest(state, artifact.sources)
            return state
        except Exception:
            await self.teardown(state)
            raise

    async def reset(self, state: StateHandle) -> None:
        state.backend_state.delete_all()

    async def teardown(self, state: StateHandle) -> None:
        if state.metadata.get("cleanup_complete") is True:
            return
        if not state.metadata.get("adapter_owned"):
            raise MemosProfileError("refusing cleanup of non-owned state")
        state_dir = Path(state.metadata["state_dir"]).resolve()
        if state_dir.parent != self.root_dir or state_dir.name != state.state_id:
            raise MemosProfileError("refusing cleanup outside adapter-owned state directory")
        memory = state.backend_state
        try:
            memory.delete_all()
        finally:
            for candidate in (
                getattr(getattr(memory, "vector_db", None), "client", None),
                getattr(memory, "client", None),
                memory,
            ):
                close = getattr(candidate, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        # State is already unpublished. Continue retiring every
                        # owned resource and remove only the proven-owned path.
                        pass
            shutil.rmtree(state_dir, ignore_errors=True)
            state.metadata["cleanup_complete"] = True
            self._live_state_ids.discard(state.state_id)
