"""Small common adapter contract for Phase 1."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from hashlib import sha256
import json
from typing import Any, Mapping, Sequence

from ufuzz.domain import SourceUnit
from ufuzz.mapping import RetrievalRecord


class CapabilityStatus(StrEnum):
    SUPPORTED = "supported"
    CONDITIONAL = "conditional"
    UNSUPPORTED = "unsupported"
    UNAVAILABLE = "unavailable"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class Capability:
    status: CapabilityStatus
    reason: str
    observable: str | None = None


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    backend: str
    package: str
    version_or_commit: str
    available: bool
    dependencies: tuple[str, ...]
    services: tuple[str, ...]
    operations: Mapping[str, Capability]
    controllable_initialization: tuple[str, ...]
    notes: tuple[str, ...] = ()


@dataclass(slots=True)
class StateHandle:
    backend: str
    state_id: str
    checkpoint_id: str
    initialization_digest: str
    backend_state: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OperationReceipt:
    backend: str
    state_id: str
    operation: str
    success: bool
    target_local_id: str | None
    affected_local_ids: tuple[str, ...]
    provenance_ids: tuple[str, ...]
    before: Any
    after: Any
    raw: Any
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class InitializationArtifact:
    checkpoint_id: str
    sources: tuple[SourceUnit, ...]
    frozen_config: Mapping[str, Any]
    digest: str

    @classmethod
    def create(
        cls,
        checkpoint_id: str,
        sources: Sequence[SourceUnit],
        frozen_config: Mapping[str, Any],
    ) -> "InitializationArtifact":
        safe_sources = tuple(
            SourceUnit(
                benchmark=source.benchmark,
                checkpoint_id=source.checkpoint_id,
                session_id=source.session_id,
                source_id=source.source_id,
                ordinal=source.ordinal,
                text=source.text,
                timestamp=source.timestamp,
                speaker=source.speaker,
                role=source.role,
                raw={},
            )
            for source in sources
        )
        payload = {
            "checkpoint_id": checkpoint_id,
            "sources": [
                {
                    "provenance_id": source.provenance_id,
                    "ordinal": source.ordinal,
                    "timestamp": source.timestamp,
                    "speaker": source.speaker,
                    "role": source.role,
                    "text": source.text,
                }
                for source in safe_sources
            ],
            "frozen_config": frozen_config,
        }
        digest = sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        return cls(
            checkpoint_id=checkpoint_id,
            sources=safe_sources,
            frozen_config=dict(frozen_config),
            digest=digest,
        )


class BackendAdapter(ABC):
    @abstractmethod
    def capabilities(self) -> BackendCapabilities: ...

    @abstractmethod
    async def create_isolated_state(
        self, artifact: InitializationArtifact
    ) -> StateHandle: ...

    @abstractmethod
    async def ingest(
        self, state: StateHandle, sources: Sequence[SourceUnit]
    ) -> tuple[OperationReceipt, ...]: ...

    @abstractmethod
    async def retrieve(
        self, state: StateHandle, query: str, top_k: int
    ) -> tuple[RetrievalRecord, ...]: ...

    @abstractmethod
    async def update(
        self,
        state: StateHandle,
        target_local_id: str,
        new_text: str,
        *,
        provenance_ids: Sequence[str],
        context: Mapping[str, Any] | None = None,
    ) -> OperationReceipt: ...

    @abstractmethod
    async def delete(
        self,
        state: StateHandle,
        target_local_id: str,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> OperationReceipt: ...

    @abstractmethod
    async def unrelated_existing_region_change(
        self,
        state: StateHandle,
        target_local_id: str,
        new_text: str,
        *,
        provenance_ids: Sequence[str],
        context: Mapping[str, Any] | None = None,
    ) -> OperationReceipt: ...

    @abstractmethod
    async def observable_state(self, state: StateHandle) -> Any: ...

    @abstractmethod
    async def clone_state(self, state: StateHandle) -> StateHandle: ...

    @abstractmethod
    async def replay_state(
        self, artifact: InitializationArtifact
    ) -> StateHandle: ...

    @abstractmethod
    async def reset(self, state: StateHandle) -> None: ...

    @abstractmethod
    async def teardown(self, state: StateHandle) -> None: ...


def source_metadata(source: SourceUnit) -> dict[str, Any]:
    return {
        "ufuzz_provenance_id": source.provenance_id,
        "ufuzz_benchmark": source.benchmark,
        "ufuzz_checkpoint_id": source.checkpoint_id,
        "ufuzz_session_id": source.session_id,
        "ufuzz_source_id": source.source_id,
        "ufuzz_ordinal": source.ordinal,
        "ufuzz_timestamp": source.timestamp,
        "ufuzz_speaker": source.speaker,
        "ufuzz_role": source.role,
    }


def public_dataclass(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return value
