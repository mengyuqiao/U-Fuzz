"""Benchmark-neutral domain records with stable provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterator, Mapping


JsonMapping = Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class DatasetArtifact:
    """Immutable identity for a benchmark artifact."""

    repository: str
    revision: str
    filename: str
    sha256: str
    size_bytes: int
    url: str

    def verify(self, path: str | Path, chunk_size: int = 1 << 20) -> None:
        candidate = Path(path)
        actual_size = candidate.stat().st_size
        if actual_size != self.size_bytes:
            raise ValueError(
                f"artifact size mismatch for {candidate}: "
                f"expected {self.size_bytes}, got {actual_size}"
            )
        digest = sha256()
        with candidate.open("rb") as stream:
            for chunk in iter(lambda: stream.read(chunk_size), b""):
                digest.update(chunk)
        actual_sha = digest.hexdigest()
        if actual_sha != self.sha256:
            raise ValueError(
                f"artifact SHA-256 mismatch for {candidate}: "
                f"expected {self.sha256}, got {actual_sha}"
            )


@dataclass(frozen=True, slots=True)
class SourceUnit:
    """One stable benchmark source unit, normally one turn or message."""

    benchmark: str
    checkpoint_id: str
    session_id: str
    source_id: str
    ordinal: int
    text: str
    timestamp: str | None
    speaker: str | None
    role: str | None
    raw: JsonMapping

    @property
    def provenance_id(self) -> str:
        return ":".join(
            (self.benchmark, self.checkpoint_id, self.session_id, self.source_id)
        )


@dataclass(frozen=True, slots=True)
class BenchmarkQuery:
    """A benchmark query without interpreting evaluator-only fields."""

    benchmark: str
    checkpoint_id: str
    query_id: str
    text: str
    timestamp: str | None
    query_type: str | int | None
    raw: JsonMapping


@dataclass(frozen=True, slots=True)
class BenchmarkCheckpoint:
    """One original checkpoint and its associated query set."""

    benchmark: str
    checkpoint_id: str
    sources: tuple[SourceUnit, ...]
    queries: tuple[BenchmarkQuery, ...]
    raw: JsonMapping
    artifact: DatasetArtifact
    metadata: JsonMapping = field(default_factory=dict)


def iter_json_array(path: str | Path, chunk_size: int = 1 << 20) -> Iterator[Any]:
    """Stream values from a top-level JSON array using only the standard library."""

    import json

    decoder = json.JSONDecoder()
    candidate = Path(path)
    with candidate.open("r", encoding="utf-8") as stream:
        buffer = ""
        position = 0
        started = False
        eof = False
        while True:
            if position >= len(buffer) and not eof:
                buffer = stream.read(chunk_size)
                position = 0
                if not buffer:
                    eof = True

            while position < len(buffer) and buffer[position].isspace():
                position += 1

            if not started:
                if position >= len(buffer):
                    if eof:
                        raise ValueError(f"empty JSON artifact: {candidate}")
                    continue
                if buffer[position] != "[":
                    raise ValueError(f"expected a top-level JSON array: {candidate}")
                started = True
                position += 1

            while True:
                while position < len(buffer) and (
                    buffer[position].isspace() or buffer[position] == ","
                ):
                    position += 1
                if position < len(buffer) and buffer[position] == "]":
                    return
                try:
                    value, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    if eof:
                        raise
                    buffer = buffer[position:] + stream.read(chunk_size)
                    position = 0
                    if not buffer:
                        eof = True
                    continue
                yield value
                position = end
                if position > chunk_size:
                    buffer = buffer[position:]
                    position = 0
                break
