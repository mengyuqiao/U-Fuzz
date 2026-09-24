"""One-epoch offline selector snapshot for PG_PREPROCESS_V2.

The builder is the only production component that calls the local nomic
service.  Semantic preprocessing consumes the immutable snapshot through
``FrozenSelectorSnapshot`` and cannot recompute membership.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence
import urllib.request

from ufuzz.benchmarks import LoCoMoLoader, LongMemEvalSLoader
from ufuzz.domain import BenchmarkCheckpoint, BenchmarkQuery, SourceUnit
from ufuzz.pg_preprocess import BENCHMARK_ARTIFACTS, pg_preprocess_v1_manifest
from ufuzz.semantic_sidecar import canonical_bytes


SNAPSHOT_LABEL = "PG_SELECTOR_SNAPSHOT_V2"
EXPECTED_COUNTS = {"locomo": 1_986, "longmemeval-s-cleaned": 500}
QUERY_ORDER = "benchmark ascending, checkpoint_id ascending, query_id ascending"
DEFAULT_OUTPUT = Path("/home/yuqiao/ufuzz-artifacts/PG_PREPROCESS_V2/selector")
DEFAULT_LOCOMO = Path("/tmp/ufuzz-datasets/locomo10.json")
DEFAULT_LONGMEMEVAL = Path("/tmp/ufuzz-datasets/longmemeval_s_cleaned.json")
DEFAULT_OLLAMA_MANIFEST = Path(
    "/home/yuqiao/.ollama/models/manifests/registry.ollama.ai/library/nomic-embed-text/latest"
)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _canonical_write(path: Path, value: Any) -> None:
    _atomic_write(path, canonical_bytes(value) + b"\n")


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


def canonical_source_ids(sources: Iterable[SourceUnit]) -> list[str]:
    return [source.provenance_id for source in sorted(sources, key=lambda source: (source.ordinal, source.provenance_id))]


class OneEpochNomicSelector:
    """Live selector used only while constructing one complete snapshot."""

    def __init__(
        self,
        url: str = "http://127.0.0.1:11500/api/embed",
        manifest_path: Path = DEFAULT_OLLAMA_MANIFEST,
    ) -> None:
        expected = pg_preprocess_v1_manifest().selector["immutable_model_manifest_sha256"]
        if sha256(manifest_path.read_bytes()).hexdigest() != expected:
            raise RuntimeError("nomic embedding identity mismatch")
        self.url = url
        self._checkpoint_key: tuple[str, str] | None = None
        self._source_ids: tuple[str, ...] = ()
        self._source_vectors: list[list[float]] = []
        self.source_embedding_batches = 0
        self.query_embeddings = 0

    def _embed(self, texts: list[str]) -> list[list[float]]:
        request = urllib.request.Request(
            self.url,
            data=json.dumps({
                "model": "nomic-embed-text:latest", "input": texts,
                "truncate": True, "keep_alive": "10m",
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            values = json.load(response)["embeddings"]
        if len(values) != len(texts) or any(len(value) != 768 for value in values):
            raise RuntimeError("embedding shape mismatch")
        return values

    def _vectors_for(self, checkpoint: BenchmarkCheckpoint) -> list[list[float]]:
        key = (checkpoint.benchmark, checkpoint.checkpoint_id)
        source_ids = tuple(source.provenance_id for source in checkpoint.sources)
        if key != self._checkpoint_key or source_ids != self._source_ids:
            vectors: list[list[float]] = []
            for start in range(0, len(checkpoint.sources), 16):
                vectors.extend(self._embed([source.text for source in checkpoint.sources[start:start + 16]]))
                self.source_embedding_batches += 1
            self._checkpoint_key = key
            self._source_ids = source_ids
            self._source_vectors = vectors
        return self._source_vectors

    def select(self, checkpoint: BenchmarkCheckpoint, query: BenchmarkQuery) -> dict[str, Any]:
        vectors = self._vectors_for(checkpoint)
        query_vector = self._embed([query.text])[0]
        self.query_embeddings += 1
        ranked = sorted(
            zip(checkpoint.sources, vectors, strict=True),
            key=lambda item: (-_cosine(query_vector, item[1]), item[0].ordinal, item[0].provenance_id),
        )
        relevant_members = [source for source, _ in ranked[: min(64, len(ranked))]]
        relevant_set = {source.provenance_id for source in relevant_members}
        event = f"{checkpoint.checkpoint_id}\0{query.query_id}\0urc-pool-v1"
        outside_members = sorted(
            (source for source in checkpoint.sources if source.provenance_id not in relevant_set),
            key=lambda source: (
                sha256((event + "\0" + source.provenance_id).encode()).hexdigest(),
                source.ordinal,
                source.provenance_id,
            ),
        )[:16]
        return {
            "query_id": query.query_id,
            "checkpoint_id": checkpoint.checkpoint_id,
            "benchmark": checkpoint.benchmark,
            "relevant_candidate_provenance_ids": canonical_source_ids(relevant_members),
            "unrelated_candidate_provenance_ids": canonical_source_ids(outside_members),
        }


def _ordered_checkpoints(locomo: Path, longmemeval: Path) -> list[BenchmarkCheckpoint]:
    checkpoints = list(LoCoMoLoader().load(locomo)) + list(LongMemEvalSLoader().load(longmemeval))
    return sorted(checkpoints, key=lambda item: (item.benchmark, item.checkpoint_id))


def validate_snapshot_records(
    records: Sequence[Mapping[str, Any]], checkpoints: Sequence[BenchmarkCheckpoint]
) -> dict[str, Any]:
    expected: dict[str, tuple[BenchmarkCheckpoint, BenchmarkQuery]] = {}
    for checkpoint in checkpoints:
        for query in checkpoint.queries:
            if query.query_id in expected:
                raise RuntimeError(f"duplicate benchmark query ID: {query.query_id}")
            expected[query.query_id] = (checkpoint, query)
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    errors: list[dict[str, str]] = []
    for record in records:
        query_id = record.get("query_id")
        if not isinstance(query_id, str) or query_id not in expected:
            errors.append({"query_id": str(query_id), "error": "unknown_query"})
            continue
        if query_id in seen:
            errors.append({"query_id": query_id, "error": "duplicate_query"})
            continue
        seen.add(query_id)
        checkpoint, _ = expected[query_id]
        if record.get("benchmark") != checkpoint.benchmark or record.get("checkpoint_id") != checkpoint.checkpoint_id:
            errors.append({"query_id": query_id, "error": "checkpoint_binding"})
        sources = {source.provenance_id: source for source in checkpoint.sources}
        relevant = record.get("relevant_candidate_provenance_ids")
        outside = record.get("unrelated_candidate_provenance_ids")
        if not isinstance(relevant, list) or len(relevant) > 64 or len(relevant) != len(set(relevant)):
            errors.append({"query_id": query_id, "error": "relevant_cardinality"})
            continue
        if not isinstance(outside, list) or len(outside) > 16 or len(outside) != len(set(outside)):
            errors.append({"query_id": query_id, "error": "outside_cardinality"})
            continue
        if set(relevant) & set(outside):
            errors.append({"query_id": query_id, "error": "candidate_overlap"})
        if any(value not in sources for value in relevant + outside):
            errors.append({"query_id": query_id, "error": "unknown_provenance"})
            continue
        order = lambda value: (sources[value].ordinal, value)
        if relevant != sorted(relevant, key=order) or outside != sorted(outside, key=order):
            errors.append({"query_id": query_id, "error": "presentation_order"})
        counts[checkpoint.benchmark] += 1
    missing = sorted(set(expected) - seen)
    if missing:
        errors.append({"query_id": missing[0], "error": f"missing_queries:{len(missing)}"})
    if dict(counts) != EXPECTED_COUNTS:
        errors.append({"query_id": "*", "error": f"benchmark_counts:{dict(counts)}"})
    return {
        "valid": not errors,
        "total_queries": len(seen),
        "query_counts": dict(sorted(counts.items())),
        "errors": errors,
        "zero_gold_or_evidence_inputs": True,
    }


def build_snapshot(
    output: Path,
    *,
    locomo: Path = DEFAULT_LOCOMO,
    longmemeval: Path = DEFAULT_LONGMEMEVAL,
    url: str = "http://127.0.0.1:11500/api/embed",
    ollama_version: str,
    manifest_path: Path = DEFAULT_OLLAMA_MANIFEST,
) -> dict[str, Any]:
    if (output / "selector-snapshot.jsonl").exists():
        raise RuntimeError("completed selector snapshot already exists; regeneration is forbidden")
    LoCoMoLoader().artifact.verify(locomo)
    LongMemEvalSLoader().artifact.verify(longmemeval)
    checkpoints = _ordered_checkpoints(locomo, longmemeval)
    ordered = [
        (checkpoint, query)
        for checkpoint in checkpoints
        for query in sorted(checkpoint.queries, key=lambda item: item.query_id)
    ]
    ordered.sort(key=lambda item: (item[0].benchmark, item[0].checkpoint_id, item[1].query_id))
    started = datetime.now(timezone.utc).isoformat()
    selector = OneEpochNomicSelector(url, manifest_path)
    records: list[dict[str, Any]] = []
    for index, (checkpoint, query) in enumerate(ordered, 1):
        records.append(selector.select(checkpoint, query))
        if index == 1 or index % 25 == 0 or index == len(ordered):
            print(f"selector-snapshot {index}/{len(ordered)} {query.query_id}", flush=True)
    validation = validate_snapshot_records(records, checkpoints)
    if not validation["valid"]:
        raise RuntimeError(f"selector snapshot validation failed: {validation['errors'][:10]}")
    snapshot_bytes = b"".join(canonical_bytes(record) + b"\n" for record in records)
    snapshot_sha256 = sha256(snapshot_bytes).hexdigest()
    manifest = {
        "artifact_label": SNAPSHOT_LABEL,
        "selector_snapshot_sha256": snapshot_sha256,
        "dataset_sha256": {
            "locomo": BENCHMARK_ARTIFACTS["locomo"]["sha256"],
            "longmemeval-s-cleaned": BENCHMARK_ARTIFACTS["longmemeval-s-cleaned"]["sha256"],
        },
        "query_counts": validation["query_counts"],
        "query_count": validation["total_queries"],
        "model": pg_preprocess_v1_manifest().selector,
        "ollama_version": ollama_version,
        "embedding_dimension": 768,
        "distance": "cosine",
        "top_k": 64,
        "outside_pool": 16,
        "source_embedding_cache": "one immutable loaded checkpoint at a time; keyed by benchmark/checkpoint/provenance sequence; raw source objects are never mutated",
        "query_ordering": QUERY_ORDER,
        "service_concurrency": 1,
        "creation_started_utc": started,
        "creation_completed_utc": datetime.now(timezone.utc).isoformat(),
        "gold_evidence_usage": "NONE",
        "information_firewall": "raw query and same-checkpoint raw SourceUnits only",
        "source_embedding_batches": selector.source_embedding_batches,
        "query_embeddings": selector.query_embeddings,
        "validation": validation,
    }
    manifest_bytes = canonical_bytes(manifest) + b"\n"
    output.mkdir(parents=True, exist_ok=True)
    _atomic_write(output / "selector-snapshot.jsonl", snapshot_bytes)
    _atomic_write(output / "selector-manifest.json", manifest_bytes)
    checksums = (
        f"{snapshot_sha256}  selector-snapshot.jsonl\n"
        f"{sha256(manifest_bytes).hexdigest()}  selector-manifest.json\n"
    )
    _atomic_write(output / "SHA256SUMS", checksums.encode("ascii"))
    return manifest


@dataclass(frozen=True, slots=True)
class FrozenSelectorSnapshot:
    path: Path
    expected_sha256: str
    _records: Mapping[str, Mapping[str, Any]]

    @classmethod
    def load(cls, path: Path, expected_sha256: str) -> "FrozenSelectorSnapshot":
        raw = path.read_bytes()
        actual = sha256(raw).hexdigest()
        if actual != expected_sha256:
            raise RuntimeError(f"selector snapshot digest mismatch: {actual}")
        records: dict[str, Mapping[str, Any]] = {}
        for line in raw.splitlines():
            value = json.loads(line)
            query_id = value.get("query_id")
            if not isinstance(query_id, str) or query_id in records:
                raise RuntimeError("selector snapshot contains invalid or duplicate query identity")
            records[query_id] = value
        return cls(path, expected_sha256, MappingProxyType(records))

    def select(self, checkpoint: BenchmarkCheckpoint, query: BenchmarkQuery) -> dict[str, Any]:
        record = self._records.get(query.query_id)
        if record is None:
            raise RuntimeError(f"query missing from selector snapshot: {query.query_id}")
        if record.get("benchmark") != checkpoint.benchmark or record.get("checkpoint_id") != checkpoint.checkpoint_id:
            raise RuntimeError("selector snapshot checkpoint binding mismatch")
        sources = {source.provenance_id: source for source in checkpoint.sources}
        relevant = list(record["relevant_candidate_provenance_ids"])
        outside = list(record["unrelated_candidate_provenance_ids"])
        if any(value not in sources for value in relevant + outside) or set(relevant) & set(outside):
            raise RuntimeError("selector snapshot candidate provenance mismatch")
        order = lambda value: (sources[value].ordinal, value)
        if relevant != sorted(relevant, key=order) or outside != sorted(outside, key=order):
            raise RuntimeError("selector snapshot presentation order mismatch")
        synthetic = "ufuzz-session-local-" + sha256(
            f"{checkpoint.checkpoint_id}\0{query.query_id}\0unsupported-v1".encode()
        ).hexdigest()[:24]
        raw_normalized = " ".join(" ".join(source.text.casefold().split()) for source in checkpoint.sources)
        return {
            "relevant_candidate_provenance_ids": relevant,
            "unrelated_candidate_provenance_ids": outside,
            "synthetic_unsupported_target": synthetic,
            "synthetic_exact_absent": synthetic.casefold() not in raw_normalized,
            "selector_identity": {
                "artifact_label": SNAPSHOT_LABEL,
                "selector_snapshot_sha256": self.expected_sha256,
            },
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--locomo", type=Path, default=DEFAULT_LOCOMO)
    parser.add_argument("--longmemeval", type=Path, default=DEFAULT_LONGMEMEVAL)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11500/api/embed")
    parser.add_argument("--ollama-version", required=True)
    parser.add_argument("--ollama-manifest", type=Path, default=DEFAULT_OLLAMA_MANIFEST)
    args = parser.parse_args(argv)
    result = build_snapshot(
        args.output, locomo=args.locomo, longmemeval=args.longmemeval,
        url=args.ollama_url, ollama_version=args.ollama_version,
        manifest_path=args.ollama_manifest,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
