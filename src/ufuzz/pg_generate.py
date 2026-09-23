"""Resumable full-benchmark generation for the frozen PG_PREPROCESS_V1 contract.

The ordinary test suite exercises planning, durability, merging, and mechanical
validation with doubles.  Only ``run`` and ``resume`` load the local Qwen model
and call the frozen localhost embedding service.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile
import time
from typing import Any, Protocol
import unicodedata
import urllib.error
import urllib.request

from ufuzz.benchmarks import LoCoMoLoader, LongMemEvalSLoader, resolve_answer_session_scope
from ufuzz.domain import BenchmarkCheckpoint, BenchmarkQuery, SourceUnit
from ufuzz.pg_preprocess import (
    PG_PREPROCESS_V1,
    PG_PREPROCESS_V1_PROMPTS,
    SEARCH_FORBIDDEN_FIELDS,
    pg_preprocess_v1_manifest,
)
from ufuzz.semantic_sidecar import canonical_bytes, resolve_locomo_gold_field


CONTRACT_DIGEST = "c342c97cdebdf89306d06a1c12ceb8702a9a6c8d5f98e0e6a821f2163a07b174"
SEMANTIC_VALIDATION_STATUS = "PENDING_HUMAN_VALIDATION"
DEFAULT_LOCOMO = Path("/tmp/ufuzz-datasets/locomo10.json")
DEFAULT_LONGMEMEVAL = Path("/tmp/ufuzz-datasets/longmemeval_s_cleaned.json")
DEFAULT_MODEL_PATH = Path("/tmp/memos-v2033-hf-model")
DEFAULT_OLLAMA_MANIFEST = Path(
    "/home/yuqiao/.ollama/models/manifests/registry.ollama.ai/library/nomic-embed-text/latest"
)
EXPECTED_COUNTS = {"locomo": 1_986, "longmemeval-s-cleaned": 500}
CALL_LIMITS = {
    "query_slot": 512,
    "bounded_proposition_extraction": 1536,
    "existing_memory_state": 128,
    "query_relevance": 128,
    "preference_slot": 512,
    "relation_entity_match": 256,
    "g_evidence_mapping": 1024,
}
PLACEHOLDERS = frozenset(("", "none", "null", "unknown", "unresolved", "n/a"))
ELIGIBLE_REASONS = frozenset(("explicit_mutable_memory_state",))
INELIGIBLE_REASONS = frozenset(
    (
        "pure_interrogative", "command_or_request_without_assertion",
        "conversational_filler", "discourse_or_meta_commentary",
        "unsupported_inference", "missing_meaningful_entity",
        "missing_meaningful_relation", "missing_mutable_value",
        "malformed_or_ungrounded", "source_not_in_parent_history",
        "not_asserted_semantic_proposition",
    )
)
UNRESOLVED_REASONS = frozenset(
    ("assertion_status_ambiguous", "entity_relation_value_ambiguous", "temporal_scope_ambiguous")
)


def _normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value)).strip())


def _identity(value: Any) -> str:
    return _normalize(value).casefold()


def _digest(value: Any) -> str:
    data = value if isinstance(value, bytes) else canonical_bytes(value)
    return sha256(data).hexdigest()


def shard_for_query(query_id: str, num_shards: int) -> int:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    return int(sha256(query_id.encode("utf-8")).hexdigest()[:16], 16) % num_shards


def atomic_write(path: Path, data: bytes) -> None:
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


def write_canonical_json(path: Path, value: Any) -> None:
    atomic_write(path, canonical_bytes(value) + b"\n")


def query_record_path(root: Path, shard: int, query_id: str) -> Path:
    name = sha256(query_id.encode("utf-8")).hexdigest() + ".json"
    return root / "shards" / f"shard-{shard:04d}" / "queries" / name


@dataclass(frozen=True, slots=True)
class PlannedQuery:
    query_id: str
    checkpoint_id: str
    benchmark: str
    native_type: str
    shard: int


def _iter_checkpoints(locomo: Path, longmemeval: Path) -> Iterable[BenchmarkCheckpoint]:
    yield from LoCoMoLoader().load(locomo)
    yield from LongMemEvalSLoader().load(longmemeval)


def create_plan(
    root: Path,
    num_shards: int,
    locomo: Path = DEFAULT_LOCOMO,
    longmemeval: Path = DEFAULT_LONGMEMEVAL,
    selected_query_ids: frozenset[str] | None = None,
) -> dict[str, Any]:
    manifest = pg_preprocess_v1_manifest()
    if manifest.sha256_digest != CONTRACT_DIGEST:
        raise RuntimeError("PG_PREPROCESS_V1 contract digest mismatch")
    LoCoMoLoader().artifact.verify(locomo)
    LongMemEvalSLoader().artifact.verify(longmemeval)
    planned: list[PlannedQuery] = []
    full_counts: Counter[str] = Counter()
    selected_seen: set[str] = set()
    for checkpoint in _iter_checkpoints(locomo, longmemeval):
        for query in checkpoint.queries:
            full_counts[checkpoint.benchmark] += 1
            if selected_query_ids is not None and query.query_id not in selected_query_ids:
                continue
            selected_seen.add(query.query_id)
            planned.append(
                PlannedQuery(
                    query.query_id,
                    checkpoint.checkpoint_id,
                    checkpoint.benchmark,
                    str(query.query_type),
                    shard_for_query(query.query_id, num_shards),
                )
            )
    if dict(full_counts) != EXPECTED_COUNTS:
        raise RuntimeError(f"benchmark query counts differ: {dict(full_counts)}")
    if selected_query_ids is not None and selected_seen != set(selected_query_ids):
        raise RuntimeError(f"selected query IDs not found: {sorted(set(selected_query_ids) - selected_seen)}")
    planned.sort(key=lambda item: item.query_id)
    plan = {
        "preprocessing_contract": PG_PREPROCESS_V1,
        "contract_digest": CONTRACT_DIGEST,
        "semantic_validation_status": SEMANTIC_VALIDATION_STATUS,
        "num_shards": num_shards,
        "dataset_artifacts": {
            "locomo": LoCoMoLoader().artifact.sha256,
            "longmemeval-s-cleaned": LongMemEvalSLoader().artifact.sha256,
        },
        "full_dataset_query_counts": dict(sorted(full_counts.items())),
        "planned_query_count": len(planned),
        "shard_rule": "int(first_16_hex(sha256(query_id)), 16) % num_shards",
        "queries": [asdict(item) for item in planned],
    }
    plan["query_manifest_sha256"] = _digest(plan["queries"])
    write_canonical_json(root / "plan" / "query-shards.json", plan)
    for shard in range(num_shards):
        write_canonical_json(
            root / "plan" / f"shard-{shard:04d}.json",
            {
                "shard": shard,
                "num_shards": num_shards,
                "query_ids": [item.query_id for item in planned if item.shard == shard],
                "contract_digest": CONTRACT_DIGEST,
            },
        )
    return plan


def load_plan(root: Path) -> dict[str, Any]:
    path = root / "plan" / "query-shards.json"
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("contract_digest") != CONTRACT_DIGEST:
        raise RuntimeError("plan contract digest mismatch")
    if plan.get("semantic_validation_status") != SEMANTIC_VALIDATION_STATUS:
        raise RuntimeError("plan semantic-validation status mismatch")
    if _digest(plan["queries"]) != plan.get("query_manifest_sha256"):
        raise RuntimeError("plan query manifest digest mismatch")
    return plan


class Generator(Protocol):
    def call(self, task_id: str, query_id: str, kind: str, payload: Mapping[str, Any]) -> dict[str, Any]: ...


class Selector(Protocol):
    def select(self, checkpoint: BenchmarkCheckpoint, query: BenchmarkQuery) -> dict[str, Any]: ...


class LocalQwenGenerator:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH) -> None:
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        random.seed(1729)
        np.random.seed(1729)
        torch.manual_seed(1729)
        torch.cuda.manual_seed_all(1729)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True,
            revision="40c069824f4251a91eefaf281ebe4c544efd3e18",
        )
        if sha256(self._tokenizer.chat_template.encode()).hexdigest() != pg_preprocess_v1_manifest().model["chat_template_sha256"]:
            raise RuntimeError("chat-template digest mismatch")
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True,
            revision="40c069824f4251a91eefaf281ebe4c544efd3e18",
            dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa",
        )
        self._model.eval()

    def call(self, task_id: str, query_id: str, kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        prompt = PG_PREPROCESS_V1_PROMPTS[kind]
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": canonical_bytes(payload).decode("utf-8")},
        ]
        rendered = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        encoded = self._tokenizer(rendered, return_tensors="pt").to(self._model.device)
        input_tokens = int(encoded.input_ids.shape[1])
        if input_tokens > 39_000:
            return self._record(task_id, query_id, kind, payload, input_tokens, "", None, "context_bound_exceeded")
        started = time.monotonic()
        with self._torch.inference_mode():
            row = self._model.generate(
                **encoded, do_sample=False, max_new_tokens=CALL_LIMITS[kind],
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id, use_cache=True,
            )[0]
        raw = self._tokenizer.decode(row[input_tokens:], skip_special_tokens=True).strip()
        value = raw
        if value.startswith("```"):
            value = re.sub(r"^```(?:json)?\s*", "", value)
            value = re.sub(r"\s*```$", "", value)
        parsed = None
        error = None
        try:
            parsed = json.loads(value)
        except Exception as exception:
            error = f"{type(exception).__name__}: {exception}"
        result = self._record(task_id, query_id, kind, payload, input_tokens, raw, parsed, error)
        result["wall_seconds"] = time.monotonic() - started
        return result

    def _record(
        self, task_id: str, query_id: str, kind: str, payload: Mapping[str, Any],
        input_tokens: int, raw: str, parsed: Any, error: str | None,
    ) -> dict[str, Any]:
        return {
            "task_id": task_id, "query_id": query_id, "call_kind": kind,
            "canonical_input_sha256": _digest(payload),
            "prompt_sha256": pg_preprocess_v1_manifest().prompt_sha256[kind],
            "model_id": "Qwen/Qwen3-14B",
            "model_revision": "40c069824f4251a91eefaf281ebe4c544efd3e18",
            "input_token_count": input_tokens,
            "output_token_count": len(self._tokenizer(raw, add_special_tokens=False).input_ids),
            "raw_output_sha256": sha256(raw.encode()).hexdigest(),
            "raw_output": raw,
            "parsed_canonical_output": parsed,
            "parse_error": error,
        }


class NomicSelector:
    def __init__(
        self,
        url: str = "http://127.0.0.1:11500/api/embed",
        manifest_path: Path = DEFAULT_OLLAMA_MANIFEST,
    ) -> None:
        expected = pg_preprocess_v1_manifest().selector["immutable_model_manifest_sha256"]
        if sha256(manifest_path.read_bytes()).hexdigest() != expected:
            raise RuntimeError("nomic embedding identity mismatch")
        self.url = url
        self._source_cache: dict[str, tuple[tuple[str, ...], list[list[float]]]] = {}

    def _embed(self, texts: list[str]) -> list[list[float]]:
        request = urllib.request.Request(
            self.url,
            data=json.dumps({
                "model": "nomic-embed-text:latest", "input": texts,
                "truncate": True, "keep_alive": "10m",
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                values = json.load(response)["embeddings"]
        except urllib.error.HTTPError as exception:
            if len(texts) > 1:
                middle = len(texts) // 2
                return self._embed(texts[:middle]) + self._embed(texts[middle:])
            raise RuntimeError(f"embedding failed for one input: {exception.read().decode(errors='replace')}") from exception
        if len(values) != len(texts) or any(len(value) != 768 for value in values):
            raise RuntimeError("embedding shape mismatch")
        return values

    def select(self, checkpoint: BenchmarkCheckpoint, query: BenchmarkQuery) -> dict[str, Any]:
        sources = list(checkpoint.sources)
        key = f"{checkpoint.benchmark}\0{checkpoint.checkpoint_id}"
        source_ids = tuple(source.provenance_id for source in sources)
        cached = self._source_cache.get(key)
        if cached is None or cached[0] != source_ids:
            vectors: list[list[float]] = []
            for start in range(0, len(sources), 16):
                vectors.extend(self._embed([source.text for source in sources[start:start + 16]]))
            self._source_cache[key] = source_ids, vectors
        else:
            vectors = cached[1]
        query_vector = self._embed([query.text])[0]
        ranked = sorted(
            zip(sources, vectors, strict=True),
            key=lambda item: (-_cosine(query_vector, item[1]), item[0].ordinal, item[0].provenance_id),
        )
        relevant = [source for source, _ in ranked[: min(64, len(ranked))]]
        relevant_ids = {source.provenance_id for source in relevant}
        event = f"{checkpoint.checkpoint_id}\0{query.query_id}\0urc-pool-v1"
        outside = sorted(
            (source for source in sources if source.provenance_id not in relevant_ids),
            key=lambda source: (
                sha256((event + "\0" + source.provenance_id).encode()).hexdigest(),
                source.ordinal, source.provenance_id,
            ),
        )[:16]
        synthetic = "ufuzz-session-local-" + sha256(
            f"{checkpoint.checkpoint_id}\0{query.query_id}\0unsupported-v1".encode()
        ).hexdigest()[:24]
        raw_normalized = " ".join(" ".join(source.text.casefold().split()) for source in sources)
        return {
            "relevant_candidate_provenance_ids": [source.provenance_id for source in relevant],
            "unrelated_candidate_provenance_ids": [source.provenance_id for source in outside],
            "synthetic_unsupported_target": synthetic,
            "synthetic_exact_absent": synthetic.casefold() not in raw_normalized,
            "selector_identity": pg_preprocess_v1_manifest().selector,
        }


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


def _source_payload(source: SourceUnit) -> dict[str, Any]:
    return {
        "provenance_id": source.provenance_id, "source_ordinal": source.ordinal,
        "timestamp": source.timestamp, "speaker_or_role": source.speaker or source.role,
        "text": source.text,
    }


def _validate_slot(call: Mapping[str, Any], checkpoint: BenchmarkCheckpoint, query: BenchmarkQuery) -> dict[str, Any]:
    value = call.get("parsed_canonical_output")
    expected = {
        "status", "entity_description", "entity_status", "relation_label",
        "anchor_kind", "query_anchor", "structural_explanation", "temporal_scope",
        "constraints", "replaceable_slots", "primary_reason",
    }
    fallback = {"status": "UNRESOLVED", "primary_reason": "genuinely_ambiguous_question", "validation_error": "schema_or_parse"}
    if not isinstance(value, dict) or set(value) != expected:
        return fallback
    if value["status"] not in {"RESOLVED", "UNRESOLVED"}:
        return fallback
    if value["status"] == "UNRESOLVED":
        reason = value.get("primary_reason")
        if reason not in {"genuinely_ambiguous_question", "implicit_relation_unresolved", "entity_coreference_unresolved", "temporal_constraint_unresolved"}:
            value["primary_reason"] = "genuinely_ambiguous_question"
        return value
    if value.get("entity_status") != "RESOLVED" or _identity(value.get("relation_label")) in PLACEHOLDERS:
        return {"status": "UNRESOLVED", "primary_reason": "entity_coreference_unresolved", "validation_error": "required"}
    if value.get("anchor_kind") == "EXPLICIT":
        anchor = value.get("query_anchor")
        if not isinstance(anchor, str) or not anchor or query.text.count(anchor) != 1:
            return {"status": "UNRESOLVED", "primary_reason": "implicit_relation_unresolved", "validation_error": "anchor"}
        query_anchor: Any = anchor
    elif value.get("anchor_kind") == "IMPLICIT" and _normalize(value.get("structural_explanation") or ""):
        query_anchor = {"implicit_explanation": _normalize(value["structural_explanation"])}
    else:
        return {"status": "UNRESOLVED", "primary_reason": "implicit_relation_unresolved", "validation_error": "explanation"}
    value["relation_slot"] = {
        "root_checkpoint_id": checkpoint.checkpoint_id, "query_id": query.query_id,
        "slot_index": 0, "normalized_relation_label": _identity(value["relation_label"]),
        "query_anchor": query_anchor,
    }
    value["primary_reason"] = "none"
    return value


def _validate_preference_slot(
    call: Mapping[str, Any], checkpoint: BenchmarkCheckpoint, query: BenchmarkQuery, expected_basis: str
) -> dict[str, Any]:
    value = call.get("parsed_canonical_output")
    keys = {
        "status", "entity_description", "entity_status", "relation_family", "answer_domain",
        "anchor_kind", "query_anchor", "structural_explanation", "temporal_scope",
        "constraints", "replaceable_slots", "resolution_basis", "primary_reason",
    }
    fallback = {"status": "UNRESOLVED", "primary_reason": "preference_domain_unresolved", "resolution_basis": "UNRESOLVED", "validation_error": "schema_or_parse"}
    if not isinstance(value, dict) or set(value) != keys:
        return fallback
    if value["status"] != "RESOLVED":
        if value.get("primary_reason") not in {"genuinely_ambiguous_question", "entity_coreference_unresolved", "preference_domain_unresolved"}:
            return fallback
        value["resolution_basis"] = "UNRESOLVED"
        return value
    if value.get("entity_status") != "RESOLVED" or _identity(value.get("relation_family")) != "preference" or not _normalize(value.get("answer_domain")):
        return fallback
    if value.get("resolution_basis") != expected_basis:
        return fallback
    if value.get("anchor_kind") == "EXPLICIT":
        anchor = value.get("query_anchor")
        if not isinstance(anchor, str) or not anchor or query.text.count(anchor) != 1:
            return fallback
        query_anchor: Any = anchor
    elif value.get("anchor_kind") == "IMPLICIT" and _normalize(value.get("structural_explanation") or ""):
        query_anchor = {"implicit_explanation": _normalize(value["structural_explanation"])}
    else:
        return fallback
    value["relation_label"] = "preference"
    value["constraints"] = list(value.get("constraints", [])) + [
        {"key": "answer_domain", "value": _normalize(value["answer_domain"])}
    ]
    value["relation_slot"] = {
        "root_checkpoint_id": checkpoint.checkpoint_id, "query_id": query.query_id,
        "slot_index": 0, "normalized_relation_label": "preference", "query_anchor": query_anchor,
    }
    value["primary_reason"] = "none"
    first_person = re.search(r"\b(?:i|me|my|mine)\b", query.text, re.IGNORECASE) is not None
    holder_ok = not first_person or any(
        token in str(value.get("entity_description", "")).casefold()
        for token in ("user", "speaker", "questioner")
    )
    if not holder_ok:
        return {"status": "UNRESOLVED", "primary_reason": "entity_coreference_unresolved", "resolution_basis": "UNRESOLVED", "validation_error": "preference_holder_mismatch"}
    return value


def _validate_proposition(value: Any, allowed: set[str], sources: Mapping[str, SourceUnit]) -> dict[str, Any] | None:
    if not isinstance(value, dict) or set(value) != {"provenance_id", "quote", "entity", "relation", "value", "temporal"}:
        return None
    source = sources.get(value["provenance_id"])
    quote = value.get("quote")
    if value["provenance_id"] not in allowed or source is None or not isinstance(quote, str) or not quote or source.text.count(quote) != 1:
        return None
    for field in ("entity", "relation", "value"):
        if _identity(value.get(field)) in PLACEHOLDERS:
            return None
    if not _normalize(value.get("temporal")):
        return None
    result = dict(value)
    result["char_start"] = source.text.index(quote)
    result["char_end"] = result["char_start"] + len(quote)
    result["source_ordinal"] = source.ordinal
    return result


def _validate_extraction(
    call: Mapping[str, Any], selector: Mapping[str, Any], sources: Mapping[str, SourceUnit]
) -> dict[str, Any]:
    parsed = call.get("parsed_canonical_output")
    result = {"relevant_propositions": [], "target_changing": [], "unrelated_propositions": [], "issues": []}
    if not isinstance(parsed, dict) or set(parsed) != {"relevant_propositions", "target_changing", "unrelated_propositions"}:
        result["issues"].append("schema_or_parse")
        return result
    relevant = set(selector["relevant_candidate_provenance_ids"])
    unrelated = set(selector["unrelated_candidate_provenance_ids"])
    for value in parsed.get("relevant_propositions", []):
        proposition = _validate_proposition(value, relevant, sources)
        if proposition is None:
            result["issues"].append("relevant_grounding")
        else:
            result["relevant_propositions"].append(proposition)
    for value in parsed.get("unrelated_propositions", []):
        proposition = _validate_proposition(value, unrelated, sources)
        if proposition is None:
            result["issues"].append("unrelated_grounding")
        else:
            result["unrelated_propositions"].append(proposition)
    for value in parsed.get("target_changing", []):
        if not isinstance(value, dict) or set(value) != {"changed_slot", "replacement_target_identity", "proposition"}:
            result["issues"].append("target_changing_schema")
            continue
        proposition = _validate_proposition(value["proposition"], relevant, sources)
        slot = value.get("changed_slot")
        if proposition is None or not isinstance(slot, str) or not (
            slot in {"entity", "relation", "temporal"} or slot.startswith("constraint:")
        ) or not _normalize(value.get("replacement_target_identity")):
            result["issues"].append("target_changing_grounding")
            continue
        result["target_changing"].append({
            "changed_slot": slot,
            "replacement_target_identity": _normalize(value["replacement_target_identity"]),
            "proposition": proposition,
        })
    return result


def _validate_eligibility(call: Mapping[str, Any]) -> dict[str, Any]:
    parsed = call.get("parsed_canonical_output")
    fallback = {"status": "UNRESOLVED", "primary_reason": "entity_relation_value_ambiguous", "validation_error": "schema_or_parse"}
    if not isinstance(parsed, dict) or set(parsed) != {"status", "primary_reason"}:
        return fallback
    allowed = {"ELIGIBLE": ELIGIBLE_REASONS, "INELIGIBLE": INELIGIBLE_REASONS, "UNRESOLVED": UNRESOLVED_REASONS}
    if parsed.get("status") not in allowed or parsed.get("primary_reason") not in allowed[parsed["status"]]:
        return fallback
    return parsed


def _validate_match(call: Mapping[str, Any]) -> dict[str, Any]:
    parsed = call.get("parsed_canonical_output")
    fallback = {"relation_match": "UNRESOLVED", "entity_match": "UNRESOLVED", "temporal_compatibility": "UNRESOLVED", "validation_error": "schema_or_parse"}
    if not isinstance(parsed, dict) or set(parsed) != {"relation_match", "entity_match", "temporal_compatibility"}:
        return fallback
    if parsed["relation_match"] not in {"MATCH", "NO_MATCH", "UNRESOLVED"} or parsed["entity_match"] not in {"MATCH", "NO_MATCH", "UNRESOLVED"} or parsed["temporal_compatibility"] not in {"COMPATIBLE", "CONTRADICTED", "UNRESOLVED"}:
        return fallback
    return parsed


def _validate_relevance(call: Mapping[str, Any]) -> dict[str, Any]:
    parsed = call.get("parsed_canonical_output")
    if not isinstance(parsed, dict) or set(parsed) != {"status"} or parsed["status"] not in {"RELEVANT", "UNRELATED", "UNRESOLVED"}:
        return {"status": "UNRESOLVED", "validation_error": "schema_or_parse"}
    return parsed


def _proposition_payload(checkpoint: BenchmarkCheckpoint, proposition: Mapping[str, Any], source: SourceUnit) -> dict[str, Any]:
    return {
        "checkpoint_id": checkpoint.checkpoint_id,
        "source_identity": {
            "provenance_id": source.provenance_id,
            "source_ordinal": source.ordinal,
            "belongs_to_parent_checkpoint": True,
        },
        "source_role_or_speaker": source.speaker or source.role,
        "source_text": source.text,
        "grounded_proposition": proposition,
    }


def _chunk_sources(sources: Sequence[SourceUnit]) -> list[list[SourceUnit]]:
    chunks: list[list[SourceUnit]] = []
    current: list[SourceUnit] = []
    characters = 0
    for source in sources:
        if current and (len(current) >= 10 or characters + len(source.text) > 12_000):
            chunks.append(current)
            current = []
            characters = 0
        current.append(source)
        characters += len(source.text)
    if current:
        chunks.append(current)
    return chunks


def _native_scope(checkpoint: BenchmarkCheckpoint, query: BenchmarkQuery) -> tuple[list[SourceUnit], list[str]]:
    if checkpoint.benchmark == "locomo":
        by_source_id = {source.source_id: source for source in checkpoint.sources}
        native_ids = [str(value) for value in (query.raw.get("evidence") or [])]
        return [by_source_id[value] for value in native_ids if value in by_source_id], [value for value in native_ids if value not in by_source_id]
    native_ids = [str(value) for value in query.raw.get("answer_session_ids", [])]
    scope = list(resolve_answer_session_scope(checkpoint, native_ids))
    narrowed = [source for source in scope if source.raw.get("has_answer") is True]
    return narrowed or scope, []


def _validate_g_call(
    call: Mapping[str, Any], allowed: set[str], sources: Mapping[str, SourceUnit]
) -> tuple[list[Any], list[dict[str, Any]], list[str]]:
    parsed = call.get("parsed_canonical_output")
    if not isinstance(parsed, dict) or set(parsed) != {"status", "accepted_options", "supports", "reason"} or parsed.get("status") not in {"RESOLVED", "UNRESOLVED"}:
        return [], [], ["schema_or_parse"]
    options = parsed["accepted_options"] if isinstance(parsed["accepted_options"], list) else []
    supports: list[dict[str, Any]] = []
    issues: list[str] = []
    for support in parsed["supports"] if isinstance(parsed["supports"], list) else []:
        if not isinstance(support, dict) or set(support) != {"component_id", "provenance_id", "quote", "independent_group"}:
            issues.append("support_schema")
            continue
        source = sources.get(support["provenance_id"])
        quote = support.get("quote")
        if support["provenance_id"] not in allowed or source is None or not isinstance(quote, str) or not quote or source.text.count(quote) != 1:
            issues.append("support_grounding")
            continue
        item = dict(support)
        item["char_start"] = source.text.index(quote)
        item["char_end"] = item["char_start"] + len(quote)
        item["source_ordinal"] = source.ordinal
        supports.append(item)
    return options, supports, issues


def process_query(
    checkpoint: BenchmarkCheckpoint,
    query: BenchmarkQuery,
    selector_engine: Selector,
    generator: Generator,
) -> dict[str, Any]:
    started = time.monotonic()
    calls: list[dict[str, Any]] = []
    selector = selector_engine.select(checkpoint, query)
    if not selector.get("synthetic_exact_absent"):
        raise RuntimeError(f"synthetic target collision for {query.query_id}")
    sources = {source.provenance_id: source for source in checkpoint.sources}

    def call(kind: str, suffix: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        result = generator.call(f"{query.query_id}:{suffix}", query.query_id, kind, payload)
        calls.append(result)
        return result

    if str(query.query_type) == "single-session-preference":
        query_only = _validate_preference_slot(
            call("preference_slot", "preference:q-only", {
                "checkpoint_id": checkpoint.checkpoint_id, "query_id": query.query_id,
                "question": query.text,
            }), checkpoint, query, "QUERY_ONLY"
        )
        hinted = _validate_preference_slot(
            call("preference_slot", "preference:hint", {
                "checkpoint_id": checkpoint.checkpoint_id, "query_id": query.query_id,
                "question": query.text, "native_question_type": "single-session-preference",
            }), checkpoint, query, "QUERY_PLUS_NATIVE_TYPE"
        )
        slot = query_only if query_only.get("status") == "RESOLVED" else hinted
    else:
        slot = _validate_slot(
            call("query_slot", "query-slot", {
                "checkpoint_id": checkpoint.checkpoint_id, "query_id": query.query_id,
                "question": query.text,
            }), checkpoint, query
        )

    relevant_ids = selector["relevant_candidate_provenance_ids"]
    outside_ids = selector["unrelated_candidate_provenance_ids"]
    extraction = _validate_extraction(
        call("bounded_proposition_extraction", "extract", {
            "query_id": query.query_id, "question": query.text, "relation_slot": slot,
            "relevant_pool": [_source_payload(sources[value]) for value in relevant_ids],
            "outside_top64_pool": [_source_payload(sources[value]) for value in outside_ids],
        }), selector, sources
    )
    resolved = slot.get("status") == "RESOLVED"
    relevant: list[dict[str, Any]] = []
    for index, proposition in enumerate(extraction["relevant_propositions"]):
        eligibility = _validate_eligibility(call(
            "existing_memory_state", f"relevant:{index}:eligibility",
            _proposition_payload(checkpoint, proposition, sources[proposition["provenance_id"]]),
        ))
        match = _validate_match(call(
            "relation_entity_match", f"relevant:{index}:match",
            {"query_id": query.query_id, "question": query.text, "relation_slot": slot, "source_proposition": proposition},
        ))
        if eligibility["status"] == "ELIGIBLE" and match["relation_match"] == "MATCH" and match["entity_match"] == "MATCH" and match["temporal_compatibility"] == "COMPATIBLE":
            relevant.append(proposition)
    relevant.sort(key=lambda value: (value["source_ordinal"], value["char_start"], canonical_bytes(value)))

    target_changing: list[dict[str, Any]] = []
    for index, opportunity in enumerate(extraction["target_changing"]):
        proposition = opportunity["proposition"]
        eligibility = _validate_eligibility(call(
            "existing_memory_state", f"target-changing:{index}:eligibility",
            _proposition_payload(checkpoint, proposition, sources[proposition["provenance_id"]]),
        ))
        match = _validate_match(call(
            "relation_entity_match", f"target-changing:{index}:match",
            {"query_id": query.query_id, "question": query.text, "relation_slot": slot,
             "changed_slot": opportunity["changed_slot"],
             "replacement_target_identity": opportunity["replacement_target_identity"],
             "source_proposition": proposition},
        ))
        if eligibility["status"] == "ELIGIBLE" and all(match[key] != "UNRESOLVED" for key in ("relation_match", "entity_match", "temporal_compatibility")) and opportunity["changed_slot"] in slot.get("replaceable_slots", []):
            target_changing.append(opportunity)
    target_changing.sort(key=lambda value: (value["proposition"]["source_ordinal"], value["proposition"]["char_start"], canonical_bytes(value)))

    unrelated: list[dict[str, Any]] = []
    for index, proposition in enumerate(extraction["unrelated_propositions"]):
        eligibility = _validate_eligibility(call(
            "existing_memory_state", f"unrelated:{index}:eligibility",
            _proposition_payload(checkpoint, proposition, sources[proposition["provenance_id"]]),
        ))
        relevance = _validate_relevance(call(
            "query_relevance", f"unrelated:{index}:relevance",
            {"query_id": query.query_id, "question": query.text, "relation_slot": slot, "source_proposition": proposition},
        ))
        if eligibility["status"] == "ELIGIBLE" and relevance["status"] == "UNRELATED":
            unrelated.append(proposition)
    unrelated.sort(key=lambda value: (value["source_ordinal"], value["char_start"], canonical_bytes(value)))
    selected_urc = unrelated[0] if resolved and unrelated else None
    p = {
        "query_id": query.query_id, "checkpoint_id": checkpoint.checkpoint_id,
        "benchmark": checkpoint.benchmark, "native_type": str(query.query_type),
        "status": "RESOLVED" if resolved else "UNRESOLVED",
        "unresolved_reason": None if resolved else slot.get("primary_reason", "query_slot_unresolved"),
        "source_selector": selector,
        "canonical_query_intent": slot,
        "opportunities": {
            "meaning_preserving": resolved,
            "target_changing": target_changing[0] if resolved and target_changing else None,
            "unsupported": selector["synthetic_unsupported_target"] if resolved and slot.get("replaceable_slots") else None,
            "update": relevant[0] if resolved and relevant else None,
            "delete": relevant[0] if resolved and relevant else None,
            "unrelated_change": None if selected_urc is None else {
                "canonical_target": {
                    "entity": _identity(selected_urc["entity"]),
                    "relation": _identity(selected_urc["relation"]),
                    "temporal_scope": _normalize(selected_urc["temporal"]),
                },
                "target_proposition": selected_urc,
            },
        },
        "extraction_issues": extraction["issues"],
    }
    _validate_search_firewall(p)

    if checkpoint.benchmark == "locomo":
        gold = resolve_locomo_gold_field(query.raw)
        gold_answer = gold.value
        gold_status = gold.status.value
    else:
        gold_answer = query.raw.get("answer")
        gold_status = "benchmark_answer"
    scope, missing_native = _native_scope(checkpoint, query)
    g_calls: list[dict[str, Any]] = []
    if gold_answer is not None and not missing_native:
        for index, chunk in enumerate(_chunk_sources(scope)):
            g_calls.append(call("g_evidence_mapping", f"g:{index:03d}", {
                "query_id": query.query_id, "question": query.text,
                "gold_answer": gold_answer,
                "native_evidence_chunk": [_source_payload(source) for source in chunk],
            }))
    allowed = {source.provenance_id for source in scope}
    options: list[Any] = []
    supports: list[dict[str, Any]] = []
    issues: list[str] = []
    for result in g_calls:
        call_options, call_supports, call_issues = _validate_g_call(result, allowed, sources)
        options.extend(call_options)
        supports.extend(call_supports)
        issues.extend(call_issues)
    options = [value for _, value in sorted({_digest(value): value for value in options}.items())]
    supports = [value for _, value in sorted({_digest(value): value for value in supports}.items())]
    g_resolved = gold_answer is not None and not missing_native and bool(options) and bool(supports)
    if gold_answer is None or missing_native:
        issues.append("gold_authority_or_native_evidence")
    image = any("img_url" in source.raw or "blip_caption" in source.raw for source in scope)
    g = {
        "query_id": query.query_id, "checkpoint_id": checkpoint.checkpoint_id,
        "benchmark": checkpoint.benchmark, "native_type": str(query.query_type),
        "status": "RESOLVED" if g_resolved else "UNRESOLVED",
        "unresolved_reason": None if g_resolved else (issues[0] if issues else "support_not_resolved"),
        "gold_answer": gold_answer,
        "gold_status": gold_status,
        "accepted_options": options if g_resolved else [],
        "E_plus": supports if g_resolved else [], "E_minus": [],
        "native_evidence_scope": sorted(allowed),
        "native_evidence_missing_ids": sorted(missing_native),
        "image_status": ("RESOLVED_TEXTUAL" if g_resolved else "UNRESOLVED_VISUAL_SUPPORT") if image else "NOT_IMAGE_BEARING",
        "issues": issues,
    }
    return {
        "preprocessing_contract": PG_PREPROCESS_V1,
        "contract_digest": CONTRACT_DIGEST,
        "semantic_validation_status": SEMANTIC_VALIDATION_STATUS,
        "query_id": query.query_id, "checkpoint_id": checkpoint.checkpoint_id,
        "benchmark": checkpoint.benchmark, "native_type": str(query.query_type),
        "terminal_status": "TERMINAL", "P": p, "G": g,
        "raw_calls": calls,
        "metrics": {
            "wall_seconds": time.monotonic() - started,
            "model_call_count": len(calls),
            "input_tokens": sum(item["input_token_count"] for item in calls),
            "output_tokens": sum(item["output_token_count"] for item in calls),
        },
    }


def _validate_search_firewall(value: Any) -> None:
    keys = {key.casefold() for key in _walk_keys(value)}
    forbidden = keys.intersection(set(SEARCH_FORBIDDEN_FIELDS) | {
        "gold_answer", "accepted_options", "e_plus", "e_minus", "native_evidence_scope",
    })
    if forbidden:
        raise RuntimeError(f"P information-flow violation: {sorted(forbidden)}")


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _walk_keys(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk_keys(item)


def _validate_terminal(record: Mapping[str, Any], checkpoint: BenchmarkCheckpoint) -> list[str]:
    errors: list[str] = []
    if record.get("contract_digest") != CONTRACT_DIGEST:
        errors.append("contract_digest")
    if record.get("semantic_validation_status") != SEMANTIC_VALIDATION_STATUS:
        errors.append("semantic_validation_status")
    if record.get("terminal_status") != "TERMINAL":
        errors.append("terminal_status")
    sources = {source.provenance_id: source for source in checkpoint.sources}
    try:
        _validate_search_firewall(record["P"])
    except RuntimeError:
        errors.append("information_flow")
    selector = record.get("P", {}).get("source_selector", {})
    synthetic = selector.get("synthetic_unsupported_target")
    if not selector.get("synthetic_exact_absent") or not isinstance(synthetic, str) or any(
        synthetic.casefold() in " ".join(source.text.casefold().split()) for source in checkpoint.sources
    ):
        errors.append("synthetic_absence")
    propositions: list[Mapping[str, Any]] = []
    opportunities = record.get("P", {}).get("opportunities", {})
    for key in ("update", "delete"):
        if isinstance(opportunities.get(key), Mapping):
            propositions.append(opportunities[key])
    tc = opportunities.get("target_changing")
    if isinstance(tc, Mapping) and isinstance(tc.get("proposition"), Mapping):
        propositions.append(tc["proposition"])
    urc = opportunities.get("unrelated_change")
    if isinstance(urc, Mapping) and isinstance(urc.get("target_proposition"), Mapping):
        propositions.append(urc["target_proposition"])
    for proposition in propositions:
        source = sources.get(proposition.get("provenance_id"))
        if source is None or source.ordinal != proposition.get("source_ordinal") or source.text[proposition.get("char_start", -1):proposition.get("char_end", -1)] != proposition.get("quote") or source.text.count(proposition.get("quote", "")) != 1:
            errors.append("P_provenance_span")
    native_allowed, missing = _native_scope(checkpoint, next(query for query in checkpoint.queries if query.query_id == record["query_id"]))
    allowed_ids = {source.provenance_id for source in native_allowed}
    if set(record.get("G", {}).get("native_evidence_scope", ())) != allowed_ids or record.get("G", {}).get("native_evidence_missing_ids", []) != sorted(missing):
        errors.append("G_native_scope")
    for evidence in record.get("G", {}).get("E_plus", ()):
        source = sources.get(evidence.get("provenance_id"))
        if source is None or evidence.get("provenance_id") not in allowed_ids or source.ordinal != evidence.get("source_ordinal") or source.text[evidence.get("char_start", -1):evidence.get("char_end", -1)] != evidence.get("quote") or source.text.count(evidence.get("quote", "")) != 1:
            errors.append("G_provenance_span")
    return sorted(set(errors))


def run_shard(
    root: Path,
    shard: int,
    selector: Selector,
    generator: Generator,
    *,
    resume: bool,
    locomo: Path = DEFAULT_LOCOMO,
    longmemeval: Path = DEFAULT_LONGMEMEVAL,
    stop_after: int | None = None,
) -> dict[str, Any]:
    plan = load_plan(root)
    if shard < 0 or shard >= plan["num_shards"]:
        raise ValueError("shard out of range")
    expected = {item["query_id"] for item in plan["queries"] if item["shard"] == shard}
    completed = 0
    skipped = 0
    for checkpoint in _iter_checkpoints(locomo, longmemeval):
        for query in checkpoint.queries:
            if query.query_id not in expected:
                continue
            path = query_record_path(root, shard, query.query_id)
            if path.exists():
                if not resume:
                    raise RuntimeError(f"terminal query already exists: {query.query_id}")
                existing = json.loads(path.read_text())
                if existing.get("query_id") != query.query_id or _validate_terminal(existing, checkpoint):
                    raise RuntimeError(f"invalid existing terminal query: {query.query_id}")
                skipped += 1
                continue
            terminal = process_query(checkpoint, query, selector, generator)
            errors = _validate_terminal(terminal, checkpoint)
            if errors:
                raise RuntimeError(f"terminal mechanical validation failed for {query.query_id}: {errors}")
            write_canonical_json(path, terminal)
            completed += 1
            if stop_after is not None and completed >= stop_after:
                break
        if stop_after is not None and completed >= stop_after:
            break
    status = {
        "shard": shard, "expected": len(expected), "completed_this_invocation": completed,
        "preexisting_terminal": skipped,
        "terminal_total": sum(query_record_path(root, shard, query_id).exists() for query_id in expected),
    }
    write_canonical_json(root / "shards" / f"shard-{shard:04d}" / "status.json", status)
    return status


def validate_root(
    root: Path,
    locomo: Path = DEFAULT_LOCOMO,
    longmemeval: Path = DEFAULT_LONGMEMEVAL,
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    plan = load_plan(root)
    expected = {item["query_id"]: item for item in plan["queries"]}
    checkpoints: dict[tuple[str, str], BenchmarkCheckpoint] = {}
    for checkpoint in _iter_checkpoints(locomo, longmemeval):
        if any(query.query_id in expected for query in checkpoint.queries):
            checkpoints[(checkpoint.benchmark, checkpoint.checkpoint_id)] = checkpoint
    seen: dict[str, Path] = {}
    errors: list[dict[str, Any]] = []
    for path in sorted((root / "shards").glob("shard-*/queries/*.json")):
        record = json.loads(path.read_text())
        query_id = record.get("query_id")
        if query_id not in expected:
            errors.append({"path": str(path), "error": "extra_query"})
            continue
        if query_id in seen:
            errors.append({"path": str(path), "error": "duplicate_query", "first": str(seen[query_id])})
            continue
        seen[query_id] = path
        planned_shard = expected[query_id]["shard"]
        actual_shard = int(path.parents[1].name.removeprefix("shard-"))
        if planned_shard != actual_shard:
            errors.append({"path": str(path), "error": "wrong_shard"})
        checkpoint = checkpoints[(record.get("benchmark"), record.get("checkpoint_id"))]
        for error in _validate_terminal(record, checkpoint):
            errors.append({"path": str(path), "error": error})
    missing = sorted(set(expected) - seen.keys())
    if require_complete and missing:
        errors.append({"error": "missing_queries", "count": len(missing), "sample": missing[:20]})
    report = {
        "preprocessing_contract": PG_PREPROCESS_V1,
        "contract_digest": CONTRACT_DIGEST,
        "semantic_validation_status": SEMANTIC_VALIDATION_STATUS,
        "expected_queries": len(expected), "terminal_queries": len(seen),
        "missing_queries": len(missing), "extra_or_invalid": len(errors),
        "errors": errors, "valid": not errors,
    }
    write_canonical_json(root / "manifests" / "mechanical-validation-report.json", report)
    return report


def merge_root(
    root: Path,
    locomo: Path = DEFAULT_LOCOMO,
    longmemeval: Path = DEFAULT_LONGMEMEVAL,
) -> dict[str, Any]:
    validation = validate_root(root, locomo, longmemeval, require_complete=True)
    if not validation["valid"]:
        raise RuntimeError("merge refused because mechanical validation failed")
    plan = load_plan(root)
    p_lines: list[bytes] = []
    g_lines: list[bytes] = []
    unresolved: Counter[str] = Counter()
    representability: dict[str, Counter[str]] = {}
    for item in sorted(plan["queries"], key=lambda value: value["query_id"]):
        record = json.loads(query_record_path(root, item["shard"], item["query_id"]).read_text())
        p_lines.append(canonical_bytes(record["P"]) + b"\n")
        g_lines.append(canonical_bytes(record["G"]) + b"\n")
        group_names = ("ALL", record["benchmark"], f'{record["benchmark"]}/{record["native_type"]}')
        for group in group_names:
            counts = representability.setdefault(group, Counter())
            counts["queries"] += 1
            counts[f'P_{record["P"]["status"]}'] += 1
            counts[f'G_{record["G"]["status"]}'] += 1
            opportunities = record["P"]["opportunities"]
            for name, value in opportunities.items():
                counts[name] += bool(value)
        if record["P"]["status"] == "UNRESOLVED":
            unresolved[f'P:{record["P"]["unresolved_reason"]}'] += 1
        if record["G"]["status"] == "UNRESOLVED":
            unresolved[f'G:{record["G"]["unresolved_reason"]}'] += 1
    p_path = root / "merged" / "P-search-safe.jsonl"
    g_path = root / "merged" / "G-evaluator-only.jsonl"
    atomic_write(p_path, b"".join(p_lines))
    atomic_write(g_path, b"".join(g_lines))
    unresolved_path = root / "merged" / "unresolved-summary.json"
    write_canonical_json(unresolved_path, dict(sorted(unresolved.items())))
    validation_path = root / "merged" / "mechanical-validation-report.json"
    write_canonical_json(validation_path, validation)
    artifact_digests = {
        "P-search-safe.jsonl": sha256(p_path.read_bytes()).hexdigest(),
        "G-evaluator-only.jsonl": sha256(g_path.read_bytes()).hexdigest(),
        "unresolved-summary.json": sha256(unresolved_path.read_bytes()).hexdigest(),
        "mechanical-validation-report.json": sha256(validation_path.read_bytes()).hexdigest(),
    }
    manifest = {
        "preprocessing_contract": PG_PREPROCESS_V1,
        "contract_digest": CONTRACT_DIGEST,
        "semantic_validation_status": SEMANTIC_VALIDATION_STATUS,
        "dataset_artifacts": plan["dataset_artifacts"],
        "query_counts": dict(Counter(item["benchmark"] for item in plan["queries"])),
        "total_queries": len(plan["queries"]),
        "num_shards": plan["num_shards"],
        "query_manifest_sha256": plan["query_manifest_sha256"],
        "artifact_sha256": artifact_digests,
        "representability": {key: dict(value) for key, value in sorted(representability.items())},
    }
    manifest_path = root / "merged" / "preprocessing-run-manifest.json"
    write_canonical_json(manifest_path, manifest)
    checksums = {
        **artifact_digests,
        "preprocessing-run-manifest.json": sha256(manifest_path.read_bytes()).hexdigest(),
    }
    checksum_text = "".join(f"{digest}  {name}\n" for name, digest in sorted(checksums.items()))
    atomic_write(root / "merged" / "SHA256SUMS", checksum_text.encode("ascii"))
    return {**manifest, "canonical_artifact_sha256": checksums}


def _load_selected(path: Path | None) -> frozenset[str] | None:
    if path is None:
        return None
    value = json.loads(path.read_text())
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("query ID selection must be a JSON array of strings")
    return frozenset(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", type=Path, required=True)
    common.add_argument("--locomo", type=Path, default=DEFAULT_LOCOMO)
    common.add_argument("--longmemeval", type=Path, default=DEFAULT_LONGMEMEVAL)
    plan_parser = sub.add_parser("plan", parents=[common])
    plan_parser.add_argument("--num-shards", type=int, required=True)
    plan_parser.add_argument("--query-ids-file", type=Path)
    for mode in ("run", "resume"):
        run_parser = sub.add_parser(mode, parents=[common])
        run_parser.add_argument("--shard", type=int, required=True)
        run_parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
        run_parser.add_argument("--ollama-url", default="http://127.0.0.1:11500/api/embed")
        run_parser.add_argument("--ollama-manifest", type=Path, default=DEFAULT_OLLAMA_MANIFEST)
        run_parser.add_argument("--stop-after", type=int)
    sub.add_parser("merge", parents=[common])
    validate_parser = sub.add_parser("validate", parents=[common])
    validate_parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "plan":
        result = create_plan(args.root, args.num_shards, args.locomo, args.longmemeval, _load_selected(args.query_ids_file))
    elif args.mode in {"run", "resume"}:
        result = run_shard(
            args.root, args.shard,
            NomicSelector(args.ollama_url, args.ollama_manifest),
            LocalQwenGenerator(args.model_path),
            resume=args.mode == "resume", locomo=args.locomo,
            longmemeval=args.longmemeval, stop_after=args.stop_after,
        )
    elif args.mode == "merge":
        result = merge_root(args.root, args.locomo, args.longmemeval)
    else:
        result = validate_root(
            args.root, args.locomo, args.longmemeval,
            require_complete=not args.allow_incomplete,
        )
    if args.mode == "plan":
        result = {
            "preprocessing_contract": result["preprocessing_contract"],
            "contract_digest": result["contract_digest"],
            "semantic_validation_status": result["semantic_validation_status"],
            "num_shards": result["num_shards"],
            "planned_queries": len(result["queries"]),
            "query_manifest_sha256": result["query_manifest_sha256"],
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
