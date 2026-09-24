from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ufuzz.benchmarks import LoCoMoLoader
from ufuzz.pg_generate import (
    CONTRACT_DIGEST,
    SEMANTIC_VALIDATION_STATUS,
    atomic_write,
    canonical_source_presentation,
    _canonicalize_g_options,
    load_plan,
    merge_root,
    process_query,
    query_record_path,
    run_shard,
    shard_for_query,
    validate_root,
    write_canonical_json,
)
from ufuzz.pg_preprocess import PG_PREPROCESS_V2, pg_preprocess_v2_manifest
from ufuzz.semantic_sidecar import canonical_bytes


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "locomo_minimal.json"


class FakeSelector:
    def select(self, checkpoint, query):
        return {
            "relevant_candidate_provenance_ids": [source.provenance_id for source in checkpoint.sources],
            "unrelated_candidate_provenance_ids": [],
            "synthetic_unsupported_target": "ufuzz-session-local-absent-fixture",
            "synthetic_exact_absent": True,
            "selector_identity": {
                "artifact_label": "PG_SELECTOR_SNAPSHOT_V2",
                "selector_snapshot_sha256": pg_preprocess_v2_manifest().selector_snapshot_sha256,
            },
        }


class FakeGenerator:
    def __init__(self):
        self.calls = []

    def call(self, task_id, query_id, kind, payload):
        if kind == "query_slot":
            parsed = {
                "status": "RESOLVED", "entity_description": "Alice",
                "entity_status": "RESOLVED", "relation_label": "residence",
                "anchor_kind": "EXPLICIT", "query_anchor": "live",
                "structural_explanation": None, "temporal_scope": "unspecified",
                "constraints": [], "replaceable_slots": ["entity", "relation"],
                "primary_reason": "none",
            }
        elif kind == "bounded_proposition_extraction":
            source = payload["relevant_pool"][0]
            parsed = {
                "relevant_propositions": [{
                    "provenance_id": source["provenance_id"],
                    "quote": "Alice residence Boston and Bob hobby tennis.",
                    "entity": "Alice", "relation": "residence",
                    "value": "Boston", "temporal": "unspecified",
                }],
                "target_changing": [], "unrelated_propositions": [],
            }
        elif kind == "existing_memory_state":
            parsed = {"status": "ELIGIBLE", "primary_reason": "explicit_mutable_memory_state"}
        elif kind == "relation_entity_match":
            parsed = {"relation_match": "MATCH", "entity_match": "MATCH", "temporal_compatibility": "COMPATIBLE"}
        elif kind == "g_evidence_mapping":
            source = payload["native_evidence_chunk"][0]
            parsed = {
                "status": "RESOLVED",
                "accepted_options": [{"required_components": [{"component_id": "place", "value": "Boston"}]}],
                "supports": [{
                    "component_id": "place", "provenance_id": source["provenance_id"],
                    "quote": "Alice residence Boston", "independent_group": "place",
                }],
                "reason": None,
            }
        else:
            raise AssertionError(kind)
        result = {
            "task_id": task_id, "query_id": query_id, "call_kind": kind,
            "canonical_input_sha256": "0" * 64, "prompt_sha256": "1" * 64,
            "model_id": "test", "model_revision": "test", "input_token_count": 1,
            "output_token_count": 1, "raw_output_sha256": "2" * 64,
            "parsed_canonical_output": parsed, "parse_error": None, "wall_seconds": 0.01,
        }
        self.calls.append(result)
        return result


def _checkpoint_pair():
    checkpoint = next(LoCoMoLoader().load(FIXTURE, verify_artifact=False))
    first = checkpoint.queries[0]
    second = replace(first, query_id=first.query_id + ":second")
    return replace(checkpoint, queries=(first, second))


def _write_plan(root: Path, checkpoint, num_shards: int = 1):
    queries = [
        {
            "query_id": query.query_id, "checkpoint_id": checkpoint.checkpoint_id,
            "benchmark": checkpoint.benchmark, "native_type": str(query.query_type),
            "shard": shard_for_query(query.query_id, num_shards),
        }
        for query in checkpoint.queries
    ]
    queries.sort(key=lambda row: row["query_id"])
    plan = {
        "preprocessing_contract": PG_PREPROCESS_V2,
        "contract_digest": CONTRACT_DIGEST,
        "executable_contract_sha256": pg_preprocess_v2_manifest().executable_contract_sha256,
        "selector_snapshot_sha256": pg_preprocess_v2_manifest().selector_snapshot_sha256,
        "semantic_validation_status": SEMANTIC_VALIDATION_STATUS,
        "num_shards": num_shards, "dataset_artifacts": {},
        "full_dataset_query_counts": {}, "planned_query_count": len(queries),
        "shard_rule": "int(first_16_hex(sha256(query_id)), 16) % num_shards",
        "queries": queries,
    }
    from hashlib import sha256
    plan["query_manifest_sha256"] = sha256(canonical_bytes(queries)).hexdigest()
    write_canonical_json(root / "plan" / "query-shards.json", plan)
    return plan


class PGGenerateTests(unittest.TestCase):
    def test_canonical_g_option_merge_ignores_generation_order(self):
        left = [{"required_components": [
            {"component_id": "b", "value": " two "},
            {"component_id": "a", "value": "one"},
        ]}]
        right = [{"required_components": list(reversed(left[0]["required_components"]))}]
        first, issues = _canonicalize_g_options(left)
        second, other_issues = _canonicalize_g_options(right)
        self.assertEqual(first, second)
        self.assertEqual(issues, [])
        self.assertEqual(other_issues, [])
        self.assertEqual([item["component_id"] for item in first[0]["required_components"]], ["a", "b"])

    def test_selected_membership_has_checkpoint_native_presentation(self):
        checkpoint = _checkpoint_pair()
        sources = list(checkpoint.sources)
        presented = canonical_source_presentation(reversed(sources))
        self.assertEqual(
            [source.provenance_id for source in presented],
            [source.provenance_id for source in sorted(sources, key=lambda source: (source.ordinal, source.provenance_id))],
        )

    def test_sharding_is_stable_and_machine_independent(self):
        self.assertEqual(shard_for_query("query:one", 7), shard_for_query("query:one", 7))
        self.assertEqual(shard_for_query("query:one", 7), 2)
        with self.assertRaises(ValueError):
            shard_for_query("query", 0)

    def test_query_record_is_separated_and_mechanically_grounded(self):
        checkpoint = _checkpoint_pair()
        generator = FakeGenerator()
        record = process_query(checkpoint, checkpoint.queries[0], FakeSelector(), generator)
        self.assertEqual(record["P"]["status"], "RESOLVED")
        self.assertEqual(record["G"]["status"], "RESOLVED")
        self.assertNotIn("gold_answer", json.dumps(record["P"]))
        self.assertEqual(record["P"]["opportunities"]["update"]["value"], "Boston")
        self.assertEqual(
            record["executable_contract_sha256"],
            pg_preprocess_v2_manifest().executable_contract_sha256,
        )
        call_ids = [call["task_id"] for call in generator.calls]
        self.assertEqual(len(call_ids), len(set(call_ids)))
        self.assertEqual(call_ids, [call["task_id"] for call in record["raw_calls"]])

    def test_atomic_resume_skips_committed_terminal_query(self):
        checkpoint = _checkpoint_pair()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_plan(root, checkpoint)
            with patch("ufuzz.pg_generate._iter_checkpoints", return_value=iter((checkpoint,))):
                first = run_shard(root, 0, FakeSelector(), FakeGenerator(), resume=False, stop_after=1)
            self.assertEqual(first["terminal_total"], 1)
            with patch("ufuzz.pg_generate._iter_checkpoints", return_value=iter((checkpoint,))):
                resumed = run_shard(root, 0, FakeSelector(), FakeGenerator(), resume=True)
            self.assertEqual(resumed["preexisting_terminal"], 1)
            self.assertEqual(resumed["terminal_total"], 2)

    def test_run_rejects_duplicate_existing_terminal(self):
        checkpoint = _checkpoint_pair()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_plan(root, checkpoint)
            with patch("ufuzz.pg_generate._iter_checkpoints", return_value=iter((checkpoint,))):
                run_shard(root, 0, FakeSelector(), FakeGenerator(), resume=False, stop_after=1)
            with patch("ufuzz.pg_generate._iter_checkpoints", return_value=iter((checkpoint,))):
                with self.assertRaisesRegex(RuntimeError, "already exists"):
                    run_shard(root, 0, FakeSelector(), FakeGenerator(), resume=False)

    def test_validation_and_merge_reject_missing_and_duplicate_queries(self):
        checkpoint = _checkpoint_pair()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = _write_plan(root, checkpoint, num_shards=2)
            with patch("ufuzz.pg_generate._iter_checkpoints", return_value=iter((checkpoint,))):
                report = validate_root(root, require_complete=True)
            self.assertFalse(report["valid"])
            self.assertEqual(report["missing_queries"], 2)
            query = checkpoint.queries[0]
            record = process_query(checkpoint, query, FakeSelector(), FakeGenerator())
            planned = next(item for item in plan["queries"] if item["query_id"] == query.query_id)
            write_canonical_json(query_record_path(root, planned["shard"], query.query_id), record)
            wrong = 1 - planned["shard"]
            write_canonical_json(query_record_path(root, wrong, query.query_id), record)
            with patch("ufuzz.pg_generate._iter_checkpoints", return_value=iter((checkpoint,))):
                report = validate_root(root, require_complete=False)
            self.assertTrue(any(error["error"] == "duplicate_query" for error in report["errors"]))
            with patch("ufuzz.pg_generate._iter_checkpoints", return_value=iter((checkpoint,))):
                with self.assertRaisesRegex(RuntimeError, "merge refused"):
                    merge_root(root)

    def test_contract_digest_mismatch_fails_closed(self):
        checkpoint = _checkpoint_pair()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_plan(root, checkpoint)
            path = root / "plan" / "query-shards.json"
            plan = json.loads(path.read_text())
            plan["contract_digest"] = "0" * 64
            write_canonical_json(path, plan)
            with self.assertRaisesRegex(RuntimeError, "contract digest"):
                load_plan(root)

    def test_selector_snapshot_digest_mismatch_fails_closed(self):
        checkpoint = _checkpoint_pair()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_plan(root, checkpoint)
            path = root / "plan" / "query-shards.json"
            plan = json.loads(path.read_text())
            plan["selector_snapshot_sha256"] = "0" * 64
            write_canonical_json(path, plan)
            with self.assertRaisesRegex(RuntimeError, "selector-snapshot"):
                load_plan(root)

    def test_dataset_hash_mismatch_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            corrupt = Path(directory) / "locomo.json"
            atomic_write(corrupt, b"[]")
            with self.assertRaisesRegex(ValueError, "artifact size mismatch"):
                LoCoMoLoader().artifact.verify(corrupt)

    def test_information_flow_and_provenance_failures_stop_validation(self):
        checkpoint = _checkpoint_pair()
        query = checkpoint.queries[0]
        record = process_query(checkpoint, query, FakeSelector(), FakeGenerator())
        record["P"]["gold_answer"] = "leak"
        record["P"]["opportunities"]["update"]["char_start"] = 3
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = _write_plan(root, replace(checkpoint, queries=(query,)))
            write_canonical_json(query_record_path(root, plan["queries"][0]["shard"], query.query_id), record)
            with patch("ufuzz.pg_generate._iter_checkpoints", return_value=iter((replace(checkpoint, queries=(query,)),))):
                report = validate_root(root)
            kinds = {error["error"] for error in report["errors"]}
            self.assertIn("information_flow", kinds)
            self.assertIn("P_provenance_span", kinds)

    def test_complete_merge_digest_is_stable(self):
        checkpoint = _checkpoint_pair()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_plan(root, checkpoint)
            with patch("ufuzz.pg_generate._iter_checkpoints", return_value=iter((checkpoint,))):
                run_shard(root, 0, FakeSelector(), FakeGenerator(), resume=False)
            with patch("ufuzz.pg_generate._iter_checkpoints", return_value=iter((checkpoint,))):
                first = merge_root(root)
            p_digest = first["artifact_sha256"]["P-search-safe.jsonl"]
            with patch("ufuzz.pg_generate._iter_checkpoints", return_value=iter((checkpoint,))):
                second = merge_root(root)
            self.assertEqual(p_digest, second["artifact_sha256"]["P-search-safe.jsonl"])
            checksums = (root / "merged" / "SHA256SUMS").read_text().splitlines()
            self.assertEqual(len(checksums), 5)
            self.assertTrue(any(line.endswith("  preprocessing-run-manifest.json") for line in checksums))
            for line in checksums:
                digest, name = line.split("  ", 1)
                from hashlib import sha256
                self.assertEqual(sha256((root / "merged" / name).read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
