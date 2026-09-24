from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ufuzz.benchmarks import LoCoMoLoader
from ufuzz.pg_selector_snapshot import (
    FrozenSelectorSnapshot,
    canonical_source_ids,
    validate_snapshot_records,
)
from ufuzz.semantic_sidecar import canonical_bytes


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "locomo_minimal.json"


class SelectorSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.checkpoint = next(LoCoMoLoader().load(FIXTURE, verify_artifact=False))
        self.query = self.checkpoint.queries[0]

    def _record(self):
        sources = list(self.checkpoint.sources)
        return {
            "query_id": self.query.query_id,
            "checkpoint_id": self.checkpoint.checkpoint_id,
            "benchmark": self.checkpoint.benchmark,
            "relevant_candidate_provenance_ids": canonical_source_ids(reversed(sources)),
            "unrelated_candidate_provenance_ids": [],
        }

    def test_schema_uniqueness_binding_and_canonical_presentation(self):
        record = self._record()
        expected = {"locomo": 1}
        with patch("ufuzz.pg_selector_snapshot.EXPECTED_COUNTS", expected):
            report = validate_snapshot_records([record], [self.checkpoint])
            self.assertTrue(report["valid"])
            duplicate = validate_snapshot_records([record, record], [self.checkpoint])
            self.assertFalse(duplicate["valid"])
            self.assertTrue(any(row["error"] == "duplicate_query" for row in duplicate["errors"]))

    def test_relevant_and_outside_must_be_disjoint(self):
        record = self._record()
        record["unrelated_candidate_provenance_ids"] = record["relevant_candidate_provenance_ids"][:1]
        with patch("ufuzz.pg_selector_snapshot.EXPECTED_COUNTS", {"locomo": 1}):
            report = validate_snapshot_records([record], [self.checkpoint])
        self.assertFalse(report["valid"])
        self.assertTrue(any(row["error"] == "candidate_overlap" for row in report["errors"]))

    def test_frozen_snapshot_digest_and_query_binding(self):
        record = self._record()
        raw = canonical_bytes(record) + b"\n"
        digest = sha256(raw).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selector-snapshot.jsonl"
            path.write_bytes(raw)
            selector = FrozenSelectorSnapshot.load(path, digest)
            selected = selector.select(self.checkpoint, self.query)
            self.assertEqual(
                selected["relevant_candidate_provenance_ids"],
                record["relevant_candidate_provenance_ids"],
            )
            self.assertEqual(selected["selector_identity"]["selector_snapshot_sha256"], digest)
            with self.assertRaisesRegex(RuntimeError, "digest mismatch"):
                FrozenSelectorSnapshot.load(path, "0" * 64)

    def test_snapshot_record_contains_no_gold_or_evidence_fields(self):
        serialized = json.dumps(self._record()).casefold()
        self.assertNotIn("gold", serialized)
        self.assertNotIn("evidence", serialized)


if __name__ == "__main__":
    unittest.main()
