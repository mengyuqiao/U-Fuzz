from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from ufuzz.pg_annotation import (
    AnnotationRecord,
    cohens_kappa,
    deterministic_record_order,
    load_adjudication,
    load_labels,
    raw_agreement,
    record_manifest,
    score_annotations,
    validate_annotation_information_flow,
    write_annotation_package,
)
from ufuzz.pg_preprocess import PG_PREPROCESS_V2


def _record(index: int, dimension: str, section: str = "SEARCH_SAFE") -> AnnotationRecord:
    return AnnotationRecord(
        f"record:{index}:{dimension}", f"query:{index}", "locomo", "1", section,
        dimension, f"candidate:{index}", {"raw_question": f"Question {index}?"},
    )


class AnnotationPackageTests(unittest.TestCase):
    def test_order_and_manifest_are_stable(self) -> None:
        records = (
            _record(1, "query_slot_validity"),
            _record(2, "target_changing_validity"),
            _record(3, "g_evidence_region_support", "EVALUATOR_ONLY"),
        )
        first = deterministic_record_order(records)
        second = deterministic_record_order(reversed(records))
        self.assertEqual(first, second)
        self.assertEqual(record_manifest(first), record_manifest(second))

    def test_information_flow_rejects_gold_in_search_safe_evidence(self) -> None:
        leaked = AnnotationRecord(
            "leak", "query", "locomo", "1", "SEARCH_SAFE",
            "query_slot_validity", "slot", {"gold_answer": "secret"},
        )
        with self.assertRaisesRegex(ValueError, "leaks evaluator fields"):
            validate_annotation_information_flow((leaked,))

    def test_package_templates_have_no_prefilled_human_labels(self) -> None:
        records = (_record(1, "query_slot_validity"),)
        with tempfile.TemporaryDirectory() as directory:
            digests = write_annotation_package(directory, records, "# Guideline")
            self.assertIn("annotator_A_labels.csv", digests)
            with (Path(directory) / "annotator_A_labels.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["annotator_label"], "")
            manifest = json.loads(
                (Path(directory) / "deterministic_record_manifest.json").read_text()
            )
            self.assertEqual(manifest["record_count"], 1)

    def test_v2_package_binds_v2_identity_without_labels(self) -> None:
        records = (_record(1, "query_slot_validity"),)
        with tempfile.TemporaryDirectory() as directory:
            write_annotation_package(
                directory, records, "# V2 Guideline", candidate_label=PG_PREPROCESS_V2
            )
            root = Path(directory)
            record_manifest_value = json.loads(
                (root / "deterministic_record_manifest.json").read_text()
            )
            candidate = json.loads((root / "frozen_candidate_manifest.json").read_text())
            self.assertEqual(record_manifest_value["candidate"], PG_PREPROCESS_V2)
            self.assertEqual(candidate["contract_label"], PG_PREPROCESS_V2)
            with (root / "annotator_A_labels.csv").open(newline="") as stream:
                self.assertEqual(next(csv.DictReader(stream))["annotator_label"], "")

    def test_label_validation_rejects_missing_duplicate_and_changed_rows(self) -> None:
        records = (_record(1, "query_slot_validity"),)
        manifest = record_manifest(records)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            self._write_labels(path, manifest, {records[0].record_id: "CORRECT"})
            self.assertEqual(load_labels(path, manifest)[records[0].record_id], "CORRECT")
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            rows[0]["query_id"] = "changed"
            self._write_raw(path, rows)
            with self.assertRaisesRegex(ValueError, "immutable annotation field changed"):
                load_labels(path, manifest)

    def test_known_agreement_and_kappa_vectors(self) -> None:
        left = ["A", "A", "B", "B"]
        right = ["A", "B", "B", "B"]
        self.assertEqual(raw_agreement(left, right), 0.75)
        self.assertAlmostEqual(cohens_kappa(left, right), 0.5)
        self.assertIsNone(cohens_kappa(["A", "A"], ["A", "A"]))

    def test_disagreements_and_adjudication_completeness(self) -> None:
        dimensions = (
            "query_slot_validity", "target_changing_validity",
            "unsupported_slot_validity", "update_validity", "delete_validity",
            "urc_existing_proposition", "urc_unrelatedness",
            "g_gold_reference_correctness", "g_support_set_correctness",
            "g_evidence_region_support",
        )
        records = tuple(
            _record(i, dimension, "EVALUATOR_ONLY" if dimension.startswith("g_") else "SEARCH_SAFE")
            for i, dimension in enumerate(dimensions)
        )
        manifest = record_manifest(records)
        a = {}
        b = {}
        for row in manifest["records"]:
            dimension = row["annotation_dimension"]
            positive = {
                "query_slot_validity": "CORRECT", "target_changing_validity": "VALID",
                "unsupported_slot_validity": "VALID", "update_validity": "VALID",
                "delete_validity": "VALID", "urc_existing_proposition": "YES",
                "urc_unrelatedness": "YES", "g_gold_reference_correctness": "YES",
                "g_support_set_correctness": "YES", "g_evidence_region_support": "SUPPORTS",
            }[dimension]
            a[row["record_id"]] = positive
            b[row["record_id"]] = positive
        changed = manifest["records"][0]
        b[changed["record_id"]] = {
            "query_slot_validity": "INCORRECT", "target_changing_validity": "INVALID",
            "unsupported_slot_validity": "INVALID", "update_validity": "INVALID",
            "delete_validity": "INVALID", "urc_existing_proposition": "NO",
            "urc_unrelatedness": "NO", "g_gold_reference_correctness": "NO",
            "g_support_set_correctness": "NO", "g_evidence_region_support": "DOES_NOT_SUPPORT",
        }[changed["annotation_dimension"]]
        initial = score_annotations(manifest, a, b)
        self.assertEqual(initial["disagreement_count"], 1)
        with self.assertRaisesRegex(ValueError, "exactly disagreements"):
            score_annotations(manifest, a, b, {})
        final = score_annotations(
            manifest, a, b, {changed["record_id"]: a[changed["record_id"]]},
            mechanical_violations={"provenance": 0, "flow": 0, "absence": 0},
            systematic_error_status="NO_SYSTEMATIC_ERROR",
        )
        self.assertTrue(final["adjudication_complete"])
        self.assertEqual(final["precision"]["target_changing"], 1.0)
        self.assertFalse(final["accuracy_gate_pass"])  # the deliberately weak IAA blocks a pass

    @staticmethod
    def _write_labels(path: Path, manifest: dict, labels: dict[str, str]) -> None:
        rows = []
        for row in manifest["records"]:
            rows.append({
                "record_id": row["record_id"], "query_id": row["query_id"],
                "benchmark": row["benchmark"], "annotation_dimension": row["annotation_dimension"],
                "candidate_id": row["candidate_id"], "allowed_labels": "|".join(row["allowed_labels"]),
                "annotator_label": labels.get(row["record_id"], ""), "reason_code": "",
                "short_rationale": "",
            })
        AnnotationPackageTests._write_raw(path, rows)

    @staticmethod
    def _write_raw(path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
