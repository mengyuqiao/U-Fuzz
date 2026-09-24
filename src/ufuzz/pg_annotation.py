"""Human annotation package and scoring for frozen P/G candidates.

The scorer is deterministic and contains no model invocation.  Candidate
evidence is immutable; annotators fill only the label, reason, and rationale
columns in copies of the supplied templates.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ufuzz.pg_preprocess import (
    PG_PREPROCESS_V1,
    PG_PREPROCESS_V2,
    pg_preprocess_v1_manifest,
    pg_preprocess_v2_manifest,
)
from ufuzz.semantic_sidecar import canonical_bytes


ANNOTATION_ORDER_SEED = "PG_PREPROCESS_V1/HUMAN_REVIEW/1729"
CSV_FIELDS = (
    "record_id", "query_id", "benchmark", "annotation_dimension",
    "candidate_id", "allowed_labels", "annotator_label", "reason_code",
    "short_rationale",
)


@dataclass(frozen=True, slots=True)
class DimensionSpec:
    labels: tuple[str, ...]
    positive: tuple[str, ...]
    negative: tuple[str, ...]
    unresolved: tuple[str, ...]
    primary: bool = True


DIMENSIONS: dict[str, DimensionSpec] = {
    "query_slot_validity": DimensionSpec(
        ("CORRECT", "INCORRECT", "UNRESOLVED"), ("CORRECT",), ("INCORRECT",), ("UNRESOLVED",)
    ),
    "query_slot_unresolved_audit": DimensionSpec(
        ("JUSTIFIED_UNRESOLVED", "SHOULD_BE_RESOLVABLE", "HUMAN_UNCERTAIN"),
        ("JUSTIFIED_UNRESOLVED",), ("SHOULD_BE_RESOLVABLE",), ("HUMAN_UNCERTAIN",), False,
    ),
    "target_changing_validity": DimensionSpec(
        ("VALID", "INVALID", "UNRESOLVED"), ("VALID",), ("INVALID",), ("UNRESOLVED",)
    ),
    "unsupported_slot_validity": DimensionSpec(
        ("VALID", "INVALID", "UNRESOLVED"), ("VALID",), ("INVALID",), ("UNRESOLVED",)
    ),
    "update_validity": DimensionSpec(
        ("VALID", "INVALID", "UNRESOLVED"), ("VALID",), ("INVALID",), ("UNRESOLVED",)
    ),
    "delete_validity": DimensionSpec(
        ("VALID", "INVALID", "UNRESOLVED"), ("VALID",), ("INVALID",), ("UNRESOLVED",)
    ),
    "urc_existing_proposition": DimensionSpec(
        ("YES", "NO", "UNRESOLVED"), ("YES",), ("NO",), ("UNRESOLVED",)
    ),
    "urc_unrelatedness": DimensionSpec(
        ("YES", "NO", "UNRESOLVED"), ("YES",), ("NO",), ("UNRESOLVED",)
    ),
    "g_gold_reference_correctness": DimensionSpec(
        ("YES", "NO", "UNRESOLVED"), ("YES",), ("NO",), ("UNRESOLVED",)
    ),
    "g_support_set_correctness": DimensionSpec(
        ("YES", "NO", "UNRESOLVED"), ("YES",), ("NO",), ("UNRESOLVED",)
    ),
    "g_evidence_region_support": DimensionSpec(
        ("SUPPORTS", "DOES_NOT_SUPPORT", "UNRESOLVED"),
        ("SUPPORTS",), ("DOES_NOT_SUPPORT",), ("UNRESOLVED",),
    ),
    "g_unresolved_audit": DimensionSpec(
        ("JUSTIFIED_UNRESOLVED", "SHOULD_BE_RESOLVABLE", "HUMAN_UNCERTAIN"),
        ("JUSTIFIED_UNRESOLVED",), ("SHOULD_BE_RESOLVABLE",), ("HUMAN_UNCERTAIN",), False,
    ),
}


@dataclass(frozen=True, slots=True)
class AnnotationRecord:
    record_id: str
    query_id: str
    benchmark: str
    native_type: str
    section: str
    annotation_dimension: str
    candidate_id: str
    evidence: Mapping[str, Any]
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.annotation_dimension not in DIMENSIONS:
            raise ValueError(f"unknown annotation dimension {self.annotation_dimension}")
        if self.section not in {"SEARCH_SAFE", "EVALUATOR_ONLY"}:
            raise ValueError("annotation section must enforce the information boundary")

    @property
    def allowed_labels(self) -> tuple[str, ...]:
        return DIMENSIONS[self.annotation_dimension].labels


def deterministic_record_order(
    records: Iterable[AnnotationRecord], order_seed: str = ANNOTATION_ORDER_SEED
) -> tuple[AnnotationRecord, ...]:
    def key(record: AnnotationRecord) -> tuple[str, str]:
        digest = sha256((order_seed + "\0" + record.record_id).encode()).hexdigest()
        return digest, record.record_id
    values = tuple(sorted(records, key=key))
    ids = [record.record_id for record in values]
    if len(ids) != len(set(ids)):
        raise ValueError("annotation record IDs must be unique")
    return values


def validate_annotation_information_flow(records: Iterable[AnnotationRecord]) -> None:
    forbidden = {
        "gold_answer", "gold_answers", "accepted_answer", "accepted_answers",
        "accepted_answer_options", "native_evidence", "native_evidence_scope",
        "answer_session_ids", "has_answer", "e_plus", "e_minus", "proposed_e_plus",
        "g_status", "pipeline_g_status", "correctness", "failure_surface", "cfs",
        "uf", "cov", "backend_result", "backend_response",
    }
    for record in records:
        if record.section != "SEARCH_SAFE":
            continue
        keys = {key.casefold() for key in _walk_mapping_keys(record.evidence)}
        leaked = sorted(keys & forbidden)
        if leaked:
            raise ValueError(
                f"search-safe annotation record {record.record_id} leaks evaluator fields: {leaked}"
            )


def record_manifest(
    records: Sequence[AnnotationRecord], *, candidate_label: str = PG_PREPROCESS_V1,
    order_seed: str = ANNOTATION_ORDER_SEED,
) -> dict[str, Any]:
    ordered = deterministic_record_order(records, order_seed)
    entries = [
        {
            "record_id": item.record_id,
            "query_id": item.query_id,
            "benchmark": item.benchmark,
            "native_type": item.native_type,
            "section": item.section,
            "annotation_dimension": item.annotation_dimension,
            "candidate_id": item.candidate_id,
            "allowed_labels": item.allowed_labels,
            "evidence_sha256": sha256(canonical_bytes(item.evidence)).hexdigest(),
        }
        for item in ordered
    ]
    return {
        "candidate": candidate_label,
        "ordering_seed": order_seed,
        "record_count": len(entries),
        "records": entries,
    }


def write_annotation_package(
    directory: str | Path,
    records: Sequence[AnnotationRecord],
    guideline: str,
    *,
    candidate_label: str = PG_PREPROCESS_V1,
) -> dict[str, str]:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    order_seed = f"{candidate_label}/HUMAN_REVIEW/1729"
    ordered = deterministic_record_order(records, order_seed)
    validate_annotation_information_flow(ordered)
    manifest = record_manifest(ordered, candidate_label=candidate_label, order_seed=order_seed)
    if candidate_label == PG_PREPROCESS_V1:
        candidate = pg_preprocess_v1_manifest()
    elif candidate_label == PG_PREPROCESS_V2:
        candidate = pg_preprocess_v2_manifest()
    else:
        raise ValueError(f"unknown preprocessing candidate {candidate_label}")
    _write_json(destination / "frozen_candidate_manifest.json", json.loads(candidate.canonical_bytes))
    _write_json(destination / "deterministic_record_manifest.json", manifest)
    with (destination / "annotation_records.jsonl").open("w", encoding="utf-8") as stream:
        for item in ordered:
            stream.write(json.dumps({
                "record_id": item.record_id,
                "query_id": item.query_id,
                "benchmark": item.benchmark,
                "native_type": item.native_type,
                "section": item.section,
                "annotation_dimension": item.annotation_dimension,
                "candidate_id": item.candidate_id,
                "allowed_labels": item.allowed_labels,
                "allowed_reason_codes": item.reason_codes,
                "evidence": item.evidence,
            }, ensure_ascii=False, sort_keys=True) + "\n")
    rows = [_template_row(item) for item in ordered]
    for filename in ("annotator_A_labels.csv", "annotator_B_labels.csv", "adjudication_template.csv"):
        _write_csv(destination / filename, rows)
    (destination / "annotation_guideline.md").write_text(guideline.rstrip() + "\n", encoding="utf-8")
    _write_json(destination / "annotator_metadata_template.json", {
        "candidate": candidate_label,
        "annotators": [
            _empty_person("annotator_A"), _empty_person("annotator_B"), _empty_person("adjudicator")
        ],
        "first_pass_independence_confirmed": False,
        "adjudicator_systematic_error_review": {
            "status": "UNFILLED",
            "allowed": ["NO_SYSTEMATIC_ERROR", "SYSTEMATIC_ERROR_FOUND", "UNRESOLVED"],
            "rationale": "",
        },
    })
    _write_json(destination / "mechanical_invariants_template.json", {
        "provenance_or_span_violations": None,
        "information_flow_violations": None,
        "synthetic_target_absence_violations": None,
    })
    (destination / "README.md").write_text(_package_readme(candidate_label), encoding="utf-8")
    digests = package_digests(destination)
    (destination / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(digests.items())), encoding="utf-8"
    )
    return package_digests(destination)


def package_digests(directory: str | Path) -> dict[str, str]:
    root = Path(directory)
    return {
        path.name: sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.iterdir())
        if path.is_file() and path.name != "SHA256SUMS"
    }


def load_labels(path: str | Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    expected = {item["record_id"]: item for item in manifest["records"]}
    values: dict[str, str] = {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            record_id = row.get("record_id", "")
            if record_id not in expected:
                raise ValueError(f"changed or unknown record ID {record_id!r}")
            if record_id in values:
                raise ValueError(f"duplicate judgment {record_id}")
            registered = expected[record_id]
            for field in ("query_id", "benchmark", "annotation_dimension", "candidate_id", "allowed_labels"):
                actual = row.get(field, "")
                wanted = "|".join(registered[field]) if field == "allowed_labels" else str(registered[field])
                if actual != wanted:
                    raise ValueError(f"immutable annotation field changed for {record_id}: {field}")
            label = row.get("annotator_label", "").strip()
            if label not in registered["allowed_labels"]:
                raise ValueError(f"invalid or missing label for {record_id}")
            values[record_id] = label
    missing = sorted(set(expected) - values.keys())
    if missing:
        raise ValueError(f"missing {len(missing)} required judgments")
    return values


def load_adjudication(
    path: str | Path,
    manifest: Mapping[str, Any],
    disagreement_ids: Iterable[str],
) -> dict[str, str]:
    expected_all = {item["record_id"]: item for item in manifest["records"]}
    expected = set(disagreement_ids)
    values: dict[str, str] = {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            record_id = row.get("record_id", "")
            label = row.get("annotator_label", "").strip()
            if not label:
                continue
            if record_id not in expected:
                raise ValueError(f"adjudication provided for a non-disagreement {record_id!r}")
            if record_id in values:
                raise ValueError(f"duplicate adjudication {record_id}")
            registered = expected_all[record_id]
            for field in ("query_id", "benchmark", "annotation_dimension", "candidate_id", "allowed_labels"):
                actual = row.get(field, "")
                wanted = "|".join(registered[field]) if field == "allowed_labels" else str(registered[field])
                if actual != wanted:
                    raise ValueError(f"immutable adjudication field changed for {record_id}: {field}")
            if label not in registered["allowed_labels"]:
                raise ValueError(f"invalid adjudication label for {record_id}")
            if not row.get("short_rationale", "").strip():
                raise ValueError(f"adjudication rationale is required for {record_id}")
            values[record_id] = label
    missing = sorted(expected - values.keys())
    if missing:
        raise ValueError(f"missing {len(missing)} disagreement adjudications")
    return values


def raw_agreement(left: Sequence[str], right: Sequence[str]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("agreement requires equal non-empty label sequences")
    return sum(a == b for a, b in zip(left, right, strict=True)) / len(left)


def cohens_kappa(left: Sequence[str], right: Sequence[str]) -> float | None:
    if len(left) != len(right) or not left:
        raise ValueError("kappa requires equal non-empty label sequences")
    labels = sorted(set(left) | set(right))
    observed = raw_agreement(left, right)
    total = len(left)
    expected = sum((left.count(label) / total) * (right.count(label) / total) for label in labels)
    if expected == 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


def score_annotations(
    manifest: Mapping[str, Any],
    labels_a: Mapping[str, str],
    labels_b: Mapping[str, str],
    adjudication: Mapping[str, str] | None = None,
    *,
    mechanical_violations: Mapping[str, int] | None = None,
    systematic_error_status: str = "UNRESOLVED",
) -> dict[str, Any]:
    rows = manifest["records"]
    by_dimension: dict[str, list[str]] = {}
    for row in rows:
        by_dimension.setdefault(row["annotation_dimension"], []).append(row["record_id"])
    agreement: dict[str, Any] = {}
    for dimension, ids in sorted(by_dimension.items()):
        left = [labels_a[item] for item in ids]
        right = [labels_b[item] for item in ids]
        agreement[dimension] = {"n": len(ids), "raw_agreement": raw_agreement(left, right), "cohens_kappa": cohens_kappa(left, right)}
    primary_ids = [row["record_id"] for row in rows if DIMENSIONS[row["annotation_dimension"]].primary]
    left_primary = [_coarse(labels_a[item], next(row["annotation_dimension"] for row in rows if row["record_id"] == item)) for item in primary_ids]
    right_primary = [_coarse(labels_b[item], next(row["annotation_dimension"] for row in rows if row["record_id"] == item)) for item in primary_ids]
    pooled = {"n": len(primary_ids), "raw_agreement": raw_agreement(left_primary, right_primary), "cohens_kappa": cohens_kappa(left_primary, right_primary)}
    disagreements = [item for item in (row["record_id"] for row in rows) if labels_a[item] != labels_b[item]]
    iaa_pass = pooled["raw_agreement"] >= 0.85 and pooled["cohens_kappa"] is not None and pooled["cohens_kappa"] >= 0.70
    report: dict[str, Any] = {
        "candidate": manifest.get("candidate", PG_PREPROCESS_V1),
        "agreement_by_dimension": agreement,
        "pooled_primary_agreement": pooled,
        "iaa_gate_pass": iaa_pass,
        "disagreement_count": len(disagreements),
        "disagreement_record_ids": disagreements,
        "adjudication_complete": False,
        "precision": None,
        "accuracy_gate_pass": False,
    }
    if adjudication is None:
        return report
    missing = sorted(set(disagreements) - adjudication.keys())
    extra = sorted(set(adjudication) - set(disagreements))
    if missing or extra:
        raise ValueError(f"adjudication must cover exactly disagreements; missing={missing}, extra={extra}")
    final = {item: (labels_a[item] if labels_a[item] == labels_b[item] else adjudication[item]) for item in labels_a}
    precision = _precision_metrics(rows, final)
    violations = dict(mechanical_violations or {})
    mechanical_pass = all(value == 0 for value in violations.values())
    thresholds = {
        "query_slot": precision["query_slot"] >= 0.95,
        "target_changing": precision["target_changing"] >= 0.90,
        "unsupported": precision["unsupported"] >= 0.90,
        "update": precision["update"] >= 0.90,
        "delete": precision["delete"] >= 0.90,
        "urc": precision["urc"] >= 0.90,
        "g_query": precision["g_query"] >= 0.95,
        "g_evidence": precision["g_evidence"] >= 0.95,
    }
    report.update({
        "adjudication_complete": True,
        "precision": precision,
        "precision_thresholds": thresholds,
        "mechanical_violations": violations,
        "mechanical_gate_pass": mechanical_pass,
        "systematic_error_status": systematic_error_status,
        "nonpositive_pattern_counts": _nonpositive_patterns(rows, final),
        "accuracy_gate_pass": iaa_pass and all(thresholds.values()) and mechanical_pass and systematic_error_status == "NO_SYSTEMATIC_ERROR",
    })
    return report


def _precision_metrics(rows: Sequence[Mapping[str, Any]], final: Mapping[str, str]) -> dict[str, float]:
    groups = {
        "query_slot": ("query_slot_validity",),
        "target_changing": ("target_changing_validity",),
        "unsupported": ("unsupported_slot_validity",),
        "update": ("update_validity",),
        "delete": ("delete_validity",),
        "g_evidence": ("g_evidence_region_support",),
    }
    result: dict[str, float] = {}
    for name, dimensions in groups.items():
        selected = [row for row in rows if row["annotation_dimension"] in dimensions]
        result[name] = _positive_fraction(selected, final)
    urc_by_candidate: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row["annotation_dimension"] in {"urc_existing_proposition", "urc_unrelatedness"}:
            urc_by_candidate.setdefault((row["query_id"], row["candidate_id"]), []).append(row)
    result["urc"] = sum(all(final[row["record_id"]] == "YES" for row in pair) and len(pair) == 2 for pair in urc_by_candidate.values()) / len(urc_by_candidate)
    g_by_query: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if row["annotation_dimension"] in {"g_gold_reference_correctness", "g_support_set_correctness"}:
            g_by_query.setdefault(row["query_id"], []).append(row)
    result["g_query"] = sum(all(final[row["record_id"]] == "YES" for row in pair) and len(pair) == 2 for pair in g_by_query.values()) / len(g_by_query)
    result["meaning_preserving"] = result["query_slot"]
    return result


def _positive_fraction(rows: Sequence[Mapping[str, Any]], final: Mapping[str, str]) -> float:
    if not rows:
        raise ValueError("precision metric has no registered opportunities")
    return sum(final[row["record_id"]] in DIMENSIONS[row["annotation_dimension"]].positive for row in rows) / len(rows)


def _nonpositive_patterns(
    rows: Sequence[Mapping[str, Any]], final: Mapping[str, str]
) -> dict[str, dict[str, int]]:
    patterns: dict[str, dict[str, int]] = {
        "benchmark": {}, "native_type": {}, "annotation_dimension": {}
    }
    for row in rows:
        spec = DIMENSIONS[row["annotation_dimension"]]
        if final[row["record_id"]] in spec.positive:
            continue
        for group, value in (
            ("benchmark", row["benchmark"]),
            ("native_type", row.get("native_type", "unknown")),
            ("annotation_dimension", row["annotation_dimension"]),
        ):
            patterns[group][value] = patterns[group].get(value, 0) + 1
    return patterns


def _coarse(label: str, dimension: str) -> str:
    spec = DIMENSIONS[dimension]
    if label in spec.positive:
        return "PASS"
    if label in spec.negative:
        return "FAIL"
    return "UNRESOLVED"


def _template_row(item: AnnotationRecord) -> dict[str, str]:
    return {
        "record_id": item.record_id, "query_id": item.query_id,
        "benchmark": item.benchmark, "annotation_dimension": item.annotation_dimension,
        "candidate_id": item.candidate_id, "allowed_labels": "|".join(item.allowed_labels),
        "annotator_label": "", "reason_code": "", "short_rationale": "",
    }


def _empty_person(identifier: str) -> dict[str, Any]:
    return {
        "assignment": identifier, "anonymous_annotator_id": "", "role": "",
        "relevant_technical_or_research_experience": "",
        "llm_or_memory_agent_evaluation_familiarity": "",
        "locomo_or_longmemeval_familiarity": "",
        "contributed_to_preprocessing_rule_design": None,
    }


def _walk_mapping_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _walk_mapping_keys(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk_mapping_keys(item)


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_bytes(value) + b"\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _package_readme(candidate_label: str = PG_PREPROCESS_V1) -> str:
    return f"""# {candidate_label} human review

Annotators A and B must independently read `annotation_guideline.md` and
`annotation_records.jsonl`, fill only the final three columns of their own
CSV, and complete their own metadata. They must not exchange labels or see
agreement statistics. Keep evaluator-only evidence out of any search-side
artifact.

After both first passes, run the scorer without adjudication to enumerate
disagreements. A third documented adjudicator fills labels for exactly those
records and records a systematic-error review. Human labels remain external
inputs and are never generated by the preprocessing model or scorer.

First-pass command:

`PYTHONPATH=src python -m ufuzz.pg_annotation score --package PACKAGE --annotator-a PACKAGE/annotator_A_labels.csv --annotator-b PACKAGE/annotator_B_labels.csv --output first_pass_score.json`

After adjudication, fill metadata and the mechanical invariant report and add
`--adjudication PACKAGE/adjudication_template.csv --metadata completed_metadata.json --mechanical-report completed_mechanical_invariants.json`.
"""


def _read_manifest(package: Path) -> dict[str, Any]:
    return json.loads((package / "deterministic_record_manifest.json").read_text())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score")
    score.add_argument("--package", type=Path, required=True)
    score.add_argument("--annotator-a", type=Path, required=True)
    score.add_argument("--annotator-b", type=Path, required=True)
    score.add_argument("--adjudication", type=Path)
    score.add_argument("--metadata", type=Path)
    score.add_argument("--mechanical-report", type=Path)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = _read_manifest(args.package)
    a = load_labels(args.annotator_a, manifest)
    b = load_labels(args.annotator_b, manifest)
    initial = score_annotations(manifest, a, b)
    adjudication = (
        load_adjudication(args.adjudication, manifest, initial["disagreement_record_ids"])
        if args.adjudication else None
    )
    systematic = "UNRESOLVED"
    if args.adjudication and (not args.metadata or not args.mechanical_report):
        raise ValueError("final adjudicated scoring requires metadata and mechanical invariant report")
    if args.metadata:
        metadata = json.loads(args.metadata.read_text())
        people = metadata.get("annotators", [])
        identifiers = [person.get("anonymous_annotator_id") for person in people]
        required_person_fields = {
            "anonymous_annotator_id", "role", "relevant_technical_or_research_experience",
            "llm_or_memory_agent_evaluation_familiarity", "locomo_or_longmemeval_familiarity",
            "contributed_to_preprocessing_rule_design",
        }
        required_text_fields = required_person_fields - {
            "contributed_to_preprocessing_rule_design"
        }
        if len(people) != 3 or len(set(identifiers)) != 3 or any(
            not required_person_fields.issubset(person)
            or any(not str(person.get(field, "")).strip() for field in required_text_fields)
            or not isinstance(person.get("contributed_to_preprocessing_rule_design"), bool)
            for person in people
        ):
            raise ValueError("two distinct annotators and one distinct adjudicator need complete metadata")
        if not metadata.get("first_pass_independence_confirmed"):
            raise ValueError("independent first-pass annotation must be confirmed")
        systematic = metadata.get("adjudicator_systematic_error_review", {}).get("status", "UNRESOLVED")
        if systematic not in {"NO_SYSTEMATIC_ERROR", "SYSTEMATIC_ERROR_FOUND", "UNRESOLVED"}:
            raise ValueError("invalid adjudicator systematic-error status")
    mechanical = json.loads(args.mechanical_report.read_text()) if args.mechanical_report else {}
    report = score_annotations(
        manifest, a, b, adjudication,
        mechanical_violations=mechanical,
        systematic_error_status=systematic,
    )
    args.output.write_bytes(canonical_bytes(report) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
