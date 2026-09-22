"""Immutable, outcome-independent semantic sidecar schemas.

Search-safe structure and evaluator-only references deliberately use distinct
top-level types and serializers.  This module performs no model inference and
contains no runtime evaluator behavior.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from enum import StrEnum
from hashlib import sha256
import json
import re
import unicodedata
from typing import Any

from ufuzz.domain import BenchmarkCheckpoint, SourceUnit


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SPACE_RE = re.compile(r"\s+")
_SEARCH_FORBIDDEN_KEYS = frozenset(
    {
        "gold_answer",
        "gold_answers",
        "accepted_answer",
        "accepted_answers",
        "accepted_answer_set",
        "correct_support",
        "evaluator_result",
        "correctness",
        "failure_surface",
        "cfs",
        "uf",
        "cov",
        "answer_session_ids",
        "has_answer",
        "evidence",
    }
)
_GLOBAL_FORBIDDEN_KEYS = frozenset(
    {
        "backend",
        "backend_id",
        "method",
        "method_id",
        "repetition",
        "campaign_id",
        "physical_memory_id",
        "fault_count",
    }
)


def normalize_text(value: str, *, case_insensitive: bool = False) -> str:
    """Apply the frozen NFKC/whitespace normalization baseline."""

    if not isinstance(value, str):
        raise TypeError("semantic text must be a string")
    normalized = _SPACE_RE.sub(" ", unicodedata.normalize("NFKC", value).strip())
    if not normalized:
        raise ValueError("semantic text must be non-empty after normalization")
    return normalized.casefold() if case_insensitive else normalized


def _require_identifier(value: str, name: str) -> str:
    value = normalize_text(value)
    if any(character in value for character in "\r\n\x00"):
        raise ValueError(f"{name} contains a forbidden control character")
    return value


def _require_sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _primitive(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _primitive(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _primitive(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_primitive(item) for item in value]
    if isinstance(value, (set, frozenset)):
        encoded = [_primitive(item) for item in value]
        return sorted(encoded, key=lambda item: json.dumps(item, sort_keys=True))
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"cannot canonically serialize {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _primitive(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return sha256(canonical_bytes(value)).hexdigest()


def _walk_keys(value: Any) -> Iterable[str]:
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            yield field.name.casefold()
            yield from _walk_keys(getattr(value, field.name))
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key).casefold()
            yield from _walk_keys(item)
    elif isinstance(value, (tuple, list, set, frozenset)):
        for item in value:
            yield from _walk_keys(item)


class AnswerabilityStatus(StrEnum):
    ANSWERABLE = "answerable"
    UNSUPPORTED = "unsupported"
    UNRESOLVED = "unresolved"


class TemporalKind(StrEnum):
    LOCAL_ISO = "local_iso"
    SYMBOLIC = "symbolic"
    UNSPECIFIED = "unspecified"


class SourceDisposition(StrEnum):
    FACTS_EXTRACTED = "facts_extracted"
    NO_ELIGIBLE_FACT = "no_eligible_fact"
    UNRESOLVED = "unresolved"


class CompletenessValidation(StrEnum):
    NOT_VALIDATED = "not_validated"
    PARTIAL = "partial"
    VALIDATED = "validated"


class NativeEvidenceStatus(StrEnum):
    VALID = "valid"
    PARTIAL = "partial"
    MALFORMED = "malformed"
    ABSENT = "absent"


class GoldFieldStatus(StrEnum):
    RESOLVED_SINGLE_FIELD = "resolved_single_field"
    RESOLVED_EQUAL_FIELDS = "resolved_equal_fields"
    UNRESOLVED_CONFLICT = "unresolved_conflict"
    MISSING = "missing"


@dataclass(frozen=True, slots=True)
class SemanticArtifactIdentity:
    benchmark: str
    benchmark_artifact_sha256: str
    checkpoint_id: str
    preprocessing_contract_version: str
    semantic_schema_version: str

    def __post_init__(self) -> None:
        for name in (
            "benchmark",
            "checkpoint_id",
            "preprocessing_contract_version",
            "semantic_schema_version",
        ):
            object.__setattr__(self, name, _require_identifier(getattr(self, name), name))
        _require_sha256(self.benchmark_artifact_sha256, "benchmark_artifact_sha256")


@dataclass(frozen=True, slots=True)
class SourceAnchor:
    provenance_id: str
    source_ordinal: int
    session_occurrence: int | None
    session_id: str
    turn_index: int | None
    char_start: int
    char_end: int
    source_text_sha256: str

    def __post_init__(self) -> None:
        for name in ("provenance_id", "session_id"):
            object.__setattr__(self, name, _require_identifier(getattr(self, name), name))
        if self.source_ordinal <= 0:
            raise ValueError("source_ordinal must be positive")
        for name in ("session_occurrence", "turn_index"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or None")
        if self.char_start < 0 or self.char_end <= self.char_start:
            raise ValueError("SourceAnchor requires a non-empty character span")
        _require_sha256(self.source_text_sha256, "source_text_sha256")

    @classmethod
    def from_source(cls, source: SourceUnit, start: int, end: int) -> "SourceAnchor":
        return cls(
            source.provenance_id,
            source.ordinal,
            source.session_occurrence,
            source.session_id,
            source.turn_index,
            start,
            end,
            sha256(source.text.encode("utf-8")).hexdigest(),
        )


@dataclass(frozen=True, order=True, slots=True)
class CanonicalEntity:
    entity_id: str
    label: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_id", _require_identifier(self.entity_id, "entity_id"))
        object.__setattr__(self, "label", normalize_text(self.label))


@dataclass(frozen=True, order=True, slots=True)
class AliasBinding:
    alias: str
    entity_id: str
    case_insensitive: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "alias", normalize_text(self.alias, case_insensitive=self.case_insensitive)
        )
        object.__setattr__(self, "entity_id", _require_identifier(self.entity_id, "entity_id"))


@dataclass(frozen=True, slots=True)
class EntityRegistry:
    entities: tuple[CanonicalEntity, ...]

    def __post_init__(self) -> None:
        values = tuple(sorted(self.entities))
        if len({item.entity_id for item in values}) != len(values):
            raise ValueError("entity IDs must be unique")
        object.__setattr__(self, "entities", values)

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(item.entity_id for item in self.entities)


@dataclass(frozen=True, slots=True)
class AliasRegistry:
    bindings: tuple[AliasBinding, ...]

    def __post_init__(self) -> None:
        values = tuple(sorted(self.bindings, key=lambda item: (item.alias, item.entity_id)))
        identities: dict[str, str] = {}
        for item in values:
            previous = identities.setdefault(item.alias, item.entity_id)
            if previous != item.entity_id:
                raise ValueError(f"ambiguous alias {item.alias!r}")
        object.__setattr__(self, "bindings", values)


@dataclass(frozen=True, order=True, slots=True)
class RelationDefinition:
    relation_id: str
    label: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "relation_id", _require_identifier(self.relation_id, "relation_id"))
        object.__setattr__(self, "label", normalize_text(self.label))


@dataclass(frozen=True, slots=True)
class RelationRegistry:
    relations: tuple[RelationDefinition, ...]

    def __post_init__(self) -> None:
        values = tuple(sorted(self.relations))
        if len({item.relation_id for item in values}) != len(values):
            raise ValueError("relation IDs must be unique")
        object.__setattr__(self, "relations", values)

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(item.relation_id for item in self.relations)


@dataclass(frozen=True, order=True, slots=True)
class CanonicalTemporalScope:
    temporal_id: str
    original_literal: str | None
    canonical_value: str | None
    kind: TemporalKind

    def __post_init__(self) -> None:
        object.__setattr__(self, "temporal_id", _require_identifier(self.temporal_id, "temporal_id"))
        if not isinstance(self.kind, TemporalKind):
            raise TypeError("kind must be TemporalKind")
        if self.kind is TemporalKind.UNSPECIFIED:
            if self.original_literal is not None or self.canonical_value is not None:
                raise ValueError("unspecified time cannot have a literal or value")
            return
        if self.original_literal is None or self.canonical_value is None:
            raise ValueError("specified time requires literal and canonical value")
        object.__setattr__(self, "original_literal", normalize_text(self.original_literal))
        object.__setattr__(self, "canonical_value", normalize_text(self.canonical_value))


def canonicalize_temporal(temporal_id: str, literal: str | None) -> CanonicalTemporalScope:
    if literal is None or not literal.strip():
        return CanonicalTemporalScope(temporal_id, None, None, TemporalKind.UNSPECIFIED)
    normalized = normalize_text(literal)
    candidate = normalized.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return CanonicalTemporalScope(
            temporal_id, normalized, normalized, TemporalKind.SYMBOLIC
        )
    return CanonicalTemporalScope(
        temporal_id,
        normalized,
        parsed.isoformat(timespec="seconds"),
        TemporalKind.LOCAL_ISO,
    )


@dataclass(frozen=True, slots=True)
class TemporalRegistry:
    scopes: tuple[CanonicalTemporalScope, ...]

    def __post_init__(self) -> None:
        values = tuple(sorted(self.scopes))
        if len({item.temporal_id for item in values}) != len(values):
            raise ValueError("temporal IDs must be unique")
        object.__setattr__(self, "scopes", values)

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(item.temporal_id for item in self.scopes)


@dataclass(frozen=True, order=True, slots=True)
class CanonicalConstraint:
    key: str
    value: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", normalize_text(self.key, case_insensitive=True))
        object.__setattr__(self, "value", normalize_text(self.value))


@dataclass(frozen=True, slots=True)
class ConstraintRegistry:
    constraints: tuple[CanonicalConstraint, ...]

    def __post_init__(self) -> None:
        values = tuple(sorted(self.constraints))
        if len(set(values)) != len(values):
            raise ValueError("constraints must be unique")
        object.__setattr__(self, "constraints", values)


@dataclass(frozen=True, slots=True)
class StructuralFact:
    fact_id: str
    canonical_entity: str
    canonical_relation: str
    canonical_value: str
    canonical_temporal_scope: str
    source_anchor: SourceAnchor
    source_order: int

    def __post_init__(self) -> None:
        for name in ("fact_id", "canonical_entity", "canonical_relation", "canonical_temporal_scope"):
            object.__setattr__(self, name, _require_identifier(getattr(self, name), name))
        object.__setattr__(self, "canonical_value", normalize_text(self.canonical_value))
        if self.source_order < 0:
            raise ValueError("source_order must be non-negative")


@dataclass(frozen=True, slots=True)
class CanonicalQueryIntent:
    query_id: str
    canonical_entity: str
    canonical_relation: str
    canonical_temporal_scope: str
    answer_affecting_constraints: tuple[CanonicalConstraint, ...]
    checkpoint_relative_answerability_status: AnswerabilityStatus

    def __post_init__(self) -> None:
        for name in ("query_id", "canonical_entity", "canonical_relation", "canonical_temporal_scope"):
            object.__setattr__(self, name, _require_identifier(getattr(self, name), name))
        object.__setattr__(
            self,
            "answer_affecting_constraints",
            tuple(sorted(set(self.answer_affecting_constraints))),
        )
        if not isinstance(self.checkpoint_relative_answerability_status, AnswerabilityStatus):
            raise TypeError("answerability must be AnswerabilityStatus")


@dataclass(frozen=True, order=True, slots=True)
class SourceCompletenessRecord:
    provenance_id: str
    disposition: SourceDisposition

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance_id", _require_identifier(self.provenance_id, "provenance_id"))
        if not isinstance(self.disposition, SourceDisposition):
            raise TypeError("disposition must be SourceDisposition")


@dataclass(frozen=True, slots=True)
class CompletenessScope:
    processed_provenance_ids: tuple[str, ...]
    claimed_fact_classes: tuple[str, ...]
    source_records: tuple[SourceCompletenessRecord, ...]
    validation_status: CompletenessValidation

    def __post_init__(self) -> None:
        processed = tuple(sorted(set(self.processed_provenance_ids)))
        records = tuple(sorted(self.source_records))
        if len({record.provenance_id for record in records}) != len(records):
            raise ValueError("completeness records must be unique by provenance")
        if set(processed) != {record.provenance_id for record in records}:
            raise ValueError("processed sources and completeness records must match")
        if self.validation_status is CompletenessValidation.VALIDATED and any(
            record.disposition is SourceDisposition.UNRESOLVED for record in records
        ):
            raise ValueError("validated completeness cannot contain unresolved sources")
        object.__setattr__(self, "processed_provenance_ids", processed)
        object.__setattr__(self, "claimed_fact_classes", tuple(sorted(set(self.claimed_fact_classes))))
        object.__setattr__(self, "source_records", records)


@dataclass(frozen=True, slots=True)
class SearchSafeSemanticSidecar:
    identity: SemanticArtifactIdentity
    facts: tuple[StructuralFact, ...]
    query_intents: tuple[CanonicalQueryIntent, ...]
    entities: EntityRegistry
    aliases: AliasRegistry
    relations: RelationRegistry
    temporals: TemporalRegistry
    constraints: ConstraintRegistry
    completeness: CompletenessScope

    @property
    def artifact_bytes(self) -> bytes:
        validate_search_safe_information_flow(self)
        return canonical_bytes(self)

    @property
    def sha256_digest(self) -> str:
        return sha256(self.artifact_bytes).hexdigest()


@dataclass(frozen=True, slots=True)
class AcceptedAnswerComponent:
    component_id: str
    canonical_value: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "component_id", _require_identifier(self.component_id, "component_id"))
        object.__setattr__(self, "canonical_value", normalize_text(self.canonical_value))


@dataclass(frozen=True, slots=True)
class AcceptedAnswerOption:
    required_components: tuple[AcceptedAnswerComponent, ...]

    def __post_init__(self) -> None:
        if not self.required_components:
            raise ValueError("an accepted answer option needs at least one component")
        if len({item.component_id for item in self.required_components}) != len(self.required_components):
            raise ValueError("component IDs must be unique within an answer option")


@dataclass(frozen=True, slots=True)
class AcceptedAnswerSet:
    accepted_options: tuple[AcceptedAnswerOption, ...]

    def __post_init__(self) -> None:
        if not self.accepted_options:
            raise ValueError("answerable references need at least one accepted option")


@dataclass(frozen=True, slots=True)
class SupportAlternativeGroup:
    fact_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        values = tuple(sorted(set(self.fact_ids)))
        if not values:
            raise ValueError("a support alternative group needs at least one fact")
        object.__setattr__(self, "fact_ids", values)

    def satisfied_by(self, available_fact_ids: Iterable[str]) -> bool:
        return bool(set(self.fact_ids).intersection(available_fact_ids))


@dataclass(frozen=True, slots=True)
class SupportRequirement:
    required_groups: tuple[SupportAlternativeGroup, ...]

    def __post_init__(self) -> None:
        if not self.required_groups:
            raise ValueError("a support requirement needs at least one required group")

    def satisfied_by(self, available_fact_ids: Iterable[str]) -> bool:
        available = frozenset(available_fact_ids)
        return all(group.satisfied_by(available) for group in self.required_groups)


@dataclass(frozen=True, slots=True)
class GoldComponentSupport:
    component_id: str
    support_requirements: tuple[SupportRequirement, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "component_id", _require_identifier(self.component_id, "component_id"))
        if not self.support_requirements:
            raise ValueError("gold component support needs at least one alternative requirement")


@dataclass(frozen=True, slots=True)
class NativeEvidenceConstraint:
    policy: str
    native_ids: tuple[str, ...]
    allowed_provenance_ids: tuple[str, ...]
    status: NativeEvidenceStatus

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy", _require_identifier(self.policy, "policy"))
        object.__setattr__(self, "native_ids", tuple(sorted(set(self.native_ids))))
        object.__setattr__(self, "allowed_provenance_ids", tuple(sorted(set(self.allowed_provenance_ids))))
        if not isinstance(self.status, NativeEvidenceStatus):
            raise TypeError("status must be NativeEvidenceStatus")


@dataclass(frozen=True, slots=True)
class EvaluationReferenceSeed:
    query_id: str
    accepted_answer_set: AcceptedAnswerSet | None
    native_evidence_constraint: NativeEvidenceConstraint
    component_support: tuple[GoldComponentSupport, ...]
    forbidden_or_superseded_fact_ids: tuple[str, ...]
    answerability: AnswerabilityStatus
    reference_construction_provenance: tuple[str, ...]
    unresolved_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_id", _require_identifier(self.query_id, "query_id"))
        object.__setattr__(
            self,
            "forbidden_or_superseded_fact_ids",
            tuple(sorted(set(self.forbidden_or_superseded_fact_ids))),
        )
        if self.answerability is AnswerabilityStatus.ANSWERABLE:
            if self.accepted_answer_set is None or not self.component_support:
                raise ValueError("answerable reference requires answers and support")
            if (
                self.native_evidence_constraint.status
                not in (NativeEvidenceStatus.VALID, NativeEvidenceStatus.PARTIAL)
                or not self.native_evidence_constraint.allowed_provenance_ids
            ):
                raise ValueError(
                    "answerable reference requires a bounded native evidence scope"
                )
        elif self.answerability is AnswerabilityStatus.UNSUPPORTED:
            if self.accepted_answer_set is not None or self.component_support:
                raise ValueError("unsupported reference cannot contain answers or support")
        elif not self.unresolved_reason:
            raise ValueError("unresolved reference requires a reason")


@dataclass(frozen=True, slots=True)
class EvaluatorReferenceSidecar:
    identity: SemanticArtifactIdentity
    references: tuple[EvaluationReferenceSeed, ...]

    @property
    def artifact_bytes(self) -> bytes:
        _reject_global_identity(self)
        return canonical_bytes(self)

    @property
    def sha256_digest(self) -> str:
        return sha256(self.artifact_bytes).hexdigest()


@dataclass(frozen=True, slots=True)
class StaticArtifactManifest:
    benchmark_artifact_sha256: str
    preprocessing_contract_version: str
    semantic_schema_version: str
    normalization_registry_version: str
    preprocessing_model_config_identity: str
    annotation_artifact_sha256: str
    validation_report_sha256: str
    native_evidence_constraint_policy: str

    def __post_init__(self) -> None:
        for name in (
            "benchmark_artifact_sha256",
            "annotation_artifact_sha256",
            "validation_report_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        for name in (
            "preprocessing_contract_version",
            "semantic_schema_version",
            "normalization_registry_version",
            "preprocessing_model_config_identity",
            "native_evidence_constraint_policy",
        ):
            object.__setattr__(self, name, _require_identifier(getattr(self, name), name))

    @property
    def artifact_bytes(self) -> bytes:
        _reject_global_identity(self)
        return canonical_bytes(self)


@dataclass(frozen=True, slots=True)
class LocomoGoldFieldResolution:
    status: GoldFieldStatus
    source_field: str | None
    value: Any = None


def resolve_locomo_gold_field(raw_query: Mapping[str, Any]) -> LocomoGoldFieldResolution:
    """Resolve only unambiguous local LoCoMo gold-field cases.

    The pinned local artifact has two category-5 records with conflicting
    ``answer`` and ``adversarial_answer`` values, while the available local
    copied evaluator skips category 5.  Those records stay unresolved until an
    authoritative benchmark convention is frozen.
    """

    has_answer = "answer" in raw_query
    has_adversarial = "adversarial_answer" in raw_query
    if not has_answer and not has_adversarial:
        return LocomoGoldFieldResolution(GoldFieldStatus.MISSING, None)
    if has_answer and has_adversarial:
        if raw_query["answer"] == raw_query["adversarial_answer"]:
            return LocomoGoldFieldResolution(
                GoldFieldStatus.RESOLVED_EQUAL_FIELDS,
                "answer+adversarial_answer",
                raw_query["answer"],
            )
        return LocomoGoldFieldResolution(GoldFieldStatus.UNRESOLVED_CONFLICT, None)
    field = "answer" if has_answer else "adversarial_answer"
    return LocomoGoldFieldResolution(
        GoldFieldStatus.RESOLVED_SINGLE_FIELD, field, raw_query[field]
    )


def validate_search_safe_information_flow(sidecar: SearchSafeSemanticSidecar) -> None:
    keys = set(_walk_keys(sidecar))
    forbidden = sorted(keys.intersection(_SEARCH_FORBIDDEN_KEYS | _GLOBAL_FORBIDDEN_KEYS))
    if forbidden:
        raise ValueError("search-safe sidecar contains forbidden fields: " + ", ".join(forbidden))


def _reject_global_identity(value: Any) -> None:
    forbidden = sorted(set(_walk_keys(value)).intersection(_GLOBAL_FORBIDDEN_KEYS))
    if forbidden:
        raise ValueError("semantic artifact contains outcome identity: " + ", ".join(forbidden))


def validate_search_safe_sidecar(
    sidecar: SearchSafeSemanticSidecar,
    checkpoint: BenchmarkCheckpoint,
) -> None:
    validate_search_safe_information_flow(sidecar)
    if sidecar.identity.benchmark != checkpoint.benchmark:
        raise ValueError("sidecar benchmark does not match checkpoint")
    if sidecar.identity.checkpoint_id != checkpoint.checkpoint_id:
        raise ValueError("sidecar checkpoint does not match")
    if sidecar.identity.benchmark_artifact_sha256 != checkpoint.artifact.sha256:
        raise ValueError("sidecar benchmark artifact digest does not match")
    sources = {source.provenance_id: source for source in checkpoint.sources}
    if len(sources) != len(checkpoint.sources):
        raise ValueError("checkpoint provenance IDs are not unique")
    fact_ids: set[str] = set()
    for fact in sidecar.facts:
        if fact.fact_id in fact_ids:
            raise ValueError(f"duplicate fact ID {fact.fact_id}")
        fact_ids.add(fact.fact_id)
        source = sources.get(fact.source_anchor.provenance_id)
        if source is None:
            raise ValueError(f"fact {fact.fact_id} has an unknown source anchor")
        anchor = fact.source_anchor
        if anchor.source_ordinal != source.ordinal or anchor.session_id != source.session_id:
            raise ValueError(f"fact {fact.fact_id} source anchor metadata differs")
        if anchor.session_occurrence != source.session_occurrence or anchor.turn_index != source.turn_index:
            raise ValueError(f"fact {fact.fact_id} occurrence anchor differs")
        if anchor.source_text_sha256 != sha256(source.text.encode("utf-8")).hexdigest():
            raise ValueError(f"fact {fact.fact_id} source digest differs")
        if anchor.char_end > len(source.text) or not source.text[anchor.char_start:anchor.char_end]:
            raise ValueError(f"fact {fact.fact_id} source span is invalid")
        if fact.canonical_entity not in sidecar.entities.ids:
            raise ValueError(f"fact {fact.fact_id} has an unregistered entity")
        if fact.canonical_relation not in sidecar.relations.ids:
            raise ValueError(f"fact {fact.fact_id} has an unregistered relation")
        if fact.canonical_temporal_scope not in sidecar.temporals.ids:
            raise ValueError(f"fact {fact.fact_id} has an unregistered temporal scope")
    if not set(sidecar.completeness.processed_provenance_ids).issubset(sources):
        raise ValueError("completeness ledger references an unknown source")
    query_ids = {query.query_id for query in checkpoint.queries}
    for intent in sidecar.query_intents:
        if intent.query_id not in query_ids:
            raise ValueError(f"unknown query intent {intent.query_id}")
        if intent.canonical_entity not in sidecar.entities.ids:
            raise ValueError(f"query {intent.query_id} has an unregistered entity")
        if intent.canonical_relation not in sidecar.relations.ids:
            raise ValueError(f"query {intent.query_id} has an unregistered relation")
        if intent.canonical_temporal_scope not in sidecar.temporals.ids:
            raise ValueError(f"query {intent.query_id} has an unregistered time")
        if not set(intent.answer_affecting_constraints).issubset(sidecar.constraints.constraints):
            raise ValueError(f"query {intent.query_id} has an unregistered constraint")


def validate_evaluator_sidecar(
    evaluator: EvaluatorReferenceSidecar,
    search_safe: SearchSafeSemanticSidecar,
) -> None:
    _reject_global_identity(evaluator)
    if evaluator.identity != search_safe.identity:
        raise ValueError("search-safe and evaluator-only identities differ")
    facts = {fact.fact_id: fact for fact in search_safe.facts}
    intents = {intent.query_id for intent in search_safe.query_intents}
    for reference in evaluator.references:
        if reference.query_id not in intents:
            raise ValueError(f"reference has unknown query {reference.query_id}")
        component_ids: set[str] = set()
        if reference.accepted_answer_set is not None:
            component_ids = {
                component.component_id
                for option in reference.accepted_answer_set.accepted_options
                for component in option.required_components
            }
        mapped_components = {item.component_id for item in reference.component_support}
        if reference.answerability is AnswerabilityStatus.ANSWERABLE and mapped_components != component_ids:
            raise ValueError("every answer component must have exactly declared support")
        allowed = set(reference.native_evidence_constraint.allowed_provenance_ids)
        if reference.component_support and not allowed:
            raise ValueError("support mapping cannot use an unbounded native evidence scope")
        missing_forbidden = set(reference.forbidden_or_superseded_fact_ids) - facts.keys()
        if missing_forbidden:
            raise ValueError(
                "reference points to absent forbidden/superseded facts: "
                + ", ".join(sorted(missing_forbidden))
            )
        for mapping in reference.component_support:
            for requirement in mapping.support_requirements:
                for group in requirement.required_groups:
                    for fact_id in group.fact_ids:
                        fact = facts.get(fact_id)
                        if fact is None:
                            raise ValueError(f"reference points to absent fact {fact_id}")
                        if allowed and fact.source_anchor.provenance_id not in allowed:
                            raise ValueError(f"support fact {fact_id} violates native evidence scope")
