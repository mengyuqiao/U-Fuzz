"""Partial-but-certified Checkpoint Structural Index records."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
import json
import unicodedata
from typing import Any, Iterable, Mapping, Protocol

from ufuzz.domain import BenchmarkCheckpoint, SourceUnit


class CertificateStatus(StrEnum):
    CERTIFIED = "certified"
    UNSTRUCTURED = "unstructured"


@dataclass(frozen=True, slots=True)
class StructuralCertificate:
    certificate_id: str
    subject_id: str
    field: str
    status: CertificateStatus
    method: str
    provenance_ids: tuple[str, ...]
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CanonicalFact:
    fact_id: str
    entity: str
    relation: str
    value: str
    temporal_scope: str | None
    provenance_id: str
    certificate_ids: tuple[str, ...]

    @classmethod
    def create(
        cls,
        *,
        entity: str,
        relation: str,
        value: str,
        temporal_scope: str | None,
        provenance_id: str,
        certificate_ids: Iterable[str],
    ) -> "CanonicalFact":
        payload = json.dumps(
            [entity, relation, value, temporal_scope, provenance_id],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return cls(
            fact_id="fact:" + sha256(payload.encode("utf-8")).hexdigest(),
            entity=entity,
            relation=relation,
            value=value,
            temporal_scope=temporal_scope,
            provenance_id=provenance_id,
            certificate_ids=tuple(certificate_ids),
        )


@dataclass(frozen=True, slots=True)
class QueryIntent:
    query_id: str
    entity: str | None
    entity_span: tuple[int, int] | None
    attribute: str | None
    attribute_span: tuple[int, int] | None
    time_scope: str | None
    constraints: tuple[str, ...]
    answerability: bool | None
    certificate_ids: tuple[str, ...]
    unstructured_reason: str | None = None


class RegionNormalizer(Protocol):
    def normalize_entity(self, value: str) -> str: ...

    def normalize_relation(self, value: str) -> str: ...


class ProvisionalIdentityNormalizer:
    """Replaceable, identity-preserving Phase-1 normalizer.

    It performs Unicode NFC and edge-whitespace cleanup only. It deliberately
    does not merge aliases, change case, or canonicalize relations.
    """

    @staticmethod
    def _identity(value: str) -> str:
        return unicodedata.normalize("NFC", value.strip())

    def normalize_entity(self, value: str) -> str:
        return self._identity(value)

    def normalize_relation(self, value: str) -> str:
        return self._identity(value)


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    provenance_id: str
    benchmark: str
    checkpoint_id: str
    session_id: str
    source_id: str
    ordinal: int
    text: str
    timestamp: str | None
    speaker: str | None
    role: str | None

    @classmethod
    def from_source(cls, source: SourceUnit) -> "ProvenanceRecord":
        return cls(
            provenance_id=source.provenance_id,
            benchmark=source.benchmark,
            checkpoint_id=source.checkpoint_id,
            session_id=source.session_id,
            source_id=source.source_id,
            ordinal=source.ordinal,
            text=source.text,
            timestamp=source.timestamp,
            speaker=source.speaker,
            role=source.role,
        )


@dataclass(frozen=True, slots=True)
class ProvenanceRegistry:
    """Search-safe stable provenance without benchmark evaluator annotations."""

    sources: Mapping[str, ProvenanceRecord]

    @classmethod
    def from_checkpoint(cls, checkpoint: BenchmarkCheckpoint) -> "ProvenanceRegistry":
        sources: dict[str, ProvenanceRecord] = {}
        for source in checkpoint.sources:
            if source.provenance_id in sources:
                raise ValueError(f"duplicate provenance ID: {source.provenance_id}")
            sources[source.provenance_id] = ProvenanceRecord.from_source(source)
        return cls(sources=sources)


@dataclass(frozen=True, slots=True)
class StructuralIndex:
    checkpoint_id: str
    facts: tuple[CanonicalFact, ...]
    query_intents: Mapping[str, QueryIntent]
    provenance: ProvenanceRegistry
    certificates: Mapping[str, StructuralCertificate]
    normalizer_name: str = "provisional-identity-v1"

    def fact_by_id(self) -> Mapping[str, CanonicalFact]:
        return {fact.fact_id: fact for fact in self.facts}

    def facts_for_source(self, provenance_id: str) -> tuple[CanonicalFact, ...]:
        return tuple(f for f in self.facts if f.provenance_id == provenance_id)


class StructuralIndexBuilder:
    """Build a conservative deterministic Phase-1 index.

    Natural-language benchmark turns and questions do not contain native
    entity/relation slots. They remain explicitly unstructured unless a caller
    supplies separately certified facts or intents.
    """

    def build(
        self,
        checkpoint: BenchmarkCheckpoint,
        *,
        facts: Iterable[CanonicalFact] = (),
        query_intents: Iterable[QueryIntent] = (),
        certificates: Iterable[StructuralCertificate] = (),
    ) -> StructuralIndex:
        provenance = ProvenanceRegistry.from_checkpoint(checkpoint)
        certificate_map = {c.certificate_id: c for c in certificates}
        fact_list = tuple(facts)
        for fact in fact_list:
            if fact.provenance_id not in provenance.sources:
                raise ValueError(
                    f"fact {fact.fact_id} has unknown provenance {fact.provenance_id}"
                )
            missing = set(fact.certificate_ids) - certificate_map.keys()
            if missing:
                raise ValueError(f"fact {fact.fact_id} lacks certificates {sorted(missing)}")

        intent_map = {intent.query_id: intent for intent in query_intents}
        for query in checkpoint.queries:
            if query.query_id in intent_map:
                continue
            certificate_id = f"unstructured:query:{query.query_id}"
            certificate_map[certificate_id] = StructuralCertificate(
                certificate_id=certificate_id,
                subject_id=query.query_id,
                field="QueryIntent",
                status=CertificateStatus.UNSTRUCTURED,
                method="benchmark-native-inspection",
                provenance_ids=(),
                details={
                    "reason": (
                        "benchmark schema provides natural-language question text "
                        "but no certified entity/relation slots"
                    )
                },
            )
            intent_map[query.query_id] = QueryIntent(
                query_id=query.query_id,
                entity=None,
                entity_span=None,
                attribute=None,
                attribute_span=None,
                time_scope=None,
                constraints=(),
                answerability=None,
                certificate_ids=(certificate_id,),
                unstructured_reason=(
                    "no deterministic benchmark-native QueryIntent annotation"
                ),
            )

        if not fact_list:
            certificate_id = f"unstructured:facts:{checkpoint.checkpoint_id}"
            certificate_map[certificate_id] = StructuralCertificate(
                certificate_id=certificate_id,
                subject_id=checkpoint.checkpoint_id,
                field="canonical_facts",
                status=CertificateStatus.UNSTRUCTURED,
                method="benchmark-native-inspection",
                provenance_ids=tuple(provenance.sources),
                details={
                    "reason": (
                        "benchmark histories contain natural-language source units "
                        "but no deterministic canonical fact table"
                    )
                },
            )

        return StructuralIndex(
            checkpoint_id=checkpoint.checkpoint_id,
            facts=fact_list,
            query_intents=intent_map,
            provenance=provenance,
            certificates=certificate_map,
        )


def exact_source_fact(
    *,
    source: SourceUnit,
    entity: str,
    relation: str,
    value: str,
    temporal_scope: str | None = None,
) -> tuple[CanonicalFact, StructuralCertificate]:
    """Certify a synthetic or structured fact using exact source substrings."""

    folded = source.text.casefold()
    missing = [part for part in (entity, relation, value) if part.casefold() not in folded]
    if missing:
        raise ValueError(
            f"cannot certify fact from {source.provenance_id}; missing exact fields {missing}"
        )
    certificate_id = "cert:" + sha256(
        json.dumps(
            [source.provenance_id, entity, relation, value, temporal_scope],
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    certificate = StructuralCertificate(
        certificate_id=certificate_id,
        subject_id=source.provenance_id,
        field="canonical_fact",
        status=CertificateStatus.CERTIFIED,
        method="exact-source-substrings",
        provenance_ids=(source.provenance_id,),
        details={
            "entity": entity,
            "relation": relation,
            "value": value,
        },
    )
    fact = CanonicalFact.create(
        entity=entity,
        relation=relation,
        value=value,
        temporal_scope=temporal_scope,
        provenance_id=source.provenance_id,
        certificate_ids=(certificate_id,),
    )
    return fact, certificate


def certify_exact_source_inventory(
    *,
    source: SourceUnit,
    facts: Iterable[CanonicalFact],
) -> StructuralCertificate:
    """Certify completeness for exact synthetic ``entity relation value.`` rows.

    This narrow deterministic helper exists for infrastructure fixtures. Natural
    benchmark prose must use a separately audited extraction procedure and is
    not certified complete by this helper.
    """

    values = tuple(facts)
    if not values:
        raise ValueError("a complete source inventory must contain at least one fact")
    if any(fact.provenance_id != source.provenance_id for fact in values):
        raise ValueError("inventory facts must share the source provenance")
    expected = " ".join(
        f"{fact.entity} {fact.relation} {fact.value}." for fact in values
    )
    if " ".join(source.text.split()) != expected:
        raise ValueError(
            "source is not an exact synthetic serialization of the proposed facts"
        )
    fact_ids = tuple(fact.fact_id for fact in values)
    certificate_id = "cert:inventory:" + sha256(
        json.dumps(
            [source.provenance_id, fact_ids],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return StructuralCertificate(
        certificate_id=certificate_id,
        subject_id=source.provenance_id,
        field="source_fact_inventory",
        status=CertificateStatus.CERTIFIED,
        method="exact-synthetic-source-inventory",
        provenance_ids=(source.provenance_id,),
        details={"fact_ids": fact_ids},
    )
