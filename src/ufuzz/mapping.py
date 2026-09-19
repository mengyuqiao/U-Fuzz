"""Conservative fact-level retrieval-to-region mapping."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Iterable, Mapping

from ufuzz.structural import (
    CanonicalFact,
    ProvisionalIdentityNormalizer,
    RegionNormalizer,
    StructuralIndex,
)


RegionKey = tuple[str, ...]


class MappingStatus(StrEnum):
    MAPPED = "mapped"
    AMBIGUOUS = "ambiguous"
    UNMAPPED = "unmapped"


@dataclass(frozen=True, slots=True)
class RetrievalRecord:
    backend: str
    local_id: str
    text: str
    rank: int
    score: float | None
    provenance_ids: tuple[str, ...]
    metadata: Mapping[str, Any]
    raw: Any = None


@dataclass(frozen=True, slots=True)
class MappingResult:
    entry_id: str
    fact_map: tuple[CanonicalFact, ...]
    semantic_regions: frozenset[RegionKey]
    fallback_regions: frozenset[RegionKey]
    regions: frozenset[RegionKey]
    status: MappingStatus
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MappingQuality:
    total: int
    mapped: int
    ambiguous: int
    unmapped: int

    @property
    def mapping_rate(self) -> float:
        return self.mapped / self.total if self.total else 0.0


class FactLevelMapper:
    def __init__(
        self,
        index: StructuralIndex,
        normalizer: RegionNormalizer | None = None,
    ) -> None:
        self.index = index
        self.normalizer = normalizer or ProvisionalIdentityNormalizer()
        self._facts = index.fact_by_id()

    def semantic_key(self, fact: CanonicalFact) -> RegionKey:
        return (
            "entity-relation",
            self.normalizer.normalize_entity(fact.entity),
            self.normalizer.normalize_relation(fact.relation),
        )
    def source_semantic_keys(self, provenance_id: str) -> frozenset[RegionKey]:
        return frozenset(
            self.semantic_key(fact)
            for fact in self.index.facts_for_source(provenance_id)
        )

    def map(self, entry: RetrievalRecord) -> MappingResult:
        reasons: list[str] = []
        mapped_facts: dict[str, CanonicalFact] = {}
        fallback: set[RegionKey] = set()
        ambiguous = False

        certified_ids = entry.metadata.get("ufuzz_canonical_fact_ids", ())
        if isinstance(certified_ids, str):
            certified_ids = (certified_ids,)
        if isinstance(certified_ids, Iterable):
            for fact_id in certified_ids:
                fact = self._facts.get(str(fact_id))
                if fact and fact.provenance_id in entry.provenance_ids:
                    mapped_facts[fact.fact_id] = fact

        for provenance_id in entry.provenance_ids:
            if provenance_id not in self.index.provenance.sources:
                reasons.append(f"unknown provenance: {provenance_id}")
                continue
            facts = self.index.facts_for_source(provenance_id)
            keys = {self.semantic_key(fact) for fact in facts}
            if not keys:
                fallback.add(("source", provenance_id))
                continue

            if len(keys) == 1:
                for fact in facts:
                    mapped_facts[fact.fact_id] = fact
                continue

            matched = self._match_fact_content(entry.text, facts)
            if matched:
                for fact in matched:
                    mapped_facts[fact.fact_id] = fact
            else:
                ambiguous = True
                reasons.append(
                    f"multi-region source {provenance_id} cannot be fact-disambiguated"
                )

        semantic = frozenset(self.semantic_key(f) for f in mapped_facts.values())
        fallback_regions = frozenset(fallback)
        regions = semantic | fallback_regions
        if regions:
            status = MappingStatus.MAPPED
        elif ambiguous:
            status = MappingStatus.AMBIGUOUS
        else:
            status = MappingStatus.UNMAPPED
            if not entry.provenance_ids:
                reasons.append("retrieved entry has no reliable provenance")

        return MappingResult(
            entry_id=entry.local_id,
            fact_map=tuple(mapped_facts.values()),
            semantic_regions=semantic,
            fallback_regions=fallback_regions,
            regions=regions,
            status=status,
            reasons=tuple(reasons),
        )

    @staticmethod
    def _match_fact_content(
        text: str, facts: tuple[CanonicalFact, ...]
    ) -> tuple[CanonicalFact, ...]:
        folded = text.casefold()
        return tuple(
            fact
            for fact in facts
            if fact.entity.casefold() in folded and fact.value.casefold() in folded
        )

    @staticmethod
    def quality(results: Iterable[MappingResult]) -> MappingQuality:
        values = tuple(results)
        return MappingQuality(
            total=len(values),
            mapped=sum(r.status is MappingStatus.MAPPED for r in values),
            ambiguous=sum(r.status is MappingStatus.AMBIGUOUS for r in values),
            unmapped=sum(r.status is MappingStatus.UNMAPPED for r in values),
        )
