from __future__ import annotations

from pathlib import Path
import unittest

from ufuzz.benchmarks import LoCoMoLoader, LongMemEvalSLoader
from ufuzz.mapping import FactLevelMapper, MappingStatus, RetrievalRecord
from ufuzz.structural import StructuralIndexBuilder, exact_source_fact


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "locomo_minimal.json"


class StructuralIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.checkpoint = next(LoCoMoLoader().load(FIXTURE, verify_artifact=False))

    def test_native_natural_language_remains_unstructured(self) -> None:
        index = StructuralIndexBuilder().build(self.checkpoint)
        self.assertEqual(index.facts, ())
        intent = index.query_intents[self.checkpoint.queries[0].query_id]
        self.assertIsNone(intent.entity)
        self.assertIsNone(intent.attribute)
        self.assertIsNotNone(intent.unstructured_reason)

    def test_structural_provenance_excludes_evaluator_turn_annotations(self) -> None:
        checkpoint = next(
            LongMemEvalSLoader().load(
                ROOT / "tests" / "fixtures" / "longmemeval_s_minimal.json",
                verify_artifact=False,
            )
        )
        self.assertIn("has_answer", checkpoint.sources[0].raw)
        indexed_source = next(
            iter(StructuralIndexBuilder().build(checkpoint).provenance.sources.values())
        )
        self.assertFalse(hasattr(indexed_source, "raw"))
        self.assertEqual(indexed_source.text, checkpoint.sources[0].text)

    def _certified_index(self):
        source = self.checkpoint.sources[0]
        alice, alice_certificate = exact_source_fact(
            source=source,
            entity="Alice",
            relation="residence",
            value="Boston",
        )
        bob, bob_certificate = exact_source_fact(
            source=source,
            entity="Bob",
            relation="hobby",
            value="tennis",
        )
        return StructuralIndexBuilder().build(
            self.checkpoint,
            facts=(alice, bob),
            certificates=(alice_certificate, bob_certificate),
        )

    def test_multifact_source_maps_only_represented_fact(self) -> None:
        index = self._certified_index()
        mapper = FactLevelMapper(index)
        source_id = self.checkpoint.sources[0].provenance_id
        result = mapper.map(
            RetrievalRecord(
                backend="synthetic",
                local_id="entry-alice",
                text="Alice lives in Boston.",
                rank=1,
                score=1.0,
                provenance_ids=(source_id,),
                metadata={},
            )
        )
        self.assertEqual(result.status, MappingStatus.MAPPED)
        self.assertEqual(
            result.regions,
            frozenset({("entity-relation", "Alice", "residence")}),
        )
        self.assertNotIn(("entity-relation", "Bob", "hobby"), result.regions)

    def test_multifact_source_without_disambiguation_is_ambiguous(self) -> None:
        mapper = FactLevelMapper(self._certified_index())
        result = mapper.map(
            RetrievalRecord(
                backend="synthetic",
                local_id="entry-ambiguous",
                text="A summary of this conversation.",
                rank=1,
                score=None,
                provenance_ids=(self.checkpoint.sources[0].provenance_id,),
                metadata={},
            )
        )
        self.assertEqual(result.status, MappingStatus.AMBIGUOUS)
        self.assertFalse(result.regions)

    def test_source_fallback_only_when_source_has_no_semantic_region(self) -> None:
        index = self._certified_index()
        mapper = FactLevelMapper(index)
        unstructured_source = self.checkpoint.sources[1].provenance_id
        result = mapper.map(
            RetrievalRecord(
                backend="synthetic",
                local_id="entry-fallback",
                text="Bob acknowledges Alice.",
                rank=1,
                score=None,
                provenance_ids=(unstructured_source,),
                metadata={},
            )
        )
        self.assertEqual(
            result.fallback_regions,
            frozenset({("source", unstructured_source)}),
        )
        self.assertFalse(result.semantic_regions)

    def test_mapping_quality_counts_statuses(self) -> None:
        mapper = FactLevelMapper(self._certified_index())
        source = self.checkpoint.sources[0].provenance_id
        results = [
            mapper.map(
                RetrievalRecord(
                    backend="synthetic",
                    local_id="mapped",
                    text="Bob enjoys tennis.",
                    rank=1,
                    score=None,
                    provenance_ids=(source,),
                    metadata={},
                )
            ),
            mapper.map(
                RetrievalRecord(
                    backend="synthetic",
                    local_id="ambiguous",
                    text="Conversation summary.",
                    rank=2,
                    score=None,
                    provenance_ids=(source,),
                    metadata={},
                )
            ),
            mapper.map(
                RetrievalRecord(
                    backend="synthetic",
                    local_id="unmapped",
                    text="No provenance.",
                    rank=3,
                    score=None,
                    provenance_ids=(),
                    metadata={},
                )
            ),
        ]
        quality = mapper.quality(results)
        self.assertEqual((quality.total, quality.mapped, quality.ambiguous, quality.unmapped), (3, 1, 1, 1))
        self.assertAlmostEqual(quality.mapping_rate, 1 / 3)


if __name__ == "__main__":
    unittest.main()
