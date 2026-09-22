from __future__ import annotations

from pathlib import Path
import unittest

from ufuzz.benchmarks import (
    LoCoMoLoader,
    LongMemEvalSLoader,
    resolve_answer_session_scope,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


class LoCoMoLoaderTests(unittest.TestCase):
    def test_preserves_schema_and_stable_provenance(self) -> None:
        checkpoint = next(
            LoCoMoLoader().load(FIXTURES / "locomo_minimal.json", verify_artifact=False)
        )
        self.assertEqual(checkpoint.checkpoint_id, "conv-test")
        self.assertEqual(len(checkpoint.sources), 2)
        self.assertEqual(
            checkpoint.sources[0].provenance_id,
            "locomo:conv-test:session_1:D1:1",
        )
        self.assertEqual(checkpoint.sources[0].raw["fixture_extra"], "preserved")
        self.assertEqual(checkpoint.queries[0].raw["answer"], "Boston")
        self.assertEqual(checkpoint.raw["fixture_top_level"], "preserved")

    def test_official_artifact_if_resolved(self) -> None:
        path = Path("/tmp/ufuzz-datasets/locomo10.json")
        if not path.exists():
            self.skipTest("official LoCoMo artifact is not resolved locally")
        checkpoints = list(LoCoMoLoader().load(path))
        self.assertEqual(len(checkpoints), 10)
        self.assertTrue(all(checkpoint.sources for checkpoint in checkpoints))
        self.assertEqual(sum(len(checkpoint.sources) for checkpoint in checkpoints), 5_882)
        self.assertEqual(sum(len(checkpoint.queries) for checkpoint in checkpoints), 1_986)


class LongMemEvalLoaderTests(unittest.TestCase):
    def test_preserves_mixed_types_optional_turn_fields_and_provenance(self) -> None:
        checkpoint = next(
            LongMemEvalSLoader().load(
                FIXTURES / "longmemeval_s_minimal.json", verify_artifact=False
            )
        )
        self.assertEqual(checkpoint.checkpoint_id, "q-test")
        self.assertEqual(len(checkpoint.sources), 4)
        self.assertEqual(
            checkpoint.sources[0].provenance_id,
            "longmemeval-s-cleaned:q-test:session-occurrence-0000:"
            "session-1:turn-0000:ordinal-000001",
        )
        self.assertIs(checkpoint.sources[0].raw["has_answer"], True)
        self.assertEqual(checkpoint.sources[-1].raw["fixture_extra"], "preserved")
        self.assertIsInstance(checkpoint.queries[0].raw["answer"], int)
        self.assertEqual(checkpoint.raw["fixture_top_level"], "preserved")

    def test_frozen_official_artifact_hash_and_schema(self) -> None:
        path = Path("/tmp/ufuzz-datasets/longmemeval_s_cleaned.json")
        if not path.exists():
            self.skipTest("frozen LongMemEval-S artifact is not resolved locally")
        loader = LongMemEvalSLoader()
        count = 0
        source_count = 0
        answer_types: set[type[object]] = set()
        for checkpoint in loader.load(path):
            count += 1
            source_count += len(checkpoint.sources)
            self.assertEqual(len(checkpoint.queries), 1)
            answer_types.add(type(checkpoint.queries[0].raw["answer"]))
            provenance = [source.provenance_id for source in checkpoint.sources]
            self.assertEqual(len(provenance), len(set(provenance)))
            scope = resolve_answer_session_scope(
                checkpoint,
                [str(value) for value in checkpoint.raw["answer_session_ids"]],
            )
            self.assertTrue(scope)
        self.assertEqual(count, 500)
        self.assertEqual(source_count, 246_750)
        self.assertEqual(answer_types, {str, int})

    def test_repeated_session_ids_are_occurrence_qualified_and_all_resolve(self) -> None:
        loader = LongMemEvalSLoader()
        first = next(
            loader.load(
                FIXTURES / "longmemeval_s_repeated_session.json",
                verify_artifact=False,
            )
        )
        second = next(
            loader.load(
                FIXTURES / "longmemeval_s_repeated_session.json",
                verify_artifact=False,
            )
        )
        first_ids = tuple(source.provenance_id for source in first.sources)
        self.assertEqual(first_ids, tuple(source.provenance_id for source in second.sources))
        self.assertEqual(len(first_ids), len(set(first_ids)))
        self.assertEqual(len(first_ids), 4)
        self.assertIn("session-occurrence-0000", first_ids[0])
        self.assertIn("session-occurrence-0001", first_ids[2])
        resolved = resolve_answer_session_scope(first, ["session-repeat"])
        self.assertEqual(tuple(source.provenance_id for source in resolved), first_ids)


if __name__ == "__main__":
    unittest.main()
