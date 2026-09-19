from __future__ import annotations

from pathlib import Path
import unittest

from ufuzz.benchmarks import LoCoMoLoader, LongMemEvalSLoader


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
            "longmemeval-s-cleaned:q-test:session-1:turn-0000",
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
        answer_types: set[type[object]] = set()
        for checkpoint in loader.load(path):
            count += 1
            self.assertEqual(len(checkpoint.queries), 1)
            answer_types.add(type(checkpoint.queries[0].raw["answer"]))
        self.assertEqual(count, 500)
        self.assertEqual(answer_types, {str, int})


if __name__ == "__main__":
    unittest.main()
