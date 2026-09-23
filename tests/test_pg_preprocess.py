from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest

from ufuzz.pg_preprocess import (
    INFORMATION_FLOW_POLICY,
    PG_PREPROCESS_V1,
    SEARCH_FORBIDDEN_FIELDS,
    VALIDATION_RULES,
    pg_preprocess_v1_manifest,
    prompt_digests,
)
from ufuzz.semantic_sidecar import canonical_sha256


class PGPreprocessCandidateTests(unittest.TestCase):
    def test_manifest_is_canonical_and_has_golden_digest(self) -> None:
        manifest = pg_preprocess_v1_manifest()
        self.assertEqual(manifest.contract_label, PG_PREPROCESS_V1)
        self.assertEqual(
            manifest.sha256_digest,
            "c342c97cdebdf89306d06a1c12ceb8702a9a6c8d5f98e0e6a821f2163a07b174",
        )
        self.assertEqual(manifest.canonical_bytes, pg_preprocess_v1_manifest().canonical_bytes)
        with self.assertRaises(FrozenInstanceError):
            manifest.contract_label = "changed"

    def test_exact_prompt_digests_and_runtime_binding(self) -> None:
        manifest = pg_preprocess_v1_manifest()
        self.assertEqual(manifest.prompt_sha256, prompt_digests())
        self.assertEqual(len(manifest.prompt_sha256), 7)
        self.assertEqual(
            manifest.model["revision"],
            "40c069824f4251a91eefaf281ebe4c544efd3e18",
        )
        self.assertFalse(manifest.model["do_sample"])
        self.assertFalse(manifest.model["enable_thinking"])
        self.assertEqual(manifest.model["retry_policy"], "none")

    def test_final_p2r3_rules_are_part_of_candidate_identity(self) -> None:
        manifest = pg_preprocess_v1_manifest()
        self.assertEqual(
            manifest.validation_rule_sha256, canonical_sha256(VALIDATION_RULES)
        )
        self.assertEqual(
            manifest.information_flow_policy_sha256,
            canonical_sha256(INFORMATION_FLOW_POLICY),
        )
        self.assertIn("unknown", manifest.rules["placeholder_precheck"]["reject_after_normalization"])
        self.assertTrue(
            manifest.rules["preference_holder"]["reject_object_only_subject"]
        )
        self.assertIn("gold_answer", SEARCH_FORBIDDEN_FIELDS)
        self.assertTrue(manifest.rules["no_global_completeness_claim"])


if __name__ == "__main__":
    unittest.main()
