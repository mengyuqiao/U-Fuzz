from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import unittest

from tests.memos_test_support import GeneralTextDouble, item_factory
from ufuzz.backends.base import InitializationArtifact
from ufuzz.backends.memos import (
    MEMOS_EMBEDDING_MANIFEST_DIGEST, MemosAdapter, MemosProfileError,
    _NonCallingExtractor,
)
from ufuzz.backends.memos_capability import MemosRetrievableEntryCapability
from ufuzz.domain import SourceUnit
from ufuzz.benchmarks import LoCoMoLoader, LongMemEvalSLoader


def source(index, text):
    return SourceUnit("synthetic", "root", "session", f"source-{index}", index,
                      text, "2025-01-01T00:00:00+00:00", "user", "user", {})


SOURCES = (
    source(0, "Alice lives in Boston."),
    source(1, "Alice lives in Boston."),
    source(2, "Bob enjoys tennis."),
)


class MemosCapabilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.adapter = MemosAdapter(
            root_dir=self.tmp.name,
            memory_factory=GeneralTextDouble,
            item_factory=item_factory,
            embedding_identity_resolver=lambda: MEMOS_EMBEDDING_MANIFEST_DIGEST,
            runtime_identity_verifier=lambda: None,
        )
        self.artifact = InitializationArtifact.create(
            "root", SOURCES, {"memos_profile_id": self.adapter.capabilities().notes[0]}
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_inventory_duplicates_retrieval_update_delete_and_unrelated_change(self):
        async def exercise():
            state = await self.adapter.create_isolated_state(self.artifact)
            receipts = await self.adapter.ingest(state, SOURCES)
            self.assertEqual(len(receipts), 3)
            capability = MemosRetrievableEntryCapability(state)
            inventory = (await capability.inventory(
                campaign_id="campaign", root_checkpoint_id="root", state_id=state.state_id
            )).require()
            self.assertEqual(len(inventory.entries), 3)
            duplicates = [entry for entry in inventory.entries
                          if entry.projection.observable["memory"] == "Alice lives in Boston."]
            self.assertEqual(len(duplicates), 2)
            self.assertNotEqual(duplicates[0].provenance_ids, duplicates[1].provenance_ids)
            ranked = (await capability.retrieve(
                campaign_id="campaign", root_checkpoint_id="root", state_id=state.state_id,
                query="Alice Boston", top_k=2,
            )).require()
            self.assertEqual(len(ranked.ranked_physical_entries), 2)
            alice = receipts[0].affected_local_ids[0]
            bob = receipts[2].affected_local_ids[0]
            await self.adapter.update(state, alice, "Alice lives in Seattle.",
                                      provenance_ids=receipts[0].provenance_ids)
            self.assertEqual(state.backend_state.get(alice).memory, "Alice lives in Seattle.")
            count = len(state.backend_state.get_all())
            await self.adapter.unrelated_existing_region_change(
                state, bob, "Bob enjoys golf.", provenance_ids=receipts[2].provenance_ids)
            self.assertEqual(len(state.backend_state.get_all()), count)
            await self.adapter.delete(state, alice)
            self.assertIsNone(state.backend_state.get(alice))
            state_dir = Path(state.metadata["state_dir"])
            await self.adapter.teardown(state)
            self.assertFalse(state_dir.exists())
            await self.adapter.teardown(state)
        asyncio.run(exercise())

    def test_root_replay_projection_and_isolation(self):
        async def exercise():
            a = await self.adapter.replay_state(self.artifact)
            with self.assertRaisesRegex(MemosProfileError, "one live state"):
                await self.adapter.replay_state(self.artifact)
            other = MemosAdapter(
                root_dir=self.tmp.name,
                memory_factory=GeneralTextDouble,
                item_factory=item_factory,
                embedding_identity_resolver=lambda: MEMOS_EMBEDDING_MANIFEST_DIGEST,
                runtime_identity_verifier=lambda: None,
            )
            b = await other.replay_state(self.artifact)
            self.assertEqual(await self.adapter.observable_state(a), await self.adapter.observable_state(b))
            target = a.backend_state.get_all()[0]
            await self.adapter.update(a, target.id, "changed", provenance_ids=(SOURCES[0].provenance_id,))
            self.assertNotEqual(await self.adapter.observable_state(a), await self.adapter.observable_state(b))
            b_dir = Path(b.metadata["state_dir"])
            await self.adapter.teardown(a)
            self.assertTrue(b_dir.exists())
            await other.teardown(b)
        asyncio.run(exercise())

    def test_embedding_guard_and_native_chat_firewall(self):
        bad = MemosAdapter(root_dir=self.tmp.name, memory_factory=GeneralTextDouble,
                           item_factory=item_factory, embedding_identity_resolver=lambda: "f" * 64,
                           runtime_identity_verifier=lambda: None)
        with self.assertRaisesRegex(MemosProfileError, "embedding digest mismatch"):
            asyncio.run(bad.create_isolated_state(self.artifact))
        with self.assertRaisesRegex(MemosProfileError, "disabled"):
            _NonCallingExtractor().generate("prompt")
        self.assertEqual(GeneralTextDouble.extract_calls, 0)

    def test_capability_rejects_search_object_absent_from_inventory(self):
        async def exercise():
            state = await self.adapter.replay_state(self.artifact)
            original = state.backend_state.search
            state.backend_state.search = lambda q, k: [item_factory("unknown", SOURCES and {
                "info": {"ufuzz_provenance_v1": {"schema": "ufuzz_provenance_v1", "provenance_id": "x"}}
            })]
            result = await MemosRetrievableEntryCapability(state).retrieve(
                campaign_id="c", root_checkpoint_id="root", state_id=state.state_id,
                query="q", top_k=1)
            self.assertIsNotNone(result.blocker)
            state.backend_state.search = original
            await self.adapter.teardown(state)
        asyncio.run(exercise())

    def test_generic_locomo_and_longmemeval_source_units_materialize_directly(self):
        root = Path(__file__).resolve().parents[1]
        checkpoints = (
            next(LoCoMoLoader().load(root / "tests/fixtures/locomo_minimal.json", verify_artifact=False)),
            next(LongMemEvalSLoader().load(
                root / "tests/fixtures/longmemeval_s_minimal.json", verify_artifact=False
            )),
        )
        async def exercise():
            for checkpoint in checkpoints:
                artifact = InitializationArtifact.create(
                    checkpoint.checkpoint_id, checkpoint.sources,
                    {"memos_profile_id": "generic-checkpoint-interface"},
                )
                state = await self.adapter.replay_state(artifact)
                inventory = state.backend_state.get_all()
                self.assertEqual(len(inventory), len(checkpoint.sources))
                self.assertEqual(
                    {entry.metadata["info"]["ufuzz_provenance_v1"]["provenance_id"] for entry in inventory},
                    {source.provenance_id for source in checkpoint.sources},
                )
                await self.adapter.teardown(state)
        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
