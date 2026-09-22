from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
import tomllib
import unittest

from ufuzz.backends import AMemAdapter, GraphitiAdapter, InitializationArtifact, Mem0Adapter, MemosAdapter
from ufuzz.benchmarks import LoCoMoLoader, LongMemEvalSLoader
from ufuzz.structural import (
    StructuralIndexBuilder,
    certify_exact_source_inventory,
    exact_source_fact,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "locomo_minimal.json"


class _Mem0Double:
    """Public-surface test double; it is not reported as a backend smoke run."""

    def __init__(self) -> None:
        self.entries: dict[str, dict] = {}
        self.next_id = 0

    def add(self, messages, **kwargs):
        self.next_id += 1
        local_id = f"m{self.next_id}"
        self.entries[local_id] = {
            "id": local_id,
            "memory": messages[0]["content"],
            "metadata": kwargs.get("metadata", {}),
        }
        return {"results": [{"id": local_id, "event": "ADD"}]}

    def search(self, query, **kwargs):
        return {"results": list(self.entries.values())[: kwargs["top_k"]]}

    def get_all(self, **kwargs):
        return {"results": list(self.entries.values())}

    def get(self, local_id):
        return self.entries.get(local_id)

    def update(self, local_id, text=None, metadata=None, **kwargs):
        self.entries[local_id]["memory"] = text
        if metadata:
            self.entries[local_id]["metadata"].update(metadata)
        return {"message": "updated"}

    def delete(self, local_id):
        self.entries.pop(local_id, None)
        return {"message": "deleted"}

    def reset(self):
        self.entries.clear()


class _AMemDouble:
    def __init__(self) -> None:
        self.memories: dict[str, SimpleNamespace] = {}
        self.next_id = 0

    def add_note(self, content, time=None, **kwargs):
        self.next_id += 1
        local_id = f"a{self.next_id}"
        self.memories[local_id] = SimpleNamespace(
            id=local_id,
            content=content,
            timestamp=time,
            tags=kwargs.get("tags", []),
        )
        return local_id

    def search(self, query, k=5):
        return [
            {"id": local_id, "content": note.content, "score": float(rank)}
            for rank, (local_id, note) in enumerate(self.memories.items())
        ][:k]

    def read(self, local_id):
        return self.memories.get(local_id)

    def update(self, local_id, **kwargs):
        note = self.memories.get(local_id)
        if note is None:
            return False
        for key, value in kwargs.items():
            setattr(note, key, value)
        return True

    def delete(self, local_id):
        return self.memories.pop(local_id, None) is not None


class _GraphitiDouble:
    def __init__(self) -> None:
        self.episodes: dict[str, SimpleNamespace] = {}

    async def add_episode(self, **kwargs):
        episode_uuid = kwargs["uuid"]
        edge = SimpleNamespace(
            uuid=f"edge:{episode_uuid}",
            fact=kwargs["episode_body"],
            episodes=[episode_uuid],
            group_id=kwargs["group_id"],
            valid_at=kwargs["reference_time"],
            invalid_at=None,
            expired_at=None,
        )
        self.episodes[episode_uuid] = edge
        return SimpleNamespace(edges=[edge], episode=SimpleNamespace(uuid=episode_uuid))

    async def search(self, query, group_ids=None, num_results=10):
        return [
            edge
            for edge in self.episodes.values()
            if edge.group_id in set(group_ids or [])
        ][:num_results]

    async def get_nodes_and_edges_by_episode(self, episode_ids):
        return SimpleNamespace(
            edges=[self.episodes[value] for value in episode_ids if value in self.episodes],
            nodes=[],
        )

    async def remove_episode(self, episode_uuid):
        self.episodes.pop(episode_uuid, None)


class BackendContractTests(unittest.TestCase):
    def test_backend_dependencies_are_separate_optional_groups(self) -> None:
        with (ROOT / "pyproject.toml").open("rb") as stream:
            project = tomllib.load(stream)["project"]
        self.assertEqual(project["dependencies"], [])
        extras = project["optional-dependencies"]
        self.assertEqual(extras["mem0"], ["mem0ai==2.0.12"])
        self.assertTrue(
            any(
                "ceffb860f0712bbae97b184d440df62bc910ca8d" in requirement
                for requirement in extras["amem"]
            )
        )
        self.assertIn("graphiti-core==0.30.2", extras["graphiti"])
        self.assertIn("neo4j==5.26.0", extras["graphiti"])
        self.assertTrue(any(
            "78a372a4fc853a24d2a78efa3b4bbbd27ab9f7ad" in requirement
            for requirement in extras["memos"]
        ))
        self.assertIn("qdrant-client==1.16.2", extras["memos"])

    def test_initialization_artifact_digest_is_stable(self) -> None:
        checkpoint = next(LoCoMoLoader().load(FIXTURE, verify_artifact=False))
        first = InitializationArtifact.create(
            checkpoint.checkpoint_id, checkpoint.sources, {"model": "fixed"}
        )
        second = InitializationArtifact.create(
            checkpoint.checkpoint_id, checkpoint.sources, {"model": "fixed"}
        )
        self.assertEqual(first.digest, second.digest)

    def test_initialization_artifact_excludes_evaluator_annotations(self) -> None:
        fixture = ROOT / "tests" / "fixtures" / "longmemeval_s_minimal.json"
        checkpoint = next(LongMemEvalSLoader().load(fixture, verify_artifact=False))
        self.assertIn("has_answer", checkpoint.sources[0].raw)
        artifact = InitializationArtifact.create(
            checkpoint.checkpoint_id,
            checkpoint.sources,
            {"model": "fixed"},
        )
        self.assertEqual(artifact.sources[0].raw, {})

    def test_capability_reports_are_explicit(self) -> None:
        reports = [
            Mem0Adapter().capabilities(),
            AMemAdapter().capabilities(),
            GraphitiAdapter().capabilities(),
            MemosAdapter().capabilities(),
        ]
        self.assertEqual({r.backend for r in reports}, {"mem0", "a-mem", "graphiti", "memos"})
        for report in reports:
            self.assertIn("initialization_equivalence", report.operations)
            self.assertIn("provenance", report.operations)

    def test_mem0_adapter_public_contract_with_test_double(self) -> None:
        async def exercise() -> None:
            checkpoint = next(LoCoMoLoader().load(FIXTURE, verify_artifact=False))
            artifact = InitializationArtifact.create(
                checkpoint.checkpoint_id, checkpoint.sources, {"infer": False}
            )
            adapter = Mem0Adapter(memory_factory=lambda _: _Mem0Double(), infer=False)
            state = await adapter.create_isolated_state(artifact)
            config = state.metadata["config"]
            vector_config = config["vector_store"]["config"]
            self.assertTrue(
                str(vector_config["path"]).startswith(str(adapter.root_dir))
            )
            self.assertTrue(
                str(config["history_db_path"]).startswith(str(adapter.root_dir))
            )
            self.assertTrue(vector_config["collection_name"].startswith("ufuzz_"))
            receipts = await adapter.ingest(state, checkpoint.sources)
            self.assertEqual(len(receipts), 2)
            retrieved = await adapter.retrieve(state, "Alice", 2)
            self.assertEqual(retrieved[0].provenance_ids, (checkpoint.sources[0].provenance_id,))
            target = retrieved[0].local_id
            update = await adapter.update(
                state,
                target,
                "Alice residence Seattle.",
                provenance_ids=retrieved[0].provenance_ids,
            )
            self.assertTrue(update.success)
            unrelated = await adapter.unrelated_existing_region_change(
                state,
                retrieved[1].local_id,
                "Bob hobby golf.",
                provenance_ids=retrieved[1].provenance_ids,
            )
            self.assertTrue(unrelated.success)
            deletion = await adapter.delete(state, target)
            self.assertTrue(deletion.success)
            await adapter.reset(state)
            self.assertEqual(await adapter.observable_state(state), [])
            await adapter.teardown(state)

        asyncio.run(exercise())

    def test_mem0_rejects_external_storage_by_default(self) -> None:
        async def exercise() -> None:
            checkpoint = next(LoCoMoLoader().load(FIXTURE, verify_artifact=False))
            artifact = InitializationArtifact.create(
                checkpoint.checkpoint_id,
                checkpoint.sources,
                {"infer": False},
            )
            adapter = Mem0Adapter(
                config={
                    "vector_store": {
                        "provider": "qdrant",
                        "config": {"url": "https://example.invalid"},
                    }
                },
                memory_factory=lambda _: _Mem0Double(),
                infer=False,
            )
            with self.assertRaisesRegex(ValueError, "external Qdrant"):
                await adapter.create_isolated_state(artifact)

        asyncio.run(exercise())

    def test_amem_adapter_public_contract_with_test_double(self) -> None:
        async def exercise() -> None:
            checkpoint = next(LoCoMoLoader().load(FIXTURE, verify_artifact=False))
            artifact = InitializationArtifact.create(
                checkpoint.checkpoint_id, checkpoint.sources, {"model": "fixed"}
            )
            adapter = AMemAdapter(memory_factory=_AMemDouble)
            state = await adapter.create_isolated_state(artifact)
            receipts = await adapter.ingest(state, checkpoint.sources)
            self.assertEqual(len(receipts), 2)
            retrieved = await adapter.retrieve(state, "Alice", 2)
            self.assertEqual(retrieved[0].provenance_ids, (checkpoint.sources[0].provenance_id,))
            updated = await adapter.update(
                state,
                retrieved[0].local_id,
                "Alice residence Seattle.",
                provenance_ids=retrieved[0].provenance_ids,
            )
            self.assertTrue(updated.success)
            unrelated = await adapter.unrelated_existing_region_change(
                state,
                retrieved[1].local_id,
                "Bob hobby golf.",
                provenance_ids=retrieved[1].provenance_ids,
            )
            self.assertTrue(unrelated.success)
            deleted = await adapter.delete(state, retrieved[0].local_id)
            self.assertTrue(deleted.success)
            await adapter.reset(state)
            self.assertEqual(await adapter.observable_state(state), [])
            await adapter.teardown(state)

        asyncio.run(exercise())

    def test_graphiti_adapter_public_contract_with_test_double(self) -> None:
        async def exercise() -> None:
            checkpoint = next(LoCoMoLoader().load(FIXTURE, verify_artifact=False))
            artifact = InitializationArtifact.create(
                checkpoint.checkpoint_id, checkpoint.sources, {"model": "fixed"}
            )
            deletion_fact, fact_certificate = exact_source_fact(
                source=checkpoint.sources[1],
                entity="Bob",
                relation="acknowledges",
                value="Alice",
            )
            alice_fact, alice_certificate = exact_source_fact(
                source=checkpoint.sources[0],
                entity="Alice",
                relation="residence",
                value="Boston",
            )
            bob_fact, bob_certificate = exact_source_fact(
                source=checkpoint.sources[0],
                entity="Bob",
                relation="hobby",
                value="tennis",
            )
            inventory_certificate = certify_exact_source_inventory(
                source=checkpoint.sources[1],
                facts=(deletion_fact,),
            )
            index = StructuralIndexBuilder().build(
                checkpoint,
                facts=(alice_fact, bob_fact, deletion_fact),
                certificates=(
                    alice_certificate,
                    bob_certificate,
                    fact_certificate,
                    inventory_certificate,
                ),
            )
            adapter = GraphitiAdapter(graphiti=_GraphitiDouble())
            state = await adapter.create_isolated_state(artifact)
            receipts = await adapter.ingest(state, checkpoint.sources)
            self.assertEqual(len(receipts), 2)
            retrieved = await adapter.retrieve(state, "Alice", 2)
            self.assertEqual(retrieved[0].provenance_ids, (checkpoint.sources[0].provenance_id,))
            updated = await adapter.update(
                state,
                retrieved[0].local_id,
                "Alice residence Seattle.",
                provenance_ids=retrieved[0].provenance_ids,
            )
            self.assertTrue(updated.success)
            unrelated = await adapter.unrelated_existing_region_change(
                state,
                retrieved[1].local_id,
                "Bob hobby golf.",
                provenance_ids=retrieved[1].provenance_ids,
            )
            self.assertTrue(unrelated.success)
            with self.assertRaisesRegex(ValueError, "exactly the intended fact"):
                await adapter.certify_episode_deletion(
                    state,
                    receipts[0].target_local_id or "",
                    alice_fact.fact_id,
                    index,
                )
            bypass = await adapter.delete(
                state,
                receipts[1].target_local_id or "",
                context={"faithful_atomic": True},
            )
            self.assertFalse(bypass.success)
            deletion_certificate = await adapter.certify_episode_deletion(
                state,
                receipts[1].target_local_id or "",
                deletion_fact.fact_id,
                index,
            )
            deleted = await adapter.delete(
                state,
                receipts[1].target_local_id or "",
                context={"deletion_certificate": deletion_certificate},
            )
            self.assertTrue(deleted.success)
            await adapter.reset(state)
            self.assertEqual(await adapter.observable_state(state), [])
            await adapter.teardown(state)

        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
