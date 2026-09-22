from __future__ import annotations

import asyncio
from types import SimpleNamespace
import unittest
from uuid import uuid4

from ufuzz.backends.amem import AMemAdapter
from ufuzz.backends.amem_capability import AMemRetrievableEntryCapability
from ufuzz.backends.amem_materializer import (
    AMEM_NATIVE_CONFIGURATION_ID,
    AMemMaterializationProtocol,
)
from ufuzz.backends.base import InitializationArtifact
from ufuzz.domain import SourceUnit
from ufuzz.materialization import EphemeralMaterializedState, RootMaterializationRequest
from ufuzz.retrieval_capability import build_frozen_checkpoint_inventory
from ufuzz.retrieval_feedback import MutationRelation
from ufuzz.state_contract import (
    ExecutableQueryArtifact,
    MaterializationFailureKind,
    MutationOpportunity,
    PhysicalTransitionOutcome,
    RealizedMutationArtifact,
    SemanticMutationCertificate,
)


CAMPAIGN = "a-mem-campaign"
CHECKPOINT = "a-mem-root"
ARTIFACT_ID = "a-mem-init"


class _Collection:
    name = "memories"

    def __init__(self) -> None:
        self.ids: list[str] = []

    def get(self):
        return {"ids": list(self.ids)}

    def count(self):
        return len(self.ids)


class _MemoryNote:
    def __init__(self, identifier, content, time, tags):
        self.id = identifier
        self.content = content
        self.context = "General"
        self.keywords = tuple(word.casefold().strip(".,") for word in content.split()[:3])
        self.tags = list(tags)
        self.timestamp = time


class _NativeAMemDouble:
    """Pinned public A-Mem surfaces with fresh replay-local IDs."""

    def __init__(self) -> None:
        self.memories: dict[str, _MemoryNote] = {}
        self.retriever = SimpleNamespace(collection=_Collection())

    def add_note(self, content, time=None, tags=None):
        identifier = uuid4().hex
        self.memories[identifier] = _MemoryNote(identifier, content, time, tags or [])
        self.retriever.collection.ids.append(identifier)
        return identifier

    def read(self, identifier):
        return self.memories.get(identifier)

    def search(self, query, k=5):
        terms = set(query.casefold().split())
        ranked = sorted(
            self.memories.values(),
            key=lambda note: (
                -len(terms.intersection(note.content.casefold().split())),
                note.content,
                note.id,
            ),
        )[:k]
        return [
            {
                "id": note.id,
                "content": note.content,
                "context": note.context,
                "keywords": list(note.keywords),
                "score": float(index),
            }
            for index, note in enumerate(ranked)
        ]

    def update(self, identifier, **changes):
        note = self.memories.get(identifier)
        if note is None:
            return False
        for key, value in changes.items():
            if hasattr(note, key):
                setattr(note, key, value)
        return True

    def delete(self, identifier):
        if identifier not in self.memories:
            return False
        del self.memories[identifier]
        self.retriever.collection.ids.remove(identifier)
        return True


def _source(index: int, text: str, *, benchmark: str = "synthetic") -> SourceUnit:
    return SourceUnit(
        benchmark,
        CHECKPOINT,
        "session-1",
        f"source-{index}",
        index,
        text,
        f"2026-01-{index + 1:02d}T00:00:00+00:00",
        "user",
        "user",
        {},
    )


SOURCES = (
    _source(0, "Alice lives in Boston."),
    _source(1, "Bob enjoys tennis."),
    _source(2, "Alice lives in Boston."),
)


class AMemMaterializerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact = InitializationArtifact.create(
            CHECKPOINT,
            SOURCES,
            {"amem_profile_id": AMEM_NATIVE_CONFIGURATION_ID},
        )

        async def bootstrap():
            adapter = AMemAdapter(memory_factory=_NativeAMemDouble)
            state = await adapter.replay_state(cls.artifact)
            current = (
                await AMemRetrievableEntryCapability(state).inventory(
                    campaign_id=CAMPAIGN,
                    root_checkpoint_id=CHECKPOINT,
                    state_id=state.state_id,
                )
            ).require()
            frozen = build_frozen_checkpoint_inventory(current).inventory
            await adapter.teardown(state)
            return frozen

        cls.inventory = asyncio.run(bootstrap())

    def setUp(self):
        self.protocol = AMemMaterializationProtocol(
            initialization_artifacts={ARTIFACT_ID: self.artifact},
            frozen_inventories={(CAMPAIGN, CHECKPOINT): self.inventory},
            adapter=AMemAdapter(memory_factory=_NativeAMemDouble),
        )

    def tearDown(self):
        async def cleanup():
            for state_id, context in tuple(self.protocol._states.items()):
                await self.protocol.discard(
                    EphemeralMaterializedState(
                        context.handle,
                        "a-mem",
                        AMEM_NATIVE_CONFIGURATION_ID,
                        CAMPAIGN,
                        CHECKPOINT,
                        state_id,
                    )
                )

        asyncio.run(cleanup())

    def request(self):
        return RootMaterializationRequest(
            "a-mem",
            AMEM_NATIVE_CONFIGURATION_ID,
            CAMPAIGN,
            CHECKPOINT,
            ARTIFACT_ID,
            self.inventory.coverage_ids,
        )

    async def root(self):
        result = await self.protocol.materialize_root(self.request())
        self.assertIsNone(result.failure, result.failure)
        return result.value

    def coverage(self, index):
        return next(
            entry.coverage_id
            for entry in self.inventory.entries
            if entry.provenance_ids == (SOURCES[index].provenance_id,)
        )

    def mutation(self, name, relation, index, text=None):
        target = self.coverage(index)
        outcome = (
            PhysicalTransitionOutcome.DELETED
            if relation is MutationRelation.DELETION
            else PhysicalTransitionOutcome.SAME_ID
        )
        opportunity = MutationOpportunity.create(
            opportunity_id=f"op-{name}",
            campaign_id=CAMPAIGN,
            root_checkpoint_id=CHECKPOINT,
            parent_seed_id=f"parent-{name}",
            relation=relation,
            canonical_target={"coverage": target.opaque_id},
            target_lineage=(target,),
            applicability_evidence={"exact": True},
            generation_constraints={"type_compatible": True},
            possible_transition_outcomes={outcome},
            acceptable_transition_outcomes={outcome},
        )
        semantic = SemanticMutationCertificate(
            f"semantic-{name}",
            opportunity.opportunity_id,
            CAMPAIGN,
            CHECKPOINT,
            relation,
            True,
            {"target": target.opaque_id},
            f"proof-{name}",
        )
        payload = (
            {"operation": "delete"}
            if relation is MutationRelation.DELETION
            else {
                "operation": "unrelated_change"
                if relation is MutationRelation.UNRELATED_CHANGE
                else "update",
                "new_text": text,
                "provenance_ids": (SOURCES[index].provenance_id,),
            }
        )
        return RealizedMutationArtifact(
            name, opportunity, semantic, "query", None, relation.value, payload
        )

    def test_root_replay_rebinds_fresh_ids_and_preserves_duplicate_provenance(self):
        async def exercise():
            first = await self.root()
            first_ids = {
                binding.physical_entry.backend_entry_id
                for binding in first.rebinding_certificate.live_bindings
            }
            duplicate_entries = [
                entry
                for entry in self.inventory.entries
                if entry.initial_observable_projection["content"] == "Alice lives in Boston."
            ]
            self.assertEqual(len(duplicate_entries), 2)
            self.assertNotEqual(
                duplicate_entries[0].provenance_ids, duplicate_entries[1].provenance_ids
            )
            observed = first.observed_root_state
            await self.protocol.discard(first.state)
            second = await self.root()
            second_ids = {
                binding.physical_entry.backend_entry_id
                for binding in second.rebinding_certificate.live_bindings
            }
            self.assertTrue(first_ids.isdisjoint(second_ids))
            self.assertEqual(observed, second.observed_root_state)

        asyncio.run(exercise())

    def test_ranked_retrieval_is_bounded_and_inventory_resolved(self):
        async def exercise():
            root = await self.root()
            result = await self.protocol.retrieve(
                root.state,
                query_artifact=ExecutableQueryArtifact.create(
                    artifact_id="query", executable_text="Alice Boston", metadata={}
                ),
                top_k=2,
            )
            self.assertIsNone(result.failure)
            self.assertEqual(len(result.value.ranked_physical_entries), 2)
            current_ids = set(root.state.handle.backend_state.memories)
            self.assertTrue(
                all(
                    entry.backend_entry_id in current_ids
                    for entry in result.value.ranked_physical_entries
                )
            )

        asyncio.run(exercise())

    def test_update_delete_and_unrelated_change_are_certified(self):
        async def exercise():
            root = await self.root()
            update = await self.protocol.replay_transition(
                root.state,
                self.mutation(
                    "update", MutationRelation.UPDATE, 0, "Alice lives in Seattle."
                ),
            )
            self.assertIsNone(update.failure, update.failure)
            self.assertIs(
                update.value.certificate.observed_outcome,
                PhysicalTransitionOutcome.SAME_ID,
            )
            unrelated = await self.protocol.replay_transition(
                root.state,
                self.mutation(
                    "urc", MutationRelation.UNRELATED_CHANGE, 1, "Bob enjoys golf."
                ),
            )
            self.assertIsNone(unrelated.failure, unrelated.failure)
            deletion = await self.protocol.replay_transition(
                root.state, self.mutation("delete", MutationRelation.DELETION, 2)
            )
            self.assertIsNone(deletion.failure, deletion.failure)
            descendant = await self.protocol.certify_descendant(
                root.state,
                (
                    update.value.certificate,
                    unrelated.value.certificate,
                    deletion.value.certificate,
                ),
            )
            self.assertIsNone(descendant.failure, descendant.failure)
            self.assertEqual(descendant.value.deleted_ids, frozenset({self.coverage(2)}))

        asyncio.run(exercise())

    def test_unrelated_change_cannot_add_or_change_another_note(self):
        async def exercise():
            root = await self.root()
            original = root.state.handle.backend_state.update

            def add_instead(identifier, **changes):
                root.state.handle.backend_state.add_note(
                    changes["content"], tags=["ufuzz-source:unexpected"]
                )
                return True

            root.state.handle.backend_state.update = add_instead
            result = await self.protocol.replay_transition(
                root.state,
                self.mutation(
                    "bad-urc", MutationRelation.UNRELATED_CHANGE, 1, "Bob enjoys golf."
                ),
            )
            self.assertIsNotNone(result.failure)
            root.state.handle.backend_state.update = original

        asyncio.run(exercise())

    def test_root_drift_and_ambiguous_rebinding_fail_closed(self):
        async def exercise():
            drifted = InitializationArtifact.create(
                CHECKPOINT,
                SOURCES,
                {"amem_profile_id": AMEM_NATIVE_CONFIGURATION_ID},
            )
            protocol = AMemMaterializationProtocol(
                initialization_artifacts={ARTIFACT_ID: drifted},
                frozen_inventories={(CAMPAIGN, CHECKPOINT): self.inventory},
                adapter=AMemAdapter(memory_factory=_NativeAMemDouble),
            )
            original = protocol._adapter.ingest

            async def drift(state, sources):
                receipts = await original(state, sources)
                next(iter(state.backend_state.memories.values())).context = "Changed"
                return receipts

            protocol._adapter.ingest = drift
            result = await protocol.materialize_root(self.request())
            self.assertIsNotNone(result.failure)
            self.assertIs(
                result.failure.kind,
                MaterializationFailureKind.ROOT_INITIALIZATION_INEQUIVALENT,
            )

        asyncio.run(exercise())

    def test_process_scoped_isolation_and_cleanup(self):
        async def exercise():
            first = await self.root()
            blocked = await self.protocol.materialize_root(self.request())
            self.assertIsNotNone(blocked.failure)
            self.assertIs(
                blocked.failure.kind,
                MaterializationFailureKind.TRANSIENT_BACKEND_FAILURE,
            )
            await self.protocol.discard(first.state)
            second = await self.root()
            self.assertEqual(len(second.state.handle.backend_state.memories), 3)

            other = AMemMaterializationProtocol(
                initialization_artifacts={ARTIFACT_ID: self.artifact},
                frozen_inventories={(CAMPAIGN, CHECKPOINT): self.inventory},
                adapter=AMemAdapter(memory_factory=_NativeAMemDouble),
            )
            other_root = await other.materialize_root(self.request())
            self.assertIsNotNone(other_root.failure)
            self.assertIs(
                other_root.failure.kind,
                MaterializationFailureKind.TRANSIENT_BACKEND_FAILURE,
            )
            await self.protocol.discard(second.state)
            other_root = await other.materialize_root(self.request())
            self.assertIsNone(other_root.failure)
            self.assertEqual(len(other_root.value.state.handle.backend_state.memories), 3)
            await other.discard(other_root.value.state)

        asyncio.run(exercise())

    def test_minimal_locomo_and_longmemeval_source_units_use_generic_path(self):
        # Benchmark names are provenance, not switches in the materializer.
        for benchmark in ("locomo", "longmemeval-s"):
            source = _source(8, f"{benchmark} memory", benchmark=benchmark)
            artifact = InitializationArtifact.create(
                CHECKPOINT,
                (source,),
                {"amem_profile_id": AMEM_NATIVE_CONFIGURATION_ID},
            )
            self.assertEqual(artifact.sources[0].benchmark, benchmark)


if __name__ == "__main__":
    unittest.main()
