from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace
import unittest

from ufuzz.backends.amem import AMEM_COMMIT, AMemAdapter
from ufuzz.backends.amem_capability import (
    AMEM_PUBLIC_RETRIEVAL_API,
    AMEM_RETRIEVAL_OBJECT_CLASS,
    AMemRetrievableEntryCapability,
    amem_observable_projection,
)
from ufuzz.backends.base import InitializationArtifact, OperationReceipt, StateHandle
from ufuzz.coverage import LineageResolutionError, PhysicalEntryRef
from ufuzz.retrieval_capability import (
    assemble_campaign_coverage,
    build_frozen_checkpoint_inventory,
    compare_initialization_inventories,
    resolve_fuzzing_retrieval,
)
from ufuzz.retrieval_feedback import (
    BehaviorHistory,
    MutationRelation,
    score_observed_retrieval,
)


class _Note:
    """Pinned MemoryNote public shape used without importing A-Mem."""

    def __init__(
        self,
        note_id: str,
        content: str,
        *,
        context: str = "General",
        keywords: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        self.id = note_id
        self.content = content
        self.context = context
        self.keywords = list(keywords or [])
        self.tags = list(tags or [])
        self.links: list[str] = []
        self.category = "Uncategorized"
        self.timestamp = "202601010000"
        self.last_accessed = "202601010001"
        self.retrieval_count = 0
        self.evolution_history: list[object] = []


class _Collection:
    def __init__(self, ids: list[str]) -> None:
        self.name = "memories"
        self.ids = list(ids)
        self.snapshots: list[list[str]] = []
        self.reported_count: int | None = None

    def get(self):
        if self.snapshots:
            self.ids = list(self.snapshots.pop(0))
        return {"ids": list(self.ids)}

    def count(self):
        return len(self.ids) if self.reported_count is None else self.reported_count

    def add(self, note_id: str) -> None:
        if note_id not in self.ids:
            self.ids.append(note_id)

    def delete(self, note_id: str) -> None:
        if note_id in self.ids:
            self.ids.remove(note_id)


class _AMemPublicDouble:
    """Source-faithful memories/Chroma/search/update/delete behavior."""

    def __init__(self, notes: list[_Note]) -> None:
        self.memories = {note.id: note for note in notes}
        self.retriever = SimpleNamespace(collection=_Collection(list(self.memories)))
        self.search_ids = list(self.memories)

    def search(self, query: str, k: int = 5):
        values = []
        for rank, note_id in enumerate(self.search_ids[:k]):
            note = self.memories.get(note_id)
            if note is None:
                continue
            values.append(
                {
                    "id": note_id,
                    "content": note.content,
                    "context": note.context,
                    "keywords": list(note.keywords),
                    "score": rank + 0.25,
                }
            )
        return values

    def read(self, note_id: str):
        return self.memories.get(note_id)

    def update(self, note_id: str, **kwargs):
        note = self.memories.get(note_id)
        if note is None:
            return False
        for key, value in kwargs.items():
            if hasattr(note, key):
                setattr(note, key, value)
        # Pinned source deletes and re-adds the Chroma document under same ID.
        self.retriever.collection.delete(note_id)
        self.retriever.collection.add(note_id)
        return True

    def delete(self, note_id: str):
        if note_id not in self.memories:
            return False
        self.retriever.collection.delete(note_id)
        del self.memories[note_id]
        return True


def _state(system: _AMemPublicDouble, *, state_id: str = "state-1") -> StateHandle:
    return StateHandle(
        backend="a-mem",
        state_id=state_id,
        checkpoint_id="checkpoint-1",
        initialization_digest="digest",
        backend_state=system,
        metadata={"artifact": "synthetic"},
    )


async def _inventory(
    capability: AMemRetrievableEntryCapability,
    *,
    campaign_id: str = "campaign-1",
):
    return (
        await capability.inventory(
            campaign_id=campaign_id,
            root_checkpoint_id="checkpoint-1",
            state_id="state-1",
        )
    ).require()


def _receipt(
    operation: str,
    note_id: str,
    *,
    after: object,
    success: bool = True,
) -> OperationReceipt:
    return OperationReceipt(
        backend="a-mem",
        state_id="state-1",
        operation=operation,
        success=success,
        target_local_id=note_id,
        affected_local_ids=(note_id,),
        provenance_ids=(),
        before={"id": note_id},
        after=after,
        raw={"success": success},
    )


class AMemCapabilityTests(unittest.TestCase):
    def test_frozen_pin_and_selected_public_object_class(self) -> None:
        self.assertEqual(AMEM_COMMIT, "ceffb860f0712bbae97b184d440df62bc910ca8d")

        async def exercise() -> None:
            note = _Note("a-1", "memory")
            inventory = await _inventory(
                AMemRetrievableEntryCapability(_state(_AMemPublicDouble([note])))
            )
            self.assertEqual(inventory.selected_public_retrieval_api, AMEM_PUBLIC_RETRIEVAL_API)
            self.assertEqual(inventory.retrieval_object_class, AMEM_RETRIEVAL_OBJECT_CLASS)
            self.assertEqual(inventory.entries[0].physical_entry.backend_entry_id, "a-1")

        asyncio.run(exercise())

    def test_projection_is_exact_public_query_independent_state(self) -> None:
        note = _Note(
            "a-1",
            "  Exact Content  ",
            context="Exact Context",
            keywords=["Beta", "alpha"],
            tags=["ufuzz-source:p1", "private-tag"],
        )
        projection = amem_observable_projection(note)
        self.assertEqual(
            dict(projection.observable),
            {
                "content": "  Exact Content  ",
                "context": "Exact Context",
                "keywords": ("Beta", "alpha"),
            },
        )
        for excluded in (
            "id",
            "score",
            "rank",
            "tags",
            "links",
            "category",
            "timestamp",
            "last_accessed",
            "retrieval_count",
            "evolution_history",
        ):
            self.assertNotIn(excluded, projection.observable)
        public_first = amem_observable_projection(
            {
                "id": "a-1",
                "content": "same",
                "context": "General",
                "keywords": ["k"],
                "score": 0.1,
            }
        )
        public_second = amem_observable_projection(
            {
                "id": "different-physical-id",
                "content": "same",
                "context": "General",
                "keywords": ["k"],
                "score": 999.0,
            }
        )
        self.assertEqual(public_first, public_second)

    def test_keyword_order_is_semantic_and_duplicates_are_preserved(self) -> None:
        ordered = amem_observable_projection(
            _Note("a-1", "memory", keywords=["primary", "secondary", "primary"])
        )
        reordered = amem_observable_projection(
            _Note("a-2", "memory", keywords=["secondary", "primary", "primary"])
        )
        self.assertEqual(
            ordered.observable["keywords"],
            ("primary", "secondary", "primary"),
        )
        self.assertNotEqual(ordered, reordered)

    def test_projection_rejects_malformed_public_fields(self) -> None:
        async def exercise() -> None:
            malformed = (
                {"content": None},
                {"context": 7},
                {"keywords": "single-string"},
                {"keywords": ["valid", 3]},
            )
            for index, changes in enumerate(malformed):
                note = _Note(f"a-{index}", "memory", keywords=["valid"])
                for field, value in changes.items():
                    setattr(note, field, value)
                result = await AMemRetrievableEntryCapability(
                    _state(_AMemPublicDouble([note]))
                ).inventory(
                    campaign_id="campaign-1",
                    root_checkpoint_id="checkpoint-1",
                    state_id="state-1",
                )
                self.assertFalse(result.supported)
                self.assertEqual(result.blocker.operation.value, "projection")

        asyncio.run(exercise())

    def test_memory_note_and_public_search_result_projection_agree(self) -> None:
        note = _Note(
            "stored-id",
            "Exact content",
            context="Exact context",
            keywords=["primary", "secondary", "primary"],
            tags=["ufuzz-source:source-1"],
        )
        result = {
            "id": "different-physical-id-is-excluded",
            "content": "Exact content",
            "context": "Exact context",
            "keywords": ["primary", "secondary", "primary"],
            "score": 123.5,
        }
        self.assertEqual(
            amem_observable_projection(note),
            amem_observable_projection(result),
        )

    def test_capability_rejects_another_checkpoint_or_state(self) -> None:
        async def exercise() -> None:
            capability = AMemRetrievableEntryCapability(
                _state(_AMemPublicDouble([_Note("a-1", "memory")]))
            )
            wrong_checkpoint = await capability.inventory(
                campaign_id="campaign-1",
                root_checkpoint_id="checkpoint-2",
                state_id="state-1",
            )
            wrong_state = await capability.inventory(
                campaign_id="campaign-1",
                root_checkpoint_id="checkpoint-1",
                state_id="state-2",
            )
            self.assertFalse(wrong_checkpoint.supported)
            self.assertFalse(wrong_state.supported)

        asyncio.run(exercise())

    def test_missing_ids_malformed_notes_and_duplicates_are_blocked(self) -> None:
        async def exercise() -> None:
            for invalid in ("", None, 4):
                note = _Note("temporary", "memory")
                note.id = invalid
                system = _AMemPublicDouble([])
                system.memories = {invalid: note}
                system.retriever.collection.ids = [invalid]
                result = await AMemRetrievableEntryCapability(_state(system)).inventory(
                    campaign_id="campaign-1",
                    root_checkpoint_id="checkpoint-1",
                    state_id="state-1",
                )
                self.assertFalse(result.supported)

            system = _AMemPublicDouble([_Note("a-1", "memory")])
            system.retriever.collection.ids = ["a-1", "a-1"]
            result = await AMemRetrievableEntryCapability(_state(system)).inventory(
                campaign_id="campaign-1",
                root_checkpoint_id="checkpoint-1",
                state_id="state-1",
            )
            self.assertFalse(result.supported)
            self.assertIn("duplicate", result.blocker.reason)

            capability = AMemRetrievableEntryCapability(
                _state(_AMemPublicDouble([_Note("a-1", "memory")]))
            )
            capability.ranked_retrieval_from_search_result(
                [{"content": "memory", "context": "General", "keywords": []}],
                campaign_id="campaign-1",
                root_checkpoint_id="checkpoint-1",
                state_id="state-1",
            )

        with self.assertRaisesRegex(ValueError, "memory ID"):
            asyncio.run(exercise())

    def test_two_sided_inventory_consistency_rejects_stale_entries(self) -> None:
        async def exercise() -> None:
            for chroma_ids, expected in (
                (["a-1", "chroma-only"], "Chroma-only"),
                ([], "self.memories-only"),
            ):
                system = _AMemPublicDouble([_Note("a-1", "memory")])
                system.retriever.collection.ids = chroma_ids
                result = await AMemRetrievableEntryCapability(_state(system)).inventory(
                    campaign_id="campaign-1",
                    root_checkpoint_id="checkpoint-1",
                    state_id="state-1",
                )
                self.assertFalse(result.supported)
                self.assertIn(expected, result.blocker.reason)

        asyncio.run(exercise())

    def test_chroma_get_must_equal_unbounded_collection_count(self) -> None:
        async def exercise() -> None:
            system = _AMemPublicDouble([_Note("a-1", "memory")])
            system.retriever.collection.reported_count = 2
            result = await AMemRetrievableEntryCapability(_state(system)).inventory(
                campaign_id="campaign-1",
                root_checkpoint_id="checkpoint-1",
                state_id="state-1",
            )
            self.assertFalse(result.supported)
            self.assertIn("incomplete", result.blocker.reason)

        asyncio.run(exercise())

    def test_repeated_snapshot_detects_id_and_projection_instability(self) -> None:
        async def exercise() -> None:
            system = _AMemPublicDouble([_Note("a-1", "memory")])
            system.retriever.collection.snapshots = [["a-1"], []]
            result = await AMemRetrievableEntryCapability(_state(system)).inventory(
                campaign_id="campaign-1",
                root_checkpoint_id="checkpoint-1",
                state_id="state-1",
            )
            self.assertFalse(result.supported)

            note = _Note("a-1", "before")
            system = _AMemPublicDouble([note])
            original_get = system.retriever.collection.get
            calls = 0

            def changing_get():
                nonlocal calls
                result = original_get()
                calls += 1
                if calls == 1:
                    note.content = "after"
                return result

            system.retriever.collection.get = changing_get
            result = await AMemRetrievableEntryCapability(_state(system)).inventory(
                campaign_id="campaign-1",
                root_checkpoint_id="checkpoint-1",
                state_id="state-1",
            )
            self.assertFalse(result.supported)
            self.assertIn("projection", result.blocker.reason)

        asyncio.run(exercise())

    def test_equivalence_ignores_ids_order_and_harness_tags_but_keeps_state(self) -> None:
        async def exercise() -> None:
            left_notes = [
                _Note("a-1", "same", keywords=["k"], tags=["ufuzz-source:p1"]),
                _Note("a-2", "same", keywords=["k"], tags=["ufuzz-source:p2"]),
            ]
            right_notes = [
                _Note("b-2", "same", keywords=["k"], tags=["ufuzz-source:q2"]),
                _Note("b-1", "same", keywords=["k"], tags=["ufuzz-source:q1"]),
            ]
            left = await _inventory(
                AMemRetrievableEntryCapability(_state(_AMemPublicDouble(left_notes))),
                campaign_id="method-a",
            )
            right = await _inventory(
                AMemRetrievableEntryCapability(_state(_AMemPublicDouble(right_notes))),
                campaign_id="method-b",
            )
            comparison = compare_initialization_inventories(left, right)
            self.assertTrue(comparison.equivalent)
            self.assertEqual(comparison.reference_count, 2)
            build = build_frozen_checkpoint_inventory(left)
            self.assertEqual(len(build.inventory.entries), 2)

            changed = _AMemPublicDouble(
                [
                    _Note("c-1", "same", context="different", keywords=["k"]),
                    _Note("c-2", "same", keywords=["k", "extra"]),
                ]
            )
            changed_inventory = await _inventory(
                AMemRetrievableEntryCapability(_state(changed)),
                campaign_id="method-c",
            )
            self.assertFalse(
                compare_initialization_inventories(left, changed_inventory).equivalent
            )

        asyncio.run(exercise())

    def test_search_order_preserved_and_unknown_lineage_rejected(self) -> None:
        async def exercise() -> None:
            system = _AMemPublicDouble(
                [_Note("a-1", "one"), _Note("a-2", "two")]
            )
            system.search_ids = ["a-2", "a-1"]
            capability = AMemRetrievableEntryCapability(_state(system))
            retrieval = (
                await capability.retrieve(
                    campaign_id="campaign-1",
                    root_checkpoint_id="checkpoint-1",
                    state_id="state-1",
                    query="query",
                    top_k=2,
                )
            ).require()
            self.assertEqual(
                [item.backend_entry_id for item in retrieval.ranked_physical_entries],
                ["a-2", "a-1"],
            )

            inventory = await _inventory(capability)
            build = build_frozen_checkpoint_inventory(inventory)
            campaign = assemble_campaign_coverage("campaign-1", (build,))
            unknown = capability.ranked_retrieval_from_search_result(
                [
                    {
                        "id": "unknown",
                        "content": "unknown",
                        "context": "General",
                        "keywords": [],
                        "score": 0.1,
                    }
                ],
                campaign_id="campaign-1",
                root_checkpoint_id="checkpoint-1",
                state_id="state-1",
            )
            with self.assertRaises(LineageResolutionError):
                resolve_fuzzing_retrieval(
                    unknown,
                    campaign.lineage,
                    campaign.coverage_state,
                )

        asyncio.run(exercise())

    def test_same_id_update_requires_receipt_and_keeps_real_state(self) -> None:
        async def exercise() -> None:
            system = _AMemPublicDouble([_Note("a-1", "before")])
            adapter = AMemAdapter(memory_factory=lambda: system)
            state = await adapter.create_isolated_state(
                InitializationArtifact(
                    checkpoint_id="checkpoint-1",
                    sources=(),
                    frozen_config={"source": "synthetic"},
                    digest="digest",
                )
            )
            capability = AMemRetrievableEntryCapability(state)
            inventory = (
                await capability.inventory(
                    campaign_id="campaign-1",
                    root_checkpoint_id="checkpoint-1",
                    state_id=state.state_id,
                )
            ).require()
            build = build_frozen_checkpoint_inventory(inventory)
            campaign = assemble_campaign_coverage("campaign-1", (build,))
            physical = build.initial_bindings[0].physical_entry

            receipt = await adapter.update(
                state,
                "a-1",
                "after",
                provenance_ids=(),
            )
            binding = capability.certify_same_id_update(
                campaign.lineage,
                physical,
                receipt=receipt,
            )
            self.assertEqual(binding.physical_entry.state_id, state.state_id)
            self.assertEqual(binding.coverage_ids, build.initial_bindings[0].coverage_ids)

            bad = replace(
                receipt,
                state_id="invented-descendant",
            )
            with self.assertRaisesRegex(ValueError, "same-ID"):
                capability.certify_same_id_update(
                    campaign.lineage,
                    physical,
                    receipt=bad,
                )
            await adapter.teardown(state)

        asyncio.run(exercise())

    def test_deletion_verifies_both_stores_and_preserves_denominator(self) -> None:
        async def exercise() -> None:
            system = _AMemPublicDouble(
                [_Note("a-1", "one"), _Note("a-2", "two")]
            )
            capability = AMemRetrievableEntryCapability(_state(system))
            build = build_frozen_checkpoint_inventory(await _inventory(capability))
            campaign = assemble_campaign_coverage("campaign-1", (build,))
            physical = next(
                binding.physical_entry
                for binding in build.initial_bindings
                if binding.physical_entry.backend_entry_id == "a-1"
            )
            system.delete("a-1")
            capability.mark_deleted(
                campaign.lineage,
                physical,
                receipt=_receipt("delete", "a-1", after=None),
            )
            self.assertTrue(campaign.lineage.is_deleted(physical))
            self.assertEqual(campaign.coverage_state.denominator_count, 2)

            divergent = _AMemPublicDouble([_Note("a-1", "one")])
            divergent.memories.pop("a-1")
            divergent_capability = AMemRetrievableEntryCapability(_state(divergent))
            # Chroma-only state cannot certify a delete.
            with self.assertRaisesRegex(Exception, "identity mismatch"):
                divergent_capability.mark_deleted(
                    campaign.lineage,
                    physical,
                    receipt=_receipt("delete", "a-1", after=None),
                )

        asyncio.run(exercise())

    def test_uncertified_replacement_merge_split_are_not_guessed(self) -> None:
        async def exercise() -> None:
            system = _AMemPublicDouble([_Note("a-1", "before")])
            capability = AMemRetrievableEntryCapability(_state(system))
            build = build_frozen_checkpoint_inventory(await _inventory(capability))
            campaign = assemble_campaign_coverage("campaign-1", (build,))
            unknown = PhysicalEntryRef(
                "campaign-1", "checkpoint-1", "state-1", "new-id"
            )
            self.assertFalse(campaign.lineage.resolution(unknown).coverage_ids)
            with self.assertRaises(LineageResolutionError):
                campaign.lineage.resolve(unknown)

        asyncio.run(exercise())

    def test_generic_end_to_end_bridge_and_feedback(self) -> None:
        async def exercise() -> None:
            system = _AMemPublicDouble(
                [_Note("a-1", "one"), _Note("a-2", "two")]
            )
            system.search_ids = ["a-2", "a-1"]
            capability = AMemRetrievableEntryCapability(_state(system))
            build = build_frozen_checkpoint_inventory(await _inventory(capability))
            campaign = assemble_campaign_coverage("campaign-1", (build,))
            retrieval = (
                await capability.retrieve(
                    campaign_id="campaign-1",
                    root_checkpoint_id="checkpoint-1",
                    state_id="state-1",
                    query="query",
                    top_k=2,
                )
            ).require()
            observation = resolve_fuzzing_retrieval(
                retrieval,
                campaign.lineage,
                campaign.coverage_state,
            )
            history = BehaviorHistory("campaign-1")
            feedback = score_observed_retrieval(
                root_checkpoint_id="checkpoint-1",
                observed_signature=observation.signature,
                coverage_update=observation.coverage_update,
                retrieval_depth=2,
                mutation_relation=MutationRelation.UPDATE,
                history=history,
            )
            self.assertEqual(observation.coverage_update.raw_gain, 2)
            self.assertEqual(feedback.coverage_guided_score, 2)
            self.assertEqual(len(observation.signature), 2)

        asyncio.run(exercise())
if __name__ == "__main__":
    unittest.main()
