from __future__ import annotations

import asyncio
import importlib.metadata
import importlib.util
import inspect
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest

from ufuzz.backends.base import InitializationArtifact, OperationReceipt, StateHandle
from ufuzz.backends.mem0 import Mem0Adapter
from ufuzz.backends.mem0_capability import (
    MEM0_PUBLIC_RETRIEVAL_API,
    MEM0_RETRIEVAL_OBJECT_CLASS,
    Mem0RetrievableEntryCapability,
    certify_mem0_same_id_update,
    mark_mem0_deleted,
    mem0_observable_projection,
)
from ufuzz.coverage import (
    LineageRelation,
    LineageResolutionError,
    LineageStatus,
    PhysicalEntryRef,
)
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


def _record(
    record_id: str,
    memory: str,
    *,
    provenance_id: str = "source-1",
    stable_metadata: dict | None = None,
    role: str = "user",
) -> dict:
    metadata = {
        "ufuzz_provenance_id": provenance_id,
        "ufuzz_checkpoint_id": "checkpoint-1",
    }
    metadata.update(stable_metadata or {})
    return {
        "id": record_id,
        "memory": memory,
        "hash": "derived-hash",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-02T00:00:00Z",
        "score": 0.91,
        "score_details": {"semantic": 0.91},
        "user_id": "replay-local-user",
        "agent_id": "replay-local-agent",
        "run_id": "replay-local-run",
        "role": role,
        "actor_id": None,
        "attributed_to": None,
        "expiration_date": None,
        "metadata": metadata,
    }


class _Mem0PublicDouble:
    def __init__(self, records: list[dict]) -> None:
        self.records = list(records)
        self.get_all_calls: list[dict] = []
        self.search_calls: list[dict] = []
        self.search_results = list(records)

    def get_all(self, **kwargs):
        self.get_all_calls.append(kwargs)
        return {"results": self.records[: kwargs["top_k"]]}

    def search(self, query, **kwargs):
        self.search_calls.append({"query": query, **kwargs})
        return {"results": list(self.search_results)[: kwargs["top_k"]]}


class _FinalSnapshotDouble(_Mem0PublicDouble):
    def __init__(self, first: list[dict], repeated: list[dict]) -> None:
        super().__init__(first)
        self.repeated = list(repeated)

    def get_all(self, **kwargs):
        self.get_all_calls.append(kwargs)
        values = self.records if len(self.get_all_calls) == 1 else self.repeated
        return {"results": values[: kwargs["top_k"]]}


def _state(memory, *, state_id: str = "state-1") -> StateHandle:
    return StateHandle(
        backend="mem0",
        state_id=state_id,
        checkpoint_id="checkpoint-1",
        initialization_digest="digest",
        backend_state=memory,
        metadata={
            "config": {
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "collection_name": f"ufuzz_{state_id.replace('-', '_')}",
                        "path": f"/tmp/ufuzz-test/{state_id}/qdrant",
                    },
                },
                "history_db_path": f"/tmp/ufuzz-test/{state_id}/history.db",
            }
        },
    )


async def _inventory(
    capability: Mem0RetrievableEntryCapability,
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


class Mem0CapabilityTests(unittest.TestCase):
    def test_inventory_uses_public_record_ids_and_object_class(self) -> None:
        async def exercise() -> None:
            memory = _Mem0PublicDouble([_record("m-1", "Alice lives in Boston.")])
            capability = Mem0RetrievableEntryCapability(_state(memory))
            inventory = await _inventory(capability)
            self.assertEqual(inventory.selected_public_retrieval_api, MEM0_PUBLIC_RETRIEVAL_API)
            self.assertEqual(inventory.retrieval_object_class, MEM0_RETRIEVAL_OBJECT_CLASS)
            self.assertEqual(
                inventory.entries[0].physical_entry.backend_entry_id,
                "m-1",
            )
            self.assertEqual(inventory.entries[0].provenance_ids, ("source-1",))

        asyncio.run(exercise())

    def test_missing_or_empty_public_record_id_is_a_capability_failure(self) -> None:
        async def exercise() -> None:
            for invalid in (None, "", 17):
                value = _record("temporary", "memory")
                value["id"] = invalid
                capability = Mem0RetrievableEntryCapability(
                    _state(_Mem0PublicDouble([value]))
                )
                result = await capability.inventory(
                    campaign_id="campaign-1",
                    root_checkpoint_id="checkpoint-1",
                    state_id="state-1",
                )
                self.assertFalse(result.supported)
                self.assertIn("'id'", result.blocker.reason)

            memory = _Mem0PublicDouble([_record("m-1", "memory")])
            memory.search_results = [{"memory": "missing id"}]
            result = await Mem0RetrievableEntryCapability(_state(memory)).retrieve(
                campaign_id="campaign-1",
                root_checkpoint_id="checkpoint-1",
                state_id="state-1",
                query="memory",
                top_k=2,
            )
            self.assertFalse(result.supported)
            self.assertIn("'id'", result.blocker.reason)

        asyncio.run(exercise())

    def test_get_all_completeness_grows_limit_and_uses_exact_scope(self) -> None:
        async def exercise() -> None:
            memory = _Mem0PublicDouble(
                [_record(f"m-{index}", f"memory {index}") for index in range(3)]
            )
            capability = Mem0RetrievableEntryCapability(
                _state(memory),
                initial_inventory_limit=2,
            )
            inventory = await _inventory(capability)
            self.assertEqual(len(inventory.entries), 3)
            self.assertEqual(
                [call["top_k"] for call in memory.get_all_calls],
                [2, 4, 4],
            )
            for call in memory.get_all_calls:
                self.assertEqual(call["filters"], {"user_id": "state-1"})
                self.assertFalse(call["show_expired"])

        asyncio.run(exercise())

    def test_projection_has_exact_frozen_fields_and_excludes_volatile_data(self) -> None:
        projection = mem0_observable_projection(_record("m-1", "  Exact Text  "))
        self.assertEqual(
            set(projection.observable),
            {
                "memory",
                "role",
                "actor_id",
                "attributed_to",
                "expiration_date",
                "stable_metadata",
            },
        )
        self.assertEqual(projection.observable["memory"], "  Exact Text  ")
        for excluded in (
            "id",
            "hash",
            "score",
            "score_details",
            "rank",
            "user_id",
            "agent_id",
            "run_id",
            "created_at",
            "updated_at",
        ):
            self.assertNotIn(excluded, projection.observable)
        self.assertEqual(projection.observable["stable_metadata"], {})

    def test_metadata_projection_filters_harness_and_preserves_stable_remainder(self) -> None:
        first = mem0_observable_projection(
            _record(
                "m-1",
                "same",
                provenance_id="source-1",
                stable_metadata={"category": "personal"},
            )
        )
        different_provenance = mem0_observable_projection(
            _record(
                "m-2",
                "same",
                provenance_id="source-2",
                stable_metadata={"category": "personal"},
            )
        )
        different_stable_metadata = mem0_observable_projection(
            _record(
                "m-3",
                "same",
                provenance_id="source-1",
                stable_metadata={"category": "work"},
            )
        )
        self.assertEqual(first, different_provenance)
        self.assertNotEqual(first, different_stable_metadata)
        self.assertEqual(
            first.observable["stable_metadata"],
            {"category": "personal"},
        )

    def test_equal_text_does_not_force_projection_equality(self) -> None:
        first = mem0_observable_projection(
            _record("m-1", "same", stable_metadata={"category": "personal"})
        )
        second = mem0_observable_projection(
            _record("m-2", "same", stable_metadata={"category": "work"})
        )
        third = mem0_observable_projection(_record("m-3", "same", role="assistant"))
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, third)

    def test_final_inventory_snapshot_requires_stable_ids_and_projections(self) -> None:
        async def exercise() -> None:
            original = [_record("m-1", "one"), _record("m-2", "two")]

            reordered = _FinalSnapshotDouble(original, list(reversed(original)))
            stable = await _inventory(
                Mem0RetrievableEntryCapability(
                    _state(reordered),
                    initial_inventory_limit=4,
                )
            )
            self.assertEqual(len(stable.entries), 2)

            changed_ids = _FinalSnapshotDouble(
                original,
                [_record("m-1", "one"), _record("m-3", "two")],
            )
            id_result = await Mem0RetrievableEntryCapability(
                _state(changed_ids),
                initial_inventory_limit=4,
            ).inventory(
                campaign_id="campaign-1",
                root_checkpoint_id="checkpoint-1",
                state_id="state-1",
            )
            self.assertFalse(id_result.supported)
            self.assertIn("physical ID set changed", id_result.blocker.reason)

            changed_projection = _FinalSnapshotDouble(
                original,
                [_record("m-1", "changed"), _record("m-2", "two")],
            )
            projection_result = await Mem0RetrievableEntryCapability(
                _state(changed_projection),
                initial_inventory_limit=4,
            ).inventory(
                campaign_id="campaign-1",
                root_checkpoint_id="checkpoint-1",
                state_id="state-1",
            )
            self.assertFalse(projection_result.supported)
            self.assertIn(
                "observable projection changed",
                projection_result.blocker.reason,
            )

        asyncio.run(exercise())

    def test_equivalence_ignores_ids_and_order_but_preserves_multiplicity(self) -> None:
        async def exercise() -> None:
            first_memory = _Mem0PublicDouble(
                [
                    _record("a-1", "same", provenance_id="source-a"),
                    _record("a-2", "same", provenance_id="source-b"),
                ]
            )
            second_memory = _Mem0PublicDouble(
                [
                    _record("b-2", "same", provenance_id="source-y"),
                    _record("b-1", "same", provenance_id="source-x"),
                ]
            )
            first = await _inventory(
                Mem0RetrievableEntryCapability(_state(first_memory)),
                campaign_id="method-a",
            )
            second = await _inventory(
                Mem0RetrievableEntryCapability(_state(second_memory)),
                campaign_id="method-b",
            )
            result = compare_initialization_inventories(first, second)
            self.assertTrue(result.equivalent)
            self.assertEqual(result.reference_count, 2)
            build = build_frozen_checkpoint_inventory(first)
            self.assertEqual(len(build.inventory.entries), 2)
            self.assertEqual(
                build.inventory.entries[0].initial_observable_projection,
                build.inventory.entries[1].initial_observable_projection,
            )

            second_memory.records.pop()
            missing = await _inventory(
                Mem0RetrievableEntryCapability(_state(second_memory)),
                campaign_id="method-b",
            )
            self.assertFalse(compare_initialization_inventories(first, missing).equivalent)

            personal = await _inventory(
                Mem0RetrievableEntryCapability(
                    _state(
                        _Mem0PublicDouble(
                            [
                                _record(
                                    "p-1",
                                    "same",
                                    stable_metadata={"category": "personal"},
                                )
                            ]
                        )
                    )
                ),
                campaign_id="metadata-a",
            )
            work = await _inventory(
                Mem0RetrievableEntryCapability(
                    _state(
                        _Mem0PublicDouble(
                            [
                                _record(
                                    "w-1",
                                    "same",
                                    stable_metadata={"category": "work"},
                                )
                            ]
                        )
                    )
                ),
                campaign_id="metadata-b",
            )
            self.assertFalse(
                compare_initialization_inventories(personal, work).equivalent
            )

        asyncio.run(exercise())

    def test_search_order_and_duplicate_physical_ids_are_preserved(self) -> None:
        async def exercise() -> None:
            memory = _Mem0PublicDouble(
                [_record("m-1", "one"), _record("m-2", "two")]
            )
            memory.search_results = [
                _record("m-2", "two"),
                _record("m-1", "one"),
                _record("m-2", "two"),
            ]
            ranked = (
                await Mem0RetrievableEntryCapability(_state(memory)).retrieve(
                    campaign_id="campaign-1",
                    root_checkpoint_id="checkpoint-1",
                    state_id="state-1",
                    query="query",
                    top_k=3,
                )
            ).require()
            self.assertEqual(
                [item.backend_entry_id for item in ranked.ranked_physical_entries],
                ["m-2", "m-1", "m-2"],
            )
            self.assertEqual(memory.search_calls[0]["filters"], {"user_id": "state-1"})

        asyncio.run(exercise())

    def test_unknown_search_id_is_rejected_during_certified_resolution(self) -> None:
        async def exercise() -> None:
            memory = _Mem0PublicDouble([_record("m-1", "one")])
            capability = Mem0RetrievableEntryCapability(_state(memory))
            inventory = await _inventory(capability)
            campaign = assemble_campaign_coverage(
                "campaign-1",
                (build_frozen_checkpoint_inventory(inventory),),
            )
            memory.search_results = [_record("m-new", "uncertified")]
            ranked = (
                await capability.retrieve(
                    campaign_id="campaign-1",
                    root_checkpoint_id="checkpoint-1",
                    state_id="state-1",
                    query="query",
                    top_k=2,
                )
            ).require()
            with self.assertRaises(LineageResolutionError):
                resolve_fuzzing_retrieval(
                    ranked,
                    campaign.lineage,
                    campaign.coverage_state,
                )

        asyncio.run(exercise())

    def test_same_id_update_remains_in_actual_state(self) -> None:
        async def exercise() -> None:
            memory = _Mem0PublicDouble([_record("m-1", "before")])
            inventory = await _inventory(Mem0RetrievableEntryCapability(_state(memory)))
            campaign = assemble_campaign_coverage(
                "campaign-1",
                (build_frozen_checkpoint_inventory(inventory),),
            )
            before = inventory.entries[0].physical_entry
            binding = certify_mem0_same_id_update(
                campaign.lineage,
                before,
                receipt=OperationReceipt(
                    backend="mem0",
                    state_id="state-1",
                    operation="update",
                    success=True,
                    target_local_id="m-1",
                    affected_local_ids=("m-1",),
                    provenance_ids=(),
                    before={"id": "m-1", "memory": "before"},
                    after={"id": "m-1", "memory": "after"},
                    raw={"message": "updated"},
                ),
            )
            self.assertIs(binding.relation, LineageRelation.SAME_ID_UPDATE)
            self.assertEqual(binding.coverage_ids, campaign.initial_ids)
            self.assertEqual(binding.physical_entry.backend_entry_id, "m-1")
            self.assertEqual(binding.physical_entry.state_id, "state-1")
            self.assertEqual(binding.physical_entry, before)
            self.assertNotIn(
                "after_state_id",
                inspect.signature(certify_mem0_same_id_update).parameters,
            )

        asyncio.run(exercise())

    def test_same_id_update_rejects_mismatched_receipt(self) -> None:
        async def exercise() -> None:
            memory = _Mem0PublicDouble([_record("m-1", "before")])
            inventory = await _inventory(Mem0RetrievableEntryCapability(_state(memory)))
            campaign = assemble_campaign_coverage(
                "campaign-1",
                (build_frozen_checkpoint_inventory(inventory),),
            )
            before = inventory.entries[0].physical_entry
            invalid = OperationReceipt(
                backend="mem0",
                state_id="synthetic-descendant",
                operation="update",
                success=True,
                target_local_id="m-1",
                affected_local_ids=("m-1",),
                provenance_ids=(),
                before={"id": "m-1"},
                after={"id": "m-1"},
                raw={"message": "updated"},
            )
            with self.assertRaisesRegex(ValueError, "same-ID transition"):
                certify_mem0_same_id_update(
                    campaign.lineage,
                    before,
                    receipt=invalid,
                )

        asyncio.run(exercise())

    def test_deletion_preserves_denominator_and_gives_no_credit(self) -> None:
        async def exercise() -> None:
            memory = _Mem0PublicDouble(
                [_record("m-1", "one"), _record("m-2", "two")]
            )
            inventory = await _inventory(Mem0RetrievableEntryCapability(_state(memory)))
            campaign = assemble_campaign_coverage(
                "campaign-1",
                (build_frozen_checkpoint_inventory(inventory),),
            )
            deleted = inventory.entries[0].physical_entry
            mark_mem0_deleted(
                campaign.lineage,
                deleted,
                receipt=OperationReceipt(
                    backend="mem0",
                    state_id="state-1",
                    operation="delete",
                    success=True,
                    target_local_id=deleted.backend_entry_id,
                    affected_local_ids=(deleted.backend_entry_id,),
                    provenance_ids=(),
                    before={"id": deleted.backend_entry_id},
                    after=None,
                    raw={"message": "deleted"},
                ),
            )
            self.assertTrue(campaign.lineage.is_deleted(deleted))
            self.assertEqual(campaign.coverage_state.denominator_count, 2)
            self.assertEqual(campaign.coverage_state.reached_count, 0)

        asyncio.run(exercise())

    def test_replacement_or_merge_is_not_guessed_without_certificate(self) -> None:
        async def exercise() -> None:
            memory = _Mem0PublicDouble([_record("m-1", "one")])
            inventory = await _inventory(Mem0RetrievableEntryCapability(_state(memory)))
            campaign = assemble_campaign_coverage(
                "campaign-1",
                (build_frozen_checkpoint_inventory(inventory),),
            )
            unknown = PhysicalEntryRef(
                "campaign-1",
                "checkpoint-1",
                "state-2",
                "replacement-or-merge",
            )
            result = campaign.lineage.resolution(unknown)
            self.assertIs(result.status, LineageStatus.CAPABILITY_FAILURE)
            self.assertEqual(result.coverage_ids, frozenset())

        asyncio.run(exercise())

    def test_concrete_observation_flows_to_post_execution_score(self) -> None:
        async def exercise() -> None:
            memory = _Mem0PublicDouble(
                [_record("m-1", "one"), _record("m-2", "two")]
            )
            capability = Mem0RetrievableEntryCapability(_state(memory))
            inventory = await _inventory(capability)
            campaign = assemble_campaign_coverage(
                "campaign-1",
                (build_frozen_checkpoint_inventory(inventory),),
            )
            ranked = (
                await capability.retrieve(
                    campaign_id="campaign-1",
                    root_checkpoint_id="checkpoint-1",
                    state_id="state-1",
                    query="one",
                    top_k=2,
                )
            ).require()
            observation = resolve_fuzzing_retrieval(
                ranked,
                campaign.lineage,
                campaign.coverage_state,
            )
            feedback = score_observed_retrieval(
                root_checkpoint_id="checkpoint-1",
                observed_signature=observation.signature,
                coverage_update=observation.coverage_update,
                retrieval_depth=2,
                mutation_relation=MutationRelation.UPDATE,
                history=BehaviorHistory("campaign-1"),
            )
            self.assertEqual(feedback.raw_coverage_gain, 2)
            self.assertEqual(feedback.bounded_coverage_gain, 1.0)

        asyncio.run(exercise())


@unittest.skipUnless(
    importlib.util.find_spec("mem0") is not None,
    "mem0ai is not installed in this backend-specific environment",
)
class Mem0InstalledPackageCapabilityTests(unittest.TestCase):
    def test_real_2012_local_qdrant_public_capability_path(self) -> None:
        self.assertEqual(importlib.metadata.version("mem0ai"), "2.0.12")

        async def exercise() -> None:
            from mem0.embeddings.mock import MockEmbeddings
            from mem0.memory import main as mem0_main

            with tempfile.TemporaryDirectory(prefix="ufuzz-mem0-capability-") as root:
                artifact = InitializationArtifact.create(
                    "checkpoint-real",
                    (),
                    {"infer": False, "purpose": "capability-validation"},
                )
                adapter = Mem0Adapter(
                    config={
                        "vector_store": {
                            "provider": "qdrant",
                            "config": {"embedding_model_dims": 10},
                        },
                        "embedder": {"provider": "openai", "config": {}},
                        "llm": {
                            "provider": "openai",
                            "config": {"api_key": "unused-local-test-key"},
                        },
                    },
                    root_dir=Path(root),
                    infer=False,
                )
                with (
                    patch.object(
                        mem0_main.EmbedderFactory,
                        "create",
                        return_value=MockEmbeddings(),
                    ),
                    patch.object(
                        mem0_main.LlmFactory,
                        "create",
                        return_value=object(),
                    ),
                    patch.object(mem0_main, "MEM0_TELEMETRY", False),
                    patch.object(mem0_main, "capture_event"),
                    patch.object(mem0_main, "display_first_run_notice"),
                    patch.object(mem0_main, "display_scale_threshold_notice"),
                ):
                    state = await adapter.create_isolated_state(artifact)
                    memory = state.backend_state
                    responses = [
                        memory.add(
                            [{"role": "user", "content": text}],
                            user_id=state.state_id,
                            metadata={"ufuzz_provenance_id": source},
                            infer=False,
                        )
                        for text, source in (
                            ("same memory", "source-1"),
                            ("same memory", "source-2"),
                            ("different memory", "source-3"),
                        )
                    ]
                    ids = [item["results"][0]["id"] for item in responses]
                    self.assertEqual(len(set(ids)), 3)

                    capability = Mem0RetrievableEntryCapability(
                        state,
                        initial_inventory_limit=2,
                    )
                    inventory = (
                        await capability.inventory(
                            campaign_id="campaign-real",
                            root_checkpoint_id="checkpoint-real",
                            state_id=state.state_id,
                        )
                    ).require()
                    self.assertEqual(len(inventory.entries), 3)
                    self.assertEqual(
                        len(
                            {
                                item.physical_entry.backend_entry_id
                                for item in inventory.entries
                            }
                        ),
                        3,
                    )
                    same = [
                        item
                        for item in inventory.entries
                        if item.projection.observable["memory"] == "same memory"
                    ]
                    self.assertEqual(len(same), 2)
                    self.assertEqual(same[0].projection, same[1].projection)

                    replayed_state = await adapter.create_isolated_state(artifact)
                    replayed_memory = replayed_state.backend_state
                    for text, source in (
                        ("same memory", "source-1"),
                        ("same memory", "source-2"),
                        ("different memory", "source-3"),
                    ):
                        replayed_memory.add(
                            [{"role": "user", "content": text}],
                            user_id=replayed_state.state_id,
                            metadata={"ufuzz_provenance_id": source},
                            infer=False,
                        )
                    replayed_inventory = (
                        await Mem0RetrievableEntryCapability(
                            replayed_state,
                            initial_inventory_limit=2,
                        ).inventory(
                            campaign_id="campaign-replayed",
                            root_checkpoint_id="checkpoint-real",
                            state_id=replayed_state.state_id,
                        )
                    ).require()
                    equivalence = compare_initialization_inventories(
                        inventory,
                        replayed_inventory,
                    )
                    self.assertTrue(equivalence.equivalent)
                    self.assertNotEqual(
                        {
                            item.physical_entry.backend_entry_id
                            for item in inventory.entries
                        },
                        {
                            item.physical_entry.backend_entry_id
                            for item in replayed_inventory.entries
                        },
                    )
                    await adapter.teardown(replayed_state)

                    ranked = (
                        await capability.retrieve(
                            campaign_id="campaign-real",
                            root_checkpoint_id="checkpoint-real",
                            state_id=state.state_id,
                            query="same memory",
                            top_k=3,
                        )
                    ).require()
                    inventory_ids = {
                        item.physical_entry.backend_entry_id
                        for item in inventory.entries
                    }
                    self.assertTrue(
                        {
                            item.backend_entry_id
                            for item in ranked.ranked_physical_entries
                        }.issubset(inventory_ids)
                    )

                    target_id = ids[0]
                    update_receipt = await adapter.update(
                        state,
                        target_id,
                        "updated memory",
                        provenance_ids=("source-1",),
                    )
                    updated = memory.get(target_id)
                    self.assertEqual(updated["id"], target_id)
                    self.assertEqual(updated["memory"], "updated memory")

                    frozen = build_frozen_checkpoint_inventory(inventory)
                    campaign = assemble_campaign_coverage(
                        "campaign-real",
                        (frozen,),
                    )
                    initial_ref = next(
                        item.physical_entry
                        for item in inventory.entries
                        if item.physical_entry.backend_entry_id == target_id
                    )
                    updated_binding = certify_mem0_same_id_update(
                        campaign.lineage,
                        initial_ref,
                        receipt=update_receipt,
                    )
                    self.assertEqual(updated_binding.physical_entry, initial_ref)
                    self.assertEqual(
                        updated_binding.coverage_ids,
                        campaign.lineage.resolve(initial_ref),
                    )
                    denominator = frozen.checkpoint_coverage_state.denominator_count
                    delete_receipt = await adapter.delete(state, target_id)
                    mark_mem0_deleted(
                        campaign.lineage,
                        initial_ref,
                        receipt=delete_receipt,
                    )
                    after_delete = (
                        await capability.inventory(
                            campaign_id="campaign-real",
                            root_checkpoint_id="checkpoint-real",
                            state_id=state.state_id,
                        )
                    ).require()
                    self.assertNotIn(
                        target_id,
                        {
                            item.physical_entry.backend_entry_id
                            for item in after_delete.entries
                        },
                    )
                    self.assertEqual(
                        frozen.checkpoint_coverage_state.denominator_count,
                        denominator,
                    )
                    await adapter.teardown(state)

        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
