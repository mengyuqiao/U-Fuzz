from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import json
from types import SimpleNamespace
import unittest

from ufuzz.backends.base import StateHandle
from ufuzz.backends.graphiti import GRAPHITI_TAG_COMMIT, GRAPHITI_VERSION
from ufuzz.backends.graphiti_capability import (
    GRAPHITI_INVENTORY_API,
    GRAPHITI_PUBLIC_RETRIEVAL_API,
    GRAPHITI_RETRIEVAL_OBJECT_CLASS,
    GraphitiRetrievableEntryCapability,
    graphiti_observable_projection,
)
from ufuzz.coverage import LineageResolutionError
from ufuzz.retrieval_capability import (
    CapabilityOperation,
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


GROUP = "graphiti-state-1"
EDGE_1 = "00000000-0000-4000-8000-000000000001"
EDGE_2 = "00000000-0000-4000-8000-000000000002"
EDGE_3 = "00000000-0000-4000-8000-000000000003"
SOURCE_1 = "10000000-0000-4000-8000-000000000001"
TARGET_1 = "10000000-0000-4000-8000-000000000002"
EPISODE_1 = "20000000-0000-4000-8000-000000000001"
EPISODE_2 = "20000000-0000-4000-8000-000000000002"
TIME = datetime(2025, 1, 2, 3, 4, tzinfo=UTC)


def _node(
    uuid: str,
    name: str,
    *,
    group: str = GROUP,
    labels: tuple[str, ...] = ("Person", "Entity"),
    summary: str | None = None,
    attributes: dict | None = None,
    created_at: datetime | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        uuid=uuid,
        group_id=group,
        name=name,
        summary=summary if summary is not None else f"Summary for {name}",
        labels=list(labels),
        attributes=attributes if attributes is not None else {},
        created_at=created_at or datetime(2026, 1, 1, tzinfo=UTC),
    )


def _episode(
    uuid: str,
    provenance: str,
    *,
    group: str = GROUP,
) -> SimpleNamespace:
    return SimpleNamespace(
        uuid=uuid,
        group_id=group,
        source_description=json.dumps({"ufuzz_provenance_id": provenance}),
    )


def _edge(
    uuid: str = EDGE_1,
    fact: str = "Alice lives in Boston",
    *,
    group: str = GROUP,
    source: str = SOURCE_1,
    target: str = TARGET_1,
    episodes: tuple[str, ...] = (EPISODE_1,),
    name: str = "LIVES_IN",
    valid_at: datetime | None = TIME,
    invalid_at: datetime | None = None,
    expired_at: datetime | None = None,
    reference_time: datetime | None = TIME,
    attributes: dict | None = None,
    created_at: datetime | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        uuid=uuid,
        group_id=group,
        source_node_uuid=source,
        target_node_uuid=target,
        name=name,
        fact=fact,
        fact_embedding=None,
        episodes=list(episodes),
        valid_at=valid_at,
        invalid_at=invalid_at,
        expired_at=expired_at,
        reference_time=reference_time,
        attributes=attributes if attributes is not None else {},
        created_at=created_at or datetime(2026, 2, 1, tzinfo=UTC),
    )


class _EntityEdges:
    def __init__(self, snapshots: list[list[SimpleNamespace]]) -> None:
        self.snapshots = snapshots
        self.calls: list[tuple[list[str], int | None, str | None]] = []

    async def get_by_group_ids(self, group_ids, limit=None, uuid_cursor=None):
        self.calls.append((list(group_ids), limit, uuid_cursor))
        index = min(len(self.calls) - 1, len(self.snapshots) - 1)
        return list(self.snapshots[index])

    async def load_embeddings(self, edge):
        edge.fact_embedding = [0.25, 0.75]


class _ObjectsByUuid:
    def __init__(self, values: list[SimpleNamespace]) -> None:
        self.values = {value.uuid: value for value in values}
        self.calls: list[tuple[str, ...]] = []

    async def get_by_uuids(self, uuids):
        self.calls.append(tuple(uuids))
        return [self.values[value] for value in uuids if value in self.values]


class _GraphitiPublicDouble:
    def __init__(
        self,
        edges: list[SimpleNamespace],
        *,
        snapshots: list[list[SimpleNamespace]] | None = None,
        nodes: list[SimpleNamespace] | None = None,
        episodes: list[SimpleNamespace] | None = None,
    ) -> None:
        self.entity_edges = _EntityEdges(snapshots or [edges, edges])
        self.entity_nodes = _ObjectsByUuid(
            nodes or [_node(SOURCE_1, "Alice"), _node(TARGET_1, "Boston")]
        )
        self.episode_nodes = _ObjectsByUuid(
            episodes or [_episode(EPISODE_1, "source-1"), _episode(EPISODE_2, "source-2")]
        )
        self.edges = SimpleNamespace(entity=self.entity_edges)
        self.nodes = SimpleNamespace(
            entity=self.entity_nodes,
            episode=self.episode_nodes,
        )
        self.search_results = list(edges)
        self.search_calls: list[dict] = []

    async def search(self, query, **kwargs):
        self.search_calls.append({"query": query, **kwargs})
        return list(self.search_results)[: kwargs["num_results"]]

    async def get_nodes_and_edges_by_episode(self, _episode_ids):
        raise AssertionError("episode traversal must not be used as Graphiti inventory")


def _state(
    backend: _GraphitiPublicDouble,
    *,
    state_id: str = GROUP,
    checkpoint: str = "checkpoint-1",
) -> StateHandle:
    return StateHandle(
        backend="graphiti",
        state_id=state_id,
        checkpoint_id=checkpoint,
        initialization_digest="digest",
        backend_state=backend,
        metadata={"group_id": state_id, "database": "ufuzz-test"},
    )


async def _inventory(
    capability: GraphitiRetrievableEntryCapability,
    *,
    campaign: str = "campaign-1",
):
    return await capability.inventory(
        campaign_id=campaign,
        root_checkpoint_id=capability.state.checkpoint_id,
        state_id=capability.state.state_id,
    )


class GraphitiCapabilityTests(unittest.TestCase):
    def test_exact_source_pin_and_public_object_contract(self) -> None:
        self.assertEqual(GRAPHITI_VERSION, "0.30.2")
        self.assertEqual(
            GRAPHITI_TAG_COMMIT,
            "eaa4128681bc53487138a4bbc22d58336ebe70d2",
        )
        self.assertEqual(GRAPHITI_PUBLIC_RETRIEVAL_API, "Graphiti.search")
        self.assertEqual(GRAPHITI_INVENTORY_API, "Graphiti.edges.entity.get_by_group_ids")
        self.assertEqual(GRAPHITI_RETRIEVAL_OBJECT_CLASS, "EntityEdge")

    def test_inventory_uses_public_unbounded_group_entity_edges(self) -> None:
        async def exercise() -> None:
            backend = _GraphitiPublicDouble([_edge()])
            result = await _inventory(GraphitiRetrievableEntryCapability(_state(backend)))
            inventory = result.require()
            self.assertEqual(len(inventory.entries), 1)
            self.assertEqual(inventory.entries[0].physical_entry.backend_entry_id, EDGE_1)
            self.assertEqual(
                backend.entity_edges.calls,
                [([GROUP], None, None), ([GROUP], None, None)],
            )

        asyncio.run(exercise())

    def test_invalid_edge_uuid_is_an_explicit_identity_blocker(self) -> None:
        async def exercise() -> None:
            result = await _inventory(
                GraphitiRetrievableEntryCapability(_state(_GraphitiPublicDouble([_edge("bad")])) )
            )
            self.assertFalse(result.supported)
            self.assertEqual(result.blocker.operation, CapabilityOperation.PHYSICAL_IDENTITY)

        asyncio.run(exercise())

    def test_duplicate_uuid_discovery_is_suppressed_only_when_exact(self) -> None:
        async def exercise() -> None:
            first = _edge()
            duplicate = _edge()
            inventory = (
                await _inventory(
                    GraphitiRetrievableEntryCapability(
                        _state(_GraphitiPublicDouble([first, duplicate]))
                    )
                )
            ).require()
            self.assertEqual(len(inventory.entries), 1)

            mismatch = _edge(fact="Alice lives in Seattle")
            result = await _inventory(
                GraphitiRetrievableEntryCapability(
                    _state(_GraphitiPublicDouble([_edge(), mismatch]))
                )
            )
            self.assertFalse(result.supported)
            self.assertIn("duplicate discovery", result.blocker.reason)

        asyncio.run(exercise())

    def test_distinct_equal_projection_edges_remain_distinct_e0_entries(self) -> None:
        async def exercise() -> None:
            backend = _GraphitiPublicDouble([_edge(EDGE_1), _edge(EDGE_2)])
            inventory = (
                await _inventory(GraphitiRetrievableEntryCapability(_state(backend)))
            ).require()
            self.assertEqual(len(inventory.entries), 2)
            self.assertEqual(
                inventory.entries[0].projection,
                inventory.entries[1].projection,
            )
            frozen = build_frozen_checkpoint_inventory(inventory)
            self.assertEqual(len(frozen.inventory.entries), 2)

        asyncio.run(exercise())

    def test_projection_excludes_replay_ids_and_preserves_temporal_state(self) -> None:
        edge = _edge(invalid_at=TIME, expired_at=TIME)
        projection = graphiti_observable_projection(
            edge,
            source_node=_node(SOURCE_1, "Alice", labels=("Entity", "Person")),
            target_node=_node(TARGET_1, "Boston"),
            episode_provenance=(("source-1",),),
        )
        observable = projection.observable
        representation = repr(observable)
        for excluded in (EDGE_1, SOURCE_1, TARGET_1, GROUP, "fact_embedding", "created_at"):
            self.assertNotIn(excluded, representation)
        self.assertIn(TIME.isoformat(), representation)
        self.assertIn("Alice", representation)
        self.assertIn("Boston", representation)

    def test_expired_at_keeps_status_but_excludes_wall_clock_timestamp(self) -> None:
        first = graphiti_observable_projection(
            _edge(expired_at=datetime(2026, 1, 1, tzinfo=UTC)),
            source_node=_node(SOURCE_1, "Alice"),
            target_node=_node(TARGET_1, "Boston"),
            episode_provenance=(("source-1",),),
        )
        second = graphiti_observable_projection(
            _edge(expired_at=datetime(2026, 9, 1, tzinfo=UTC)),
            source_node=_node(SOURCE_1, "Alice"),
            target_node=_node(TARGET_1, "Boston"),
            episode_provenance=(("source-1",),),
        )
        active = graphiti_observable_projection(
            _edge(expired_at=None),
            source_node=_node(SOURCE_1, "Alice"),
            target_node=_node(TARGET_1, "Boston"),
            episode_provenance=(("source-1",),),
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first, active)
        self.assertTrue(first.observable["is_expired"])

    def test_attributes_remove_only_pinned_storage_and_harness_keys(self) -> None:
        projection = graphiti_observable_projection(
            _edge(
                attributes={
                    "confidence": "high",
                    "uuid": "replay-edge",
                    "fact_embedding": [1.0],
                    "ufuzz_provenance_id": "harness",
                }
            ),
            source_node=_node(
                SOURCE_1,
                "Alice",
                attributes={
                    "occupation": "engineer",
                    "created_at": "runtime",
                    "name_embedding": [1.0],
                    "ufuzz_checkpoint_id": "harness",
                },
            ),
            target_node=_node(TARGET_1, "Boston"),
            episode_provenance=(("source-1",),),
        )
        self.assertEqual(dict(projection.observable["attributes"]), {"confidence": "high"})
        source = projection.observable["source"]
        self.assertEqual(dict(source["attributes"]), {"occupation": "engineer"})

    def test_episode_position_is_behavioral_and_inner_provenance_is_set_like(self) -> None:
        edge = _edge(episodes=(EPISODE_1, EPISODE_2))
        source = _node(SOURCE_1, "Alice")
        target = _node(TARGET_1, "Boston")
        ordered = graphiti_observable_projection(
            edge,
            source_node=source,
            target_node=target,
            episode_provenance=(("source-b", "source-a"), ("source-c",)),
        )
        same_inner_set = graphiti_observable_projection(
            edge,
            source_node=source,
            target_node=target,
            episode_provenance=(("source-a", "source-b"), ("source-c",)),
        )
        reversed_episodes = graphiti_observable_projection(
            edge,
            source_node=source,
            target_node=target,
            episode_provenance=(("source-c",), ("source-a", "source-b")),
        )
        self.assertEqual(ordered, same_inner_set)
        self.assertNotEqual(ordered, reversed_episodes)

    def test_replay_local_fields_are_ignored_but_each_semantic_field_matters(self) -> None:
        replay_edge = "40000000-0000-4000-8000-000000000001"
        replay_source = "40000000-0000-4000-8000-000000000002"
        replay_target = "40000000-0000-4000-8000-000000000003"
        replay_episode = "40000000-0000-4000-8000-000000000004"
        replay_group = "graphiti-replay-state"

        baseline = graphiti_observable_projection(
            _edge(),
            source_node=_node(SOURCE_1, "Alice"),
            target_node=_node(TARGET_1, "Boston"),
            episode_provenance=(("source-1",),),
        )
        replay = graphiti_observable_projection(
            _edge(
                replay_edge,
                group=replay_group,
                source=replay_source,
                target=replay_target,
                episodes=(replay_episode,),
                created_at=datetime(2030, 1, 1, tzinfo=UTC),
            ),
            source_node=_node(
                replay_source,
                "Alice",
                group=replay_group,
                created_at=datetime(2030, 1, 2, tzinfo=UTC),
            ),
            target_node=_node(
                replay_target,
                "Boston",
                group=replay_group,
                created_at=datetime(2030, 1, 3, tzinfo=UTC),
            ),
            episode_provenance=(("source-1",),),
        )
        self.assertEqual(baseline, replay)

        changed = {
            "fact": graphiti_observable_projection(
                _edge(fact="Alice lives in Seattle"),
                source_node=_node(SOURCE_1, "Alice"),
                target_node=_node(TARGET_1, "Boston"),
                episode_provenance=(("source-1",),),
            ),
            "relation": graphiti_observable_projection(
                _edge(name="RESIDES_IN"),
                source_node=_node(SOURCE_1, "Alice"),
                target_node=_node(TARGET_1, "Boston"),
                episode_provenance=(("source-1",),),
            ),
            "endpoint": graphiti_observable_projection(
                _edge(),
                source_node=_node(SOURCE_1, "Alicia"),
                target_node=_node(TARGET_1, "Boston"),
                episode_provenance=(("source-1",),),
            ),
            "endpoint_summary": graphiti_observable_projection(
                _edge(),
                source_node=_node(SOURCE_1, "Alice", summary="Changed summary"),
                target_node=_node(TARGET_1, "Boston"),
                episode_provenance=(("source-1",),),
            ),
            "endpoint_labels": graphiti_observable_projection(
                _edge(),
                source_node=_node(SOURCE_1, "Alice", labels=("Entity", "Place")),
                target_node=_node(TARGET_1, "Boston"),
                episode_provenance=(("source-1",),),
            ),
            "endpoint_attributes": graphiti_observable_projection(
                _edge(),
                source_node=_node(
                    SOURCE_1, "Alice", attributes={"kind": "changed"}
                ),
                target_node=_node(TARGET_1, "Boston"),
                episode_provenance=(("source-1",),),
            ),
            "provenance": graphiti_observable_projection(
                _edge(),
                source_node=_node(SOURCE_1, "Alice"),
                target_node=_node(TARGET_1, "Boston"),
                episode_provenance=(("source-2",),),
            ),
            "valid_at": graphiti_observable_projection(
                _edge(valid_at=datetime(2024, 1, 1, tzinfo=UTC)),
                source_node=_node(SOURCE_1, "Alice"),
                target_node=_node(TARGET_1, "Boston"),
                episode_provenance=(("source-1",),),
            ),
            "reference_time": graphiti_observable_projection(
                _edge(reference_time=datetime(2024, 1, 1, tzinfo=UTC)),
                source_node=_node(SOURCE_1, "Alice"),
                target_node=_node(TARGET_1, "Boston"),
                episode_provenance=(("source-1",),),
            ),
            "invalid_at": graphiti_observable_projection(
                _edge(invalid_at=datetime(2025, 2, 1, tzinfo=UTC)),
                source_node=_node(SOURCE_1, "Alice"),
                target_node=_node(TARGET_1, "Boston"),
                episode_provenance=(("source-1",),),
            ),
            "expiration_state": graphiti_observable_projection(
                _edge(expired_at=datetime(2026, 1, 1, tzinfo=UTC)),
                source_node=_node(SOURCE_1, "Alice"),
                target_node=_node(TARGET_1, "Boston"),
                episode_provenance=(("source-1",),),
            ),
            "attributes": graphiti_observable_projection(
                _edge(attributes={"confidence": "derived"}),
                source_node=_node(SOURCE_1, "Alice"),
                target_node=_node(TARGET_1, "Boston"),
                episode_provenance=(("source-1",),),
            ),
        }
        for field, projection in changed.items():
            with self.subTest(field=field):
                self.assertNotEqual(baseline, projection)

    def test_stable_snapshot_rejects_uuid_or_projection_change(self) -> None:
        async def exercise() -> None:
            for snapshots, message in (
                ([[_edge(EDGE_1)], [_edge(EDGE_2)]], "UUID set changed"),
                ([[_edge(EDGE_1)], [_edge(EDGE_1, fact="changed")]], "projection"),
            ):
                backend = _GraphitiPublicDouble(snapshots[0], snapshots=snapshots)
                result = await _inventory(
                    GraphitiRetrievableEntryCapability(_state(backend))
                )
                self.assertFalse(result.supported)
                self.assertIn(message, result.blocker.reason)

        asyncio.run(exercise())

    def test_inventory_enumeration_order_does_not_affect_snapshot(self) -> None:
        async def exercise() -> None:
            backend = _GraphitiPublicDouble(
                [_edge(EDGE_1), _edge(EDGE_2)],
                snapshots=[
                    [_edge(EDGE_1), _edge(EDGE_2)],
                    [_edge(EDGE_2), _edge(EDGE_1)],
                ],
            )
            inventory = (
                await _inventory(GraphitiRetrievableEntryCapability(_state(backend)))
            ).require()
            self.assertEqual(len(inventory.entries), 2)

        asyncio.run(exercise())

    def test_bm25_returnable_edge_does_not_require_fact_embedding(self) -> None:
        async def exercise() -> None:
            backend = _GraphitiPublicDouble([_edge()])

            async def must_not_load_embedding(_edge_value):
                raise AssertionError("BM25-returnable inventory must not require an embedding")

            backend.entity_edges.load_embeddings = must_not_load_embedding
            inventory = (
                await _inventory(GraphitiRetrievableEntryCapability(_state(backend)))
            ).require()
            self.assertEqual(len(inventory.entries), 1)

        asyncio.run(exercise())

    def test_multi_episode_provenance_is_one_physical_entry(self) -> None:
        async def exercise() -> None:
            edge = _edge(episodes=(EPISODE_1, EPISODE_2))
            inventory = (
                await _inventory(
                    GraphitiRetrievableEntryCapability(
                        _state(_GraphitiPublicDouble([edge]))
                    )
                )
            ).require()
            self.assertEqual(len(inventory.entries), 1)
            self.assertEqual(inventory.entries[0].provenance_ids, ("source-1", "source-2"))

        asyncio.run(exercise())

    def test_search_order_is_preserved_and_unknown_uuid_gets_no_lineage(self) -> None:
        async def exercise() -> None:
            backend = _GraphitiPublicDouble([_edge(EDGE_1), _edge(EDGE_2)])
            capability = GraphitiRetrievableEntryCapability(_state(backend))
            inventory = (await _inventory(capability)).require()
            frozen = build_frozen_checkpoint_inventory(inventory)
            campaign = assemble_campaign_coverage("campaign-1", (frozen,))

            backend.search_results = [_edge(EDGE_2), _edge(EDGE_1)]
            ranked = (
                await capability.retrieve(
                    campaign_id="campaign-1",
                    root_checkpoint_id="checkpoint-1",
                    state_id=GROUP,
                    query="where",
                    top_k=2,
                )
            ).require()
            self.assertEqual(
                tuple(value.backend_entry_id for value in ranked.ranked_physical_entries),
                (EDGE_2, EDGE_1),
            )
            observation = resolve_fuzzing_retrieval(
                ranked, campaign.lineage, campaign.coverage_state
            )
            self.assertEqual(observation.coverage_update.raw_gain, 2)

            backend.search_results = [_edge(EDGE_3)]
            unknown = (
                await capability.retrieve(
                    campaign_id="campaign-1",
                    root_checkpoint_id="checkpoint-1",
                    state_id=GROUP,
                    query="new",
                    top_k=2,
                )
            ).require()
            with self.assertRaises(LineageResolutionError):
                resolve_fuzzing_retrieval(
                    unknown, campaign.lineage, observation.coverage_update.state
                )

        asyncio.run(exercise())

    def test_invalid_search_uuid_or_cross_group_result_is_not_omitted(self) -> None:
        async def exercise() -> None:
            backend = _GraphitiPublicDouble([_edge()])
            capability = GraphitiRetrievableEntryCapability(_state(backend))
            for result_edge, phrase in (
                (_edge("not-a-uuid"), "malformed UUID"),
                (_edge(EDGE_2, group="another-group"), "another group"),
            ):
                backend.search_results = [result_edge]
                result = await capability.retrieve(
                    campaign_id="campaign-1",
                    root_checkpoint_id="checkpoint-1",
                    state_id=GROUP,
                    query="query",
                    top_k=2,
                )
                self.assertFalse(result.supported)
                self.assertIn(phrase, result.blocker.reason)

        asyncio.run(exercise())

    def test_initialization_equivalence_ignores_replay_uuids_not_state(self) -> None:
        async def exercise() -> None:
            other_group = "graphiti-state-2"
            other_edge = "30000000-0000-4000-8000-000000000001"
            other_source = "30000000-0000-4000-8000-000000000002"
            other_target = "30000000-0000-4000-8000-000000000003"
            other_episode = "30000000-0000-4000-8000-000000000004"
            first = GraphitiRetrievableEntryCapability(
                _state(_GraphitiPublicDouble([_edge()]))
            )
            second_backend = _GraphitiPublicDouble(
                [
                    _edge(
                        other_edge,
                        group=other_group,
                        source=other_source,
                        target=other_target,
                        episodes=(other_episode,),
                    )
                ],
                nodes=[
                    _node(other_source, "Alice", group=other_group),
                    _node(other_target, "Boston", group=other_group),
                ],
                episodes=[_episode(other_episode, "source-1", group=other_group)],
            )
            second = GraphitiRetrievableEntryCapability(
                _state(second_backend, state_id=other_group)
            )
            reference = (await _inventory(first, campaign="reference")).require()
            candidate = (await _inventory(second, campaign="candidate")).require()
            self.assertTrue(compare_initialization_inventories(reference, candidate).equivalent)

            second_backend.entity_edges.snapshots = [
                [
                    _edge(
                        other_edge,
                        group=other_group,
                        source=other_source,
                        target=other_target,
                        episodes=(other_episode,),
                        invalid_at=TIME,
                    )
                ]
            ]
            changed = (await _inventory(second, campaign="changed")).require()
            self.assertFalse(compare_initialization_inventories(reference, changed).equivalent)

        asyncio.run(exercise())

    def test_generic_bridge_feeds_post_execution_score(self) -> None:
        async def exercise() -> None:
            backend = _GraphitiPublicDouble([_edge()])
            capability = GraphitiRetrievableEntryCapability(_state(backend))
            inventory = (await _inventory(capability)).require()
            frozen = build_frozen_checkpoint_inventory(inventory)
            campaign = assemble_campaign_coverage("campaign-1", (frozen,))
            ranked = (
                await capability.retrieve(
                    campaign_id="campaign-1",
                    root_checkpoint_id="checkpoint-1",
                    state_id=GROUP,
                    query="Alice",
                    top_k=2,
                )
            ).require()
            observation = resolve_fuzzing_retrieval(
                ranked, campaign.lineage, campaign.coverage_state
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
            self.assertEqual(feedback.raw_coverage_gain, 1)
            self.assertEqual(feedback.signature, observation.signature)

        asyncio.run(exercise())

    def test_episode_deletion_and_new_edges_do_not_gain_automatic_lineage(self) -> None:
        async def exercise() -> None:
            backend = _GraphitiPublicDouble([_edge()])
            capability = GraphitiRetrievableEntryCapability(_state(backend))
            inventory = (await _inventory(capability)).require()
            frozen = build_frozen_checkpoint_inventory(inventory)
            initial = frozen.initial_bindings[0].physical_entry
            self.assertFalse(frozen.lineage.is_deleted(initial))
            self.assertFalse(hasattr(capability, "mark_deleted"))
            self.assertFalse(hasattr(capability, "certify_replacement"))

        asyncio.run(exercise())

    def test_empty_inventory_is_supported_but_campaign_eligibility_is_generic(self) -> None:
        async def exercise() -> None:
            inventory = (
                await _inventory(
                    GraphitiRetrievableEntryCapability(
                        _state(_GraphitiPublicDouble([], nodes=[], episodes=[]))
                    )
                )
            ).require()
            self.assertEqual(inventory.entries, ())
            campaign = assemble_campaign_coverage(
                "campaign-1", (build_frozen_checkpoint_inventory(inventory),)
            )
            self.assertIsNone(campaign.coverage_state.cov_fraction)

        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
