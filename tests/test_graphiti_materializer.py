from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import json
from types import SimpleNamespace
import unittest
from uuid import NAMESPACE_URL, uuid5

from ufuzz.backends.base import InitializationArtifact
from ufuzz.backends.graphiti import GraphitiAdapter
from ufuzz.backends.graphiti_capability import GraphitiRetrievableEntryCapability
from ufuzz.backends.graphiti_materializer import (
    GRAPHITI_NATIVE_CONFIGURATION_ID,
    GraphitiMaterializationProtocol,
)
from ufuzz.domain import BenchmarkCheckpoint, BenchmarkQuery, DatasetArtifact, SourceUnit
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
from ufuzz.structural import (
    StructuralIndexBuilder,
    certify_exact_source_inventory,
    exact_source_fact,
)


CAMPAIGN, CHECKPOINT, ARTIFACT_ID = "graphiti-campaign", "graphiti-root", "graphiti-init"


def _source(index, text, benchmark="synthetic"):
    return SourceUnit(
        benchmark,
        CHECKPOINT,
        "session",
        f"source-{index}",
        index,
        text,
        f"2026-01-{index + 1:02d}T00:00:00+00:00",
        "user",
        "user",
        {},
    )


SOURCES = (
    _source(0, "Alice residence Boston."),
    _source(1, "Bob hobby tennis."),
    _source(2, "Carol pet cat."),
)


def _node(identifier, group, name):
    return SimpleNamespace(
        uuid=identifier,
        group_id=group,
        name=name,
        summary=f"Summary for {name}",
        labels=["Entity"],
        attributes={},
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _edge(identifier, group, source_id, target_id, episode_id, fact, relation, when):
    return SimpleNamespace(
        uuid=identifier,
        group_id=group,
        source_node_uuid=source_id,
        target_node_uuid=target_id,
        name=relation.upper(),
        fact=fact,
        fact_embedding=None,
        episodes=[episode_id],
        valid_at=when,
        invalid_at=None,
        expired_at=None,
        reference_time=when,
        attributes={},
        created_at=datetime(2026, 2, 1, tzinfo=UTC),
    )


class _ByUuid:
    def __init__(self, values):
        self.values = values

    async def get_by_uuids(self, identifiers):
        return [self.values[value] for value in identifiers if value in self.values]


class _Edges:
    def __init__(self, graph):
        self.graph = graph

    async def get_by_group_ids(self, group_ids, limit=None, uuid_cursor=None):
        values = [
            edge for edge in self.graph.edge_values.values() if edge.group_id in group_ids
        ]
        return sorted(values, key=lambda edge: edge.uuid)


class _GraphitiDouble:
    """Deterministic public Graphiti surface with replacement updates."""

    def __init__(self, *, retain_predecessor=False):
        self.edge_values = {}
        self.node_values = {}
        self.episode_values = {}
        self.edges = SimpleNamespace(entity=_Edges(self))
        self.nodes = SimpleNamespace(
            entity=_ByUuid(self.node_values), episode=_ByUuid(self.episode_values)
        )
        self.retain_predecessor = retain_predecessor

    @staticmethod
    def _parts(text):
        parts = text.rstrip(".").split()
        if len(parts) != 3:
            raise ValueError("test episode must be entity relation value")
        return parts

    async def add_episode(
        self,
        *,
        name,
        episode_body,
        source_description,
        reference_time,
        source,
        group_id,
        uuid,
    ):
        entity, relation, value = self._parts(episode_body)
        source_id = str(uuid5(NAMESPACE_URL, f"{group_id}:node:{entity}"))
        target_id = str(uuid5(NAMESPACE_URL, f"{group_id}:node:{value}"))
        self.node_values.setdefault(source_id, _node(source_id, group_id, entity))
        self.node_values.setdefault(target_id, _node(target_id, group_id, value))
        self.episode_values[uuid] = SimpleNamespace(
            uuid=uuid,
            group_id=group_id,
            source_description=source_description,
        )
        if name.startswith("update:"):
            old_id = name.removeprefix("update:")
            if self.retain_predecessor and old_id in self.edge_values:
                self.edge_values[old_id].invalid_at = reference_time
                self.edge_values[old_id].expired_at = reference_time
            else:
                self.edge_values.pop(old_id, None)
            edge_id = str(uuid5(NAMESPACE_URL, f"{uuid}:replacement"))
        else:
            edge_id = str(uuid5(NAMESPACE_URL, f"{uuid}:root"))
        edge = _edge(
            edge_id,
            group_id,
            source_id,
            target_id,
            uuid,
            episode_body,
            relation,
            reference_time,
        )
        self.edge_values[edge_id] = edge
        return SimpleNamespace(edges=[edge], episode=self.episode_values[uuid])

    async def search(self, query, group_ids=None, num_results=10):
        terms = set(query.casefold().split())
        values = [edge for edge in self.edge_values.values() if edge.group_id in group_ids]
        return sorted(
            values,
            key=lambda edge: (
                -len(terms.intersection(edge.fact.casefold().split())),
                edge.fact,
                edge.uuid,
            ),
        )[:num_results]

    async def get_nodes_and_edges_by_episode(self, episode_ids):
        edges = [
            edge
            for edge in self.edge_values.values()
            if set(edge.episodes).intersection(episode_ids)
        ]
        nodes = {
            identifier: self.node_values[identifier]
            for edge in edges
            for identifier in (edge.source_node_uuid, edge.target_node_uuid)
        }
        return SimpleNamespace(edges=edges, nodes=list(nodes.values()))

    async def remove_episode(self, episode_uuid):
        for identifier, edge in tuple(self.edge_values.items()):
            if episode_uuid in edge.episodes:
                del self.edge_values[identifier]
        self.episode_values.pop(episode_uuid, None)

    async def clear_group(self, group_id):
        self.edge_values = {
            key: value
            for key, value in self.edge_values.items()
            if value.group_id != group_id
        }
        self.node_values = {
            key: value
            for key, value in self.node_values.items()
            if value.group_id != group_id
        }
        self.episode_values = {
            key: value
            for key, value in self.episode_values.items()
            if value.group_id != group_id
        }
        self.nodes.entity.values = self.node_values
        self.nodes.episode.values = self.episode_values


def _checkpoint_and_index():
    checkpoint = BenchmarkCheckpoint(
        "synthetic",
        CHECKPOINT,
        SOURCES,
        (
            BenchmarkQuery(
                "synthetic", CHECKPOINT, "query", "Where does Alice live?", None, None, {}
            ),
        ),
        {},
        DatasetArtifact("synthetic", "v1", "memory", "0" * 64, 0, "synthetic://v1"),
    )
    facts, certificates = [], []
    for source, entity, relation, value in zip(
        SOURCES,
        ("Alice", "Bob", "Carol"),
        ("residence", "hobby", "pet"),
        ("Boston", "tennis", "cat"),
    ):
        fact, certificate = exact_source_fact(
            source=source, entity=entity, relation=relation, value=value
        )
        facts.append(fact)
        certificates.append(certificate)
    certificates.append(certify_exact_source_inventory(source=SOURCES[2], facts=(facts[2],)))
    return checkpoint, StructuralIndexBuilder().build(
        checkpoint, facts=facts, certificates=certificates
    ), facts


class GraphitiMaterializerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint, cls.index, cls.facts = _checkpoint_and_index()
        cls.artifact = InitializationArtifact.create(
            CHECKPOINT,
            SOURCES,
            {"graphiti_profile_id": GRAPHITI_NATIVE_CONFIGURATION_ID},
        )

        async def bootstrap():
            adapter = GraphitiAdapter(graphiti=_GraphitiDouble(), database="test")
            state = await adapter.replay_state(cls.artifact)
            inventory = (
                await GraphitiRetrievableEntryCapability(state).inventory(
                    campaign_id=CAMPAIGN,
                    root_checkpoint_id=CHECKPOINT,
                    state_id=state.state_id,
                )
            ).require()
            frozen = build_frozen_checkpoint_inventory(inventory).inventory
            await adapter.teardown(state)
            return frozen

        cls.inventory = asyncio.run(bootstrap())

    def setUp(self):
        self.protocol = self.make_protocol()

    def make_protocol(self, graph=None):
        return GraphitiMaterializationProtocol(
            initialization_artifacts={ARTIFACT_ID: self.artifact},
            frozen_inventories={(CAMPAIGN, CHECKPOINT): self.inventory},
            structural_indexes={(CAMPAIGN, CHECKPOINT): self.index},
            adapter=GraphitiAdapter(graphiti=graph or _GraphitiDouble(), database="test"),
        )

    def tearDown(self):
        async def cleanup():
            for state_id, context in tuple(self.protocol._states.items()):
                await self.protocol.discard(
                    EphemeralMaterializedState(
                        context.handle,
                        "graphiti",
                        GRAPHITI_NATIVE_CONFIGURATION_ID,
                        CAMPAIGN,
                        CHECKPOINT,
                        state_id,
                    )
                )

        asyncio.run(cleanup())

    def request(self):
        return RootMaterializationRequest(
            "graphiti",
            GRAPHITI_NATIVE_CONFIGURATION_ID,
            CAMPAIGN,
            CHECKPOINT,
            ARTIFACT_ID,
            self.inventory.coverage_ids,
        )

    async def root(self):
        result = await self.protocol.materialize_root(self.request())
        self.assertIsNone(result.failure, result.failure)
        return result.value

    def entry(self, index):
        return next(
            entry
            for entry in self.inventory.entries
            if entry.provenance_ids == (SOURCES[index].provenance_id,)
        )

    def _expected(self, index, new_text, *, seconds=3):
        before = dict(self.entry(index).initial_observable_projection)
        entity, relation, value = new_text.rstrip(".").split()
        before["fact"] = new_text
        before["name"] = relation.upper()
        before["source"] = {
            "name": entity,
            "summary": f"Summary for {entity}",
            "labels": ("Entity",),
            "attributes": {},
        }
        before["target"] = {
            "name": value,
            "summary": f"Summary for {value}",
            "labels": ("Entity",),
            "attributes": {},
        }
        before["valid_at"] = f"2100-01-01T00:00:{seconds:02d}+00:00"
        before["reference_time"] = f"2100-01-01T00:00:{seconds:02d}+00:00"
        return before

    def mutation(
        self,
        name,
        relation,
        index,
        new_text=None,
        *,
        seconds=3,
        update_outcome=PhysicalTransitionOutcome.REPLACED,
    ):
        target = self.entry(index).coverage_id
        acceptable = (
            {PhysicalTransitionOutcome.DELETED}
            if relation is MutationRelation.DELETION
            else {update_outcome}
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
            possible_transition_outcomes=acceptable,
            acceptable_transition_outcomes=acceptable,
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
        if relation is MutationRelation.DELETION:
            payload = {"operation": "delete", "intended_fact_id": self.facts[index].fact_id}
        else:
            payload = {
                "operation": "unrelated_change"
                if relation is MutationRelation.UNRELATED_CHANGE
                else "update",
                "new_text": new_text,
                "provenance_ids": (SOURCES[index].provenance_id,),
                "context": {
                    "expected_successor_observable": self._expected(
                        index, new_text, seconds=seconds
                    )
                },
            }
        return RealizedMutationArtifact(
            name, opportunity, semantic, "query", None, relation.value, payload
        )

    def test_root_replay_and_ranked_retrieval(self):
        async def exercise():
            first = await self.root()
            observed = first.observed_root_state
            first_ids = {
                item.physical_entry.backend_entry_id
                for item in first.rebinding_certificate.live_bindings
            }
            retrieval = await self.protocol.retrieve(
                first.state,
                query_artifact=ExecutableQueryArtifact.create(
                    artifact_id="query", executable_text="Alice Boston", metadata={}
                ),
                top_k=2,
            )
            self.assertIsNone(retrieval.failure)
            self.assertEqual(len(retrieval.value.ranked_physical_entries), 2)
            await self.protocol.discard(first.state)
            second = await self.root()
            second_ids = {
                item.physical_entry.backend_entry_id
                for item in second.rebinding_certificate.live_bindings
            }
            self.assertTrue(first_ids.isdisjoint(second_ids))
            self.assertEqual(observed, second.observed_root_state)

        asyncio.run(exercise())

    def test_update_delete_and_unrelated_change_certification(self):
        async def exercise():
            root = await self.root()
            update = await self.protocol.replay_transition(
                root.state,
                self.mutation(
                    "update", MutationRelation.UPDATE, 0, "Alice residence Seattle."
                ),
            )
            self.assertIsNone(update.failure, update.failure)
            self.assertIs(
                update.value.certificate.observed_outcome,
                PhysicalTransitionOutcome.REPLACED,
            )
            unrelated = await self.protocol.replay_transition(
                root.state,
                self.mutation(
                    "urc",
                    MutationRelation.UNRELATED_CHANGE,
                    1,
                    "Bob hobby golf.",
                    seconds=4,
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
            self.assertEqual(descendant.value.deleted_ids, frozenset({self.entry(2).coverage_id}))

        asyncio.run(exercise())

    def test_collateral_or_unbound_successor_fails_closed(self):
        async def exercise():
            root = await self.root()
            artifact = self.mutation(
                "bad", MutationRelation.UPDATE, 0, "Alice residence Seattle."
            )
            values = dict(artifact.operation_payload)
            values["context"] = {
                "expected_successor_observable": {"fact": "not the native edge"}
            }
            bad = RealizedMutationArtifact(
                artifact.artifact_id,
                artifact.opportunity,
                artifact.semantic_certificate,
                artifact.parent_query_id,
                None,
                artifact.subtype,
                values,
            )
            result = await self.protocol.replay_transition(root.state, bad)
            self.assertIsNotNone(result.failure)
            self.assertIs(
                result.failure.kind, MaterializationFailureKind.TRANSITION_REPLAY_DIVERGED
            )

        asyncio.run(exercise())

    def test_native_historical_predecessor_is_certified_as_split(self):
        async def exercise():
            protocol = self.make_protocol(_GraphitiDouble(retain_predecessor=True))
            result = await protocol.materialize_root(self.request())
            self.assertIsNone(result.failure, result.failure)
            mutation = self.mutation(
                "split-update",
                MutationRelation.UPDATE,
                0,
                "Alice residence Seattle.",
                update_outcome=PhysicalTransitionOutcome.SPLIT,
            )
            replay = await protocol.replay_transition(result.value.state, mutation)
            self.assertIsNone(replay.failure, replay.failure)
            self.assertIs(
                replay.value.certificate.observed_outcome,
                PhysicalTransitionOutcome.SPLIT,
            )
            self.assertEqual(
                len(replay.value.certificate.affected[0].successors), 2
            )
            descendant = await protocol.certify_descendant(
                result.value.state, (replay.value.certificate,)
            )
            self.assertIsNone(descendant.failure, descendant.failure)
            await protocol.discard(result.value.state)

        asyncio.run(exercise())

    def test_process_scoped_search_configuration_isolation_and_cleanup(self):
        async def exercise():
            first = await self.root()
            second_protocol = self.make_protocol()
            blocked = await second_protocol.materialize_root(self.request())
            self.assertIsNotNone(blocked.failure)
            await self.protocol.discard(first.state)
            admitted = await second_protocol.materialize_root(self.request())
            self.assertIsNone(admitted.failure, admitted.failure)
            graph = admitted.value.state.handle.backend_state
            await second_protocol.discard(admitted.value.state)
            self.assertEqual(graph.edge_values, {})
            self.assertEqual(graph.node_values, {})
            self.assertEqual(graph.episode_values, {})

        asyncio.run(exercise())

    def test_minimal_benchmark_sources_need_no_backend_specific_transform(self):
        for benchmark in ("locomo", "longmemeval-s"):
            source = _source(9, f"Alice residence {benchmark}.", benchmark)
            artifact = InitializationArtifact.create(
                CHECKPOINT,
                (source,),
                {"graphiti_profile_id": GRAPHITI_NATIVE_CONFIGURATION_ID},
            )
            self.assertEqual(artifact.sources[0].benchmark, benchmark)


if __name__ == "__main__":
    unittest.main()
