from __future__ import annotations

import asyncio
from dataclasses import fields, is_dataclass
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping
import unittest
from uuid import NAMESPACE_URL, uuid5

from ufuzz.backends import AMemAdapter, GraphitiAdapter, InitializationArtifact, Mem0Adapter
from ufuzz.backends.amem_capability import AMemRetrievableEntryCapability
from ufuzz.backends.base import source_metadata
from ufuzz.backends.graphiti_capability import GraphitiRetrievableEntryCapability
from ufuzz.backends.mem0_capability import Mem0RetrievableEntryCapability
from ufuzz.benchmarks import LongMemEvalSLoader
from ufuzz.coverage import (
    CoverageEntryId,
    CoverageState,
    FrozenCheckpointInventory,
    InitialCoverageEntry,
    InitialEntryLineage,
    LineageRelation,
    PhysicalEntryRef,
)
from ufuzz.materialization import (
    CertifiedExecutionObservation,
    CertifiedMaterialization,
    EphemeralMaterializedState,
    RootMaterializationRequest,
    TransitionReplay,
)
from ufuzz.evaluation_contract import Backend, NativeMemoryLLMProvider
from ufuzz.production_config_contract import (
    InvariantBackendSubstrateManifest,
    MEMOS_GENERAL_TEXT_PROFILE_MANIFEST,
    NativeMemoryLLMConfigurationManifest,
    ProviderSubstrateBinding,
    RQ4EmbeddingControlEvidence,
)
from ufuzz.retrieval_feedback import (
    BehaviorHistory,
    MutationRelation,
    retrieval_signature,
    score_observed_retrieval,
)
from ufuzz.retrieval_capability import (
    CapabilityOperation,
    CapabilityResult,
    RankedPhysicalRetrieval,
    RetrievableEntryInventory,
    RetrievableEntryProjection,
    RetrievableInventoryEntry,
    RetrievalCapabilityBlocker,
    assemble_campaign_coverage,
    build_frozen_checkpoint_inventory,
    compare_initialization_inventories,
    resolve_fuzzing_retrieval,
    resolve_initialization_signature,
)
from ufuzz.state_contract import (
    AffectedPhysicalTransition,
    AffectedTransitionKind,
    CertificationStatus,
    DescendantStateCertificate,
    ExecutableQueryArtifact,
    LiveLineageState,
    LogicalSeed,
    MaterializationFailure,
    MaterializationFailureKind,
    MaterializationResult,
    MutationOpportunity,
    PhysicalLineageEndpoint,
    PhysicalTransitionCertificate,
    PhysicalTransitionOutcome,
    RealizedMutationArtifact,
    RebindingKind,
    ReplayBinding,
    ReplayRebindingCertificate,
    SeedAdmission,
    SemanticMutationCertificate,
    build_rebound_lineage,
)
from ufuzz.structural import StructuralIndexBuilder


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "longmemeval_s_minimal.json"
EVALUATOR_KEYS = {
    "answer",
    "answers",
    "answer_session_ids",
    "has_answer",
    "evidence",
    "gold_answer",
    "gold_answers",
    "failure_label",
    "oracle_verdict",
    "failure_surface",
    "uf_at_b",
    "api_key",
    "api_secret",
    "bearer_token",
    "credential",
    "credentials",
}


def _structured_key_paths(value: Any, path: str = "$", seen: set[int] | None = None):
    """Yield paths to forbidden structured keys while ignoring string content."""

    seen = seen or set()
    if isinstance(value, (str, bytes, int, float, bool, type(None))):
        return
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)

    if is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            if item.name.casefold() in EVALUATOR_KEYS:
                yield f"{path}.{item.name}"
            yield from _structured_key_paths(
                getattr(value, item.name),
                f"{path}.{item.name}",
                seen,
            )
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str) and key.casefold() in EVALUATOR_KEYS:
                yield f"{path}[{key!r}]"
            yield from _structured_key_paths(item, f"{path}[{key!r}]", seen)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for index, item in enumerate(value):
            yield from _structured_key_paths(item, f"{path}[{index}]", seen)
        return
    if hasattr(value, "__dict__"):
        yield from _structured_key_paths(vars(value), path, seen)


class _Mem0Sink:
    def __init__(self) -> None:
        self.entries: dict[str, dict[str, Any]] = {}

    def add(self, messages, **kwargs):
        local_id = f"m{len(self.entries) + 1}"
        self.entries[local_id] = {
            "id": local_id,
            "memory": messages[0]["content"],
            "metadata": kwargs.get("metadata", {}),
        }
        return {"results": [{"id": local_id}]}

    def get_all(self, **kwargs):
        return {"results": list(self.entries.values())}

    def reset(self):
        self.entries.clear()


class _AMemSink:
    def __init__(self) -> None:
        self.memories: dict[str, SimpleNamespace] = {}
        self.retriever = SimpleNamespace(collection=_AMemCollection(self.memories))

    def add_note(self, content, time=None, **kwargs):
        local_id = f"a{len(self.memories) + 1}"
        self.memories[local_id] = SimpleNamespace(
            id=local_id,
            content=content,
            context="General",
            keywords=[],
            timestamp=time,
            tags=kwargs.get("tags", []),
        )
        self.retriever.collection.ids.append(local_id)
        return local_id

    def delete(self, local_id):
        if local_id in self.retriever.collection.ids:
            self.retriever.collection.ids.remove(local_id)
        return self.memories.pop(local_id, None) is not None


class _AMemCollection:
    def __init__(self, memories) -> None:
        self.name = "memories"
        self.memories = memories
        self.ids: list[str] = []

    def get(self):
        return {"ids": list(self.ids)}

    def count(self):
        return len(self.ids)


class _GraphitiSink:
    def __init__(self) -> None:
        self.episodes: dict[str, SimpleNamespace] = {}
        self.episode_nodes: dict[str, SimpleNamespace] = {}
        self.entity_nodes: dict[str, SimpleNamespace] = {}
        self.add_calls: list[dict[str, Any]] = []
        self.edges = SimpleNamespace(entity=_GraphitiEntityEdges(self))
        self.nodes = SimpleNamespace(
            entity=_GraphitiObjectsByUuid(self.entity_nodes),
            episode=_GraphitiObjectsByUuid(self.episode_nodes),
        )

    async def add_episode(self, **kwargs):
        self.add_calls.append(kwargs)
        episode_uuid = kwargs["uuid"]
        source_uuid = str(uuid5(NAMESPACE_URL, f"source:{episode_uuid}"))
        target_uuid = str(uuid5(NAMESPACE_URL, f"target:{episode_uuid}"))
        edge_uuid = str(uuid5(NAMESPACE_URL, f"edge:{episode_uuid}"))
        group_id = kwargs["group_id"]
        self.entity_nodes[source_uuid] = SimpleNamespace(
            uuid=source_uuid,
            group_id=group_id,
            name="Source",
            summary="Synthetic source",
            labels=["Entity"],
            attributes={},
        )
        self.entity_nodes[target_uuid] = SimpleNamespace(
            uuid=target_uuid,
            group_id=group_id,
            name="Target",
            summary="Synthetic target",
            labels=["Entity"],
            attributes={},
        )
        self.episode_nodes[episode_uuid] = SimpleNamespace(
            uuid=episode_uuid,
            group_id=group_id,
            source_description=kwargs["source_description"],
        )
        edge = SimpleNamespace(
            uuid=edge_uuid,
            source_node_uuid=source_uuid,
            target_node_uuid=target_uuid,
            name="REMEMBERS",
            fact=kwargs["episode_body"],
            fact_embedding=None,
            episodes=[episode_uuid],
            group_id=group_id,
            valid_at=kwargs["reference_time"],
            invalid_at=None,
            expired_at=None,
            reference_time=kwargs["reference_time"],
            attributes={},
        )
        self.episodes[episode_uuid] = edge
        return SimpleNamespace(edges=[edge])

    async def get_nodes_and_edges_by_episode(self, episode_ids):
        return SimpleNamespace(
            edges=[self.episodes[value] for value in episode_ids],
            nodes=[],
        )

    async def remove_episode(self, episode_uuid):
        self.episodes.pop(episode_uuid, None)
        self.episode_nodes.pop(episode_uuid, None)

    async def search(self, _query, *, group_ids, num_results):
        return [
            edge
            for edge in self.episodes.values()
            if edge.group_id in group_ids
        ][:num_results]


class _GraphitiEntityEdges:
    def __init__(self, sink: _GraphitiSink) -> None:
        self.sink = sink

    async def get_by_group_ids(self, group_ids, limit=None, uuid_cursor=None):
        del limit, uuid_cursor
        return [
            edge
            for edge in self.sink.episodes.values()
            if edge.group_id in group_ids
        ]

    async def load_embeddings(self, edge):
        edge.fact_embedding = [0.25, 0.75]


class _GraphitiObjectsByUuid:
    def __init__(self, values: dict[str, SimpleNamespace]) -> None:
        self.values = values

    async def get_by_uuids(self, uuids):
        return [self.values[value] for value in uuids if value in self.values]


class InformationFlowRegressionTests(unittest.TestCase):
    def test_memos_profile_manifest_contains_no_credentials_or_evaluator_state(self) -> None:
        manifest = MEMOS_GENERAL_TEXT_PROFILE_MANIFEST
        self.assertSearchSafe(manifest)
        artifact = manifest.artifact_bytes.lower()
        for forbidden in (
            b"api_key", b"api_secret", b"bearer_token", b"credential",
            b"gold_answer", b"failure_surface", b"uf_at_b",
        ):
            self.assertNotIn(forbidden, artifact)

    def test_native_memory_llm_manifest_contains_configuration_not_credentials(self) -> None:
        manifest = NativeMemoryLLMConfigurationManifest(
            NativeMemoryLLMProvider.OPENAI,
            "model-id",
            "model-revision",
            b'{"system":"memory extraction"}',
            b'{"type":"object"}',
            "parser-v1",
            b'{"temperature":0}',
            b'{"transport_retries":3,"resamples":2}',
            "api-runtime-v1",
        )
        self.assertSearchSafe(manifest)
        self.assertFalse(
            EVALUATOR_KEYS.intersection(
                field.name.casefold() for field in fields(manifest)
            )
        )
        artifact = manifest.artifact_bytes.lower()
        for forbidden in (b"api_key", b"api_secret", b"bearer_token", b"credential"):
            self.assertNotIn(forbidden, artifact)

    def test_rq4_substrate_evidence_contains_public_configuration_not_credentials(self) -> None:
        substrate = InvariantBackendSubstrateManifest(
            Backend.MEM0,
            "mem0-source-2.0.12",
            "native-profile-b-v1",
            b'{"database":"isolated-qdrant"}',
            "huggingface",
            "embedding-model-id",
            "embedding-revision",
            768,
            "l2-normalized",
            b'{"search":"dense-cosine"}',
            b'{"reranker":"none"}',
            10,
            "isolated-replay-v1",
        )
        evidence = RQ4EmbeddingControlEvidence(
            Backend.MEM0,
            tuple(
                ProviderSubstrateBinding(provider, substrate)
                for provider in NativeMemoryLLMProvider
            ),
        )
        self.assertSearchSafe(evidence)
        artifact = substrate.artifact_bytes.lower()
        for forbidden in (b"api_key", b"api_secret", b"bearer_token", b"credential"):
            self.assertNotIn(forbidden, artifact)

    def assertSearchSafe(self, value: Any) -> None:
        self.assertEqual(list(_structured_key_paths(value)), [])

    def test_state_replay_contract_objects_are_search_safe(self) -> None:
        campaign = "state-contract-campaign"
        root = "checkpoint-1"
        coverage_id = CoverageEntryId(campaign, root, "e1")
        before = PhysicalEntryRef(campaign, root, "state-before", "memory-1")
        after = PhysicalEntryRef(campaign, root, "state-after", "memory-1")
        opportunity = MutationOpportunity.create(
            opportunity_id="op-1",
            campaign_id=campaign,
            root_checkpoint_id=root,
            parent_seed_id="seed-parent",
            relation=MutationRelation.UPDATE,
            canonical_target={"region": ["Ada", "employer"]},
            target_lineage=(coverage_id,),
            applicability_evidence={"semantic_region": "Ada/employer"},
            generation_constraints={"replacement_type": "organization"},
            possible_transition_outcomes={PhysicalTransitionOutcome.SAME_ID},
            acceptable_transition_outcomes={PhysicalTransitionOutcome.SAME_ID},
        )
        semantic = SemanticMutationCertificate(
            "semantic-1",
            opportunity.opportunity_id,
            campaign,
            root,
            opportunity.relation,
            True,
            {"obligation": "replace exact semantic value"},
            "semantic-proof",
        )
        artifact = RealizedMutationArtifact(
            "artifact-1",
            opportunity,
            semantic,
            "query-1",
            None,
            "replace_value",
            {"replacement": "Analytical Engines Ltd"},
        )
        affected = AffectedPhysicalTransition(
            "transition-1",
            AffectedTransitionKind.SAME_ID_CHANGED,
            (PhysicalLineageEndpoint(before, (coverage_id,)),),
            (PhysicalLineageEndpoint(after, (coverage_id,)),),
            backend_proof_digest="backend-proof",
        )
        transition = PhysicalTransitionCertificate(
            "physical-1",
            artifact.artifact_id,
            semantic.certificate_id,
            campaign,
            root,
            frozenset({coverage_id}),
            CertificationStatus.CERTIFIED,
            PhysicalTransitionOutcome.SAME_ID,
            affected.transition_id,
            (affected,),
            affected.predecessor_refs,
            affected.successor_refs,
            True,
            "complete-proof",
        )
        replay_binding = ReplayBinding(after, (coverage_id,), "binding-proof")
        rebinding = ReplayRebindingCertificate(
            "rebinding-1",
            RebindingKind.DESCENDANT,
            "synthetic",
            "config-v1",
            campaign,
            root,
            frozenset({coverage_id}),
            (replay_binding,),
            ((coverage_id,),),
            frozenset(),
            "rebinding-proof",
        )
        descendant = DescendantStateCertificate(
            "descendant-1",
            "synthetic",
            "config-v1",
            campaign,
            root,
            frozenset({coverage_id}),
            (LiveLineageState((coverage_id,), "projection", "provenance"),),
            frozenset(),
            "observable-state",
            "transition-state",
            (transition.certificate_id,),
        )
        seed = LogicalSeed(
            "seed-parent",
            campaign,
            "synthetic",
            root,
            "config-v1",
            ExecutableQueryArtifact.create(
                artifact_id="query-artifact-1",
                executable_text="Where does Ada work?",
                metadata={"query_id": "query-1"},
            ),
            (artifact,),
            descendant,
            ((coverage_id,),),
        )
        replay_inventory = FrozenCheckpointInventory(
            campaign,
            root,
            (
                InitialCoverageEntry(
                    coverage_id,
                    root,
                    "memory-1",
                    {"content": "search-safe state"},
                ),
            ),
        )
        replay_lineage = build_rebound_lineage(rebinding, replay_inventory)
        ephemeral = EphemeralMaterializedState(
            "opaque-live-handle",
            "synthetic",
            "config-v1",
            campaign,
            root,
            "state-after",
        )
        transition_replay = TransitionReplay(ephemeral, transition)
        root_request = RootMaterializationRequest(
            "synthetic",
            "config-v1",
            campaign,
            root,
            "initialization-artifact-1",
            frozenset({coverage_id}),
        )
        certified_materialization = CertifiedMaterialization(
            seed.seed_id,
            ephemeral,
            replay_lineage,
            rebinding,
            descendant,
            (transition,),
        )
        ranked = RankedPhysicalRetrieval(
            "synthetic",
            "Synthetic.search",
            "SyntheticMemory",
            campaign,
            root,
            "state-after",
            (after,),
        )
        replay_observation = resolve_fuzzing_retrieval(
            ranked,
            replay_lineage,
            CoverageState(frozenset({coverage_id})),
        )
        certified_execution = CertifiedExecutionObservation(
            seed.seed_id,
            certified_materialization,
            replay_observation,
        )
        admission = SeedAdmission.for_executed_child(
            execution_valid=True,
            rematerializable=True,
        )
        failure = MaterializationFailure(
            MaterializationFailureKind.TRANSIENT_BACKEND_FAILURE,
            campaign,
            root,
            "temporary local backend failure",
        )
        result = MaterializationResult[LogicalSeed].failed(failure)
        for value in (
            opportunity,
            semantic,
            artifact,
            affected,
            transition,
            replay_binding,
            rebinding,
            descendant,
            seed,
            root_request,
            ephemeral,
            transition_replay,
            certified_materialization,
            certified_execution,
            admission,
            failure,
            result,
        ):
            self.assertSearchSafe(value)

    def test_evaluator_keys_do_not_cross_into_search_or_ingestion(self) -> None:
        checkpoint = next(LongMemEvalSLoader().load(FIXTURE, verify_artifact=False))
        archival_paths = list(_structured_key_paths(checkpoint.raw))
        self.assertTrue(archival_paths)

        index = StructuralIndexBuilder().build(checkpoint)
        artifact = InitializationArtifact.create(
            checkpoint.checkpoint_id,
            checkpoint.sources,
            {"purpose": "information-flow-regression"},
        )
        self.assertSearchSafe(index)
        self.assertSearchSafe(index.provenance)
        for intent in index.query_intents.values():
            self.assertSearchSafe(intent)
        self.assertSearchSafe(artifact)
        for source in artifact.sources:
            self.assertSearchSafe(source_metadata(source))

        coverage_id = CoverageEntryId(
            "information-flow-campaign",
            checkpoint.checkpoint_id,
            "entry-1",
        )
        coverage_entry = InitialCoverageEntry(
            coverage_id=coverage_id,
            root_checkpoint_id=checkpoint.checkpoint_id,
            initial_backend_entry_id="backend-entry-1",
            initial_observable_projection={"content": "search-safe projection"},
            provenance_ids=(checkpoint.sources[0].provenance_id,),
        )
        inventory = FrozenCheckpointInventory(
            "information-flow-campaign",
            checkpoint.checkpoint_id,
            (coverage_entry,),
        )
        lineage = InitialEntryLineage("information-flow-campaign", (inventory,))
        physical_entry = PhysicalEntryRef(
            "information-flow-campaign",
            checkpoint.checkpoint_id,
            "state-1",
            "backend-entry-1",
        )
        binding = lineage.certify_binding(
            physical_entry,
            (coverage_id,),
            LineageRelation.INITIAL,
        )
        signature = retrieval_signature((binding.coverage_ids,))
        history = BehaviorHistory(
            "information-flow-campaign",
            {checkpoint.checkpoint_id: (signature,)},
        )
        coverage = CoverageState.from_entries((coverage_entry,))
        coverage_update = coverage.observe((binding.coverage_ids,))
        feedback = score_observed_retrieval(
            root_checkpoint_id=checkpoint.checkpoint_id,
            observed_signature=signature,
            coverage_update=coverage_update,
            retrieval_depth=2,
            mutation_relation=MutationRelation.UPDATE,
            history=history,
        )

        entry_projection = RetrievableEntryProjection(
            "synthetic",
            "SyntheticMemory",
            {"content": "search-safe projection"},
        )
        capability_inventory = RetrievableEntryInventory(
            backend="synthetic",
            selected_public_retrieval_api="Synthetic.search",
            retrieval_object_class="SyntheticMemory",
            retrieval_returnability_basis="same public result object class",
            campaign_id="information-flow-campaign",
            root_checkpoint_id=checkpoint.checkpoint_id,
            state_id="state-1",
            entries=(
                RetrievableInventoryEntry(
                    physical_entry,
                    entry_projection,
                    (checkpoint.sources[0].provenance_id,),
                ),
            ),
        )
        equivalence = compare_initialization_inventories(
            capability_inventory,
            capability_inventory,
        )
        frozen_build = build_frozen_checkpoint_inventory(capability_inventory)
        campaign_coverage = assemble_campaign_coverage(
            "information-flow-campaign",
            (frozen_build,),
        )
        ranked_retrieval = RankedPhysicalRetrieval(
            "synthetic",
            "Synthetic.search",
            "SyntheticMemory",
            "information-flow-campaign",
            checkpoint.checkpoint_id,
            "state-1",
            (physical_entry,),
        )
        initialization_signature = resolve_initialization_signature(
            ranked_retrieval,
            campaign_coverage.lineage,
        )
        resolved_observation = resolve_fuzzing_retrieval(
            ranked_retrieval,
            campaign_coverage.lineage,
            campaign_coverage.coverage_state,
        )
        blocker = RetrievalCapabilityBlocker(
            "synthetic",
            checkpoint.checkpoint_id,
            CapabilityOperation.INVENTORY,
            "search-side capability diagnostic",
        )
        capability_failure = CapabilityResult.failure(blocker)
        for search_side_value in (
            coverage_id,
            coverage_entry,
            inventory,
            lineage,
            physical_entry,
            binding,
            signature,
            history,
            coverage,
            coverage_update,
            feedback,
            entry_projection,
            capability_inventory,
            equivalence,
            frozen_build,
            campaign_coverage,
            ranked_retrieval,
            initialization_signature,
            resolved_observation,
            blocker,
            capability_failure,
        ):
            self.assertSearchSafe(search_side_value)

        async def exercise_adapters() -> None:
            mem0_sink = _Mem0Sink()
            mem0 = Mem0Adapter(memory_factory=lambda _: mem0_sink, infer=False)
            mem0_state = await mem0.create_isolated_state(artifact)
            await mem0.ingest(mem0_state, artifact.sources[:1])
            mem0_capability = Mem0RetrievableEntryCapability(mem0_state)
            mem0_inventory = await mem0_capability.inventory(
                campaign_id="information-flow-campaign",
                root_checkpoint_id=checkpoint.checkpoint_id,
                state_id=mem0_state.state_id,
            )
            self.assertSearchSafe(mem0_state.metadata)
            self.assertSearchSafe(mem0._provenance)
            self.assertSearchSafe(mem0_sink.entries)
            self.assertSearchSafe(mem0_capability.scope)
            self.assertSearchSafe(mem0_inventory)
            await mem0.teardown(mem0_state)

            amem_sink = _AMemSink()
            amem = AMemAdapter(memory_factory=lambda: amem_sink)
            amem_state = await amem.create_isolated_state(artifact)
            await amem.ingest(amem_state, artifact.sources[:1])
            amem_capability = AMemRetrievableEntryCapability(amem_state)
            amem_inventory = await amem_capability.inventory(
                campaign_id="information-flow-campaign",
                root_checkpoint_id=checkpoint.checkpoint_id,
                state_id=amem_state.state_id,
            )
            self.assertSearchSafe(amem_state.metadata)
            self.assertSearchSafe(amem._provenance)
            self.assertSearchSafe(amem_sink.memories)
            self.assertSearchSafe(amem_capability.scope)
            self.assertSearchSafe(amem_inventory)
            await amem.teardown(amem_state)

            graphiti_sink = _GraphitiSink()
            graphiti = GraphitiAdapter(graphiti=graphiti_sink)
            graphiti_state = await graphiti.create_isolated_state(artifact)
            await graphiti.ingest(graphiti_state, artifact.sources[:1])
            graphiti_capability = GraphitiRetrievableEntryCapability(graphiti_state)
            graphiti_inventory = await graphiti_capability.inventory(
                campaign_id="information-flow-campaign",
                root_checkpoint_id=checkpoint.checkpoint_id,
                state_id=graphiti_state.state_id,
            )
            self.assertSearchSafe(graphiti_state.metadata)
            self.assertSearchSafe(graphiti._episode_provenance)
            self.assertSearchSafe(graphiti_capability.scope)
            self.assertSearchSafe(graphiti_inventory)
            descriptions = [
                json.loads(call["source_description"])
                for call in graphiti_sink.add_calls
            ]
            self.assertSearchSafe(descriptions)
            await graphiti.teardown(graphiti_state)

        asyncio.run(exercise_adapters())


if __name__ == "__main__":
    unittest.main()
