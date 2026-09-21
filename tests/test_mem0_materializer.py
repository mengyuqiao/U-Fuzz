from __future__ import annotations

import asyncio
from dataclasses import replace
import importlib.metadata
import importlib.util
import math
from pathlib import Path
import tempfile
from unittest.mock import AsyncMock, patch
import unittest
from uuid import uuid4

from ufuzz.backends.base import InitializationArtifact, OperationReceipt
from ufuzz.backends.mem0 import Mem0Adapter
from ufuzz.backends.mem0_capability import Mem0RetrievableEntryCapability
import ufuzz.backends.mem0_materializer as mem0_materializer
from ufuzz.backends.mem0_materializer import (
    MEM0_PROFILE_A_CONFIGURATION_ID,
    Mem0ProfileAMaterializationProtocol,
)
from ufuzz.coverage import CoverageEntryId, CoverageState, FrozenMapping
from ufuzz.domain import SourceUnit
from ufuzz.materialization import (
    EphemeralMaterializedState,
    RootMaterializationRequest,
    materialize_and_observe,
    materialize_seed,
    observe_materialization,
)
from ufuzz.retrieval_capability import build_frozen_checkpoint_inventory
from ufuzz.retrieval_feedback import MutationRelation
from ufuzz.state_contract import (
    ExecutableQueryArtifact,
    LogicalSeed,
    MaterializationFailureKind,
    MutationOpportunity,
    PhysicalTransitionOutcome,
    RealizedMutationArtifact,
    SemanticMutationCertificate,
    descendant_states_equivalent,
)


CAMPAIGN = "mem0-profile-a-campaign"
CHECKPOINT = "mem0-profile-a-checkpoint"
INITIALIZATION_ID = "mem0-profile-a-initialization"


def _source(index: int, text: str) -> SourceUnit:
    return SourceUnit(
        benchmark="profile-a-audit",
        checkpoint_id=CHECKPOINT,
        session_id="session-1",
        source_id=f"source-{index}",
        ordinal=index,
        text=text,
        timestamp=None,
        speaker="user",
        role="user",
        raw={},
    )


SOURCES = (
    _source(1, "Alice enjoys watercolor painting in Boston."),
    _source(2, "Bob repairs bicycles in Seattle."),
    _source(3, "Carol studies marine biology in Miami."),
)


def _artifact() -> InitializationArtifact:
    return InitializationArtifact.create(CHECKPOINT, SOURCES, {"infer": False})


async def _close_bootstrap(adapter: Mem0Adapter, handle) -> None:
    with mem0_materializer._profile_a_runtime():
        await adapter.teardown(handle)
    try:
        handle.backend_state.close()
    except Exception:
        pass


class Mem0ProfileAMaterializerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._class_tmp = tempfile.TemporaryDirectory()
        cls.artifact = _artifact()

        async def bootstrap():
            adapter = Mem0Adapter(
                config=mem0_materializer._profile_a_config(),
                root_dir=Path(cls._class_tmp.name) / "bootstrap",
                memory_factory=mem0_materializer._create_profile_a_memory,
                infer=False,
                allow_external_store=False,
            )
            with mem0_materializer._profile_a_runtime():
                handle = await adapter.replay_state(cls.artifact)
                result = await Mem0RetrievableEntryCapability(handle).inventory(
                    campaign_id=CAMPAIGN,
                    root_checkpoint_id=CHECKPOINT,
                    state_id=handle.state_id,
                )
            inventory = result.require()
            frozen = build_frozen_checkpoint_inventory(inventory).inventory
            await _close_bootstrap(adapter, handle)
            return frozen

        cls.inventory = asyncio.run(bootstrap())
        cls.coverage = CoverageState.from_entries(cls.inventory.entries)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._class_tmp.cleanup()

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.protocol = self._new_protocol()

    def tearDown(self) -> None:
        async def discard_all() -> None:
            for context in tuple(self.protocol._states.values()):
                state = EphemeralMaterializedState(
                    context.handle,
                    "mem0",
                    MEM0_PROFILE_A_CONFIGURATION_ID,
                    CAMPAIGN,
                    CHECKPOINT,
                    context.handle.state_id,
                )
                await self.protocol.discard(state)

        asyncio.run(discard_all())
        self._tmp.cleanup()

    def _new_protocol(self, *, inventory=None, artifact=None):
        return Mem0ProfileAMaterializationProtocol(
            initialization_artifacts={INITIALIZATION_ID: artifact or self.artifact},
            frozen_inventories={
                (CAMPAIGN, CHECKPOINT): inventory or self.inventory
            },
            root_dir=Path(self._tmp.name) / f"protocol-{uuid4().hex}",
        )

    def _root_request(self, *, frozen_e0=None, initialization_id=INITIALIZATION_ID):
        return RootMaterializationRequest(
            "mem0",
            MEM0_PROFILE_A_CONFIGURATION_ID,
            CAMPAIGN,
            CHECKPOINT,
            initialization_id,
            self.inventory.coverage_ids if frozen_e0 is None else frozen_e0,
        )

    async def _root(self, protocol=None):
        result = await (protocol or self.protocol).materialize_root(
            self._root_request()
        )
        self.assertIsNone(result.failure)
        self.assertIsNotNone(result.value)
        return result.value

    def _coverage_for_source(self, source: SourceUnit) -> CoverageEntryId:
        return next(
            entry.coverage_id
            for entry in self.inventory.entries
            if entry.provenance_ids == (source.provenance_id,)
        )

    def _mutation(
        self,
        artifact_id: str,
        relation: MutationRelation,
        target: CoverageEntryId,
        *,
        new_text: str | None = None,
    ) -> RealizedMutationArtifact:
        outcome = (
            PhysicalTransitionOutcome.DELETED
            if relation is MutationRelation.DELETION
            else PhysicalTransitionOutcome.SAME_ID
        )
        opportunity = MutationOpportunity.create(
            opportunity_id=f"opportunity-{artifact_id}",
            campaign_id=CAMPAIGN,
            root_checkpoint_id=CHECKPOINT,
            parent_seed_id=f"parent-{artifact_id}",
            relation=relation,
            canonical_target={"coverage_id": target.opaque_id},
            target_lineage=(target,),
            applicability_evidence={"profile": "mem0-a"},
            generation_constraints={"exact": True},
            possible_transition_outcomes={outcome},
            acceptable_transition_outcomes={outcome},
        )
        semantic = SemanticMutationCertificate(
            f"semantic-{artifact_id}",
            opportunity.opportunity_id,
            CAMPAIGN,
            CHECKPOINT,
            relation,
            True,
            {"selected_target": target.opaque_id},
            f"semantic-proof-{artifact_id}",
        )
        if relation is MutationRelation.DELETION:
            payload = {"operation": "delete"}
        else:
            operation = (
                "unrelated_change"
                if relation is MutationRelation.UNRELATED_CHANGE
                else "update"
            )
            source = next(
                source
                for source in SOURCES
                if self._coverage_for_source(source) == target
            )
            payload = {
                "operation": operation,
                "new_text": new_text or "Bob works at OpenAI in San Francisco.",
                "provenance_ids": (source.provenance_id,),
            }
        return RealizedMutationArtifact(
            artifact_id,
            opportunity,
            semantic,
            "original-query",
            None,
            relation.value,
            payload,
        )

    async def _expected(self, artifacts=(), protocol=None):
        chosen = protocol or self.protocol
        root = await self._root(chosen)
        certificates = []
        for artifact in artifacts:
            result = await chosen.replay_transition(root.state, artifact)
            self.assertIsNone(result.failure, result.failure)
            certificates.append(result.value.certificate)
        result = await chosen.certify_descendant(root.state, tuple(certificates))
        self.assertIsNone(result.failure, result.failure)
        expected = result.value
        await chosen.discard(root.state)
        return expected

    def _query(self, text="What is OpenAI doing in San Francisco?"):
        return ExecutableQueryArtifact.create(
            artifact_id="query-artifact-1",
            executable_text=text,
            metadata={"benchmark_query_id": "query-1"},
        )

    def _seed(self, expected, artifacts=(), *, query=None):
        first = min(self.inventory.coverage_ids)
        return LogicalSeed(
            "logical-seed-1",
            CAMPAIGN,
            "mem0",
            CHECKPOINT,
            MEM0_PROFILE_A_CONFIGURATION_ID,
            query or self._query(),
            tuple(artifacts),
            expected,
            ((first,),),
        )

    def test_exact_profile_gate_and_profile_b_rejected(self) -> None:
        self.assertEqual(importlib.metadata.version("mem0ai"), "2.0.12")
        self.assertEqual(importlib.metadata.version("qdrant-client"), "1.18.0")
        self.assertEqual(importlib.metadata.version("spacy"), "3.8.14")
        self.assertEqual(importlib.metadata.version("en-core-web-sm"), "3.8.0")
        self.assertIsNone(importlib.util.find_spec("fastembed"))
        with self.assertRaisesRegex(ValueError, "infer=False"):
            self._new_protocol(
                artifact=InitializationArtifact.create(
                    CHECKPOINT, SOURCES, {"infer": True}
                )
            )
        with self.assertRaisesRegex(ValueError, "evaluator-only"):
            self._new_protocol(
                artifact=InitializationArtifact.create(
                    CHECKPOINT,
                    SOURCES,
                    {"infer": False, "audit": {"gold_answer": "secret"}},
                )
            )

    def test_state_evidence_is_type_preserving_and_sensitive(self) -> None:
        coverage = self._coverage_for_source(SOURCES[0])
        lineage = mem0_materializer._lineage_evidence((coverage,))
        base = {
            "lineage": lineage,
            "projection": FrozenMapping({"memory": "Exact"}),
            "provenance": (SOURCES[0].provenance_id,),
            "multiplicity": 1,
        }
        digest = mem0_materializer._stable_digest(base)
        self.assertEqual(digest, mem0_materializer._stable_digest(dict(base)))
        for changed in (
            {**base, "projection": FrozenMapping({"memory": "Changed"})},
            {**base, "provenance": (SOURCES[1].provenance_id,)},
            {**base, "lineage": mem0_materializer._lineage_evidence((self._coverage_for_source(SOURCES[1]),))},
            {**base, "multiplicity": 2},
        ):
            self.assertNotEqual(digest, mem0_materializer._stable_digest(changed))

    def test_root_rebinding_fresh_ids_and_empty_auxiliary_state(self) -> None:
        async def exercise() -> None:
            first = await self._root()
            second = await self._root()
            first_ids = {
                binding.physical_entry.backend_entry_id
                for binding in first.rebinding_certificate.live_bindings
            }
            second_ids = {
                binding.physical_entry.backend_entry_id
                for binding in second.rebinding_certificate.live_bindings
            }
            self.assertTrue(first_ids.isdisjoint(second_ids))
            self.assertEqual(
                {binding.lineage for binding in first.rebinding_certificate.live_bindings},
                {binding.lineage for binding in second.rebinding_certificate.live_bindings},
            )
            self.assertTrue(
                descendant_states_equivalent(
                    first.observed_root_state, second.observed_root_state
                )
            )
            for root in (first, second):
                context = self.protocol._states[root.state.state_id]
                public = await self.protocol._public_inventory(root.state, context)
                self.assertEqual(
                    self.protocol._auxiliary_snapshot(
                        root.state, context, public
                    ).canonical_rows,
                    (),
                )
                self.assertEqual(
                    len(root.observed_root_state.frozen_e0), len(SOURCES)
                )
                await self.protocol.discard(root.state)

        asyncio.run(exercise())

    def test_equal_projection_with_distinct_provenance_rebinds_uniquely(self) -> None:
        async def exercise() -> None:
            equal_sources = (
                _source(21, "The exact same public memory."),
                _source(22, "The exact same public memory."),
            )
            artifact = InitializationArtifact.create(
                CHECKPOINT, equal_sources, {"infer": False}
            )
            adapter = Mem0Adapter(
                config=mem0_materializer._profile_a_config(),
                root_dir=Path(self._tmp.name) / "distinct-provenance-bootstrap",
                memory_factory=mem0_materializer._create_profile_a_memory,
                infer=False,
                allow_external_store=False,
            )
            with mem0_materializer._profile_a_runtime():
                handle = await adapter.replay_state(artifact)
                established = (
                    await Mem0RetrievableEntryCapability(handle).inventory(
                        campaign_id=CAMPAIGN,
                        root_checkpoint_id=CHECKPOINT,
                        state_id=handle.state_id,
                    )
                ).require()
            self.assertEqual(
                established.entries[0].projection,
                established.entries[1].projection,
            )
            self.assertNotEqual(
                established.entries[0].provenance_ids,
                established.entries[1].provenance_ids,
            )
            frozen = build_frozen_checkpoint_inventory(established).inventory
            await _close_bootstrap(adapter, handle)
            protocol = self._new_protocol(inventory=frozen, artifact=artifact)
            result = await protocol.materialize_root(
                RootMaterializationRequest(
                    "mem0",
                    MEM0_PROFILE_A_CONFIGURATION_ID,
                    CAMPAIGN,
                    CHECKPOINT,
                    INITIALIZATION_ID,
                    frozen.coverage_ids,
                )
            )
            self.assertIsNone(result.failure, result.failure)
            self.assertEqual(
                len(result.value.rebinding_certificate.live_bindings), 2
            )
            await protocol.discard(result.value.state)

        asyncio.run(exercise())

    def test_materialize_then_observe_uses_exact_query_and_does_not_apply_coverage(self) -> None:
        async def exercise() -> None:
            expected = await self._expected()
            query = self._query("  Exact Query\nWith CASE!  ")
            seed = self._seed(expected, query=query)
            before = self.coverage
            result = await materialize_seed(
                seed=seed,
                protocol=self.protocol,
                inventory=self.inventory,
                expected_root_state=expected,
                initialization_artifact_id=INITIALIZATION_ID,
            )
            self.assertIsNone(result.failure)
            certified = result.value
            context = self.protocol._states[certified.state.state_id]
            self.assertIsNone(context.handle.backend_state._entity_store)
            with patch.object(
                context.capability,
                "retrieve",
                wraps=context.capability.retrieve,
            ) as retrieve:
                observed = await observe_materialization(
                    seed_id=seed.seed_id,
                    materialization=certified,
                    protocol=self.protocol,
                    query_artifact=query,
                    top_k=3,
                    coverage_state=before,
                )
            self.assertIsNone(observed.failure, observed.failure)
            self.assertEqual(
                retrieve.call_args.kwargs["query"], query.executable_text
            )
            self.assertIs(before, self.coverage)
            self.assertEqual(self.coverage.reached_ids, frozenset())
            public = await self.protocol._public_inventory(certified.state, context)
            self.assertEqual(
                self.protocol._auxiliary_snapshot(
                    certified.state, context, public
                ).canonical_rows,
                (),
            )
            await self.protocol.discard(certified.state)

        asyncio.run(exercise())

    def test_real_update_ranking_effect_and_three_run_replay_stability(self) -> None:
        async def exercise() -> None:
            root_expected = await self._expected()
            root_seed = self._seed(root_expected)
            initial = await materialize_and_observe(
                seed=root_seed,
                protocol=self.protocol,
                inventory=self.inventory,
                expected_root_state=root_expected,
                initialization_artifact_id=INITIALIZATION_ID,
                coverage_state=self.coverage,
                top_k=3,
            )
            self.assertIsNone(initial.failure, initial.failure)
            initial_signature = initial.value.observation.signature
            self.assertEqual(len(initial_signature), 3)
            target = initial_signature[-1][0]
            self.assertNotEqual(initial_signature[0], (target,))
            await self.protocol.discard(initial.value.materialization.state)

            update = self._mutation(
                "update-ranking",
                MutationRelation.UPDATE,
                target,
                new_text="Bob works at OpenAI in San Francisco.",
            )
            expected = await self._expected((update,))
            seed = self._seed(expected, (update,))
            descendant_certificates = []
            signatures = []
            public_id_sets = []
            entity_ids = []
            entity_vectors = []
            for _ in range(3):
                result = await materialize_and_observe(
                    seed=seed,
                    protocol=self.protocol,
                    inventory=self.inventory,
                    expected_root_state=root_expected,
                    initialization_artifact_id=INITIALIZATION_ID,
                    coverage_state=self.coverage,
                    top_k=3,
                )
                self.assertIsNone(result.failure, result.failure)
                value = result.value
                signatures.append(value.observation.signature)
                self.assertEqual(value.observation.signature[0], (target,))
                certified = value.materialization
                update_certificate = certified.transition_certificates[0]
                self.assertEqual(
                    update_certificate.observed_outcome,
                    PhysicalTransitionOutcome.SAME_ID,
                )
                self.assertEqual(
                    update_certificate.certificate_id,
                    "mem0-profile-a:transition:update-ranking",
                )
                descendant_certificates.append(certified.observed_descendant_state)
                context = self.protocol._states[certified.state.state_id]
                public = await self.protocol._public_inventory(certified.state, context)
                public_id_sets.append(
                    {entry.physical_entry.backend_entry_id for entry in public.entries}
                )
                points = self.protocol._enumerate_entity_points(
                    certified.state.handle.backend_state
                )
                self.assertEqual(len(points), 1)
                entity_ids.append(str(points[0].id))
                auxiliary = self.protocol._auxiliary_snapshot(
                    certified.state, context, public
                )
                self.assertEqual(len(auxiliary.canonical_rows), 1)
                entity_row = auxiliary.canonical_rows[0]
                vector = entity_row["dense_vector"]
                self.assertEqual(len(vector), 10)
                self.assertTrue(all(math.isfinite(value) for value in vector))
                entity_vectors.append(vector)
                self.assertEqual(
                    entity_row["scope"],
                    "current-materialization-user-scope",
                )
                self.assertEqual(entity_row["sparse_vector_state"], "absent")
                self.assertEqual(len(certified.observed_descendant_state.frozen_e0), 3)
                await self.protocol.discard(certified.state)

            self.assertEqual(signatures[0], signatures[1])
            self.assertEqual(signatures[1], signatures[2])
            self.assertTrue(public_id_sets[0].isdisjoint(public_id_sets[1]))
            self.assertTrue(public_id_sets[1].isdisjoint(public_id_sets[2]))
            self.assertEqual(len(set(entity_ids)), 3)
            self.assertEqual(entity_vectors[0], entity_vectors[1])
            self.assertEqual(entity_vectors[1], entity_vectors[2])
            self.assertEqual(
                {
                    certificate.observable_state_digest
                    for certificate in descendant_certificates
                },
                {descendant_certificates[0].observable_state_digest},
            )
            self.assertEqual(
                {
                    certificate.transition_state_digest
                    for certificate in descendant_certificates
                },
                {descendant_certificates[0].transition_state_digest},
            )
            self.assertEqual(
                {certificate.certificate_id for certificate in descendant_certificates},
                {descendant_certificates[0].certificate_id},
            )
            self.assertTrue(
                all(
                    descendant_states_equivalent(
                        descendant_certificates[0], candidate
                    )
                    for candidate in descendant_certificates[1:]
                )
            )

        asyncio.run(exercise())

    def test_delete_after_entity_creation_removes_link_and_keeps_denominator(self) -> None:
        async def exercise() -> None:
            root_expected = await self._expected()
            target = self._coverage_for_source(SOURCES[1])
            update = self._mutation(
                "before-delete", MutationRelation.UPDATE, target
            )
            delete = self._mutation(
                "delete-linked", MutationRelation.DELETION, target
            )
            expected = await self._expected((update, delete))
            seed = self._seed(expected, (update, delete))
            result = await materialize_and_observe(
                seed=seed,
                protocol=self.protocol,
                inventory=self.inventory,
                expected_root_state=root_expected,
                initialization_artifact_id=INITIALIZATION_ID,
                coverage_state=self.coverage,
                top_k=3,
            )
            self.assertIsNone(result.failure, result.failure)
            certified = result.value.materialization
            self.assertEqual(
                certified.transition_certificates[-1].observed_outcome,
                PhysicalTransitionOutcome.DELETED,
            )
            self.assertIn(target, certified.observed_descendant_state.deleted_ids)
            self.assertEqual(certified.observed_descendant_state.frozen_e0, self.inventory.coverage_ids)
            self.assertNotIn((target,), result.value.observation.signature)
            context = self.protocol._states[certified.state.state_id]
            public = await self.protocol._public_inventory(certified.state, context)
            auxiliary = self.protocol._auxiliary_snapshot(
                certified.state, context, public
            )
            self.assertEqual(auxiliary.canonical_rows, ())
            self.assertEqual(self.coverage.reached_ids, frozenset())
            await self.protocol.discard(certified.state)

        asyncio.run(exercise())

    def test_shared_entity_row_survives_with_remaining_lineage(self) -> None:
        async def exercise() -> None:
            first = self._coverage_for_source(SOURCES[0])
            second = self._coverage_for_source(SOURCES[1])
            artifacts = (
                self._mutation(
                    "shared-one",
                    MutationRelation.UPDATE,
                    first,
                    new_text="Alice works at OpenAI in San Francisco.",
                ),
                self._mutation(
                    "shared-two",
                    MutationRelation.UPDATE,
                    second,
                    new_text="Bob works at OpenAI in San Francisco.",
                ),
                self._mutation("shared-delete", MutationRelation.DELETION, first),
            )
            root = await self._root()
            certificates = []
            for artifact in artifacts:
                transition = await self.protocol.replay_transition(root.state, artifact)
                self.assertIsNone(transition.failure, transition.failure)
                certificates.append(transition.value.certificate)
            descendant = await self.protocol.certify_descendant(
                root.state, tuple(certificates)
            )
            self.assertIsNone(descendant.failure, descendant.failure)
            context = self.protocol._states[root.state.state_id]
            public = await self.protocol._public_inventory(root.state, context)
            auxiliary = self.protocol._auxiliary_snapshot(root.state, context, public)
            self.assertEqual(len(auxiliary.canonical_rows), 1)
            linked = auxiliary.canonical_rows[0]["linked_lineages"]
            self.assertEqual(
                linked,
                (((second.campaign_id, second.root_checkpoint_id, second.opaque_id),),),
            )
            await self.protocol.discard(root.state)

        asyncio.run(exercise())

    def test_unrelated_change_accepts_certified_native_ranking_state(self) -> None:
        async def exercise() -> None:
            root_expected = await self._expected()
            initial = await materialize_and_observe(
                seed=self._seed(root_expected),
                protocol=self.protocol,
                inventory=self.inventory,
                expected_root_state=root_expected,
                initialization_artifact_id=INITIALIZATION_ID,
                coverage_state=self.coverage,
                top_k=3,
            )
            self.assertIsNone(initial.failure, initial.failure)
            initial_signature = initial.value.observation.signature
            target = initial_signature[-1][0]
            await self.protocol.discard(initial.value.materialization.state)
            unrelated = self._mutation(
                "unrelated",
                MutationRelation.UNRELATED_CHANGE,
                target,
                new_text="This unrelated memory mentions OpenAI in San Francisco.",
            )
            expected = await self._expected((unrelated,))
            result = await materialize_and_observe(
                seed=self._seed(expected, (unrelated,)),
                protocol=self.protocol,
                inventory=self.inventory,
                expected_root_state=root_expected,
                initialization_artifact_id=INITIALIZATION_ID,
                coverage_state=self.coverage,
                top_k=3,
            )
            self.assertIsNone(result.failure, result.failure)
            self.assertNotEqual(result.value.observation.signature, initial_signature)
            self.assertEqual(result.value.observation.signature[0], (target,))
            certified = result.value.materialization
            certificate = certified.transition_certificates[0]
            self.assertEqual(
                certificate.certificate_id,
                f"mem0-profile-a:transition:{unrelated.artifact_id}",
            )
            self.assertEqual(
                certificate.realized_artifact_id, unrelated.artifact_id
            )
            self.assertEqual(
                certificate.semantic_certificate_id,
                unrelated.semantic_certificate.certificate_id,
            )
            self.assertEqual(
                certificate.observed_outcome, PhysicalTransitionOutcome.SAME_ID
            )
            self.assertTrue(certificate.semantic_postcondition_satisfied)
            target_group = certificate.affected[0]
            self.assertEqual(
                target_group.predecessors[0].physical_entry,
                target_group.successors[0].physical_entry,
            )
            context = self.protocol._states[certified.state.state_id]
            public = await self.protocol._public_inventory(certified.state, context)
            self.assertEqual(
                len(
                    self.protocol._auxiliary_snapshot(
                        certified.state, context, public
                    ).canonical_rows
                ),
                1,
            )
            await self.protocol.discard(certified.state)

        asyncio.run(exercise())

    def test_profile_a_lifecycle_never_initializes_or_sends_telemetry(self) -> None:
        async def exercise() -> None:
            protocol = self._new_protocol()
            with (
                patch(
                    "mem0.memory.telemetry._get_oss_telemetry",
                    side_effect=AssertionError("telemetry singleton requested"),
                ) as telemetry_singleton,
                patch(
                    "mem0.memory.telemetry.Posthog",
                    side_effect=AssertionError("PostHog client constructed"),
                ) as posthog_constructor,
                patch(
                    "posthog.client.Client.capture",
                    side_effect=AssertionError("PostHog capture invoked"),
                ) as posthog_capture,
                patch(
                    "posthog.client.Client.flush",
                    side_effect=AssertionError("PostHog flush invoked"),
                ) as posthog_flush,
                patch(
                    "posthog.client.Client.evaluate_flags",
                    side_effect=AssertionError("PostHog notices requested"),
                ) as posthog_flags,
            ):
                root = await self._root(protocol)
                context = protocol._states[root.state.state_id]
                await protocol._public_inventory(root.state, context)
                target = self._coverage_for_source(SOURCES[1])
                update = self._mutation(
                    "telemetry-update", MutationRelation.UPDATE, target
                )
                updated = await protocol.replay_transition(root.state, update)
                self.assertIsNone(updated.failure, updated.failure)
                retrieval = await protocol.retrieve(
                    root.state, query_artifact=self._query(), top_k=3
                )
                self.assertIsNone(retrieval.failure, retrieval.failure)
                deletion = self._mutation(
                    "telemetry-delete", MutationRelation.DELETION, target
                )
                deleted = await protocol.replay_transition(root.state, deletion)
                self.assertIsNone(deleted.failure, deleted.failure)
                await protocol.discard(root.state)

            for guarded_call in (
                telemetry_singleton,
                posthog_constructor,
                posthog_capture,
                posthog_flush,
                posthog_flags,
            ):
                guarded_call.assert_not_called()

        asyncio.run(exercise())

    def test_unexpected_backend_exceptions_are_transient_failures(self) -> None:
        async def exercise() -> None:
            protocol = self._new_protocol()
            with patch.object(
                protocol._adapter,
                "replay_state",
                new=AsyncMock(side_effect=RuntimeError("temporary root failure")),
            ):
                root_failure = await protocol.materialize_root(self._root_request())
            self.assertEqual(
                root_failure.failure.kind,
                MaterializationFailureKind.TRANSIENT_BACKEND_FAILURE,
            )

            root = await self._root(protocol)
            target = self._coverage_for_source(SOURCES[1])
            update = self._mutation(
                "transient-transition", MutationRelation.UPDATE, target
            )
            with patch.object(
                protocol,
                "_public_inventory",
                new=AsyncMock(side_effect=RuntimeError("temporary inventory failure")),
            ):
                transition_failure = await protocol.replay_transition(
                    root.state, update
                )
            self.assertEqual(
                transition_failure.failure.kind,
                MaterializationFailureKind.TRANSIENT_BACKEND_FAILURE,
            )
            with patch.object(
                protocol,
                "_public_inventory",
                new=AsyncMock(side_effect=RuntimeError("temporary inventory failure")),
            ):
                descendant_failure = await protocol.certify_descendant(
                    root.state, ()
                )
            self.assertEqual(
                descendant_failure.failure.kind,
                MaterializationFailureKind.TRANSIENT_BACKEND_FAILURE,
            )
            await protocol.discard(root.state)

        asyncio.run(exercise())

    def test_multi_state_isolation_and_cleanup(self) -> None:
        async def exercise() -> None:
            first = await self._root()
            second = await self._root()
            self.assertNotEqual(first.state.state_id, second.state.state_id)
            self.assertIsNot(first.state.handle.backend_state, second.state.handle.backend_state)
            target = self._coverage_for_source(SOURCES[1])
            update = self._mutation("isolated", MutationRelation.UPDATE, target)
            transitioned = await self.protocol.replay_transition(first.state, update)
            self.assertIsNone(transitioned.failure, transitioned.failure)
            first_context = self.protocol._states[first.state.state_id]
            second_context = self.protocol._states[second.state.state_id]
            first_public = await self.protocol._public_inventory(first.state, first_context)
            second_public = await self.protocol._public_inventory(second.state, second_context)
            self.assertEqual(
                len(self.protocol._auxiliary_snapshot(first.state, first_context, first_public).canonical_rows),
                1,
            )
            self.assertEqual(
                self.protocol._auxiliary_snapshot(second.state, second_context, second_public).canonical_rows,
                (),
            )
            first_ids = {
                entry.physical_entry.backend_entry_id for entry in first_public.entries
            }
            second_ids = {
                entry.physical_entry.backend_entry_id for entry in second_public.entries
            }
            self.assertTrue(first_ids.isdisjoint(second_ids))
            await self.protocol.discard(first.state)
            still_usable = await self.protocol.retrieve(
                second.state, query_artifact=self._query(), top_k=3
            )
            self.assertIsNone(still_usable.failure, still_usable.failure)
            await self.protocol.discard(first.state)
            await self.protocol.discard(second.state)

        asyncio.run(exercise())

    def test_root_failure_classes_and_duplicate_replay_key(self) -> None:
        async def exercise() -> None:
            unknown = await self.protocol.materialize_root(
                self._root_request(initialization_id="unknown")
            )
            self.assertEqual(
                unknown.failure.kind,
                MaterializationFailureKind.BACKEND_CAPABILITY_FAILURE,
            )
            extra = CoverageEntryId(CAMPAIGN, CHECKPOINT, "extra")
            wrong = await self.protocol.materialize_root(
                self._root_request(
                    frozen_e0=self.inventory.coverage_ids | {extra}
                )
            )
            self.assertEqual(
                wrong.failure.kind,
                MaterializationFailureKind.BACKEND_CAPABILITY_FAILURE,
            )

            first = self.inventory.entries[0]
            altered_projection = FrozenMapping(
                {
                    **dict(first.initial_observable_projection),
                    "memory": "different frozen root projection",
                }
            )
            altered_inventory = replace(
                self.inventory,
                entries=(
                    replace(
                        first,
                        initial_observable_projection=altered_projection,
                    ),
                    *self.inventory.entries[1:],
                ),
            )
            inequivalent_protocol = self._new_protocol(
                inventory=altered_inventory
            )
            inequivalent = await inequivalent_protocol.materialize_root(
                RootMaterializationRequest(
                    "mem0",
                    MEM0_PROFILE_A_CONFIGURATION_ID,
                    CAMPAIGN,
                    CHECKPOINT,
                    INITIALIZATION_ID,
                    altered_inventory.coverage_ids,
                )
            )
            self.assertEqual(
                inequivalent.failure.kind,
                MaterializationFailureKind.ROOT_INITIALIZATION_INEQUIVALENT,
            )

            duplicate_source = _source(10, "identical")
            duplicate_artifact = InitializationArtifact.create(
                CHECKPOINT,
                (duplicate_source, duplicate_source),
                {"infer": False},
            )
            adapter = Mem0Adapter(
                config=mem0_materializer._profile_a_config(),
                root_dir=Path(self._tmp.name) / "duplicate-bootstrap",
                memory_factory=mem0_materializer._create_profile_a_memory,
                infer=False,
                allow_external_store=False,
            )
            with mem0_materializer._profile_a_runtime():
                handle = await adapter.replay_state(duplicate_artifact)
                established = (
                    await Mem0RetrievableEntryCapability(handle).inventory(
                        campaign_id=CAMPAIGN,
                        root_checkpoint_id=CHECKPOINT,
                        state_id=handle.state_id,
                    )
                ).require()
            duplicate_inventory = build_frozen_checkpoint_inventory(established).inventory
            await _close_bootstrap(adapter, handle)
            duplicate_protocol = self._new_protocol(
                inventory=duplicate_inventory, artifact=duplicate_artifact
            )
            result = await duplicate_protocol.materialize_root(
                RootMaterializationRequest(
                    "mem0",
                    MEM0_PROFILE_A_CONFIGURATION_ID,
                    CAMPAIGN,
                    CHECKPOINT,
                    INITIALIZATION_ID,
                    duplicate_inventory.coverage_ids,
                )
            )
            self.assertEqual(
                result.failure.kind,
                MaterializationFailureKind.E0_REBINDING_AMBIGUOUS,
            )

        asyncio.run(exercise())

    def test_auxiliary_schema_failures_are_not_certified(self) -> None:
        async def exercise() -> None:
            root = await self._root()
            state = root.state
            context = self.protocol._states[state.state_id]
            memory = state.handle.backend_state
            store = memory.entity_store
            vector = [value / 10 for value in range(1, 11)]
            valid = {
                "data": "OpenAI",
                "entity_type": "ORG",
                "linked_memory_ids": [
                    root.rebinding_certificate.live_bindings[0].physical_entry.backend_entry_id
                ],
                "user_id": state.state_id,
            }
            entity_one = str(uuid4())
            entity_two = str(uuid4())
            store.insert(
                vectors=[vector, vector],
                ids=[entity_one, entity_two],
                payloads=[valid, valid],
            )
            result = await self.protocol.certify_descendant(state, ())
            self.assertEqual(
                result.failure.kind,
                MaterializationFailureKind.DESCENDANT_STATE_INEQUIVALENT,
            )
            store.delete(vector_id=entity_two)
            store.update(
                vector_id=entity_one,
                vector=None,
                payload={**valid, "unknown": True},
            )
            result = await self.protocol.certify_descendant(state, ())
            self.assertEqual(
                result.failure.kind,
                MaterializationFailureKind.DESCENDANT_STATE_INEQUIVALENT,
            )
            store.update(
                vector_id=entity_one,
                vector=None,
                payload={**valid, "linked_memory_ids": ["unknown-memory"]},
            )
            result = await self.protocol.certify_descendant(state, ())
            self.assertEqual(
                result.failure.kind,
                MaterializationFailureKind.DESCENDANT_STATE_INEQUIVALENT,
            )
            store.update(
                vector_id=entity_one,
                vector=None,
                payload={**valid, "user_id": "another-state"},
            )
            result = await self.protocol.certify_descendant(state, ())
            self.assertEqual(
                result.failure.kind,
                MaterializationFailureKind.DESCENDANT_STATE_INEQUIVALENT,
            )
            with self.assertRaisesRegex(ValueError, "unexpected vector slot"):
                self.protocol._dense_vector({"": vector, "bm25": [1.0]})
            await self.protocol.discard(state)

        asyncio.run(exercise())

    def test_transition_failures_do_not_infer_replacement_or_choose_targets(self) -> None:
        async def exercise() -> None:
            root = await self._root()
            target = self._coverage_for_source(SOURCES[1])
            update = self._mutation("bad-receipt", MutationRelation.UPDATE, target)
            bad_receipt = OperationReceipt(
                "mem0",
                root.state.state_id,
                "update",
                False,
                "wrong",
                (),
                (),
                None,
                None,
                None,
                "mismatch",
            )
            with patch.object(
                self.protocol._adapter,
                "update",
                new=AsyncMock(return_value=bad_receipt),
            ):
                result = await self.protocol.replay_transition(root.state, update)
            self.assertEqual(
                result.failure.kind,
                MaterializationFailureKind.TRANSITION_REPLAY_DIVERGED,
            )
            await self.protocol.discard(root.state)

            root = await self._root()
            malformed = replace(
                update,
                operation_payload=FrozenMapping(
                    {
                        **dict(update.operation_payload),
                        "unused_extra": "must not be ignored",
                    }
                ),
            )
            result = await self.protocol.replay_transition(root.state, malformed)
            self.assertEqual(
                result.failure.kind,
                MaterializationFailureKind.TRANSITION_REPLAY_DIVERGED,
            )
            await self.protocol.discard(root.state)

            root = await self._root()
            context = self.protocol._states[root.state.state_id]
            original = self.protocol._adapter.update

            async def update_and_add(*args, **kwargs):
                receipt = await original(*args, **kwargs)
                with mem0_materializer._profile_a_runtime():
                    context.handle.backend_state.add(
                        "unexpected public record",
                        user_id=root.state.state_id,
                        infer=False,
                    )
                return receipt

            with patch.object(self.protocol._adapter, "update", new=update_and_add):
                result = await self.protocol.replay_transition(root.state, update)
            self.assertEqual(
                result.failure.kind,
                MaterializationFailureKind.TRANSITION_REPLAY_DIVERGED,
            )
            await self.protocol.discard(root.state)

            root = await self._root()
            deletion = self._mutation(
                "remove-before-replay", MutationRelation.DELETION, target
            )
            deleted = await self.protocol.replay_transition(root.state, deletion)
            self.assertIsNone(deleted.failure, deleted.failure)
            missing = await self.protocol.replay_transition(root.state, update)
            self.assertEqual(
                missing.failure.kind,
                MaterializationFailureKind.TRANSITION_REPLAY_DIVERGED,
            )
            await self.protocol.discard(root.state)

            root = await self._root()
            context = self.protocol._states[root.state.state_id]
            public = await self.protocol._public_inventory(root.state, context)
            real_mapping = self.protocol._current_lineage_map(context, public)
            target_token = (target,)
            backend_ids = [
                entry.physical_entry.backend_entry_id for entry in public.entries
            ]
            duplicate_mapping = dict(real_mapping)
            duplicate_mapping[backend_ids[0]] = target_token
            duplicate_mapping[backend_ids[1]] = target_token
            with patch.object(
                self.protocol,
                "_current_lineage_map",
                return_value=duplicate_mapping,
            ):
                multiple = await self.protocol.replay_transition(root.state, update)
            self.assertEqual(
                multiple.failure.kind,
                MaterializationFailureKind.TRANSITION_REPLAY_DIVERGED,
            )
            await self.protocol.discard(root.state)

            root = await self._root()
            context = self.protocol._states[root.state.state_id]
            public = await self.protocol._public_inventory(root.state, context)
            target_id = next(
                entry.physical_entry.backend_entry_id
                for entry in public.entries
                if self.protocol._current_lineage_map(context, public)[
                    entry.physical_entry.backend_entry_id
                ]
                == (target,)
            )
            collateral_id = next(
                entry.physical_entry.backend_entry_id
                for entry in public.entries
                if entry.physical_entry.backend_entry_id != target_id
            )
            original = self.protocol._adapter.update

            async def update_with_collateral(*args, **kwargs):
                receipt = await original(*args, **kwargs)
                with mem0_materializer._profile_a_runtime():
                    context.handle.backend_state.update(
                        collateral_id,
                        text="unexpected collateral public change",
                    )
                return receipt

            with patch.object(
                self.protocol._adapter,
                "update",
                new=update_with_collateral,
            ):
                collateral = await self.protocol.replay_transition(root.state, update)
            self.assertEqual(
                collateral.failure.kind,
                MaterializationFailureKind.TRANSITION_REPLAY_DIVERGED,
            )
            await self.protocol.discard(root.state)

        asyncio.run(exercise())

    def test_descendant_auxiliary_mismatch_and_wrong_state_retrieval(self) -> None:
        async def exercise() -> None:
            root_expected = await self._expected()
            target = self._coverage_for_source(SOURCES[1])
            update = self._mutation("digest-mismatch", MutationRelation.UPDATE, target)
            expected = await self._expected((update,))
            wrong = replace(expected, transition_state_digest="wrong-auxiliary-digest")
            seed = self._seed(wrong, (update,))
            result = await materialize_seed(
                seed=seed,
                protocol=self.protocol,
                inventory=self.inventory,
                expected_root_state=root_expected,
                initialization_artifact_id=INITIALIZATION_ID,
            )
            self.assertEqual(
                result.failure.kind,
                MaterializationFailureKind.DESCENDANT_STATE_INEQUIVALENT,
            )

            root = await self._root()
            wrong_state = EphemeralMaterializedState(
                root.state.handle,
                root.state.backend,
                root.state.frozen_configuration_id,
                root.state.campaign_id,
                root.state.root_checkpoint_id,
                "another-state",
            )
            retrieval = await self.protocol.retrieve(
                wrong_state, query_artifact=self._query(), top_k=3
            )
            self.assertEqual(
                retrieval.failure.kind,
                MaterializationFailureKind.BACKEND_CAPABILITY_FAILURE,
            )
            await self.protocol.discard(root.state)

        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
