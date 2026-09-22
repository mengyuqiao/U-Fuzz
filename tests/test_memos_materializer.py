from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import unittest

from tests.memos_test_support import GeneralTextDouble, item_factory
from ufuzz.backends.base import InitializationArtifact
from ufuzz.backends.memos import MEMOS_EMBEDDING_MANIFEST_DIGEST, MEMOS_PROFILE_ID, MemosAdapter
from ufuzz.backends.memos_capability import MemosRetrievableEntryCapability
from ufuzz.backends.memos_materializer import (
    MEMOS_GENERAL_TEXT_CONFIGURATION_ID, MemosGeneralTextMaterializationProtocol,
)
from ufuzz.domain import SourceUnit
from ufuzz.materialization import RootMaterializationRequest
from ufuzz.retrieval_capability import build_frozen_checkpoint_inventory
from ufuzz.retrieval_feedback import MutationRelation
from ufuzz.state_contract import (
    ExecutableQueryArtifact, MutationOpportunity, PhysicalTransitionOutcome,
    RealizedMutationArtifact, SemanticMutationCertificate,
)


CAMPAIGN, CHECKPOINT, ARTIFACT_ID = "memos-campaign", "memos-root", "memos-init"


def source(index, text):
    return SourceUnit("synthetic", CHECKPOINT, "session", f"source-{index}", index,
                      text, f"2025-01-0{index + 1}T00:00:00+00:00", "user", "user", {})


SOURCES = (
    source(0, "Alice lives in Boston."),
    source(1, "Bob enjoys tennis."),
    source(2, "Carol owns a cat."),
)


class MemosMaterializerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.class_tmp = tempfile.TemporaryDirectory()
        cls.artifact = InitializationArtifact.create(
            CHECKPOINT, SOURCES, {"memos_profile_id": MEMOS_PROFILE_ID}
        )
        async def bootstrap():
            adapter = cls.adapter(Path(cls.class_tmp.name) / "bootstrap")
            state = await adapter.replay_state(cls.artifact)
            current = (await MemosRetrievableEntryCapability(state).inventory(
                campaign_id=CAMPAIGN, root_checkpoint_id=CHECKPOINT, state_id=state.state_id
            )).require()
            frozen = build_frozen_checkpoint_inventory(current).inventory
            await adapter.teardown(state)
            return frozen
        cls.inventory = asyncio.run(bootstrap())

    @classmethod
    def tearDownClass(cls):
        cls.class_tmp.cleanup()

    @staticmethod
    def adapter(root):
        return MemosAdapter(root_dir=root, memory_factory=GeneralTextDouble,
                            item_factory=item_factory,
                            embedding_identity_resolver=lambda: MEMOS_EMBEDDING_MANIFEST_DIGEST,
                            runtime_identity_verifier=lambda: None)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.protocol = MemosGeneralTextMaterializationProtocol(
            initialization_artifacts={ARTIFACT_ID: self.artifact},
            frozen_inventories={(CAMPAIGN, CHECKPOINT): self.inventory},
            adapter=self.adapter(Path(self.tmp.name) / "states"),
        )

    def tearDown(self):
        async def cleanup():
            for state_id, context in tuple(self.protocol._states.items()):
                from ufuzz.materialization import EphemeralMaterializedState
                await self.protocol.discard(EphemeralMaterializedState(
                    context.handle, "memos", MEMOS_GENERAL_TEXT_CONFIGURATION_ID,
                    CAMPAIGN, CHECKPOINT, state_id))
        asyncio.run(cleanup())
        self.tmp.cleanup()

    def request(self):
        return RootMaterializationRequest(
            "memos", MEMOS_GENERAL_TEXT_CONFIGURATION_ID, CAMPAIGN, CHECKPOINT,
            ARTIFACT_ID, self.inventory.coverage_ids)

    async def root(self):
        result = await self.protocol.materialize_root(self.request())
        self.assertIsNone(result.failure, result.failure)
        return result.value

    def coverage(self, index):
        return next(entry.coverage_id for entry in self.inventory.entries
                    if entry.provenance_ids == (SOURCES[index].provenance_id,))

    def mutation(self, name, relation, index, text=None):
        outcome = PhysicalTransitionOutcome.DELETED if relation is MutationRelation.DELETION else PhysicalTransitionOutcome.SAME_ID
        target = self.coverage(index)
        opportunity = MutationOpportunity.create(
            opportunity_id=f"op-{name}", campaign_id=CAMPAIGN,
            root_checkpoint_id=CHECKPOINT, parent_seed_id=f"parent-{name}",
            relation=relation, canonical_target={"coverage": target.opaque_id},
            target_lineage=(target,), applicability_evidence={"exact": True},
            generation_constraints={"type_compatible": True},
            possible_transition_outcomes={outcome}, acceptable_transition_outcomes={outcome})
        semantic = SemanticMutationCertificate(
            f"semantic-{name}", opportunity.opportunity_id, CAMPAIGN, CHECKPOINT,
            relation, True, {"target": target.opaque_id}, f"proof-{name}")
        payload = {"operation": "delete"} if relation is MutationRelation.DELETION else {
            "operation": "unrelated_change" if relation is MutationRelation.UNRELATED_CHANGE else "update",
            "new_text": text,
            "provenance_ids": (SOURCES[index].provenance_id,),
        }
        return RealizedMutationArtifact(name, opportunity, semantic, "query", None,
                                        relation.value, payload)

    def test_root_rebinding_retrieval_and_reopen(self):
        async def exercise():
            first = await self.root()
            self.assertEqual(len(first.rebinding_certificate.live_bindings), 3)
            result = await self.protocol.retrieve(
                first.state, query_artifact=ExecutableQueryArtifact.create(
                    artifact_id="query", executable_text="Alice Boston", metadata={}), top_k=2)
            self.assertIsNone(result.failure)
            self.assertLessEqual(len(result.value.ranked_physical_entries), 2)
            first_observed = first.observed_root_state
            await self.protocol.discard(first.state)
            second = await self.root()
            self.assertEqual(first_observed, second.observed_root_state)
        asyncio.run(exercise())

    def test_update_delete_and_unrelated_change_certificates(self):
        async def exercise():
            root = await self.root()
            update = self.mutation("update", MutationRelation.UPDATE, 0, "Alice lives in Seattle.")
            result = await self.protocol.replay_transition(root.state, update)
            self.assertIsNone(result.failure, result.failure)
            self.assertIs(result.value.certificate.observed_outcome, PhysicalTransitionOutcome.SAME_ID)
            unrelated = self.mutation("urc", MutationRelation.UNRELATED_CHANGE, 1, "Bob enjoys golf.")
            result2 = await self.protocol.replay_transition(root.state, unrelated)
            self.assertIsNone(result2.failure, result2.failure)
            deletion = self.mutation("delete", MutationRelation.DELETION, 2)
            result3 = await self.protocol.replay_transition(root.state, deletion)
            self.assertIsNone(result3.failure, result3.failure)
            self.assertIs(result3.value.certificate.observed_outcome, PhysicalTransitionOutcome.DELETED)
            descendant = await self.protocol.certify_descendant(
                root.state, (result.value.certificate, result2.value.certificate, result3.value.certificate))
            self.assertIsNone(descendant.failure, descendant.failure)
            self.assertEqual(descendant.value.deleted_ids, frozenset({self.coverage(2)}))
        asyncio.run(exercise())

    def test_rejects_independent_addition_during_unrelated_change(self):
        async def exercise():
            root = await self.root()
            original = root.state.handle.backend_state.update
            def add_instead(identifier, item):
                root.state.handle.backend_state.add([item])
            root.state.handle.backend_state.update = add_instead
            artifact = self.mutation("bad-urc", MutationRelation.UNRELATED_CHANGE, 1, "Bob enjoys golf.")
            result = await self.protocol.replay_transition(root.state, artifact)
            self.assertIsNotNone(result.failure)
            root.state.handle.backend_state.update = original
        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
