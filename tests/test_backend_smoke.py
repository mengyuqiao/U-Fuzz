from __future__ import annotations

import asyncio
import importlib.util
import os
import unittest

from ufuzz.backends import AMemAdapter, GraphitiAdapter, InitializationArtifact, Mem0Adapter, MemosAdapter
from ufuzz.domain import (
    BenchmarkCheckpoint,
    BenchmarkQuery,
    DatasetArtifact,
    SourceUnit,
)
from ufuzz.mapping import FactLevelMapper, MappingStatus
from ufuzz.structural import (
    StructuralIndexBuilder,
    certify_exact_source_inventory,
    exact_source_fact,
)


# Test-only retrieval bound; this does not set the RQ1 experimental top-k.
SMOKE_TOP_K = 10


def _synthetic_checkpoint() -> tuple[BenchmarkCheckpoint, object]:
    artifact = DatasetArtifact(
        repository="synthetic",
        revision="phase-1-smoke-v1",
        filename="in-memory",
        sha256="0" * 64,
        size_bytes=0,
        url="synthetic://phase-1-smoke-v1",
    )
    rows = (
        ("turn-1", "Alice residence Boston.", "2024-01-01T00:00:00+00:00"),
        ("turn-2", "Bob hobby tennis.", "2024-01-02T00:00:00+00:00"),
        ("turn-3", "Carol pet cat.", "2024-01-03T00:00:00+00:00"),
        (
            "turn-4",
            "David occupation carpenter and Eve favorite_color green.",
            "2024-01-04T00:00:00+00:00",
        ),
    )
    sources = tuple(
        SourceUnit(
            benchmark="synthetic",
            checkpoint_id="phase-1-smoke",
            session_id="session-1",
            source_id=source_id,
            ordinal=ordinal,
            text=text,
            timestamp=timestamp,
            speaker="user",
            role="user",
            raw={"source_id": source_id, "text": text, "timestamp": timestamp},
        )
        for ordinal, (source_id, text, timestamp) in enumerate(rows)
    )
    checkpoint = BenchmarkCheckpoint(
        benchmark="synthetic",
        checkpoint_id="phase-1-smoke",
        sources=sources,
        queries=(
            BenchmarkQuery(
                benchmark="synthetic",
                checkpoint_id="phase-1-smoke",
                query_id="query-1",
                text="Where does Alice live?",
                timestamp=None,
                query_type="smoke",
                raw={"question": "Where does Alice live?"},
            ),
        ),
        raw={"checkpoint_id": "phase-1-smoke"},
        artifact=artifact,
    )

    fact_specs = (
        (sources[0], "Alice", "residence", "Boston"),
        (sources[1], "Bob", "hobby", "tennis"),
        (sources[2], "Carol", "pet", "cat"),
        (sources[3], "David", "occupation", "carpenter"),
        (sources[3], "Eve", "favorite_color", "green"),
    )
    facts = []
    certificates = []
    for source, entity, relation, value in fact_specs:
        fact, certificate = exact_source_fact(
            source=source,
            entity=entity,
            relation=relation,
            value=value,
        )
        facts.append(fact)
        certificates.append(certificate)
    carol_fact = facts[2]
    certificates.append(
        certify_exact_source_inventory(
            source=sources[2],
            facts=(carol_fact,),
        )
    )
    index = StructuralIndexBuilder().build(
        checkpoint,
        facts=facts,
        certificates=certificates,
    )
    return checkpoint, index


async def _exercise_backend(adapter, *, graphiti: bool = False) -> None:
    checkpoint, index = _synthetic_checkpoint()
    artifact = InitializationArtifact.create(
        checkpoint.checkpoint_id,
        checkpoint.sources,
        {"purpose": "phase-1-live-smoke", "ingestion_order": "source-ordinal"},
    )
    mapper = FactLevelMapper(index)

    state = await adapter.create_isolated_state(artifact)
    receipts = await adapter.ingest(state, artifact.sources)
    if len(receipts) != len(artifact.sources):
        raise AssertionError("backend did not return one ingestion receipt per source")
    baseline = await adapter.observable_state(state)

    retrieved = await adapter.retrieve(state, "Alice residence", SMOKE_TOP_K)
    if not retrieved:
        raise AssertionError("ranked retrieval returned no entries")
    if not any(entry.provenance_ids for entry in retrieved):
        raise AssertionError("retrieval did not recover stable benchmark provenance")
    mapped = tuple(mapper.map(entry) for entry in retrieved)
    if not any(result.status is MappingStatus.MAPPED for result in mapped):
        raise AssertionError("retrieval yielded no reliable fact-level or fallback mapping")

    alice_id = receipts[0].affected_local_ids[0]
    bob_id = receipts[1].affected_local_ids[0]
    carol_id = receipts[2].affected_local_ids[0]
    update = await adapter.update(
        state,
        alice_id,
        "Alice residence Seattle.",
        provenance_ids=(checkpoint.sources[0].provenance_id,),
    )
    if not update.success:
        raise AssertionError(f"native update failed: {update.reason}")
    unrelated = await adapter.unrelated_existing_region_change(
        state,
        bob_id,
        "Bob hobby golf.",
        provenance_ids=(checkpoint.sources[1].provenance_id,),
    )
    if not unrelated.success:
        raise AssertionError(f"unrelated existing-region change failed: {unrelated.reason}")

    deletion_context = None
    if graphiti:
        carol_fact = next(
            fact
            for fact in index.facts_for_source(checkpoint.sources[2].provenance_id)
            if fact.entity == "Carol" and fact.relation == "pet"
        )
        certificate = await adapter.certify_episode_deletion(
            state,
            carol_id,
            carol_fact.fact_id,
            index,
        )
        deletion_context = {"deletion_certificate": certificate}
    deletion = await adapter.delete(state, carol_id, context=deletion_context)
    if not deletion.success:
        raise AssertionError(f"faithful native deletion failed: {deletion.reason}")

    await adapter.teardown(state)
    replayed = await adapter.replay_state(artifact)
    replay_projection = await adapter.observable_state(replayed)
    try:
        if replay_projection != baseline:
            raise AssertionError(
                "replay did not reproduce the same available observable projection; "
                "initialization equivalence remains unestablished"
            )
    finally:
        await adapter.teardown(replayed)


class LiveBackendPrerequisiteTests(unittest.TestCase):
    def test_mem0_prerequisites_are_explicit(self) -> None:
        report = Mem0Adapter().capabilities()
        self.assertTrue(report.available)
        self.assertEqual(report.version_or_commit, "2.0.12")

    def test_amem_prerequisites_are_explicit(self) -> None:
        report = AMemAdapter().capabilities()
        self.assertEqual(
            report.available,
            importlib.util.find_spec("agentic_memory") is not None,
        )
        self.assertIn("ceffb860f0712bbae97b184d440df62bc910ca8d", report.version_or_commit)

    def test_graphiti_prerequisites_are_explicit(self) -> None:
        report = GraphitiAdapter().capabilities()
        self.assertEqual(
            report.available,
            importlib.util.find_spec("graphiti_core") is not None,
        )
        self.assertIn("0.30.2", report.version_or_commit)

    def test_memos_prerequisites_are_explicit(self) -> None:
        report = MemosAdapter().capabilities()
        self.assertEqual(report.available, importlib.util.find_spec("memos") is not None)
        self.assertIn("78a372a4fc853a24d2a78efa3b4bbbd27ab9f7ad", report.version_or_commit)


@unittest.skipUnless(
    os.environ.get("UFUZZ_RUN_LIVE_MEM0") == "1",
    "set UFUZZ_RUN_LIVE_MEM0=1 to run the real Mem0 smoke test",
)
class Mem0LiveSmokeTests(unittest.TestCase):
    def test_full_public_api_smoke(self) -> None:
        if not os.environ.get("OPENAI_API_KEY"):
            self.fail("OPENAI_API_KEY is required for the configured Mem0 ingestion path")
        asyncio.run(_exercise_backend(Mem0Adapter(), graphiti=False))


@unittest.skipUnless(
    os.environ.get("UFUZZ_RUN_LIVE_AMEM") == "1",
    "set UFUZZ_RUN_LIVE_AMEM=1 to run the real A-Mem smoke test",
)
class AMemLiveSmokeTests(unittest.TestCase):
    def test_full_public_api_smoke(self) -> None:
        if os.environ.get("UFUZZ_AMEM_DEDICATED_PROCESS") != "1":
            self.fail(
                "UFUZZ_AMEM_DEDICATED_PROCESS=1 is required to acknowledge "
                "A-Mem's process-scoped ephemeral Chroma reset"
            )
        if importlib.util.find_spec("agentic_memory") is None:
            self.fail("the pinned A-Mem package is not installed")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            self.fail("OPENAI_API_KEY is required for the configured A-Mem ingestion path")
        asyncio.run(_exercise_backend(AMemAdapter(api_key=api_key), graphiti=False))


@unittest.skipUnless(
    os.environ.get("UFUZZ_RUN_LIVE_GRAPHITI") == "1",
    "set UFUZZ_RUN_LIVE_GRAPHITI=1 to run the real Graphiti smoke test",
)
class GraphitiLiveSmokeTests(unittest.TestCase):
    def test_full_public_api_smoke(self) -> None:
        if os.environ.get("UFUZZ_GRAPHITI_ISOLATED_TEST_INSTANCE") != "1":
            self.fail(
                "UFUZZ_GRAPHITI_ISOLATED_TEST_INSTANCE=1 is required; "
                "the smoke test must not target a shared Neo4j service"
            )
        if importlib.util.find_spec("graphiti_core") is None:
            self.fail("graphiti-core==0.30.2 is not installed")
        required = ("OPENAI_API_KEY", "NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD")
        missing = [name for name in required if not os.environ.get(name)]
        if missing:
            self.fail("missing live Graphiti configuration: " + ", ".join(missing))
        asyncio.run(
            _exercise_backend(
                GraphitiAdapter(
                    uri=os.environ["NEO4J_URI"],
                    user=os.environ["NEO4J_USER"],
                    password=os.environ["NEO4J_PASSWORD"],
                    database=os.environ.get("NEO4J_DATABASE"),
                ),
                graphiti=True,
            )
        )


@unittest.skipUnless(
    os.environ.get("UFUZZ_RUN_LIVE_MEMOS") == "1",
    "set UFUZZ_RUN_LIVE_MEMOS=1 to run the pinned MemOS GeneralText smoke test",
)
class MemosLiveSmokeTests(unittest.TestCase):
    def test_full_public_api_smoke(self) -> None:
        if os.environ.get("UFUZZ_MEMOS_DEDICATED_PROCESS") != "1":
            self.fail("UFUZZ_MEMOS_DEDICATED_PROCESS=1 is required")
        if importlib.util.find_spec("memos") is None:
            self.fail("pinned MemoryOS v2.0.33 is not installed")
        async def exercise() -> None:
            adapter = MemosAdapter()
            await _exercise_backend(adapter, graphiti=False)
            checkpoint, _ = _synthetic_checkpoint()
            artifact = InitializationArtifact.create(
                checkpoint.checkpoint_id,
                checkpoint.sources,
                {"memos_profile_id": "live-smoke-isolation"},
            )
            a = await adapter.replay_state(artifact)
            adapter_b = MemosAdapter()
            b = await adapter_b.replay_state(artifact)
            try:
                before_b = await adapter_b.observable_state(b)
                target = a.backend_state.get_all()[0]
                # Public retrieval supplies exact provenance without using
                # private storage or enumeration order as replay identity.
                record = (await adapter.retrieve(a, target.memory, 1))[0]
                await adapter.update(
                    a, record.local_id, record.text + " Updated.",
                    provenance_ids=record.provenance_ids,
                )
                if await adapter_b.observable_state(b) != before_b:
                    raise AssertionError("MemOS state A mutation changed isolated state B")
                b_dir = b.metadata["state_dir"]
                await adapter.teardown(a)
                if not os.path.isdir(b_dir):
                    raise AssertionError("MemOS cleanup A removed isolated state B")
            finally:
                await adapter.teardown(a)
                await adapter_b.teardown(b)

        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
