from __future__ import annotations

import asyncio
from dataclasses import fields, is_dataclass
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping
import unittest

from ufuzz.backends import AMemAdapter, GraphitiAdapter, InitializationArtifact, Mem0Adapter
from ufuzz.backends.base import source_metadata
from ufuzz.benchmarks import LongMemEvalSLoader
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

    def add_note(self, content, time=None, **kwargs):
        local_id = f"a{len(self.memories) + 1}"
        self.memories[local_id] = SimpleNamespace(
            id=local_id,
            content=content,
            timestamp=time,
            tags=kwargs.get("tags", []),
        )
        return local_id

    def delete(self, local_id):
        return self.memories.pop(local_id, None) is not None


class _GraphitiSink:
    def __init__(self) -> None:
        self.episodes: dict[str, SimpleNamespace] = {}
        self.add_calls: list[dict[str, Any]] = []

    async def add_episode(self, **kwargs):
        self.add_calls.append(kwargs)
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
        return SimpleNamespace(edges=[edge])

    async def get_nodes_and_edges_by_episode(self, episode_ids):
        return SimpleNamespace(
            edges=[self.episodes[value] for value in episode_ids],
            nodes=[],
        )

    async def remove_episode(self, episode_uuid):
        self.episodes.pop(episode_uuid, None)


class InformationFlowRegressionTests(unittest.TestCase):
    def assertSearchSafe(self, value: Any) -> None:
        self.assertEqual(list(_structured_key_paths(value)), [])

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

        async def exercise_adapters() -> None:
            mem0_sink = _Mem0Sink()
            mem0 = Mem0Adapter(memory_factory=lambda _: mem0_sink, infer=False)
            mem0_state = await mem0.create_isolated_state(artifact)
            await mem0.ingest(mem0_state, artifact.sources[:1])
            self.assertSearchSafe(mem0_state.metadata)
            self.assertSearchSafe(mem0._provenance)
            self.assertSearchSafe(mem0_sink.entries)
            await mem0.teardown(mem0_state)

            amem_sink = _AMemSink()
            amem = AMemAdapter(memory_factory=lambda: amem_sink)
            amem_state = await amem.create_isolated_state(artifact)
            await amem.ingest(amem_state, artifact.sources[:1])
            self.assertSearchSafe(amem_state.metadata)
            self.assertSearchSafe(amem._provenance)
            self.assertSearchSafe(amem_sink.memories)
            await amem.teardown(amem_state)

            graphiti_sink = _GraphitiSink()
            graphiti = GraphitiAdapter(graphiti=graphiti_sink)
            graphiti_state = await graphiti.create_isolated_state(artifact)
            await graphiti.ingest(graphiti_state, artifact.sources[:1])
            self.assertSearchSafe(graphiti_state.metadata)
            self.assertSearchSafe(graphiti._episode_provenance)
            descriptions = [
                json.loads(call["source_description"])
                for call in graphiti_sink.add_calls
            ]
            self.assertSearchSafe(descriptions)
            await graphiti.teardown(graphiti_state)

        asyncio.run(exercise_adapters())


if __name__ == "__main__":
    unittest.main()
