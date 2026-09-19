"""Loader for the official LoCoMo ``locomo10.json`` schema."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator

from ufuzz.benchmarks.base import BenchmarkLoader
from ufuzz.domain import (
    BenchmarkCheckpoint,
    BenchmarkQuery,
    DatasetArtifact,
    SourceUnit,
    iter_json_array,
)


LOCOMO_ARTIFACT = DatasetArtifact(
    repository="snap-research/locomo",
    revision="3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376",
    filename="data/locomo10.json",
    sha256="79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4",
    size_bytes=2_805_274,
    url=(
        "https://raw.githubusercontent.com/snap-research/locomo/"
        "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/data/locomo10.json"
    ),
)

_SESSION_RE = re.compile(r"^session_(\d+)$")


class LoCoMoLoader(BenchmarkLoader):
    artifact = LOCOMO_ARTIFACT

    def load(
        self, path: str | Path, *, verify_artifact: bool = True
    ) -> Iterator[BenchmarkCheckpoint]:
        if verify_artifact:
            self.artifact.verify(path)
        for record in iter_json_array(path):
            yield self._parse_record(record)

    def _parse_record(self, record: Any) -> BenchmarkCheckpoint:
        if not isinstance(record, dict):
            raise ValueError("LoCoMo records must be JSON objects")
        for required in ("sample_id", "conversation", "qa"):
            if required not in record:
                raise ValueError(f"LoCoMo record lacks required field {required!r}")

        checkpoint_id = str(record["sample_id"])
        conversation = record["conversation"]
        if not isinstance(conversation, dict):
            raise ValueError(f"LoCoMo {checkpoint_id}: conversation must be an object")

        sources: list[SourceUnit] = []
        ordinal = 0
        sessions: list[tuple[int, str, list[Any]]] = []
        for key, value in conversation.items():
            match = _SESSION_RE.match(key)
            if match and isinstance(value, list):
                sessions.append((int(match.group(1)), key, value))
        sessions.sort()

        for session_number, session_id, turns in sessions:
            timestamp_value = conversation.get(f"session_{session_number}_date_time")
            timestamp = str(timestamp_value) if timestamp_value is not None else None
            for turn_index, turn in enumerate(turns):
                if not isinstance(turn, dict):
                    raise ValueError(
                        f"LoCoMo {checkpoint_id}/{session_id}: turn must be an object"
                    )
                if "dia_id" not in turn or "text" not in turn:
                    raise ValueError(
                        f"LoCoMo {checkpoint_id}/{session_id}: turn lacks dia_id or text"
                    )
                ordinal += 1
                speaker_value = turn.get("speaker")
                sources.append(
                    SourceUnit(
                        benchmark="locomo",
                        checkpoint_id=checkpoint_id,
                        session_id=session_id,
                        source_id=str(turn["dia_id"]),
                        ordinal=ordinal,
                        text=str(turn["text"]),
                        timestamp=timestamp,
                        speaker=str(speaker_value) if speaker_value is not None else None,
                        role=None,
                        raw=turn,
                    )
                )

        qa = record["qa"]
        if not isinstance(qa, list):
            raise ValueError(f"LoCoMo {checkpoint_id}: qa must be a list")
        queries: list[BenchmarkQuery] = []
        for index, item in enumerate(qa):
            if not isinstance(item, dict) or "question" not in item:
                raise ValueError(f"LoCoMo {checkpoint_id}: invalid QA item {index}")
            category = item.get("category")
            queries.append(
                BenchmarkQuery(
                    benchmark="locomo",
                    checkpoint_id=checkpoint_id,
                    query_id=f"locomo:{checkpoint_id}:qa:{index:04d}",
                    text=str(item["question"]),
                    timestamp=None,
                    query_type=category if isinstance(category, (str, int)) else None,
                    raw=item,
                )
            )

        return BenchmarkCheckpoint(
            benchmark="locomo",
            checkpoint_id=checkpoint_id,
            sources=tuple(sources),
            queries=tuple(queries),
            raw=record,
            artifact=self.artifact,
            metadata={
                "speaker_a": conversation.get("speaker_a"),
                "speaker_b": conversation.get("speaker_b"),
            },
        )
