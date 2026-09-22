"""Streaming loader for the frozen cleaned LongMemEval-S artifact."""

from __future__ import annotations

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


LONGMEMEVAL_S_ARTIFACT = DatasetArtifact(
    repository="xiaowu0162/longmemeval-cleaned",
    revision="98d7416c24c778c2fee6e6f3006e7a073259d48f",
    filename="longmemeval_s_cleaned.json",
    sha256="d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",
    size_bytes=277_383_467,
    url=(
        "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/"
        "98d7416c24c778c2fee6e6f3006e7a073259d48f/"
        "longmemeval_s_cleaned.json"
    ),
)

_REQUIRED_FIELDS = {
    "question_id",
    "question_type",
    "question",
    "question_date",
    "answer",
    "answer_session_ids",
    "haystack_dates",
    "haystack_session_ids",
    "haystack_sessions",
}


class LongMemEvalSLoader(BenchmarkLoader):
    artifact = LONGMEMEVAL_S_ARTIFACT

    def load(
        self, path: str | Path, *, verify_artifact: bool = True
    ) -> Iterator[BenchmarkCheckpoint]:
        if verify_artifact:
            self.artifact.verify(path)
        for record in iter_json_array(path):
            yield self._parse_record(record)

    def _parse_record(self, record: Any) -> BenchmarkCheckpoint:
        if not isinstance(record, dict):
            raise ValueError("LongMemEval-S records must be JSON objects")
        missing = _REQUIRED_FIELDS - record.keys()
        if missing:
            raise ValueError(f"LongMemEval-S record lacks fields: {sorted(missing)}")

        checkpoint_id = str(record["question_id"])
        sessions = record["haystack_sessions"]
        session_ids = record["haystack_session_ids"]
        dates = record["haystack_dates"]
        if not all(isinstance(value, list) for value in (sessions, session_ids, dates)):
            raise ValueError(
                f"LongMemEval-S {checkpoint_id}: haystack fields must be lists"
            )
        if not (len(sessions) == len(session_ids) == len(dates)):
            raise ValueError(
                f"LongMemEval-S {checkpoint_id}: session/date/id lengths differ"
            )

        sources: list[SourceUnit] = []
        ordinal = 0
        for session_index, (session, session_id_value, date_value) in enumerate(
            zip(sessions, session_ids, dates, strict=True)
        ):
            if not isinstance(session, list):
                raise ValueError(
                    f"LongMemEval-S {checkpoint_id}: session {session_index} is not a list"
                )
            session_id = str(session_id_value)
            timestamp = str(date_value) if date_value is not None else None
            for turn_index, turn in enumerate(session):
                if not isinstance(turn, dict):
                    raise ValueError(
                        f"LongMemEval-S {checkpoint_id}/{session_id}: turn is not an object"
                    )
                if "role" not in turn or "content" not in turn:
                    raise ValueError(
                        f"LongMemEval-S {checkpoint_id}/{session_id}: "
                        "turn lacks role or content"
                    )
                ordinal += 1
                role = str(turn["role"])
                sources.append(
                    SourceUnit(
                        benchmark="longmemeval-s-cleaned",
                        checkpoint_id=checkpoint_id,
                        session_id=session_id,
                        source_id=f"turn-{turn_index:04d}",
                        ordinal=ordinal,
                        text=str(turn["content"]),
                        timestamp=timestamp,
                        speaker=None,
                        role=role,
                        raw=turn,
                        session_occurrence=session_index,
                        turn_index=turn_index,
                    )
                )

        question_type = record.get("question_type")
        query = BenchmarkQuery(
            benchmark="longmemeval-s-cleaned",
            checkpoint_id=checkpoint_id,
            query_id=f"longmemeval-s-cleaned:{checkpoint_id}:query",
            text=str(record["question"]),
            timestamp=(
                str(record["question_date"])
                if record.get("question_date") is not None
                else None
            ),
            query_type=(
                question_type if isinstance(question_type, (str, int)) else None
            ),
            raw=record,
        )

        return BenchmarkCheckpoint(
            benchmark="longmemeval-s-cleaned",
            checkpoint_id=checkpoint_id,
            sources=tuple(sources),
            queries=(query,),
            raw=record,
            artifact=self.artifact,
            metadata={
                "schema_fields": tuple(record.keys()),
            },
        )


def resolve_answer_session_scope(
    checkpoint: BenchmarkCheckpoint,
    answer_session_ids: tuple[str, ...] | list[str],
) -> tuple[SourceUnit, ...]:
    """Resolve every occurrence of each native LongMemEval session ID.

    Native ``answer_session_ids`` identify the benchmark session label, not a
    unique occurrence. Repeated labels therefore intentionally resolve to all
    matching occurrences until evaluator-only semantic preprocessing selects
    exact supporting turns or spans.
    """

    if checkpoint.benchmark != "longmemeval-s-cleaned":
        raise ValueError("answer-session scope requires a LongMemEval-S checkpoint")
    requested_set = {str(value) for value in answer_session_ids}
    resolved = tuple(
        source for source in checkpoint.sources if source.session_id in requested_set
    )
    missing = sorted(requested_set - {source.session_id for source in resolved})
    if missing:
        raise ValueError(f"answer_session_ids do not resolve: {missing}")
    return resolved
