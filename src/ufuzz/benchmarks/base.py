"""Common benchmark-loader contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from ufuzz.domain import BenchmarkCheckpoint, DatasetArtifact


class BenchmarkLoader(ABC):
    artifact: DatasetArtifact

    @abstractmethod
    def load(
        self, path: str | Path, *, verify_artifact: bool = True
    ) -> Iterator[BenchmarkCheckpoint]:
        """Yield checkpoints without discarding raw benchmark fields."""
