"""U-Fuzz Phase-1 infrastructure."""

from .domain import BenchmarkCheckpoint, BenchmarkQuery, DatasetArtifact, SourceUnit
from .mapping import FactLevelMapper, MappingResult, MappingStatus
from .structural import CanonicalFact, QueryIntent, StructuralIndex

__all__ = [
    "BenchmarkCheckpoint",
    "BenchmarkQuery",
    "CanonicalFact",
    "DatasetArtifact",
    "FactLevelMapper",
    "MappingResult",
    "MappingStatus",
    "QueryIntent",
    "SourceUnit",
    "StructuralIndex",
]
