"""Initialized-entry lineage and pure campaign coverage state."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


EVALUATOR_ONLY_KEYS = frozenset(
    {
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
    }
)


type FrozenValue = (
    None
    | bool
    | int
    | float
    | str
    | bytes
    | tuple["FrozenValue", ...]
    | frozenset["FrozenValue"]
    | FrozenMapping
)


class FrozenMapping(Mapping[str, FrozenValue]):
    """Small immutable mapping used for observable entry projections."""

    __slots__ = ("_items", "_lookup")

    def __init__(self, values: Mapping[str, Any]) -> None:
        if any(not isinstance(key, str) for key in values):
            raise TypeError("observable projection keys must be strings")
        forbidden = sorted(
            key for key in values if key.casefold() in EVALUATOR_ONLY_KEYS
        )
        if forbidden:
            raise ValueError(
                "observable projection contains evaluator-only keys: "
                + ", ".join(forbidden)
            )
        items = tuple(
            sorted(
                ((key, freeze_value(value)) for key, value in values.items()),
                key=lambda item: item[0],
            )
        )
        self._items = items
        self._lookup = dict(items)

    def __getitem__(self, key: str) -> FrozenValue:
        return self._lookup[key]

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __hash__(self) -> int:
        return hash(self._items)

    def __repr__(self) -> str:
        return f"FrozenMapping({dict(self._items)!r})"


def freeze_value(value: Any) -> FrozenValue:
    """Recursively freeze a JSON-like observable projection."""

    if isinstance(value, FrozenMapping):
        return value
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, Mapping):
        return FrozenMapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(freeze_value(item) for item in value)
    raise TypeError(
        "observable projection must contain only immutable JSON-like values; "
        f"got {type(value).__name__}"
    )


@dataclass(frozen=True, order=True, slots=True)
class CoverageEntryId:
    """Opaque coverage identity scoped to one campaign and root checkpoint.

    Lexicographic field order is the canonical total order used by retrieval
    tokens. Backend-local identifiers are deliberately absent.
    """

    campaign_id: str
    root_checkpoint_id: str
    opaque_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("campaign_id", self.campaign_id),
            ("root_checkpoint_id", self.root_checkpoint_id),
            ("opaque_id", self.opaque_id),
        ):
            if not value:
                raise ValueError(f"{name} must be non-empty")


@dataclass(frozen=True, slots=True)
class InitialCoverageEntry:
    """One frozen retrieval-eligible entry in a campaign checkpoint state."""

    coverage_id: CoverageEntryId
    root_checkpoint_id: str
    initial_backend_entry_id: str | None
    initial_observable_projection: FrozenValue
    provenance_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.root_checkpoint_id != self.coverage_id.root_checkpoint_id:
            raise ValueError("entry root checkpoint must match its coverage ID")
        object.__setattr__(
            self,
            "initial_observable_projection",
            freeze_value(self.initial_observable_projection),
        )
        object.__setattr__(
            self,
            "provenance_ids",
            tuple(dict.fromkeys(self.provenance_ids)),
        )


@dataclass(frozen=True, slots=True)
class FrozenCheckpointInventory:
    """Frozen retrieval-eligible entries for one initialized checkpoint state."""

    campaign_id: str
    root_checkpoint_id: str
    entries: tuple[InitialCoverageEntry, ...]

    def __post_init__(self) -> None:
        if not self.campaign_id or not self.root_checkpoint_id:
            raise ValueError("campaign and root checkpoint IDs must be non-empty")
        ids: set[CoverageEntryId] = set()
        for entry in self.entries:
            if entry.coverage_id.campaign_id != self.campaign_id:
                raise ValueError("inventory contains an entry from another campaign")
            if entry.root_checkpoint_id != self.root_checkpoint_id:
                raise ValueError("inventory contains an entry from another checkpoint")
            if entry.coverage_id in ids:
                raise ValueError(f"duplicate coverage ID: {entry.coverage_id}")
            ids.add(entry.coverage_id)

    @property
    def coverage_ids(self) -> frozenset[CoverageEntryId]:
        return frozenset(entry.coverage_id for entry in self.entries)


@dataclass(frozen=True, order=True, slots=True)
class PhysicalEntryRef:
    """Concrete backend entry identity within one campaign state."""

    campaign_id: str
    root_checkpoint_id: str
    state_id: str
    backend_entry_id: str

    def __post_init__(self) -> None:
        if not all(
            (
                self.campaign_id,
                self.root_checkpoint_id,
                self.state_id,
                self.backend_entry_id,
            )
        ):
            raise ValueError("physical entry reference fields must be non-empty")


class LineageRelation(StrEnum):
    INITIAL = "initial"
    SAME_ID_UPDATE = "same_id_update"
    REPLACEMENT_UPDATE = "replacement_update"
    CERTIFIED_MERGE = "certified_merge"
    UNRELATED_CHANGE = "unrelated_change"
    UNKNOWN = "unknown"


class LineageStatus(StrEnum):
    CERTIFIED = "certified"
    INAPPLICABLE = "inapplicable"
    CAPABILITY_FAILURE = "capability_failure"


@dataclass(frozen=True, slots=True)
class LineageBinding:
    """Certified lineage or an explicit inability to certify it."""

    physical_entry: PhysicalEntryRef
    coverage_ids: frozenset[CoverageEntryId]
    relation: LineageRelation
    status: LineageStatus
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status is LineageStatus.CERTIFIED:
            if not self.coverage_ids:
                raise ValueError("certified lineage must resolve to a non-empty set")
            if self.reason is not None:
                raise ValueError("certified lineage cannot have a failure reason")
        else:
            if self.coverage_ids:
                raise ValueError("uncertifiable lineage cannot invent coverage IDs")
            if not self.reason:
                raise ValueError("uncertifiable lineage requires a reason")


class LineageResolutionError(RuntimeError):
    """Raised when a physical entry has no certified initialized-entry lineage."""

    def __init__(self, binding: LineageBinding) -> None:
        self.binding = binding
        super().__init__(binding.reason or "lineage is not certified")


class InitialEntryLineage:
    """Campaign-local registry from physical entries to frozen coverage IDs."""

    def __init__(
        self,
        campaign_id: str,
        inventories: Iterable[FrozenCheckpointInventory],
    ) -> None:
        if not campaign_id:
            raise ValueError("campaign_id must be non-empty")
        self.campaign_id = campaign_id
        self._entries: dict[CoverageEntryId, InitialCoverageEntry] = {}
        for inventory in inventories:
            if inventory.campaign_id != campaign_id:
                raise ValueError("cannot combine inventories from different campaigns")
            for entry in inventory.entries:
                if entry.coverage_id in self._entries:
                    raise ValueError(f"duplicate coverage ID: {entry.coverage_id}")
                self._entries[entry.coverage_id] = entry
        self._bindings: dict[PhysicalEntryRef, LineageBinding] = {}
        self._deleted: set[PhysicalEntryRef] = set()

    @property
    def entries(self) -> tuple[InitialCoverageEntry, ...]:
        return tuple(self._entries.values())

    @property
    def coverage_ids(self) -> frozenset[CoverageEntryId]:
        return frozenset(self._entries)

    def certify_binding(
        self,
        physical_entry: PhysicalEntryRef,
        coverage_ids: Collection[CoverageEntryId],
        relation: LineageRelation,
    ) -> LineageBinding:
        """Bind one concrete backend entry to certified initialized entries."""

        ids = frozenset(coverage_ids)
        self._validate_scope(physical_entry, ids)
        binding = LineageBinding(
            physical_entry=physical_entry,
            coverage_ids=ids,
            relation=relation,
            status=LineageStatus.CERTIFIED,
        )
        current = self._bindings.get(physical_entry)
        if current is not None and current != binding:
            raise ValueError("physical entry already has a different lineage result")
        self._bindings[physical_entry] = binding
        return binding

    def certify_same_id_update(self, physical_entry: PhysicalEntryRef) -> LineageBinding:
        """Keep an updated in-place physical record on its existing lineage."""

        ids = self.resolve(physical_entry)
        binding = LineageBinding(
            physical_entry=physical_entry,
            coverage_ids=ids,
            relation=LineageRelation.SAME_ID_UPDATE,
            status=LineageStatus.CERTIFIED,
        )
        self._bindings[physical_entry] = binding
        return binding

    def certify_replacement(
        self,
        replaced_entry: PhysicalEntryRef,
        replacement_entry: PhysicalEntryRef,
        *,
        relation: LineageRelation = LineageRelation.REPLACEMENT_UPDATE,
    ) -> LineageBinding:
        """Bind a replacement physical record to the replaced entry's lineage."""

        return self.certify_binding(
            replacement_entry,
            self.resolve(replaced_entry),
            relation,
        )

    def certify_merge(
        self,
        source_entries: Collection[PhysicalEntryRef],
        merged_entry: PhysicalEntryRef,
    ) -> LineageBinding:
        """Union certified source lineages into one physical merged record."""

        if not source_entries:
            raise ValueError("certified merge requires at least one source entry")
        ids: set[CoverageEntryId] = set()
        for source in source_entries:
            ids.update(self.resolve(source))
        return self.certify_binding(
            merged_entry,
            ids,
            LineageRelation.CERTIFIED_MERGE,
        )

    def mark_uncertifiable(
        self,
        physical_entry: PhysicalEntryRef,
        *,
        status: LineageStatus,
        reason: str,
        relation: LineageRelation = LineageRelation.UNKNOWN,
    ) -> LineageBinding:
        """Record inapplicability/capability failure without inventing identity."""

        if status is LineageStatus.CERTIFIED:
            raise ValueError("use certify_binding for certified lineage")
        self._validate_physical_scope(physical_entry)
        if physical_entry in self._bindings:
            raise ValueError("physical entry already has a lineage result")
        binding = LineageBinding(
            physical_entry=physical_entry,
            coverage_ids=frozenset(),
            relation=relation,
            status=status,
            reason=reason,
        )
        self._bindings[physical_entry] = binding
        return binding

    def resolution(self, physical_entry: PhysicalEntryRef) -> LineageBinding:
        """Return the explicit lineage result for a physical entry."""

        self._validate_physical_scope(physical_entry)
        return self._bindings.get(
            physical_entry,
            LineageBinding(
                physical_entry=physical_entry,
                coverage_ids=frozenset(),
                relation=LineageRelation.UNKNOWN,
                status=LineageStatus.CAPABILITY_FAILURE,
                reason="no certified initialized-entry lineage",
            ),
        )

    def resolve(self, physical_entry: PhysicalEntryRef) -> frozenset[CoverageEntryId]:
        """Resolve a physical record or raise its explicit lineage failure."""

        binding = self.resolution(physical_entry)
        if binding.status is not LineageStatus.CERTIFIED:
            raise LineageResolutionError(binding)
        return binding.coverage_ids

    def mark_deleted(self, physical_entry: PhysicalEntryRef) -> None:
        """Record deletion while retaining lineage and the frozen denominator."""

        self.resolve(physical_entry)
        self._deleted.add(physical_entry)

    def is_deleted(self, physical_entry: PhysicalEntryRef) -> bool:
        return physical_entry in self._deleted

    def _validate_scope(
        self,
        physical_entry: PhysicalEntryRef,
        coverage_ids: frozenset[CoverageEntryId],
    ) -> None:
        self._validate_physical_scope(physical_entry)
        if not coverage_ids:
            raise ValueError("lineage binding must be non-empty")
        unknown = coverage_ids.difference(self._entries)
        if unknown:
            raise ValueError(f"coverage IDs are outside frozen E_0: {sorted(unknown)!r}")
        if any(
            item.root_checkpoint_id != physical_entry.root_checkpoint_id
            for item in coverage_ids
        ):
            raise ValueError("lineage cannot cross root checkpoints")

    def _validate_physical_scope(self, physical_entry: PhysicalEntryRef) -> None:
        if physical_entry.campaign_id != self.campaign_id:
            raise ValueError("physical entry belongs to another campaign")


class CoverageEligibility(StrEnum):
    ELIGIBLE = "eligible"
    RQ1_INELIGIBLE_EMPTY_INVENTORY = "rq1_ineligible_empty_inventory"


class CoverageIneligibleError(RuntimeError):
    """Raised when coverage execution is attempted with an empty E_0."""


@dataclass(frozen=True, slots=True)
class CoverageState:
    """Immutable campaign coverage state with frozen E_0 and reached C_t."""

    initial_ids: frozenset[CoverageEntryId]
    reached_ids: frozenset[CoverageEntryId] = frozenset()

    def __post_init__(self) -> None:
        if not self.reached_ids.issubset(self.initial_ids):
            raise ValueError("reached coverage IDs must be contained in frozen E_0")
        campaigns = {item.campaign_id for item in self.initial_ids}
        if len(campaigns) > 1:
            raise ValueError("coverage state cannot mix campaign-local IDs")

    @classmethod
    def from_entries(cls, entries: Iterable[InitialCoverageEntry]) -> "CoverageState":
        values = tuple(entries)
        ids = frozenset(entry.coverage_id for entry in values)
        if len(ids) != len(values):
            raise ValueError("coverage entries contain duplicate IDs")
        return cls(ids)

    @property
    def eligibility(self) -> CoverageEligibility:
        if self.initial_ids:
            return CoverageEligibility.ELIGIBLE
        return CoverageEligibility.RQ1_INELIGIBLE_EMPTY_INVENTORY

    @property
    def reached_count(self) -> int:
        return len(self.reached_ids)

    @property
    def denominator_count(self) -> int:
        return len(self.initial_ids)

    @property
    def cov_fraction(self) -> float | None:
        if not self.initial_ids:
            return None
        return len(self.reached_ids) / len(self.initial_ids)

    def observe(
        self,
        resolved_lineages: Iterable[Collection[CoverageEntryId]],
    ) -> "CoverageUpdate":
        """Apply one valid fuzzing retrieval after all lineages are certified."""

        if self.eligibility is not CoverageEligibility.ELIGIBLE:
            raise CoverageIneligibleError(
                "campaign has empty E_0 and is RQ1-ineligible"
            )
        retrieved: set[CoverageEntryId] = set()
        for lineage in resolved_lineages:
            ids = frozenset(lineage)
            if not ids:
                raise ValueError("resolved retrieval lineage must be non-empty")
            retrieved.update(ids)
        unknown = retrieved.difference(self.initial_ids)
        if unknown:
            raise ValueError(f"retrieval resolved outside frozen E_0: {sorted(unknown)!r}")
        retrieved_ids = frozenset(retrieved)
        newly_reached = retrieved_ids.difference(self.reached_ids)
        next_state = CoverageState(
            initial_ids=self.initial_ids,
            reached_ids=self.reached_ids.union(retrieved_ids),
        )
        return CoverageUpdate(
            state=next_state,
            retrieved_ids=retrieved_ids,
            newly_reached_ids=frozenset(newly_reached),
            raw_gain=len(newly_reached),
        )


@dataclass(frozen=True, slots=True)
class CoverageUpdate:
    """Result of applying one observed valid fuzzing retrieval."""

    state: CoverageState
    retrieved_ids: frozenset[CoverageEntryId]
    newly_reached_ids: frozenset[CoverageEntryId]
    raw_gain: int
