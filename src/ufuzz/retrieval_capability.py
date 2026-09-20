"""Retrievable-entry capability and resolved-observation primitives.

This module bridges backend public-retrieval objects to Phase 2A initialized
entry lineage.  It deliberately contains no scheduler or evaluator policy.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, Protocol, TypeVar, runtime_checkable

from ufuzz.coverage import (
    CoverageEntryId,
    CoverageState,
    CoverageUpdate,
    FrozenCheckpointInventory,
    FrozenValue,
    InitialCoverageEntry,
    InitialEntryLineage,
    LineageBinding,
    LineageRelation,
    LineageResolutionError,
    LineageStatus,
    PhysicalEntryRef,
    freeze_value,
)
from ufuzz.retrieval_feedback import RetrievalSignature, retrieval_signature


@dataclass(frozen=True, slots=True)
class RetrievableEntryProjection:
    """Backend-defined immutable projection of one public-retrieval object.

    ``observable`` must exclude replay-local IDs.  Exact equality of the full
    projection is the only equivalence relation used here; this layer performs
    no semantic or embedding-based matching.
    """

    backend: str
    retrieval_object_class: str
    observable: FrozenValue

    def __post_init__(self) -> None:
        if not self.backend or not self.retrieval_object_class:
            raise ValueError("projection backend and object class must be non-empty")
        object.__setattr__(self, "observable", freeze_value(self.observable))


@dataclass(frozen=True, slots=True)
class RetrievableInventoryEntry:
    """One initialized object of the selected public-retrieval result class."""

    physical_entry: PhysicalEntryRef
    projection: RetrievableEntryProjection
    provenance_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "provenance_ids",
            tuple(dict.fromkeys(self.provenance_ids)),
        )


@dataclass(frozen=True, slots=True)
class RetrievableEntryInventory:
    """Established initialized inventory for one checkpoint backend state.

    ``retrieval_returnability_basis`` records why the enumerated object class
    is the same class returned by ``selected_public_retrieval_api``.  Merely
    enumerating internal storage does not satisfy this contract.
    """

    backend: str
    selected_public_retrieval_api: str
    retrieval_object_class: str
    retrieval_returnability_basis: str
    campaign_id: str
    root_checkpoint_id: str
    state_id: str
    entries: tuple[RetrievableInventoryEntry, ...]

    def __post_init__(self) -> None:
        required = (
            self.backend,
            self.selected_public_retrieval_api,
            self.retrieval_object_class,
            self.retrieval_returnability_basis,
            self.campaign_id,
            self.root_checkpoint_id,
            self.state_id,
        )
        if not all(required):
            raise ValueError("inventory identity and capability basis must be non-empty")
        object.__setattr__(self, "entries", tuple(self.entries))
        physical_entries: set[PhysicalEntryRef] = set()
        for entry in self.entries:
            physical = entry.physical_entry
            if physical.campaign_id != self.campaign_id:
                raise ValueError("inventory contains a physical entry from another campaign")
            if physical.root_checkpoint_id != self.root_checkpoint_id:
                raise ValueError("inventory contains a physical entry from another checkpoint")
            if physical.state_id != self.state_id:
                raise ValueError("inventory contains a physical entry from another state")
            if entry.projection.backend != self.backend:
                raise ValueError("entry projection belongs to another backend")
            if entry.projection.retrieval_object_class != self.retrieval_object_class:
                raise ValueError("entry projection uses another retrieval object class")
            if physical in physical_entries:
                raise ValueError(
                    "duplicate physical identity in retrievable-entry inventory: "
                    f"{physical.backend_entry_id}"
                )
            physical_entries.add(physical)


class CapabilityOperation(StrEnum):
    """Search-side operation whose faithful support may be blocked."""

    INVENTORY = "inventory"
    PHYSICAL_IDENTITY = "physical_identity"
    PROJECTION = "projection"
    RANKED_RETRIEVAL = "ranked_retrieval"
    LINEAGE = "lineage"
    INITIALIZATION_EQUIVALENCE = "initialization_equivalence"


@dataclass(frozen=True, slots=True)
class RetrievalCapabilityBlocker:
    """Explicit condition-level failure instead of a fabricated inventory."""

    backend: str
    root_checkpoint_id: str
    operation: CapabilityOperation
    reason: str

    def __post_init__(self) -> None:
        if not self.backend or not self.root_checkpoint_id or not self.reason:
            raise ValueError("capability blocker fields must be non-empty")


class RetrievalCapabilityError(RuntimeError):
    """Raised when a caller requires a result blocked by backend capability."""

    def __init__(self, blocker: RetrievalCapabilityBlocker) -> None:
        self.blocker = blocker
        super().__init__(f"{blocker.operation}: {blocker.reason}")


CapabilityValue = TypeVar("CapabilityValue")


@dataclass(frozen=True, slots=True)
class CapabilityResult(Generic[CapabilityValue]):
    """Exactly one successful value or explicit capability blocker."""

    value: CapabilityValue | None = None
    blocker: RetrievalCapabilityBlocker | None = None

    def __post_init__(self) -> None:
        if (self.value is None) == (self.blocker is None):
            raise ValueError("capability result requires exactly one value or blocker")

    @classmethod
    def success(cls, value: CapabilityValue) -> "CapabilityResult[CapabilityValue]":
        return cls(value=value)

    @classmethod
    def failure(
        cls, blocker: RetrievalCapabilityBlocker
    ) -> "CapabilityResult[CapabilityValue]":
        return cls(blocker=blocker)

    @property
    def supported(self) -> bool:
        return self.blocker is None

    def require(self) -> CapabilityValue:
        if self.blocker is not None:
            raise RetrievalCapabilityError(self.blocker)
        assert self.value is not None
        return self.value


@dataclass(frozen=True, slots=True)
class RankedPhysicalRetrieval:
    """Ranked physical objects returned by one selected public API call."""

    backend: str
    selected_public_retrieval_api: str
    retrieval_object_class: str
    campaign_id: str
    root_checkpoint_id: str
    state_id: str
    ranked_physical_entries: tuple[PhysicalEntryRef, ...]

    def __post_init__(self) -> None:
        required = (
            self.backend,
            self.selected_public_retrieval_api,
            self.retrieval_object_class,
            self.campaign_id,
            self.root_checkpoint_id,
            self.state_id,
        )
        if not all(required):
            raise ValueError("ranked retrieval scope fields must be non-empty")
        object.__setattr__(
            self,
            "ranked_physical_entries",
            tuple(self.ranked_physical_entries),
        )
        for physical in self.ranked_physical_entries:
            if physical.campaign_id != self.campaign_id:
                raise ValueError("ranked retrieval contains another campaign")
            if physical.root_checkpoint_id != self.root_checkpoint_id:
                raise ValueError("ranked retrieval contains another checkpoint")
            if physical.state_id != self.state_id:
                raise ValueError("ranked retrieval contains another backend state")


@runtime_checkable
class RetrievableEntryCapability(Protocol):
    """Optional backend capability for the selected public result class.

    Concrete adapters may implement this separately from ``BackendAdapter``.
    Inventory and retrieval methods return explicit blockers when physical
    identity, projection, enumeration, or ranked resolution is not faithful.
    """

    backend: str
    selected_public_retrieval_api: str
    retrieval_object_class: str

    async def inventory(
        self,
        *,
        campaign_id: str,
        root_checkpoint_id: str,
        state_id: str,
    ) -> CapabilityResult[RetrievableEntryInventory]: ...

    async def retrieve(
        self,
        *,
        campaign_id: str,
        root_checkpoint_id: str,
        state_id: str,
        query: str,
        top_k: int,
    ) -> CapabilityResult[RankedPhysicalRetrieval]: ...


@dataclass(frozen=True, slots=True)
class ProjectionMultiplicity:
    projection: RetrievableEntryProjection
    count: int

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("projection multiplicity must be positive")


@dataclass(frozen=True, slots=True)
class InitializationEquivalenceResult:
    """Exact multiset comparison of retrievable-entry projections."""

    equivalent: bool
    reference_count: int
    candidate_count: int
    missing_from_candidate: tuple[ProjectionMultiplicity, ...]
    excess_in_candidate: tuple[ProjectionMultiplicity, ...]
    condition_mismatches: tuple[str, ...] = ()

    def require_equivalent(self) -> "InitializationEquivalenceResult":
        """Require equivalence before admitting this checkpoint to campaign E_0."""

        if not self.equivalent:
            raise InitializationEquivalenceError(self)
        return self


class InitializationEquivalenceError(RuntimeError):
    """Raised when a non-equivalent initialization is admitted for RQ1."""

    def __init__(self, result: InitializationEquivalenceResult) -> None:
        self.result = result
        super().__init__(
            "retrievable-entry initialization projections are not equivalent"
        )


def _projection_counts(
    values: Mapping[RetrievableEntryProjection, int],
) -> tuple[ProjectionMultiplicity, ...]:
    return tuple(
        ProjectionMultiplicity(projection, count)
        for projection, count in sorted(values.items(), key=lambda item: repr(item[0]))
        if count > 0
    )


def compare_initialization_inventories(
    reference: RetrievableEntryInventory,
    candidate: RetrievableEntryInventory,
) -> InitializationEquivalenceResult:
    """Compare exact observable projections as an order-independent multiset.

    Campaign-local coverage IDs and physical backend IDs are absent from this
    comparison.  Duplicate equal projections retain their multiplicity.
    """

    mismatches: list[str] = []
    for name in (
        "backend",
        "selected_public_retrieval_api",
        "retrieval_object_class",
        "root_checkpoint_id",
    ):
        if getattr(reference, name) != getattr(candidate, name):
            mismatches.append(name)
    reference_counts = Counter(entry.projection for entry in reference.entries)
    candidate_counts = Counter(entry.projection for entry in candidate.entries)
    missing = reference_counts - candidate_counts
    excess = candidate_counts - reference_counts
    equivalent = not mismatches and not missing and not excess
    return InitializationEquivalenceResult(
        equivalent=equivalent,
        reference_count=len(reference.entries),
        candidate_count=len(candidate.entries),
        missing_from_candidate=_projection_counts(missing),
        excess_in_candidate=_projection_counts(excess),
        condition_mismatches=tuple(mismatches),
    )


@dataclass(frozen=True, slots=True)
class FrozenInventoryBuild:
    """Checkpoint-local frozen entries and initial physical bindings.

    ``checkpoint_coverage_state`` is an intermediate local state.  It is not
    the authoritative RQ1 condition-level eligibility decision; campaign E_0
    is assembled across all admitted checkpoints by
    :func:`assemble_campaign_coverage`.
    """

    inventory: FrozenCheckpointInventory
    lineage: InitialEntryLineage
    initial_bindings: tuple[LineageBinding, ...]
    checkpoint_coverage_state: CoverageState


def build_frozen_checkpoint_inventory(
    established: RetrievableEntryInventory,
) -> FrozenInventoryBuild:
    """Assign one opaque local coverage ID per initialized physical object.

    Entries are ordered by their backend-local physical identity only to make
    local opaque assignment deterministic.  Projection equality never
    deduplicates initialized objects, and IDs are not compared across methods.

    This local construction does not prove initialization equivalence and does
    not itself authorize admission to the campaign E_0.  The caller must first
    compare the method initialization against the shared condition and require
    an equivalent result before passing the build to campaign assembly.
    """

    ordered = tuple(
        sorted(
            established.entries,
            key=lambda entry: entry.physical_entry.backend_entry_id,
        )
    )
    entries = tuple(
        InitialCoverageEntry(
            coverage_id=CoverageEntryId(
                established.campaign_id,
                established.root_checkpoint_id,
                f"entry-{index:06d}",
            ),
            root_checkpoint_id=established.root_checkpoint_id,
            initial_backend_entry_id=item.physical_entry.backend_entry_id,
            initial_observable_projection=item.projection.observable,
            provenance_ids=item.provenance_ids,
        )
        for index, item in enumerate(ordered, start=1)
    )
    frozen = FrozenCheckpointInventory(
        established.campaign_id,
        established.root_checkpoint_id,
        entries,
    )
    lineage = InitialEntryLineage(established.campaign_id, (frozen,))
    bindings = tuple(
        lineage.certify_binding(
            inventory_entry.physical_entry,
            (coverage_entry.coverage_id,),
            LineageRelation.INITIAL,
        )
        for inventory_entry, coverage_entry in zip(ordered, entries, strict=True)
    )
    return FrozenInventoryBuild(
        inventory=frozen,
        lineage=lineage,
        initial_bindings=bindings,
        checkpoint_coverage_state=CoverageState.from_entries(entries),
    )


@dataclass(frozen=True, slots=True)
class FrozenCampaignCoverage:
    """Authoritative campaign E_0 assembled as a checkpoint-disjoint union."""

    campaign_id: str
    checkpoint_inventories: tuple[FrozenCheckpointInventory, ...]
    lineage: InitialEntryLineage
    coverage_state: CoverageState

    def __post_init__(self) -> None:
        if not self.campaign_id:
            raise ValueError("campaign_id must be non-empty")
        if self.lineage.campaign_id != self.campaign_id:
            raise ValueError("campaign lineage belongs to another campaign")
        expected_ids = frozenset(
            coverage_id
            for inventory in self.checkpoint_inventories
            for coverage_id in inventory.coverage_ids
        )
        if self.lineage.coverage_ids != expected_ids:
            raise ValueError("campaign lineage does not match assembled inventories")
        if self.coverage_state.initial_ids != expected_ids:
            raise ValueError("campaign coverage E_0 does not match assembled inventories")
        if self.coverage_state.reached_ids:
            raise ValueError("campaign coverage must begin with an empty reached set")

    @property
    def initial_ids(self) -> frozenset[CoverageEntryId]:
        return self.coverage_state.initial_ids

    @property
    def entries(self) -> tuple[InitialCoverageEntry, ...]:
        return tuple(
            entry
            for inventory in self.checkpoint_inventories
            for entry in inventory.entries
        )


def assemble_campaign_coverage(
    campaign_id: str,
    checkpoint_builds: Sequence[FrozenInventoryBuild],
) -> FrozenCampaignCoverage:
    """Assemble campaign E_0 after per-checkpoint equivalence is established.

    Each input build must already have passed the applicable
    ``InitializationEquivalenceResult.require_equivalent()`` check.  This
    function cannot infer that experimental prerequisite from a locally frozen
    representation.  Empty checkpoint inventories contribute no IDs; only an
    empty union across every admitted checkpoint makes campaign coverage
    RQ1-ineligible.
    """

    if not campaign_id:
        raise ValueError("campaign_id must be non-empty")
    builds = tuple(checkpoint_builds)
    inventories = tuple(build.inventory for build in builds)
    roots: set[str] = set()
    for build in builds:
        inventory = build.inventory
        if inventory.campaign_id != campaign_id:
            raise ValueError("campaign assembly contains another campaign")
        if inventory.root_checkpoint_id in roots:
            raise ValueError("campaign assembly contains a duplicate checkpoint")
        roots.add(inventory.root_checkpoint_id)
        if build.checkpoint_coverage_state.reached_ids:
            raise ValueError("checkpoint-local coverage must be empty at assembly")
        if (
            build.checkpoint_coverage_state.initial_ids
            != inventory.coverage_ids
        ):
            raise ValueError("checkpoint-local coverage does not match its inventory")

    lineage = InitialEntryLineage(campaign_id, inventories)
    for build in builds:
        for binding in build.initial_bindings:
            if binding.status is not LineageStatus.CERTIFIED:
                raise ValueError("initial campaign lineage must be certified")
            if binding.relation is not LineageRelation.INITIAL:
                raise ValueError("campaign assembly accepts only INITIAL bindings")
            lineage.certify_binding(
                binding.physical_entry,
                binding.coverage_ids,
                LineageRelation.INITIAL,
            )

    entries = tuple(
        entry for inventory in inventories for entry in inventory.entries
    )
    coverage_state = CoverageState.from_entries(entries)
    return FrozenCampaignCoverage(
        campaign_id=campaign_id,
        checkpoint_inventories=inventories,
        lineage=lineage,
        coverage_state=coverage_state,
    )


def _resolve_ranked_bindings(
    retrieval: RankedPhysicalRetrieval,
    lineage: InitialEntryLineage,
) -> tuple[LineageBinding, ...]:
    if retrieval.campaign_id != lineage.campaign_id:
        raise ValueError("ranked retrieval belongs to another lineage campaign")
    bindings: list[LineageBinding] = []
    for physical in retrieval.ranked_physical_entries:
        binding = lineage.resolution(physical)
        if binding.status is not LineageStatus.CERTIFIED:
            raise LineageResolutionError(binding)
        bindings.append(binding)
    return tuple(bindings)


def _validate_resolved_observation(
    retrieval: RankedPhysicalRetrieval,
    bindings: Sequence[LineageBinding],
    signature: RetrievalSignature,
) -> frozenset[CoverageEntryId]:
    if any(binding.status is not LineageStatus.CERTIFIED for binding in bindings):
        raise ValueError("resolved observation contains uncertified lineage")
    if tuple(binding.physical_entry for binding in bindings) != (
        retrieval.ranked_physical_entries
    ):
        raise ValueError("resolved bindings do not preserve physical retrieval rank")
    expected_signature = retrieval_signature(
        binding.coverage_ids for binding in bindings
    )
    if signature != expected_signature:
        raise ValueError("retrieval signature is inconsistent with resolved lineages")
    return frozenset(
        coverage_id
        for binding in bindings
        for coverage_id in binding.coverage_ids
    )


@dataclass(frozen=True, slots=True)
class InitializationRetrievalSignature:
    """Resolved initialization behavior with deliberately no coverage update."""

    retrieval: RankedPhysicalRetrieval
    resolved_lineages: tuple[LineageBinding, ...]
    signature: RetrievalSignature

    def __post_init__(self) -> None:
        _validate_resolved_observation(
            self.retrieval,
            self.resolved_lineages,
            self.signature,
        )


def resolve_initialization_signature(
    retrieval: RankedPhysicalRetrieval,
    lineage: InitialEntryLineage,
) -> InitializationRetrievalSignature:
    """Resolve H_i initialization behavior without changing coverage C_0."""

    bindings = _resolve_ranked_bindings(retrieval, lineage)
    signature = retrieval_signature(binding.coverage_ids for binding in bindings)
    return InitializationRetrievalSignature(retrieval, bindings, signature)


@dataclass(frozen=True, slots=True)
class ResolvedRetrievalObservation:
    """One valid fuzzing retrieval resolved once for behavior and coverage."""

    retrieval: RankedPhysicalRetrieval
    resolved_lineages: tuple[LineageBinding, ...]
    signature: RetrievalSignature
    coverage_update: CoverageUpdate

    def __post_init__(self) -> None:
        expected_ids = _validate_resolved_observation(
            self.retrieval,
            self.resolved_lineages,
            self.signature,
        )
        if self.coverage_update.retrieved_ids != expected_ids:
            raise ValueError(
                "coverage update retrieved IDs differ from resolved retrieval lineage"
            )
        update = self.coverage_update
        if update.raw_gain != len(update.newly_reached_ids):
            raise ValueError("coverage update raw gain does not match newly reached IDs")
        if not update.newly_reached_ids.issubset(update.retrieved_ids):
            raise ValueError("newly reached IDs must be a subset of retrieved IDs")
        if not update.retrieved_ids.issubset(update.state.initial_ids):
            raise ValueError("retrieved IDs must be contained in frozen E_0")
        if not update.retrieved_ids.issubset(update.state.reached_ids):
            raise ValueError("retrieved IDs must be reached in the next coverage state")
        if not update.newly_reached_ids.issubset(update.state.reached_ids):
            raise ValueError("newly reached IDs must be reached in the next coverage state")


def resolve_fuzzing_retrieval(
    retrieval: RankedPhysicalRetrieval,
    lineage: InitialEntryLineage,
    coverage_state: CoverageState,
) -> ResolvedRetrievalObservation:
    """Derive sigma and CoverageUpdate from the same ranked certified lineages."""

    coverage_campaigns = {
        coverage_id.campaign_id for coverage_id in coverage_state.initial_ids
    }
    if coverage_campaigns and coverage_campaigns != {retrieval.campaign_id}:
        raise ValueError("coverage state belongs to another campaign")
    bindings = _resolve_ranked_bindings(retrieval, lineage)
    resolved_ids = tuple(binding.coverage_ids for binding in bindings)
    signature = retrieval_signature(resolved_ids)
    coverage_update = coverage_state.observe(resolved_ids)
    return ResolvedRetrievalObservation(
        retrieval=retrieval,
        resolved_lineages=bindings,
        signature=signature,
        coverage_update=coverage_update,
    )
