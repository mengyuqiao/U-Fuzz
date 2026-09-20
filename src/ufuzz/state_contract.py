"""Immutable, backend-independent state, replay, and transition contracts.

This module validates certificates supplied by later backend integrations.  It
does not materialize states, execute mutations, or infer replay bindings from
observable projections.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Generic, TypeVar

from ufuzz.coverage import (
    CoverageEntryId,
    FrozenCheckpointInventory,
    FrozenMapping,
    FrozenValue,
    InitialEntryLineage,
    LineageRelation,
    PhysicalEntryRef,
    freeze_value,
)
from ufuzz.retrieval_feedback import (
    CoverageToken,
    MutationRelation,
    RetrievalSignature,
    canonical_tuple,
)


_QUERY_RELATIONS = frozenset(
    {
        MutationRelation.MEANING_PRESERVING_QUERY,
        MutationRelation.TARGET_CHANGING_QUERY,
        MutationRelation.UNSUPPORTED_QUERY,
    }
)
_MEMORY_RELATIONS = frozenset(
    {
        MutationRelation.UPDATE,
        MutationRelation.DELETION,
        MutationRelation.UNRELATED_CHANGE,
    }
)


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _freeze_mapping(value: Mapping[str, Any] | FrozenMapping) -> FrozenMapping:
    return value if isinstance(value, FrozenMapping) else FrozenMapping(value)


def _canonical_optional_lineage(
    values: Collection[CoverageEntryId],
) -> CoverageToken:
    ids = tuple(values)
    return canonical_tuple(ids) if ids else ()


def _validate_id_scope(
    coverage_id: CoverageEntryId,
    campaign_id: str,
    root_checkpoint_id: str,
) -> None:
    if coverage_id.campaign_id != campaign_id:
        raise ValueError("coverage ID belongs to another campaign")
    if coverage_id.root_checkpoint_id != root_checkpoint_id:
        raise ValueError("coverage ID belongs to another root checkpoint")


def _validate_physical_scope(
    physical_entry: PhysicalEntryRef,
    campaign_id: str,
    root_checkpoint_id: str,
) -> None:
    if physical_entry.campaign_id != campaign_id:
        raise ValueError("physical entry belongs to another campaign")
    if physical_entry.root_checkpoint_id != root_checkpoint_id:
        raise ValueError("physical entry belongs to another root checkpoint")


def _validate_signature_scope(
    signature: RetrievalSignature,
    campaign_id: str,
    root_checkpoint_id: str,
) -> None:
    if len(set(signature)) != len(signature):
        raise ValueError("retrieval signature cannot repeat canonical tokens")
    for token in signature:
        if token != canonical_tuple(token):
            raise ValueError("retrieval signature contains a non-canonical token")
        for coverage_id in token:
            _validate_id_scope(coverage_id, campaign_id, root_checkpoint_id)


class PhysicalTransitionOutcome(StrEnum):
    """An actual physical outcome; inability to certify is deliberately absent."""

    SAME_ID = "same_id"
    DELETED = "deleted"
    REPLACED = "replaced"
    MERGED = "merged"
    SPLIT = "split"
    UNCHANGED = "unchanged"


class CertificationStatus(StrEnum):
    """Whether one physical outcome was completely established."""

    CERTIFIED = "certified"
    UNRESOLVED = "unresolved"


class AffectedTransitionKind(StrEnum):
    """Classification of one group in the complete physically affected set."""

    UNCHANGED_LINEAGE = "unchanged_lineage"
    SAME_ID_CHANGED = "same_id_changed"
    DELETED = "deleted"
    REPLACED = "replaced"
    MERGE = "merge"
    SPLIT = "split"
    NEWLY_OBSERVED_UNCERTIFIED = "newly_observed_uncertified"
    OTHERWISE_UNRESOLVED = "otherwise_unresolved"


class RebindingKind(StrEnum):
    ROOT = "root"
    DESCENDANT = "descendant"


class MaterializationFailureKind(StrEnum):
    """Frozen failure distinctions with different methodological treatment."""

    ROOT_INITIALIZATION_INEQUIVALENT = "root_initialization_inequivalent"
    E0_REBINDING_AMBIGUOUS = "e0_rebinding_ambiguous"
    TRANSITION_REPLAY_DIVERGED = "transition_replay_diverged"
    DESCENDANT_STATE_INEQUIVALENT = "descendant_state_inequivalent"
    BACKEND_CAPABILITY_FAILURE = "backend_capability_failure"
    MUTATION_INAPPLICABLE = "mutation_inapplicable"
    TRANSIENT_BACKEND_FAILURE = "transient_backend_failure"
    POST_RETRIEVAL_LINEAGE_VIOLATION = "post_retrieval_lineage_violation"
    TRANSITION_SEMANTIC_DRIFT = "transition_semantic_drift"


def validate_transition_outcome_contract(
    relation: MutationRelation,
    possible: Collection[PhysicalTransitionOutcome],
    acceptable: Collection[PhysicalTransitionOutcome],
) -> None:
    """Validate method-independent possible/acceptable outcome semantics."""

    if not isinstance(relation, MutationRelation):
        raise TypeError("relation must use the canonical MutationRelation")
    possible_set = frozenset(possible)
    acceptable_set = frozenset(acceptable)
    if any(not isinstance(item, PhysicalTransitionOutcome) for item in possible_set):
        raise TypeError("possible outcomes must be PhysicalTransitionOutcome values")
    if any(not isinstance(item, PhysicalTransitionOutcome) for item in acceptable_set):
        raise TypeError("acceptable outcomes must be PhysicalTransitionOutcome values")
    if not acceptable_set.issubset(possible_set):
        raise ValueError("acceptable transition outcomes must be a subset of possible")
    if relation in _QUERY_RELATIONS:
        if possible_set or acceptable_set:
            raise ValueError("query-only opportunities have no physical transition")
        return
    if relation not in _MEMORY_RELATIONS:
        raise ValueError("unknown mutation relation")
    if not possible_set or not acceptable_set:
        raise ValueError("memory opportunities require possible and acceptable outcomes")
    if PhysicalTransitionOutcome.UNCHANGED in acceptable_set:
        raise ValueError(
            "UNCHANGED cannot satisfy Update, Deletion, or Unrelated Change"
        )


def validate_certification_status(
    status: CertificationStatus,
    observed_outcome: PhysicalTransitionOutcome | None,
) -> None:
    """Validate that certification status and physical outcome do not conflict."""

    if not isinstance(status, CertificationStatus):
        raise TypeError("status must be a CertificationStatus")
    if observed_outcome is not None and not isinstance(
        observed_outcome, PhysicalTransitionOutcome
    ):
        raise TypeError("observed outcome must be a PhysicalTransitionOutcome")
    if status is CertificationStatus.CERTIFIED and observed_outcome is None:
        raise ValueError("CERTIFIED requires exactly one observed outcome")
    if status is CertificationStatus.UNRESOLVED and observed_outcome is not None:
        raise ValueError("UNRESOLVED cannot claim an observed outcome")


@dataclass(frozen=True, slots=True)
class MutationOpportunity:
    """One method-independent legal mutation opportunity before selection."""

    opportunity_id: str
    campaign_id: str
    root_checkpoint_id: str
    parent_seed_id: str
    relation: MutationRelation
    canonical_target: FrozenValue
    target_lineage: CoverageToken
    applicability_evidence: FrozenMapping
    generation_constraints: FrozenMapping
    possible_transition_outcomes: frozenset[PhysicalTransitionOutcome]
    acceptable_transition_outcomes: frozenset[PhysicalTransitionOutcome]

    def __post_init__(self) -> None:
        for name in (
            "opportunity_id",
            "campaign_id",
            "root_checkpoint_id",
            "parent_seed_id",
        ):
            _require_text(name, getattr(self, name))
        if not isinstance(self.relation, MutationRelation):
            raise TypeError("relation must use the canonical MutationRelation")
        object.__setattr__(self, "canonical_target", freeze_value(self.canonical_target))
        object.__setattr__(
            self,
            "target_lineage",
            _canonical_optional_lineage(self.target_lineage),
        )
        object.__setattr__(
            self,
            "applicability_evidence",
            _freeze_mapping(self.applicability_evidence),
        )
        object.__setattr__(
            self,
            "generation_constraints",
            _freeze_mapping(self.generation_constraints),
        )
        possible = frozenset(self.possible_transition_outcomes)
        acceptable = frozenset(self.acceptable_transition_outcomes)
        object.__setattr__(self, "possible_transition_outcomes", possible)
        object.__setattr__(self, "acceptable_transition_outcomes", acceptable)
        validate_transition_outcome_contract(self.relation, possible, acceptable)
        self._validate_transition_outcomes()

    @classmethod
    def create(
        cls,
        *,
        opportunity_id: str,
        campaign_id: str,
        root_checkpoint_id: str,
        parent_seed_id: str,
        relation: MutationRelation,
        canonical_target: Any,
        target_lineage: Collection[CoverageEntryId] = (),
        applicability_evidence: Mapping[str, Any] | FrozenMapping,
        generation_constraints: Mapping[str, Any] | FrozenMapping,
        possible_transition_outcomes: Collection[PhysicalTransitionOutcome] = (),
        acceptable_transition_outcomes: Collection[PhysicalTransitionOutcome] = (),
    ) -> MutationOpportunity:
        return cls(
            opportunity_id,
            campaign_id,
            root_checkpoint_id,
            parent_seed_id,
            relation,
            freeze_value(canonical_target),
            _canonical_optional_lineage(target_lineage),
            _freeze_mapping(applicability_evidence),
            _freeze_mapping(generation_constraints),
            frozenset(possible_transition_outcomes),
            frozenset(acceptable_transition_outcomes),
        )

    def _validate_transition_outcomes(self) -> None:
        if self.relation in _QUERY_RELATIONS:
            if self.target_lineage:
                raise ValueError("query-only opportunities cannot target physical lineage")
            return
        if not self.target_lineage:
            raise ValueError("memory opportunities require an existing target lineage")
        for coverage_id in self.target_lineage:
            _validate_id_scope(coverage_id, self.campaign_id, self.root_checkpoint_id)


@dataclass(frozen=True, slots=True)
class SemanticMutationCertificate:
    """Pre-execution proof of a mutation's semantic obligation and applicability."""

    certificate_id: str
    opportunity_id: str
    campaign_id: str
    root_checkpoint_id: str
    relation: MutationRelation
    valid: bool
    obligation: FrozenMapping
    proof_digest: str

    def __post_init__(self) -> None:
        for name in (
            "certificate_id",
            "opportunity_id",
            "campaign_id",
            "root_checkpoint_id",
            "proof_digest",
        ):
            _require_text(name, getattr(self, name))
        if not isinstance(self.relation, MutationRelation):
            raise TypeError("relation must use the canonical MutationRelation")
        if not isinstance(self.valid, bool):
            raise TypeError("valid must be a bool")
        object.__setattr__(self, "obligation", _freeze_mapping(self.obligation))


@dataclass(frozen=True, slots=True)
class RealizedMutationArtifact:
    """Exact accepted mutation payload replayed without regeneration."""

    artifact_id: str
    opportunity: MutationOpportunity
    semantic_certificate: SemanticMutationCertificate
    parent_query_id: str
    exact_mutant_query: str | None
    subtype: str | None
    operation_payload: FrozenMapping

    def __post_init__(self) -> None:
        _require_text("artifact_id", self.artifact_id)
        _require_text("parent_query_id", self.parent_query_id)
        if self.subtype is not None:
            _require_text("subtype", self.subtype)
        object.__setattr__(
            self,
            "operation_payload",
            _freeze_mapping(self.operation_payload),
        )
        if not self.operation_payload:
            raise ValueError("realized mutation requires a non-empty exact payload")
        certificate = self.semantic_certificate
        opportunity = self.opportunity
        if not certificate.valid:
            raise ValueError("a realized artifact requires a valid semantic certificate")
        if certificate.opportunity_id != opportunity.opportunity_id:
            raise ValueError("semantic certificate belongs to another opportunity")
        if (
            certificate.campaign_id,
            certificate.root_checkpoint_id,
            certificate.relation,
        ) != (
            opportunity.campaign_id,
            opportunity.root_checkpoint_id,
            opportunity.relation,
        ):
            raise ValueError("semantic certificate scope/relation does not match")
        if opportunity.relation in _QUERY_RELATIONS:
            _require_text("exact_mutant_query", self.exact_mutant_query or "")
        elif self.exact_mutant_query is not None:
            raise ValueError("memory mutation artifact cannot replace the query")


@dataclass(frozen=True, slots=True)
class PhysicalLineageEndpoint:
    """One physical object and its already-frozen inherited E_0 lineage."""

    physical_entry: PhysicalEntryRef
    lineage: CoverageToken

    def __post_init__(self) -> None:
        object.__setattr__(self, "lineage", canonical_tuple(self.lineage))


@dataclass(frozen=True, slots=True)
class AffectedPhysicalTransition:
    """One fully described group within the complete affected physical set."""

    transition_id: str
    kind: AffectedTransitionKind
    predecessors: tuple[PhysicalLineageEndpoint, ...]
    successors: tuple[PhysicalLineageEndpoint, ...]
    unresolved_predecessors: tuple[PhysicalEntryRef, ...] = ()
    unresolved_successors: tuple[PhysicalEntryRef, ...] = ()
    backend_proof_digest: str = ""

    def __post_init__(self) -> None:
        _require_text("transition_id", self.transition_id)
        _require_text("backend_proof_digest", self.backend_proof_digest)
        object.__setattr__(self, "predecessors", tuple(self.predecessors))
        object.__setattr__(self, "successors", tuple(self.successors))
        object.__setattr__(
            self, "unresolved_predecessors", tuple(self.unresolved_predecessors)
        )
        object.__setattr__(
            self, "unresolved_successors", tuple(self.unresolved_successors)
        )
        if len({item.physical_entry for item in self.predecessors}) != len(
            self.predecessors
        ):
            raise ValueError("transition repeats a predecessor physical object")
        if len({item.physical_entry for item in self.successors}) != len(
            self.successors
        ):
            raise ValueError("transition repeats a successor physical object")
        if len(set(self.unresolved_predecessors)) != len(
            self.unresolved_predecessors
        ):
            raise ValueError("transition repeats an unresolved predecessor")
        if len(set(self.unresolved_successors)) != len(self.unresolved_successors):
            raise ValueError("transition repeats an unresolved successor")
        if set(self.unresolved_predecessors).intersection(
            item.physical_entry for item in self.predecessors
        ):
            raise ValueError("predecessor cannot be both resolved and unresolved")
        if set(self.unresolved_successors).intersection(
            item.physical_entry for item in self.successors
        ):
            raise ValueError("successor cannot be both resolved and unresolved")
        self._validate_shape()

    @property
    def predecessor_refs(self) -> frozenset[PhysicalEntryRef]:
        return frozenset(
            [item.physical_entry for item in self.predecessors]
            + list(self.unresolved_predecessors)
        )

    @property
    def successor_refs(self) -> frozenset[PhysicalEntryRef]:
        return frozenset(
            [item.physical_entry for item in self.successors]
            + list(self.unresolved_successors)
        )

    @property
    def inherited_ids(self) -> frozenset[CoverageEntryId]:
        return frozenset(
            coverage_id
            for endpoint in self.predecessors + self.successors
            for coverage_id in endpoint.lineage
        )

    def _validate_shape(self) -> None:
        predecessors = self.predecessors
        successors = self.successors
        unresolved = self.unresolved_predecessors + self.unresolved_successors
        if self.kind in {
            AffectedTransitionKind.NEWLY_OBSERVED_UNCERTIFIED,
            AffectedTransitionKind.OTHERWISE_UNRESOLVED,
        }:
            if not unresolved:
                raise ValueError("unresolved transition group requires unresolved objects")
            if self.kind is AffectedTransitionKind.NEWLY_OBSERVED_UNCERTIFIED:
                if predecessors or successors or self.unresolved_predecessors:
                    raise ValueError("newly observed uncertified object has no predecessor")
            return
        if unresolved:
            raise ValueError("certifiable transition kind cannot contain unresolved objects")
        if self.kind in {
            AffectedTransitionKind.UNCHANGED_LINEAGE,
            AffectedTransitionKind.SAME_ID_CHANGED,
            AffectedTransitionKind.REPLACED,
        }:
            if len(predecessors) != 1 or len(successors) != 1:
                raise ValueError("one-to-one transition requires one predecessor/successor")
            if predecessors[0].lineage != successors[0].lineage:
                raise ValueError("one-to-one transition must preserve inherited lineage")
            if (
                self.kind is AffectedTransitionKind.SAME_ID_CHANGED
                and predecessors[0].physical_entry.backend_entry_id
                != successors[0].physical_entry.backend_entry_id
            ):
                raise ValueError("same-ID transition must preserve backend entry ID")
            return
        if self.kind is AffectedTransitionKind.DELETED:
            if len(predecessors) != 1 or successors:
                raise ValueError("deletion requires one predecessor and no successor")
            return
        if self.kind is AffectedTransitionKind.MERGE:
            if len(predecessors) < 2 or len(successors) != 1:
                raise ValueError("merge requires multiple predecessors and one successor")
            predecessor_ids = frozenset(
                value for item in predecessors for value in item.lineage
            )
            if len(predecessor_ids) < 2:
                raise ValueError("merge requires multiple initialized lineages")
            if successors[0].lineage != canonical_tuple(predecessor_ids):
                raise ValueError("merged successor must inherit the predecessor union")
            return
        if self.kind is AffectedTransitionKind.SPLIT:
            if len(predecessors) != 1 or len(successors) < 2:
                raise ValueError("split requires one predecessor and multiple successors")
            if len(predecessors[0].lineage) != 1:
                raise ValueError("split requires one initialized predecessor lineage")
            if any(item.lineage != predecessors[0].lineage for item in successors):
                raise ValueError("every split successor must inherit the same lineage")
            return
        raise ValueError("unsupported affected transition kind")


_OUTCOME_KIND = {
    PhysicalTransitionOutcome.SAME_ID: AffectedTransitionKind.SAME_ID_CHANGED,
    PhysicalTransitionOutcome.DELETED: AffectedTransitionKind.DELETED,
    PhysicalTransitionOutcome.REPLACED: AffectedTransitionKind.REPLACED,
    PhysicalTransitionOutcome.MERGED: AffectedTransitionKind.MERGE,
    PhysicalTransitionOutcome.SPLIT: AffectedTransitionKind.SPLIT,
    PhysicalTransitionOutcome.UNCHANGED: AffectedTransitionKind.UNCHANGED_LINEAGE,
}


@dataclass(frozen=True, slots=True)
class PhysicalTransitionCertificate:
    """Complete physical transition accounting for one executed memory mutation."""

    certificate_id: str
    realized_artifact_id: str
    semantic_certificate_id: str
    campaign_id: str
    root_checkpoint_id: str
    frozen_e0: frozenset[CoverageEntryId]
    status: CertificationStatus
    observed_outcome: PhysicalTransitionOutcome | None
    target_transition_id: str | None
    affected: tuple[AffectedPhysicalTransition, ...]
    observed_affected_predecessors: frozenset[PhysicalEntryRef]
    observed_affected_successors: frozenset[PhysicalEntryRef]
    semantic_postcondition_satisfied: bool | None
    backend_proof_digest: str
    unresolved_reason: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "certificate_id",
            "realized_artifact_id",
            "semantic_certificate_id",
            "campaign_id",
            "root_checkpoint_id",
        ):
            _require_text(name, getattr(self, name))
        _require_text("backend_proof_digest", self.backend_proof_digest)
        object.__setattr__(self, "frozen_e0", frozenset(self.frozen_e0))
        object.__setattr__(self, "affected", tuple(self.affected))
        object.__setattr__(
            self,
            "observed_affected_predecessors",
            frozenset(self.observed_affected_predecessors),
        )
        object.__setattr__(
            self,
            "observed_affected_successors",
            frozenset(self.observed_affected_successors),
        )
        self._validate()

    def _validate(self) -> None:
        validate_certification_status(self.status, self.observed_outcome)
        for coverage_id in self.frozen_e0:
            _validate_id_scope(coverage_id, self.campaign_id, self.root_checkpoint_id)
        transition_ids = [item.transition_id for item in self.affected]
        if len(set(transition_ids)) != len(transition_ids):
            raise ValueError("affected transition IDs must be unique")
        predecessor_refs: set[PhysicalEntryRef] = set()
        successor_refs: set[PhysicalEntryRef] = set()
        resolved_refs: set[PhysicalEntryRef] = set()
        unresolved_refs: set[PhysicalEntryRef] = set()
        for transition in self.affected:
            for physical_entry in transition.predecessor_refs | transition.successor_refs:
                _validate_physical_scope(
                    physical_entry, self.campaign_id, self.root_checkpoint_id
                )
            unknown = transition.inherited_ids.difference(self.frozen_e0)
            if unknown:
                raise ValueError("transition introduces coverage IDs outside frozen E_0")
            if predecessor_refs.intersection(transition.predecessor_refs):
                raise ValueError("predecessor is classified by multiple transitions")
            if successor_refs.intersection(transition.successor_refs):
                raise ValueError("successor is classified by multiple transitions")
            predecessor_refs.update(transition.predecessor_refs)
            successor_refs.update(transition.successor_refs)
            resolved_refs.update(
                item.physical_entry
                for item in transition.predecessors + transition.successors
            )
            unresolved_refs.update(transition.unresolved_predecessors)
            unresolved_refs.update(transition.unresolved_successors)
        if resolved_refs.intersection(unresolved_refs):
            raise ValueError(
                "physical object cannot be both resolved and unresolved"
            )
        if predecessor_refs != set(self.observed_affected_predecessors):
            raise ValueError("transition certificate does not account for all predecessors")
        if successor_refs != set(self.observed_affected_successors):
            raise ValueError("transition certificate does not account for all successors")

        unresolved_kinds = {
            AffectedTransitionKind.NEWLY_OBSERVED_UNCERTIFIED,
            AffectedTransitionKind.OTHERWISE_UNRESOLVED,
        }
        if self.status is CertificationStatus.CERTIFIED:
            if not self.target_transition_id:
                raise ValueError("CERTIFIED requires a target transition")
            if self.unresolved_reason is not None:
                raise ValueError("CERTIFIED cannot carry an unresolved reason")
            if self.semantic_postcondition_satisfied is None:
                raise ValueError("CERTIFIED must record the semantic postcondition result")
            if not isinstance(self.semantic_postcondition_satisfied, bool):
                raise TypeError("semantic postcondition result must be a bool")
            if any(item.kind in unresolved_kinds for item in self.affected):
                raise ValueError("CERTIFIED cannot contain unresolved affected objects")
            target = next(
                (
                    item
                    for item in self.affected
                    if item.transition_id == self.target_transition_id
                ),
                None,
            )
            if target is None:
                raise ValueError("target transition is absent from affected set")
            if target.kind is not _OUTCOME_KIND[self.observed_outcome]:
                raise ValueError("observed outcome disagrees with target transition")
        elif self.status is CertificationStatus.UNRESOLVED:
            if self.target_transition_id is not None:
                raise ValueError("UNRESOLVED cannot claim a certified target transition")
            if self.semantic_postcondition_satisfied is not None:
                raise ValueError("UNRESOLVED cannot certify a semantic postcondition")
            if not self.unresolved_reason:
                raise ValueError("UNRESOLVED requires a reason")


@dataclass(frozen=True, slots=True)
class ReplayBinding:
    """Backend-certified binding of one fresh physical object to frozen E_0."""

    physical_entry: PhysicalEntryRef
    lineage: CoverageToken
    binding_proof_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "lineage", canonical_tuple(self.lineage))
        _require_text("binding_proof_digest", self.binding_proof_digest)


@dataclass(frozen=True, slots=True)
class ReplayRebindingCertificate:
    """Atomic complete binding for one freshly materialized checkpoint state.

    The caller must supply exact backend-certified bindings.  This generic
    object never matches entries by projection, inventory order, generated ID,
    or semantic similarity.
    """

    certificate_id: str
    kind: RebindingKind
    backend: str
    frozen_configuration_id: str
    campaign_id: str
    root_checkpoint_id: str
    frozen_e0: frozenset[CoverageEntryId]
    live_bindings: tuple[ReplayBinding, ...]
    expected_live_lineage_multiset: tuple[CoverageToken, ...]
    deleted_ids: frozenset[CoverageEntryId]
    backend_proof_digest: str

    def __post_init__(self) -> None:
        for name in (
            "certificate_id",
            "backend",
            "frozen_configuration_id",
            "campaign_id",
            "root_checkpoint_id",
            "backend_proof_digest",
        ):
            _require_text(name, getattr(self, name))
        frozen_e0 = frozenset(self.frozen_e0)
        deleted_ids = frozenset(self.deleted_ids)
        bindings = tuple(self.live_bindings)
        expected = tuple(canonical_tuple(token) for token in self.expected_live_lineage_multiset)
        object.__setattr__(self, "frozen_e0", frozen_e0)
        object.__setattr__(self, "deleted_ids", deleted_ids)
        object.__setattr__(self, "live_bindings", bindings)
        object.__setattr__(self, "expected_live_lineage_multiset", expected)
        self._validate()

    def _validate(self) -> None:
        if not isinstance(self.kind, RebindingKind):
            raise TypeError("kind must be a RebindingKind")
        for coverage_id in self.frozen_e0 | self.deleted_ids:
            _validate_id_scope(coverage_id, self.campaign_id, self.root_checkpoint_id)
        if not self.deleted_ids.issubset(self.frozen_e0):
            raise ValueError("deleted IDs must belong to frozen E_0")
        physical_refs: set[PhysicalEntryRef] = set()
        live_ids: set[CoverageEntryId] = set()
        actual_tokens: list[CoverageToken] = []
        for token in self.expected_live_lineage_multiset:
            for coverage_id in token:
                _validate_id_scope(
                    coverage_id, self.campaign_id, self.root_checkpoint_id
                )
                if coverage_id not in self.frozen_e0:
                    raise ValueError(
                        "expected lineage contains coverage IDs outside frozen E_0"
                    )
        for binding in self.live_bindings:
            _validate_physical_scope(
                binding.physical_entry, self.campaign_id, self.root_checkpoint_id
            )
            if binding.physical_entry in physical_refs:
                raise ValueError("a live physical object may be bound only once")
            physical_refs.add(binding.physical_entry)
            unknown = set(binding.lineage).difference(self.frozen_e0)
            if unknown:
                raise ValueError("re-binding introduces coverage IDs outside frozen E_0")
            live_ids.update(binding.lineage)
            actual_tokens.append(binding.lineage)
        if live_ids.intersection(self.deleted_ids):
            raise ValueError("deleted and live initialized IDs must be disjoint")
        if live_ids.union(self.deleted_ids) != set(self.frozen_e0):
            raise ValueError("re-binding must account for every frozen E_0 entry")
        if Counter(actual_tokens) != Counter(self.expected_live_lineage_multiset):
            raise ValueError("live bindings do not match expected lineage multiset")
        if self.kind is RebindingKind.ROOT:
            if self.deleted_ids:
                raise ValueError("root re-binding cannot mark initialized IDs deleted")
            if any(len(binding.lineage) != 1 for binding in self.live_bindings):
                raise ValueError("root re-binding permits singleton lineages only")
            counts = Counter(
                coverage_id
                for binding in self.live_bindings
                for coverage_id in binding.lineage
            )
            if set(counts) != set(self.frozen_e0) or any(
                count != 1 for count in counts.values()
            ):
                raise ValueError("root re-binding must map each E_0 entry exactly once")


def build_rebound_lineage(
    certificate: ReplayRebindingCertificate,
    inventory: FrozenCheckpointInventory,
) -> InitialEntryLineage:
    """Validate completely, then return a fresh live-only lineage registry.

    No caller-owned registry is mutated.  Descendant deletions remain explicit
    in the certificate because a deleted entry has no fresh physical object.
    """

    if inventory.campaign_id != certificate.campaign_id:
        raise ValueError("inventory belongs to another campaign")
    if inventory.root_checkpoint_id != certificate.root_checkpoint_id:
        raise ValueError("inventory belongs to another root checkpoint")
    if inventory.coverage_ids != certificate.frozen_e0:
        raise ValueError("inventory does not equal the certificate's frozen E_0")

    # Certificate construction has already validated every binding, so the
    # mutable registry remains local until all inserts finish successfully.
    lineage = InitialEntryLineage(certificate.campaign_id, (inventory,))
    for binding in certificate.live_bindings:
        relation = (
            LineageRelation.CERTIFIED_MERGE
            if len(binding.lineage) > 1
            else LineageRelation.INITIAL
        )
        lineage.certify_binding(binding.physical_entry, binding.lineage, relation)
    return lineage


@dataclass(frozen=True, slots=True)
class LiveLineageState:
    """Persistent observable state attached to one inherited lineage token."""

    lineage: CoverageToken
    observable_state_digest: str
    stable_provenance_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "lineage", canonical_tuple(self.lineage))
        _require_text("observable_state_digest", self.observable_state_digest)
        _require_text("stable_provenance_digest", self.stable_provenance_digest)


@dataclass(frozen=True, slots=True)
class DescendantStateCertificate:
    """Lineage-aware persistent state expected from any rematerialization."""

    certificate_id: str
    backend: str
    frozen_configuration_id: str
    campaign_id: str
    root_checkpoint_id: str
    frozen_e0: frozenset[CoverageEntryId]
    live_lineage_states: tuple[LiveLineageState, ...]
    deleted_ids: frozenset[CoverageEntryId]
    observable_state_digest: str
    transition_state_digest: str
    accepted_transition_certificate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "certificate_id",
            "backend",
            "frozen_configuration_id",
            "campaign_id",
            "root_checkpoint_id",
            "observable_state_digest",
            "transition_state_digest",
        ):
            _require_text(name, getattr(self, name))
        frozen_e0 = frozenset(self.frozen_e0)
        deleted = frozenset(self.deleted_ids)
        live = tuple(
            sorted(
                self.live_lineage_states,
                key=lambda item: (
                    item.lineage,
                    item.observable_state_digest,
                    item.stable_provenance_digest,
                ),
            )
        )
        transition_ids = tuple(self.accepted_transition_certificate_ids)
        if any(not isinstance(item, str) or not item for item in transition_ids):
            raise ValueError("accepted transition certificate IDs must be non-empty")
        if len(set(transition_ids)) != len(transition_ids):
            raise ValueError("accepted transition certificate IDs cannot repeat")
        object.__setattr__(self, "frozen_e0", frozen_e0)
        object.__setattr__(self, "deleted_ids", deleted)
        object.__setattr__(self, "live_lineage_states", live)
        object.__setattr__(self, "accepted_transition_certificate_ids", transition_ids)
        live_ids: set[CoverageEntryId] = set()
        for coverage_id in frozen_e0 | deleted:
            _validate_id_scope(coverage_id, self.campaign_id, self.root_checkpoint_id)
        if not deleted.issubset(frozen_e0):
            raise ValueError("deleted IDs must belong to frozen E_0")
        for state in live:
            for coverage_id in state.lineage:
                _validate_id_scope(
                    coverage_id, self.campaign_id, self.root_checkpoint_id
                )
                if coverage_id not in frozen_e0:
                    raise ValueError("live lineage contains an ID outside frozen E_0")
                live_ids.add(coverage_id)
        if live_ids.intersection(deleted):
            raise ValueError("deleted and live initialized IDs must be disjoint")
        if live_ids.union(deleted) != set(frozen_e0):
            raise ValueError("descendant certificate must account for all frozen E_0")

    @property
    def expected_live_lineage_multiset(self) -> tuple[CoverageToken, ...]:
        return tuple(item.lineage for item in self.live_lineage_states)


def descendant_states_equivalent(
    expected: DescendantStateCertificate,
    observed: DescendantStateCertificate,
) -> bool:
    """Compare complete lineage-aware persistent state, excluding certificate ID."""

    return (
        expected.backend,
        expected.frozen_configuration_id,
        expected.campaign_id,
        expected.root_checkpoint_id,
        expected.frozen_e0,
        expected.live_lineage_states,
        expected.deleted_ids,
        expected.observable_state_digest,
        expected.transition_state_digest,
        expected.accepted_transition_certificate_ids,
    ) == (
        observed.backend,
        observed.frozen_configuration_id,
        observed.campaign_id,
        observed.root_checkpoint_id,
        observed.frozen_e0,
        observed.live_lineage_states,
        observed.deleted_ids,
        observed.observable_state_digest,
        observed.transition_state_digest,
        observed.accepted_transition_certificate_ids,
    )


@dataclass(frozen=True, slots=True)
class LogicalSeed:
    """Persistent logical state identity, independent of live backend objects."""

    seed_id: str
    campaign_id: str
    backend: str
    root_checkpoint_id: str
    frozen_configuration_id: str
    current_query_artifact: FrozenMapping
    memory_transition_artifacts: tuple[RealizedMutationArtifact, ...]
    expected_descendant_state: DescendantStateCertificate
    authoritative_parent_signature: RetrievalSignature

    def __post_init__(self) -> None:
        for name in (
            "seed_id",
            "campaign_id",
            "backend",
            "root_checkpoint_id",
            "frozen_configuration_id",
        ):
            _require_text(name, getattr(self, name))
        object.__setattr__(
            self,
            "current_query_artifact",
            _freeze_mapping(self.current_query_artifact),
        )
        artifacts = tuple(self.memory_transition_artifacts)
        object.__setattr__(self, "memory_transition_artifacts", artifacts)
        state = self.expected_descendant_state
        if (
            state.campaign_id,
            state.backend,
            state.root_checkpoint_id,
            state.frozen_configuration_id,
        ) != (
            self.campaign_id,
            self.backend,
            self.root_checkpoint_id,
            self.frozen_configuration_id,
        ):
            raise ValueError("descendant state certificate does not match seed scope")
        for artifact in artifacts:
            opportunity = artifact.opportunity
            if opportunity.relation not in _MEMORY_RELATIONS:
                raise ValueError("seed memory path may contain memory mutations only")
            if (
                opportunity.campaign_id,
                opportunity.root_checkpoint_id,
            ) != (self.campaign_id, self.root_checkpoint_id):
                raise ValueError("memory transition artifact belongs to another seed scope")
        if len(artifacts) != len(state.accepted_transition_certificate_ids):
            raise ValueError(
                "memory artifact path and accepted transition path must have equal length"
            )
        _validate_signature_scope(
            self.authoritative_parent_signature,
            self.campaign_id,
            self.root_checkpoint_id,
        )


def validate_opportunity_parent(
    opportunity: MutationOpportunity,
    parent: LogicalSeed,
) -> None:
    """Require an opportunity to reference the exact persistent parent seed."""

    if not isinstance(opportunity, MutationOpportunity):
        raise TypeError("opportunity must be a MutationOpportunity")
    if not isinstance(parent, LogicalSeed):
        raise TypeError("parent must be a LogicalSeed")
    if opportunity.parent_seed_id != parent.seed_id:
        raise ValueError("opportunity references another parent seed")
    if (
        opportunity.campaign_id,
        opportunity.root_checkpoint_id,
    ) != (parent.campaign_id, parent.root_checkpoint_id):
        raise ValueError("opportunity and parent seed scopes do not match")


@dataclass(frozen=True, slots=True)
class SeedAdmission:
    """Execution validity and method-independent future-parent eligibility."""

    execution_valid: bool
    rematerializable: bool
    retainable_as_parent: bool

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, bool)
            for value in (
                self.execution_valid,
                self.rematerializable,
                self.retainable_as_parent,
            )
        ):
            raise TypeError("seed admission values must be bools")
        if self.retainable_as_parent and not (
            self.execution_valid and self.rematerializable
        ):
            raise ValueError(
                "retainable parent must be execution-valid and rematerializable"
            )

    @classmethod
    def for_executed_child(
        cls, *, execution_valid: bool, rematerializable: bool
    ) -> SeedAdmission:
        return cls(
            execution_valid,
            rematerializable,
            execution_valid and rematerializable,
        )


@dataclass(frozen=True, slots=True)
class MaterializationFailure:
    """One classified failure without retry-count or abort-policy decisions."""

    kind: MaterializationFailureKind
    campaign_id: str
    root_checkpoint_id: str
    reason: str

    def __post_init__(self) -> None:
        for name in ("campaign_id", "root_checkpoint_id", "reason"):
            _require_text(name, getattr(self, name))
        if not isinstance(self.kind, MaterializationFailureKind):
            raise TypeError("kind must be a MaterializationFailureKind")


_ResultValue = TypeVar("_ResultValue")


@dataclass(frozen=True, slots=True)
class MaterializationResult(Generic[_ResultValue]):
    """Pure success/failure carrier; retry and abort policy remain external."""

    value: _ResultValue | None = None
    failure: MaterializationFailure | None = None

    def __post_init__(self) -> None:
        if (self.value is None) == (self.failure is None):
            raise ValueError("materialization result requires exactly one value or failure")

    @classmethod
    def success(cls, value: _ResultValue) -> MaterializationResult[_ResultValue]:
        if value is None:
            raise ValueError("successful materialization value cannot be None")
        return cls(value=value)

    @classmethod
    def failed(
        cls, failure: MaterializationFailure
    ) -> MaterializationResult[_ResultValue]:
        return cls(failure=failure)


def validate_memory_child_contract(
    artifact: RealizedMutationArtifact,
    transition: PhysicalTransitionCertificate,
    descendant: DescendantStateCertificate,
) -> None:
    """Validate pure certificate composition for a persistent memory child."""

    opportunity = artifact.opportunity
    if opportunity.relation not in _MEMORY_RELATIONS:
        raise ValueError("physical transition is unavailable for a query-only artifact")
    if transition.status is not CertificationStatus.CERTIFIED:
        raise ValueError("memory child requires a certified physical transition")
    if transition.realized_artifact_id != artifact.artifact_id:
        raise ValueError("physical transition certifies another realized artifact")
    if (
        transition.semantic_certificate_id
        != artifact.semantic_certificate.certificate_id
    ):
        raise ValueError("physical transition certifies another semantic certificate")
    if transition.observed_outcome not in opportunity.acceptable_transition_outcomes:
        raise ValueError("observed physical outcome is not acceptable")
    if transition.observed_outcome is PhysicalTransitionOutcome.UNCHANGED:
        raise ValueError("memory child requires an actual semantic state change")
    if transition.semantic_postcondition_satisfied is not True:
        raise ValueError("memory child failed its semantic postcondition")
    target = next(
        item
        for item in transition.affected
        if item.transition_id == transition.target_transition_id
    )
    if not set(opportunity.target_lineage).issubset(target.inherited_ids):
        raise ValueError("physical target does not include the opportunity target lineage")
    if (
        transition.campaign_id,
        transition.root_checkpoint_id,
        transition.frozen_e0,
    ) != (
        opportunity.campaign_id,
        opportunity.root_checkpoint_id,
        descendant.frozen_e0,
    ):
        raise ValueError("transition/opportunity/descendant scopes do not match")
    if (
        descendant.campaign_id,
        descendant.root_checkpoint_id,
    ) != (opportunity.campaign_id, opportunity.root_checkpoint_id):
        raise ValueError("descendant certificate belongs to another scope")
    if not descendant.accepted_transition_certificate_ids:
        raise ValueError("descendant must record the accepted transition certificate")
    if descendant.accepted_transition_certificate_ids[-1] != transition.certificate_id:
        raise ValueError("descendant does not end with the certified transition")
