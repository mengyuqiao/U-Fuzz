"""Async orchestration for certified logical-seed materialization.

The protocol and carriers in this module are backend-independent.  Live
handles and materialization-local lineage are ephemeral; persistent identity
continues to live exclusively in the frozen state-contract objects.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from ufuzz.coverage import (
    CoverageEntryId,
    CoverageState,
    FrozenCheckpointInventory,
    FrozenMapping,
    InitialEntryLineage,
    LineageResolutionError,
)
from ufuzz.retrieval_capability import (
    RankedPhysicalRetrieval,
    ResolvedRetrievalObservation,
    resolve_fuzzing_retrieval,
)
from ufuzz.state_contract import (
    CertificationStatus,
    DescendantStateCertificate,
    LogicalSeed,
    MaterializationFailure,
    MaterializationFailureKind,
    MaterializationResult,
    PhysicalTransitionCertificate,
    PhysicalTransitionOutcome,
    RealizedMutationArtifact,
    RebindingKind,
    ReplayBinding,
    ReplayRebindingCertificate,
    build_rebound_lineage,
    descendant_states_equivalent,
    validate_memory_child_contract,
)


HandleT = TypeVar("HandleT")


@dataclass(frozen=True, slots=True)
class RootMaterializationRequest:
    """Search-safe identity needed to reconstruct one campaign root.

    The request deliberately excludes the descendant query, transition path,
    expected descendant state, parent signature, and search feedback.  A
    concrete materializer resolves ``initialization_artifact_id`` to the
    already-frozen ``InitializationArtifact`` owned by its campaign layer.
    """

    backend: str
    frozen_configuration_id: str
    campaign_id: str
    root_checkpoint_id: str
    initialization_artifact_id: str
    frozen_e0: frozenset[CoverageEntryId]

    def __post_init__(self) -> None:
        for name in (
            "backend",
            "frozen_configuration_id",
            "campaign_id",
            "root_checkpoint_id",
            "initialization_artifact_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        frozen_e0 = frozenset(self.frozen_e0)
        for coverage_id in frozen_e0:
            if (
                coverage_id.campaign_id,
                coverage_id.root_checkpoint_id,
            ) != (self.campaign_id, self.root_checkpoint_id):
                raise ValueError("root request E_0 contains an ID from another scope")
        object.__setattr__(self, "frozen_e0", frozen_e0)


@dataclass(slots=True)
class EphemeralMaterializedState(Generic[HandleT]):
    """One live backend state, deliberately excluded from persistent identity."""

    handle: HandleT
    backend: str
    frozen_configuration_id: str
    campaign_id: str
    root_checkpoint_id: str
    state_id: str

    def __post_init__(self) -> None:
        for name in (
            "backend",
            "frozen_configuration_id",
            "campaign_id",
            "root_checkpoint_id",
            "state_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")


@dataclass(slots=True)
class RootMaterialization(Generic[HandleT]):
    """Protocol output for a newly created and observed root backend state."""

    state: EphemeralMaterializedState[HandleT]
    rebinding_certificate: ReplayRebindingCertificate
    observed_root_state: DescendantStateCertificate


@dataclass(slots=True)
class TransitionReplay(Generic[HandleT]):
    """Protocol output after executing one exact realized artifact."""

    state: EphemeralMaterializedState[HandleT]
    certificate: PhysicalTransitionCertificate


@dataclass(slots=True)
class AppliedTransitionLineage:
    """Fresh materialization-local lineage derived from one certified transition."""

    lineage: InitialEntryLineage
    rebinding_certificate: ReplayRebindingCertificate


@dataclass(slots=True)
class CertifiedMaterialization(Generic[HandleT]):
    """Ephemeral live state published only after complete certification."""

    seed_id: str
    state: EphemeralMaterializedState[HandleT]
    lineage: InitialEntryLineage
    current_rebinding: ReplayRebindingCertificate
    observed_descendant_state: DescendantStateCertificate
    transition_certificates: tuple[PhysicalTransitionCertificate, ...]


@dataclass(slots=True)
class CertifiedExecutionObservation(Generic[HandleT]):
    """Certified ephemeral materialization plus its resolved public retrieval."""

    seed_id: str
    materialization: CertifiedMaterialization[HandleT]
    observation: ResolvedRetrievalObservation


class MaterializationProtocol(Protocol[HandleT]):
    """Minimum async contract implemented later by concrete backend materializers."""

    backend: str
    frozen_configuration_id: str

    async def materialize_root(
        self,
        request: RootMaterializationRequest,
    ) -> MaterializationResult[RootMaterialization[HandleT]]:
        """Create a root and transfer its live handle only on success.

        A failing implementation must retire every handle it created because
        no unpublished handle is available to the generic orchestrator.
        """

        ...

    async def replay_transition(
        self,
        state: EphemeralMaterializedState[HandleT],
        artifact: RealizedMutationArtifact,
    ) -> MaterializationResult[TransitionReplay[HandleT]]:
        """Execute exactly one artifact and transfer live-state ownership.

        On success ownership transfers to the returned state.  If it contains
        a distinct successor handle, the protocol retires the superseded
        handle before returning.  On failure the supplied current state stays
        owned by the orchestrator and remains discardable; the protocol must
        retire any unpublished handle that it created internally.
        """

        ...

    async def certify_descendant(
        self,
        state: EphemeralMaterializedState[HandleT],
        transition_certificates: tuple[PhysicalTransitionCertificate, ...],
    ) -> MaterializationResult[DescendantStateCertificate]: ...

    async def retrieve(
        self,
        state: EphemeralMaterializedState[HandleT],
        *,
        query_artifact: FrozenMapping,
        top_k: int,
    ) -> MaterializationResult[RankedPhysicalRetrieval]: ...

    async def discard(self, state: EphemeralMaterializedState[HandleT]) -> None:
        """Retire the one live state currently owned by the caller."""

        ...


def _failure(
    seed: LogicalSeed,
    kind: MaterializationFailureKind,
    reason: str,
) -> MaterializationResult[CertifiedExecutionObservation[HandleT]]:
    return MaterializationResult.failed(
        MaterializationFailure(
            kind,
            seed.campaign_id,
            seed.root_checkpoint_id,
            reason,
        )
    )


def _normalize_protocol_failure(
    seed: LogicalSeed,
    failure: MaterializationFailure,
) -> MaterializationFailure:
    return _normalize_protocol_failure_for_scope(
        seed.campaign_id,
        seed.root_checkpoint_id,
        failure,
    )


def _normalize_protocol_failure_for_scope(
    campaign_id: str,
    root_checkpoint_id: str,
    failure: MaterializationFailure,
) -> MaterializationFailure:
    if (failure.campaign_id, failure.root_checkpoint_id) == (
        campaign_id,
        root_checkpoint_id,
    ):
        return failure
    return MaterializationFailure(
        MaterializationFailureKind.BACKEND_CAPABILITY_FAILURE,
        campaign_id,
        root_checkpoint_id,
        "protocol returned a failure for another campaign/checkpoint",
    )


async def _discard_and_fail(
    protocol: MaterializationProtocol[HandleT],
    state: EphemeralMaterializedState[HandleT] | None,
    failure: MaterializationFailure,
) -> MaterializationResult[CertifiedExecutionObservation[HandleT]]:
    if state is not None:
        try:
            await protocol.discard(state)
        except Exception:
            # The failed state is never published.  Cleanup policy and retry
            # remain outside this semantic orchestration primitive.
            pass
    return MaterializationResult.failed(failure)


def _validate_state_scope(
    state: EphemeralMaterializedState[HandleT],
    seed: LogicalSeed,
) -> None:
    if (
        state.backend,
        state.frozen_configuration_id,
        state.campaign_id,
        state.root_checkpoint_id,
    ) != (
        seed.backend,
        seed.frozen_configuration_id,
        seed.campaign_id,
        seed.root_checkpoint_id,
    ):
        raise ValueError("ephemeral backend state does not match logical seed scope")


def _validate_rebinding_state_id(
    certificate: ReplayRebindingCertificate,
    state_id: str,
) -> None:
    if any(
        binding.physical_entry.state_id != state_id
        for binding in certificate.live_bindings
    ):
        raise ValueError("re-binding contains a physical object from another live state")


def _validate_descendant_against_rebinding(
    descendant: DescendantStateCertificate,
    rebinding: ReplayRebindingCertificate,
) -> None:
    if (
        descendant.backend,
        descendant.frozen_configuration_id,
        descendant.campaign_id,
        descendant.root_checkpoint_id,
        descendant.frozen_e0,
    ) != (
        rebinding.backend,
        rebinding.frozen_configuration_id,
        rebinding.campaign_id,
        rebinding.root_checkpoint_id,
        rebinding.frozen_e0,
    ):
        raise ValueError("descendant state and live re-binding scopes differ")
    if Counter(descendant.expected_live_lineage_multiset) != Counter(
        rebinding.expected_live_lineage_multiset
    ):
        raise ValueError("descendant live lineage multiset differs from re-binding")
    if descendant.deleted_ids != rebinding.deleted_ids:
        raise ValueError("descendant deleted IDs differ from re-binding")


def apply_certified_transition(
    *,
    current: ReplayRebindingCertificate,
    transition: PhysicalTransitionCertificate,
    inventory: FrozenCheckpointInventory,
    current_state_id: str,
    successor_state_id: str,
) -> AppliedTransitionLineage:
    """Apply one certified transition to a fresh materialization-local lineage.

    The transition certificate proves the physical change.  This function
    stores its resulting live bindings in a new ``InitialEntryLineage``; it
    never treats the registry itself as transition evidence.
    """

    if transition.status is not CertificationStatus.CERTIFIED:
        raise ValueError("cannot apply an unresolved physical transition")
    if (
        transition.campaign_id,
        transition.root_checkpoint_id,
        transition.frozen_e0,
    ) != (current.campaign_id, current.root_checkpoint_id, current.frozen_e0):
        raise ValueError("physical transition and current re-binding scopes differ")
    if inventory.coverage_ids != current.frozen_e0:
        raise ValueError("inventory does not match frozen re-binding E_0")
    _validate_rebinding_state_id(current, current_state_id)

    live = {
        binding.physical_entry: binding.lineage
        for binding in current.live_bindings
    }
    for affected in transition.affected:
        for predecessor in affected.predecessors:
            if predecessor.physical_entry.state_id != current_state_id:
                raise ValueError("transition predecessor belongs to another live state")
            current_lineage = live.get(predecessor.physical_entry)
            if current_lineage is None:
                raise ValueError("transition predecessor is not live")
            if current_lineage != predecessor.lineage:
                raise ValueError("transition predecessor lineage disagrees with live state")

    for affected in transition.affected:
        for predecessor in affected.predecessors:
            del live[predecessor.physical_entry]

    for affected in transition.affected:
        for successor in affected.successors:
            if successor.physical_entry.state_id != successor_state_id:
                raise ValueError("transition successor belongs to another live state")
            if successor.physical_entry in live:
                raise ValueError("transition successor collides with an unaffected object")
            live[successor.physical_entry] = successor.lineage

    live_ids = frozenset(
        coverage_id for token in live.values() for coverage_id in token
    )
    deleted_ids = current.frozen_e0.difference(live_ids)
    bindings = tuple(
        ReplayBinding(
            physical_entry,
            token,
            f"physical-transition:{transition.certificate_id}",
        )
        for physical_entry, token in sorted(live.items())
    )
    derived = ReplayRebindingCertificate(
        certificate_id=f"applied:{transition.certificate_id}",
        kind=RebindingKind.DESCENDANT,
        backend=current.backend,
        frozen_configuration_id=current.frozen_configuration_id,
        campaign_id=current.campaign_id,
        root_checkpoint_id=current.root_checkpoint_id,
        frozen_e0=current.frozen_e0,
        live_bindings=bindings,
        expected_live_lineage_multiset=tuple(item.lineage for item in bindings),
        deleted_ids=deleted_ids,
        backend_proof_digest=f"derived-from:{transition.certificate_id}",
    )
    lineage = build_rebound_lineage(derived, inventory)
    return AppliedTransitionLineage(lineage, derived)


def _validate_transition_for_artifact(
    seed: LogicalSeed,
    artifact: RealizedMutationArtifact,
    transition: PhysicalTransitionCertificate,
    inventory: FrozenCheckpointInventory,
) -> MaterializationFailure | None:
    if transition.status is not CertificationStatus.CERTIFIED:
        return MaterializationFailure(
            MaterializationFailureKind.TRANSITION_REPLAY_DIVERGED,
            seed.campaign_id,
            seed.root_checkpoint_id,
            "replayed transition was not certified",
        )
    if (
        transition.realized_artifact_id != artifact.artifact_id
        or transition.semantic_certificate_id
        != artifact.semantic_certificate.certificate_id
    ):
        return MaterializationFailure(
            MaterializationFailureKind.TRANSITION_REPLAY_DIVERGED,
            seed.campaign_id,
            seed.root_checkpoint_id,
            "transition certificate does not bind the exact replayed artifact",
        )
    if (
        transition.campaign_id,
        transition.root_checkpoint_id,
        transition.frozen_e0,
    ) != (seed.campaign_id, seed.root_checkpoint_id, inventory.coverage_ids):
        return MaterializationFailure(
            MaterializationFailureKind.TRANSITION_REPLAY_DIVERGED,
            seed.campaign_id,
            seed.root_checkpoint_id,
            "transition certificate scope or frozen E_0 diverged",
        )
    if transition.observed_outcome not in artifact.opportunity.acceptable_transition_outcomes:
        return MaterializationFailure(
            MaterializationFailureKind.TRANSITION_REPLAY_DIVERGED,
            seed.campaign_id,
            seed.root_checkpoint_id,
            "replayed physical outcome is not acceptable for the artifact",
        )
    if transition.observed_outcome is PhysicalTransitionOutcome.UNCHANGED:
        return MaterializationFailure(
            MaterializationFailureKind.TRANSITION_REPLAY_DIVERGED,
            seed.campaign_id,
            seed.root_checkpoint_id,
            "replayed memory transition made no physical change",
        )
    if transition.semantic_postcondition_satisfied is not True:
        return MaterializationFailure(
            MaterializationFailureKind.TRANSITION_SEMANTIC_DRIFT,
            seed.campaign_id,
            seed.root_checkpoint_id,
            "replayed transition violated its semantic postcondition",
        )
    target = next(
        item
        for item in transition.affected
        if item.transition_id == transition.target_transition_id
    )
    if not set(artifact.opportunity.target_lineage).issubset(target.inherited_ids):
        return MaterializationFailure(
            MaterializationFailureKind.TRANSITION_REPLAY_DIVERGED,
            seed.campaign_id,
            seed.root_checkpoint_id,
            "replayed physical target does not contain the artifact target lineage",
        )
    return None


async def materialize_seed(
    *,
    seed: LogicalSeed,
    protocol: MaterializationProtocol[HandleT],
    inventory: FrozenCheckpointInventory,
    expected_root_state: DescendantStateCertificate,
    initialization_artifact_id: str,
) -> MaterializationResult[CertifiedMaterialization[HandleT]]:
    """Rematerialize and certify a logical seed without executing retrieval.

    Every live state created by this orchestration is discarded on failure.
    Success transfers ownership of the returned materialization to the caller.
    """

    if (
        inventory.campaign_id,
        inventory.root_checkpoint_id,
        inventory.coverage_ids,
    ) != (seed.campaign_id, seed.root_checkpoint_id, seed.expected_descendant_state.frozen_e0):
        return _failure(
            seed,
            MaterializationFailureKind.BACKEND_CAPABILITY_FAILURE,
            "supplied inventory does not match logical seed frozen E_0",
        )
    if (
        protocol.backend,
        protocol.frozen_configuration_id,
    ) != (seed.backend, seed.frozen_configuration_id):
        return _failure(
            seed,
            MaterializationFailureKind.BACKEND_CAPABILITY_FAILURE,
            "materialization protocol does not match seed backend/configuration",
        )
    if expected_root_state.accepted_transition_certificate_ids:
        return _failure(
            seed,
            MaterializationFailureKind.ROOT_INITIALIZATION_INEQUIVALENT,
            "expected root state cannot contain memory transitions",
        )

    try:
        root_request = RootMaterializationRequest(
            backend=seed.backend,
            frozen_configuration_id=seed.frozen_configuration_id,
            campaign_id=seed.campaign_id,
            root_checkpoint_id=seed.root_checkpoint_id,
            initialization_artifact_id=initialization_artifact_id,
            frozen_e0=inventory.coverage_ids,
        )
    except (TypeError, ValueError) as error:
        return _failure(
            seed,
            MaterializationFailureKind.BACKEND_CAPABILITY_FAILURE,
            str(error),
        )

    state: EphemeralMaterializedState[HandleT] | None = None
    try:
        root_result = await protocol.materialize_root(root_request)
    except Exception as error:
        return _failure(
            seed,
            MaterializationFailureKind.BACKEND_CAPABILITY_FAILURE,
            f"root materialization raised {type(error).__name__}",
        )
    if root_result.failure is not None:
        return MaterializationResult.failed(
            _normalize_protocol_failure(seed, root_result.failure)
        )
    root = root_result.value
    if not isinstance(root, RootMaterialization):
        return _failure(
            seed,
            MaterializationFailureKind.BACKEND_CAPABILITY_FAILURE,
            "protocol returned an invalid root materialization object",
        )
    state = root.state

    try:
        _validate_state_scope(state, seed)
        root_certificate = root.rebinding_certificate
        if root_certificate.kind is not RebindingKind.ROOT:
            raise ValueError("root materialization requires a root re-binding")
        if (
            root_certificate.backend,
            root_certificate.frozen_configuration_id,
            root_certificate.campaign_id,
            root_certificate.root_checkpoint_id,
            root_certificate.frozen_e0,
        ) != (
            seed.backend,
            seed.frozen_configuration_id,
            seed.campaign_id,
            seed.root_checkpoint_id,
            inventory.coverage_ids,
        ):
            raise ValueError("root re-binding scope does not match logical seed")
        _validate_rebinding_state_id(root_certificate, state.state_id)
        lineage = build_rebound_lineage(root_certificate, inventory)
    except (TypeError, ValueError) as error:
        return await _discard_and_fail(
            protocol,
            state,
            MaterializationFailure(
                MaterializationFailureKind.E0_REBINDING_AMBIGUOUS,
                seed.campaign_id,
                seed.root_checkpoint_id,
                str(error),
            ),
        )

    try:
        _validate_descendant_against_rebinding(
            root.observed_root_state, root_certificate
        )
        if not descendant_states_equivalent(
            expected_root_state, root.observed_root_state
        ):
            raise ValueError("observed root state differs from expected root state")
    except (TypeError, ValueError) as error:
        return await _discard_and_fail(
            protocol,
            state,
            MaterializationFailure(
                MaterializationFailureKind.ROOT_INITIALIZATION_INEQUIVALENT,
                seed.campaign_id,
                seed.root_checkpoint_id,
                str(error),
            ),
        )

    current_rebinding = root_certificate
    certificates: list[PhysicalTransitionCertificate] = []
    for artifact in seed.memory_transition_artifacts:
        previous_state = state
        try:
            replay_result = await protocol.replay_transition(previous_state, artifact)
        except Exception as error:
            return await _discard_and_fail(
                protocol,
                state,
                MaterializationFailure(
                    MaterializationFailureKind.BACKEND_CAPABILITY_FAILURE,
                    seed.campaign_id,
                    seed.root_checkpoint_id,
                    f"transition replay raised {type(error).__name__}",
                ),
            )
        if replay_result.failure is not None:
            return await _discard_and_fail(
                protocol,
                state,
                _normalize_protocol_failure(seed, replay_result.failure),
            )
        replay = replay_result.value
        if not isinstance(replay, TransitionReplay):
            return await _discard_and_fail(
                protocol,
                state,
                MaterializationFailure(
                    MaterializationFailureKind.BACKEND_CAPABILITY_FAILURE,
                    seed.campaign_id,
                    seed.root_checkpoint_id,
                    "protocol returned an invalid transition replay object",
                ),
            )
        state = replay.state
        try:
            _validate_state_scope(state, seed)
        except ValueError as error:
            return await _discard_and_fail(
                protocol,
                state,
                MaterializationFailure(
                    MaterializationFailureKind.TRANSITION_REPLAY_DIVERGED,
                    seed.campaign_id,
                    seed.root_checkpoint_id,
                    str(error),
                ),
            )
        transition_failure = _validate_transition_for_artifact(
            seed, artifact, replay.certificate, inventory
        )
        if transition_failure is not None:
            return await _discard_and_fail(protocol, state, transition_failure)
        try:
            applied = apply_certified_transition(
                current=current_rebinding,
                transition=replay.certificate,
                inventory=inventory,
                current_state_id=previous_state.state_id,
                successor_state_id=state.state_id,
            )
            _validate_rebinding_state_id(
                applied.rebinding_certificate, state.state_id
            )
        except (TypeError, ValueError) as error:
            return await _discard_and_fail(
                protocol,
                state,
                MaterializationFailure(
                    MaterializationFailureKind.TRANSITION_REPLAY_DIVERGED,
                    seed.campaign_id,
                    seed.root_checkpoint_id,
                    str(error),
                ),
            )
        lineage = applied.lineage
        current_rebinding = applied.rebinding_certificate
        certificates.append(replay.certificate)

    try:
        final_result = await protocol.certify_descendant(state, tuple(certificates))
    except Exception as error:
        return await _discard_and_fail(
            protocol,
            state,
            MaterializationFailure(
                MaterializationFailureKind.BACKEND_CAPABILITY_FAILURE,
                seed.campaign_id,
                seed.root_checkpoint_id,
                f"descendant certification raised {type(error).__name__}",
            ),
        )
    if final_result.failure is not None:
        return await _discard_and_fail(
            protocol,
            state,
            _normalize_protocol_failure(seed, final_result.failure),
        )
    observed_descendant = final_result.value
    if not isinstance(observed_descendant, DescendantStateCertificate):
        return await _discard_and_fail(
            protocol,
            state,
            MaterializationFailure(
                MaterializationFailureKind.BACKEND_CAPABILITY_FAILURE,
                seed.campaign_id,
                seed.root_checkpoint_id,
                "protocol returned an invalid descendant certificate",
            ),
        )
    try:
        _validate_descendant_against_rebinding(
            observed_descendant, current_rebinding
        )
        if tuple(item.certificate_id for item in certificates) != (
            observed_descendant.accepted_transition_certificate_ids
        ):
            raise ValueError("descendant certificate transition order differs from replay")
        if not descendant_states_equivalent(
            seed.expected_descendant_state, observed_descendant
        ):
            raise ValueError("observed descendant state differs from logical seed")
        if certificates:
            validate_memory_child_contract(
                seed.memory_transition_artifacts[-1],
                certificates[-1],
                observed_descendant,
            )
    except (TypeError, ValueError) as error:
        return await _discard_and_fail(
            protocol,
            state,
            MaterializationFailure(
                MaterializationFailureKind.DESCENDANT_STATE_INEQUIVALENT,
                seed.campaign_id,
                seed.root_checkpoint_id,
                str(error),
            ),
        )

    materialization = CertifiedMaterialization(
        seed.seed_id,
        state,
        lineage,
        current_rebinding,
        observed_descendant,
        tuple(certificates),
    )
    return MaterializationResult.success(materialization)


async def observe_materialization(
    *,
    seed_id: str,
    protocol: MaterializationProtocol[HandleT],
    materialization: CertifiedMaterialization[HandleT],
    query_artifact: FrozenMapping,
    coverage_state: CoverageState,
    top_k: int,
) -> MaterializationResult[CertifiedExecutionObservation[HandleT]]:
    """Execute and resolve retrieval from an already-certified materialization.

    This function borrows the successful materialization.  It never discards
    it and never applies the immutable ``CoverageUpdate`` returned inside the
    resolved observation.
    """

    if not isinstance(seed_id, str) or not seed_id:
        raise ValueError("seed_id must be a non-empty string")
    if seed_id != materialization.seed_id:
        return MaterializationResult.failed(
            MaterializationFailure(
                MaterializationFailureKind.BACKEND_CAPABILITY_FAILURE,
                materialization.state.campaign_id,
                materialization.state.root_checkpoint_id,
                "observation seed identity differs from certified materialization",
            )
        )
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    state = materialization.state
    if (
        protocol.backend,
        protocol.frozen_configuration_id,
    ) != (state.backend, state.frozen_configuration_id):
        return MaterializationResult.failed(
            MaterializationFailure(
                MaterializationFailureKind.BACKEND_CAPABILITY_FAILURE,
                state.campaign_id,
                state.root_checkpoint_id,
                "observation protocol differs from certified materialization",
            )
        )
    if not materialization.current_rebinding.frozen_e0.issubset(
        coverage_state.initial_ids
    ):
        return MaterializationResult.failed(
            MaterializationFailure(
                MaterializationFailureKind.BACKEND_CAPABILITY_FAILURE,
                state.campaign_id,
                state.root_checkpoint_id,
                "campaign CoverageState does not contain checkpoint frozen E_0",
            )
        )
    try:
        _validate_rebinding_state_id(
            materialization.current_rebinding, state.state_id
        )
        for binding in materialization.current_rebinding.live_bindings:
            if materialization.lineage.resolve(binding.physical_entry) != frozenset(
                binding.lineage
            ):
                raise ValueError("materialization lineage differs from re-binding")
    except (LineageResolutionError, TypeError, ValueError) as error:
        return MaterializationResult.failed(
            MaterializationFailure(
                MaterializationFailureKind.POST_RETRIEVAL_LINEAGE_VIOLATION,
                state.campaign_id,
                state.root_checkpoint_id,
                str(error),
            )
        )

    try:
        retrieval_result = await protocol.retrieve(
            state,
            query_artifact=query_artifact,
            top_k=top_k,
        )
    except Exception as error:
        return MaterializationResult.failed(
            MaterializationFailure(
                MaterializationFailureKind.BACKEND_CAPABILITY_FAILURE,
                state.campaign_id,
                state.root_checkpoint_id,
                f"public retrieval raised {type(error).__name__}",
            )
        )
    if retrieval_result.failure is not None:
        return MaterializationResult.failed(
            _normalize_protocol_failure_for_scope(
                state.campaign_id,
                state.root_checkpoint_id,
                retrieval_result.failure,
            )
        )
    retrieval = retrieval_result.value
    if not isinstance(retrieval, RankedPhysicalRetrieval):
        return MaterializationResult.failed(
            MaterializationFailure(
                MaterializationFailureKind.BACKEND_CAPABILITY_FAILURE,
                state.campaign_id,
                state.root_checkpoint_id,
                "protocol returned an invalid ranked retrieval object",
            )
        )
    if (
        retrieval.backend,
        retrieval.campaign_id,
        retrieval.root_checkpoint_id,
        retrieval.state_id,
    ) != (
        state.backend,
        state.campaign_id,
        state.root_checkpoint_id,
        state.state_id,
    ):
        return MaterializationResult.failed(
            MaterializationFailure(
                MaterializationFailureKind.POST_RETRIEVAL_LINEAGE_VIOLATION,
                state.campaign_id,
                state.root_checkpoint_id,
                "ranked retrieval scope differs from certified materialization",
            )
        )
    try:
        observation = resolve_fuzzing_retrieval(
            retrieval, materialization.lineage, coverage_state
        )
    except (LineageResolutionError, TypeError, ValueError) as error:
        return MaterializationResult.failed(
            MaterializationFailure(
                MaterializationFailureKind.POST_RETRIEVAL_LINEAGE_VIOLATION,
                state.campaign_id,
                state.root_checkpoint_id,
                str(error),
            )
        )
    return MaterializationResult.success(
        CertifiedExecutionObservation(seed_id, materialization, observation)
    )


async def materialize_and_observe(
    *,
    seed: LogicalSeed,
    protocol: MaterializationProtocol[HandleT],
    inventory: FrozenCheckpointInventory,
    expected_root_state: DescendantStateCertificate,
    initialization_artifact_id: str,
    coverage_state: CoverageState,
    top_k: int,
) -> MaterializationResult[CertifiedExecutionObservation[HandleT]]:
    """Compose explicit materialization and observation boundaries."""

    materialized = await materialize_seed(
        seed=seed,
        protocol=protocol,
        inventory=inventory,
        expected_root_state=expected_root_state,
        initialization_artifact_id=initialization_artifact_id,
    )
    if materialized.failure is not None:
        return MaterializationResult.failed(materialized.failure)
    materialization = materialized.value
    observed = await observe_materialization(
        seed_id=seed.seed_id,
        protocol=protocol,
        materialization=materialization,
        query_artifact=seed.current_query_artifact,
        coverage_state=coverage_state,
        top_k=top_k,
    )
    if observed.failure is not None:
        try:
            await protocol.discard(materialization.state)
        except Exception:
            pass
    return observed
