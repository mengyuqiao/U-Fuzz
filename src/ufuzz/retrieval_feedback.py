"""Pure retrieval signatures, RBO feedback, and post-execution scores."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Hashable, TypeVar

from ufuzz.coverage import CoverageEntryId, CoverageUpdate


CoverageToken = tuple[CoverageEntryId, ...]
RetrievalSignature = tuple[CoverageToken, ...]
RankedToken = TypeVar("RankedToken", bound=Hashable)


def canonical_tuple(coverage_ids: Collection[CoverageEntryId]) -> CoverageToken:
    """Return one atomic lineage token in CoverageEntryId's total order."""

    ids = frozenset(coverage_ids)
    if not ids:
        raise ValueError("canonical lineage token must be non-empty")
    if any(not isinstance(item, CoverageEntryId) for item in ids):
        raise TypeError("canonical lineage tokens require CoverageEntryId values")
    scopes = {(item.campaign_id, item.root_checkpoint_id) for item in ids}
    if len(scopes) != 1:
        raise ValueError("canonical lineage token cannot cross campaign/checkpoint scope")
    return tuple(sorted(ids))


def retrieval_signature(
    ranked_lineages: Iterable[Collection[CoverageEntryId]],
) -> RetrievalSignature:
    """Build sigma by first-occurrence deduplication of atomic lineage tuples."""

    signature: list[CoverageToken] = []
    seen: set[CoverageToken] = set()
    scope: tuple[str, str] | None = None
    for lineage in ranked_lineages:
        token = canonical_tuple(lineage)
        token_scope = (token[0].campaign_id, token[0].root_checkpoint_id)
        if scope is None:
            scope = token_scope
        elif token_scope != scope:
            raise ValueError("retrieval signature cannot cross campaign/checkpoint scope")
        if token not in seen:
            signature.append(token)
            seen.add(token)
    return tuple(signature)


def rbo_p_for_depth(k: int) -> float:
    """Return the frozen RBO persistence p=1-1/k; primary k must be at least 2."""

    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("retrieval depth k must be an integer")
    if k < 2:
        raise ValueError("primary retrieval depth k must be at least 2")
    return 1.0 - (1.0 / k)


def _ranked_values(values: Sequence[RankedToken]) -> tuple[RankedToken, ...]:
    ranked = tuple(values)
    try:
        unique = set(ranked)
    except TypeError as error:
        raise TypeError("RBO tokens must be hashable") from error
    if len(unique) != len(ranked):
        raise ValueError("RBO ranked lists must not contain duplicate tokens")
    return ranked


def _validate_p(p: float) -> float:
    value = float(p)
    if not 0.0 < value < 1.0:
        raise ValueError("RBO persistence p must lie strictly between 0 and 1")
    return value


def rbo_ext(
    first: Sequence[RankedToken],
    second: Sequence[RankedToken],
    p: float,
) -> float:
    """Compute standard extrapolated RBO for finite, untied ranked lists.

    This implements Webber, Moffat, and Zobel (2010), Equation 32 for uneven
    lists, which reduces to Equation 23 for equal lengths. Each sequence item
    is an atomic token; tuple-valued merge tokens are never flattened.
    """

    persistence = _validate_p(p)
    left = _ranked_values(first)
    right = _ranked_values(second)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0

    if len(left) <= len(right):
        short, long = left, right
    else:
        short, long = right, left
    short_length = len(short)
    long_length = len(long)

    short_prefix: set[RankedToken] = set()
    long_prefix: set[RankedToken] = set()
    overlaps = [0] * (long_length + 1)
    for depth in range(1, long_length + 1):
        if depth <= short_length:
            short_prefix.add(short[depth - 1])
        long_prefix.add(long[depth - 1])
        overlaps[depth] = len(short_prefix.intersection(long_prefix))

    observed = sum(
        (overlaps[depth] / depth) * (persistence**depth)
        for depth in range(1, long_length + 1)
    )
    overlap_at_short = overlaps[short_length]
    uneven_extrapolation = sum(
        (
            overlap_at_short
            * (depth - short_length)
            / (short_length * depth)
        )
        * (persistence**depth)
        for depth in range(short_length + 1, long_length + 1)
    )
    overlap_at_long = overlaps[long_length]
    tail_agreement = (
        ((overlap_at_long - overlap_at_short) / long_length)
        + (overlap_at_short / short_length)
    ) * (persistence**long_length)
    score = (
        ((1.0 - persistence) / persistence)
        * (observed + uneven_extrapolation)
        + tail_agreement
    )
    return min(1.0, max(0.0, score))


def rbo_distance(
    first: Sequence[RankedToken],
    second: Sequence[RankedToken],
    p: float,
) -> float:
    """Return frozen RBO distance, including explicit empty-list semantics."""

    similarity = rbo_ext(first, second, p)
    return min(1.0, max(0.0, 1.0 - similarity))


@dataclass(frozen=True, slots=True)
class NoveltyResult:
    """Checkpoint-local novelty plus the empty-history diagnostic."""

    score: float
    prior_history_empty: bool


class BehaviorHistory:
    """Root-checkpoint-local archive of observed retrieval signatures."""

    def __init__(
        self,
        campaign_id: str,
        initialization_signatures: Mapping[
            str, Iterable[RetrievalSignature]
        ] | None = None,
    ) -> None:
        if not campaign_id:
            raise ValueError("campaign_id must be non-empty")
        self.campaign_id = campaign_id
        self._history: dict[str, set[RetrievalSignature]] = {}
        for root_checkpoint_id, signatures in (initialization_signatures or {}).items():
            self.seed(root_checkpoint_id, signatures)

    def seed(
        self,
        root_checkpoint_id: str,
        signatures: Iterable[RetrievalSignature],
    ) -> None:
        """Seed H_i from cached initialization observations only."""

        self._validate_root(root_checkpoint_id)
        archive = self._history.setdefault(root_checkpoint_id, set())
        for signature in signatures:
            self._validate_signature_root(root_checkpoint_id, signature)
            archive.add(signature)

    def signatures(self, root_checkpoint_id: str) -> frozenset[RetrievalSignature]:
        """Return a read-only snapshot of one checkpoint's archive."""

        self._validate_root(root_checkpoint_id)
        return frozenset(self._history.get(root_checkpoint_id, set()))

    def is_new_behavior(
        self,
        root_checkpoint_id: str,
        signature: RetrievalSignature,
    ) -> bool:
        """Evaluate 1[sigma not in H_i] without changing the archive."""

        self._validate_signature_root(root_checkpoint_id, signature)
        return signature not in self.signatures(root_checkpoint_id)

    def novelty(
        self,
        root_checkpoint_id: str,
        signature: RetrievalSignature,
        p: float,
    ) -> NoveltyResult:
        """Return nearest-history RBO distance for one root checkpoint."""

        self._validate_signature_root(root_checkpoint_id, signature)
        archive = self.signatures(root_checkpoint_id)
        if not archive:
            return NoveltyResult(score=1.0, prior_history_empty=True)
        return NoveltyResult(
            score=min(rbo_distance(signature, prior, p) for prior in archive),
            prior_history_empty=False,
        )

    def observe(
        self,
        root_checkpoint_id: str,
        signature: RetrievalSignature,
    ) -> None:
        """Insert sigma after it has been scored, independent of retention."""

        self._validate_root(root_checkpoint_id)
        self._validate_signature_root(root_checkpoint_id, signature)
        self._history.setdefault(root_checkpoint_id, set()).add(signature)

    @staticmethod
    def _validate_root(root_checkpoint_id: str) -> None:
        if not root_checkpoint_id:
            raise ValueError("root_checkpoint_id must be non-empty")

    def _validate_signature_root(
        self,
        root_checkpoint_id: str,
        signature: RetrievalSignature,
    ) -> None:
        if any(
            coverage_id.campaign_id != self.campaign_id
            for token in signature
            for coverage_id in token
        ):
            raise ValueError("retrieval signature belongs to another campaign")
        if any(
            coverage_id.root_checkpoint_id != root_checkpoint_id
            for token in signature
            for coverage_id in token
        ):
            raise ValueError("retrieval signature belongs to another root checkpoint")


def parent_child_divergence(
    parent_signature: RetrievalSignature,
    child_signature: RetrievalSignature,
    p: float,
) -> float:
    """Return pure exact-parent retrieval divergence."""

    return rbo_distance(parent_signature, child_signature, p)


class MutationRelation(StrEnum):
    """Canonical semantic mutation-relation vocabulary for all later phases.

    Mutation and scheduler modules must reuse these six relations rather than
    defining another enum or string vocabulary. Surface operator subtypes do
    not belong here.
    """

    MEANING_PRESERVING_QUERY = "meaning_preserving_query"
    TARGET_CHANGING_QUERY = "target_changing_query"
    UNSUPPORTED_QUERY = "unsupported_query"
    UPDATE = "update"
    DELETION = "deletion"
    UNRELATED_CHANGE = "unrelated_change"


class MissingExactParentSignatureError(ValueError):
    """Raised when meaning-preserving feedback lacks its exact cached parent."""


def _validate_gain(raw_gain: int) -> int:
    if isinstance(raw_gain, bool) or not isinstance(raw_gain, int):
        raise TypeError("raw coverage gain must be an integer")
    if raw_gain < 0:
        raise ValueError("raw coverage gain must be non-negative")
    return raw_gain


def _validate_unit_score(name: str, score: float) -> float:
    value = float(score)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return value


def coverage_guided_score(raw_gain: int) -> int:
    """Return the raw post-execution Coverage-Guided score S_CG=G_t."""

    return _validate_gain(raw_gain)


def bounded_coverage_gain(raw_gain: int, k: int) -> float:
    """Return U-Fuzz-only clipped gain min(1, G_t/k)."""

    gain = _validate_gain(raw_gain)
    rbo_p_for_depth(k)
    return min(1.0, gain / k)


def behavior_score(
    novelty: float,
    relation: MutationRelation,
    divergence: float | None = None,
) -> float:
    """Return S_beh with divergence only for Meaning-Preserving Query."""

    if not isinstance(relation, MutationRelation):
        raise TypeError("relation must be a MutationRelation")
    novelty_value = _validate_unit_score("novelty", novelty)
    if relation is MutationRelation.MEANING_PRESERVING_QUERY:
        if divergence is None:
            raise MissingExactParentSignatureError(
                "meaning-preserving feedback requires exact-parent divergence"
            )
        divergence_value = _validate_unit_score("divergence", divergence)
        return (novelty_value + divergence_value) / 2.0
    if divergence is not None:
        raise ValueError("divergence is unavailable for this mutation relation")
    return novelty_value


def ufuzz_score(bounded_gain: float, retrieval_behavior_score: float) -> float:
    """Return fixed equal-weight U-Fuzz score."""

    coverage_value = _validate_unit_score("bounded coverage gain", bounded_gain)
    behavior_value = _validate_unit_score(
        "retrieval behavior score", retrieval_behavior_score
    )
    return (coverage_value + behavior_value) / 2.0


@dataclass(frozen=True, slots=True)
class RetrievalFeedback:
    """Search-side feedback for one already executed valid mutant."""

    root_checkpoint_id: str
    mutation_relation: MutationRelation
    signature: RetrievalSignature
    new_behavior: bool
    novelty: float
    prior_history_empty: bool
    divergence: float | None
    raw_coverage_gain: int
    bounded_coverage_gain: float
    coverage_guided_score: int
    behavior_score: float
    ufuzz_score: float


def score_observed_retrieval(
    *,
    root_checkpoint_id: str,
    observed_signature: RetrievalSignature,
    coverage_update: CoverageUpdate,
    retrieval_depth: int,
    mutation_relation: MutationRelation,
    history: BehaviorHistory,
    exact_parent_signature: RetrievalSignature | None = None,
) -> RetrievalFeedback:
    """Score one observed retrieval after its valid execution consumed budget.

    The function scores against H_i before inserting the observed signature.
    CoverageState.observe remains authoritative for E_0 validation and G_t.
    The later campaign layer must ensure that ``observed_signature`` and
    ``coverage_update`` came from the same completed backend execution; this
    primitive deliberately does not implement that orchestration. It performs
    no candidate ranking, retention, queueing, or evaluator work.
    """

    if not root_checkpoint_id:
        raise ValueError("root_checkpoint_id must be non-empty")
    if not isinstance(mutation_relation, MutationRelation):
        raise TypeError("mutation_relation must be a MutationRelation")
    if not isinstance(coverage_update, CoverageUpdate):
        raise TypeError("coverage_update must be a CoverageUpdate")
    persistence = rbo_p_for_depth(retrieval_depth)
    new_behavior = history.is_new_behavior(root_checkpoint_id, observed_signature)
    novelty_result = history.novelty(
        root_checkpoint_id,
        observed_signature,
        persistence,
    )

    divergence: float | None = None
    if mutation_relation is MutationRelation.MEANING_PRESERVING_QUERY:
        if exact_parent_signature is None:
            raise MissingExactParentSignatureError(
                "exact cached parent signature is required"
            )
        divergence = parent_child_divergence(
            exact_parent_signature,
            observed_signature,
            persistence,
        )
    elif exact_parent_signature is not None:
        raise ValueError(
            "exact parent signature must not supply divergence for this relation"
        )

    raw_gain = coverage_guided_score(coverage_update.raw_gain)
    bounded_gain = bounded_coverage_gain(raw_gain, retrieval_depth)
    retrieval_score = behavior_score(
        novelty_result.score,
        mutation_relation,
        divergence,
    )
    final_score = ufuzz_score(bounded_gain, retrieval_score)
    history.observe(root_checkpoint_id, observed_signature)
    return RetrievalFeedback(
        root_checkpoint_id=root_checkpoint_id,
        mutation_relation=mutation_relation,
        signature=observed_signature,
        new_behavior=new_behavior,
        novelty=novelty_result.score,
        prior_history_empty=novelty_result.prior_history_empty,
        divergence=divergence,
        raw_coverage_gain=raw_gain,
        bounded_coverage_gain=bounded_gain,
        coverage_guided_score=raw_gain,
        behavior_score=retrieval_score,
        ufuzz_score=final_score,
    )
