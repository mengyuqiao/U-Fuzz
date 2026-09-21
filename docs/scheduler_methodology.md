# Scheduler scientific methodology

This document freezes how an RQ1--RQ4 campaign chooses its next scientific
action. It complements the unified evaluation contract and the existing state,
materialization, retrieval, coverage, and feedback contracts. It does not
define a scheduler implementation, runner, backend worker, model client,
evaluator, or result store.

## Root rounds

The outer scheduling unit is `root_checkpoint_id`. At campaign start, a named
`root_order` random stream creates one persistent permutation of the eligible
root corpus from the later-frozen repetition seed. The event identity contains
the benchmark/root corpus and repetition, but no method name. Compared methods
therefore receive the same logical root order.

One round is one pass over the currently non-exhausted roots. One root visit is
exactly one frontier-item decision. After the method selects that item, only
its opportunity may receive realization attempts during the visit. A terminal
success ends the visit with one valid execution. Terminal generation
exhaustion ends it with zero valid executions. Neither outcome permits a
second frontier selection for that root in the same round; the cycle advances
to the next root. A transport retry of the same realized artifact remains
inside the same visit and creates no new frontier decision. Exceeding the
transport retry cap blocks the campaign as resumable infrastructure failure
rather than advancing as though scientific exhaustion occurred.

Roots with no currently eligible frontier item are skipped. A scientifically
exhausted root leaves future cycles. When the campaign reaches B in the middle
of a round, it stops immediately rather than finishing the round for symmetry.

This method-neutral outer cycle prevents one checkpoint from consuming the
campaign while behavioral novelty remains root-local. Scores are never
compared across roots. There is no additional relation quota or relation
round-robin policy.

## Frontier and opportunity lifecycle

Within the selected root, one frontier item is exactly:

```text
(persistent parent LogicalSeed, applicable MutationOpportunity)
```

It contains no replay-local backend identity. It is eligible only while the
parent is retained, the relation belongs to the method's frozen mutation
space, and the parent/opportunity pair is nonterminal.

Every seed receives one complete scheduler-visible opportunity enumeration at
admission. Initial seeds are enumerated before round 1. A valid child is
enumerated as part of its retain/admit step in the atomic valid-execution
transition, but the child and its complete set remain unschedulable until the
next round.
The certified set is immutable for that `LogicalSeed`. Opportunity existence
depends only on frozen logical seed state, structural index/certification, the
exact mutation-relation mask, and frozen scientific configuration. Retrieval feedback,
priority, evaluator output, queue pressure, worker order, GPU batching, and
wall clock may neither add nor remove an opportunity. Scores control future
order only.

An uncertified enumeration caused by a capability or infrastructure problem
is blocked evidence, not scientific exhaustion, and no partial opportunity set
may be exposed. A certified complete empty set is locally exhausted. A root is
scientifically exhausted only when every retained seed has a certified
complete set and each opportunity is terminal or the set is empty. The
scheduler-policy version owns admission timing, completeness, and immutability;
the `mutation_index_opportunity_enumeration` production binding owns the exact
structural index and enumeration implementation/version.

One realization attempt yields one exact realized mutant. There is no generic
pool of unexecuted or backend-executed candidates available to Coverage-Guided
or U-Fuzz. Their feedback exists only after a valid execution and affects only
later scheduling.

The parent/opportunity lifecycle has two terminal outcomes:

- `terminal_success` after its first valid executed child;
- `terminal_generation_exhausted` after its later-configured realization cap
  is exhausted without a valid mutant.

Unselected opportunities persist across rounds. A terminal pair is never
silently revisited. A transient replay of the same exact artifact belongs to
the same scientific attempt and does not reset this lifecycle.

All compared full-space methods use the same realization interface,
applicability rules, semantic validator, and relation semantics. One method may
choose a different opportunity, but it receives no richer mutation language or
looser validator. Realization randomness is event-keyed by stable scientific
identity and realization-attempt index so worker arrival and batching cannot
change the artifact. The parent and opportunity parts of that event key are
method-neutral logical identities, not campaign-local object IDs; the later
PRNG freeze must specify their exact canonical derivation.

## Retention, chaining, and priority

Every valid executed child that meets the existing persistent
rematerializability requirement becomes a retained `LogicalSeed`, including
zero-score, no-new-coverage, and non-novel children. Scores change exploration
order; they never reject admission.

The logical retained corpus is monotone. A parent remains available until all
members of its certified complete opportunity set are terminal. The scientific
policy has no queue capacity, score threshold, eviction, score aging, or
maximum depth. Physical caching and replay memoization are later engineering
choices and must reproduce the same logical state and certificates.

Children are retained immediately as evidence but cannot enter the schedulable
frontier until the next round. Query and memory descendants may chain, while
each individual edge continues to change exactly query or memory.

Initial benchmark seeds have exact numeric bootstrap priority `0.0` for every
guided method. Coverage priority is exact nonnegative `G_t`; U-Fuzz-family
priority is the frozen score in `[0, 1]`; and online-judge priority is a finite
scalar in `[0, 1]`. A positive child score outranks bootstrap, while a
zero-score child ties bootstrap and uses the event-keyed tie mechanism. There
is no unordered priority sentinel. A child's execution assigns only that
child's future parent priority and never overwrites its parent's score.

Within a selected root:

- Random Mutation chooses uniformly over all eligible frontier items.
- Unguided LLM uses its controller to choose one eligible frontier item.
- Coverage-Guided chooses a frontier item whose parent has maximal stored raw
  coverage-gain priority.
- U-Fuzz-Q, U-Fuzz-M, full U-Fuzz, and the RQ3 ablations choose by the parent's
  corresponding already-frozen U-Fuzz method score.
- LLM-as-Judge chooses by the parent's stored online-judge priority.

Equal parent priorities use the frozen event-keyed tie mechanism. Guided scores
are post-execution observations and cannot rank unexecuted mutants.

## Baseline information boundaries

Random Mutation is uniform over frontier items themselves. It is not
relation-first, parent-first, or score-guided.

Unguided LLM may receive root/query semantic context, the persistent logical
parent recipe, the mutation relation/opportunity, and prior mutation-path
structure needed to understand the action. It cannot receive coverage state or
gain, retrieval signatures, novelty, parent divergence, U-Fuzz score, final
evaluator output, failure surface, CFS, ReferenceVault/gold data, or answer
correctness. After selection it uses the shared realization contract.

LLM-as-Judge is post-execution online feedback. After B is consumed it may see
the exact query and mutation artifacts, ranked public memory
content/projections, resolved retrieval structure/ranks, and necessary
non-evaluator campaign context. It cannot see gold answers, ReferenceVault,
the final evaluator, correctness labels, failure surface, CFS, CoverageState or
gain, retrieval novelty, parent divergence, or the U-Fuzz score. It does not
require final response `y`. It emits one finite numeric scalar in `[0, 1]` for
the retained child. Unparseable, NaN, infinite, or out-of-range output is never
clamped and receives no default; it follows the model-output resampling path.
Exact model, prompt, serialized representation, parser, decoding, and retry cap
remain configuration work. A missing judge score blocks any later decision
that could depend on it.

## Duplicates, retries, and exhaustion

The same parent/opportunity pair cannot execute successfully twice. Replaying
one exact artifact after a transient infrastructure failure remains one
scientific attempt. Distinct legitimate opportunities are not globally
suppressed merely because they realize equal text or semantic content; each
valid execution consumes B.

Retrieval signatures continue to use the frozen root-local history rule,
coverage continues to use E0 set semantics, and CFS deduplication remains a
post-search evaluator operation. None substitutes for scheduler opportunity
deduplication.

A root is scientifically exhausted only when it has no nonterminal applicable
frontier items. A campaign is scientifically exhausted only when every root is
exhausted. If this happens before target B, the campaign stops at achieved B,
records an explicit shortfall, fabricates no probes, repeats no terminal
opportunity, and does not extrapolate UF/Cov. Its target-budget table value is
incomplete until a separate reporting decision handles the observed shortfall.

Transient transport or service failure is not scientific exhaustion. Its retry
replays the exact same scientific event, artifact where applicable, and
stochastic sample. It does not increment a scientific attempt index or draw a
new sample. Exceeding its later-configured cap creates an
infrastructure-blocked resumable campaign state and remains associated with
the selected root visit.

Model-output or realization resampling is scientifically different. An
invalid generated mutation, invalid Unguided action, or invalid online-judge
score may receive another draw only under the configured role-specific cap.
That draw retains the parent scientific event, increments the role-specific
attempt index, and obtains a new event-keyed sample. Controller failure never
falls back to Random. Judge failure after B leaves B consumed and never
receives a default score. Cap exhaustion creates the applicable blocked or
terminal state; transport replay is never counted as resampling.

## Event-keyed randomness

There is no shared mutable global scientific RNG. Named streams include:

- `root_order`;
- `random_baseline_choice`;
- `scheduler_tie_break`;
- `mutation_realization`;
- `unguided_controller_sampling`;
- `online_judge_sampling`;
- `semantic_validation_sampling`;
- `response_generation_sampling`;
- `final_evaluator_cfs_sampling`.

Every stochastic component that may change the search trajectory or final
UF/CFS output is deterministic under frozen configuration or is derived from
the later-frozen master/repetition seed, role stream, stable scientific event
identity, and role-specific attempt index. Resampling uses that attempt index,
not a mutable global retry stream. Worker assignment/completion, worker or
device identity, request arrival, GPU batch order, replica choice, wall clock,
container iteration, Python hash, and runtime backend UUID cannot affect the
logical result. Numeric seeds and the derivation/KDF version remain unbound.

## Atomic valid-execution transition

After successful lineage resolution, the exact logical order is:

1. construct `ResolvedRetrievalObservation`;
2. cross the valid-execution B boundary;
3. increment the campaign valid-execution index;
4. adopt the proposed immutable `CoverageUpdate.state`;
5. compute post-execution feedback, with G from pre-observation coverage, N
   against pre-insertion root history, and MP-query D from the authoritative
   parent signature;
6. insert the child signature into root-local history after scoring;
7. retain the valid child unconditionally;
8. assign its future priority or method-specific scheduling metadata;
9. persist the ordered sealed witness;
10. perform or defer post-B response generation and final evaluation.

For LLM-as-Judge, step 8 requires a successful online judge result before the
child can participate in the next round.

Response generation is:

```text
y = ResponseModel(exact query, exact resolved retrieved content,
                  frozen response configuration)
```

It is post-B and search-inert. It may be deferred or batched; failure does not
refund B and leaves a sealed pending record. The final evaluator follows
response generation, is also search-inert, and may be deferred. UF at a
checkpoint filters sealed records by execution index before campaign-local CFS
deduplication.

For fixed scientific configuration and event identity, deferred response `y`
and final evaluator/CFS output must be invariant to batch order, worker, GPU
replica, device, and request arrival. If a serving topology, precision, or
batching mode cannot preserve that equivalence, it becomes causal scientific
configuration or is ineligible; it cannot remain an engineering-only setting.

RQ2 checkpoint recording is observational only. It may snapshot execution
index, cumulative coverage state/digest, ordered witness boundary, and timing
or attempt counters. It cannot change the root cycle, frontier, retained seeds,
priorities, history, coverage, randomness, or model state.

## Retrieval depth and production configuration

Numeric retrieval depth `k` remains unbound, but production must choose one
`k >= 2` shared across all RQ1--RQ4 methods. The same value controls backend
top-k, initialization and parent/child signatures, `p = 1 - 1/k`, and
`G_tilde = min(1, G/k)`. It is causal scientific configuration and cannot vary
for performance reasons.

Every production campaign must transitively bind identities for:

- benchmark version and preprocessing;
- backend runtime/profile;
- scheduler-policy version;
- PRNG derivation version and repetition seed;
- retrieval `k`;
- mutation realization;
- exact mutation index/opportunity-enumeration implementation and version;
- mutation generation/realization attempt cap;
- transient infrastructure replay cap;
- semantic validation and structural extraction;
- response generation;
- final evaluator and CFS construction.

Unguided LLM additionally binds its controller configuration and its
controller model-output resampling cap. LLM-as-Judge additionally binds its
online-judge configuration and judge model-output resampling cap. These are
explicit search-affecting dimensions. Model/version, prompt digest, decoding,
output/parse schema, and relevant runtime versions are included where
applicable.

Post-B operational transport retry counts need not form separate campaign
identity dimensions when they replay the exact same event and cannot alter
response/evaluator output. Any post-B resampling capable of changing `y` or CFS
must instead be bound transitively by the response-generation or final
evaluator/CFS scientific configuration.

Worker count, GPU model, batch size, serving queue depth, and replica count stay
outside scientific identity only when they preserve exact semantic outcomes.
Their concrete values remain engineering decisions.

## Exact relation-mask authority

Every admitted seed is enumerated only over `MethodSpec.enabled_relations`. Full methods expose six relations, U-Fuzz-Q and U-Fuzz-M expose their exact three-relation masks, and each operator ablation exposes five. Excluded relations are absent from certified enumeration and can never enter the frontier later. The `root-cyclic-retain-all-relation-mask-v2` scheduler and `complete-at-seed-admission-relation-mask-v2` enumeration versions make the exact mask causal provenance. Scores still control order only; they never create or remove opportunities. RQ4 reuses the existing Random, Coverage-Guided, and U-Fuzz policies.
