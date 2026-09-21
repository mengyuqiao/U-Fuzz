# Unified evaluation contract for RQ1, RQ2, and RQ3

This contract separates a **scientific campaign** from a **research-question
view**. A scientific campaign is identified by its benchmark, backend, causal
method key, repetition index, maximum valid-execution budget, and the frozen
model/backend/runtime configuration identities that must be supplied before a
production run. The causal method key contains the stable method ID, selection
family, mutation space, online search signal, feedback-component mask, and
feedback-combination rule. Presentation fields such as method display name,
fairness-group annotation, RQ membership, table-row label, and observational
checkpoint are excluded. Changing presentation metadata therefore does not
create a new raw campaign, while changing causal method semantics does.

Configuration bindings remain causal scientific identity. Their ordering is
canonicalized, but changing a bound configuration ID changes the campaign key,
and duplicate binding dimensions are invalid. The exported `EVALUATION_PLAN`
is the canonical **planning topology** for RQ1-RQ3: it fixes cardinality, reuse,
budgets, methods, backends, repetitions, and views with empty configuration
bindings. It is not sufficient authorization for production. A later explicit
pre-run freeze must supply and approve every required configuration binding;
merely observing some nonempty bindings is not a production-readiness test.

RQ membership is view metadata. If two RQ entries have the same scientific
campaign configuration, they refer to one raw campaign and one trajectory;
they are never rerun for each RQ. RQ2 checkpoints are also view metadata: the
maximum budget remains campaign identity, while each checkpoint selects a
prefix of that one trajectory.

The contract plans campaigns only. It does not implement scheduling, mutation
generation, materialization, evaluator execution, logging, or plotting. RQ4 is
pending a separate freeze and contributes no campaign, budget, provider, or
metric here.

## RQ1: fixed-budget effectiveness

RQ1 asks how effectively U-Fuzz discovers memory-use failures under a fixed
valid-execution budget. It uses LoCoMo and LongMemEval-S, the Mem0, A-Mem, and
Graphiti backends, and repetition indices 0, 1, and 2. LoCoMo campaigns have
`B_max = 8000`; LongMemEval-S campaigns have `B_max = 4000`.

The seven displayed methods and stable identifiers are:

| Method | Identifier | Mutation space |
|---|---|---|
| Random Mutation | `random-mutation` | Full |
| Unguided LLM | `unguided-llm` | Full |
| LLM-as-Judge | `llm-as-judge` | Full |
| Coverage-Guided | `coverage-guided` | Full |
| U-Fuzz-Q | `ufuzz-q` | Query only |
| U-Fuzz-M | `ufuzz-m` | Memory only |
| U-Fuzz | `ufuzz` | Full |

Random Mutation, Unguided LLM, LLM-as-Judge, Coverage-Guided, and U-Fuzz form
the same-full-space fairness group. They share the complete eligible initial
seed corpus, all six mutation relations, applicability and semantic validation,
materialization and valid-execution rules, candidate-generation limits, final
evaluator, backend, and benchmark. They differ only in their defined
exploration or prioritization logic. U-Fuzz-Q and U-Fuzz-M are restricted-space
variants and must never be described as using the same mutation space as the
five full-space methods.

Query-only permits Meaning-Preserving Query, Target-Changing Query, and
Unsupported Query. Memory-only permits Update, Deletion, and Unrelated Change.
Full permits all six existing `MutationRelation` values.

Random Mutation selects applicable opportunities uniformly and receives no
retrieval feedback. Unguided LLM uses an LLM for exploration without retrieval
feedback. LLM-as-Judge uses a distinct online judge to rank valid candidates by
estimated fault-exposure likelihood, without gold answers or final-evaluator
references. Coverage-Guided uses only newly covered memory-entry feedback.
U-Fuzz variants use the frozen U-Fuzz retrieval feedback.

The primary metrics are `UF@B` and `Cov@B`. `UF@B` is the number of distinct,
evaluator-confirmed canonical fault signatures discovered in the first `B`
valid executions of one campaign, deduplicated at campaign level. It is not a
sum of per-subset counts. `Cov@B` is the cumulative fraction of frozen campaign
E₀ lineages reached during those executions. Deleted E₀ remains in the
denominator, and certified replacement/merge/split lineage follows the frozen
backend contracts. Coverage granularity is backend-specific and is primarily
compared within a backend.

The valid-execution definition is not redefined here. The authoritative rule
remains creation of a valid `ResolvedRetrievalObservation` after semantic,
materialization, transition, retrieval, and returned-lineage certification.
The final evaluator runs afterward and cannot refund budget. Initialization,
invalid attempts, failures before resolved observation, evaluator work, and
free materialization work are outside the valid-execution count as already
specified by the frozen methodology.

Every reported value uses the same three raw repetition values. Aggregation is
the arithmetic mean and sample standard deviation with `ddof = 1`. Experimental
data are not interpolated and standard deviations are not smoothed or scaled.

## RQ2: budget scaling from RQ1 trajectories

RQ2 uses only LoCoMo, the same three backends, U-Fuzz-Q, U-Fuzz-M, and U-Fuzz,
and checkpoints 1000 through 8000 in increments of 1000. It adds zero campaigns.
For each backend, method, and repetition, RQ2 reads cumulative snapshots from
the one LoCoMo `B_max = 8000` trajectory already executed for RQ1. Checkpoints
are not independent runs and do not restart the campaign.

For each raw repetition, both cumulative series must be monotone:

    UF@1000 <= UF@2000 <= ... <= UF@8000
    Cov@1000 <= Cov@2000 <= ... <= Cov@8000

A decrease is a logger or aggregation error. The RQ2 endpoint at 8000 and the
corresponding RQ1 entry are the same campaign object and underlying raw value,
not copied or independently generated values.

Checkpoint observation is side-effect free. Recording a checkpoint cannot
reset history or coverage, reseed randomness, alter queue or retention state,
change model state, or influence the next mutation decision.

## RQ3: full-space U-Fuzz ablations

RQ3 uses LoCoMo at `B = 8000`, all three backends, and repetitions 0, 1, and 2.
Its displayed rows are Coverage-Guided (a reused reference), U-Fuzz w/o
Coverage, U-Fuzz w/o Novelty, U-Fuzz w/o Parent Divergence, and full U-Fuzz.
Every true U-Fuzz ablation uses the full mutation space.

Let `G_tilde = min(1, G_t/k)`, `N_t` be behavioral novelty, and `D_t` be
parent-child retrieval divergence, available only for Meaning-Preserving Query.
The frozen full score is:

    Meaning-Preserving Query: S_beh = (N_t + D_t) / 2
    Other relations:          S_beh = N_t
    Full U-Fuzz:              S = (G_tilde + S_beh) / 2

An ablation sets exactly one component to zero and **does not renormalize** the
remaining components or weights:

- **w/o Coverage:** `G_tilde = 0`; retain `S_beh`; therefore `S = S_beh / 2`.
- **w/o Novelty:** `N_t = 0`. For Meaning-Preserving Query,
  `S_beh = D_t/2` and `S = (G_tilde + D_t/2)/2`. For every other relation,
  `S_beh = 0` and `S = G_tilde/2`.
- **w/o Parent Divergence:** `D_t = 0`. For Meaning-Preserving Query,
  `S_beh = N_t/2` and `S = (G_tilde + N_t/2)/2`. For other relations,
  `S_beh = N_t` and `S = (G_tilde + N_t)/2`.
- **Full U-Fuzz:** unchanged.

RQ3 reuses the RQ1 LoCoMo Coverage-Guided and full U-Fuzz campaigns. Full
U-Fuzz is also the same raw campaign used by RQ2. Only the three component
ablations add campaigns.

## Deduplicated campaign and budget totals

LoCoMo has seven RQ1 method configurations plus three new RQ3 ablations:

    10 methods x 3 backends x 3 repetitions = 90 campaigns
    90 x 8000 = 720000 planned valid executions

LongMemEval-S has only the seven RQ1 configurations:

    7 methods x 3 backends x 3 repetitions = 63 campaigns
    63 x 4000 = 252000 planned valid executions

The unified plan therefore contains exactly **153 unique campaigns** and
**972000 planned valid executions**. The execution total excludes invalid
attempts, initialization and materialization overhead, evaluator calls,
nonvalid retries, and RQ4.

## Final evaluator and adaptive search

The final fault evaluator runs after the valid-execution boundary. Its verdict,
failure surface, references, and canonical-CFS inputs are unavailable to all
search methods and ablations. LLM-as-Judge has a separate online judge signal;
that signal is not the final evaluator and cannot access evaluator-only data.
The final evaluator may later be deferred or batched if execution indices are
preserved, every valid observation is evaluated consistently, and each
`UF@checkpoint` uses only executions at or before that checkpoint.

Independent campaigns may run in parallel using CPU/backend workers,
asynchronous queues, batched LLM requests, several GPU inference replicas,
deferred evaluator batches, and backend-specific concurrency caps. Within one
adaptive campaign, execution `t`, required search feedback and state updates,
and the decision for execution `t+1` must retain their semantic order.
Speculative intra-campaign execution that changes that decision sequence is
forbidden. Cross-campaign parallelism is the primary scaling mechanism.

## Engineering target and future instrumentation

Six strong GPUs are available, and completing RQ1-RQ3 within 24 wall-clock
hours is an engineering target rather than a scientific parameter. It is not
part of campaign identity, method definitions, metrics, or reported values.
Using the whole 24 hours for execution alone implies:

    972000 / 86400 = 11.25 valid executions per second

Required attempt throughput is higher when the valid-attempt rate is below
one. This is a planning target, not a guarantee.

The future local logger must expose campaign ID, benchmark, backend, method,
repetition, attempts, valid executions, valid-attempt rate, total wall time,
backend failures by frozen failure kind, and time spent in mutation generation,
semantic validation, materialization, transition certification, retrieval,
lineage resolution, search feedback, the online judge where applicable, and
the final evaluator. It must also record LLM request and input/output-token
counts plus batching and queue metrics where available. No external telemetry
service is allowed.

Before production, a throughput pilot must estimate valid executions and
attempts per second, valid-attempt rate, per-stage latency, GPU utilization and
inference throughput, backend bottlenecks, and projected full-suite wall time.
Pilot size remains unfrozen.

## Items still requiring a pre-run freeze

- exact generator model and version;
- exact LLM-as-Judge model and version;
- exact final evaluator model and version;
- prompt templates;
- retrieval depth `k`;
- candidate-generation limit;
- retry semantics beyond the frozen budget rule;
- scheduler queue, retention, chaining, capacity, and eviction policies;
- tie-breaking;
- campaign exhaustion and budget-shortfall handling;
- exact PRNG seed derivation from repetition indices;
- batching configuration;
- worker count and GPU serving topology;
- backend-specific concurrency caps;
- final RQ4 design.

These items are not implied by the planning contract and must not be filled in
silently by the runner.
