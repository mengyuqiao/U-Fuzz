# Unified RQ1-RQ4 evaluation contract

This document and `src/ufuzz/evaluation_contract.py` are authoritative for the scientific campaign topology. A campaign is one raw adaptive trajectory. Its scientific key contains benchmark, backend, causal method key, repetition index, maximum budget, the optional RQ4 provider-family condition, and normalized production configuration bindings. The causal method key includes the exact enabled-relation configuration. Display names, fairness annotations, RQ membership, table labels, observational checkpoints, and engineering settings are excluded. A view points to a campaign and checkpoint; equal scientific configurations reuse the same canonical campaign object rather than copying results or rerunning.

Configuration bindings are causal identity. Binding order is canonicalized, changing a configuration ID changes the campaign key, and duplicate dimensions are rejected. `EVALUATION_PLAN` is a configuration-unbound planning topology; empty bindings do not establish production readiness.

## Shared rules

Repetitions are exactly 0, 1, and 2. Reported values use the arithmetic mean and sample standard deviation (`ddof=1`) over those three raw values; values and standard deviations are neither interpolated, smoothed, nor rescaled.

One unit of B is consumed only after semantic validation, exact certified materialization and transition where applicable, selected public retrieval, complete returned-object lineage resolution, and construction of a valid `ResolvedRetrievalObservation`. Initialization, materialization/replay work, invalid generation, inapplicability, uncertifiable transition, semantic drift, retrieval failure, and unresolved returned lineage consume zero B under the frozen failure taxonomy. Response generation and final evaluation occur after this boundary. Evaluator failure never refunds B.

`UF@B` is the number of distinct evaluator-confirmed canonical fault signatures among executions 1 through B, deduplicated within the campaign rather than summed across subsets. A canonical fault signature contains `root_checkpoint_id`, `mutation_relation`, `canonical_query_intent`, `canonical_mutation_target`, and `failure_surface`. `Cov@B` is cumulative reached frozen E0 lineage divided by the campaign's frozen E0 cardinality. Deleted E0 entries remain in the denominator, and replacement, merge, and split lineage follow the frozen backend contracts. Coverage is backend-entry-granularity specific and is primarily compared within one backend.

The primary backends for RQ1-RQ3 are Mem0, A-Mem, Graphiti, and MemOS. MemOS is planning identity only until its native capability, materialization, replay, lineage, isolation, and cleanup proofs are complete.

Methods carry an exact immutable relation configuration. Full methods enable all six frozen `MutationRelation` values. U-Fuzz-Q enables exactly the three query relations; U-Fuzz-M enables exactly the three memory relations. A leave-one-operator-out method has broad space FULL and exactly five relations. The exact relation set is causal campaign identity.

## RQ1: fixed-budget effectiveness

RQ1 uses LoCoMo at B=8000 and LongMemEval-S at B=4000, four backends, three repetitions, and exactly seven methods:

1. Random Mutation
2. Unguided LLM
3. LLM-as-Judge
4. Coverage-Guided
5. U-Fuzz-Q
6. U-Fuzz-M
7. U-Fuzz

Random Mutation, Unguided LLM, LLM-as-Judge, Coverage-Guided, and U-Fuzz use the same full six-relation mutation space and shared realization, applicability, certification, materialization, B, reader, and evaluator contracts. U-Fuzz-Q and U-Fuzz-M are explicitly restricted-space variants.

Each benchmark contributes `7 × 4 × 3 = 84` campaigns. RQ1 therefore has 168 campaigns and `84×8000 + 84×4000 = 1,008,000` planned valid executions.

## RQ2: budget scaling

RQ2 uses LoCoMo, all four backends, U-Fuzz-Q, U-Fuzz-M, and U-Fuzz at checkpoints 1000 through 8000 in steps of 1000. Every point is an observational prefix of the corresponding RQ1 LoCoMo Bmax=8000 trajectory. Checkpoint recording cannot reset or alter history, coverage, frontier, retention, priority, randomness, model state, or selection. RQ2 adds zero campaigns and zero planned executions. Per-repetition UF and Cov prefix series must be monotone.

## RQ3: component ablation

RQ3 asks how removing one mutation operator or one feedback component affects
U-Fuzz relative to Full U-Fuzz. It uses LoCoMo at displayed B=4000, all four
backends, and repetitions 0, 1, and 2. Its ten conceptual settings are one
Full U-Fuzz reference and nine one-component-at-a-time ablations.

For every backend and repetition, the Full U-Fuzz reference is the B=4000
observational prefix of the exact canonical RQ1 LoCoMo Full U-Fuzz Bmax=8000
campaign. Full therefore adds no raw campaign. U-Fuzz-Q, U-Fuzz-M,
Coverage-Guided, and Random Mutation remain RQ1 methods but are not RQ3 rows.

Six new operator ablations retain full U-Fuzz feedback and remove exactly one relation from the original six-relation universe:

- U-Fuzz w/o Meaning-Preserving Query
- U-Fuzz w/o Target-Changing Query
- U-Fuzz w/o Unsupported Query
- U-Fuzz w/o Update
- U-Fuzz w/o Deletion
- U-Fuzz w/o Unrelated Change

The removed relation produces no opportunities. There is no relation quota, probability renormalization, or compensation for the smaller opportunity set.

Three new feedback ablations retain all six relations:

- U-Fuzz w/o Coverage
- U-Fuzz w/o Novelty
- U-Fuzz w/o Parent Divergence

A disabled feedback component is set to zero without renormalizing the original combination weights. The frozen equations remain:

- full: `S=(G_tilde+S_beh)/2`
- no coverage: `S=S_beh/2`
- no novelty: MP query `S=(G_tilde+D_t/2)/2`; otherwise `S=G_tilde/2`
- no parent divergence: MP query `S=(G_tilde+N_t/2)/2`; otherwise full non-MP scoring

Only the nine new ablation configurations create raw RQ3 campaigns: `9×4×3=108` at B=4000, or 432,000 planned valid executions. Scheduler-mechanics ablations are outside the primary RQ3 plan.

RQ3 reports derived paired deltas at B=4000. For backend `b`, ablation `a`,
and matched repetition `r`:

    DeltaUF[b,a,r] = UF@4000[b,a,r] - UF@4000[b,Full-U-Fuzz,r]
    DeltaCov[b,a,r] = Cov@4000[b,a,r] - Cov@4000[b,Full-U-Fuzz,r]

The subtraction is performed within repetitions 0, 1, and 2 first. The table
then reports the arithmetic mean and sample standard deviation (`ddof=1`) of
those three paired deltas. It must not subtract aggregate means and synthesize
an unpaired error bar. Positive values mean the ablation outperformed Full on
that metric; zero means no observed change; negative values mean removing the
component reduced the metric. DeltaCov is stored as a difference between raw
normalized fractions. A presentation such as `-7.0 pp` is permitted only as
an exact, clearly labeled rendering of `-0.07`.

The main-paper RQ3 table shows Mem0 only. Its nine rows are grouped into the
six mutation ablations and three feedback ablations, with `DeltaUF@4K` and
`DeltaCov@4K` columns containing paired-delta mean ± sample standard
deviation. Full need not appear as a literal zero row because it defines the
reference. Its absolute UF@4000 and Cov@4000 may be reported elsewhere for
scale, but they are not additional RQ3 metrics. A-Mem, Graphiti, and MemOS
report the same nine rows and paired
metrics in the appendix, separately by backend; results are never averaged
across systems. Main-paper versus appendix placement is reporting metadata and
does not affect campaign identity, execution, seeds, budgets, or metrics.

RQ1-RQ3 therefore contain 276 unique campaigns and 1,440,000 planned valid executions.

## RQ4: API-memory-LLM portability on LoCoMo

RQ4 asks whether U-Fuzz remains effective when the memory system's native LLM-dependent processing uses different proprietary API model families. Its primary scope is LoCoMo only, B=2000, Mem0 and Graphiti, repetitions 0-2, and three methods: Random Mutation, Coverage-Guided, and U-Fuzz. These provide no-feedback, simple-coverage-feedback, and full-feedback comparisons.

The planning provider families are OpenAI, Anthropic, and Google. Exact API model IDs and revisions remain production-unbound. The provider condition changes only the memory system's native LLM processing, such as memory/update inference or graph/entity/relation/temporal extraction. It does not choose the mutation generator, scheduler, validator, common response reader, evaluator, embeddings, storage, search, reranker, retrieval k, or repetition procedure.

RQ4 campaign identity includes a typed native-memory-LLM condition. Production identity must additionally bind the exact provider/model revision, memory-system prompt/configuration, structured-output schema, parser, decoding, retry/resampling behavior, and causal API/runtime version. Credentials never enter campaign identity, manifests, logs, evidence, or witnesses.

Within one memory system, all three provider conditions must share one invariant backend substrate: source/runtime, embeddings and revision, vector dimension and normalization, storage, search, reranker/cross-encoder, replay/isolation profile, and retrieval k. A provider integration that cannot preserve this control is ineligible.

RQ4 reuses UF@B and Cov@B and retains E0 cardinality and retrievable-entry profile identity for interpretation. A condition without complete E0 and lineage is ineligible. RQ4 has `2×3×3×3=54` campaigns and 108,000 planned valid executions. These campaigns are distinct from RQ1 because the native-memory-LLM configuration is causal.

## Counts and engineering targets

The complete planning topology is:

- local RQ1-RQ3: 276 campaigns, 1,440,000 valid executions;
- RQ4 API suite: 54 campaigns, 108,000 valid executions;
- combined: 330 campaigns, 1,548,000 valid executions.

The six-local-GPU, 24-hour engineering target applies only to RQ1-RQ3. Its execution-only floor is `1,440,000/86,400 = 16.666666...` valid executions/s, with higher attempt and post-B capacity required in practice. RQ4 has a separate, still-unbound API SLA because quotas, rate limits, latency, and cost are external constraints.

## Raw-campaign legality

Construction fails closed on the complete scientific tuple. The legal raw
families are RQ1 LoCoMo original-method campaigns at 8000, RQ1 LongMemEval-S
original-method campaigns at 4000, new RQ3 ablation LoCoMo campaigns at 4000,
and RQ4 LoCoMo Mem0/Graphiti campaigns at 2000 with one typed provider
condition. In particular, a reused RQ3 reference is never a separate 4000-run
campaign: its view checkpoint is 4000 on the canonical RQ1 LoCoMo 8000
trajectory. RQ membership remains excluded from campaign identity because
legality is derivable from benchmark, backend, causal method, budget, and the
optional native-memory-LLM condition.

## Final evaluator and production binding

The final evaluator is post-B and search-inert. Its references, verdict, failure surface, and CFS inputs are unavailable to search. LLM-as-Judge uses a separate online, gold-blind priority signal and cannot access final-evaluator-only records. Response generation and final evaluation may be deferred if execution ordering is preserved, every valid observation is evaluated consistently, and UF at a checkpoint uses only execution indices at or below that checkpoint.

`EVALUATION_PLAN` is the canonical planning topology. Its empty configuration bindings do not authorize production. A later readiness freeze must bind exact benchmark manifests, native backend profiles, scheduler/opportunity-enumeration version, PRNG, k, realization and validation, response reader, evaluator/CFS, retry policy, and method-specific roles. RQ4 additionally requires the exact native-memory-LLM configuration and invariant-substrate/embedding-control proof.
