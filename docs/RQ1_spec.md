# RQ1 Experimental Specification

## Goal and experimental conditions

RQ1 evaluates whether U-Fuzz improves memory exploration and confirmed
memory-use failure discovery under a fixed execution budget.

Memory systems:

- Mem0
- A-Mem
- Graphiti

Benchmarks:

- LoCoMo
- LongMemEval-S

The frozen LongMemEval-S artifact is the authors' cleaned release:

    repository: xiaowu0162/longmemeval-cleaned
    revision: 98d7416c24c778c2fee6e6f3006e7a073259d48f
    filename: longmemeval_s_cleaned.json
    sha256: d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442
    size_bytes: 277383467

The immutable revision, rather than the repository's moving main branch, is
the reproducibility identity for RQ1.

Methods:

- Random Mutation
- Unguided LLM Mutation
- U-Fuzz

Within each backend/benchmark condition, all methods use the same initial
seeds, frozen structural indexes, mutation operators and targets, backend
applicability masks, validators, and valid-execution budget. Only their search
and selection strategies differ.

## Fuzzing campaign and budget

One evaluated fuzzing campaign is:

    one original checkpoint M_i
    x one backend b
    x one method h
    x one repetition seed s

Each campaign receives exactly B valid new mutant executions. Thus, B is a
per-checkpoint campaign budget of valid mutant executions, not a global budget
shared across checkpoints.

For a fixed checkpoint M_i, backend b, and repetition seed s, the backend
checkpoint is constructed once. Random Mutation, Unguided LLM Mutation, and
U-Fuzz preferably start from exact clones of that state. If exact cloning is
unavailable, every method replays the same frozen initialization artifact with
all controllable randomness, model versions, prompts, decoding settings, and
ingestion configuration fixed. Identical replay input alone is not evidence of
equivalent initialized state. Before method-specific fuzzing, perform an
initialization-equivalence check and record a stable state identity,
fingerprint, or canonical state projection sufficient to establish equivalent
starting states. The exact fingerprinting procedure is deferred to the backend
capability-test phase.

If initialization equivalence cannot be established, record an initialization
or capability failure and exclude that condition from the paired RQ1
comparison until it is resolved. Once equivalence is established, the original
benchmark queries and all required parent baseline observations are executed
during shared initialization. Their coverage is denoted:

    C_0^(i,b,s)

C_0 does not have a method index because it is shared across methods. Method-
specific fuzzing begins only after shared initialization.

Budget accounting is:

    valid new mutant execution -> consumes one unit of B
    INVALID generated candidate -> consumes no unit of B
    INAPPLICABLE opportunity -> is not generated and consumes no unit of B

Initialization executions do not consume B. Invalid candidates may be retried
under a generation-attempt cap selected after the pilot and frozen before the
full evaluation.

Every selected parent must already have an observation for its exact state and
query identity. Initial parent observations come from shared initialization.
When an executed mutant is retained, its observation is stored with the seed
and reused if that exact seed later becomes a parent. Normal pair evaluation is:

    cached U(parent)
        +
    execute U(mutant) -> consumes one unit of B

A cached observation may be reused only when exact state and query identity are
proven identical. If identity cannot be proven, reconstruction or parent
execution overhead is recorded separately and the same protocol is applied to
all methods. This overhead does not silently enter or disappear from B.

For N checkpoints, a fixed method/backend/repetition condition therefore uses
N*B valid new mutant executions, plus separately logged initialization and
reconstruction overhead.

## Seed and one-input invariant

A seed is:

    x = (M, q)

Each parent-mutant pair changes exactly one input:

    (M, q) -> (M, q')

or:

    (M, q) -> (M', q)

## Operator set

Query mutations:

- Meaning-preserving
  - paraphrase
  - shorten removable context
  - add nonrestrictive context
  - syntactic-form change
- Target-changing
- Unsupported

Memory-state mutations:

- Update
- Deletion
- Unrelated Change

The following are not part of the current design:

- Addition as a mutation operator
- Conflict as a mutation operator
- poisoning
- MCTS
- gold-guided search

## Checkpoint Structural Index

For each benchmark checkpoint M, construct a frozen Checkpoint Structural
Index from its history and stable benchmark provenance:

    I(M) = (F_M, Q_M, P_M, Z_M)

where:

- F_M contains certified canonical facts;
- Q_M contains certified QueryIntent records;
- P_M is the registry of stable benchmark source units;
- Z_M contains structural certificates used for applicability and validation.

A canonical fact is:

    f = (e, r, v, tau, p)

where e is the entity, r is the attribute or relation, v is the value, tau is
the temporal scope or timestamp when available, and p is stable benchmark
provenance such as a session, turn, or source ID.

Backend-generated IDs are not canonical fact identities. They may be stored
only as adapter-local metadata for resolving backend entries to stable
benchmark provenance.

The structural index contains non-oracle structural information. Access to it
does not by itself constitute evaluator leakage. The preferred flow is:

    Structural Index
        -> Obligation Builder
        -> frozen Omega_r
        -> Scheduler

The scheduler normally operates on the frozen obligation projection rather
than enumerating the full index. This is an architecture choice. Independently,
the scheduler must not use the complete region universe, its size, or a list of
unseen regions as search guidance.

The same frozen index is used by all methods in a backend/checkpoint/
repetition condition.

## Partial-but-certified extraction

The framework does not require complete human annotation of every benchmark
record. Structured fields may be produced through deterministic parsing,
existing benchmark structure, rule-based extraction, or LLM-assisted
extraction. An LLM judgment alone cannot certify a structured field.

Every field used for mutation applicability or validation must have checks
independent of mutant execution. These checks may verify source IDs and text
spans, trace values to stable provenance, check source support and type
compatibility, or establish temporal order from checkpoint history.

If a field cannot be certified, it remains unstructured and corresponding
structured mutation opportunities are inapplicable. Partial extraction is
acceptable; metadata must not be fabricated.

## QueryIntent and unsupportedness

For a query that supports structured mutation, define:

    QueryIntent(q) =
        (entity,
         entity_span,
         attribute,
         attribute_span,
         time_scope,
         constraints,
         answerability)

QueryIntent contains structural request information, not the gold answer.

Meaning-preserving mutations preserve the entity, attribute, time scope,
answerability, and other answer-affecting constraints. A target-changing
mutation changes exactly one designated primary slot from entity, attribute,
or time scope while preserving the remaining primary slots and constraints.

Absence from the partial canonical-fact table is insufficient to establish an
unsupported query. Define a checkpoint-relative absence certificate:

    Absent(z, r, tau, M)

or, when no temporal scope applies:

    Absent(z, r, M)

The certificate must be established before execution by a checkpoint-wide
provenance verification procedure. The verifier inspects the full checkpoint
provenance relevant to the requested entity, relation, and scope, not only
F_M. It may use deterministic matching, structured lookup, and LLM-assisted
analysis, but an LLM assertion alone is insufficient.

Unsupported means unsupported with respect to the current checkpoint, not
false or unknown in the real world.

## ReferenceVault and information boundary

Maintain a separate evaluator-only ReferenceVault:

    G(x) = (a, E+, E-)

where a is the expected answer, E+ is valid supporting evidence, and E- is
evidence that must not support the current answer. Use a = bottom for a
checkpoint-relative unsupported query.

Evaluator-only information includes:

- gold answers;
- G(x) and G(x');
- E+ and E-;
- stale, current, and deleted correctness annotations;
- unsupported expected answer bottom;
- failure and answer predicates;
- confirmed failure verdicts and Fail@B results.

These values must not affect mutation selection, seed retention, scheduling,
search feedback, or search-value computation. The scheduler must not import
or query the ReferenceVault.

Search-side information may include seed identity, QueryIntent fields required
for mutation, operators and concrete targets, applicability, construction
provenance, backend retrieval output, reached canonical regions, online
retrieval divergence, online coverage gain, and retrieval-pattern novelty.

## Mutation obligations

A mutation obligation is:

    omega = (x, o, t)

where x is a current seed, o is an operator, and t is one concrete
structurally certified target. Obligations are generated from the frozen
structural index rather than manually pre-written.

For backend b, define:

    T_{o,b}(x) =
        {t :
            structural_precondition_o(x,t) = 1
            and backend_faithful_b(x,o,t) = 1}

and:

    Omega_{r,b} =
        {(x,o,t) : x in S_r and t in T_{o,b}(x)}

Omega_{r,b} contains only applicable obligations and is fixed within a round.

For target-changing mutation, the replacement must be supported by a
provenance-backed canonical fact for the unchanged requested relation. The
scheduler-facing obligation identifies the slot and replacement target, not
its evaluator gold answer.

For unsupported mutation, applicability requires a checkpoint-relative
absence certificate for the resulting request. Failed retrieval, an incorrect
response, an LLM assertion by itself, or absence from the partial fact table
cannot establish unsupportedness.

For update, the old and new facts preserve canonical entity and relation
identity, use different type-compatible values, and specify a later state or
temporal position. The backend applies its native update semantics; the new
value inherits the target's canonical region key.

For deletion, the fact must exist in the structural index and map to a
faithful public or native backend operation. If the backend cannot delete the
designated fact without changing other relevant information, the opportunity
is inapplicable. Low-level backend internals must not emulate an unsupported
semantic deletion. For Graphiti, episode deletion is usable only when it
faithfully realizes the designated one-change transition.

For Unrelated Change, the designated target must be an existing canonical
region that is structurally unrelated to the current QueryIntent. The
operation may change the value or state within that region. It cannot introduce
a new canonical region.

## Inapplicable and invalid outcomes

INAPPLICABLE means an operator-target pair cannot be faithfully used for the
seed and backend. Examples include an absent type-compatible replacement,
failure to certify checkpoint-relative unsupportedness, non-atomic backend
deletion, uncertified required fields, or an Unrelated Change that would
require a new region.

Inapplicable opportunities are excluded from Omega_{r,b}. They do not count as
failed generation attempts or appear in the applicable-obligation denominator.

INVALID means an applicable obligation was selected and a candidate was
generated, but the candidate failed pre-execution validation. An invalid
candidate leaves its obligation uncovered and may be retried under the frozen
generation-attempt cap. It does not consume B.

Validation is completed before mutant execution. It may use the structural
index, parent and candidate forms, backend operation results, and observable
backend state. It cannot use the mutant retrieval, mutant response,
ReferenceVault, or failure oracle. An LLM judgment alone cannot accept a
candidate.

## Backend applicability mask

For every seed, operator, target, and backend, record a frozen applicability
status and exclusion reason. All three methods receive the same applicability
mask in a backend/checkpoint/repetition condition.

Report total structurally enumerated opportunities, applicable opportunities,
backend-specific exclusions by reason, invalid-generation counts, and valid-
execution counts. Unsupported backend behavior must not be silently emulated.

## Canonical regions and source fallback

For a fact with certified entity and relation, define its semantic key as:

    kappa(f) =
        ("entity-relation", norm(e), norm(r))

Facts with different values or times for the same certified entity-relation
pair belong to the same logical region. The normalization procedure is
selected after the pilot and frozen before full evaluation.

For each stable source unit p, define:

    K_i(p) =
        {kappa(f) :
         f has provenance p
         and has a certified entity-relation key}

and:

    RegionsForSource_i(p) =
        K_i(p)                         if K_i(p) is non-empty
        {("source", p)}                otherwise

A source that yields at least one certified semantic region contributes only
those semantic regions. Residual unstructured text from that source does not
create an additional source fallback. A source fallback is used only when the
source yields no certified entity-relation region. Semantic regions are never
invented for uncertified material.

For original checkpoint M_i, define the frozen campaign universe once:

    U_i := U(M_i)
         = union over checkpoint source units p
           of RegionsForSource_i(p)

Descendant states do not define new region universes. Every descendant state
is evaluated against U_i. Update changes a value within an existing region;
deletion may deactivate its content while the region remains in U_i; and
Unrelated Change modifies an existing unrelated region. No valid memory-state
mutation may create a region outside U_i.

Backend-generated memory, note, node, edge, episode, or other replay-unstable
IDs must not be canonical region keys.

## Retrieval-to-region mapping

For every probe derived from M_i, including a probe on a descendant state, the
adapter resolves backend-local provenance to the original checkpoint sources.
For retrieved entry m, define a certified fact-level mapping:

    FactMap_{b,i}(m) =
        {canonical facts that m can reliably be shown to represent}

Define the semantic region set:

    R_sem_{b,i}(m) =
        {kappa(f) : f in FactMap_{b,i}(m)}

Separately define:

    R_fallback_{b,i}(m) =
        {("source", p) :
         m has reliable provenance to p
         and K_i(p) is empty}

The source fallback is not represented through FactMap. Define:

    rho_{b,i}(m) =
        R_sem_{b,i}(m) union R_fallback_{b,i}(m)

subject to these conservative rules:

- If source p has no certified semantic region, reliable provenance to p may
  map m to ("source", p).
- If p has exactly one certified semantic region, reliable provenance to p is
  sufficient to map m to that region.
- If p has multiple certified semantic regions and m reliably identifies a
  subset of their canonical facts, map only that subset.
- If p has multiple certified semantic regions and m cannot be reliably
  disambiguated to particular facts, mark it ambiguous/unmapped for RegionCov.
  Do not assign every region in K_i(p).

One entry may map to multiple regions when the entry itself reliably
represents multiple certified facts. Mutation-generated values inherit the
region of their designated target. If no reliable checkpoint provenance
exists, record the entry as unmapped rather than inventing a region.

Thus:

    rho_{b,i}(m) subseteq U_i

RegionCov counts regions represented in the retrieved entry, not every region
present in its source unit.

For ranked retrieval R_k, define:

    Regions_i(R_k) =
        union over m in R_k of rho_{b,i}(m)

## Retrieval-mapping quality diagnostic

Because ambiguous and unmapped entries are conservatively excluded from
RegionCov, report:

    MappingRate =
        number of retrieved entries with non-empty rho
        ------------------------------------------------
        total number of retrieved entries considered

Also log the counts of mapped entries, ambiguous entries, and unmapped entries.
Report MappingRate by backend/benchmark condition and optionally by method as a
sanity check.

MappingRate is diagnostic only. It does not guide scheduling, seed retention,
or mutation selection; it does not enter the RegionCov numerator or
denominator; and it is not a primary RQ1 effectiveness metric. Its purpose is
to distinguish low observed region coverage from limited retrieval-to-region
observability.

## Online coverage signal

Let C_{t-1}^online contain only regions observed through shared initialization
and prior valid probes in the campaign. For a probe executed on its actual
state M_{i,t}, let:

    R_t = Regions_i(R_k(q_t, M_{i,t}))

Then:

    R_cov,t =
        |R_t \ C_{t-1}^online|
        -----------------------
        max(1, |R_t|)

    C_t^online = C_{t-1}^online union R_t

This online signal uses only observed regions. It does not use U_i, |U_i|,
gold answers, evaluator evidence, or failure labels.

## Offline RegionCov@B

For fixed checkpoint M_i, backend b, and repetition seed s, let S_0^(i) be
the associated original benchmark query set. Shared initial coverage is:

    C_0^(i,b,s) =
        union over q in S_0^(i)
        of Regions_i(R_k(q, M_i))

These initialization executions do not consume B.

For method h, let its B valid new mutant executions be:

    x_j = (M_{i,j}, q_j),  1 <= j <= B

where M_{i,j} is the actual original or descendant state used for mutant j.
Define:

    C_B^(i,b,h,s) =
        C_0^(i,b,s)
        union
        union over j = 1..B
        of Regions_i(R_k(q_j, M_{i,j}))

Because every retrieval mapping is anchored to U_i:

    C_B^(i,b,h,s) subseteq U_i

The per-campaign metrics are:

    RegionCov_(i,b,h,s)@B =
        |C_B^(i,b,h,s)|
        -----------------
        |U_i|

    DeltaRegionCov_(i,b,h,s)@B =
        |C_B^(i,b,h,s)| - |C_0^(i,b,s)|
        ---------------------------------
        |U_i|

For fixed backend b, method h, and repetition seed s, primary aggregation
across N checkpoint campaigns is:

    RegionCov_(b,h,s)@B =
        (1/N) * sum_i RegionCov_(i,b,h,s)@B

and analogously for DeltaRegionCov@B. A pooled or micro result may be reported
only as a secondary diagnostic.

## Fail@B

Fail@B reports confirmed memory-use failures found within the same B valid new
mutant executions. Evaluation occurs offline through the ReferenceVault and
cannot influence search. The rule for deduplicating distinct failures remains
intentionally unspecified until it is separately approved.

## Manual audit

After the automatic structural-index and mutation pipeline operates, perform a
stratified manual audit across benchmark, mutation operator, and backend when
backend semantics affect applicability. Inspect QueryIntent, entity-relation
extraction, provenance mapping, fact-level retrieval-to-region mapping
accuracy, target-changing validity, unsupportedness, update relations,
deletion applicability, and Unrelated Change validity.

Report the sampling procedure, sample counts, exclusion reasons, and empirical
metadata, mutation validity, and retrieval-mapping accuracy rates. Report
MappingRate and mapped, ambiguous, and unmapped entry counts separately from
the primary effectiveness metrics. The audit is quality control rather than
the primary construction mechanism.

## Information-flow regression test

The implementation must include a regression test that:

1. seals an observable execution trace;
2. permutes evaluator-only gold answers and failure labels;
3. replays scheduling and retention from the same observable retrieval trace;
4. verifies that every scheduling and retention decision is identical.

## Parameters and definitions to fix later

The following experimental parameters are selected after the pilot and frozen
before full evaluation:

- numerical B;
- retrieval depth top-k;
- generation retry cap;
- repetition count;
- extraction model and version;
- entity/relation normalization algorithm;
- manual-audit sample size.

The following methodological definitions will be specified separately before
full evaluation:

- the deduplication rule for distinct failures in Fail@B;
- the executable evaluator predicates C_o and Ans.

No implementation may invent these parameters or definitions implicitly.
