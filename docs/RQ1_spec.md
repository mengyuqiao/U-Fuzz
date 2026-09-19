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
- LLM-as-Judge
- Coverage-Guided
- U-Fuzz

These five primary methods use the same benchmark seed corpus, eligible
checkpoint set, frozen structural indexes, full mutation operators and targets,
backend applicability masks, validators, generation-attempt limits, eligibility
for valid executed mutants to become later seeds, and valid-execution budget.
Only their search and selection strategies differ. U-Fuzz-Q and U-Fuzz-M are
restricted ablations: U-Fuzz-Q uses only query mutations, U-Fuzz-M uses only
memory-state mutations, and full U-Fuzz uses both classes. The restricted
ablations are not part of the same-full-space fairness comparison.

## Fuzzing campaign and budget

One evaluated fuzzing campaign is:

    one benchmark corpus d
    x one backend b
    x one method h
    x one repetition seed s

Each campaign starts from the complete eligible benchmark seed corpus and
receives exactly B valid new mutant executions across that corpus. B is a total
per-campaign budget, not a budget allocated independently to each checkpoint.

For a fixed benchmark, backend, and repetition seed, every method initializes
all eligible checkpoint states under the same frozen initialization protocol.
Exact state cloning is preferred. If cloning is unavailable, every method
replays the same frozen initialization artifacts with all controllable
randomness, model versions, prompts, decoding settings, and ingestion
configuration fixed. Identical replay input alone is not evidence of equivalent
initialized state. Before method-specific fuzzing, compare the observable
multiset of retrievable-entry projections for every checkpoint state. The exact
backend-specific projection is finalized during capability testing.

If initialization equivalence cannot be established, record an initialization
or capability failure and exclude that condition from the paired RQ1
comparison until it is resolved. Equivalent states need not share backend UUIDs
or global coverage-entry IDs. Each method campaign assigns its own opaque IDs
to its own initialized retrievable entries. Once equivalence is established,
the original benchmark queries and all required parent baseline observations
are executed during initialization. Method-specific fuzzing begins afterward.

Budget accounting is:

    valid new mutant execution -> consumes one unit of B
    INVALID generated candidate -> consumes no unit of B
    INAPPLICABLE opportunity -> is not generated and consumes no unit of B

Initialization executions do not consume B. Invalid candidates may be retried
under a generation-attempt cap selected after the pilot and frozen before the
full evaluation. Initialization retrievals do not contribute to Cov@B and do
not populate the Coverage-Guided seen-entry set.

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

A campaign therefore uses B valid new mutant executions in total, plus
separately logged initialization and reconstruction overhead.

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
than enumerating the full index. This is an architecture choice.

The same frozen index for each eligible checkpoint is used by all methods in a
backend/benchmark/repetition condition.

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
- confirmed failure verdicts and UF@B results.

These values must not affect mutation selection, seed retention, scheduling,
search feedback, or search-value computation. The scheduler must not import
or query the ReferenceVault.

Search-side information may include seed identity, QueryIntent fields required
for mutation, operators and concrete targets, applicability, construction
provenance, backend retrieval output, campaign-local coverage-entry IDs, online
entry-lineage certificates, ranked retrieval order, cached parent retrieval
signatures, checkpoint-local signature histories, observed new-entry coverage,
retrieval-pattern novelty, and meaning-preserving parent-child divergence.

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
status and exclusion reason. The five primary methods receive the same full-
space applicability mask in a backend/benchmark/repetition condition. U-Fuzz-Q
and U-Fuzz-M receive the corresponding query-only and memory-only restrictions.

Report total structurally enumerated opportunities, applicable opportunities,
backend-specific exclusions by reason, invalid-generation counts, and valid-
execution counts. Unsupported backend behavior must not be silently emulated.

## Search methods and restricted ablations

Random Mutation uniformly selects among applicable obligations. Unguided LLM
uses an LLM to choose among the same applicable obligations and realize the
chosen mutation without retrieval feedback; it may not leave the shared
applicable space. LLM-as-Judge ranks valid candidates by their estimated
likelihood of exposing a memory-use fault, without gold answers, evaluator
references, failure labels, oracle verdicts, or retrieval feedback. Its
candidate-pool size and judging cost are fixed after the pilot.

Coverage-Guided uses only the memory-entry coverage signal defined below.
U-Fuzz uses observed new-memory-entry coverage together with its approved
retrieval-pattern novelty and, for meaning-preserving mutations, ranked
retrieval divergence. U-Fuzz-Q and U-Fuzz-M apply the same U-Fuzz strategy to
the query-only and memory-only operator subsets, respectively.

## Canonical structural regions and provenance mapping

For a fact with certified entity and relation, define its structural key as:

    kappa(f) =
        ("entity-relation", norm(e), norm(r))

Facts with different values or times for the same certified entity-relation
pair belong to the same structural region. Backend-generated memory, note,
node, edge, episode, or other replay-unstable IDs are not canonical structural
keys. The normalization procedure is selected after the pilot and frozen
before full evaluation.

For each stable source unit p, define:

    K_i(p) =
        {kappa(f) :
         f has provenance p
         and has a certified entity-relation key}

and use ("source", p) as a conservative fallback only when K_i(p) is empty.
Residual unstructured text does not create an additional fallback for a source
that already has a certified semantic key.

For a retrieved entry m, define:

    FactMap_{b,i}(m) =
        {canonical facts that m can reliably be shown to represent}

    R_sem_{b,i}(m) =
        {kappa(f) : f in FactMap_{b,i}(m)}

    R_fallback_{b,i}(m) =
        {("source", p) :
         m has reliable provenance to p
         and K_i(p) is empty}

    rho_{b,i}(m) =
        R_sem_{b,i}(m) union R_fallback_{b,i}(m)

For a source with one semantic region, reliable provenance may establish that
mapping. For a source with multiple semantic regions, map only the facts that
the entry reliably represents; otherwise mark the semantic mapping ambiguous
or unmapped. Never assign every region merely from source provenance.

Canonical regions, FactMap, semantic/fallback mapping, and MappingQuality are
structural infrastructure for mutation construction and validation,
unrelatedness checks, deletion certification, provenance auditing, and backend
capability analysis. They do not define, qualify, or enter Cov@B. MappingRate
may be retained as an internal provenance-mapping diagnostic, but it is not
required to interpret Cov@B and cannot guide search.

## Initial retrievable-entry inventory and identity

For each method campaign, initialize all eligible checkpoint states and first
establish entry-level equivalence against the shared initialization condition.
Equivalence compares the observable multiset of retrievable-entry projections,
not backend UUIDs. Each method then freezes its own initialized retrievable-
entry set E_0 and assigns opaque campaign-local coverage IDs.

Coverage entries are checkpoint-local. E_0 is the disjoint union of the
initialized retrievable entries from all eligible checkpoint states. Similar
or identical content in different checkpoints remains distinct. A coverage
entry must have the same backend object granularity that the selected public
retrieval API can return: memory records for Mem0, memory notes for A-Mem, and
fact/edge objects for Graphiti. Storage enumeration alone does not establish
retrieval eligibility. If the backend cannot establish E_0 at this granularity,
Cov@B for that backend/benchmark condition is a capability blocker.
If the complete verified campaign-level inventory is empty, |E_0| = 0, the
backend x benchmark condition is RQ1-ineligible. Cov@B is undefined, UF@B is
not reported, and no numeric 0 or 1 is fabricated. Record the condition as a
capability/ineligibility result. An individual checkpoint with no retrievable
entry does not exclude a campaign whose campaign-level E_0 is non-empty.

An in-place update retains the target entry's coverage ID. A replacement or
later physical record created for an update inherits the designated target's
coverage ID. Deletion does not remove the target from E_0 and does not itself
count as retrieval. Unrelated Change remains associated with its selected
initialized target. No valid memory-state mutation creates a coverage ID
outside E_0. If a particular split, merge, or replacement cannot be certified
against its initialized target, that mutation opportunity is INAPPLICABLE; it
does not automatically exclude the complete backend/benchmark condition.

## Cov@B

For a campaign identified by benchmark d, backend b, method h, and repetition
seed s, initialize the reached-entry set as:

    C_0 = empty set

For valid fuzzing execution t, let R_t be its ranked retrieval result and let
CoverageIDs(R_t) contain the initialized coverage IDs represented by its
retrieved physical entries. Define:

    C_t = C_{t-1} union CoverageIDs(R_t)

After B valid fuzzing executions:

    Cov@B = |C_B| / |E_0|

Original benchmark-query executions and parent-observation construction do not
consume B, contribute to C_0, or populate any online coverage seen set. Thus
Cov@B is in [0,1]. Because backends expose different entry granularities,
absolute Cov@B values are interpreted primarily within the same backend.

## Coverage-guided and U-Fuzz coverage feedback

For valid fuzzing execution t, the selected public retrieval API returns the
ranked list:

    R_t = (m_{t,1}, ..., m_{t,l_t}), where l_t <= k

and k is the frozen retrieval depth. For each retrieved physical object m, let:

    Gamma_t(m) subseteq E_0

be the non-empty, certified set of initialized coverage entries represented by
m. Gamma_t(m) is normally a singleton. A faithfully certified merge may bind m
to multiple initialized entries. Define the deterministic canonical token:

    tau_t(m) = canonical_tuple(Gamma_t(m))

where canonical_tuple orders campaign-local coverage-entry IDs using one fixed
deterministic canonical ordering. Singleton and multi-entry bindings use the
same tuple type: {e_1} becomes (e_1), and either {e_2,e_1} or {e_1,e_2}
becomes (e_1,e_2) under that ordering. If Gamma_t(m) cannot be established
faithfully, the implementation must report an unresolved lineage or capability
condition. It must not silently drop m, invent a token, or fall back to FactMap
or canonical structural regions.

The ranked retrieval behavior signature is:

    sigma_t =
        dedup_first(
            tau_t(m_{t,1}),
            ...,
            tau_t(m_{t,l_t}))

where dedup_first retains only the first ranked occurrence of each identical
canonical token. The empty retrieval has the valid signature sigma_t = <>.
Thus sigma_t records which initialized entries were returned and their ranked
order; it contains no raw retrieved text, embedding, backend score, backend
UUID, canonical region, FactMap value, or evaluator information.

Each complete tuple tau_t(m) is one atomic ranked token for sigma_t and RBO. A
merged physical object is not expanded into artificial ranks for its members.
For example, if m_1 represents {e_1,e_2} and m_2 represents {e_3}, then:

    sigma_t = [(e_1,e_2), (e_3)]

not [e_1,e_2,e_3]. By contrast, CoverageIDs(R_t) unions the underlying entries
as {e_1,e_2,e_3} for G_t, C_t, and Cov@B. Consequently [(e_1,e_2),(e_3)] and
[(e_1),(e_2),(e_3)] are distinct behaviors even though they represent the same
coverage-entry set. dedup_first removes only identical complete tuple tokens.

### Checkpoint-local behavior history and exact new behavior

For every original root checkpoint i, maintain a separate behavior archive
H_i^(t). Do not compare signatures across checkpoints. Initialize it from the
cached original parent or baseline retrieval observations for that checkpoint:

    H_i^(0) =
        {canonical retrieval signatures observed during initialization
         for root checkpoint i}

These initialization signatures seed behavioral history only. Coverage still
starts with C_0 = empty, Coverage-Guided still starts with seen = empty, and
initialization receives no Cov@B or Coverage-Guided credit.

For a valid execution t descended from checkpoint i, score sigma_t against
H_i^(t-1) and then update:

    H_i^(t) = H_i^(t-1) union {sigma_t}

regardless of whether the executed mutant is retained. Archives for other root
checkpoints are unchanged. Define the exact new-behavior indicator:

    NewBehavior_t = 1[sigma_t not in H_i^(t-1)]

NewBehavior_t is logged for analysis and is not a fourth independent scheduler
objective.

### RBO distance, novelty, and parent-child divergence

Let RBO_EXT,p(S,T) be standard extrapolated rank-biased overlap for finite
ranked lists, and define:

    d_RBO(S,T) = 1 - RBO_EXT,p(S,T)

so 0 <= d_RBO(S,T) <= 1. Larger values denote more different ranked
retrieval behavior.

Once k is frozen, set:

    p = 1 - 1/k

and require k >= 2 in the primary experiment. The persistence p is not tuned
on UF@B or Cov@B. Freeze the empty-list cases as:

    d_RBO(<>, <>) = 0
    d_RBO(<>, T) = d_RBO(T, <>) = 1, for non-empty T

For execution t from root checkpoint i, retrieval-pattern novelty is distance
to the closest previously observed behavior for that checkpoint:

    N_t = min_{sigma in H_i^(t-1)} d_RBO(sigma_t, sigma)

equivalently:

    N_t =
        1 - max_{sigma in H_i^(t-1)} RBO_EXT,p(sigma_t, sigma)

If H_i^(t-1) is empty despite initialization, define N_t = 1 and log the
absence of a prior checkpoint-local signature.

Parent-child divergence is defined only for a validated Meaning-Preserving
Query mutation. Let sigma_parent(t) be the signature stored in the exact cached
parent observation. Then:

    D_t = d_RBO(sigma_parent(t), sigma_t)
        = 1 - RBO_EXT,p(sigma_parent(t), sigma_t)

Do not compute D_t from a different parent, an unverified reconstruction, or a
merely similar query. If exact parent identity is unavailable, D_t is not
validly available and no substitute is used. D_t is not a primary signal for
Target-Changing Query, Unsupported Query, Update, Deletion, or Unrelated
Change.

N_t and D_t measure distinct phenomena. N_t measures global checkpoint-local
novelty against all prior signatures, whereas D_t measures local instability
relative to the exact parent. For example, if a parent retrieves [A,B,C,D] and
its child retrieves [X,Y,Z,W], but [X,Y,Z,W] was observed earlier, N_t may be
approximately 0 while D_t remains approximately 1.

### Coverage gain and search priorities

All retrieval-derived quantities and priorities below are computed only after
a valid mutant has executed and its actual R_t has been observed. The search
chronology is:

    select eligible parent and mutation opportunity
        -> generate mutant
        -> validate mutant
        -> execute valid mutant
        -> observe R_t
        -> compute G_t, N_t, and D_t when applicable
        -> compute S_CG,t or S_UF,t
        -> use the score for retention or future expansion priority

The valid execution consumes one unit of B before its feedback score becomes
available. S_CG,t and S_UF,t prioritize executed mutants as possible future
seeds or parents; neither can rank an unexecuted candidate whose retrieval has
not been observed. Cached exact-parent observations supply sigma_parent(t) for
D_t but do not change this chronology.

For execution t, define:

    A_t = CoverageIDs(R_t)
    G_t = |A_t \ C_{t-1}|
    C_t = C_{t-1} union A_t

CoverageIDs(R_t) contains every initialized entry certified through the
Gamma_t bindings. Therefore a certified multi-entry retrieval can make G_t
larger than the physical retrieval depth k. Cov@B and C_t retain all such
entries.

Coverage-Guided uses only the raw new-entry count:

    S_CG,t = G_t

and updates its initially empty seen set with A_t. It does not clip G_t and
cannot use N_t, D_t, NewBehavior_t, canonical-region coverage, evaluator
references, gold answers, failure labels, or oracle verdicts.

Only for the bounded U-Fuzz priority component, normalize coverage gain as:

    G_t_tilde = min(1, G_t / k)
              = min(G_t, k) / k

In the usual singleton-lineage case G_t <= k, this reduces to G_t / k. A
certified merge can make G_t > k; clipping keeps the scheduler component in
[0,1] without clipping G_t, A_t, C_t, Cov@B, or the Coverage-Guided score. It
introduces no parameter and uses neither |E_0| nor |A_t| as a denominator.

Define the bounded retrieval-behavior score:

    S_beh,t = N_t
        for every mutation other than Meaning-Preserving Query

    S_beh,t = (N_t + D_t) / 2
        for a validated Meaning-Preserving Query mutation

and the final U-Fuzz priority:

    S_UF,t = (G_t_tilde + S_beh,t) / 2

Both S_beh,t and S_UF,t lie in [0,1]. Meaning-Preserving Query mutations split
the fixed behavior-feedback mass equally between novelty and local divergence,
rather than receiving an extra objective. With singleton lineage and G_t <= k,
the effective weights are 0.50 coverage, 0.25 novelty, and 0.25 divergence for
Meaning-Preserving Query, and 0.50 coverage plus 0.50 novelty for other
relations. When a certified merge makes G_t > k, the coverage component
saturates at 1.

These weights are fixed a priori and are not optimized using pilot outcomes,
UF@B, Cov@B, gold answers, or evaluator labels. Strict lexicographic coverage-
first priority is not used because novelty and divergence would then affect
search only on coverage ties, making U-Fuzz too similar to Coverage-Guided.
Pareto selection is not used because it does not define a unique next action
without another rule such as crowding distance, hypervolume, or a second
ranking procedure. No tuned alpha, beta, or gamma weights are introduced.

Coverage-Guided and U-Fuzz resolve exact score ties using the same deterministic
principle controlled by the repetition seed. Queue capacity, eviction,
batching, and other scheduler mechanics remain to be fixed during scheduler
implementation.

All terms above are search-side observables. They may use root checkpoint
identity, campaign-local coverage IDs, certified physical-entry lineage,
ranked retrieval order, exact cached parent signatures, and checkpoint-local
signature histories. They cannot use gold answers, E+ or E-, ReferenceVault
content, CFS failure surfaces, R/A/RA labels, fault verdicts, UF@B, evaluator
correctness predicates, response correctness, or generated-answer behavior.
CanonicalFact, structural regions, FactMap, semantic/fallback provenance
mapping, and MappingQuality remain available for structural and auditing uses
listed above, but they do not define sigma_t, N_t, D_t, G_t, S_beh,t, S_UF,t,
or Cov@B.

## UF@B

We operationalize fault uniqueness using canonical semantic fault signatures
rather than attempting to infer latent implementation root causes. For every
evaluator-confirmed failing execution, define:

    CFS =
        (root_checkpoint_id,
         mutation_relation,
         canonical_query_intent,
         canonical_mutation_target,
         failure_surface)

Two confirmed failures are the same unique fault if and only if their complete
CFS values are identical. For one campaign:

    UF@B =
        |{CFS(x_t, x'_t) :
            1 <= t <= B
            and execution t is evaluator-confirmed as a fault}|

The root checkpoint ID is the original benchmark checkpoint from which the
seed lineage descends. It is not a descendant state ID, replay state ID,
backend UUID, or execution number.

The canonical mutation relations are Meaning-Preserving Query,
Target-Changing Query, Unsupported Query, Update, Deletion, and Unrelated
Change. Paraphrase, shortening, nonrestrictive-context addition, and
syntactic-form change all normalize to Meaning-Preserving Query.

The canonical query intent is the normalized and certified QueryIntent of the
mutant query: entity, relation or attribute, temporal scope, answer-affecting
constraints, and checkpoint-relative answerability. Raw query wording is not
part of the signature.

The canonical mutation target is operator-specific:

- Meaning-Preserving Query: EMPTY.
- Target-Changing Query: designated changed slot plus canonical replacement.
- Unsupported Query: designated changed slot or requested unsupported target,
  represented canonically.
- Update: canonical entity-relation target, including temporal scope when
  structurally relevant. The replacement value is excluded.
- Deletion: canonical semantic deletion target, including entity, relation or
  attribute, value, and temporally relevant scope. Stable provenance, duplicate
  source copies, and backend physical IDs do not distinguish the UF signature
  when they represent the same semantic deletion target.
- Unrelated Change: canonical unrelated entity-relation target, including
  temporal scope when structurally relevant. The replacement value is excluded.

The mutually exclusive failure surfaces are R for retrieval-side failure only,
A for answer/use-side failure only, and RA for both. Failure-surface
classification is evaluator-only and must not affect mutation selection,
scheduling, retention, or search feedback. The executable C_o, retrieval-side,
and Ans predicates remain unresolved until the evaluator specification is
finalized.

The primary CFS excludes raw query or paraphrase wording, exact response text,
backend or retrieval-entry UUIDs, exact retrieval-rank perturbations, concrete
update or unrelated-change replacement values, random seeds, execution indices,
scheduling paths, descendant state IDs, repetition IDs, and method names. These
values may be retained only in raw witness records. Every CFS retains all of its
confirmed failing executions as witnesses, while UF@B counts that CFS once.

Deduplication occurs independently within each campaign and repetition.
Primary reporting aggregates the resulting per-run UF@B values; it does not
union CFS values across repetitions. A cross-repetition CFS union may be
reported only as a secondary corpus-discovery diagnostic. Thus, “unique fault”
means unique under this declared CFS equivalence relation.

The CFS field schema, operator-specific target semantics, excluded fields,
equality rule over canonicalized CFS values, per-repetition deduplication, and
witness retention are fixed. Executable construction of canonical CFS values
depends on the separately frozen entity/relation/time/constraint normalization
procedure and evaluator predicates, which must be fixed before RQ1 execution.

## Reporting across repetitions

Primary RQ1 results remain separated by benchmark. For each benchmark x
backend x method cell, aggregate repeated campaign results over repetition
seeds. Do not pool LoCoMo and LongMemEval-S into one primary result. The number
of repetitions and uncertainty statistic are selected after the pilot and
frozen before full evaluation.

## Manual audit

After the automatic structural-index and mutation pipeline operates, perform a
stratified manual audit across benchmark, mutation operator, and backend when
backend semantics affect applicability. Inspect QueryIntent, entity-relation
extraction, provenance mapping, fact-level retrieval-to-region mapping
accuracy, target-changing validity, unsupportedness, update relations,
deletion applicability, and Unrelated Change validity.

Report the sampling procedure, sample counts, exclusion reasons, and empirical
metadata, mutation validity, and retrieval-mapping accuracy rates. Report
MappingRate and mapped, ambiguous, and unmapped fact/provenance-mapping counts
only as internal quality diagnostics. They neither define nor qualify Cov@B.
The audit is quality control rather than the primary construction mechanism.

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
- uncertainty statistic;
- LLM-as-Judge candidate-pool size and judging cost;
- extraction model and version;
- entity/relation/time/constraint canonicalization procedure;
- manual-audit sample size.

The following methodological definitions will be specified separately before
full evaluation:

- the executable evaluator predicates C_o and Ans, including retrieval-side
  and answer/use-side correctness and the resulting R/A/RA assignment;
- backend-specific retrievable-entry projection implementations;
- concrete scheduler queue capacity, eviction, and batching mechanics;
- backend capability issues not yet live-validated.

No implementation may invent these parameters or definitions implicitly.
