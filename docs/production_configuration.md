# Production configuration and pilot protocol

This document freezes the configuration principles and blinded decision
protocols that sit between the RQ1--RQ4 evaluation plan and a production run.
It does not select models, prompts, retrieval depth, retry caps, or pilot data,
and it does not authorize a campaign. The evaluation and scheduler contracts
remain authoritative for campaign identity, valid-execution accounting, and
trajectory-producing behavior.

## Native backend fidelity

Primary RQ1--RQ3 results use each backend's intended native reportable
production profile. Mem0, A-Mem, Graphiti, and MemOS need not use identical internal
embeddings, extraction, graph logic, or storage components. The experiment is
about fuzzing the actual memory systems, not synthetic implementations made
component-identical across backends.

Fairness is enforced within a backend: all methods use the same native profile,
initialization, retrieval depth, common response reader, evaluator, and causal
scientific configuration for that backend.

Deterministic or synthetic profiles may be used for capability,
materialization, replay, and reproducibility validation. They must be labeled
as validation tracks and cannot silently replace the native primary profile.

### Current backend status

- **Mem0 Profile A** is the validated deterministic validation profile. It uses
  `infer=False`, deterministic mock embeddings, a noncalling LLM, isolated
  local Qdrant, and the already-certified deterministic entity-state behavior.
  It is not automatically the primary paper profile.
- **Mem0 Profile B** is the intended native `infer=True` primary family. Its
  exact LLM, embedding, extraction, reranking, and search configuration remain
  unbound. It needs live proof of extraction multiplicity, initialization and
  inventory equivalence, provenance and E0 rebinding, auxiliary entity state,
  update/delete behavior, replacement/merge/split evidence where applicable,
  exact lineage, and replay certifiability.
- **A-Mem** requires a native primary profile, dedicated-process isolation, and
  a materializer with replay, inventory, lineage, transition, and cleanup proof.
- **Graphiti** requires a native primary profile, isolated database/search
  configuration, protection from shared mutable search recipes, and a
  materializer with replay, inventory, graph-transition, lineage, and cleanup
  proof.
- **MemOS** uses the frozen profile **MemOS v2.0.33 using its
  GeneralTextMemory textual-memory backend**, source commit
  `78a372a4fc853a24d2a78efa3b4bbbd27ab9f7ad`. It directly constructs
  deterministic `TextualMemoryItem` entries and uses native
  `add/search/get_all/update/delete` with isolated embedded Qdrant, cosine
  retrieval, and a guarded 768-dimensional local `nomic-embed-text` manifest.
  The profile does not use Tree memory, MemReader/native extraction, native
  feedback interpretation, MemScheduler, native chat inference, conflict or
  merge inference, or agentic/deep retrieval. Merge and split are inapplicable;
  updates preserve physical ID and initialized lineage.

None of the intended native profiles is production-ready merely because a
generic backend contract exists.

## Common response reader

The primary analysis uses one common response-reader scientific configuration
across every method and all four RQ1--RQ3 backends. It receives the exact executable
query, the exact selected and resolved public retrieval content, and one frozen
serialization/prompt configuration. Backend identity changes the retrieved
content but does not select a different reader.

This isolates memory retrieval and use from reader-model variation. A future
backend-native-reader analysis may be reported only as a secondary analysis and
cannot replace the common-reader primary result. The exact common reader is
selected by the registered configuration pilot.

## Model roles and two-tier allocation

The scientific roles are mutation generator, Unguided controller, online
judge, optional model-assisted semantic validator, common response reader, and
final evaluator/CFS. Each role has its own identifier, prompt/config artifact,
information boundary, parser/output schema, and stochastic stream when needed.

Roles may share base weights or a physical serving process. Such sharing is an
engineering choice only when it produces the same scientific outputs as
isolated serving and does not leak role context or evaluator/gold data.

The default allocation principle is two-tier:

- online mutation generation, controller selection, online judging, and
  optional lightweight model assistance should use a fast common local model
  family when it meets the quality contract;
- offline response reading and final evaluation may use stronger
  configurations when required.

This does not require different weights between tiers and does not select any
model. Online roles cannot be starved by deferred response/evaluator backlogs.

## Structural certification and evaluator

Structural and semantic certification is deterministic first. It uses
benchmark evidence, normalized structural evidence, certified provenance, and
backend-state evidence wherever those can decide the obligation. A model may
propose an interpretation where deterministic extraction is insufficient, but
the model's assertion cannot be the sole certificate of the mutation it or
another model generated. LLM-only self-certification is inadmissible.

The final evaluator is hybrid. Deterministic rules handle exact relation,
intent, target, reference, retrieval-lineage/content, normalized-answer, and
temporal decisions. A semantic model handles only cases that cannot be decided
exactly. Every valid execution is processed. An exact router may finalize an
exact verdict or send every uncertain case to the full evaluator; it cannot
drop a potential fault for speed. Evaluation is post-B and search-inert.

Production remains blocked until the evaluator/CFS implementation specifies:

- relation-specific correctness `C_o`;
- retrieval and answer correctness;
- R, A, and RA failure-surface classification;
- QueryIntent and mutation-target canonicalization;
- entity, relation, temporal, and answer-constraint normalization;
- open-answer equivalence;
- ReferenceVault construction; and
- deterministic CFS serialization and deduplication.

## Decoding policy family

Mutation generation and the Unguided controller prefer seeded stochastic
decoding to preserve reproducible diversity and exploration. The online judge,
common response reader, final evaluator/CFS, and model-assisted validator use
deterministic/greedy decoding as their default pilot candidate. Exact
temperature, top-p, stop, and other decoding values remain unbound until the
pilot verifies quality and repeat consistency.

## PRNG derivation

Scientific randomness uses HMAC-SHA-256 under version `UFUZZ_PRNG_V1` and the
public domain separator `ufuzz/evaluation/prng/v1`. Typed values use versioned,
length-prefixed UTF-8/binary fields with fixed big-endian lengths and unsigned
integers. Python `hash`, `repr`, unspecified JSON ordering, and native-endian
serialization are forbidden.

For each benchmark and repetition 0, 1, or 2, the repetition key is derived
from the frozen benchmark-manifest SHA-256 digest and repetition index. It does
not contain method identity. Role/event seeds are then:

```text
HMAC-SHA-256(
    repetition_key,
    versioned_frame(stream_name,
                    stable_scientific_event_identity,
                    role_attempt_index)
)
```

This preserves method-neutral root ordering and shared realization sampling
where the scheduler contract requires it. Changing the benchmark manifest
changes campaign identity and the derived randomness. The later implementation
must freeze output-to-integer/float/choice mapping with test vectors.

## Prompt and model configuration artifacts

Every model role stores a canonical artifact plus its SHA-256 digest. The
artifact contains the role, provider/runtime family, exact model and revision
or weights digest, tokenizer and chat-template revisions, exact system and user
template bytes, ordered few-shot examples, serialization format and field
order, output schema, parser version, decoding parameters, maximum output
tokens, stop rules, and precision/quantization when it can change output.

Labels and digests without the underlying artifact are insufficient.

## Pilot/final-data firewall

Configuration-development items and final RQ1--RQ3 roots must be disjoint and
recorded in a `CONFIGURATION_DEV_MANIFEST` before final outputs are opened. No
development items are selected in this phase because the repository does not
currently expose a clearly declared disjoint development split.

Pilots may inspect parser/schema success, invalid-output rate, blinded semantic
or structural agreement, support-evidence truncation, replay and repeat
consistency, latency, token count, GPU memory, throughput, backend correctness,
and infrastructure reliability. They cannot inspect UF@B, Cov@B, method fault
advantages, or U-Fuzz-versus-baseline gaps. Configurations and their artifacts
are frozen before final results are opened.

## Registered scientific configuration pilots

### Retrieval depth

The candidates are exactly `k in {3, 5, 10, 20}`. The pilot uses the intended
native profiles and configuration-development data, and measures reference
support within top-k where references permit, truncation, payload/context size,
latency, and empty/truncated behavior. One global k is selected for RQ1--RQ3.
The rule is to prefer the smallest k that preserves adequate support evidence
across all four intended native profiles. No arbitrary adequacy threshold is invented
here. If blinded measurements do not determine a value unambiguously, they
return for explicit approval without UF/Cov visibility.

### Mutation generator

Candidate capability classes are fast local 7B/8B, local approximately 14B,
and a larger class only if necessary. Measure schema validity, semantic
validity, certification pass rate, redundant realization, tokens, p50/p95
latency, batched throughput, and event-keyed replay consistency. Prefer the
smallest and fastest model satisfying the frozen validity/certification rules.

### Unguided controller

The controller must choose from every eligible frontier item. Measure item
count, serialized token count, and growth with depth and B, then first attempt
a complete lossless serialization. A hidden top-N shortlist is forbidden. If
the complete frontier does not fit, stop and separately freeze a lossless
paged or hierarchical method that covers every item. This phase defines no
fallback.

### Online judge

Use the same candidate capability classes as the generator. Measure finite
`[0,1]` parse success, repeat consistency, blinded qualitative relevance of
the priority ordering, tokens, latency, and batched throughput. Select the
smallest stable model without downstream UF/Cov.

### Common response reader

Use one frozen input/prompt format across methods and backends. Compare the
same three capability classes using normalized reference-answer agreement,
malformed/nonanswer rate, repeat consistency, context truncation, tokens,
latency, and throughput. Select the smallest reader that does not make reader
weakness dominate the memory-system measurement; method-specific fault counts
are unavailable to this decision.

### Evaluator/CFS

Implement the exact predicates and canonicalization before selecting a model.
Then compare deterministic branches, hybrid deterministic-plus-semantic
evaluation, and stronger fallback classes where needed using blinded manual
agreement, repeat and parser consistency, canonicalization consistency,
router false-route rate, latency, and throughput. The final evaluator is
identical across methods and is never selected from resulting UF counts.

### Retry caps

No numeric cap is set here. Measure mutation schema/semantic rejection by
relation, transient failure/recovery by backend, Unguided invalid/unparseable
selection, and online-judge invalid/nonfinite/out-of-range output. The mutation
cap is shared by methods using the same realization contract. Controller and
judge caps are role-specific. Infrastructure caps may vary by backend only
when reliability evidence supports that choice, and remain identical across
methods for that backend.

## Backend and throughput dependency

Native profiles and materializers are validated before final retrieval-k and
model pilots. MemOS GeneralText capability and materialization are frozen;
Mem0 Profile B and the A-Mem and Graphiti production materializers remain to
be validated, followed by benchmark-scale isolation and throughput proof.

The local RQ1--RQ3 plan has 1,440,000 valid executions and, without a proven
exact evaluator router, the same number of response and evaluator records. The
six-local-GPU 24-hour target implies at least 16.6666666667 valid executions
per second plus enough post-B capacity. The separate RQ4 API load has 108,000
of each record and no frozen local 24-hour SLA. The combined planning count is
1,548,000, but it has no single shared SLA.

After exact models and profiles are frozen, the throughput pilot measures
role calls and tokens per second, batch fill, queue wait, GPU utilization and
VRAM, p50/p95 latency, backend latency, replay depth/cost, invalid-attempt rate,
and response/evaluator backlog drain. Production does not launch when the
projected makespan materially exceeds 24 hours. Engineering topology may be
optimized only while semantic equivalence is preserved.

## Production readiness

A readiness manifest is bound directly to the canonical `CampaignSpec`; its
method, backend, benchmark, budget, repetition, and optional RQ4 condition are
derived from that campaign rather than repeated as caller-controlled fields.
A primary campaign fails closed unless every base and method-specific binding
required by the scheduler contract is present exactly once and nonprovisional.
It also requires a validated native backend profile and materializer, complete
opportunity enumeration, complete evaluator/CFS implementation, a frozen
development/final data manifest, and proof that unbound serving choices are
semantically equivalent.

Unguided LLM additionally binds its controller and controller resampling cap.
LLM-as-Judge additionally binds its judge and judge resampling cap. Validation
Profile A cannot pass primary Mem0 readiness. Worker count, GPU model, batch
size, queue depth, and replica count remain outside scientific identity only
when equivalence has been established.

Exact models, prompt contents, decoding values, k, retry caps, development
items, final native backend configurations, evaluator predicates, and serving
topology remain intentionally unbound.

## RQ4 native-memory-LLM control

RQ4 varies only the memory system's native LLM-dependent processing for Mem0 and Graphiti across planning provider families OpenAI, Anthropic, and Google. Production readiness requires a `NATIVE_MEMORY_LLM_CONFIGURATION` binding whose value equals the SHA-256 digest of a self-contained canonical manifest. That manifest stores the exact provider, model identifier and revision, memory-system prompt/config bytes, structured-output schema bytes, parser version, decoding-parameter bytes, retry/resampling-policy bytes, and causal API/runtime version. A digest or display label without the artifact is insufficient. API keys, bearer tokens, secrets, and credential objects are absent from the schema and forbidden from manifests and scientific evidence.

For each memory system, an exact canonical invariant-substrate manifest stores the backend and source version, profile version, storage/database artifact, embedding provider/family, exact embedding model and revision/weights digest, vector dimension and normalization, retrieval/search artifact, reranker/cross-encoder artifact, retrieval k, and isolation/replay profile version. Its canonical bytes and SHA-256 digest are retained. Cross-provider evidence binds the same backend and requires the OpenAI, Anthropic, and Google substrate-manifest digests to be identical. Human-authored labels are not evidence. A provider integration that changes any causal substrate field is ineligible. The API model is not the common response reader, generator, validator, evaluator, or scheduler.

## Split workload acceptance

`LOCAL_RQ1_RQ3_LOAD` contains 1,440,000 valid executions and the same number of response and evaluator records, with the 24-hour local target. `RQ4_API_LOAD` contains 108,000 of each record and has a separate unbound API SLA. `COMBINED_PLANNING_LOAD` is 1,548,000 records and does not imply one shared SLA.

The HMAC-SHA-256 `UFUZZ_PRNG_V1` domain is `ufuzz/evaluation/prng/v1`. This RQ-independent separator replaced the uncommitted RQ1--RQ3 wording before any production use; framing and method-neutral benchmark/repetition derivation are unchanged.
