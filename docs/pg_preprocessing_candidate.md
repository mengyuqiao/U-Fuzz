# Query-centered P/G preprocessing candidate

`PG_PREPROCESS_V1` is the frozen production candidate awaiting independent
human validation. It binds the two query-centered artifacts used by the
current Method:

- `BuildDescriptor(q, H_1:t, Z)` produces the search-safe descriptor `P(x)`.
- `BuildReference(q, H_1:t, Z)` produces the evaluator-only reference `G(x)`.

The candidate does not construct a checkpoint-wide fact graph and does not
claim global semantic completeness. `P(x)` registers a finite set of
source-grounded applicable mutation opportunities. Mutation coverage is over
that registered set. Ambiguous slots, propositions, or relations fail closed.

## Frozen inference and selection configuration

The semantic model is `Qwen/Qwen3-14B` at revision
`40c069824f4251a91eefaf281ebe4c544efd3e18`, loaded by Transformers 4.56.2
and Torch 2.8.0+cu128 in BF16 with SDPA. Decoding uses
`do_sample=False`, `enable_thinking=False`, seed 1729, and no retries. The
tokenizer, tokenizer configuration, model configuration, chat template, exact
prompt texts, output limits, and their SHA-256 digests are part of the
canonical manifest returned by
`ufuzz.pg_preprocess.pg_preprocess_v1_manifest()`.

Source selection uses the frozen 768-dimensional `nomic-embed-text` weights
whose immutable manifest digest is
`0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f`.
Cosine ranking selects the top 64 source units, with source ordinal and
provenance ID as deterministic ties. An unrelated-change candidate pool takes
at most 16 sources outside the top 64 in event-keyed SHA-256 order. The moving
text label `nomic-embed-text:latest` is transport metadata only.

## Search-safe `P(x)`

The query parser creates one seed-local answer-bearing relation slot. Its
identity binds checkpoint, query, slot index, normalized relation label, and
an explicit query anchor or fixed implicit explanation. Meaning-preserving
children inherit this identity. Equivalent slots in unrelated roots need not
share a global ontology ID.

Candidate propositions require an exact unique source span, stable provenance,
source ordinal, entity, relation, mutable value, and temporal scope when
needed. NFKC normalization trims and collapses whitespace; identity fields are
casefolded while meaningful punctuation and numbers are retained. No fuzzy or
embedding-based repair is allowed.

The final deterministic prechecks reject placeholder entity, relation, or
value labels after normalization: empty, `none`, `null`, `unknown`,
`unresolved`, and `n/a`. First-person preference queries must identify the
preference holder as the user, speaker, or questioner; an object-only subject
fails closed. The native `single-session-preference` type may constrain only
the relation family. The query-only parse is used when resolved; otherwise a
separately fixed hinted parse is considered. Neither pass receives a gold
answer or evidence label.

The registered relations are:

- Meaning-Preserving Query: a resolved canonical seed-local intent.
- Target-Changing Query: one replaceable slot and an exact grounded supported
  alternative, with no expected child answer in `P(x)`.
- Unsupported Query: an event-keyed synthetic session-local identifier whose
  normalized form is mechanically absent from all raw checkpoint text and
  registered identifiers.
- Update: an exact grounded query-relevant proposition with a mutable value.
- Deletion: an exact grounded query-relevant proposition.
- Unrelated Change: a proposition independently classified as an eligible
  existing memory-state proposition and, in a separate query-aware decision,
  as unrelated to the seed slot.

Existing memory-state eligibility is query-independent. It admits explicit
states, events, preferences, relations, plans, attributes, possessions,
locations, counts, and times with a meaningful entity, relation, and mutable
value. Temporary and one-time propositions may qualify. Query relevance is a
separate three-way `RELEVANT`/`UNRELATED`/`UNRESOLVED` decision.

## Evaluator-only `G(x)`

`G(x)` may use benchmark gold answers and native evidence restrictions.
LoCoMo support remains within valid native evidence turns. LongMemEval-S
support remains within all occurrences of nominated `answer_session_ids`;
`has_answer=true` may narrow candidates but does not itself establish support.
The LoCoMo gold resolver accepts a sole field or equal `answer` and
`adversarial_answer` values and fails closed on conflicts.

Native evidence is processed in chunks of at most 10 source units and 12,000
source characters. Every accepted `E+` region has an exact provenance ID and
unique quote/span. A required support missing from the bounded native scope
makes `G(x)` unresolved. `G(x)` never creates a search-safe proposition.

`P(x)` never contains a gold answer, accepted-answer structure, native support
label, evaluator verdict, failure surface, CFS, UF, or Cov. `G(x)` is never
available to mutation enumeration, generation, scheduling, retrieval
feedback, or search priority. The candidate is frozen before fuzzing and is
shared across every method, backend, and repetition.

## Human-validation freeze

The fixed validation population contains all 55 P2R1–P2R3 records: five
LoCoMo queries from each category 1–5 and five LongMemEval-S queries from each
of its six native question types. Two annotators label independently; a third
documented adjudicator resolves every disagreement. Primary agreement must be
at least 0.85 with Cohen's kappa at least 0.70.

Adjudicated precision must be at least 0.95 for query slots, 0.90 for each
registered Target-Changing, Unsupported, Update, Deletion, and Unrelated
Change class, 0.95 for resolved `G(x)` references, and 0.95 for accepted `E+`
regions. Provenance/span, synthetic-target absence, and information-flow
violations must remain zero. A repeated same-cause error in a benchmark,
native type, or mutation relation blocks a pass even when aggregate thresholds
hold. Human accuracy is not validated until this gate is completed.

Any rule change after the first human label requires a new preprocessing
contract version and a new independent validation protocol.
