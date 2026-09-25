# When Correct Memory Goes Wrong: Fuzzing Persistent Memory Use in LLM Agents

U-Fuzz is a behavior-guided fuzzing framework for testing how LLM agents use
persistent memory. Existing memory benchmarks primarily evaluate fixed queries
or the correctness of stored content. U-Fuzz instead searches for cases where a
memory system stores valid information but retrieves or uses it incorrectly
after a valid change to the query or memory state.

This repository contains the research infrastructure accompanying the U-Fuzz
paper. It is an active research artifact: the current release provides the
foundational data, provenance, structural, backend, and validation components,
but it does not include completed RQ1--RQ3 campaigns or final paper tables.

## Overview

Persistent-memory systems let an agent carry facts, preferences, events, and
other context across long interactions. Correct storage alone does not ensure
correct behavior. A harmless paraphrase may displace the relevant memory, an
update may leave stale evidence ranked first, or a deletion may fail to remove
the affected support.

U-Fuzz treats this as a search problem over valid changes to a seed
\(x=(M,q)\), where \(M\) is a materialized memory checkpoint and \(q\) is a
query. Each mutant changes exactly one input:

- query mutations keep \(M\) fixed and produce \((M,q')\);
- memory-state mutations keep \(q\) fixed and produce \((M',q)\).

The mutation space covers meaning-preserving, target-changing, and unsupported
query mutations, together with update, deletion, and unrelated memory changes.
Candidates are validated against frozen structural and provenance constraints
before backend execution. Search then uses observable memory behavior, such as
retrieval coverage and parent-mutant divergence, to prioritize later tests.

U-Fuzz enforces an information-flow boundary between search and evaluation.
Search-safe descriptors and `MutationOpportunity` records contain only the
information needed to construct and validate mutations. Gold answers, positive
and forbidden evidence, and fault labels live in an evaluator-only
`ReferenceVault`. They cannot influence mutation generation, scheduling,
coverage, or retention.

## Supported Systems

The codebase uses a common adapter interface for isolated state creation,
ingestion, ranked retrieval, update, deletion, unrelated change, replay, and
teardown. The current adapters cover:

- **Mem0** (`mem0ai==2.0.12`)
- **A-Mem** (`agentic-memory==0.0.1` at a pinned source commit)
- **Graphiti** (`graphiti-core==0.30.2`)
- **MemOS** (MemoryOS v2.0.33 at a pinned source commit, using the GeneralText
  profile)

Capabilities differ by backend. An operator is admitted only when the adapter
can perform it faithfully and certify its provenance and state transition.
Unsupported or unresolved capabilities fail closed rather than being emulated
through private storage edits.

## Benchmarks

U-Fuzz targets two long-term-memory benchmarks:

- **LoCoMo**, using `data/locomo10.json` from the pinned upstream revision
  recorded in `src/ufuzz/benchmarks/locomo.py`;
- **LongMemEval-S**, using the authors' cleaned
  `longmemeval_s_cleaned.json` artifact at the pinned revision recorded in
  `src/ufuzz/benchmarks/longmemeval.py`.

The datasets are not redistributed in this repository. Obtain them from their
upstream projects and retain their original licenses and terms. The loaders
verify the expected file size and SHA-256 digest by default, preserve native
fields, and derive stable checkpoint, query, session, and source provenance.

Example loader use:

```python
from ufuzz.benchmarks import LoCoMoLoader, LongMemEvalSLoader

locomo = tuple(LoCoMoLoader().load("/path/to/locomo10.json"))
longmemeval = tuple(
    LongMemEvalSLoader().load("/path/to/longmemeval_s_cleaned.json")
)
```

## Methodology

The implementation follows six stages described in the paper.

1. **Seed initialization.** Interaction histories are converted into immutable
   initialization artifacts and materialized as isolated backend states. A seed
   \(x=(M,q)\) retains stable benchmark provenance rather than treating
   backend-generated IDs as scientific identity.
2. **Structural indexing.** A frozen, partial-but-certified structural index
   records canonical facts, query intent, provenance, and applicability
   certificates. Uncertified information remains unstructured.
3. **Mutation opportunities.** Applicable query and memory-state changes are
   represented as finite, method-neutral `MutationOpportunity` records. Backend
   capability masks and exclusion reasons are explicit.
4. **Validation.** Generated candidates must satisfy relation-specific semantic
   checks before execution. Validation uses the parent, candidate, structural
   index, and provenance; it cannot use mutant retrieval, responses, gold
   answers, or evaluator output.
5. **Execution.** Valid mutants run from isolated parent checkpoint states.
   Retrieval observations are resolved through canonical lineage before they
   can count toward an execution budget or coverage.
6. **Evaluation.** Retrieval correctness and answer correctness are assessed
   downstream using the evaluator-only `ReferenceVault`. Confirmed failures are
   excluded from all search feedback.

This separation allows different search methods to share the same seeds,
mutation space, validators, and backend applicability while differing only in
how they prioritize the next opportunity.

## Repository Structure

```text
src/ufuzz/
  backends/          Backend adapters, capability checks, and materializers
  benchmarks/        Provenance-preserving LoCoMo and LongMemEval-S loaders
  coverage.py        Canonical initialized-entry coverage state
  domain.py          Shared benchmark and provenance data contracts
  mapping.py         Backend-neutral retrieval-to-region mapping
  materialization.py Logical seed materialization contracts
  pg_*.py            Search-safe P / evaluator-only G preprocessing tools
  retrieval_*.py     Retrieval capability and feedback contracts
  scheduler_contract.py
                     Frozen scheduler data contracts
  semantic_sidecar.py
                     Evaluator-only semantic records
  state_contract.py  Logical seed and lineage contracts
  structural.py      Certified structural index and QueryIntent
tests/               Unit tests, deterministic fixtures, and opt-in live smokes
pyproject.toml       Package metadata and pinned optional backend dependencies
```

## Installation

U-Fuzz requires Python 3.12 or newer. Create an isolated environment and
install the base package:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The base package has no required third-party runtime dependencies. Install the
extra for the backend you intend to exercise:

```bash
python -m pip install -e '.[mem0]'
python -m pip install -e '.[amem]'
python -m pip install -e '.[graphiti]'
python -m pip install -e '.[memos]'
```

Using a separate environment for each backend is recommended because the
upstream systems have independent dependency stacks and service requirements.
Some extras install an upstream project directly from its pinned Git commit and
therefore require Git during installation.

## Running Tests

Run the complete default unit suite from the repository root:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

The default suite uses deterministic fixtures and does not enable live backend
tests. Live smoke tests are opt-in and may incur provider or service costs. Set
only the flag for the backend you have configured:

```bash
UFUZZ_RUN_LIVE_MEM0=1 \
  python -m unittest discover -s tests -p 'test_backend_smoke.py' -v

UFUZZ_RUN_LIVE_AMEM=1 UFUZZ_AMEM_DEDICATED_PROCESS=1 \
  python -m unittest discover -s tests -p 'test_backend_smoke.py' -v

UFUZZ_RUN_LIVE_GRAPHITI=1 UFUZZ_GRAPHITI_ISOLATED_TEST_INSTANCE=1 \
  python -m unittest discover -s tests -p 'test_backend_smoke.py' -v

UFUZZ_RUN_LIVE_MEMOS=1 UFUZZ_MEMOS_DEDICATED_PROCESS=1 \
  python -m unittest discover -s tests -p 'test_backend_smoke.py' -v
```

Consult the failure message from a live smoke test for any missing package,
credential, local model, or service prerequisite.

## Backend Configuration

Backend configuration is deliberately explicit and fail-closed:

| Backend | Required setup for the current live path |
|---|---|
| Mem0 | The pinned `mem0ai` extra, an isolated local Qdrant state, and provider credentials required by the configured ingestion model. |
| A-Mem | The pinned A-Mem extra, a dedicated process, its embedding model, and credentials for the configured LLM backend. |
| Graphiti | The pinned Graphiti extra, a dedicated Neo4j service or database, and explicit LLM, embedder, and reranker configuration. The smoke harness reads `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, and optional `NEO4J_DATABASE`. |
| MemOS | The pinned MemoryOS extra, a dedicated process, adapter-owned embedded Qdrant storage, and the exact locally available `nomic-embed-text` Ollama artifact expected by the profile. |

The current Mem0, A-Mem, and Graphiti live smoke configurations use
`OPENAI_API_KEY`. Export credentials in the shell or inject them through your
secret manager. Never write API keys, Neo4j passwords, access tokens, or
provider configuration containing secrets into source-controlled files.

Backend-native physical IDs are runtime-local. Scientific evidence identity is
derived from stable benchmark provenance and certified lineage, so a replay is
not accepted merely because its text or enumeration order looks similar.

## Reproducibility

The repository records or checks:

- immutable benchmark revisions, file sizes, and SHA-256 digests;
- pinned backend package versions or source commits;
- stable benchmark provenance and canonical structural-region identities;
- deterministic JSON serialization and digest-bound artifacts;
- explicit backend capability and materialization certificates;
- strict separation between search-safe descriptors and evaluator-only data;
- deterministic fixtures for the default test suite.

Live backends may also depend on external services, model artifacts, and
provider behavior. A reproducible experiment must record those runtime
bindings, service versions, model revisions, decoding settings, seeds, and
environment details in its experiment manifest. A moving model alias is not an
immutable experimental identity.

## Current Status and Limitations

This release currently provides:

- checksum-verifying benchmark loaders;
- stable source provenance and canonical identity contracts;
- partial-but-certified structural indexing and query-intent records;
- adapters, capability checks, and materializers for four memory systems;
- retrieval mapping, coverage, feedback, and scheduler contracts;
- search-safe P and evaluator-only G preprocessing infrastructure;
- semantic sidecars and information-flow regression tests;
- deterministic unit tests and opt-in live backend smoke tests.

This release does **not** claim that the full RQ1--RQ3 experiment matrix has
been executed from this public snapshot. It does not publish final benchmark
results, reproduce final paper tables, or bundle the benchmark datasets,
provider credentials, model weights, or external services. Some backend
capabilities are conditional on exact runtime configuration and are reported as
unavailable or unresolved when they cannot be certified.

## Citation

Citation metadata will be added with the public paper record. Until then,
please cite the accompanying anonymous manuscript, *When Correct Memory Goes
Wrong: Fuzzing Persistent Memory Use in LLM Agents*.
