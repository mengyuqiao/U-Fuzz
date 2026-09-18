RQ1 goal:
Evaluate whether U-Fuzz improves memory exploration and confirmed
memory-use failure discovery under a fixed execution budget.

Open-source memory systems:
- Mem0
- A-Mem
- Graphiti

Benchmarks:
- LoCoMo
- LongMemEval-S

Baselines:
- Random Mutation
- Unguided LLM Mutation

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
- Unrelated change

Do NOT implement:
- Addition
- Conflict
- Poisoning
- MCTS
- gold-guided search

Seed:
x = (M, q)

Reference:
G(x) = (a, E+, E-)
References are not part of search feedback.

Critical invariant:
Each parent-mutant pair changes only one input:
(M, q) -> (M, q')
or
(M, q) -> (M', q)

Validation happens before execution and is independent of retrieval
or response produced by the mutant.

Search feedback:
- region coverage
- retrieval divergence for meaning-preserving mutations
- retrieval-pattern novelty

Confirmed failure labels must NEVER affect seed retention or scheduling.

RQ1 metrics:
- Fail@B: number of distinct confirmed memory-use failures within B valid executions
- RegionCov@B: memory-region coverage within the same budget

All methods must use:
- same initial seed corpus
- same applicable mutation space
- same validation
- same execution budget