# Wave 1: Implement All Primitive Layout Operations

## Mission

Implement the complete set of primitive layout operations for dagua's modular
framework. These ops are the "vocabulary" -- atomic, composable steps sufficient
to express EVERY layout algorithm dagua supports (23 classic reimplementations +
native engine). This is NOT algorithm translation yet; we are building the toolkit.

## Context

Last sprint built the data structures in `dagua/layout/ops/`:
- `state.py`: LayoutProblem, SolveState, RuntimeContext, HierarchyLevel, etc.
- `base.py`: Op, LossOp, Pipeline, Repeat, Conditional, LossGroup, MultilevelVCycle, EarlyBreak
- `taxonomy.py`: OpCategory enum (21 categories), @register_op registry
- `DESIGN.md`: Catalog of ~80 op names -- treat as HYPOTHESIS to validate

The existing algorithms live in `dagua/layout/classic/` (22 files) and
`dagua/layout/engine.py` (native engine). Do NOT modify these files.

## File Organization

One file per OpCategory, inside `dagua/layout/ops/`:
```
ops/
  __init__.py          # existing -- update exports
  base.py              # existing -- do not modify
  state.py             # existing -- do not modify
  taxonomy.py          # existing -- do not modify
  DESIGN.md            # existing -- update when done
  init.py              # INIT ops
  preprocess.py        # PREPROCESS ops
  distance.py          # DISTANCE ops
  layering.py          # LAYERING ops
  ordering.py          # ORDERING ops
  coordinate.py        # COORDINATE ops
  coarsen.py           # COARSEN ops
  prolong.py           # PROLONG ops
  force.py             # FORCE ops
  loss.py              # LOSS ops
  embed.py             # EMBED ops
  optimize.py          # OPTIMIZE ops
  project.py           # PROJECT ops
  anneal.py            # ANNEAL ops
  context.py           # CONTEXT ops
  converge.py          # CONVERGE ops
  postprocess.py       # POSTPROCESS ops
  edge_route.py        # EDGE_ROUTE ops
  utility.py           # UTILITY ops
```

Each file contains:
- All Op subclasses for that category
- Frozen dataclass configs (XxxConfig) for each configurable op
- @register_op on every concrete op
- Proper reads/writes/requires metadata
- Docstrings with algorithm provenance ("Used by: FR, FA2, SFDP")

## Execution Plan

You are an architect. Do NOT write implementation code yourself -- dispatch
Codex agents for all implementation work. Follow this sequence:

### Phase 1: Research (parallel Codex agents)

Dispatch 4-5 read-only Codex agents to crawl the algorithm code. Each agent
reads a subset of algorithms and reports back:
- Every distinct computational step
- Parameters and their defaults
- Which steps repeat across algorithms (the gold -- reusable ops)
- Anything hardcoded that should be configurable

Suggested split:
- Agent A: Force-directed family (fr.py, fa2.py, gem.py, graphopt.py, linlog.py, drl.py, lgl.py)
- Agent B: Distance/stress family (kk.py, stress_majorization.py, stress_sgd.py, maxent_stress.py, classical_mds.py, pivot_mds.py)
- Agent C: Hierarchical + tree (sugiyama.py, reingold_tilford.py, sfdp.py, fmmm.py, davidson_harel.py)
- Agent D: Embedding/neural (spectral.py, tsnet.py, umap_layout.py, neulay.py, sgd2_multi.py)
- Agent E: Native engine (engine.py -- this is the biggest single file, ~1200 lines)

Each agent prompt MUST include:
```xml
<task>
Read the following algorithm files in dagua/layout/classic/ (or engine.py).
For EACH file, extract:
1. Every distinct computational step, in execution order
2. All configurable parameters with defaults and types
3. Data structures used (what fields of pos, layers, forces, etc.)
4. Optimizer/solver used and how
5. Convergence criteria
6. Steps shared with other algorithms (name them)
7. Anything hardcoded that could be a parameter

Also read dagua/layout/ops/DESIGN.md for the existing op catalog.
Cross-reference: which cataloged ops match real code? Which are missing?
Which cataloged ops should be split or merged?

Files to read: [LIST EXACT FILES]

Report as structured markdown with one section per algorithm.
End with a "Cross-Algorithm Patterns" section listing reused motifs.
</task>
<default_follow_through_policy>
This is read-only research. Read every file thoroughly.
Default to the most reasonable interpretation and keep going.
</default_follow_through_policy>
```

### Phase 2: Synthesis

After all research agents return, synthesize their findings:
1. Build the DEFINITIVE op catalog -- validate/refine DESIGN.md's ~80 ops
2. Identify ops to add, remove, split, or merge
3. For each op: name, category, config fields, reads/writes, which algos use it
4. List followup questions (ambiguities, design forks)

### Phase 3: Followup Research (if needed)

Dispatch targeted Codex agents to answer specific questions from Phase 2.
Keep these narrow and focused. One question per agent.

### Phase 4: Implementation Plan

Write a comprehensive plan document at `.project-context/plans/wave1_ops.md`:
- Final op catalog with every op, its config, its interface
- File assignments (which ops go in which file)
- Implementation priority (foundational ops first)
- Testing strategy (unit test per op, compose into mini-pipelines)
- Estimated Codex agent split for implementation

### Phase 5: Adversarial Review

Dispatch a Codex adversarial review of the plan:
```
/codex:adversarial-review --background "Review the Wave 1 primitive ops plan at
.project-context/plans/wave1_ops.md. The plan must define operations sufficient
to express ALL 23 classic algorithms + the native engine. Check:
1. COMPLETENESS: can every algorithm step be expressed? Walk through each algo.
2. GRANULARITY: are ops too coarse (doing multiple things) or too fine (trivial)?
3. CONFIGURABILITY: are hardcoded choices exposed as config?
4. COMPOSABILITY: can ops actually compose via Pipeline/Repeat/LossGroup?
5. STATE CONTRACT: do reads/writes match SolveState fields?
6. MISSING OPS: anything the algorithms need that isn't in the plan?
Cross-reference against every file in dagua/layout/classic/ and engine.py."
```

Iterate with the adversarial reviewer until they have NO substantive objections.
Do NOT skip this. Do NOT declare it done if they have remaining concerns.

### Phase 6: Implementation (parallel Codex agents)

Split implementation across Codex agents by file. Each agent writes ONE
category file. Include in each prompt:
- The final op catalog for that category
- The base classes from base.py (paste the code, don't reference)
- The state model from state.py (paste relevant dataclass definitions)
- The taxonomy from taxonomy.py (paste OpCategory enum)
- Example ops from other categories if helpful for pattern consistency
- Test requirements: pytest tests in tests/test_ops_{category}.py

Pre-dispatch checklist:
- No two agents write the same file
- Shared imports resolved (each file imports from base, state, taxonomy)
- Test files are separate per agent

Suggested agent split (adjust based on final catalog):
- Agent 1: init.py + tests
- Agent 2: preprocess.py + distance.py + tests
- Agent 3: layering.py + ordering.py + coordinate.py + tests
- Agent 4: coarsen.py + prolong.py + tests
- Agent 5: force.py + tests (likely the biggest -- ~9 ops)
- Agent 6: loss.py + tests (also big -- ~16 ops)
- Agent 7: embed.py + tests
- Agent 8: optimize.py + project.py + tests
- Agent 9: anneal.py + converge.py + tests
- Agent 10: context.py + postprocess.py + edge_route.py + utility.py + tests

Each agent prompt MUST include:
```xml
<task>
Implement the following layout operations in dagua/layout/ops/{file}.py.

[PASTE: full op specifications from the plan]
[PASTE: base.py Op/LossOp class definitions]
[PASTE: state.py relevant dataclass definitions]
[PASTE: taxonomy.py OpCategory enum]

Requirements:
- Subclass Op (or LossOp for differentiable losses)
- Decorate with @register_op
- Set name, category, reads, writes, requires class attributes
- Implement apply(problem, state, ctx) -> SolveState
- Create frozen dataclass XxxConfig for each configurable op
- Config goes in the same file, above the op that uses it
- Include docstring with: what the op does, which algorithms use it
- Handle edge cases (empty graph, single node, disconnected)
- All tensor ops must respect ctx.plan.device
- Return state (modified or new) -- never return None

Also write tests in tests/test_ops_{category}.py:
- Test each op individually on small graphs (5-20 nodes)
- Test config variations
- Test edge cases (empty, single node, disconnected)
- Test that reads/writes metadata is accurate
- At least one composition test (2+ ops in Pipeline)

Read dagua/layout/ops/base.py, state.py, taxonomy.py before writing.
Read the algorithm files listed under "Used by" to match behavior.

Reference: AGENTS.md and .project-context/conventions.md for code style.
</task>
<completeness_contract>
Every op listed in the spec MUST be implemented. Every op MUST have tests.
Do not skip ops. Do not write stub implementations.
</completeness_contract>
<verification_loop>
After writing all ops and tests, run: pytest tests/test_ops_{category}.py -x --tb=short
Fix any failures before finishing.
</verification_loop>
```

### Phase 7: Review and Commit

After all implementation agents complete:
1. Dispatch a Codex review of the full ops/ directory
2. Run full test suite: `pytest tests/test_ops_*.py -x --tb=short`
3. Update `dagua/layout/ops/__init__.py` exports
4. Update `dagua/layout/ops/DESIGN.md` with final catalog
5. Commit: `feat(ops): implement complete primitive operation library`

## Random Seed Fidelity

This is non-negotiable: ops that involve randomness MUST produce EXACTLY the
same output as the existing classic/ reimplementations when given the same seed.
This is how we verify fidelity when we later translate algorithms to the op
framework.

Concretely:
- Every op that uses randomness must accept a seed (via config or from
  LayoutProblem.seed) and use torch.Generator for reproducibility
- The RNG sequence must match what the classic/ code does. If fr.py calls
  `torch.rand(N, 2, generator=g)` for init, then RandomUniformInit must
  produce the identical tensor given the same seed.
- Research agents: when cataloging each algorithm, note EXACTLY how seeds
  are consumed -- which calls, in what order, with what generator. This is
  critical for fidelity verification.
- Implementation agents: read the corresponding classic/ file and match the
  RNG call sequence. Do not "improve" the randomness -- match it.
- Test strategy: for each op that uses randomness, write a fidelity test that
  runs the op AND the equivalent code path from the classic/ file with the
  same seed and asserts torch.allclose() on the output.

The goal: `Pipeline([RandomUniformInit(seed=42), ...])` must produce bit-identical
positions to `fr.py` with `seed=42` at the same stage of execution. When we
later compose full algorithm pipelines from ops, we will diff against the
classic/ implementations. Any deviation is a bug.

## Critical Rules

- Do NOT modify existing algorithm files in dagua/layout/classic/ or engine.py
- Do NOT modify base.py, state.py, or taxonomy.py (unless a bug is found)
- Every op MUST be a proper Op subclass with @register_op
- Every op MUST have unit tests
- Lean on Codex for ALL implementation -- Claude orchestrates only
- The adversarial reviewer MUST be satisfied before implementation begins
- When in doubt about an op's behavior, READ THE ALGORITHM CODE -- it is ground truth
- Make hardcoded choices configurable (optimizer type, cooling schedule, etc.)
- Ops that appear in 3+ algorithms are HIGH PRIORITY for correct abstraction

## Resource Awareness

Before launching parallel Codex agents:
```bash
free -h && df -h / && uptime
```
Stagger launches if system is under pressure. Monitor with `/codex:status`.
