<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`.

Round 19 ADVERSARIAL DIFF for **graphopt** family. Round 16 already
verified hyperparameters match. Find every OTHER divergence,
line by line.

## Inputs

**Dagua side (READ ALL):**
- `dagua/layout/ops/pipelines/graphopt.py`
- `dagua/layout/ops/init.py` -- specifically `GraphOptInitializePositions`
  and `GraphOptInitializePositionsConfig` (lines 436-490)
- `dagua/layout/ops/force.py` -- look for the GraphOpt Coulomb constant
  and force-law math
- `dagua/layout/ops/postprocess.py` -- if any GraphOpt finalize logic

**igraph reference (READ ALL):**
- `/home/jtaylor/projects/_references/igraph/src/layout/graphopt.c` (entire ~500 lines)

Round 16 confirmed defaults match. So the divergence is in:
- Force-law math (Coulomb formula details)
- Newton-style move/displacement code
- max_sa_movement clamping behavior
- Random init range (already aligned in Round 16)
- RNG semantics (numpy vs torch)
- Edge weight handling

## What to do

**DIAGNOSIS-ONLY.** Produce ONE document:
`.project-context/research/sprint_algo_fidelity/ROUND_19_DIFF_graphopt.md`

Cover:

1. **Overall flow** (initialize -> n iterations of force/move -> done)
2. **Coulomb repulsion formula** (igraph: `8987500000 * q^2 / d^2`,
   exact line in graphopt.c, vs dagua's force.py impl)
3. **Spring/Hooke attraction formula** (igraph: `force = -k*(d-L)`,
   line ref, vs dagua)
4. **Force vector direction conventions** (igraph: line 145; dagua: ?)
5. **Newton's second law step** (`displacement = force / mass`,
   clamping at `max_sa_movement`)
6. **Edge weight handling** (igraph respects per-edge weight?)
7. **RNG semantics** for initial layout (numpy vs torch behavior on
   same seed -- often produces different sequences)
8. **Self-loops handling**
9. **Hyperparameter alignment table** (confirm)
10. **Ranked fix list** (even small ones)
11. **Recommended Round 20 fix scope**

## Constraints

DIAGNOSIS ONLY. No file edits. No commits.
</task>

<scope_constraints>
DIAGNOSIS-ONLY. Read-only.
</scope_constraints>

<verification_loop>
File ROUND_19_DIFF_graphopt.md exists, exhaustive, with line:line refs.
</verification_loop>
