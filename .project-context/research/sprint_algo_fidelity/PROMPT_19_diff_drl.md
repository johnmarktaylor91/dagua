<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`.

Round 19 ADVERSARIAL DIFF for **drl** family. The user wants every
divergence between dagua DRL and igraph DRL, line by line.

## Inputs

**Dagua side (READ ALL):**
- `dagua/layout/ops/drl.py` (or whatever the drl ops file is -- locate
  it via grep)
- `dagua/layout/ops/pipelines/drl.py`
- `dagua/eval/variants.py` for drl variant configs

**igraph reference (READ ALL of these C files):**
- `/home/jtaylor/projects/_references/igraph/src/layout/drl/drl_graph.cpp` (main SA loop)
- `/home/jtaylor/projects/_references/igraph/src/layout/drl/drl_layout.cpp` (entry + defaults)
- `/home/jtaylor/projects/_references/igraph/src/layout/drl/DensityGrid.cpp` (density grid for repulsion)
- `/home/jtaylor/projects/_references/igraph/src/layout/drl/drl_Node.cpp` (per-node update)
- Plus headers: drl_graph.h, drl_layout.h, DensityGrid.h, drl_Node.h

**Round 14 prior diagnosis:**
- `eval_output/algo_fidelity/round_14/SUMMARY.md`

## What to do

**DIAGNOSIS-ONLY.** Produce ONE document:
`.project-context/research/sprint_algo_fidelity/ROUND_19_DIFF_drl.md`

Cover:

1. **Overall pipeline structure**: 6 phases (init, liquid, expansion,
   cooldown, crunch, simmer). Compare per-phase iteration counts,
   temperatures, attractions, damping mults to igraph's defaults at
   `drl_layout.cpp:240-310`. Make a full table.

2. **Per-phase parameter table** for all 5 templates (DEFAULT,
   COARSEN, COARSEST, REFINE, FINAL) -- igraph values vs dagua values
   for each.

3. **Density grid implementation**: igraph uses a grid-based repulsion
   (`DensityGrid.cpp`) where each node contributes to a fixed-resolution
   grid; repulsion comes from grid-density gradient. Does dagua
   implement this? If not (plain O(N^2)?), call out as biggest divergence.

4. **Per-node update step**: igraph's `drl_Node.cpp::update_position`
   handles cell membership transitions in the density grid. Compare
   to dagua's per-node updates.

5. **Edge cutting**: Round 14 noted "igraph removes the selected long
   edge from only the current node's neighbor map; dagua removes
   symmetrically". Verify this and characterize the impact.

6. **Move-acceptance + cooling**: simulated annealing schedule per phase.

7. **RNG**: graph initialization, node iteration order, move
   randomization. igraph uses specific RNG sequences; check dagua's.

8. **Hyperparameter alignment** (full table)

9. **Ranked fix list** (top 10+ items)

10. **Recommended Round 20 fix scope**

Be exhaustive. Cite line:line refs.

## Constraints

DIAGNOSIS ONLY. No file edits. No commits.
</task>

<scope_constraints>
DIAGNOSIS-ONLY. NO file edits. NO commits.
Allowed: read files, run greps, write ONE markdown report.
</scope_constraints>

<verification_loop>
File ROUND_19_DIFF_drl.md exists, exhaustive, with line:line refs.
</verification_loop>
