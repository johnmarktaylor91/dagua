# Sprint 23 Area C: Long-edge-aware Sugiyama ordering for dense DAGs

## Mandate

dependency_500 (-1.92 post sprint-22e), clustered_medium_5x20 (-1.41),
outerplanar_dag_20 (-0.74), and likely multi_component_80 (-0.64) all
share a structural limitation: dagua's layered_dag pipeline does dummy-
node expansion for long edges but its ordering pass (median or
barycenter) is weaker than graphviz_dot's median-with-transpose-phase.

Sprint-22 area D's prototype (gap_validated_layer_swaps, sprint-22e)
was a tactical patch on dependency_500 that closed +0.99 by permuting
within-layer x. The structural fix is a stronger ordering pass BEFORE
coordinate assignment: dummy-node expansion + barycenter or median +
transpose phase + weighted Sugiyama coordinate.

## Research questions

1. Audit dagua's current layered_dag pipeline ordering. Read
   `dagua/layout/ops/pipelines/native_layered_dag.py` and
   `dagua/layout/ops/ordering.py`. Identify which ordering ops run
   today, in what sequence, and where the median-transpose phase
   would slot in.

2. Implement a "median-transpose" ordering pass in /tmp/sprint23_c/.
   Pseudocode:
   ```
   for sweep in range(24):
       if sweep % 2 == 0:
           # downward sweep: each layer ordered by median position
           # of upper-layer neighbors
           for layer in range(L+1):
               sort layer nodes by median upper-neighbor x
       else:
           # upward sweep
           for layer in range(L-1, -1, -1):
               sort layer nodes by median lower-neighbor x
       # transpose phase: try swapping adjacent same-layer nodes
       # if it reduces crossings, accept
       changed = True
       while changed:
           changed = False
           for layer in range(L+1):
               for i in range(len(layer_nodes) - 1):
                   if swap_reduces_crossings(layer_nodes, i, i+1):
                       swap; changed = True
   ```

3. Empirically measure on:
   - dependency_500 (primary target, currently -1.92)
   - clustered_medium_5x20 (currently -1.41)
   - outerplanar_dag_20 (-0.74)
   - multi_component_80 (-0.64)
   - random_dag_200 (protected win)
   - org_chart_deep, hub_fanout_label_skew (protected wins)
   - linear_3layer_mlp, deep_chain_20 (metric ceiling -- must not
     regress)

4. Decide: ship as a polish candidate (post-pipeline projection)
   or as a replacement for the existing ordering pass (forced).
   The picker margin gate is more robust; the forced replacement
   gives stronger lift but risks regression on protected wins.

## Output spec

File: `.project-context/research/sprint_23_finish_line/C_dense_dag_ordering__<agent>.md`

Sections:
- TL;DR (5 bullets)
- Audit: what dagua does today vs what dot does
- Algorithm sketch (Python pseudocode, 100-150 LOC)
- Empirical validation: per-graph table including protected wins
- Polish-candidate vs forced-replacement decision: which one and
  why; if polish candidate, what's the gate
- Implementation: where it slots in dagua/, LOC estimate

## Constraints

- READ-ONLY on dagua/
- HEAD = sprint-22e finalize commit `d27fced`
- Reference sprint-22 area D research at
  `.project-context/research/sprint_22_algo_bets/D_dependency_500_escape__codex.md`
  -- this is the natural extension of that note's "deeper bet"

## Citations

- Eades-Sugiyama-Tamassia "Algorithms for Drawing Graphs" (1981)
  for the original layered drawing
- Gansner-Koutsofios-North-Vo 1993 IEEE TSE 19(3) for median-transpose
- Junger-Mutzel "2-Layer Straightline Crossing Minimization"
  (Algorithmica, 1997) for the transpose phase analysis

## Word budget

2500-4000 words.
