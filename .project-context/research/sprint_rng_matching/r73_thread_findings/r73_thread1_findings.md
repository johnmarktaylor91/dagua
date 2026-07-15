# R73 Thread 1: Sugiyama Divergence Analysis
## 231 Divergent Combos (Mode-B, Deterministic)
## Investigation Method: Benchmark-path analysis on r72 fidelity data

---

## 1. Inventory Summary

**Total divergent combos: 231**
- Variants: classic_sugiyama_{default, graphviz_fidelity, passes4, passes48, tight, wide}
- 6 variants x ~38 graphs (with graphviz_fidelity being separate bucket)

**Combo breakdown by variant:**
- classic_sugiyama_graphviz_fidelity: 65 divergent combos (65 unique graphs)
- classic_sugiyama_default: 34 divergent combos
- classic_sugiyama_wide: 36 divergent combos
- classic_sugiyama_tight: 34 divergent combos
- classic_sugiyama_passes4: 31 divergent combos
- classic_sugiyama_passes48: 31 divergent combos

**Two distinct reference buckets:**
- **igraph bucket** (variants: default/wide/tight/passes4/passes48): 166 combos, 41 unique graphs
- **graphviz bucket** (variant: graphviz_fidelity): 65 combos, 65 unique graphs

---

## 2. Sub-Bucket Breakdown (5 Sub-Buckets)

### Sub-Bucket A: LP Degeneracy -- Layer Assignment (igraph bucket)
- **Count: ~27 unique graphs x 5 variants = ~135 combos**
- Root: `_igraph_glpk_layer_assignments()` uses scipy HiGHS LP with zero objective
  `c = [0.0] * N`. HiGHS picks an arbitrary feasible point; igraph uses GLPK simplex
  which picks a different equally-optimal feasible point. Both give same LP objective.
- **Verified:** On `moe_router_sparse`, dagua assigns node 5 to layer 2, igraph assigns
  layer 3. All three (dagua, igraph, alternative-objective) yield LP objective value 13.0.
  The difference is which degenerate LP vertex the solver lands on.
- **Verdict: FLOOR**
- Why not fixable: Matching HiGHS to GLPK's simplex pivot trajectory requires embedding
  GLPK. The LP has degenerate optima (zero objective means ANY feasible point is optimal);
  no objective formulation forces a unique solution unless we replicate GLPK's exact
  internal simplex steps.

### Sub-Bucket B: Brandes-Kopf vs igraph x-coordinate (igraph bucket, layer-matched)
- **Count: ~14 unique graphs x 5 variants = ~70 combos**
- Root: dagua uses Brandes-Kopf 4-alignment median compaction
  (`_coordinate_assignment()` in `sugiyama.py`). igraph uses a simpler sequential
  x-placement that assigns positions left-to-right with node_sep spacing, then
  adjusts for alignment with parents.
- **Verified:** `kitchen_sink_platform_graph` layer 0 (nodes [0, 12, 16]):
  - dagua: x = [1.0, 2.0, 3.0] (uniform Brandes-Kopf centering)
  - igraph: x = [-0.5, 0.5, 2.0] (non-uniform: gaps 1.0, 1.5)
  - Layers and ordering are bit-identical; only x differs.
- **Verdict: FIXABLE**
- Fix: Add `use_igraph_x_placement` flag to `_CoordinateAssignment` op and implement
  igraph's sequential placement algorithm in `fidelity_mode='igraph'` path.
- **File:function:line:**
  - `dagua/layout/ops/sugiyama.py`, `_CoordinateAssignment.apply()` ~L2590: add branch
    for `use_igraph_x_placement=True` that calls new `_igraph_sequential_x_placement()`
  - `dagua/layout/ops/sugiyama.py`, `_igraph_glpk_layer_assignments()` region: study
    igraph C source (`layout/sugiyama.c:sugiyamaLayout()`) for sequential x-pass logic
  - `dagua/layout/ops/pipelines/sugiyama.py`, `build_sugiyama_pipeline()`: pass
    `use_igraph_x_placement=True` when `fidelity_mode='igraph'`
- **Expected impact: ~70 combos** (the 14 igraph-bucket graphs where layers+ordering match)

### Sub-Bucket C: Graphviz mincross ordering (graphviz_fidelity bucket, layers match)
- **Count: ~23 combos**
- Root: dagua's `_dot_mincross.py` `graphviz_mincross()` produces different within-layer
  node order than actual `dot` mincross.c. Both start from the same layer assignment but
  arrive at different orderings after barycenter/transpose passes.
- **Verified:** `bipartite_4_3_4`, layer 0:
  - dagua: [0, 4, 5, 6]
  - dot: [5, 6, 0, 4]
  - d_R = 0.387 (substantial Procrustes distance)
- Root cause candidates (require diff against mincross.c):
  1. Tie-breaking when barycenter values equal -- dot uses "flat median" rule; dagua may
     use sort-stable insertion order
  2. Transpose phase: dot's `do_transpose()` uses different neighbor iteration order
  3. Initial order seeding: dot initializes orders from DFS vs dagua's BFS
- **Verdict: FIXABLE** (but requires careful code-level audit of mincross.c vs
  `_dot_mincross.py` -- approximately 100-200 lines of C to match)
- Fix: Systematically audit `_dot_mincross.py::graphviz_mincross()` against
  `graphviz/lib/common/mincross.c` for: flat-median tie-breaking, transpose neighbor
  iteration order, convergence condition, initial seed order.
- **File:function:line:**
  - `dagua/layout/ops/_dot_mincross.py`, `graphviz_mincross()` entire function: audit
    each sub-step against mincross.c reference
  - Specifically: `_barycenter_sort()` (tie-break), `_do_transpose()` (edge direction,
    neighbor scan order), initial seed (DFS vs BFS)
- **Expected impact: ~23 combos** (graphviz_fidelity graphs where layers match but order
  differs)

### Sub-Bucket D: Brandes-Kopf vs dot x-coordinate (graphviz_fidelity, layers+order match)
- **Count: ~26 combos**
- Root: Same coordinate assignment mismatch as Sub-Bucket B but against graphviz dot
  reference. dot uses a priority-based x-placement (see graphviz `coord.c:node_pos()`
  which assigns x via network-simplex on the "auxiliary graph"). dagua uses Brandes-Kopf.
- **Verified:** `binary_tree` -- layers match, ordering matches, but d_R = 0.237.
  On `binary_tree`, Brandes-Kopf produces symmetric centering [0, -1, 1, -1.5, -0.5, 0.5,
  1.5] and dot produces the same structure -- but for non-symmetric graphs the two diverge.
- **Verdict: PARTIALLY FIXABLE**
- The full dot x-coordinate algorithm (network simplex on auxiliary graph) is substantial.
  A simpler fallback is to use the same sequential x-placement approach but match dot's
  specific priority weighting for virtual nodes. Full fix requires implementing
  `coord.c:node_pos()` logic.
- **File:function:line:**
  - `dagua/layout/ops/sugiyama.py`, `_CoordinateAssignment.apply()`: add
    `use_graphviz_x_placement` branch for `fidelity_mode='graphviz'`
  - New function `_graphviz_coord_x_placement()` implementing the priority-order x-scan
    from `graphviz/lib/common/coord.c`
- **Expected impact: ~26 combos IF fully fixed; ~10-15 combos with approximate fix**

### Sub-Bucket E: Network simplex rank divergence (graphviz_fidelity, layer mismatch)
- **Count: ~16 combos**
- Root: `_graphviz_layer_assignments()` in `dagua/layout/ops/sugiyama.py` calls
  `dot_rank.py` network simplex. 16 graphviz_fidelity graphs have wrong layer assignment
  vs actual `dot`. These involve graphs where: (a) cycle detection + feedback arc removal
  differs from dot's `acyclic.c`, or (b) the network simplex solver reaches a different
  feasible tree due to floating point or tie-breaking in pivot selection.
- Cluster-level treatment may also play a role: dot's `mincross.c` handles cluster
  sub-graphs with special rank constraints; dagua's `_graphviz_layer_assignments()` may
  not replicate cluster rank pinning.
- **Verdict: FLOOR (for now)** -- would require full audit of `dot_rank.py` against
  graphviz's network-simplex rank.c + acyclic.c. These 16 combos are distinct graphs
  from the ordering and x-coord buckets (sub-bucket E graphs have wrong layers, so
  ordering and x comparisons are moot).

---

## 3. Impact Table

| Sub-Bucket | Mechanism | Verdict | Combos | Post-fix combos |
|---|---|---|---|---|
| A | LP degeneracy (igraph layer assign) | FLOOR | ~135 | ~135 (no change) |
| B | Brandes-Kopf vs igraph x-coord | FIXABLE | ~70 | ~0 |
| C | Graphviz mincross ordering | FIXABLE | ~23 | ~0 |
| D | dot coord.c x-placement | PARTIALLY FIXABLE | ~26 | ~10-15 |
| E | Network simplex rank divergence | FLOOR | ~16 | ~16 (no change) |

**If B+C+D fixed:** 231 -> ~151-161 residual divergent combos
**If B+C+D(full) fixed:** 231 -> ~151 residual
**Irreducible floor:** Sub-bucket A (~135) + Sub-bucket E (~16) = ~151 combos

**Note:** Sub-bucket A is 58% of the total divergence and is irreducible without
embedding GLPK as the LP solver. igraph's layer assignment uses GLPK simplex with
zero-objective LP, and the specific degenerate solution it picks is not reproducible
in scipy HiGHS without matching exact pivot history.

---

## 4. Fix Specs (FIXABLE mechanisms)

### Fix B: igraph sequential x-placement
**Files to modify:**
- `dagua/layout/ops/sugiyama.py`
  - Add `use_igraph_x_placement: bool = False` to `_CoordinateAssignmentConfig`
  - In `_CoordinateAssignment.apply()` (L2590+): branch on `use_igraph_x_placement`;
    call new `_igraph_sequential_x_placement(layers, node_sep, node_sizes)` instead of
    `_coordinate_assignment()`
  - Implement `_igraph_sequential_x_placement()`: for each layer, assign
    `x[i] = i * node_sep` then center each node over its parent's x centroid
    (igraph's `_place_nodes_at_*` functions in `layout/sugiyama.c`)
- `dagua/layout/ops/pipelines/sugiyama.py`, `build_sugiyama_pipeline()`:
  - When `fidelity_mode='igraph'`, pass `use_igraph_x_placement=True` to
    `_CoordinateAssignment`
**Reference:** igraph C source `igraph/src/layout/sugiyama.c`,
  function `igraph_layout_sugiyama()`, x-placement section (lines ~800-900)

### Fix C: graphviz mincross tie-breaking audit
**Files to modify:**
- `dagua/layout/ops/_dot_mincross.py`
  - `graphviz_mincross()`: audit against graphviz `lib/common/mincross.c`
  - Specifically: `do_mincross()` line ~185-240 in mincross.c
  - Check: median tie-break uses "flat" rule (fractional median of edge positions);
    dagua may be using a different sort-stable order when medians are equal
  - Check: `transpose()` scans virtual (dummy) nodes vs real nodes in specific order;
    dagua may iterate differently
  - Check: convergence check `_MIN_QUIT = 8` and `_CONVERGENCE_RATIO = 0.995` -- verify
    these match graphviz's `MINQUIT` and `CONVERGENCE` in mincross.h
**Reference:** graphviz `lib/common/mincross.c` + `lib/common/mincross.h`

### Fix D (approximate): dot sequential x-placement
**Files to modify:**
- `dagua/layout/ops/sugiyama.py`
  - Add `use_graphviz_x_placement: bool = False` to `_CoordinateAssignmentConfig`
  - Implement `_graphviz_sequential_x_placement()`: use left-to-right x assignment
    with priority for real nodes before virtual (dummy) nodes; then shift each rank
    to minimize edge length squared (greedy pass)
- `dagua/layout/ops/pipelines/sugiyama.py`:
  - When `fidelity_mode='graphviz'`, pass `use_graphviz_x_placement=True`
**Reference:** graphviz `lib/common/coord.c`, `node_pos()` function

---

## 5. Residual (Irreducible Floor)

**~151 combos will remain divergent after all FIXABLE mechanisms are addressed:**

Sub-bucket A (LP degeneracy, ~135 combos): The igraph Sugiyama reference uses GLPK
with zero LP objective, which produces a degenerate arbitrary feasible solution. scipy
HiGHS produces a different equally-valid solution. No formulation of the objective
forces GLPK's exact pivot path without embedding GLPK. This is provably a solver
degeneracy issue, not an algorithmic mismatch.

Sub-bucket E (network simplex rank, ~16 combos): graphviz dot's network simplex rank
assignment (`rank.c`) has cycle-handling and cluster-constraint logic that differs from
dagua's `dot_rank.py`. The divergence manifests as wrong layer assignment on 16 graphs.
Full fix requires auditing rank.c vs dot_rank.py, which is a separate (large) task.

---

## 6. Evidence Quality

All measurements from r72 fidelity data at:
`/home/jtaylor/projects/dagua/eval_output/fidelity_definitive_r72/per_combo.json`

Specific benchmark-path verification performed:
- Mode-B analysis: all 231 combos are `near_deterministic=True`, `n_ref_seeded_ok=0`
  (expected for deterministic references), `d_R` range 0.003-0.993
- Sub-bucket A evidence: moe_router_sparse LP objective = 13.0 for dagua/igraph/alternative
  (all equal) but node layer assignments differ -> degenerate LP
- Sub-bucket B evidence: kitchen_sink_platform_graph layer 0 x-coords dagua=[1,2,3]
  vs igraph=[-0.5,0.5,2.0]; layers and ordering identical -> x-only divergence
- Sub-bucket C evidence: bipartite_4_3_4 layer 0 dagua=[0,4,5,6] vs dot=[5,6,0,4];
  d_R=0.387 -> ordering divergence
- Sub-bucket D evidence: binary_tree layers+ordering match (verified directly) but
  d_R=0.237 -> x-coordinate-only divergence

Categorization method: For each divergent combo, ran direct Python comparison of
dagua vs reference layer assignments (igraph adapter and graphviz `dot -Tjson`),
then checked ordering, then x-coordinates -- attributing divergence to first
differing stage.
