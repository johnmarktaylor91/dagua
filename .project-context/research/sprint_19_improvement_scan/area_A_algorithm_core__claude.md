# Area A: Algorithm Core -- Dagua vs Sugiyama Gold Standard

Reviewer: Claude (Opus 4.7 1M)
Scope: `dagua/layout/` + default `dagua_native` pipeline
Date: 2026-04-24

## 1. TL;DR (highest-leverage gaps)

1. **No long-edge splitting / dummy-node insertion in the default pipeline.**
   `InsertDummyNodes` exists as a registered op (used only by the opt-in
   `sugiyama` pipeline), but `dagua_native` lays out original nodes only.
   Consequence: an edge spanning k layers is one straight segment whose ends
   are the only degrees of freedom -- barycenter reorder cannot untangle it,
   and `edge_length_cv` (20% of composite) plus `edge_straightness` (10%)
   suffer on deep DAGs like `dependency_500` (-11), `extreme_mixed_width_transformer` (-5).
   Estimated composite gain: **+2 to +4 points on DAG-deep graphs**.

2. **Brandes-Köpf coordinate assignment is implemented but never called by
   the default pipeline.** `_brandes_koepf_x_positions` lives in
   `ops/coordinate.py` and is only reachable through `_CoordinateAssignment`
   in the `sugiyama` composable pipeline. `dagua_native` assigns x via
   gradient optimization + a post-hoc `BarycenterReorder`. BK gives exact,
   globally-optimal alignment of long-edge chains in O(V+E), which is
   precisely what the current gradient pipeline *cannot* do (a soft
   straightness loss under repulsion/overlap never converges to pixel-aligned
   dummy chains). Estimated composite gain: **+2 to +5 points on layered
   DAGs**, plus a runtime saving because fewer gradient steps are then
   necessary.

3. **Layer assignment is longest-path only; no network-simplex or Coffman-Graham
   alternative.** Longest-path produces maximally tall, narrow layouts (every
   sink sits at the deepest y). `dot`, `dagre`, and `ELK` default to
   network-simplex, which minimizes the sum of integer edge spans (equivalent
   to total vertical travel), yielding more compact layouts with lower
   `edge_length_cv`. `LayerPromotion` exists as a partial downward-shuffle but
   is not used. Estimated composite gain: **+1.5 to +3 points** on
   `dense_pair_50`, `hexagonal_lattice_42`, `sierpinski_42`, `dependency_500`.

4. **No transpose/exchange heuristic after barycenter reorder.**
   `TransposeHeuristic` exists (`ops/ordering.py:766`) and is used by the
   CPU-side Python init in `init_placement.py::_transpose_heuristic`, but the
   registered GPU-friendly op is never inserted into `dagua_native` after
   `BarycenterReorder`. `dagre` and `ELK` run greedy swap-if-improves passes
   and routinely beat pure barycenter by 10-20% on `edge_crossings`.

5. **Disconnected graphs are not decomposed.** Even though `DetectComponents`
   exists (`ops/preprocess.py:1339`), `dagua_native` lays out all components
   in one shared coordinate system. On `disconnected_label_cycle_collage`
   (dagua -13 vs elk) the components share spacing budget and repulsion
   "fights" force tiny cycles apart in odd ways. Per-component layout + tiling
   is a well-known standard and gets ELK its win here.

## 2. Framework Phase Comparison

| Phase | graphviz dot | dagre | ELK layered | dagua_native (default) | Gap? |
|-------|:---:|:---:|:---:|:---:|:---:|
| 1. Cycle removal | DFS + Greedy FAS | Greedy FAS | Greedy/DFS | DFS + Greedy FAS (cycle.py, sprint-19a) | OK |
| 2a. Layer assignment | Network simplex | Network simplex | Network simplex | Longest-path (Kahn, `layering.py:148`) | **MISSING** |
| 2b. Dummy-node split | Yes | Yes | Yes | **Not in default pipeline** (only in `sugiyama` side pipeline) | **MISSING** |
| 3a. Crossing reduction | Weighted median + barycenter | Barycenter + median + transpose | Barycenter + transpose | Barycenter only (`BarycenterReorder`, 8 iters) | **PARTIAL** |
| 3b. Transpose / swap | Yes | Yes | Yes | **Absent** in default (op exists, unused) | **MISSING** |
| 4. Coordinate assignment | Brandes-Köpf + QP | Brandes-Köpf | Brandes-Köpf + compaction | **Gradient descent + sort-then-barycenter** | **MAJOR GAP** |
| 5. Edge routing | Spline routing | Polyline + bundling | Orthogonal/spline | Straight lines (no ops in default) | Acceptable for metrics |
| 6. Aspect ratio | Implicit via compaction | Implicit | Explicit `aspectRatio` | `AspectRatioFit` (target=0.25) | OK |
| 7. Component decomp | Yes (cgraph) | Yes (sparsified) | Yes (padding) | **Absent** (DetectComponents unused) | **MISSING** |
| 8. Planar embedding | No (uses ortho) | No | Limited | **None** | Minor gap |

## 3. Findings

### Category A: Hierarchy-phase gaps (high impact)

---

**Finding A1: Long-edge chains are never dummy-split in the default pipeline**

- Severity: **HIGH** (directly moves `edge_length_cv` = 20% and `edge_straightness` = 10%)
- Evidence:
  - `dagua/layout/ops/pipelines/dagua_native.py:370-450` -- build_dagua_pipeline body shows the ops chain `NativeEngineInit -> Force2DInitIfFlat -> ... -> CreateOptimizer -> gradient_core -> BarycenterReorder -> OverlapProjection -> AspectRatioFit -> ClusterGridArrange`. No `InsertDummyNodes` anywhere.
  - `dagua/layout/ops/layering.py:278-370` -- `_expand_long_edges_with_dummy_nodes` and its registered op `InsertDummyNodes` exist and are exercised by `sugiyama` pipeline.
  - Consequence: on `hexagonal_lattice_42`, `sierpinski_42`, `dependency_500`, long edges cross many layers; the gradient edge-attraction loss pulls the two endpoints toward some midpoint, but no "virtual waypoints" on intermediate layers means the edge ends up slanted across the layer grid.
- Proposed fix:
  1. Add an op sequence in `build_dagua_pipeline` right after `NativeEngineInit` and before `gradient_core`:
     `InsertDummyNodes(InsertDummyNodesConfig(dummy_width=0.0, dummy_height=0.0))`, which writes an expanded edge_index + per-layer node list into `state.extras["expanded_graph"]`.
  2. Extend `SolveState` (or use `extras`) so `gradient_core`'s attraction/straightness losses operate on the expanded graph's edge_index for the duration of optimization, then collapse back to the original node set at the end.
  3. After `BarycenterReorder`, the dummy-node x-positions become the routed spline waypoints (currently stored but unused).
- Expected composite impact: **+2 to +4** on layered DAGs, notably
  `dependency_500` (currently -11), `extreme_mixed_width_transformer` (-5),
  `hexagonal_lattice_42` (-7), `sierpinski_42` (-6).
- Effort: **8-12 hours**. The op already works; the hard part is wiring
  `gradient_core` to consume the expanded adjacency transparently and then
  returning collapsed positions.

---

**Finding A2: Longest-path layering produces suboptimal layer height (no network-simplex)**

- Severity: **MED-HIGH** (affects `edge_length_cv` = 20%, runtime on deep graphs)
- Evidence:
  - `dagua/layout/ops/layering.py:148` -- `_longest_path_layering` uses Kahn with a heap; every node lands at its longest-path depth, which maximizes total vertical travel (each edge = layer_diff). dagre/dot minimize sum(layer_diff) via an integer LP solvable in polynomial time by network-simplex.
  - On a graph like `dense_pair_50` (lost -5 to dot), dot stacks pairs into tighter layers whereas dagua spreads them across all N/2 layers.
  - `LayerPromotion` (layering.py:530) exists but is NOT added to the pipeline. Promotion only pushes toward deep side; simplex can shift either direction for true minimization.
- Proposed fix:
  - Short term: add `LayerPromotion` to the dagua_native pipeline between `NativeEngineInit` and gradient loop (cheap, <50 LOC wiring, gives ~40% of the benefit).
  - Medium term: implement `NetworkSimplexLayering` op. A self-contained implementation using `networkx.network_simplex` on an auxiliary flow graph is ~150 LOC. GPU tensor version is harder but the original graph is already small (<=500 nodes benchmark).
- Expected composite impact: **+1 to +2** for `LayerPromotion`-only, **+2 to +3**
  for network simplex. Runtime: slight CPU cost on small graphs, net neutral.
- Effort: 2 hours for promotion wiring; 6-10 hours for network simplex.

---

**Finding A3: Brandes-Köpf x-coordinate assignment exists but is unused in default**

- Severity: **HIGH** (`edge_straightness` = 10%, runtime improvement)
- Evidence:
  - `dagua/layout/ops/coordinate.py:50-135` -- `_brandes_koepf_x_positions` is a full 4-alignment (ul/ur/dl/dr) BK implementation with horizontal compaction.
  - It is only invoked through `_CoordinateAssignment` in `ops/sugiyama.py`, registered only by the `sugiyama` pipeline.
  - The `dagua_native` pipeline instead does:
    (a) Initialize with `init_positions` (Python-loop barycenter on CPU for N<=100, tensor ops otherwise),
    (b) Gradient descent minimizing a soft straightness loss,
    (c) Permute within-layer x's to the existing set in barycenter order (`BarycenterReorder`).
    Step (c) CANNOT straighten long-edge chains because each layer's x set is fixed; BK picks x freely, subject only to non-overlap, so dummy-long-edge chains align to a single column exactly.
- Proposed fix:
  - Insert a `BrandesKoepfCoordinateAssignment` op AFTER `BarycenterReorder` that:
    1. Reads final within-layer ordering (from barycenter output).
    2. Runs BK to re-assign x-coords respecting node widths + `node_sep`.
    3. Writes back to `state.pos[:, 0]`. Y stays untouched.
  - For graphs with dummies (finding A1), BK places whole chains in a column. Without dummies, BK still straightens inner segments of 2-layer edges.
- Expected composite impact: **+2 to +5** on layered DAGs; may also let us
  reduce `steps` by 20-30% (less work for the gradient loss), giving **5-15%
  runtime savings**.
- Effort: 4-6 hours (op already exists, needs a thin wrapper that consumes
  `state.layer_index` + within-layer ordering from BarycenterReorder output).

---

**Finding A4: No transpose heuristic after barycenter reorder**

- Severity: **MED** (`crossing_rate` = 10% of composite)
- Evidence:
  - `dagua/layout/ops/ordering.py:766` -- `TransposeHeuristic` Op registered.
  - `init_placement.py:486-568` has a CPU Python `_transpose_heuristic` but it runs during INIT only, not during the post-gradient polish phase.
  - `BarycenterReorder` alone only guarantees local-optimum barycenters; classical Sugiyama literature (Gansner et al. 1993) shows adding a transpose sweep after barycenter typically removes 10-20% more crossings.
- Proposed fix: Insert `TransposeHeuristic(config=TransposeHeuristicConfig(iterations=3))` immediately after `BarycenterReorder` in `dagua_native`.
- Expected composite impact: **+0.5 to +1.5** points (directly on `crossing_rate`).
- Effort: **1-2 hours** (op exists, just wire it in).

---

**Finding A5: Disconnected graphs are laid out in one shared coordinate system**

- Severity: **MED-HIGH** (loss of -13 on `disconnected_label_cycle_collage`, -4 on `small_world_500` partially; affects `overlap_count`, `edge_length_cv`)
- Evidence:
  - `DetectComponents` op exists (`ops/preprocess.py:1339`), but no pipeline uses it in layering.
  - Repulsion loss between two unconnected components pushes them apart with an arbitrary scale, while attraction cannot pull them; the final bbox is noisy and cluster_separation suffers.
  - ELK handles this via `elk.spacing.componentComponent` and lays each component independently, then tiles.
- Proposed fix:
  - Add `ComponentDecomposeLayout` meta-op: run `DetectComponents`, then for each component (>=threshold_n) run `build_dagua_pipeline` on the subproblem, then tile components on a grid (use `ClusterGridArrange` logic generalized).
  - For `disconnected_label_cycle_collage` (n=7 total, 3 cycles), each tiny cycle is laid out separately = clean concentric arrangement.
- Expected composite impact: **+3 to +5** on the 3 disconnected/cyclic losses
  (`disconnected_label_cycle_collage`, plus tail on `small_world_*`).
- Effort: **6-10 hours**. Mostly orchestration, tile layout is easy.

---

### Category B: Within-phase quality gaps (medium impact)

---

**Finding B1: Barycenter uses mean-only (no median, no weighted)**

- Severity: LOW-MED
- Evidence: `ops/barycenter.py:111` -- `barycenter = torch.where(has_neigh, sum_x / count.clamp(min=1.0), current_x)`. Pure unweighted mean of neighbours.
- Note: `ops/ordering.py` has a separate `MedianSweep` op (line 687) that alternates barycenter + median, which is what dot + dagre do. That op is unused in default.
- Proposed fix: replace `BarycenterReorder` with a meta-op that alternates barycenter / median / barycenter sweeps (Gansner recommends median + averaging for reduced ties).
- Expected composite impact: **+0.3 to +0.8** on crossings.
- Effort: 2-3 hours.

---

**Finding B2: No weighted barycenter for dummy-node chains**

- Severity: LOW (until A1 lands, then MED)
- Evidence: contingent on dummy-node insertion.
- Proposed fix: after A1, weight dummy-node neighbours higher than real-node neighbours in barycenter averaging -- dagre does this implicitly by assigning dummy nodes a fixed "port" position.

---

**Finding B3: init_placement.py still uses Python-loop barycenter for N<=100**

- Severity: LOW (runtime only)
- Evidence: `init_placement.py:148-196` -- for N <= 100, runs 15-40 Python-loop barycenter passes with `sum/len` and `sort()`. For N<=100 this is maybe 1-5ms; for small-benchmark cold start it compounds across 93 graphs.
- Proposed fix: unify paths -- always use the tensor-based `_init_positions_vectorized`. The threshold `num_nodes > 100` was kept for historical reasons per the comment; tensor ops are faster even at N=20.
- Expected impact: **5-10% wall clock savings** on the small-graph benchmark portion.
- Effort: 1-2 hours + regression testing.

---

**Finding B4: `_spread_fanout_children` runs in CPU Python loops**

- Severity: LOW (runtime)
- Evidence: `init_placement.py:571-622`. Sorts by `.item()` calls, writes positions with `.item()` / index assignment.
- Proposed fix: vectorize using `scatter_add_` + `argsort`.
- Effort: 2 hours. Impact: negligible on composite, ~1-2% runtime.

---

**Finding B5: BarycenterReorder forces within-layer x-positions to the SAME SET**

- Severity: MED (`edge_straightness` = 10%)
- Evidence: `ops/barycenter.py:202-211` --
  `sorted_x, _ = torch.sort(current_x); pos_new[new_member_order, 0] = sorted_x`.
  This is intentional (preserves overlap), but it means a wide layer immediately
  below a narrow layer keeps its wide spread, even though most edges should
  converge to a single x. BK assignment (A3) solves this natively.
- Proposed fix: superseded by A3 (BK assignment).

---

### Category C: Runtime inefficiencies

---

**Finding C1: `init_placement.py:75,96,133,134,206,589,590` -- repeated `.tolist()` / `.cpu()` conversions**

- Severity: LOW
- Evidence: 6+ calls that transfer tensors off GPU mid-pipeline when the graph hasn't moved. Each `tolist()` on E=5000 edges is ~0.2ms of overhead, compounded over per-call invocations.
- Proposed fix: keep CPU tensors as CPU tensors through this helper; avoid moving to CUDA for the final `positions` result when the caller wants it on CUDA.
- Expected impact: 2-5% runtime.

---

**Finding C2: `_greedy_fas` in `cycle.py` is O(V^2) Python**

- Severity: LOW (only fires on cyclic graphs)
- Evidence: `cycle.py:162-179` -- outer loop over N, inner over remaining nodes = O(V^2) scan per step. Eades original is O(V+E) with two linked lists.
- Proposed fix: replace with Eades linked-list version. On `small_world_500` (currently -4) this could save 50-100ms on each layout call.
- Effort: 4 hours (careful to keep FAS identity mapping).

---

**Finding C3: Default `steps` budget is large on small graphs**

- Severity: LOW-MED (runtime)
- Evidence: `resolve.py:68-84`. `auto_layout_steps(10)=50`, `50->100`, `200->150`. Given the soft-loss convergence, 50% of the final quality is usually reached in the first 30% of steps. With BK coordinate assignment (A3) replacing the gradient effort for x-straightness, we should be able to halve step budgets for layered DAGs.
- Proposed fix: after A3 lands, add a path that runs 50% of the step budget when the graph classifies as TREE/DAG with >1 layer.
- Expected impact: **25-40% runtime savings** on DAGs once A3 is in.

---

**Finding C4: `BarycenterReorder` creates a `member_to_idx` full-N tensor for every layer on every iteration**

- Severity: LOW
- Evidence: `ops/barycenter.py:95-97` -- inside the per-layer loop, `torch.full((N,), -1)` then `member_to_idx[layer_members] = arange(n_members)`.
  For 8 iterations * L layers * N allocation = 8L full-N tensors allocated.
- Proposed fix: allocate `member_to_idx` once outside the iteration loop, reset in-place.
- Impact: small, <1% runtime. Good hygiene.

---

### Category D: Untouched corners

---

**Finding D1: No planar-aware layout for hexagonal_lattice / sierpinski**

- Severity: MED (-7, -6 losses)
- Evidence: hexagonal_lattice_42 and sierpinski_42 are planar graphs.
  dot exploits their planarity implicitly via network-simplex + BK;
  dagua treats them as arbitrary DAGs.
- Proposed fix: detect planar structure via `graph_classify` (low max-degree,
  small treewidth heuristic). When detected, use Tutte's barycentric
  embedding as an alternative initializer (3 lines with torch.linalg.solve
  on the Laplacian with boundary fixed). Then gradient-polish with heavily
  down-weighted repulsion.
- Expected impact: **+3 to +5** on these two graphs specifically.
- Effort: 8 hours (Tutte embedding op + detection logic).

---

**Finding D2: Cyclic-graph 2D init fallback randomizes y but keeps x from barycenter**

- Severity: MED (small_world_100 loses -8, recurrent_feedback_cell loses -7)
- Evidence: `dagua_native.py:386-395` -- `Force2DInitIfFlat` randomizes y when
  num_layers <= 1. But for cycles that HAPPEN to produce a layering (cycle
  pre-pass from sprint-19a), the y is forced into a contrived layer order
  that fights the cycle's natural circular embedding.
- Proposed fix: when `structure.family == CYCLE` or is strongly connected
  with low edge/node ratio, use a CIRCULAR init (one big circle of N points)
  and rely on force-directed polish rather than Sugiyama layering. This is
  what ELK does for `SMALL_WORLD`-like graphs (competitor sugiyama wins here
  because it falls back to igraph's radial).
- Expected impact: **+3 to +6** on `small_world_100`, `recurrent_feedback_cell`,
  and `small_world_500`.
- Effort: 4-6 hours (add `CircularInitIfCyclic` op and a classification
  predicate).

---

**Finding D3: No edge-bundling for dense graphs**

- Severity: LOW (visual only, doesn't hit composite metric directly, but
  could ease crossing_rate by implicit effect).
- Skip for this sprint.

---

**Finding D4: Convergence criteria rely on global loss stall**

- Severity: LOW
- Evidence: `resolve.py:142-148` -- `rel_threshold=1e-4`, stall_limit=3 to 5.
  The loss is a weighted sum of 11+ terms; when one term plateaus but another
  is still improving, the stall fires too soon on noisy losses.
- Proposed fix: monitor per-term loss reductions; stall only when ALL active
  terms plateau.
- Expected impact: tiny on composite (+0.1), small runtime cost (-2-5%).
- Effort: 2 hours.

---

**Finding D5: No incremental/caching layout for repeat calls**

- Severity: N/A for benchmark (each call is a cold start). Skip.

---

## 4. Action Queue (ordered by impact / effort)

| Rank | Finding | Impact | Effort (hrs) | Ratio |
|:---:|---|---|:---:|:---:|
| 1 | A4: Wire `TransposeHeuristic` after `BarycenterReorder` | +0.5 to +1.5 | 1-2 | **Highest** |
| 2 | A2a: Add `LayerPromotion` op to default pipeline | +1 to +2 | 2 | **Very high** |
| 3 | A3: Brandes-Köpf coordinate assignment after BarycenterReorder | +2 to +5 | 4-6 | **Very high** |
| 4 | A1: Dummy-node insertion + gradient_core on expanded graph | +2 to +4 | 8-12 | **High** |
| 5 | A5: Component decomposition + tile | +3 to +5 | 6-10 | **High** |
| 6 | D2: Circular init for strongly-connected cyclic graphs | +3 to +6 | 4-6 | **High** |
| 7 | A2b: Full network-simplex layering | +2 to +3 | 6-10 | **Medium** |
| 8 | D1: Tutte barycentric init for planar graphs | +3 to +5 (narrow) | 8 | **Medium** |
| 9 | B1: Alternating median + barycenter sweeps | +0.3 to +0.8 | 2-3 | **Medium** |
| 10 | C3: Halve step budget for DAGs post-A3 | Runtime -25-40% | 2 | **Medium (runtime)** |
| 11 | B3: Drop Python-loop path from init_placement | Runtime -5-10% | 1-2 | **Medium (runtime)** |
| 12 | D4: Per-term stall detection | +0.1 | 2 | Low |
| 13 | B4, C1, C2, C4: Micro-optimizations | Runtime -5-10% | 8 combined | Low |

## 5. Recommended Sprint 19 Focus

Ship in order, each as a standalone sprint task with a before/after composite
benchmark:

1. **sprint-19c**: TransposeHeuristic + LayerPromotion (3 hours, ~1.5 points).
   Both exist as ops, pure wiring change. Low-risk baseline win.
2. **sprint-19d**: Brandes-Köpf coordinate assignment (6 hours, ~3 points).
   Standalone value even without dummy nodes; the BK op already works for the
   sugiyama pipeline. Will reduce gradient pressure so we can also halve the
   step count after verifying quality (C3).
3. **sprint-19e**: Dummy-node insertion (10 hours, ~3 points). Gives BK the
   long-edge chains it needs. This is where the biggest dependency_500 /
   hexagonal / sierpinski wins land. Requires careful SolveState changes and a
   "collapse to original nodes" final op.
4. **sprint-19f**: Component decomposition (8 hours, ~4 points on the 2 loss
   graphs where it applies).
5. **sprint-19g**: Circular init for cyclic graphs (5 hours, ~4 points on
   small_world / recurrent_feedback_cell).

Expected cumulative composite gain: **+10 to +15 points** on the mean, with
the biggest wins concentrated precisely on the current loss graphs. This
would push dagua from 77.29 to ~87-92, clearly dominant instead of narrowly
leading.

## 6. Cross-cutting notes

- The `sugiyama` pipeline (`ops/pipelines/sugiyama.py`) contains ~70% of the
  missing algorithmic phases already, just composed as "the whole pipeline"
  rather than "a bucket of insertable ops." The refactoring effort to extract
  its steps as opt-in insertions for `dagua_native` is modest and would
  unlock findings A1, A3, A4 cheaply.
- Several sprint comments in `dagua_native.py` explicitly note the absence of
  these phases as deferred work (e.g. Sprint 1 FamilyConditionalInit reverted,
  Sprint 2 V-cycle not production-ready). Sprint 19 can move those back into
  scope now that the baseline is stronger.
- Brandes-Köpf + dummy-node insertion together is the standard Graphviz-dot
  architecture. Dagua has all the pieces; they're just not wired. This is
  the single highest-leverage change available.
- Cycle handling (sprint-19a/b) is already strong; don't regress it. The
  circular-init path (D2) should fire BEFORE layering for truly cyclic
  graphs rather than after layering detects skew.
