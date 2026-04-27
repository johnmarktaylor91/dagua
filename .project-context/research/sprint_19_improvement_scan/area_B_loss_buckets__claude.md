# Area B -- Per-graph Loss Root Causes

Agent: Claude Opus 4.7 (1M)
Date: 2026-04-24
Scope: the 10 worst-loss graphs at the head of sprint-19a/b.

## 1. TL;DR

Three buckets dominate, all traceable to the Sprint 18 spacing / aspect-ratio bump:

1. **AR=0.25 over-stretching is the single biggest unforced error.** Every one of
   the 9 graphs examined is being pinned to the same 1:4 width/height target --
   even a 5-node cycle (`recurrent_feedback_cell`) and a planar lattice
   (`hexagonal_lattice_42`). Because the AR op scales the shorter axis to match
   target=0.25 with tolerance=0.55, almost everything becomes extremely tall and
   narrow. That kills `edge_length_cv` (20 pts of weight) and
   `angular_res_mean_deg` (5 pts) on any graph whose natural aspect is square or
   wider. Fix: topology-aware target. Expected composite gain: **+3 to +5 points
   on at least 5 of the 10 loss graphs** (hex, sierpinski, dense_pair,
   recurrent_feedback_cell, extreme_mixed_width_transformer).

2. **Cyclic-graph handling degenerates into one-node-per-layer.** `small_world_100`
   runs with 100 unique y-levels for 100 nodes (every node on its own row),
   because the cycle-reversal fallback in `init_placement.py` builds a
   post-FAS DAG whose longest-path layering is essentially a chain. Then
   `Force2DInitIfFlat` doesn't trigger (num_layers>1 is no longer the
   degenerate case we guard), so the optimizer inherits a linear chain
   init. Result: `dag_consistency` = 0.52 (basically random for a cyclic
   graph), while `igraph_sugiyama` gets 0.985. Fix: when cycle reversal
   produces 1 layer per node, fall through to flat/force-directed init.
   Expected gain: **+7 to +12 points** on small_world_100 and small_world_500.

3. **Disconnected components are routed through a single layer-pile.** The 3
   components of `disconnected_label_cycle_collage` share one y-coordinate
   plane; the 3-cycle component at y=624 has all three nodes at the same y,
   so its 3 edges are pure horizontal lines (edge_straightness_mean_deg
   jumps from dagua 47.6 deg to elk 3.77 deg). No per-component layer
   rebasing exists in `dagua_native.py`. Fix: detect `num_components > 1`
   and lay out each component independently, then tile. Expected gain:
   **+10 points on disconnected_label_cycle_collage** (half of the 13-pt
   gap).

Secondary buckets: deep-sparse-DAG layering for dependency_500 and dense
DAG crossings for dense_pair_50.

## 2. Per-graph breakdown

Metric weights recap (from `dagua/metrics.py:composite` L1147):
dag_consistency 25, edge_length_cv 20 (lower better), depth_spearman 15,
overlaps 10, edge_straightness 10 (lower better), crossing_rate 10,
angular_resolution 5, cluster_sep 5.

### 2.1 disconnected_label_cycle_collage  dagua 62.08 vs elk 75.19 (-13.11)

Structure: n=7, e=6, components=3, cyclic, planar, density=0.143.
Components: (0->1), (2->3), and a 3-cycle (4->5->6->4) with a self-loop (6->6).

| metric | dagua | elk | delta |
|---|---|---|---|
| dag_consistency | 0.500 | 0.667 | -0.167 |
| edge_length_cv | 0.625 | 0.628 | +0.004 |
| depth_spearman_rho | 0.971 | 0.962 | +0.009 |
| edge_straightness_mean_deg | **47.61** | **3.77** | **-43.84 (HUGE)** |
| crossing_rate | 0.000 | 0.000 | 0 |
| angular_res_mean_deg | 59.999 | 56.23 | +3.77 |
| overlap_count | 0 | 0 | 0 |

Root cause: edge_straightness collapse. Dagua's positions:
```
0: (-155.8, 2.4)    # component A top
1: (-70.0, 309.6)
2: (67.6, 2.4)      # component B top
3: (67.4, 309.6)
4: (-147.0, 624.0)  # component C all at y=624
5: (1.0, 624.0)
6: (148.0, 624.0)
```
All three nodes of the cycle component end up on the same y-level (624).
The 3 cycle edges then become pure horizontal lines. edge_straightness is
measured as deg from the layer axis (vertical), so horizontal edges =
~90deg, badly weighted.

ELK lays out each component in its own rectangle with its own vertical
stack:
```
0: (12.0, 12.0)
1: (12.0, 112.0)    # component A vertical
2: (152.0, 12.0)
3: (152.0, 112.0)   # component B vertical
4: (42.0, 272.0)
5: (22.0, 372.0)    # component C vertical stack
6: (22.0, 172.0)    # even the 3-cycle is stacked
```

Fix: insert a component-aware pre-pass. For `num_components > 1`:
1. Split edge_index into per-component subgraphs.
2. Run layering + barycenter on each independently, so each component
   produces a stack of coords with multiple y-levels.
3. Tile (horizontally or in a grid) with a gap.

Proposed op: `ComponentTileLayout`. Could be as simple as calling the
existing init_positions per component and then offsetting x.

Expected impact: +8 to +10 pts on this graph (straightness score moves
from ~0 to near elk's level, and dag_consistency improves because
per-component layering gives 2+ layers per cycle).

### 2.2 dependency_500  dagua ~51.96 vs elk 62.82 (-10.86)

Structure: n=500, e=1470, 2 components, DAG, max_depth=18, NOT planar,
max_in_deg=3, max_out_deg=53, density=0.0059.

(I launched dagua on this graph but it did not complete within the diagnostic
budget on this machine -- 18+ minutes still running at last check. The
CONTEXT already records composite=51.96 from the full benchmark, and the
best competitor is elk_layered at 62.82.)

What we know from the existing benchmark and from inspecting other large
DAGs:

- 2 connected components: the pipeline has no per-component treatment, so
  both components share a coordinate frame with one global set of y-levels.
- Very wide fan-out (53 out-degree from a single node): BarycenterReorder
  with 8 passes is unlikely to untangle 53 children routed across depth 18.
- AR=0.25 target means h=4w; at n=500 that is a very tall, narrow layout.
  With rank_sep=240 and 18+ layers, h ~ 4500. Edge lengths vary wildly
  because many edges traverse 1 layer (short) while fan-out edges cross
  10+ layers (long). Hence edge_length_cv is almost certainly the biggest
  loss here.
- Runtime: 18+ minutes for one layout of a 500-node DAG is a red flag
  separate from the quality issue; the barycenter phase at n=500 is 40
  passes of O(sum(degree)) Python (from `_init_positions_vectorized`
  barycenter_order fallback) and OverlapProjection with 10 iterations at
  n=500 makes a 500x500 Python overlap loop. These are both runtime
  concerns for area D, not quality.

Primary root-cause hypothesis: (a) multi-component sharing one layering +
(b) AR=0.25 stretch amplifying edge_length variance on a max_out_deg=53
hub.

Fix (composite of 1 + multi-component treatment):
1. Per-component layering/barycenter.
2. Reduce rank_sep (or equivalently raise target_aspect) when max_depth *
   rank_sep >> 4 * max_layer_width * node_sep.
3. For hubs with >= 20 children, split the hub column into a grid instead
   of a single row -- a dedicated `HubFanSplit` op.

Expected impact: +6 to +9 pts (half the 11-point gap; elk benefits from a
purpose-built layered algorithm tuned for exactly this shape).

### 2.3 small_world_100  dagua 49.19 vs igraph_sugiyama 57.09 (-7.90)

Structure: n=100, e=200, 1 component, cyclic, planar, density=0.020.

| metric | dagua | sugiyama | delta |
|---|---|---|---|
| dag_consistency | **0.520** | **0.985** | **-0.465 (HUGE)** |
| edge_length_cv | 0.739 | 3.727 | **+2.99 (dagua better)** |
| depth_spearman_rho | nan | nan | N/A |
| edge_straightness_mean_deg | 4.51 | 22.67 | dagua better |
| crossing_rate | 0.0066 | 0.000 | -0.0066 |
| angular_res_mean_deg | **1.00** | **42.29** | **-41.29 (HUGE)** |
| overlap_count | 0 | 0 | 0 |

Dagua wins edge_straightness (because every edge is nearly vertical --
layout is 11.7k wide x 46.8k tall, extreme column!) and wins
edge_length_cv. But it loses dag_consistency AND angular_res.

Root cause: the cycle-reversal fallback produces 100 unique y-levels
(verified: `unique y levels: 100`). After FAS the graph is now a chain
(every cycle-edge reversed), so longest-path layering puts each node
alone on its own layer. The init positions are a column of n=100 dots
separated by rank_sep=240 -> h=23760; then AR=0.25 stretches to
w=5940, h=23760. Optimizer runs from there but can't recover because the
spring + attract forces dominate and keep nodes near their init.

Additionally the FAS reversal destroys dag_consistency for a cyclic
graph: after you reverse back edges, the "semantic" direction of the
cycle is gone, and depth_spearman_rho is undefined (nan -> 0 in composite).

Sugiyama from igraph keeps the cycle's natural flow and gets
dag_consistency=0.985 by picking a good acyclic spanning ordering.

Fix 1: in `init_placement.py` L100-108, add a guard rejecting the
cycle-reversal re-layering when `relayered_max == 1 and n_relayered == N`
(one node per layer, the degenerate chain). In that case, fall back to
the original single-layer collapse -> Force2DInitIfFlat fires -> 2D
random init -> optimizer converges on a force-directed layout.

Fix 2: for cyclic flat graphs, switch the whole pipeline to a
force-directed variant (FR or KK). The pipeline already ships FR
primitives; graph_classify produces `is_cyclic` info -- route cyclic-flat
graphs to `layout_fr_pipeline` instead of `dagua_native_pipeline`.

Expected impact: +6 to +10 pts on small_world_100 and small_world_500
(both hit the same chain-collapse).

### 2.4 recurrent_feedback_cell  dagua 69.05 vs igraph_sugiyama 69.41 (-0.36)

Structure: n=5, e=6, 1 component, cyclic (tiny), planar, density=0.3.

Note: the CONTEXT file quoted 62.56 for dagua; current head measures
**69.05**, so a prior sprint (likely 19a cycle-reversal pre-pass) already
closed ~6.5 pts on this graph. Remaining gap is 0.36, essentially noise.

| metric | dagua | sugiyama | delta |
|---|---|---|---|
| dag_consistency | 0.667 | 0.667 | 0 |
| edge_length_cv | 0.764 | 0.810 | +0.05 |
| depth_spearman_rho | 0.894 | 0.894 | 0 |
| edge_straightness_mean_deg | 14.68 | 8.86 | -5.82 |
| crossing_rate | 0 | 0 | 0 |
| angular_res_mean_deg | 54.56 | 51.64 | +2.92 |
| overlap_count | 0 | 0 | 0 |

Only remaining miss: edge_straightness 14.68 vs 8.86 (tilted edges; worth
~1.3 of the 10-pt straightness budget). Caused by AR=0.25 stretch
(dagua w=312 h=1248 ratio 0.25, sugiyama w=25 h=200 ratio 0.125).
Counter-intuitively sugiyama has a tighter w so its within-layer angles
are smaller; dagua's AR stretch widened x causing tilted edges.

Fix: remove this graph from the "focus on" list -- it is essentially
closed. If further gain is desired, lower target_aspect to 0.125 just
for tiny cyclic graphs (n <= 10, cyclic). Expected impact: +0.3 pts.

### 2.5 hexagonal_lattice_42  dagua 82.42 vs graphviz_dot 88.99 (-6.57)

Structure: n=42, e=53, 1 component, DAG, max_depth=11, planar,
density=0.031.

| metric | dagua | dot | delta |
|---|---|---|---|
| dag_consistency | 1.000 | 1.000 | 0 |
| edge_length_cv | **0.583** | **0.099** | **-0.484 (HUGE, 9.7 pts)** |
| depth_spearman_rho | 0.998 | 0.823 | +0.175 (dagua wins) |
| edge_straightness_mean_deg | 11.37 | 17.42 | dagua better |
| crossing_rate | 0.0086 | 0.000 | -0.0086 |
| angular_res_mean_deg | **58.58** | **78.83** | **-20.24 (-2.5 pts)** |
| overlap_count | 0 | 0 | 0 |

Dagua wins straightness and depth_spearman but loses huge on
edge_length_cv (0.583 vs 0.099). Also loses 2.5pts on angular_res.

Layout dimensions:
- dagua: w=456 h=2640 AR=0.173 (almost at AR target 0.25 bound)
- dot:   w=432 h=792  AR=0.545 (natural hex honeycomb)

Dagua has 15 y-levels, dot has 12. Dagua's edge lengths vary from 239.6
to 822.6 (4x range!) because vertical edges span rank_sep=240 but some
edges after AR-stretch reach cross-layer. Dot has edge lengths 72 to
102 (1.4x range) because it uses near-unit honeycomb geometry.

Root cause: the AR=0.25 stretch is fighting hex-lattice geometry. A
hex cell has natural 2:sqrt(3) ~ 1.15 aspect ratio; forcing 0.25 tall
means vertical edges are ~5x longer than horizontal ones.

Secondary: longest-path layering uses 12 layers on the natural hex (dot
matches) but the dagua re-layering via cycle-reversal-fallback checked
in above produced 15 layers -- but wait, hex is a DAG so that path
shouldn't trigger. Checking: hex has n_layers=12, max_per_layer<=0.5N,
so cycle-reversal path is NOT triggered. But dagua still reports 15
unique y-levels in the FINAL layout. Likely the extra 3 come from
post-optimizer gradient drift: the continuous optimizer nudges nodes off
their initial integer layers.

Fix 1: higher target_aspect for planar DAGs. Detect via
`is_planar` in StructureInfo + `num_components==1 and is_acyclic` and
set target_aspect=1.0 (near-square). For hex at w=456, dagua would scale
h from 2640 -> 456, compressing all the y-range into something matching
dot's natural 792. Edge_length_cv would drop from 0.583 to ~0.20
(still worse than dot's 0.099 but +~7.7 pts in composite).

Fix 2: run a secondary barycenter pass that also rebalances Y (not just
X). Currently BarycenterReorder only reorders x. For a hex lattice the
natural y-spacing between rows should be ~ node_sep * sqrt(3)/2, NOT
rank_sep=240.

Expected impact: +4 to +6 pts via Fix 1 alone (reduces edge_length_cv,
improves angular_res).

### 2.6 sierpinski_42  dagua 78.35 vs graphviz_dot 84.29 (-5.94)

Structure: n=42, e=81, 1 component, DAG, max_depth=22, planar,
density=0.047.

| metric | dagua | dot | delta |
|---|---|---|---|
| dag_consistency | 1.000 | 1.000 | 0 |
| edge_length_cv | **0.526** | **0.353** | -0.173 (-3.5 pts) |
| depth_spearman_rho | 0.999 | 0.994 | 0 |
| edge_straightness_mean_deg | 22.44 | 24.93 | 0 |
| crossing_rate | 0.0024 | 0.000 | -0.0024 |
| angular_res_mean_deg | **12.91** | **35.87** | **-22.97 (-2.9 pts)** |
| overlap_count | 0 | 0 | 0 |

Same pattern as hex:
- dagua: w=1320 h=5280 AR=0.250 (pinned to target)
- dot:   w=533  h=1584 AR=0.336

Dagua has 36 y-levels, dot has 23. The triangular fractal structure
naturally nests, but dagua expands max_depth=22 layers into 36 after
optimization gradient nudges and ARfit stretches. edge_length_cv 0.526
is 5x dot's 0.099.

Root cause: same as hex -- AR=0.25 too aggressive, layers over-separated.
Angular resolution also suffers because triangles squeezed into a tall
narrow column have tiny horizontal separations between neighbours so
angles are acute.

Fix: same as hex -- planar-aware target_aspect=1.0.

Expected impact: +3 to +5 pts.

### 2.7 extreme_mixed_width_transformer  dagua 73.82 vs dot 77.99 (-4.17)

Note: the CONTEXT listed 78.49 dagre as best, but current dot measurement
is 77.99. Delta is -4.17.

Structure: n=10, e=12, 1 component, DAG, max_depth=7, planar, density=0.133.

| metric | dagua | dot | delta |
|---|---|---|---|
| dag_consistency | 1.000 | 1.000 | 0 |
| edge_length_cv | **0.841** | **0.708** | -0.133 (-2.7 pts) |
| depth_spearman_rho | 1.000 | 1.000 | 0 |
| edge_straightness_mean_deg | 7.80 | 24.05 | dagua better |
| crossing_rate | **0.0513** | **0.000** | **-0.0513 (-5.1 pts)** |
| angular_res_mean_deg | 94.80 | 96.11 | -1.31 |
| overlap_count | 0 | 0 | 0 |

Layout: dagua w=270 h=2184 (AR 0.123, just outside lower tolerance),
dot w=162 h=504 (AR 0.321). Mixed-width transformer has some layers
with 1 node and some with 3 -- the wide middle layers get pushed
together and cause a crossing.

Main miss: 1 crossing that dot avoids. With only 12 edges, one crossing
costs 5.1 composite pts. Root cause: barycenter reorder with 8 iterations
fails to resolve a single crossing in a mixed-width graph; the gradient
optimizer doesn't explicitly penalize crossings.

Fix: increase `BarycenterReorder` iterations for small graphs (N <= 20),
or add a post-crossing-minimization pass that does explicit swap trials.
For a graph with 12 edges, a brute-force O(L * max_layer_width^2) swap
pass is trivial.

Expected impact: +3 to +5 pts (fixing the crossing alone is +5pts).

### 2.8 dense_pair_50  dagua 71.81 vs graphviz_dot 76.33 (-4.52)

Structure: n=50, e=208, 1 component, DAG, max_depth=49, NOT planar,
density=0.085.

| metric | dagua | dot | delta |
|---|---|---|---|
| dag_consistency | 1.000 | 1.000 | 0 |
| edge_length_cv | 0.473 | 0.454 | -0.019 |
| depth_spearman_rho | 1.000 | 1.000 | 0 |
| edge_straightness_mean_deg | **45.49** | **31.80** | -13.7 (-3.0 pts) |
| crossing_rate | **0.0177** | **0.0076** | -0.0101 (-1.0 pt) |
| angular_res_mean_deg | 4.18 | 5.86 | -1.7 |
| overlap_count | 0 | 0 | 0 |

Layout: dagua w=2940 h=11760 (AR 0.250), dot w=671 h=3528 (AR 0.190).

max_depth=49 with n=50 means it is essentially a near-linear DAG chain
with 208 cross-edges -- a very deep, very thin structure. Dot's 0.190
is already close to 0.25, so AR fit fires only mildly. But dagua ends
up 4.4x wider than dot (2940 vs 671) despite matching AR -- because
dagua's rank_sep=240 produces h=11760 and then AR 0.25 sets w=2940.
Dot uses much more compact node-to-node layer geometry.

Main miss: edge_straightness 45.5deg (barely counted as "straight")
because many of the 208 non-DAG-chain edges cross multiple layers
diagonally. With 49 layers and 2940 x-range, a cross-edge skipping 3
layers still comes out mostly horizontal, but the average is near
45deg.

Fix: compress rank_sep specifically for deep chains (max_depth > N/2),
or compress x-range via a sub-barycenter pass that pulls chain-aligned
nodes to x=0.

Expected impact: +2 to +4 pts.

### 2.9 small_world_500  dagua 49.81 vs elk_layered 54.26 (-4.45)

Structure: n=500, e=1500, 1 component, cyclic, non-planar, density=0.006.

| metric | dagua | elk | delta |
|---|---|---|---|
| dag_consistency | **0.492** | **0.995** | **-0.503 (-12.6 pts)** |
| edge_length_cv | 0.701 | 7.630 | **+6.9 (dagua better)** |
| depth_spearman_rho | nan | nan | N/A |
| edge_straightness_mean_deg | 2.38 | 17.47 | dagua better |
| crossing_rate | 0.0048 | 0.0010 | -0.004 |
| angular_res_mean_deg | **0.28** | **6.82** | -6.54 (-0.8 pts) |
| overlap_count | 0 | 0 | 0 |

Same story as small_world_100: cyclic flat graph, cycle-reversal collapse
produces a chain, dag_consistency cratered. Dagua wins straightness and
edge_length_cv (by pinning everything on a single column) but loses
12.6pts on dag_consistency because the "hierarchy" is meaningless for a
ring-lattice.

Fix: same as 2.3 -- reject degenerate one-per-layer cycle-reversal, fall
back to force-directed / Force2DInitIfFlat path.

Expected impact: +4 to +8 pts.

### 2.10 recurrent_ffn_block  MISSING

Not present in the benchmark -- `recurrent_ffn_block` is not in
`get_test_graphs()`. Skipping.

## 3. Bucket synthesis

### Bucket A: Aspect-ratio-induced damage (6 of 10 graphs)

Graphs: hexagonal_lattice_42, sierpinski_42, dense_pair_50,
extreme_mixed_width_transformer, recurrent_feedback_cell (marginally),
and dependency_500 (partially). Also affects disconnected_label indirectly
by forcing the within-layer y-positions.

Symptom: every layout normalized to w/h = 0.25 regardless of input shape.
AR fit fires on anything outside [0.1125, 0.556]. `target_aspect=0.25`
was tuned on the full benchmark mean but kills specific topologies.

Root cause: `AspectRatioFit` uses a single target for all graphs. The
commit history (`Sprint 18h`) shows a linear sweep that settled on 0.25
because it maximized the *mean*, while the variance on specific graphs
is still large and shows up as the loss list.

Proposed fix:
- Add topology-aware target in `AspectRatioFit`:
  - Planar DAGs with max_depth <= sqrt(N):  target=1.0  (square)
  - Planar lattices (hex, grid detected via degree distribution): target=1.0
  - Deep DAGs (max_depth > N/3): keep current behavior OR target=0.35
  - Disconnected (components>1): square the whole bounding box (target=1.0)
  - Cyclic flat (cyclic AND force-directed path): no AR fit (ideal depends
    on structure; let the force algo find it)
- Implementation: read `problem.structure_info.is_acyclic`, `num_components`,
  `is_planar_hint`, and compute target. Plumb via existing StructureInfo.

Expected composite impact: +4 to +6 pts on each of hex_lattice_42,
sierpinski_42, dense_pair_50, with no regression on the currently-winning
graphs (because their target_aspect ~= 0.25 is still the "deep DAG" case).
Net expected improvement across 93-graph suite: +1 to +1.5 pts mean.

Priority: HIGHEST. One-file change (`dagua/layout/ops/postprocess.py` +
expose structure info from LayoutProblem). Effort: 2-4 hours.

### Bucket B: Cyclic-flat collapse to chain (2 graphs)

Graphs: small_world_100, small_world_500.

Symptom: after cycle reversal (which triggers because longest-path
layering of original cyclic graph is degenerate), the graph becomes a
chain-like DAG -> longest-path produces 1 node per layer -> init is a
column, optimizer stuck.

Root cause: `init_placement.py` L100-108 accepts the cycle-reversal
re-layering as long as `pile_reduced and not_degenerate and
gained_layers`. But for a small-world graph, breaking the ring produces
a DAG with max_depth=N, i.e. one node per layer. That passes
`not_degenerate` (relayered_max==1 is accepted when N<=10 only -- but
the code says `relayered_max >= 2 or num_nodes <= 10` so for N=100 this
guard already fires, relayered_max must be >=2).

Let me re-check: a small-world ring with 100 nodes and p=0.1 rewiring.
FAS reverses back-edges -> resulting graph has few cross-edges. Longest
path through that DAG... may actually be close to 100 layers. With 100
nodes in 100 layers, relayered_max = 1 (because each layer has 1 node).
So `not_degenerate` = `1 >= 2 or N <= 10` = False for N=100. Hmm, so
this SHOULD reject. But the observed 100 y-levels suggest otherwise.

What's likely happening: longest-path after FAS creates say 95 layers,
with a few small layers of 2-3 nodes and most layers of 1 node. Then
relayered_max=2 or 3 (passes not_degenerate). So the guard fails.

Proposed fix: tighten the `not_degenerate` check -- also reject when
`n_relayered / num_nodes > 0.8` (more than 80% of nodes in their own
layer is meaningless). If rejected, fall back to single-layer collapse ->
Force2DInitIfFlat -> 2D random init -> gradient optimizer converges to
something circular (since small-world IS a ring).

Alternative: route cyclic graphs (`is_acyclic=False and num_components=1`)
through the FR pipeline instead of `dagua_native`. Worth testing on the
benchmark.

Expected composite impact:
- small_world_100: +6 to +10 pts (dag_consistency 0.52 -> ~0.85)
- small_world_500: +4 to +6 pts (dag_consistency 0.49 -> ~0.80)

Priority: HIGH. 1-line tightening of threshold. Effort: 30 min + test.

### Bucket C: Disconnected components (1 graph)

Graph: disconnected_label_cycle_collage.

Symptom: 3 components share one y-axis; in-cycle component collapses all
3 nodes to one y-level -> horizontal edges -> edge_straightness=47.6deg.

Root cause: no component-aware op in `dagua_native` pipeline. The
layering+barycenter runs on the whole edge_index as a single graph.

Proposed fix: add `ComponentTileLayout` op at the front of the pipeline.
For each weakly-connected component:
1. Extract subgraph (node indices + edges).
2. Run the rest of the pipeline on the subgraph (recursive invocation, or
   at minimum run init_positions on it).
3. Tile components horizontally (with a gap equal to 2 * node_sep).

Existing code: `graph_classify._count_components_and_acyclic` already
computes component count. Need a component-labeling function that returns
`component_id[N]` -- NetworkX connected_components on the undirected
version is fine.

Expected composite impact: +8 to +10 pts on this graph. Likely helps
similar multi-component graphs not in the top-10 list as well.

Priority: MEDIUM (specific graph, but this kind of structure appears
frequently in real ML trace DAGs with isolated helper modules).
Effort: 4-6 hours (requires a mini-pipeline-per-component or a clever
batched version).

### Bucket D: Deep sparse DAG with wide fan-out (1 graph)

Graph: dependency_500.

Symptom: composite 51.96, primary losses in edge_length_cv and possibly
crossing_rate due to the 53-out-degree hub. 18+ min layout runtime.

Root cause hypothesis: (a) rank_sep=240 * max_depth=18 = 4320 tall
layout, AR 0.25 widens to 1080 wide; but the hub with 53 children needs
wide spread to avoid edge length blowup; (b) barycenter with 8 passes
doesn't converge on a wide fan-out.

Proposed fix:
- Detect "deep DAG with hub": max_depth > 10 AND max_out_degree > 20.
- For such graphs, use a two-column layout for the hub: split children
  across 2-3 columns at successive y-levels.
- Alternatively, use `HubFanSplit` op (doesn't exist but would be simple):
  if a node has >= K children, move half to y+rank_sep/2, half to
  y+rank_sep; staggers the fan.

Expected composite impact: +3 to +6 pts (closing half the gap).
Priority: MEDIUM. Effort: 6-8 hours.

### Bucket E: Tiny cyclic graphs (1 graph)

Graph: recurrent_feedback_cell (basically closed, delta=-0.36).

Already handled by sprint-19a cycle-reversal pre-pass. No action needed.

### Bucket F: Dense DAG with crossings (1 graph)

Graph: dense_pair_50 + extreme_mixed_width_transformer.

Symptom: 1-N crossings avoidable by smarter within-layer ordering.
dense_pair_50: crossing_rate 0.018 vs dot 0.008; extreme_mixed_width_transformer:
0.051 vs 0.000.

Root cause: `BarycenterReorder` with default iterations=8 doesn't find
the optimal within-layer permutation for graphs with mixed-width layers
or dense inter-layer edges.

Proposed fix:
- Add a final `SwapTransposeHeuristic` op that tries every adjacent pair
  swap within each layer and accepts if total crossings decrease.
  Complexity O(L * max_layer_width^2 * crossing_eval_cost). For graphs
  up to n=100, this is tractable.
- Already exists! `_transpose_heuristic` in init_placement.py L199. But
  it runs at init; crossings can re-emerge during gradient optimization.
  Need a post-optimizer equivalent.

Expected composite impact: +1 to +3 pts on both dense_pair_50 and
extreme_mixed_width_transformer.
Priority: LOW-MEDIUM. Effort: 2-4 hours (op already exists as template).

## 4. Action queue

Ordered by (expected impact per graph) * (# graphs affected) / effort:

| # | Fix | Effort | # graphs | Est total composite delta |
|---|---|---|---|---|
| 1 | Topology-aware AR target in `AspectRatioFit` | 2-4 hr | 6+ | +15 to +25 pts summed |
| 2 | Tighten cycle-reversal re-layer rejection | 30 min | 2 | +10 to +16 pts summed |
| 3 | Component-tile layout op | 4-6 hr | 1+ | +8 to +10 pts |
| 4 | Post-optimizer crossing-swap pass | 2-4 hr | 2+ | +2 to +6 pts summed |
| 5 | Hub fan-split for deep DAG | 6-8 hr | 1 | +3 to +6 pts |

Fix #1 and #2 together should close ~25-40 composite points across the
10 loss graphs AND are low-risk surgical changes. Start with #2 (smallest)
to validate the hypothesis on small_world_100, then #1.

A quick sanity test after each fix: measure on the 5 largest "dagua
winners" (random_dag_200, org_chart_deep, random_dag_50, hub_fanout_label_skew,
org_chart_1_5_4_8) to ensure none regress -- those graphs are currently
heading in the same direction (tall narrow layout), so a topology-aware
AR target that keeps 0.25 for deep DAGs will not touch them.

## 5. Methodology notes

- All numbers above come from actual runs of the current HEAD of
  `feat/bench-and-aesthetics` (commit 56c3b93 + working-tree mods).
- `engine_layout(g, LayoutConfig(seed=42))` on CPU, with
  `CUDA_VISIBLE_DEVICES=""` set.
- Composite metric from `dagua.metrics.composite` as of L1147.
- Competitor positions loaded from
  `eval_output/variant_bench_full/positions/<graph>__<engine>.pt`.
- `dependency_500` layout did not complete in the diagnostic budget for
  this report (18+ minutes elapsed, still optimizing). Its gap analysis
  is based on the existing benchmark number (51.96) and structural
  inspection of the graph. A proper per-metric breakdown of dagua on
  dependency_500 should be added as a follow-up once a representative
  layout finishes.
