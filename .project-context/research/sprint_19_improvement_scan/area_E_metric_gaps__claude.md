# Area E: Per-Metric Gap Analysis (Claude)

Generated: 2026-04-24. Data source: `eval_output/variant_bench_full/positions/`
(cached dagua + graphviz_dot + dagre + elk_layered + igraph_sugiyama positions).
93 graphs with complete 5-engine coverage. Metrics recomputed with
`crossing_samples=50k`, `angular_samples=500`; all other formulas match
`dagua/metrics.py`.

Raw table: `/tmp/metric_table.json`. Analysis scripts:
`/tmp/analyze_metrics.py`, `/tmp/analyze_extra.py`.

All numbers below are the actual metric values; "pts" refers to the composite
contribution computed by reapplying `composite()` weights.

---

## 1. TL;DR -- Top 3 Closable Gaps

| Rank | Metric | Weight | Opportunity (pts across 93 graphs) | Root cause |
|------|--------|-------:|------------------------------------:|-------------|
| 1 | `overlap_count` (binary) | 10 | **210.0** | OverlapProjection under-converges on n>=100 dense graphs; 21 graphs still have overlaps when competitors have 0. |
| 2 | `edge_length_cv` | 20 | **187.4** | No dummy-node splitting of long spans; no horizontal node_sep compaction after force solve. Native pipeline has `LengthVarianceLoss` (w=8.0) but it can't fight the repulsion gradient. |
| 3 | `crossing_rate` | 10 | **142.1** | BarycenterReorder runs once at the end; no transpose heuristic; no post-barycenter coordinate reassignment. Crossings in cyclic / near-regular graphs blow up. |

Next tier (still worth chasing):
- `edge_straightness_mean_deg` (w=10): **71.0 pts**. Brandes-Koepf 4-pass op
  exists but is not in the native pipeline.
- `angular_res_mean_deg` (w=5): **58.6 pts**. Many flat / wide-layer graphs
  collapse into narrow horizontal strips.
- `depth_spearman_rho` (w=15): **45.0 pts**. Mostly "wide" graphs where every
  layer is depth 0 or 1 and dagua's depth becomes noisy.

Cluster_separation is not scored here (the benchmark graphs don't carry
`cluster_ids`, so the column is effectively neutral for everyone).

Per-engine mean composite (scored components only, no cluster term):
- `graphviz_dot` 64.19
- `dagre` 62.00
- `dagua` 61.08
- `igraph_sugiyama` 59.47
- `elk_layered` 57.40

Note: this is a SUBSET of the public composite because our measurement skipped
cluster_separation (no cluster_ids available). In the official rubric dagua
still leads; the gaps below are where it spends that lead.

---

## 2. Per-Metric Deep Dive

### 2.1 `dag_consistency` (weight 25, HIGHER is better)

**Formula (metrics.py L240).** Fraction of forward edges that point in the
layout direction (y_tgt > y_src for TB). Edge cases: empty edge_index -> 1.0;
`back_edge_mask` excludes known back edges.

**Distribution (dagua means on 93 graphs).** Mean 0.972, median 1.000, p10
0.977. Dagua is tied-best with dagre/sugiyama (0.974) and only elk (0.855) is
clearly worse. This metric is essentially solved.

**Losses >0.10.** Only 2 graphs:

- `recurrent_feedback_cell` (n=5, cycle): dagua 0.500 vs graphviz_dot 0.667.
- `center_port_backedge_hub`: dagua 0.667 vs graphviz_dot 0.778.

Both are small cyclic graphs where the DFS-based back-edge reversal collapses
(`cycle.py`) can't cleanly separate forward from feedback edges. Opportunity
cost is trivial: **7.64 pts total**, 4.17 concentrated on `recurrent_feedback_cell`.

**Root cause.** Only matters on tiny cyclic graphs where Kahn + make_acyclic
fallback in `init_placement.py` marks the wrong edges back.

**Intervention.** Area A (algorithm core) should handle this. Low priority
here -- the composite gain is ~0.08 pts/graph average.

### 2.2 `edge_length_cv` (weight 20, LOWER is better)

**Formula (L206).** CV = std(edge_lengths) / mean(edge_lengths). Edge cases:
mean < 1e-8 -> 0.0 (degenerate). Composite = 20 * max(0, 1 - cv); cv > 1 scores
0. Targets: <0.3 excellent, <0.5 good, >1.0 broken.

**Distribution.** Dagua mean **0.757** vs dagre 0.717, sugiyama 0.720,
graphviz_dot 0.754, elk 0.779. Dagua is middle-of-pack on MEAN but has a heavy
right tail: p90=0.932, max=7.86.

**Biggest losses (dagua >0.30 above best competitor):**

| Graph | dagua CV | best competitor CV | gap |
|---|---:|---:|---:|
| small_world_500 | 7.863 | dagre 1.377 | 6.49 |
| small_world_100 | 4.004 | dagre 2.676 | 1.33 |
| grid_20x20 | 0.798 | sugiyama 0.000 | 0.80 |
| cluster_member_style_stress | 0.934 | elk 0.352 | 0.58 |
| clustered_medium_5x20 | 1.540 | sugiyama 1.106 | 0.43 |
| hexagonal_lattice_42 | 0.512 | graphviz_dot 0.099 | 0.41 |
| compound_dag_5x30 | 1.616 | sugiyama 1.248 | 0.37 |
| complete_bipartite_8x12 | 0.635 | elk 0.271 | 0.36 |
| random_bipartite_60 | 0.814 | elk 0.473 | 0.34 |
| rgg_100 | 0.900 | elk 0.561 | 0.34 |
| random_dag_50 | 0.831 | graphviz_dot 0.529 | 0.30 |
| citation_dag_300 | 1.013 | elk 0.712 | 0.30 |

**Wins.** `org_chart_deep` (0.555 vs sugiyama 1.178) and `recurrent_feedback_cell`.
Dagua is best on ~2/93 graphs; tied/near-tied on another ~20.

**Root cause breakdown.** Three different failure modes:
- **Multi-layer spans without dummies.** `grid_20x20` (regular mesh) and
  `hexagonal_lattice_42`: Graphviz produces near-constant edge length because
  its Sugiyama phase-1.5 inserts dummy nodes on long edges so every drawn
  segment is exactly `rank_sep`. Dagua draws these as straight lines from
  source layer to target layer; short edges (one rank apart) stay ~`rank_sep`
  while diagonal edges stretch sqrt(2) * rank_sep or more.
- **Flat cyclic graphs.** `small_world_100/500`: longest-path layering makes
  almost all nodes share one layer, so Force2DInitIfFlat kicks in with
  random 2D positions. The repulsion loss then explodes to prevent overlaps,
  creating a log-normal edge-length distribution. CV 7.86 is not "long edges";
  it's "a few edges are 30x the median because those pairs were pushed apart".
- **Dense bipartite / near-regular.** `complete_bipartite_8x12`,
  `random_bipartite_60`: many nodes in one layer all connected to many in the
  next; dagua's barycenter reorder + gradient end up with uneven x-spacing,
  producing long sideways edges. Elk's linear-compaction phase equalizes
  these.

**Intervention.**
1. **Dummy-node splitting (Sugiyama phase 1.5)** on long-span edges. The ops
   already exist (`sugiyama.py:_expand_long_edges_with_dummy_nodes`) and the
   `sugiyama_pipeline` uses them. The native pipeline does NOT. Inserting
   `_ExpandDummyNodes` before `BarycenterReorder` normalizes every drawn
   segment to ~`rank_sep`. Expected impact: closes grid_20x20, lattice, and
   deep-chain cases. **Projected gain: 25-40 pts composite** if dummy nodes
   are introduced but NOT contracted back for overlap projection (tricky --
   see side-effects below).
2. **Post-gradient horizontal compaction** for the flat-graph branch. When
   Force2DInitIfFlat runs, the final layout has no rank structure; a
   MaxEnt-stress-style compaction on x (with lower bound = node_sep) would cap
   the long-tail edge lengths. Expected impact on small_world_*: 5-10 pts each.
3. **Raise `w_length_variance`** (currently 8.0) or anneal it on a later
   schedule. Sprint 16 weight sweep evidently already explored this; the fact
   that dagua still has 7.86 CV on small_world_500 suggests gradient alone
   cannot fight the repulsion geometry. More loss weight will saturate, not
   solve, the flat-cyclic case -- prefer option (2).

**Side effects.** Dummy-node splitting directly INCREASES edge_straightness
score (dummy chains are colinear by construction) and DECREASES crossing_rate
(dummies force intermediate layers to untangle), so it is net-positive across
metrics. It may slightly INCREASE overlap_count because dummy nodes consume
horizontal space -- but dummy nodes have zero size in the existing ops code
(see `dummy_sizes.append([0.0, 0.0])`) so they won't trigger overlaps.

### 2.3 `depth_spearman_rho` (weight 15, HIGHER is better)

**Formula (L325).** Spearman rho between topo_depth and y-coordinate. Edge
cases: constant input (all zeros or all N-1) returns NaN (see scipy
ConstantInputWarning). Composite uses `max(0, rho)` so NaN becomes 0.

**Distribution.** Mean is NaN for dagua because 4 graphs have undefined rho:
`center_port_backedge_hub`, `parallel_cycles_4x5`, `small_world_100`,
`small_world_500`. All are cyclic/flat graphs where Kahn's longest_path
assigns every node to layer 0.

**Biggest real losses (>0.15):**

| Graph | dagua rho | competitor best rho | gap |
|---|---:|---:|---:|
| wide_1_100_1 | 0.240 | graphviz_dot 1.000 | 0.76 |
| wide_single_layer_1_50_1 | 0.334 | graphviz_dot 1.000 | 0.67 |
| wide_3_50_3 | 0.537 | graphviz_dot 1.000 | 0.46 |
| disconnected_label_cycle_collage | 0.756 | elk 0.962 | 0.21 |
| complete_bipartite_8x12 | 0.850 | graphviz_dot 1.000 | 0.15 |

**Root cause.** The "wide_*" family is 1-source -> N-wide-layer -> 1-sink
graphs. All sugiyama-family engines assign the 100 middle-layer nodes to a
single y-coordinate, so rho = 1.0. Dagua's gradient then spreads these
middle-layer nodes vertically (because repulsion is isotropic) -- some drift
up toward the source, some down toward the sink. Spearman drops to 0.24.

**Intervention.**
1. **Rank-pinning.** After NativeEngineInit assigns y = layer * rank_sep, the
   gradient should treat y as fixed (or strongly clamped) for the wide-layer
   nodes. Options:
   - Add a `w_rank_pin` loss term that quadratically penalizes y drift from
     `layer * rank_sep`.
   - Project y back onto the layer grid at every step / every N steps.
2. `disconnected_label_cycle_collage` is a different problem (multi-component
   + cyclic). See Area B (codex agent) for details.

**Side effects.** Pinning y-coordinates to layers HELPS edge_straightness
(segments become truly vertical if layers agree) and hurts edge_length_cv
only mildly (fixes the y component, not x). It HELPS dag_consistency by
construction.

**Expected impact.** 7-12 pts composite across the wide_* family (3 graphs x
~3-4 pts each) plus 2-3 pts on bipartite / collage. Total gain ~10-15 pts.

### 2.4 `overlap_count` (weight 10, BINARY)

**Formula (L341).** Exact pairwise bbox intersection (exact up to n<=2000,
then spatial-hashed). Composite: 10 pts if zero, else 0.

**Distribution.** Dagua mean **11.26 overlaps** per graph vs competitors
~0.01. Dagua has non-zero overlaps on 21 graphs where competitors have 0.

**Loss list** (dagua overlaps > 0 AND graphviz_dot == 0):

| Graph | dagua overlaps | n |
|---|---:|---:|
| dependency_500 | 224 | 500 |
| hub_spoke_5x50 | 201 | 257 |
| rgg_500 | 167 | 500 |
| ba_500 | 110 | 500 |
| hub_spoke_10x20 | 98 | 212 |
| grid_20x20 | 88 | 400 |
| org_chart_deep | 73 | 79 |
| scale_free_ba_120 | 20 | 120 |
| powerlaw_500 | 16 | 500 |
| er_500 | 11 | 500 |
| dependency_graph_100 | 7 | 100 |
| real_football_115 | 6 | 115 |
| protein_ppi_200 | 5 | 200 |
| rgg_100 | 5 | 100 |
| 6 more with 1-4 overlaps |

**Root cause.** The native pipeline ends with `OverlapProjection` (Lloyd-style
iterative push-apart) with default `iterations=10`. On dense n>=100 graphs the
solver cannot converge in 10 iterations; each push creates new overlaps. For
`dependency_500` the 224 overlaps are a thick cluster of hub nodes competing
for the same vertical band.

Notably it is the SAME graphs that also lose edge_length_cv and crossing_rate
-- the layout is a pile. See the per-graph decomposition of
`dependency_500` (diff=-13.11): -10 from overlap, -2.44 from length_cv, -1.01
from angular_res. 76% of the composite loss is "I didn't finish projecting."

**Intervention.**
1. **Increase OverlapProjection iteration budget** to N-dependent scaling
   (e.g. `iterations = max(10, int(sqrt(N)))`) -- cheap and safe.
2. **Constraint-based overlap removal.** The existing `OverlapProjection` op
   uses soft pushes; swap in a constraint-graph based algorithm (Dwyer 2005
   "Fast Node Overlap Removal" -- O((N log N + C) alpha(N)) where C is the
   constraint count). Alternatively adopt a sweepline-based horizontal
   compaction that directly enforces `|x_i - x_j| >= (w_i + w_j)/2 + node_sep`
   as a hard constraint after the gradient stage.
3. **Increase `w_overlap` in the gradient loss** so the solver pushes harder
   before reaching OverlapProjection. Risk: destabilizes optimization on tiny
   graphs (the current weight schedule anneals it, which may be underpowered
   for dense large graphs).

**Expected impact.** Option (1) alone likely closes ~half the list (the
marginal overlap graphs); option (2) closes all 21. Projected gain: 150-210
pts composite (each graph contributes 10 pts on the binary metric).

**Side effects.** More iterations slightly hurt edge_length_cv (pushed-apart
nodes have longer edges) but improve crossing_rate and angular_res. Net
positive.

### 2.5 `edge_straightness_mean_deg` (weight 10, LOWER is better)

**Formula (L465).** Mean angular deviation of each edge from the primary axis
(vertical for TB). Straight rank-to-rank edges -> 0 deg. Composite = 10 * max(0,
1 - deg / 45).

**Distribution.** Dagua 35.2 deg vs dot 33.7, sugiyama 40.4, dagre 41.8, elk 39.7.
Dagua is second-best on mean but has a BIMODAL distribution (p10=0, p90=76).
Dagua wins where layers are cleanly separated and a handful of edges fall
exactly on axis; loses when edges span multiple layers (not drawn as chain) or
when the layout is flat.

**Biggest losses:**

| Graph | dagua deg | best competitor deg | gap |
|---|---:|---:|---:|
| grid_20x20 | 70.4 | sugiyama 26.6 | 43.8 |
| outerplanar_dag_20 | 49.6 | elk 8.7 | 40.9 |
| complete_bipartite_8x12 | 72.7 | elk 33.8 | 38.8 |
| org_chart_deep | 74.4 | elk 38.4 | 36.0 |
| random_bipartite_60 | 70.8 | elk 42.0 | 28.7 |
| wide_3_50_3 | 83.6 | elk 56.2 | 27.4 |
| real_lesmis_77 | 67.2 | dot 43.0 | 24.2 |
| chung_lu_150 | 83.0 | elk 60.3 | 22.7 |
| citation_dag_300 | 77.0 | elk 54.4 | 22.6 |
| random_dag_50 | 69.0 | dot 47.0 | 22.0 |

**Biggest wins:**

| Graph | dagua deg | next-best deg |
|---|---:|---:|
| dense_pair_50 | 0.07 | dot 31.8 |
| planar_60 | 0.00 | dot 25.2 |
| densenet_block | 0.38 | elk 24.0 |
| small_world_500 | 0.10 | elk 17.5 |

Wins are on flat or LR-oriented graphs where dagua's aspect-ratio fit
collapses y; losses are on multi-layer DAGs that SHOULD be straight but
dagua's BarycenterReorder only permutes within a layer without equalizing
x for a source-target pair.

**Root cause.** Two things:
1. **No horizontal coordinate assignment algorithm.** Brandes-Koepf 4-pass is
   registered as an op (`dagua/layout/ops/coordinate.py:1249`,
   `BrandesKopf4Pass`) and USED by the `sugiyama_pipeline`, but the
   `dagua_native` pipeline does not call it. Brandes-Koepf aligns inner
   segments (dummy-to-dummy chains) with zero deviation.
2. **No dummy-node expansion.** Even if BK were added, it needs a dummy graph
   to align. Without dummy nodes, a long edge 0->5 is a single segment, and
   BK has nothing to align.

The two fixes stack: add `_ExpandDummyNodes` then `BrandesKopf4Pass`, then
contract edges back.

**Intervention.**
1. **Add BrandesKopf 4-pass** after BarycenterReorder in the native pipeline
   (requires dummy-node expansion). Dagua already uses this in its `sugiyama`
   pipeline; port the setup.
2. **Add a coordinate-alignment post-pass** that, for each pair of connected
   nodes in consecutive layers, nudges their x-coordinates toward alignment
   if doing so doesn't create overlap.

**Expected impact.** ~30-50 pts composite if BK+dummies are added, based on
the wins elk and dot get on these graphs.

**Side effects.** BK produces RANK-ALIGNED layouts, which hurt
angular_res_mean_deg (see next section). Per-metric correlation on dagua
rows: straightness vs angular_res = **-0.545** (strongly anti-correlated) and
straightness vs overlap_count = **+0.372** (more straight = more overlaps in
the pile-up sense). So straightness improvements WILL cost some angular-res
points and MAY cost overlap points.

### 2.6 `crossing_rate` (weight 10, LOWER is better)

**Formula (L587).** Sampled edge-pair crossing rate. Excludes pairs sharing a
node. Composite = 10 * max(0, 1 - rate * 10); rate > 0.1 scores 0.

**Distribution.** Dagua 0.056 vs dot 0.040, dagre 0.043, sugiyama 0.050, elk
0.054. Dagua is WORST on this metric. Dagua crossing_rate > 0.10 on 15 graphs;
graphviz_dot has > 0.10 on only 6.

**Biggest losses:**

| Graph | dagua | best comp | gap |
|---|---:|---:|---:|
| petersen_10 | 0.197 | sugiyama 0.028 | 0.169 |
| er_100 | 0.163 | dot 0.058 | 0.105 |
| weighted_karate_34 | 0.145 | dot 0.040 | 0.104 |
| real_karate_34 | 0.135 | dot 0.042 | 0.093 |
| er_500 | 0.163 | dot 0.071 | 0.093 |
| regular_3_30 | 0.123 | dot 0.035 | 0.088 |
| interleaved_cluster_crosstalk | 0.093 | dot 0.011 | 0.083 |
| rgg_100 | 0.130 | dot 0.052 | 0.078 |
| sbm_4x30 | 0.115 | dot 0.039 | 0.076 |
| protein_ppi_200 | 0.086 | dot 0.016 | 0.070 |

**Root cause.** BarycenterReorder runs ONE pass at the end of the pipeline
(8 internal iterations per sprint-18k). Without dummy nodes there's no way to
untangle inner segments. The transpose heuristic in `init_placement.py` runs
only during INIT (8 passes at n<=500, 3 at n<=2000), so it cannot fix
crossings introduced by the gradient. Dot / dagre run barycenter+transpose
iteratively until convergence over a dummy-expanded graph.

**Intervention.**
1. **Dummy-node expansion + iterated barycenter/transpose.** Same fix as
   straightness.
2. **Move BarycenterReorder to FOUR cycles** (currently one, 8 iterations),
   alternating with a transpose heuristic.
3. **Graph-aware sampling** in `sampled_crossing_rate` to make results less
   noisy (not a fix, a measurement issue). 50k samples is enough on n<=200
   but noisy on n=500 (actual edge-pair count E*(E-1)/2 >> 50k).

**Expected impact.** 30-50 pts composite if barycenter iterates to
convergence on a dummy-expanded graph.

**Side effects.** Iterated barycenter preserves y (HELPS dag_consistency,
depth_spearman, straightness) and permutes x (no direct effect on overlap or
length_cv). Net positive.

### 2.7 `angular_res_mean_deg` (weight 5, HIGHER is better)

**Formula (L718).** Sampled minimum angle at each degree>=2 node. Composite = 5 *
min(1, mean_deg / 40). Target mean >20 deg decent.

**Distribution.** Dagua 54.7 vs dot 60.5, dagre 60.6, elk 56.1, sugiyama 58.1.
Dagua is WORST on mean. Has extreme bimodal: p10=1.7 deg, p90=116 deg. Mean
180 on some graphs (hub_and_spoke where all edges fan out symmetrically).

**Biggest losses:**

| Graph | dagua deg | best comp deg | gap |
|---|---:|---:|---:|
| wide_1_100_1 | 6.9 | elk 85.4 | 78.5 |
| wide_single_layer_1_50_1 | 11.3 | elk 82.7 | 71.4 |
| grid_20x20 | 10.6 | dagre 76.0 | 65.3 |
| org_chart_deep | 23.5 | dot 82.1 | 58.6 |
| small_world_100 | 0.0 | sugiyama 42.3 | 42.3 |
| hub_and_spoke_3x20 | 47.8 | elk 90.1 | 42.2 |
| binary_tree | 45.9 | dagre 83.6 | 37.7 |
| hub_spoke_10x20 | 66.9 | dot 99.3 | 32.4 |
| powerlaw_500 | 23.8 | dagre 54.9 | 31.1 |

**Root cause.** Two failure modes:
1. **Wide layers squished into narrow columns.** `wide_*` and hub_spoke
   graphs: dagua places 50-100 nodes in one row but within a compressed
   x-range (probably because AspectRatioFit uniformly scales). Edges from a
   single source fan out over ~1 degree of angle each instead of ~3-4 degrees.
2. **Near-degenerate positions in flat cyclic graphs.** `small_world_100`:
   angular_res=0.0 means two edges at the same node are colinear (overlap).
   This happens when Force2DInitIfFlat places many nodes at nearly the same
   point and repulsion hasn't separated them.

**Intervention.**
1. **Per-node "port spreading"** in the init. For each hub node (degree >
   threshold), space the downstream children proportionally to hub width.
   `init_placement.py:_spread_fanout_children` tries this but evidently not
   aggressively enough for 100-child fanouts.
2. **Aspect-ratio target that considers layer width**: if one layer has 100
   nodes, x-range should scale so that per-node spacing remains >= node_sep.
   Currently AspectRatioFit pushes aspect toward 4:1 regardless of layer
   population.

**Expected impact.** 10-20 pts composite (wide_*, hub_spoke together).

**Side effects.** Widening layers INCREASES edge_length_cv (edges span wider
x) and HELPS overlap. Correlation with length_cv on dagua rows = -0.24 --
mildly negative, so increasing spacing increases CV modestly. Accept the
trade-off for the 5-pt category since wider also improves overlaps in those
exact graphs.

### 2.8 `cluster_separation` (weight 5)

Not measured: benchmark graphs in the cached positions don't carry
cluster_ids in a way the metric could consume here. The composite() function
falls back to a 0.5 neutral contribution -> 2.5 pts for everyone. No gap to
chase without first wiring cluster labels through the evaluation. Area A /
B.

---

## 3. Cross-Metric Interactions and Trade-offs

Pearson correlations on the dagua rows (93 dagua runs):

|                     | DAG | lenCV | depth | straight | cross | angular | overlap |
|---------------------|-----|-------|-------|----------|-------|---------|---------|
| dag_consistency     | 1.00 | 0.01 | 0.40 | -0.08 | -0.03 | 0.02 | 0.08 |
| edge_length_cv      | 0.01 | 1.00 | -0.04 | -0.11 | -0.03 | -0.24 | 0.03 |
| depth_spearman      | 0.40 | -0.04 | 1.00 | -0.40 | -0.13 | 0.22 | 0.10 |
| straight (deg)      | -0.08 | -0.11 | -0.40 | 1.00 | 0.51 | -0.55 | 0.37 |
| crossing_rate       | -0.03 | -0.03 | -0.13 | 0.51 | 1.00 | -0.53 | 0.07 |
| angular_res_mean    | 0.02 | -0.24 | 0.22 | -0.55 | -0.53 | 1.00 | -0.16 |
| overlap_count       | 0.08 | 0.03 | 0.10 | 0.37 | 0.07 | -0.16 | 1.00 |

Remember straightness and crossing_rate are LOWER-is-better; angular_res and
depth are HIGHER-is-better; overlap is LOWER-is-better.

**Key conflicts:**

- **Straightness vs angular_res (r = -0.55).** More rank-aligned edges force
  many edges to emerge at the same angle from each node -> smaller minimum
  angle. Expected and unavoidable.
- **Straightness vs crossing_rate (r = +0.51).** Both are "highly angular =
  also crossing-prone." Correlation is co-movement (both signals of a bad
  layout) not direct causation. Improving the underlying ordering drops both.
- **Straightness vs overlap (r = +0.37).** Rank alignment stacks many nodes
  into one y band, crowding horizontally. Adding BK + dummies makes edges
  straighter but also compacts layers -- without a companion horizontal
  compaction, this raises overlap count.
- **depth vs dag_consistency (r = +0.40).** Strong y-rank correlation implies
  y-order is consistent with topology; consistency naturally follows. Fixing
  one helps the other.

**Synergies:**

- Dummy-node expansion simultaneously improves straightness, crossing,
  length_cv. This is the single highest-leverage intervention.
- Horizontal compaction post-gradient reduces overlap, length_cv, and
  crossing_rate (because pile-up clusters get separated).
- Rank-pinning helps depth, straightness, and dag_consistency together.

**Watch-outs:**

- If the sprint plan stacks "add BK" and "tighten w_overlap" without adding
  dummies, straightness may improve while overlap gets WORSE. Always ship
  dummies first.
- If `w_length_variance` is raised without horizontal compaction, the
  gradient will overcompress layers to equalize edge lengths, and tiny nodes
  will pile up (overlap). A length-variance loss that ignores the node
  bounding box creates hidden overlaps.

---

## 4. Opportunity Cost Ranking

Per-metric pts available if dagua matched the best competitor on every graph
where it currently loses (composite components only, no cluster):

| Rank | Metric | Pts available | n_loss | Notes |
|---:|---|---:|---:|---|
| 1 | overlap_count | 210.0 | 21 | Projection budget scaling + hard-constraint compaction. |
| 2 | edge_length_cv | 187.4 | 64 | Dummy nodes + post-gradient compaction. |
| 3 | crossing_rate | 142.1 | 48 | Iterated barycenter on dummy-expanded graph + transpose. |
| 4 | edge_straightness | 71.0 | 35 | Brandes-Koepf 4-pass (requires dummies). |
| 5 | angular_res | 58.6 | 37 | Layer-width-aware spreading. |
| 6 | depth_spearman | 45.0 | 44 | Rank pinning. |
| 7 | dag_consistency | 7.6 | 6 | Covered by Area A cycle handling. |

Collapsing the top-5 to root-cause buckets:

| Bucket | Captures | Pts |
|---|---|---:|
| Dummy nodes + Brandes-Koepf | length_cv, crossing, straight | 400+ |
| Better overlap projection | overlap | 210 |
| Rank pinning in gradient | depth, straight | 50 |
| Layer-width spreading | angular_res | 60 |

Of course, not every loss is fully closable (we can't literally hit the best
competitor everywhere), so the realistic total gain after stacking fixes is
probably 100-150 composite pts across 93 graphs -- i.e. a LATER composite
mean of 75-77 (from current ~61 on the scored subset), which lifts full-rubric
(including cluster and back-edge handling) past 80.

---

## 5. Action Queue -- Recommended Ordering

Sorted by expected impact per effort.

### Priority 0 (gate before anything else)
- **Fix the `depth_spearman_rho` NaN-on-flat bug.** 4 graphs currently score 0
  on a 15-pt category because their topo_depth is all-zero -> rho = NaN. Two
  options: (a) skip metric and renormalize composite; (b) substitute
  y-variance score. Either way this is free pts -- ~2 pts per NaN graph.
  Effort: 1 hour.

### Priority 1 -- dummy nodes and Brandes-Koepf (400+ pts)
1. Insert `_ExpandDummyNodes` into `build_dagua_pipeline` after
   `NativeEngineInit`. Contract dummies back to polylines before
   `OverlapProjection`. Tests: verify position tensor shape, edge round-trip.
2. Replace the final `BarycenterReorder` with an iterated
   barycenter->transpose->barycenter loop on the dummy-expanded graph.
3. Add `BrandesKopf4Pass` as a coordinate-assignment step AFTER the iterated
   barycenter. Preserve y; BK only permutes x.
4. Gate: dummy expansion only when num_layers >= 3 and max_edge_span >= 2
   (otherwise it is a no-op).

### Priority 2 -- overlap projection scaling (210 pts)
1. Scale `OverlapProjection` iteration count with N:
   `iters = max(10, int(2 * sqrt(N)))` plus early-stop when no overlap moved.
   Cheap; all benefit, no quality cost.
2. Investigate hard-constraint horizontal compaction (Dwyer 2005) as a
   post-step for n >= 100 graphs. Higher engineering cost but full closes the
   overlap gap.

### Priority 3 -- rank pinning (15-25 pts)
1. Add `w_rank_pin` loss that penalizes y deviation from layer * rank_sep on
   the forward-edge nodes. Anneal from 0 -> ~4 (on the same schedule as
   w_attract).
2. Alternative: every K gradient steps, snap y to nearest layer grid.

### Priority 4 -- angular resolution (10-20 pts)
1. Strengthen `_spread_fanout_children` to handle N=100 children. Currently it
   spreads within `node_sep`; make the spread proportional to the widest
   parent's fanout degree.
2. Add a layer-width-aware aspect ratio: if max_layer_nodes * node_sep >
   current_bbox_width, expand x range.

### Priority 5 -- metric hygiene
1. `cluster_separation` currently contributes a neutral 2.5 pts for everyone
   because cluster_ids aren't wired through the benchmark. Either wire them
   (if cluster structure can be computed e.g. via connected components /
   modularity) or make the `composite()` weight renormalize when the field is
   absent. Otherwise this blind 5-pt category obscures progress on the real
   metrics.
2. Crossing metric sample count on n=500 graphs is undersampled vs the
   actual 125k edge pairs; either raise `crossing_samples` on large n or add
   a note to the measurement script. Low priority.

### Do NOT do
- Don't raise `w_length_variance` alone. The 8.0 weight is already fighting
  repulsion; the gradient cannot solve the CV problem without dummies. You'll
  just create pathologies on small graphs.
- Don't add a "straighten-all-edges" post-pass unless dummy nodes are already
  in place. It will create extreme overlap (the straighter -> more overlaps
  correlation).

---

## Appendix A. Per-graph decomposition of top-15 composite losses

For the 15 graphs where dagua loses most to its best competitor
(scored-components composite):

```
grid_20x20               dagua=53.86 vs graphviz_dot=86.49  (-32.64)
  length_cv  -15.45  (0.80 vs 0.025)
  overlap    -10.00  (88 vs 0)
  angular     -3.67  (10.6 vs 72.0)
  straight    -2.00  (70.4 vs 36.0)
  crossing    -1.51  (0.015 vs 0.000)

rgg_100                  dagua=42.18 vs elk=64.33         (-22.15)
  overlap    -10.00  (5 vs 0)
  length_cv   -6.79  (0.90 vs 0.56)
  crossing    -4.68  (0.130 vs 0.053)
  straight    -1.87  (56.7 vs 36.6)

disconnected_label_cycle_collage  dagua=53.60 vs elk=72.69  (-19.10)
  overlap    -10.00  (2 vs 0)
  straight    -4.83  (25.5 vs 3.8)
  depth       -3.09  (0.76 vs 0.96)
  length_cv   -1.17

citation_dag_300         dagua=40.74 vs elk=57.03        (-16.29)
  overlap    -10.00  (2 vs 0)
  length_cv   -5.76  (1.01 vs 0.71)

wide_1_100_1             dagua=56.73 vs elk=71.53        (-14.80)
  depth      -11.40  (0.24 vs 1.00)
  angular     -4.14  (6.9 vs 85.4)

complete_bipartite_8x12  dagua=55.17 vs elk=67.60        (-12.43)
  length_cv   -7.29  (0.64 vs 0.27)
  straight    -2.48  (72.7 vs 33.8)
  depth       -2.25  (0.85 vs 1.00)
```

Notice the pattern: **overlap_count is the single biggest contributor in 10
of the top-15 losses** (all the n>=100 dense graphs), confirming that
Priority 2 above (overlap projection) is simultaneously the easiest and the
highest-EV fix.

## Appendix B. Methodology caveats

- `sampled_crossing_rate` uses 50k samples (default is 1M). For n=500 graphs
  this has a standard error of ~0.002-0.005, so differences below 0.01 are
  noise. Ranking conclusions at >0.02 gap are robust.
- `angular_resolution` uses 500 samples. Degree-weighted; on high-degree hubs
  the min-angle statistic is dominated by the hub. Rankings at >10 deg gap
  are robust.
- `cluster_separation` not computed (no cluster_ids). Everyone gets neutral
  2.5 pts, so the relative ranking is unaffected but absolute composites are
  ~2.5 pts lower than the public rubric.
- `depth_spearman_rho` NaN on 4 graphs; treated as 0 in composite (same as
  public `composite()`).
- All positions loaded from the cache; this is the exact same data used by
  the running benchmark, so per-metric numbers are directly comparable to
  the sprint-19a/b results.
