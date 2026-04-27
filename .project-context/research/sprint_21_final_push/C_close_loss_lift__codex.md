# Area C — Close-loss + tie bucket lift analysis (codex)

## TL;DR

- The highest-leverage low-cost lift is **stronger discrete crossing/order polish**. It directly targets `weighted_clusters_3x10`, `triangular_lattice_36`, `multi_component_80`, and tie graph `densenet_block`; expected delivery is 2-4 strict-dominate flips if the polish is scored and rejected unless it improves composite.
- The cleanest single-graph flip is **component depth-rank restacking** for `disconnected_label_cycle_collage`: dagua loses 2.89 weighted points on `depth_spearman_rho` while winning CV/straightness. Restacking disconnected tiles by component depth should flip it with low visual risk.
- The best structural lift is **a cyclic-regular route/best-of choice** for `parallel_cycles_4x5` and possibly `small_world_500`. Competitors win by abandoning layered straightness and making all edges near-equal length or DAG-consistent; this can flip one or two graphs but has the highest runtime/regression risk.
- Several ties are saturated (`deep_chain_20`, `linear_3layer_mlp`, `nested_shallow_enc_dec`, `weighted_chain_20`, `parallel_multiedge_bundle`). Do not spend implementation budget there.
- I ignored cached `__dagua.pt` positions for authoritative scoring because several are stale. Bucket enumeration came from `/tmp/h2h_buckets_seeded.py`; per-component evidence came from fresh `layout(g, LayoutConfig(seed=42))` scored with `torch.manual_seed(0); composite(full(...))`.

## Evidence

HEAD is `97286e4`. Required bucket command:

```text
Total: 93
WIN strong: 32
WIN modest: 42
TIE: 8
close LOSS: 8
moderate LOSS: 3
big LOSS: 0
BEST or TIED: 82/93 = 88%
COMPETITIVE: 90/93 = 97%
```

The 16 close-loss/tie candidates from the fresh enumerator were:

`disconnected_label_cycle_collage`, `small_world_500`, `weighted_clusters_3x10`, `triangular_lattice_36`, `clustered_medium_5x20`, `outerplanar_dag_20`, `multi_component_80`, `parallel_cycles_4x5`, `recurrent_feedback_cell`, `parallel_multiedge_bundle`, `deep_chain_20`, `linear_3layer_mlp`, `nested_shallow_enc_dec`, `weighted_chain_20`, `small_world_100`, `densenet_block`.

The component diffs below are weighted composite-point diffs, not raw metric diffs.

One measurement nuance: `composite()` still includes small add-on terms outside
the prompt's 100-point component list when the underlying metric dictionary
contains them, notably edge-node crossing and label overlap. Those did not
change the diagnosis for this bucket, but they explain why summing only the
eight headline rows can be off by a few tenths. I ranked the dominant loss by
the weighted contribution used by `composite()`, then sanity-checked the raw
metric values for cases where the net delta looked inconsistent with the top
loss. That is why `parallel_cycles_4x5` is classified as structural even though
the net loss is only -0.62: dagua is making a very large, bidirectional trade
between equal edge lengths and layered straightness.

## Per-Graph Breakdown

| graph | dagua | comp | delta | losing-metric | dagua winning on | comp-strategy | recommendation | est-delta |
|---|---:|---|---:|---|---|---|---|---:|
| disconnected_label_cycle_collage | 77.37 | elk_layered | -1.99 | depth_spearman -2.89 | CV +0.86, straightness +0.66 | ELK stacks disconnected pieces so local depth also ranks globally by y | Component depth-rank restack after independent tiling | +2.0..+2.5 |
| small_world_500 | 52.19 | elk_layered | -1.96 | dag_consistency -12.35, straightness -6.12 | edge_length_cv +15.45 | ELK forces a layered DAG-like view; terrible CV but excellent directionality | Best-of stress vs layered for cyclic dense small-world, or n>200 density gate | +1.5..+2.5, high risk |
| weighted_clusters_3x10 | 65.14 | graphviz_dot | -1.61 | crossing_rate -5.32 | straightness +4.06, depth +3.65 | dot gets a better rank order via barycentric/median crossing minimization | Stronger crossing/order polish on small layered graphs | +1.0..+4.0 |
| triangular_lattice_36 | 85.48 | graphviz_dot | -1.61 | angular_res -0.76, crossing -0.50, CV -0.44 | straightness +0.20 | dot preserves regular hex-grid angles and zero crossings | Lattice grid/angle snap plus scored crossing cleanup | +1.0..+2.0 |
| clustered_medium_5x20 | 69.78 | graphviz_dot | -1.41 | straightness -3.55 | angular +1.07, depth +0.86, DAG +0.39 | dot keeps cluster/rank internals more vertical | Cluster-aware straightness/short-edge compression polish | +1.5..+3.0 |
| outerplanar_dag_20 | 72.42 | igraph_sugiyama | -0.74 | angular_res -1.87, CV -1.34 | straightness +2.24 | sugiyama gives cleaner fanout angles and more uniform rank spans | Fanout-angle polish or rank-quantile spacing | +0.7..+1.5 |
| multi_component_80 | 74.46 | graphviz_dot | -0.64 | crossing_rate -0.49 | effectively tied elsewhere | dot lays components independently with no residual crossings | Component tile reorder plus small-graph crossing polish | +0.5..+1.0 |
| parallel_cycles_4x5 | 62.11 | graphviz_sfdp | -0.62 | edge_length_cv -15.20 | straightness +9.58, DAG +5.00 | sfdp ring-like force layout equalizes every cycle edge | Route regular cyclic components to force/stress or score best-of layered vs force | +1.0..+3.0, medium/high risk |
| recurrent_feedback_cell | 73.18 | igraph_sugiyama | -0.39 | straightness -1.36 | CV +0.97 | sugiyama routes the feedback edge as a cleaner horizontal/back arc | Relax back-edge/arc straightness polish gate for n<=10 | +0.5..+1.0 |
| parallel_multiedge_bundle | 85.50 | graphviz_dot | -0.00 | none meaningful | none meaningful | identical/saturated | Leave alone | 0 |
| deep_chain_20 | 97.50 | graphviz_dot | +0.00 | none | saturated | both perfect chain layouts | Leave alone | 0 |
| linear_3layer_mlp | 97.50 | graphviz_dot | +0.00 | none | saturated | both perfect small layered layouts | Leave alone | 0 |
| nested_shallow_enc_dec | 97.50 | igraph_sugiyama | +0.00 | none | saturated | both at structural ceiling | Leave alone | 0 |
| weighted_chain_20 | 97.50 | graphviz_dot | +0.00 | none | saturated | both perfect chain layouts | Leave alone | 0 |
| small_world_100 | 57.18 | igraph_sugiyama | +0.09 | dag_consistency -12.13, straightness -4.92 | edge_length_cv +17.15 | sugiyama forces hierarchy; dagua wins CV enough to tie | Do not tune for this directly; only let a scored best-of choose layered if it wins | 0..+0.5 |
| densenet_block | 69.00 | dagre | +0.32 | crossing_rate -2.50 | straightness +1.77, CV +1.12 | dagre accepts worse straightness to reduce crossings | Same crossing/order polish as weighted_clusters | +0.8..+2.0 |

## Recommendation Clusters

### 1. Stronger discrete crossing/order polish

Targets: `weighted_clusters_3x10`, `triangular_lattice_36`, `multi_component_80`, `densenet_block`.

The repeated pattern is that dagua already wins smooth continuous geometry, while competitors win the scorer's discrete crossing term. The clearest case is `weighted_clusters_3x10`: dagua wins straightness by +4.06 and depth by +3.65, but loses crossing by -5.32 and angular resolution by -1.33. `densenet_block` is already a tie-to-win, but still leaves -2.50 crossing points to dagre. `multi_component_80` is only -0.64 overall and the visible loss is a -0.49 crossing term, so one accepted crossing improvement is enough.

The lowest-effort implementation is not another gradient weight. The context already says crossing and length weights saturate. The change should be a **scored local order polish**: find non-incident crossing pairs on graphs under a small cap, try adjacent same-layer x swaps or targeted barycenter reordering of crossing endpoints, and keep the candidate only if `composite(full(...))` improves by the existing polish margin. There is also an existing `dagua/layout/ops/crossing_swap.py` Sugiyama adjacent-swap op; if it is not wired into this route, reusing or adapting it is lower risk than inventing another crossing counter.

Expected delta is +0.5 to +5.0 per crossing-sensitive graph, with 2-4 strict-dominate flips. Runtime risk is bounded if gated to `n <= 200`, `E <= 400`, and a max swap budget.

### 2. Component depth-rank restacking

Targets: `disconnected_label_cycle_collage`, possibly `multi_component_80`.

`disconnected_label_cycle_collage` is the cleanest miss in the bucket. The component breakdown says dagua is not losing visual basics: edge CV is +0.86 and straightness is +0.66. The dominant loss is `depth_spearman_rho` at -2.89 weighted points. This is exactly the kind of gap produced by disconnected component packing: each component can be internally good, but row-major tiling scrambles the global y ordering used by the metric.

The lowest-effort change is in the component tiler path (`_tile_component_positions` in `dagua/layout/ops/pipelines/dagua_native_legacy.py`, called by `dagua_native.py`). After solving components independently, compute each component's local depth range or mean node depth from its child problem, then assign tile rows so deeper components land lower in global y. A cheaper first pass is to sort components by `(max_depth, size)` instead of only size/area, then keep the existing grid chooser.

Expected delta is +2.0 to +2.5 on `disconnected_label_cycle_collage`, enough to flip it. `multi_component_80` could gain +0.5 to +1.0 if the restack also reduces residual crossings, but it should be measured because its component loss is smaller.

### 3. Cluster-aware straightness and short-edge compression

Targets: `clustered_medium_5x20`, partial help for `weighted_clusters_3x10` and `outerplanar_dag_20`.

`clustered_medium_5x20` loses primarily on straightness: 26.50 degrees vs dot's 10.51, a -3.55 point deficit. Dagua wins depth and angular resolution, which means the broad layering is fine; the problem is local edge geometry inside repeated clustered ranks. `outerplanar_dag_20` loses angular and CV while winning straightness, so it needs a gentler version focused on fanout spacing rather than aggressive verticalization.

The simplest knob is a scored polish candidate that compresses short-rank or intra-cluster edges and tests whether straightness/CV improve without breaking angular separation. For clustered graphs, target edges whose endpoints share a cluster or adjacent rank: move endpoints toward the local edge midpoint along the non-dominant axis, then score. For outerplanar graphs, try rank-quantile y spacing plus fanout x spreading so high-degree fanouts get angular room.

Expected delta is +1.5 to +3.0 on `clustered_medium_5x20`; +0.5 to +1.5 on `outerplanar_dag_20`; smaller secondary lift on `weighted_clusters_3x10`.

### 4. Cyclic regular route/best-of

Targets: `parallel_cycles_4x5`, `small_world_500`; avoid harming `small_world_100`.

These are structural tradeoffs, not polish misses. `parallel_cycles_4x5` gives dagua +9.58 straightness and +5.00 DAG consistency, but graphviz_sfdp gets +15.20 edge-length uniformity. Dagua's layered representation is too "correct" for a cyclic regular graph whose scorer rewards equal edge lengths. `small_world_500` is the inverse shape: dagua wins CV by +15.45, but ELK wins DAG consistency by +12.35 and straightness by +6.12, leaving dagua -1.96 overall.

The lowest-effort safe version is **score best-of two existing routes** for a narrow topology: regular cyclic components and dense small-world graphs. Try current native route plus a force/stress/layered alternative, score both under `composite(full(...))`, return the better. The risky but cheaper runtime version is a heuristic gate: regular low-degree cyclic graphs go force/stress; dense `n > 200` small-world with many back edges goes layered.

Expected delta is +1 to +3 on `parallel_cycles_4x5`, and +1.5 to +2.5 on `small_world_500`. Risk is higher than polish because route choice can affect runtime and can re-break `small_world_100`, which currently ties by winning CV hard enough to offset its DAG/straightness loss.

### 5. Back-edge micro-polish

Target: `recurrent_feedback_cell`.

This is a singleton and should not drive architecture. Dagua loses -1.36 on straightness while winning +0.97 on CV. Sugiyama's specific advantage is a cleaner feedback edge/back arc on a 5-node graph. A low-risk tweak is to relax any back-edge arc or compactness gate for very small graphs (`n <= 10`, at least one back edge), score it, and keep only if composite improves. Expected delta is +0.5 to +1.0.

## Risk / Protected Wins

The fragile protected wins near the +0.5 threshold are `interleaved_cluster_crosstalk` (+0.706), `transformer_layer` (+0.751), `transformer_full_4h_2l` (+0.855), `dependency_graph_100` (+0.909), `long_skip_only_24` (+1.065), `sierpinski_42` (+1.135), `braided_feedback_tails` (+1.471), `resnet_stack_4x16` (+1.489), `er_100` (+1.643), `sparse_pair_50` (+1.914), and `compound_dag_5x30` (+1.978).

- Crossing/order polish risk: low if implemented as a scored candidate. The risk is runtime on dense graphs, so cap graph size and swap attempts.
- Component restacking risk: medium for disconnected/clustered wins such as `sparse_pair_50` and `compound_dag_5x30`. Gate to truly disconnected graphs and preserve the existing size/area ordering as a tiebreaker.
- Cluster-aware compression risk: medium. `interleaved_cluster_crosstalk` is only +0.706 and could regress if intra-cluster compression creates crossings. Only accept via polish score.
- Cyclic best-of route risk: high relative to payoff. It can double runtime on hard graphs and may undo the sprint-20 small-world fix. Use last, and only with a narrow topology gate or scorer-based selection.
- Back-edge micro-polish risk: low if gated to small graphs and scored.

## Implementation Order

1. **Crossing/order polish** first. It covers the most candidates and maps directly to measured lost points.
2. **Component depth-rank restack** second. It is the cleanest one-graph flip and may help `multi_component_80`.
3. **Cluster-aware straightness/short-edge polish** third. It should flip `clustered_medium_5x20`, but needs a careful trigger.
4. **Back-edge micro-polish** fourth. Cheap singleton.
5. **Cyclic best-of route** last. It has real upside, but also the largest runtime and regression surface.

## Assumptions / Concerns

- I treated the eight prompt-listed components as the primary target, but inspected the raw fields actually consumed by `composite()`: `edge_straightness_mean_deg`, `crossing_rate`, `angular_res_mean_deg`, and `cluster_mean_sep_ratio`.
- The per-component direct run gave `disconnected_label_cycle_collage` at 78.00 vs the bucket's 77.37 in one repeat, but the dominant loss stayed the same (`depth_spearman_rho`). All other close/tie direct scores matched the bucket closely enough to support the recommendations.
- No source files were modified by this survey. This is research only.

## Knowledge

- The current strict-dominate ceiling is not blocked by one universal weakness.
  The close/tie set splits into three different mechanisms: discrete crossing
  order, disconnected-component global depth ranking, and structural route
  choice for cyclic graphs.
- The polish-picker pattern remains the safest implementation vehicle. Almost
  every low-cost recommendation should be a scored candidate, not a default
  geometry change, because the protected wins near +0.5 are mostly adjacent
  graph classes.
- Cached competitor positions are reliable for the seven competitor engines,
  but cached dagua positions can lag the current native pipeline. Future surveys
  should recompute dagua with `LayoutConfig(seed=42)` and use cache only for
  competitors unless the cache has just been regenerated.
