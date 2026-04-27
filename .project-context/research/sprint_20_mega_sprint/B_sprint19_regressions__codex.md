# Sprint 20 Research B: Sprint-19 Regression Ablation

## TL;DR

- `planar_60` is the cleanest sprint-19 regression. It is not a broad aspect or median issue; it is the **dummy-node + BK interaction**. Current is `65.82` vs ELK `75.04`. Disabling either `insert_dummy_nodes` or `brandes_koepf_refine` restores Dagua to `78.74`, a `+12.92` recovery and a win over ELK. Root cause: dummy expansion turns 60 real nodes into 1,166 active nodes because 97 long edges create 1,106 dummy nodes, then BK compacts that expanded chain structure into a crossing/angle disaster.
- `ragged_feature_pyramid` is a **median/transpose regression**. Current is `69.52` vs ELK `79.56`. Disabling `use_native_median_transpose` lifts Dagua to `75.82`. No other sprint-19 flag matters. Median/transpose increases crossing rate from `0.0370` to `0.1358`, zeroing the 10-point crossing contribution.
- `regular_3_30` is also primarily **median/transpose**, with dummy nodes as a secondary contributor. Current is `68.37` vs graphviz_dot `71.84` in my fresh rescore. Disabling median/transpose lifts to `72.07`; disabling dummy nodes lifts to `70.92`; disabling BK alone does not help (`68.11`).
- `transformer_layer` is not materially explained by sprint-19. Current is `76.18` vs graphviz_dot `80.19`. Disabling BK only lifts to `76.35`; all other flags are no-ops. The remaining loss is structural: graphviz_dot has zero crossings while Dagua has `crossing_rate=0.0458`, and dot gets much better edge-length CV (`0.7276` vs `0.7766`).
- `parallel_cycles_4x5` is not a sprint-19 regression under the tested flags. Every ablation is exactly `58.24` vs graphviz_sfdp `62.73`. The graph is cyclic and one-layer, so median, dummy nodes, BK, topology aspect, and decomposition all skip or become no-ops. It belongs to the cyclic / disconnected force-layout bucket, not sprint-19 cleanup.
- The safest fix is not a blanket rollback. Keep sprint-19 wins by adding acceptance/gating around the two bad compositions: (1) only keep median/transpose if measured crossing count does not worsen, and skip it for small ragged low-width DAGs; (2) cap dummy expansion and prevent BK from running on expanded graphs with extreme dummy-to-real ratios or one-node-per-layer mesh orientations.

## Method

I ran the requested CPU ablation against the current branch in `/home/jtaylor/projects/dagua`, using cached competitor positions from `eval_output/variant_bench_full/positions`. The exact graph set was:

- `ragged_feature_pyramid`
- `planar_60`
- `parallel_cycles_4x5`
- `transformer_layer`
- `regular_3_30`

For each graph I measured:

- `current`
- `no_decomp`: `LayoutConfig(decompose_components=False)`
- `no_bk`: `LayoutConfig(brandes_koepf_refine=False)`
- `no_median`: `LayoutConfig(use_native_median_transpose=False)`
- `no_dummy`: `LayoutConfig(insert_dummy_nodes=False)`
- `no_aspect`: current flags, but monkeypatched `dagua.layout.resolve.resolve_topology_aware_aspect` to return `(0.25, 1.0)`
- `none`: all four flags disabled plus the same aspect monkeypatch

I also ran interaction checks for `no_dummy_no_bk`, `no_median_no_dummy`, and `no_median_no_bk` on the graphs where the one-at-a-time ablation showed coupled behavior.

One assumption: sprint-19e does not expose a public `LayoutConfig` flag, so I treated “disable aspect-target override” as monkeypatching the resolver back to the pre-policy default target. That is conservative because the implemented resolver is centralized in `resolve_topology_aware_aspect()` and threaded into `_dagua_native_target_aspect` during config preparation (`dagua/layout/resolve.py:130-157`, `dagua/layout/resolve.py:350-357`).

The fresh competitor rescoring is close but not byte-for-byte identical to the numbers in the prompt for every graph. I use my measured numbers below because the task asked for real measurements from this run.

## Current Sprint-19 Mechanisms In Scope

The current native pipeline applies the sprint-19 features in this order:

1. Optional per-component wrapper, gated by `_should_decompose_components()` (`dagua/layout/ops/pipelines/dagua_native.py:562-610`, invoked at `dagua/layout/ops/pipelines/dagua_native.py:1272-1282`).
2. Optional dummy insertion, gated by `_should_use_native_dummy_nodes()` (`dagua/layout/ops/pipelines/dagua_native.py:151-189`).
3. Gradient core on the active graph.
4. `BarycenterReorder`, optional `MedianSweep`, optional `TransposeHeuristic`, then `BrandesKoepfHorizontalRefine` (`dagua/layout/ops/pipelines/dagua_native.py:999-1020`, applied at `dagua/layout/ops/pipelines/dagua_native.py:1148-1149`).
5. `StripDummyNodes()`, then `AspectRatioFit()` using `_dagua_native_target_aspect` (`dagua/layout/ops/pipelines/dagua_native.py:1150-1166`).

The two implementation details that matter most for these regressions are:

- Median and transpose directly rewrite x positions after deriving an ordering (`dagua/layout/ops/ordering.py:936-985`, `dagua/layout/ops/ordering.py:1036-1075`). They do not currently run an acceptance check against the actual metric crossing rate after the rewrite.
- BK runs on the active graph, which can be the dummy-expanded graph (`dagua/layout/ops/coordinate.py:1548-1579`). Its gate checks family, layer count, weak components, and strict forward layering (`dagua/layout/ops/coordinate.py:987-1035`), but it does not know whether the active graph is a pathological dummy expansion of a non-layered mesh.

## Per-Graph Results

### 1. `ragged_feature_pyramid`: median/transpose regression

Classifier and gates:

- `N=12`, `E=15`
- `family=GENERAL`, `num_components=1`, `num_layers=10`
- `max_layer_width=2`, `max_degree=4`, `edge_to_node_ratio=1.25`
- `is_directed_acyclic=True`, `topology_tags=('lattice_like',)`
- prepared config: `target_aspect=0.05`, `dummy=False`, `steps=100`

Measured ablation:

| Variant | Dagua | Best competitor | Delta | Crossing rate | Edge CV | Straight deg | Aspect |
|---|---:|---:|---:|---:|---:|---:|---:|
| current | 69.52 | ELK 79.56 | -10.04 | 0.1358 | 0.8342 | 5.58 | 0.083 |
| no_decomp | 69.52 | ELK 79.56 | -10.04 | 0.1358 | 0.8342 | 5.58 | 0.083 |
| no_bk | 69.52 | ELK 79.56 | -10.04 | 0.1358 | 0.8342 | 5.58 | 0.083 |
| no_median | 75.82 | ELK 79.56 | -3.74 | 0.0370 | 0.8342 | 5.58 | 0.083 |
| no_dummy | 69.52 | ELK 79.56 | -10.04 | 0.1358 | 0.8342 | 5.58 | 0.083 |
| no_aspect | 68.70 | ELK 79.56 | -10.85 | 0.1358 | 0.7399 | 17.75 | 0.273 |
| none | 75.02 | ELK 79.56 | -4.54 | 0.0370 | 0.7394 | 17.73 | 0.273 |

Diagnosis:

The primary source is `use_native_median_transpose=True`. The score recovery from disabling it is `+6.30`, and the crossing contribution explains nearly all of it. With median/transpose on, the crossing rate is `0.1358`, which makes the composite crossing term `0.00 / 10`. With median/transpose off, crossing rate is `0.0370`, giving `6.30 / 10`. Edge CV and straightness do not move when median is disabled, so this is not a continuous-geometry issue.

Dummy nodes do not run here because `_DUMMY_NODE_MIN_NODES = 20` and the graph has only 12 nodes. BK also does not run meaningfully because the BK gate skips `lattice_like` tags (`dagua/layout/ops/coordinate.py:1024-1025`). Decomposition is skipped because the graph has one component.

The aspect result is counterintuitive but useful: disabling topology-aware aspect hurts slightly. The `lattice_like` target of `0.05` improves straightness from `17.75` degrees to `5.58` degrees, although it worsens CV. Net effect is `+0.82`, so sprint-19e is not the regression source for this graph.

Proposed fix:

Add an acceptance check around `MedianSweep + TransposeHeuristic`: compute the exact crossing count/rate on the active layered graph before and after the ordering rewrite, and restore the prior order if crossings increase. For this graph that would preserve the `no_median` behavior automatically. A narrower gate would also work: skip native median/transpose for very small ragged DAGs where `num_nodes < 20`, `max_layer_width <= 2`, and `long_edge_count > 0`. The acceptance check is better because it protects other small hand-authored DAGs without hardcoding one family shape.

Risk:

Median/transpose was added for crossing reduction, so an acceptance check should preserve its wins. It only rejects cases where the patch objectively worsens the thing it is supposed to improve. Runtime risk is low at these sizes; for larger graphs, use the same crossing sampler already used by metrics or restrict the exact check to `E <= 2,000`.

### 2. `planar_60`: dummy-node + BK interaction

Classifier and gates:

- `N=60`, `E=156`
- `family=GENERAL`, `num_components=1`, `num_layers=60`
- `max_layer_width=1`, `max_degree=6`, `edge_to_node_ratio=2.60`
- `is_directed_acyclic=True`, `topology_tags=()`
- prepared config: `target_aspect=0.25`, `dummy=True`, `steps=150`

Layer-span diagnostic:

- 59 edges span 1 layer.
- 5 edges span 11 layers.
- 48 edges span 12 layers.
- 44 edges span 13 layers.
- 97 of 156 edges are long edges.
- Dummy expansion would add 1,106 dummy nodes, so the active graph becomes 1,166 nodes before stripping.

Measured ablation:

| Variant | Dagua | Best competitor | Delta | Crossing rate | Edge CV | Straight deg | Angular deg | Aspect |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current | 65.82 | ELK 75.04 | -9.22 | 0.0713 | 0.6898 | 26.15 | 0.47 | 0.481 |
| no_decomp | 65.82 | ELK 75.04 | -9.22 | 0.0713 | 0.6898 | 26.15 | 0.47 | 0.481 |
| no_bk | 78.74 | ELK 75.04 | +3.70 | 0.0000 | 0.6881 | 0.00 | 0.00 | 0.003 |
| no_median | 65.82 | ELK 75.04 | -9.22 | 0.0713 | 0.6898 | 26.15 | 0.47 | 0.481 |
| no_dummy | 78.74 | ELK 75.04 | +3.70 | 0.0000 | 0.6881 | 0.00 | 0.00 | 0.003 |
| no_aspect | 65.82 | ELK 75.04 | -9.22 | 0.0713 | 0.6898 | 26.15 | 0.47 | 0.481 |
| none | 78.74 | ELK 75.04 | +3.70 | 0.0000 | 0.6881 | 0.00 | 0.00 | 0.003 |

Interaction checks:

- `no_dummy_no_bk`: `78.74`
- `no_median_no_dummy`: `78.74`
- `no_median_no_bk`: `78.74`
- `no_median` alone: still `65.82`

Diagnosis:

This is not “dummy nodes are always bad” and not “BK is always bad.” It is specifically BK acting on the dummy-expanded representation of a graph that is planar but not truly layered in the Sugiyama sense.

The original graph is a dense planar nested-cycle graph. The classifier sees a DAG because the generator’s edge orientation is acyclic, but the topology is an undirected-looking mesh. Longest-path layering degenerates into 60 layers with one real node per layer. That creates a huge count of long-span edges, so the dummy gate passes (`dagua/layout/ops/pipelines/dagua_native.py:179-189`). Dummy expansion then inserts 1,106 invisible nodes. BK sees the active expanded graph and passes its own gate because the expanded graph is connected, deep, and strictly forward-layered (`dagua/layout/ops/coordinate.py:987-1035`). The resulting x compaction optimizes the wrong object: the dummy chain scaffold, not the real planar drawing.

The metric signature is decisive. Current has `crossing_rate=0.0713`, `edge_straightness_mean_deg=26.15`, and `angular_res_mean_deg=0.47`. Disabling either dummy nodes or BK makes crossings zero and straightness zero. That `+12.92` score recovery is almost entirely crossing (`+7.13`) and straightness (`+5.81`).

Proposed fix:

Add a dummy expansion budget and a BK-on-expanded budget:

1. In `_should_use_native_dummy_nodes()`, estimate `dummy_nodes = sum(max(span - 1, 0))`. Skip expansion when `dummy_nodes > max(4 * num_nodes, 3 * num_edges)` or when `expanded_num_nodes / num_nodes > 5`. `planar_60` has `dummy_nodes=1106`, `18.4x` real nodes, and fails this immediately.
2. Add a mesh-orientation escape: skip dummy nodes when `max_layer_width <= 1`, `edge_to_node_ratio > 2.0`, and `long_edge_fraction > 0.5`. This catches one-node-per-layer planar orientations without affecting normal wide layered DAGs.
3. In `BrandesKoepfHorizontalRefine`, skip BK on an expanded graph when `active_num_nodes / problem.num_nodes` exceeds the same budget. That is a second guardrail in case dummy expansion is useful for the optimizer but BK should not compact the expanded scaffold.

Risk:

The cap could hurt legitimate long-skip DAGs such as `long_skip_only_24` or dependency chains. That is why the cap should be generous and measured. A graph with 24 real nodes and a few long skips can still pass if expansion stays under `5x`. The high-risk case is exactly what we see here: mesh-like orientation where almost every edge becomes a long dummy chain.

### 3. `parallel_cycles_4x5`: not sprint-19

Classifier and gates:

- `N=20`, `E=20`
- `family=GENERAL`, `num_components=1` by the classifier, `num_layers=1`
- `max_layer_width=20`, `max_degree=2`, `edge_to_node_ratio=1.00`
- `is_directed_acyclic=False`, `topology_tags=()`
- prepared config: `target_aspect=0.25`, `dummy=False`, `steps=100`

Measured ablation:

| Variant | Dagua | Best competitor | Delta | Crossing rate | Edge CV | DAG consistency | Aspect |
|---|---:|---:|---:|---:|---:|---:|---:|
| current | 58.24 | graphviz_sfdp 62.73 | -4.49 | 0.0248 | 0.4992 | 0.5500 | 0.288 |
| no_decomp | 58.24 | graphviz_sfdp 62.73 | -4.49 | 0.0248 | 0.4992 | 0.5500 | 0.288 |
| no_bk | 58.24 | graphviz_sfdp 62.73 | -4.49 | 0.0248 | 0.4992 | 0.5500 | 0.288 |
| no_median | 58.24 | graphviz_sfdp 62.73 | -4.49 | 0.0248 | 0.4992 | 0.5500 | 0.288 |
| no_dummy | 58.24 | graphviz_sfdp 62.73 | -4.49 | 0.0248 | 0.4992 | 0.5500 | 0.288 |
| no_aspect | 58.24 | graphviz_sfdp 62.73 | -4.49 | 0.0248 | 0.4992 | 0.5500 | 0.288 |
| none | 58.24 | graphviz_sfdp 62.73 | -4.49 | 0.0248 | 0.4992 | 0.5500 | 0.288 |

Diagnosis:

None of the sprint-19 flags affects this graph in the current pipeline. It is cyclic, flat, and effectively one-layer after native init. Median/transpose is gated by `is_acyclic` (`dagua/layout/ops/pipelines/dagua_native.py:1006-1012`), dummy nodes require directed acyclicity and more than one layer (`dagua/layout/ops/pipelines/dagua_native.py:179-184`), and BK cannot pass the strict forward-layering gate. The prompt notes this may overlap with agent A, and the data supports that: this belongs to cyclic / disconnected component handling, not sprint-19 regression cleanup.

Proposed fix:

Do not change sprint-19 flags for this graph. Route cyclic regular components to a force/stress sub-pipeline or improve component decomposition for directed cycles. The graph description says “four independent directed 5-cycles,” but `classify_graph()` reports one component, so the generator likely connects them in a way that the description abstracts away or the classifier’s dense early exit hides the intended weak components. Either way, the current native layered objective is the wrong model: it optimizes DAG consistency on a cycle and scores only `0.55` on that 25-point term.

Risk:

Low for sprint-19 cleanup. High if someone tries to fix this by relaxing DAG ordering globally; that could regress the top DAG wins. Keep it in a cyclic topology-dispatch track.

### 4. `transformer_layer`: mostly pre-existing structural gap

Classifier and gates:

- `N=16`, `E=19`
- `family=GENERAL`, `num_components=1`, `num_layers=14`
- `max_layer_width=3`, `max_degree=5`, `edge_to_node_ratio=1.19`
- `is_directed_acyclic=True`, `topology_tags=()`
- prepared config: `target_aspect=0.25`, `dummy=False`, `steps=100`

Measured ablation:

| Variant | Dagua | Best competitor | Delta | Crossing rate | Edge CV | Straight deg | Angular deg | Aspect |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current | 76.18 | graphviz_dot 80.19 | -4.00 | 0.0458 | 0.7766 | 5.42 | 99.52 | 0.090 |
| no_decomp | 76.18 | graphviz_dot 80.19 | -4.00 | 0.0458 | 0.7766 | 5.42 | 99.52 | 0.090 |
| no_bk | 76.35 | graphviz_dot 80.19 | -3.83 | 0.0458 | 0.7743 | 4.86 | 99.13 | 0.090 |
| no_median | 76.18 | graphviz_dot 80.19 | -4.00 | 0.0458 | 0.7766 | 5.42 | 99.52 | 0.090 |
| no_dummy | 76.18 | graphviz_dot 80.19 | -4.00 | 0.0458 | 0.7766 | 5.42 | 99.52 | 0.090 |
| no_aspect | 76.18 | graphviz_dot 80.19 | -4.00 | 0.0458 | 0.7766 | 5.42 | 99.52 | 0.090 |
| none | 76.35 | graphviz_dot 80.19 | -3.83 | 0.0458 | 0.7743 | 4.86 | 99.13 | 0.090 |

Diagnosis:

The only sprint-19 effect is a tiny BK cost of `-0.17`. This is too small to call the root cause. Dummy nodes skip because `N < 20`. Median/transpose produces no measurable change. Aspect target is default in both current and `no_aspect`.

The actual gap is against dot’s ordering/coordinate assignment. Graphviz_dot scores `80.19` with `crossing_rate=0.0000`, `edge_length_cv=0.7276`, and `straightness=12.13`. Dagua scores lower despite better straightness because it has seven estimated crossings (`crossing_rate=0.0458`) and worse edge-length CV. In other words, dot accepts less vertical straightness to preserve a cleaner branch ordering.

Proposed fix:

Do not gate sprint-19 globally for this graph. Add a specific shallow/neural DAG ordering improvement: for small clustered model graphs with `N < 20`, `num_layers > 8`, and `max_layer_width <= 3`, try a dot-like ordering seed or an exact adjacent-swap crossing optimizer after barycenter. The median/transpose stack is currently no-op here; it needs either better initial ordering or an exact crossing-minimizing pass for tiny DAGs.

Risk:

Small. This can be tried as a tiny-graph post-ordering pass with exact crossing acceptance. Avoid changing dummy thresholds because dummy nodes do not run here.

### 5. `regular_3_30`: median primary, dummy secondary

Classifier and gates:

- `N=30`, `E=45`
- `family=GENERAL`, `num_components=1`, `num_layers=7`
- `max_layer_width=7`, `max_degree=3`, `edge_to_node_ratio=1.50`
- `is_directed_acyclic=True`, `topology_tags=('planar_dag',)`
- prepared config: `target_aspect=0.08`, `dummy=True`, `steps=100`

Layer-span diagnostic:

- 28 edges span 1 layer.
- 7 edges span 2 layers.
- 3 edges span 3 layers.
- 5 edges span 4 layers.
- 2 edges span 5 layers.
- Dummy expansion would add 36 dummy nodes.

Measured ablation:

| Variant | Dagua | Best competitor | Delta | Crossing rate | Edge CV | Straight deg | Angular deg | Aspect |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current | 68.37 | graphviz_dot 71.84 | -3.47 | 0.1022 | 0.6895 | 3.47 | 5.51 | 0.082 |
| no_decomp | 68.37 | graphviz_dot 71.84 | -3.47 | 0.1022 | 0.6895 | 3.47 | 5.51 | 0.082 |
| no_bk | 68.11 | graphviz_dot 71.84 | -3.73 | 0.1114 | 0.6968 | 4.09 | 5.76 | 0.085 |
| no_median | 72.07 | graphviz_dot 71.84 | +0.23 | 0.0620 | 0.6914 | 2.86 | 3.96 | 0.082 |
| no_dummy | 70.92 | graphviz_dot 71.84 | -0.92 | 0.0762 | 0.6854 | 3.06 | 4.04 | 0.084 |
| no_aspect | 68.67 | graphviz_dot 71.84 | -3.17 | 0.1022 | 0.6647 | 9.75 | 15.13 | 0.254 |
| none | 67.68 | graphviz_dot 71.84 | -4.16 | 0.0967 | 0.6604 | 18.12 | 17.69 | 0.494 |

Interaction checks:

- `no_median`: `72.07`
- `no_dummy`: `70.92`
- `no_bk`: `68.11`
- `no_median_no_bk`: `71.59`
- `no_median_no_dummy`: `69.48`
- `no_dummy_no_bk`: `69.80`

Diagnosis:

Median/transpose is the primary regression source. It raises crossing rate from `0.0620` to `0.1022`, dropping the crossing contribution from `3.80 / 10` to `0.00 / 10`. Dummy nodes are secondary: disabling dummy nodes improves current by `+2.55`, mainly by reducing crossing rate to `0.0762`. BK is not the culprit here; disabling BK alone makes the score slightly worse.

This graph is a random 3-regular graph, but the classifier tags it as `planar_dag` because the directed orientation is acyclic and the planar sparsity hint is true. That tag gives it `target_aspect=0.08`, much narrower than the wave-2 plan’s proposed `0.45`. The aspect ablation only recovers `+0.30`, so aspect is not the primary regression, but the tag is semantically suspicious. A random regular graph is not a planar layered DAG even if `E < 3N - 6`.

Proposed fix:

Use the same median/transpose acceptance check proposed for `ragged_feature_pyramid`. It would reject the current median order because crossing rate worsens. Add a topology gate to keep dummy nodes off random regular-like graphs: skip dummy expansion when `max_degree <= 3`, `edge_to_node_ratio <= 1.6`, `num_layers <= 8`, and the graph lacks a wide-layered / neural / explicit DAG tag. For implementation, this should not depend on benchmark names; derive it from `GraphStructure` plus edge span stats.

Also tighten `planar_dag` tagging. The current rule in `graph_classify.py` treats sparse, bounded-degree, acyclic orientations as planar DAGs even when they are random regular graphs. Add a `layer_width_cv` ceiling for `planar_dag`, similar to `lattice_like`, or require an explicit low crossing structural hint. `regular_3_30` has `layer_width_cv=0.525`, already outside the lattice threshold of `0.45`; that is a reasonable first cutoff.

Risk:

The dummy gate could affect legitimate small planar DAGs with degree 3. Keep the first implementation as an acceptance check or use an expansion budget instead of a hard family ban. The aspect tag tightening is low risk because the current `planar_dag=0.08` target is far from the wave-2 design and does not drive a measurable win here.

## Cross-Graph Root Cause Ranking

| Graph | Primary source | Secondary source | Evidence |
|---|---|---|---|
| `ragged_feature_pyramid` | median/transpose | none | `no_median` `69.52 -> 75.82`; all other single toggles unchanged |
| `planar_60` | dummy + BK interaction | none | `no_dummy` and `no_bk` both `65.82 -> 78.74`; `no_median` unchanged |
| `parallel_cycles_4x5` | none of sprint-19 | cyclic/force structural | all variants `58.24` |
| `transformer_layer` | not sprint-19 | tiny DAG ordering gap | `none` only `76.18 -> 76.35` |
| `regular_3_30` | median/transpose | dummy nodes | `no_median` `68.37 -> 72.07`; `no_dummy` `68.37 -> 70.92` |

## Proposed Implementation Order

1. Add an acceptance guard to median/transpose.

   This is the highest-impact, lowest-risk change. It fixes `ragged_feature_pyramid` and `regular_3_30` while preserving all cases where median/transpose actually reduces crossings. The implementation can snapshot `state.pos` and `state.ordering` before `MedianSweep`, run median+transpose, compute crossings on the active graph, and restore if crossing count increases. For `E <= 2,000`, exact crossing is fine. For larger graphs, use a deterministic sample or only apply the guard to the sprint-19 small-graph path.

2. Add dummy expansion budget before expanding.

   Estimate dummy count from layer spans in `_should_use_native_dummy_nodes()`. Skip when the expansion ratio is extreme. This directly prevents `planar_60` from expanding from 60 to 1,166 active nodes. The budget should be private initially, for example:

   - `dummy_nodes <= 4 * num_nodes`
   - `expanded_nodes <= 5 * num_nodes`
   - `dummy_nodes <= 3 * num_edges`

   The exact constants should be swept against sprint-19 winners before landing.

3. Add BK-on-expanded escape condition.

   Even if dummy nodes are useful for gradient optimization, BK should not necessarily consume the expanded scaffold. In `BrandesKoepfHorizontalRefine.apply()`, if `active_num_nodes > problem.num_nodes` and `active_num_nodes / problem.num_nodes > 5`, return unchanged. That prevents BK from magnifying dummy artifacts.

4. Tighten topology tags.

   `regular_3_30` should not be `planar_dag` merely because it is sparse, bounded-degree, and acyclic under the benchmark orientation. Add `layer_width_cv <= 0.45` or a similar regular-layer criterion to the `planar_dag` rule. Revisit the implemented aspect targets: `0.05` and `0.08` are much narrower than the wave-2 design (`0.60` lattice, `0.45` planar), but this ablation shows they are not the primary cause of these five losses.

5. Leave `parallel_cycles_4x5` and `transformer_layer` to topology-dispatch work.

   Do not spend sprint-19 cleanup complexity on graphs whose ablations are no-ops. `parallel_cycles_4x5` needs cyclic/force handling. `transformer_layer` needs a tiny-DAG exact ordering improvement.

## Risk / Regression Analysis

The protected winners in sprint-20 context include `org_chart_deep`, `random_dag_200`, `hub_fanout_label_skew`, `org_chart_1_5_4_8`, `random_dag_50`, `random_bipartite_60`, `edge_label_braid`, `bipartite_4_3_4`, and the karate graphs. The proposed fixes should preserve them for these reasons:

- Median/transpose acceptance only rejects objectively worse crossing outcomes. If a winner benefits from median, it stays. If a winner is harmed, this is a bonus.
- Dummy expansion budget only rejects pathological expansion ratios. Normal long-edge DAGs with modest dummy count still run. The riskiest protected graph is `random_dag_200` if it has many skip edges, so it must be in the validation set.
- BK-on-expanded budget does not disable BK on original graphs and does not disable BK on reasonable dummy expansions.
- Topology tag tightening for `planar_dag` should not touch obvious `wide_layered`, `bipartite_dag`, or tree/forest cases.

The main regression risk is under-fixing dummy nodes: a graph like `dependency_500` may genuinely need dummies despite a large absolute dummy count. Use ratios, not absolute counts, and validate against `dependency_500`, `hexagonal_lattice_42`, `sierpinski_42`, `dense_pair_50`, and `long_skip_only_24` before landing.

## Big-Bet Proposals

This ablation reinforces the sprint-20 context warning about Frankenstein risk. The default pipeline is mixing DAG ordering, force relaxation, dummy expansion, BK compaction, topology aspect, and component packing in one path. The immediate fixes above are targeted guardrails, but the cleaner long-term direction is topology dispatch:

- A layered-DAG sub-pipeline for true Sugiyama-like graphs: dummy nodes, crossing minimization, BK, strip, aspect fit.
- A planar/mesh sub-pipeline that does not force longest-path layers into one-node-per-layer dummy chains.
- A cyclic/regular/force sub-pipeline for `parallel_cycles_4x5`, `regular_*`, and small-world graphs, using stress or force-directed objectives without DAG consistency dominating.
- A tiny-DAG exact ordering pass for `transformer_layer`-scale graphs where exact adjacent-swap optimization is cheap and dot is still winning.

Projected impact: the targeted guardrails should recover about `+6.3` on `ragged_feature_pyramid`, `+12.9` on `planar_60`, and `+3.7` on `regular_3_30` in isolation, with little suite risk. The dispatch architecture is larger but addresses the graphs that this ablation shows sprint-19 flags cannot touch.

## Concerns To Verify

- The metric rescoring differs slightly from the prompt on `regular_3_30` and `planar_60`. My run measured graphviz_dot `71.84` for `regular_3_30`, not `72.23`; ELK `75.04` for `planar_60`, effectively matching the prompt.
- The aspect targets in code differ materially from the wave-2 plan. Code uses `lattice_like=0.05`, `planar_dag=0.08`, and `dense_dag=0.05` (`dagua/layout/resolve.py:148-156`), while the plan proposed much wider targets. This did not drive the biggest regressions here, but it should be audited separately.
- `parallel_cycles_4x5` is tagged in the graph catalog as disconnected, but the classifier reports one weak component. I did not chase that discrepancy because every sprint-19 ablation was a no-op; agent A’s cyclic/component work is the better owner.
