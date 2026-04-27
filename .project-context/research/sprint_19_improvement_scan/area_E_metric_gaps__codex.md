# Area E Metric Gaps

## Method

- Read first: `CONTEXT.md`, `dagua/metrics.py`, `dagua/layout/engine.py`, and `dagua/layout/ops/pipelines/dagua_native.py`.
- Data source: cached benchmark positions from `eval_output/variant_bench_full/positions` for `dagua`, `graphviz_dot`, `dagre`, `elk_layered`, and `igraph_sugiyama` across the 93 graphs with `N <= 500`.
- Metrics recomputed directly from those positions using the formulas in `dagua/metrics.py`. All reported means are empirical over the 93-graph roster, except `cluster_separation`, which only exists on the 19 clustered graphs.
- For the “>5 metric points” cutoff, I used the per-metric 0-100 normalization induced by `composite()` rather than weighted composite contribution. That is the only interpretation that makes sense for the 5-point metrics (`angular_resolution`, `cluster_separation`).
- `crossing_rate` is itself a sampled metric. I used a 20k pair sample per layout for this report, so crossing deltas should be read as stable directional evidence rather than exact counts.
- Important benchmark caveat: the current `benchmark.py` path calls `full()` without `cluster_ids`, so `cluster_separation` is neutral-scored in the live composite today. I still recomputed it explicitly because the rubric names it as one of the eight target metrics.

## TL;DR

- The biggest real headroom is not mysterious: `overlap_count` and `edge_length_cv` are the fastest composite wins. `overlap_count` is being reintroduced by post-processing after the last overlap projection, and `edge_length_cv` is fundamentally limited because Dagua optimizes original long-span edges instead of dummy-split edges.
- `depth_spearman`, `edge_straightness`, and `crossing_rate` all point to the same missing discrete layered phase: Dagua has barycenter reordering, but it still lacks a final transpose pass and a final coordinate assignment pass such as the already-implemented `BrandesKopf4Pass`.
- The strongest observed trade-off is not `edge_straightness` vs `edge_length_cv`; it is `cluster_separation` vs `edge_length_cv` on clustered graphs (`r = -0.46` on Dagua). Spreading clusters apart lengthens inter-cluster bridges. `crossing_rate` and `angular_resolution` move together (`r = +0.54`), so improving ordering usually helps both.

## 1. `dag_consistency`

- A. Formula summary: `dag_consistency` is the fraction of edges whose target is below the source for `TB` layouts. In `composite()`, it contributes `25 * dag_consistency`, so every violated edge costs directly.
- B. Distribution: Dagua mean is `0.974 / 97.4 score`; competitor average is `0.945 / 94.5 score` (`dot 97.3`, `dagre 97.6`, `elk 85.6`, `igraph 97.6`). Dagua loses by >5 score points on only two graphs: `recurrent_feedback_cell` (`50.0` vs `66.7`) and `center_port_backedge_hub` (`66.7` vs `77.8`). Dagua wins by >5 on 18 graphs, especially clustered residual / compound DAGs such as `compound_dag_5x30`, `nested_shallow_enc_dec`, `multiscale_skip_cascade`, `resnet_stack_4x16`, and `compound_10x20`.
- C. Root cause: the losses are both cyclic/self-loop motifs. In `build_loss_ops()`, `DagOrderingLoss` is skipped on cyclic graphs by design. After that, the default path relies on `init_positions()` plus `Force2DInitIfFlat`, but no later op hard-projects the “forward” edges of a feedback graph back onto ranks. That is why Dagua is usually excellent on DAGs and only slips on tiny recurrent motifs.
- D. Fix: add a feedback-aware rank projection for cyclic graphs: compute a `back_edge_mask` from the cycle breaker, optimize only forward edges for ordering, then run a cheap `ForwardEdgeRankProjection` after `gradient_core`. Estimated suite impact: `+0.05` to `+0.10` composite. The theoretical max headroom here is only `+0.08`, so this is a cleanup item, not the main lever.

## 2. `edge_length_cv`

- A. Formula summary: `edge_length_cv = std(edge_length) / mean(edge_length)`. `composite()` uses `20 * max(0, 1 - CV)`. The metric rewards uniform edge lengths and punishes even a small set of very long bridges.
- B. Distribution: Dagua mean is `0.754 / 36.9 score`; competitor average is `0.742 / 35.1 score`, but that average hides a fatter left tail for Dagua on the layered cases that matter. Dagua loses by >5 score points on 46 graphs. Worst cases are `grid_20x20` (`20.2` vs `100.0`), `cluster_member_style_stress`, `hexagonal_lattice_42`, `complete_bipartite_8x12`, `random_bipartite_60`, `rgg_100`, `citation_dag_300`, and `interleaved_cluster_crosstalk`. The dominant loss tags are `wide-parallel`, `diamond`, `skip-light`, `mixed-width`, and `skip-heavy`. Dagua wins by >5 on 52 graphs, especially trees and already-compact graphs such as `org_chart_1_5_4_8`, `org_chart_deep`, `grid_5x5`, `binary_tree`, and `dense_pair_50`.
- C. Root cause: this is the clearest structural gap in the current pipeline. `EdgeLengthVarianceLoss` and `EdgeAttractionLoss` operate on the original edge set, so an edge that spans many logical ranks is still optimized as one literal long segment. Dagua never inserts virtual nodes or split segments, so long skip edges dominate the variance even when ordering is good. This is why wide DAGs, bipartite layers, and cluster handoff graphs keep showing up.
- D. Fix: add Sugiyama Phase 1.5 dummy-node edge splitting before ordering / coordinate assignment, then optimize length variance on the split graph and collapse the virtual chain back to the original edge at the end. Pair that with a span-aware weighting so a 5-rank bridge does not compete with a 1-rank local edge as if they were the same aesthetic object. Estimated suite impact: `+0.8` to `+1.2` composite. The theoretical metric-only headroom is `+1.96`, so this is the single biggest structural opportunity after overlap cleanup.

## 3. `depth_spearman_rho`

- A. Formula summary: this is the Spearman rank correlation between topological depth and `y`. `composite()` uses `15 * max(0, rho)`. It rewards exact rank monotonicity, not just “mostly downward” edges.
- B. Distribution: Dagua mean is `0.951 / 95.1 score`; competitor average is `0.887 / 89.5 score`. Dagua loses by >5 score points on 10 graphs, and the bad ones are concentrated: `wide_1_100_1` (`24.0` vs `100.0`), `wide_single_layer_1_50_1` (`33.4` vs `100.0`), `wide_3_50_3` (`53.7` vs `100.0`), plus `disconnected_label_cycle_collage`, `complete_bipartite_8x12`, `random_bipartite_60`, `inception_block`, `random_dag_50`, `hub_fanout_label_skew`, and `bipartite_4_3_4`. Dagua wins by >5 on 48 graphs, especially clustered compound DAGs.
- C. Root cause: `NativeEngineInit` starts from longest-path layers, but `gradient_core` does not preserve exact layer `y` coordinates. `DagOrderingLoss` only penalizes inversions and encourages a minimum separation; it does not lock nodes to discrete ranks. On very wide 3-layer graphs, repulsion and overlap avoidance spread the middle layer vertically, so Dagua keeps directionality while losing exact depth correlation.
- D. Fix: add a DAG-only `RankProjectY` or `LayerLockY` phase after `gradient_core`, or reuse the already computed `state.layers` in a final coordinate assignment pass. This can also be folded into `BrandesKopf4Pass` if the ordering state is materialized. Estimated suite impact: `+0.20` to `+0.35` composite. Theoretical headroom is `+0.49`, so a hard rank restore should recover most of it.

## 4. `overlap_count`

- A. Formula summary: `count_overlaps_detailed()` counts overlapping node bounding-box pairs. The composite is binary here: `10` points for zero overlaps, `0` otherwise. One collision is enough to drop the full metric.
- B. Distribution: Dagua mean is `11.26 overlaps / 77.4 score`; competitor average is `3.32 overlaps / 89.0 score`, and the three main layered competitors are basically overlap-free on almost every graph (`~98.9 score`). Dagua loses by >5 score points on 21 graphs, including `hub_fanout_label_skew`, `disconnected_label_cycle_collage`, `cluster_member_style_stress`, `protein_ppi_200`, `citation_dag_300`, `multi_component_80`, `scale_free_ba_120`, `dependency_graph_100`, `real_lesmis_77`, `real_football_115`, `er_500`, and `rgg_100`. Dagua wins by >5 on 29 graphs, mostly against `igraph_sugiyama`.
- C. Root cause: the pipeline order is the bug. `dagua_native.py` runs `OverlapProjection`, then `AspectRatioFit`, then `ClusterGridArrange`, and stops. Both post-process ops can move nodes after the last collision fix. That exactly matches the loss profile: mostly graphs with label width skew, cluster rearrangement, or component reshaping.
- D. Fix: run one more `OverlapProjection` after `AspectRatioFit` and `ClusterGridArrange`, or make those ops collision-aware. This is the cleanest short-term win in the whole file because it attacks a binary 10-point cliff with minimal algorithmic risk. Estimated suite impact: `+1.0` to `+1.5` composite. Theoretical headroom is `+2.26`, so even a partial cleanup is large.

## 5. `edge_straightness_mean_deg`

- A. Formula summary: the metric is the mean angular deviation from the primary axis. For `TB`, it is `atan2(|dx|, |dy|)` in degrees; smaller is better. `composite()` uses `10 * max(0, 1 - deg / 45)`.
- B. Distribution: Dagua mean is `35.1 deg / 40.3 score`; competitor average is `38.9 deg / 31.6 score`, but the graph-by-graph tail still hurts Dagua on 31 graphs. Worst losses are `outerplanar_dag_20` (`0.0` vs `80.6`), `disconnected_label_cycle_collage`, `grid_20x20`, `org_chart_1_5_4_8`, `recurrent_feedback_cell`, `center_port_backedge_hub`, `shape_and_routing_matrix`, `rgg_500`, `hexagonal_lattice_42`, and `grid_rect_6x8`. Dagua wins by >5 on 42 graphs, especially clustered residual / compound DAGs.
- C. Root cause: Dagua has a straightness loss and a late `BarycenterReorder`, but it does not have a final layered coordinate assignment. `BarycenterReorder` only permutes the existing `x` values within each layer; it never computes new `x` coordinates from the order. So if the continuous optimizer has already produced slanted geometry, the last pass cannot really make edges vertical again.
- D. Fix: integrate the existing `BrandesKopf4Pass` op after ordering. The op is already in `dagua/layout/ops/coordinate.py`; the gap is wiring, not invention. The likely good sequence is `BarycenterReorder -> TransposeHeuristic -> BrandesKopf4Pass -> OverlapProjection`. Estimated suite impact: `+0.25` to `+0.45` composite standalone, and more if paired with dummy-node splitting.

## 6. `crossing_rate`

- A. Formula summary: `sampled_crossing_rate()` samples non-adjacent edge pairs and estimates the fraction that geometrically intersect. `composite()` uses `10 * max(0, 1 - 10 * crossing_rate)`.
- B. Distribution: Dagua mean is `0.0283 / 73.7 score`; competitor average is `0.0220 / 80.2 score`. Dagua loses by >5 score points on 46 graphs. The biggest misses are `interleaved_cluster_crosstalk` (`34.1` vs `100.0`), `regular_3_30`, `real_karate_34`, `weighted_karate_34`, `er_100`, `er_500`, `edge_label_braid`, `regular_4_40`, `rgg_100`, `ragged_feature_pyramid`, and `sbm_4x30`. `graphviz_dot` is the best competitor on 33 of those 46 losses. Dagua wins by >5 on 23 graphs, mainly on deeply hierarchical nets where ELK or igraph route merges poorly.
- C. Root cause: Dagua’s crossing term is still a soft proxy with `max_pairs=500`, and the only discrete polish is barycenter reordering. The init path has a transpose heuristic, but the final pipeline does not. That means Dagua often exits with a reasonable ordering but not the local adjacent swaps that classical Sugiyama implementations use to squeeze out the last crossings.
- D. Fix: add a registered transpose pass after barycenter reordering, and raise crossing attention only on layered DAG families. The repo already has `TransposeHeuristic` in `dagua/layout/ops/ordering.py`; the missing piece is using it in `dagua_native`. Estimated suite impact: `+0.4` to `+0.7` composite. Dummy-edge splitting will also help here by turning one long crossing-prone bridge into rank-local segments.

## 7. `angular_res_mean_deg`

- A. Formula summary: for each node with degree at least 2, the metric takes the minimum angle between incident edges and averages those minima. `composite()` caps it at `5 * min(1, angle / 40)`.
- B. Distribution: Dagua mean is `54.7 deg / 68.7 score`; competitor average is `58.8 deg / 75.3 score`. Dagua loses by >5 score points on 33 graphs, especially `small_world_100` (`0.0` vs `100.0`), `wide_1_100_1`, `grid_20x20`, `wide_single_layer_1_50_1`, `sierpinski_42`, `random_bipartite_60`, `er_500`, `real_karate_34`, `weighted_karate_34`, `er_100`, `org_chart_deep`, and `powerlaw_500`. Dagua wins by >5 on 16 graphs, such as `outerplanar_dag_20`, `interleaved_cluster_crosstalk`, `small_label_storm`, and `moe_router_sparse`.
- C. Root cause: Dagua has a very weak `FanoutDistributionLoss` (`w_fanout = 0.3`) and no explicit port-order / incident-angle phase. The init code has `_spread_fanout_children()`, but that happens before the optimizer and only for high-degree hubs. Later attraction, crossing, and cluster forces can collapse incident edges back together.
- D. Fix: add a degree-gated post-process such as `IncidentAngleSpread` or a light port-order projection for nodes with degree >= 4, and strengthen `w_fanout` only on families where center-port bunching is common. Estimated suite impact: `+0.10` to `+0.25` composite. This metric is only worth 5 points, so it should stay gated and low-risk.

## 8. `cluster_separation`

- A. Formula summary: `cluster_separation()` computes centroid distance divided by average intra-cluster spread for each cluster pair, then averages those ratios. `composite()` caps it at `5 * min(1, ratio / 5)`.
- B. Distribution: this metric exists on 19 clustered graphs. On that subset, Dagua mean is `6.51 / 56.3 score`; competitor average is `7.10 / 53.3 score`. Dagua loses by >5 score points on 11 clustered graphs, especially `clustered_medium_5x20`, `moe_router_sparse`, `dependency_graph_100`, `multiscale_skip_cascade`, `kitchen_sink_platform_graph`, `small_label_storm`, `transformer_full_4h_2l`, `clustered_longlabel_handoffs`, `dependency_500`, `nested_cluster_label_stack`, and `interleaved_cluster_crosstalk`. Dagua wins by >5 on 11 too, led by `compound_dag_5x30`, `compound_10x20`, `dependency_500`, and `resnet_stack_4x16`.
- C. Root cause: there are two separate stories. First, the metric is not actually in the live benchmark because `benchmark.py` does not pass `cluster_ids` into `full()`. Second, on the geometry side, Dagua’s current cluster shaping is split between low-weight cluster losses (`w_cluster = 1.0`, separation default `0.5`) and a very late `ClusterGridArrange`. That late arrange helps compound DAGs but can worsen edge length and ordering because nothing repairs geometry afterward.
- D. Fix: if this metric is meant to matter, first fix the benchmark path so clustered graphs are not neutral-scored. Then replace the current all-or-nothing late arrange with a cluster-centroid phase that is DAG-aware and followed by final overlap / coordinate repair. Estimated impact is `+0.00` composite on the current benchmark as written, and `+0.05` to `+0.15` if the benchmark starts scoring the metric correctly.

## Composite Couplings

- Strongest observed anti-correlation in Dagua is `cluster_separation` vs `edge_length_cv` on the 19 clustered graphs (`r = -0.46`). The meaning is straightforward: pushing sibling clusters farther apart stretches bridge edges and worsens length variance. This is the one trade-off that should be explicit in any cluster-aware change.
- Weaker anti-correlations are `dag_consistency` vs `cluster_separation` (`r = -0.32`) and `depth_spearman` vs `cluster_separation` (`r = -0.31`). Cluster spreading can force extra vertical distortion unless the rank structure is re-applied afterward.
- The commonly feared `edge_straightness` vs `edge_length_cv` trade-off did not show up as a real suite-level blocker. It is mildly positive (`r = +0.11` on Dagua, `+0.26` across all layouts). In other words, the repo is currently leaving “good layered geometry” on the table rather than sitting on a hard Pareto frontier there.
- `crossing_rate` and `angular_resolution` are positively correlated (`r = +0.54`). Better discrete ordering tends to separate incident bundles and reduce crossings at the same time.
- `overlap_count` is special because it is binary in the composite. One post-process collision wipes a full 10 points, so a final collision-repair pass is worth more than small smooth improvements in several other metrics.

## Theoretical Headroom

- If Dagua matched the per-graph best cached competitor on exactly one metric and everything else stayed unchanged, the maximum mean composite gain would be about `+2.26` from `overlap_count`, `+1.96` from `edge_length_cv`, `+1.28` from `crossing_rate`, `+0.76` from `edge_straightness`, `+0.65` from `angular_resolution`, `+0.49` from `depth_spearman`, `+0.15` from `cluster_separation`, and `+0.08` from `dag_consistency`.
- That ranking matters because it separates “structural” work from “cleanup” work. `overlap_count` and `edge_length_cv` are not just the most visible failures; they are the two largest remaining composite drains even before any secondary synergies are counted.
- The headroom ordering also explains why the best bundle is not eight separate metric-specific tweaks. A dummy-node + transpose + Brandes-Kopf layered finish should hit `edge_length_cv`, `depth_spearman`, `edge_straightness`, `crossing_rate`, and some `angular_resolution` together, while a final overlap repair is a largely independent binary win.

## Recommended Queue

1. Re-run `OverlapProjection` after `AspectRatioFit` and `ClusterGridArrange`. Expected net: `+1.0` to `+1.5` composite.
2. Add dummy-node edge splitting for long-span edges, then score / optimize length variance on the split graph. Expected net: `+0.8` to `+1.2`.
3. Wire in `TransposeHeuristic` and `BrandesKopf4Pass` as the actual last layered phase. Expected net: `+0.7` to `+1.1` combined across `crossing_rate`, `edge_straightness`, and `depth_spearman`.
4. Add a DAG-only rank projection after `gradient_core` so wide layered graphs cannot drift vertically. Expected net: `+0.2` to `+0.35`.
5. Add a degree-gated incident-angle / port-spread polish pass. Expected net: `+0.1` to `+0.25`.
6. Only after the benchmark path is fixed, tune cluster separation more aggressively; until then its composite ROI is literally zero.
