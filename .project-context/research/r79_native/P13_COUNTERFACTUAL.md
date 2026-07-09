# P13: r80 Sprint Counterfactual -- Pre-Sprint Dagua on the Honest Ruler

## Method

Goal: isolate how much of the r80 sprint's reported improvement is real algorithm
gain vs. how much is a change in the scoring ruler (size-aware externals + prism
overlap + guarded composite, landed in the S9/P11-P12 honesty work).

1. Pulled the pre-sprint frozen store from git history: `38a13e1^` (= `7b5308d`,
   "S4 complete -- 87/108 best-or-tied verified", dated 2026-07-06), before the
   `r80/undirected-portfolio` merge. Extracted all 108 `*__dagua.pt` position
   tensors from that commit via `git show 38a13e1^:eval_output/r79_baseline/positions/<graph>__dagua.pt`.
2. Loaded the current corpus (`dagua.eval.graphs.get_test_graphs(max_nodes=500)`,
   108 graphs -- identical graph set to the pre-sprint store, zero corpus drift).
3. Rescored every pre-sprint dagua position tensor with the CURRENT scoring code
   exactly as `scripts/r79_baseline.py::run_engine()` does for dagua rows:
   `dagua.metrics.evaluate(graph, positions, tier="full")` then
   `dagua.metrics.composite_auto(metrics, is_semantically_directed(test_graph))`.
   No reimplementation -- imported and called the real functions.
4. Compared each pre-sprint composite against the CURRENT store's best external
   row per graph (`eval_output/r79_baseline/results.json`, HEAD `53050bf`,
   the honest re-freeze: size-aware + prism externals, guarded composite),
   same 0.5 tie band used everywhere else in this sprint.
5. Also recomputed the CURRENT store's own dagua-vs-external verdicts the same
   way, as a sanity check against the frozen headline (`52/13/28 + 6/3/6 =
   74/108`) -- reproduced exactly (58/16/34 combined = 52+6/13+3/28+6).

Sanity check: the freshly recomputed pre-sprint composites are bit-identical
(max abs diff 0.0 across all 108 graphs) to the composites originally stored
in the `38a13e1^` results.json. This confirms dagua's own composite formula
is unchanged by the P6/P11/P12 honesty work -- only the EXTERNAL side of the
comparison changed (size-aware node sizing + prism overlap for sfdp/neato/fdp
+ the degeneracy-guarded composite). The counterfactual isolates exactly that:
same dagua layouts, same graphs, honest ruler on both sides.

No corpus drift: all 108 pre-sprint graphs are present in the current corpus
and vice versa (`set difference == {}` both directions). Zero N/A rows.

## Headline

| | Pre-sprint dagua (honest ruler) | Current dagua (honest ruler, HEAD) | Delta |
|---|---|---|---|
| Legacy (93 graphs) | 50 W / 5 T / 38 L (55/93 best-or-tied) | 52 W / 13 T / 28 L (65/93 best-or-tied) | +2 W, +8 T, -10 L |
| Extended (15 graphs) | 5 W / 3 T / 7 L (8/15 best-or-tied) | 6 W / 3 T / 6 L (9/15 best-or-tied) | +1 W, +0 T, -1 L |
| **Combined (108 graphs)** | **55 W / 8 T / 45 L (63/108 best-or-tied)** | **58 W / 16 T / 34 L (74/108 best-or-tied)** | **+3 W, +8 T, -11 L** |

**True sprint value on the honest ruler: 63/108 -> 74/108 best-or-tied (+11
net), zero regressions.** The r80 sprint's raw headline (74 -> 87/108 on the
old ruler, from the `undirected-portfolio` merge) overstated the gain because
part of that jump came from the ruler getting stricter on externals at the
same time the algorithm improved. Re-scoring the OLD dagua layouts on the NEW
ruler shows the honest starting point was 63/108, not 74/108 -- so the honest
sprint gain is +11/108, and all 11 flips are pre-L -> current-W/T with none
going the other way (verified below).

## Verdict flips (pre-sprint -> current, 11 of 108)

All 11 movers flip from Loss to Win/Tie; **zero graphs regress** (no W/T -> L
flip anywhere in the corpus). All 11 are undirected-class graphs, consistent
with the `r80/undirected-portfolio` route being the mechanism (not a scoring
artifact):

| Graph | Population | Pre verdict | Current verdict | Pre composite | Current composite | Dagua delta |
|---|---|---|---|---|---|---|
| grid_5x5 | legacy | L | T | 94.21 | 94.53 | +0.32 |
| grid_rect_6x8 | legacy | L | T | 92.87 | 94.26 | +1.39 |
| hexagonal_lattice_42 | legacy | L | W | 78.13 | 93.92 | +15.79 |
| multi_component_80 | legacy | L | W | 81.55 | 92.52 | +10.97 |
| petersen_10 | legacy | L | T | 57.22 | 79.02 | +21.80 |
| planar_60 | legacy | L | T | 57.31 | 77.72 | +20.42 |
| random_bipartite_60 | legacy | L | T | 65.24 | 79.09 | +13.84 |
| regular_3_30 | legacy | L | T | 62.32 | 84.34 | +22.02 |
| sierpinski_42 | legacy | L | T | 88.33 | 90.40 | +2.08 |
| triangular_lattice_36 | legacy | L | T | 82.26 | 94.48 | +12.22 |
| r79_undirected_sbm_mid_mix_5x20 | extended | L | W | 49.61 | 55.98 | +6.37 |

## Full per-graph table (108 rows)

| Graph | Pop | Pre-sprint dagua | Current dagua | Best external (engine) | Verdict (pre) | Verdict (current) | Moved? |
|---|---|---|---|---|---|---|---|
| r79_directed_scc_120_2cores | extended | 58.38 | 58.38 | 59.99 (graphviz_dot) | L | L |  |
| r79_directed_scc_90_3cores | extended | 57.18 | 57.18 | 57.06 (graphviz_dot) | T | T |  |
| r79_nested_clusters_2x3x12 | extended | 77.77 | 78.35 | 71.60 (graphviz_dot) | W | W |  |
| r79_nested_clusters_3x2x10 | extended | 70.90 | 70.92 | 71.12 (graphviz_dot) | T | T |  |
| r79_nested_clusters_4x2x8 | extended | 71.52 | 71.41 | 69.77 (graphviz_dot) | W | W |  |
| r79_undirected_sbm_high_mix_3x30 | extended | 36.54 | 46.88 | 47.54 (graphviz_sfdp) | L | L |  |
| r79_undirected_sbm_low_mix_4x25 | extended | 52.78 | 62.39 | 64.21 (graphviz_neato) | L | L |  |
| r79_undirected_sbm_mid_mix_5x20 | extended | 49.61 | 55.98 | 53.72 (graphviz_sfdp) | L | W | yes |
| r79_weighted_bipartite_16x24 | extended | 64.71 | 64.71 | 64.60 (graphviz_neato) | T | T |  |
| r79_weighted_community_4x18 | extended | 46.21 | 58.37 | 68.36 (graphviz_neato) | L | L |  |
| r79_weighted_hub_spoke_4x18 | extended | 75.50 | 75.50 | 62.63 (elk_layered) | W | W |  |
| r79_weighted_ladder_40 | extended | 94.70 | 94.70 | 89.50 (elk_layered) | W | W |  |
| r79_weighted_mesh_10x12 | extended | 87.84 | 87.84 | 89.88 (graphviz_neato) | L | L |  |
| r79_weighted_skew_dag_6x10 | extended | 83.44 | 83.44 | 77.83 (graphviz_dot) | W | W |  |
| r79_weighted_small_world_120 | extended | 32.65 | 45.28 | 68.15 (graphviz_neato) | L | L |  |
| asymmetric_hourglass_hub | legacy | 79.17 | 79.17 | 73.24 (graphviz_dot) | W | W |  |
| ba_500 | legacy | 44.30 | 44.30 | 60.83 (graphviz_sfdp) | L | L |  |
| binary_tree | legacy | 86.40 | 86.40 | 84.19 (graphviz_dot) | W | W |  |
| bipartite_4_3_4 | legacy | 78.75 | 78.75 | 68.94 (elk_layered) | W | W |  |
| braided_feedback_tails | legacy | 79.89 | 79.89 | 77.90 (graphviz_dot) | W | W |  |
| broken_symmetry_residual_pair | legacy | 74.78 | 74.78 | 76.22 (elk_layered) | L | L |  |
| center_port_backedge_hub | legacy | 59.98 | 59.98 | 57.25 (dagre) | W | W |  |
| chung_lu_150 | legacy | 39.62 | 50.99 | 67.40 (graphviz_sfdp) | L | L |  |
| citation_dag_300 | legacy | 59.30 | 59.30 | 57.36 (elk_layered) | W | W |  |
| cluster_member_style_stress | legacy | 81.37 | 81.37 | 74.87 (dagre) | W | W |  |
| clustered_longlabel_handoffs | legacy | 85.49 | 85.49 | 78.93 (dagre) | W | W |  |
| clustered_medium_5x20 | legacy | 65.27 | 65.31 | 66.88 (graphviz_dot) | L | L |  |
| complete_bipartite_8x12 | legacy | 63.65 | 63.65 | 61.15 (elk_layered) | W | W |  |
| compound_10x20 | legacy | 71.46 | 71.46 | 70.72 (graphviz_dot) | W | W |  |
| compound_dag_5x30 | legacy | 71.38 | 71.38 | 71.00 (graphviz_dot) | T | T |  |
| deep_chain_20 | legacy | 87.93 | 87.93 | 88.28 (elk_layered) | T | T |  |
| dense_pair_50 | legacy | 70.22 | 70.22 | 69.72 (graphviz_dot) | T | T |  |
| densenet_block | legacy | 68.48 | 68.48 | 67.98 (graphviz_dot) | W | W |  |
| dependency_500 | legacy | 54.56 | 54.56 | 57.71 (elk_layered) | L | L |  |
| dependency_graph_100 | legacy | 57.11 | 57.11 | 58.10 (elk_layered) | L | L |  |
| disconnected_encoder_residual | legacy | 77.20 | 77.20 | 81.29 (elk_layered) | L | L |  |
| disconnected_label_cycle_collage | legacy | 77.86 | 77.86 | 73.20 (graphviz_dot) | W | W |  |
| edge_label_braid | legacy | 81.98 | 81.98 | 77.97 (dagre) | W | W |  |
| er_100 | legacy | 56.02 | 56.02 | 75.48 (graphviz_sfdp) | L | L |  |
| er_500 | legacy | 46.82 | 51.72 | 70.93 (graphviz_sfdp) | L | L |  |
| extreme_mixed_width_transformer | legacy | 81.78 | 81.78 | 74.00 (dagre) | W | W |  |
| grid_20x20 | legacy | 93.44 | 93.44 | 94.45 (graphviz_dot) | L | L |  |
| grid_5x5 | legacy | 94.21 | 94.53 | 95.00 (igraph_sugiyama) | L | T | yes |
| grid_rect_6x8 | legacy | 92.87 | 94.26 | 94.26 (graphviz_neato) | L | T | yes |
| heavy_tail_weights_50 | legacy | 67.47 | 67.47 | 80.53 (graphviz_neato) | L | L |  |
| hexagonal_lattice_42 | legacy | 78.13 | 93.92 | 91.75 (graphviz_dot) | L | W | yes |
| hierarchical_residual_stage | legacy | 84.96 | 84.96 | 83.62 (dagre) | W | W |  |
| hub_and_spoke_3x20 | legacy | 77.27 | 77.27 | 67.82 (graphviz_dot) | W | W |  |
| hub_fanout_label_skew | legacy | 88.35 | 88.35 | 74.96 (dagre) | W | W |  |
| hub_skip_superfan | legacy | 75.59 | 75.59 | 74.31 (graphviz_dot) | W | W |  |
| hub_spoke_10x20 | legacy | 73.34 | 73.34 | 64.48 (graphviz_dot) | W | W |  |
| hub_spoke_5x50 | legacy | 75.10 | 75.10 | 68.27 (elk_layered) | W | W |  |
| inception_block | legacy | 89.64 | 89.64 | 79.70 (dagre) | W | W |  |
| interleaved_cluster_crosstalk | legacy | 75.15 | 75.15 | 67.08 (graphviz_dot) | W | W |  |
| kitchen_sink_hybrid_net | legacy | 71.54 | 71.54 | 67.88 (graphviz_neato) | W | W |  |
| kitchen_sink_platform_graph | legacy | 82.15 | 82.15 | 78.49 (graphviz_dot) | W | W |  |
| linear_3layer_mlp | legacy | 90.60 | 90.60 | 91.64 (elk_layered) | L | L |  |
| long_range_residual_ladder | legacy | 74.91 | 74.91 | 69.27 (graphviz_dot) | W | W |  |
| long_skip_only_24 | legacy | 81.06 | 81.06 | 78.33 (graphviz_dot) | W | W |  |
| mixed_width_labels | legacy | 84.14 | 84.14 | 80.46 (dagre) | W | W |  |
| moe_router_sparse | legacy | 83.93 | 83.93 | 78.45 (dagre) | W | W |  |
| multi_component_80 | legacy | 81.55 | 92.52 | 88.22 (graphviz_neato) | L | W | yes |
| multiscale_skip_cascade | legacy | 70.89 | 70.89 | 67.02 (dagre) | W | W |  |
| nested_cluster_label_stack | legacy | 88.48 | 88.48 | 87.74 (graphviz_dot) | W | W |  |
| nested_shallow_enc_dec | legacy | 89.11 | 89.11 | 89.11 (igraph_sugiyama) | T | T |  |
| org_chart_1_5_4_8 | legacy | 89.13 | 89.13 | 74.90 (graphviz_dot) | W | W |  |
| org_chart_deep | legacy | 83.89 | 83.89 | 63.42 (elk_layered) | W | W |  |
| outerplanar_dag_20 | legacy | 69.33 | 69.33 | 69.93 (igraph_sugiyama) | L | L |  |
| parallel_cycles_4x5 | legacy | 58.38 | 58.38 | 60.46 (elk_layered) | L | L |  |
| parallel_multiedge_bundle | legacy | 84.03 | 84.03 | 84.97 (elk_layered) | L | L |  |
| petersen_10 | legacy | 57.22 | 79.02 | 79.41 (graphviz_neato) | L | T | yes |
| planar_60 | legacy | 57.31 | 77.72 | 77.80 (graphviz_sfdp) | L | T | yes |
| powerlaw_500 | legacy | 63.78 | 63.78 | 60.99 (elk_layered) | W | W |  |
| protein_ppi_200 | legacy | 52.05 | 52.05 | 70.61 (graphviz_neato) | L | L |  |
| ragged_feature_pyramid | legacy | 76.92 | 76.92 | 74.45 (graphviz_dot) | W | W |  |
| random_bipartite_60 | legacy | 65.24 | 79.09 | 78.77 (graphviz_sfdp) | L | T | yes |
| random_dag_200 | legacy | 67.48 | 67.48 | 57.09 (igraph_sugiyama) | W | W |  |
| random_dag_50 | legacy | 67.98 | 67.98 | 61.31 (igraph_sugiyama) | W | W |  |
| real_football_115 | legacy | 30.64 | 31.61 | 53.76 (graphviz_neato) | L | L |  |
| real_karate_34 | legacy | 50.11 | 68.79 | 74.67 (graphviz_neato) | L | L |  |
| real_lesmis_77 | legacy | 50.62 | 50.62 | 67.14 (graphviz_sfdp) | L | L |  |
| recurrent_feedback_cell | legacy | 74.69 | 74.69 | 73.27 (dagre) | W | W |  |
| regular_3_30 | legacy | 62.32 | 84.34 | 84.74 (graphviz_neato) | L | T | yes |
| regular_4_40 | legacy | 50.74 | 71.71 | 73.34 (graphviz_sfdp) | L | L |  |
| residual_block | legacy | 79.23 | 79.23 | 77.29 (graphviz_dot) | W | W |  |
| resnet_stack_4x16 | legacy | 73.66 | 73.66 | 72.42 (dagre) | W | W |  |
| rgg_100 | legacy | 47.38 | 47.38 | 67.75 (graphviz_neato) | L | L |  |
| rgg_500 | legacy | 53.49 | 53.49 | 69.87 (graphviz_neato) | L | L |  |
| sbm_4x30 | legacy | 46.15 | 46.15 | 61.33 (graphviz_neato) | L | L |  |
| sbm_5x50 | legacy | 45.79 | 45.79 | 61.92 (graphviz_neato) | L | L |  |
| scale_free_ba_120 | legacy | 34.86 | 36.74 | 61.19 (graphviz_sfdp) | L | L |  |
| shape_and_routing_matrix | legacy | 92.34 | 92.34 | 85.81 (dagre) | W | W |  |
| sierpinski_42 | legacy | 88.33 | 90.40 | 90.47 (graphviz_neato) | L | T | yes |
| small_label_storm | legacy | 91.23 | 91.23 | 85.59 (dagre) | W | W |  |
| small_world_100 | legacy | 91.75 | 91.75 | 84.71 (graphviz_neato) | W | W |  |
| small_world_500 | legacy | 42.88 | 42.88 | 66.39 (graphviz_sfdp) | L | L |  |
| sparse_pair_50 | legacy | 80.66 | 80.66 | 78.12 (elk_layered) | W | W |  |
| transformer_full_4h_2l | legacy | 74.47 | 74.47 | 71.64 (dagre) | W | W |  |
| transformer_layer | legacy | 75.03 | 75.03 | 74.40 (graphviz_dot) | W | W |  |
| triangular_lattice_36 | legacy | 82.26 | 94.48 | 94.48 (graphviz_neato) | L | T | yes |
| unet_small | legacy | 76.84 | 76.84 | 74.13 (elk_layered) | W | W |  |
| weighted_chain_20 | legacy | 88.02 | 88.02 | 88.02 (dagre) | T | T |  |
| weighted_clusters_3x10 | legacy | 45.08 | 68.05 | 72.87 (graphviz_neato) | L | L |  |
| weighted_karate_34 | legacy | 50.11 | 69.55 | 74.67 (graphviz_neato) | L | L |  |
| wide_1_100_1 | legacy | 86.72 | 86.72 | 69.62 (elk_layered) | W | W |  |
| wide_3_50_3 | legacy | 70.90 | 70.90 | 59.44 (elk_layered) | W | W |  |
| wide_single_layer_1_50_1 | legacy | 86.47 | 86.47 | 69.93 (elk_layered) | W | W |  |
| width_skew_late_merge | legacy | 76.65 | 76.65 | 75.99 (graphviz_dot) | W | W |  |
