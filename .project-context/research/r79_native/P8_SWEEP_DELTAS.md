# r80-S4 Gate 3: Full Sweep Before/After Deltas

## W/T/L by population

| population | before W/T/L | after W/T/L |
|---|---|---|
| extended | 8/2/5 | 8/2/5 |
| legacy | 56/8/29 | 63/14/16 |

**Undirected class best-or-tied: 12 -> 25 (gain +13; acceptance >= +6)**

**WIN->LOSS flips: 0** []

**GATE 3 PASSES**

Undirected-class dagua wall time: 1144.2s -> 1649.6s (frozen store recorded vs this sweep; note this sweep ran under heavy shared-machine load, so absolute multipliers overstate the route cost -- see directed-graph runtimes in the same sweep for the load factor).

## Per-graph table (all scored graphs)

| graph | pop | undirected | before | after | delta | best-ext | verdict before->after | runtime before->after (s) |
|---|---|---|---|---|---|---|---|---|
| weighted_clusters_3x10 | legacy | Y | 45.08 | 68.05 | +22.98 | dagre 57.13 | LOSS->WIN | 9.0->10.1 |
| regular_3_30 | legacy | Y | 62.32 | 84.34 | +22.02 | graphviz_dot 68.28 | LOSS->WIN | 0.5->3.7 |
| petersen_10 | legacy | Y | 57.22 | 79.02 | +21.80 | graphviz_sfdp 78.44 | LOSS->WIN | 0.3->2.5 |
| weighted_karate_34 | legacy | Y | 50.11 | 69.55 | +19.44 | graphviz_dot 58.56 | LOSS->WIN | 5.2->8.0 |
| real_karate_34 | legacy | Y | 50.11 | 68.79 | +18.68 | graphviz_dot 58.56 | LOSS->WIN | 0.6->6.4 |
| hexagonal_lattice_42 | legacy | Y | 78.13 | 93.92 | +15.79 | graphviz_neato 93.92 | LOSS->TIE | 0.7->6.2 |
| triangular_lattice_36 | legacy | Y | 82.26 | 94.48 | +12.22 | graphviz_neato 94.48 | LOSS->TIE | 0.8->8.9 |
| chung_lu_150 | legacy | Y | 39.62 | 50.99 | +11.37 | igraph_kamada_kawai 46.84 | LOSS->WIN | 3.2->25.8 |
| multi_component_80 | legacy | Y | 81.55 | 92.52 | +10.97 | graphviz_neato 92.52 | LOSS->TIE | 0.9->14.0 |
| regular_4_40 | legacy | Y | 50.74 | 56.32 | +5.58 | igraph_kamada_kawai 54.74 | LOSS->WIN | 1.0->10.8 |
| sierpinski_42 | legacy | Y | 88.33 | 90.40 | +2.08 | graphviz_neato 90.39 | LOSS->TIE | 0.7->9.9 |
| grid_rect_6x8 | legacy | Y | 92.87 | 94.26 | +1.39 | graphviz_neato 94.26 | LOSS->TIE | 0.9->12.5 |
| planar_60 | legacy | Y | 57.31 | 57.86 | +0.55 | dagre 62.37 | LOSS->LOSS | 2.1->20.8 |
| grid_5x5 | legacy | Y | 94.21 | 94.53 | +0.32 | igraph_sugiyama 95.00 | LOSS->TIE | 0.5->6.6 |
| linear_3layer_mlp | legacy |  | 90.60 | 90.60 | +0.00 | dagre 90.60 | TIE->TIE | 0.6->2.3 |
| deep_chain_20 | legacy |  | 87.93 | 87.93 | +0.00 | dagre 87.93 | TIE->TIE | 0.5->1.4 |
| inception_block | legacy |  | 89.64 | 89.64 | +0.00 | dagre 76.88 | WIN->WIN | 1.3->3.8 |
| residual_block | legacy |  | 79.23 | 79.23 | +0.00 | graphviz_dot 77.12 | WIN->WIN | 0.3->1.1 |
| densenet_block | legacy |  | 68.48 | 68.48 | +0.00 | graphviz_dot 67.91 | WIN->WIN | 0.3->2.4 |
| binary_tree | legacy |  | 86.40 | 86.40 | +0.00 | graphviz_dot 83.32 | WIN->WIN | 0.2->1.2 |
| unet_small | legacy |  | 76.84 | 76.84 | +0.00 | dagre 73.82 | WIN->WIN | 0.2->2.3 |
| nested_shallow_enc_dec | legacy |  | 89.11 | 89.11 | +0.00 | igraph_sugiyama 89.11 | TIE->TIE | 0.2->1.9 |
| transformer_layer | legacy |  | 75.03 | 75.03 | +0.00 | graphviz_dot 74.22 | WIN->WIN | 0.4->2.6 |
| mixed_width_labels | legacy |  | 84.14 | 84.14 | +0.00 | elk_layered 81.74 | WIN->WIN | 0.2->1.7 |
| random_dag_50 | legacy |  | 67.98 | 67.98 | +0.00 | igraph_sugiyama 61.31 | WIN->WIN | 0.6->3.3 |
| random_dag_200 | legacy |  | 67.48 | 67.48 | +0.00 | igraph_sugiyama 57.09 | WIN->WIN | 3.1->14.0 |
| bipartite_4_3_4 | legacy |  | 78.75 | 78.75 | +0.00 | igraph_sugiyama 67.57 | WIN->WIN | 0.3->1.1 |
| hierarchical_residual_stage | legacy |  | 84.96 | 84.96 | +0.00 | dagre 83.58 | WIN->WIN | 0.3->2.0 |
| recurrent_feedback_cell | legacy |  | 74.69 | 74.69 | +0.00 | igraph_sugiyama 73.24 | WIN->WIN | 0.2->3.2 |
| parallel_multiedge_bundle | legacy |  | 84.03 | 84.03 | +0.00 | elk_layered 84.74 | LOSS->LOSS | 0.0->0.0 |
| disconnected_encoder_residual | legacy |  | 77.20 | 77.20 | +0.00 | elk_layered 81.28 | LOSS->LOSS | 0.5->2.4 |
| moe_router_sparse | legacy |  | 83.93 | 83.93 | +0.00 | graphviz_dot 76.31 | WIN->WIN | 0.3->1.7 |
| ragged_feature_pyramid | legacy |  | 76.92 | 76.92 | +0.00 | graphviz_dot 74.41 | WIN->WIN | 0.3->2.6 |
| kitchen_sink_hybrid_net | legacy |  | 71.54 | 71.54 | +0.00 | dagre 67.60 | WIN->WIN | 0.4->2.4 |
| kitchen_sink_platform_graph | legacy |  | 82.15 | 82.15 | +0.00 | graphviz_dot 76.61 | WIN->WIN | 0.4->2.6 |
| extreme_mixed_width_transformer | legacy |  | 81.78 | 81.78 | +0.00 | graphviz_dot 74.19 | WIN->WIN | 0.2->1.8 |
| hub_fanout_label_skew | legacy |  | 88.35 | 88.35 | +0.00 | graphviz_dot 71.41 | WIN->WIN | 0.4->4.2 |
| clustered_longlabel_handoffs | legacy |  | 85.49 | 85.49 | +0.00 | dagre 79.03 | WIN->WIN | 0.3->3.1 |
| disconnected_label_cycle_collage | legacy |  | 77.86 | 77.86 | +0.00 | elk_layered 76.40 | WIN->WIN | 0.2->2.7 |
| shape_and_routing_matrix | legacy |  | 92.34 | 92.34 | +0.00 | dagre 85.60 | WIN->WIN | 0.2->4.0 |
| center_port_backedge_hub | legacy |  | 59.98 | 59.98 | +0.00 | dagre 56.35 | WIN->WIN | 0.2->3.3 |
| cluster_member_style_stress | legacy |  | 81.37 | 81.37 | +0.00 | dagre 76.26 | WIN->WIN | 0.2->2.6 |
| edge_label_braid | legacy |  | 81.98 | 81.98 | +0.00 | igraph_sugiyama 74.76 | WIN->WIN | 0.2->4.0 |
| nested_cluster_label_stack | legacy |  | 88.48 | 88.48 | +0.00 | graphviz_dot 87.26 | WIN->WIN | 0.2->3.4 |
| small_label_storm | legacy |  | 91.23 | 91.23 | +0.00 | dagre 84.37 | WIN->WIN | 0.2->2.8 |
| long_range_residual_ladder | legacy |  | 74.91 | 74.91 | +0.00 | graphviz_dot 69.31 | WIN->WIN | 0.6->4.3 |
| interleaved_cluster_crosstalk | legacy |  | 75.15 | 75.15 | +0.00 | graphviz_dot 65.33 | WIN->WIN | 0.3->5.7 |
| asymmetric_hourglass_hub | legacy |  | 79.17 | 79.17 | +0.00 | elk_layered 73.82 | WIN->WIN | 0.3->3.8 |
| multiscale_skip_cascade | legacy |  | 70.89 | 70.89 | +0.00 | dagre 66.05 | WIN->WIN | 0.4->4.4 |
| braided_feedback_tails | legacy |  | 79.89 | 79.89 | +0.00 | igraph_sugiyama 77.69 | WIN->WIN | 0.3->3.6 |
| width_skew_late_merge | legacy |  | 76.65 | 76.65 | +0.00 | graphviz_dot 75.35 | WIN->WIN | 0.4->4.2 |
| broken_symmetry_residual_pair | legacy |  | 74.78 | 74.78 | +0.00 | graphviz_dot 74.86 | TIE->TIE | 0.3->3.7 |
| hub_skip_superfan | legacy |  | 75.59 | 75.59 | +0.00 | elk_layered 72.70 | WIN->WIN | 0.3->3.4 |
| outerplanar_dag_20 | legacy |  | 69.33 | 69.33 | +0.00 | igraph_sugiyama 69.93 | LOSS->LOSS | 0.4->4.0 |
| protein_ppi_200 | legacy | Y | 52.05 | 52.05 | +0.00 | igraph_kamada_kawai 52.97 | LOSS->LOSS | 7.7->43.4 |
| citation_dag_300 | legacy |  | 59.30 | 59.30 | +0.00 | elk_layered 55.42 | WIN->WIN | 11.0->64.6 |
| random_bipartite_60 | legacy | Y | 65.24 | 65.24 | +0.00 | igraph_kamada_kawai 60.54 | WIN->WIN | 0.8->14.1 |
| heavy_tail_weights_50 | legacy | Y | 67.47 | 67.47 | +0.00 | igraph_kamada_kawai 62.23 | WIN->WIN | 0.7->48.3 |
| scale_free_ba_120 | legacy | Y | 34.86 | 34.86 | +0.00 | graphviz_sfdp 41.61 | LOSS->LOSS | 2.5->86.8 |
| complete_bipartite_8x12 | legacy | Y | 63.65 | 63.65 | +0.00 | graphviz_sfdp 57.91 | WIN->WIN | 1.5->12.8 |
| clustered_medium_5x20 | legacy |  | 65.31 | 65.31 | +0.00 | graphviz_dot 66.86 | LOSS->LOSS | 4.9->15.6 |
| hub_and_spoke_3x20 | legacy |  | 77.27 | 77.27 | +0.00 | elk_layered 69.89 | WIN->WIN | 2.9->13.6 |
| wide_single_layer_1_50_1 | legacy |  | 86.47 | 86.47 | +0.00 | elk_layered 69.30 | WIN->WIN | 1.4->5.9 |
| sparse_pair_50 | legacy |  | 80.66 | 80.66 | +0.00 | elk_layered 77.90 | WIN->WIN | 0.9->4.7 |
| dense_pair_50 | legacy |  | 70.22 | 70.22 | +0.00 | graphviz_dot 69.77 | TIE->TIE | 1.6->4.8 |
| compound_dag_5x30 | legacy |  | 71.38 | 71.38 | +0.00 | graphviz_dot 70.92 | TIE->TIE | 2.5->13.8 |
| long_skip_only_24 | legacy |  | 81.06 | 81.06 | +0.00 | graphviz_dot 77.59 | WIN->WIN | 0.5->3.4 |
| parallel_cycles_4x5 | legacy |  | 58.38 | 58.38 | +0.00 | graphviz_sfdp 60.41 | LOSS->LOSS | 0.3->2.5 |
| resnet_stack_4x16 | legacy |  | 73.66 | 73.66 | +0.00 | dagre 72.32 | WIN->WIN | 0.5->2.6 |
| transformer_full_4h_2l | legacy |  | 74.47 | 74.47 | +0.00 | dagre 71.30 | WIN->WIN | 0.7->3.4 |
| dependency_graph_100 | legacy |  | 57.11 | 57.11 | +0.00 | elk_layered 56.20 | WIN->WIN | 3.8->33.5 |
| org_chart_1_5_4_8 | legacy |  | 89.13 | 89.13 | +0.00 | graphviz_dot 71.55 | WIN->WIN | 0.4->0.8 |
| small_world_100 | legacy | Y | 91.75 | 91.75 | +0.00 | igraph_kamada_kawai 68.62 | WIN->WIN | 2.0->9.6 |
| real_lesmis_77 | legacy | Y | 50.62 | 50.62 | +0.00 | graphviz_dot 52.81 | LOSS->LOSS | 2.3->53.8 |
| real_football_115 | legacy | Y | 30.64 | 30.64 | +0.00 | graphviz_dot 37.77 | LOSS->LOSS | 21.7->45.8 |
| er_100 | legacy | Y | 56.02 | 56.02 | +0.00 | graphviz_sfdp 56.30 | TIE->TIE | 1.9->11.0 |
| er_500 | legacy | Y | 46.82 | 46.82 | +0.00 | graphviz_sfdp 54.48 | LOSS->LOSS | 20.3->95.5 |
| rgg_100 | legacy | Y | 47.38 | 47.38 | +0.00 | elk_layered 52.31 | LOSS->LOSS | 9.9->27.2 |
| rgg_500 | legacy | Y | 53.49 | 53.49 | +0.00 | elk_layered 52.38 | WIN->WIN | 294.0->450.8 |
| ba_500 | legacy | Y | 44.30 | 44.30 | +0.00 | graphviz_sfdp 41.81 | WIN->WIN | 248.4->158.0 |
| sbm_4x30 | legacy | Y | 46.15 | 46.15 | +0.00 | graphviz_dot 48.18 | LOSS->LOSS | 28.4->22.0 |
| sbm_5x50 | legacy | Y | 45.79 | 45.79 | +0.00 | graphviz_dot 44.49 | WIN->WIN | 98.1->89.3 |
| grid_20x20 | legacy | Y | 93.44 | 93.44 | +0.00 | graphviz_neato 94.48 | LOSS->LOSS | 72.2->42.4 |
| wide_1_100_1 | legacy |  | 86.72 | 86.72 | +0.00 | elk_layered 69.05 | WIN->WIN | 12.5->11.3 |
| wide_3_50_3 | legacy |  | 70.90 | 70.90 | +0.00 | elk_layered 58.93 | WIN->WIN | 16.4->12.3 |
| hub_spoke_5x50 | legacy |  | 75.10 | 75.10 | +0.00 | elk_layered 67.86 | WIN->WIN | 178.2->41.9 |
| hub_spoke_10x20 | legacy |  | 73.34 | 73.34 | +0.00 | graphviz_dot 65.60 | WIN->WIN | 143.3->29.3 |
| compound_10x20 | legacy |  | 71.46 | 71.46 | +0.00 | graphviz_dot 70.57 | WIN->WIN | 37.4->20.1 |
| small_world_500 | legacy | Y | 42.88 | 42.88 | +0.00 | graphviz_neato 58.66 | LOSS->LOSS | 124.8->75.9 |
| dependency_500 | legacy |  | 54.56 | 54.56 | +0.00 | elk_layered 55.66 | LOSS->LOSS | 527.9->168.7 |
| org_chart_deep | legacy |  | 83.89 | 83.89 | +0.00 | elk_layered 62.85 | WIN->WIN | 14.5->5.3 |
| powerlaw_500 | legacy |  | 63.78 | 63.78 | +0.00 | elk_layered 58.99 | WIN->WIN | 274.0->92.9 |
| r79_weighted_community_4x18 | extended | Y | 46.21 | 46.21 | +0.00 | graphviz_dot 51.88 | LOSS->LOSS | 13.5->59.5 |
| r79_weighted_mesh_10x12 | extended | Y | 87.84 | 87.84 | +0.00 | graphviz_neato 92.52 | LOSS->LOSS | 19.8->15.0 |
| r79_weighted_skew_dag_6x10 | extended |  | 83.44 | 83.44 | +0.00 | graphviz_dot 76.37 | WIN->WIN | 23.8->11.7 |
| r79_weighted_hub_spoke_4x18 | extended |  | 75.50 | 75.50 | +0.00 | elk_layered 61.82 | WIN->WIN | 21.4->15.5 |
| r79_weighted_small_world_120 | extended | Y | 32.65 | 32.65 | +0.00 | igraph_kamada_kawai 51.69 | LOSS->LOSS | 40.4->18.3 |
| r79_weighted_ladder_40 | extended | Y | 94.70 | 94.70 | +0.00 | igraph_sugiyama 88.22 | WIN->WIN | 8.1->26.5 |
| r79_weighted_bipartite_16x24 | extended | Y | 64.71 | 64.71 | +0.00 | elk_layered 51.33 | WIN->WIN | 13.6->15.9 |
| r79_nested_clusters_3x2x10 | extended |  | 70.92 | 70.92 | +0.00 | graphviz_dot 71.28 | TIE->TIE | 9.1->7.0 |
| r79_nested_clusters_2x3x12 | extended |  | 78.35 | 78.35 | +0.00 | graphviz_dot 71.82 | WIN->WIN | 14.0->5.1 |
| r79_nested_clusters_4x2x8 | extended |  | 71.41 | 71.41 | +0.00 | graphviz_dot 70.01 | WIN->WIN | 19.0->11.9 |
| r79_undirected_sbm_low_mix_4x25 | extended | Y | 52.78 | 52.78 | +0.00 | elk_layered 50.12 | WIN->WIN | 34.8->13.8 |
| r79_undirected_sbm_mid_mix_5x20 | extended | Y | 49.61 | 49.61 | +0.00 | elk_layered 37.83 | WIN->WIN | 23.0->40.6 |
| r79_undirected_sbm_high_mix_3x30 | extended | Y | 36.54 | 36.54 | +0.00 | elk_layered 40.38 | LOSS->LOSS | 27.7->27.3 |
| r79_directed_scc_90_3cores | extended |  | 57.18 | 57.18 | +0.00 | graphviz_dot 56.99 | TIE->TIE | 12.6->10.6 |
| r79_directed_scc_120_2cores | extended |  | 58.38 | 58.38 | +0.00 | graphviz_dot 59.88 | LOSS->LOSS | 16.5->13.6 |
| weighted_chain_20 | legacy |  | 88.02 | 88.02 | +0.00 | dagre 88.02 | TIE->TIE | 2.5->2.1 |
