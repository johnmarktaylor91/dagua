# Graph Catalog

Node and edge counts below reflect the default generator parameters in
`dagua/eval/graphs.py`. Use this catalog to pick graphs by structure, failure
mode, and benchmark purpose.

## Baselines And Trees

| Name | Nodes | Edges | Structure type | Tags | What it tests |
| --- | ---: | ---: | --- | --- | --- |
| `linear_3layer_mlp` | 6 | 5 | Shallow chain | `linear-shallow` | Baseline layered layout sanity |
| `deep_chain_20` | 22 | 21 | Deep chain | `linear-deep` | Tall layouts and edge-length consistency |
| `binary_tree` | 11 | 10 | Balanced tree | `tree` | Symmetry and breadth spreading |
| `org_chart_1_5_4_8` | 18 | 17 | Non-uniform tree | `tree` | Uneven level widths and parent-child spacing |

## DAG Width, Density, And Skip Pressure

| Name | Nodes | Edges | Structure type | Tags | What it tests |
| --- | ---: | ---: | --- | --- | --- |
| `inception_block` | 7 | 9 | Parallel fan-out/fan-in | `wide-parallel` | Branch alignment in wide shallow DAGs |
| `residual_block` | 10 | 10 | Residual diamond | `diamond`, `skip-light` | Merge alignment with a simple skip |
| `densenet_block` | 8 | 22 | Dense skip DAG | `large-dense`, `skip-heavy` | Crossing pressure from heavy skip reuse |
| `unet_small` | 9 | 11 | Encoder-decoder diamond | `diamond`, `skip-light` | Symmetry with lateral skips |
| `random_dag_50` | 97 | 70 | Sparse random DAG | `large-sparse` | Medium-scale general readability |
| `random_dag_200` | 383 | 300 | Sparse random DAG | `large-sparse` | Scaling pressure without hierarchy |
| `grid_5x5` | 25 | 40 | Square grid DAG | `diamond`, `large-dense` | Regularity and local crossing avoidance |
| `bipartite_4_3_4` | 11 | 24 | Two-stage bipartite DAG | `large-dense`, `wide-parallel` | Crossing minimization across layers |
| `ragged_feature_pyramid` | 12 | 15 | Ragged multiscale DAG | `diamond`, `skip-heavy`, `wide-parallel` | Long lateral skips and uneven branch widths |
| `long_range_residual_ladder` | 38 | 41 | Deep ladder with skips | `linear-deep`, `skip-heavy`, `wide-parallel` | Long residual routing without losing backbone flow |
| `asymmetric_hourglass_hub` | 14 | 15 | Asymmetric hourglass | `diamond`, `skip-light`, `wide-parallel` | Balance around a dominant hub |
| `multiscale_skip_cascade` | 15 | 23 | Cross-scale cascade | `nested-shallow`, `skip-heavy`, `wide-parallel` | Skip routing across repeated resolutions |
| `braided_feedback_tails` | 12 | 17 | Braided near-layered DAG | `diamond`, `linear-deep`, `skip-heavy` | Crossing-heavy braids plus late feedback |
| `width_skew_late_merge` | 17 | 23 | Wide-to-narrow funnel | `diamond`, `skip-heavy`, `wide-parallel` | Extreme width collapse into a late merge |
| `broken_symmetry_residual_pair` | 12 | 16 | Near-symmetric residual pair | `diamond`, `skip-heavy`, `wide-parallel` | Pattern preservation with a deliberate asymmetry |
| `hub_skip_superfan` | 13 | 19 | Deep spine plus hub | `linear-deep`, `skip-heavy`, `wide-parallel` | Central skip knots around a dominant hub |
| `scale_free_ba_120` | 120 | 354 | Preferential-attachment DAG | `large-sparse`, `scale-free` | Hub dominance and bundled long fan-out |
| `grid_rect_6x8` | 48 | 82 | Rectangular grid DAG | `diamond`, `grid` | Regular spacing in non-square grids |
| `complete_bipartite_8x12` | 20 | 96 | Complete bipartite DAG | `bipartite`, `wide-parallel` | Dense two-layer ordering stress |
| `hub_and_spoke_3x20` | 65 | 125 | Hub-and-spoke DAG | `hub-spoke`, `wide-parallel` | Very high-degree fan-out/fan-in nodes |
| `wide_single_layer_1_50_1` | 52 | 100 | Three-layer wide DAG | `wide-layer`, `wide-parallel` | Extremely wide middle-layer spacing |
| `sparse_pair_50` | 50 | 61 | Sparse matched DAG | `large-sparse` | Density-controlled baseline for paired comparison |
| `dense_pair_50` | 50 | 208 | Dense matched DAG | `large-dense` | Same nodes as sparse pair with much higher crossing pressure |
| `long_skip_only_24` | 24 | 41 | Skip-only DAG | `linear-deep`, `skip-heavy` | Pathological long-edge routing with no local edges |

## Clustered And Compound Hierarchies

| Name | Nodes | Edges | Structure type | Tags | What it tests |
| --- | ---: | ---: | --- | --- | --- |
| `nested_shallow_enc_dec` | 6 | 5 | Two-cluster DAG | `nested-shallow` | Basic cluster boxing and labeling |
| `transformer_layer` | 16 | 19 | Clustered transformer layer | `nested-deep`, `skip-light`, `wide-parallel` | Parallel heads plus residual hierarchy |
| `hierarchical_residual_stage` | 10 | 11 | Multi-level residual hierarchy | `nested-deep`, `skip-light` | Cross-boundary skips inside deep clusters |
| `moe_router_sparse` | 9 | 11 | Clustered expert routing | `large-dense`, `nested-shallow`, `wide-parallel` | Sparse router fan-out into clustered experts |
| `clustered_medium_5x20` | 100 | 193 | Medium clustered DAG | `clustered`, `nested-shallow` | Medium-scale cluster separation and bridge routing |
| `compound_dag_5x30` | 150 | 210 | Compound DAG | `clustered`, `compound`, `nested-shallow` | Stage-level DAG flow with explicit clusters |
| `interleaved_cluster_crosstalk` | 12 | 17 | Sibling-cluster cross-talk | `nested-deep`, `skip-heavy`, `wide-parallel` | Keeping sibling identity legible under cross-links |
| `dependency_graph_100` | 100 | 285 | Dependency DAG with core cluster | `clustered`, `dependency`, `scale-free` | Core-library dominance and package fan-in |

## Cycles, Components, And Feedback

| Name | Nodes | Edges | Structure type | Tags | What it tests |
| --- | ---: | ---: | --- | --- | --- |
| `recurrent_feedback_cell` | 5 | 6 | Small recurrent cell | `self-loops`, `skip-light` | Cycle breaking and self-loop routing |
| `disconnected_encoder_residual` | 9 | 8 | Two disconnected DAGs | `disconnected`, `skip-light` | Component packing stability |
| `disconnected_label_cycle_collage` | 7 | 6 | Disconnected mixed collage | `disconnected`, `mixed-width`, `self-loops` | Packing components with very different local scales |
| `center_port_backedge_hub` | 6 | 9 | Cyclic hub graph | `self-loops`, `skip-heavy`, `wide-parallel` | Back-edge pressure around a dominant hub |
| `parallel_cycles_4x5` | 20 | 20 | Parallel independent cycles | `cyclic`, `disconnected` | Cycle handling plus disconnected placement |
| `small_world_100` | 100 | 200 | Directed small-world graph | `cyclic`, `small-world` | Short paths and non-layered global geometry |
| `kitchen_sink_platform_graph` | 18 | 21 | Platform system graph | `disconnected`, `mixed-width`, `nested-deep`, `self-loops`, `skip-heavy`, `wide-parallel` | Combined subsystem, loop, and hierarchy stress |

## Label, Style, And Routing Stress

| Name | Nodes | Edges | Structure type | Tags | What it tests |
| --- | ---: | ---: | --- | --- | --- |
| `mixed_width_labels` | 6 | 6 | Mixed-label residual DAG | `mixed-width`, `skip-light` | Node sizing under varied label widths |
| `parallel_multiedge_bundle` | 3 | 6 | Multi-edge bundle | `diamond`, `multi-edge` | Duplicate-edge separation |
| `kitchen_sink_hybrid_net` | 19 | 25 | Hybrid stress collage | `mixed-width`, `multi-edge`, `nested-deep`, `self-loops`, `skip-heavy`, `wide-parallel` | Combined loops, clusters, width skew, and duplicate edges |
| `extreme_mixed_width_transformer` | 10 | 12 | Extreme label-width transformer | `mixed-width`, `skip-light`, `wide-parallel` | Severe short-vs-long label imbalance |
| `hub_fanout_label_skew` | 10 | 13 | Label-skewed fan-out | `diamond`, `mixed-width`, `wide-parallel` | Balance under asymmetric branch widths |
| `clustered_longlabel_handoffs` | 10 | 12 | Long-label clustered handoff | `mixed-width`, `multi-edge`, `nested-deep`, `skip-light` | Cluster sizing with repeated long-label handoffs |
| `shape_and_routing_matrix` | 6 | 6 | Shape/routing matrix | `diamond`, `mixed-width`, `wide-parallel` | Shape-aware routing across edge modes |
| `cluster_member_style_stress` | 8 | 8 | Cluster style override DAG | `diamond`, `nested-shallow`, `skip-light` | Cluster-scoped node and edge style cascades |
| `edge_label_braid` | 8 | 10 | Edge-label braid | `diamond`, `mixed-width`, `wide-parallel` | Edge-label collision avoidance near crossings |
| `nested_cluster_label_stack` | 8 | 9 | Nested cluster-label stack | `mixed-width`, `nested-deep`, `skip-light` | Cluster-title stacking near labeled handoffs |
| `small_label_storm` | 6 | 6 | Compact label storm | `mixed-width`, `nested-shallow`, `wide-parallel` | Edge-label and cluster-label crowding in a small footprint |

## Realistic Neural-Net Motifs

| Name | Nodes | Edges | Structure type | Tags | What it tests |
| --- | ---: | ---: | --- | --- | --- |
| `resnet_stack_4x16` | 30 | 33 | Residual stack | `nested-shallow`, `neural-net`, `skip-light` | Repeated residual blocks with cluster grouping |
| `transformer_full_4h_2l` | 26 | 35 | Transformer stack | `nested-deep`, `neural-net`, `wide-parallel` | Full attention/residual hierarchy over multiple layers |

## Optional TorchLens Traces

These appear only when TorchLens and its example dependencies are available.

| Name | Nodes | Edges | Structure type | Tags | What it tests |
| --- | ---: | ---: | --- | --- | --- |
| `tl_mlp_3layer` | 7 | 6 | Traced MLP | `linear-shallow` | Real traced operator graph for a simple feed-forward model |
| `tl_cnn_small` | 10 | 9 | Traced CNN | `linear-shallow` | Real traced convolution/pooling pipeline |
| `tl_resnet_2block` | 20 | 21 | Traced residual CNN | `nested-shallow`, `skip-light` | Real residual operator graph with repeated blocks |
| `tl_transformer_1layer` | 38 | 41 | Traced transformer encoder | `nested-deep`, `skip-light`, `wide-parallel` | Real attention operator graph with residual structure |

## Benchmark Fit

- Baseline sanity checks: `linear_3layer_mlp`, `deep_chain_20`, `binary_tree`, `org_chart_1_5_4_8`
- Crossing and ordering stress: `complete_bipartite_8x12`, `dense_pair_50`, `grid_rect_6x8`, `long_skip_only_24`, `scale_free_ba_120`
- Cluster and hierarchy evaluation: `clustered_medium_5x20`, `compound_dag_5x30`, `interleaved_cluster_crosstalk`, `transformer_full_4h_2l`
- Cycle and component robustness: `parallel_cycles_4x5`, `small_world_100`, `recurrent_feedback_cell`, `kitchen_sink_platform_graph`
- Label and rendering stress: `edge_label_braid`, `small_label_storm`, `clustered_longlabel_handoffs`, `shape_and_routing_matrix`
- Realistic model comparisons: `resnet_stack_4x16`, `transformer_full_4h_2l`, `dependency_graph_100`, TorchLens traces
