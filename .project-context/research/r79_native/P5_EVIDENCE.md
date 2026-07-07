# R79 P5 Cluster Evidence

## Scope

Partial shipment.

- Default `dagua_native` no longer emits the native cluster fallback warning.
- `native_stress` is wired through `ClusterAwareDriver` for recursive cluster placement.
- The default `dagua_native` path remains on the current flat native placement for clustered DAG/layered workloads. Honest recursive attempts regressed the quality gate badly, so they are not enabled for the default path.
- `ClusterTree.from_flat_membership()` now expands direct-member parent clusters with child descendants. This fixes containment checking for corpus graphs that store parent clusters as direct members instead of already-flat descendants.

## Warning And Containment

- Fallback warning repro before: `nested_shallow_enc_dec` emitted `cluster_aware=True is not yet supported for algorithm='dagua_native'; falling back to legacy flat placement`.
- After: all 27 clustered corpus graphs `<=500` nodes emitted 0 fallback warnings.
- Containment: 0 violations across all 27 clustered corpus graphs `<=500` nodes, including node-in-cluster and child-box-in-parent checks with expanded cluster descendants.

## Sweep

Branch point run: `scripts/r79_baseline.py --dagua-only`

- Git SHA: `e0ea4905a66138f8b06d459a74bfff237940a9c7`
- Wall time: `4474.980s`
- W/T/L: legacy `56/8/29`, extended `8/2/5`

Final run: `scripts/r79_baseline.py --dagua-only`

- Git SHA: `e0ea4905a66138f8b06d459a74bfff237940a9c7`
- Wall time: `702.267s`
- W/T/L: legacy `56/8/29`, extended `8/2/5`

## Cluster Metrics

| Graph | Composite Before | Composite After | Delta | Mean Sep Before | Mean Sep After | Min Sep Before | Min Sep After |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cluster_member_style_stress` | 81.369 | 81.369 | 0.000 | 0.945 | 0.945 | 0.429 | 0.429 |
| `clustered_longlabel_handoffs` | 85.489 | 85.489 | 0.000 | 3.072 | 3.072 | 0.111 | 0.111 |
| `clustered_medium_5x20` | 65.307 | 65.307 | 0.000 | 3.240 | 3.240 | 0.761 | 0.761 |
| `compound_10x20` | 71.465 | 71.465 | 0.000 | 12.369 | 12.369 | 3.366 | 3.366 |
| `compound_dag_5x30` | 71.381 | 71.381 | 0.000 | 6.815 | 6.815 | 3.405 | 3.405 |
| `dependency_500` | 54.556 | 54.556 | 0.000 | 4.434 | 4.434 | 4.434 | 4.434 |
| `dependency_graph_100` | 57.114 | 57.114 | 0.000 | 1.657 | 1.657 | 1.657 | 1.657 |
| `hierarchical_residual_stage` | 84.960 | 84.960 | 0.000 | 271.569 | 271.569 | 0.173 | 0.173 |
| `interleaved_cluster_crosstalk` | 75.148 | 75.148 | 0.000 | 1.460 | 1.460 | 0.197 | 0.197 |
| `kitchen_sink_hybrid_net` | 71.540 | 71.540 | 0.000 | 16.453 | 16.453 | 0.264 | 0.264 |
| `kitchen_sink_platform_graph` | 82.146 | 82.146 | 0.000 | 3.479 | 3.479 | 0.180 | 0.180 |
| `moe_router_sparse` | 83.932 | 83.932 | 0.000 | 0.294 | 0.294 | 0.294 | 0.294 |
| `multiscale_skip_cascade` | 70.895 | 70.895 | 0.000 | 1.355 | 1.355 | 0.159 | 0.159 |
| `nested_cluster_label_stack` | 88.481 | 88.481 | 0.000 | 3.187 | 3.187 | 0.431 | 0.431 |
| `nested_shallow_enc_dec` | 89.108 | 89.108 | 0.000 | 1.257 | 1.257 | 0.471 | 0.471 |
| `r79_directed_scc_120_2cores` | 58.381 | 58.381 | 0.000 | 2.727 | 2.727 | 1.235 | 1.235 |
| `r79_directed_scc_90_3cores` | 57.178 | 57.178 | 0.000 | 4.959 | 4.959 | 1.472 | 1.472 |
| `r79_nested_clusters_2x3x12` | 78.347 | 78.347 | 0.000 | 6.248 | 6.248 | 0.894 | 0.894 |
| `r79_nested_clusters_3x2x10` | 70.921 | 70.921 | 0.000 | 3.814 | 3.814 | 0.702 | 0.702 |
| `r79_nested_clusters_4x2x8` | 71.408 | 71.408 | 0.000 | 4.560 | 4.560 | 0.126 | 0.126 |
| `r79_undirected_sbm_high_mix_3x30` | 36.543 | 36.543 | 0.000 | 0.237 | 0.237 | 0.148 | 0.148 |
| `r79_undirected_sbm_low_mix_4x25` | 52.780 | 52.780 | 0.000 | 1.669 | 1.669 | 0.252 | 0.252 |
| `r79_undirected_sbm_mid_mix_5x20` | 49.609 | 49.609 | 0.000 | 0.429 | 0.429 | 0.213 | 0.213 |
| `resnet_stack_4x16` | 73.659 | 73.659 | 0.000 | 2.613 | 2.613 | 0.136 | 0.136 |
| `small_label_storm` | 91.228 | 91.228 | 0.000 | 1.804 | 1.804 | 0.194 | 0.194 |
| `transformer_full_4h_2l` | 74.467 | 74.467 | 0.000 | 1.958 | 1.958 | 0.233 | 0.233 |
| `transformer_layer` | 75.027 | 75.027 | 0.000 | 2.471 | 2.471 | 0.660 | 0.660 |

## Determinism And Jitter

- Determinism: `clustered_medium_5x20` and `r79_nested_clusters_3x2x10` were two-run bit-identical with max absolute delta `0.0`.
- Jitter: no quality improvement is claimed; the shipped default keeps composite and cluster-separation metrics unchanged.

## Renders

Generated 2-panel before/after PNGs, each `1800x812`:

- `.project-context/research/r79_native/gallery_p5/clustered_medium_5x20_before_after.png`
- `.project-context/research/r79_native/gallery_p5/r79_nested_clusters_3x2x10_before_after.png`
- `.project-context/research/r79_native/gallery_p5/r79_undirected_sbm_high_mix_3x30_before_after.png`
- `.project-context/research/r79_native/gallery_p5/r79_directed_scc_120_2cores_before_after.png`

## Failed Recursive Attempts

- Full recursive `dagua_native` driver with per-level native router reproduced no warning and passed containment, but regressed the sweep to legacy `49/9/35`, extended `6/1/8`; several clustered DAGs dropped more than 25 composite points.
- Stress-only recursive placement for all default clustered graphs was worse on the clustered subset, with typical drops from 20 to 55 composite points.
- Conclusion: recursive cluster support is safe to expose for `native_stress`, but layered/DAG recursive native placement remains a deeper algorithm issue.

## Warning honesty fix

- Restored a flat-placement warning when clustered `algorithm="dagua_native"` layouts do not enter the recursive cluster driver.
- Kept the warning suppressed for explicit `algorithm="native_stress"` clustered layouts because that path is handled by `ClusterAwareDriver`.
- The change only gates warning emission after recursive cluster dispatch returns `None`; layout geometry is unchanged.
