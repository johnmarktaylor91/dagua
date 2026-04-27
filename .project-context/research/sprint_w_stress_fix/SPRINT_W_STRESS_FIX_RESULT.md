# Sprint W Stress Fix Result

Date: 2026-04-27
Branch: `codex/sprint-31a-gate-refinement`

## Diagnosis

`PivotApproxStressLoss` crashed when the native layered-DAG pipeline activated
dummy-node expansion. The stress precompute ops (`PivotSelection` and
`PivotDistanceQueries`) run on the original graph, so `state.pivot_indices` and
`state.pivot_distances` are original-node sized. During optimization, however,
the active `state.pos` may include dummy routing nodes appended after the
original node block.

Smallest repro: `asymmetric_hourglass_hub`.

Diagnostic script: `/tmp/sprint_w_stress_fix_diagnose.py`.

Observed at the crash site:

| input | shape/value |
|---|---:|
| `problem.num_nodes` | 14 |
| `state.pos` | `[21, 2]` |
| `state.pivot_indices` | `[14]` |
| `state.pivot_distances` | `[14, 14]` |
| `expanded_graph.num_nodes` | 21 |

The crash happened in `_maxent_stress_term`: `torch.cdist(positions,
pivot_positions)` produced `[21, 14]`, but the target distance matrix was
`[14, 14]`.

## Fix

Stress is only defined for original graph nodes. Dummy nodes are routing
artifacts and do not have graph-distance semantics. `PivotApproxStressLoss`
now slices the active position tensor to `pos[:problem.num_nodes]` before
calling the maxent stress implementation whenever dummy nodes are present.
Gradients flow through original node positions only.

Regression test added:
`tests/test_ops_loss_classic.py::test_pivot_approx_stress_loss_ignores_dummy_node_tail`.

## Empirical Results

Probe: `/tmp/sprint_w_stress_probe.py`

Output:

- CSV: `/tmp/w_stress_probe.csv`
- Log: `/tmp/w_stress_probe_after.log`
- Pre-fix CSV preserved as `/tmp/w_stress_probe_before_fix.csv`

All 15 graphs now succeed for `w_stress in {0.0, 0.05, 0.1, 0.2}`.

| graph | comp@0 | stress@0 | d_comp@0.05 | d_stress@0.05 | d_comp@0.1 | d_stress@0.1 | d_comp@0.2 | d_stress@0.2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `asymmetric_hourglass_hub` | 84.8361 | 0.653931 | -0.0019 | +0.000001 | -0.0019 | +0.000001 | -0.0022 | +0.000000 |
| `deep_chain_20` | 97.5000 | 0.907029 | +0.0000 | +0.000000 | +0.0000 | +0.000000 | +0.0000 | +0.000000 |
| `small_world_100` | 58.9997 | 0.896408 | +0.0000 | +0.000000 | +0.0000 | +0.000000 | +0.0000 | +0.000000 |
| `rgg_100` | 72.7725 | 0.644869 | -0.0015 | +0.000070 | -0.0015 | +0.000070 | -0.0015 | +0.000070 |
| `real_lesmis_77` | 71.9202 | 0.665653 | -0.1544 | +0.042872 | -0.1544 | +0.042872 | -0.1544 | +0.042872 |
| `scale_free_ba_120` | 64.1313 | 0.680538 | +0.0272 | +0.000009 | +0.0272 | +0.000009 | +0.0272 | +0.000009 |
| `compound_dag_5x30` | 76.1037 | 0.936235 | -0.0002 | +0.000001 | -0.0002 | +0.000001 | -0.0002 | +0.000001 |
| `dependency_graph_100` | 59.7055 | 0.667521 | -0.1337 | +0.041037 | -0.1337 | +0.041037 | -0.1337 | +0.041037 |
| `real_football_115` | 60.8907 | 0.635815 | +0.0018 | +0.000002 | +0.0018 | +0.000002 | +0.0018 | +0.000002 |
| `ba_500` | 63.1384 | 0.748140 | -0.0876 | +0.000682 | -0.0879 | +0.000636 | -0.0876 | +0.000682 |
| `er_500` | 70.2977 | 0.837471 | +0.6012 | -0.001408 | +0.0977 | -0.002389 | +0.4226 | +0.000156 |
| `dense_pair_50` | 72.7130 | 0.686862 | +0.0000 | +0.000000 | +0.0000 | +0.000000 | +0.0000 | +0.000000 |
| `random_dag_200` | 74.5799 | 0.866847 | +0.2809 | -0.000458 | +0.2178 | +0.000183 | +0.2186 | +0.000140 |
| `unet_small` | 80.4757 | 0.480432 | +0.0000 | +0.000000 | +0.0000 | +0.000000 | +0.0000 | +0.000000 |
| `mixed_width_labels` | 87.1363 | 0.353655 | +0.0011 | -0.000072 | +0.0011 | -0.000072 | +0.0011 | -0.000072 |

Aggregate:

| w_stress | graphs with lower sampled_stress | mean composite delta | worst composite delta | mean sampled_stress delta |
|---:|---:|---:|---:|---:|
| 0.05 | 3/15 | +0.0355 | -0.1544 | +0.005516 |
| 0.10 | 2/15 | -0.0023 | -0.1544 | +0.005490 |
| 0.20 | 1/15 | +0.0194 | -0.1544 | +0.005660 |

## Decision

Option A: bug fix only. Leave default `w_stress = 0.0`.

Reason: the crash is fixed, but enabling explicit stress does not satisfy the
default-on pass condition. At `w_stress=0.05`, only 3 of 15 graphs improve
sampled stress, far below the required 8 of 15. Mean sampled stress gets worse
by `+0.005516` even though composite stays within the allowed aggregate band.

## Honest Answer

Enabling `w_stress > 0` does not meaningfully lift `sampled_stress` on this
probe. It mostly leaves layouts unchanged, slightly improves a few graphs, and
materially worsens stress on `real_lesmis_77` and `dependency_graph_100`.
Composite cost is small, but the target metric does not improve enough to
justify default-on behavior.
