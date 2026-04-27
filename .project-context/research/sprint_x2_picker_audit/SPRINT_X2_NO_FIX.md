# Sprint-X2 No-Fix Report

Date: 2026-04-26
Branch: `codex/sprint-31a-gate-refinement`
Baseline: git HEAD `3eaa01c`

## Outcome

No fix shipped. The picker audit found several zero-win candidates, but removing
the safely-dead set did not satisfy the Sprint-X2 runtime gate.

Trial removal set:

- `_POLISH_SETTINGS`: `(30, 0.02)` plus its generated chained candidates.
- Direct candidate entries: `overlap_jitter`, `back_edge_relayer_quarter`.
- Candidate functions and entries: `_per_layer_x_kmeans`,
  `_gap_validated_layer_swaps`, `_multi_component_row_major_repack`.
- Additional zero-win chained slots were also tried while keeping their
  edge-equalize seeds.

Result:

- 93-graph CPU slice (`get_test_graphs()`, `num_nodes <= 500`): `451.742s ->
  407.788s`, `1.108x`.
- 8-graph picker probe from Sprint-41 context: `53.424s -> 49.540s`, `1.078x`
  on the first trim; a second chained-slot trim rerun measured `1.017x`.
- Composite was unchanged on almost all graphs in the first full rerun, but the
  run exposed benchmark reproducibility noise for `random_dag_200` because
  graph construction depends on Python set iteration order unless
  `PYTHONHASHSEED` is fixed.

The pass condition requires `>=1.2x` aggregate speedup on the 8-graph picker
probe. The empirical trim did not reach that threshold, so committing it would
ship code churn without satisfying Sprint-X2.

## Audit Method

Temporary env-gated instrumentation was added to `_best_of_polish` and then
removed before this no-fix report. With `DAGUA_PICKER_AUDIT=1`, it appended one
row per picker invocation to `/tmp/picker_winners_full.csv`:

`graph_id,winner_name,winner_score,margin_over_second,margin_over_baseline`

Behavior check on a fixed tensor input:

- Audit disabled vs enabled: `torch.equal(out1, out2) == True`
- Maximum absolute position difference: `0.0`

The full audit run covered 93 graphs with `num_nodes <= 500`. There were 93
picker invocations across 92 graphs; `parallel_multiedge_bundle` has only 3
nodes and does not enter the polish gate, while `disconnected_encoder_residual`
entered the picker twice.

## Winner Distribution

| winner | wins |
|---|---:|
| `baseline` | 17 |
| `orthogonal_align` | 15 |
| `y_layer_snap` | 8 |
| `edge_equalize_50_0.2` | 7 |
| `edge_equalize_50_0.05` | 4 |
| `orthogonal_align_after_edge_equalize_5_0.05` | 4 |
| `edge_equalize_10_0.1` | 3 |
| `orthogonal_align_overlap_jitter_after_edge_equalize_5_0.05` | 3 |
| `orthogonal_align_overlap_jitter_after_edge_equalize_50_0.2` | 3 |
| `edge_equalize_20_0.03` | 3 |
| `orthogonal_align_after_edge_equalize_10_0.1` | 3 |
| `swap_2opt_anti_crossing` | 3 |
| `orthogonal_align_after_edge_equalize_50_0.2` | 2 |
| `orthogonal_align_after_edge_equalize_10_0.05` | 2 |
| `lattice_uniform_centered_slots` | 2 |
| `dot_lattice_lp` | 2 |
| `edge_equalize_5_0.05` | 2 |
| `global_depth_align` | 1 |
| `y_layer_snap_after_edge_equalize_5_0.05` | 1 |
| `back_edge_relayer_half` | 1 |
| `orthogonal_align_overlap_jitter_after_edge_equalize_50_0.05` | 1 |
| `outerplanar_source_fan_spine` | 1 |
| `tutte_cyclic_planar` | 1 |
| `orthogonal_align_after_edge_equalize_20_0.03` | 1 |
| `y_layer_snap_after_edge_equalize_10_0.1` | 1 |
| `median_transpose_polish` | 1 |
| `back_edge_relayer_full` | 1 |

Zero-win static candidates observed:

- `edge_equalize_10_0.05` as a direct winner. Kept because its chained
  `orthogonal_align_after_edge_equalize_10_0.05` won on two graphs.
- `edge_equalize_30_0.02`.
- `overlap_jitter` as a direct winner. Kept as a function because chained
  overlap-jitter candidates won on seven graphs.
- `per_layer_x_kmeans`.
- `back_edge_relayer_quarter`.
- `gap_validated_layer_swaps`.
- `multi_component_row_major_repack`.

## Why No Safe Drop Shipped

The zero-win candidates are not the dominant remaining cost. The Sprint-41
profile already pointed at `_dot_lattice_lp` and repeated metric scoring as the
hot work on expensive graphs such as `real_football_115`. `dot_lattice_lp` won
on `dense_pair_50` and `planar_60`, so it is not dead and cannot be dropped
under the Sprint-X2 rules.

The audit therefore found removable clutter, but not enough removable clutter
to meet the required speedup. A principled follow-up should target candidate
gating for expensive candidates that rarely win, especially `dot_lattice_lp`,
without modifying the kept candidate functions blindly.

## Validation Artifacts

- Audit CSV: `/tmp/picker_winners_full.csv`
- Baseline run log: `/tmp/sprint_x2_audit_run.log`
- Baseline scores/runtimes: `/tmp/sprint_x2_baseline.json`
- First trim rerun log: `/tmp/sprint_x2_post_run.log`
- First trim scores/runtimes: `/tmp/sprint_x2_post.json`
- Second trim 8-graph probe: `/tmp/sprint_x2_probe_post2.json`

No commit was made.
