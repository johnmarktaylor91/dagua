# Sprint 31a Result -- Gate Refinement

## Summary

Shipped the scoped gate refinement in
`dagua/layout/ops/pipelines/dagua_native_legacy.py`.

- Small DAGs below `_DUMMY_NODE_MIN_NODES` now bypass the floor only when they
  have resolved layer assignments, at least one long layer edge, and a
  resolvable layer width.
- The `max_layer_width <= 1` dummy-node short-circuit was removed.
- The `max_layer_width <= 1` Brandes-Koepf short-circuit was removed.
- No new constants, topology signatures, polish candidates, or score pickers
  were added.

## Fixed Deltas

Validation command: `python /tmp/sprint31a_validate.py`.

Scores use `dagua.metrics.full(..., node_sizes=[[40, 20]] * N)` with
`LayoutConfig(seed=42, device="cpu")`.

| graph | before | after | delta |
|---|---:|---:|---:|
| mixed_width_labels | 77.583644 | 77.583644 | +0.000000 |
| unet_small | 70.785209 | 70.785209 | +0.000000 |
| extreme_mixed_width_transformer | 74.459398 | 86.408259 | +11.948861 |
| hierarchical_residual_stage | 82.285323 | 82.285323 | +0.000000 |
| deep_chain_20 | 97.500000 | 97.500000 | +0.000000 |
| random_dag_200 | 74.770455 | 74.805200 | +0.034745 |
| ba_500 | 63.138364 | 63.138364 | +0.000000 |
| org_chart_deep | 92.440685 | 92.440685 | +0.000000 |
| hub_fanout_label_skew | 93.736517 | 93.736517 | +0.000000 |

Protected-win condition passed: all fixed deltas are within +/-0.5.

## Jitter Table

Sigma=0.5 jitter, 8 trials, seed base=42. Deltas are `after - before`.

| graph | mean | std | min | max |
|---|---:|---:|---:|---:|
| mixed_width_labels | +0.000001 | 0.000001 | -0.000000 | +0.000002 |
| unet_small | +0.000000 | 0.000000 | +0.000000 | +0.000000 |
| extreme_mixed_width_transformer | +11.734629 | 2.271648 | +9.780256 | +15.343316 |
| hierarchical_residual_stage | +0.000000 | 0.000001 | -0.000001 | +0.000003 |

Target condition passed literally: at least 3 targets had jitter mean > 0 and
jitter min > -1.0. The material fixed-seed lift is concentrated in
`extreme_mixed_width_transformer`; the two other passing targets are
effectively neutral within floating-point noise but not jitter-negative.

## H2H Impact Estimate

Using the sprint-31 context best-engine scores:

- `extreme_mixed_width_transformer` moves from a -3.53 loss
  (`74.46` vs `77.99`) to an estimated +8.42 win (`86.41` vs `77.99`).
- `mixed_width_labels`, `unet_small`, and `hierarchical_residual_stage` remain
  fixed-seed neutral under this 3-conditional gate-only change.
- Aggregate target gap improves by about +11.95 composite points, all from the
  small mixed-width transformer topology.

The result is principled and protected-win safe, but it should not be read as a
complete geometric fix for width-1 residual chains. Those remain visible future
work because dummy insertion now gates on, but the current downstream machinery
does not materially move real nodes on the pure width-1 fixed-seed cases.
