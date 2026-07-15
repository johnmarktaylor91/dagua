# CX1 Findings: kNN / Neighborhood-Preservation (`np`) Metric

## Verdict

`np` is **BASIN-INVARIANT**, not basin-sensitive. It does **not** compare Dagua's spatial neighbors to the reference layout's spatial neighbors. It computes each layout's preservation of the graph's shortest-path neighborhoods, then compares the two scalar preservation scores.

Evidence:

- `stress_quality_equivalence()` builds one graph shortest-path matrix (`graph_dist`) from `edge_index`, then computes `neighborhood_preservation(dagua, graph_dist, k=k)` and independently `neighborhood_preservation(reference, graph_dist, k=k)`, storing only `abs(neighborhood_dagua - neighborhood_reference)` as the delta: `dagua/eval/equivalence_metrics.py:456-477`.
- `neighborhood_preservation()` explicitly says it compares layout nearest neighbors against graph-distance neighbors: `dagua/eval/equivalence_metrics.py:481-501`.
- Inside the loop, `graph_neighbors` comes from `all_pairs_distances[node]`, while `layout_neighbors` comes from pairwise distances within that same layout, and the score is their overlap fraction: `dagua/eval/equivalence_metrics.py:508-525`.
- The distributional analysis feeds the same way: `quality_metric_samples()` computes `np_d = [neighborhood_preservation(layout, dists, k=10) for layout in d_layouts]` and separately `np_r = [...] for layout in r_layouts`: `scripts/definitive_fidelity_analysis.py:1275-1305`.
- Mode A compares paired scalar score differences `metrics["np_d"] - metrics["np_r"]` against the fixed absolute margin, not spatial neighborhoods between layouts: `scripts/definitive_fidelity_analysis.py:1163-1179`.
- Mode B compares Dagua scalar samples to one reference scalar target: `scripts/definitive_fidelity_analysis.py:1206-1224`.

## Gate Mechanics

The strict 3Q battery requires all three metric-specific tests to pass. The record sets `quality_identical_raw = stress_ok and cross_ok and np_ok`: `scripts/definitive_fidelity_analysis.py:1344-1355`. `metric_equivalent()` is a strict raw-alpha gate: degenerate-SD direct pass, otherwise finite `p_tost < 0.05`: `scripts/definitive_fidelity_analysis.py:1408-1424`. The current fixed `np` margin is `QUALITY_NP_ABS_MARGIN = 0.02`: `scripts/definitive_fidelity_analysis.py:48-53`, and the report documents k-NN preservation at `0.02` absolute: `scripts/definitive_fidelity_report.py:2445-2449`.

## Does `np` Bind the 574 Rung-4 Rows?

No. Stress/cross bind far more often.

From `eval_output/fidelity_definitive_r73/per_combo.json`:

- Total rows: 3,955.
- Rung-4 rows: 574.
- Rung-4 rows with strict quality battery fields: 570. Four additional rung-4 rows only have the older stress-only quality fields, so they are excluded from per-metric battery counts.

Per-metric failures among those 570 rung-4 battery rows:

| Failed metric | Count |
|---|---:|
| stress | 548 |
| crossings | 385 |
| `np` | 295 |

Failure patterns:

| Failed metrics | Count |
|---|---:|
| stress + crossings + `np` | 204 |
| stress + crossings | 164 |
| stress only | 96 |
| stress + `np` | 84 |
| crossings only | 15 |
| `np` only | 5 |
| crossings + `np` | 2 |

So `np` is the sole binding constraint for only **5 / 570 = 0.9%** of battery-scored rung-4 rows, versus stress-only for 96 and crossings-only for 15. The battery is mainly stress-bound, with crossings also materially binding.

Across all 3,692 scored rows with battery fields, the same pattern holds: stress fails 1,828, crossings 1,075, `np` 400; sole failures are stress 783, crossings 137, `np` 11.

## Is Dagua Worse or Better on `np`?

Overall, Dagua is more often better on `np`, especially where `np` fails:

- Rung-4 rows with finite `np` means: Dagua better in 296, worse in 171, tied in 103; mean `np_D - np_R = +0.0243`.
- `np`-failed rung-4 rows: Dagua better in 204, worse in 91; mean `np_D - np_R = +0.0461`.

For the five `np`-only binders, Dagua is worse in 4 and better in 1:

| Combo | `np_D` | `np_R` | D-R | Stored `np_p_tost` |
|---|---:|---:|---:|---:|
| `disconnected_encoder_residual::classic_fmmm_steps10` | 0.8748 | 0.7076 | +0.1673 | 1.0 |
| `kitchen_sink_hybrid_net::classic_fmmm_steps10` | 0.8201 | 0.8416 | -0.0215 | 0.6913 |
| `org_chart_1_5_4_8::classic_classical_mds_default` | 0.8000 | 0.8389 | -0.0389 | 1.0 |
| `org_chart_1_5_4_8::classic_classical_mds_igraph_fidelity` | 0.8000 | 0.8389 | -0.0389 | 1.0 |
| `parallel_cycles_4x5::classic_neato` | 0.8333 | 0.9644 | -0.1310 | 1.0 |

Interpretation: current `np` equivalence penalizes both worse and better preservation. The first `np`-only binder is a clear example: Dagua preserves graph neighborhoods much better (+0.1673) yet fails equivalence because absolute delta exceeds 0.02.

## Real Deficit or Margin Artifact?

For the binding `np` rows, mixed:

- `disconnected_encoder_residual::classic_fmmm_steps10`: not a deficit. It is an improvement punished by symmetric equivalence.
- `kitchen_sink_hybrid_net::classic_fmmm_steps10`: likely a margin-edge artifact from the stored means alone. The deficit is -0.02149, only 0.00149 beyond the absolute 0.02 band.
- The two org-chart MDS rows (-0.0389) and `parallel_cycles_4x5::classic_neato` (-0.1310) are real `np` shortfalls under the current scalar metric.

I attempted direct per-seed variance recomputation for the five binders from stored layouts, but the result-store/graph-loading path did not complete within the audit window. Therefore I am not claiming a seed-variance-calibrated reclassification. The stored means are enough to show that `np` is rarely binding and sometimes punishes improvements, but not enough to prove all `np`-only deficits are seed-noise artifacts.

## Corrected Gate

Keep `np` as a basin-invariant graph-neighborhood preservation metric, but change its gate from symmetric equivalence to directional non-inferiority:

```text
np passes if np_D_mean >= np_R_mean - np_margin
```

For stochastic Mode A, implement this as a one-sided non-inferiority TOST/CI on `np_d - np_r` with lower bound `-np_margin`, not `abs(mean difference) <= margin`. For Mode B, use the same directional rule against the deterministic reference scalar.

Recommended margin:

```text
np_margin = max(0.02, q95 reference self-split |mean(np_A) - mean(np_B)|, graph_discrete_step)
```

where `graph_discrete_step` should account for the score quantum induced by finite `N` and `k` (roughly one neighbor-overlap unit averaged over eligible nodes). This matches the report's stated calibration philosophy (`q95 self-reference split rule`) at `scripts/definitive_fidelity_report.py:2438` while preserving the current 0.02 floor.

This corrected gate would stop laundering worse layouts, avoid penalizing better graph-neighborhood preservation, and make the `np` threshold robust to small/discrete graphs.
