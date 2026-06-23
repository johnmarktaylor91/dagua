# CX2: Stress metric scale / normalization / alignment audit

## Verdict

The stress used inside the 3Q strict quality battery is **not scale-normalized**. It is computed on raw coordinate distances by `dagua.eval.equivalence_metrics.normalized_stress`, with no Procrustes alignment and no fitted scalar alpha before residuals.

There is a separate diagnostic / Task-A stress path that **is scale-normalized**: `dagua.eval.distributional_fidelity.stress_per_layout` fits closed-form alpha before residuals.

This means a pure dagua-vs-reference coordinate scale difference can inflate `battery_stress_*` with no quality difference. However, on the current r73 final `per_combo.json`, this does **not** explain most of the 574 rung-4 rows. Only **4/574** rung-4 rows have scale-fitted diagnostic stress means within the strict 2% stress margin while raw battery stress fails.

## Code Trace

Scale-normalized diagnostic stress:

- `dagua/eval/distributional_fidelity.py:241` defines `stress_per_layout`.
- `dagua/eval/distributional_fidelity.py:263-267` computes:
  - `denominator_alpha = sum(layout_d^2)`
  - `alpha = sum(layout_d * graph_d) / denominator_alpha`
  - residual on `(alpha * layout_d) - graph_d`
- `scripts/definitive_fidelity_analysis.py:1026-1029` uses `df.stress_per_layout(...)` for `stress_D_mean`, `stress_R_mean`, and `stress_p_tost`.

Raw, un-scale-normalized battery stress:

- `dagua/eval/equivalence_metrics.py:384` defines `normalized_stress`.
- `dagua/eval/equivalence_metrics.py:417-420` computes layout distances, weights, and:
  - `numerator = sum(weights * (d_layout - d_graph)^2)`
  - no alpha fit; no Procrustes; no centering needed because pairwise distances are translation-invariant.
- `scripts/definitive_fidelity_analysis.py:1275-1281` uses `normalized_stress(...)` for 3Q battery `stress_d` / `stress_r`.
- `scripts/definitive_fidelity_analysis.py:1336-1359` turns those raw stress arrays into `battery_stress_D_mean`, `battery_stress_R_mean`, `battery_stress_p_tost`, and the stress leg of `quality_identical_raw`.
- `scripts/definitive_fidelity_report.py:449-450` passes `quality_identical_raw` into final rung assignment; `q_battery` is computed but not used to override that raw conjunction.

Deterministic rows have the same raw-battery issue:

- `scripts/definitive_fidelity_analysis.py:2220-2221` calls `normalized_stress(...)` in `deterministic_quality_metrics`.

## Quantification On Current Data

Source: `eval_output/fidelity_definitive_r73/per_combo.json`

Rung counts:

- Total rows: 3,955
- Rung 4: 574

For the 574 rung-4 rows:

- `battery_stress` fails: 552/574
- `battery_stress` is the sole failed battery leg: 96/574
- Diagnostic scale-fitted stress mean relative delta <= 2%: 4/574
- Signature rows where diagnostic scale-fitted stress mean is within 2% but raw battery stress is >2% and fails: 4/574
- Among the 96 rows where battery stress is the sole blocker: 0/96 show diagnostic scale-fitted stress equivalence by the available row fields.

Threshold sweep, using diagnostic scale-fitted mean relative delta:

- <=2% and raw battery relative delta >2%: 4/574
- <=5% and raw battery relative delta >2%: 28/574
- <=10% and raw battery relative delta >2%: 141/574
- <=25% and raw battery relative delta >2%: 278/574

Do not interpret the wider thresholds as a fix; the registered strict margin is 2%, and relaxing it would launder differences.

## Specific Rung-4 Trace Rows

These are the 4 rows with the scale-artifact signature under the strict 2% mean-delta check:

| combo | scale-fitted stress D/R | fitted rel delta | raw battery stress D/R | raw rel delta | battery stress p | other failing leg |
|---|---:|---:|---:|---:|---:|---|
| `small_world_100::classic_sfdp_steps200` | 0.017307 / 0.017010 | 1.75% | 9758.233 / 689.468 | 1315.33% | 1.0 | NP p=1.967e-05 |
| `random_bipartite_60::classic_sfdp_theta08` | 0.101641 / 0.100299 | 1.34% | 4.349 / 9431.177 | 99.95% | 1.0 | crossing p=0.240, NP p=0.0389 also not both safely passing |
| `real_karate_34::classic_drl_default` | 0.109230 / 0.107420 | 1.69% | 839.251 / 1286.244 | 34.75% | 1.0 | NP p=0.000213 |
| `multiscale_skip_cascade::classic_sfdp_p_neg2` | 0.059618 / 0.060643 | 1.69% | 27803.161 / 28513.329 | 2.49% | 0.555 | NP p=0.000209 |

Important: these are not obvious final-rung flips because at least NP or crossing also fails in the same battery. They do prove the raw stress leg is scale-sensitive.

## Exact Fix

Do not relax margins.

Add alpha fitting to the battery stress function, not to the TOST threshold.

Best concrete fix:

1. Add a keyword to `dagua/eval/equivalence_metrics.py::normalized_stress`, e.g. `fit_scale: bool = False`, preserving existing default behavior for callers that expect raw stress.
2. When `fit_scale=True`, compute alpha after `weights` and `d_layout`:

```python
denominator_alpha = float(np.sum(weights * np.square(d_layout)))
alpha = 0.0
if denominator_alpha >= EPSILON:
    alpha = float(np.sum(weights * d_layout * d_graph_valid) / denominator_alpha)
residual = (alpha * d_layout) - d_graph_valid
numerator = float(np.sum(weights * np.square(residual)))
```

3. In `scripts/definitive_fidelity_analysis.py`, call:

```python
normalized_stress(layout, edge_index, all_pairs_distances=dists, fit_scale=True)
```

at the battery call sites:

- `quality_metric_samples`: lines 1275-1281
- `deterministic_quality_metrics`: lines 2220-2221

4. Add a regression test with `positions_reference = positions_dagua * c`; raw stress should differ, but `fit_scale=True` stress should match.

## Confidence

High on the code-path verdict: the alpha/no-alpha split is explicit in the traced functions.

Medium-high on the 574-row quantification: it uses the finalized r73 `per_combo.json` fields rather than reloading all seed layouts. That is the correct report artifact for current final rungs, but a full post-fix experiment should recompute the battery on stored layouts with `fit_scale=True` to confirm exact rung movement.
