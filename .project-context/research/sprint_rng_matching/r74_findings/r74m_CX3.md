# CX3 Audit: 3Q TOST/IUT Battery, Margins, and Rung-4 vs 3Q

## Scope and Sources

Read `/tmp/r74m_cx_shared.md` and audited the r73 fidelity artifacts and implementation paths:

- `scripts/definitive_fidelity_analysis.py`
- `scripts/definitive_fidelity_report.py`
- `dagua/eval/distributional_fidelity.py`
- `eval_output/fidelity_definitive_r73/per_combo.json`
- `eval_output/fidelity_definitive_r73/DEFINITIVE_FIDELITY_REPORT.md`
- `eval_output/fidelity_definitive_r73/controls/gate_results.json`

No repo code or benchmark output was modified. I attempted a direct reference metric variance recompute from stored layouts, but stopped it because graph registry/loading was not returning under the concurrent live r74 reanalysis load. I did not write into `eval_output`.

## 1. How 3Q Is Decided

### Metric-level tests

For Mode A seeded-reference rows, the quality battery samples up to 60 matched seed pairs and runs paired TOSTs:

- stress: `paired_tost(stress_d - stress_r, quality_stress_margin(stress_r))`
- crossings: `paired_tost(cross_d - cross_r, quality_cross_margin(cross_r))`
- neighborhood preservation: `paired_tost(np_d - np_r, QUALITY_NP_ABS_MARGIN)`

For Mode B deterministic-reference rows, it samples up to 60 Dagua seeds against one fixed reference layout and runs one-sample TOSTs:

- stress: `one_sample_tost(stress_d, stress_target, quality_stress_margin([stress_target]))`
- crossings: `one_sample_tost(cross_d, cross_target, quality_cross_margin([cross_target]))`
- NP: `one_sample_tost(np_d, np_target, QUALITY_NP_ABS_MARGIN)`

Evidence:

- `scripts/definitive_fidelity_analysis.py:1163-1179`
- `scripts/definitive_fidelity_analysis.py:1206-1224`

### IUT p-value

The battery p-value is an intersection-union max-p conjunction:

- `battery_p_iut = max(battery_stress_p_tost, cross_p_tost, np_p_tost)` when all three are finite.
- `quality_identical_raw = stress_ok and cross_ok and np_ok`.
- A metric is raw-ok if degenerate-SD direct equivalence is true, or finite `p_tost < 0.05`.

Evidence:

- `scripts/definitive_fidelity_analysis.py:1336-1352`
- `scripts/definitive_fidelity_analysis.py:1409-1424`

### BH correction and the actual rung gate

The report applies BH to the full `battery_p_iut` family:

- `apply_bh_family(eligible, "battery_p_iut", "q_battery")`
- r73 family size: 3692.

Evidence:

- `scripts/definitive_fidelity_report.py:433-436`
- `eval_output/fidelity_definitive_r73/DEFINITIVE_FIDELITY_REPORT.md:64`

Important implementation nuance: finalization sets `row["quality_identical"] = bool(row.get("quality_identical_raw", False))` before `assign_rung`. Then `assign_rung` computes:

- `quality_identical = record["quality_identical"] or q_battery < 0.05`

So current 3Q is not purely BH-gated. A raw all-three metric pass is sufficient, and BH q can also pass. This is conservative for rung-4 accounting here because 0/574 rung-4 rows have `quality_identical_raw = True`, but it is a design/wording mismatch if the intended rule was "BH-corrected IUT only."

Evidence:

- `scripts/definitive_fidelity_report.py:449-450`
- `dagua/eval/distributional_fidelity.py:361-365`
- `dagua/eval/distributional_fidelity.py:389-390`

## 2. Exact 3Q Equivalence Margins

The strict battery margins are:

- normalized stress: `max(0.02 * mean(reference stress), 1e-6)`
- crossings: `max(0.02 * mean(reference crossings), 0.5)`
- neighborhood preservation: absolute `0.02`

Evidence:

- `scripts/definitive_fidelity_analysis.py:1376-1389`
- `scripts/definitive_fidelity_analysis.py:1392-1404`
- `eval_output/fidelity_definitive_r73/DEFINITIVE_FIDELITY_REPORT.md:12`

Rung 3 is different: it is the loose stress-only 5% fallback. 3Q is the stricter three-metric battery.

Evidence:

- `eval_output/fidelity_definitive_r73/DEFINITIVE_FIDELITY_REPORT.md:328-331`

## 3. Current r73 Rung-4 Battery Facts

r73 definitive counts:

- total rows: 3955
- rung 4: 574
- 3Q: 36
- finite `battery_p_iut`: 3692
- finite `q_battery`: 3692

Rung-4 metric pass counts under current margins:

- stress pass: 22/574
- crossings pass: 185/574
- NP pass: 275/574
- all-three raw pass: 0/574

This confirms the r74 plan statement that no current rung-4 rows pass the strict 3Q gate even before BH.

Family distribution of the 574 rung-4 rows:

- sugiyama: 231
- sfdp: 184
- fmmm: 85
- classical_mds: 34
- gem: 22
- umap: 8
- drl: 5
- maxent_stress: 3
- neato: 2

Mode split:

- Mode A seeded reference: 339
- Mode B deterministic reference: 235

Mode B rows do not expose reference seed-to-seed variance in this artifact because the reference side is a single deterministic draw by design.

## 4. Reference Seed-to-Seed Variance Calibration

Question: are the margins tighter than the reference engine's own seed-to-seed variance?

I could not complete the full reference metric variance recompute in this turn. The data needed for the exact answer is not stored in `per_combo.json`: it stores D/R metric means and D stress SD, but not per-seed reference quality metric vectors or reference SDs for stress/crossings/NP. The calculation must reload stored layouts and recompute per-seed reference metrics.

I attempted that via the same analysis loader and newest benchmark roots, but graph registry/materialization did not return promptly under the concurrent r74 benchmark/reanalysis load, so I terminated my probe to avoid competing with the active run.

This is not a pass. It is a validation gap in the persisted artifact: the definitive row does not record the quantities needed to prove that `margin >= reference self-variance` for stress, crossings, and NP.

Minimum fix: persist for every battery row:

- `battery_stress_R_sd`
- `cross_R_sd`
- `np_R_sd`
- `battery_stress_R_split_delta` or equivalent disjoint self-pair statistic
- `cross_R_split_delta`
- `np_R_split_delta`
- a positive-control row/result for `reference split A` vs `reference split B`

## 5. Positive Control: Can the Battery Certify Known-Equivalent Pairs?

The existing controls do not prove this for the 3Q battery.

`gate_results.json` contains:

- positive Mode A: 39/39 pass, but the implementation checks final rungs in `{0,1,2}`. This is a distributional/seed-tracking positive control, not a quality-battery sensitivity control.
- positive Mode B: 39/39 pass, but it checks final rungs in `{2',0}`. This is a typicality/deterministic-reference control, not a quality-battery sensitivity control.
- quality-identical anti-laundering: 0/40 negative+chance controls become 3Q. This proves specificity against laundering, not sensitivity for known-equivalent pairs.

Evidence:

- `eval_output/fidelity_definitive_r73/controls/gate_results.json`
- `scripts/definitive_fidelity_report.py:2260-2270`
- `scripts/definitive_fidelity_report.py:2280-2290`
- `scripts/definitive_fidelity_report.py:2356-2384`

Verdict: critical validation gap. There is no persisted positive control showing the quality battery can certify `reference vs reference` on disjoint seed splits. The battery's false-positive control is checked; its sensitivity is not.

## 6. Variance-Tied Margins: How Many of the 574 Would Pass?

Not safely computable from the persisted definitive artifact alone.

A defensible variance-tied policy would be metric-wise:

- `stress_margin_vt = max(current_stress_margin, reference_stress_seed_sd_or_split_delta)`
- `cross_margin_vt = max(current_cross_margin, reference_cross_seed_sd_or_split_delta)`
- `np_margin_vt = max(0.02, reference_np_seed_sd_or_split_delta)`

Then recompute the three TOST p-values, the max-p IUT, and the full-run BH family. Because current finalization also accepts `quality_identical_raw`, the report must state both raw all-three pass and BH pass.

I cannot honestly report a count without recomputing reference metric vectors. Any count derived only from D/R means would be fabricated because TOST p-values depend on the per-seed paired-difference SD, and the calibration question specifically depends on reference seed-to-seed metric variance.

Expected direction: if the strict margins are below reference SD for many rows, some current rung-4 rows are margin artifacts. But the current data also shows many genuine quality differences, especially where stress gaps are huge in scale-invariant terms.

## 7. Genuine Quality Gaps vs Margin-Artifact Floor

Using persisted D/R stress means, many rows are not plausible margin artifacts:

Rung-4 scale-invariant stress gap by family, median `(D - R) / R`:

- sfdp: `+0.343` median, max `+75.424` -> many genuine D-worse stress gaps.
- fmmm: `-0.014` median, but max `+3.967` -> mixed; some D-better/near, some genuine D-worse.
- sugiyama: `-0.9998` median -> D has much lower normalized stress than reference, so stress failure is not the explanation; these fail other battery dimensions and/or one-sided TOST framing. Need crossing/NP diagnosis, not stress-only "quality gap" labeling.
- umap: median `-0.506`, max `+102.410` -> mixed with severe outliers.
- gem: median `-0.129` -> not stress-worse on median; crossing/NP likely blockers.

Examples from persisted rows:

- `ba_2000::classic_sugiyama_default`: stress D `108370.103` vs R `281021448.763`, but crossings D `1907883` vs R `2105797`, NP D `0.03895` vs R `0.00895`; stress p is `1.0` despite D being far lower, because the TOST is symmetric equivalence, not one-sided quality-better acceptance.
- `asymmetric_hourglass_hub::classic_sugiyama_graphviz_fidelity`: stress D `541.006` vs R `26715.760`, crossings both `0`, NP D `0.7929` vs R `0.8929`; failure is NP loss, not stress.

So the audit should not flatten all rung-4 rows into "FP floor." There are at least three buckets:

1. Genuine D-worse stress gaps: e.g. many sfdp rows.
2. Quality-different but D-better on stress: e.g. many sugiyama rows; symmetric equivalence fails, but not because D stress is worse.
3. Possible margin-artifact floor: rows with small D/R mean differences where the current 2%/0.5/0.02 margins may be tighter than reference self-variance. This bucket requires the missing reference self-variance computation.

## 8. Recommendations

1. Add and persist a reference self-variance calibration table for the battery metrics.
2. Add a 3Q positive control: `reference seeds split A` vs `reference seeds split B`, same graphs/engines, same battery code, disjoint seeds.
3. Report two 3Q decisions separately:
   - raw IUT all-three pass (`quality_identical_raw`)
   - BH-adjusted IUT pass (`q_battery < 0.05`)
4. Decide whether final 3Q should be raw all-three OR BH, or BH only. Current code effectively allows raw all-three to bypass BH.
5. For variance-tied margins, recompute from per-seed layout metrics and publish both raw and BH counts for the 574 rung-4 rows.

## Bottom Line

The 3Q battery is strict and currently blocks all 574 rung-4 rows, but its calibration is not fully validated. The anti-laundering/specificity control passes at 0/40, yet there is no corresponding sensitivity positive control for known-equivalent reference-vs-reference splits. The exact reference self-variance and variance-tied pass count are not derivable from `per_combo.json`; they require a new layout-metric recompute or, preferably, persisted per-row reference metric SD/split fields.
