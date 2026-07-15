# AXIS CX4: anti-laundering controls and principled quality-equivalence test

## Scope and data read

Read-only audit of the fidelity/evaluation pipeline. I read:

- `/tmp/r74m_cx_shared.md`
- `scripts/definitive_fidelity_analysis.py`
- `scripts/definitive_fidelity_report.py`
- `dagua/eval/distributional_fidelity.py`
- stored artifacts:
  - `eval_output/fidelity_definitive/controls/gate3_negative.jsonl`
  - `eval_output/fidelity_definitive/controls/gate4_chance.jsonl`
  - `eval_output/fidelity_definitive_r73/per_combo.json`
  - `eval_output/fidelity_definitive_r73/controls/gate_results.json`

No code or benchmark artifacts were modified. This file is the only output.

## How the controls are constructed

### Chance controls

Chance controls are not synthetic wrong-reference rows. They are 20 real Mode-A-eligible `(graph, engine)` pairs drawn from the same combo universe. The selector:

- resolves the registered reference for each engine,
- requires at least `MIN_MODE_SEEDS` matched successful seeds between reimplementation and reference,
- appends eligible pairs,
- deterministically samples 20 with `draw_sorted(..., "r70::chance")`.

Evidence: `scripts/definitive_fidelity_analysis.py:1732-1739`.

Interpretation: these are designed to make seed-tracking p-values look uniform/recoverable under a null-ish stochastic pairing regime. They are not designed to be adversarial wrong-quality examples. For quality anti-laundering, they are useful only if a proposed 3Q rule does not accidentally promote stochastic clouds whose reference has no canonical layout.

### Negative controls

Negative controls are deliberate mispairs on the same graph. The selector:

- groups engines by graph,
- considers every `(engine, other)` pair on that graph,
- rejects same engine,
- rejects same pre-registered algorithm token,
- rejects same base reference engine,
- requires the reimplementation cloud to be moderately informative (`offdiag_mean(W_D) <= 1.0` from first 30 layouts),
- requires the true reference draw and wrong reference draw to differ by Procrustes distance `> 0.1`,
- encodes the synthetic pair as `(graph, "engine\tother_ref")`, then samples 20 with `draw_sorted(..., "r70::negctl")`.

Evidence: `scripts/definitive_fidelity_analysis.py:1766-1790`, `scripts/definitive_fidelity_analysis.py:1872-1910`, and `scripts/definitive_fidelity_analysis.py:1913-1920` onward.

Interpretation: negative controls are not obviously mis-built. They avoid identical/same-family references and require a reference-distance difference. They are still allowed to be high-variance stochastic reference clouds; the selector does not require the quality metrics to be discriminative for that graph/reference.

## Current quality battery

For Mode A, the existing strict quality battery is paired by matched seed:

- select up to 60 deterministic seed indices,
- compute per-layout stress, crossings, and kNN neighborhood preservation,
- run paired TOST on `D - R` for each metric,
- combine as an IUT by taking the max of the three p-values,
- set `quality_identical_raw = stress_ok and cross_ok and np_ok`.

Evidence: `scripts/definitive_fidelity_analysis.py:1163-1179`, `scripts/definitive_fidelity_analysis.py:1227-1245`, `scripts/definitive_fidelity_analysis.py:1248-1309`, `scripts/definitive_fidelity_analysis.py:1312-1373`.

For Mode B, the same quality metrics are compared one-sample against the single deterministic reference layout. Evidence: `scripts/definitive_fidelity_analysis.py:1206-1224`.

TOST behavior is standard t-based equivalence, with a degenerate-SD exact branch. Evidence: `dagua/eval/distributional_fidelity.py:272-309` and `dagua/eval/distributional_fidelity.py:917-970`.

The report applies BH to `battery_p_iut` into `q_battery`, but `assign_rung()` can also use raw `quality_identical`; final 3Q promotion happens when quality-identical is true. Evidence: `scripts/definitive_fidelity_report.py:432-450` and `dagua/eval/distributional_fidelity.py:343-390`.

The hard anti-laundering gate checks finalized negative + chance controls and passes if 3Q rate is at most 5%. Evidence: `scripts/definitive_fidelity_report.py:2353-2384`.

Stored r73 gate result: gate 5 passed with 0/40 3Q controls. `eval_output/fidelity_definitive_r73/controls/gate_results.json` reports `three_q_count=0`, `scored=40`.

## Why the crude `D <= R` parity rule leaks 22/40 controls

I reproduced the r73 crude rule on the 40 stored control rows. The rule that matches the reported 22/40 leakage is `mean_W_D <= mean_W_R` (same count with `plain_mean_W_D <= plain_mean_W_R`):

- all controls: 22/40
- chance controls: 10/20
- negative controls: 12/20

Other simple quality-direction checks also leak because many controls are not worse on scalar quality axes:

- stress not worse: 16/40
- crossings not worse: 26/40
- neighborhood preservation not worse: 29/40
- at least 2 of 3 scalar quality metrics not worse: 27/40
- all 3 scalar quality metrics not worse: 8/40

This is not primarily evidence that the controls are mis-built. It is evidence that `D <= R` and scalar quality parity are non-discriminative when the reference distribution is stochastic or quality-flat.

The reference-cloud spreads in the 40 controls are large:

- `mean_W_R` median: 0.7874
- min: 0.2341
- max: 1.3772
- controls with `mean_W_R <= 0.3`: 3/40
- controls with `mean_W_R <= 0.5`: 6/40
- controls with `mean_W_R <= 1.0`: 22/40

So for many controls, a draw from the wrong side can be no farther from the reference cloud than another reference draw. In that regime, `distance-to-shuffled ≈ distance-to-correct` is exactly what the stochastic cloud predicts.

There is a smaller control-design weakness: negative controls only require one reference draw distance `> 0.1`, not a metric-space discriminability proof. That is adequate for placement anti-laundering, but insufficient for quality anti-laundering.

## Proposed principled quality-equivalence test

I would replace the current 3Q promotion rule with a two-arm test that distinguishes canonical references from stochastic references.

### Inputs

For each combo, compute the existing per-seed arrays for:

- normalized stress, lower is better,
- crossing count or seeded estimate, lower is better,
- neighborhood preservation, higher is better.

Use all available 100 seeds where possible; 60-seed capping is acceptable only as a runtime fallback. The hypothesis in the brief is explicitly aggregate over 100 seeds, so 100 should be the default for 3Q certification.

### Arm A: canonical/discriminable references

Use this arm only when the reference cloud is canonical enough for controls to separate. Pre-screen:

- Mode A: `plain_mean_W_R <= 1.0` and `n_ref_seeded_ok >= 30`.
- Mode B deterministic reference: allowed only if existing Mode B typicality is informative, or the deterministic reference is near-deterministic by existing flags.
- Additional validation gate before release: this arm must certify 0/40 negative+chance controls on the stored controls.

Then require the strict IUT quality battery:

- stress population mean equivalence within `max(0.02 * mean(R_stress), 1e-6)`,
- crossing population mean equivalence within `max(0.02 * mean(R_cross), 0.5)`,
- neighborhood population mean equivalence within `0.02`,
- all three pass after report-family BH on the IUT max p-value.

This arm certifies genuine FP-floor cases where the reference is tight enough that “same quality” is meaningful. On r73 stored summaries, the proxy implementation `mean_W_R <= 1.0` plus existing `quality_identical_raw or q_battery < 0.05` certifies 1,152 non-control combos and 0/40 controls. The current r73 final 3Q count is only 36, so this would be a substantial but guarded expansion.

Why `1.0`: it is already the code's coarse “informative cloud” cutoff for negative-control construction (`W_D <= 1.0`) and, on the stored 40 controls, the only raw 3Q-like control passes have `mean_W_R=1.2482` and `1.3377`; adding `mean_W_R <= 1.0` blocks them. This should not be frozen forever: the threshold should be registered by control validation, not by intuition.

### Arm B: stochastic-reference population equivalence

For stochastic references with `plain_mean_W_R > 1.0`, do not use paired seed deltas. Seed identity is not canonical. Instead compare the two populations of quality metrics:

- `D_metric[100]` vs `R_metric[100]` as independent samples,
- Welch/TOST or bootstrap percentile TOST on mean difference,
- same metric margins as above,
- max-p IUT across the three metrics,
- BH across the report family.

This arm should also require a discriminability calibration, because stochastic quality distributions can still be flat. For each graph/reference family, build pseudo-negative controls by comparing the reference quality population to wrong-reference populations selected by the negative-control logic. Certify stochastic 3Q only if the exact population-equivalence rule rejects all 40 registered controls plus the pseudo-negative family at 0 certifications. If any wrong population is equivalent, mark the combo `QUALITY_UNDISCRIMINABLE`, not 3Q.

This population arm is the correct treatment for true FP-floor: rounding chaos can move a run to another basin, but across 100 seeds it should not shift the quality distribution. A paired per-seed test is too strong and sometimes conceptually wrong when seed labels do not identify a canonical basin.

### Explicit exclusion rule for genuine gaps

Before any 3Q certification, apply a hard practical-difference veto:

- For each lower-is-better metric, fail if `mean(D) - mean(R) > margin`.
- For neighborhood preservation, fail if `mean(R) - mean(D) > margin`.
- This is separate from the statistical p-value; it prevents high-variance or sample-size artifacts from certifying obvious practical gaps.

A Sugiyama row with crossings `22344` vs `2805` is excluded immediately: crossing delta is `19539`, while the 2% crossing margin is `max(0.02 * 2805, 0.5) = 56.1`. It fails by ~348x the allowed margin. This is a real algorithmic/quality gap, not FP-floor.

## Validation plan

1. Recompute quality metric arrays for the 40 registered controls, not just summary means.
2. Run the exact new rule over those controls.
3. Required result: 0/40 certifications. This is stricter than the current gate's <=5% rule.
4. Recompute over all r73 non-control combos.
5. Report counts by arm:
   - canonical Arm A certified,
   - stochastic Arm B certified,
   - quality-undiscriminable stochastic rows,
   - vetoed practical gaps.
6. Add a regression artifact with per-control decision reasons: failed pre-screen, failed stress, failed crossings, failed neighborhood, or veto.
7. Keep the existing chance-control seed-tracking KS gate separate; it tests `p_track`, not quality discriminability.

Using currently stored r73 summaries, the Arm-A proxy certifies 1,152 non-control combos and 0/40 controls. The exact Arm-B count cannot be obtained from `per_combo.json` because it does not retain the per-seed quality arrays needed for independent two-sample TOST; it must be recomputed from stored layouts.

## Verdict

The 22/40 crude-rule leakage is mainly a metric/test-design artifact: `D <= R` and scalar quality parity are not discriminative for stochastic reference clouds. The controls are not obviously malformed; they are doing enough to expose the laundering risk. The fix is not to weaken the controls, but to require a discriminability pre-screen for canonical references and a true population-equivalence test for stochastic references, with a hard practical-gap veto.
