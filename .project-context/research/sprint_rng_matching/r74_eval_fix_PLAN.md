# r74 EVAL-PIPELINE FIX PLAN (reconciled Opus O1-O5 + Codex CX1-CX5 metric audit)

Trigger: JMT -- "a systematic aggregate quality difference CANNOT be mere rounding (averages out over 100
seeds); it implies a metric artifact OR a real non-trivial difference = more work." Verdict: BOTH are true.
Full findings: r74_findings/r74m_O{1..5}.md + r74m_CX{1..5}.md.

## What the audit established (Opus+Codex agree unless noted)
- **np / neighborhood-preservation is BASIN-INVARIANT and FINE** (O1+CX1). It scores each layout's
  preservation of the GRAPH's neighborhoods (equivalence_metrics.py kernel), not dagua-vs-ref positions.
  np is NOT the binding constraint (binds ~5-19); dagua is generally BETTER on np (mean +0.024). O6's
  "kNN binding" claim is RETRACTED.
- **STRESS is the binding constraint** (~508-548/574) (O1,O3,CX1,CX3).
- **Battery stress has a SCALE ARTIFACT** (O2+CX2): the 3Q battery uses raw `normalized_stress`
  (equivalence_metrics.py:417-420, NO optimal-scale alpha) while the DIAGNOSTIC stress fits alpha
  (distributional_fidelity.py:263-267). No Procrustes/scale alignment precedes battery stress. Traced
  20.2x battery vs 1.05x scale-invariant on the same combo; ref battery stress maxes 9.8e9 (unnormalized).
  **SCOPE DISPUTED -> reconciled:** O2 said 32% by a loose criterion; CX2/CX3 (stricter) found the scale
  fix ALONE flips ~0 at the strict 2% margin. So scale-alpha is REQUIRED FOR CORRECTNESS but is not what
  reclassifies -- the MARGIN is.
- **Margins are TIGHTER than the reference's own seed-to-seed variance** (O1,O3,O5,CX1,CX3): the decisive
  miscalibration. gem's 2% stress margin is ~50-100x tighter than gem's reference self-SD -> the test
  demands dagua match the reference closer than the reference matches ITSELF across seeds. gem floor:
  dagua UNIFORMLY BETTER (0/19 worse). CAVEAT (CX3/CX4): per_combo.json does NOT persist reference
  per-seed SDs, so the variance-tied pass count is NOT derivable without recomputing per-seed quality
  arrays from stored layouts.
- **NO POSITIVE CONTROL** (O3+CX3): the battery has chance/negative controls (specificity) but nothing
  proving it CAN certify a known-equivalent pair (sensitivity) -> unverified. CRITICAL gap.

## The LAUNDERING LIMIT (the deep, honest finding -- O4+CX4)
For STOCHASTIC-reference floor combos (gem/fmmm/sfdp), the chance control = the correct reference cloud
with seed-labels permuted -> its MARGINAL quality distribution is IDENTICAL to the real reference. So ANY
per-combo quality-equivalence test that certifies these ALSO certifies chance = laundering. The crude
"D<=R parity" rule reproduces exactly 22/40 control passes for this reason. **You cannot per-combo-certify
quality-equivalence against a stochastic cloud without certifying chance -- an information-theoretic limit,
not a fixable bug.** O4's pre-screen (mean_W_R<=~1.0) keeps 0/40 controls but, honestly, certifies ~0 of
the stochastic-ref combos (correctly refuses what can't be distinguished from chance).

## THE HONEST THREE-WAY SPLIT of the 574
1. **Eval-artifact / margin-floor** (gem uniformly-better; much of fmmm/sfdp connected): fail ONLY due to
   the scale artifact + margins tighter than reference self-variance. dagua is equal-or-better. The
   DEFENSIBLE claim is AGGREGATE quality-neutrality (not per-combo 3Q, which the laundering limit blocks
   for stochastic refs). Size needs the per-seed recomputation to quantify (~roughly 150-250, UNVERIFIED).
2. **Real per-combo deficits = genuine MORE WORK** (CX5, survive scale+margin correction):
   - `deep_chain_20::classic_fmmm_steps10` stress 0.145 vs 0.032 (4.5x worse), np 0.81 vs 0.98.
   - `asymmetric_hourglass_hub::classic_sfdp_default` systematic stress deficit 0.030 vs 0.0267.
   - sugiyama graphviz-fidelity contrasts: scale-invariant stress + np gaps.
   Suspect cause files (CX5): fmmm.py:1723 (budget/refinement), sfdp.py:426/:848 (p clamp/adaptive refine),
   sugiyama.py:60. THESE are the real algorithm targets JMT predicted -- now pinpointed by better metrics.
3. **Genuine large gaps** (~231 sugiyama-led, deterministic, real 100% scale-invariant gaps; ba_500
   22344 vs 2805 crossings): need the deep ports (C1 partly done). Correctly divergent.

## THE EVAL FIXES (implement after re-bench frees the files; gate EVERY change on controls)
Phase 1 -- CORRECTNESS (clear, safe):
  1. **Scale-alpha in battery stress**: add `fit_scale=True` to `normalized_stress`
     (equivalence_metrics.py:417-420), apply at battery call sites (analysis.py quality_metric_samples
     + deterministic_quality_metrics). Re-derive the O(1) margin. (O2+CX2.)
  2. **Non-inferiority on np** (and stress): `np passes if np_D_mean >= np_R_mean - margin` (one-sided) --
     dagua-better never fails a QUALITY test. (O1+CX1.)
  3. **POSITIVE CONTROL**: reference-split-A vs reference-split-B (disjoint seed halves of the SAME
     reference) MUST pass the battery. If it fails, the battery is mis-calibrated -- this is the
     sensitivity check the pipeline lacks. (O3+CX3.)
Phase 2 -- CALIBRATION (trickier, laundering-sensitive):
  4. **Persist reference per-seed metric SDs** (data-pipeline change) and set **variance-tied margins**
     (margin = max(absolute_floor, k * reference_self_spread / q95 of ref self-split)). A layout cannot be
     required to match the reference tighter than the reference matches itself. (O1,O3,O5,CX1,CX3.)
  5. **Discriminability pre-screen + practical-gap veto** (O4+CX4): only attempt per-combo 3Q where the
     reference is canonical (mean_W_R<=~1.0) AND metrics non-saturated; hard-veto genuine gaps (sugiyama
     19539 crossing delta vs margin 56). For stochastic refs, do NOT per-combo-certify (laundering limit).
  6. **Aggregate quality-neutrality report**: the defensible answer to JMT for the stochastic-ref floor --
     across those combos, dagua's quality distribution is statistically indistinguishable from (or better
     than) the reference's. Population-level, not per-combo. (O5+CX5.)

## VALIDATION GATES (non-negotiable -- this is the anti-laundering line)
- After EVERY metric change: re-run `definitive_fidelity_report.py --controls` -> chance+negative MUST stay
  0/40 in 3Q. If a change lets any control through, it LAUNDERS -> revert.
- The NEW positive control (ref-split-A vs ref-split-B) MUST pass (proves sensitivity).
- Genuine gaps (sugiyama) MUST stay divergent (the veto).

## SEQUENCING
Re-bench (pid 216733) finishes -> current-metric r74 scorecard (baseline). THEN implement Phase 1 in main
tree -> validate controls+positive -> re-score -> Phase 2 -> re-score -> report BOTH the corrected rung
distribution AND the aggregate quality-neutrality result. Then the real-deficit algo targets (split 2) are
the next fidelity work (separate from eval fixes).
