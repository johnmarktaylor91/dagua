# BUCKET: metrics/criteria (the eval system itself -- r74's biggest wins came from HERE)

No target JSON; your subjects are the metric/criteria code paths. r74 found two real eval
defects (scale artifact in stress; margins tighter than reference self-noise) that reclassified
72 combos. The sprint lead wants the SAME audit lens applied to what remains. Four subtasks:

## S1. Crossings-equivalence audit (highest priority)
Crossings now fail on 235 of 337 remaining divergent combos. Audit dagua/metrics count_crossings
+ its use in scripts/definitive_fidelity_analysis.py (quality_metric_samples ~line 1297,
cross_sampled = E>500 ~line 1362, the TOST at ~line 1260, margin = max(floor, cross_ref_self_spread)).
Questions:
(a) DISCRETENESS: crossings is an integer count. On small/sparse graphs the reference's 100 seeds
    can produce identical counts -> ref_self_spread = 0 -> margin collapses to the floor. What IS
    the floor for crossings? Is a 1-2 crossing absolute difference on a 50-node graph failing
    combos that any honest reading calls identical quality? Quantify: for the 235 cross-failing
    combos, what are the absolute deltas and the margins? (Data: r74_phase2_rescore.jsonl fields
    cross_D_mean/cross_R_mean/cross_margin/cross_ref_self_spread.)
(b) SAMPLING: cross_sampled=True when E>500 -- how is the sampled estimator computed (same edge
    sample both sides? seeded? unbiased?), and is sampling noise reflected in the margin? A
    sampled count compared against a margin built from a DIFFERENT noise source is miscalibrated.
(c) SCALE/DEGENERACY: any analogue of the r74 stress scale-artifact for crossings (e.g., near-
    collinear layouts where epsilon decisions flip counts)?
(d) Propose the calibrated fix (e.g., count-noise-aware margin, exact counting cutoff raise, or
    rate-based normalization), expected reclassification impact, and the regression tests +
    anti-laundering controls (gate_5 0/40 chance+negative) any change MUST pass.

## S2. Hang-safe scoring for the ~165 huge-graph divergent combos (>300 nodes)
The r74 Phase-2 rescore skipped >300-node graphs because crossings + APSP grind (ba_2000/ba_5000
hung the analysis). Design (do not implement) a bounded-time scoring path: landmark/pivot APSP
for stress (with error bound), sampled crossings with CI, np via approximate kNN, per-combo time
budget + timeout fallback semantics, and how margins stay honest under approximation (the margin
and the metric must share the estimator). Enumerate the actual >300 divergent combos from
eval_output/fidelity_definitive/r74_analysis.jsonl (rows NOT in the 409 rescored <=300 set;
cross-check combo counts) so the follow-up task has an exact worklist.

## S3. Population two-sample equivalence for stochastic-reference floors
For combos where the reference is stochastic and per-combo 3Q certification would certify chance
(the r73/r74 "laundering limit"), design the aggregate claim: a two-sample distributional
equivalence test (e.g., per-seed quality metric samples dagua-vs-ref, Wilcoxon/TOST hybrid or
KS-based equivalence with pre-registered margin) at the ENGINE level with BH correction, plus the
controls that keep it honest. State exactly what headline it licenses ("engine X is
population-quality-equivalent on graph class Y") vs what it does NOT.

## S4. Dagua-better policy (criteria question -- analysis only)
93 of 337 remaining divergent combos are dagua-BETTER on every failing leg (two-sided TOST fails
because we're outside the margin on the GOOD side). Current tiering lumps them into "divergent".
Analyze and recommend: root-cause-first policy (better = suspected comparison bug), and IF a
better-cause is proven benign, what honest tier should exist (e.g. "quality-superior,
distributionally distinct")? It must NOT count toward the identical headline. Check how np's
one-sided non-inferiority (r74 d4ef688) handles this asymmetry vs stress/crossings' two-sided.

OUTPUT: same contract as shared preamble; BUCKET=metrics. Rank by expected reclassification
impact. Include per-change control/gate requirements. NO code changes in this task.
