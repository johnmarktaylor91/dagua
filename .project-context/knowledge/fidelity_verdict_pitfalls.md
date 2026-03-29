# Fidelity Verdict Pitfalls

Hard-won lessons from the 2026-03-28 fidelity analysis session. These are
the traps that produced incorrect verdicts and must be handled correctly
in any future fidelity/comparison pipeline.

## 1. Mirror Reflections Are Valid Equivalence

**Problem:** Procrustes forces proper rotation (det=+1). If a reimplementation
produces a mirror reflection of the original, the non-reflected alignment
shows artificially high RMSD and displacement, causing false "divergent" verdicts.

**Fix:** Treat `mirror_match` as an explainable anomaly, not a fidelity failure.
Most layout algorithms have arbitrary axis orientation (SVD sign ambiguity,
arbitrary coordinate frame). A mirrored layout is a valid equivalent output.

**Proper fix (TODO):** Use the reflected Procrustes RMSD when a mirror is
detected, so displacement values are accurate rather than just treating the
anomaly as benign.

**Where:** `explainable_only()` in `scripts/fidelity_analysis.py`

## 2. Scale Normalization Mismatch

**Problem:** Procrustes normalizes layouts to unit Frobenius norm (pure shape
comparison), but quality metrics (edge_length_mean, etc.) are computed on raw
coordinates. Two implementations producing identical shapes at different scales
get "identical" Procrustes but different quality metrics.

**Impact:** Scale ratio anomalies (`scale_ratio_out_of_range`) trigger false
downgrades. An algorithm producing correct shape at 2x scale is faithful.

**Where:** `finalize_group_row()` anomaly checks, scale ratio thresholds

## 3. Identical Distributions Break Statistical Tests

**Problem:** When orig and reimpl produce byte-identical outputs for the same
seed, the quality metric distributions are identical. TOST gets pooled SD = 0,
causing divide-by-zero -> NaN p-values. NaN < 0.05 is False, so TOST "fails"
and the verdict becomes "divergent" -- the exact opposite of reality.

**Fix:** When TOST returns NaN, check if orig_mean == reimpl_mean. If so,
equivalence is trivially satisfied -> pass. Same logic for Mann-Whitney.

**Rule:** Identical distributions must ALWAYS resolve to strong_equivalent,
never divergent. A statistical test failing on identical data is a test bug,
not a fidelity failure.

**Where:** `_tost_passes()` and `_mw_significant()` in `finalize_group_row()`

## 4. Within-vs-Between Is the Right Fidelity Test

**Problem:** The verdict logic uses TOST on quality metrics to judge stochastic
algorithm fidelity. But the natural test is comparing within-implementation
Procrustes variance to between-implementation Procrustes variance. If the
between/within ratio is ~1.0, the reimplementation is indistinguishable from
"just another seed of the original."

**Data:** The pairwise_similarity.csv already has orig-orig, reimpl-reimpl, and
orig-reimpl Procrustes comparisons. The within-vs-between ratio showed most
algorithms at ~1.0 even when quality metric TOST said "divergent."

**TODO:** Consider making within-vs-between Procrustes ratio the primary
fidelity signal, with quality metrics as secondary diagnostics.

## 5. Family-Level Aggregation Is Conservative

**Problem:** Family verdict requires ALL per-graph verdicts to match. One bad
graph (insufficient data, edge case, outlier) downgrades the entire family.
With 105 test graphs, even a 95% pass rate means the family gets downgraded.

**Impact:** Every family showed "divergent" or "partial_match" even when 90%+
of per-graph pairs were equivalent.

**Consideration:** Family verdict could use majority-rule or percentile-based
thresholds instead of all-or-nothing.

## 6. Pre-Loading 406K HDF5 Entries Is Unnecessary When Serial

**Problem:** The HDF5 pre-load was added to avoid thread contention, but after
removing ThreadPoolExecutor (GIL hang fix), serial processing can read HDF5
on demand. The pre-load added 30+ min of dead time with no progress output.

**Rule:** When switching from parallel to serial execution, remove parallel
infrastructure (pre-loads, caches, batching) that only existed to support
concurrency.

## 7. Always Add Progress Logging to Long Loops

**Problem:** The HDF5 pre-load loop (406K iterations, 30+ min) had no progress
output. Impossible to tell if it was working, stuck, or how long remained.

**Rule:** Any loop that might exceed 10 seconds MUST have progress logging
with count, elapsed time, and ETA. No exceptions.
