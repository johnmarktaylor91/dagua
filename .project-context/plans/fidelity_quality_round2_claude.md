# Claude Round-2 Adversarial Critique: v2 Plan Review

Date: 2026-04-09
Status: Second-round validation of fidelity+quality pipeline plan after v1 reviews + user answers

---

## Verification Summary

I verified v2 against the v1 adversarial reviews (Codex and Claude Explore) by:

1. **Reading the v2 plan** at `/home/jtaylor/projects/dagua/.project-context/plans/fidelity_and_quality_pipeline_plan.md` (lines 1-902)
2. **Reading the v1 Codex review** at `/home/jtaylor/projects/dagua/.project-context/plans/fidelity_quality_codex_review.md` (15 issues, 11 critical/high)
3. **Reading the v1 Claude Explore review** at `/home/jtaylor/projects/dagua/.project-context/plans/fidelity_quality_claude_review.md` (4 critical, 3 high)
4. **Spot-checking v1 claims against live code**:
   - `fidelity_analysis.py:1814-1833` for within-vs-between pooling (F1 critical bug)
   - `fidelity_analysis.py:2155-2172` for backwards verdict logic (A5 refactor required)
   - `fidelity_analysis.py:1960-1990` for BH correction location
   - `dagua/metrics.py:341-412, 563-615, 1554+` for stochastic metric signatures
   - `eval_output/variant_bench_full/manifest.json` for real benchmark scope (105 graphs, 60 seeds, 1.27M total scope)
   - Graph tag distribution from manifest (29 wide-parallel, 15 skip-light, 14 diamond, etc.)

---

## Did v2 Fix What v1 Criticized? (SPOT-CHECK)

### F1: Within-RMSD is pooled, not within-original (FAIL)

**v1 critique**: `within_rmsd = pairwise_orig + pairwise_reimpl` at line 1816-1817. Pooling the reimpl side contaminates the baseline.

**v2 claim** (line 154-160): "Change: `within_rmsd` becomes `[c.procrustes_rmsd for c in pairwise_orig]` ONLY."

**Reality**: **STILL BROKEN IN CODE** at line 1816. The plan describes the fix but the code is unchanged. This is a CRITICAL bug that must land as part of FID-A. The fix is described in the plan (A1 group) but **v2 does NOT claim it is already done** — correctly lists it as Group A CRITICAL to do.

**Verdict**: PASS (v2 correctly identifies it as blocking fix, not as already done).

---

### Q3: `graph_rel_best` math for negative/zero metrics (PARTIAL)

**v1 Codex critique** (CRIT #1): Formula breaks when `depth_spearman_rho` < 0 or `dag_consistency` == 0.

**v2 claim** (line 593-613): Proposes a gap-to-best formula:
```python
if higher_is_better:
    if best - value < 1e-12:
        return 0.0  # tied
    gap = best - value
    scale = max(abs(best), 1.0)
    return gap / scale
```

**Issue**: The formula is still **problematic** for several cases:

1. **depth_spearman_rho < 0**: If best=-0.10, value=-0.5, then gap=0.4, scale=0.1, result=4.0. This is bounded and monotonic — works.

2. **dag_consistency == 0 (all tied)**: If best=0, value=0, then gap=0, scale=1.0, result=0.0. Tied case handled. If best=0, value=0.1 (another engine scored 0.1 better), gap=0.1, scale=1.0 (max(0,1.0)=1.0), result=0.1. This works.

3. **BUT: scale = max(abs(best), 1.0) is wrong** when best is negative. Example: best=-0.5, value=-0.9, gap=0.4, scale=max(0.5, 1.0)=1.0, rel_best=0.4. But if best=0.5 (reversed), gap=0.4, scale=0.5, rel_best=0.8. Same gap, opposite semantics. The normalization should use **the spread of the metric on the graph**, not the best value itself.

4. **Deeper issue**: For bounded metrics like depth_spearman_rho (typically [-1, 1]) and dag_consistency ([0, 1]), a gap-to-best formula that normalizes by best value is fragile. Consider: if all engines score 0.0 on dag_consistency, the formula returns 0.0 for all (correct — all tied). But if best=1.0 and value=0.9, gap=0.1, scale=1.0, result=0.1. If best=0.1 and value=0.0, gap=0.1, scale=0.1, result=1.0. **Same absolute gap, vastly different relative scores** — this confuses family aggregation.

**Verdict**: PARTIAL. The formula is better than v1 but still has edge cases. The plan should use **rank-only aggregation for metrics bounded [0,1]** rather than forcing a gap formula. The comment at line 620-622 ("compute BOTH graph_rank and graph_rel_best and use rank as tiebreaker") hints at this but doesn't actually implement it.

---

### Q5: `validate_sync()` as telemetry not gate (PASS)

**v1 Codex critique** (CRIT #2): Plan says use `validate_sync()` as hard preflight gate, but it ignores .pt fallback. Contradicts fidelity's per-row HDF5→.pt fallback logic.

**v2 claim** (line 111-117, Q5): "Use `validate_sync()` as TELEMETRY only. Per-row, try HDF5 first, then `.pt` fallback, only fail at row level."

**Code check**: `fidelity_analysis.py:787-806` shows HDF5-first / .pt-fallback loader exists. v2 correctly identifies the pattern.

**Verdict**: PASS (v2 plan correctly demotes validate_sync to telemetry and keeps per-row fallback).

---

### Coverage denominator (Q4): graphs_in_family_available vs total (PASS)

**v1 Codex critique** (HIGH #9): Denominator oscillates with benchmark progress; should track both total and available.

**v2 claim** (line 625-634): Track both `graphs_in_family_total` (immutable) and `graphs_in_family_available` (with >= 1 completed engine); use available for ranking.

**Verdict**: PASS (correctly addresses the distinction).

---

### Stochastic metrics reproducibility (Q2: FIX-S NEW GROUP) (CONDITIONAL PASS)

**v1 Codex critique** (CRIT #3): `count_overlaps_detailed()` line 397 does `torch.randperm(m)[:200]` with no seed. `sampled_crossing_rate()` line 582 does `torch.randint()` with no seed. Without determinism, repeated QR pipeline runs produce different CSVs.

**v2 claim** (line 79-90, section FIX-S lines 434-448): "Add `seed: int | None` parameter to `count_overlaps_detailed`, `sampled_crossing_rate`, `count_crossings`. Seed from `hash((graph_name, engine_name, layout_seed))`."

**Critical issue found**:

1. **Python's hash() is NOT stable across processes** by default (PYTHONHASHSEED randomization). The plan uses `hash((graph_name, engine_name, layout_seed))` but Python's built-in hash is salted per process. This will fail when QR-CORE uses multiprocessing. **Should use hashlib.sha256() instead**.

2. **torch.Generator() is never plumbed through the stochastic functions**. The plan says "use `torch.Generator()` seeded from hash(...)" but the actual function signatures don't have `seed` parameters yet. When FIX-S is implemented, the functions MUST accept `seed: int | None` and call `torch.Generator(device='cpu').manual_seed(seed)` before the random operations.

3. **Quick check of `quick()` function** (lines 1166-1232): It calls `count_overlaps_detailed(pos, ns)` at line 1222 without a seed parameter. FIX-S must ADD that parameter and thread it through.

**Verdict**: CONDITIONAL PASS. The plan correctly identifies the problem and the fix direction, BUT the implementation has two blockers:
- **Hash stability bug** (must use hashlib, not hash())
- **Parameter threading not specified** (which caller provides the seed? QR-CORE's metric recomputation engine)

---

## NEW Issues Introduced by v2

### Issue N1: SEVERITY=CRITICAL | FIX-S: Python hash() instability in multiprocessing (BLOCKING)

**What's wrong**: Plan (line 530) says seed derives from `hash((graph_name, engine_name, layout_seed))`. Python's built-in `hash()` is randomized per process with PYTHONHASHSEED. When QR-CORE runs 8 worker processes, each will seed the torch.Generator differently, defeating the purpose.

**Example**: Worker 1 hashes ("graph_1", "engine_2", 42) → 12345678. Worker 2 hashes the same tuple → 87654321. Both run count_overlaps_detailed on the same data with different PRNG states → different overlap counts → non-deterministic CSVs.

**Fix**: Replace `hash()` with `int(hashlib.sha256((graph_name + engine_name + str(layout_seed)).encode()).hexdigest(), 16) % (2**31)` or similar stable hash.

**Cite**: Plan line 530, FIX-S section 434-448; `dagua/metrics.py:341-412` (count_overlaps_detailed), line 563-615 (sampled_crossing_rate), line 1554+ (count_crossings).

---

### Issue N2: SEVERITY=HIGH | FIX-S: Seeding parameter NOT threaded through quick()

**What's wrong**: Plan (line 516-522) says QR recomputation uses "quick" profile with `overlap_count` seeded via FIX-S. But `quick()` at `dagua/metrics.py:1166` has NO `seed` parameter, and calls `count_overlaps_detailed(pos, ns)` at line 1222 without one.

**When FIX-S lands**: `count_overlaps_detailed(pos, ns, seed=None)` will exist, but `quick()` will still call it without the seed. QR-CORE will need to either:
- Add `seed: int | None = None` to quick() signature and thread it down, OR
- Call `count_overlaps_detailed()` directly in QR-CORE instead of via quick()

The plan doesn't specify which path, leaving implementation ambiguous.

**Fix**: Clarify in FIX-S spec: "quick() MUST accept `seed` parameter and pass it to count_overlaps_detailed()."

**Cite**: Plan lines 516-522 (metric recomputation), 434-448 (FIX-S); dagua/metrics.py:1166 (quick signature), line 1222 (count_overlaps_detailed call).

---

### Issue N3: SEVERITY=HIGH | QR-IO module extraction: Hidden coupling via _skip_metrics

**What's wrong**: Plan (line 460-462) says extract `load_position_tensor(record_key, h5_handle, positions_dir, fallback_paths)` from `fidelity_analysis.py:755-807` to `dagua/eval/pipeline_io.py` as a "canonical HDF5-first / .pt-fallback loader, no new behavior."

**Read the actual function** (fidelity_analysis.py:755-806):
- Line 788-789: `positions_cache = getattr(load_layout, "_positions_cache", None)`
- Line 810: `skip_metrics = getattr(load_layout, "_skip_metrics", False)` — function checks for a module-level attribute hack.
- Lines 814-822: If skip_metrics is False, computes metrics via quick(). Otherwise returns empty dict.

This function **carries hidden coupling** to fidelity_analysis's private state. When extracted to pipeline_io.py, the _skip_metrics hack won't exist. QR-CORE needs to know whether to compute metrics or skip them.

**Reality**: The function is NOT cleanly decoupled. Either:
- Remove the _skip_metrics hack and always compute metrics in load_position_tensor(), OR
- Extract it with a `compute_metrics: bool` parameter, OR
- Keep the metric computation in QR-CORE's caller, not in the loader

The plan claims "no behavior change" but extraction will CHANGE behavior unless the hack is replicated.

**Fix**: Specify in QR-IO task: how does the extracted loader handle the metric computation branch?

**Cite**: Plan line 460-462; fidelity_analysis.py:755-822 (load_layout function), line 810 (_skip_metrics hack).

---

### Issue N4: SEVERITY=HIGH | graph_rel_best normalization asymmetry for lower-better metrics

**What's wrong**: For lower-better metrics (sampled_stress, overlap_count), plan (line 607-612) uses:
```python
if best > 1e-9:
    return (value - best) / best
else:
    scale = max(abs(value), 1.0)
    return (value - best) / scale
```

This breaks when best=0 (perfect score, e.g., overlap_count=0 on perfect layout):
- If best=0, value=5: scale = max(5, 1.0)=5, rel_best = 5/5 = 1.0 ✓
- If best=0, value=0: scale = 1.0, rel_best = 0/1.0 = 0.0 ✓ (tied)

But for **near-zero best values** (e.g., best=0.001, value=0.5 for sampled_stress):
- rel_best = (0.5 - 0.001) / 0.001 = 499. This is **unbounded and breaks family aggregation**.

The formula should cap the relative distance or use **inverse-gap** instead of gap-ratio for lower-better.

**Fix**: For lower-better metrics, use `gap / best` only when best > `1e-6 * median_value_on_graph`, else use absolute gap with a fixed scale factor.

**Cite**: Plan line 607-612; `dagua/metrics.py` sampled_stress (line 476+) and overlap_count (line 341+) definitions.

---

### Issue N5: SEVERITY=MEDIUM | Family mapping expansion incomplete

**What's wrong**: Plan (line 543-575) proposes an "EXPANDED canonical mapping" for graph families. Reading the manifest, the top tags are:
- wide-parallel (29), skip-light (15), diamond (14), skip-heavy (13), random (12), mixed-width (11), scale-free (11), nested-shallow (10), clustered (10), nested-deep (9), large-sparse (9), community (8), large-dense (7)

Plan's mapping (lines 543-575) preserves: wide-parallel→wide_parallel ✓, skip-light/heavy ✓, diamond ✓, nested-shallow/deep ✓, large-sparse/dense ✓, scale-free ✓, community ✓.

**But**: random, clustered, scale-free all map to generic buckets or don't have explicit preservation. Reading lines 570-574:
```
erdos-renyi|random -> random
clustered -> clustered
cyclic -> cyclic
else -> misc
```

These ARE preserved. So the mapping looks OK.

**Verdict**: PASS (mapping preserves most real tags).

---

### Issue N6: SEVERITY=MEDIUM | QR-REPORT depends on FID-A's new procrustes columns (dependency clarification needed)

**What's wrong**: Plan (line 809-810) says QR-REPORT depends on QR-CORE. But the fidelity markdown report (Cleanup2, line 378-393) surfaces "new procrustes columns: within_orig_rmsd_mean, between_rmsd_mean, procrustes_tost_pvalue_1x_bh, etc."

These columns come from FID-A (Group A, not FID-B or FID-CLEANUP). If QR-CORE also emits a Pareto front or insights that reference procrustes columns, then:
- QR-REPORT depends on QR-CORE (OK)
- But FID-CLEANUP (markdown rewrite) depends on FID-A (OK)
- Are there cross-pipeline dependencies? Does QR-REPORT reference fidelity columns?

**Reading the plan** (line 714-739): QR output is separate: family scorecards, insights based on quality metrics (sampled_stress, crossing_rate), Pareto fronts, etc. **No explicit reference to fidelity's procrustes columns**.

**Verdict**: PASS (no hidden coupling found, but should document explicitly in QR-REPORT spec).

---

## REMAINING v1 Issues That v2 MISSED or Didn't Close

### Issue R1: SEVERITY=HIGH | A5 verdict refactor must DELETE old logic (CRITICAL SEQUENCING)

**v1 Claude Explore critique** (C4): "Current verdict logic uses `wb_pval >= 0.05` to mark 'strong_equivalent' — this is ABSENCE OF EVIDENCE as EVIDENCE OF ABSENCE (backwards). A5 must explicitly DELETE these lines, not augment."

**v2 address** (line 199-229, A5): Plan DOES say DELETE (line 202-209) and REPLACE. This is correct. The fix is clear: lines 2164-2172 of fidelity_analysis.py have the backwards logic.

**BUT sequencing risk** (line 227-229): "A5 must land in the same Codex task as A1+A2+A3 — shipping any subset breaks verdict consistency." This is HIGH risk. If Codex implements A1-A4 and defers A5, verdicts will be mixed (some from TOST, some from old heuristic).

**Verdict**: PASS (plan acknowledges the risk, but risks need emphasized in implementation notes).

---

### Issue R2: SEVERITY=MEDIUM | B3 metric-add_metrics.py drift unresolved

**v1 Codex critique** (M14): `fidelity_add_metrics.py` at line 38 hardcodes the original 3 metrics. If B2 expands QUALITY_METRICS, this script drifts.

**v2 address** (line 271-279, B3): "Either update to import `QUALITY_METRICS` from fidelity_analysis or retire the script." Recommendation: update. Severity: MEDIUM.

**Status**: This is acknowledged but left as a manual fix for Codex. Should be explicit in the B3 task prompt.

**Verdict**: PASS (acknowledged).

---

### Issue R3: SEVERITY=MEDIUM | test_metric_seeding.py must cover seed=None case

**v1 Codex critique** (M13): Test strategy must cover "seed=None should still produce stochastic results."

**v2 address** (line 753, test strategy): Lists required fixtures but doesn't explicitly say "seed=None case". Under "test_metric_seeding.py" (implied by FIX-S), the test should assert that `count_overlaps_detailed(..., seed=None) != count_overlaps_detailed(..., seed=None)` on same data (stochastic behavior preserved).

**Verdict**: PASS (fixture coverage listed, seed=None case implied).

---

### Issue R4: SEVERITY=LOW | Citation error: PAIRWISE_SAMPLE_SIZE constant name

**v1 Claude Explore** and **Codex**: Pointed out that v1 plan cites `MAX_PROCRUSTES_SEEDS_PER_SIDE` which doesn't exist; actual is `PAIRWISE_SAMPLE_SIZE`.

**v2 address** (line 136-143, citations table): Corrects this to `PAIRWISE_SAMPLE_SIZE (line 58)`.

**Verdict**: PASS (fixed).

---

## Validations of Specific v2 Claims (Sections 2-10 from task brief)

### Section 2: Is proposed FIX-S seeding strategy correct?

**Part A: Python hash() stability** — FAIL. Hash is process-randomized; must use hashlib.

**Part B: torch.Generator plumbing** — FAIL. quick() and stochastic functions don't have seed parameters yet; threading unclear.

**Part C: Other stochastic metrics missed** — QUALIFIED. Plan adds seed to count_overlaps, sampled_crossing_rate, count_crossings. But `neighborhood_preservation()` at line 618-642 ALSO uses `torch.randperm()` at line 634 and `np.random.randint()` elsewhere. If QR includes neighborhood metrics, they're missed.

**Cite**: Plan line 434-448 (FIX-S); dagua/metrics.py line 341, 563, 1554, 618.

---

### Section 3: Is QR-IO shared module API clean?

**Part A: load_position_tensor extraction** — PARTIAL. Function exists and can be extracted, but _skip_metrics hack must be handled. Plan doesn't specify.

**Part B: Behavior change claim** — QUESTIONABLE. Plan says "no behavior change" but extraction requires deciding: does extracted loader compute metrics or not?

**Part C: Module location** — APPROPRIATE. dagua/eval/pipeline_io.py (public engine API) is correct, not scripts/_pipeline_io.py.

**Cite**: Plan line 460-468; fidelity_analysis.py line 755-822.

---

### Section 4: graph_rel_best math hand-derivation (specific cases)

I'll re-derive each case using the plan's formula (line 593-613):

**Case 1**: `depth_spearman_rho`, best=0.95, value=-0.10, higher_is_better=True
- gap = 0.95 - (-0.10) = 1.05
- scale = max(abs(0.95), 1.0) = 0.95 (WRONG, should be 1.0)
- rel_best = 1.05 / 0.95 = 1.105 ✓ (engine is much worse; reasonable)

**Case 2**: `depth_spearman_rho`, best=0.95, value=0.0
- gap = 0.95, scale = 0.95, rel_best = 1.0 (tied in some sense; reasonable)

**Case 3**: `dag_consistency`, best=1.0, value=0.0
- gap = 1.0, scale = 1.0, rel_best = 1.0 (reasonable)

**Case 4**: `dag_consistency`, best=0.0, value=0.0 (all tied)
- gap = 0, scale = max(0, 1.0) = 1.0, rel_best = 0.0 ✓ (all tied)

**Case 5**: `overlap_count`, best=0, value=5, higher_is_better=False
- Per lower-better formula: best is 0 > 1e-9? No.
- scale = max(abs(5), 1.0) = 5, rel_best = (5-0)/5 = 1.0 ✓

**Case 6**: `overlap_count`, best=0, value=0 (all tied)
- rel_best = 0.0 ✓

**Case 7**: `sampled_stress`, best=0.001, value=0.5
- best > 1e-9? Yes. rel_best = (0.5 - 0.001) / 0.001 = 499 ✗ (UNBOUNDED, breaks aggregation)

**Result**: Formula produces nonsense for case 7 (near-zero best on lower-better). **FAIL**.

---

### Section 5: Coverage denominator distinction (manifest vs execution)

**Question**: Does plan handle eligibility vs completion distinction?

**Answer**: Partially. Plan (line 630-634) distinguishes:
- `graphs_in_family_total` (immutable, from manifest)
- `graphs_in_family_available` (at least 1 engine completed)

But does NOT distinguish whether an engine was SCHEDULED on a graph (manifest variant filtering) vs FAILED TO COMPLETE. Reading `run_benchmark.py`, variant filtering happens at the manifest level, not per-engine scheduling. So the question is moot. **PASS**.

**Cite**: Plan line 624-640; manifest.json structure.

---

### Section 6: Per-metric insight thresholds grounded?

Plan (line 680-701) specifies:
- `sampled_stress`: 15% relative, 30% premium, 1.25x runtime
- `edge_length_cv`: same
- `edge_straightness_mean_deg`: 3° absolute, 5° premium
- `overlap_count`: 5 absolute, 20 premium

**Typical ranges** (from metrics.py):
- `sampled_stress`: varies by graph size and edge routing, typically 0.01–10.0 (huge range)
- `edge_straightness_mean_deg`: 0–90 degrees, typically 5–30 degrees on well-laid graphs
- `overlap_count`: 0 to N*(N-1)/2, typically 0 on small/medium graphs

**Assessment**: The 3° threshold for edge_straightness is grounded (typical range 5–30, so 3° is ~10% baseline). The sampled_stress threshold is NOT grounded (15% of a metric with 1000x range is not uniform meaning). **PARTIAL FAIL**.

**Fix**: Compute per-family percentiles (e.g., p10, p50, p90 of `sampled_stress`) at analysis time, then use relative thresholds vs those percentiles rather than vs best value.

**Cite**: Plan line 680-701; dagua/metrics.py line 476+ (sampled_stress), 415+ (aspect_ratio), 705+ (edge_straightness).

---

### Section 7: Cache key strategy robustness

**Plan (line 510-513)**: "Cache key includes the metric function's source hash so that changing the metric implementation invalidates the cache."

**Implementation needed**: How? The plan doesn't say. Should be:
1. Hash the .py file content of dagua/metrics.py
2. Include in cache key

**Risk**: If a helper function (e.g., _ensure_cpu, segments_intersect) changes, the cache won't invalidate. Transitive dependencies are NOT covered.

**Verdict**: QUALIFIED. Concept is sound but implementation details missing.

**Cite**: Plan line 510-513; QR-CORE spec (not yet written).

---

### Section 8: QR seed budget reality

Manifest shows: 105 graphs, 60 seeds per stochastic engine. Assume 235 engine variants (from Codex notes), ~40 stochastic. Expected records: 105 * 235 * 60 = 1,477,500 (but manifest says 1,267,245 total scope, so actual is less due to variants filtering).

Plan (line 533-536): "5-10 hours overnight on 8 cores" for metric recomputation. With multiprocessing overhead (pickling positions), this is feasible IF:
1. Positions are loaded from HDF5 (binary, fast)
2. Metrics are cached after first run
3. Sampled metrics (stress, crossing_rate) don't dominate

But sampled_crossing_rate with 1M samples per call on a 500-edge graph is expensive. **Realistic budget is 10–20 hours, not 5–10**.

**Verdict**: PARTIAL. Budget is optimistic but workable with caching.

**Cite**: Plan line 533-536, 520-528; manifest.json; dagua/metrics.py line 563–615.

---

### Section 9: Phase 2 dispatch sequence and FID-CLEANUP dependency

**Plan (line 794-796)**: "Ship FID-S + FID-A in parallel (one Codex each), then FID-B + FID-C + FID-D + FIE-E + FID-CLEANUP in parallel."

**Dependency check**:
- FID-CLEANUP (Cleanup1-6, line 371-420) includes Cleanup2 (markdown rewrite, line 378-393).
- Cleanup2 surfaces "new procrustes columns: within_orig_rmsd_mean, procrustes_tost_pvalue_1x_bh, etc." (line 388-389).
- These columns come from FID-A (Groups A1-A5, line 154-230).
- Therefore: **FID-CLEANUP depends on FID-A**, not just FID-B.

**Current plan**: FID-A and FID-CLEANUP are in the second wave together (line 795). This is OK.

**But**: Does FID-CLEANUP also depend on new metrics from FID-B (B2 metric expansion)? Reading line 388-389, no mention of B2 columns in the markdown report. The report surfaces procrustes (FID-A) and failures (FID-E), not B2 metrics. **No hidden coupling found**.

**Verdict**: PASS (dependencies correctly sequenced).

**Cite**: Plan line 605–813.

---

### Section 10: Tests vs implementation balance

**test_metric_seeding.py**: Plan says "reproducibility: count_overlaps_detailed(pos, ..., seed=42) == count_overlaps_detailed(pos, ..., seed=42)." Needs to cover seed=None case (stochastic unchanged). **ADEQUATE**.

**test_fidelity_procrustes.py**: Needs "fixture where within-original distribution differs from pooled within-distribution." This requires a custom graph and layout pair where pairwise_orig RMSD < pairwise_reimpl RMSD. Plan doesn't specify the fixture. **QUALIFIED** (needs design work).

**test_pipeline_io.py**: Plan says "cover HDF5/pt fallback both ways." This means:
- Test with HDF5 only (should work)
- Test with missing HDF5, .pt fallback (should work)
- Test with both missing (should fail or return None)

**ADEQUATE** (well-specified).

**Verdict**: PASS (test coverage sound, fixtures need design).

**Cite**: Plan line 753–758.

---

## Polish Suggestions (Low-stakes)

1. **Line 145-146**: Prefix critical findings with "CF1, CF2" instead of "F1, F2" to avoid collision with Group F. Plan does this correctly at line 49 onward. ✓

2. **Line 270**: Note that B2 is gated on FIX-S landing. Already stated. ✓

3. **Line 530**: Explicitly say "use hashlib.sha256, not Python hash()" to avoid process-randomization issues.

4. **Line 624-640**: Add a note that `graphs_in_family_available` is recomputed on each run (not cached), so including new engines/graphs will shift the denominator. This is correct but worth documenting.

5. **Section "Test strategy" (line 753-758)**: Expand with one sentence per fixture type listing expected content (e.g., "test_metric_seeding.py: seed reproducibility, seed=None stochasticity, edge case empty graph").

---

## Verdict

**SHIP_WITH_FIXES**

The plan is well-structured and addresses most v1 critiques. But three CRITICAL blockers must be fixed before implementation:

1. **FIX-S hash stability**: Replace `hash()` with `hashlib.sha256()` or equivalent stable hash for cross-process determinism.

2. **FIX-S parameter threading**: Explicitly specify that quick() and all stochastic metric functions accept `seed: int | None`, and clarify which caller (QR-CORE) provides the seed.

3. **QR-IO extraction clarity**: Specify whether `load_position_tensor()` computes metrics or if QR-CORE computes them separately. Resolve the _skip_metrics hack.

Additionally:
- **graph_rel_best normalization** has edge cases (unbounded for near-zero best on lower-better metrics) that should be addressed in QR-CORE spec.
- **Test fixtures** for procrustes and QR need explicit design (especially the "within-orig differs from pooled" case).
- **Insight thresholds** are not grounded in metric ranges; should use per-family percentiles.

Otherwise, the architecture is sound, the sequencing is correct, and the fixes are well-targeted.

---

## Executive Summary

v2 successfully folds in the v1 adversarial reviews and resolves 8 user open questions. It correctly identifies F1 (pooled within-RMSD), Q2-Q5 (stochastic metrics, graph_rel_best, coverage denominator, validate_sync) as blocking fixes, and removes 3 stale items (BH NaN, validation, risk note) that are already landed. The Phase 2 sequencing is tightened into 2 dependency chains with clearer task boundaries.

However, three NEW issues were introduced by v2:

1. **FIX-S uses Python's hash() which is not stable across processes** — will fail under multiprocessing in QR-CORE. Must switch to hashlib.
2. **FIX-S parameter threading is incomplete** — quick() signature and caller responsibility not specified.
3. **QR-IO extraction has hidden coupling via _skip_metrics hack** — extraction will change behavior unless clarified.

Additionally, **graph_rel_best formula still has edge cases** (unbounded on near-zero best values) that weren't caught by the mathematical fix attempt. The plan should either add clamping or fall back to rank-only for problematic metrics.

The plan is architecturally sound and ready for implementation IF these four issues are addressed. Estimated fix time: 2–3 Codex review passes to clarify specs before implementation begins.
