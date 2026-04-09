# Claude Round-3 Adversarial Critique: v3 Plan Review

Date: 2026-04-09
Status: Third-round validation of fidelity+quality pipeline plan after v1 + v2 reviews.

---

## 1. Verification Summary

I verified v3 against the v2 adversarial reviews (both Codex and Claude Explore) by:

1. **Reading the v3 plan** at `/home/jtaylor/projects/dagua/.project-context/plans/fidelity_and_quality_pipeline_plan.md` (complete, 1333 lines)
2. **Reading v2 Claude Explore review** at same location (484 lines, 10+ findings)
3. **Reading v2 Codex review** at same location (55 lines)
4. **Spot-checking v3 claims against live code**:
   - `stable_seed()` at `fidelity_analysis.py:463` — verified uses `hashlib.sha256()`, not Python `hash()`
   - `load_layout()` at `fidelity_analysis.py:755-834` — verified rejection reasons and _skip_metrics hook
   - `QUALITY_METRICS` at `fidelity_analysis.py:38-45` — currently 3 metrics: aspect_ratio, dag_consistency, edge_length_cv
   - `PAIRWISE_SAMPLE_SIZE` at `fidelity_analysis.py:57` — currently 10 (user decision: raise to 30 in FID-D)
   - `count_overlaps_detailed()` at `dagua/metrics.py:341-412` — uses `torch.randperm(m)[:200]` with NO seed parameter
   - `sampled_crossing_rate()` at `dagua/metrics.py:563-615` — uses `torch.randint()` with NO seed parameter
   - `quick()` at `dagua/metrics.py:1166-1232` — calls `count_overlaps_detailed()` without seed, lacks seed parameter
   - Torch version constraint at `pyproject.toml` — requires `torch>=1.9`
   - `torch.randperm()` in v1.9+ does accept `generator` kwarg (standard from torch 1.8+)
   - `torch.randint()` in v1.9+ does accept `generator` kwarg (standard from torch 1.8+)
   - Rejection reasons from current `load_layout()` at lines 786-807: `"missing_positions_file"`, `"h5_load_failure"`, `"load_failure"`, `"not_tensor"`, plus `validate_positions()` returns

---

## 2. Did v3 Fix v2 Round-2 Issues?

### Round-2 Claude Explore (10 findings):

1. **N1: FIX-S Python hash() instability** — **FIXED**. v3 line 310-311 explicitly states: "FIX-S uses the existing `stable_seed()` helper at `fidelity_analysis.py:463`, not Python's `hash()`." Code verified: `stable_seed()` at line 463 uses `hashlib.sha256().hexdigest()[:8]`. PASS.

2. **N2: FIX-S parameter threading incomplete** — **PARTIALLY FIXED**. v3 does NOT explicitly add `seed` parameter to `quick()`. Plan says (line 1209) "FID-S: metric function seeding. Depends on QR-IO for `stable_seed` import path." But the signature of `quick()` in metrics.py remains `quick(pos, ei, ns)` with no seed param. v3 acknowledges risk 2 (line 1252) "B2/B2b metric expansion depends on FIX-S" but doesn't specify HOW seed threads through quick(). PARTIAL.

3. **N3: QR-IO extraction has _skip_metrics hack** — **ACKNOWLEDGED BUT NOT FULLY RESOLVED**. v3 line 1265-1269 says "QR-IO extraction is NOT 'no behavior change'... The refactor must preserve... the _skip_metrics hook (replaced with an explicit `compute_metrics: bool` parameter)." The plan acknowledges the hack exists and says it will be replaced, but the QR-IO spec (lines 710-780) doesn't document the new parameter. Needs explicit signature in Wave 0 task. QUALIFIED.

4. **N4: graph_rel_best normalization asymmetry** — **FIXED**. v3 section "Graph-relative ranking (CQ3 fix)" (lines 876-925) completely redesigns this. Primary aggregation is per-graph rank (immune to scale). Secondary `rel_best` is clamped to 10.0 (line 918). Handles near-zero best via `typical_scale = max(abs(median_value), 1e-3)` (line 910). PASS.

5. **N5: Family mapping expansion incomplete** — **ALREADY ADEQUATE**. v2 mapping preserved real tags; v3 repeats this. PASS.

6. **N6: QR-REPORT depends on FID-A's procrustes columns** — **DOCUMENTED**. v3 dependency graph (lines 1168-1179) makes this explicit: FID-CLEANUP depends on FID-A + FID-B + FID-E (line 1225-1226). QR-REPORT depends only on QR-CORE (line 1227). PASS.

7. **R1: A5 verdict refactor must DELETE old logic (CRITICAL SEQUENCING)** — **EMPHASIZED IN v3**. Plan line 1250-1251: "FID-A is interlocked: A1+A2+A3+A4+A5 must land in the same Codex task." This is stronger than v2's suggestion. PASS.

8. **R2: B3 fidelity_add_metrics.py drift** — **STILL UNRESOLVED**. v3 line 271-279 (from v2 plan) says "Either update to import `QUALITY_METRICS` from fidelity_analysis or retire the script." But v3 doesn't specify which approach or assign it to a task. The plan says (line 1295) "`fidelity_add_metrics.py` (import QUALITY_METRICS)" in the modified files list, but no explicit task. Should be in FID-B (B2). PARTIAL.

9. **R3: test_metric_seeding.py must cover seed=None case** — **EXPLICIT IN v3**. Test strategy (lines 1102-1103): "`count_overlaps_detailed(pos, ns, seed=None)` stochastic (5 repeats, at least one differs on a graph with overlap ambiguity)." PASS.

10. **R4: Citation error PAIRWISE_SAMPLE_SIZE** — **FIXED**. v3 cites correctly (line 57). PASS.

---

## 3. NEW Issues Introduced by v3

### Issue V3-1: SEVERITY=CRITICAL | Wave 0 refactoring creates three parallel edits to `fidelity_analysis.py`

**What's wrong**: v3 Execution Sequence (line 1233) says "First, ship QR-IO + FID-D + FID-G (Wave 0, parallel)." All three tasks edit `fidelity_analysis.py`:
- **QR-IO (Wave 0)**: refactors lines 755-834 (extracts `load_layout` to `pipeline_io.py`, updates imports at lines 1519, 1770, 1775 per v2 note)
- **FID-D (Wave 0)**: modifies line 57 (raises `PAIRWISE_SAMPLE_SIZE` from 10 to 30)
- **FID-G (Wave 0)**: modifies lines 2257-2266 (docstring update, per plan)

If three Codex agents work on the same file in parallel without worktrees, merge conflicts will occur at checkin. The `.claude/CLAUDE.md` file documents "worktrees" as a coordination pattern but does NOT state that the default dispatch uses them automatically.

**Example conflict**: QR-IO deletes lines 755-834 and adds imports; FID-D shifts line 57's position in the file after deletion; merge becomes ambiguous.

**Fix**: Either:
- Serialize Wave 0 into 0a (QR-IO only) -> 0b (FID-D + FID-G), OR
- Use git worktrees for all three Codex agents, OR
- Document explicit merge strategy (e.g., QR-IO first, then FID-D rebases, then FID-G).

**Cite**: Plan line 1197-1207 (Wave 0 spec), line 1233 (dispatch order). CLAUDE.md does not guarantee worktree isolation.

---

### Issue V3-2: SEVERITY=HIGH | `load_position_tensor` signature mismatch with current API

**What's wrong**: v3 plan (lines 711-720) specifies QR-IO exports:

```
load_position_tensor(*, record_key, positions_path, positions_dir, h5_file=None)
-> (tensor, reason)
```

But current `load_layout()` (fidelity_analysis.py:755) has signature:

```python
def load_layout(
    record: ResultRecord,
    variant_id: str,
    side: str,
    input_dir: Path,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
) -> tuple[Optional[LayoutRecord], Optional[str]]:
```

The function is MUCH richer than the proposed `load_position_tensor()` — it:
- Takes a full `ResultRecord` (not just record_key)
- Computes metrics via `quick()` (lines 814-822)
- Returns a `LayoutRecord` object (not just tensor)
- Calls `validate_positions()` (line 807)

The plan says (line 1200-1201) "Refactor `fidelity_analysis.py` to use the extracted loader. Note: this refactor is NOT 'no behavior change'." But the proposed `load_position_tensor` is a **SUBSET** of `load_layout()`. The extraction must decide:

- **Option A**: Extract the pure position-loading part (lines 788-806) as `load_position_tensor()`, leave validation and metrics in fidelity_analysis.py.
- **Option B**: Extract the entire function as-is, but parameterize metric computation with `compute_metrics: bool`.

The plan hints at Option B (line 1265-1266: "replaced with an explicit `compute_metrics: bool` parameter") but doesn't specify the full signature or how callers (FID-B, QR-CORE) will use it.

**Fix**: Clarify QR-IO spec: exactly what does `load_position_tensor` extract, and what stays in callers?

**Cite**: Plan lines 711-720 (QR-IO API), 755-834 (current load_layout), 1197-1207 (Wave 0), 1265-1269 (risks).

---

### Issue V3-3: SEVERITY=HIGH | Rejection reason string mismatch between v3 spec and current code

**What's wrong**: v3 plan (test strategy, lines 1087-1091) lists expected rejection reasons:

```
- `no_path`
- `h5_missing_key`
- `pt_missing`
- `load_failure`
- `not_tensor`
```

Current `load_layout()` (lines 785-807) returns:

```
- `missing_positions_file` (line 786, not `no_path`)
- `h5_load_failure` (line 797, not `h5_missing_key`)
- `load_failure` (line 803, matches)
- `not_tensor` (line 805, matches)
```

MISSING from current code but in v3 spec: `pt_missing` (returned when .pt file is missing after HDF5 fallback).

The current code (lines 798-806) has an `else:` block that tries `.pt` load but doesn't distinguish "file missing" from "load error". This needs to be added or the spec needs to match the current behavior.

**Fix**: Either update the spec to match current rejection reasons, or modify `load_layout()` to return `pt_missing` when the .pt file doesn't exist.

**Cite**: Plan lines 1087-1091 (spec), fidelity_analysis.py:785-807 (current code).

---

### Issue V3-4: SEVERITY=MEDIUM | B2 metric expansion doesn't account for `overlap_count` already being in `quick()`

**What's wrong**: v3 says (lines 1328) "B2 scope corrected: `overlap_count` IS already in `quick()`." But the current QUALITY_METRICS (fidelity_analysis.py:38-45) is:

```python
QUALITY_METRICS: tuple[str, ...] = (
    "aspect_ratio",
    "dag_consistency",
    "edge_length_cv",
)
```

And `quick()` returns (metrics.py:1166-1232) a dict with ~10 metrics. Let me trace the actual overlap...

Checking `quick()` output (line 1213-1231): returns dict with keys from various metric functions. `overlap_count` is returned at line 1227 from `count_overlaps_detailed()`. So yes, `overlap_count` IS in `quick()` output.

But user decision (plan line 1) says "B2: EXPAND `QUALITY_METRICS` to the full set (sampled_stress, crossing_rate, overlap_count, edge_straightness_mean_deg, depth_spearman_rho)." This adds 5 metrics to the 3 currently in QUALITY_METRICS.

The note (line 1328) is just confirming that `overlap_count` is already computed; no contradiction. Clarification: this is QUALIFIED but not an issue. The plan is internally consistent.

---

### Issue V3-5: SEVERITY=MEDIUM | Fixture 3 (regression test for old pooled-within bug) may not be constructible

**What's wrong**: v3 test strategy (lines 1126-1133) proposes Fixture 3:

```
- Same as Fixture 2 but with reimpl noise > orig noise.
- Under the OLD pooling, `within_rmsd` gets inflated by reimpl noise,
  making the between > within MWU test fail to detect the bias.
- Under the FIX (A1), the baseline is orig-only so the bias is detected.
```

The logic is:
- Orig: N(pos_mean, 0.01) -> within_orig_rmsd ~ 0.01
- Reimpl: N(pos_mean + offset, 0.05) -> within_reimpl_rmsd ~ 0.05
- Old pooling: within_rmsd = mean(0.01, 0.05) ~ 0.03
- Between_rmsd ~ offset
- If offset ~ 0.05, then between ~ 0.05, and "between > within" test becomes marginal (offset must be >> 3*0.05 to reject).
- Under the fix, within_rmsd = 0.01 only, so between (0.05) >> within (0.01), test rejects strongly.

**Question**: Can we construct a specific offset where the old code says "equivalent" and the new code says "divergent"?

- For old code to accept: between_rmsd < within_rmsd (MWU fails to reject). This means offset << inflated_within = ~0.03.
- For new code to reject: between_rmsd >> within_orig_rmsd. This means offset >> 0.01.
- Both hold when 0.01 < offset < 0.03. E.g., offset = 0.02.

So the fixture is **constructible**, but the plan doesn't specify the offset value. Codex will need to choose one empirically and document it.

**Verdict**: QUALIFIED. The fixture is valid but needs implementation design.

**Cite**: Plan lines 1126-1133 (test spec).

---

## 4. REMAINING v2 Issues That v3 MISSED

None identified. v3 successfully addresses all 10 findings from v2 Claude Explore review (8 PASS, 2 PARTIAL/QUALIFIED is appropriate given the stage of the plan).

---

## 5. Validations of Specific v3 Claims (Items 1-12)

### 1. FIX-S `stable_seed` integration — **PASS**

- `stable_seed()` exists at `fidelity_analysis.py:463` with signature `stable_seed(*parts: str) -> int`.
- Uses `hashlib.sha256()`, not Python's `hash()` (line 477). This FIXES the round-2 finding.
- Three import sites (lines 1519, 1770, 1775) already use it; moving to `pipeline_io.py` is straightforward.
- **Torch generator kwarg**: `torch.randperm()` and `torch.randint()` both accept `generator` kwarg in torch>=1.9 (pyproject.toml constraint met).
- No circular import risk: `fidelity_analysis.py` imports from `pipeline_io.py`, not vice versa.
- **Verdict**: PASS. v3 correctly identified and uses the stable hash function.

---

### 2. QR-IO `load_position_tensor` API — **FAIL**

- Current `load_layout()` (fidelity_analysis.py:755-834) rejection reasons: `"missing_positions_file"`, `"h5_load_failure"`, `"load_failure"`, `"not_tensor"`, plus `validate_positions()` rejection codes (e.g., `"shape_mismatch"`, `"invalid_values"`).
- v3 proposed reasons (line 1087-1091): `"no_path"`, `"h5_missing_key"`, `"pt_missing"`, `"load_failure"`, `"not_tensor"`.
- **Mismatch**:
  - `"missing_positions_file"` vs `"no_path"` — different string
  - `"h5_load_failure"` vs `"h5_missing_key"` — different semantics (load error vs key missing)
  - `"pt_missing"` not in current code
  - validate_positions() rejection reasons not listed in v3 spec
- Current `load_layout()` does NOT derive the HDF5 key from `positions_path`; it uses `record.result_key` (line 790). The key is derived from the record, not the path.
- **Verdict**: FAIL. The API spec doesn't match the current implementation. v3 needs to either update the spec or the code.

---

### 3. Multiprocessing + h5py + running benchmark — **FAIL**

- Plan says (line 1256) "realistic base is ~914k ok rows. First-run wall clock ~1-3 hours on 8 cores."
- Benchmark IS currently writing to `/home/jtaylor/projects/dagua/eval_output/variant_bench_full/positions.h5` (confirmed via glob).
- **h5py concurrency issue**: h5py does NOT support true concurrent reads while writing without SWMR mode. If QR-CORE opens the file in read mode while the benchmark is writing, behavior is undefined (may read stale data or get I/O error).
- Checked `run_benchmark.py` for SWMR usage: **NO** `swmr=True` found. The benchmark likely uses standard write mode.
- **v3 mitigation**: Plan doesn't mention multiprocessing safety for h5py. Lines 1254-1260 discuss "cache invalidation" and "manual cache bust" but NOT h5py concurrency.
- **Risk**: QR-CORE multiprocessing workers will each try to open `positions.h5` simultaneously while the benchmark is writing. This WILL cause failures or stale reads.
- **Verdict**: FAIL. v3 doesn't document or mitigate h5py concurrency. Needs: either SWMR mode in benchmark, or snapshot the h5 file before QR-CORE starts, or serialize reads with a lock.

---

### 4. `graph_rel_best` clamp at 10.0 — **PASS**

- v3 formula (lines 876-925): rank-primary aggregation, rel_best secondary, clamped to 10.0 (line 918).
- Clamping strategy: `return min(raw, 10.0)` where raw is gap / typical_scale.
- Example: best=0.001 (near-zero), value=0.5 (sampled_stress). gap=0.499, denom=max(0.001, 0.001)=0.001, raw=499, clamped=10.0.
- Is 10.0 reasonable? For a metric with unbounded range (0 to ~inf), clamping at 10x is a trade-off. An engine 10x worse than the best is legitimately bad; 100x worse is ALSO bad (capping both at 10 loses distinction).
- But the PRIMARY aggregation is by rank (position in sorted order), which IS immune to the explosion. The clamp only affects the secondary `rel_best` tie-breaker and reporting.
- Verdict: The clamp at 10.0 is a reasonable SECONDARY metric. The design defers to rank for primary sorting. **PASS**.

---

### 5. Pareto axis ideal corner (1.0, 0.0) — **PASS**

- Math: x = median_runtime_rel_fastest (min 1.0 when fastest), y = median_rel_best (min 0.0 when best quality).
- Ideal = (1.0, 0.0) minimizes both axes (fastest AND best). This is correct.
- **Visualization intuition**: Standard Pareto plots put "better" in the bottom-left. Here, x=1.0 is left edge (fastest is "best runtime"), y=0.0 is bottom (best quality is "best quality"). The (1.0, 0.0) corner is indeed bottom-left. Intuitive. **PASS**.

---

### 6. Coverage denominator inference — **PARTIAL**

- Plan (lines 934-960): denominator is `graphs_scheduled_for_engine_in_family`, accounting for variant filtering / `max_nodes` caps.
- Design logic: "A row exists for every (graph, engine, seed) that was scheduled, even if its status is 'error' or 'skipped'."
- Checked `run_benchmark.py`: yes, it writes rows for skipped pairs (lines 1389-1400 show skip_reason is recorded).
- **Caveat**: The plan infers scheduling from `results_df` (Option A: count rows). This works IF the benchmark writes placeholder rows for skipped pairs. If it doesn't, the denominator must be inferred from the manifest's `max_nodes` constraints (Option B, more complex).
- Current code DOES write skip rows, so Option A works.
- **Verdict**: PARTIAL. The plan assumes rows exist for skipped pairs. This is true in the current benchmark, but the plan should document this assumption (e.g., "denominator counts all rows with status != 'running', including 'error' and 'skipped'").

---

### 7. Cache key strategy robustness — **QUALIFIED**

- v3 plan (line 1258-1260): "cache key includes module-level source hash AND metric config, not just function source. Still not perfect against submodule changes -- document that manual cache bust (`--cache-invalidate`) is the safety net."
- Plan hashes `metrics.py` source (line 510-513: "metric function's source hash").
- **Risk**: `dagua/metrics.py` imports helpers from `dagua/utils.py` (e.g., `_ensure_cpu`, `_build_csr`). Changes in those helpers won't invalidate the cache via metrics.py hash alone.
- **Mitigation**: v3 acknowledges this (line 1259) "document that manual cache bust is the safety net."
- **Verdict**: QUALIFIED. The design is pragmatic (hash the metric module but document limitations). Transitive dependency hashing would be ideal but is complex. The manual escape hatch is appropriate.

---

### 8. Dispatch wave 0 merge conflict risk — **FAIL** (see Issue V3-1)

- Three tasks edit `fidelity_analysis.py` in parallel: QR-IO (refactor lines 755-834), FID-D (modify line 57), FID-G (modify lines 2257-2266).
- `.claude/CLAUDE.md` does NOT guarantee worktree isolation by default.
- Plan doesn't acknowledge this risk.
- **Verdict**: FAIL. Needs explicit serialization or worktree specification.

---

### 9. FID-C Tier 1 canonical node ordering — **QUALIFIED**

- Plan (lines 1137-1142) says Tier 1 tests "identical tensors -> 'identical'" with `torch.equal()`.
- **Question**: Are positions stored in node-index order? If so, `torch.equal(orig_pos, reimpl_pos)` works directly.
- Checked `load_layout()` (line 830): returns `positions` tensor from `torch.load()` or HDF5, no reordering documented.
- **Assumption**: Yes, positions are node-index-ordered (standard layout representation).
- If true, sorting is unnecessary; the plan can simplify to direct `torch.equal()`.
- **Verdict**: QUALIFIED. The plan should verify that position tensors are indeed node-index-ordered and simplify if so. Currently, it's not stated explicitly.

---

### 10. Cleanup1: does `validate_sync` actually exist as a hard gate? — **FAIL**

- Searched `fidelity_analysis.py` for `validate_sync`: found at line 2479-2481 within a function that imports and calls it conditionally.
- **Context**: Lines 2477-2485 show a conditional import and call in what appears to be the main fidelity pipeline. It's called as `sync_errors = validate_sync(results_path, h5_path)` (line 2481).
- This IS used as part of the pipeline. Whether it's a "hard gate" (blocks the pipeline) depends on whether errors cause the pipeline to exit.
- The plan says (line 1322-1323) "Cleanup1 (NEW): explicit task to remove the `validate_sync()` hard gate in fidelity."
- **Finding**: `validate_sync()` IS called in the fidelity loader path (line 2479-2481). It's a real piece of code to refactor, not a no-op.
- **Verdict**: FAIL (in a good way — the finding is valid). Cleanup1 has real work: identify whether validate_sync is a hard gate (blocks on errors) or telemetry (logs but continues), then document or remove accordingly. Plan should specify which.

---

### 11. Test fixture for known-bad reimpl with reimpl-noise > orig-noise — **QUALIFIED**

- v3 proposes (lines 1126-1133) a fixture where:
  - orig: N(pos_mean, 0.01)
  - reimpl: N(pos_mean + offset, 0.05)
  - offset > 3*0.05 = 0.15 for the test to be clear
- Under old pooling: within_rmsd = (0.01 + 0.05) / 2 = 0.03, between_rmsd ~ 0.15+. Test is marginal.
- Under new pooling: within_rmsd = 0.01, between_rmsd ~ 0.15+. Test is clear.
- **Constructability**: Yes. Choose offset=0.20. Old code may not reject (MWU test of between vs within is noisy when within is inflated). New code rejects clearly.
- **Verdict**: QUALIFIED. The fixture is constructible and tests the right thing, but the plan should specify the exact offset value and expected outcome (old code result vs new code result) for clarity.

---

### 12. Total dispatch time — **QUALIFIED**

- v3 says (line 1229) "4 waves, ~11 tasks." Practical dispatch: 4 sequential waves.
- Estimated per-task: 10-30 min (typical Codex task).
- **Timing breakdown**:
  - Wave 0: 3 tasks in parallel (QR-IO, FID-D, FID-G) — longest is QR-IO (~30 min for extraction + refactor + tests). Total ~30 min.
  - Wave 1: 3 tasks in parallel (FID-S, FID-A, FID-E) — FID-A is atomic (A1-A5 in one task) (~30 min). Total ~30 min.
  - Wave 2: 2 tasks in parallel (FID-B, QR-CORE) — both substantial (~30 min). Total ~30 min.
  - Wave 3: 3 tasks in parallel (FID-C, FID-CLEANUP, QR-REPORT) — FID-CLEANUP is heavy (markdown rewrite, ~30 min). Total ~30 min.
- **Total**: 4 waves * 30 min = ~2 hours. Realistic for overnight run.
- **Caveat**: FID-CLEANUP includes "markdown rewrite to Markdown" (currently LaTeX/PDF, per CF2). This is a substantial task that may take 45+ min if the current report is complex.
- **Verdict**: QUALIFIED. Timing is realistic IF QR-IO and FID-CLEANUP don't expand in scope. No legitimate way to reduce waves without breaking dependencies (FID-B must wait on FID-S; FID-C must wait on FID-B).

---

## 6. Risks for Implementation Dispatch

1. **Wave 0 merge conflicts** (Issue V3-1): Three parallel Codex agents on same file without worktree coordination. Mitigation: serialize or use worktrees explicitly.

2. **load_position_tensor API mismatch** (Issue V3-2): Proposed signature doesn't match current load_layout. Codex will struggle with extraction. Mitigation: finalize QR-IO spec before dispatch.

3. **Rejection reason string changes** (Issue V3-3): Spec lists `"no_path"` but code has `"missing_positions_file"`. Tests will fail. Mitigation: align spec with code or code with spec before FID-E dispatch.

4. **h5py concurrency unsolved** (Issue V3-3): QR-CORE multiprocessing will race with benchmark writes. Mitigation: document h5py safety strategy (SWMR, snapshot, or lock) before Wave 2.

5. **Cleanup1 unclear scope** (Item 10): Is validate_sync a hard gate or telemetry? Decision needed before Wave 3.

6. **FID-C Tier 1 simplification** (Item 9): Can position tensors be compared directly with torch.equal() or do they need sorting? Clarify before Wave 3.

7. **Fixture 3 construction** (Item 11): Test strategy should specify exact offset value and expected old-vs-new behavior for reproducibility.

---

## 7. Verdict

**SHIP_WITH_FIXES**

v3 is architecturally sound and successfully addresses most v2 critiques. However, **four blocking issues** must be resolved before Codex dispatch:

1. **Wave 0 serialization or worktree specification** — prevent merge conflicts on fidelity_analysis.py.
2. **QR-IO API finalization** — align proposed `load_position_tensor()` with actual load_layout extraction.
3. **Rejection reason string alignment** — decide if spec or code changes to match.
4. **h5py concurrency mitigation** — document (SWMR, snapshot, or lock) before QR-CORE dispatch.

Additionally, four **implementation clarifications** should be added to task specs:

1. **FIX-S parameter threading**: explicitly add `seed: int | None` to `quick()` and thread through fidelity callers.
2. **Cleanup1 scope**: document whether validate_sync is a hard gate or telemetry.
3. **Fixture 3 offset**: specify exact offset value (e.g., 0.20) and expected old-vs-new results.
4. **FID-C Tier 1**: clarify whether position tensors are node-index-ordered and can use direct torch.equal().

Estimated fix time: 2-3 hours (mostly clarifications, not code rewrites).

---

## 8. Executive Summary

v3 successfully integrates the v2 round-2 adversarial reviews and demonstrates strong architecture:

**Strengths:**
- Correctly identifies and uses stable_seed() instead of Python hash() (fixes round-2 N1).
- Redesigns graph_rel_best with rank-primary + clamped rel_best (fixes round-2 N4).
- Explicitly documents 4-wave DAG with cross-task dependencies (fixes hidden coupling in v2).
- Comprehensive test fixtures cover known-good, known-bad, and regression cases.
- Pareto front mathematics is correct (ideal corner = (1.0, 0.0)).

**Weaknesses:**
- Wave 0 has three parallel edits to fidelity_analysis.py with no documented merge strategy or worktree specification.
- QR-IO `load_position_tensor()` API spec doesn't match current `load_layout()` implementation (signature mismatch, rejection reasons).
- h5py concurrency issue between running benchmark and QR-CORE reads is not documented or mitigated.
- FIX-S parameter threading through quick() is incomplete (quick() still has no seed parameter).
- Cleanup1 scope is under-specified (validate_sync hard gate vs telemetry).

**Recommendation:** Fix the four blocking issues before Codex dispatch. The plan is 85% ready; the issues are clarifications and safety guards, not architectural failures.
