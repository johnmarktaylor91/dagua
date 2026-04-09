# Baton -- Fidelity + Quality/Runtime Pipelines COMPLETE

Date: 2026-04-09
Branch: feat/bench-and-aesthetics
Status: Implementation complete. Awaiting benchmark completion for final runs.

---

## Summary

Both analysis pipelines (fidelity + new quality/runtime) are built, tested,
and smoke-verified on the in-progress benchmark data. All 11 implementation
tasks dispatched in 4 waves landed successfully. 91/91 new tests pass.

## What shipped

### New files
- `dagua/eval/pipeline_io.py` -- shared loader + seeded metric helpers +
  `stable_seed` + `validate_positions` + `aspect_ratio_deviation`
- `scripts/quality_runtime_analysis.py` -- the new QR pipeline (1,812 lines)
- `scripts/generate_quality_runtime_report.py` -- markdown report renderer
- `scripts/run_quality_runtime_pipeline.sh` -- QR shell driver
- `tests/test_pipeline_io.py` (29 tests)
- `tests/test_metric_seeding.py` (12 tests)
- `tests/test_fidelity_procrustes.py` (3 tests -- known-good / known-bad /
  pooled-within regression)
- `tests/test_fidelity_rejection_reasons.py` (3 tests)
- `tests/test_fidelity_pairwise_columns.py` (1 test)
- `tests/test_fidelity_metric_expansion.py` (4 tests)
- `tests/test_fidelity_deterministic.py` (4 tests)
- `tests/test_fidelity_report_markdown.py` (7 tests)
- `tests/test_quality_runtime_analysis.py` (18 tests)
- `tests/test_quality_runtime_report.py` (10 tests)

### Modified files
- `dagua/metrics.py` -- seeded `count_overlaps_detailed`,
  `sampled_crossing_rate`, `count_crossings`, `quick`
- `scripts/fidelity_analysis.py` -- A1-A5 atomic procrustes fix, B1 Welch,
  B2 metric expansion, B2b sampled metrics, C1 three-tier deterministic
  comparator, D1 seed cap, D2 pairwise CSV columns, E1 rejection
  preservation, G1 docstring, Cleanup1 validate_sync telemetry
- `scripts/fidelity_recompute_verdicts.py` -- mirrored Welch + expanded
  metrics
- `scripts/fidelity_add_metrics.py` -- canonical metric tuples + seeded
  sampled metrics
- `scripts/generate_fidelity_report.py` -- full rewrite to markdown (dropped
  LaTeX)
- `scripts/run_fidelity_pipeline.sh` -- dropped pdflatex, wired validator
- `scripts/merge_fidelity_csvs.py` -- README preservation

## Critical bug fixes landed

1. **Pooled within-RMSD (CF1)**: `within_rmsd` was pooling orig + reimpl
   pairwise distances, contaminating the baseline. A1 fixed it to
   within-original only with diagnostic reimpl column.
2. **LaTeX report (CF2)**: full rewrite to markdown. `pdflatex` dropped from
   shell driver.
3. **Python hash() instability**: FIX-S uses the existing SHA-256 based
   `stable_seed()` for cross-process reproducibility. Verified with a
   cross-process test in `test_metric_seeding.py`.
4. **QR-IO rejection reason mismatch**: canonical enum now uses the EXACT
   strings from the existing fidelity loader (missing_positions_file,
   h5_load_failure, load_failure, not_tensor, tensor_not_2d, tensor_not_xy,
   too_few_nodes, node_count_mismatch, contains_nan, contains_inf).
5. **validate_sync hard gate**: Cleanup1 downgraded the sys.exit(1) at
   line 2479-2499 to telemetry + warning. Verified on real benchmark data
   (267 desyncs detected, pipeline continued).
6. **Deterministic comparator**: three-tier comparator with raw equality
   (Tier 1), rigid alignment (Tier 2 -- new `procrustes_align_rigid`),
   metric near-equality (Tier 3). No over-engineered node ordering.
7. **Coverage denominator**: QR uses `graphs_scheduled / graphs_covered`
   derived from all-status records_df (not just ok rows), accounting for
   variant filtering caps (max_nodes).
8. **Graph-relative ranking**: rank is primary; rel_best is secondary with
   clamp at 10.0 + floor at 1e-3 + typical_scale normalization.
9. **Pareto ideal corner**: (1.0, 0.0) -- x = runtime_rel_fastest (min 1.0),
   y = rel_best (min 0.0).

## Dispatch summary (4 waves, ~55 minutes total Codex time)

| Wave | Task | Duration | Tests |
|---|---|---|---|
| 0a | QR-IO (pipeline_io + fidelity refactor) | 7m 22s | 26 |
| 0b | FID-D + FID-G | 4m 25s + manual | 1 |
| 1 | FID-S (metric seeding) | 7m 46s | 12 |
| 1 | FID-A (procrustes atomic A1-A5) | 7m 40s | 3 |
| 1 | FID-E (rejection reasons) | 5m 3s | 3 |
| 2 | FID-B (Welch + metric expansion) | 9m 18s | 4 |
| 2 | QR-CORE (quality/runtime analysis) | 16m 49s | 18 |
| 3 | FID-C (deterministic comparator) | 7m 7s | 4 |
| 3 | QR-REPORT (markdown renderer) | 5m 54s | 10 |
| 3 | FID-CLEANUP (validate_sync + markdown) | 6m 27s | 7 |
| **Total new tests** | | | **91** |

## Benchmark status

- Progress: 989,125 / 1,267,245 = 78.1% (as of 2026-04-09 17:06)
- Process: PID 2698780 (rescue wrapper 1799105), RSS 3.6GB, state Sl
- Error counts: 1,444 timeouts + 99 disconnected + 90 acyclic + 12 connected
  (all legitimate)
- Rate: ~55-60 records/min (heavier stochastic zone)
- ETA: ~2 more days to reach 100%

## Next steps (when benchmark finishes)

1. **Consolidate positions.h5** (one-time ~3 hours for ~900k files):
   ```bash
   python scripts/consolidate_positions_hdf5.py \
       --input eval_output/variant_bench_full \
       --output eval_output/variant_bench_full/positions.h5
   ```
   This refreshes the stale HDF5 store so analysis loads are 75x faster.

2. **Run fidelity pipeline**:
   ```bash
   ./scripts/run_fidelity_pipeline.sh
   ```
   Output: `eval_output/fidelity_report/data/*.csv` + `report.md`.

3. **Run QR pipeline**:
   ```bash
   ./scripts/run_quality_runtime_pipeline.sh
   ```
   Output: `eval_output/quality_runtime_report/*.csv` + `report.md` +
   per-family Pareto PNGs.

## Known limitations (documented, not bugs)

- `graph_rel_best` clamps at 10.0 for pathological near-zero cases. Rank is
  the primary ordering; clamp is a safety floor.
- Insight thresholds (15%/30%/1.25x/2.0x) are policy constants. The report
  prints per-family p25/p50/p75 of each metric alongside so the user can
  eyeball calibration.
- Cache key hashes the whole `dagua/metrics.py` file content. Changes in
  modules metrics.py imports from won't invalidate the cache; use
  `--cache-invalidate` as the safety net.
- QR first-run time on full benchmark: 1-3 hours with 8 workers + cache.
  Second run: minutes.

## Files NOT deleted (deferred)

- `scripts/compare_classic.py` -- still referenced by `_final_run.py` and
  `_overnight.py`.
- `scripts/compare_reimpl_vs_original.py` -- same.

## Plan docs

- `.project-context/plans/fidelity_and_quality_pipeline_plan.md` (v4)
- `.project-context/plans/fidelity_quality_codex_review.md` (round 1)
- `.project-context/plans/fidelity_quality_claude_review.md` (round 1)
- `.project-context/plans/fidelity_quality_round2_codex.md`
- `.project-context/plans/fidelity_quality_round2_claude.md`
- `.project-context/plans/fidelity_quality_round3_codex.md`
- `.project-context/plans/fidelity_quality_round3_claude.md`

## Specs for each Codex task

Saved at `/tmp/dagua_specs/`:
- `wave0a_qr_io.md`
- `wave0b_fid_d.md`
- `wave0b_fid_g.md` (executed manually via Edit tool, not dispatched)
- `wave1_fid_s.md`
- `wave1_fid_a.md`
- `wave1_fid_e.md`
- `wave2_fid_b.md`
- `wave2_qr_core.md`
- `wave3_fid_c.md`
- `wave3_fid_cleanup.md`
- `wave3_qr_report.md`

## Smoke test results

Both pipelines were smoke-tested against the in-progress
`eval_output/variant_bench_full/` (at 78.1% completion):

- **Fidelity**: wrote `validate_sync_telemetry.json` listing 267 desyncs
  (because positions.h5 is 9 days stale); continued instead of aborting
  (Cleanup1 working).
- **QR**: wrote `validate_sync_telemetry.json` + 129 metric cache entries
  within 90 seconds on 2 workers. Multiprocessing + cache working.

## Autonomous execution notes

- Dispatched all 11 tasks via the codex-companion broker autonomously.
- No critical blocking issues surfaced during the implementation.
- Pre-existing unrelated issues NOT addressed (out of scope):
  - `tests/test_classic_drl.py` fails to import `layout_drl` (archived
    classic code).
  - `scripts/purge_fixable_errors.py:219` has an E501 lint violation.
