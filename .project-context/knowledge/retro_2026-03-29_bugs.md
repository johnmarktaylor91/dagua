# Retro 2026-03-29: Bug & Process Notes

## Category: Benchmark Infrastructure

### Bug 1: --resume Without --engines Runs Everything
- **Symptoms:** Benchmark queued 528K jobs instead of ~15K. Progress nearly
  stalled after initial burst. Result count exceeded original target.
- **Root cause:** `--resume` scans ALL registered engines/variants, not just
  recently modified ones. Deterministic engines with 105 entries get backfilled
  to 3150 (identical results, wasted compute).
- **Fix:** Always pair `--resume` with `--engines <list>` when re-running
  specific variants.
- **Architectural lesson:** The benchmark script has no concept of "dirty"
  variants. It can't distinguish "I purged these on purpose" from "these were
  never run." The `--engines` filter is the only scoping mechanism.

### Bug 2: HDF5 Individual Key Deletion is Catastrophically Slow
- **Symptoms:** Purging 39K entries from 400K-key H5 file took 50+ minutes.
- **Root cause:** HDF5 delete is O(N) per operation. 39K deletes on a 1.4GB
  file = massive I/O.
- **Fix options:** (a) Skip H5 purge -- benchmark overwrites stale data.
  (b) Write new H5 with only kept keys. (c) Use h5repack.
- **Architectural lesson:** For HDF5 bulk operations, always prefer bulk
  writes over individual deletes.

### Bug 3: Piping Background Commands Through head/tail
- **Symptoms:** Benchmark killed by SIGPIPE (exit 144) after 30 lines output.
- **Root cause:** `| head -30` in background command closes pipe after N lines.
- **Fix:** Never pipe long-running commands through head/tail. Use separate
  `tail -f` or `grep` on the output file to monitor.

## Category: Fidelity Analysis Pipeline

### Bug 4: float('') in CSV Verdict Logic
- **Symptoms:** `ValueError: could not convert string to float: ''`
- **Root cause:** `finalize_group_row()` used bare `float()` on CSV values
  that could be empty strings from the recompute partial-population path.
- **Fix:** Added `_safe_float()` helper, replaced all bare `float()` calls
  in the verdict path.

### Bug 5: Procrustes Mirror Detection Without Correction
- **Symptoms:** 3 near-miss families divergent due to max_displacement just
  over 1.0. All had mirror_match + extreme scale_ratio.
- **Root cause:** `fidelity_procrustes()` tested reflected alignment, flagged
  it as `reflected=True`, but returned the WORSE non-reflected RMSD. A TODO
  in the code acknowledged this.
- **Fix:** Modified to return whichever alignment (reflected or not) gives
  lower RMSD.

## Category: Process

### Bug 6: Passive Monitoring Instead of Active Diagnosis
- **Symptoms:** User asked "now?" 10 times over 2 hours. Each response was
  "still going, X more to go" without investigating slowness.
- **Root cause:** Checking result count instead of reading the benchmark log
  to see what engines were actually running.
- **Fix:** On first sign of unexpected slowness, read the actual process
  output to diagnose what's happening.

## Summary Table

| # | Bug | Category | Time Lost | Severity |
|---|-----|----------|-----------|----------|
| 1 | --resume without --engines | Benchmark | 6+ hours | Critical |
| 2 | H5 individual deletes | Benchmark | 50 min | High |
| 3 | head pipe kill | Benchmark | 5 min | Low |
| 4 | float('') in verdicts | Analysis | 20 min | Medium |
| 5 | Mirror detection w/o correction | Analysis | 5 min | Medium |
| 6 | Passive monitoring | Process | 30 min user | High |
