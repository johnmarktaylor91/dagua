# Fidelity Analysis Pipeline Retro -- Incident Log (2026-03-27)

## Timeline of Errors

### Incident 1: Scale normalization omission (~4 hours wasted)
**What happened:** Adversarial critic said "scale is meaningful fidelity info,
don't normalize." I accepted without questioning. First report showed ALL 62
paired algorithms as "divergent" with RMSD 2-5700. Different implementations
use different coordinate systems (NetworkX [0,1], igraph [0,100], PyTorch
arbitrary). Scale difference dominated shape difference.
**Root cause:** Deferred to critic's authority without applying domain judgment.
The critic was technically right (scale IS information) but practically wrong
(it's not meaningful for fidelity comparison).
**Fix:** Normalize both layouts to unit Frobenius norm before alignment.
**Time lost:** First full run (75 min) + analysis + head-scratching before diagnosis.

### Incident 2: Frozen dataclass broke pairing silently (~3+ hours wasted)
**What happened:** ResultRecord was @dataclass(frozen=True). Adding result_key
field for HDF5 lookup required mutation. The `_with_key` function tried to set
an attribute on a frozen instance -> FrozenInstanceError. But this error was
SILENT -- it happened inside a dict comprehension and the load_results function
returned records WITHOUT result_keys. All downstream pairing worked but with
empty keys, so HDF5 lookups fell through to .pt file loading. 35 algorithm
families appeared "unpaired" (0 paired_ok) because the pairing logic couldn't
match records.
**Root cause:** Changed a field on a frozen dataclass without changing frozen=False.
**Why silent:** The error occurred in a try/except somewhere OR the code path
that used result_key had a fallback that masked the error.
**Fix:** Changed @dataclass(frozen=True) to @dataclass.
**Time lost:** Ran the full analysis twice (75 min + 37 min) before discovering
35 families were unpaired when they shouldn't be.

### Incident 3: Bytecache not cleared between fix and rerun (~37 min wasted)
**What happened:** After committing scale normalization + frozen fix + HDF5
support, dispatched a new analysis run. The run produced IDENTICAL results to
the pre-fix run. All three fixes were absent.
**Root cause:** Python cached the old .pyc bytecode. The dispatch launched
`python scripts/fidelity_analysis.py` which loaded the cached bytecode from
`scripts/__pycache__/`, not the updated .py source.
**Fix:** `find scripts/ -name "__pycache__" -exec rm -rf {} +` before launch.
**Time lost:** 37 minutes of the second run, plus diagnosis time.

### Incident 4: ThreadPoolExecutor + GIL for CPU-bound work (~40 min wasted)
**What happened:** Used ThreadPoolExecutor(max_workers=12) to parallelize
Procrustes SVD computation. First attempt: 0% CPU, process hung. Threads
were blocked on h5py reads which hold the GIL.
**Fix attempt 1:** Pre-load all positions into memory dict, then threads only
do math (no I/O). CPU went to 100% but only on ONE core -- GIL prevents
actual parallelism for CPU-bound torch.linalg.svd on small tensors.
**Root cause:** Python GIL. ThreadPoolExecutor only helps for I/O-bound work.
CPU-bound numpy/torch operations on small tensors don't release the GIL.
Should have used ProcessPoolExecutor with shared memory.
**Time lost:** ~40 min diagnosing + two hung/slow attempts.

### Incident 5: --skip-metrics didn't skip metric ANALYSIS (~3+ hours wasted)
**What happened:** Added --skip-metrics flag to skip quality metrics computation
(the expensive metrics.quick() call). But the flag only skipped metric COLLECTION
in load_layout(). The metric ANALYSIS loop (bootstrap_ci, KS test, Mann-Whitney,
TOST at 4 margins, Cohen's d, Cliff's delta) still ran on every group.
With empty metrics, collect_metric_values() returned NaN arrays. All 9,104 groups
x 5 metrics x 10,000 bootstrap samples = 455M NaN operations. Plus KS, MW, TOST
on NaN arrays -- all producing NaN results.
**Root cause:** The skip flag was set in one function (load_layout) but the
consuming function (add_metric_tests_to_row) didn't check it.
**Fix:** Added `if skip: return` at the top of add_metric_tests_to_row.
**Time lost:** 3+ hours of the run before diagnosis. Would have been 180+ min total.

### Incident 6: 10,000 bootstrap samples (design error)
**What happened:** Adversarial critic spec said "10000 bootstrap samples" and
it was written directly into the default. 1000 would have been equally valid
statistically (1% vs 0.3% CI precision). 10x the compute for negligible benefit.
**Root cause:** Academic reflex -- bigger number sounds more rigorous. No one
questioned whether 1000 was sufficient.
**Fix:** Change default to 1000 (for future runs).
**Time lost:** 10x multiplier on every run's bootstrap phase.

### Incident 7: Unbuffered output not set (ongoing frustration)
**What happened:** Multiple runs had no visible progress because Python's stderr
was buffered. Couldn't tell if process was making progress, stuck, or dead.
Led to repeated "now?" checks with no information gained.
**Root cause:** Default Python stderr buffering. Need PYTHONUNBUFFERED=1.
**Fix:** Added PYTHONUNBUFFERED=1 to launch command (eventually).
**Time lost:** Hours of anxiety + inability to estimate completion time.

### Incident 8: Time estimates consistently wrong
**What happened:** Every estimate was optimistic by 2-5x.
- "45 min" -> 75 min (first run)
- "10 min with HDF5" -> 12 min just for pre-load
- "5-10 min Procrustes only" -> 40+ min
- "3 hour upper bound" -> exceeded
**Root cause:** Failure to account for: bootstrap overhead, GIL contention,
h5py per-key overhead, stat test overhead beyond Procrustes.
**Fix:** Stop estimating. Just run it and report progress.

### Incident 9: dispatch.sh reliability issues
**What happened:** dispatch.sh sometimes relaunched processes after kill,
failed silently, or reported wrong status. Multiple confusing kill-restart
cycles where processes kept reappearing.
**Root cause:** dispatch.sh runs commands in a subshell with `&`. Killing
the Python process doesn't always kill the wrapper. The wrapper doesn't
always detect child death.
**Fix:** Switched to direct `nohup python ... &` for reliability.

### Incident 10: layout.metrics[key] KeyError with --skip-metrics
**What happened:** First --skip-metrics run crashed immediately with
KeyError: 'aspect_ratio'. The metrics dict was empty but code accessed
it with dict[key] instead of dict.get(key, default).
**Root cause:** 8 places in the code used direct dict access instead of .get().
**Fix:** Replace all layout.metrics[metric_name] with .get(metric_name, math.nan).
**Time lost:** One restart cycle (~5 min).

## Summary

| Incident | Time wasted | Category |
|----------|------------|----------|
| Scale normalization | ~4 hrs | Design decision |
| Frozen dataclass | ~3 hrs | Silent failure |
| Bytecache | ~37 min | Python gotcha |
| ThreadPool + GIL | ~40 min | Wrong tool |
| Skip-metrics incomplete | ~3+ hrs | Incomplete implementation |
| 10K bootstrap | ~10x multiplier | Over-engineering |
| Unbuffered output | hours of anxiety | Missing basics |
| Bad time estimates | cumulative | Optimism bias |
| dispatch.sh issues | ~30 min | Infrastructure |
| KeyError on skip | ~5 min | Missing .get() |

**Total time wasted: ~12+ hours on a task that should have taken ~1 hour.**
