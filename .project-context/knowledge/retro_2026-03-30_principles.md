# Retro 2026-03-30: Principles (v2 -- post-adversarial)

## Why the Previous Retro Failed

The 2026-03-29 retro produced prose principles that were ignored within
hours. Root cause: principles written as instructions depend on voluntary
recall by a stateless agent under cognitive load. Evidence from 2 days:
voluntary recall does not work.

## What This Retro Produced: CODE, Not Prose

### Script 1: `scripts/validate_benchmark_integrity.py`
- Validates results.json and positions.h5 are in sync
- Every "ok" record must have a corresponding H5 key
- Called as a hard gate at the top of `fidelity_analysis.py`
- Prevents: the exact 2-day failure (results.json said "done", H5 had 0 positions)

### Script 2: `scripts/validate_fidelity_output.py`
- Post-flight check for fidelity analysis output
- Checks: row counts, NaN columns, paired data counts
- Delta comparison: if previous run exists, flags identical results
- Prevents: 9-hour analysis on empty data producing identical results unnoticed

### Script 3: `scripts/safe_purge_variants.py`
- Purges BOTH results.json AND positions.h5 atomically
- Dry-run by default, requires --confirm
- Post-purge sync validation
- Prevents: purging one data store without the other

### Integration: `fidelity_analysis.py` hard gate
- At startup, calls `validate_benchmark_integrity.validate_sync()`
- Warns loudly if results.json and H5 are out of sync
- Prevents: starting a 9-hour analysis on desynchronized data

## Remaining Principles (prose, because code enforcement isn't feasible)

### Delta comparison after re-runs
After any re-run, compare key metrics against previous run BEFORE
reporting results. If all metrics are identical after code changes,
something is wrong.

Enforcement: `validate_fidelity_output.py --previous <old_dir>`

### Assume corrupt state after kills
After any process kill, re-validate ALL coupled state from scratch.
Do not trust cached results.json counts.

Enforcement: `validate_benchmark_integrity.py` before any --resume.

## The Rule

**The deliverable of a retro is a commit containing code changes, not
a markdown file.** If the enforcement exists only as prose, the retro
has failed. Every principle that CAN be code MUST be code.
