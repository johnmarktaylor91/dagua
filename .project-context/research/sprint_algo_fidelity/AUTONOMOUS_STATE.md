---
run: algo_fidelity_completionist
created: 2026-04-30T15:00:00-04:00
state: R28_RUNNING
phase: r28_fixes_plus_ogdf_infra
last_updated: 2026-04-30T15:00:00-04:00
parallel_codexes:
  r28_sfdp:    {pid: 2590923, log: /tmp/algo_fid_28_sfdp.log,    status: running}
  r28_neato:   {pid: 2591053, log: /tmp/algo_fid_28_neato.log,   status: running}
  r28_dot:     {pid: 2591350, log: /tmp/algo_fid_28_dot.log,     status: running}
  r28_ogdf:    {pid: 2591596, log: /tmp/algo_fid_28_ogdf.log,    status: running}
benchmark_pid: null
benchmark_log: null
benchmark_progress_file: null
stop_criterion: 100_seed_benchmark_completed_with_fidelity_pipeline_run
max_phases: 8
fallback_chain: [codex_medium_effort, codex_high_effort, claude_subagent_opus, schedule_wakeup]
---

# Algo-fidelity completionist autonomous loop

## Goal

Drive `algo_fidelity` sprint to genuine ceiling. After this run, JMT
should be able to claim "dagua matches every audited reference algorithm
modulo documented infrastructure residuals."

User directive (verbatim, 2026-04-30 15:00):
> 1. completionist option, zero stones unturned
> 2. fix the ogdf runner rebuild and multiseed cache thing and deal with
>    any associated issues, get it fully fixed. don't put anything off
> 3. then kick off the 100 seed benchmark run once everything is fixed
> 4. cook in background while I do other things for next 4-5 days
> 5. iMessage at beginning and end of each major step

## Major step structure

| # | Phase                              | Beginning iMessage              | End iMessage                       |
|---|------------------------------------|---------------------------------|------------------------------------|
| 1 | R28 fixes (sfdp + neato + dot)     | "R28 fixes starting"            | "R28 done: <verdicts>"             |
| 2 | OGDF runner rebuild + seed plumb   | (combined w/ #1)                | "OGDF runner rebuilt"              |
| 3 | Multi-seed OGDF cache regen        | "OGDF cache regen starting"     | "OGDF cache regen done"            |
| 4 | R29 verification sweep             | "R29 verification starting"     | "R29 verification done"            |
| 5 | 100-seed full benchmark            | "100-seed benchmark starting"   | "100-seed benchmark done (~5 days)"|
| 6 | Fidelity pipeline post-benchmark   | "Fidelity pipeline starting"    | "Fidelity pipeline done. RESULT"   |

## Wake-up case routing

EVERY new turn (Monitor event, schedule wakeup, user ping, anything):
1. Read this file. Note `state` and `phase`.
2. Check observable state (`git log -3`, `pgrep`, log size, file
   existence). The case routing below picks next action based on
   observable, NOT on what was last done.

### Cases

**A. R28 codex still running** (any of pid in parallel_codexes alive)
   → ack briefly, yield. Watcher will surface.

**B. R28 codex(es) finished**
   → triage logs, commit any uncommitted fix code, move to phase #2 / #3.
   - sfdp/neato/dot: re-run live_compare 30 seeds; record post-fix
   - ogdf: verify runner binary built and `./scripts/ogdf_runner --help`
     works (or equivalent); if good, advance to phase #3.
   → Update phase. iMessage R28 done.

**C. OGDF rebuild failed** (libs unavailable, can't apt-install, can't build)
   → mark phase as ogdf_blocked, iMessage user with specific error,
     SKIP cache regen, advance to R29 with whatever cache exists.
   → Document residual: "OGDF multi-seed cache requires ogdf-dev libs
     (couldn't install)".

**D. Cache regen running**
   → poll `eval_output/competitor_cache/` for new files; if 80%+ done
     and stalled 30 min, mark complete; else keep polling.

**E. Cache regen done**
   → advance to R29. Re-run `scripts/round_26_sweep.sh` →
     `scripts/round_24_aggregate.py round_26`. iMessage results.

**F. R29 done, no regressions**
   → kick off 100-seed full benchmark. iMessage start.

**G. R29 found regressions**
   → dispatch one round of codex fixes targeting only regressions. Loop
     back to R29.

**H. 100-seed benchmark running**
   → check `progress.json` (if benchmark uses it) or ETA from
     `dagua benchmark-status`. Yield. Check back in 30-60 min via
     ScheduleWakeup. Restart if crashed.

**I. 100-seed benchmark done**
   → run fidelity pipeline:
     - `dagua placement-tune --output-dir eval_output/report`
     - `python scripts/fidelity_analysis.py`
     - `python scripts/validate_fidelity_output.py`
   → iMessage final report links + key numbers.

**J. Anything fails 3 times in a row**
   → ANTI-FLAIL: mark as principled_residual, iMessage user with
     specific blocker, advance.

**K. Codex quota exhausted**
   → fallback chain:
     - effort=high (one retry)
     - claude subagent (opus) for code work
     - schedule wakeup for codex reset (check error msg)
     - if all blocked: iMessage user "blocked", stop until they say resume.

## Shutdown procedure (when phase #6 done)

1. Update `state: DONE_FINAL`.
2. Write `.project-context/research/sprint_algo_fidelity/algo_fidelity_FINAL_FINAL_SUMMARY.md`
   with full verdicts, residuals, infra notes.
3. iMessage final summary including:
   - Per-family final TOST verdict
   - Path to benchmark report
   - Path to placement_tuning.md
   - Total commits
   - Any residuals still open
4. Mark all todos complete.

## Iteration log

| phase | started | ended | commits | notes |
|---|---|---|---|---|
| R28_dispatch | 2026-04-30T15:00:00-04:00 | (running) | n/a | 4 parallel codexes (sfdp/neato/dot fixes + OGDF infra) |
