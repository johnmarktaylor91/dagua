# R39+ Autonomous Loop State

**User directive (2026-05-25):** "Do it all man how many times do I have to tell you!!!! EVERYTHING!!!"

Keep iterating until every graphviz-paired engine is bit-exact (RMSD <0.001)
or honestly floored with documented architectural reason.

## Active In-Flight

- R39 sfdp gv_random RNG port -- codex PID 2148396, log /tmp/r39_sfdp_rng.log
- R39 neato PCA+CG solver port -- codex PID 2150603, log /tmp/r39_neato_pca.log
- R39 fdp tLayout/xLayout/packGraphs port -- codex PID 2152815, log /tmp/r39_fdp_kernels.log
- R35 comprehensive rerun (separate sprint) -- PID 1193554, log /tmp/r35_comprehensive_rerun.log

## Case Routing (every wake-up event)

### On CODEX_DONE label=codex pid=<R39_pid>
1. Read SUMMARY.md for that round
2. Look at smoke RMSD
3. If overall RMSD < 0.001: variant is bit-exact, mark done
4. If RMSD > 0.001: identify residual root cause in SUMMARY
5. **Auto-dispatch R(N+1) codex** for that engine targeting the named residual.
   Do NOT ask user. Same prompt pattern as R39_<engine>.md.
6. Cap at R45 (6 rounds beyond R39) before forcing "architectural floor"
   verdict and dropping the variant.

### On R35_RERUN_DONE label=r35-rerun
1. Verify auto-pipeline ran (fidelity_analysis + QR)
2. Read eval_output/fidelity_report_100seed_r35/report.md
3. Focal-rerun the 3 (or 4 if fdp re-enabled) new R37/R39 variants
4. Re-run fidelity_analysis on top
5. iMessage delta vs original report

### On CODEX_FAILED
1. Read log for failure mode
2. If quota: pause + schedule wake-up for reset + 5min, redispatch
3. If actual error: read codex output, write targeted retry spec, redispatch ONCE
4. If fails again same way: escalate to user

## Stop Criteria

Loop terminates when EITHER:
- (A) All 4 graphviz-paired engines have smoke RMSD < 0.001 (bit-exact)
- (B) Each remaining variant has architectural floor documented at
      `eval_output/algo_fidelity/round_N/<engine>/FLOOR_VERDICT.md` with
      specific kernel/algorithm citing why bit-exact isn't reachable
      from current dagua architecture
- (C) R45 reached without convergence (6 rounds beyond R39)

## Iteration Log

- R39 dispatched 2026-05-25 14:19 (3 codexes, medium effort)
- R40+ to be auto-dispatched on R39 completion based on SUMMARY residuals

## Anti-Flail Rule

If round N+1 produces identical smoke RMSD to round N for the SAME engine,
declare architectural floor and drop / lock variant. Do NOT do round N+2
on the same engine in that case.
