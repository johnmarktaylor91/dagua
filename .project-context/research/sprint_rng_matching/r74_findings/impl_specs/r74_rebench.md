<task>
Re-benchmark the 6 algorithms changed by the dagua r74 fidelity sprint, overlay-analyze against the r73
baseline, and produce a scorecard with a PER-ENGINE REGRESSION CHECK. This VERIFIES the r74 code fixes;
you produce NUMBERS + EVIDENCE only -- do NOT write final result docs, do NOT make fidelity claims, do
NOT relabel anything as "floor", do NOT touch git or any code. The orchestrator interprets your output.

r74 made 8 commits on develop (26eb9b2..169ce7b) changing these 6 reimpl engines:
  sfdp (p_neg2 force-law clamp + disconnected-component packing),
  classical_mds (disconnected per-component + TileToRows pack),
  maxent_stress (disconnected per-component pack),
  fmmm (vectorized graphviz-fdp grid repulsion -- targets the ~9 fmmm_graphviz_fdp TIMEOUTS),
  umap (n_neighbors clamp -- targets the ~10 umap_nn30 CRASH combos),
  sugiyama (igraph-faithful GLPK layer objective + directed<=1000 gating, igraph variants only;
            + iterative cycle-break fixing the small_world_2000 recursion CRASH).

MUST READ FIRST (methodology is intricate; follow r73 exactly):
- .project-context/research/sprint_rng_matching/r73_drive_to_zero_STATE.md  (the re-bench commands +
  the TWO overlay-trap postmortems: (1) overlay must match base SEED COUNT=100 or it Frankensteins;
  (2) overlay of a SEEDED-ref engine must re-run the ref WITH --seed-refs or stale ref keys persist ->
  mixed ref cloud -> cosmetic mode B->A relabel.)
- .project-context/research/sprint_rng_matching/r74_PLAN.md and r74_close_all_gaps_STATE.md (what each fix targets)
- scripts/run_benchmark.py, scripts/definitive_fidelity_analysis.py, scripts/definitive_fidelity_report.py (--help/argparse)

SEEDABLE ref bases (must pass to --seed-refs for the engines that touch them):
  graphviz_sfdp (sfdp), ogdf_fmmm + graphviz_fdp (fmmm), ogdf_stress (maxent_stress), igraph_mds (classical_mds).
  (umap and sugiyama are NOT in the seedable set.)

STEP 0 -- BASELINE REPRODUCTION SELF-CHECK (do this FIRST, it guards against a wrong overlay chain):
  Determine the EXACT --data-dir overlay chain that r73's per_combo (eval_output/fidelity_definitive_r73/
  per_combo.json, 574 divergent) was built from (read r73 STATE + any provenance; the chain involves
  benchmark_100seed_escalation_final, benchmark_100seed_seeded_refs, benchmark_100seed_r73_fixes and
  possibly earlier r72 dirs -- freshest LAST). Run definitive_fidelity_analysis.py over JUST that chain
  (no r74 dir) and CONFIRM it reproduces ~574 total divergent / 309 escalation-divergent. If it does
  NOT reproduce ~574, STOP and report the chain you tried and the number you got -- do not proceed with a
  wrong baseline.

STEP 1 -- RE-BENCH the 29 changed-engine variants (reuse cached references; recompute dagua side):
  python3 scripts/run_benchmark.py --variants --engines <the 6 reimpl bases: classic_sfdp,classic_umap,
    classic_maxent_stress,classic_classical_mds,classic_fmmm,classic_sugiyama -- or the explicit 29
    variant names; confirm the right filter syntax from argparse> --seeds 100 --seed-start 42
    --seed-refs graphviz_sfdp,graphviz_fdp,ogdf_fmmm,ogdf_stress,igraph_mds
    --max-nodes 300 --timeout 90 --watchdog-timeout 7200
    --output-dir eval_output/benchmark_100seed_r74_fixes --resume
  HARD GUARDS: seeds MUST be 100 (NOT 5) to match the base seed count. --seed-refs MUST include every
  seedable base above. After the run, SELF-REPORT per engine: number of reimpl seeds and number of
  ref-seed keys per combo (they must be 100 / consistent -- prove no Frankenstein).
  THEN a TARGETED LARGE-GRAPH ADDENDUM (the <=300 cap drops graphs needed to validate two fixes):
  re-run (a) ALL classic_sugiyama variants and (b) classic_fmmm_graphviz_fdp on the large graphs the
  fixes target -- at minimum small_world_2000 (sugiyama recursion) and the fmmm_graphviz_fdp timeout
  graphs -- with --timeout 600 --watchdog-timeout 7200, appended into the same output dir (--resume).

STEP 2 -- RE-ANALYZE: definitive_fidelity_analysis.py --mode full with the r73 overlay chain from STEP 0
  PLUS --data-dir eval_output/benchmark_100seed_r74_fixes as the FRESHEST (LAST) dir; --combos-file =
  all combos of the 6 changed engines (generate it); --output eval_output/fidelity_definitive/r74_analysis.jsonl.

STEP 3 -- SCORECARD + REGRESSION CHECK:
  - Run definitive_fidelity_report.py with --controls (the anti-laundering gate) on the r74 analysis.
  - Compute vs the r73 baseline (per_combo.json): total divergent (was 574), escalation-divergent ModeA
    (was 309), 3Q (was 36), and PER-ENGINE divergent counts before/after.
  - HARD REGRESSION CHECK: list every combo that was rung 1/2/2'/3/3Q at r73 baseline and is now rung 4
    (final_rung is a STRING "4" -- compare as strings). ZERO is the target; list any that regressed.
  - INSUFFICIENT recovered: how many of the 257 INSUFFICIENT are now scored (esp. umap_nn30 crashes,
    fmmm_graphviz_fdp timeouts, sugiyama small_world_2000 crash).
  - Also verify the anti-laundering controls still pass 0/40 (no laundering introduced).
</task>

<default_follow_through_policy>
Proceed autonomously EXCEPT: STOP and report if STEP 0 cannot reproduce ~574 (wrong overlay chain), or if
you cannot guarantee 100-seed / seed-refs consistency (Frankenstein risk), or any irreversible concern.
Resume-safe: if a benchmark sub-run dies, relaunch with --resume.
</default_follow_through_policy>

<constraints>
- Do NOT edit any source code or git. Do NOT write final RESULTS/scorecard docs or make claims -- output
  raw numbers + evidence to /tmp/r74_rebench_results.md (+ the analysis jsonl + report dir). Do NOT
  relabel combos as floor (no FP experiments here). final_rung is a STRING.
- Match params+seed to references; NO runtime delegation introduced. Verify on the BENCHMARK PATH (that
  is exactly what this is). ASCII only.
- Mind disk: positions are written; check df before the run and warn if < 15G free.
</constraints>

<verification_loop>
Report: the STEP-0 baseline number (must be ~574), the per-engine seed/ref-key self-check, total divergent
r73->r74 + per-engine deltas, the regression list (target: empty), INSUFFICIENT recovered counts, and the
controls 0/40 result. Put it all in /tmp/r74_rebench_results.md.
</verification_loop>
