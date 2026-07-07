# Resume after reboot -- r78 targeted fidelity benches (2026-07-07)

Machine reboot planned. All benches stopped CLEANLY (SIGTERM, results.json flushed).
Everything below is `--resume`-safe: relaunch the SAME command, completed combos are skipped.
`cd /home/jtaylor/projects/dagua` first. PYTHONPATH not needed (editable install).

## State at stop
- nodrl bench: 275/5670 complete (~5%), results.json 4515 recs OK
- slow bench:  475/1925 complete (~25%), results.json 1575 recs OK
- drl bench: NEVER LAUNCHED (held -- drl ~500s/seed pathology; decide: perf-fix subagent vs grind)
- small tier: NEVER LAUNCHED (180 rows, mostly stale sgd2 pre-hang-fix + missing refs)

## Relaunch commands (identical -> --resume continues)

SEED_REFS="graphviz_sfdp,graphviz_fdp,graphviz_neato,igraph_davidson_harel,igraph_drl,igraph_mds,ogdf_fmmm,ogdf_gem,ogdf_stress,umap_graph,nx_spring,sgd2,sgd2_multi_ref"

### nodrl (500-node fast engines: 53 big-insuff + 24 gem-stale)
NODRL_ENG="classic_fr_steps50,classic_fr_steps100,classic_fr_steps200,classic_fr_steps500,classic_gem_iters100,classic_gem_iters500,classic_gem_iters2000,classic_maxent_stress_steps50,classic_neato,classic_sfdp_default,classic_sfdp_graphviz_fidelity,classic_sfdp_p_neg2,classic_sfdp_steps200,classic_sfdp_theta04,classic_sfdp_theta08,classic_stress_maj_iter50,classic_classical_mds_default,classic_classical_mds_igraph_fidelity"
setsid nice -n 5 python3 scripts/run_benchmark.py --variants --engines "$NODRL_ENG" --seed-refs "$SEED_REFS" --graphs ba_500,dependency_500,er_500,grid_20x20,grid_50x50,powerlaw_500,rgg_500,sbm_8x100,small_world_500 --max-nodes 0 --seeds 35 --seed-start 100 --workers 4 --timeout 3600 --watchdog-timeout 7200 --resume --output-dir eval_output/benchmark_100seed_r78_targeted_nodrl > /tmp/r78_targeted_nodrl.log 2>&1 &

### slow (2000-node non-drl + drl_refine: 34 hard rows)
SLOW_ENG="classic_classical_mds_default,classic_classical_mds_igraph_fidelity,classic_drl_refine,classic_fr_steps200,classic_fr_steps500,classic_sfdp_default,classic_sfdp_graphviz_fidelity,classic_sfdp_p_neg2,classic_sfdp_steps200,classic_sfdp_theta04,classic_sfdp_theta08"
setsid nice -n 5 python3 scripts/run_benchmark.py --variants --engines "$SLOW_ENG" --seed-refs "$SEED_REFS" --graphs ba_2000,er_2000,powerlaw_2000,rgg_2000,small_world_2000 --max-nodes 0 --seeds 35 --seed-start 100 --workers 5 --timeout 21600 --watchdog-timeout 28800 --resume --output-dir eval_output/benchmark_100seed_r78_targeted_slow > /tmp/r78_targeted_slow.log 2>&1 &

After relaunch: pgrep -of "run_benchmark.py.*targeted_nodrl" (and _slow) -> write pidfiles,
arm exit monitors (inline `until ! kill -0 PID` loops -- pkill-proof).

## Then (once benches land)
- Rescore: definitive_fidelity_analysis.py --mode full --data-dir <all ~35 dirs + the 2 new
  targeted dirs> --combos-file /tmp/r78_targeted_combos.txt (291 combos) --workers >1 OK now
  (spawn-pool fix merged ad0df11). NOTE: /tmp/r78_targeted_combos.txt dies on reboot --
  regenerate from r77 per_combo.json (insufficient + gem-not-clean); recipe in
  r76_final_sprint_STATE.md 2026-07-07 entry.
- drl decision, small tier, then DEFINITIVE FINAL RE-LEDGER (supersedes r77).

## Reboot hygiene
- Codex sentinel SET (out of credits, JMT 2026-07-07) -> all dispatched work = Claude subagents.
- /tmp files (combos list, logs, pidfiles) DIE on reboot -- only the eval_output/ dirs + git persist.
- Monster-bench full command banked at r78_scratch/MONSTER_BENCH_FULL_CMD.txt (deferred due-diligence run).
