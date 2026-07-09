# r78 FINAL DEFINITIVE RESCORE RUNBOOK (written 2026-07-08, resume-safe)

Preconditions: ALL benches done (nodrl, slow, drl, sgd2, 3x fmmm; stale_mds/umap/neato_tl already done).
Check: pgrep -f run_benchmark.py returns nothing.

## 1. Full rescore (707 contested combos, all dirs)
cd /home/jtaylor/projects/dagua
python3 scripts/definitive_fidelity_analysis.py --mode full \
  --data-dir eval_output/benchmark_100seed_drlref_realfix eval_output/escalation_final eval_output/benchmark_100seed_fmmm_r3 eval_output/benchmark_100seed_gem_realfix eval_output/benchmark_100seed_r72_fixes eval_output/benchmark_100seed_r73_fixes eval_output/benchmark_100seed_r75_fixes eval_output/benchmark_100seed_r75_topup2 eval_output/benchmark_100seed_r76_gem_fix eval_output/benchmark_100seed_r76_refs eval_output/benchmark_100seed_r76_refs3 eval_output/benchmark_100seed_r76_sfdp_fix eval_output/benchmark_100seed_r76_sfdp_fix2 eval_output/benchmark_100seed_r76_sfdp_refs eval_output/benchmark_100seed_r76_umap_fix2 eval_output/benchmark_100seed_r76_umap_refs eval_output/benchmark_100seed_r76_umap_refs2 eval_output/benchmark_100seed_r77_era_refs eval_output/benchmark_100seed_r77_igraph_bk eval_output/benchmark_100seed_r77_mds2 eval_output/benchmark_100seed_r77_randomdag eval_output/benchmark_100seed_r77_sfdp_pack2 eval_output/benchmark_100seed_r77_sugiyama_final eval_output/benchmark_100seed_seeded_refs eval_output/benchmark_100seed_umap_realfix eval_output/benchmark_100seed_r78_biggraph eval_output/benchmark_100seed_r78_bigtail eval_output/benchmark_100seed_r78_bk2 eval_output/benchmark_100seed_r78_close18 eval_output/benchmark_100seed_r78_neato eval_output/benchmark_100seed_r78_neato_rd200 eval_output/benchmark_100seed_r78_neato_tl eval_output/benchmark_100seed_r78_prism eval_output/benchmark_100seed_r78_small_sgd2 eval_output/benchmark_100seed_r78_small_umap eval_output/benchmark_100seed_r78_stale_fmmm_fdpfid eval_output/benchmark_100seed_r78_stale_fmmm_steps10 eval_output/benchmark_100seed_r78_stale_fmmm_steps100_200 eval_output/benchmark_100seed_r78_stale_mds eval_output/benchmark_100seed_r78_targeted_drl eval_output/benchmark_100seed_r78_targeted_fastbig eval_output/benchmark_100seed_r78_targeted_nodrl eval_output/benchmark_100seed_r78_targeted_slow \
  --combos-file .project-context/research/sprint_rng_matching/r78_scratch/r78_final_rescore_combos.txt \
  --workers 6 --output eval_output/fidelity_definitive/per_combo_r78.jsonl
(NOTE: verify each dir path resolves -- some r77 dirs may live at eval_output/<name> not
benchmark_100seed_<name>; check the overlay resolver in the analysis script.)

## 2. Merge: overlay per_combo_r78 rows onto per_combo_r77 for untouched combos
(the analysis script's overlay handles this if given the r77 jsonl as base; else merge by combo key,
new wins.)

## 3. Build the definitive ledger
python3 scripts/build_definitive_ledger.py \
  --per-combo <merged jsonl> \
  --output-dir eval_output/fidelity_definitive_ledger_r78 \
  --stale-map '{"classical_mds": "2026-07-04", "fmmm": "2026-06-22", "sugiyama": "2026-07-05", "neato": "2026-07-06"}' \
  --winners <regenerated winners map for the new chain>
Exit code 0 required (zero DIVERGENT_UNEXPLAINED) unless every unexplained row gets adjudicated
with evidence and moved to the named-cause sidecar.

## 4. Adjudications pending (write into causes sidecar with evidence):
- parallel_cycles_4x5::fdp -> SUPERIOR_DISTINCT (dagua stress 0.082 vs ref-at-equilibrium 0.130; r78 neato agent dossier /tmp/r78_neato/FINDINGS.md -- copy to r78_evidence/ BEFORE any reboot)
- parallel_cycles_4x5::classic_sfdp_default + ::classic_sfdp_graphviz_fidelity -> verify per-component match, then SFDP label-box/packing family (needs probe evidence, do not name blind)
- 2x sgd2 era rows -> full-power rescore resolves (in combos list)

## 5. Gates: rerun definitive_fidelity_report gates over new chain; gate2+gate3 must both be 100% (post-b15d08b).

## 6. Then: honest headline table, STATE close-out, iMessage JMT.

## Addendum 2026-07-08 evening: family-2 resolution + remaining launches
- FAMILY-2 ROOT CAUSE: no bug. classic_maxent_stress default = 572s standalone on grid_20x20
  (steps50 162s); June watchdog rows + probe failures were runtime-vs-timeout artifacts.
- LAUNCHED: fam2_maxent3 (pid 1956834) + fam2_maxent400 (pid 1956835), 2h timeouts, ogdf_stress refs.
- 57 drl rows covered by targeted_drl bench; maxent_steps50/stress_maj_iter50 500-node rows by nodrl.
- STILL TO LAUNCH when a lane frees (fast tranche, ~3600s timeouts):
  A) --engines classic_davidson_harel_rounds50,classic_davidson_harel_rounds100,classic_neato,classic_pivot_mds_50,classic_stress_sgd_eps001,classic_stress_sgd_eps01,classic_stress_sgd_steps300 --seed-refs igraph_davidson_harel,graphviz_neato,igraph_mds,ogdf_stress,sgd2 --graphs grid_rect_6x8,multi_component_80,org_chart_deep,random_bipartite_60,regular_4_40,sierpinski_42,sparse_pair_50,triangular_lattice_36,wide_single_layer_1_50_1,clustered_medium_5x20,er_100,real_lesmis_77,sbm_4x30,citation_dag_300,heavy_tail_weights_50 --workers 3 --output-dir eval_output/benchmark_100seed_r78_fam2_fast (NOTE: DH rows on >50-node graphs will skip on engine max_nodes -- that outcome is itself the adjudication evidence: capacity-limited, not divergent)
  B) --engines classic_fmmm_steps10,classic_fmmm_steps100,classic_fmmm_steps200,classic_stress_maj_default,classic_stress_maj_iter500 --seed-refs ogdf_fmmm,graphviz_fdp,ogdf_stress --graphs chung_lu_150,citation_dag_300,compound_10x20,compound_dag_5x30,dependency_graph_100,hub_spoke_10x20,protein_ppi_200,multi_component_80,org_chart_deep,real_lesmis_77,rgg_100,small_world_100 --workers 2 --output-dir eval_output/benchmark_100seed_r78_fam2_fmst
  (both: --variants --max-nodes 0 --seeds 35 --seed-start 100 --timeout 3600 --watchdog-timeout 7200 --resume)

## Addendum 2026-07-09 morning: SELF-INFLICTED OVERSUBSCRIPTION INCIDENT + repair plan
- Overnight fleet (22 workers) x torch intra-op threads (~165% CPU/worker) -> load 80-90 ->
  run inflation -> ~257 TIMEOUT ERRORS: maxent3 140, fdpfid 105, fam2_fast 12. SAME mechanism
  as June family-2 artifacts. Lesson: worker budget must count THREADS not workers;
  cap fleet at ~8-10 workers total.
- --resume SKIPS non-ok records (run_benchmark.py:~1443) -> errors are POISONED in-dir.
  REPAIR (per dir, AFTER its bench exits): backup results.json; drop records with
  status in (error, timeout); relaunch same command with --resume (records absent -> retried).
- STOPPED early to shed load (resume-safe): fam2_fast, fam2_fmst, fmmm_s100_200.
- RELAUNCH QUEUE (strictly one-at-a-time as lanes finish, verify load <20 before each):
  1) fdpfid error-retry (after it exits; ~3 combos worth)
  2) maxent3 error-retry (4 combos; 2 workers, 7200s timeouts held)
  3) fmmm_s100_200 --resume (245 runs left)
  4) fam2_fast --resume (+12-error purge)
  5) fam2_fmst --resume
- Orphan-killed 3 init-parented forkservers post-SIGTERM (incl one 9h54m -- verify fdpfid/
  steps10 progress advances; if a lane stalls >30min with log silence, SIGTERM + resume it).
