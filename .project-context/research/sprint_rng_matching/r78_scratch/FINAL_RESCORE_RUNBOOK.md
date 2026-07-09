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
