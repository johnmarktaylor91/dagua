#!/usr/bin/env bash
# r70 phase CB: control + refresh benchmarks (spec secs. 1, 8).
# 1. Tier-1 positive-control bench: 5 control engines + refs, 8 pre-screened graphs, 100 seeds.
# 2. Deterministic refresh: the 8 DETERMINISTIC_DIFFERENT engines + refs, all graphs, 5 seeds.
# 3. Sugiyama rung-0 reverify: 6 sugiyama variants + refs, all graphs, 5 seeds (same dir as 2).
set -uo pipefail
cd /home/jtaylor/projects/dagua

CTL_GRAPHS="center_port_backedge_hub,clustered_medium_5x20,heavy_tail_weights_50,planar_60,real_lesmis_77,sbm_5x50,sparse_pair_50,weighted_clusters_3x10"
CTL_ENGINES="classic_fa2_default,fa2_ref__for__classic_fa2_default,classic_graphopt_default,igraph_graphopt__for__classic_graphopt_default,classic_lgl_default,igraph_lgl__for__classic_lgl_default,classic_tsnet_default,tsne_graph__for__classic_tsnet_default,classic_linlog_default,linlog__for__classic_linlog_default"

DET_ENGINES="classic_kk_steps1000,nx_kamada_kawai__for__classic_kk_steps1000,classic_kk_steps100,nx_kamada_kawai__for__classic_kk_steps100,classic_kk_steps300,nx_kamada_kawai__for__classic_kk_steps300,classic_rt_horizontal,igraph_rt_horizontal__for__classic_rt_horizontal,classic_spectral_default,nx_spectral__for__classic_spectral_default,classic_spectral_nx_fidelity,nx_spectral__for__classic_spectral_nx_fidelity,classic_spectral_random_walk,nx_spectral_random_walk__for__classic_spectral_random_walk,classic_spectral_unnormalized,nx_spectral__for__classic_spectral_unnormalized"

SUG_ENGINES="classic_sugiyama_default,igraph_sugiyama__for__classic_sugiyama_default,classic_sugiyama_graphviz_fidelity,graphviz_dot__for__classic_sugiyama_graphviz_fidelity,classic_sugiyama_passes48,igraph_sugiyama__for__classic_sugiyama_passes48,classic_sugiyama_passes4,igraph_sugiyama__for__classic_sugiyama_passes4,classic_sugiyama_tight,igraph_sugiyama__for__classic_sugiyama_tight,classic_sugiyama_wide,igraph_sugiyama__for__classic_sugiyama_wide"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "=== CB1: tier1 controls (100 seeds x 8 graphs x 10 engines) ==="
python3 scripts/run_benchmark.py --seeds 100 --seed-start 42 --variants \
  --engines "$CTL_ENGINES" --graphs "$CTL_GRAPHS" \
  --output-dir eval_output/benchmark_100seed_tier1_controls \
  --resume --workers 8 --timeout 300 --watchdog-timeout 600 \
  || { echo "CB1_FAILED rc=$?"; exit 1; }
echo "CB1_DONE"

echo "=== CB2: deterministic refresh (5 seeds x all graphs x 16 engines) ==="
python3 scripts/run_benchmark.py --seeds 5 --seed-start 42 --variants \
  --engines "$DET_ENGINES" \
  --output-dir eval_output/benchmark_5seed_deterministic_refresh \
  --resume --workers 8 --timeout 300 --watchdog-timeout 600 \
  || { echo "CB2_FAILED rc=$?"; exit 1; }
echo "CB2_DONE"

echo "=== CB3: sugiyama rung-0 reverify (5 seeds x all graphs x 12 engines) ==="
python3 scripts/run_benchmark.py --seeds 5 --seed-start 42 --variants \
  --engines "$SUG_ENGINES" \
  --output-dir eval_output/benchmark_5seed_deterministic_refresh \
  --resume --workers 8 --timeout 300 --watchdog-timeout 600 \
  || { echo "CB3_FAILED rc=$?"; exit 1; }
echo "CB3_DONE"
echo "PHASE_CB_COMPLETE"
