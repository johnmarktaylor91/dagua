#!/usr/bin/env bash
# Round 29: full verification sweep using new multi-seed OGDF cache (R28 ogdf-infra)
# 30 seeds across all 16 R22/R23 families + sfdp + dot.
#
# Output: eval_output/algo_fidelity/round_29/<family>/multi_seed_summary.json
#
# OGDF families use the new multi-seed cache at
# eval_output/algo_fidelity/round_28/ogdf_seeded_cache_30/

set -u
cd "$(dirname "$0")/.."

LOG=/tmp/round_29_sweep.log
OUT_BASE=eval_output/algo_fidelity/round_29
GRAPHS="linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels"
SEEDS=30
OGDF_CACHE=eval_output/algo_fidelity/round_28/ogdf_seeded_cache_30
mkdir -p "$OUT_BASE"

# Family -> (dagua_engine, target_engine, use_ogdf_cache)
declare -a FAMILIES=(
  "classical_mds:classic_classical_mds:igraph_mds:0"
  "fa2:classic_fa2:fa2_ref:0"
  "fmmm:classic_fmmm:ogdf_fmmm:1"
  "fr:classic_fr:nx_spring:0"
  "gem:classic_gem:ogdf_gem:1"
  "kk:classic_kk:nx_kamada_kawai:0"
  "lgl:classic_lgl:igraph_lgl:0"
  "maxent_stress:classic_maxent_stress:ogdf_stress:1"
  "pivot_mds:classic_pivot_mds:ogdf_pivot_mds:1"
  "rt:classic_rt:igraph_rt:0"
  "sgd2_multi:classic_sgd2_multi:sgd2_multi_ref:0"
  "spectral:classic_spectral:nx_spectral:0"
  "stress_maj:classic_stress_maj:ogdf_stress:1"
  "stress_sgd:classic_stress_sgd:sgd2:0"
  "sugiyama:classic_sugiyama:igraph_sugiyama:0"
  "umap:classic_umap:umap_graph:0"
  "sfdp:classic_sfdp:graphviz_sfdp:0"
)

echo "=== Round 29 sweep started $(date -Iseconds) ===" | tee -a "$LOG"
PASSED=0
FAILED=0
START_T=$SECONDS

for entry in "${FAMILIES[@]}"; do
  IFS=':' read -r fam dagua target use_ogdf <<< "$entry"
  out_dir="$OUT_BASE/$fam"
  rm -rf "$out_dir"
  mkdir -p "$out_dir"
  fam_start=$SECONDS
  echo "--- $fam: $dagua vs $target (ogdf_cache=$use_ogdf) ---" | tee -a "$LOG"

  cmd=(timeout 1800 python3 scripts/algo_fidelity_live_compare.py
       "$dagua" "$target" --seeds "$SEEDS" --graphs "$GRAPHS"
       --output-dir "$out_dir")
  if [ "$use_ogdf" = "1" ]; then
    cmd+=(--graphviz-cache-dir "$OGDF_CACHE")
  fi

  if "${cmd[@]}" >> "$LOG" 2>&1; then
    elapsed=$((SECONDS - fam_start))
    PASSED=$((PASSED + 1))
    echo "    OK [${elapsed}s]" | tee -a "$LOG"
  else
    elapsed=$((SECONDS - fam_start))
    FAILED=$((FAILED + 1))
    echo "    FAIL [${elapsed}s]" | tee -a "$LOG"
  fi
done

TOTAL_T=$((SECONDS - START_T))
echo "=== Round 29 sweep done $(date -Iseconds) total=${TOTAL_T}s passed=$PASSED failed=$FAILED ===" | tee -a "$LOG"
echo "ROUND_29_DONE" | tee -a "$LOG"
