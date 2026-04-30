#!/usr/bin/env bash
# Round 26: 30-seed final verification sweep across all 16 families.
# Confirms Round 25 fixes landed and no regressions occurred.
#
# Output: eval_output/algo_fidelity/round_26/<family>/multi_seed_summary.json
# Usage:  bash scripts/round_26_sweep.sh
# Tail:   tail -f /tmp/round_26_sweep.log

set -u
cd "$(dirname "$0")/.."

LOG=/tmp/round_26_sweep.log
OUT_BASE=eval_output/algo_fidelity/round_26
GRAPHS="linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels"
SEEDS=30
mkdir -p "$OUT_BASE"

# Family -> (dagua_engine, target_engine)
declare -a FAMILIES=(
  "classical_mds:classic_classical_mds:igraph_mds"
  "fa2:classic_fa2:fa2_ref"
  "fmmm:classic_fmmm:ogdf_fmmm"
  "fr:classic_fr:nx_spring"
  "gem:classic_gem:ogdf_gem"
  "kk:classic_kk:nx_kamada_kawai"
  "lgl:classic_lgl:igraph_lgl"
  "maxent_stress:classic_maxent_stress:ogdf_stress"
  "pivot_mds:classic_pivot_mds:ogdf_pivot_mds"
  "rt:classic_rt:igraph_rt"
  "sgd2_multi:classic_sgd2_multi:sgd2_multi_ref"
  "spectral:classic_spectral:nx_spectral"
  "stress_maj:classic_stress_maj:ogdf_stress"
  "stress_sgd:classic_stress_sgd:sgd2"
  "sugiyama:classic_sugiyama:igraph_sugiyama"
  "umap:classic_umap:umap_graph"
)

echo "=== Round 26 verification sweep started $(date -Iseconds) ===" | tee -a "$LOG"
echo "Seeds: $SEEDS" | tee -a "$LOG"
echo "Graphs: $GRAPHS" | tee -a "$LOG"

PASSED=0
FAILED=0
START_T=$SECONDS

for entry in "${FAMILIES[@]}"; do
  IFS=':' read -r fam dagua target <<< "$entry"
  out_dir="$OUT_BASE/$fam"
  rm -rf "$out_dir"
  mkdir -p "$out_dir"
  fam_start=$SECONDS
  echo "--- $fam: $dagua vs $target ---" | tee -a "$LOG"
  if timeout 1800 python3 scripts/algo_fidelity_live_compare.py \
      "$dagua" "$target" \
      --seeds "$SEEDS" \
      --graphs "$GRAPHS" \
      --output-dir "$out_dir" >> "$LOG" 2>&1; then
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
echo "=== Round 26 sweep done $(date -Iseconds) total=${TOTAL_T}s passed=$PASSED failed=$FAILED ===" | tee -a "$LOG"
echo "ROUND_26_DONE" | tee -a "$LOG"
