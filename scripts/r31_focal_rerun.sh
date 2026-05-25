#!/usr/bin/env bash
# R31 focal rerun + fidelity refresh.
# Affected families: umap, lgl, graphopt, neulay, sgd2_multi, davidson_harel.
# Results.json already purged of these (backup at results.json.r31_pre_purge_backup).
# This rerun fills them back with post-R31 code.
#
# Usage: nohup bash scripts/r31_focal_rerun.sh > /dev/null 2>&1 & disown
# Tail:  tail -f /tmp/r31_focal_rerun.log

set -u
cd "$(dirname "$0")/.."

LOG=/tmp/r31_focal_rerun.log
BENCH_OUT=eval_output/benchmark_100seed_final
SEND=$HOME/.claude/scripts/send-to-jmt.sh

# All affected engines (classic side + their R31 paired originals)
ENGINES="classic_umap_default,classic_umap_mindist001,classic_umap_mindist05,classic_umap_nn30,classic_umap_nn5,classic_umap_spread2,classic_lgl_cool1,classic_lgl_cool2,classic_lgl_default,classic_lgl_iter300,classic_lgl_iter50,classic_graphopt_default,classic_graphopt_charge_high,classic_graphopt_charge_low,classic_graphopt_mass_high,classic_graphopt_mass_low,classic_graphopt_spring2,classic_neulay_default,classic_neulay_lr001,classic_neulay_lr05,classic_neulay_no_gcn,classic_neulay_radius02,classic_neulay_radius08,classic_sgd2_multi_default,classic_sgd2_multi_batch8,classic_sgd2_multi_batch128,classic_sgd2_multi_lr001,classic_sgd2_multi_lr01,classic_sgd2_multi_stress_only,classic_sgd2_multi_with_aspect,classic_sgd2_multi_with_crossing,classic_davidson_harel_rounds50,classic_davidson_harel_rounds100,classic_davidson_harel_rounds200"

exec >> "$LOG" 2>&1
echo ""
echo "=== R31 focal rerun started $(date -Iseconds) PID=$$ ==="

"$SEND" "R31 focal rerun starting -- ~34 affected variants x ~95 graphs x 100 seeds. ETA hours-to-day. Will iMessage on completion." || true

# Focal benchmark rerun. Use --resume so entries already in results.json (for non-affected engines)
# are skipped; affected engines were just purged so they'll be refilled.
MAX_RETRIES=10
ATTEMPT=0
SUCCESS=0
while [ $ATTEMPT -lt $MAX_RETRIES ]; do
  ATTEMPT=$((ATTEMPT + 1))
  echo "=== Attempt $ATTEMPT $(date -Iseconds) ==="

  if python3 scripts/run_benchmark.py \
      --seeds 100 --variants \
      --output-dir "$BENCH_OUT" \
      --resume \
      --workers 8 \
      --engines "$ENGINES" \
      --timeout 300 \
      --watchdog-timeout 600; then
    echo "=== Benchmark phase succeeded on attempt $ATTEMPT $(date -Iseconds) ==="
    SUCCESS=1
    break
  else
    EXIT_CODE=$?
    echo "=== Exit code $EXIT_CODE; retrying in 60s ==="
    if [ $((ATTEMPT % 3)) -eq 0 ]; then
      "$SEND" "R31 focal rerun crashed attempt $ATTEMPT (exit $EXIT_CODE). Auto-retrying." || true
    fi
    sleep 60
  fi
done

if [ $SUCCESS -eq 0 ]; then
  "$SEND" "R31 focal rerun FAILED after $MAX_RETRIES attempts. See $LOG." || true
  exit 1
fi

"$SEND" "R31 focal benchmark done. Re-aggregating fidelity report (~50h)." || true

# Refresh HDF5
echo "--- consolidate_positions_hdf5 $(date -Iseconds) ---"
python3 scripts/consolidate_positions_hdf5.py \
    --input "$BENCH_OUT" \
    --output "$BENCH_OUT/positions.h5" || echo "consolidate failed (continuing)"

# Refresh fidelity report
FIDELITY_OUT=eval_output/fidelity_report_100seed_r31
echo "--- fidelity_analysis $(date -Iseconds) ---"
export LD_LIBRARY_PATH=/home/jtaylor/anaconda3/envs/py311/lib:${LD_LIBRARY_PATH:-}

python3 scripts/fidelity_analysis.py \
    --input "$BENCH_OUT" \
    --output "$FIDELITY_OUT/data" || { echo "fidelity_analysis failed"; "$SEND" "R31 fidelity_analysis failed -- see $LOG"; exit 1; }

python3 scripts/validate_fidelity_output.py --data "$FIDELITY_OUT/data" || true

python3 scripts/generate_fidelity_report.py \
    --input "$FIDELITY_OUT/data" \
    --output "$FIDELITY_OUT/report.md" || true

"$SEND" "R31 COMPLETE. Updated fidelity report: $FIDELITY_OUT/report.md" || true
echo "=== R31 focal rerun done $(date -Iseconds) ==="
