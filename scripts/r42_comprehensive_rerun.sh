#!/usr/bin/env bash
# R42 comprehensive rerun.
# Purges every R36-R41-affected classic_* + re-paired references, then refills
# under current code, then auto-runs fidelity_analysis + QR pipelines.

set -u
cd "$(dirname "$0")/.."

LOG=/tmp/r42_comprehensive_rerun.log
BENCH_OUT=eval_output/benchmark_100seed_final
FIDELITY_OUT=eval_output/fidelity_report_100seed_r42
QR_OUT=eval_output/quality_runtime_report_100seed_r42
SEND=$HOME/.claude/scripts/send-to-jmt.sh

exec >> "$LOG" 2>&1
echo ""
echo "=== R42 comprehensive rerun started $(date -Iseconds) PID=$$ ==="

"$SEND" "R42 comprehensive rerun starting. Purging all R36-R41-affected entries then refilling under current code. ETA hours to ~1 day." || true

# Phase A: purge
echo "--- r42_comprehensive_purge $(date -Iseconds) ---"
python3 scripts/r42_comprehensive_purge.py || { echo "purge failed"; exit 1; }

# Phase B: benchmark refill with --resume + --engines all = skip unaffected, refill purged
MAX_RETRIES=20
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
      --engines all \
      --timeout 180 \
      --watchdog-timeout 360; then
    echo "=== Benchmark phase succeeded on attempt $ATTEMPT $(date -Iseconds) ==="
    SUCCESS=1
    break
  else
    EXIT_CODE=$?
    echo "=== Exit code $EXIT_CODE; retrying in 60s ==="
    if [ $((ATTEMPT % 3)) -eq 0 ]; then
      "$SEND" "R42 rerun crashed attempt $ATTEMPT (exit $EXIT_CODE). Auto-retrying." || true
    fi
    sleep 60
  fi
done

if [ $SUCCESS -eq 0 ]; then
  "$SEND" "R42 rerun FAILED after $MAX_RETRIES attempts. See $LOG." || true
  exit 1
fi

"$SEND" "R42 benchmark done. Re-aggregating fidelity + QR." || true

# Phase C: refresh HDF5
echo "--- consolidate_positions_hdf5 $(date -Iseconds) ---"
export LD_LIBRARY_PATH=/home/jtaylor/anaconda3/envs/py311/lib:${LD_LIBRARY_PATH:-}

python3 scripts/consolidate_positions_hdf5.py \
    --input "$BENCH_OUT" \
    --output "$BENCH_OUT/positions.h5" || echo "consolidate failed (continuing)"

# Phase D: refresh fidelity report
echo "--- fidelity_analysis $(date -Iseconds) ---"
mkdir -p "$FIDELITY_OUT/data"
python3 scripts/fidelity_analysis.py \
    --input "$BENCH_OUT" \
    --output "$FIDELITY_OUT/data" || { echo "fidelity_analysis failed"; "$SEND" "R42 fidelity_analysis failed -- see $LOG"; }

python3 scripts/validate_fidelity_output.py --data "$FIDELITY_OUT/data" || true

python3 scripts/generate_fidelity_report.py \
    --input "$FIDELITY_OUT/data" \
    --output "$FIDELITY_OUT/report.md" || true

"$SEND" "R42 fidelity report ready: $FIDELITY_OUT/report.md. Running QR pipeline now." || true

# Phase E: refresh QR pipeline
echo "--- run_quality_runtime_pipeline $(date -Iseconds) ---"
WORKERS=8 bash scripts/run_quality_runtime_pipeline.sh "$BENCH_OUT" "$QR_OUT" || echo "QR pipeline failed (continuing)"

"$SEND" "R42 COMPLETE. Fidelity: $FIDELITY_OUT/report.md , QR: $QR_OUT/report.md" || true
echo "=== R42 comprehensive rerun done $(date -Iseconds) ==="
