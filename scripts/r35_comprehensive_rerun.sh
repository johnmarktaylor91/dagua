#!/usr/bin/env bash
# R35 comprehensive focal rerun.
# Results.json already purged of R31-R35 affected entries
# (scripts/r35_purge_comprehensive.py). This rerun fills them back with
# current post-R35 code, then auto-runs fidelity + QR pipelines.

set -u
cd "$(dirname "$0")/.."

LOG=/tmp/r35_comprehensive_rerun.log
BENCH_OUT=eval_output/benchmark_100seed_final
FIDELITY_OUT=eval_output/fidelity_report_100seed_r35
QR_OUT=eval_output/quality_runtime_report_100seed_r35
SEND=$HOME/.claude/scripts/send-to-jmt.sh

exec >> "$LOG" 2>&1
echo ""
echo "=== R35 comprehensive rerun started $(date -Iseconds) PID=$$ ==="

"$SEND" "R35 comprehensive rerun starting. ~828k purged entries to refill with current code. Then fidelity_analysis + QR pipelines auto-run. ETA hours to ~1 day." || true

# Benchmark with --resume + --engines all = skip unaffected (740k), refill purged (828k)
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
      --timeout 300 \
      --watchdog-timeout 600; then
    echo "=== Benchmark phase succeeded on attempt $ATTEMPT $(date -Iseconds) ==="
    SUCCESS=1
    break
  else
    EXIT_CODE=$?
    echo "=== Exit code $EXIT_CODE; retrying in 60s ==="
    if [ $((ATTEMPT % 3)) -eq 0 ]; then
      "$SEND" "R35 rerun crashed attempt $ATTEMPT (exit $EXIT_CODE). Auto-retrying." || true
    fi
    sleep 60
  fi
done

if [ $SUCCESS -eq 0 ]; then
  "$SEND" "R35 rerun FAILED after $MAX_RETRIES attempts. See $LOG." || true
  exit 1
fi

"$SEND" "R35 benchmark done. Re-aggregating fidelity + QR (~50-60h)." || true

# Refresh HDF5
echo "--- consolidate_positions_hdf5 $(date -Iseconds) ---"
export LD_LIBRARY_PATH=/home/jtaylor/anaconda3/envs/py311/lib:${LD_LIBRARY_PATH:-}

python3 scripts/consolidate_positions_hdf5.py \
    --input "$BENCH_OUT" \
    --output "$BENCH_OUT/positions.h5" || echo "consolidate failed (continuing)"

# Refresh fidelity report
echo "--- fidelity_analysis $(date -Iseconds) ---"
mkdir -p "$FIDELITY_OUT/data"
python3 scripts/fidelity_analysis.py \
    --input "$BENCH_OUT" \
    --output "$FIDELITY_OUT/data" || { echo "fidelity_analysis failed"; "$SEND" "R35 fidelity_analysis failed -- see $LOG"; }

python3 scripts/validate_fidelity_output.py --data "$FIDELITY_OUT/data" || true

python3 scripts/generate_fidelity_report.py \
    --input "$FIDELITY_OUT/data" \
    --output "$FIDELITY_OUT/report.md" || true

"$SEND" "R35 fidelity report ready: $FIDELITY_OUT/report.md. Running QR pipeline now." || true

# Refresh QR pipeline
echo "--- run_quality_runtime_pipeline $(date -Iseconds) ---"
WORKERS=8 bash scripts/run_quality_runtime_pipeline.sh "$BENCH_OUT" "$QR_OUT" || echo "QR pipeline failed (continuing)"

"$SEND" "R35 COMPLETE. Fidelity: $FIDELITY_OUT/report.md , QR: $QR_OUT/report.md" || true
echo "=== R35 comprehensive rerun done $(date -Iseconds) ==="
