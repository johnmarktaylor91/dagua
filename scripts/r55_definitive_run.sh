#!/usr/bin/env bash
# R55 DEFINITIVE RUN: full 100-seed benchmark, float64 default (already in
# place via R44), every classic_* engine refilled with current bit-exact code.
# Original (reference) engine outputs left as-is per JMT directive.
#
# Compressed timeouts (60s/120s) and 3-consecutive-skip rule should drain
# the slow tail faster than R35/R42 (which used 300s).

set -u
cd "$(dirname "$0")/.."

LOG=/tmp/r55_definitive_run.log
BENCH_OUT=eval_output/benchmark_100seed_final
FIDELITY_OUT=eval_output/fidelity_report_100seed_r55
QR_OUT=eval_output/quality_runtime_report_100seed_r55
SEND=$HOME/.claude/scripts/send-to-jmt.sh

exec >> "$LOG" 2>&1
echo ""
echo "=== R55 DEFINITIVE RUN started $(date -Iseconds) PID=$$ ==="

"$SEND" "R55 definitive run starting. Full 100 seeds, float64 fidelity, every classic_* refilled with R36-R53 bit-exact code. ETA hours to ~1.5 days." || true

# Phase A: purge (reuse R45 smart purge -- purges classic_* + re-paired refs)
echo "--- r45_smart_purge $(date -Iseconds) ---"
python3 scripts/r45_smart_purge.py || { echo "purge failed"; exit 1; }

# Phase B: benchmark refill with --seeds 100 + 60s timeout
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
      --timeout 60 \
      --watchdog-timeout 120; then
    echo "=== Benchmark phase succeeded on attempt $ATTEMPT $(date -Iseconds) ==="
    SUCCESS=1
    break
  else
    EXIT_CODE=$?
    echo "=== Exit code $EXIT_CODE; retrying in 60s ==="
    if [ $((ATTEMPT % 3)) -eq 0 ]; then
      "$SEND" "R55 rerun crashed attempt $ATTEMPT (exit $EXIT_CODE). Auto-retrying." || true
    fi
    sleep 60
  fi
done

if [ $SUCCESS -eq 0 ]; then
  "$SEND" "R55 FAILED after $MAX_RETRIES attempts. See $LOG." || true
  exit 1
fi

"$SEND" "R55 benchmark done. Running fidelity_analysis." || true

# Phase C
echo "--- consolidate_positions_hdf5 $(date -Iseconds) ---"
export LD_LIBRARY_PATH=/home/jtaylor/anaconda3/envs/py311/lib:${LD_LIBRARY_PATH:-}

python3 scripts/consolidate_positions_hdf5.py \
    --input "$BENCH_OUT" \
    --output "$BENCH_OUT/positions.h5" || echo "consolidate failed (continuing)"

echo "--- fidelity_analysis $(date -Iseconds) ---"
mkdir -p "$FIDELITY_OUT/data"
python3 scripts/fidelity_analysis.py \
    --input "$BENCH_OUT" \
    --output "$FIDELITY_OUT/data" || { echo "fidelity_analysis failed"; "$SEND" "R55 fidelity_analysis failed -- see $LOG"; }

python3 scripts/validate_fidelity_output.py --data "$FIDELITY_OUT/data" || true

python3 scripts/generate_fidelity_report.py \
    --input "$FIDELITY_OUT/data" \
    --output "$FIDELITY_OUT/report.md" || true

"$SEND" "R55 fidelity report ready: $FIDELITY_OUT/report.md. Running QR." || true

echo "--- run_quality_runtime_pipeline $(date -Iseconds) ---"
WORKERS=8 bash scripts/run_quality_runtime_pipeline.sh "$BENCH_OUT" "$QR_OUT" || echo "QR pipeline failed (continuing)"

"$SEND" "R55 DEFINITIVE COMPLETE. Fidelity: $FIDELITY_OUT/report.md . QR: $QR_OUT/report.md" || true
echo "=== R55 done $(date -Iseconds) ==="
