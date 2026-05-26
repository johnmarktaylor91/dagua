#!/usr/bin/env bash
# R66 final verification: purge all R36-R65 affected entries, refill with
# current code (5 seeds, float64, instrumented graphviz), run fidelity_analysis,
# generate report.

set -u
cd "$(dirname "$0")/.."

LOG=/tmp/r66_final_verify.log
BENCH_OUT=eval_output/benchmark_100seed_final
FIDELITY_OUT=eval_output/fidelity_report_100seed_r66
QR_OUT=eval_output/quality_runtime_report_100seed_r66
SEND=$HOME/.claude/scripts/send-to-jmt.sh

# Use instrumented graphviz
export PATH=/tmp/graphviz_instr/bin:$PATH
export LD_LIBRARY_PATH=/home/jtaylor/anaconda3/envs/py311/lib:${LD_LIBRARY_PATH:-}

exec >> "$LOG" 2>&1
echo ""
echo "=== R66 final verification started $(date -Iseconds) PID=$$ ==="

"$SEND" "R66 final verification: 5 seeds, float64, instrumented graphviz, all R36-R65 ports active." || true

echo "--- purge $(date -Iseconds) ---"
python3 scripts/r45_smart_purge.py || { echo "purge failed"; exit 1; }

MAX_RETRIES=10
ATTEMPT=0
SUCCESS=0
while [ $ATTEMPT -lt $MAX_RETRIES ]; do
  ATTEMPT=$((ATTEMPT + 1))
  echo "=== Attempt $ATTEMPT $(date -Iseconds) ==="

  if python3 scripts/run_benchmark.py \
      --seeds 5 --variants \
      --output-dir "$BENCH_OUT" \
      --resume \
      --workers 8 \
      --engines all \
      --timeout 1200 \
      --watchdog-timeout 1500; then
    echo "=== Benchmark phase succeeded $(date -Iseconds) ==="
    SUCCESS=1
    break
  else
    EXIT_CODE=$?
    echo "=== Exit code $EXIT_CODE; retrying in 60s ==="
    sleep 60
  fi
done

if [ $SUCCESS -eq 0 ]; then
  "$SEND" "R66 FAILED after $MAX_RETRIES attempts. See $LOG." || true
  exit 1
fi

"$SEND" "R66 benchmark done. Running fidelity_analysis." || true

echo "--- consolidate_positions_hdf5 $(date -Iseconds) ---"
python3 scripts/consolidate_positions_hdf5.py \
    --input "$BENCH_OUT" \
    --output "$BENCH_OUT/positions.h5" || echo "consolidate failed (continuing)"

echo "--- fidelity_analysis $(date -Iseconds) ---"
mkdir -p "$FIDELITY_OUT/data"
python3 scripts/fidelity_analysis.py \
    --input "$BENCH_OUT" \
    --output "$FIDELITY_OUT/data" || { echo "fidelity_analysis failed"; "$SEND" "R66 fidelity_analysis failed"; }

python3 scripts/validate_fidelity_output.py --data "$FIDELITY_OUT/data" || true

python3 scripts/generate_fidelity_report.py \
    --input "$FIDELITY_OUT/data" \
    --output "$FIDELITY_OUT/report.md" || true

"$SEND" "R66 FIDELITY REPORT READY: $FIDELITY_OUT/report.md" || true
echo "=== R66 done $(date -Iseconds) ==="
