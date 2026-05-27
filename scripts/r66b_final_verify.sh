#!/usr/bin/env bash
# R66b final verification with 5-minute timeout to auto-skip slow tail.
# Continues from R66 partial state (results.json already has 91% of entries).

set -u
cd "$(dirname "$0")/.."

LOG=/tmp/r66b_final_verify.log
BENCH_OUT=eval_output/benchmark_100seed_final
FIDELITY_OUT=eval_output/fidelity_report_100seed_r66b
SEND=$HOME/.claude/scripts/send-to-jmt.sh

export PATH=/tmp/graphviz_instr/bin:$PATH
export LD_LIBRARY_PATH=/home/jtaylor/anaconda3/envs/py311/lib:${LD_LIBRARY_PATH:-}

exec >> "$LOG" 2>&1
echo ""
echo "=== R66b final verification started $(date -Iseconds) PID=$$ ==="

"$SEND" "R66b restarted with 5min timeout to auto-skip slow tail variants. Continuing from 91% state." || true

MAX_RETRIES=5
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
      --timeout 300 \
      --watchdog-timeout 420; then
    echo "=== Benchmark succeeded $(date -Iseconds) ==="
    SUCCESS=1
    break
  else
    EXIT_CODE=$?
    echo "=== Exit code $EXIT_CODE; retrying in 60s ==="
    sleep 60
  fi
done

if [ $SUCCESS -eq 0 ]; then
  "$SEND" "R66b FAILED after $MAX_RETRIES attempts." || true
  exit 1
fi

"$SEND" "R66b benchmark done. Running fidelity_analysis." || true

echo "--- consolidate_positions_hdf5 $(date -Iseconds) ---"
python3 scripts/consolidate_positions_hdf5.py \
    --input "$BENCH_OUT" \
    --output "$BENCH_OUT/positions.h5" || echo "consolidate failed (continuing)"

echo "--- fidelity_analysis $(date -Iseconds) ---"
mkdir -p "$FIDELITY_OUT/data"
python3 scripts/fidelity_analysis.py \
    --input "$BENCH_OUT" \
    --output "$FIDELITY_OUT/data" || { echo "fidelity_analysis failed"; "$SEND" "R66b fidelity_analysis failed"; }

python3 scripts/generate_fidelity_report.py \
    --input "$FIDELITY_OUT/data" \
    --output "$FIDELITY_OUT/report.md" || true

"$SEND" "R66b FIDELITY REPORT READY: $FIDELITY_OUT/report.md" || true
echo "=== R66b done $(date -Iseconds) ==="
