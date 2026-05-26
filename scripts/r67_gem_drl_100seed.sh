#!/usr/bin/env bash
# R67: 100-seed rerun for gem + drl only. To get TOST statistical equivalence
# data on the 2 engines with documented chaotic floors.
#
# Should be run AFTER R66 completes. R66 produces the 5-seed report for the
# other 22 engines; R67 adds 100-seed TOST data for gem + drl.

set -u
cd "$(dirname "$0")/.."

LOG=/tmp/r67_gem_drl_100seed.log
BENCH_OUT=eval_output/benchmark_100seed_final
FIDELITY_OUT=eval_output/fidelity_report_100seed_r67
SEND=$HOME/.claude/scripts/send-to-jmt.sh

export PATH=/tmp/graphviz_instr/bin:$PATH
export LD_LIBRARY_PATH=/home/jtaylor/anaconda3/envs/py311/lib:${LD_LIBRARY_PATH:-}

exec >> "$LOG" 2>&1
echo ""
echo "=== R67 gem+drl 100-seed rerun started $(date -Iseconds) PID=$$ ==="

"$SEND" "R67 100-seed gem+drl rerun starting. TOST statistical equivalence on the 2 engines with documented chaotic floors." || true

echo "--- r67_gem_drl_100seed purge $(date -Iseconds) ---"
python3 scripts/r67_gem_drl_100seed.py || { echo "purge failed"; exit 1; }

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
      --engines all \
      --timeout 1200 \
      --watchdog-timeout 1500; then
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
  "$SEND" "R67 FAILED after $MAX_RETRIES attempts. See $LOG." || true
  exit 1
fi

"$SEND" "R67 benchmark done. Running fidelity_analysis (TOST + per-variant)." || true

echo "--- consolidate_positions_hdf5 $(date -Iseconds) ---"
python3 scripts/consolidate_positions_hdf5.py \
    --input "$BENCH_OUT" \
    --output "$BENCH_OUT/positions.h5" || echo "consolidate failed (continuing)"

echo "--- fidelity_analysis $(date -Iseconds) ---"
mkdir -p "$FIDELITY_OUT/data"
python3 scripts/fidelity_analysis.py \
    --input "$BENCH_OUT" \
    --output "$FIDELITY_OUT/data" || { echo "fidelity_analysis failed"; "$SEND" "R67 fidelity_analysis failed"; }

python3 scripts/generate_fidelity_report.py \
    --input "$FIDELITY_OUT/data" \
    --output "$FIDELITY_OUT/report.md" || true

"$SEND" "R67 FINAL FIDELITY REPORT (with TOST gem+drl): $FIDELITY_OUT/report.md" || true
echo "=== R67 done $(date -Iseconds) ==="
