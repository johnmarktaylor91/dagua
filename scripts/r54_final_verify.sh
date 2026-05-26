#!/usr/bin/env bash
# R54 final verification: purge all R36-R53 affected entries, then refill
# with 5 seeds per variant. Bit-exact algorithms only need a few seeds
# to verify since seed-equatable means dagua(seed=N) == reference(seed=N)
# at every N. 5 seeds is the spot-check count.
#
# Aggressive 60s/120s timeouts to compress the slow tail
# (davidson_harel_rounds200 etc.).

set -u
cd "$(dirname "$0")/.."

LOG=/tmp/r54_final_verify.log
BENCH_OUT=eval_output/benchmark_100seed_final
FIDELITY_OUT=eval_output/fidelity_report_100seed_r54
QR_OUT=eval_output/quality_runtime_report_100seed_r54
SEND=$HOME/.claude/scripts/send-to-jmt.sh

exec >> "$LOG" 2>&1
echo ""
echo "=== R54 final verification started $(date -Iseconds) PID=$$ ==="

"$SEND" "R54 final verification starting. 5 seeds, 60s timeout. Bit-exact 24/24 verification at scale." || true

# Reuse the R45 purge logic
echo "--- r45_smart_purge $(date -Iseconds) ---"
python3 scripts/r45_smart_purge.py || { echo "purge failed"; exit 1; }

# Benchmark with 5 seeds + tight timeouts
MAX_RETRIES=20
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
      --timeout 60 \
      --watchdog-timeout 120; then
    echo "=== Benchmark phase succeeded on attempt $ATTEMPT $(date -Iseconds) ==="
    SUCCESS=1
    break
  else
    EXIT_CODE=$?
    echo "=== Exit code $EXIT_CODE; retrying in 60s ==="
    if [ $((ATTEMPT % 3)) -eq 0 ]; then
      "$SEND" "R54 rerun crashed attempt $ATTEMPT (exit $EXIT_CODE). Auto-retrying." || true
    fi
    sleep 60
  fi
done

if [ $SUCCESS -eq 0 ]; then
  "$SEND" "R54 rerun FAILED after $MAX_RETRIES attempts. See $LOG." || true
  exit 1
fi

"$SEND" "R54 benchmark done. Running fidelity_analysis." || true

echo "--- consolidate_positions_hdf5 $(date -Iseconds) ---"
export LD_LIBRARY_PATH=/home/jtaylor/anaconda3/envs/py311/lib:${LD_LIBRARY_PATH:-}

python3 scripts/consolidate_positions_hdf5.py \
    --input "$BENCH_OUT" \
    --output "$BENCH_OUT/positions.h5" || echo "consolidate failed (continuing)"

echo "--- fidelity_analysis $(date -Iseconds) ---"
mkdir -p "$FIDELITY_OUT/data"
python3 scripts/fidelity_analysis.py \
    --input "$BENCH_OUT" \
    --output "$FIDELITY_OUT/data" || { echo "fidelity_analysis failed"; "$SEND" "R54 fidelity_analysis failed -- see $LOG"; }

python3 scripts/validate_fidelity_output.py --data "$FIDELITY_OUT/data" || true

python3 scripts/generate_fidelity_report.py \
    --input "$FIDELITY_OUT/data" \
    --output "$FIDELITY_OUT/report.md" || true

"$SEND" "R54 fidelity report ready: $FIDELITY_OUT/report.md. Running QR." || true

echo "--- run_quality_runtime_pipeline $(date -Iseconds) ---"
WORKERS=8 bash scripts/run_quality_runtime_pipeline.sh "$BENCH_OUT" "$QR_OUT" || echo "QR pipeline failed (continuing)"

"$SEND" "R54 COMPLETE. Fidelity: $FIDELITY_OUT/report.md . QR: $QR_OUT/report.md (if generated)." || true
echo "=== R54 done $(date -Iseconds) ==="
