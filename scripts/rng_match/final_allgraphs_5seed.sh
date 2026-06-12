#!/usr/bin/env bash
# RNG-matching FINAL: 5-seed sweep across ALL graphs x ALL engines, on the current
# matched-params + RNG-ported + OGDF-rebuilt code. Classifies every (graph, algo) combo.
# Fresh output dir (the old benchmark_5seed_fidelity predates the RNG-matching sprint).
set -u
cd "$(dirname "$0")/../.."

LOG=/tmp/rng_final_allgraphs.log
BENCH=eval_output/benchmark_5seed_final
REPORT=eval_output/fidelity_report_final
SEND=$HOME/.claude/scripts/send-to-jmt.sh

export LD_LIBRARY_PATH=/home/jtaylor/anaconda3/envs/py311/lib:${LD_LIBRARY_PATH:-}
# carry the speedup: heavy fidelity ports are single-threaded sequential
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

exec >> "$LOG" 2>&1
echo ""
echo "=== RNG-match FINAL all-graphs 5-seed sweep started $(date -Iseconds) PID=$$ ==="

# sanity: matched params present (param-matching landed) + linlog not delegating
if [ "$(grep -c '"fidelity_mode"' dagua/eval/variants.py)" -lt 80 ]; then
  echo "FATAL: variants.py not fidelity-matched"; "$SEND" "RNG final abort: variants not matched" || true; exit 1
fi

MAX_RETRIES=6; ATTEMPT=0; SUCCESS=0
while [ $ATTEMPT -lt $MAX_RETRIES ]; do
  ATTEMPT=$((ATTEMPT+1)); echo "=== run_benchmark attempt $ATTEMPT $(date -Iseconds) ==="
  if python3 scripts/run_benchmark.py \
      --seeds 5 --seed-start 42 --variants --engines all \
      --output-dir "$BENCH" --resume \
      --workers 18 --timeout 300 --watchdog-timeout 420; then
    echo "=== benchmark OK $(date -Iseconds) ==="; SUCCESS=1; break
  else echo "=== exit $?; retry 60s ==="; sleep 60; fi
done
[ $SUCCESS -eq 0 ] && { "$SEND" "RNG final FAILED after $MAX_RETRIES attempts" || true; exit 1; }

echo "--- consolidate $(date -Iseconds) ---"
python3 scripts/consolidate_positions_hdf5.py --input "$BENCH" --output "$BENCH/positions.h5" || echo "consolidate failed (continuing)"

echo "--- per-(graph,seed) Procrustes report $(date -Iseconds) ---"
mkdir -p "$REPORT"
python3 scripts/fast_fidelity_report.py \
  --results "$BENCH/results.json" --positions "$BENCH/positions.h5" \
  --output "$REPORT" --max-seeds 5 --bit-exact-threshold 1e-3 || echo "report failed (continuing)"

echo "=== RNG-match FINAL done $(date -Iseconds) ==="
"$SEND" "RNG-match FINAL all-graphs 5-seed sweep COMPLETE. Per-(graph,algo) bit-exact classification at $REPORT/report.md -- orchestrator will summarize which graphs/algos diverge." || true
