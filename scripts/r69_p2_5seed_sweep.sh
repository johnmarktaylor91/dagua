#!/usr/bin/env bash
# R69 P2 -- Stage-1 5-seed fidelity sweep (ALL combos, fidelity-matched variants).
#
# Fresh output dir (NO purge -- the un-matched benchmark_100seed_final is left alone).
# Steps: run_benchmark (5 seeds) -> consolidate positions.h5 -> per-seed Procrustes report.
# Classification (bit-identical / timeout / escalate) is read from stage1/per_variant.json
# by the orchestrator afterwards. NO TOST here (that is P3).
set -u
cd "$(dirname "$0")/.."

LOG=/tmp/r69_p2_5seed.log
BENCH_OUT=eval_output/benchmark_5seed_fidelity
FIDELITY_OUT=eval_output/fidelity_report_r69/stage1
SEND=$HOME/.claude/scripts/send-to-jmt.sh

export LD_LIBRARY_PATH=/home/jtaylor/anaconda3/envs/py311/lib:${LD_LIBRARY_PATH:-}
# R69 P2 speedup (2026-06-01): heavy fidelity ports are single-threaded sequential
# Python loops -- torch intra-op threads do nothing for them but cost cores. Pin each
# worker to 1 thread and run more workers so the timeout-bound tail clears faster.
# Classification-preserving (same 300s timeout, same algorithm).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

exec >> "$LOG" 2>&1
echo ""
echo "=== R69 P2 5-seed sweep started $(date -Iseconds) PID=$$ ==="

# Sanity: variants must be fidelity-matched (P1b) and linlog must not delegate (P1a).
if [ "$(grep -c '"fidelity_mode"' dagua/eval/variants.py)" -lt 80 ]; then
  echo "FATAL: variants.py has < 80 fidelity_mode entries -- P1b not applied?"; "$SEND" "R69 P2 abort: variants not fidelity-matched" || true; exit 1
fi
if grep -q "eval.competitors" dagua/layout/ops/pipelines/linlog.py; then
  echo "FATAL: linlog still delegates -- P1a not applied?"; "$SEND" "R69 P2 abort: linlog still delegates" || true; exit 1
fi

MAX_RETRIES=5; ATTEMPT=0; SUCCESS=0
while [ $ATTEMPT -lt $MAX_RETRIES ]; do
  ATTEMPT=$((ATTEMPT + 1))
  echo "=== run_benchmark attempt $ATTEMPT $(date -Iseconds) ==="
  if python3 scripts/run_benchmark.py \
      --seeds 5 --seed-start 42 --variants \
      --output-dir "$BENCH_OUT" --resume \
      --workers 18 --engines all \
      --timeout 300 --watchdog-timeout 420; then
    echo "=== benchmark OK $(date -Iseconds) ==="; SUCCESS=1; break
  else
    echo "=== exit $?; retry in 60s ==="; sleep 60
  fi
done
if [ $SUCCESS -eq 0 ]; then "$SEND" "R69 P2 FAILED after $MAX_RETRIES attempts." || true; exit 1; fi

echo "--- consolidate positions.h5 $(date -Iseconds) ---"
python3 scripts/consolidate_positions_hdf5.py \
  --input "$BENCH_OUT" --output "$BENCH_OUT/positions.h5" || echo "consolidate failed (continuing)"

echo "--- fast_fidelity_report (per-seed Procrustes) $(date -Iseconds) ---"
mkdir -p "$FIDELITY_OUT"
python3 scripts/fast_fidelity_report.py \
  --results "$BENCH_OUT/results.json" \
  --positions "$BENCH_OUT/positions.h5" \
  --output "$FIDELITY_OUT" \
  --max-seeds 5 --bit-exact-threshold 1e-3 || echo "fast report failed (continuing)"

echo "=== R69 P2 done $(date -Iseconds) ==="
"$SEND" "R69 P2 5-seed sweep COMPLETE. Per-seed Procrustes at $FIDELITY_OUT/report.md -- orchestrator will triage tiers." || true
