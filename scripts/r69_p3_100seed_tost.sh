#!/usr/bin/env bash
# R69 P3 -- 100-seed escalation run + TOST + combined 4-tier report.
#
# Runs ONLY the escalation subset: variants that at 5 seeds were NOT bit-identical
# AND did NOT time out (the chaotic-but-completing engines). The engine list (the
# classic_* variants AND their paired reference adapters, comma-separated) must be in
# /tmp/r69_escalation_engines.txt, written by the triage step.
#
# These complete fast (they finish, just differ by basin), so unlike P2 this is quick.
set -u
cd "$(dirname "$0")/.."

LOG=/tmp/r69_p3_100seed.log
ENGINE_FILE=/tmp/r69_escalation_engines.txt
BENCH_OUT=eval_output/benchmark_100seed_escalation
STAGE1=eval_output/fidelity_report_r69/stage1
TOST_OUT=eval_output/fidelity_report_r69/tost
REPORT=eval_output/fidelity_report_r69/report.md
SEND=$HOME/.claude/scripts/send-to-jmt.sh

export LD_LIBRARY_PATH=/home/jtaylor/anaconda3/envs/py311/lib:${LD_LIBRARY_PATH:-}
# carry the P2 speedup: heavy fidelity ports are single-threaded; pin threads, more workers
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

exec >> "$LOG" 2>&1
echo ""
echo "=== R69 P3 100-seed escalation + TOST started $(date -Iseconds) PID=$$ ==="

if [ ! -s "$ENGINE_FILE" ]; then
  echo "FATAL: $ENGINE_FILE missing/empty -- triage must write the escalation engine list first"
  "$SEND" "R69 P3 abort: no escalation engine list" || true; exit 1
fi
ENGINES="$(cat "$ENGINE_FILE")"
echo "escalation engines: $ENGINES"

MAX_RETRIES=5; ATTEMPT=0; SUCCESS=0
while [ $ATTEMPT -lt $MAX_RETRIES ]; do
  ATTEMPT=$((ATTEMPT + 1))
  echo "=== run_benchmark attempt $ATTEMPT $(date -Iseconds) ==="
  if python3 scripts/run_benchmark.py \
      --seeds 100 --seed-start 42 --variants \
      --engines "$ENGINES" \
      --output-dir "$BENCH_OUT" --resume \
      --workers 18 --timeout 300 --watchdog-timeout 420; then
    echo "=== benchmark OK $(date -Iseconds) ==="; SUCCESS=1; break
  else
    echo "=== exit $?; retry in 60s ==="; sleep 60
  fi
done
if [ $SUCCESS -eq 0 ]; then "$SEND" "R69 P3 FAILED after $MAX_RETRIES attempts." || true; exit 1; fi

echo "--- consolidate positions.h5 $(date -Iseconds) ---"
python3 scripts/consolidate_positions_hdf5.py \
  --input "$BENCH_OUT" --output "$BENCH_OUT/positions.h5" || echo "consolidate failed (continuing)"

echo "--- TOST equivalence on escalation variants $(date -Iseconds) ---"
mkdir -p "$TOST_OUT"
python3 scripts/r68_tost_followup.py \
  --per-variant-json "$STAGE1/per_variant.json" \
  --results "$BENCH_OUT/results.json" \
  --positions "$BENCH_OUT/positions.h5" \
  --output "$TOST_OUT" || echo "TOST failed (continuing)"

echo "--- combined report $(date -Iseconds) ---"
python3 scripts/r68_combined_report.py \
  --per-seed "$STAGE1" \
  --tost "$TOST_OUT" \
  --output "$REPORT" || echo "combined report failed (continuing)"

echo "=== R69 P3 done $(date -Iseconds) ==="
"$SEND" "R69 P3 COMPLETE (100-seed + TOST). Combined report: $REPORT -- orchestrator will finalize 4-tier verdict (P4)." || true
