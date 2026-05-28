#!/usr/bin/env bash
# R68 100-seed benchmark + TOST analysis pipeline.
#
# Prerequisites (RUN FIRST, separately, via codex):
#   ~/.claude/scripts/codex-bg.sh /tmp/r68_variants.log /tmp/PROMPT_68_variant_fidelity_mode.md \
#     --cd /home/jtaylor/projects/dagua --sandbox danger-full-access \
#     -c model_reasoning_effort=high
#
# That codex patches dagua/eval/variants.py to add fidelity_mode to every
# benchmark variant so the bit-exact ports actually run.
#
# Then THIS script:
#   1. Purges affected classic_* + reference rows from results.json
#   2. Reruns benchmark with --seeds 100
#   3. Refreshes positions.h5
#   4. Runs fast_fidelity_report.py for the per-seed Procrustes verdict
#   5. For variants where max RMSD >= 1e-3, runs the legacy fidelity_analysis.py
#      with TOST aggregation (only on those variants, not all 63)
#   6. Combined report at eval_output/fidelity_report_r68/

set -u
cd "$(dirname "$0")/.."

LOG=/tmp/r68_100seed_run.log
BENCH_OUT=eval_output/benchmark_100seed_final
FIDELITY_OUT=eval_output/fidelity_report_r68
SEND=$HOME/.claude/scripts/send-to-jmt.sh

export PATH=/tmp/graphviz_instr/bin:$PATH
export LD_LIBRARY_PATH=/home/jtaylor/anaconda3/envs/py311/lib:${LD_LIBRARY_PATH:-}

exec >> "$LOG" 2>&1
echo ""
echo "=== R68 100-seed + TOST started $(date -Iseconds) PID=$$ ==="

# Sanity check: R68 codex must have patched variants.py first
if ! grep -q '"fidelity_mode"' dagua/eval/variants.py; then
  echo "FATAL: dagua/eval/variants.py does not contain 'fidelity_mode' -- run the R68 codex first"
  "$SEND" "R68 failed: variants.py not patched. Run /tmp/PROMPT_68_variant_fidelity_mode.md codex first" || true
  exit 1
fi

"$SEND" "R68 100-seed + TOST starting. variants.py patched to use bit-exact ports." || true

echo "--- purge $(date -Iseconds) ---"
python3 scripts/r45_smart_purge.py || { echo "purge failed"; exit 1; }

MAX_RETRIES=5
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
  "$SEND" "R68 FAILED after $MAX_RETRIES attempts." || true
  exit 1
fi

"$SEND" "R68 benchmark done. Consolidating + per-seed Procrustes." || true

echo "--- consolidate_positions_hdf5 $(date -Iseconds) ---"
python3 scripts/consolidate_positions_hdf5.py \
    --input "$BENCH_OUT" \
    --output "$BENCH_OUT/positions.h5" || echo "consolidate failed (continuing)"

echo "--- fast_fidelity_report (per-seed Procrustes) $(date -Iseconds) ---"
mkdir -p "$FIDELITY_OUT"
python3 scripts/fast_fidelity_report.py \
    --results "$BENCH_OUT/results.json" \
    --positions "$BENCH_OUT/positions.h5" \
    --output "$FIDELITY_OUT/per_seed" \
    --max-seeds 100 \
    --bit-exact-threshold 1e-3 || echo "fast report failed (continuing)"

echo "--- TOST on non-bit-exact variants $(date -Iseconds) ---"
# Identify variants where max RMSD >= 1e-3 from the fast report, run TOST on them
python3 scripts/r68_tost_followup.py \
    --per-variant-json "$FIDELITY_OUT/per_seed/per_variant.json" \
    --results "$BENCH_OUT/results.json" \
    --positions "$BENCH_OUT/positions.h5" \
    --output "$FIDELITY_OUT/tost" || echo "TOST followup failed (continuing)"

# Combined report
echo "--- combined report $(date -Iseconds) ---"
python3 scripts/r68_combined_report.py \
    --per-seed "$FIDELITY_OUT/per_seed" \
    --tost "$FIDELITY_OUT/tost" \
    --output "$FIDELITY_OUT/report.md" || echo "combined report failed (continuing)"

"$SEND" "R68 COMPLETE. Combined fidelity report: $FIDELITY_OUT/report.md" || true
echo "=== R68 done $(date -Iseconds) ==="
