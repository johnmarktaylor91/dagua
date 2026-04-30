#!/usr/bin/env bash
# Multi-day supervisor for 100-seed benchmark + post-pipeline.
# Auto-resumes on crash, iMessages at major step boundaries, runs post-
# pipeline when benchmark finishes cleanly.
#
# Usage: nohup bash scripts/supervisor_100seed.sh > /tmp/sup_100seed.log 2>&1 & disown
# Tail:  tail -f /tmp/benchmark_100seed_supervisor.log

set -u
cd "$(dirname "$0")/.."

BENCH_OUT="eval_output/benchmark_100seed_final"
BENCH_LOG="/tmp/benchmark_100seed.log"
PIPELINE_LOG="/tmp/post_pipeline_100seed.log"
SUPERVISOR_LOG="/tmp/benchmark_100seed_supervisor.log"
PROGRESS_FILE="${BENCH_OUT}/progress.json"

mkdir -p "$BENCH_OUT"
exec >> "$SUPERVISOR_LOG" 2>&1

echo ""
echo "=== Supervisor started $(date -Iseconds) PID=$$ ==="

# iMessage start
~/.claude/scripts/send-to-jmt.sh "100-seed benchmark supervisor starting (PID $$). Output: $BENCH_OUT. ETA 4-5 days. Will iMessage on completion." >/dev/null 2>&1 || true

# Benchmark phase: run with auto-restart on crash
MAX_RETRIES=20
ATTEMPT=0
SUCCESS=0
while [ $ATTEMPT -lt $MAX_RETRIES ]; do
  ATTEMPT=$((ATTEMPT + 1))
  echo "=== Benchmark attempt $ATTEMPT $(date -Iseconds) ==="

  if python3 scripts/run_benchmark.py \
      --seeds 100 \
      --variants \
      --output-dir "$BENCH_OUT" \
      --resume \
      --workers 8 \
      --timeout 600 \
      --watchdog-timeout 7200; then
    echo "=== Benchmark succeeded on attempt $ATTEMPT $(date -Iseconds) ==="
    SUCCESS=1
    break
  else
    EXIT_CODE=$?
    echo "=== Benchmark exited with code $EXIT_CODE on attempt $ATTEMPT, retrying in 60s ==="
    if [ $((ATTEMPT % 3)) -eq 0 ]; then
      ~/.claude/scripts/send-to-jmt.sh "100-seed benchmark crashed attempt $ATTEMPT (exit $EXIT_CODE). Auto-retrying with --resume..." >/dev/null 2>&1 || true
    fi
    sleep 60
  fi
done

if [ $SUCCESS -eq 0 ]; then
  ~/.claude/scripts/send-to-jmt.sh "100-seed benchmark FAILED after $MAX_RETRIES attempts. See $BENCH_LOG. Manual intervention needed." >/dev/null 2>&1 || true
  echo "=== Supervisor giving up $(date -Iseconds) ==="
  exit 1
fi

~/.claude/scripts/send-to-jmt.sh "100-seed benchmark DONE. Running post-pipeline (HDF5 consolidate -> fidelity analysis -> quality/runtime). ETA ~30min." >/dev/null 2>&1 || true

echo "=== Post-pipeline phase starting $(date -Iseconds) ==="

# Phase 1: Consolidate positions to HDF5
echo "--- consolidate_positions_hdf5 $(date -Iseconds) ---"
python3 scripts/consolidate_positions_hdf5.py \
    --input "$BENCH_OUT" \
    --output "$BENCH_OUT/positions.h5" >> "$PIPELINE_LOG" 2>&1 || \
    echo "consolidate failed (continuing)"

# Phase 2: Fidelity pipeline
FIDELITY_OUT="eval_output/fidelity_report_100seed_final"
echo "--- fidelity_analysis $(date -Iseconds) ---"
python3 scripts/fidelity_analysis.py \
    --input "$BENCH_OUT" \
    --output "$FIDELITY_OUT/data" >> "$PIPELINE_LOG" 2>&1 || \
    { echo "fidelity_analysis failed"; ~/.claude/scripts/send-to-jmt.sh "fidelity_analysis FAILED. See $PIPELINE_LOG"; }

echo "--- validate_fidelity_output $(date -Iseconds) ---"
python3 scripts/validate_fidelity_output.py \
    --data "$FIDELITY_OUT/data" >> "$PIPELINE_LOG" 2>&1 || true

echo "--- generate_fidelity_report $(date -Iseconds) ---"
python3 scripts/generate_fidelity_report.py \
    --input "$FIDELITY_OUT/data" \
    --output "$FIDELITY_OUT/report.md" >> "$PIPELINE_LOG" 2>&1 || true

~/.claude/scripts/send-to-jmt.sh "Fidelity report generated: $FIDELITY_OUT/report.md. Now running quality/runtime pipeline." >/dev/null 2>&1 || true

# Phase 3: Quality/runtime pipeline
QR_OUT="eval_output/quality_runtime_report_100seed_final"
echo "--- run_quality_runtime_pipeline $(date -Iseconds) ---"
WORKERS=8 bash scripts/run_quality_runtime_pipeline.sh "$BENCH_OUT" "$QR_OUT" >> "$PIPELINE_LOG" 2>&1 || true

# Final iMessage with summary
SUMMARY="$(python3 -c "
import json, os
out = '$BENCH_OUT'
results_path = os.path.join(out, 'results.json')
if os.path.exists(results_path):
    r = json.load(open(results_path))
    n_total = len(r)
    n_ok = sum(1 for v in r.values() if v.get('status') == 'ok')
    n_err = sum(1 for v in r.values() if v.get('status') == 'error')
    n_skip = sum(1 for v in r.values() if v.get('status') == 'skipped')
    print(f'total={n_total} ok={n_ok} err={n_err} skip={n_skip}')
" 2>/dev/null)"

~/.claude/scripts/send-to-jmt.sh "100-seed run COMPLETE. Benchmark: $SUMMARY. Reports: $FIDELITY_OUT/report.md, $QR_OUT/report.md" >/dev/null 2>&1 || true

echo "=== Supervisor done $(date -Iseconds) ==="
