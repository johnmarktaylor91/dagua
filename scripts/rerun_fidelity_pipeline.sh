#!/usr/bin/env bash
# Re-run the post-benchmark fidelity + quality_runtime pipelines that crashed
# on 2026-05-21 due to a transient llvmlite/numba load failure.
# HDF5 consolidate output already exists at eval_output/benchmark_100seed_final/positions.h5

set -u
cd "$(dirname "$0")/.."

BENCH_OUT=eval_output/benchmark_100seed_final
FIDELITY_OUT=eval_output/fidelity_report_100seed_final
QR_OUT=eval_output/quality_runtime_report_100seed_final
LOG=/tmp/rerun_fidelity_pipeline.log

exec >> "$LOG" 2>&1
echo ""
echo "=== rerun_fidelity_pipeline started $(date -Iseconds) ==="

~/.claude/scripts/send-to-jmt.sh "Re-running fidelity + quality_runtime pipelines (the post-benchmark crashed on llvmlite/numba load -- now resolved). Output: $FIDELITY_OUT/report.md and $QR_OUT/report.md" || true

# Phase 1: Fidelity pipeline
echo "--- fidelity_analysis $(date -Iseconds) ---"
rm -rf "$FIDELITY_OUT/data"
mkdir -p "$FIDELITY_OUT/data"
if python3 scripts/fidelity_analysis.py \
    --input "$BENCH_OUT" \
    --output "$FIDELITY_OUT/data"; then
    echo "  fidelity_analysis OK"
else
    echo "  fidelity_analysis FAILED"
    ~/.claude/scripts/send-to-jmt.sh "fidelity_analysis FAILED again. See $LOG" || true
    exit 1
fi

echo "--- validate_fidelity_output $(date -Iseconds) ---"
python3 scripts/validate_fidelity_output.py --data "$FIDELITY_OUT/data" || true

echo "--- generate_fidelity_report $(date -Iseconds) ---"
python3 scripts/generate_fidelity_report.py \
    --input "$FIDELITY_OUT/data" \
    --output "$FIDELITY_OUT/report.md" || true

if [ -s "$FIDELITY_OUT/report.md" ]; then
    echo "  fidelity report: $FIDELITY_OUT/report.md ($(wc -l < "$FIDELITY_OUT/report.md") lines)"
    ~/.claude/scripts/send-to-jmt.sh "Fidelity report ready: $FIDELITY_OUT/report.md ($(wc -l < "$FIDELITY_OUT/report.md") lines). Now running quality/runtime pipeline." || true
fi

# Phase 2: Quality/Runtime pipeline
echo "--- run_quality_runtime_pipeline $(date -Iseconds) ---"
if WORKERS=8 bash scripts/run_quality_runtime_pipeline.sh "$BENCH_OUT" "$QR_OUT"; then
    echo "  quality_runtime OK"
else
    echo "  quality_runtime FAILED (continuing)"
fi

echo "=== rerun_fidelity_pipeline done $(date -Iseconds) ==="
~/.claude/scripts/send-to-jmt.sh "Post-benchmark pipeline COMPLETE. Reports: $FIDELITY_OUT/report.md , $QR_OUT/report.md" || true
