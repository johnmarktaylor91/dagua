#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${1:-eval_output/variant_bench_full}"
OUTPUT_DIR="${2:-eval_output/quality_runtime_report}"
WORKERS="${WORKERS:-8}"

mkdir -p "$OUTPUT_DIR"

python scripts/quality_runtime_analysis.py \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --workers "$WORKERS" \
    --verbose

python scripts/generate_quality_runtime_report.py \
    --input "$OUTPUT_DIR" \
    --output "$OUTPUT_DIR/report.md"

echo "QR pipeline complete. Report: $OUTPUT_DIR/report.md"
