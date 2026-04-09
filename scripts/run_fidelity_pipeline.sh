#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${1:-eval_output/variant_bench_full}"
OUTPUT_DIR="${2:-eval_output/fidelity_report}"

# Phase 1: run analysis
python scripts/fidelity_analysis.py \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR/data"

# Phase 2: validate output
python scripts/validate_fidelity_output.py \
    --data "$OUTPUT_DIR/data"

# Phase 3: generate markdown report
python scripts/generate_fidelity_report.py \
    --input "$OUTPUT_DIR/data" \
    --output "$OUTPUT_DIR/report.md"

echo "Fidelity pipeline complete. Report: $OUTPUT_DIR/report.md"
