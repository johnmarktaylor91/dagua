#!/bin/bash
set -euo pipefail

echo "=== Phase 1: Statistical Analysis ==="
python scripts/fidelity_analysis.py \
    --input eval_output/variant_bench_full \
    --output eval_output/fidelity_report/data

echo "=== Phase 2: Generate Report ==="
python scripts/generate_fidelity_report.py \
    --data eval_output/fidelity_report/data \
    --output eval_output/fidelity_report

echo "=== Phase 3: Compile PDF ==="
if command -v pdflatex &> /dev/null; then
    cd eval_output/fidelity_report
    pdflatex -interaction=nonstopmode report.tex
    pdflatex -interaction=nonstopmode report.tex
    echo "=== Done: eval_output/fidelity_report/report.pdf ==="
else
    echo "=== Done: LaTeX at eval_output/fidelity_report/report.tex ==="
fi
