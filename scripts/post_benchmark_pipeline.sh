#!/bin/bash
# One-shot post-benchmark pipeline. Scheduled for 4am 2026-04-16.
# Self-removes its crontab entry when done.
set -euo pipefail

# Use conda py311 env explicitly -- cron's minimal PATH finds /usr/bin/python (2.7) otherwise
export PATH="/home/jtaylor/anaconda3/envs/py311/bin:$PATH"

cd /home/jtaylor/projects/dagua
LOG="eval_output/post_benchmark_pipeline.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== Post-benchmark pipeline started at $(date) ==="

# Step 1: Verify benchmark is done
RUNNING=$(python3 -c "
import json
r = json.load(open('eval_output/variant_bench_full/results.json'))
print(sum(1 for v in r.values() if v.get('status') == 'running'))
")

if [ "$RUNNING" -ne 0 ]; then
    echo "ABORT: Benchmark still running ($RUNNING remaining). Exiting."
    exit 1
fi

echo "=== Benchmark complete. Starting consolidation at $(date) ==="

# Step 2: Consolidate positions to HDF5
python scripts/consolidate_positions_hdf5.py \
    --input eval_output/variant_bench_full \
    --output eval_output/variant_bench_full/positions.h5

echo "=== HDF5 consolidation done at $(date). Starting fidelity pipeline ==="

# Step 3: Fidelity pipeline
./scripts/run_fidelity_pipeline.sh

echo "=== Fidelity pipeline done at $(date). Starting quality/runtime pipeline ==="

# Step 4: Quality/runtime pipeline
./scripts/run_quality_runtime_pipeline.sh

echo "=== All pipelines complete at $(date) ==="

# Remove the crontab entry
crontab -l 2>/dev/null | grep -v 'post_benchmark_pipeline' | crontab - || true

echo "=== Crontab entry removed. Pipeline finished. ==="
