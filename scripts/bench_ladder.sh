#!/usr/bin/env bash
# Benchmark ladder: 10 → 1B nodes, sequential, with timing.
# Writes results to eval_output/ladder_results.csv
set -euo pipefail

RESULTS="eval_output/ladder_results.csv"
mkdir -p eval_output
echo "nodes,time_seconds,status" > "$RESULTS"

SIZES=(10 20 50 100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000 1000000 2000000 5000000 10000000 20000000 50000000 100000000 200000000 500000000 1000000000)

for SIZE in "${SIZES[@]}"; do
    echo "=== $SIZE nodes ==="
    START=$(date +%s.%N)
    if stdbuf -oL python scripts/bench_large.py "$SIZE" 2>&1 | tee /tmp/dagua_ladder_${SIZE}.log; then
        END=$(date +%s.%N)
        ELAPSED=$(echo "$END - $START" | bc)
        echo "$SIZE,$ELAPSED,ok" >> "$RESULTS"
        echo "=== $SIZE nodes: ${ELAPSED}s OK ==="
    else
        END=$(date +%s.%N)
        ELAPSED=$(echo "$END - $START" | bc)
        echo "$SIZE,$ELAPSED,FAILED" >> "$RESULTS"
        echo "=== $SIZE nodes: FAILED after ${ELAPSED}s ==="
        echo "FAILED at $SIZE nodes. Stopping ladder."
        exit 1
    fi
done

echo "=== LADDER COMPLETE ==="
cat "$RESULTS"
