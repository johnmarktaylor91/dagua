#!/usr/bin/env bash
# bench_scaling_ladder.sh — Run bench_large.py at 200M, 500M, 1B, 1.5B, 2B sequentially.
# Uses --resume to pick up from existing checkpoints.
# Resource checks between runs to avoid OOM.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

SIZES=("50000000" "100000000" "200000000" "500000000" "1000000000" "1500000000" "2000000000")
LABELS=("50M" "100M" "200M" "500M" "1B" "1.5B" "2B")
MIN_RAM_GB=32

check_resources() {
    local avail_kb
    avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
    local avail_gb=$(( avail_kb / 1048576 ))
    echo "[resource] Available RAM: ${avail_gb}GB"
    if [ "$avail_gb" -lt "$MIN_RAM_GB" ]; then
        echo "ABORT: Only ${avail_gb}GB available, need at least ${MIN_RAM_GB}GB. Skipping remaining runs."
        return 1
    fi
    return 0
}

echo "=== Scaling Ladder: ${LABELS[*]} ==="
echo "Started: $(date -Iseconds)"
echo ""

for i in "${!SIZES[@]}"; do
    size="${SIZES[$i]}"
    label="${LABELS[$i]}"

    echo "──────────────────────────────────────"
    echo "[${label}] Starting at $(date -Iseconds)"

    if ! check_resources; then
        echo "[${label}] SKIPPED — insufficient RAM"
        continue
    fi

    echo "[${label}] Running: python scripts/bench_large.py ${size} --resume --fast-final --device cuda"
    if python scripts/bench_large.py "${size}" --resume --fast-final --device cuda; then
        echo "[${label}] DONE at $(date -Iseconds)"
    else
        echo "[${label}] FAILED (exit $?) at $(date -Iseconds)"
    fi

    echo ""
done

echo "=== Scaling Ladder Complete ==="
echo "Finished: $(date -Iseconds)"
