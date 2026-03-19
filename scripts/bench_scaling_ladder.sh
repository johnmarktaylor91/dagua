#!/usr/bin/env bash
# bench_scaling_ladder.sh — Scaling ladder from START_FROM to 2B.
# Uses --resume to pick up from existing checkpoints.
# Resource checks between runs to avoid OOM.
# Usage: bench_scaling_ladder.sh [START_FROM]
#   START_FROM: node count to begin at (default: 10). Skips smaller sizes.
#   Example: bench_scaling_ladder.sh 200000000  (start from 200M)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

START_FROM="${1:-10}"

SIZES=(
    "10" "20" "50" "100" "200" "500"
    "1000" "2000" "5000" "10000" "20000" "50000"
    "100000" "200000" "500000"
    "1000000" "2000000" "5000000"
    "10000000" "20000000" "50000000"
    "100000000" "200000000" "500000000"
    "1000000000"
)
LABELS=(
    "10" "20" "50" "100" "200" "500"
    "1K" "2K" "5K" "10K" "20K" "50K"
    "100K" "200K" "500K"
    "1M" "2M" "5M"
    "10M" "20M" "50M"
    "100M" "200M" "500M"
    "1B"
)
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

# Pre-compute graph + layering for sizes >= 1B that don't have cached layering.
PRECOMPUTE_THRESHOLD=1000000000
PRECOMPUTE_SIZES=()
LOCKER="/mnt/locker/jt3295/dagua_bench_large"
for i in "${!SIZES[@]}"; do
    s="${SIZES[$i]}"
    if [ "$s" -ge "$PRECOMPUTE_THRESHOLD" ] && [ "$s" -ge "$START_FROM" ]; then
        if [ ! -f "${LOCKER}/${s}/layer_assignments.pt" ]; then
            PRECOMPUTE_SIZES+=("$s")
        fi
    fi
done
if [ ${#PRECOMPUTE_SIZES[@]} -gt 0 ]; then
    echo "=== Pre-computing layering for ${#PRECOMPUTE_SIZES[@]} large sizes ==="
    python scripts/precompute_layering.py "${PRECOMPUTE_SIZES[@]}"
    echo ""
else
    echo "=== All large-graph layerings cached ==="
fi

echo "=== Scaling Ladder: ${START_FROM} → 2B ==="
echo "Started: $(date -Iseconds)"
echo ""

for i in "${!SIZES[@]}"; do
    size="${SIZES[$i]}"
    label="${LABELS[$i]}"

    if [ "$size" -lt "$START_FROM" ]; then
        echo "[${label}] SKIPPED — below start threshold ${START_FROM}"
        continue
    fi

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
