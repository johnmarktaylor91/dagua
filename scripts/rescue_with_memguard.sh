#!/bin/bash
# Benchmark rescue wrapper with memory guard.
#
# Launches run_benchmark.py --resume and monitors its RSS every 15s.
# If RSS exceeds MAX_RSS_GB, sends 2x SIGINT (triggers Python's graceful
# shutdown handler which saves results via the finally block), waits, and
# falls back to SIGKILL only if absolutely stuck. Then restarts with
# --resume. --resume picks up from the last completed eval so no work is
# lost. Loops until benchmark finishes cleanly (exit 0) or we've tried
# too many restarts without progress.

set -u

MAX_RSS_GB=${MAX_RSS_GB:-70}
CHECK_INTERVAL=${CHECK_INTERVAL:-15}
LOG=${LOG:-eval_output/benchmark_unified.log}
MAX_RESTARTS=${MAX_RESTARTS:-50}
TIMEOUT=${TIMEOUT:-120}
WORKERS=${WORKERS:-4}
# The watchdog kills the worker pool when no future completes in N seconds.
# Default in run_benchmark.py is 300s, which false-positives on slow stochastic
# work groups (60 seeds x 600s timeout = up to 36000s per work group).  For
# repair runs set it generously so the watchdog only catches real worker
# crashes, not slow-but-progressing work.
WATCHDOG_TIMEOUT=${WATCHDOG_TIMEOUT:-7200}
# Extra args appended to the run_benchmark.py invocation.  For variant_bench_full
# repair set EXTRA_ARGS to something like:
#   "--variants --output-dir eval_output/variant_bench_full --seeds 60"
EXTRA_ARGS=${EXTRA_ARGS:-}
# Stagnation guard: if N consecutive restarts make no progress (measured
# by total visible-log-line count growing), bail out.
STAGNATION_LIMIT=5
GRACEFUL_WAIT=60
HARD_WAIT=20

cd "$(dirname "$0")/.." || exit 1

count_progress_lines() {
    # Count every observable benchmark output line -- we want to tell if
    # the child is doing anything at all, even if it's all timeouts.
    grep -c -E "^\[benchmark\] (ERROR|.*\| |.*ok|.*skipped|SKIP|Resume)" \
        "$LOG" 2>/dev/null || echo 0
}

: > "$LOG"
echo "[memguard] cap=${MAX_RSS_GB}GB interval=${CHECK_INTERVAL}s log=$LOG" >> "$LOG"

restart_count=0
stagnation=0
last_lines=0

while [ $restart_count -lt $MAX_RESTARTS ]; do
    echo "[memguard] Launch attempt $((restart_count + 1))" >> "$LOG"
    # python -u = unbuffered stdout, so progress actually flushes to disk
    # shellcheck disable=SC2086  # we want EXTRA_ARGS word-split for flags
    python -u scripts/run_benchmark.py --resume --workers $WORKERS \
        --timeout $TIMEOUT --watchdog-timeout $WATCHDOG_TIMEOUT \
        $EXTRA_ARGS \
        >> "$LOG" 2>&1 &
    pid=$!
    echo "[memguard] Started PID $pid" >> "$LOG"

    killed_for_memory=0
    while kill -0 $pid 2>/dev/null; do
        rss_kb=$(ps -o rss= -p $pid 2>/dev/null | tr -d ' ')
        if [ -z "$rss_kb" ]; then
            break
        fi
        rss_gb=$((rss_kb / 1024 / 1024))
        if [ $rss_gb -ge $MAX_RSS_GB ]; then
            echo "[memguard] PID $pid RSS=${rss_gb}GB >= ${MAX_RSS_GB}GB cap -- SIGINT #1" >> "$LOG"
            kill -INT $pid
            # Give Python up to $GRACEFUL_WAIT seconds to drain inflight
            # and flush the finally-block save.
            for _ in $(seq 1 $GRACEFUL_WAIT); do
                kill -0 $pid 2>/dev/null || break
                sleep 1
            done
            if kill -0 $pid 2>/dev/null; then
                echo "[memguard] Still alive after ${GRACEFUL_WAIT}s -- SIGINT #2 (force save)" >> "$LOG"
                kill -INT $pid
                for _ in $(seq 1 $HARD_WAIT); do
                    kill -0 $pid 2>/dev/null || break
                    sleep 1
                done
            fi
            if kill -0 $pid 2>/dev/null; then
                echo "[memguard] Still alive after 2x SIGINT -- SIGKILL" >> "$LOG"
                kill -KILL $pid
                pkill -KILL -P $pid 2>/dev/null
            fi
            killed_for_memory=1
            break
        fi
        sleep $CHECK_INTERVAL
    done

    wait $pid 2>/dev/null
    exit_code=$?
    echo "[memguard] PID $pid exited (code=$exit_code, killed_for_memory=$killed_for_memory)" >> "$LOG"

    if [ $killed_for_memory -eq 0 ] && [ $exit_code -eq 0 ]; then
        echo "[memguard] Benchmark completed successfully" >> "$LOG"
        exit 0
    fi

    cur_lines=$(count_progress_lines)
    cur_lines=${cur_lines:-0}
    echo "[memguard] Log progress lines: $cur_lines (was $last_lines)" >> "$LOG"

    if [ "$cur_lines" -le "$last_lines" ]; then
        stagnation=$((stagnation + 1))
        echo "[memguard] No new progress since last attempt (stagnation=$stagnation)" >> "$LOG"
        if [ $stagnation -ge $STAGNATION_LIMIT ]; then
            echo "[memguard] Stagnation limit hit -- giving up" >> "$LOG"
            exit 2
        fi
    else
        stagnation=0
    fi
    last_lines=$cur_lines

    restart_count=$((restart_count + 1))
    # Brief pause so forkserver sockets unwind before next launch.
    sleep 5
done

echo "[memguard] Max restarts ($MAX_RESTARTS) hit -- stopping" >> "$LOG"
exit 3
