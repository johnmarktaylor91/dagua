#!/usr/bin/env bash
# Daily check-in for the 100-seed benchmark supervisor.
# Reads supervisor + benchmark process state, summarizes progress, iMessages JMT.
# Self-removes its crontab entry when the supervisor finishes ("100-seed run COMPLETE").
#
# Install: crontab -e, add:  0 7 * * * /home/jtaylor/projects/dagua/scripts/daily_benchmark_check.sh

set -u
cd /home/jtaylor/projects/dagua

SUPERVISOR_LOG=/tmp/benchmark_100seed_supervisor.log
BENCH_OUT=eval_output/benchmark_100seed_final
RESULTS=$BENCH_OUT/results.json
SEND=$HOME/.claude/scripts/send-to-jmt.sh
DAILY_LOG=/tmp/daily_benchmark_check.log

exec >> "$DAILY_LOG" 2>&1
echo ""
echo "=== daily_benchmark_check $(date -Iseconds) ==="

# --- Check completion first; if done, self-remove crontab ---
if grep -q "100-seed run COMPLETE" "$SUPERVISOR_LOG" 2>/dev/null; then
    echo "Supervisor already reported COMPLETE; self-removing crontab entry."
    crontab -l 2>/dev/null | grep -v daily_benchmark_check.sh | crontab -
    "$SEND" "Daily check: 100-seed benchmark already COMPLETE; removing daily check cron." || true
    exit 0
fi

# --- Process state ---
SUPERVISOR_ALIVE=$(pgrep -af supervisor_100seed.sh | grep -v grep | wc -l)
BENCHMARK_ALIVE=$(pgrep -af "run_benchmark.py.*benchmark_100seed_final" | grep -v grep | wc -l)

# --- Progress stats from results.json ---
if [ -f "$RESULTS" ]; then
    STATS=$(python3 -c "
import json
try:
    r = json.load(open('$RESULTS'))
    n = len(r)
    ok = sum(1 for v in r.values() if v.get('status') == 'ok')
    err = sum(1 for v in r.values() if v.get('status') == 'error')
    run = sum(1 for v in r.values() if v.get('status') == 'running')
    skip = sum(1 for v in r.values() if v.get('status') == 'skipped')
    pct = (ok + err + skip) * 100.0 / n if n else 0
    print(f'total={n} ok={ok} err={err} run={run} skip={skip} pct={pct:.1f}')
except Exception as e:
    print(f'results_read_error={e}')
")
else
    STATS="results.json missing"
fi

# --- Recent supervisor activity ---
RECENT=$(tail -20 "$SUPERVISOR_LOG" 2>/dev/null | grep -E "Benchmark attempt|crashed|retrying|DONE|FAILED" | tail -3 | tr '\n' ';' | head -c 200)

# --- Hours running ---
START_TS=$(grep "Supervisor started" "$SUPERVISOR_LOG" 2>/dev/null | head -1 | grep -oE "[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}")
if [ -n "$START_TS" ]; then
    START_EPOCH=$(date -d "$START_TS" +%s 2>/dev/null || echo 0)
    NOW_EPOCH=$(date +%s)
    HOURS_RUN=$(( (NOW_EPOCH - START_EPOCH) / 3600 ))
else
    HOURS_RUN="?"
fi

# --- Status summary ---
if [ "$SUPERVISOR_ALIVE" -ge 1 ] && [ "$BENCHMARK_ALIVE" -ge 1 ]; then
    STATUS="OK supervisor+benchmark both alive"
elif [ "$SUPERVISOR_ALIVE" -ge 1 ] && [ "$BENCHMARK_ALIVE" -eq 0 ]; then
    STATUS="WARN supervisor alive but no benchmark process (between retries?)"
elif [ "$SUPERVISOR_ALIVE" -eq 0 ]; then
    STATUS="PROBLEM supervisor DEAD"
else
    STATUS="UNKNOWN"
fi

MSG="100-seed daily check ${HOURS_RUN}h running. $STATUS. Progress: $STATS. Recent: ${RECENT:-(no notable events)}"
echo "$MSG"

# --- Auto-restart if supervisor died and not done ---
if [ "$SUPERVISOR_ALIVE" -eq 0 ]; then
    if ! grep -q "100-seed run COMPLETE" "$SUPERVISOR_LOG" 2>/dev/null; then
        echo "Supervisor dead, restarting..."
        nohup bash scripts/supervisor_100seed.sh > /dev/null 2>&1 &
        disown
        sleep 3
        NEW_PID=$(pgrep -af supervisor_100seed.sh | grep -v grep | awk '{print $1}' | head -1)
        MSG="$MSG | RESTARTED supervisor as PID $NEW_PID"
    fi
fi

# --- iMessage ---
"$SEND" "$MSG" || echo "iMessage failed"

echo "=== daily_benchmark_check done $(date -Iseconds) ==="
