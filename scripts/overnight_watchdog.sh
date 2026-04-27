#!/bin/bash
# Overnight watchdog: at TARGET_TIME, if the benchmark hasn't finished,
# gracefully kill it, flip any "running" rows to "skipped", then run the
# post-benchmark pipeline so the user wakes up to reports.
#
# If the benchmark already finished naturally and overnight_finish_layouts.sh
# already ran the post-pipeline, this script detects the "done" status and
# exits quietly without doing anything.
#
# Coordinates via .project-context/tasks/overnight-finish.status:
#   "running" -> benchmark or post-pipeline still active; intervene
#   "done"    -> everything already complete; exit
#   "failed"  -> treat like "running" for intervention safety
set -u

cd /home/jtaylor/projects/dagua
export PATH="/home/jtaylor/anaconda3/envs/py311/bin:$PATH"
export PYTHONUNBUFFERED=1

TARGET="2026-04-17 05:30:00"
TARGET_TS=$(date -d "$TARGET" +%s)
echo "=== [watchdog] armed, will fire at $TARGET (ts=$TARGET_TS) ==="
echo "=== [watchdog] started at $(date -Iseconds) ==="

# --- Sleep until target time -------------------------------------------------
while [ "$(date +%s)" -lt "$TARGET_TS" ]; do
    STATUS="$(cat .project-context/tasks/overnight-finish.status 2>/dev/null || echo unknown)"
    if [ "$STATUS" = "done" ]; then
        echo "=== [watchdog] overnight-finish already 'done' at $(date -Iseconds); exiting ==="
        exit 0
    fi
    sleep 60
done

echo "=== [watchdog] TARGET TIME reached at $(date -Iseconds) ==="

# --- Re-check: if overnight-finish is already done, exit quietly ------------
STATUS="$(cat .project-context/tasks/overnight-finish.status 2>/dev/null || echo unknown)"
if [ "$STATUS" = "done" ]; then
    echo "=== [watchdog] overnight-finish already 'done'; exiting ==="
    exit 0
fi

echo "=== [watchdog] overnight-finish status='$STATUS'; intervening ==="

# --- Kill benchmark python processes gracefully, then hard ------------------
KILL_BENCH() {
    local pids
    pids=$(pgrep -f "run_benchmark.py.*variant_bench_full" | tr '\n' ' ')
    echo "KILL_BENCH pids=[$pids]"
    [ -z "$pids" ] && return
    for p in $pids; do kill -SIGINT  $p 2>/dev/null; done
    sleep 15
    for p in $pids; do kill -SIGINT  $p 2>/dev/null; done
    sleep 60
    for p in $pids; do kill -SIGTERM $p 2>/dev/null; done
    sleep 15
    for p in $pids; do kill -SIGKILL $p 2>/dev/null; done
}

KILL_BENCH

# Wait for actual exit
while pgrep -f "run_benchmark.py.*variant_bench_full" > /dev/null 2>&1; do
    echo "[watchdog] still waiting for benchmark to exit..."
    sleep 5
done
echo "=== [watchdog] benchmark killed ==="

# --- Kill the wrapper overnight_finish_layouts.sh -- it would also try
#     to run post-pipeline if BENCH_EXIT was successful, but we're preempting
for p in $(pgrep -f "scripts/overnight_finish_layouts.sh" 2>/dev/null); do
    kill -SIGTERM $p 2>/dev/null && echo "[watchdog] killed wrapper $p"
done
sleep 2

# --- Kill any orphaned forkserver workers older than 5 min ------------------
pgrep -f "from multiprocessing.forkserver" 2>/dev/null | while read p; do
    ET=$(ps -o etimes= -p "$p" 2>/dev/null | xargs)
    if [ -n "$ET" ] && [ "$ET" -gt 300 ]; then
        kill -SIGKILL "$p" 2>/dev/null && echo "[watchdog] killed orphan forkserver $p (age=${ET}s)"
    fi
done
sleep 2

# --- Flip running -> skipped ------------------------------------------------
echo "=== [watchdog] flipping running rows to skipped ==="
python3 scripts/flip_running_to_skipped.py --reason overnight_time_limit_5am
FLIP_EXIT=$?
if [ $FLIP_EXIT -ne 0 ]; then
    ~/.claude/scripts/send-to-jmt.sh "Dagua watchdog FAILED to flip running rows (exit=$FLIP_EXIT). Check eval_output/variant_bench_full/results.json manually." || true
    exit $FLIP_EXIT
fi

# --- Run post-benchmark pipeline --------------------------------------------
echo "=== [watchdog] running post-benchmark pipeline ==="
./scripts/post_benchmark_pipeline.sh
POST_EXIT=$?

# --- Notify ------------------------------------------------------------------
SUMMARY=$(python3 -c "
import json
r = json.load(open('eval_output/variant_bench_full/results.json'))
from collections import Counter
c = Counter(v.get('status','') for v in r.values())
print(f\"ok={c.get('ok',0):,} err={c.get('error',0):,} skip={c.get('skipped',0):,} timeout={c.get('timeout',0):,}\")
" 2>/dev/null || echo "summary unavailable")

if [ $POST_EXIT -eq 0 ]; then
    ~/.claude/scripts/send-to-jmt.sh "Dagua overnight FORCE-COMPLETE at 5:30am. $SUMMARY. Reports in eval_output/report/. Remaining 'running' rows (mostly slow neulay retries) flipped to skipped." || true
else
    ~/.claude/scripts/send-to-jmt.sh "Dagua overnight PARTIAL: watchdog at 5:30am, post-pipeline exit=$POST_EXIT. $SUMMARY. See eval_output/post_benchmark_pipeline.log" || true
fi

echo "=== [watchdog] done at $(date -Iseconds) post_exit=$POST_EXIT ==="
exit $POST_EXIT
