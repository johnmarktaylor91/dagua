#!/usr/bin/env bash
# Self-healing watchdog for the 100-seed layouts run (scripts/r69_p3b_layouts_only.py).
#
# Failure mode it fixes (observed 2026-06-04, engine 9 drl_final): run_benchmark finishes its work
# ("Done" printed, results.json written) but a multiprocessing WORKER stuck in an uninterruptible
# igraph C call never terminates, so the pool join hangs forever -> the runner waits on the subprocess
# indefinitely. The per-combo timeout/watchdog can't catch this (it's post-work, in shutdown).
#
# Strategy: if a run_benchmark is alive but results.json hasn't been written in STALL_S seconds
# (>> the 420s per-combo watchdog, so only a genuine hang trips it), SIGKILL the run_benchmark main(s)
# + any orphaned multiprocessing workers. The runner's subprocess.run then returns nonzero -> it
# retries (--resume skips the completed combos -> fast) or, after 3 tries, advances to the next engine.
# Bounds each hang to ~STALL_S instead of indefinite. Exits when the runner exits.
set -u
RUNNER="${1:?runner pid}"
RESULTS="eval_output/benchmark_100seed_escalation_final/results.json"
STALL_S="${2:-900}"   # 15 min; >> 420s combo watchdog so legit slow combos don't trip it
POLL=120

# Reap orphaned multiprocessing workers: any python3 with PPID=1 + "multiprocessing.forks" is an
# orphan by definition (legit workers are always children of a LIVE run_benchmark; the runner itself
# is PPID=1 but its args are the runner script, not multiprocessing.forks -> excluded). Safe to kill
# every cycle, unconditionally -- this closes the gap where workers reparent to init AFTER a stall-kill's
# one-shot sweep and then spin at 99% CPU forever.
reap_orphans() {
  local o
  o=$(ps -C python3 -o pid=,ppid=,args= 2>/dev/null | awk '$2==1 && /multiprocessing.forks/{print $1}')
  if [ -n "$o" ]; then
    kill -KILL $o 2>/dev/null
    echo "$(date -Iseconds) ORPHAN_REAP killed=[$(echo $o | tr '\n' ' ')]"
  fi
}

echo "$(date -Iseconds) STALL_KILLER_STARTED runner=$RUNNER stall=${STALL_S}s"
while kill -0 "$RUNNER" 2>/dev/null; do
  sleep "$POLL"
  reap_orphans                                  # every cycle -- orphans are always PPID=1 multiprocessing
  RB=$(ps -C python3 -o pid=,args= 2>/dev/null | awk '/run_benchmark\.py/{print $1}')
  [ -z "$RB" ] && continue                      # between engines / not running -> nothing to watch
  now=$(date +%s)
  m=$(stat -c %Y "$RESULTS" 2>/dev/null || echo "$now")
  age=$(( now - m ))
  if [ "$age" -gt "$STALL_S" ]; then
    echo "$(date -Iseconds) STALL_KILL results_age=${age}s killing run_benchmark=[$RB]"
    kill -KILL $RB 2>/dev/null
    sleep 6; reap_orphans      # immediate sweep; the every-cycle reap above catches late stragglers
    echo "$(date -Iseconds) STALL_KILL_DONE -- runner will retry/advance"
    sleep 90                                     # let the runner spawn its retry before re-checking
  fi
done
echo "$(date -Iseconds) STALL_KILLER_EXIT runner gone"
