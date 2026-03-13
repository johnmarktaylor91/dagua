#!/usr/bin/env bash
# check-tasks.sh — Quick status sweep of background tasks
# Usage: ./scripts/check-tasks.sh [task-id]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)" 2>/dev/null || true
TASK_DIR=".project-context/tasks"

if [ ! -d "$TASK_DIR" ]; then echo "No tasks directory."; exit 0; fi

if [ -n "${1:-}" ]; then
  TASK_ID="$1"
  if [ -f "$TASK_DIR/$TASK_ID.status" ]; then
    echo "=== $TASK_ID ==="
    cat "$TASK_DIR/$TASK_ID.result" 2>/dev/null
    echo ""; echo "--- Status: $(cat "$TASK_DIR/$TASK_ID.status") ---"
    if [ -f "$TASK_DIR/$TASK_ID.diff" ] && [ -s "$TASK_DIR/$TASK_ID.diff" ]; then
      echo ""; echo "--- Diff ---"; cat "$TASK_DIR/$TASK_ID.diff"
    fi
    if [ -f "$TASK_DIR/$TASK_ID.log" ]; then
      echo ""; echo "--- Log (last 30) ---"; tail -30 "$TASK_DIR/$TASK_ID.log"
    fi
  else echo "No task: $TASK_ID"; fi
  exit 0
fi

FOUND=0
for f in "$TASK_DIR"/*.status 2>/dev/null; do
  [ -f "$f" ] || continue; FOUND=1
  TID=$(basename "$f" .status); S=$(cat "$f")
  case "$S" in done) I="✓";; failed) I="✗";; running) I="⟳";; *) I="?";; esac
  D=$(grep -oP 'Duration: \K.*' "$TASK_DIR/$TID.result" 2>/dev/null || echo "—")
  printf "%-4s %-30s %s\n" "$I" "$TID" "($D)"
done
[ $FOUND -eq 0 ] && echo "No active tasks."
