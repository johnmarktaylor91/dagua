#!/usr/bin/env bash
# clean-tasks.sh — Remove old task artifacts
# Usage: ./scripts/clean-tasks.sh [days|all]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)" 2>/dev/null || true
TASK_DIR=".project-context/tasks"
[ ! -d "$TASK_DIR" ] && echo "No tasks." && exit 0

if [ "${1:-}" = "all" ]; then
  rm -f "$TASK_DIR"/*.{spec.md,status,result,log,diff} 2>/dev/null
  echo "Cleared all tasks."
else
  DAYS="${1:-7}"
  find "$TASK_DIR" -name "*.status" -mtime +"$DAYS" -exec basename {} .status \; | \
    while read -r TID; do
      rm -f "$TASK_DIR/$TID".{spec.md,status,result,log,diff}
      echo "Cleaned: $TID"
    done
fi
