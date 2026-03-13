#!/usr/bin/env bash
# dispatch.sh — Run command in background, write status files, send Pushover
# Usage: ./scripts/dispatch.sh <task-id> <command> [args...]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)" 2>/dev/null || true

TASK_ID="${1:?Usage: dispatch.sh <task-id> <command> [args...]}"
shift
TASK_DIR=".project-context/tasks"

mkdir -p "$TASK_DIR"
rm -f "$TASK_DIR/$TASK_ID".{status,result,log,diff}

echo "running" > "$TASK_DIR/$TASK_ID.status"
cat > "$TASK_DIR/$TASK_ID.result" <<EOF
Task:    $TASK_ID
Command: $*
Started: $(date -Iseconds)
EOF

(
  SECONDS=0
  if OUTPUT=$("$@" 2>&1); then EXIT_CODE=0; else EXIT_CODE=$?; fi
  DURATION=$SECONDS

  echo "$OUTPUT" > "$TASK_DIR/$TASK_ID.log"
  if [ $EXIT_CODE -eq 0 ]; then
    echo "done" > "$TASK_DIR/$TASK_ID.status"
    MSG="✓ $TASK_ID done (${DURATION}s)"
  else
    echo "failed" > "$TASK_DIR/$TASK_ID.status"
    MSG="✗ $TASK_ID FAILED (exit $EXIT_CODE, ${DURATION}s)"
  fi
  cat >> "$TASK_DIR/$TASK_ID.result" <<EOF
Exit:     $EXIT_CODE
Duration: ${DURATION}s
Finished: $(date -Iseconds)
EOF
  git diff --stat > "$TASK_DIR/$TASK_ID.diff" 2>/dev/null || true

  # Notify on failures always; successes only for test/lint/check/build tasks
  SEND=false
  if [ $EXIT_CODE -ne 0 ]; then SEND=true; fi
  if [[ "$TASK_ID" =~ (test|lint|check|build|review|bench) ]]; then SEND=true; fi

  if [ "$SEND" = true ] && [ -n "${PUSHOVER_TOKEN:-}" ] && [ -n "${PUSHOVER_USER:-}" ]; then
    curl -s \
      -F "token=$PUSHOVER_TOKEN" \
      -F "user=$PUSHOVER_USER" \
      -F "title=$TASK_ID" \
      -F "message=$MSG" \
      https://api.pushover.net/1/messages.json > /dev/null 2>&1 || true
  fi
) &

echo "⚡ Dispatched '$TASK_ID' (PID: $!)"
