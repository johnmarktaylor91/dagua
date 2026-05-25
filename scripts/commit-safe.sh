#!/usr/bin/env bash
# Pre-commit-safe commit helper.
#
# Problem: pre-commit hooks (end-of-file-fixer, ruff-format) auto-fix staged
# files during `git commit`, then can fail to re-apply the unstaged-changes
# stash cleanly, causing pre-commit to ROLL BACK the entire commit. This bit
# us in R31 when drl + tsnet got silently rolled back.
#
# Solution: run the auto-fixing hooks on staged files BEFORE invoking
# `git commit`, so the commit-time hooks find nothing to fix.
#
# Usage:
#   git add <files>
#   bash scripts/commit-safe.sh "commit message"
#   bash scripts/commit-safe.sh -F message_file.txt
#
# The wrapper:
# 1. Lists currently staged files
# 2. Runs `pre-commit run --files <staged>` to apply auto-fixes
# 3. Re-stages the now-fixed files
# 4. Invokes `git commit` with passed-through args (no -m/-F flags get
#    consumed by the wrapper -- they pass to git)

set -u
cd "$(dirname "$0")/.."

STAGED=$(git diff --cached --name-only --diff-filter=ACMR)
if [ -z "$STAGED" ]; then
    echo "commit-safe: no staged files; aborting" >&2
    exit 2
fi

echo "[commit-safe] staged files:"
echo "$STAGED" | sed 's/^/  /'
echo

echo "[commit-safe] running pre-commit auto-fixes on staged files..."
if ! pre-commit run --files $STAGED; then
    echo
    echo "[commit-safe] pre-commit hooks made changes. Re-staging."
    # Re-stage the files (hooks may have auto-fixed them)
    git add $STAGED
    # Run pre-commit again to verify all hooks now pass cleanly
    if ! pre-commit run --files $STAGED; then
        echo
        echo "[commit-safe] pre-commit still failing -- manual fix needed" >&2
        exit 3
    fi
fi

echo
echo "[commit-safe] all hooks clean; invoking git commit..."
git commit "$@"
