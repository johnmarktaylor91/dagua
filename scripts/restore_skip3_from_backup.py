"""Re-insert the ``skipped after 3 consecutive errors`` rows back into
results.json from the pre-salvage backup.

Rationale: the salvage cleanup removed 10,450 of these (pairs where some
seed succeeded elsewhere) on the theory they were transiently bad. Turns
out that retry is slow -- each of those pairs needs 3 x 120s timeouts
before auto-skip, and almost all still fail. The throughput hit makes the
remaining recovery infeasible within a reasonable window.

This script:
  1. Reads both the live results.json and the pre-salvage backup.
  2. For each key that:
       a) exists in the backup with skip_reason ==
          "skipped after 3 consecutive errors"
       b) exists in the live file with status == "ok" (meaning the current
          benchmark DID finish it successfully -- keep the ok),
          OR is NOT a live ok row
     -> restore the backup row, but update skip_reason to
        "heavy_pair_not_retried" so we can distinguish it from the
        original population.
  3. Writes atomically.

Only rows that actually succeeded in the running salvage benchmark are
preserved; everything else is rolled back to skipped so the next --resume
doesn't re-process them.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("eval_output/variant_bench_full/results.json"),
    )
    parser.add_argument(
        "--backup",
        type=Path,
        required=True,
        help="Path to the pre-salvage-cleanup backup (still has skip3 rows).",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


SKIP3_REASON = "skipped after 3 consecutive errors"


def main() -> int:
    args = parse_args()
    if not args.results.exists():
        print(f"ERROR: {args.results} missing", file=sys.stderr)
        return 2
    if not args.backup.exists():
        print(f"ERROR: backup {args.backup} missing", file=sys.stderr)
        return 2

    print(f"[restore] Loading live {args.results}")
    live = json.loads(args.results.read_text(encoding="utf-8"))
    print(f"[restore] Loading backup {args.backup}")
    bak = json.loads(args.backup.read_text(encoding="utf-8"))

    reasons: Counter[str] = Counter()
    to_add = 0
    live_already_has_row = 0

    # Only restore keys that are MISSING from live (cleared by salvage cleanup).
    # Keys still in live weren't cleared -- don't touch them.
    for k, v in bak.items():
        if v.get("skip_reason") != SKIP3_REASON:
            continue
        if k in live:
            live_already_has_row += 1
            continue
        restored = dict(v)
        restored["skip_reason"] = "heavy_pair_not_retried"
        to_add += 1
        if not args.dry_run:
            live[k] = restored
        reasons["heavy_pair_not_retried"] += 1

    skip3_count = sum(1 for v in bak.values() if v.get("skip_reason") == SKIP3_REASON)
    print(f"[restore] Backup skip3 rows reviewed: {skip3_count:,}")
    print(f"[restore] Already in live (untouched): {live_already_has_row:,}")
    print(f"[restore] To restore (as heavy_pair_not_retried): {to_add:,}")

    if args.dry_run:
        print("[restore] DRY RUN -- no write")
        return 0

    if to_add == 0:
        print("[restore] Nothing to do")
        return 0

    stamp = time.strftime("%Y%m%d-%H%M%S")
    live_backup = args.results.parent / f"{args.results.name}.pre-restore-skip3-{stamp}.bak"
    print(f"[restore] Extra backup of current live: {live_backup}")
    shutil.copy2(args.results, live_backup)

    fd, tmp_str = tempfile.mkstemp(
        dir=str(args.results.parent),
        prefix=f"{args.results.stem}.",
        suffix=".tmp",
    )
    tmp = Path(tmp_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(live, f, indent=2, default=str)
        tmp.replace(args.results)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

    print(f"[restore] Live now has {len(live):,} records (added {to_add:,})")
    print("[restore] Atomic rewrite complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
