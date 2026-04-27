"""Remove watchdog-collateral error rows from results.json.

Watchdog-errored rows (``status=error`` with ``error="watchdog: worker pool stuck"``)
are collateral damage -- when any worker hangs past the watchdog timeout, ALL
in-flight rolling-window futures get marked errored, including many that were
still healthy. Removing these lets ``--resume`` retry them.

Writes atomically through a tmp file (tmp -> rename). Makes an additional
timestamped backup alongside the original.
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
        help="Path to results.json to clean (atomic rewrite).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report counts and exit without writing.",
    )
    return parser.parse_args()


def is_watchdog_error(record: dict) -> bool:
    if record.get("status") != "error":
        return False
    err = (record.get("error") or "").lower()
    return "watchdog: worker pool stuck" in err


def main() -> int:
    args = parse_args()
    results_path: Path = args.results
    if not results_path.exists():
        print(f"ERROR: {results_path} does not exist", file=sys.stderr)
        return 2

    print(f"[cleanup] Loading {results_path}...")
    data = json.loads(results_path.read_text(encoding="utf-8"))
    total_before = len(data)
    print(f"[cleanup] Loaded {total_before:,} records")

    status_before = Counter(v.get("status", "") for v in data.values())

    removed_keys = [k for k, v in data.items() if is_watchdog_error(v)]
    print(f"[cleanup] Identified {len(removed_keys):,} watchdog-collateral rows to remove")

    if args.dry_run:
        print("[cleanup] DRY RUN -- no files written")
        return 0

    if not removed_keys:
        print("[cleanup] Nothing to do -- no watchdog errors present")
        return 0

    timestamped_backup = (
        results_path.parent
        / f"{results_path.name}.pre-watchdog-cleanup-{time.strftime('%Y%m%d-%H%M%S')}.bak"
    )
    print(f"[cleanup] Writing extra backup: {timestamped_backup}")
    shutil.copy2(results_path, timestamped_backup)

    for k in removed_keys:
        del data[k]

    total_after = len(data)
    status_after = Counter(v.get("status", "") for v in data.values())

    print(f"[cleanup] Records before: {total_before:,}")
    print(f"[cleanup] Records after:  {total_after:,}")
    print(f"[cleanup] Removed:        {total_before - total_after:,}")
    print("[cleanup] Status breakdown:")
    all_keys = set(status_before) | set(status_after)
    for k in sorted(all_keys):
        b = status_before.get(k, 0)
        a = status_after.get(k, 0)
        print(f"            {k:>10}: {b:,} -> {a:,} (delta {a - b:+,})")

    fd, tmp_path_str = tempfile.mkstemp(
        dir=str(results_path.parent),
        prefix=f"{results_path.stem}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        tmp_path.replace(results_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    print(f"[cleanup] Atomic rewrite complete: {results_path}")
    print("[cleanup] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
