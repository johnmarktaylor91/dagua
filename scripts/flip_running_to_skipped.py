"""Flip ``status=running`` rows in results.json to ``status=skipped``.

Used as a time-budget safety net: when the overnight benchmark cannot process
every retry in the available window, we need the remaining in-flight or
abandoned rows marked with a terminal status so that downstream tooling
(post_benchmark_pipeline.sh) stops aborting on the "still running" check.

Writes atomically through a tmp file and preserves the existing JSON shape.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("eval_output/variant_bench_full/results.json"),
    )
    parser.add_argument(
        "--reason",
        default="overnight_time_limit",
        help="Value to write into skip_reason on flipped rows.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_path: Path = args.results
    if not results_path.exists():
        print(f"ERROR: {results_path} does not exist", file=sys.stderr)
        return 2

    print(f"[flip] Loading {results_path}...")
    data = json.loads(results_path.read_text(encoding="utf-8"))
    total = len(data)
    print(f"[flip] Loaded {total:,} records")

    candidates = [k for k, v in data.items() if v.get("status") == "running"]
    print(
        f"[flip] {len(candidates):,} rows currently 'running' -> will flip to "
        f"'skipped' with reason='{args.reason}'"
    )

    if args.dry_run or not candidates:
        if args.dry_run:
            print("[flip] DRY RUN -- no write")
        return 0

    backup = (
        results_path.parent / f"{results_path.name}.pre-flip-{time.strftime('%Y%m%d-%H%M%S')}.bak"
    )
    print(f"[flip] Backup: {backup}")
    shutil.copy2(results_path, backup)

    for k in candidates:
        row = data[k]
        row["status"] = "skipped"
        row["skip_reason"] = args.reason
        # Keep runtime_seconds/error/positions_file as they are (usually null)

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

    print(f"[flip] Flipped {len(candidates):,} running -> skipped")
    print(f"[flip] Atomic rewrite complete: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
