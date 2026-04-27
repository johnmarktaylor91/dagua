"""Remove rows from results.json that should be retried in a salvage round.

Targets three categories deemed recoverable:

1. ``skipped`` rows with ``skip_reason == "overnight_time_limit_5am"`` -- the
   watchdog flipped these during the overnight cutoff; they never got to run.
2. ``error`` rows with ``error == "watchdog: worker pool stuck"`` -- collateral
   damage from the second overnight's watchdog firings (new rows this time,
   not the original round-1 watchdog rows that we already cleared).
3. ``skipped`` rows from ``"skipped after 3 consecutive errors"`` where the
   same (engine, graph) pair has >=1 ``ok`` at a different seed. The three
   consecutive errors may have been a transient patch (memory pressure,
   race) rather than a deterministic fail -- other seeds succeeded, so the
   pair isn't broken.

Does NOT touch: genuine timeouts (120s), preconditions (disconnected/DAG/SCC),
OOM, max_nodes, or "skipped after 3 consecutive errors" on pairs that never
once succeeded -- those are wasted retries.

Writes atomically through a tmp file.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("eval_output/variant_bench_full/results.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_path: Path = args.results
    if not results_path.exists():
        print(f"ERROR: {results_path} does not exist", file=sys.stderr)
        return 2

    print(f"[salvage] Loading {results_path}...")
    data = json.loads(results_path.read_text(encoding="utf-8"))
    total_before = len(data)
    print(f"[salvage] Loaded {total_before:,} records")

    # Pre-compute which (engine, graph) pairs have at least one ok record
    pair_has_ok: dict[tuple[str, str], bool] = defaultdict(bool)
    for v in data.values():
        if v.get("status") == "ok":
            pair_has_ok[(v.get("engine_name", ""), v.get("graph_name", ""))] = True

    # Classify each record
    to_remove: list[tuple[str, str]] = []  # (key, reason-tag)
    reason_counts: Counter[str] = Counter()
    for k, v in data.items():
        status = v.get("status")
        skip_reason = v.get("skip_reason") or ""
        error = (v.get("error") or "").lower()

        if status == "skipped" and skip_reason == "overnight_time_limit_5am":
            to_remove.append((k, "cutoff"))
            reason_counts["cutoff"] += 1
        elif status == "error" and "watchdog: worker pool stuck" in error:
            to_remove.append((k, "watchdog_residual"))
            reason_counts["watchdog_residual"] += 1
        elif status == "skipped" and skip_reason == "skipped after 3 consecutive errors":
            pair = (v.get("engine_name", ""), v.get("graph_name", ""))
            if pair_has_ok.get(pair, False):
                to_remove.append((k, "skip3_pair_succeeded_elsewhere"))
                reason_counts["skip3_pair_succeeded_elsewhere"] += 1

    print("[salvage] Rows selected for removal (will be retried via --resume):")
    for r, n in reason_counts.most_common():
        print(f"  {n:>6,}  {r}")
    print(f"[salvage] Total to remove: {len(to_remove):,}")

    if args.dry_run or not to_remove:
        if args.dry_run:
            print("[salvage] DRY RUN -- no write")
        return 0

    backup = (
        results_path.parent
        / f"{results_path.name}.pre-salvage-cleanup-{time.strftime('%Y%m%d-%H%M%S')}.bak"
    )
    print(f"[salvage] Additional backup: {backup}")
    shutil.copy2(results_path, backup)

    for k, _ in to_remove:
        del data[k]

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

    print(f"[salvage] Records before: {total_before:,}")
    print(f"[salvage] Records after:  {len(data):,}")
    print(f"[salvage] Removed:        {total_before - len(data):,}")
    print(f"[salvage] Atomic rewrite complete: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
