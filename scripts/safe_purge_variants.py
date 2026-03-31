#!/usr/bin/env python3
"""Safely purge benchmark data for specified variants.

Removes BOTH results.json entries AND positions.h5 keys atomically.
Refuses to purge one without the other. Shows what will be removed
and requires --confirm.

Written as enforcement code from retro 2026-03-30 after purging H5
without purging results.json caused a 2-day cascade of failures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    """Parse arguments and perform safe purge."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "engines",
        nargs="+",
        help="Engine names to purge (e.g., classic_fa2_linlog)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("eval_output/variant_bench_full"),
        help="Directory containing results.json and positions.h5",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually perform the purge (without this, dry-run only)",
    )
    args = parser.parse_args()

    engine_set = set(args.engines)
    results_path = args.data_dir / "results.json"
    h5_path = args.data_dir / "positions.h5"

    # Count what would be removed
    print("Loading results.json...", file=sys.stderr)
    with open(results_path) as f:
        data = json.load(f)

    rj_keys = [
        k for k, v in data.items() if v.get("engine_name", v.get("engine", "")) in engine_set
    ]

    h5_count = 0
    if h5_path.exists():
        import h5py

        with h5py.File(h5_path, "r") as h5f:
            h5_keys = [k for k in h5f.keys() if "::" in k and k.split("::")[1] in engine_set]
            h5_count = len(h5_keys)

    # Report
    print(f"\nPurge plan for {len(engine_set)} engines:")
    for eng in sorted(engine_set):
        eng_rj = sum(1 for k, v in data.items() if v.get("engine_name", v.get("engine", "")) == eng)
        print(f"  {eng}: {eng_rj} results.json entries")
    print(f"\nTotal results.json entries to remove: {len(rj_keys)}")
    print(f"Total positions.h5 keys to remove:   {h5_count}")
    print(f"Total results.json entries remaining:  {len(data) - len(rj_keys)}")

    if not args.confirm:
        print(
            "\nDry run. Add --confirm to actually purge.",
            file=sys.stderr,
        )
        return

    # Purge results.json
    for k in rj_keys:
        del data[k]
    with open(results_path, "w") as f:
        json.dump(data, f)
    print(f"\nPurged {len(rj_keys)} entries from results.json")

    # Purge positions.h5
    if h5_path.exists() and h5_count > 0:
        import h5py

        with h5py.File(h5_path, "a") as h5f:
            removed = 0
            for k in list(h5f.keys()):
                if "::" in k and k.split("::")[1] in engine_set:
                    del h5f[k]
                    removed += 1
        print(f"Purged {removed} keys from positions.h5")
    else:
        print("No positions.h5 keys to purge")

    # Verify sync after purge
    print("\nPost-purge validation...")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from validate_benchmark_integrity import validate_sync

    errors = validate_sync(results_path, h5_path, engine_set)
    if errors:
        print(f"WARNING: {len(errors)} sync issues after purge:")
        for err in errors:
            print(f"  {err}")
    else:
        print("Post-purge sync OK: both stores consistent")


if __name__ == "__main__":
    main()
