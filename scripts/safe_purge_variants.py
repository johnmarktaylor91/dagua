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
import os
import sys
from pathlib import Path


def stage_json_write(path: Path, payload: dict[str, object]) -> Path:
    """Write JSON to a temporary sibling file and return its staged path.

    Parameters
    ----------
    path : Path
        Final JSON destination.
    payload : dict[str, object]
        JSON-serializable object to persist.

    Returns
    -------
    Path
        Temporary file path ready to replace ``path``.
    """
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return temp_path


def stage_purged_hdf5(h5_path: Path, engine_set: set[str]) -> tuple[Path, int]:
    """Stage a rewritten ``positions.h5`` without the selected engine keys.

    Parameters
    ----------
    h5_path : Path
        Existing HDF5 cache path.
    engine_set : set[str]
        Engine names to purge.

    Returns
    -------
    tuple[Path, int]
        Temporary file path and number of HDF5 datasets removed.
    """
    import h5py

    temp_path = h5_path.with_suffix(f"{h5_path.suffix}.tmp")
    temp_path.unlink(missing_ok=True)
    removed = 0
    with h5py.File(h5_path, "r") as source_h5, h5py.File(temp_path, "w") as target_h5:
        for key in source_h5.keys():
            if "::" in key and key.split("::")[1] in engine_set:
                removed += 1
                continue
            source_h5.copy(source_h5[key], target_h5, name=key)
    return temp_path, removed


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
    staged_results_path = stage_json_write(results_path, data)
    staged_h5_path: Path | None = None
    removed = 0
    if h5_path.exists() and h5_count > 0:
        staged_h5_path, removed = stage_purged_hdf5(h5_path, engine_set)

    if staged_h5_path is not None:
        os.rename(staged_h5_path, h5_path)
    os.rename(staged_results_path, results_path)
    print(f"\nPurged {len(rj_keys)} entries from results.json")

    # Purge positions.h5
    if h5_path.exists() and h5_count > 0:
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
