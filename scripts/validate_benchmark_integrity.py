#!/usr/bin/env python3
"""Validate that results.json and positions.h5 are in sync.

Every 'ok' record in results.json must have a corresponding key in
positions.h5. Every key in positions.h5 must have a corresponding
record in results.json. Exits nonzero if ANY desync found.

Written as enforcement code from retro 2026-03-30 after 2 days of
wasted compute due to results.json/positions.h5 desync.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def validate_sync(
    results_path: Path,
    h5_path: Path,
    engines: set[str] | None = None,
) -> list[str]:
    """Check results.json and positions.h5 are in sync.

    Parameters
    ----------
    results_path : Path
        Path to results.json.
    h5_path : Path
        Path to positions.h5.
    engines : set[str] | None
        If provided, only validate these engine names.

    Returns
    -------
    list[str]
        Error messages. Empty list means sync is valid.
    """
    import h5py

    errors: list[str] = []

    with open(results_path) as f:
        results = json.load(f)

    # Build set of keys that should have positions (status=ok, not --no-positions)
    rj_ok_keys: set[str] = set()
    for key, record in results.items():
        eng = record.get("engine_name", record.get("engine", ""))
        if engines and eng not in engines:
            continue
        status = record.get("status", "")
        if status == "ok":
            rj_ok_keys.add(key)

    # Build set of H5 keys
    if not h5_path.exists():
        if rj_ok_keys:
            errors.append(
                f"positions.h5 does not exist but results.json has {len(rj_ok_keys)} ok records"
            )
        return errors

    with h5py.File(h5_path, "r") as h5f:
        h5_keys: set[str] = set(h5f.keys())
        if engines:
            h5_keys = {k for k in h5_keys if "::" in k and k.split("::")[1] in engines}

    # Check for results without positions
    missing_positions = rj_ok_keys - h5_keys
    if missing_positions:
        # Group by engine for readable output
        by_engine: dict[str, int] = {}
        for key in missing_positions:
            parts = key.split("::")
            eng = parts[1] if len(parts) >= 2 else "unknown"
            by_engine[eng] = by_engine.get(eng, 0) + 1
        for eng, count in sorted(by_engine.items(), key=lambda x: -x[1]):
            errors.append(f"DESYNC: {eng} has {count} ok results but missing positions")

    # Check for orphaned positions (H5 keys without results)
    orphaned = h5_keys - rj_ok_keys
    if orphaned:
        by_engine_orphan: dict[str, int] = {}
        for key in orphaned:
            parts = key.split("::")
            eng = parts[1] if len(parts) >= 2 else "unknown"
            by_engine_orphan[eng] = by_engine_orphan.get(eng, 0) + 1
        for eng, count in sorted(by_engine_orphan.items(), key=lambda x: -x[1]):
            errors.append(f"ORPHAN: {eng} has {count} positions but no ok result")

    return errors


def main() -> None:
    """Run sync validation from CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("eval_output/variant_bench_full"),
        help="Directory containing results.json and positions.h5",
    )
    parser.add_argument(
        "--engines",
        type=str,
        default=None,
        help="Comma-separated engine names to validate (default: all)",
    )
    args = parser.parse_args()

    results_path = args.data_dir / "results.json"
    h5_path = args.data_dir / "positions.h5"
    engine_set = set(args.engines.split(",")) if args.engines else None

    if not results_path.exists():
        print(f"ERROR: {results_path} not found", file=sys.stderr)
        sys.exit(1)

    errors = validate_sync(results_path, h5_path, engine_set)

    if errors:
        print(
            f"VALIDATION FAILED: {len(errors)} sync errors",
            file=sys.stderr,
        )
        for err in errors:
            print(f"  {err}", file=sys.stderr)
        sys.exit(1)
    else:
        print("Benchmark integrity OK: results.json and positions.h5 in sync")


if __name__ == "__main__":
    main()
