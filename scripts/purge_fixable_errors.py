"""Purge only the fixable-error records from a benchmark results.json.

Identifies records that match known fixable error patterns (stale API, chain
pos kwarg mismatch, watchdog stuck, sgd2 wrapper bugs) and removes them so a
subsequent ``--resume`` run re-attempts them with the fixed code.

Non-fixable errors (legitimate timeouts, unsupported algorithms, hard graph
constraints, igraph dqueue OOM) are preserved so the run does not waste time
re-attempting them.

Usage
-----
    python scripts/purge_fixable_errors.py \
        --results eval_output/benchmark_full/results.json \
        --positions-dir eval_output/benchmark_full/positions \
        [--dry-run]

The script prints a per-bucket breakdown before purging, backs up the
original results.json next to it with a ``.bak.<timestamp>`` suffix, and
removes any associated positions files in one atomic sweep.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

# Each bucket is (name, predicate, description). Predicates take an
# ``(error_message, engine_name)`` tuple because a few buckets are
# disambiguated by engine name even when the message overlaps.
Predicate = Callable[[str, str], bool]


def _match_stale_api(err: str, _eng: str) -> bool:
    """Records from before the ``_pipeline`` suffix rename."""
    return "has no attribute 'layout_" in err and "module 'dagua.layout.ops.pipelines." in err


def _match_kwarg_mismatch(err: str, _eng: str) -> bool:
    """``classic_kk_fr`` chain variant passing ``pos`` to ``layout_fr_pipeline``."""
    return "second pass failed" in err and "unexpected keyword argument 'pos'" in err


def _match_watchdog_stuck(err: str, _eng: str) -> bool:
    """Transient worker-pool watchdog kills — re-running usually succeeds."""
    return "watchdog: worker pool stuck" in err


def _match_sgd2_pos_nan(err: str, eng: str) -> bool:
    """sgd2_multi_ref 'pos is nan' caused by wrapper's broken edge filter."""
    return eng.startswith("sgd2_multi_ref") and err == "pos is nan"


def _match_sgd2_0d_tensor(err: str, eng: str) -> bool:
    """Cached 'len() of a 0-d tensor' records from before the criteria patch."""
    return eng.startswith("sgd2_multi_ref") and "len() of a 0-d tensor" in err


def _match_sgd2_num_samples(err: str, eng: str) -> bool:
    """sgd2_multi_ref DataLoader num_samples=0 on tiny graphs."""
    return eng.startswith("sgd2_multi_ref") and "num_samples should be a positive integer" in err


def _match_nan_divergence(err: str, eng: str) -> bool:
    """Our wrapper's own NaN-guard message (same root cause as pos is nan)."""
    return eng.startswith("sgd2_multi_ref") and "optimization diverged" in err


def _match_tsne_max_iter(err: str, _eng: str) -> bool:
    """sklearn 1.5+ rejects ``max_iter < 250`` in TSNE; fixed by clamp."""
    return "'max_iter' parameter of TSNE must be an int in the range" in err


FIXABLE_BUCKETS: list[tuple[str, Predicate, str]] = [
    (
        "STALE_API",
        _match_stale_api,
        "Cached from before the _pipeline suffix rename",
    ),
    (
        "KWARG_MISMATCH_KK_FR",
        _match_kwarg_mismatch,
        "classic_kk_fr chain passes pos= to layout_fr_pipeline",
    ),
    (
        "WATCHDOG_STUCK",
        _match_watchdog_stuck,
        "Transient worker-pool watchdog kill",
    ),
    (
        "SGD2_POS_NAN",
        _match_sgd2_pos_nan,
        "sgd2_multi_ref wrapper dropped edges where source > target",
    ),
    (
        "SGD2_0D_TENSOR",
        _match_sgd2_0d_tensor,
        "Stale — already fixed by _compat_criteria_patches",
    ),
    (
        "SGD2_NUM_SAMPLES",
        _match_sgd2_num_samples,
        "sgd2 crossings DataLoader on empty non_incident_edge_pairs",
    ),
    (
        "SGD2_NAN_DIVERGENCE",
        _match_nan_divergence,
        "Wrapper-level NaN guard (same root cause as SGD2_POS_NAN)",
    ),
    (
        "TSNE_MAX_ITER",
        _match_tsne_max_iter,
        "sklearn 1.5+ rejects max_iter<250; fixed by clamp in tsne_competitor",
    ),
]


def classify(err: str, eng: str) -> Optional[str]:
    """Return the bucket name for a fixable error, or None.

    Parameters
    ----------
    err : str
        Error message from the record.
    eng : str
        Engine name from the record.

    Returns
    -------
    str or None
        Bucket name when the error is fixable, else ``None``.
    """
    for name, predicate, _desc in FIXABLE_BUCKETS:
        if predicate(err, eng):
            return name
    return None


def find_fixable_keys(records: dict) -> dict[str, list[str]]:
    """Group record keys by fixable bucket.

    Parameters
    ----------
    records : dict
        Loaded ``results.json`` payload.

    Returns
    -------
    dict[str, list[str]]
        Mapping from bucket name to list of record keys to remove.
    """
    buckets: dict[str, list[str]] = {name: [] for name, _, _ in FIXABLE_BUCKETS}
    for key, value in records.items():
        if not isinstance(value, dict):
            continue
        if value.get("status") != "error":
            continue
        err = value.get("error") or ""
        eng = value.get("engine_name") or ""
        bucket = classify(err, eng)
        if bucket is not None:
            buckets[bucket].append(key)
    return buckets


def print_summary(buckets: dict[str, list[str]]) -> int:
    """Print a per-bucket summary to stdout and return total fixable count.

    Parameters
    ----------
    buckets : dict[str, list[str]]
        Fixable records grouped by bucket.

    Returns
    -------
    int
        Total number of records across all buckets.
    """
    total = 0
    print("=" * 60)
    print("Fixable error buckets")
    print("=" * 60)
    bucket_desc = {name: desc for name, _, desc in FIXABLE_BUCKETS}
    for name in buckets:
        cnt = len(buckets[name])
        total += cnt
        print(f"  {name:<22}  {cnt:>6}  {bucket_desc[name]}")
    print("-" * 60)
    print(f"  {'TOTAL':<22}  {total:>6}")
    return total


def positions_path_for(record: dict, positions_dir: Path) -> Optional[Path]:
    """Return the positions file path for a record, if one exists.

    Parameters
    ----------
    record : dict
        One benchmark record.
    positions_dir : Path
        Directory containing per-run ``.pt`` files.

    Returns
    -------
    Path or None
        Absolute path to the positions file if it exists, else ``None``.
    """
    positions_file = record.get("positions_file")
    if not positions_file:
        return None
    candidate = (
        positions_dir / positions_file
        if not Path(positions_file).is_absolute()
        else Path(positions_file)
    )
    if candidate.exists():
        return candidate
    return None


def atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON to ``path`` atomically via a temp file rename.

    Parameters
    ----------
    path : Path
        Destination results.json path.
    payload : dict
        Content to serialize.

    Returns
    -------
    None
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    tmp.replace(path)


def main(argv: Optional[list[str]] = None) -> int:
    """Script entry point.

    Parameters
    ----------
    argv : list[str] or None
        Command-line arguments, or ``None`` to read from ``sys.argv``.

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(description="Purge fixable error records.")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("eval_output/benchmark_full/results.json"),
        help="Path to results.json to purge.",
    )
    parser.add_argument(
        "--positions-dir",
        type=Path,
        default=Path("eval_output/benchmark_full/positions"),
        help="Directory holding per-run .pt position files (optional).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary without writing any changes.",
    )
    args = parser.parse_args(argv)

    if not args.results.exists():
        print(f"ERROR: results file not found: {args.results}", file=sys.stderr)
        return 2

    with args.results.open("r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"Loaded {len(records)} records from {args.results}")
    buckets = find_fixable_keys(records)
    total = print_summary(buckets)

    if total == 0:
        print("\nNothing to purge.")
        return 0

    # Use a set so ``k not in keys_to_remove`` stays O(1) during the dict
    # comprehension below.  Building a fresh ``set()`` inside the comprehension
    # is an O(N*M) trap on 1M+ record datasets.
    keys_to_remove: set[str] = {key for keys in buckets.values() for key in keys}
    print(f"\nWill remove {len(keys_to_remove)} records.")

    if args.dry_run:
        print("Dry run — no changes written.")
        return 0

    # Back up the original results.json.
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = args.results.with_suffix(f".json.bak.{timestamp}")
    shutil.copy2(args.results, backup_path)
    print(f"Backed up original to {backup_path}")

    # Remove positions files for the records we're removing.
    positions_removed = 0
    if args.positions_dir.exists():
        for key in keys_to_remove:
            record = records[key]
            pos_path = positions_path_for(record, args.positions_dir)
            if pos_path is not None:
                try:
                    pos_path.unlink()
                    positions_removed += 1
                except OSError as exc:
                    print(f"  warning: could not unlink {pos_path}: {exc}", file=sys.stderr)

    # Remove the records and write back.
    new_records = {k: v for k, v in records.items() if k not in keys_to_remove}
    atomic_write_json(args.results, new_records)
    print(f"Wrote {len(new_records)} surviving records to {args.results}")
    print(f"Removed {positions_removed} positions files from {args.positions_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
