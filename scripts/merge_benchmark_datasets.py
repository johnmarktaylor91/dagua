"""Merge ``benchmark_full`` into ``variant_bench_full`` for a single unified dataset.

The two benchmark output directories were populated by historical runs that
each partially covered the engine space. This script merges the records and
positions files from ``benchmark_full`` INTO ``variant_bench_full`` so there is
one unified dataset at ``eval_output/variant_bench_full/``.

Merge rules
-----------
- For every key in ``benchmark_full/results.json``:
    * If the key is not in ``variant_bench_full``, copy it in directly.
    * If the key is in both, prefer the record whose status is ``ok`` over any
      other status. Ties go to whichever has a valid ``positions_file``. Final
      tie-breaker is ``benchmark_full`` (the more recent dataset after the
      repair run).
- For every positions file referenced by a surviving benchmark_full record,
  copy it into ``variant_bench_full/positions/``. Skip if the destination
  already exists with the same SHA-256 to avoid duplicate work.
- Back up both ``results.json`` files before writing anything.

After a successful merge ``benchmark_full/`` still exists but should be
considered read-only archive data. Use ``--delete-source`` only after a clean
run to rename it to ``benchmark_full.merged_<timestamp>``.

Usage
-----
    python scripts/merge_benchmark_datasets.py
    python scripts/merge_benchmark_datasets.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _timestamp() -> str:
    """Return a UTC timestamp suitable for backup filenames."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_json(path: Path) -> dict:
    """Load a JSON file and return its decoded contents."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json_atomic(path: Path, payload: dict) -> None:
    """Write ``payload`` to ``path`` via a temp-file rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    tmp.replace(path)


def _with_source_dir(record: Any, source_dir: str) -> Any:
    """Return a record tagged with its originating benchmark directory.

    Parameters
    ----------
    record : Any
        JSON-decoded benchmark record.
    source_dir : str
        Directory name for the benchmark store that supplied ``record``.

    Returns
    -------
    Any
        Copy of ``record`` with ``source_dir`` populated when it is a mapping;
        non-mapping payloads are returned unchanged.
    """
    if not isinstance(record, dict):
        return record
    tagged = dict(record)
    tagged.setdefault("source_dir", source_dir)
    return tagged


def _prefer(record_bf: dict, record_vbf: dict) -> dict:
    """Return the preferred record among a pair that share a key.

    Preference:
    1. ``status == "ok"`` wins.
    2. If both are ok or both are non-ok, prefer the one with a
       ``positions_file`` that is not ``None``.
    3. Final tie-breaker: ``benchmark_full`` (more recent after the repair run).

    Parameters
    ----------
    record_bf : dict
        Record from ``benchmark_full``.
    record_vbf : dict
        Record from ``variant_bench_full``.
    """
    status_bf = record_bf.get("status") if isinstance(record_bf, dict) else None
    status_vbf = record_vbf.get("status") if isinstance(record_vbf, dict) else None
    if status_bf == "ok" and status_vbf != "ok":
        return record_bf
    if status_vbf == "ok" and status_bf != "ok":
        return record_vbf
    pos_bf = record_bf.get("positions_file") if isinstance(record_bf, dict) else None
    pos_vbf = record_vbf.get("positions_file") if isinstance(record_vbf, dict) else None
    if pos_bf and not pos_vbf:
        return record_bf
    if pos_vbf and not pos_bf:
        return record_vbf
    return record_bf


def _hash_file(path: Path) -> Optional[str]:
    """Return the SHA-256 hex digest of ``path``, or ``None`` if missing."""
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_positions(
    surviving_records: dict,
    bf_output_dir: Path,
    vbf_output_dir: Path,
    dry_run: bool,
) -> tuple[int, int, int]:
    """Copy positions files from benchmark_full to variant_bench_full.

    Only copies when the destination is missing or has a different hash.
    The ``positions_file`` field in each record is stored relative to the
    benchmark output directory (e.g. ``"positions/foo__bar.pt"``), not the
    positions subdirectory.

    Parameters
    ----------
    surviving_records : dict
        Records from the merged dataset whose positions we need.
    bf_output_dir : Path
        Source benchmark output directory (the parent of ``positions/``).
    vbf_output_dir : Path
        Destination benchmark output directory.
    dry_run : bool
        If True, count but do not copy.

    Returns
    -------
    (copied, skipped, missing) : tuple[int, int, int]
    """
    copied = 0
    skipped = 0
    missing = 0
    for rec in surviving_records.values():
        if not isinstance(rec, dict):
            continue
        positions_file = rec.get("positions_file")
        if not positions_file:
            continue
        src = bf_output_dir / positions_file
        if not src.exists():
            missing += 1
            continue
        dst = vbf_output_dir / positions_file
        if dst.exists():
            src_hash = _hash_file(src)
            dst_hash = _hash_file(dst)
            if src_hash == dst_hash:
                skipped += 1
                continue
        if dry_run:
            copied += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    return copied, skipped, missing


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point.

    Parameters
    ----------
    argv : list[str] or None
        CLI args, or ``None`` to use ``sys.argv``.

    Returns
    -------
    int
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Merge benchmark_full into variant_bench_full.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("eval_output/benchmark_full"),
        help="Source benchmark directory (will be merged FROM here).",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("eval_output/variant_bench_full"),
        help="Target benchmark directory (will be merged INTO here).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would happen without writing anything.",
    )
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="After a successful merge, rename the source dir with a merged_<ts> suffix.",
    )
    args = parser.parse_args(argv)

    src_results = args.source / "results.json"
    dst_results = args.target / "results.json"
    if not src_results.exists():
        print(f"ERROR: source results.json not found: {src_results}", file=sys.stderr)
        return 2
    if not dst_results.exists():
        print(f"ERROR: target results.json not found: {dst_results}", file=sys.stderr)
        return 2

    print(f"Loading source: {src_results}")
    src_records = _load_json(src_results)
    print(f"Loading target: {dst_results}")
    dst_records = _load_json(dst_results)

    print(f"  source records:  {len(src_records):>10}")
    print(f"  target records:  {len(dst_records):>10}")

    # Merge records.
    src_source_dir = args.source.name
    dst_source_dir = args.target.name
    src_records = {
        key: _with_source_dir(record, src_source_dir) for key, record in src_records.items()
    }
    dst_records = {
        key: _with_source_dir(record, dst_source_dir) for key, record in dst_records.items()
    }
    collisions = 0
    new_from_source = 0
    source_preferred = 0
    target_preferred = 0
    merged: dict = dict(dst_records)
    for key, rec in src_records.items():
        if key not in merged:
            merged[key] = rec
            new_from_source += 1
            continue
        collisions += 1
        preferred = _prefer(rec, merged[key])
        if preferred is rec:
            merged[key] = rec
            source_preferred += 1
        else:
            target_preferred += 1

    src_status = Counter(r.get("status", "?") for r in src_records.values() if isinstance(r, dict))
    dst_status = Counter(r.get("status", "?") for r in dst_records.values() if isinstance(r, dict))
    merged_status = Counter(r.get("status", "?") for r in merged.values() if isinstance(r, dict))

    print()
    print(f"Collisions:            {collisions:>10}")
    print(f"  source preferred:    {source_preferred:>10}")
    print(f"  target preferred:    {target_preferred:>10}")
    print(f"New records from src:  {new_from_source:>10}")
    print(f"Merged total:          {len(merged):>10}")
    print()
    print("Status breakdown:")
    print(f"{'':<20} {'source':>10} {'target':>10} {'merged':>10}")
    keys = sorted(set(src_status) | set(dst_status) | set(merged_status))
    for k in keys:
        print(
            f"  {k:<18} {src_status.get(k, 0):>10} {dst_status.get(k, 0):>10} "
            f"{merged_status.get(k, 0):>10}"
        )

    if args.dry_run:
        copied, skipped, missing = _copy_positions(
            surviving_records={k: v for k, v in src_records.items()},
            bf_output_dir=args.source,
            vbf_output_dir=args.target,
            dry_run=True,
        )
        print()
        print(
            f"[dry-run] Would copy: {copied} positions files "
            f"(skip {skipped} identical, {missing} missing on disk)."
        )
        print("Dry run complete — no changes written.")
        return 0

    # Back up both results.json.
    ts = _timestamp()
    src_backup = src_results.with_suffix(f".json.bak.{ts}")
    dst_backup = dst_results.with_suffix(f".json.bak.{ts}")
    shutil.copy2(src_results, src_backup)
    shutil.copy2(dst_results, dst_backup)
    print()
    print(f"Backed up source -> {src_backup}")
    print(f"Backed up target -> {dst_backup}")

    # Write merged results to target.
    _save_json_atomic(dst_results, merged)
    print(f"Wrote merged results to {dst_results} ({len(merged)} records)")

    # Copy positions files (paths are relative to each benchmark output dir).
    copied, skipped, missing = _copy_positions(
        surviving_records={k: v for k, v in src_records.items()},
        bf_output_dir=args.source,
        vbf_output_dir=args.target,
        dry_run=False,
    )
    print(f"Copied {copied} positions files from {args.source} to {args.target}")
    print(f"Skipped {skipped} already-identical positions files")
    if missing:
        print(f"Warning: {missing} positions files referenced in records but missing on disk")

    if args.delete_source:
        archive_path = args.source.parent / f"{args.source.name}.merged_{ts}"
        args.source.rename(archive_path)
        print(f"Renamed source to {archive_path}")

    print()
    print("Merge complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
