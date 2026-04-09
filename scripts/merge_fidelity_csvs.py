#!/usr/bin/env python3
"""Merge fidelity analysis CSVs from a partial re-run into existing full CSVs.

Replaces rows for specified algorithm families in the existing CSVs with
rows from the partial re-run. Preserves all unchanged families.

Usage:
    python scripts/merge_fidelity_csvs.py \
        --existing eval_output/fidelity_report/data \
        --partial /tmp/fidelity_changed \
        --output eval_output/fidelity_report/data \
        --families classic_fa2_linlog,classic_neulay_default,...

Written for the incremental fidelity re-analysis workflow where only
changed algorithm families need re-processing.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from datetime import datetime
from pathlib import Path


def merge_csv(
    existing_path: Path,
    partial_path: Path,
    output_path: Path,
    families: set[str],
    family_column: str = "algorithm_family",
) -> tuple[int, int, int]:
    """Merge partial CSV rows into existing CSV.

    Parameters
    ----------
    existing_path : Path
        Full CSV from previous run.
    partial_path : Path
        Partial CSV with only changed families.
    output_path : Path
        Where to write merged result.
    families : set[str]
        Algorithm families to replace from partial.
    family_column : str
        Column name containing the family identifier.

    Returns
    -------
    tuple[int, int, int]
        (kept_from_existing, added_from_partial, total_rows)
    """
    # Read existing rows
    with open(existing_path, newline="") as f:
        reader = csv.DictReader(f)
        existing_fieldnames = list(reader.fieldnames or [])
        existing_rows = list(reader)

    # Read partial rows
    with open(partial_path, newline="") as f:
        reader = csv.DictReader(f)
        partial_fieldnames = list(reader.fieldnames or [])
        partial_rows = list(reader)

    # Validate: partial must contain all requested families that exist in existing
    existing_target_families = {r.get(family_column, "") for r in existing_rows} & families
    partial_row_families = {r.get(family_column, "") for r in partial_rows}
    missing_families = existing_target_families - partial_row_families
    if missing_families:
        raise ValueError(
            f"{partial_path} is missing requested families present in "
            f"{existing_path}: {sorted(missing_families)}"
        )

    # Validate: partial must have same columns as existing
    missing_columns = set(existing_fieldnames) - set(partial_fieldnames)
    if missing_columns:
        raise ValueError(
            f"{partial_path} is missing columns present in "
            f"{existing_path}: {sorted(missing_columns)}"
        )

    # Keep only non-target rows from existing, filter partial to target only
    kept = [r for r in existing_rows if r.get(family_column, "") not in existing_target_families]
    replacement = [r for r in partial_rows if r.get(family_column, "") in families]

    fieldnames = existing_fieldnames.copy()
    for fn in partial_fieldnames:
        if fn not in fieldnames:
            fieldnames.append(fn)

    merged = kept + replacement
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged)

    return len(kept), len(replacement), len(merged)


def main() -> None:
    """Parse arguments and merge CSVs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing", type=Path, required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--families",
        type=str,
        required=True,
        help="Comma-separated algorithm family names to replace",
    )
    args = parser.parse_args()

    families = set(args.families.split(","))
    print(f"Merging {len(families)} families from partial into existing", file=sys.stderr)

    csvs = [
        ("algorithm_summary.csv", "algorithm_family"),
        ("per_graph_detail.csv", "algorithm_family"),
        ("per_seed_detail.csv", "algorithm_family"),
        ("pairwise_similarity.csv", "algorithm_family"),
    ]

    for csv_name, col in csvs:
        existing = args.existing / csv_name
        partial = args.partial / csv_name
        output = args.output / csv_name

        if not existing.exists():
            print(f"  SKIP {csv_name}: existing not found", file=sys.stderr)
            continue
        if not partial.exists():
            print(f"  SKIP {csv_name}: partial not found", file=sys.stderr)
            continue

        kept, added, total = merge_csv(existing, partial, output, families, col)
        print(
            f"  {csv_name}: kept {kept} + added {added} = {total} rows",
            file=sys.stderr,
        )

    # Preserve the merged README when present so incremental merges do not
    # clobber provenance already recorded in the full output directory.
    merged_readme = args.output / "README.md"
    partial_readme = args.partial / "README.md"
    if merged_readme.exists() and partial_readme.exists():
        existing = merged_readme.read_text(encoding="utf-8")
        note = (
            "\n\n## Merge note\n\n"
            f"Merged in records from {args.partial} on {datetime.now().isoformat()}\n"
        )
        merged_readme.write_text(existing + note, encoding="utf-8")
    elif partial_readme.exists() and not merged_readme.exists():
        shutil.copy2(partial_readme, merged_readme)

    print("Merge complete", file=sys.stderr)


if __name__ == "__main__":
    main()
