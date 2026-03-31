#!/usr/bin/env python3
"""Post-flight validation for fidelity analysis output.

Checks that output CSVs are non-empty, have expected columns, have
non-trivial data, and differ from any previous run. Exits nonzero
if validation fails.

Written as enforcement code from retro 2026-03-30 after an overnight
analysis ran 9 hours on empty position data and produced identical
results to the pre-fix run without anyone noticing.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


def validate_output(data_dir: Path, previous_dir: Path | None = None) -> list[str]:
    """Validate fidelity analysis output.

    Parameters
    ----------
    data_dir : Path
        Directory containing fidelity CSVs.
    previous_dir : Path | None
        If provided, compare against previous run's CSVs.

    Returns
    -------
    list[str]
        Error/warning messages. Empty means all checks passed.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Check algorithm_summary.csv
    summary_path = data_dir / "algorithm_summary.csv"
    if not summary_path.exists():
        errors.append("algorithm_summary.csv does not exist")
        return errors

    with open(summary_path) as f:
        summary_rows = list(csv.DictReader(f))

    if len(summary_rows) == 0:
        errors.append("algorithm_summary.csv has 0 rows")
        return errors

    verdicts = Counter(r["verdict"] for r in summary_rows)
    total = len(summary_rows)
    print(
        f"Verdict breakdown: {dict(verdicts)} (total: {total})",
        file=sys.stderr,
    )

    # Check per_graph_detail.csv
    pg_path = data_dir / "per_graph_detail.csv"
    if not pg_path.exists():
        errors.append("per_graph_detail.csv does not exist")
    else:
        with open(pg_path) as f:
            pg_rows = list(csv.DictReader(f))

        if len(pg_rows) == 0:
            errors.append("per_graph_detail.csv has 0 rows")
        else:
            # Check for all-NaN RMSD (sign of missing positions)
            nan_rmsd = sum(
                1
                for r in pg_rows
                if not r.get("procrustes_rmsd_mean") or r["procrustes_rmsd_mean"] == "nan"
            )
            if nan_rmsd > len(pg_rows) * 0.5:
                errors.append(
                    f"per_graph_detail.csv: {nan_rmsd}/{len(pg_rows)} rows "
                    f"have NaN RMSD -- likely missing positions"
                )

            # Check paired counts
            zero_paired = sum(1 for r in pg_rows if r.get("verdict") == "insufficient_data")
            if zero_paired > len(pg_rows) * 0.5:
                warnings.append(
                    f"per_graph_detail.csv: {zero_paired}/{len(pg_rows)} rows are insufficient_data"
                )

    # Delta comparison against previous run
    if previous_dir and (previous_dir / "algorithm_summary.csv").exists():
        with open(previous_dir / "algorithm_summary.csv") as f:
            prev_rows = list(csv.DictReader(f))

        prev_verdicts = {r["algorithm_family"]: r["verdict"] for r in prev_rows}
        curr_verdicts = {r["algorithm_family"]: r["verdict"] for r in summary_rows}

        changed = 0
        for family in curr_verdicts:
            if family in prev_verdicts and curr_verdicts[family] != prev_verdicts[family]:
                changed += 1

        if changed == 0 and len(curr_verdicts) == len(prev_verdicts):
            # Check RMSD values too
            prev_rmsd = {
                r["algorithm_family"]: r.get("procrustes_rmsd_mean", "") for r in prev_rows
            }
            curr_rmsd = {
                r["algorithm_family"]: r.get("procrustes_rmsd_mean", "") for r in summary_rows
            }
            rmsd_changed = sum(
                1 for f in curr_rmsd if f in prev_rmsd and curr_rmsd[f] != prev_rmsd[f]
            )
            if rmsd_changed == 0:
                errors.append(
                    "OUTPUT IDENTICAL TO PREVIOUS RUN. "
                    "All verdicts AND RMSD values unchanged. "
                    "Did your code changes take effect? "
                    "Check that positions.h5 has fresh data."
                )
            elif changed == 0:
                warnings.append(
                    f"All verdicts identical to previous run, but "
                    f"{rmsd_changed} families have different RMSD. "
                    f"Verdicts may not have moved despite data changes."
                )
        else:
            # Print delta summary (informational, not an error)
            print(
                f"Delta from previous run: {changed} families changed verdict",
                file=sys.stderr,
            )
            for family in sorted(curr_verdicts):
                if family in prev_verdicts and curr_verdicts[family] != prev_verdicts[family]:
                    print(
                        f"  {family}: {prev_verdicts[family]} -> {curr_verdicts[family]}",
                        file=sys.stderr,
                    )

    all_issues = errors + warnings
    return all_issues


def main() -> None:
    """Run output validation from CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Directory containing fidelity CSVs",
    )
    parser.add_argument(
        "--previous",
        type=Path,
        default=None,
        help="Previous run's data directory for delta comparison",
    )
    args = parser.parse_args()

    issues = validate_output(args.data, args.previous)

    if issues:
        print(
            f"VALIDATION: {len(issues)} issues found",
            file=sys.stderr,
        )
        for issue in issues:
            print(f"  {issue}", file=sys.stderr)
        # Exit 1 only for errors (not warnings)
        has_errors = any(
            "IDENTICAL" in i or "does not exist" in i or "0 rows" in i or "NaN RMSD" in i
            for i in issues
        )
        sys.exit(1 if has_errors else 0)
    else:
        print("Fidelity output validation OK")


if __name__ == "__main__":
    main()
