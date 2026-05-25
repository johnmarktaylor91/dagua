#!/usr/bin/env python3
"""Check R41 fidelity verdict robustness under 30-seed subsampling."""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Optional, Sequence

import h5py

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.fidelity_analysis import family_summary_rows, load_results  # noqa: E402
from scripts.r35_robustness_check import (  # noqa: E402
    SUBSET_COUNT,
    SUBSET_SIZE,
    build_seed_index,
    build_subset_rows,
    classify_frequency,
    load_seed_metrics,
    read_csv_rows,
    sample_seed_sets,
    write_csv,
)

DEFAULT_CANDIDATE_BENCHMARK_DIRS: tuple[Path, ...] = (
    Path("eval_output/benchmark_100seed_final"),
    Path("eval_output/fidelity_report_100seed_final"),
)
DEFAULT_REPORT_DATA_DIR = Path("eval_output/fidelity_report_100seed_final/data")
DEFAULT_OUTPUT_DIR = Path("eval_output/algo_fidelity/round_41/robustness")
RANDOM_SEED = 41_041


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed CLI options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=None,
        help=(
            "Benchmark directory containing results.json and positions.h5. "
            "Defaults to the first 100-seed final directory with both files."
        ),
    )
    parser.add_argument(
        "--report-data-dir",
        type=Path,
        default=DEFAULT_REPORT_DATA_DIR,
        help="Final fidelity report data directory with per-seed and pairwise CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination directory for robustness CSV and summary.",
    )
    parser.add_argument("--subset-count", type=int, default=SUBSET_COUNT)
    parser.add_argument("--subset-size", type=int, default=SUBSET_SIZE)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def resolve_benchmark_dir(explicit_dir: Optional[Path]) -> Path:
    """Resolve the read-only benchmark artifact directory.

    Parameters
    ----------
    explicit_dir : pathlib.Path | None
        User-provided benchmark directory, if any.

    Returns
    -------
    pathlib.Path
        Directory containing both ``results.json`` and ``positions.h5``.
    """
    candidates = (explicit_dir,) if explicit_dir is not None else DEFAULT_CANDIDATE_BENCHMARK_DIRS
    for candidate in candidates:
        if candidate is None:
            continue
        if (candidate / "results.json").exists() and (candidate / "positions.h5").exists():
            return candidate
    joined = ", ".join(str(path) for path in candidates if path is not None)
    raise FileNotFoundError(f"No benchmark directory with results.json and positions.h5: {joined}")


def validate_inputs(benchmark_dir: Path, report_data_dir: Path) -> None:
    """Validate required input artifacts before writing outputs.

    Parameters
    ----------
    benchmark_dir : pathlib.Path
        Benchmark directory selected for ``results.json`` and ``positions.h5``.
    report_data_dir : pathlib.Path
        Fidelity report data directory used for cached metric aggregation rows.
    """
    required_paths = [
        benchmark_dir / "results.json",
        benchmark_dir / "positions.h5",
        report_data_dir / "algorithm_summary.csv",
        report_data_dir / "per_graph_detail.csv",
        report_data_dir / "per_seed_detail.csv",
        report_data_dir / "pairwise_similarity.csv",
    ]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        joined = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing required robustness inputs:\n{joined}")


def write_summary(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    subset_seeds: Sequence[set[int]],
    benchmark_dir: Path,
    report_data_dir: Path,
) -> None:
    """Write the Markdown robustness summary.

    Parameters
    ----------
    path : pathlib.Path
        Summary path.
    rows : Sequence[Mapping[str, object]]
        Per-variant robustness rows.
    subset_seeds : Sequence[set[int]]
        Seed subsets used for the run.
    benchmark_dir : pathlib.Path
        Source benchmark artifact directory.
    report_data_dir : pathlib.Path
        Source fidelity report data directory.
    """
    classification_counts = Counter(str(row["robustness_class"]) for row in rows)
    baseline_counts = Counter(str(row["full_verdict"]) for row in rows)
    lines = [
        "# R41 Robustness Check",
        "",
        f"- Benchmark artifacts: `{benchmark_dir}`",
        f"- Fidelity report data: `{report_data_dir}`",
        f"- Variants checked: {len(rows)}",
        f"- Subsamples: {len(subset_seeds)} x {len(next(iter(subset_seeds), []))} seeds",
        f"- Robust: {classification_counts.get('robust', 0)}",
        f"- Borderline: {classification_counts.get('borderline', 0)}",
        f"- Noisy: {classification_counts.get('noisy', 0)}",
        "",
        "## Robustness Tiers",
        "",
    ]
    for tier, count in sorted(classification_counts.items()):
        lines.append(f"- {tier}: {count}")
    lines.extend(["", "## Full-Verdict Counts", ""])
    for verdict, count in sorted(baseline_counts.items()):
        lines.append(f"- {verdict}: {count}")
    lines.extend(["", "## Subset Seeds", ""])
    for index, seeds in enumerate(subset_seeds, start=1):
        joined = " ".join(str(seed) for seed in sorted(seeds))
        lines.append(f"- subset_{index}: {joined}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalize_subset_rows(rows: Sequence[dict[str, object]]) -> None:
    """Fill compatibility defaults expected by the current aggregator.

    Parameters
    ----------
    rows : Sequence[dict[str, object]]
        Mutable per-graph subset rows built from cached report CSVs.
    """
    for row in rows:
        row.setdefault("hungarian_rmsd_mean", math.nan)
        row.setdefault("hungarian_rmsd_median", math.nan)
        row.setdefault("hungarian_rmsd_max", math.nan)


def main() -> None:
    """Run the R41 robustness check.

    Returns
    -------
    None
        Writes robustness CSV and Markdown summary to the output directory.
    """
    args = parse_args()
    benchmark_dir = resolve_benchmark_dir(args.benchmark_dir)
    validate_inputs(benchmark_dir, args.report_data_dir)

    results_path = benchmark_dir / "results.json"
    positions_path = benchmark_dir / "positions.h5"
    base_rows = read_csv_rows(args.report_data_dir / "per_graph_detail.csv")
    full_summary_rows = read_csv_rows(args.report_data_dir / "algorithm_summary.csv")
    full_verdict_by_family = {
        row["algorithm_family"]: row.get("verdict", "unknown") for row in full_summary_rows
    }
    records = load_results(results_path)
    all_seeds = sorted({record.seed for record in records.values() if record.seed is not None})
    subset_seeds = sample_seed_sets(
        all_seeds,
        args.subset_count,
        args.subset_size,
        args.random_seed,
    )
    seed_metrics = load_seed_metrics(args.report_data_dir / "per_seed_detail.csv")
    seed_index = build_seed_index(seed_metrics)

    verdicts_by_family: dict[str, list[str]] = defaultdict(list)
    with h5py.File(positions_path, "r"):
        for subset_index, selected in enumerate(subset_seeds, start=1):
            print(f"[r41] subset {subset_index}/{len(subset_seeds)}", file=sys.stderr)
            subset_rows = build_subset_rows(
                base_rows=base_rows,
                seed_metrics=seed_metrics,
                seed_index=seed_index,
                pairwise_path=args.report_data_dir / "pairwise_similarity.csv",
                selected_seeds=selected,
            )
            normalize_subset_rows(subset_rows)
            for summary_row in family_summary_rows(subset_rows):
                verdicts_by_family[str(summary_row["algorithm_family"])].append(
                    str(summary_row["verdict"])
                )

    output_rows: list[dict[str, object]] = []
    for family in sorted(full_verdict_by_family):
        verdicts = verdicts_by_family.get(family, [])
        counts = Counter(verdicts)
        output_rows.append(
            {
                "algorithm_family": family,
                "full_verdict": full_verdict_by_family.get(family, "unknown"),
                "robustness_class": classify_frequency(verdicts),
                "subset_verdicts": ";".join(verdicts),
                "strong_equivalent": counts.get("strong_equivalent", 0),
                "weak_equivalent": counts.get("weak_equivalent", 0),
                "partial_match": counts.get("partial_match", 0),
                "divergent": counts.get("divergent", 0),
                "identical": counts.get("identical", 0),
                "insufficient_data": counts.get("insufficient_data", 0),
            }
        )

    output_csv = args.output_dir / "robustness_per_variant.csv"
    write_csv(
        output_csv,
        output_rows,
        [
            "algorithm_family",
            "full_verdict",
            "robustness_class",
            "subset_verdicts",
            "strong_equivalent",
            "weak_equivalent",
            "partial_match",
            "divergent",
            "identical",
            "insufficient_data",
        ],
    )
    write_summary(
        args.output_dir / "SUMMARY.md",
        output_rows,
        subset_seeds,
        benchmark_dir,
        args.report_data_dir,
    )
    print(f"[r41] wrote {output_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
