#!/usr/bin/env python3
"""Check R35 fidelity verdict robustness under 30-seed subsampling."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import h5py
import numpy as np
from scipy.stats import mannwhitneyu

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dagua.eval.variants import VARIANT_REGISTRY  # noqa: E402
from scripts.fidelity_analysis import (  # noqa: E402
    ALL_QUALITY_METRICS,
    TOST_MARGIN_FACTORS,
    TOST_MARGIN_LABELS,
    PValueBucket,
    apply_bh_correction,
    cliffs_delta,
    cohens_d,
    family_summary_rows,
    finalize_group_row,
    initialize_metric_columns,
    load_results,
    margin_for_metric,
    metric_regression_pct,
    pairwise_statistics,
    rank_biserial_from_delta,
    relative_delta_pct,
    safe_mean,
    safe_std,
    tost_pvalue,
)

DEFAULT_BENCHMARK_DIR = Path("eval_output/benchmark_100seed_final")
DEFAULT_REPORT_DATA_DIR = Path("eval_output/fidelity_report_100seed_final/data")
DEFAULT_OUTPUT_DIR = Path("eval_output/algo_fidelity/round_35/robustness")
SUBSET_COUNT = 5
SUBSET_SIZE = 30
RANDOM_SEED = 35_035


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed CLI options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--report-data-dir", type=Path, default=DEFAULT_REPORT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--subset-count", type=int, default=SUBSET_COUNT)
    parser.add_argument("--subset-size", type=int, default=SUBSET_SIZE)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def optional_seed(value: object) -> Optional[int]:
    """Convert CSV or JSON seed values to an optional integer.

    Parameters
    ----------
    value : object
        Raw seed value.

    Returns
    -------
    int | None
        Parsed seed, or ``None`` for deterministic blank seeds.
    """
    if value in (None, "", "nan"):
        return None
    return int(float(str(value)))


def optional_float(value: object) -> float:
    """Convert a CSV value to float, preserving missing values as NaN.

    Parameters
    ----------
    value : object
        Raw numeric value.

    Returns
    -------
    float
        Parsed value or ``math.nan``.
    """
    if value in (None, ""):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into dictionaries.

    Parameters
    ----------
    path : pathlib.Path
        CSV path.

    Returns
    -------
    list[dict[str, str]]
        CSV rows.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    """Write dictionaries to a CSV file.

    Parameters
    ----------
    path : pathlib.Path
        Output CSV path.
    rows : Sequence[Mapping[str, object]]
        Output rows.
    fieldnames : Sequence[str]
        Column order.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def load_seed_metrics(
    per_seed_path: Path,
) -> dict[tuple[str, str, str, Optional[int]], dict[str, float]]:
    """Load per-seed quality metrics from the final fidelity report data.

    Parameters
    ----------
    per_seed_path : pathlib.Path
        Existing ``per_seed_detail.csv`` path.

    Returns
    -------
    dict[tuple[str, str, str, int | None], dict[str, float]]
        Metrics keyed by ``(variant_id, graph_name, side, seed)``.
    """
    metrics: dict[tuple[str, str, str, Optional[int]], dict[str, float]] = {}
    with per_seed_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (
                row["variant_id"],
                row["graph_name"],
                row["side"],
                optional_seed(row["seed"]),
            )
            metrics[key] = {
                metric_name: optional_float(row.get(metric_name))
                for metric_name in ALL_QUALITY_METRICS
            }
            metrics[key]["runtime_seconds"] = optional_float(row.get("runtime_seconds"))
    return metrics


def make_pvalue_buckets() -> dict[str, PValueBucket]:
    """Create the p-value buckets used by the fidelity analysis.

    Returns
    -------
    dict[str, PValueBucket]
        Empty correction buckets.
    """
    buckets = {
        "ks": PValueBucket(),
        "mannwhitney": PValueBucket(),
        "welch": PValueBucket(),
        "procrustes_one_sided": PValueBucket(),
        "procrustes_mannwhitney": PValueBucket(),
    }
    for label in TOST_MARGIN_LABELS.values():
        buckets[f"procrustes_tost_{label}"] = PValueBucket()
        buckets[f"tost_{label}"] = PValueBucket()
    return buckets


def subset_side_seeds(
    available_seeds: Iterable[Optional[int]],
    selected_seeds: set[int],
) -> list[Optional[int]]:
    """Keep deterministic seeds and selected stochastic seeds.

    Parameters
    ----------
    available_seeds : Iterable[int | None]
        Available per-side seeds.
    selected_seeds : set[int]
        Seed subset for this simulation.

    Returns
    -------
    list[int | None]
        Seeds included in the subsample.
    """
    ordered = sorted(available_seeds, key=lambda seed: (seed is None, seed))
    if len(ordered) <= 1:
        return ordered
    selected = [seed for seed in ordered if seed is None or seed in selected_seeds]
    return selected if selected else ordered


def metric_array(
    seed_metrics: Mapping[tuple[str, str, str, Optional[int]], Mapping[str, float]],
    variant_id: str,
    graph_name: str,
    side: str,
    seeds: Iterable[Optional[int]],
    metric_name: str,
) -> np.ndarray:
    """Collect one metric for selected seeds.

    Parameters
    ----------
    seed_metrics : Mapping[tuple[str, str, str, int | None], Mapping[str, float]]
        Per-seed metric lookup.
    variant_id : str
        Variant identifier.
    graph_name : str
        Graph identifier.
    side : str
        ``"orig"`` or ``"reimpl"``.
    seeds : Iterable[int | None]
        Seeds included in the subsample.
    metric_name : str
        Metric column to collect.

    Returns
    -------
    numpy.ndarray
        Finite metric values.
    """
    values = [
        seed_metrics.get((variant_id, graph_name, side, seed), {}).get(metric_name, math.nan)
        for seed in seeds
    ]
    return np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)


def fill_metric_tests(
    row: dict[str, Any],
    row_index: int,
    pvalue_buckets: dict[str, PValueBucket],
    seed_metrics: Mapping[tuple[str, str, str, Optional[int]], Mapping[str, float]],
    orig_seeds: Sequence[Optional[int]],
    reimpl_seeds: Sequence[Optional[int]],
) -> None:
    """Compute per-metric subset summaries and TOST values.

    Parameters
    ----------
    row : dict[str, Any]
        Pending per-graph row.
    row_index : int
        Row index for BH correction buckets.
    pvalue_buckets : dict[str, PValueBucket]
        Deferred p-value buckets.
    seed_metrics : Mapping[tuple[str, str, str, int | None], Mapping[str, float]]
        Per-seed metric lookup.
    orig_seeds : Sequence[int | None]
        Original-side seeds included in the subset.
    reimpl_seeds : Sequence[int | None]
        Reimplementation-side seeds included in the subset.
    """
    variant_id = str(row["variant_id"])
    graph_name = str(row["graph_name"])
    for metric_name in ALL_QUALITY_METRICS:
        orig_values = metric_array(
            seed_metrics,
            variant_id,
            graph_name,
            "orig",
            orig_seeds,
            metric_name,
        )
        reimpl_values = metric_array(
            seed_metrics,
            variant_id,
            graph_name,
            "reimpl",
            reimpl_seeds,
            metric_name,
        )
        if orig_values.size == 0 or reimpl_values.size == 0:
            continue
        row[f"{metric_name}_orig_mean"] = float(orig_values.mean())
        row[f"{metric_name}_orig_std"] = (
            float(orig_values.std(ddof=1)) if orig_values.size > 1 else 0.0
        )
        row[f"{metric_name}_reimpl_mean"] = float(reimpl_values.mean())
        row[f"{metric_name}_reimpl_std"] = (
            float(reimpl_values.std(ddof=1)) if reimpl_values.size > 1 else 0.0
        )
        row[f"{metric_name}_delta"] = (
            row[f"{metric_name}_reimpl_mean"] - row[f"{metric_name}_orig_mean"]
        )
        row[f"{metric_name}_delta_pct"] = relative_delta_pct(
            row[f"{metric_name}_orig_mean"],
            row[f"{metric_name}_reimpl_mean"],
            1e-12,
        )
        row[f"{metric_name}_regression_pct"] = metric_regression_pct(
            metric_name,
            row[f"{metric_name}_orig_mean"],
            row[f"{metric_name}_reimpl_mean"],
        )
        row[f"{metric_name}_cohens_d"] = cohens_d(orig_values, reimpl_values)
        delta = cliffs_delta(orig_values, reimpl_values)
        row[f"{metric_name}_cliffs_delta"] = delta
        row[f"{metric_name}_rank_biserial"] = rank_biserial_from_delta(delta)
        for factor in TOST_MARGIN_FACTORS:
            label = TOST_MARGIN_LABELS[factor]
            margin = margin_for_metric(metric_name, orig_values, factor)
            pvalue = tost_pvalue(orig_values, reimpl_values, margin)
            row[f"{metric_name}_tost_margin_{label}"] = margin
            row[f"{metric_name}_tost_pvalue_{label}_raw"] = pvalue
            if math.isfinite(pvalue):
                pvalue_buckets[f"tost_{label}"].entries.append(
                    (row_index, f"{metric_name}_tost_pvalue_{label}_bh", pvalue)
                )


def summarize_runtimes(
    row: dict[str, Any],
    seed_metrics: Mapping[tuple[str, str, str, Optional[int]], Mapping[str, float]],
    orig_seeds: Sequence[Optional[int]],
    reimpl_seeds: Sequence[Optional[int]],
) -> None:
    """Fill subset runtime summary columns.

    Parameters
    ----------
    row : dict[str, Any]
        Pending row.
    seed_metrics : Mapping[tuple[str, str, str, int | None], Mapping[str, float]]
        Per-seed metric lookup with runtime values.
    orig_seeds : Sequence[int | None]
        Original-side seeds.
    reimpl_seeds : Sequence[int | None]
        Reimplementation-side seeds.
    """
    variant_id = str(row["variant_id"])
    graph_name = str(row["graph_name"])
    orig_runtime = [
        seed_metrics.get((variant_id, graph_name, "orig", seed), {}).get(
            "runtime_seconds",
            math.nan,
        )
        for seed in orig_seeds
    ]
    reimpl_runtime = [
        seed_metrics.get((variant_id, graph_name, "reimpl", seed), {}).get(
            "runtime_seconds",
            math.nan,
        )
        for seed in reimpl_seeds
    ]
    orig_finite = [value for value in orig_runtime if math.isfinite(value)]
    reimpl_finite = [value for value in reimpl_runtime if math.isfinite(value)]
    row["runtime_orig_mean"] = safe_mean(orig_finite)
    row["runtime_reimpl_mean"] = safe_mean(reimpl_finite)
    if math.isfinite(row["runtime_orig_mean"]) and abs(row["runtime_orig_mean"]) > 1e-12:
        row["runtime_ratio"] = row["runtime_reimpl_mean"] / row["runtime_orig_mean"]


def build_seed_index(
    seed_metrics: Mapping[tuple[str, str, str, Optional[int]], Mapping[str, float]],
) -> dict[tuple[str, str, str], set[Optional[int]]]:
    """Build available seed sets from per-seed metrics.

    Parameters
    ----------
    seed_metrics : Mapping[tuple[str, str, str, int | None], Mapping[str, float]]
        Per-seed metric lookup.

    Returns
    -------
    dict[tuple[str, str, str], set[int | None]]
        Available seeds keyed by ``(variant_id, graph_name, side)``.
    """
    seed_index: dict[tuple[str, str, str], set[Optional[int]]] = defaultdict(set)
    for variant_id, graph_name, side, seed in seed_metrics:
        seed_index[(variant_id, graph_name, side)].add(seed)
    return seed_index


def pair_seed_selected(seed: Optional[int], selected_seeds: set[int]) -> bool:
    """Return whether a pairwise seed belongs to a subset.

    Parameters
    ----------
    seed : int | None
        Pairwise seed value.
    selected_seeds : set[int]
        Selected stochastic seeds.

    Returns
    -------
    bool
        ``True`` for deterministic blank seeds or selected stochastic seeds.
    """
    return seed is None or seed in selected_seeds


def fill_pairwise_from_csv(
    rows: Sequence[dict[str, Any]],
    row_index_by_key: Mapping[tuple[str, str], int],
    pairwise_path: Path,
    selected_seeds: set[int],
    pvalue_buckets: dict[str, PValueBucket],
) -> None:
    """Fill Procrustes subset summaries from existing pairwise report rows.

    Parameters
    ----------
    rows : Sequence[dict[str, Any]]
        Pending per-graph rows.
    row_index_by_key : Mapping[tuple[str, str], int]
        Row index by ``(variant_id, graph_name)``.
    pairwise_path : pathlib.Path
        Existing ``pairwise_similarity.csv`` path.
    selected_seeds : set[int]
        Selected stochastic seeds.
    pvalue_buckets : dict[str, PValueBucket]
        Deferred p-value buckets.
    """
    values: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {
            "between": [],
            "within_orig": [],
            "within_reimpl": [],
            "scale": [],
            "max_displacement": [],
            "reflected": [],
        }
    )
    with pairwise_path.open(newline="", encoding="utf-8") as handle:
        for pair_row in csv.DictReader(handle):
            seed_a = optional_seed(pair_row.get("seed_a"))
            seed_b = optional_seed(pair_row.get("seed_b"))
            if not (
                pair_seed_selected(seed_a, selected_seeds)
                and pair_seed_selected(seed_b, selected_seeds)
            ):
                continue
            key = (pair_row["variant_id"], pair_row["graph_name"])
            if key not in row_index_by_key:
                continue
            comparison_type = pair_row["comparison_type"]
            bucket = values[key]
            rmsd = optional_float(pair_row.get("procrustes_rmsd"))
            if comparison_type == "orig-reimpl":
                bucket["between"].append(rmsd)
                bucket["scale"].append(optional_float(pair_row.get("scale_ratio")))
                bucket["max_displacement"].append(
                    optional_float(pair_row.get("max_node_displacement"))
                )
                if str(pair_row.get("reflected", "")).lower() in {"true", "1"}:
                    bucket["reflected"].append(1.0)
            elif comparison_type == "orig-orig":
                bucket["within_orig"].append(rmsd)
            elif comparison_type == "reimpl-reimpl":
                bucket["within_reimpl"].append(rmsd)

    for key, bucket in values.items():
        row_index = row_index_by_key[key]
        row = rows[row_index]
        between = [value for value in bucket["between"] if math.isfinite(value)]
        within_orig = [value for value in bucket["within_orig"] if math.isfinite(value)]
        within_reimpl = [value for value in bucket["within_reimpl"] if math.isfinite(value)]
        if not between:
            continue
        summary = pairwise_statistics(between)
        row["procrustes_rmsd_mean"] = summary["mean"]
        row["procrustes_rmsd_std"] = summary["std"]
        row["procrustes_rmsd_max"] = summary["max"]
        row["scale_ratio_mean"] = safe_mean(
            [value for value in bucket["scale"] if math.isfinite(value)]
        )
        row["scale_ratio_std"] = safe_std(
            [value for value in bucket["scale"] if math.isfinite(value)]
        )
        row["reflected"] = bool(bucket["reflected"])
        max_values = [value for value in bucket["max_displacement"] if math.isfinite(value)]
        row["max_node_displacement"] = max(max_values) if max_values else math.nan
        row["within_rmsd_mean"] = safe_mean(within_orig)
        row["within_rmsd_std"] = safe_std(within_orig)
        row["reimpl_rmsd_mean"] = safe_mean(within_reimpl)
        row["reimpl_rmsd_std"] = safe_std(within_reimpl)
        row["between_rmsd_mean"] = safe_mean(between)
        if within_orig and safe_mean(within_orig) > 0.0:
            row["rmsd_ratio"] = safe_mean(between) / max(safe_mean(within_orig), 1e-12)
        if len(within_orig) >= 2 and len(between) >= 2:
            _, wb_pvalue = mannwhitneyu(between, within_orig, alternative="greater")
            row["within_vs_between_pvalue"] = float(wb_pvalue)
            pvalue_buckets["procrustes_one_sided"].entries.append(
                (row_index, "within_vs_between_pvalue_bh", float(wb_pvalue))
            )
            std_floor = max(float(np.std(within_orig, ddof=1)), 1e-6)
            for factor in TOST_MARGIN_FACTORS:
                label = TOST_MARGIN_LABELS[factor]
                margin = factor * std_floor
                pvalue = tost_pvalue(
                    np.asarray(within_orig, dtype=np.float64),
                    np.asarray(between, dtype=np.float64),
                    margin,
                )
                row[f"procrustes_tost_margin_{label}"] = margin
                row[f"procrustes_tost_pvalue_{label}_raw"] = pvalue
                if math.isfinite(pvalue):
                    pvalue_buckets[f"procrustes_tost_{label}"].entries.append(
                        (row_index, f"procrustes_tost_pvalue_{label}_bh", pvalue)
                    )


def build_subset_rows(
    base_rows: Sequence[Mapping[str, str]],
    seed_metrics: Mapping[tuple[str, str, str, Optional[int]], Mapping[str, float]],
    seed_index: Mapping[tuple[str, str, str], set[Optional[int]]],
    pairwise_path: Path,
    selected_seeds: set[int],
) -> list[dict[str, Any]]:
    """Build per-graph fidelity rows for one seed subset.

    Parameters
    ----------
    base_rows : Sequence[Mapping[str, str]]
        Full-report per-graph rows used for metadata defaults.
    seed_metrics : Mapping[tuple[str, str, str, int | None], Mapping[str, float]]
        Per-seed quality metric lookup.
    seed_index : Mapping[tuple[str, str, str], set[int | None]]
        Available seeds by variant, graph, and side.
    pairwise_path : pathlib.Path
        Existing ``pairwise_similarity.csv`` path.
    selected_seeds : set[int]
        Seeds for this subset.

    Returns
    -------
    list[dict[str, Any]]
        Finalized per-graph rows for family aggregation.
    """
    variant_by_id = {variant.variant_id: variant for variant in VARIANT_REGISTRY}
    pvalue_buckets = make_pvalue_buckets()
    rows: list[dict[str, Any]] = []
    row_index_by_key: dict[tuple[str, str], int] = {}
    for row_index, base_row in enumerate(base_rows):
        variant_id = base_row["variant_id"]
        graph_name = base_row["graph_name"]
        variant = variant_by_id.get(variant_id)
        row: dict[str, Any] = dict(base_row)
        initialize_metric_columns(row)
        row["_variant_is_stochastic"] = bool(variant.is_stochastic) if variant else True
        row["_deterministic_tier"] = int(float(str(row.get("_deterministic_tier") or 0)))
        row["_deterministic_verdict"] = ""
        row["_deterministic_rejection_reasons"] = ""
        row["_rejection_count"] = int(float(str(row.get("total_rejected") or 0)))
        row["verdict"] = "insufficient_data"
        row["anomaly_reason"] = ""
        orig_seeds = subset_side_seeds(
            seed_index.get((variant_id, graph_name, "orig"), set()),
            selected_seeds,
        )
        reimpl_seeds = subset_side_seeds(
            seed_index.get((variant_id, graph_name, "reimpl"), set()),
            selected_seeds,
        )
        row["num_orig_seeds"] = len(orig_seeds)
        row["num_reimpl_seeds"] = len(reimpl_seeds)
        summarize_runtimes(row, seed_metrics, orig_seeds, reimpl_seeds)
        fill_metric_tests(row, row_index, pvalue_buckets, seed_metrics, orig_seeds, reimpl_seeds)
        row_index_by_key[(variant_id, graph_name)] = len(rows)
        rows.append(row)

    fill_pairwise_from_csv(rows, row_index_by_key, pairwise_path, selected_seeds, pvalue_buckets)
    apply_bh_correction(rows, pvalue_buckets)
    for row in rows:
        finalize_group_row(row)
    return rows


def sample_seed_sets(
    all_seeds: Sequence[int],
    subset_count: int,
    subset_size: int,
    seed: int,
) -> list[set[int]]:
    """Sample deterministic seed subsets.

    Parameters
    ----------
    all_seeds : Sequence[int]
        Available benchmark seeds.
    subset_count : int
        Number of subsets to draw.
    subset_size : int
        Number of seeds per subset.
    seed : int
        Random generator seed.

    Returns
    -------
    list[set[int]]
        Seed subsets.
    """
    rng = np.random.default_rng(seed)
    ordered = np.asarray(sorted(all_seeds), dtype=np.int64)
    return [
        set(int(value) for value in rng.choice(ordered, size=subset_size, replace=False))
        for _ in range(subset_count)
    ]


def classify_frequency(verdicts: Sequence[str]) -> str:
    """Classify verdict robustness from subset verdicts.

    Parameters
    ----------
    verdicts : Sequence[str]
        Verdicts from all subsamples.

    Returns
    -------
    str
        ``robust``, ``borderline``, or ``noisy``.
    """
    counts = Counter(verdicts)
    top_count = max(counts.values()) if counts else 0
    if top_count >= 4:
        return "robust"
    if len(counts) >= 3 or top_count <= 2:
        return "noisy"
    return "borderline"


def write_summary(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    subset_seeds: Sequence[set[int]],
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
    """
    classification_counts = Counter(str(row["robustness_class"]) for row in rows)
    baseline_counts = Counter(str(row["full_verdict"]) for row in rows)
    lines = [
        "# R35 Robustness Check",
        "",
        f"- Variants checked: {len(rows)}",
        f"- Subsamples: {len(subset_seeds)} x {len(next(iter(subset_seeds), []))} seeds",
        f"- Robust: {classification_counts.get('robust', 0)}",
        f"- Borderline: {classification_counts.get('borderline', 0)}",
        f"- Noisy: {classification_counts.get('noisy', 0)}",
        "",
        "## Full-Verdict Counts",
        "",
    ]
    for verdict, count in sorted(baseline_counts.items()):
        lines.append(f"- {verdict}: {count}")
    lines.extend(["", "## Subset Seeds", ""])
    for index, seeds in enumerate(subset_seeds, start=1):
        joined = " ".join(str(seed) for seed in sorted(seeds))
        lines.append(f"- subset_{index}: {joined}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the R35 robustness check."""
    args = parse_args()
    results_path = args.benchmark_dir / "results.json"
    positions_path = args.benchmark_dir / "positions.h5"
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
            print(f"[r35] subset {subset_index}/{len(subset_seeds)}", file=sys.stderr)
            subset_rows = build_subset_rows(
                base_rows=base_rows,
                seed_metrics=seed_metrics,
                seed_index=seed_index,
                pairwise_path=args.report_data_dir / "pairwise_similarity.csv",
                selected_seeds=selected,
            )
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
    write_summary(args.output_dir / "SUMMARY.md", output_rows, subset_seeds)
    print(f"[r35] wrote {output_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
