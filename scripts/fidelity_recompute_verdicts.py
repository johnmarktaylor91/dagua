#!/usr/bin/env python3
"""Recompute quality-metric statistical tests and verdicts from existing CSVs.

Reads per_seed_detail.csv (with quality metrics filled in by fidelity_add_metrics.py)
and per_graph_detail.csv, runs KS / Mann-Whitney / TOST / effect-size tests on the
quality metrics, applies BH correction, recomputes verdicts, and rewrites
per_graph_detail.csv and algorithm_summary.csv.

Usage:
    python scripts/fidelity_recompute_verdicts.py --data eval_output/fidelity_report/data
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Import shared logic from fidelity_analysis
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fidelity_analysis import (  # noqa: E402
    ALL_QUALITY_METRICS,
    TOST_MARGIN_FACTORS,
    TOST_MARGIN_LABELS,
    VARIANT_REGISTRY,
    PValueBucket,
    algorithm_summary_fieldnames,
    apply_bh_correction,
    bootstrap_mean_difference_ci,
    cliffs_delta,
    cohens_d,
    family_summary_rows,
    finalize_group_row,
    margin_for_metric,
    per_graph_fieldnames,
    rank_biserial_from_delta,
    tost_pvalue,
    write_csv,
)
from scipy.stats import ks_2samp, mannwhitneyu, ttest_ind  # noqa: E402

from dagua.eval.pipeline_io import stable_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute verdicts from existing CSVs")
    parser.add_argument("--data", type=Path, required=True, help="Directory with CSVs")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    return parser.parse_args()


def collect_metric_array(seed_rows: list[dict[str, str]], metric_name: str) -> np.ndarray:
    """Extract finite metric values from seed rows."""
    values = []
    for row in seed_rows:
        raw = row.get(metric_name, "nan")
        val = float(raw) if raw and raw != "nan" else math.nan
        if math.isfinite(val):
            values.append(val)
    return np.array(values, dtype=np.float64) if values else np.array([], dtype=np.float64)


def main() -> None:
    args = parse_args()
    data_dir = args.data
    bootstrap_samples = args.bootstrap_samples

    # Read per_seed_detail.csv
    seed_path = data_dir / "per_seed_detail.csv"
    print(f"[recompute] Reading {seed_path}", file=sys.stderr)
    with open(seed_path) as f:
        seed_rows = list(csv.DictReader(f))
    print(f"[recompute] {len(seed_rows)} seed rows loaded", file=sys.stderr)

    # Group seeds by (variant_id, graph_name, side)
    seed_groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in seed_rows:
        key = (row["variant_id"], row["graph_name"], row["side"])
        seed_groups[key].append(row)

    # Read per_graph_detail.csv
    graph_path = data_dir / "per_graph_detail.csv"
    print(f"[recompute] Reading {graph_path}", file=sys.stderr)
    with open(graph_path) as f:
        graph_rows = list(csv.DictReader(f))
    print(f"[recompute] {len(graph_rows)} graph rows loaded", file=sys.stderr)

    # Build variant lookup for stochastic flag
    variant_by_id = {v.variant_id: v for v in VARIANT_REGISTRY}

    # Populate internal fields needed by finalize_group_row / family_summary_rows
    for row in graph_rows:
        vid = row["variant_id"]
        variant = variant_by_id.get(vid)
        row["_variant_is_stochastic"] = variant.is_stochastic if variant else False
        row.setdefault("_rejection_count", 0)
        row.setdefault("_row_index", 0)

    # Set up p-value buckets for BH correction
    pvalue_buckets: dict[str, PValueBucket] = {
        "ks": PValueBucket(),
        "mannwhitney": PValueBucket(),
        "welch": PValueBucket(),
        "tost_0_5x": PValueBucket(),
        "tost_1x": PValueBucket(),
        "tost_1_5x": PValueBucket(),
        "tost_2x": PValueBucket(),
    }

    # For each per_graph row, compute metric tests
    updated = 0
    skipped_no_metrics = 0
    import time as _time

    _t_start = _time.monotonic()
    for row_idx, row in enumerate(graph_rows):
        variant_id = row["variant_id"]
        graph_name = row["graph_name"]
        variant = variant_by_id.get(variant_id)
        stochastic = variant.is_stochastic if variant else False

        if not stochastic:
            # Deterministic -- verdict based on displacement, no metric tests needed
            continue

        orig_seeds = seed_groups.get((variant_id, graph_name, "orig"), [])
        reimpl_seeds = seed_groups.get((variant_id, graph_name, "reimpl"), [])

        if not orig_seeds or not reimpl_seeds:
            continue

        has_metrics = False
        for metric_name in ALL_QUALITY_METRICS:
            orig_values = collect_metric_array(orig_seeds, metric_name)
            reimpl_values = collect_metric_array(reimpl_seeds, metric_name)

            if orig_values.size < 2 or reimpl_values.size < 2:
                continue

            has_metrics = True
            delta = cliffs_delta(orig_values, reimpl_values)
            ci_low, ci_high = bootstrap_mean_difference_ci(
                orig_values,
                reimpl_values,
                samples=bootstrap_samples,
                seed=stable_seed(variant_id, graph_name, metric_name),
            )
            row[f"{metric_name}_orig_mean"] = float(orig_values.mean())
            row[f"{metric_name}_orig_std"] = float(orig_values.std(ddof=1))
            row[f"{metric_name}_reimpl_mean"] = float(reimpl_values.mean())
            row[f"{metric_name}_reimpl_std"] = float(reimpl_values.std(ddof=1))
            row[f"{metric_name}_cohens_d"] = cohens_d(orig_values, reimpl_values)
            row[f"{metric_name}_cliffs_delta"] = delta
            row[f"{metric_name}_rank_biserial"] = rank_biserial_from_delta(delta)
            row[f"{metric_name}_bootstrap_diff_ci_low"] = ci_low
            row[f"{metric_name}_bootstrap_diff_ci_high"] = ci_high

            ks_pvalue = float(ks_2samp(orig_values, reimpl_values).pvalue)
            mw_pvalue = float(
                mannwhitneyu(orig_values, reimpl_values, alternative="two-sided").pvalue
            )
            _, welch_pvalue = ttest_ind(
                orig_values,
                reimpl_values,
                equal_var=False,
                alternative="two-sided",
            )
            row[f"{metric_name}_ks_pvalue_raw"] = ks_pvalue
            row[f"{metric_name}_mannwhitney_pvalue_raw"] = mw_pvalue
            row[f"{metric_name}_welch_pvalue_raw"] = (
                float(welch_pvalue) if np.isfinite(welch_pvalue) else math.nan
            )
            pvalue_buckets["ks"].entries.append((row_idx, f"{metric_name}_ks_pvalue_bh", ks_pvalue))
            pvalue_buckets["mannwhitney"].entries.append(
                (row_idx, f"{metric_name}_mannwhitney_pvalue_bh", mw_pvalue)
            )
            pvalue_buckets["welch"].entries.append(
                (row_idx, f"{metric_name}_welch_pvalue_bh", float(welch_pvalue))
            )

            for factor in TOST_MARGIN_FACTORS:
                label = TOST_MARGIN_LABELS[factor]
                margin = margin_for_metric(metric_name, orig_values, factor)
                pvalue = tost_pvalue(orig_values, reimpl_values, margin)
                row[f"{metric_name}_tost_margin_{label}"] = margin
                row[f"{metric_name}_tost_pvalue_{label}_raw"] = pvalue
                pvalue_buckets[f"tost_{label}"].entries.append(
                    (row_idx, f"{metric_name}_tost_pvalue_{label}_bh", pvalue)
                )

        if has_metrics:
            updated += 1
        else:
            skipped_no_metrics += 1

        if (row_idx + 1) % 500 == 0:
            elapsed = _time.monotonic() - _t_start
            print(
                f"[recompute] {row_idx + 1}/{len(graph_rows)} rows processed "
                f"({elapsed:.0f}s, {updated} updated)",
                file=sys.stderr,
            )

    elapsed = _time.monotonic() - _t_start
    print(
        f"[recompute] All {len(graph_rows)} rows processed in {elapsed:.0f}s "
        f"({updated} updated, {skipped_no_metrics} skipped/no metrics)",
        file=sys.stderr,
    )

    # Apply BH correction across all p-values
    apply_bh_correction(graph_rows, pvalue_buckets)

    # Recompute verdicts
    for row in graph_rows:
        finalize_group_row(row)

    # Recompute family summaries
    summary_rows = family_summary_rows(graph_rows)

    # Write updated CSVs
    write_csv(graph_path, graph_rows, per_graph_fieldnames())
    write_csv(
        data_dir / "algorithm_summary.csv",
        summary_rows,
        algorithm_summary_fieldnames(),
    )
    print(
        "[recompute] Wrote updated per_graph_detail.csv and algorithm_summary.csv", file=sys.stderr
    )

    # Verdict breakdown
    verdicts: dict[str, int] = defaultdict(int)
    for row in summary_rows:
        verdicts[str(row.get("verdict", "unknown"))] += 1
    print("[recompute] Verdict breakdown:", file=sys.stderr)
    for v, c in sorted(verdicts.items(), key=lambda x: -x[1]):
        print(f"  {v}: {c}", file=sys.stderr)


if __name__ == "__main__":
    main()
