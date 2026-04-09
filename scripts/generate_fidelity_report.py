#!/usr/bin/env python3
"""Generate a short markdown fidelity report from the analysis CSVs.

Reads the four CSVs written by ``scripts/fidelity_analysis.py`` and emits
a markdown summary at ``<output_dir>/report.md``. Large detail tables are
kept in the sidecar CSVs; the markdown inlines only family-level
summaries and a failures section.

Usage
-----
python scripts/generate_fidelity_report.py \
    --input eval_output/fidelity_report/data \
    --output eval_output/fidelity_report/report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

QUALITY_METRICS: tuple[str, ...] = (
    "aspect_ratio",
    "dag_consistency",
    "edge_length_cv",
    "edge_straightness_mean_deg",
    "depth_spearman_rho",
    "overlap_count",
    "sampled_stress",
    "crossing_rate",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for markdown report generation.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate markdown fidelity report.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("eval_output/fidelity_report/data"),
        help="Directory containing fidelity CSV outputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_output/fidelity_report/report.md"),
        help="Markdown report destination.",
    )
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, Any]]:
    """Load a CSV into memory.

    Parameters
    ----------
    path : Path
        CSV file to load.

    Returns
    -------
    list[dict[str, Any]]
        Parsed CSV rows. Returns an empty list when the file is absent.
    """
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fmt_num(value: Any, precision: int = 3) -> str:
    """Format a numeric value for markdown output.

    Parameters
    ----------
    value : Any
        Raw value to format.
    precision : int, default=3
        Number of digits after the decimal point.

    Returns
    -------
    str
        Formatted number or ``"-"`` when the value is missing or non-finite.
    """
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "-"
    if math.isnan(parsed) or math.isinf(parsed):
        return "-"
    return f"{parsed:.{precision}f}"


def column_pass_rate(rows: list[dict[str, Any]], column_name: str) -> float:
    """Compute the fraction of rows whose numeric column value is below 0.05.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Input rows to inspect.
    column_name : str
        Column whose values are interpreted as p-values.

    Returns
    -------
    float
        Fraction of finite values below 0.05, or ``nan`` when no finite
        values are present.
    """
    passes = 0
    eligible = 0
    for row in rows:
        try:
            value = float(row.get(column_name, ""))
        except (TypeError, ValueError):
            continue
        if math.isnan(value):
            continue
        eligible += 1
        if value < 0.05:
            passes += 1
    if eligible == 0:
        return math.nan
    return passes / eligible


def build_executive_summary_table(algorithm_summary: list[dict[str, Any]]) -> str:
    """Build the family-level executive summary markdown table.

    Parameters
    ----------
    algorithm_summary : list[dict[str, Any]]
        Rows from ``algorithm_summary.csv``.

    Returns
    -------
    str
        Markdown table or a placeholder when no rows are present.
    """
    if not algorithm_summary:
        return "_(no algorithm summaries)_"
    headers = [
        "Family",
        "Verdict",
        "Stochastic",
        "N OK",
        "N Strong",
        "N Weak",
        "N Partial",
        "N Divergent",
        "Median Procrustes RMSD",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in sorted(algorithm_summary, key=lambda item: str(item.get("algorithm_family", ""))):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("algorithm_family", "-")),
                    str(row.get("family_verdict", row.get("verdict", "-"))),
                    str(row.get("is_stochastic", "-")),
                    str(row.get("num_graphs_paired_ok", "-")),
                    str(row.get("num_strong_equivalent", "-")),
                    str(row.get("num_weak_equivalent", "-")),
                    str(row.get("num_partial_match", "-")),
                    str(row.get("num_divergent", "-")),
                    fmt_num(row.get("procrustes_rmsd_median")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def build_failures_section(
    per_graph_detail: list[dict[str, Any]],
    per_seed_detail: list[dict[str, Any]],
    pairwise_similarity: list[dict[str, Any]],
) -> str:
    """Build a markdown section for divergent and partial-match variants.

    Parameters
    ----------
    per_graph_detail : list[dict[str, Any]]
        Rows from ``per_graph_detail.csv``.
    per_seed_detail : list[dict[str, Any]]
        Rows from ``per_seed_detail.csv``.
    pairwise_similarity : list[dict[str, Any]]
        Rows from ``pairwise_similarity.csv``.

    Returns
    -------
    str
        Markdown section summarizing the most important failures.
    """
    failing = [
        row
        for row in per_graph_detail
        if str(row.get("verdict", "")).lower() in {"divergent", "partial_match"}
    ]
    if not failing:
        return "No variants were flagged as divergent or partial_match.\n"

    lines = [
        f"## Failure breakdown ({len(failing)} variants)",
        "",
        (
            f"Context: {len(per_seed_detail)} per-seed rows and "
            f"{len(pairwise_similarity)} pairwise rows are available for forensic follow-up."
        ),
        "",
        "| Variant | Graph | Verdict | Total Rejected | Top rejection reason |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in failing[:50]:
        breakdown_json = row.get("rejection_breakdown_json", "{}")
        try:
            breakdown = json.loads(breakdown_json)
        except (TypeError, ValueError):
            breakdown = {}
        if breakdown:
            top_reason_name, top_reason_count = max(
                breakdown.items(),
                key=lambda item: item[1],
            )
            top_reason = f"{top_reason_name}={top_reason_count}"
        else:
            top_reason = "-"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("variant_id", row.get("variant", "-"))),
                    str(row.get("graph_name", "-")),
                    str(row.get("verdict", "-")),
                    str(row.get("total_rejected", "-")),
                    top_reason,
                ]
            )
            + " |"
        )
    if len(failing) > 50:
        lines.extend(
            [
                "",
                f"_({len(failing) - 50} more rows in per_graph_detail.csv)_",
            ]
        )
    return "\n".join(lines)


def build_procrustes_summary(per_graph_detail: list[dict[str, Any]]) -> str:
    """Summarize Procrustes equivalence columns across all variants.

    Parameters
    ----------
    per_graph_detail : list[dict[str, Any]]
        Rows from ``per_graph_detail.csv``.

    Returns
    -------
    str
        Markdown section summarizing TOST pass rates and RMSD sidecar columns.
    """
    if not per_graph_detail:
        return "_(no per-graph data)_"

    total = len(per_graph_detail)
    tost_1x = column_pass_rate(per_graph_detail, "procrustes_tost_pvalue_1x_bh")
    tost_2x = column_pass_rate(per_graph_detail, "procrustes_tost_pvalue_2x_bh")
    lines = [
        "## Procrustes equivalence (TOST)",
        "",
        f"- 1x margin pass rate: {fmt_num(tost_1x, 4)} ({total} variants)",
        f"- 2x margin pass rate: {fmt_num(tost_2x, 4)}",
        (
            "- Sidecar columns include `within_orig_rmsd_mean`, `reimpl_rmsd_mean`, "
            "`between_rmsd_mean`, `procrustes_tost_pvalue_1x_bh`, and "
            "`procrustes_tost_pvalue_2x_bh`."
        ),
        "",
        "See `per_graph_detail.csv` for per-variant Procrustes and TOST outputs.",
    ]
    return "\n".join(lines)


def build_metric_surface_section() -> str:
    """Describe the quality-metric and Welch-test surface tracked in the CSVs.

    Returns
    -------
    str
        Markdown section describing metric columns and test families.
    """
    metrics_label = ", ".join(QUALITY_METRICS)
    return "\n".join(
        [
            "## Metric surface",
            "",
            f"- Quality metrics surfaced in the CSVs: {metrics_label}.",
            (
                "- Per-metric statistical columns include `*_tost_pvalue_{margin}_bh`, "
                "`*_mannwhitney_pvalue_bh`, `*_ks_pvalue_bh`, and "
                "`*_welch_pvalue_bh`."
            ),
            (
                "- The markdown keeps these as sidecar evidence instead of inlining a large "
                "table for every metric-family combination."
            ),
        ]
    )


def build_markdown_report(data_dir: Path, output_path: Path) -> str:
    """Assemble the complete markdown report.

    Parameters
    ----------
    data_dir : Path
        Directory containing the analysis CSV artifacts.
    output_path : Path
        Destination markdown file path.

    Returns
    -------
    str
        Complete markdown report.
    """
    algorithm_summary = load_csv(data_dir / "algorithm_summary.csv")
    per_graph_detail = load_csv(data_dir / "per_graph_detail.csv")
    per_seed_detail = load_csv(data_dir / "per_seed_detail.csv")
    pairwise_similarity = load_csv(data_dir / "pairwise_similarity.csv")

    lines = [
        "# Dagua Fidelity Analysis Report",
        "",
        "Short markdown summary. Detail tables live in the sidecar CSVs.",
        "",
        f"Report target: `{output_path}`",
        "",
        "## Executive Summary",
        "",
        build_executive_summary_table(algorithm_summary),
        "",
        build_procrustes_summary(per_graph_detail),
        "",
        build_metric_surface_section(),
        "",
        build_failures_section(per_graph_detail, per_seed_detail, pairwise_similarity),
        "",
        "## Methodology",
        "",
        (
            "- **Procrustes alignment**: per-seed-pair SVD-based alignment accepts the "
            "better of reflected and non-reflected fits."
        ),
        (
            "- **Within-vs-between test**: the within-distribution is computed from "
            "within-original seed pairs only, not pooled with reimplementation seeds."
        ),
        (
            "- **Equivalence tests**: TOST at factors 0.5x, 1.0x, 1.5x, and 2.0x of "
            "std(within-original) for Procrustes distances, plus per-metric TOST."
        ),
        (
            "- **Difference tests**: KS, Mann-Whitney U, Welch t-test per metric, plus "
            "one-sided and two-sided Mann-Whitney U on Procrustes distributions."
        ),
        "- **BH correction**: applied per test bucket across all variants before verdict routing.",
        f"- **Quality metrics**: {', '.join(QUALITY_METRICS)}.",
        (
            "- **Seed counts**: deterministic originals may contribute a single seed; "
            "stochastic verdict routing uses the asymmetric seed-handling implemented "
            "in the analysis pipeline."
        ),
        "",
        "## Artifacts",
        "",
        f"- `{data_dir / 'algorithm_summary.csv'}` -- family-level verdicts",
        (
            f"- `{data_dir / 'per_graph_detail.csv'}` -- per-variant Procrustes, "
            "TOST, MWU, KS, and Welch p-values"
        ),
        f"- `{data_dir / 'per_seed_detail.csv'}` -- per-seed layout samples",
        f"- `{data_dir / 'pairwise_similarity.csv'}` -- pairwise Procrustes RMSDs",
        (
            f"- `{data_dir / 'validate_sync_telemetry.json'}` -- sync mismatch telemetry "
            "when HDF5 coverage lags results.json"
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    """Run the markdown report generator CLI.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args()
    markdown = build_markdown_report(args.input, args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(markdown)
    print(f"Wrote {args.output} ({len(markdown)} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
