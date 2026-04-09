#!/usr/bin/env python3
"""Render a compact markdown quality/runtime report from QR sidecar CSVs.

Reads the sidecar CSVs produced by ``scripts/quality_runtime_analysis.py`` and
emits a short markdown summary. Large detail tables remain in CSV sidecars.
Pareto front plots can be generated as PNG sidecars.

Usage
-----
python scripts/generate_quality_runtime_report.py \
    --input eval_output/quality_runtime_report \
    --output eval_output/quality_runtime_report/report.md \
    [--plots]  # enabled by default; use --no-plots to disable
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def load_csv(path: Path) -> List[Dict[str, Any]]:
    """Load one CSV file into a list of row dictionaries.

    Parameters
    ----------
    path : Path
        CSV path to load.

    Returns
    -------
    List[Dict[str, Any]]
        Parsed rows. Missing files return an empty list.
    """
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_float(value: Any) -> Optional[float]:
    """Parse a possibly missing numeric value.

    Parameters
    ----------
    value : Any
        Candidate scalar value.

    Returns
    -------
    Optional[float]
        Parsed float when finite, else ``None``.
    """
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric_value) or math.isinf(numeric_value):
        return None
    return numeric_value


def parse_int(value: Any) -> Optional[int]:
    """Parse a possibly missing integer value.

    Parameters
    ----------
    value : Any
        Candidate scalar value.

    Returns
    -------
    Optional[int]
        Parsed integer when conversion succeeds, else ``None``.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def median(values: Iterable[float]) -> Optional[float]:
    """Return the median of finite numeric values.

    Parameters
    ----------
    values : Iterable[float]
        Numeric values to aggregate.

    Returns
    -------
    Optional[float]
        Median value, or ``None`` when empty.
    """
    finite_values = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite_values:
        return None
    midpoint = len(finite_values) // 2
    if len(finite_values) % 2 == 1:
        return finite_values[midpoint]
    return (finite_values[midpoint - 1] + finite_values[midpoint]) / 2.0


def fmt_num(value: Any, precision: int = 3) -> str:
    """Format one numeric value for markdown output.

    Parameters
    ----------
    value : Any
        Candidate scalar value.
    precision : int, optional
        Decimal precision for finite numeric values.

    Returns
    -------
    str
        Formatted number, or ``"-"`` for invalid values.
    """
    numeric_value = parse_float(value)
    if numeric_value is None:
        return "-"
    return f"{numeric_value:.{precision}f}"


def is_truthy(value: Any) -> bool:
    """Interpret common CSV truthy values.

    Parameters
    ----------
    value : Any
        Raw CSV field value.

    Returns
    -------
    bool
        ``True`` for common truthy string and numeric forms.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def build_dataset_snapshot(records_snapshot: List[Dict[str, Any]]) -> str:
    """Build the dataset snapshot section.

    Parameters
    ----------
    records_snapshot : List[Dict[str, Any]]
        Rows from ``analysis_records_snapshot.csv``.

    Returns
    -------
    str
        Markdown section.
    """
    if not records_snapshot:
        return "## Dataset Snapshot\n\n_(no records loaded)_"

    counts: Dict[str, int] = {}
    for row in records_snapshot:
        status = str(row.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1

    total_records = sum(counts.values())
    ordered_statuses = ["ok", "error", "skipped", "timeout", "running"]
    remaining_statuses = sorted(status for status in counts if status not in ordered_statuses)

    lines = ["## Dataset Snapshot", "", f"Total records: {total_records}"]
    for status in ordered_statuses + remaining_statuses:
        if status in counts:
            lines.append(f"- {status}: {counts[status]}")
    return "\n".join(lines)


def build_coverage_section(family_summary: List[Dict[str, Any]]) -> str:
    """Build the per-family coverage table.

    Parameters
    ----------
    family_summary : List[Dict[str, Any]]
        Rows from ``family_metric_summary.csv``.

    Returns
    -------
    str
        Markdown section.
    """
    if not family_summary:
        return "## Coverage\n\n_(no family summary)_"

    per_family: Dict[str, Dict[str, Any]] = {}
    for row in family_summary:
        family_name = str(row.get("graph_family", "-"))
        family_state = per_family.setdefault(
            family_name,
            {
                "graphs_total": parse_int(row.get("graphs_in_family_total")) or 0,
                "graphs_available": parse_int(row.get("graphs_in_family_available")) or 0,
                "dagua_coverage_ratio": None,
            },
        )
        total_value = parse_int(row.get("graphs_in_family_total"))
        available_value = parse_int(row.get("graphs_in_family_available"))
        if total_value is not None:
            family_state["graphs_total"] = total_value
        if available_value is not None:
            family_state["graphs_available"] = available_value
        if str(row.get("engine_name", "")) == "dagua":
            family_state["dagua_coverage_ratio"] = parse_float(row.get("coverage_ratio"))

    lines = [
        "## Coverage",
        "",
        "| Family | Total | Available | Dagua Coverage Ratio |",
        "| --- | --- | --- | --- |",
    ]
    for family_name in sorted(per_family):
        family_state = per_family[family_name]
        lines.append(
            "| "
            + " | ".join(
                [
                    family_name,
                    str(family_state["graphs_total"]),
                    str(family_state["graphs_available"]),
                    fmt_num(family_state["dagua_coverage_ratio"], 3),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def build_family_scorecards(family_summary: List[Dict[str, Any]]) -> str:
    """Build one compact family leader table per graph family.

    Parameters
    ----------
    family_summary : List[Dict[str, Any]]
        Rows from ``family_metric_summary.csv``.

    Returns
    -------
    str
        Markdown section.
    """
    if not family_summary:
        return "## Family Scorecards\n\n_(no family scorecards)_"

    by_family: Dict[str, List[Dict[str, Any]]] = {}
    for row in family_summary:
        family_name = str(row.get("graph_family", "-"))
        by_family.setdefault(family_name, []).append(row)

    lines = ["## Family Scorecards", ""]
    for family_name in sorted(by_family):
        family_rows = by_family[family_name]
        eligible_rows = [
            row
            for row in family_rows
            if "scorecard_eligible" not in row or is_truthy(row.get("scorecard_eligible"))
        ]
        scorecard_rows = eligible_rows or family_rows

        per_engine: Dict[str, Dict[str, Any]] = {}
        for row in scorecard_rows:
            engine_name = str(row.get("engine_name", "-"))
            engine_state = per_engine.setdefault(
                engine_name,
                {
                    "median_graph_rank": [],
                    "median_rel_best": [],
                    "median_runtime_rel_fastest": [],
                    "coverage_ratio": [],
                },
            )
            for key in (
                "median_graph_rank",
                "median_rel_best",
                "median_runtime_rel_fastest",
                "coverage_ratio",
            ):
                numeric_value = parse_float(row.get(key))
                if numeric_value is not None:
                    engine_state[key].append(numeric_value)

        aggregated_rows: List[Dict[str, Any]] = []
        for engine_name, engine_state in per_engine.items():
            aggregated_rows.append(
                {
                    "engine_name": engine_name,
                    "median_graph_rank": median(engine_state["median_graph_rank"]),
                    "median_rel_best": median(engine_state["median_rel_best"]),
                    "median_runtime_rel_fastest": median(
                        engine_state["median_runtime_rel_fastest"]
                    ),
                    "coverage_ratio": median(engine_state["coverage_ratio"]),
                }
            )

        if not aggregated_rows:
            continue

        aggregated_rows.sort(
            key=lambda row: (
                parse_float(row.get("median_graph_rank"))
                if parse_float(row.get("median_graph_rank")) is not None
                else float("inf"),
                parse_float(row.get("median_rel_best"))
                if parse_float(row.get("median_rel_best")) is not None
                else float("inf"),
                parse_float(row.get("median_runtime_rel_fastest"))
                if parse_float(row.get("median_runtime_rel_fastest")) is not None
                else float("inf"),
                str(row.get("engine_name", "")),
            )
        )

        metrics_ranked = sorted(
            {
                str(row.get("metric_name", "-"))
                for row in scorecard_rows
                if str(row.get("engine_name", "")) == str(aggregated_rows[0]["engine_name"])
            }
        )
        lines.append(f"### {family_name}")
        lines.append("")
        lines.append(
            f"Leaders aggregated across {len(metrics_ranked)} metrics for scorecard-eligible rows."
        )
        lines.append("")
        lines.append("| Engine | Median Rank | Median Rel Best | Median Runtime Rel | Coverage |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in aggregated_rows[:3]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("engine_name", "-")),
                        fmt_num(row.get("median_graph_rank"), 2),
                        fmt_num(row.get("median_rel_best"), 3),
                        fmt_num(row.get("median_runtime_rel_fastest"), 2),
                        fmt_num(row.get("coverage_ratio"), 2),
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def build_dagua_insights_section(insights: List[Dict[str, Any]]) -> str:
    """Build the dagua default insight table.

    Parameters
    ----------
    insights : List[Dict[str, Any]]
        Rows from ``dagua_default_insights.csv``.

    Returns
    -------
    str
        Markdown section.
    """
    if not insights:
        return "## Dagua Default Insights\n\n_(no actionable insights)_"

    priority = {
        "dagua_dominated": 0,
        "premium_quality": 1,
        "steal_from": 2,
        "dagua_competitor_winner": 3,
    }
    sorted_insights = sorted(
        insights,
        key=lambda row: (
            priority.get(str(row.get("insight_type", "")), 99),
            -(parse_float(row.get("quality_advantage_norm")) or 0.0),
            -(parse_float(row.get("quality_advantage")) or 0.0),
            str(row.get("graph_family", "")),
            str(row.get("metric_name", "")),
        ),
    )[:30]

    lines = [
        "## Dagua Default Insights",
        "",
        "| Insight | Family | Metric | Competitor | Delta | Runtime Ratio | Exemplar Range |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in sorted_insights:
        p25 = fmt_num(row.get("family_metric_p25"), 3)
        p50 = fmt_num(row.get("family_metric_p50"), 3)
        p75 = fmt_num(row.get("family_metric_p75"), 3)
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("insight_type", "-")),
                    str(row.get("graph_family", "-")),
                    str(row.get("metric_name", "-")),
                    str(row.get("competitor_engine_name", "-")),
                    fmt_num(row.get("quality_advantage"), 3),
                    fmt_num(row.get("runtime_ratio"), 2),
                    f"{p25}/{p50}/{p75}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def build_best_of_breed_section(best_of_breed: List[Dict[str, Any]]) -> str:
    """Build the cross-family Pareto summary section.

    Parameters
    ----------
    best_of_breed : List[Dict[str, Any]]
        Rows from ``best_of_breed_configs.csv``.

    Returns
    -------
    str
        Markdown section.
    """
    if not best_of_breed:
        return "## Best-of-Breed Configs\n\n_(no best-of-breed data)_"

    sorted_rows = sorted(
        best_of_breed,
        key=lambda row: (
            -(parse_int(row.get("pareto_family_count")) or 0),
            -(parse_int(row.get("pareto_metric_count")) or 0),
            -(parse_int(row.get("best_quality_count")) or 0),
            str(row.get("engine_name", "")),
        ),
    )[:20]

    lines = [
        "## Best-of-Breed Configs",
        "",
        "Engines appearing on Pareto fronts across multiple families.",
        "",
        "| Engine | Families | Best Quality | Fastest | Balanced |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in sorted_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("engine_name", "-")),
                    str(row.get("pareto_family_count", "-")),
                    str(row.get("best_quality_count", "-")),
                    str(row.get("fastest_count", "-")),
                    str(row.get("balanced_count", "-")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def build_artifact_index(data_dir: Path, plots_enabled: bool) -> str:
    """Build the artifact listing section.

    Parameters
    ----------
    data_dir : Path
        Directory containing sidecar artifacts.
    plots_enabled : bool
        Whether PNG plot generation was requested.

    Returns
    -------
    str
        Markdown section.
    """
    artifact_rows = load_csv(data_dir / "artifact_index.csv")
    lines = ["## Artifact Index", ""]

    if artifact_rows:
        lines.append("| Type | Path | Rows |")
        lines.append("| --- | --- | --- |")
        for row in artifact_rows[:50]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("artifact_type", "-")),
                        f"`{row.get('path', '-')}`",
                        str(row.get("rows", "-")),
                    ]
                )
                + " |"
            )
        return "\n".join(lines)

    lines.append("Artifacts present in the output directory:")
    lines.append("")
    for path in sorted(data_dir.glob("*.csv"))[:50]:
        lines.append(f"- `{path.name}`")
    if plots_enabled:
        png_paths = sorted(data_dir.glob("*.png"))
        if png_paths:
            lines.append("")
            lines.append("Pareto plots:")
            for path in png_paths[:30]:
                lines.append(f"- `{path.name}`")
    return "\n".join(lines)


def generate_pareto_plots(pareto_csvs: List[Path], output_dir: Path) -> List[Path]:
    """Generate Pareto PNGs from per-family/metric Pareto CSVs.

    Parameters
    ----------
    pareto_csvs : List[Path]
        Pareto CSV paths.
    output_dir : Path
        Directory to receive PNG files.

    Returns
    -------
    List[Path]
        Paths of plots written successfully.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    written_paths: List[Path] = []
    for csv_path in pareto_csvs:
        rows = load_csv(csv_path)
        points: List[tuple[float, float, str]] = []
        for row in rows:
            runtime_value = parse_float(
                row.get("median_runtime_rel_fastest", row.get("runtime_rel_fastest"))
            )
            quality_value = parse_float(row.get("median_rel_best", row.get("rel_best")))
            if runtime_value is None or quality_value is None:
                continue
            points.append((runtime_value, quality_value, str(row.get("engine_name", ""))))

        if not points:
            continue

        figure, axis = plt.subplots(figsize=(6.0, 5.0))
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        labels = [point[2] for point in points]
        axis.scatter(xs, ys, s=30)
        for x_value, y_value, label in zip(xs, ys, labels):
            axis.annotate(label, (x_value, y_value), fontsize=6)
        axis.set_xlabel("Runtime (rel fastest)")
        axis.set_ylabel("Quality gap to best (median rel_best)")
        if xs and max(xs) / max(min(xs), 1e-6) > 10.0:
            axis.set_xscale("log")
        axis.axhline(0.0, color="g", linestyle=":", alpha=0.3)
        axis.axvline(1.0, color="g", linestyle=":", alpha=0.3)
        axis.set_title(csv_path.stem)
        png_path = output_dir / f"{csv_path.stem}.png"
        figure.tight_layout()
        figure.savefig(png_path, dpi=100)
        plt.close(figure)
        written_paths.append(png_path)
    return written_paths


def build_markdown_report(data_dir: Path, plots_enabled: bool) -> str:
    """Build the markdown report body from QR sidecar CSVs.

    Parameters
    ----------
    data_dir : Path
        QR report output directory.
    plots_enabled : bool
        Whether plotting is enabled for this report.

    Returns
    -------
    str
        Complete markdown report.
    """
    records_snapshot = load_csv(data_dir / "analysis_records_snapshot.csv")
    family_summary = load_csv(data_dir / "family_metric_summary.csv")
    insights = load_csv(data_dir / "dagua_default_insights.csv")
    best_of_breed = load_csv(data_dir / "best_of_breed_configs.csv")

    sections = [
        "# Quality/Runtime Analysis",
        "",
        "Short markdown summary. Detailed tables remain in the sidecar CSVs.",
        "",
        build_dataset_snapshot(records_snapshot),
        "",
        build_coverage_section(family_summary),
        "",
        build_family_scorecards(family_summary),
        "",
        build_dagua_insights_section(insights),
        "",
        build_best_of_breed_section(best_of_breed),
        "",
        build_artifact_index(data_dir, plots_enabled),
        "",
    ]
    return "\n".join(sections)


def main() -> int:
    """Parse arguments and write the QR markdown report.

    Returns
    -------
    int
        Process exit status code.
    """
    parser = argparse.ArgumentParser(description="Render QR markdown report.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("eval_output/quality_runtime_report"),
        help="Directory containing quality/runtime analysis sidecar CSVs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_output/quality_runtime_report/report.md"),
        help="Markdown report path to write.",
    )
    parser.add_argument("--plots", dest="plots", action="store_true", default=True)
    parser.add_argument("--no-plots", dest="plots", action="store_false")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    written_plots: List[Path] = []
    if args.plots:
        pareto_csvs = sorted(args.input.glob("family_*__metric_*__pareto.csv"))
        written_plots = generate_pareto_plots(pareto_csvs, args.input)
        print(f"Wrote {len(written_plots)} Pareto PNGs", file=sys.stderr)

    markdown = build_markdown_report(args.input, args.plots)
    with args.output.open("w", encoding="utf-8") as handle:
        handle.write(markdown)

    print(f"Wrote {args.output} ({len(markdown)} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
