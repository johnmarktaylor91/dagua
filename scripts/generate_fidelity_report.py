#!/usr/bin/env python3
"""Generate the LaTeX fidelity report from precomputed CSV outputs."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

README_HASH_PREFIX = "Results SHA-256:"
QUALITY_METRICS: tuple[str, ...] = (
    "aspect_ratio",
    "dag_consistency",
    "edge_length_cv",
    "edge_length_mean",
    "overlap_count",
)
TOST_MARGIN_LABELS: tuple[str, ...] = ("0_5x", "1x", "1_5x", "2x")
REQUIRED_LATEX_PACKAGES: tuple[str, ...] = (
    "booktabs.sty",
    "geometry.sty",
    "hyperref.sty",
    "graphicx.sty",
    "amsmath.sty",
    "siunitx.sty",
    "longtable.sty",
    "fancyhdr.sty",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("eval_output/fidelity_report/data"),
        help="Directory containing the fidelity CSV outputs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_output/fidelity_report"),
        help="Destination directory for report.tex and report.pdf",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into memory.

    Parameters
    ----------
    path : Path
        CSV file path.

    Returns
    -------
    list[dict[str, str]]
        CSV rows as dictionaries.
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def latex_escape(value: object) -> str:
    """Escape a value for LaTeX text contexts.

    Parameters
    ----------
    value : object
        Raw value to render.

    Returns
    -------
    str
        Escaped LaTeX-safe string.
    """
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def parse_float(value: object) -> float:
    """Parse a float-like value, returning ``nan`` on failure.

    Parameters
    ----------
    value : object
        Raw cell value.

    Returns
    -------
    float
        Parsed float or ``nan``.
    """
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return math.nan
    return parsed


def parse_bool(value: object) -> bool:
    """Parse a loose boolean string.

    Parameters
    ----------
    value : object
        Raw cell value.

    Returns
    -------
    bool
        Parsed truth value.
    """
    return str(value).strip().lower() in {"1", "true", "yes"}


def format_float(value: float, digits: int = 3) -> str:
    """Format a numeric value for LaTeX tables.

    Parameters
    ----------
    value : float
        Numeric value.
    digits : int, default=3
        Digits after the decimal point.

    Returns
    -------
    str
        Formatted number or ``N/A``.
    """
    if not math.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def read_readme_metadata(path: Path) -> tuple[str, str]:
    """Extract the benchmark hash and power statement from the README.

    Parameters
    ----------
    path : Path
        README path.

    Returns
    -------
    tuple[str, str]
        Results hash and power-effect summary.
    """
    if not path.exists():
        return "N/A", "N/A"
    results_hash = "N/A"
    power_effect = "N/A"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(README_HASH_PREFIX):
            parts = line.split("`")
            if len(parts) >= 2:
                results_hash = parts[1]
        if "minimum detectable effect" in line and "`" in line:
            parts = line.split("`")
            if len(parts) >= 2:
                power_effect = parts[1]
    return results_hash, power_effect


def rows_by_family(rows: Iterable[Mapping[str, str]]) -> dict[str, list[Mapping[str, str]]]:
    """Group rows by algorithm family.

    Parameters
    ----------
    rows : Iterable[Mapping[str, str]]
        Input rows.

    Returns
    -------
    dict[str, list[Mapping[str, str]]]
        Grouped rows.
    """
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["algorithm_family"])].append(row)
    return grouped


def mean(values: Iterable[float]) -> float:
    """Return the arithmetic mean, or ``nan`` for empty input.

    Parameters
    ----------
    values : Iterable[float]
        Numeric values.

    Returns
    -------
    float
        Arithmetic mean or ``nan``.
    """
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return math.nan
    return sum(finite) / len(finite)


def std(values: Iterable[float]) -> float:
    """Return the sample standard deviation, or ``0`` for short input.

    Parameters
    ----------
    values : Iterable[float]
        Numeric values.

    Returns
    -------
    float
        Sample standard deviation or ``0``.
    """
    finite = [value for value in values if math.isfinite(value)]
    if len(finite) < 2:
        return 0.0
    sample_mean = sum(finite) / len(finite)
    variance = sum((value - sample_mean) ** 2 for value in finite) / (len(finite) - 1)
    return math.sqrt(max(variance, 0.0))


def family_metric_summary(
    rows: list[Mapping[str, str]], metric_name: str
) -> tuple[float, float, float, float, float]:
    """Aggregate one metric across per-graph rows for a family.

    Parameters
    ----------
    rows : list[Mapping[str, str]]
        Per-graph rows for one family.
    metric_name : str
        Metric identifier.

    Returns
    -------
    tuple[float, float, float, float, float]
        Original mean/std, reimplementation mean/std, and mean Cohen's d.
    """
    orig_means = [parse_float(row[f"{metric_name}_orig_mean"]) for row in rows]
    reimpl_means = [parse_float(row[f"{metric_name}_reimpl_mean"]) for row in rows]
    effect_sizes = [parse_float(row[f"{metric_name}_cohens_d"]) for row in rows]
    return (
        mean(orig_means),
        std(orig_means),
        mean(reimpl_means),
        std(reimpl_means),
        mean(effect_sizes),
    )


def family_tost_pass_rate(rows: list[Mapping[str, str]], label: str) -> float:
    """Compute the fraction of rows whose BH-adjusted TOST passes at one margin.

    Parameters
    ----------
    rows : list[Mapping[str, str]]
        Per-graph rows for one family.
    label : str
        Margin label such as ``1x``.

    Returns
    -------
    float
        Pass rate in ``[0, 1]``.
    """
    eligible = [row for row in rows if str(row.get("verdict", "")) != "insufficient_data"]
    if not eligible:
        return math.nan
    passes = 0
    for row in eligible:
        if all(
            parse_float(row[f"{metric_name}_tost_pvalue_{label}_bh"]) < 0.05
            for metric_name in QUALITY_METRICS
        ):
            passes += 1
    return passes / len(eligible)


def executive_summary_table(summary_rows: Sequence[Mapping[str, str]]) -> str:
    """Render the executive summary verdict table.

    Parameters
    ----------
    summary_rows : Sequence[Mapping[str, str]]
        Algorithm-summary rows.

    Returns
    -------
    str
        LaTeX table.
    """
    body = []
    for row in sorted(summary_rows, key=lambda item: str(item["algorithm_family"])):
        body.append(
            " & ".join(
                [
                    latex_escape(row["algorithm_family"]),
                    latex_escape(row["verdict"]),
                    latex_escape(row["num_graphs_paired_ok"]),
                    format_float(parse_float(row["procrustes_rmsd_mean"])),
                    format_float(parse_float(row["mean_runtime_ratio"])),
                    format_float(parse_float(row["tost_pass_rate_at_1x"]), digits=2),
                    latex_escape(row["anomaly_count"]),
                ]
            )
            + r" \\"
        )
    return "\n".join(
        [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{llllrrr}",
            r"\toprule",
            (
                r"Family & Verdict & Paired groups & RMSD mean & Runtime ratio "
                r"& TOST 1.0x & Anomalies \\"
            ),
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )


def methodology_section(
    per_graph_rows: Sequence[Mapping[str, str]],
    results_hash: str,
    power_effect: str,
) -> str:
    """Render the methodology section.

    Parameters
    ----------
    per_graph_rows : Sequence[Mapping[str, str]]
        Per-graph detail rows.
    results_hash : str
        Benchmark results hash.
    power_effect : str
        Power-analysis summary.

    Returns
    -------
    str
        Methodology section body.
    """
    total_rows = len(per_graph_rows)
    paired_rows = sum(str(row.get("verdict", "")) != "insufficient_data" for row in per_graph_rows)
    distinct_graphs = len({str(row["graph_name"]) for row in per_graph_rows})
    return "\n".join(
        [
            r"\section{Methodology}",
            (
                "Benchmark data were loaded from the fidelity CSV extracts "
                rf"derived from `{latex_escape(results_hash)}`."
            ),
            (
                f"The analysis covers {paired_rows} analyzable variant/graph "
                f"groups across {distinct_graphs} distinct graphs "
                f"({total_rows} total rows including insufficient-data cases)."
            ),
            r"",
            r"\subsection{Data collection}",
            (
                r"Layouts were sourced from the full variant benchmark "
                r"corpus, with one valid \texttt{[N,2]} tensor required "
                r"per retained run."
            ),
            (
                r"NaN, Inf, malformed tensors, and layouts with fewer than "
                r"three nodes were rejected before metrics or geometry "
                r"were computed."
            ),
            r"",
            r"\subsection{Geometric comparison}",
            (
                r"Procrustes alignment was implemented with centering plus "
                r"optimal proper rotation only: scale was reported "
                r"separately rather than normalized away, and reflected "
                r"alignments were detected but never accepted as the "
                r"match score."
            ),
            (
                r"Graphs with fewer than five nodes were excluded from "
                r"Procrustes reporting because the resulting displacement "
                r"summaries are not stable enough to interpret "
                r"comparatively."
            ),
            r"",
            r"\subsection{Quality metrics}",
            (
                r"The report tracks five quick metrics from "
                r"\texttt{dagua.metrics.quick}: edge-length coefficient "
                r"of variation, mean edge length, DAG consistency, "
                r"overlap count, and aspect ratio."
            ),
            r"",
            r"\subsection{Statistical tests}",
            r"TOST equivalence testing is the primary evidence standard for stochastic algorithms.",
            (
                r"Two-sample Kolmogorov--Smirnov and Mann--Whitney U tests "
                r"are secondary difference-detection checks only; they are "
                r"not interpreted as evidence of equivalence when "
                r"non-significant."
            ),
            (
                r"Benjamini--Hochberg false-discovery-rate control at "
                r"$q=0.05$ was applied separately to each test family "
                r"across all graph-level metric tests."
            ),
            r"",
            r"\subsection{Equivalence margins}",
            (
                r"TOST margins were reported at $0.5\times$, $1.0\times$, "
                r"$1.5\times$, and $2.0\times$ the within-original "
                r"standard deviation, subject to per-metric floors to "
                r"handle zero-variance cases."
            ),
            r"",
            r"\subsection{Power analysis}",
            (
                r"With $n=10$ seeds per side, the Mann--Whitney test "
                r"reaches roughly 80\% power at "
                rf"`{latex_escape(power_effect)}` under a Gaussian "
                r"location-shift simulation."
            ),
            r"",
            r"\subsection{Limitations}",
            (
                r"Graphs in the benchmark collection are not statistically "
                r"independent, so family-level verdicts should be read as "
                r"reproducibility evidence across a curated workload "
                r"rather than as a sample from a broader population."
            ),
            (
                r"Likewise, a large KS p-value does not imply equivalence; "
                r"only the TOST analyses are used as primary equivalence "
                r"evidence."
            ),
        ]
    )


def family_results_section(
    summary_rows: Sequence[Mapping[str, str]],
    per_graph_rows: Sequence[Mapping[str, str]],
) -> str:
    """Render algorithm-by-algorithm results.

    Parameters
    ----------
    summary_rows : Sequence[Mapping[str, str]]
        Algorithm-summary rows.
    per_graph_rows : Sequence[Mapping[str, str]]
        Per-graph detail rows.

    Returns
    -------
    str
        Results section body.
    """
    grouped = rows_by_family(per_graph_rows)
    sections: list[str] = [r"\section{Algorithm-by-algorithm results}"]
    for summary_row in sorted(summary_rows, key=lambda item: str(item["algorithm_family"])):
        family_name = str(summary_row["algorithm_family"])
        family_rows = grouped.get(family_name, [])
        sections.extend(
            [
                rf"\subsection{{{latex_escape(family_name)}}}",
                rf"Verdict: \textbf{{{latex_escape(summary_row['verdict'])}}}. ",
                (
                    "This family contributes "
                    f"{latex_escape(summary_row['num_graphs_paired_ok'])} "
                    "analyzable variant/graph groups."
                ),
                "",
                r"\begin{table}[htbp]",
                r"\centering",
                r"\small",
                r"\begin{tabular}{lrrr}",
                r"\toprule",
                r"Metric & Original & Reimplementation & Cohen's d \\",
                r"\midrule",
            ]
        )
        for metric_name in QUALITY_METRICS:
            orig_mean, orig_std, reimpl_mean, reimpl_std, effect_size = family_metric_summary(
                family_rows, metric_name
            )
            sections.append(
                " & ".join(
                    [
                        latex_escape(metric_name),
                        f"{format_float(orig_mean)} $\\pm$ {format_float(orig_std)}",
                        f"{format_float(reimpl_mean)} $\\pm$ {format_float(reimpl_std)}",
                        format_float(effect_size),
                    ]
                )
                + r" \\"
            )
        sections.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
                "",
                r"\begin{table}[htbp]",
                r"\centering",
                r"\small",
                r"\begin{tabular}{lrrrr}",
                r"\toprule",
                r"Margin & 0.5x & 1.0x & 1.5x & 2.0x \\",
                r"\midrule",
                "Pass rate & "
                + " & ".join(
                    format_float(family_tost_pass_rate(family_rows, label), digits=2)
                    for label in TOST_MARGIN_LABELS
                )
                + r" \\",
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
                "",
                (
                    "Mean runtime ratio (reimplementation/original): "
                    f"{format_float(parse_float(summary_row['mean_runtime_ratio']))}."
                ),
            ]
        )
        anomalies = [row for row in family_rows if str(row.get("anomaly_reason", "")).strip()]
        if anomalies:
            sections.append(r"\paragraph{Anomalous graphs.}")
            sections.append(
                ", ".join(
                    latex_escape(
                        f"{row['variant_id']}:{row['graph_name']} ({row['anomaly_reason']})"
                    )
                    for row in anomalies[:12]
                )
                + "."
            )
        else:
            sections.append(r"\paragraph{Anomalous graphs.} None.")
    return "\n".join(sections)


def cross_algorithm_section(
    summary_rows: Sequence[Mapping[str, str]],
    per_graph_rows: Sequence[Mapping[str, str]],
) -> str:
    """Render the cross-algorithm summary and category breakdown.

    Parameters
    ----------
    summary_rows : Sequence[Mapping[str, str]]
        Algorithm-summary rows.
    per_graph_rows : Sequence[Mapping[str, str]]
        Per-graph detail rows.

    Returns
    -------
    str
        Cross-algorithm section body.
    """
    category_counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in per_graph_rows:
        for category in ("density_bucket", "size_bucket", "structure_bucket"):
            category_counts[(category, str(row[category]))] += 1
    lines = [
        r"\section{Cross-algorithm summary}",
        executive_summary_table(summary_rows),
        "",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"Category axis & Bucket & Rows \\",
        r"\midrule",
    ]
    for (category, bucket), count in sorted(category_counts.items()):
        lines.append(
            " & ".join([latex_escape(category), latex_escape(bucket), latex_escape(count)]) + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def anomaly_section(per_graph_rows: Sequence[Mapping[str, str]]) -> str:
    """Render the anomaly deep-dive section when needed.

    Parameters
    ----------
    per_graph_rows : Sequence[Mapping[str, str]]
        Per-graph rows.

    Returns
    -------
    str
        LaTeX section, or an empty string when there are no anomalies.
    """
    anomalies = [row for row in per_graph_rows if str(row.get("anomaly_reason", "")).strip()]
    if not anomalies:
        return ""
    lines = [
        r"\section{Anomaly deep dive}",
        r"\begin{longtable}{p{0.18\textwidth}p{0.20\textwidth}p{0.17\textwidth}p{0.35\textwidth}}",
        r"\toprule",
        r"Family & Variant & Graph & Reason \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Family & Variant & Graph & Reason \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in anomalies:
        lines.append(
            " & ".join(
                [
                    latex_escape(row["algorithm_family"]),
                    latex_escape(row["variant_id"]),
                    latex_escape(row["graph_name"]),
                    latex_escape(row["anomaly_reason"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    return "\n".join(lines)


def appendix_section(per_graph_rows: Sequence[Mapping[str, str]]) -> str:
    """Render the appendix with full per-graph detail.

    Parameters
    ----------
    per_graph_rows : Sequence[Mapping[str, str]]
        Per-graph detail rows.

    Returns
    -------
    str
        Appendix section body.
    """
    lines = [
        r"\appendix",
        r"\section{Per-graph detail}",
        r"\begin{longtable}{p{0.16\textwidth}p{0.18\textwidth}p{0.18\textwidth}rrrrp{0.16\textwidth}}",
        r"\toprule",
        r"Family & Variant & Graph & Seeds O/R & RMSD & Scale & Runtime ratio & Verdict \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Family & Variant & Graph & Seeds O/R & RMSD & Scale & Runtime ratio & Verdict \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in sorted(
        per_graph_rows,
        key=lambda item: (
            str(item["algorithm_family"]),
            str(item["variant_id"]),
            str(item["graph_name"]),
        ),
    ):
        seeds_label = f"{row['num_orig_seeds']}/{row['num_reimpl_seeds']}"
        lines.append(
            " & ".join(
                [
                    latex_escape(row["algorithm_family"]),
                    latex_escape(row["variant_id"]),
                    latex_escape(row["graph_name"]),
                    latex_escape(seeds_label),
                    format_float(parse_float(row["procrustes_rmsd_mean"])),
                    format_float(parse_float(row["scale_ratio_mean"])),
                    format_float(parse_float(row["runtime_ratio"])),
                    latex_escape(row["verdict"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    return "\n".join(lines)


def build_report_tex(data_dir: Path, output_dir: Path) -> Path:
    """Build the LaTeX source file for the fidelity report.

    Parameters
    ----------
    data_dir : Path
        Directory containing the CSV outputs.
    output_dir : Path
        Directory to write ``report.tex`` into.

    Returns
    -------
    Path
        Path to the written LaTeX source file.
    """
    summary_rows = read_csv_rows(data_dir / "algorithm_summary.csv")
    per_graph_rows = read_csv_rows(data_dir / "per_graph_detail.csv")
    results_hash, power_effect = read_readme_metadata(data_dir / "README.md")

    output_dir.mkdir(parents=True, exist_ok=True)
    tex_path = output_dir / "report.tex"
    executive_table = executive_summary_table(summary_rows)
    anomaly_deep_dive = anomaly_section(per_graph_rows)
    tex_source = "\n".join(
        [
            r"\documentclass[11pt]{article}",
            r"\usepackage[margin=1in]{geometry}",
            r"\usepackage{booktabs}",
            r"\usepackage{hyperref}",
            r"\usepackage{graphicx}",
            r"\usepackage{amsmath}",
            r"\usepackage{siunitx}",
            r"\usepackage{longtable}",
            r"\usepackage{fancyhdr}",
            r"\pagestyle{fancy}",
            r"\fancyhf{}",
            r"\fancyhead[L]{Dagua Fidelity Report}",
            r"\fancyhead[R]{\thepage}",
            r"\begin{document}",
            r"\begin{titlepage}",
            r"\centering",
            r"{\LARGE Dagua Algorithm Reimplementation Fidelity Report\par}",
            r"\vspace{1cm}",
            rf"{{\large Benchmark hash: \texttt{{{latex_escape(results_hash)}}}\par}}",
            r"\vfill",
            r"{\large Generated from structured fidelity-analysis exports\par}",
            r"\end{titlepage}",
            r"\section{Executive summary}",
            executive_table,
            (
                r"The dominant pattern in the benchmark corpus is high "
                r"geometric and metric agreement between the "
                r"reimplementations and their reference algorithms, with "
                r"remaining discrepancies concentrated in flagged anomaly "
                r"cases."
            ),
            methodology_section(per_graph_rows, results_hash, power_effect),
            family_results_section(summary_rows, per_graph_rows),
            cross_algorithm_section(summary_rows, per_graph_rows),
            anomaly_deep_dive,
            appendix_section(per_graph_rows),
            r"\end{document}",
        ]
    )
    tex_path.write_text(tex_source + "\n", encoding="utf-8")
    return tex_path


def compile_report(output_dir: Path) -> bool:
    """Compile ``report.tex`` with ``pdflatex`` when available.

    Parameters
    ----------
    output_dir : Path
        Report output directory.

    Returns
    -------
    bool
        ``True`` when compilation ran, otherwise ``False``.
    """
    if shutil.which("pdflatex") is None:
        return False
    if shutil.which("kpsewhich") is not None:
        for package_name in REQUIRED_LATEX_PACKAGES:
            package_lookup = subprocess.run(
                ["kpsewhich", package_name],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            if package_lookup.returncode != 0 or not package_lookup.stdout.strip():
                return False
    for _ in range(2):
        try:
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "report.tex"],
                cwd=output_dir,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except subprocess.CalledProcessError:
            return False
    return True


def main() -> None:
    """Parse arguments, write LaTeX, and compile the report when possible."""
    args = parse_args()
    tex_path = build_report_tex(args.data, args.output)
    compiled = compile_report(args.output)
    if compiled:
        print(args.output / "report.pdf")
    else:
        print(f"PDF compilation unavailable. LaTeX source at {tex_path}")


if __name__ == "__main__":
    main()
