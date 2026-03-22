#!/usr/bin/env python3
"""Compare benchmarked reimplementation variants against their originals."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dagua.eval.graphs import get_test_graphs  # noqa: E402
from dagua.eval.variants import (  # noqa: E402
    VARIANT_REGISTRY,
    algorithm_family,
    engine_is_stochastic,
    original_variant_name,
)
from scripts.run_benchmark import BenchmarkRecord  # noqa: E402

METRICS_TO_COMPARE = (
    "aspect_ratio",
    "dag_consistency",
    "edge_length_cv",
    "edge_length_mean",
    "overlap_count",
)
DETERMINISTIC_MATCH_TOLERANCE = 1e-6


@dataclass(frozen=True)
class MetricAssessment:
    """Comparison summary for one metric on one graph.

    Parameters
    ----------
    metric_name : str
        Metric identifier from ``dagua.metrics.quick``.
    reimpl_summary : str
        Human-readable reimplementation summary.
    original_summary : str
        Human-readable original summary.
    assessment : str
        Short match label for this metric.
    """

    metric_name: str
    reimpl_summary: str
    original_summary: str
    assessment: str


@dataclass(frozen=True)
class GraphVariantReport:
    """One report row for a variant/graph comparison.

    Parameters
    ----------
    variant_id : str
        Reimplementation variant competitor name.
    graph_name : str
        Stable graph name.
    sample_summary : str
        Human-readable sample count summary.
    metrics_summary : str
        Compact metric comparison summary.
    match_assessment : str
        Overall graph-level assessment.
    """

    variant_id: str
    graph_name: str
    sample_summary: str
    metrics_summary: str
    match_assessment: str


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed CLI options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=str,
        default="eval_output/benchmark_full/results.json",
        help="Path to the benchmark results.json file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_output/benchmark_variants/comparison_report.md",
        help="Destination markdown report path",
    )
    parser.add_argument(
        "--graphs",
        type=str,
        default=None,
        help="Optional comma-separated graph filter",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Bootstrap iterations for stochastic comparisons",
    )
    return parser.parse_args()


def load_results(path: Path) -> dict[str, BenchmarkRecord]:
    """Load benchmark records from ``results.json``.

    Parameters
    ----------
    path : Path
        Results path.

    Returns
    -------
    dict[str, BenchmarkRecord]
        Parsed benchmark records keyed by record key.
    """
    if not path.exists():
        raise FileNotFoundError(f"results file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"results file is not a JSON object: {path}")
    return {
        str(key): BenchmarkRecord.from_dict(value)
        for key, value in payload.items()
        if isinstance(value, dict)
    }


def select_graph_names(graph_filter: Optional[str]) -> set[str]:
    """Resolve the optional graph-name filter.

    Parameters
    ----------
    graph_filter : str | None
        Raw CLI graph filter.

    Returns
    -------
    set[str]
        Selected graph names, or an empty set when no filter is supplied.
    """
    if graph_filter is None:
        return set()
    return {name.strip() for name in graph_filter.split(",") if name.strip()}


def load_graphs() -> dict[str, Any]:
    """Load benchmark graph objects keyed by name.

    Returns
    -------
    dict[str, Any]
        Test-graph registry keyed by graph name.
    """
    return {test_graph.name: test_graph for test_graph in get_test_graphs()}


def load_position_tensor(path: Path) -> Optional[torch.Tensor]:
    """Load one saved position tensor when valid.

    Parameters
    ----------
    path : Path
        Position file path.

    Returns
    -------
    torch.Tensor | None
        Loaded ``[N, 2]`` tensor, or ``None`` when loading/validation fails.
    """
    try:
        tensor = torch.load(path, map_location="cpu")
    except Exception:
        return None
    if not isinstance(tensor, torch.Tensor):
        return None
    if tensor.ndim != 2 or tensor.shape[1] != 2:
        return None
    return tensor


def compute_metrics(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
) -> dict[str, float]:
    """Compute quick quality metrics for one layout.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor shaped ``[N, 2]``.
    edge_index : torch.Tensor
        Graph edge index shaped ``[2, E]``.
    node_sizes : torch.Tensor
        Node-size tensor shaped ``[N, 2]``.

    Returns
    -------
    dict[str, float]
        Quick layout metrics.
    """
    from dagua.metrics import quick

    return {
        metric_name: float(metric_value)
        for metric_name, metric_value in quick(
            positions,
            edge_index,
            node_sizes=node_sizes,
        ).items()
        if metric_name in METRICS_TO_COMPARE
    }


def percentile_bounds(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Return equal-tailed percentile bounds.

    Parameters
    ----------
    values : numpy.ndarray
        Bootstrap statistic samples.
    alpha : float, default=0.05
        Two-sided tail probability.

    Returns
    -------
    tuple[float, float]
        Lower and upper percentile bounds.
    """
    lower = float(np.percentile(values, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(values, 100.0 * (1.0 - alpha / 2.0)))
    return lower, upper


def deterministic_metric_assessment(
    metric_name: str,
    reimpl_value: float,
    original_value: float,
) -> MetricAssessment:
    """Compare deterministic metric values directly.

    Parameters
    ----------
    metric_name : str
        Metric identifier.
    reimpl_value : float
        Reimplementation metric value.
    original_value : float
        Original metric value.

    Returns
    -------
    MetricAssessment
        Per-metric deterministic assessment.
    """
    delta = reimpl_value - original_value
    is_match = math.isclose(
        reimpl_value,
        original_value,
        rel_tol=0.0,
        abs_tol=DETERMINISTIC_MATCH_TOLERANCE,
    )
    return MetricAssessment(
        metric_name=metric_name,
        reimpl_summary=f"{reimpl_value:.4f}",
        original_summary=f"{original_value:.4f}",
        assessment="exact" if is_match else f"delta={delta:.4f}",
    )


def stochastic_metric_assessment(
    metric_name: str,
    reimpl_values: list[float],
    original_values: list[float],
    bootstrap_samples: int,
) -> MetricAssessment:
    """Compare stochastic metric distributions using bootstrap CIs.

    Parameters
    ----------
    metric_name : str
        Metric identifier.
    reimpl_values : list[float]
        Reimplementation metric samples.
    original_values : list[float]
        Original metric samples.
    bootstrap_samples : int
        Number of bootstrap iterations.

    Returns
    -------
    MetricAssessment
        Per-metric stochastic assessment.
    """
    reimpl_array = np.asarray(reimpl_values, dtype=np.float64)
    original_array = np.asarray(original_values, dtype=np.float64)
    rng = np.random.default_rng(42)

    boot_deltas = np.empty(bootstrap_samples, dtype=np.float64)
    for index in range(bootstrap_samples):
        reimpl_sample = rng.choice(reimpl_array, size=reimpl_array.size, replace=True)
        original_sample = rng.choice(original_array, size=original_array.size, replace=True)
        boot_deltas[index] = float(np.mean(reimpl_sample) - np.mean(original_sample))

    ci_low, ci_high = percentile_bounds(boot_deltas)
    within_scale = float(max(np.std(reimpl_array), np.std(original_array), 1e-9))
    ratio = abs(float(np.mean(reimpl_array) - np.mean(original_array))) / within_scale
    is_match = max(abs(ci_low), abs(ci_high)) <= within_scale
    return MetricAssessment(
        metric_name=metric_name,
        reimpl_summary=f"{np.mean(reimpl_array):.4f} +/- {np.std(reimpl_array):.4f}",
        original_summary=f"{np.mean(original_array):.4f} +/- {np.std(original_array):.4f}",
        assessment=(
            f"bootstrap match (ratio={ratio:.2f})"
            if is_match
            else f"bootstrap mismatch (ci=[{ci_low:.4f}, {ci_high:.4f}], ratio={ratio:.2f})"
        ),
    )


def summarize_metric_assessments(assessments: list[MetricAssessment]) -> tuple[str, str]:
    """Build compact markdown summaries for a graph-level comparison.

    Parameters
    ----------
    assessments : list[MetricAssessment]
        Per-metric assessments for one variant/graph pair.

    Returns
    -------
    tuple[str, str]
        ``(metrics_summary, overall_assessment)``.
    """
    metrics_summary = "; ".join(
        (
            f"{assessment.metric_name}: "
            f"{assessment.reimpl_summary} vs {assessment.original_summary} "
            f"({assessment.assessment})"
        )
        for assessment in assessments
    )
    is_exact = all(assessment.assessment == "exact" for assessment in assessments)
    has_mismatch = any("mismatch" in assessment.assessment for assessment in assessments)
    if is_exact:
        overall = "match"
    elif has_mismatch:
        overall = "mismatch"
    else:
        overall = "close"
    return metrics_summary, overall


def build_report_rows(
    results_path: Path,
    records: dict[str, BenchmarkRecord],
    graphs: dict[str, Any],
    graph_filter: set[str],
    bootstrap_samples: int,
) -> dict[str, list[GraphVariantReport]]:
    """Build grouped report rows for all benchmarked variants.

    Parameters
    ----------
    results_path : Path
        Results path used to resolve position files.
    records : dict[str, BenchmarkRecord]
        Parsed benchmark records.
    graphs : dict[str, Any]
        Graph registry keyed by graph name.
    graph_filter : set[str]
        Optional selected graph names.
    bootstrap_samples : int
        Bootstrap iterations for stochastic comparisons.

    Returns
    -------
    dict[str, list[GraphVariantReport]]
        Report rows grouped by algorithm family.
    """
    output_root = results_path.parent
    ok_records_by_engine_graph: dict[tuple[str, str], list[BenchmarkRecord]] = {}
    for record in records.values():
        if record.status != "ok" or record.positions_file is None:
            continue
        ok_records_by_engine_graph.setdefault(
            (record.engine_name, record.graph_name),
            [],
        ).append(record)

    graph_rows_by_family: dict[str, list[GraphVariantReport]] = {}
    metric_cache: dict[tuple[str, str], dict[str, float]] = {}
    for variant in VARIANT_REGISTRY:
        original_engine_name = original_variant_name(variant)
        if original_engine_name is None:
            continue

        family = algorithm_family(variant.variant_id)
        for graph_name, test_graph in graphs.items():
            if graph_filter and graph_name not in graph_filter:
                continue

            reimpl_records = ok_records_by_engine_graph.get(
                (variant.variant_id, graph_name),
                [],
            )
            original_records = ok_records_by_engine_graph.get(
                (original_engine_name, graph_name),
                [],
            )
            if not reimpl_records or not original_records:
                continue

            graph = test_graph.graph
            graph.compute_node_sizes()
            if graph.node_sizes is None:
                continue

            reimpl_metrics_by_metric: dict[str, list[float]] = {
                name: [] for name in METRICS_TO_COMPARE
            }
            original_metrics_by_metric: dict[str, list[float]] = {
                name: [] for name in METRICS_TO_COMPARE
            }

            for record_group, store in (
                (reimpl_records, reimpl_metrics_by_metric),
                (original_records, original_metrics_by_metric),
            ):
                for record in record_group:
                    if record.positions_file is None:
                        continue
                    cache_key = (graph_name, record.positions_file)
                    if cache_key not in metric_cache:
                        positions = load_position_tensor(output_root / record.positions_file)
                        if positions is None or positions.shape[0] != graph.num_nodes:
                            continue
                        metric_cache[cache_key] = compute_metrics(
                            positions=positions,
                            edge_index=graph.edge_index,
                            node_sizes=graph.node_sizes,
                        )
                    for metric_name, metric_value in metric_cache[cache_key].items():
                        store[metric_name].append(metric_value)

            assessments: list[MetricAssessment] = []
            stochastic_original = engine_is_stochastic(original_engine_name)
            for metric_name in METRICS_TO_COMPARE:
                reimpl_values = reimpl_metrics_by_metric[metric_name]
                original_values = original_metrics_by_metric[metric_name]
                if not reimpl_values or not original_values:
                    continue

                if (not variant.is_stochastic) and (not stochastic_original):
                    assessments.append(
                        deterministic_metric_assessment(
                            metric_name=metric_name,
                            reimpl_value=reimpl_values[0],
                            original_value=original_values[0],
                        )
                    )
                else:
                    assessments.append(
                        stochastic_metric_assessment(
                            metric_name=metric_name,
                            reimpl_values=reimpl_values,
                            original_values=original_values,
                            bootstrap_samples=bootstrap_samples,
                        )
                    )

            if not assessments:
                continue

            metrics_summary, overall_assessment = summarize_metric_assessments(assessments)
            sample_summary = f"reimpl={len(reimpl_records)}, original={len(original_records)}"
            graph_rows_by_family.setdefault(family, []).append(
                GraphVariantReport(
                    variant_id=variant.variant_id,
                    graph_name=graph_name,
                    sample_summary=sample_summary,
                    metrics_summary=metrics_summary,
                    match_assessment=overall_assessment,
                )
            )

    return graph_rows_by_family


def render_markdown(grouped_rows: dict[str, list[GraphVariantReport]]) -> str:
    """Render the grouped markdown comparison report.

    Parameters
    ----------
    grouped_rows : dict[str, list[GraphVariantReport]]
        Report rows grouped by algorithm family.

    Returns
    -------
    str
        Markdown report content.
    """
    lines = [
        "# Variant Comparison Report",
        "",
        (
            "Each row compares one reimplementation variant against its canonical "
            "original or proxy variant run."
        ),
        "",
    ]
    for family in sorted(grouped_rows):
        lines.extend(
            [
                f"## {family}",
                "",
                "| Variant | Graph | Samples | Metrics | Match Assessment |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for row in sorted(
            grouped_rows[family],
            key=lambda item: (item.variant_id, item.graph_name),
        ):
            lines.append(
                f"| {row.variant_id} | {row.graph_name} | {row.sample_summary} | "
                f"{row.metrics_summary} | {row.match_assessment} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    """Generate the variant comparison report.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args()
    results_path = Path(args.results)
    output_path = Path(args.output)
    records = load_results(results_path)
    graphs = load_graphs()
    graph_filter = select_graph_names(args.graphs)
    grouped_rows = build_report_rows(
        results_path=results_path,
        records=records,
        graphs=graphs,
        graph_filter=graph_filter,
        bootstrap_samples=args.bootstrap_samples,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown(grouped_rows), encoding="utf-8")
    print(f"[compare] wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
