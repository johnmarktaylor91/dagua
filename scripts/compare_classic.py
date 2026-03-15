"""Compare classic reimplementations against reference implementations.

Runs each algorithm pair on the standard test graph collection and generates
a markdown report showing metric-by-metric comparison.

Usage
-----
python scripts/compare_classic.py
python scripts/compare_classic.py --max-nodes 200
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from dagua.eval.competitors import CompetitorBase, get_available_competitors
from dagua.eval.graphs import TestGraph, get_test_graphs
from dagua.metrics import compute_all_metrics

PAIRS: list[tuple[str, list[str]]] = [
    ("classic_fr", ["nx_spring", "igraph_fr"]),
    ("classic_kk", ["nx_kamada_kawai"]),
    ("classic_sugiyama", ["graphviz_dot", "igraph_sugiyama"]),
    ("classic_fa2", []),
    ("classic_stress_sgd", []),
]

KEY_METRICS: list[str] = [
    "edge_crossings",
    "node_overlaps",
    "dag_fraction",
    "edge_length_std",
    "overall_quality",
]
DISPLAY_METRICS: list[str] = [*KEY_METRICS, "runtime_seconds"]
FAITHFUL_THRESHOLD = 0.8

SECTION_TITLES: dict[str, str] = {
    "classic_fr": "Fruchterman-Reingold: Our Implementation vs References",
    "classic_kk": "Kamada-Kawai: Our Implementation vs Reference",
    "classic_sugiyama": "Sugiyama: Our Implementation vs References",
    "classic_fa2": "ForceAtlas2 (no reference — standalone metrics)",
    "classic_stress_sgd": "Stress-SGD (no reference — standalone metrics)",
}


@dataclass
class EngineRun:
    """Execution outcome for one engine on one graph."""

    engine_name: str
    status: str
    runtime_seconds: Optional[float] = None
    metrics: Optional[dict[str, float]] = None
    error: Optional[str] = None
    note: Optional[str] = None


@dataclass
class SectionAssessment:
    """Aggregate fidelity summary for one paired section."""

    section_title: str
    engine_name: str
    within_tolerance: int = 0
    total_comparisons: int = 0
    worst_metric: Optional[str] = None
    worst_graph: Optional[str] = None
    worst_reference: Optional[str] = None
    worst_factor: float = 1.0
    worst_our_value: Optional[float] = None
    worst_reference_value: Optional[float] = None
    notes: list[str] = field(default_factory=list)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-nodes", type=int, default=500)
    parser.add_argument("--output", default="eval_output/classic_comparison.md")
    return parser.parse_args()


def _required_engine_names() -> list[str]:
    """Return the ordered set of engines needed for the report.

    Returns
    -------
    list[str]
        Unique engine names in report order.
    """
    names: list[str] = []
    for engine_name, references in PAIRS:
        if engine_name not in names:
            names.append(engine_name)
        for reference in references:
            if reference not in names:
                names.append(reference)
    return names


def _ensure_node_sizes(graph: TestGraph) -> None:
    """Populate node sizes before metric computation.

    Parameters
    ----------
    graph : TestGraph
        Graph wrapper from the evaluation catalog.

    Returns
    -------
    None
        Node sizes are computed in place on the embedded ``DaguaGraph``.
    """
    graph.graph.compute_node_sizes()
    if graph.graph.node_sizes is None:
        raise ValueError("graph.node_sizes is unavailable after compute_node_sizes()")


def _run_engine(
    engine_name: str,
    competitors: dict[str, CompetitorBase],
    graph: TestGraph,
) -> EngineRun:
    """Run one competitor and collect metrics or failure details.

    Parameters
    ----------
    engine_name : str
        Registered competitor name.
    competitors : dict[str, CompetitorBase]
        Available competitors keyed by name.
    graph : TestGraph
        Evaluation graph entry.

    Returns
    -------
    EngineRun
        Structured result with status, metrics, and failure notes.
    """
    competitor = competitors.get(engine_name)
    if competitor is None:
        return EngineRun(
            engine_name=engine_name,
            status="unavailable",
            note="engine not available in this environment",
        )

    if graph.graph.num_nodes > competitor.max_nodes:
        return EngineRun(
            engine_name=engine_name,
            status="skipped",
            note=f"graph has {graph.graph.num_nodes} nodes; limit is {competitor.max_nodes}",
        )

    try:
        result = competitor.layout(graph.graph)
    except Exception as exc:
        return EngineRun(
            engine_name=engine_name,
            status="error",
            error=str(exc),
            note="layout call raised before returning a CompetitorResult",
        )

    if result.pos is None:
        return EngineRun(
            engine_name=engine_name,
            status="error",
            runtime_seconds=result.runtime_seconds,
            error=result.error or "layout failed without coordinates",
        )

    if tuple(result.pos.shape) != (graph.graph.num_nodes, 2):
        return EngineRun(
            engine_name=engine_name,
            status="error",
            runtime_seconds=result.runtime_seconds,
            error=f"invalid position shape {tuple(result.pos.shape)}",
        )

    assert graph.graph.node_sizes is not None
    metrics = compute_all_metrics(
        result.pos,
        graph.graph.edge_index,
        graph.graph.node_sizes,
        direction=graph.graph.direction,
    )
    metrics["runtime_seconds"] = result.runtime_seconds
    return EngineRun(
        engine_name=engine_name,
        status="ok",
        runtime_seconds=result.runtime_seconds,
        metrics=metrics,
    )


def _collect_runs(
    graphs: Sequence[TestGraph],
    competitors: dict[str, CompetitorBase],
) -> dict[str, dict[str, EngineRun]]:
    """Run every required engine on every graph.

    Parameters
    ----------
    graphs : Sequence[TestGraph]
        Graph collection for the report.
    competitors : dict[str, CompetitorBase]
        Available competitors keyed by name.

    Returns
    -------
    dict[str, dict[str, EngineRun]]
        Nested mapping ``graph_name -> engine_name -> EngineRun``.
    """
    runs: dict[str, dict[str, EngineRun]] = {}
    engine_names = _required_engine_names()

    for test_graph in graphs:
        _ensure_node_sizes(test_graph)
        graph_runs: dict[str, EngineRun] = {}
        for engine_name in engine_names:
            graph_runs[engine_name] = _run_engine(engine_name, competitors, test_graph)
        runs[test_graph.name] = graph_runs

    return runs


def _format_numeric(metric_name: str, value: float) -> str:
    """Format a metric value for markdown output.

    Parameters
    ----------
    metric_name : str
        Metric identifier.
    value : float
        Numeric value to format.

    Returns
    -------
    str
        Human-readable value string.
    """
    if not math.isfinite(value):
        return "inf"
    if metric_name in {"edge_crossings", "node_overlaps"}:
        return str(int(round(value)))
    if metric_name == "runtime_seconds":
        return f"{value:.3f}"
    return f"{value:.4f}"


def _format_run_cell(run: EngineRun, metric_name: str) -> str:
    """Format one table cell for an engine/metric pair.

    Parameters
    ----------
    run : EngineRun
        Engine execution result.
    metric_name : str
        Metric to display.

    Returns
    -------
    str
        Markdown table cell contents.
    """
    if run.status != "ok" or run.metrics is None:
        if run.status == "unavailable":
            return "unavailable"
        if run.status == "skipped":
            return "skipped"
        return "error"
    return _format_numeric(metric_name, float(run.metrics[metric_name]))


def _run_issue_line(graph_name: str, run: EngineRun) -> Optional[str]:
    """Return a markdown note for a non-successful run.

    Parameters
    ----------
    graph_name : str
        Graph name in the report.
    run : EngineRun
        Engine execution result.

    Returns
    -------
    Optional[str]
        Bullet text for report notes, or ``None`` for successful runs.
    """
    if run.status == "ok":
        return None
    detail = run.error or run.note or run.status
    return f"`{graph_name}` / `{run.engine_name}`: {run.status} ({detail})"


def _relative_factor(left: float, right: float) -> float:
    """Compute a symmetric factor difference between two numeric values.

    Parameters
    ----------
    left : float
        First value.
    right : float
        Second value.

    Returns
    -------
    float
        Factor >= 1.0, or ``math.inf`` when one side is zero and the other is not
        or when the values have opposite signs.
    """
    if math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9):
        return 1.0

    if left == 0.0 or right == 0.0:
        return math.inf

    if math.copysign(1.0, left) != math.copysign(1.0, right):
        return math.inf

    abs_left = abs(left)
    abs_right = abs(right)
    smaller = min(abs_left, abs_right)
    larger = max(abs_left, abs_right)
    if smaller <= 0.0:
        return math.inf
    return larger / smaller


def _append_paired_assessment(
    assessment: SectionAssessment,
    graph_name: str,
    our_run: EngineRun,
    reference_name: str,
    reference_run: EngineRun,
) -> None:
    """Update a section summary with paired metric comparisons.

    Parameters
    ----------
    assessment : SectionAssessment
        Mutable section summary accumulator.
    graph_name : str
        Name of the current graph.
    our_run : EngineRun
        Our engine run.
    reference_name : str
        Reference engine name.
    reference_run : EngineRun
        Reference run.

    Returns
    -------
    None
        ``assessment`` is updated in place.
    """
    if our_run.status != "ok" or reference_run.status != "ok":
        return

    assert our_run.metrics is not None
    assert reference_run.metrics is not None

    for metric_name in DISPLAY_METRICS:
        our_value = float(our_run.metrics[metric_name])
        reference_value = float(reference_run.metrics[metric_name])
        factor = _relative_factor(our_value, reference_value)
        assessment.total_comparisons += 1
        if factor <= 2.0:
            assessment.within_tolerance += 1
        if factor > assessment.worst_factor:
            assessment.worst_factor = factor
            assessment.worst_metric = metric_name
            assessment.worst_graph = graph_name
            assessment.worst_reference = reference_name
            assessment.worst_our_value = our_value
            assessment.worst_reference_value = reference_value


def _render_paired_section(
    title: str,
    our_engine: str,
    references: Sequence[str],
    graphs: Sequence[TestGraph],
    runs: dict[str, dict[str, EngineRun]],
) -> tuple[str, SectionAssessment]:
    """Render a markdown section for one paired algorithm comparison.

    Parameters
    ----------
    title : str
        Section title.
    our_engine : str
        Our competitor engine name.
    references : Sequence[str]
        Reference competitor engine names.
    graphs : Sequence[TestGraph]
        Graphs included in the report.
    runs : dict[str, dict[str, EngineRun]]
        Cached engine runs keyed by graph and engine name.

    Returns
    -------
    tuple[str, SectionAssessment]
        Markdown body and aggregate section assessment.
    """
    engines = [our_engine, *references]
    assessment = SectionAssessment(section_title=title, engine_name=our_engine)
    lines = [f"## {title}", ""]
    lines.append("| Graph | Metric | " + " | ".join(engines) + " |")
    lines.append("|-------|--------|" + "|".join(["-----------"] * len(engines)) + "|")

    for test_graph in graphs:
        graph_runs = runs[test_graph.name]
        for metric_name in DISPLAY_METRICS:
            cells = [test_graph.name, metric_name]
            for engine_name in engines:
                cells.append(_format_run_cell(graph_runs[engine_name], metric_name))
            lines.append("| " + " | ".join(cells) + " |")

        our_run = graph_runs[our_engine]
        for reference_name in references:
            reference_run = graph_runs[reference_name]
            _append_paired_assessment(
                assessment=assessment,
                graph_name=test_graph.name,
                our_run=our_run,
                reference_name=reference_name,
                reference_run=reference_run,
            )

        for engine_name in engines:
            note = _run_issue_line(test_graph.name, graph_runs[engine_name])
            if note is not None and note not in assessment.notes:
                assessment.notes.append(note)

    lines.extend(["", "### Summary"])
    if assessment.total_comparisons == 0:
        lines.append("- No paired metric comparisons were possible.")
    else:
        lines.append(
            "- Metrics within 2x tolerance on "
            f"{assessment.within_tolerance}/{assessment.total_comparisons} paired metric checks."
        )
        if assessment.worst_metric is not None:
            worst_factor = (
                "inf"
                if not math.isfinite(assessment.worst_factor)
                else (f"{assessment.worst_factor:.2f}x")
            )
            assert assessment.worst_graph is not None
            assert assessment.worst_reference is not None
            assert assessment.worst_our_value is not None
            assert assessment.worst_reference_value is not None
            our_formatted = _format_numeric(
                assessment.worst_metric,
                assessment.worst_our_value,
            )
            reference_formatted = _format_numeric(
                assessment.worst_metric,
                assessment.worst_reference_value,
            )
            lines.append(
                "- Largest deviation: "
                f"{assessment.worst_metric} on {assessment.worst_graph} against "
                f"{assessment.worst_reference} ({our_formatted} vs "
                f"{reference_formatted}, factor={worst_factor})."
            )
    if assessment.notes:
        lines.append("- Notes:")
        for note in assessment.notes:
            lines.append(f"  - {note}")

    lines.append("")
    return "\n".join(lines), assessment


def _render_solo_section(
    title: str,
    engine_name: str,
    graphs: Sequence[TestGraph],
    runs: dict[str, dict[str, EngineRun]],
) -> str:
    """Render a markdown section for a solo algorithm without references.

    Parameters
    ----------
    title : str
        Section title.
    engine_name : str
        Engine to display.
    graphs : Sequence[TestGraph]
        Graphs included in the report.
    runs : dict[str, dict[str, EngineRun]]
        Cached engine runs keyed by graph and engine name.

    Returns
    -------
    str
        Markdown section body.
    """
    lines = [f"## {title}", ""]
    lines.append(f"| Graph | Metric | {engine_name} |")
    lines.append("|-------|--------|-----------|")

    successes = 0
    total_quality = 0.0
    total_runtime = 0.0
    notes: list[str] = []

    for test_graph in graphs:
        run = runs[test_graph.name][engine_name]
        for metric_name in DISPLAY_METRICS:
            lines.append(
                "| "
                + " | ".join([test_graph.name, metric_name, _format_run_cell(run, metric_name)])
                + " |"
            )
        if run.status == "ok" and run.metrics is not None:
            successes += 1
            total_quality += float(run.metrics["overall_quality"])
            total_runtime += float(run.metrics["runtime_seconds"])
        else:
            note = _run_issue_line(test_graph.name, run)
            if note is not None:
                notes.append(note)

    lines.extend(["", "### Summary"])
    lines.append(f"- Successful runs: {successes}/{len(graphs)}.")
    if successes > 0:
        lines.append(f"- Mean overall_quality: {total_quality / successes:.4f}.")
        lines.append(f"- Mean runtime_seconds: {total_runtime / successes:.4f}.")
    if notes:
        lines.append("- Notes:")
        for note in notes:
            lines.append(f"  - {note}")

    lines.append("")
    return "\n".join(lines)


def _overall_assessment(sections: Sequence[SectionAssessment]) -> str:
    """Render the report-level summary section.

    Parameters
    ----------
    sections : Sequence[SectionAssessment]
        Per-section paired summaries.

    Returns
    -------
    str
        Markdown section body.
    """
    total_within = sum(section.within_tolerance for section in sections)
    total_checks = sum(section.total_comparisons for section in sections)

    faithful: list[str] = []
    deviations: list[str] = []
    for section in sections:
        if section.total_comparisons == 0:
            deviations.append(f"{section.engine_name} (no paired comparisons)")
            continue

        ratio = section.within_tolerance / section.total_comparisons
        if ratio >= FAITHFUL_THRESHOLD:
            faithful.append(section.engine_name)
        else:
            if section.worst_metric is None or section.worst_graph is None:
                deviations.append(section.engine_name)
            else:
                worst_factor = (
                    "inf"
                    if not math.isfinite(section.worst_factor)
                    else (f"{section.worst_factor:.2f}x")
                )
                deviations.append(
                    f"{section.engine_name} "
                    f"({section.worst_metric} on {section.worst_graph}, {worst_factor})"
                )

    lines = ["## Overall Assessment", ""]
    lines.append(f"- {total_within}/{total_checks} paired metric checks were within 2x tolerance.")
    lines.append(
        "- Algorithms faithful to originals: "
        + (", ".join(faithful) if faithful else "none under the >=80% within-2x heuristic")
    )
    lines.append(
        "- Notable deviations: " + (", ".join(deviations) if deviations else "none observed")
    )
    lines.append("- Algorithms without bundled references: classic_fa2, classic_stress_sgd")
    lines.append("")
    return "\n".join(lines)


def _build_report(
    graphs: Sequence[TestGraph],
    runs: dict[str, dict[str, EngineRun]],
) -> str:
    """Assemble the full markdown report.

    Parameters
    ----------
    graphs : Sequence[TestGraph]
        Graphs included in the report.
    runs : dict[str, dict[str, EngineRun]]
        Cached engine runs keyed by graph and engine name.

    Returns
    -------
    str
        Complete markdown report text.
    """
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    parts = ["# Classic Algorithm Comparison Report", "", f"Generated: {timestamp}", ""]
    paired_assessments: list[SectionAssessment] = []

    for engine_name, references in PAIRS:
        title = SECTION_TITLES[engine_name]
        if references:
            section_body, assessment = _render_paired_section(
                title=title,
                our_engine=engine_name,
                references=references,
                graphs=graphs,
                runs=runs,
            )
            parts.append(section_body)
            paired_assessments.append(assessment)
        else:
            parts.append(
                _render_solo_section(
                    title=title,
                    engine_name=engine_name,
                    graphs=graphs,
                    runs=runs,
                )
            )

    parts.append(_overall_assessment(paired_assessments))
    return "\n".join(parts)


def main() -> None:
    """Run the classic-vs-reference comparison and write a markdown report.

    Returns
    -------
    None
        The report is written to disk and the output path is printed.
    """
    args = _parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    competitors = {competitor.name: competitor for competitor in get_available_competitors()}
    graphs = get_test_graphs(max_nodes=args.max_nodes)
    runs = _collect_runs(graphs=graphs, competitors=competitors)
    report = _build_report(graphs=graphs, runs=runs)

    output_path.write_text(report, encoding="utf-8")
    print(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
