#!/usr/bin/env python
"""Run all layout algorithms on all test graphs with parallel execution.

Usage
-----
python scripts/run_all_layouts.py
python scripts/run_all_layouts.py --workers 8 --timeout 180
python scripts/run_all_layouts.py --resume
python scripts/run_all_layouts.py --max-nodes 10 --workers 2 --timeout 30
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

OUTPUT_DIR = Path("eval_output/all_layouts")
RESULTS_FILE = OUTPUT_DIR / "results.json"
REPORT_FILE = OUTPUT_DIR / "comparison_report.md"
SKIPS_FILE = OUTPUT_DIR / "skipped.md"

DEFAULT_TIMEOUT = 120
DEFAULT_WORKERS = 4
PROGRESS_INTERVAL = 25
POLL_INTERVAL_SECONDS = 1.0
WORKER_TIMEOUT_GRACE_SECONDS = 5.0
QUALITY_METRIC = "overall_quality"

# Engines known to be extremely slow on certain graphs.
# These get a shorter timeout to avoid blocking the run.
SLOW_ENGINES: dict[str, int] = {
    "classic_davidson_harel": 60,
    "classic_stress_sgd": 90,
}

HEAD_TO_HEAD_PAIRS: list[tuple[str, list[str]]] = [
    ("classic_fr", ["nx_spring", "igraph_fr"]),
    ("classic_kk", ["nx_kamada_kawai"]),
    ("classic_sugiyama", ["graphviz_dot", "igraph_sugiyama"]),
]


class _WorkerLayoutTimeoutError(TimeoutError):
    """Raised when a worker exceeds its wall-clock budget."""


@dataclass(frozen=True)
class WorkItem:
    """One engine/graph task scheduled on the worker pool.

    Parameters
    ----------
    engine_name : str
        Registered competitor name.
    graph_name : str
        Test graph name.
    timeout_seconds : float
        Wall-clock timeout budget for the worker task.
    """

    engine_name: str
    graph_name: str
    timeout_seconds: float

    @property
    def key(self) -> str:
        """Return the stable results key for resume support.

        Returns
        -------
        str
            Flat ``graph:engine`` key used in ``results.json``.
        """
        return f"{self.graph_name}:{self.engine_name}"


@dataclass
class LayoutRunRecord:
    """Serializable outcome for one engine on one graph.

    Parameters
    ----------
    engine : str
        Registered competitor name.
    graph : str
        Test graph name.
    status : str
        Execution status: ``ok``, ``skipped``, ``timeout``, ``error``, or
        ``unavailable``.
    runtime_seconds : float | None, optional
        Measured runtime for successful or partially completed runs.
    metrics : dict[str, float | None]
        Serialized metrics payload for successful runs.
    error : str | None, optional
        Error message for failed runs.
    reason : str | None, optional
        Human-readable reason for skipped, timeout, or unavailable runs.
    timeout_seconds : float | None, optional
        Configured timeout budget when ``status == "timeout"``.
    """

    engine: str
    graph: str
    status: str
    runtime_seconds: Optional[float] = None
    metrics: dict[str, Optional[float]] = field(default_factory=dict)
    error: Optional[str] = None
    reason: Optional[str] = None
    timeout_seconds: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the record to a JSON-serializable payload.

        Returns
        -------
        dict[str, Any]
            Plain dictionary suitable for ``json.dumps``.
        """
        return {
            "engine": self.engine,
            "graph": self.graph,
            "status": self.status,
            "runtime_seconds": self.runtime_seconds,
            "metrics": self.metrics,
            "error": self.error,
            "reason": self.reason,
            "timeout_seconds": self.timeout_seconds,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LayoutRunRecord":
        """Build a record from a JSON-decoded payload.

        Parameters
        ----------
        payload : dict[str, Any]
            Serialized record payload.

        Returns
        -------
        LayoutRunRecord
            Parsed record with normalized optional fields.
        """
        raw_metrics = payload.get("metrics", {})
        metrics: dict[str, Optional[float]] = {}
        if isinstance(raw_metrics, dict):
            for key, value in raw_metrics.items():
                metrics[str(key)] = _optional_float(value)
        return cls(
            engine=str(payload.get("engine", "")),
            graph=str(payload.get("graph", "")),
            status=str(payload.get("status", "error")),
            runtime_seconds=_optional_float(payload.get("runtime_seconds")),
            metrics=metrics,
            error=_optional_str(payload.get("error")),
            reason=_optional_str(payload.get("reason")),
            timeout_seconds=_optional_float(payload.get("timeout_seconds")),
        )

    def detail(self) -> str:
        """Return the most useful explanation string for non-success statuses.

        Returns
        -------
        str
            Human-readable failure or skip reason.
        """
        if self.reason:
            return self.reason
        if self.error:
            return self.error
        if self.status == "timeout" and self.timeout_seconds is not None:
            return f"exceeded {self.timeout_seconds:.0f}s limit"
        return ""


def _optional_float(value: Any) -> Optional[float]:
    """Convert a JSON value into a finite float when possible.

    Parameters
    ----------
    value : Any
        Raw value decoded from JSON.

    Returns
    -------
    float | None
        Finite float value or ``None`` when conversion is not possible.
    """
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(converted):
        return None
    return converted


def _optional_str(value: Any) -> Optional[str]:
    """Convert a JSON value into a string when present.

    Parameters
    ----------
    value : Any
        Raw value decoded from JSON.

    Returns
    -------
    str | None
        String representation or ``None`` when the value is absent.
    """
    if value is None:
        return None
    return str(value)


def _sanitize_metrics(metrics: dict[str, Any]) -> dict[str, Optional[float]]:
    """Normalize metric values for JSON persistence.

    Parameters
    ----------
    metrics : dict[str, Any]
        Metrics returned by ``compute_all_metrics``.

    Returns
    -------
    dict[str, float | None]
        Same keys with non-finite values replaced by ``None``.
    """
    serialized: dict[str, Optional[float]] = {}
    for key, value in metrics.items():
        serialized[key] = _optional_float(value)
    return serialized


def _timeout_signal_handler(signum: int, frame: Any) -> None:
    """Raise a worker timeout when the alarm signal fires.

    Parameters
    ----------
    signum : int
        Signal number delivered by the kernel.
    frame : Any
        Current execution frame supplied by ``signal``.

    Returns
    -------
    None
        This function always raises ``_WorkerLayoutTimeoutError``.
    """
    del signum, frame
    raise _WorkerLayoutTimeoutError("worker layout timeout exceeded")


def _enable_worker_timeout(timeout_seconds: float) -> Optional[Any]:
    """Install a wall-clock timeout for the current worker process.

    Parameters
    ----------
    timeout_seconds : float
        Maximum allowed wall-clock time.

    Returns
    -------
    object | None
        Previous signal handler when ``SIGALRM`` is available, otherwise
        ``None``.
    """
    if timeout_seconds <= 0 or not hasattr(signal, "SIGALRM"):
        return None
    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_signal_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    return previous_handler


def _disable_worker_timeout(previous_handler: Optional[Any]) -> None:
    """Clear the worker alarm and restore the previous signal handler.

    Parameters
    ----------
    previous_handler : object | None
        Handler returned by ``_enable_worker_timeout``.

    Returns
    -------
    None
        The current process signal configuration is updated in place.
    """
    if not hasattr(signal, "SIGALRM"):
        return
    signal.setitimer(signal.ITIMER_REAL, 0.0)
    if previous_handler is not None:
        signal.signal(signal.SIGALRM, previous_handler)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of worker processes to use (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Default per-layout timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip graph/engine pairs already present in the results file",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=0,
        help="Only run graphs with at most this many nodes (0 = all graphs)",
    )
    return parser.parse_args()


_WORKER_COMPETITORS: dict[str, Any] | None = None
_WORKER_GRAPHS: dict[str, Any] | None = None


def _ensure_worker_cache() -> tuple[dict[str, Any], dict[str, Any]]:
    """Lazy-init the per-worker competitor and graph caches.

    Returns
    -------
    tuple[dict, dict]
        Competitors dict and graphs dict, cached after first call.
    """
    global _WORKER_COMPETITORS, _WORKER_GRAPHS  # noqa: PLW0603
    if _WORKER_COMPETITORS is None:
        from dagua.eval.competitors import get_available_competitors

        _WORKER_COMPETITORS = {c.name: c for c in get_available_competitors()}
    if _WORKER_GRAPHS is None:
        from dagua.eval.graphs import get_test_graphs

        _WORKER_GRAPHS = {tg.name: tg for tg in get_test_graphs()}
    return _WORKER_COMPETITORS, _WORKER_GRAPHS


def _run_single_layout(engine_name: str, graph_name: str) -> dict[str, Any]:
    """Run one engine on one graph inside a worker process.

    Parameters
    ----------
    engine_name : str
        Registered competitor name.
    graph_name : str
        Test graph name.

    Returns
    -------
    dict[str, Any]
        Serializable result payload.
    """
    from dagua.metrics import compute_all_metrics

    competitors, graphs = _ensure_worker_cache()

    competitor = competitors.get(engine_name)
    if competitor is None:
        return LayoutRunRecord(
            engine=engine_name,
            graph=graph_name,
            status="unavailable",
            reason="engine not available in this environment",
        ).to_dict()

    graph = graphs.get(graph_name)
    if graph is None:
        return LayoutRunRecord(
            engine=engine_name,
            graph=graph_name,
            status="error",
            error="graph not found",
        ).to_dict()

    graph.graph.compute_node_sizes()
    if graph.graph.node_sizes is None:
        return LayoutRunRecord(
            engine=engine_name,
            graph=graph_name,
            status="error",
            error="graph.node_sizes is unavailable after compute_node_sizes()",
        ).to_dict()

    if graph.graph.num_nodes > competitor.max_nodes:
        return LayoutRunRecord(
            engine=engine_name,
            graph=graph_name,
            status="skipped",
            reason=f"nodes={graph.graph.num_nodes} > max={competitor.max_nodes}",
        ).to_dict()

    try:
        result = competitor.layout(graph.graph)
    except Exception as exc:
        return LayoutRunRecord(
            engine=engine_name,
            graph=graph_name,
            status="error",
            error=str(exc),
        ).to_dict()

    if result.pos is None:
        return LayoutRunRecord(
            engine=engine_name,
            graph=graph_name,
            status="error",
            runtime_seconds=result.runtime_seconds,
            error=result.error or "no positions returned",
        ).to_dict()

    if tuple(result.pos.shape) != (graph.graph.num_nodes, 2):
        return LayoutRunRecord(
            engine=engine_name,
            graph=graph_name,
            status="error",
            runtime_seconds=result.runtime_seconds,
            error=f"invalid position shape {tuple(result.pos.shape)}",
        ).to_dict()

    metrics = compute_all_metrics(
        result.pos,
        graph.graph.edge_index,
        graph.graph.node_sizes,
        direction=graph.graph.direction,
    )
    metrics["runtime_seconds"] = result.runtime_seconds
    return LayoutRunRecord(
        engine=engine_name,
        graph=graph_name,
        status="ok",
        runtime_seconds=result.runtime_seconds,
        metrics=_sanitize_metrics(metrics),
    ).to_dict()


def _run_single_layout_with_timeout(
    engine_name: str, graph_name: str, timeout_seconds: float
) -> dict[str, Any]:
    """Run one layout task with a worker-side wall-clock timeout.

    Parameters
    ----------
    engine_name : str
        Registered competitor name.
    graph_name : str
        Test graph name.
    timeout_seconds : float
        Maximum allowed wall-clock time.

    Returns
    -------
    dict[str, Any]
        Serializable result payload.
    """
    # Warm the cache BEFORE starting the timeout clock — import overhead
    # (loading 60 TorchLens models) can take 60-90s on first call.
    _ensure_worker_cache()
    start_time = time.perf_counter()
    previous_handler = _enable_worker_timeout(timeout_seconds)
    try:
        return _run_single_layout(engine_name, graph_name)
    except _WorkerLayoutTimeoutError:
        elapsed = time.perf_counter() - start_time
        return LayoutRunRecord(
            engine=engine_name,
            graph=graph_name,
            status="timeout",
            runtime_seconds=elapsed,
            reason=f"exceeded {timeout_seconds:.0f}s limit",
            timeout_seconds=timeout_seconds,
        ).to_dict()
    finally:
        _disable_worker_timeout(previous_handler)


def _load_results(path: Path) -> dict[str, LayoutRunRecord]:
    """Load existing run results from disk.

    Parameters
    ----------
    path : Path
        JSON file containing flat ``graph:engine`` result records.

    Returns
    -------
    dict[str, LayoutRunRecord]
        Parsed results keyed by ``graph:engine``.
    """
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    results: dict[str, LayoutRunRecord] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            results[str(key)] = LayoutRunRecord.from_dict(value)
    return results


def _save_results(path: Path, results: dict[str, LayoutRunRecord]) -> None:
    """Persist all results to disk.

    Parameters
    ----------
    path : Path
        Destination JSON file.
    results : dict[str, LayoutRunRecord]
        Flat results mapping.

    Returns
    -------
    None
        The JSON file is written atomically via replacement.
    """
    serialized = {key: record.to_dict() for key, record in sorted(results.items())}
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(serialized, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(path)


def _scoped_results(
    results: dict[str, LayoutRunRecord],
    engines: Sequence[str],
    graph_names: Sequence[str],
) -> dict[str, LayoutRunRecord]:
    """Filter results down to the current engine and graph scope.

    Parameters
    ----------
    results : dict[str, LayoutRunRecord]
        All persisted results.
    engines : Sequence[str]
        Engines selected for this run.
    graph_names : Sequence[str]
        Graphs selected for this run.

    Returns
    -------
    dict[str, LayoutRunRecord]
        Results limited to the requested scope.
    """
    engine_set = set(engines)
    graph_set = set(graph_names)
    scoped: dict[str, LayoutRunRecord] = {}
    for key, record in results.items():
        if record.engine in engine_set and record.graph in graph_set:
            scoped[key] = record
    return scoped


def _collect_issue_records(results: dict[str, LayoutRunRecord]) -> list[LayoutRunRecord]:
    """Return all non-success results in stable display order.

    Parameters
    ----------
    results : dict[str, LayoutRunRecord]
        Flat results mapping.

    Returns
    -------
    list[LayoutRunRecord]
        All records whose status is not ``ok``.
    """
    issues = [record for record in results.values() if record.status != "ok"]
    issues.sort(key=lambda record: (record.engine, record.graph, record.status))
    return issues


def _write_skips_report(records: Sequence[LayoutRunRecord]) -> None:
    """Write the markdown table for skipped, timed out, or failed runs.

    Parameters
    ----------
    records : Sequence[LayoutRunRecord]
        Non-success records to include in the report.

    Returns
    -------
    None
        ``SKIPS_FILE`` is written in markdown format.
    """
    lines = [
        "# Skipped / Timed Out / Failed Layouts",
        "",
        "| Engine | Graph | Status | Reason |",
        "|--------|-------|--------|--------|",
    ]
    for record in records:
        reason = record.detail().replace("\n", " ").replace("|", "\\|")
        lines.append(f"| {record.engine} | {record.graph} | {record.status} | {reason or '-'} |")
    SKIPS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mean(values: Sequence[float]) -> Optional[float]:
    """Compute the arithmetic mean for a sequence of floats.

    Parameters
    ----------
    values : Sequence[float]
        Numeric values to average.

    Returns
    -------
    float | None
        Mean of the sequence or ``None`` when empty.
    """
    if not values:
        return None
    return sum(values) / len(values)


def _format_float(value: Optional[float], digits: int = 2) -> str:
    """Format an optional float for markdown output.

    Parameters
    ----------
    value : float | None
        Value to format.
    digits : int, default=2
        Decimal precision.

    Returns
    -------
    str
        Rendered numeric string or ``-`` when missing.
    """
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _engine_records(
    results: dict[str, LayoutRunRecord],
    engine_name: str,
    graph_names: Sequence[str],
) -> list[LayoutRunRecord]:
    """Collect scoped records for one engine.

    Parameters
    ----------
    results : dict[str, LayoutRunRecord]
        Scoped result mapping.
    engine_name : str
        Engine to extract.
    graph_names : Sequence[str]
        Graph ordering to preserve.

    Returns
    -------
    list[LayoutRunRecord]
        Records for the requested engine.
    """
    records: list[LayoutRunRecord] = []
    for graph_name in graph_names:
        key = f"{graph_name}:{engine_name}"
        record = results.get(key)
        if record is not None:
            records.append(record)
    return records


def _graph_records(
    results: dict[str, LayoutRunRecord],
    graph_name: str,
    engines: Sequence[str],
) -> list[LayoutRunRecord]:
    """Collect scoped records for one graph.

    Parameters
    ----------
    results : dict[str, LayoutRunRecord]
        Scoped result mapping.
    graph_name : str
        Graph to extract.
    engines : Sequence[str]
        Engine ordering to preserve.

    Returns
    -------
    list[LayoutRunRecord]
        Records for the requested graph.
    """
    records: list[LayoutRunRecord] = []
    for engine_name in engines:
        key = f"{graph_name}:{engine_name}"
        record = results.get(key)
        if record is not None:
            records.append(record)
    return records


def _quality_value(record: LayoutRunRecord) -> Optional[float]:
    """Extract the report quality metric from a successful record.

    Parameters
    ----------
    record : LayoutRunRecord
        Result record to inspect.

    Returns
    -------
    float | None
        Quality score when available.
    """
    return record.metrics.get(QUALITY_METRIC)


def _build_engine_summary_table(
    results: dict[str, LayoutRunRecord],
    engines: Sequence[str],
    graph_names: Sequence[str],
) -> list[str]:
    """Build the per-engine markdown summary table.

    Parameters
    ----------
    results : dict[str, LayoutRunRecord]
        Scoped result mapping.
    engines : Sequence[str]
        Engine ordering.
    graph_names : Sequence[str]
        Graph ordering.

    Returns
    -------
    list[str]
        Markdown lines for the engine summary section.
    """
    lines = [
        "## Per-Engine Summary",
        "",
        (
            "| Engine | Success | Skipped | Timeouts | Errors | Unavailable | "
            "Mean Runtime (s) | Mean Quality |"
        ),
        (
            "|--------|---------|---------|----------|--------|-------------|"
            "------------------|--------------|"
        ),
    ]
    for engine_name in engines:
        records = _engine_records(results, engine_name, graph_names)
        ok_records = [record for record in records if record.status == "ok"]
        runtimes = [
            record.runtime_seconds for record in ok_records if record.runtime_seconds is not None
        ]
        qualities = [
            quality
            for quality in (_quality_value(record) for record in ok_records)
            if quality is not None
        ]
        skipped = sum(record.status == "skipped" for record in records)
        timed_out = sum(record.status == "timeout" for record in records)
        errors = sum(record.status == "error" for record in records)
        unavailable = sum(record.status == "unavailable" for record in records)
        lines.append(
            "| "
            f"{engine_name} | {len(ok_records)} | {skipped} | {timed_out} | "
            f"{errors} | {unavailable} | "
            f"{_format_float(_mean(runtimes), digits=3)} | "
            f"{_format_float(_mean(qualities), digits=2)} |"
        )
    lines.append("")
    return lines


def _build_graph_summary_table(
    results: dict[str, LayoutRunRecord],
    engines: Sequence[str],
    graphs: Sequence[Any],
) -> list[str]:
    """Build the per-graph markdown summary table.

    Parameters
    ----------
    results : dict[str, LayoutRunRecord]
        Scoped result mapping.
    engines : Sequence[str]
        Engine ordering.
    graphs : Sequence[Any]
        Test graph objects with ``name`` and ``graph.num_nodes`` attributes.

    Returns
    -------
    list[str]
        Markdown lines for the graph summary section.
    """
    lines = [
        "## Per-Graph Summary",
        "",
        "| Graph | Nodes | Successes | Successful Engines | Best Quality | Worst Quality |",
        "|-------|-------|-----------|--------------------|--------------|---------------|",
    ]
    for test_graph in graphs:
        records = _graph_records(results, test_graph.name, engines)
        ok_records = [record for record in records if record.status == "ok"]
        quality_records = [
            (record.engine, quality)
            for record in ok_records
            for quality in [_quality_value(record)]
            if quality is not None
        ]
        quality_records.sort(key=lambda item: item[1], reverse=True)
        succeeded_names = ", ".join(record.engine for record in ok_records) or "-"
        best_text = "-"
        worst_text = "-"
        if quality_records:
            best_engine, best_quality = quality_records[0]
            worst_engine, worst_quality = quality_records[-1]
            best_text = f"{best_engine} ({best_quality:.2f})"
            worst_text = f"{worst_engine} ({worst_quality:.2f})"
        lines.append(
            "| "
            f"{test_graph.name} | {test_graph.graph.num_nodes} | {len(ok_records)} | "
            f"{succeeded_names} | {best_text} | {worst_text} |"
        )
    lines.append("")
    return lines


def _build_head_to_head_section(
    results: dict[str, LayoutRunRecord],
    graphs: Sequence[Any],
) -> list[str]:
    """Build the reimplementation-versus-reference markdown section.

    Parameters
    ----------
    results : dict[str, LayoutRunRecord]
        Scoped result mapping.
    graphs : Sequence[Any]
        Test graph objects in display order.

    Returns
    -------
    list[str]
        Markdown lines for the head-to-head section.
    """
    graph_names = [test_graph.name for test_graph in graphs]
    lines = ["## Head-to-Head", ""]
    for reimpl_name, references in HEAD_TO_HEAD_PAIRS:
        lines.append(f"### {reimpl_name}")
        lines.append("")
        lines.append(
            "| Reference | Comparable Graphs | Reimpl Wins | Reference Wins | "
            "Mean Quality Delta | Mean Runtime Delta (s) |"
        )
        lines.append(
            "|-----------|-------------------|-------------|----------------|"
            "--------------------|------------------------|"
        )
        for reference_name in references:
            quality_deltas: list[float] = []
            runtime_deltas: list[float] = []
            reimpl_wins = 0
            reference_wins = 0
            comparable = 0
            for graph_name in graph_names:
                reimpl_record = results.get(f"{graph_name}:{reimpl_name}")
                reference_record = results.get(f"{graph_name}:{reference_name}")
                if reimpl_record is None or reference_record is None:
                    continue
                if reimpl_record.status != "ok" or reference_record.status != "ok":
                    continue
                reimpl_quality = _quality_value(reimpl_record)
                reference_quality = _quality_value(reference_record)
                if reimpl_quality is None or reference_quality is None:
                    continue
                comparable += 1
                quality_delta = reimpl_quality - reference_quality
                quality_deltas.append(quality_delta)
                if quality_delta >= 0:
                    reimpl_wins += 1
                else:
                    reference_wins += 1
                if (
                    reimpl_record.runtime_seconds is not None
                    and reference_record.runtime_seconds is not None
                ):
                    runtime_deltas.append(
                        reimpl_record.runtime_seconds - reference_record.runtime_seconds
                    )
            lines.append(
                "| "
                f"{reference_name} | {comparable} | {reimpl_wins} | {reference_wins} | "
                f"{_format_float(_mean(quality_deltas), digits=2)} | "
                f"{_format_float(_mean(runtime_deltas), digits=3)} |"
            )
        lines.append("")
    return lines


def _write_comparison_report(
    results: dict[str, LayoutRunRecord],
    engines: Sequence[str],
    graphs: Sequence[Any],
) -> None:
    """Write the markdown comparison report.

    Parameters
    ----------
    results : dict[str, LayoutRunRecord]
        Scoped result mapping.
    engines : Sequence[str]
        Engine ordering.
    graphs : Sequence[Any]
        Test graph objects in display order.

    Returns
    -------
    None
        ``REPORT_FILE`` is written in markdown format.
    """
    graph_names = [test_graph.name for test_graph in graphs]
    ok_count = sum(record.status == "ok" for record in results.values())
    timeout_count = sum(record.status == "timeout" for record in results.values())
    error_count = sum(record.status == "error" for record in results.values())
    skipped_count = sum(record.status == "skipped" for record in results.values())
    unavailable_count = sum(record.status == "unavailable" for record in results.values())

    lines = [
        "# All Layout Comparison Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"- Graphs: {len(graphs)}",
        f"- Engines: {len(engines)}",
        f"- Total scoped results: {len(results)}",
        f"- Successes: {ok_count}",
        f"- Timeouts: {timeout_count}",
        f"- Errors: {error_count}",
        f"- Skipped: {skipped_count}",
        f"- Unavailable: {unavailable_count}",
        "",
    ]
    lines.extend(_build_engine_summary_table(results, engines, graph_names))
    lines.extend(_build_graph_summary_table(results, engines, graphs))
    lines.extend(_build_head_to_head_section(results, graphs))
    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _graph_completion_targets(graphs: Sequence[Any], engines: Sequence[str]) -> dict[str, int]:
    """Return the required completion count for each graph.

    Parameters
    ----------
    graphs : Sequence[Any]
        Test graph objects in scope.
    engines : Sequence[str]
        Engines in scope.

    Returns
    -------
    dict[str, int]
        Mapping from graph name to expected result count.
    """
    return {test_graph.name: len(engines) for test_graph in graphs}


def _progress_counts(results: dict[str, LayoutRunRecord]) -> tuple[int, int, int, int, int]:
    """Compute status counts for a result mapping.

    Parameters
    ----------
    results : dict[str, LayoutRunRecord]
        Results to summarize.

    Returns
    -------
    tuple[int, int, int, int, int]
        Counts for successes, timeouts, errors, skipped, and unavailable.
    """
    succeeded = sum(record.status == "ok" for record in results.values())
    timed_out = sum(record.status == "timeout" for record in results.values())
    errors = sum(record.status == "error" for record in results.values())
    skipped = sum(record.status == "skipped" for record in results.values())
    unavailable = sum(record.status == "unavailable" for record in results.values())
    return succeeded, timed_out, errors, skipped, unavailable


def main() -> int:
    """Run the full all-engines evaluation pipeline.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    existing_results = _load_results(RESULTS_FILE) if args.resume else {}

    from dagua.eval.competitors import get_available_competitors
    from dagua.eval.graphs import get_test_graphs

    max_nodes = args.max_nodes if args.max_nodes > 0 else None
    engines = [competitor.name for competitor in get_available_competitors()]
    graphs = get_test_graphs(max_nodes=max_nodes)
    graph_names = [test_graph.name for test_graph in graphs]
    graph_targets = _graph_completion_targets(graphs, engines)

    results = dict(existing_results)
    scoped_existing = _scoped_results(existing_results, engines, graph_names)
    graph_counts = {graph_name: 0 for graph_name in graph_names}
    for record in scoped_existing.values():
        graph_counts[record.graph] = graph_counts.get(record.graph, 0) + 1
    completed_graphs = {
        graph_name
        for graph_name, expected_count in graph_targets.items()
        if graph_counts.get(graph_name, 0) >= expected_count
    }

    work_items: list[WorkItem] = []
    for test_graph in graphs:
        for engine_name in engines:
            key = f"{test_graph.name}:{engine_name}"
            if key in existing_results:
                continue
            timeout_seconds = float(SLOW_ENGINES.get(engine_name, args.timeout))
            work_items.append(
                WorkItem(
                    engine_name=engine_name,
                    graph_name=test_graph.name,
                    timeout_seconds=timeout_seconds,
                )
            )

    total_scope = len(engines) * len(graphs)
    print(
        f"Running {len(work_items)} new layouts ({len(engines)} engines × {len(graphs)} graphs; "
        f"{total_scope} scoped total)"
    )
    print(f"Workers: {args.workers}, Default timeout: {args.timeout}s")

    if not work_items:
        scoped_results = _scoped_results(results, engines, graph_names)
        _save_results(RESULTS_FILE, results)
        _write_skips_report(_collect_issue_records(scoped_results))
        _write_comparison_report(scoped_results, engines, graphs)
        succeeded, timed_out, errors, skipped, unavailable = _progress_counts(scoped_results)
        print(
            "Done. "
            f"{succeeded} succeeded, {timed_out} timed out, {errors} errors, {skipped} skipped"
            + (f", {unavailable} unavailable" if unavailable else "")
        )
        print(f"Results: {RESULTS_FILE}")
        print(f"Report: {REPORT_FILE}")
        print(f"Skipped: {SKIPS_FILE}")
        return 0

    try:
        executor = ProcessPoolExecutor(max_workers=args.workers)
    except PermissionError:
        print(
            "Failed to start ProcessPoolExecutor: this environment denies the "
            "semaphore primitives required for multiprocessing."
        )
        print("Run the script in a less restricted environment to enable parallel execution.")
        return 1
    futures: dict[Future[dict[str, Any]], WorkItem] = {}
    deadlines: dict[Future[dict[str, Any]], float] = {}
    abandoned_futures: set[Future[dict[str, Any]]] = set()
    completed_tasks = 0

    try:
        # Submit in batches to avoid deadline expiry while queued.
        # Only keep at most 2× workers in-flight at a time.
        max_inflight = args.workers * 2
        work_iter = iter(work_items)

        def _submit_next() -> bool:
            """Submit the next work item if the pool has capacity."""
            if len(futures) >= max_inflight:
                return False
            item = next(work_iter, None)
            if item is None:
                return False
            future = executor.submit(
                _run_single_layout_with_timeout,
                item.engine_name,
                item.graph_name,
                item.timeout_seconds,
            )
            futures[future] = item
            # Deadline starts from NOW (submission ≈ near-immediate pickup
            # because we cap in-flight count).
            deadlines[future] = (
                time.monotonic() + item.timeout_seconds + WORKER_TIMEOUT_GRACE_SECONDS
            )
            return True

        # Seed the pool
        while _submit_next():
            pass

        while futures or completed_tasks < len(work_items):
            done, _ = wait(
                tuple(futures.keys()),
                timeout=POLL_INTERVAL_SECONDS,
                return_when=FIRST_COMPLETED,
            )

            now = time.monotonic()
            timed_out_futures = [
                future
                for future, deadline in deadlines.items()
                if future in futures and not future.done() and now >= deadline
            ]
            for future in timed_out_futures:
                item = futures.pop(future)
                deadlines.pop(future, None)
                abandoned_futures.add(future)
                results[item.key] = LayoutRunRecord(
                    engine=item.engine_name,
                    graph=item.graph_name,
                    status="timeout",
                    reason=f"exceeded {item.timeout_seconds:.0f}s limit",
                    timeout_seconds=item.timeout_seconds,
                )
                future.cancel()
                completed_tasks += 1
                graph_counts[item.graph_name] = graph_counts.get(item.graph_name, 0) + 1

            for future in done:
                item = futures.pop(future, None)
                deadlines.pop(future, None)
                if item is None:
                    continue
                try:
                    payload = future.result()
                    results[item.key] = LayoutRunRecord.from_dict(payload)
                except Exception as exc:
                    results[item.key] = LayoutRunRecord(
                        engine=item.engine_name,
                        graph=item.graph_name,
                        status="error",
                        error=str(exc),
                    )
                completed_tasks += 1
                graph_counts[item.graph_name] = graph_counts.get(item.graph_name, 0) + 1

            newly_completed_graphs = [
                graph_name
                for graph_name, expected_count in graph_targets.items()
                if graph_name not in completed_graphs
                and graph_counts.get(graph_name, 0) >= expected_count
            ]
            for graph_name in sorted(newly_completed_graphs):
                completed_graphs.add(graph_name)
                _save_results(RESULTS_FILE, results)
                scoped_results = _scoped_results(results, engines, graph_names)
                _write_skips_report(_collect_issue_records(scoped_results))
                print(f"  graph complete: {graph_name}")

            # Refill the pool after completions
            while _submit_next():
                pass

            if completed_tasks > 0 and completed_tasks % PROGRESS_INTERVAL == 0:
                print(f"  [{completed_tasks}/{len(work_items)}] completed")
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    scoped_results = _scoped_results(results, engines, graph_names)
    _save_results(RESULTS_FILE, results)
    _write_skips_report(_collect_issue_records(scoped_results))
    _write_comparison_report(scoped_results, engines, graphs)

    succeeded, timed_out, errors, skipped, unavailable = _progress_counts(scoped_results)
    print(
        "Done. "
        f"{succeeded} succeeded, {timed_out} timed out, {errors} errors, {skipped} skipped"
        + (f", {unavailable} unavailable" if unavailable else "")
    )
    print(f"Results: {RESULTS_FILE}")
    print(f"Report: {REPORT_FILE}")
    print(f"Skipped: {SKIPS_FILE}")
    if abandoned_futures:
        print(
            "Note: "
            f"{len(abandoned_futures)} executor futures were abandoned "
            "after timeout tracking."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
