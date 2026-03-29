#!/usr/bin/env python3
"""Run the unified full benchmark across all selected graphs and engines.

This script supersedes the older one-off benchmark helpers by combining:

- all-graph/all-engine execution
- multi-seed runs for stochastic engines
- saved position tensors and runtime metadata
- resume support through a flat ``results.json`` file
- human-readable progress logging and a generated ``summary.md``

Usage
-----
python scripts/run_benchmark.py
python scripts/run_benchmark.py --workers 8 --timeout 180
python scripts/run_benchmark.py --max-nodes 10 --workers 2 --timeout 30 --seeds 2
python scripts/run_benchmark.py --resume
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import random
import re
import signal
import statistics
import sys
import tempfile
import time
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

from dagua.eval.variants import (
    base_pairings,
    engine_is_heavy,
    engine_is_stochastic,
    engine_timeout_cap,
    get_variant,
    get_variant_for_original_name,
    original_variant_engine_names,
    original_variant_name,
    variant_pairings,
    variants_for_base_engine,
)

# Use forkserver to avoid deadlocks when torch is imported in the main process.
# fork + torch's internal threading locks = hang on large job counts.
_MP_CONTEXT = multiprocessing.get_context("forkserver")

DEFAULT_OUTPUT_DIR = Path("eval_output/benchmark_full")
DEFAULT_TIMEOUT = 120
DEFAULT_WORKERS = 4
DEFAULT_SEED_COUNT = 10
DEFAULT_SEED_START = 42
PROGRESS_INTERVAL = 25
SAVE_INTERVAL = 100  # flush results to disk every N completions
MAX_INFLIGHT_GROUPS = 200  # rolling window of concurrent futures
WATCHDOG_TIMEOUT = 300.0  # seconds to wait for any future before assuming dead worker
RUNNING_STATUS = "running"
POSITION_DIRNAME = "positions"
CONSECUTIVE_FAILURE_SKIP_THRESHOLD = 3  # skip remaining seeds after N consecutive failures
# Minimum timeout (seconds) for tiny graphs -- no layout should need more
# than this on a graph with <100 nodes.
MIN_TIMEOUT_SECONDS = 30.0
# Graph size at which the full global timeout applies.
FULL_TIMEOUT_NODE_COUNT = 500


class _WorkerLayoutTimeoutError(TimeoutError):
    """Raised when a worker exceeds its wall-clock budget."""


@dataclass(frozen=True)
class GraphSummary:
    """Minimal graph metadata used by the benchmark planner.

    Parameters
    ----------
    name : str
        Stable test-graph name.
    num_nodes : int
        Number of graph nodes.
    num_edges : int
        Number of graph edges.
    tags : list[str]
        Sorted tag list from the graph registry.
    """

    name: str
    num_nodes: int
    num_edges: int
    tags: list[str]


@dataclass(frozen=True)
class WorkItem:
    """One scheduled engine/graph/seed execution.

    Parameters
    ----------
    graph_name : str
        Stable test-graph name.
    engine_name : str
        Registered competitor adapter name.
    seed : int | None
        Explicit random seed for stochastic engines, or ``None`` for
        deterministic runs.
    timeout_seconds : float
        Wall-clock timeout budget for the worker.
    output_dir : str
        Root benchmark output directory.
    save_positions : bool
        Whether worker results should persist ``[N, 2]`` tensors.
    """

    graph_name: str
    engine_name: str
    seed: Optional[int]
    timeout_seconds: float
    output_dir: str
    save_positions: bool

    @property
    def key(self) -> str:
        """Return the stable record key used in ``results.json``.

        Returns
        -------
        str
            Flat key that uniquely identifies one graph/engine/seed tuple.
        """
        return build_record_key(self.graph_name, self.engine_name, self.seed)


@dataclass
class BenchmarkRecord:
    """Serializable outcome for one graph/engine/seed attempt.

    Parameters
    ----------
    graph_name : str
        Stable test-graph name.
    engine_name : str
        Registered competitor adapter name.
    seed : int | None
        Explicit seed for stochastic engines, or ``None`` for deterministic
        runs.
    status : str
        Execution state: ``running``, ``ok``, ``skipped``, ``error``, or
        ``timeout``.
    runtime_seconds : float | None
        Measured runtime when available.
    error : str | None
        Error message for failed runs.
    positions_file : str | None
        Relative path to the saved position tensor.
    num_nodes : int
        Number of graph nodes.
    num_edges : int
        Number of graph edges.
    is_stochastic : bool
        Whether the engine is treated as stochastic by the benchmark harness.
    skip_reason : str | None
        Human-readable reason for skipped runs.
    original_for : list[str]
        Reimplementation engines for which this engine is a reference.
    reimpl_of : list[str]
        Original engines that this engine reimplements.
    """

    graph_name: str
    engine_name: str
    seed: Optional[int]
    status: str
    runtime_seconds: Optional[float]
    error: Optional[str]
    positions_file: Optional[str]
    num_nodes: int
    num_edges: int
    is_stochastic: bool
    skip_reason: Optional[str]
    original_for: list[str]
    reimpl_of: list[str]

    @property
    def key(self) -> str:
        """Return the stable record key used in ``results.json``.

        Returns
        -------
        str
            Flat key that uniquely identifies this record.
        """
        return build_record_key(self.graph_name, self.engine_name, self.seed)

    def to_dict(self) -> dict[str, Any]:
        """Convert the record into a JSON-serializable mapping.

        Returns
        -------
        dict[str, Any]
            Plain dictionary suitable for ``json.dump``.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BenchmarkRecord":
        """Build a record from a JSON-decoded mapping.

        Parameters
        ----------
        payload : dict[str, Any]
            Serialized record payload.

        Returns
        -------
        BenchmarkRecord
            Parsed record with normalized optional fields.
        """
        return cls(
            graph_name=str(payload.get("graph_name", "")),
            engine_name=str(payload.get("engine_name", "")),
            seed=_optional_int(payload.get("seed")),
            status=str(payload.get("status", "error")),
            runtime_seconds=_optional_float(payload.get("runtime_seconds")),
            error=_optional_str(payload.get("error")),
            positions_file=_optional_str(payload.get("positions_file")),
            num_nodes=int(payload.get("num_nodes", 0)),
            num_edges=int(payload.get("num_edges", 0)),
            is_stochastic=bool(payload.get("is_stochastic", False)),
            skip_reason=_optional_str(payload.get("skip_reason")),
            original_for=_string_list(payload.get("original_for")),
            reimpl_of=_string_list(payload.get("reimpl_of")),
        )

    def detail(self) -> str:
        """Return the most useful human-readable explanation for the record.

        Returns
        -------
        str
            Status detail string for progress logging.
        """
        if self.skip_reason:
            return self.skip_reason
        if self.error:
            return self.error
        if self.runtime_seconds is not None:
            return f"{self.runtime_seconds:.1f}s"
        return self.status


_WORKER_COMPETITORS: dict[str, Any] | None = None
_WORKER_GRAPHS: dict[str, Any] | None = None


def _optional_float(value: Any) -> Optional[float]:
    """Convert a raw value into a finite ``float`` when possible.

    Parameters
    ----------
    value : Any
        Raw JSON-decoded value.

    Returns
    -------
    float | None
        Finite float value, otherwise ``None``.
    """
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if not (converted == converted) or converted in (float("inf"), float("-inf")):
        return None
    return converted


def _optional_int(value: Any) -> Optional[int]:
    """Convert a raw value into ``int`` when possible.

    Parameters
    ----------
    value : Any
        Raw JSON-decoded value.

    Returns
    -------
    int | None
        Integer value, otherwise ``None``.
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> Optional[str]:
    """Convert a raw value into ``str`` when present.

    Parameters
    ----------
    value : Any
        Raw JSON-decoded value.

    Returns
    -------
    str | None
        String value, otherwise ``None``.
    """
    if value is None:
        return None
    return str(value)


def _string_list(value: Any) -> list[str]:
    """Normalize an unknown value into a list of strings.

    Parameters
    ----------
    value : Any
        Raw JSON-decoded value.

    Returns
    -------
    list[str]
        String list, or an empty list when conversion is not possible.
    """
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def build_record_key(graph_name: str, engine_name: str, seed: Optional[int]) -> str:
    """Build the flat key for one benchmark record.

    Parameters
    ----------
    graph_name : str
        Stable test-graph name.
    engine_name : str
        Registered competitor adapter name.
    seed : int | None
        Optional stochastic seed.

    Returns
    -------
    str
        Stable key for ``results.json``.
    """
    seed_part = "deterministic" if seed is None else f"seed{seed}"
    return f"{graph_name}::{engine_name}::{seed_part}"


def sanitize_component(value: str) -> str:
    """Return a filename-safe component for graph and engine names.

    Parameters
    ----------
    value : str
        Raw graph or engine name.

    Returns
    -------
    str
        Sanitized component containing only simple ASCII filename chars.
    """
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    sanitized = sanitized.strip("._")
    return sanitized or "item"


def position_relative_path(graph_name: str, engine_name: str, seed: Optional[int]) -> Path:
    """Build the relative output path for a position tensor.

    Parameters
    ----------
    graph_name : str
        Stable test-graph name.
    engine_name : str
        Registered competitor adapter name.
    seed : int | None
        Optional stochastic seed.

    Returns
    -------
    Path
        Relative path rooted under the benchmark output directory.
    """
    graph_component = sanitize_component(graph_name)
    engine_component = sanitize_component(engine_name)
    if seed is None:
        filename = f"{graph_component}__{engine_component}.pt"
    else:
        filename = f"{graph_component}__{engine_component}__seed{seed}.pt"
    return Path(POSITION_DIRNAME) / filename


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed CLI options.
    """
    parser = argparse.ArgumentParser(
        description="Run all layout algorithms on all selected test graphs."
    )
    parser.add_argument("--workers", type=str, default=str(DEFAULT_WORKERS))
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Per-layout timeout in seconds",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=DEFAULT_SEED_COUNT,
        help="Number of seeds for stochastic engines",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=0,
        help="Skip graphs with more nodes (0 = no graph-level limit)",
    )
    parser.add_argument(
        "--engines",
        type=str,
        default="all",
        help="Filter: 'all', 'originals', 'reimpl', or comma-separated names",
    )
    parser.add_argument(
        "--graphs",
        type=str,
        default=None,
        help="Comma-separated graph names to run (default: all filtered graphs)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed work from a previous run in the same output directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
    )
    parser.add_argument(
        "--no-positions",
        action="store_true",
        help="Skip saving position tensors and only record timing/status metadata",
    )
    parser.add_argument(
        "--variants",
        action="store_true",
        help="Expand classic and original engines into parameterized variant competitors",
    )
    return parser.parse_args()


def _pairings_for_engine_name(engine_name: str) -> tuple[list[str], list[str]]:
    """Return pairing metadata for one engine or synthetic variant name.

    Parameters
    ----------
    engine_name : str
        Base competitor name or synthetic variant competitor name.

    Returns
    -------
    tuple[list[str], list[str]]
        ``(original_for, reimpl_of)`` lists for ``BenchmarkRecord`` metadata.
    """
    variant = get_variant(engine_name)
    if variant is not None:
        original_name = original_variant_name(variant)
        return [], ([] if original_name is None else [original_name])

    original_variant = get_variant_for_original_name(engine_name)
    if original_variant is not None:
        return [original_variant.variant_id], []

    pairings = base_pairings()
    reversed_pairings = reverse_pairings(use_variants=False)
    return reversed_pairings.get(engine_name, []), pairings.get(engine_name, [])


def reverse_pairings(use_variants: bool) -> dict[str, list[str]]:
    """Build the reverse original-to-reimplementation pairing map.

    Parameters
    ----------
    use_variants : bool
        Whether to use synthetic variant pairings instead of base-engine
        pairings.

    Returns
    -------
    dict[str, list[str]]
        Mapping from original engine name to the reimplementations that cite it
        as a reference.
    """
    pairings = variant_pairings() if use_variants else base_pairings()
    reversed_map: dict[str, list[str]] = {}
    for reimpl_name, original_names in pairings.items():
        for original_name in original_names:
            reversed_map.setdefault(original_name, []).append(reimpl_name)
    for original_names in reversed_map.values():
        original_names.sort()
    return reversed_map


def seeds_for_engine(engine_name: str, seed_count: int) -> list[Optional[int]]:
    """Return the benchmark seeds for one engine.

    Parameters
    ----------
    engine_name : str
        Registered competitor adapter name.
    seed_count : int
        Number of seeds requested for stochastic engines.

    Returns
    -------
    list[int | None]
        One deterministic ``None`` entry or a sequence starting at 42.
    """
    if not engine_is_stochastic(engine_name):
        return [None]
    return list(range(DEFAULT_SEED_START, DEFAULT_SEED_START + seed_count))


def resolve_worker_count(workers_arg: str) -> int:
    """Resolve ``--workers`` into a concrete worker count.

    Parameters
    ----------
    workers_arg : str
        Raw CLI value, either a positive integer or ``"auto"``.

    Returns
    -------
    int
        Concrete worker count.
    """
    if workers_arg != "auto":
        return int(workers_arg)

    try:
        import psutil
    except ImportError:
        cores = os.cpu_count() or 1
        return max(1, min(cores - 1, 6))

    mem = psutil.virtual_memory()
    cores = os.cpu_count() or 1
    max_by_ram = max(1, int(mem.available / (4 * 1024**3)))
    max_by_cpu = max(1, cores - 1)
    return min(max_by_ram, max_by_cpu, 6)


def scaled_timeout(global_timeout: float, num_nodes: int) -> float:
    """Scale the per-job timeout based on graph size.

    Small graphs get a reduced timeout to avoid wasting minutes when an engine
    hangs on a trivial input.  The timeout scales linearly from
    ``MIN_TIMEOUT_SECONDS`` up to the full ``global_timeout`` at
    ``FULL_TIMEOUT_NODE_COUNT`` nodes.

    Parameters
    ----------
    global_timeout : float
        The ``--timeout`` value from the CLI.
    num_nodes : int
        Number of nodes in the graph.

    Returns
    -------
    float
        Effective timeout in seconds.
    """
    if num_nodes >= FULL_TIMEOUT_NODE_COUNT:
        return global_timeout
    size_ratio = max(0.15, num_nodes / FULL_TIMEOUT_NODE_COUNT)
    return max(MIN_TIMEOUT_SECONDS, global_timeout * size_ratio)


def graph_summary(test_graph: Any) -> GraphSummary:
    """Extract stable metadata from a ``TestGraph`` instance.

    Parameters
    ----------
    test_graph : Any
        Graph registry object returned by ``get_test_graphs``.

    Returns
    -------
    GraphSummary
        Lightweight summary used by the planner and manifest.
    """
    edge_index = test_graph.graph.edge_index
    num_edges = int(edge_index.shape[1]) if edge_index.numel() > 0 else 0
    return GraphSummary(
        name=str(test_graph.name),
        num_nodes=int(test_graph.graph.num_nodes),
        num_edges=num_edges,
        tags=sorted(str(tag) for tag in test_graph.tags),
    )


def select_graphs(graph_filter: Optional[str], max_nodes: int) -> list[Any]:
    """Select benchmark graphs based on the CLI filters.

    Parameters
    ----------
    graph_filter : str | None
        Optional comma-separated graph-name filter.
    max_nodes : int
        Graph-level node-count limit. ``0`` means no limit.

    Returns
    -------
    list[Any]
        Selected ``TestGraph`` objects.

    Raises
    ------
    ValueError
        Raised when a requested graph name is unknown.
    """
    from dagua.eval.graphs import get_test_graphs

    selected = get_test_graphs(max_nodes=max_nodes if max_nodes > 0 else None)
    # Sort small-to-large so tiny graphs finish fast and give early feedback,
    # while the expensive large-graph + slow-engine combos concentrate at the
    # end where timeout skipping has the most leverage.
    selected = sorted(selected, key=lambda tg: (tg.graph.num_nodes, tg.name))
    if graph_filter is None:
        return selected

    requested_names = [name.strip() for name in graph_filter.split(",") if name.strip()]
    if "tiny_graph" in requested_names and selected:
        smallest_graph = min(selected, key=lambda test_graph: test_graph.graph.num_nodes)
        requested_names = [
            smallest_graph.name if requested_name == "tiny_graph" else requested_name
            for requested_name in requested_names
        ]
    by_name = {test_graph.name: test_graph for test_graph in selected}
    missing = [name for name in requested_names if name not in by_name]
    if missing:
        raise ValueError(f"Unknown graph selection: {', '.join(sorted(missing))}")
    return [by_name[name] for name in requested_names]


def _expand_engine_name_for_variants(engine_name: str) -> list[str]:
    """Expand one engine selection into variant competitor names.

    Parameters
    ----------
    engine_name : str
        Base competitor name or already-expanded variant name.

    Returns
    -------
    list[str]
        Expanded engine names in benchmark order.
    """
    is_variant = get_variant(engine_name) is not None
    is_original_variant = get_variant_for_original_name(engine_name) is not None
    if is_variant or is_original_variant:
        return [engine_name]

    reimpl_variants = variants_for_base_engine(engine_name)
    if reimpl_variants:
        return [variant.variant_id for variant in reimpl_variants]

    original_names = [
        original_name
        for variant in sum((variants_for_base_engine(base) for base in base_pairings().keys()), [])
        for original_name in [original_variant_name(variant)]
        if variant.original_engine == engine_name and original_name is not None
    ]
    if original_names:
        return original_names
    return [engine_name]


def _dedupe_preserving_order(values: Sequence[str]) -> list[str]:
    """Return values with duplicates removed and order preserved.

    Parameters
    ----------
    values : Sequence[str]
        Values to deduplicate.

    Returns
    -------
    list[str]
        Unique values in first-seen order.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _build_engine_instance(
    engine_name: str,
    competitor_by_name: dict[str, Any],
) -> Any:
    """Build a concrete competitor or synthetic variant wrapper.

    Parameters
    ----------
    engine_name : str
        Base competitor name or synthetic variant competitor name.
    competitor_by_name : dict[str, Any]
        Registered base competitors keyed by name.

    Returns
    -------
    Any
        Base competitor instance or ``VariantCompetitor`` wrapper.
    """
    from dagua.eval.competitors.classic_competitor import VariantCompetitor

    variant = get_variant(engine_name)
    if variant is not None:
        base_competitor = competitor_by_name[variant.base_engine]
        return VariantCompetitor(
            base_competitor=base_competitor,
            variant_params=variant.reimpl_params,
            name=variant.variant_id,
            display_name=variant.display_name,
            is_heavy=variant.is_heavy,
        )

    original_variant = get_variant_for_original_name(engine_name)
    if original_variant is not None and original_variant.original_engine is not None:
        base_competitor = competitor_by_name[original_variant.original_engine]
        return VariantCompetitor(
            base_competitor=base_competitor,
            variant_params=original_variant.original_params,
            name=engine_name,
            display_name=f"{original_variant.display_name} [original]",
            is_heavy=engine_is_heavy(engine_name),
        )

    return competitor_by_name[engine_name]


def select_engines(engine_filter: str, include_variants: bool) -> list[Any]:
    """Select competitor adapters based on the CLI filter.

    Parameters
    ----------
    engine_filter : str
        Filter string from ``--engines``.
    include_variants : bool
        Whether classic and original engines should expand into synthetic
        variant competitors.

    Returns
    -------
    list[Any]
        Selected competitor adapter instances.

    Raises
    ------
    ValueError
        Raised when explicitly requested engine names are unknown.
    """
    from dagua.eval.competitors import get_available_competitors, get_competitors

    all_competitors = get_competitors()
    competitor_by_name = {competitor.name: competitor for competitor in all_competitors}
    available_names = {competitor.name for competitor in get_available_competitors()}

    if engine_filter == "all":
        selected_names = [
            competitor.name for competitor in all_competitors if competitor.name in available_names
        ]
    elif engine_filter == "originals":
        selected_names = [
            competitor.name
            for competitor in all_competitors
            if competitor.name in available_names
            and not competitor.name.startswith("classic_")
            and competitor.name != "dagua"
        ]
    elif engine_filter == "reimpl":
        selected_names = [
            competitor.name
            for competitor in all_competitors
            if competitor.name in available_names and competitor.name.startswith("classic_")
        ]
    else:
        requested_names = [name.strip() for name in engine_filter.split(",") if name.strip()]
        known_names = {
            *competitor_by_name.keys(),
            *(
                variant.variant_id
                for base in base_pairings()
                for variant in variants_for_base_engine(base)
            ),
            *original_variant_engine_names(),
        }
        missing = [name for name in requested_names if name not in known_names]
        if missing:
            raise ValueError(f"Unknown engine selection: {', '.join(sorted(missing))}")
        selected_names = requested_names

    if include_variants:
        expanded_names = _dedupe_preserving_order(
            [
                expanded_name
                for selected_name in selected_names
                for expanded_name in _expand_engine_name_for_variants(selected_name)
            ]
        )
        return [
            _build_engine_instance(engine_name, competitor_by_name)
            for engine_name in expanded_names
        ]

    if engine_filter == "all":
        return [c for c in all_competitors if c.name in available_names]
    if engine_filter == "originals":
        return [
            c
            for c in all_competitors
            if c.name in available_names and not c.name.startswith("classic_") and c.name != "dagua"
        ]
    if engine_filter == "reimpl":
        return [
            c
            for c in all_competitors
            if c.name in available_names and c.name.startswith("classic_")
        ]
    return [competitor_by_name[name] for name in selected_names]


def _load_results(path: Path) -> dict[str, BenchmarkRecord]:
    """Load existing benchmark results from disk.

    Parameters
    ----------
    path : Path
        Path to ``results.json``.

    Returns
    -------
    dict[str, BenchmarkRecord]
        Parsed records keyed by ``build_record_key(...)``.
    """
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    records: dict[str, BenchmarkRecord] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            records[str(key)] = BenchmarkRecord.from_dict(value)
    return records


def _save_json_atomic(path: Path, data: dict[str, Any]) -> None:
    """Write a JSON payload atomically through a temporary file.

    Parameters
    ----------
    path : Path
        Destination file path.
    data : dict[str, Any]
        Serializable payload.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f"{path.stem}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file_obj:
            json.dump(data, file_obj, indent=2, default=str)
        Path(temp_path).replace(path)
    except BaseException:
        Path(temp_path).unlink(missing_ok=True)
        raise


def _save_tensor_atomic(path: Path, tensor: torch.Tensor) -> None:
    """Persist a tensor atomically through a temporary file.

    Parameters
    ----------
    path : Path
        Destination ``.pt`` path.
    tensor : torch.Tensor
        Position tensor to serialize.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f"{path.stem}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    try:
        torch.save(tensor, temp_path)
        Path(temp_path).replace(path)
    except BaseException:
        Path(temp_path).unlink(missing_ok=True)
        raise


def save_results(path: Path, results: dict[str, BenchmarkRecord]) -> None:
    """Persist the flat results mapping to disk.

    Parameters
    ----------
    path : Path
        Destination ``results.json`` path.
    results : dict[str, BenchmarkRecord]
        Records keyed by ``build_record_key(...)``.
    """
    payload = {key: record.to_dict() for key, record in sorted(results.items())}
    _save_json_atomic(path, payload)


def recover_results_from_positions(
    output_dir: Path,
    graphs: Sequence[Any],
    engines: Sequence[Any],
    seed_count: int,
) -> dict[str, BenchmarkRecord]:
    """Rebuild best-effort successful records from saved position tensors.

    Parameters
    ----------
    output_dir : Path
        Benchmark output directory containing ``positions/``.
    graphs : Sequence[Any]
        Selected test-graph objects.
    engines : Sequence[Any]
        Selected competitor instances, including synthetic variants.
    seed_count : int
        Seed count requested for stochastic engines.

    Returns
    -------
    dict[str, BenchmarkRecord]
        Recovered successful records keyed by benchmark record key.
    """
    positions_dir = output_dir / POSITION_DIRNAME
    if not positions_dir.exists():
        return {}

    expected_by_filename: dict[str, tuple[str, str, Optional[int], GraphSummary]] = {}
    for test_graph in graphs:
        summary = graph_summary(test_graph)
        for competitor in engines:
            for seed in seeds_for_engine(competitor.name, seed_count):
                relative_path = position_relative_path(summary.name, competitor.name, seed)
                expected_by_filename[relative_path.name] = (
                    summary.name,
                    competitor.name,
                    seed,
                    summary,
                )

    recovered: dict[str, BenchmarkRecord] = {}
    for path in positions_dir.glob("*.pt"):
        expected = expected_by_filename.get(path.name)
        if expected is None:
            continue
        graph_name, engine_name, seed, summary = expected
        try:
            tensor = torch.load(path, map_location="cpu")
        except Exception:
            continue
        if not isinstance(tensor, torch.Tensor):
            continue
        if tuple(tensor.shape) != (summary.num_nodes, 2):
            continue

        record = _record_with_pairings(
            graph_name=graph_name,
            engine_name=engine_name,
            seed=seed,
            num_nodes=summary.num_nodes,
            num_edges=summary.num_edges,
            status="ok",
            runtime_seconds=None,
            error=None,
            positions_file=str(Path(POSITION_DIRNAME) / path.name),
            skip_reason=None,
        )
        recovered[record.key] = record
    return recovered


def merge_recovered_results(
    existing_results: dict[str, BenchmarkRecord],
    recovered_results: dict[str, BenchmarkRecord],
) -> dict[str, BenchmarkRecord]:
    """Merge recovered position-backed records into existing results.

    Parameters
    ----------
    existing_results : dict[str, BenchmarkRecord]
        Existing ``results.json`` contents.
    recovered_results : dict[str, BenchmarkRecord]
        Best-effort position-backed successful records.

    Returns
    -------
    dict[str, BenchmarkRecord]
        Merged results preferring recovered ``status="ok"`` records while
        preserving existing runtime metadata when available.
    """
    merged = dict(existing_results)
    for key, recovered_record in recovered_results.items():
        existing_record = merged.get(key)
        if existing_record is not None and existing_record.runtime_seconds is not None:
            recovered_record.runtime_seconds = existing_record.runtime_seconds
        merged[key] = recovered_record
    return merged


def is_record_complete(
    record: Optional[BenchmarkRecord],
    output_dir: Path,
    save_positions: bool,
) -> bool:
    """Return whether a record is complete enough for ``--resume``.

    Parameters
    ----------
    record : BenchmarkRecord | None
        Existing record to evaluate.
    output_dir : Path
        Benchmark output directory.
    save_positions : bool
        Whether the current run expects position tensors to exist.

    Returns
    -------
    bool
        ``True`` when the record should be skipped on resume.
    """
    if record is None or record.status == RUNNING_STATUS:
        return False
    if not save_positions:
        return True
    if record.status != "ok":
        return True
    if record.positions_file is None:
        return False
    return (output_dir / record.positions_file).exists()


def scoped_results(
    results: dict[str, BenchmarkRecord],
    scoped_keys: Sequence[str],
) -> dict[str, BenchmarkRecord]:
    """Filter the global result mapping to the current benchmark scope.

    Parameters
    ----------
    results : dict[str, BenchmarkRecord]
        All records currently stored for the output directory.
    scoped_keys : Sequence[str]
        Keys planned for the current invocation.

    Returns
    -------
    dict[str, BenchmarkRecord]
        Filtered mapping containing only records from the current scope.
    """
    scoped_key_set = set(scoped_keys)
    return {key: record for key, record in results.items() if key in scoped_key_set}


def _timeout_signal_handler(signum: int, frame: Any) -> None:
    """Raise a worker timeout when the alarm signal fires.

    Parameters
    ----------
    signum : int
        Delivered signal number.
    frame : Any
        Current frame from the signal handler API.
    """
    del signum, frame
    raise _WorkerLayoutTimeoutError("worker layout timeout exceeded")


def _enable_worker_timeout(timeout_seconds: float) -> Optional[Any]:
    """Install a wall-clock timeout in the current worker process.

    Parameters
    ----------
    timeout_seconds : float
        Maximum allowed wall-clock runtime.

    Returns
    -------
    object | None
        Previous signal handler when ``SIGALRM`` is supported.
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
        Previous signal handler returned by ``_enable_worker_timeout``.
    """
    if not hasattr(signal, "SIGALRM"):
        return
    signal.setitimer(signal.ITIMER_REAL, 0.0)
    if previous_handler is not None:
        signal.signal(signal.SIGALRM, previous_handler)


def set_runtime_seed(seed: Optional[int]) -> None:
    """Configure process-local RNG state for one benchmark attempt.

    Parameters
    ----------
    seed : int | None
        Explicit seed for stochastic engines, or ``None`` to clear the
        benchmark-specific environment override.
    """
    if seed is None:
        os.environ.pop("DAGUA_COMPETITOR_SEED", None)
        return

    os.environ["DAGUA_COMPETITOR_SEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def _ensure_worker_cache() -> tuple[dict[str, Any], dict[str, Any]]:
    """Lazy-initialize competitor and graph caches inside a worker process.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        Competitor registry and graph registry keyed by name.
    """
    global _WORKER_COMPETITORS, _WORKER_GRAPHS  # noqa: PLW0603

    if _WORKER_COMPETITORS is None:
        from dagua.eval.competitors import get_competitors

        base_competitors = get_competitors()
        _WORKER_COMPETITORS = {competitor.name: competitor for competitor in base_competitors}
        for variant_name in _dedupe_preserving_order(
            [
                variant.variant_id
                for base_engine in base_pairings().keys()
                for variant in variants_for_base_engine(base_engine)
            ]
            + original_variant_engine_names()
        ):
            _WORKER_COMPETITORS[variant_name] = _build_engine_instance(
                variant_name,
                _WORKER_COMPETITORS,
            )
    if _WORKER_GRAPHS is None:
        from dagua.eval.graphs import get_test_graphs

        _WORKER_GRAPHS = {test_graph.name: test_graph for test_graph in get_test_graphs()}
    return _WORKER_COMPETITORS, _WORKER_GRAPHS


def _record_with_pairings(
    graph_name: str,
    engine_name: str,
    seed: Optional[int],
    num_nodes: int,
    num_edges: int,
    status: str,
    runtime_seconds: Optional[float],
    error: Optional[str],
    positions_file: Optional[str],
    skip_reason: Optional[str],
) -> BenchmarkRecord:
    """Build a ``BenchmarkRecord`` enriched with pairing metadata.

    Parameters
    ----------
    graph_name : str
        Stable test-graph name.
    engine_name : str
        Registered competitor adapter name.
    seed : int | None
        Optional seed.
    num_nodes : int
        Number of graph nodes.
    num_edges : int
        Number of graph edges.
    status : str
        Execution status string.
    runtime_seconds : float | None
        Measured runtime if available.
    error : str | None
        Error message for failed runs.
    positions_file : str | None
        Relative path to the saved position tensor.
    skip_reason : str | None
        Human-readable skip reason.

    Returns
    -------
    BenchmarkRecord
        Fully populated record.
    """
    original_for, reimpl_of = _pairings_for_engine_name(engine_name)
    return BenchmarkRecord(
        graph_name=graph_name,
        engine_name=engine_name,
        seed=seed,
        status=status,
        runtime_seconds=runtime_seconds,
        error=error,
        positions_file=positions_file,
        num_nodes=num_nodes,
        num_edges=num_edges,
        is_stochastic=engine_is_stochastic(engine_name),
        skip_reason=skip_reason,
        original_for=original_for,
        reimpl_of=reimpl_of,
    )


def _run_single_work_item(work_item: WorkItem) -> dict[str, Any]:
    """Run one graph/engine/seed benchmark attempt in a worker process.

    Parameters
    ----------
    work_item : WorkItem
        Worker execution payload.

    Returns
    -------
    dict[str, Any]
        Serialized ``BenchmarkRecord``.
    """
    competitors, graphs = _ensure_worker_cache()

    competitor = competitors.get(work_item.engine_name)
    test_graph = graphs.get(work_item.graph_name)
    if competitor is None:
        record = _record_with_pairings(
            graph_name=work_item.graph_name,
            engine_name=work_item.engine_name,
            seed=work_item.seed,
            num_nodes=0,
            num_edges=0,
            status="skipped",
            runtime_seconds=None,
            error=None,
            positions_file=None,
            skip_reason="not installed",
        )
        return record.to_dict()
    if test_graph is None:
        record = _record_with_pairings(
            graph_name=work_item.graph_name,
            engine_name=work_item.engine_name,
            seed=work_item.seed,
            num_nodes=0,
            num_edges=0,
            status="error",
            runtime_seconds=None,
            error="graph not found",
            positions_file=None,
            skip_reason=None,
        )
        return record.to_dict()

    graph = test_graph.graph
    graph.compute_node_sizes()
    edge_index = graph.edge_index
    num_edges = int(edge_index.shape[1]) if edge_index.numel() > 0 else 0

    if competitor.max_nodes > 0 and graph.num_nodes > competitor.max_nodes:
        record = _record_with_pairings(
            graph_name=work_item.graph_name,
            engine_name=work_item.engine_name,
            seed=work_item.seed,
            num_nodes=graph.num_nodes,
            num_edges=num_edges,
            status="skipped",
            runtime_seconds=None,
            error=None,
            positions_file=None,
            skip_reason=f"exceeds max_nodes={competitor.max_nodes}",
        )
        return record.to_dict()

    previous_handler = _enable_worker_timeout(work_item.timeout_seconds)
    set_runtime_seed(work_item.seed)
    try:
        start = time.perf_counter()
        result = competitor.layout(graph, timeout=work_item.timeout_seconds, seed=work_item.seed)
        elapsed = result.runtime_seconds
        if result.error is not None and result.error.lower() == "timeout":
            record = _record_with_pairings(
                graph_name=work_item.graph_name,
                engine_name=work_item.engine_name,
                seed=work_item.seed,
                num_nodes=graph.num_nodes,
                num_edges=num_edges,
                status="timeout",
                runtime_seconds=elapsed,
                error=None,
                positions_file=None,
                skip_reason=None,
            )
            return record.to_dict()
        if result.pos is None:
            record = _record_with_pairings(
                graph_name=work_item.graph_name,
                engine_name=work_item.engine_name,
                seed=work_item.seed,
                num_nodes=graph.num_nodes,
                num_edges=num_edges,
                status="error",
                runtime_seconds=elapsed,
                error=result.error or "no positions returned",
                positions_file=None,
                skip_reason=None,
            )
            return record.to_dict()
        if tuple(result.pos.shape) != (graph.num_nodes, 2):
            record = _record_with_pairings(
                graph_name=work_item.graph_name,
                engine_name=work_item.engine_name,
                seed=work_item.seed,
                num_nodes=graph.num_nodes,
                num_edges=num_edges,
                status="error",
                runtime_seconds=elapsed,
                error=f"invalid position shape {tuple(result.pos.shape)}",
                positions_file=None,
                skip_reason=None,
            )
            return record.to_dict()

        positions_file: Optional[str] = None
        if work_item.save_positions:
            relative_path = position_relative_path(
                graph_name=work_item.graph_name,
                engine_name=work_item.engine_name,
                seed=work_item.seed,
            )
            output_path = Path(work_item.output_dir) / relative_path
            _save_tensor_atomic(output_path, result.pos.detach().cpu())
            positions_file = str(relative_path)

        record = _record_with_pairings(
            graph_name=work_item.graph_name,
            engine_name=work_item.engine_name,
            seed=work_item.seed,
            num_nodes=graph.num_nodes,
            num_edges=num_edges,
            status="ok",
            runtime_seconds=elapsed,
            error=None,
            positions_file=positions_file,
            skip_reason=None,
        )
        return record.to_dict()
    except _WorkerLayoutTimeoutError:
        elapsed = time.perf_counter() - start
        record = _record_with_pairings(
            graph_name=work_item.graph_name,
            engine_name=work_item.engine_name,
            seed=work_item.seed,
            num_nodes=graph.num_nodes,
            num_edges=num_edges,
            status="timeout",
            runtime_seconds=elapsed,
            error=None,
            positions_file=None,
            skip_reason=None,
        )
        return record.to_dict()
    except Exception as exc:
        elapsed = time.perf_counter() - start
        record = _record_with_pairings(
            graph_name=work_item.graph_name,
            engine_name=work_item.engine_name,
            seed=work_item.seed,
            num_nodes=graph.num_nodes,
            num_edges=num_edges,
            status="error",
            runtime_seconds=elapsed,
            error=f"{type(exc).__name__}: {exc}",
            positions_file=None,
            skip_reason=None,
        )
        return record.to_dict()
    finally:
        set_runtime_seed(None)
        _disable_worker_timeout(previous_handler)


def _is_failure_record(record: BenchmarkRecord) -> bool:
    """Return whether a record should count against the consecutive failure threshold.

    Timeouts and errors both count.  Successful and skipped records reset the
    consecutive failure counter.

    Parameters
    ----------
    record : BenchmarkRecord
        Completed benchmark record.

    Returns
    -------
    bool
        ``True`` when the record represents a timeout or error.
    """
    return record.status in {"timeout", "error"}


def group_work_items(work_items: Sequence[WorkItem]) -> list[list[WorkItem]]:
    """Group work items by ``(engine, graph)`` while preserving order.

    Parameters
    ----------
    work_items : Sequence[WorkItem]
        Planned benchmark attempts.

    Returns
    -------
    list[list[WorkItem]]
        Work-item groups whose seeds should execute sequentially together.
    """
    grouped: dict[tuple[str, str], list[WorkItem]] = {}
    ordered_keys: list[tuple[str, str]] = []
    for work_item in work_items:
        group_key = (work_item.engine_name, work_item.graph_name)
        if group_key not in grouped:
            grouped[group_key] = []
            ordered_keys.append(group_key)
        grouped[group_key].append(work_item)
    return [grouped[group_key] for group_key in ordered_keys]


def _run_work_group(work_group: Sequence[WorkItem]) -> list[dict[str, Any]]:
    """Run one ``(engine, graph)`` seed group sequentially.

    Parameters
    ----------
    work_group : Sequence[WorkItem]
        All seeds for one engine/graph combination.

    Returns
    -------
    list[dict[str, Any]]
        Serialized benchmark records for the group.
    """
    if not work_group:
        return []

    competitors, graphs = _ensure_worker_cache()
    del competitors
    test_graph = graphs.get(work_group[0].graph_name)
    num_nodes = 0
    num_edges = 0
    if test_graph is not None:
        edge_index = test_graph.graph.edge_index
        num_nodes = int(test_graph.graph.num_nodes)
        num_edges = int(edge_index.shape[1]) if edge_index.numel() > 0 else 0

    records: list[dict[str, Any]] = []
    consecutive_failures = 0
    last_failure_error: str = "failures"
    for work_item in work_group:
        if consecutive_failures >= CONSECUTIVE_FAILURE_SKIP_THRESHOLD:
            records.append(
                _record_with_pairings(
                    graph_name=work_item.graph_name,
                    engine_name=work_item.engine_name,
                    seed=work_item.seed,
                    num_nodes=num_nodes,
                    num_edges=num_edges,
                    status="skipped",
                    runtime_seconds=None,
                    error=None,
                    positions_file=None,
                    skip_reason=(
                        f"skipped after {CONSECUTIVE_FAILURE_SKIP_THRESHOLD} "
                        f"consecutive {last_failure_error}"
                    ),
                ).to_dict()
            )
            continue

        record = BenchmarkRecord.from_dict(_run_single_work_item(work_item))
        if _is_failure_record(record):
            consecutive_failures += 1
            last_failure_error = "timeouts" if record.status == "timeout" else "errors"
        else:
            consecutive_failures = 0
        records.append(record.to_dict())
    return records


def running_record(
    graph_name: str,
    engine_name: str,
    seed: Optional[int],
    num_nodes: int,
    num_edges: int,
) -> BenchmarkRecord:
    """Build a placeholder record used while a task is in flight.

    Parameters
    ----------
    graph_name : str
        Stable test-graph name.
    engine_name : str
        Registered competitor adapter name.
    seed : int | None
        Optional stochastic seed.
    num_nodes : int
        Number of graph nodes.
    num_edges : int
        Number of graph edges.

    Returns
    -------
    BenchmarkRecord
        Placeholder record with ``status="running"``.
    """
    return _record_with_pairings(
        graph_name=graph_name,
        engine_name=engine_name,
        seed=seed,
        num_nodes=num_nodes,
        num_edges=num_edges,
        status=RUNNING_STATUS,
        runtime_seconds=None,
        error=None,
        positions_file=None,
        skip_reason=None,
    )


def progress_line(prefix: str, completed: int, total: int, record: BenchmarkRecord) -> str:
    """Format a concise benchmark progress line.

    Parameters
    ----------
    prefix : str
        Leading log label such as ``"complete"``.
    completed : int
        Number of completed scoped tasks.
    total : int
        Total scoped task count.
    record : BenchmarkRecord
        Most recently completed record.

    Returns
    -------
    str
        Human-readable progress line.
    """
    percent = 100.0 if total == 0 else (100.0 * completed / total)
    seed_part = "" if record.seed is None else f" seed={record.seed}"
    detail = record.detail()
    return (
        f"[benchmark] {completed}/{total} complete ({percent:.1f}%) | "
        f"{record.graph_name} x {record.engine_name}{seed_part}: {record.status}"
        f"{f' ({detail})' if detail else ''}"
    )


def status_counts(records: Sequence[BenchmarkRecord]) -> dict[str, int]:
    """Count benchmark records by status.

    Parameters
    ----------
    records : Sequence[BenchmarkRecord]
        Records to summarize.

    Returns
    -------
    dict[str, int]
        Counts keyed by status string.
    """
    counts = {"ok": 0, "skipped": 0, "error": 0, "timeout": 0, RUNNING_STATUS: 0}
    for record in records:
        counts[record.status] = counts.get(record.status, 0) + 1
    return counts


def _format_float(value: Optional[float]) -> str:
    """Format an optional float for summary tables.

    Parameters
    ----------
    value : float | None
        Numeric value to format.

    Returns
    -------
    str
        Decimal string or ``"-"``.
    """
    if value is None:
        return "-"
    return f"{value:.2f}"


def engine_summary_rows(
    records: Sequence[BenchmarkRecord],
    engine_names: Sequence[str],
) -> list[str]:
    """Build the markdown table rows for the per-engine summary.

    Parameters
    ----------
    records : Sequence[BenchmarkRecord]
        Scoped benchmark records.
    engine_names : Sequence[str]
        Selected engine names in display order.

    Returns
    -------
    list[str]
        Markdown table rows.
    """
    rows = ["| Engine | OK | Skipped | Error | Timeout | Success % | Mean s | Median s | Max s |"]
    rows.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for engine_name in engine_names:
        engine_records = [record for record in records if record.engine_name == engine_name]
        runtimes = [record.runtime_seconds for record in engine_records if record.status == "ok"]
        finite_runtimes = [runtime for runtime in runtimes if runtime is not None]
        total = len(engine_records)
        ok_count = sum(record.status == "ok" for record in engine_records)
        skipped_count = sum(record.status == "skipped" for record in engine_records)
        error_count = sum(record.status == "error" for record in engine_records)
        timeout_count = sum(record.status == "timeout" for record in engine_records)
        success_rate = 0.0 if total == 0 else (100.0 * ok_count / total)
        mean_runtime = statistics.mean(finite_runtimes) if finite_runtimes else None
        median_runtime = statistics.median(finite_runtimes) if finite_runtimes else None
        max_runtime = max(finite_runtimes) if finite_runtimes else None
        rows.append(
            "| "
            f"{engine_name} | {ok_count} | {skipped_count} | {error_count} | {timeout_count} | "
            f"{success_rate:.1f} | {_format_float(mean_runtime)} | "
            f"{_format_float(median_runtime)} | {_format_float(max_runtime)} |"
        )
    return rows


def graph_summary_rows(records: Sequence[BenchmarkRecord], graph_names: Sequence[str]) -> list[str]:
    """Build the markdown table rows for the per-graph summary.

    Parameters
    ----------
    records : Sequence[BenchmarkRecord]
        Scoped benchmark records.
    graph_names : Sequence[str]
        Selected graph names in display order.

    Returns
    -------
    list[str]
        Markdown table rows.
    """
    rows = ["| Graph | Total | OK | Skipped | Error | Timeout |"]
    rows.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for graph_name in graph_names:
        graph_records = [record for record in records if record.graph_name == graph_name]
        rows.append(
            "| "
            f"{graph_name} | {len(graph_records)} | "
            f"{sum(record.status == 'ok' for record in graph_records)} | "
            f"{sum(record.status == 'skipped' for record in graph_records)} | "
            f"{sum(record.status == 'error' for record in graph_records)} | "
            f"{sum(record.status == 'timeout' for record in graph_records)} |"
        )
    return rows


def pairing_summary_rows(records: Sequence[BenchmarkRecord]) -> list[str]:
    """Build the markdown rows for reimplementation pairing coverage.

    Parameters
    ----------
    records : Sequence[BenchmarkRecord]
        Scoped benchmark records.

    Returns
    -------
    list[str]
        Markdown table rows.
    """
    rows = ["| Reimpl | Originals | Reimpl OK | Original OK | Status |"]
    rows.append("| --- | --- | ---: | ---: | --- |")
    ok_counts_by_engine: dict[str, int] = {}
    for record in records:
        if record.status == "ok":
            ok_counts_by_engine[record.engine_name] = (
                ok_counts_by_engine.get(record.engine_name, 0) + 1
            )

    pairings: dict[str, list[str]] = {}
    for record in records:
        if record.reimpl_of:
            pairings.setdefault(record.engine_name, list(record.reimpl_of))

    for reimpl_name, original_names in sorted(pairings.items()):
        reimpl_ok = ok_counts_by_engine.get(reimpl_name, 0)
        original_ok = sum(
            ok_counts_by_engine.get(original_name, 0) for original_name in original_names
        )
        if not original_names:
            status = "no original configured"
        elif reimpl_ok > 0 and original_ok > 0:
            status = "ready"
        elif reimpl_ok > 0:
            status = "missing originals"
        elif original_ok > 0:
            status = "missing reimpl"
        else:
            status = "not run"
        rows.append(
            "| "
            f"{reimpl_name} | {', '.join(original_names) or '-'} | "
            f"{reimpl_ok} | {original_ok} | {status} |"
        )
    return rows


def skip_reason_lines(records: Sequence[BenchmarkRecord]) -> list[str]:
    """Aggregate skip reasons for the summary report.

    Parameters
    ----------
    records : Sequence[BenchmarkRecord]
        Scoped benchmark records.

    Returns
    -------
    list[str]
        Bullet lines for the skip-reason section.
    """
    counts: dict[str, int] = {}
    for record in records:
        if record.status == "skipped" and record.skip_reason:
            counts[record.skip_reason] = counts.get(record.skip_reason, 0) + 1
    if not counts:
        return ["- None"]
    return [f"- {reason}: {count}" for reason, count in sorted(counts.items())]


def generate_summary(
    output_dir: Path,
    records: Sequence[BenchmarkRecord],
    engine_names: Sequence[str],
    graph_names: Sequence[str],
) -> None:
    """Write ``summary.md`` for the current benchmark scope.

    Parameters
    ----------
    output_dir : Path
        Root benchmark output directory.
    records : Sequence[BenchmarkRecord]
        Scoped benchmark records.
    engine_names : Sequence[str]
        Selected engine names in display order.
    graph_names : Sequence[str]
        Selected graph names in display order.
    """
    counts = status_counts(records)
    lines = [
        "# Unified Benchmark Summary",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Records: {len(records)}",
        f"- OK: {counts.get('ok', 0)}",
        f"- Skipped: {counts.get('skipped', 0)}",
        f"- Errors: {counts.get('error', 0)}",
        f"- Timeouts: {counts.get('timeout', 0)}",
        "",
        "## Per-Engine Success",
        "",
        *engine_summary_rows(records, engine_names),
        "",
        "## Per-Graph Completion",
        "",
        *graph_summary_rows(records, graph_names),
        "",
        "## Reimpl vs Original Pairings",
        "",
        *pairing_summary_rows(records),
        "",
        "## Skip Reasons",
        "",
        *skip_reason_lines(records),
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def manifest_payload(
    args: argparse.Namespace,
    graphs: Sequence[Any],
    engines: Sequence[Any],
    total_scope: int,
    completed_scope: int,
) -> dict[str, Any]:
    """Build the benchmark manifest payload.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    graphs : Sequence[Any]
        Selected ``TestGraph`` objects.
    engines : Sequence[Any]
        Selected competitor adapters.
    total_scope : int
        Total graph/engine/seed record count in scope.
    completed_scope : int
        Number of completed records currently available for that scope.

    Returns
    -------
    dict[str, Any]
        JSON-serializable manifest payload.
    """
    use_variants = bool(args.variants)
    pairings = variant_pairings() if use_variants else base_pairings()
    original_to_reimpl = reverse_pairings(use_variants=use_variants)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "workers": str(args.workers),
            "resolved_workers": int(args.resolved_workers),
            "timeout": int(args.timeout),
            "seeds": int(args.seeds),
            "max_nodes": int(args.max_nodes),
            "engines": str(args.engines),
            "graphs": args.graphs,
            "resume": bool(args.resume),
            "output_dir": str(args.output_dir),
            "save_positions": not bool(args.no_positions),
            "variants": use_variants,
        },
        "seed_values": list(range(DEFAULT_SEED_START, DEFAULT_SEED_START + int(args.seeds))),
        "stochastic_engines": sorted(
            competitor.name for competitor in engines if engine_is_stochastic(competitor.name)
        ),
        "pairings": {
            "reimpl_to_original": pairings,
            "original_to_reimpl": original_to_reimpl,
        },
        "graphs": [asdict(graph_summary(test_graph)) for test_graph in graphs],
        "engines": [
            {
                "name": competitor.name,
                "max_nodes": int(getattr(competitor, "max_nodes", 0)),
                "available": bool(competitor.available()),
                "is_stochastic": engine_is_stochastic(competitor.name),
                "is_heavy": engine_is_heavy(competitor.name),
                "original_for": original_to_reimpl.get(competitor.name, []),
                "reimpl_of": pairings.get(competitor.name, []),
            }
            for competitor in engines
        ],
        "total_scope": total_scope,
        "completed_scope": completed_scope,
    }


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments before starting the benchmark.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Raises
    ------
    ValueError
        Raised when a numeric argument is out of range.
    """
    if args.workers != "auto":
        try:
            if int(args.workers) <= 0:
                raise ValueError("--workers must be positive")
        except ValueError as exc:
            raise ValueError("--workers must be a positive integer or 'auto'") from exc
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if args.seeds <= 0:
        raise ValueError("--seeds must be positive")
    if args.max_nodes < 0:
        raise ValueError("--max-nodes must be non-negative")


def main() -> int:
    """Run the unified benchmark and write results to disk.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args()
    validate_args(args)
    args.resolved_workers = resolve_worker_count(args.workers)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)
    print("[benchmark] Loading graphs and engines...")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    manifest_path = output_dir / "manifest.json"

    try:
        selected_engines = select_engines(args.engines, include_variants=bool(args.variants))
        selected_graphs = select_graphs(args.graphs, args.max_nodes)
    except ValueError as exc:
        print(f"[benchmark] ERROR: {exc}")
        return 2

    disk_results = _load_results(results_path)
    recovered_results = (
        {}
        if args.no_positions
        else recover_results_from_positions(
            output_dir=output_dir,
            graphs=selected_graphs,
            engines=selected_engines,
            seed_count=args.seeds,
        )
    )
    existing_results = merge_recovered_results(disk_results, recovered_results)
    results = dict(existing_results if args.resume else recovered_results)

    engine_names = [competitor.name for competitor in selected_engines]
    graph_names = [test_graph.name for test_graph in selected_graphs]
    graph_summaries = {
        summary.name: summary
        for summary in (graph_summary(test_graph) for test_graph in selected_graphs)
    }

    total_scope = 0
    completed_scope = 0
    graph_completion_targets = {graph_name: 0 for graph_name in graph_names}
    graph_completion_counts = {graph_name: 0 for graph_name in graph_names}
    scoped_keys: list[str] = []
    work_items: list[WorkItem] = []
    immediate_records: list[BenchmarkRecord] = []

    for test_graph in selected_graphs:
        summary = graph_summaries[test_graph.name]
        for competitor in selected_engines:
            for seed in seeds_for_engine(competitor.name, args.seeds):
                key = build_record_key(summary.name, competitor.name, seed)
                scoped_keys.append(key)
                total_scope += 1
                graph_completion_targets[summary.name] += 1

                existing_record = existing_results.get(key) if args.resume else None
                if is_record_complete(
                    existing_record,
                    output_dir=output_dir,
                    save_positions=not args.no_positions,
                ):
                    completed_scope += 1
                    graph_completion_counts[summary.name] += 1
                    continue

                if not competitor.available():
                    record = _record_with_pairings(
                        graph_name=summary.name,
                        engine_name=competitor.name,
                        seed=seed,
                        num_nodes=summary.num_nodes,
                        num_edges=summary.num_edges,
                        status="skipped",
                        runtime_seconds=None,
                        error=None,
                        positions_file=None,
                        skip_reason="not installed",
                    )
                    results[key] = record
                    immediate_records.append(record)
                    completed_scope += 1
                    graph_completion_counts[summary.name] += 1
                    continue

                if competitor.max_nodes > 0 and summary.num_nodes > competitor.max_nodes:
                    record = _record_with_pairings(
                        graph_name=summary.name,
                        engine_name=competitor.name,
                        seed=seed,
                        num_nodes=summary.num_nodes,
                        num_edges=summary.num_edges,
                        status="skipped",
                        runtime_seconds=None,
                        error=None,
                        positions_file=None,
                        skip_reason=f"exceeds max_nodes={competitor.max_nodes}",
                    )
                    results[key] = record
                    immediate_records.append(record)
                    completed_scope += 1
                    graph_completion_counts[summary.name] += 1
                    continue

                graph_timeout = scaled_timeout(float(args.timeout), summary.num_nodes)
                cap = engine_timeout_cap(competitor.name)
                timeout_seconds = min(graph_timeout, cap) if cap is not None else graph_timeout
                work_items.append(
                    WorkItem(
                        graph_name=summary.name,
                        engine_name=competitor.name,
                        seed=seed,
                        timeout_seconds=timeout_seconds,
                        output_dir=str(output_dir),
                        save_positions=not args.no_positions,
                    )
                )

    save_results(results_path, results)
    _save_json_atomic(
        manifest_path,
        manifest_payload(
            args=args,
            graphs=selected_graphs,
            engines=selected_engines,
            total_scope=total_scope,
            completed_scope=completed_scope,
        ),
    )

    stochastic_engine_count = sum(engine_is_stochastic(name) for name in engine_names)
    print(
        "[benchmark] Starting: "
        f"{len(graph_names)} graphs x {len(engine_names)} engines, "
        f"{args.seeds} seeds for stochastic ({stochastic_engine_count} stochastic engines)"
    )
    if args.resume and completed_scope:
        print(f"[benchmark] Resume: {completed_scope}/{total_scope} already complete")
    for record in immediate_records:
        seed_suffix = "" if record.seed is None else f" seed={record.seed}"
        print(
            f"[benchmark] SKIP: {record.engine_name} on {record.graph_name}{seed_suffix} "
            f"({record.detail()})"
        )

    if not work_items:
        current_records = list(scoped_results(results, scoped_keys).values())
        generate_summary(output_dir, current_records, engine_names, graph_names)
        counts = status_counts(current_records)
        print(
            "[benchmark] Done: "
            f"{total_scope} total, {counts.get('ok', 0)} ok, {counts.get('skipped', 0)} skipped, "
            f"{counts.get('error', 0)} errors, {counts.get('timeout', 0)} timeouts"
        )
        return 0

    # ── Serial execution (--workers 1) or parallel ─────────────────────────
    def _process_record(record: BenchmarkRecord) -> None:
        """Handle bookkeeping after one work item completes."""
        nonlocal completed_scope
        results[record.key] = record
        completed_scope += 1
        graph_completion_counts[record.graph_name] = (
            graph_completion_counts.get(record.graph_name, 0) + 1
        )

        # Throttle disk saves -- every SAVE_INTERVAL completions instead of every record
        if completed_scope % SAVE_INTERVAL == 0 or completed_scope == total_scope:
            save_results(results_path, results)

        if record.status in {"error", "timeout"}:
            seed_suffix = "" if record.seed is None else f" seed={record.seed}"
            print(
                f"[benchmark] {record.status.upper()}: {record.engine_name} on "
                f"{record.graph_name}{seed_suffix} ({record.detail()})"
            )

        if completed_scope % PROGRESS_INTERVAL == 0 or completed_scope == total_scope:
            print(progress_line("complete", completed_scope, total_scope, record))

        if graph_completion_counts.get(record.graph_name, 0) == graph_completion_targets.get(
            record.graph_name, 0
        ):
            print(f"[benchmark] Graph complete: {record.graph_name}")

    work_groups = group_work_items(work_items)

    def _mark_group_running(work_group: Sequence[WorkItem]) -> None:
        """Persist running placeholders for one grouped submission."""
        for work_item in work_group:
            summary_record = graph_summaries[work_item.graph_name]
            results[work_item.key] = running_record(
                graph_name=work_item.graph_name,
                engine_name=work_item.engine_name,
                seed=work_item.seed,
                num_nodes=summary_record.num_nodes,
                num_edges=summary_record.num_edges,
            )

    # ── SIGINT handler: stop accepting new work, drain inflight, save. ──
    _shutdown_requested = False

    def _sigint_handler(signum: int, frame: Any) -> None:
        nonlocal _shutdown_requested
        if _shutdown_requested:
            # Second SIGINT -- hard exit after saving.
            print("\n[benchmark] Forced shutdown -- saving results...")
            save_results(results_path, results)
            sys.exit(1)
        _shutdown_requested = True
        print(
            "\n[benchmark] Graceful shutdown requested (SIGINT). "
            "Finishing inflight work, then saving..."
        )

    previous_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _sigint_handler)

    def _collect_future(
        future: Future[list[dict[str, Any]]],
        work_group: Sequence[WorkItem],
    ) -> None:
        """Collect results from one completed future."""
        try:
            payloads = future.result()
        except Exception as exc:
            payloads = [
                _record_with_pairings(
                    graph_name=wi.graph_name,
                    engine_name=wi.engine_name,
                    seed=wi.seed,
                    num_nodes=graph_summaries[wi.graph_name].num_nodes,
                    num_edges=graph_summaries[wi.graph_name].num_edges,
                    status="error",
                    runtime_seconds=None,
                    error=f"{type(exc).__name__}: {exc}",
                    positions_file=None,
                    skip_reason=None,
                ).to_dict()
                for wi in work_group
            ]
        for payload in payloads:
            _process_record(BenchmarkRecord.from_dict(payload))

    try:
        if args.resolved_workers <= 1:
            for work_group in work_groups:
                if _shutdown_requested:
                    break
                _mark_group_running(work_group)
                for payload in _run_work_group(work_group):
                    _process_record(BenchmarkRecord.from_dict(payload))
        else:
            # Treat all groups as light for parallel execution when graphs are
            # small enough (max_nodes <= 1000).  The "heavy" designation was for
            # GPU-memory safety on huge graphs, but CPU-only engines on small
            # graphs are safe to parallelize.
            if args.max_nodes and args.max_nodes <= 1000:
                light_groups = list(work_groups)
                heavy_groups = []
            else:
                light_groups = [g for g in work_groups if not engine_is_heavy(g[0].engine_name)]
                heavy_groups = [g for g in work_groups if engine_is_heavy(g[0].engine_name)]

            try:
                executor = ProcessPoolExecutor(
                    max_workers=args.resolved_workers, mp_context=_MP_CONTEXT
                )
            except PermissionError:
                print("[benchmark] ERROR: failed to start ProcessPoolExecutor")
                return 1

            print(
                f"[benchmark] Parallel: {len(light_groups)} light groups "
                f"(rolling window={MAX_INFLIGHT_GROUPS}), "
                f"{len(heavy_groups)} heavy groups (serial)"
            )

            # Rolling submission: keep up to MAX_INFLIGHT_GROUPS futures alive
            # at all times.  As each completes, immediately submit the next
            # group.  This eliminates idle gaps between batches.
            inflight: dict[Future[list[dict[str, Any]]], Sequence[WorkItem]] = {}
            group_iter = iter(light_groups)

            def _fill_inflight() -> None:
                """Submit work groups until the inflight window is full."""
                while len(inflight) < MAX_INFLIGHT_GROUPS and not _shutdown_requested:
                    try:
                        work_group = next(group_iter)
                    except StopIteration:
                        break
                    _mark_group_running(work_group)
                    fut = executor.submit(_run_work_group, work_group)
                    inflight[fut] = work_group

            try:
                _fill_inflight()
                while inflight and not _shutdown_requested:
                    # Wait for any one future, with watchdog timeout.
                    # If no future completes within WATCHDOG_TIMEOUT, a worker
                    # likely died (C-level crash in igraph/DRL/etc.).  Record
                    # the stuck futures as errors and rebuild the executor.
                    got_result = False
                    try:
                        for fut in as_completed(inflight, timeout=WATCHDOG_TIMEOUT):
                            _collect_future(fut, inflight.pop(fut))
                            _fill_inflight()
                            got_result = True
                            if _shutdown_requested:
                                break
                            break  # Process one at a time to stay responsive.
                    except TimeoutError:
                        pass  # Watchdog fired -- handled below.
                    if not got_result and not _shutdown_requested:
                        # Watchdog fired -- worker pool is stuck.
                        print(
                            f"[benchmark] WATCHDOG: no future completed in "
                            f"{WATCHDOG_TIMEOUT:.0f}s, recycling executor "
                            f"({len(inflight)} stuck futures)"
                        )
                        save_results(results_path, results)
                        # Record stuck futures as errors.
                        for stuck_fut, stuck_group in list(inflight.items()):
                            for wi in stuck_group:
                                record = _record_with_pairings(
                                    graph_name=wi.graph_name,
                                    engine_name=wi.engine_name,
                                    seed=wi.seed,
                                    num_nodes=graph_summaries[wi.graph_name].num_nodes,
                                    num_edges=graph_summaries[wi.graph_name].num_edges,
                                    status="error",
                                    runtime_seconds=None,
                                    error="watchdog: worker pool stuck",
                                    positions_file=None,
                                    skip_reason=None,
                                )
                                _process_record(record)
                        inflight.clear()
                        # Kill and rebuild the executor.
                        executor.shutdown(wait=False, cancel_futures=True)
                        executor = ProcessPoolExecutor(
                            max_workers=args.resolved_workers,
                            mp_context=_MP_CONTEXT,
                        )
                        _fill_inflight()
                    # If shutdown requested, drain remaining inflight futures.
                    if _shutdown_requested:
                        for fut in as_completed(inflight):
                            _collect_future(fut, inflight.pop(fut))
            finally:
                executor.shutdown(wait=True, cancel_futures=False)

            # Heavy groups -- run in parallel too (memory is abundant).
            # Reuse the same rolling-window + watchdog pattern.
            if heavy_groups and not _shutdown_requested:
                print(
                    f"[benchmark] Heavy phase: {len(heavy_groups)} groups "
                    f"(parallel, rolling window={MAX_INFLIGHT_GROUPS})"
                )
                executor = ProcessPoolExecutor(
                    max_workers=args.resolved_workers,
                    mp_context=_MP_CONTEXT,
                )
                group_iter = iter(heavy_groups)
                try:
                    _fill_inflight()
                    while inflight and not _shutdown_requested:
                        got_result = False
                        try:
                            for fut in as_completed(inflight, timeout=WATCHDOG_TIMEOUT):
                                _collect_future(fut, inflight.pop(fut))
                                _fill_inflight()
                                got_result = True
                                if _shutdown_requested:
                                    break
                                break
                        except TimeoutError:
                            pass
                        if not got_result and not _shutdown_requested:
                            print(
                                f"[benchmark] WATCHDOG: recycling executor "
                                f"({len(inflight)} stuck futures)"
                            )
                            save_results(results_path, results)
                            for stuck_fut, stuck_group in list(inflight.items()):
                                for wi in stuck_group:
                                    record = _record_with_pairings(
                                        graph_name=wi.graph_name,
                                        engine_name=wi.engine_name,
                                        seed=wi.seed,
                                        num_nodes=graph_summaries[wi.graph_name].num_nodes,
                                        num_edges=graph_summaries[wi.graph_name].num_edges,
                                        status="error",
                                        runtime_seconds=None,
                                        error="watchdog: worker pool stuck",
                                        positions_file=None,
                                        skip_reason=None,
                                    )
                                    _process_record(record)
                            inflight.clear()
                            executor.shutdown(wait=False, cancel_futures=True)
                            executor = ProcessPoolExecutor(
                                max_workers=args.resolved_workers,
                                mp_context=_MP_CONTEXT,
                            )
                            _fill_inflight()
                        if _shutdown_requested:
                            for fut in as_completed(inflight):
                                _collect_future(fut, inflight.pop(fut))
                finally:
                    executor.shutdown(wait=True, cancel_futures=False)

    finally:
        # Always save results on exit -- whether normal, SIGINT, or exception.
        save_results(results_path, results)
        signal.signal(signal.SIGINT, previous_sigint)
        if _shutdown_requested:
            print(
                f"[benchmark] Saved {completed_scope}/{total_scope} results "
                f"after graceful shutdown."
            )

    current_records = list(scoped_results(results, scoped_keys).values())
    _save_json_atomic(
        manifest_path,
        manifest_payload(
            args=args,
            graphs=selected_graphs,
            engines=selected_engines,
            total_scope=total_scope,
            completed_scope=len(current_records),
        ),
    )
    generate_summary(output_dir, current_records, engine_names, graph_names)

    counts = status_counts(current_records)
    print(
        "[benchmark] Done: "
        f"{total_scope} total, {counts.get('ok', 0)} ok, {counts.get('skipped', 0)} skipped, "
        f"{counts.get('error', 0)} errors, {counts.get('timeout', 0)} timeouts"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
