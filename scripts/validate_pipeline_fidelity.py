"""Validate classic-to-pipeline layout fidelity across all benchmark variants.

This script compares the classic reimplementation callables in
``dagua.layout.classic`` against their pipeline translations in
``dagua.layout.ops.pipelines`` for every supported variant in
``dagua.eval.variants.VARIANT_REGISTRY``.

The validation contract is intentionally strict:

- cover all 23 requested base engines
- cover all 105 benchmark graphs from ``get_test_graphs()``
- cover seeds ``[42, 99, 7]`` for stochastic variants and ``[42]`` otherwise
- compare output tensors with ``torch.equal()``
- record mismatches and execution errors without aborting the run
"""

from __future__ import annotations

import csv
import importlib
import inspect
import math
import os
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

import torch

from dagua.eval.competitors.classic_competitor import _CLASSIC_LAYOUT_SPECS
from dagua.eval.graphs import TestGraph, get_test_graphs
from dagua.eval.variants import VARIANT_REGISTRY, AlgorithmVariant

EXPECTED_GRAPH_COUNT = 105
PROGRESS_INTERVAL = 100
OUTPUT_PATH = Path("eval_output/pipeline_fidelity.csv")
STOCHASTIC_SEEDS = (42, 99, 7)
DETERMINISTIC_SEEDS = (42,)
SKIPPED_BASE_ENGINES = {
    "classic_fr_kk",
    "classic_kk_fr",
    "cytoscape_fcose",
    "gephi_yifanhu",
}
TARGET_BASE_ENGINES = (
    "classic_fr",
    "classic_kk",
    "classic_fa2",
    "classic_graphopt",
    "classic_stress_sgd",
    "classic_spectral",
    "classic_classical_mds",
    "classic_stress_maj",
    "classic_sugiyama",
    "classic_tsnet",
    "classic_gem",
    "classic_fmmm",
    "classic_maxent_stress",
    "classic_davidson_harel",
    "classic_linlog",
    "classic_pivot_mds",
    "classic_rt",
    "classic_drl",
    "classic_lgl",
    "classic_sfdp",
    "classic_umap",
    "classic_neulay",
    "classic_sgd2_multi",
)
FAILURE_DETAIL_LIMIT = 50
DEFAULT_MAX_WORKERS = 1  # single-threaded to avoid memory spikes

LayoutCallable = Callable[..., Any]
_WORKER_ALGORITHM_MAP: dict[str, AlgorithmPair] = {}
_WORKER_GRAPHS: list[TestGraph] = []
_WORKER_VARIANTS: list[AlgorithmVariant] = []


@dataclass(frozen=True)
class AlgorithmPair:
    """Resolved classic and pipeline callables for one base engine.

    Parameters
    ----------
    base_engine : str
        Canonical base engine name from ``VARIANT_REGISTRY``.
    classic_fn : Callable[..., Any]
        Classic layout function.
    pipeline_fn : Callable[..., Any]
        Pipeline layout function.
    default_params : dict[str, Any]
        Benchmark default parameters merged with each variant's
        ``reimpl_params`` to mirror the classic competitor adapter behavior.
    classic_param_names : frozenset[str]
        Accepted classic keyword names from ``inspect.signature``.
    pipeline_param_names : frozenset[str]
        Accepted pipeline keyword names from ``inspect.signature``.
    """

    base_engine: str
    classic_fn: LayoutCallable
    pipeline_fn: LayoutCallable
    default_params: Dict[str, Any]
    classic_param_names: frozenset[str]
    pipeline_param_names: frozenset[str]


@dataclass
class ComparisonRow:
    """Recorded result for one variant/graph/seed comparison.

    Parameters
    ----------
    algorithm : str
        Base engine name.
    variant_id : str
        Concrete variant identifier from ``VARIANT_REGISTRY``.
    graph_name : str
        Test-graph name.
    seed : int
        Seed used for this comparison.
    match : bool
        Whether the normalized output tensors matched exactly.
    max_abs_diff : float | None
        Maximum absolute tensor difference when both calls returned tensors
        with the same shape. ``None`` for execution errors or shape mismatch.
    error : str | None
        Execution or normalization error message, when present.
    """

    algorithm: str
    variant_id: str
    graph_name: str
    seed: int
    match: bool
    max_abs_diff: Optional[float]
    error: Optional[str]


@dataclass(frozen=True)
class ComparisonTask:
    """One worker task covering a variant and graph across its required seeds.

    Parameters
    ----------
    variant_index : int
        Index into the shared variant list.
    graph_index : int
        Index into the shared graph list.
    """

    variant_index: int
    graph_index: int


@dataclass
class AlgorithmSummary:
    """Aggregate counts for one base engine.

    Parameters
    ----------
    variants : set[str]
        Seen variant identifiers for the engine.
    total : int
        Total comparisons attempted.
    matches : int
        Exact matches.
    mismatches : int
        Non-error output mismatches.
    errors : int
        Exceptions or invalid return values.
    """

    variants: set[str]
    total: int = 0
    matches: int = 0
    mismatches: int = 0
    errors: int = 0


def _load_callable(import_path: str, function_name: str) -> LayoutCallable:
    """Import and return one named callable.

    Parameters
    ----------
    import_path : str
        Module path to import.
    function_name : str
        Callable attribute name inside ``import_path``.

    Returns
    -------
    Callable[..., Any]
        Imported callable.

    Raises
    ------
    AttributeError
        Raised when the module does not expose ``function_name``.
    """
    module = importlib.import_module(import_path)
    return getattr(module, function_name)


def _load_pipeline_callable(import_path: str) -> LayoutCallable:
    """Resolve the unique pipeline translation for one classic module.

    Parameters
    ----------
    import_path : str
        Classic module path such as ``dagua.layout.classic.fr``.

    Returns
    -------
    Callable[..., Any]
        Pipeline callable whose name matches ``layout_*_pipeline``.

    Raises
    ------
    RuntimeError
        Raised when zero or multiple pipeline layout functions are found.
    """
    pipeline_module_path = import_path.replace(
        "dagua.layout.classic",
        "dagua.layout.ops.pipelines",
    )
    module = importlib.import_module(pipeline_module_path)
    candidates = [
        getattr(module, name)
        for name in dir(module)
        if name.startswith("layout_") and name.endswith("_pipeline")
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one pipeline layout callable in {pipeline_module_path}, "
            f"found {len(candidates)}."
        )
    return candidates[0]


def build_algorithm_map() -> dict[str, AlgorithmPair]:
    """Build the requested base-engine map for classic and pipeline layouts.

    Returns
    -------
    dict[str, AlgorithmPair]
        Mapping from base engine name to resolved callables and metadata.

    Raises
    ------
    RuntimeError
        Raised when the resolved engine set does not match the required
        23-engine target list.
    """
    algorithm_map: dict[str, AlgorithmPair] = {}
    for base_engine in TARGET_BASE_ENGINES:
        spec = _CLASSIC_LAYOUT_SPECS[base_engine]
        classic_fn = _load_callable(spec.import_path, spec.function_name)
        pipeline_fn = _load_pipeline_callable(spec.import_path)
        classic_param_names = frozenset(inspect.signature(classic_fn).parameters)
        pipeline_param_names = frozenset(inspect.signature(pipeline_fn).parameters)
        algorithm_map[base_engine] = AlgorithmPair(
            base_engine=base_engine,
            classic_fn=classic_fn,
            pipeline_fn=pipeline_fn,
            default_params=dict(spec.default_params),
            classic_param_names=classic_param_names,
            pipeline_param_names=pipeline_param_names,
        )

    if set(algorithm_map) != set(TARGET_BASE_ENGINES):
        raise RuntimeError(
            "Algorithm map did not resolve the expected 23 base engines. "
            f"Expected {sorted(TARGET_BASE_ENGINES)}, got {sorted(algorithm_map)}."
        )
    return algorithm_map


def iter_target_variants(algorithm_map: Mapping[str, AlgorithmPair]) -> list[AlgorithmVariant]:
    """Return all registry variants covered by the fidelity run.

    Parameters
    ----------
    algorithm_map : Mapping[str, AlgorithmPair]
        Resolved algorithm map keyed by base engine.

    Returns
    -------
    list[AlgorithmVariant]
        Registry entries whose base engines are in ``algorithm_map``.
    """
    return [variant for variant in VARIANT_REGISTRY if variant.base_engine in algorithm_map]


def get_validation_graphs(max_nodes: int = 0) -> list[TestGraph]:
    """Load and optionally filter the evaluation graph corpus.

    Parameters
    ----------
    max_nodes : int, default=0
        When positive, skip graphs with more than this many nodes.
        When zero or negative, include all graphs.

    Returns
    -------
    list[TestGraph]
        Test graphs from ``get_test_graphs()``, optionally filtered.
    """
    graphs = get_test_graphs()
    if max_nodes > 0:
        graphs = [g for g in graphs if g.graph.num_nodes <= max_nodes]
        print(f"Filtered to {len(graphs)} graphs with <= {max_nodes} nodes")
    else:
        if len(graphs) != EXPECTED_GRAPH_COUNT:
            raise RuntimeError(f"Expected {EXPECTED_GRAPH_COUNT} test graphs, found {len(graphs)}.")
    return graphs


def comparison_seeds(variant: AlgorithmVariant) -> Sequence[int]:
    """Return the seed sequence for one variant.

    Parameters
    ----------
    variant : AlgorithmVariant
        Variant metadata entry.

    Returns
    -------
    Sequence[int]
        Three seeds for stochastic variants and one seed otherwise.
    """
    if variant.is_stochastic:
        return STOCHASTIC_SEEDS
    return DETERMINISTIC_SEEDS


def clone_optional_tensor(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Clone one optional tensor input for layout isolation.

    Parameters
    ----------
    tensor : torch.Tensor | None
        Tensor to clone.

    Returns
    -------
    torch.Tensor | None
        Detached clone when ``tensor`` is present, else ``None``.
    """
    if tensor is None:
        return None
    return tensor.detach().clone()


def build_call_kwargs(
    test_graph: TestGraph,
    seed: int,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Build raw call kwargs shared by classic and pipeline functions.

    Parameters
    ----------
    test_graph : TestGraph
        Graph case under test.
    seed : int
        Seed for the current comparison.
    params : Mapping[str, Any]
        Variant parameters already merged with benchmark defaults.

    Returns
    -------
    dict[str, Any]
        Raw keyword arguments before signature filtering.
    """
    graph = test_graph.graph
    call_kwargs: dict[str, Any] = {
        "edge_index": graph.edge_index.detach().clone(),
        "num_nodes": graph.num_nodes,
        "node_sizes": clone_optional_tensor(graph.node_sizes),
        "seed": seed,
        "edge_weights": clone_optional_tensor(graph.edge_weights),
    }
    call_kwargs.update(dict(params))
    return call_kwargs


def filter_kwargs(
    kwargs: Mapping[str, Any],
    accepted_param_names: frozenset[str],
) -> dict[str, Any]:
    """Drop unsupported keyword arguments for one callable.

    Parameters
    ----------
    kwargs : Mapping[str, Any]
        Candidate keyword arguments.
    accepted_param_names : frozenset[str]
        Names accepted by the callable signature.

    Returns
    -------
    dict[str, Any]
        Filtered kwargs accepted by the target callable.
    """
    return {name: value for name, value in kwargs.items() if name in accepted_param_names}


def normalize_layout_output(result: Any) -> torch.Tensor:
    """Extract the position tensor from one layout result.

    Parameters
    ----------
    result : Any
        Raw layout return value.

    Returns
    -------
    torch.Tensor
        Position tensor from the layout result.

    Raises
    ------
    TypeError
        Raised when the result is not a tensor or a tuple whose first element
        is a tensor.
    """
    if isinstance(result, torch.Tensor):
        return result
    if isinstance(result, tuple) and result and isinstance(result[0], torch.Tensor):
        return result[0]
    raise TypeError(f"Unsupported layout return value: {type(result)!r}")


def tensor_max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> Optional[float]:
    """Return the maximum absolute difference between two tensors.

    Parameters
    ----------
    left : torch.Tensor
        First tensor.
    right : torch.Tensor
        Second tensor.

    Returns
    -------
    float | None
        Maximum absolute difference, or ``None`` when the shapes differ.
    """
    if left.shape != right.shape:
        return None
    if left.numel() == 0:
        return 0.0
    diff = (left.to(dtype=torch.float64) - right.to(dtype=torch.float64)).abs().max().item()
    return float(diff)


def compare_variant_graph_seed(
    pair: AlgorithmPair,
    variant: AlgorithmVariant,
    test_graph: TestGraph,
    seed: int,
) -> ComparisonRow:
    """Run and compare one classic/pipeline pair for a single graph and seed.

    Parameters
    ----------
    pair : AlgorithmPair
        Resolved classic/pipeline callables for the base engine.
    variant : AlgorithmVariant
        Variant metadata entry.
    test_graph : TestGraph
        Graph case under test.
    seed : int
        Seed for the current comparison.

    Returns
    -------
    ComparisonRow
        Recorded outcome for CSV output and summary reporting.
    """
    merged_params = dict(pair.default_params)
    merged_params.update(dict(variant.reimpl_params))
    raw_kwargs = build_call_kwargs(test_graph=test_graph, seed=seed, params=merged_params)
    classic_kwargs = filter_kwargs(raw_kwargs, pair.classic_param_names)
    pipeline_kwargs = filter_kwargs(raw_kwargs, pair.pipeline_param_names)

    classic_exc = None
    pipeline_exc = None
    classic_raw = None
    pipeline_raw = None

    try:
        classic_raw = pair.classic_fn(**classic_kwargs)
    except Exception as exc:  # noqa: BLE001
        classic_exc = exc

    try:
        pipeline_raw = pair.pipeline_fn(**pipeline_kwargs)
    except Exception as exc:  # noqa: BLE001
        pipeline_exc = exc

    # Both errored with same type+message = identical behavior = pass
    if classic_exc is not None and pipeline_exc is not None:
        same_error = type(classic_exc).__name__ == type(pipeline_exc).__name__ and str(
            classic_exc
        ) == str(pipeline_exc)
        return ComparisonRow(
            algorithm=pair.base_engine,
            variant_id=variant.variant_id,
            graph_name=test_graph.name,
            seed=seed,
            match=same_error,
            max_abs_diff=0.0 if same_error else None,
            error=None if same_error else f"classic: {classic_exc} | pipeline: {pipeline_exc}",
        )

    # Only one errored = divergent behavior
    if classic_exc is not None or pipeline_exc is not None:
        which = "classic" if classic_exc is not None else "pipeline"
        exc = classic_exc or pipeline_exc
        return ComparisonRow(
            algorithm=pair.base_engine,
            variant_id=variant.variant_id,
            graph_name=test_graph.name,
            seed=seed,
            match=False,
            max_abs_diff=None,
            error=f"{which} only: {type(exc).__name__}: {exc}",
        )

    # Both succeeded -- compare outputs
    try:
        classic_pos = normalize_layout_output(classic_raw).detach().cpu()
        pipeline_pos = normalize_layout_output(pipeline_raw).detach().cpu()
        match = torch.equal(classic_pos, pipeline_pos)
        max_abs_diff = 0.0 if match else tensor_max_abs_diff(classic_pos, pipeline_pos)
        return ComparisonRow(
            algorithm=pair.base_engine,
            variant_id=variant.variant_id,
            graph_name=test_graph.name,
            seed=seed,
            match=match,
            max_abs_diff=max_abs_diff,
            error=None,
        )
    except Exception as exc:  # noqa: BLE001
        return ComparisonRow(
            algorithm=pair.base_engine,
            variant_id=variant.variant_id,
            graph_name=test_graph.name,
            seed=seed,
            match=False,
            max_abs_diff=None,
            error=f"comparison: {type(exc).__name__}: {exc}",
        )


def update_summary(summary: AlgorithmSummary, row: ComparisonRow) -> None:
    """Fold one comparison row into the per-algorithm summary.

    Parameters
    ----------
    summary : AlgorithmSummary
        Mutable summary record to update.
    row : ComparisonRow
        Completed comparison outcome.

    Returns
    -------
    None
        Updates ``summary`` in place.
    """
    summary.variants.add(row.variant_id)
    summary.total += 1
    if row.error is not None:
        summary.errors += 1
    elif row.match:
        summary.matches += 1
    else:
        summary.mismatches += 1


def csv_row(row: ComparisonRow) -> dict[str, Any]:
    """Convert one comparison row into the CSV schema.

    Parameters
    ----------
    row : ComparisonRow
        Comparison outcome to serialize.

    Returns
    -------
    dict[str, Any]
        Row dictionary matching the required CSV columns.
    """
    return {
        "algorithm": row.algorithm,
        "variant_id": row.variant_id,
        "graph_name": row.graph_name,
        "seed": row.seed,
        "match": row.match,
        "max_abs_diff": "" if row.max_abs_diff is None else row.max_abs_diff,
        "error": "" if row.error is None else row.error,
    }


def print_progress(completed: int, total: int, start_time: float) -> None:
    """Print a periodic progress update.

    Parameters
    ----------
    completed : int
        Completed comparison count.
    total : int
        Planned comparison count.
    start_time : float
        Wall-clock start time from ``time.perf_counter()``.

    Returns
    -------
    None
        Prints one progress line to stdout.
    """
    elapsed = time.perf_counter() - start_time
    print(
        f"[progress] {completed}/{total} comparisons completed ({elapsed:.1f}s elapsed)",
        flush=True,
    )


def total_comparison_count(variants: Iterable[AlgorithmVariant], graph_count: int) -> int:
    """Return the total planned comparison count.

    Parameters
    ----------
    variants : Iterable[AlgorithmVariant]
        Variants covered by the validation run.
    graph_count : int
        Number of test graphs.

    Returns
    -------
    int
        Total graph/seed comparisons.
    """
    total = 0
    for variant in variants:
        total += graph_count * len(comparison_seeds(variant))
    return total


def format_rate(numerator: int, denominator: int) -> str:
    """Format one ratio as a percentage string.

    Parameters
    ----------
    numerator : int
        Numerator count.
    denominator : int
        Denominator count.

    Returns
    -------
    str
        Percentage string with one decimal place.
    """
    if denominator == 0:
        return "0.0%"
    return f"{(100.0 * numerator) / denominator:.1f}%"


def print_summary_table(summaries: Mapping[str, AlgorithmSummary]) -> None:
    """Print the per-algorithm result table.

    Parameters
    ----------
    summaries : Mapping[str, AlgorithmSummary]
        Summary records keyed by base engine.

    Returns
    -------
    None
        Prints the summary table to stdout.
    """
    header = (
        f"{'algorithm':<24} {'variants':>8} {'total':>8} {'match':>8} "
        f"{'mismatch':>10} {'error':>8} {'match_rate':>11}"
    )
    print(header)
    print("-" * len(header))
    for algorithm in TARGET_BASE_ENGINES:
        summary = summaries[algorithm]
        print(
            f"{algorithm:<24} {len(summary.variants):>8} {summary.total:>8} "
            f"{summary.matches:>8} {summary.mismatches:>10} {summary.errors:>8} "
            f"{format_rate(summary.matches, summary.total):>11}"
        )


def print_failure_details(rows: Sequence[ComparisonRow]) -> None:
    """Print mismatch and error details.

    Parameters
    ----------
    rows : Sequence[ComparisonRow]
        Failed comparison rows.

    Returns
    -------
    None
        Prints up to ``FAILURE_DETAIL_LIMIT`` detailed lines.
    """
    if not rows:
        print("Failure details: none")
        return

    print("Failure details:")
    for row in rows[:FAILURE_DETAIL_LIMIT]:
        if row.error is not None:
            print(
                f"  ERROR {row.algorithm} {row.variant_id} {row.graph_name} "
                f"seed={row.seed}: {row.error}"
            )
            continue
        diff_text = (
            "nan"
            if row.max_abs_diff is None or math.isnan(row.max_abs_diff)
            else (f"{row.max_abs_diff:.8g}")
        )
        print(
            f"  MISMATCH {row.algorithm} {row.variant_id} {row.graph_name} "
            f"seed={row.seed}: max_abs_diff={diff_text}"
        )
    remaining = len(rows) - FAILURE_DETAIL_LIMIT
    if remaining > 0:
        print(f"  ... {remaining} additional failures omitted")


def set_worker_state(
    algorithm_map: Mapping[str, AlgorithmPair],
    variants: Sequence[AlgorithmVariant],
    graphs: Sequence[TestGraph],
) -> None:
    """Publish shared validation state for forked workers.

    Parameters
    ----------
    algorithm_map : Mapping[str, AlgorithmPair]
        Resolved classic and pipeline callables.
    variants : Sequence[AlgorithmVariant]
        Covered registry variants.
    graphs : Sequence[TestGraph]
        Full test-graph corpus.

    Returns
    -------
    None
        Updates module-level worker state.
    """
    global _WORKER_ALGORITHM_MAP
    global _WORKER_GRAPHS
    global _WORKER_VARIANTS

    _WORKER_ALGORITHM_MAP = dict(algorithm_map)
    _WORKER_VARIANTS = list(variants)
    _WORKER_GRAPHS = list(graphs)


def init_worker() -> None:
    """Configure worker execution for predictable CPU utilization.

    Returns
    -------
    None
        Restricts intra-op threading so multiple thread-pool workers can
        coexist without severe oversubscription.
    """
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def compare_task(task: ComparisonTask) -> list[ComparisonRow]:
    """Run all required seeds for one variant and graph in a worker.

    Parameters
    ----------
    task : ComparisonTask
        Variant/graph work item.

    Returns
    -------
    list[ComparisonRow]
        Comparison rows for the graph across that variant's configured seeds.
    """
    variant = _WORKER_VARIANTS[task.variant_index]
    pair = _WORKER_ALGORITHM_MAP[variant.base_engine]
    test_graph = _WORKER_GRAPHS[task.graph_index]
    return [
        compare_variant_graph_seed(
            pair=pair,
            variant=variant,
            test_graph=test_graph,
            seed=seed,
        )
        for seed in comparison_seeds(variant)
    ]


def iter_tasks(
    variants: Sequence[AlgorithmVariant],
    graphs: Sequence[TestGraph],
) -> Iterable[ComparisonTask]:
    """Yield worker tasks for the full variant/graph matrix.

    Parameters
    ----------
    variants : Sequence[AlgorithmVariant]
        Covered registry variants.
    graphs : Sequence[TestGraph]
        Full test-graph corpus.

    Returns
    -------
    Iterable[ComparisonTask]
        Worker tasks spanning every variant and every graph.
    """
    for variant_index, _variant in enumerate(variants):
        for graph_index, _graph in enumerate(graphs):
            yield ComparisonTask(variant_index=variant_index, graph_index=graph_index)


def requested_worker_count() -> int:
    """Return the worker count for this run.

    Returns
    -------
    int
        Worker count from ``DAGUA_PIPELINE_FIDELITY_WORKERS`` when set, else
        the conservative default cap.
    """
    raw_value = os.environ.get("DAGUA_PIPELINE_FIDELITY_WORKERS")
    if raw_value is None:
        return DEFAULT_MAX_WORKERS
    try:
        parsed = int(raw_value)
    except ValueError as exc:
        raise RuntimeError("DAGUA_PIPELINE_FIDELITY_WORKERS must be an integer when set.") from exc
    if parsed < 1:
        raise RuntimeError("DAGUA_PIPELINE_FIDELITY_WORKERS must be >= 1.")
    return parsed


def iter_threaded_task_rows(
    tasks: Iterable[ComparisonTask],
    worker_count: int,
) -> Iterable[list[ComparisonRow]]:
    """Yield completed task rows from a bounded thread pool.

    Parameters
    ----------
    tasks : Iterable[ComparisonTask]
        Task stream for the full validation matrix.
    worker_count : int
        Maximum number of concurrent worker threads.

    Returns
    -------
    Iterable[list[ComparisonRow]]
        Completed task outputs in completion order.
    """
    task_iter = iter(tasks)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        pending: dict[Future[list[ComparisonRow]], ComparisonTask] = {}

        for _ in range(worker_count * 2):
            try:
                task = next(task_iter)
            except StopIteration:
                break
            pending[executor.submit(compare_task, task)] = task

        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                del pending[future]
                yield future.result()
                try:
                    task = next(task_iter)
                except StopIteration:
                    continue
                pending[executor.submit(compare_task, task)] = task


def _run_single_engine(base_engine: str, max_nodes: int, csv_path: str) -> str:
    """Run validation for one base engine in a subprocess-safe function.

    Parameters
    ----------
    base_engine : str
        Algorithm base engine name.
    max_nodes : int
        Graph node-count filter.
    csv_path : str
        Path to write per-engine CSV fragment.

    Returns
    -------
    str
        JSON-encoded summary: matches, mismatches, errors, rows, failed details.
    """
    import gc
    import json

    algorithm_map = build_algorithm_map()
    all_variants = iter_target_variants(algorithm_map)
    engine_variants = [v for v in all_variants if v.base_engine == base_engine]
    graphs = get_validation_graphs(max_nodes=max_nodes)

    set_worker_state(algorithm_map=algorithm_map, variants=engine_variants, graphs=graphs)
    init_worker()

    matches = 0
    mismatches = 0
    errors = 0
    failed: list[dict] = []

    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "algorithm",
                "variant_id",
                "graph_name",
                "seed",
                "match",
                "max_abs_diff",
                "error",
            ],
        )
        completed = 0
        for task_rows in iter_threaded_task_rows(
            tasks=iter_tasks(engine_variants, graphs),
            worker_count=1,
        ):
            for row in task_rows:
                writer.writerow(csv_row(row))
                if row.error is not None:
                    errors += 1
                    failed.append(
                        {
                            "v": row.variant_id,
                            "g": row.graph_name,
                            "s": row.seed,
                            "e": str(row.error),
                        }
                    )
                elif not row.match:
                    mismatches += 1
                    failed.append(
                        {
                            "v": row.variant_id,
                            "g": row.graph_name,
                            "s": row.seed,
                            "d": str(row.max_abs_diff),
                        }
                    )
                else:
                    matches += 1
                completed += 1
                if completed % 100 == 0:
                    gc.collect()
                    handle.flush()
        handle.flush()

    return json.dumps(
        {
            "engine": base_engine,
            "matches": matches,
            "mismatches": mismatches,
            "errors": errors,
            "failed": failed,
        }
    )


def run_validation(max_nodes: int = 0) -> int:
    """Execute the full pipeline fidelity validation with subprocess isolation.

    Each base engine runs in its own subprocess so memory is fully reclaimed
    between algorithms. Results are merged into a single CSV.

    Parameters
    ----------
    max_nodes : int
        When positive, skip graphs with more than this many nodes.

    Returns
    -------
    int
        Exit code: 0 = all match, 1 = any mismatch or error.
    """
    import json
    import subprocess
    import sys
    import textwrap

    algorithm_map = build_algorithm_map()
    engines = sorted(algorithm_map.keys())
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_matches = 0
    total_mismatches = 0
    total_errors = 0
    all_failed: list[dict] = []
    start_time = time.perf_counter()

    # Write CSV header
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "algorithm",
                "variant_id",
                "graph_name",
                "seed",
                "match",
                "max_abs_diff",
                "error",
            ],
        )
        writer.writeheader()

    for i, engine in enumerate(engines):
        tmp_csv = str(OUTPUT_PATH) + f".{engine}.tmp"
        print(f"[{i + 1}/{len(engines)}] {engine} ...", flush=True)

        # Run in subprocess for memory isolation
        code = textwrap.dedent(f"""\
            import sys
            sys.path.insert(0, '.')
            from scripts.validate_pipeline_fidelity import _run_single_engine
            result = _run_single_engine(
                base_engine={engine!r},
                max_nodes={max_nodes!r},
                csv_path={tmp_csv!r},
            )
            print(result)
        """)
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=7200,
            )
        except subprocess.TimeoutExpired:
            print("  TIMEOUT (2h) -- skipping", flush=True)
            all_failed.append({"engine": engine, "e": "timeout after 2h"})
            if Path(tmp_csv).exists():
                Path(tmp_csv).unlink()
            continue
        if result.returncode != 0:
            print(f"  CRASH: {result.stderr[-500:]}", flush=True)
            total_errors += 1
            all_failed.append({"engine": engine, "e": "subprocess crash"})
            continue

        summary = json.loads(result.stdout.strip().split("\n")[-1])
        m, mm, e = summary["matches"], summary["mismatches"], summary["errors"]
        total_matches += m
        total_mismatches += mm
        total_errors += e
        all_failed.extend(summary["failed"])
        status = "PASS" if mm == 0 and e == 0 else "FAIL"
        print(f"  {status}: {m} match, {mm} mismatch, {e} error", flush=True)

        # Append tmp CSV to main CSV
        if Path(tmp_csv).exists():
            with (
                open(tmp_csv, encoding="utf-8") as src,
                OUTPUT_PATH.open("a", encoding="utf-8") as dst,
            ):
                dst.write(src.read())
            Path(tmp_csv).unlink()

    elapsed = time.perf_counter() - start_time
    overall = "PASS" if total_mismatches == 0 and total_errors == 0 else "FAIL"

    print()
    print(
        f"Overall: {overall} | matches={total_matches} mismatches={total_mismatches} "
        f"errors={total_errors} elapsed={elapsed:.1f}s"
    )
    if all_failed:
        print(f"\nFailures ({len(all_failed)}):")
        for f in all_failed[:20]:
            print(f"  {f}")

    return 0 if overall == "PASS" else 1


def main() -> int:
    """Run the validator and convert unexpected top-level failures to exit 1.

    Returns
    -------
    int
        Process exit code for the CLI.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline fidelity validation")
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=0,
        help="Skip graphs with more than N nodes (0 = no limit)",
    )
    args = parser.parse_args()

    try:
        return run_validation(max_nodes=args.max_nodes)
    except Exception:  # noqa: BLE001
        print("Top-level failure while validating pipeline fidelity:", flush=True)
        print(traceback.format_exc(), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
