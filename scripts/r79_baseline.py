"""Build the frozen r79 native-layout quality baseline."""

from __future__ import annotations

import argparse
import functools
import json
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch

from dagua.eval.competitors import get_competitor
from dagua.eval.competitors.base import CompetitorBase, CompetitorResult
from dagua.eval.graphs import TestGraph, get_test_graphs, is_semantically_directed
from dagua.eval.size_policy import set_size_aware_externals
from dagua.metrics import composite_auto, composite_large, evaluate

OUTPUT_DIR = Path("eval_output/r79_baseline")
SEED = 42
TIMEOUT_SECONDS = 120.0
TIE_BAND = 0.5

R79_NEW_GRAPH_NAMES = {
    "r79_weighted_community_4x18",
    "r79_weighted_mesh_10x12",
    "r79_weighted_skew_dag_6x10",
    "r79_weighted_hub_spoke_4x18",
    "r79_weighted_small_world_120",
    "r79_weighted_ladder_40",
    "r79_weighted_bipartite_16x24",
    "r79_nested_clusters_3x2x10",
    "r79_nested_clusters_2x3x12",
    "r79_nested_clusters_4x2x8",
    "r79_undirected_sbm_low_mix_4x25",
    "r79_undirected_sbm_mid_mix_5x20",
    "r79_undirected_sbm_high_mix_3x30",
    "r79_directed_scc_90_3cores",
    "r79_directed_scc_120_2cores",
}

ENGINE_NAMES = [
    "dagua",
    "graphviz_dot",
    "graphviz_sfdp",
    "graphviz_neato",
    "elk_layered",
    "dagre",
    "nx_spring",
    "igraph_kamada_kawai",
    "igraph_sugiyama",
]

EXTERNAL_ENGINE_NAMES = [name for name in ENGINE_NAMES if name != "dagua"]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : Optional[Sequence[str]], optional
        Argument vector to parse. ``None`` parses ``sys.argv[1:]`` (the CLI
        default); tests pass an explicit list.

    Returns
    -------
    argparse.Namespace
        Parsed CLI options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dagua-only",
        action="store_true",
        help="Rerun only Dagua rows against frozen external rows and positions.",
    )
    parser.add_argument(
        "--rescore-only",
        action="store_true",
        help="Recompute composites from stored metrics without rerunning layouts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Baseline output directory.",
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        action="store_true",
        help="Resume from results.rows.jsonl in the staging store.",
    )
    resume_group.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Provably-fresh sweep: clear/ignore any resumable staging store so no "
            "row is silently reused from a prior (possibly stale, pre-code-change) "
            "run. Mutually exclusive with --resume. Every row this run writes is "
            "stamped with the current git SHA and timestamp regardless of this flag."
        ),
    )
    parser.add_argument(
        "--graphs",
        nargs="+",
        default=None,
        help="Optional graph-name filter for targeted verification runs.",
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        default=None,
        help="Optional engine-name filter for targeted verification runs.",
    )
    parser.add_argument(
        "--size-blind-externals",
        action="store_true",
        help=(
            "Restore the old size-blind behavior for size-capable external "
            "adapters (graphviz dot/sfdp/neato, elk_layered, dagre): every node "
            "is laid out at a fixed placeholder size instead of its real "
            "label-measured size. Kept ONLY for store-compatibility experiments "
            "against pre-r80-P6 frozen data, which was produced size-blind "
            "throughout. Default is size-aware (the honest comparison)."
        ),
    )
    return parser.parse_args(argv)


@functools.lru_cache(maxsize=1)
def git_sha() -> str:
    """Return the current git commit SHA.

    Cached: the SHA cannot change mid-process, and this is called once per
    written row (see ``append_row``) so an uncached ``subprocess.run`` per
    row would be wasteful on large sweeps.

    Returns
    -------
    str
        Current commit SHA, or ``unknown`` if git cannot answer.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def safe_component(value: str) -> str:
    """Return a filesystem-safe artifact name component.

    Parameters
    ----------
    value : str
        Raw graph or engine name.

    Returns
    -------
    str
        Sanitized name suitable for a position tensor filename.
    """
    return "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in value)


def position_relpath(graph_name: str, engine_name: str) -> str:
    """Return the relative position-tensor path for a graph and engine.

    Parameters
    ----------
    graph_name : str
        Benchmark graph name.
    engine_name : str
        Benchmark engine name.

    Returns
    -------
    str
        POSIX-style relative path below the output directory.
    """
    filename = f"{safe_component(graph_name)}__{safe_component(engine_name)}.pt"
    return str(Path("positions") / filename)


def json_clean(value: Any) -> Any:
    """Convert metric payload values into JSON-safe primitives.

    Parameters
    ----------
    value : Any
        Raw value from metrics or runtime metadata.

    Returns
    -------
    Any
        JSON-serializable value.
    """
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return json_clean(value.item())
        return [json_clean(item) for item in value.detach().cpu().flatten().tolist()]
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return numeric if math.isfinite(numeric) else None


def load_existing_results(output_dir: Path) -> Dict[str, Any]:
    """Load an existing results payload.

    Parameters
    ----------
    output_dir : Path
        Baseline output directory containing ``results.json``.

    Returns
    -------
    Dict[str, Any]
        Parsed results payload.

    Raises
    ------
    FileNotFoundError
        If the existing results file is absent.
    """
    with (output_dir / "results.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_results(output_dir: Path, payload: Dict[str, Any]) -> None:
    """Write the results payload to disk.

    Parameters
    ----------
    output_dir : Path
        Baseline output directory.
    payload : Dict[str, Any]
        JSON-serializable results payload.

    Returns
    -------
    None
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def rows_path(output_dir: Path) -> Path:
    """Return the JSONL path for incrementally completed rows.

    Parameters
    ----------
    output_dir : Path
        Baseline output directory.

    Returns
    -------
    Path
        Path to ``results.rows.jsonl``.
    """
    return output_dir / "results.rows.jsonl"


def stamp_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of a row stamped with provenance for the current process.

    Every row written to the resumable JSONL store carries the exact git SHA
    and wall-clock time it was computed under. This makes stale-resume rows
    detectable after the fact: if a later audit finds ``row_git_sha`` values
    that predate a code change relevant to the metric, those rows are
    provably stale and must be recomputed with ``--fresh``.

    Parameters
    ----------
    row : Dict[str, Any]
        Row about to be written.

    Returns
    -------
    Dict[str, Any]
        A new dict with ``row_git_sha`` and ``row_written_at`` added. Does
        not mutate ``row``.
    """
    return {
        **row,
        "row_git_sha": git_sha(),
        "row_written_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
    }


def append_row(output_dir: Path, row: Dict[str, Any]) -> None:
    """Append one completed row to the resumable JSONL store.

    The row is stamped with the current git SHA and timestamp (see
    ``stamp_row``) before it is written.

    Parameters
    ----------
    output_dir : Path
        Baseline output directory.
    row : Dict[str, Any]
        Completed result row.

    Returns
    -------
    None
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with rows_path(output_dir).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(stamp_row(row), sort_keys=True))
        handle.write("\n")


def warn_resumed_rows(skipped_keys: Set[Tuple[str, str]], resume: bool) -> None:
    """Print a loud warning when a sweep reuses rows from a prior run.

    Silent row reuse across a resumed sweep is the stale-resume hole this
    batch closes: a partially-completed staging store can carry rows that
    were computed under OLDER code than the current git SHA, and ``--resume``
    would otherwise skip recomputing them without any signal to the operator.

    Parameters
    ----------
    skipped_keys : Set[Tuple[str, str]]
        Graph-engine keys that will be skipped because a matching row already
        exists in the staging store.
    resume : bool
        Whether this run was invoked with ``--resume``. No warning is printed
        when the run is not a resume (``skipped_keys`` should be empty in
        that case, but the flag makes the no-op path explicit).

    Returns
    -------
    None
    """
    if not resume or not skipped_keys:
        return
    banner = "=" * 78
    print(banner, flush=True)
    print(
        f"WARNING: RESUMED RUN REUSING {len(skipped_keys)} CACHED ROW(S) FROM A "
        "PRIOR STAGING STORE",
        flush=True,
    )
    print(
        "These rows were NOT recomputed under the current git SHA "
        f"({git_sha()}). If code relevant to their metrics changed since they "
        "were written, they are STALE. Use --fresh to force a provably fresh "
        "sweep with zero cached rows.",
        flush=True,
    )
    sample = sorted(skipped_keys)[:10]
    print(
        f"Sample reused keys ({min(10, len(skipped_keys))} of {len(skipped_keys)}): {sample}",
        flush=True,
    )
    print(banner, flush=True)


def load_jsonl_rows(output_dir: Path) -> List[Dict[str, Any]]:
    """Load completed rows from ``results.rows.jsonl``.

    Parameters
    ----------
    output_dir : Path
        Baseline output directory.

    Returns
    -------
    List[Dict[str, Any]]
        Rows in append order.
    """
    path = rows_path(output_dir)
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_jsonl_rows(output_dir: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Rewrite the JSONL row store with a known row sequence.

    Parameters
    ----------
    output_dir : Path
        Baseline output directory.
    rows : Sequence[Dict[str, Any]]
        Rows to persist.

    Returns
    -------
    None
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with rows_path(output_dir).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def staging_dir(output_dir: Path) -> Path:
    """Return the temporary store path used before atomic publish.

    Parameters
    ----------
    output_dir : Path
        Final baseline output directory.

    Returns
    -------
    Path
        Sibling staging directory.
    """
    return output_dir.with_name(f"{output_dir.name}.tmp")


def previous_dir(output_dir: Path) -> Path:
    """Return the temporary backup path used during atomic publish.

    Parameters
    ----------
    output_dir : Path
        Final baseline output directory.

    Returns
    -------
    Path
        Sibling directory that briefly holds the previous store.
    """
    return output_dir.with_name(f"{output_dir.name}.prev")


def _write_worker_message(result_path: str, message: Dict[str, Any]) -> None:
    """Write a child-process result message and flush it to disk.

    Parameters
    ----------
    result_path : str
        JSON file path for the child result message.
    message : Dict[str, Any]
        JSON-serializable result metadata.

    Returns
    -------
    None
    """
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(message, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _external_layout_worker(
    graph: Any,
    engine_name: str,
    temp_position_path: str,
    result_path: str,
) -> None:
    """Run one external layout in an isolated child process.

    Parameters
    ----------
    graph : Any
        ``DaguaGraph`` to lay out.
    engine_name : str
        External competitor name.
    temp_position_path : str
        Path where successful positions are written by the child.
    result_path : str
        JSON file used to return non-crash status metadata to the parent.

    Returns
    -------
    None
    """
    try:
        competitor = get_competitor(engine_name)
        if competitor is None:
            _write_worker_message(
                result_path,
                {
                    "status": "ERROR",
                    "runtime_s": 0.0,
                    "error": "adapter not registered",
                    "temp_positions_path": None,
                },
            )
            os._exit(0)
        try:
            result: CompetitorResult = competitor.layout(graph, timeout=TIMEOUT_SECONDS, seed=SEED)
        except Exception as exc:  # noqa: BLE001
            _write_worker_message(
                result_path,
                {
                    "status": "ERROR",
                    "runtime_s": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                    "temp_positions_path": None,
                },
            )
            os._exit(0)
        if result.pos is None:
            _write_worker_message(
                result_path,
                {
                    "status": "ERROR",
                    "runtime_s": float(result.runtime_seconds),
                    "error": result.error or "adapter returned no positions",
                    "temp_positions_path": None,
                },
            )
            os._exit(0)
        positions = result.pos.detach().cpu().to(dtype=torch.float32)
        torch.save(positions, temp_position_path)
        _write_worker_message(
            result_path,
            {
                "status": "OK",
                "runtime_s": float(result.runtime_seconds),
                "error": result.error,
                "temp_positions_path": temp_position_path,
            },
        )
        os._exit(0)
    except BaseException:
        os._exit(1)


def build_corpus() -> List[TestGraph]:
    """Build the benchmark corpus once and filter it to small graphs.

    Returns
    -------
    List[TestGraph]
        All registered test graphs with at most 500 nodes.
    """
    graphs = get_test_graphs(max_nodes=500)
    for test_graph in graphs:
        test_graph.graph.compute_node_sizes()
    return graphs


def graph_population(graph_name: str) -> str:
    """Return the r79 reporting population for a graph.

    Parameters
    ----------
    graph_name : str
        Benchmark graph name.

    Returns
    -------
    str
        ``"extended"`` for r79 additions, otherwise ``"legacy"``.
    """
    return "extended" if graph_name in R79_NEW_GRAPH_NAMES else "legacy"


def engine_availability(engine_names: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Inspect availability of requested competitor adapters.

    Parameters
    ----------
    engine_names : Sequence[str]
        Engine names to check.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Availability metadata keyed by engine name.
    """
    availability: Dict[str, Dict[str, Any]] = {}
    for engine_name in engine_names:
        competitor = get_competitor(engine_name)
        if competitor is None:
            availability[engine_name] = {"available": False, "reason": "adapter not registered"}
            continue
        try:
            available = competitor.available()
        except Exception as exc:  # noqa: BLE001
            availability[engine_name] = {"available": False, "reason": str(exc)}
            continue
        reason = None if available else "adapter unavailable"
        availability[engine_name] = {"available": bool(available), "reason": reason}
    return availability


def row_key(row: Dict[str, Any]) -> Tuple[str, str]:
    """Return the graph-engine key for a results row.

    Parameters
    ----------
    row : Dict[str, Any]
        Results row.

    Returns
    -------
    Tuple[str, str]
        Graph name and engine name.
    """
    return str(row["graph"]), str(row["engine"])


def make_skip_row(
    test_graph: TestGraph,
    engine_name: str,
    reason: str,
) -> Dict[str, Any]:
    """Create a SKIP row for one graph-engine pair.

    Parameters
    ----------
    test_graph : TestGraph
        Benchmark graph metadata.
    engine_name : str
        Engine name.
    reason : str
        Skip explanation.

    Returns
    -------
    Dict[str, Any]
        Results row.
    """
    return {
        "graph": test_graph.name,
        "population": graph_population(test_graph.name),
        "engine": engine_name,
        "status": "SKIP",
        "runtime_s": 0.0,
        "metrics": {},
        "composite": None,
        "positions_path": None,
        "nodes": test_graph.graph.num_nodes,
        "edges": int(test_graph.graph.edge_index.shape[1]),
        "reason": reason,
    }


def make_error_row(
    test_graph: TestGraph,
    engine_name: str,
    runtime_s: float,
    error: str,
    status_detail: Optional[str] = None,
) -> Dict[str, Any]:
    """Create an ERROR row for one graph-engine pair.

    Parameters
    ----------
    test_graph : TestGraph
        Benchmark graph metadata.
    engine_name : str
        Engine name.
    runtime_s : float
        Runtime recorded before the error was reported.
    error : str
        Error detail for the row.
    status_detail : Optional[str], default=None
        Optional machine-readable status detail such as ``"timeout"``.

    Returns
    -------
    Dict[str, Any]
        Results row.
    """
    row = {
        "graph": test_graph.name,
        "population": graph_population(test_graph.name),
        "engine": engine_name,
        "status": "ERROR",
        "runtime_s": float(runtime_s),
        "metrics": {},
        "composite": None,
        "positions_path": None,
        "nodes": test_graph.graph.num_nodes,
        "edges": int(test_graph.graph.edge_index.shape[1]),
        "error": error,
    }
    if status_detail is not None:
        row["status_detail"] = status_detail
    return row


def external_layout_result(
    test_graph: TestGraph,
    engine_name: str,
    output_dir: Path,
) -> Tuple[Optional[CompetitorResult], Optional[Dict[str, Any]]]:
    """Run one external layout in a short-lived child process.

    Parameters
    ----------
    test_graph : TestGraph
        Benchmark graph metadata.
    engine_name : str
        External competitor name.
    output_dir : Path
        Baseline output directory used for temporary position files.

    Returns
    -------
    Tuple[Optional[CompetitorResult], Optional[Dict[str, Any]]]
        Successful layout result for parent-side metrics, or an ERROR row.
    """
    context = mp.get_context("fork")
    temp_path = (
        output_dir
        / "positions"
        / (f".{safe_component(test_graph.name)}__{safe_component(engine_name)}.child.pt")
    )
    result_path = (
        output_dir
        / "positions"
        / (f".{safe_component(test_graph.name)}__{safe_component(engine_name)}.child.json")
    )
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    for path in (temp_path, result_path):
        if path.exists():
            path.unlink()
    process = context.Process(
        target=_external_layout_worker,
        args=(test_graph.graph, engine_name, str(temp_path), str(result_path)),
    )
    start = time.perf_counter()
    process.start()
    process.join(TIMEOUT_SECONDS)
    elapsed = time.perf_counter() - start
    if process.is_alive():
        process.terminate()
        process.join(5.0)
        if process.is_alive():
            process.kill()
            process.join()
        if temp_path.exists():
            temp_path.unlink()
        if result_path.exists():
            result_path.unlink()
        return None, make_error_row(
            test_graph,
            engine_name,
            elapsed,
            "timeout",
            status_detail="timeout",
        )
    exitcode = process.exitcode
    if exitcode is not None and exitcode < 0:
        if temp_path.exists():
            temp_path.unlink()
        if result_path.exists():
            result_path.unlink()
        return None, make_error_row(
            test_graph,
            engine_name,
            elapsed,
            f"crash(signal {-exitcode})",
        )
    if not result_path.is_file():
        if temp_path.exists():
            temp_path.unlink()
        detail = f"child exited {exitcode}" if exitcode is not None else "child exited"
        return None, make_error_row(test_graph, engine_name, elapsed, detail)
    with result_path.open("r", encoding="utf-8") as handle:
        message = json.load(handle)
    result_path.unlink()
    if exitcode not in (0, None):
        if temp_path.exists():
            temp_path.unlink()
        detail = f"child exited {exitcode}" if exitcode is not None else "child exited"
        return None, make_error_row(test_graph, engine_name, elapsed, detail)
    if message.get("status") != "OK":
        if temp_path.exists():
            temp_path.unlink()
        return None, make_error_row(
            test_graph,
            engine_name,
            float(message.get("runtime_s") or elapsed),
            str(message.get("error") or "external child error"),
        )
    position_path = message.get("temp_positions_path")
    if not position_path or not Path(str(position_path)).is_file():
        return None, make_error_row(
            test_graph,
            engine_name,
            float(message.get("runtime_s") or elapsed),
            "external child returned no position file",
        )
    positions = torch.load(str(position_path), map_location="cpu")
    Path(str(position_path)).unlink()
    return (
        CompetitorResult(
            name=engine_name,
            pos=positions,
            runtime_seconds=float(message.get("runtime_s") or elapsed),
            error=message.get("error"),
        ),
        None,
    )


def run_engine(
    test_graph: TestGraph,
    competitor: CompetitorBase,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run one engine on one graph and persist positions for OK rows.

    Parameters
    ----------
    test_graph : TestGraph
        Benchmark graph metadata.
    competitor : CompetitorBase
        Layout competitor adapter.
    output_dir : Path
        Baseline output directory.

    Returns
    -------
    Dict[str, Any]
        Results row.
    """
    graph = test_graph.graph
    if competitor.name in EXTERNAL_ENGINE_NAMES:
        result, error_row = external_layout_result(test_graph, competitor.name, output_dir)
        if error_row is not None:
            return error_row
        if result is None:
            return make_error_row(test_graph, competitor.name, 0.0, "external child failed")
    else:
        try:
            result = competitor.layout(graph, timeout=TIMEOUT_SECONDS, seed=SEED)
        except Exception as exc:  # noqa: BLE001
            return make_error_row(test_graph, competitor.name, 0.0, f"{type(exc).__name__}: {exc}")

    base_row = {
        "graph": test_graph.name,
        "population": graph_population(test_graph.name),
        "engine": competitor.name,
        "runtime_s": float(result.runtime_seconds),
        "nodes": graph.num_nodes,
        "edges": int(graph.edge_index.shape[1]),
    }
    if result.pos is None:
        return {
            **base_row,
            "status": "ERROR",
            "metrics": {},
            "composite": None,
            "positions_path": None,
            "error": result.error or "adapter returned no positions",
        }

    positions = result.pos.detach().cpu().to(dtype=torch.float32)
    try:
        metrics = evaluate(graph, positions, tier="full")
        composite = composite_auto(metrics, is_semantically_directed(test_graph))
    except Exception as exc:  # noqa: BLE001
        return {
            **base_row,
            "status": "ERROR",
            "metrics": {},
            "composite": None,
            "positions_path": None,
            "error": f"metrics {type(exc).__name__}: {exc}",
        }

    relpath = position_relpath(test_graph.name, competitor.name)
    position_path = output_dir / relpath
    position_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(positions, position_path)
    return {
        **base_row,
        "status": "OK",
        "metrics": json_clean(metrics),
        "composite": float(composite),
        "positions_path": relpath,
        "error": None,
    }


def validate_store(output_dir: Path) -> None:
    """Validate one-to-one consistency between OK rows and position tensors.

    Parameters
    ----------
    output_dir : Path
        Baseline output directory.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If a row or position artifact is missing its counterpart.
    """
    payload = load_existing_results(output_dir)
    rows = payload.get("rows", [])
    ok_paths = {
        str(row.get("positions_path"))
        for row in rows
        if row.get("status") == "OK" and row.get("positions_path")
    }
    missing = sorted(path for path in ok_paths if not (output_dir / path).is_file())
    position_paths = {
        str(path.relative_to(output_dir))
        for path in (output_dir / "positions").glob("*.pt")
        if path.is_file()
    }
    orphaned = sorted(position_paths - ok_paths)
    if missing or orphaned:
        details = []
        if missing:
            details.append(f"missing position files for OK rows: {missing}")
        if orphaned:
            details.append(f"orphaned position files without OK rows: {orphaned}")
        raise RuntimeError("; ".join(details))


def graph_best_external(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Find the best external OK row for each graph.

    Parameters
    ----------
    rows : Iterable[Dict[str, Any]]
        Results rows.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Best external row keyed by graph name.
    """
    best: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if row.get("engine") not in EXTERNAL_ENGINE_NAMES or row.get("status") != "OK":
            continue
        composite = row.get("composite")
        if composite is None:
            continue
        graph_name = str(row["graph"])
        if graph_name not in best or float(composite) > float(best[graph_name]["composite"]):
            best[graph_name] = row
    return best


def summarize_wtl(rows: List[Dict[str, Any]], population: str) -> Tuple[int, int, int]:
    """Summarize Dagua wins, ties, and losses for one population.

    Parameters
    ----------
    rows : List[Dict[str, Any]]
        Results rows.
    population : str
        Reporting population: ``legacy`` or ``extended``.

    Returns
    -------
    Tuple[int, int, int]
        Win, tie, and loss counts.
    """
    rows_for_population = [row for row in rows if row.get("population") == population]
    best_external = graph_best_external(rows_for_population)
    dagua_rows = {
        str(row["graph"]): row
        for row in rows_for_population
        if row.get("engine") == "dagua" and row.get("status") == "OK"
    }
    wins = ties = losses = 0
    for graph_name, dagua_row in dagua_rows.items():
        external_row = best_external.get(graph_name)
        if external_row is None:
            continue
        delta = float(dagua_row["composite"]) - float(external_row["composite"])
        if delta > TIE_BAND:
            wins += 1
        elif delta >= -TIE_BAND:
            ties += 1
        else:
            losses += 1
    return wins, ties, losses


def score_stored_metrics(row: Dict[str, Any], test_graph: TestGraph) -> float:
    """Recompute a row composite from persisted metrics and graph semantics.

    Parameters
    ----------
    row : Dict[str, Any]
        Stored benchmark row containing a ``metrics`` mapping.
    test_graph : TestGraph
        Corpus graph metadata used to resolve semantic directedness.

    Returns
    -------
    float
        Recomputed composite score.
    """
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"row {row_key(row)} has no metric dictionary")
    clean_metrics = {str(key): value for key, value in metrics.items() if value is not None}
    full_fields = {"crossing_rate", "sampled_stress", "angular_res_mean_deg"}
    if full_fields.issubset(clean_metrics):
        return float(composite_auto(clean_metrics, is_semantically_directed(test_graph)))
    return float(composite_large(clean_metrics))


def rescore_rows(
    rows: Sequence[Dict[str, Any]],
    graphs: Sequence[TestGraph],
) -> List[Dict[str, Any]]:
    """Return rows with OK composites recomputed from stored metric payloads.

    Parameters
    ----------
    rows : Sequence[Dict[str, Any]]
        Stored result rows.
    graphs : Sequence[TestGraph]
        Benchmark graph metadata.

    Returns
    -------
    List[Dict[str, Any]]
        Rows with recomputed ``composite`` values for successful layouts.
    """
    graph_by_name = {graph.name: graph for graph in graphs}
    rescored: List[Dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        if row.get("status") == "OK" and row.get("composite") is not None:
            graph = graph_by_name.get(str(row.get("graph")))
            if graph is None:
                raise RuntimeError(f"stored row references unknown graph: {row.get('graph')}")
            new_row["composite"] = score_stored_metrics(row, graph)
        rescored.append(new_row)
    return rescored


def wtl_table(rows: List[Dict[str, Any]]) -> Dict[str, Tuple[int, int, int]]:
    """Summarize Dagua W/T/L for every reporting population.

    Parameters
    ----------
    rows : List[Dict[str, Any]]
        Result rows to summarize.

    Returns
    -------
    Dict[str, Tuple[int, int, int]]
        Win, tie, and loss counts keyed by population.
    """
    return {population: summarize_wtl(rows, population) for population in ("legacy", "extended")}


def biggest_composite_movers(
    old_rows: Sequence[Dict[str, Any]],
    new_rows: Sequence[Dict[str, Any]],
    limit: int = 12,
) -> List[Dict[str, Any]]:
    """Find the largest absolute composite changes after semantic rescoring.

    Parameters
    ----------
    old_rows : Sequence[Dict[str, Any]]
        Rows before rescoring.
    new_rows : Sequence[Dict[str, Any]]
        Rows after rescoring.
    limit : int, default=12
        Maximum number of movers to return.

    Returns
    -------
    List[Dict[str, Any]]
        Largest row-level composite deltas.
    """
    old_by_key = {row_key(row): row for row in old_rows if row.get("composite") is not None}
    movers: List[Dict[str, Any]] = []
    for row in new_rows:
        key = row_key(row)
        old_row = old_by_key.get(key)
        if old_row is None or row.get("composite") is None:
            continue
        before = float(old_row["composite"])
        after = float(row["composite"])
        delta = after - before
        if abs(delta) <= 1e-9:
            continue
        movers.append(
            {
                "graph": key[0],
                "engine": key[1],
                "before": before,
                "after": after,
                "delta": delta,
            }
        )
    return sorted(movers, key=lambda item: abs(float(item["delta"])), reverse=True)[:limit]


def directedness_audit(graphs: Sequence[TestGraph]) -> List[Dict[str, str]]:
    """Build graph-level directedness verdicts for the report.

    Parameters
    ----------
    graphs : Sequence[TestGraph]
        Benchmark corpus.

    Returns
    -------
    List[Dict[str, str]]
        Directedness verdicts sorted by graph name.
    """
    rows: List[Dict[str, str]] = []
    for graph in graphs:
        verdict = "directed" if is_semantically_directed(graph) else "undirected"
        if "undirected" in graph.tags:
            reason = (
                "storage orientation is arbitrary for social, random, mesh, or symmetric structure"
            )
        elif "dag" in graph.tags or "dependency" in graph.tags:
            reason = "construction encodes DAG/dependency flow"
        elif graph.name.startswith(("linear_", "deep_", "random_dag_", "r79_directed_scc_")):
            reason = "construction encodes directed path, DAG, or SCC semantics"
        elif any(tag in graph.tags for tag in {"neural-net", "wide-parallel", "skip-heavy"}):
            reason = "construction encodes dataflow-style source-to-sink edges"
        else:
            reason = "construction uses explicit source-to-target flow semantics"
        rows.append({"graph": graph.name, "verdict": verdict, "reason": reason})
    return sorted(rows, key=lambda item: item["graph"])


def per_graph_comparison(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build per-graph Dagua-vs-best-external comparison rows.

    Parameters
    ----------
    rows : List[Dict[str, Any]]
        Results rows.

    Returns
    -------
    List[Dict[str, Any]]
        Comparison rows sorted by population then graph name.
    """
    best_external = graph_best_external(rows)
    dagua_rows = {
        str(row["graph"]): row
        for row in rows
        if row.get("engine") == "dagua" and row.get("status") == "OK"
    }
    comparisons: List[Dict[str, Any]] = []
    for graph_name, dagua_row in sorted(dagua_rows.items()):
        external_row = best_external.get(graph_name)
        if external_row is None:
            continue
        delta = float(dagua_row["composite"]) - float(external_row["composite"])
        comparisons.append(
            {
                "graph": graph_name,
                "population": dagua_row["population"],
                "dagua": float(dagua_row["composite"]),
                "best_external": float(external_row["composite"]),
                "delta": delta,
                "winner": "dagua" if delta > TIE_BAND else str(external_row["engine"]),
            }
        )
    return sorted(comparisons, key=lambda item: (str(item["population"]), str(item["graph"])))


def generate_report(output_dir: Path, payload: Dict[str, Any]) -> None:
    """Generate the Markdown baseline report.

    Parameters
    ----------
    output_dir : Path
        Baseline output directory.
    payload : Dict[str, Any]
        Results payload.

    Returns
    -------
    None
    """
    rows = list(payload["rows"])
    comparisons = per_graph_comparison(rows)
    losses = sorted(
        [item for item in comparisons if float(item["delta"]) < -TIE_BAND],
        key=lambda item: float(item["delta"]),
    )
    lines = [
        "# R79 Baseline",
        "",
        f"- Date: {payload['metadata']['date']}",
        f"- Git SHA: {payload['metadata']['git_sha']}",
        f"- Corpus <=500 nodes: {payload['metadata']['graph_count']}",
        f"- Legacy graphs: {payload['metadata']['legacy_count']}",
        f"- R79 extension graphs: {payload['metadata']['extended_count']}",
        f"- Tie band: +/-{TIE_BAND:.1f} composite points",
        "",
        "## Engine Availability",
        "",
        "| Engine | Available | Reason |",
        "| --- | ---: | --- |",
    ]
    for engine_name in ENGINE_NAMES:
        info = payload["metadata"]["engine_availability"].get(engine_name, {})
        lines.append(
            f"| {engine_name} | {str(bool(info.get('available')))} | {info.get('reason') or ''} |"
        )

    lines.extend(
        [
            "",
            "## Scoreboards",
            "",
            "| Population | W | T | L |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for population in ("legacy", "extended"):
        wins, ties, losses_count = summarize_wtl(rows, population)
        lines.append(f"| {population} | {wins} | {ties} | {losses_count} |")

    semantics_fix = payload["metadata"].get("semantics_fix")
    if isinstance(semantics_fix, dict):
        lines.extend(
            [
                "",
                "## Semantic Directedness Fix",
                "",
                "| Population | Before W/T/L | After W/T/L |",
                "| --- | ---: | ---: |",
            ]
        )
        before_wtl = semantics_fix.get("before_wtl", {})
        after_wtl = wtl_table(rows)
        for population in ("legacy", "extended"):
            before = before_wtl.get(population, [0, 0, 0])
            after = after_wtl[population]
            lines.append(
                "| {population} | {before_w}/{before_t}/{before_l} | "
                "{after_w}/{after_t}/{after_l} |".format(
                    population=population,
                    before_w=int(before[0]),
                    before_t=int(before[1]),
                    before_l=int(before[2]),
                    after_w=after[0],
                    after_t=after[1],
                    after_l=after[2],
                )
            )
        movers = semantics_fix.get("biggest_movers", [])
        if movers:
            lines.extend(
                [
                    "",
                    "### Biggest Composite Movers",
                    "",
                    "| Graph | Engine | Before | After | Delta |",
                    "| --- | --- | ---: | ---: | ---: |",
                ]
            )
            for mover in movers:
                lines.append(
                    "| {graph} | {engine} | {before:.3f} | {after:.3f} | {delta:.3f} |".format(
                        graph=mover["graph"],
                        engine=mover["engine"],
                        before=float(mover["before"]),
                        after=float(mover["after"]),
                        delta=float(mover["delta"]),
                    )
                )

    lines.extend(
        [
            "",
            "## Per-Graph Comparison",
            "",
            "| Population | Graph | Dagua | Best External | Delta | Winning Engine |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for item in comparisons:
        lines.append(
            "| {population} | {graph} | {dagua:.3f} | {best_external:.3f} | "
            "{delta:.3f} | {winner} |".format(**item)
        )

    lines.extend(
        [
            "",
            "## Losses Worst First",
            "",
            "| Population | Graph | Delta | Winning Engine |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for item in losses:
        lines.append("| {population} | {graph} | {delta:.3f} | {winner} |".format(**item))

    if payload["metadata"].get("positions_note"):
        lines.extend(["", "## Position Store", "", str(payload["metadata"]["positions_note"])])

    audit_rows = payload["metadata"].get("directedness_audit", [])
    if audit_rows:
        lines.extend(
            [
                "",
                "## Directedness Audit",
                "",
                "| Graph | Verdict | Reason |",
                "| --- | --- | --- |",
            ]
        )
        for item in audit_rows:
            lines.append(f"| {item['graph']} | {item['verdict']} | {item['reason']} |")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "BASELINE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def complete_keys(output_dir: Path) -> Set[Tuple[str, str]]:
    """Return resumable graph-engine keys already complete in JSONL.

    Parameters
    ----------
    output_dir : Path
        Baseline output directory.

    Returns
    -------
    Set[Tuple[str, str]]
        Completed row keys. OK rows require their positions file to exist.
    """
    keys: Set[Tuple[str, str]] = set()
    for row in load_jsonl_rows(output_dir):
        key = row_key(row)
        if row.get("status") == "OK":
            position_path = row.get("positions_path")
            if position_path and (output_dir / str(position_path)).is_file():
                keys.add(key)
        elif row.get("status") in {"ERROR", "SKIP"}:
            keys.add(key)
    return keys


def filter_graphs(graphs: List[TestGraph], graph_names: Optional[Sequence[str]]) -> List[TestGraph]:
    """Filter the corpus by optional graph names.

    Parameters
    ----------
    graphs : List[TestGraph]
        Full benchmark corpus.
    graph_names : Optional[Sequence[str]]
        Names to keep, or ``None`` for all graphs.

    Returns
    -------
    List[TestGraph]
        Filtered benchmark corpus.
    """
    if graph_names is None:
        return graphs
    wanted = set(graph_names)
    selected = [graph for graph in graphs if graph.name in wanted]
    missing = sorted(wanted - {graph.name for graph in selected})
    if missing:
        raise RuntimeError(f"unknown graph filter(s): {missing}")
    return selected


def filter_engines(
    engine_names: Sequence[str],
    selected_names: Optional[Sequence[str]],
) -> List[str]:
    """Filter benchmark engines by optional names.

    Parameters
    ----------
    engine_names : Sequence[str]
        Full engine list.
    selected_names : Optional[Sequence[str]]
        Names to keep, or ``None`` for all engines.

    Returns
    -------
    List[str]
        Filtered engine list.
    """
    if selected_names is None:
        return list(engine_names)
    wanted = set(selected_names)
    unknown = sorted(wanted - set(engine_names))
    if unknown:
        raise RuntimeError(f"unknown engine filter(s): {unknown}")
    return [engine_name for engine_name in engine_names if engine_name in wanted]


def prepare_full_store(output_dir: Path, resume: bool) -> Path:
    """Prepare the staging store for a full baseline run.

    Parameters
    ----------
    output_dir : Path
        Final baseline output directory.
    resume : bool
        Whether to preserve an existing staging row log.

    Returns
    -------
    Path
        Staging directory where the run should write.
    """
    work_dir = staging_dir(output_dir)
    if resume:
        if not work_dir.exists() and output_dir.exists():
            shutil.copytree(output_dir, work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "positions").mkdir(parents=True, exist_ok=True)
        return work_dir
    if work_dir.exists():
        shutil.rmtree(work_dir)
    (work_dir / "positions").mkdir(parents=True, exist_ok=True)
    return work_dir


def prepare_dagua_only_store(output_dir: Path, graphs: List[TestGraph], resume: bool) -> Path:
    """Prepare the staging store for a Dagua-only rerun.

    Parameters
    ----------
    output_dir : Path
        Final baseline output directory.
    graphs : List[TestGraph]
        Graphs whose Dagua rows will be rerun.
    resume : bool
        Whether to preserve a partially rerun staging store.

    Returns
    -------
    Path
        Staging directory where the run should write.
    """
    work_dir = staging_dir(output_dir)
    if not resume or not work_dir.exists():
        if work_dir.exists():
            shutil.rmtree(work_dir)
        shutil.copytree(output_dir, work_dir)
        existing = load_existing_results(work_dir)
        graph_names = {graph.name for graph in graphs}
        external_rows = [
            row
            for row in existing["rows"]
            if row.get("engine") != "dagua" or row.get("graph") not in graph_names
        ]
        for graph in graphs:
            dagua_path = work_dir / position_relpath(graph.name, "dagua")
            if dagua_path.exists():
                dagua_path.unlink()
        write_jsonl_rows(work_dir, external_rows)
    (work_dir / "positions").mkdir(parents=True, exist_ok=True)
    return work_dir


def run_full(
    output_dir: Path,
    graphs: List[TestGraph],
    engine_names: Sequence[str],
    resume: bool,
    fresh: bool = False,
) -> Tuple[Dict[str, Any], Path]:
    """Run the complete r79 baseline from scratch.

    Parameters
    ----------
    output_dir : Path
        Baseline output directory.
    graphs : List[TestGraph]
        Corpus graphs.
    engine_names : Sequence[str]
        Engines to run.
    resume : bool
        Whether to skip complete rows in the staging JSONL store. Must be
        ``False`` when ``fresh`` is ``True`` (enforced by the CLI's
        mutually-exclusive group).
    fresh : bool, default=False
        When ``True``, refuse to proceed if any row survives staging-store
        preparation -- guarantees a provably fresh sweep with zero resumed
        rows.

    Returns
    -------
    Tuple[Dict[str, Any], Path]
        Results payload and staging directory.
    """
    work_dir = prepare_full_store(output_dir, resume)
    if fresh:
        assert_fresh_store(work_dir)
    availability = engine_availability(engine_names)
    skipped_keys = complete_keys(work_dir) if resume else set()
    warn_resumed_rows(skipped_keys, resume)
    for test_graph in graphs:
        for engine_name in engine_names:
            if (test_graph.name, engine_name) in skipped_keys:
                print(f"SKIP existing {test_graph.name} {engine_name}", flush=True)
                continue
            competitor = get_competitor(engine_name)
            info = availability[engine_name]
            if competitor is None or not info["available"]:
                row = make_skip_row(test_graph, engine_name, str(info.get("reason")))
                append_row(work_dir, row)
                continue
            print(f"RUN {test_graph.name} {engine_name}", flush=True)
            append_row(work_dir, run_engine(test_graph, competitor, work_dir))
    rows = sorted(load_jsonl_rows(work_dir), key=lambda row: (row["graph"], row["engine"]))
    return build_payload(rows, graphs, availability, engine_names), work_dir


def assert_fresh_store(work_dir: Path) -> None:
    """Refuse to proceed if the staging store still carries any cached rows.

    ``--fresh`` clears the resume path (see ``prepare_full_store`` /
    ``prepare_dagua_only_store`` with ``resume=False``), which should already
    leave zero resumable rows. This is a belt-and-suspenders check so a
    provably-fresh sweep cannot silently reuse rows even if that invariant
    ever breaks.

    Parameters
    ----------
    work_dir : Path
        Staging directory prepared for the run.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If any graph-engine row is already complete in the staging store.
    """
    stale = complete_keys(work_dir)
    if stale:
        raise RuntimeError(
            f"--fresh requested but {len(stale)} row(s) remain resumable in "
            f"the staging store after preparation; refusing to reuse them: "
            f"{sorted(stale)[:10]}"
        )
    print("FRESH RUN: staging store carries zero cached rows.", flush=True)


def run_dagua_only(
    output_dir: Path,
    graphs: List[TestGraph],
    resume: bool,
    fresh: bool = False,
) -> Tuple[Dict[str, Any], Path]:
    """Rerun only Dagua rows while preserving frozen external rows.

    Parameters
    ----------
    output_dir : Path
        Baseline output directory.
    graphs : List[TestGraph]
        Corpus graphs.
    resume : bool
        Whether to skip complete Dagua rows in the staging JSONL store. Must
        be ``False`` when ``fresh`` is ``True``.
    fresh : bool, default=False
        When ``True``, refuse to proceed if any Dagua row survives staging
        preparation.

    Returns
    -------
    Tuple[Dict[str, Any], Path]
        Updated results payload and staging directory.
    """
    work_dir = prepare_dagua_only_store(output_dir, graphs, resume)
    if fresh:
        stale = {key for key in complete_keys(work_dir) if key[1] == "dagua"}
        if stale:
            raise RuntimeError(
                f"--fresh requested but {len(stale)} dagua row(s) remain resumable "
                f"in the staging store after preparation; refusing to reuse them: "
                f"{sorted(stale)[:10]}"
            )
        print("FRESH RUN: staging store carries zero cached dagua rows.", flush=True)
    existing = load_existing_results(work_dir)
    semantics_fix = existing["metadata"].get("semantics_fix")
    availability = dict(existing["metadata"].get("engine_availability", {}))
    availability["dagua"] = engine_availability(["dagua"])["dagua"]
    competitor = get_competitor("dagua")
    if competitor is None:
        raise RuntimeError("dagua adapter not registered")
    skipped_keys = complete_keys(work_dir) if resume else set()
    warn_resumed_rows(skipped_keys, resume)
    for test_graph in graphs:
        if (test_graph.name, "dagua") in skipped_keys:
            print(f"SKIP existing {test_graph.name} dagua", flush=True)
            continue
        print(f"RUN {test_graph.name} dagua", flush=True)
        append_row(work_dir, run_engine(test_graph, competitor, work_dir))
    rows = sorted(load_jsonl_rows(work_dir), key=lambda row: (row["graph"], row["engine"]))
    return build_payload(rows, graphs, availability, ENGINE_NAMES, semantics_fix), work_dir


def build_payload(
    rows: List[Dict[str, Any]],
    graphs: List[TestGraph],
    availability: Dict[str, Dict[str, Any]],
    engine_names: Sequence[str],
    semantics_fix: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the persisted results payload.

    Parameters
    ----------
    rows : List[Dict[str, Any]]
        Results rows.
    graphs : List[TestGraph]
        Corpus graphs.
    availability : Dict[str, Dict[str, Any]]
        Engine availability metadata.
    engine_names : Sequence[str]
        Engines included in the payload.
    semantics_fix : Optional[Dict[str, Any]], optional
        Directedness-fix metadata to preserve across Dagua-only reruns.

    Returns
    -------
    Dict[str, Any]
        Results payload ready to serialize.
    """
    legacy_count = sum(1 for graph in graphs if graph_population(graph.name) == "legacy")
    extended_count = sum(1 for graph in graphs if graph_population(graph.name) == "extended")
    metadata: Dict[str, Any] = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": git_sha(),
        "seed": SEED,
        "timeout_s": TIMEOUT_SECONDS,
        "graph_count": len(graphs),
        "legacy_count": legacy_count,
        "extended_count": extended_count,
        "new_graph_names": sorted(R79_NEW_GRAPH_NAMES),
        "engine_names": list(engine_names),
        "engine_availability": availability,
        "directedness_audit": directedness_audit(graphs),
    }
    if semantics_fix is not None:
        metadata["semantics_fix"] = semantics_fix
    return {
        "metadata": {
            **metadata,
        },
        "rows": rows,
    }


def publish_store(work_dir: Path, output_dir: Path) -> None:
    """Atomically publish a validated staging store.

    Parameters
    ----------
    work_dir : Path
        Validated staging directory.
    output_dir : Path
        Final baseline output directory.

    Returns
    -------
    None
    """
    backup_dir = previous_dir(output_dir)
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    try:
        if output_dir.exists():
            output_dir.rename(backup_dir)
        work_dir.rename(output_dir)
    except Exception:
        if output_dir.exists() and output_dir != work_dir:
            shutil.rmtree(output_dir)
        if backup_dir.exists() and not output_dir.exists():
            backup_dir.rename(output_dir)
        raise
    if backup_dir.exists():
        shutil.rmtree(backup_dir)


def rescore_existing_store(output_dir: Path, graphs: List[TestGraph]) -> Dict[str, Any]:
    """Recompute stored composites in place without rerunning any layout.

    Parameters
    ----------
    output_dir : Path
        Existing baseline output directory.
    graphs : List[TestGraph]
        Current benchmark corpus metadata with corrected directedness tags.

    Returns
    -------
    Dict[str, Any]
        Updated results payload.
    """
    payload = load_existing_results(output_dir)
    old_rows = list(payload["rows"])
    old_baseline = output_dir / "BASELINE.md"
    archived_baseline = output_dir / "BASELINE_pre_semantics_fix.md"
    if old_baseline.is_file() and not archived_baseline.exists():
        shutil.copy2(old_baseline, archived_baseline)

    new_rows = rescore_rows(old_rows, graphs)
    availability = dict(payload["metadata"].get("engine_availability", {}))
    semantics_fix = {
        "before_wtl": {key: list(value) for key, value in wtl_table(old_rows).items()},
        "after_wtl": {key: list(value) for key, value in wtl_table(new_rows).items()},
        "biggest_movers": biggest_composite_movers(old_rows, new_rows),
        "archived_baseline": str(archived_baseline.relative_to(output_dir)),
    }
    updated = build_payload(
        rows=sorted(new_rows, key=lambda row: (row["graph"], row["engine"])),
        graphs=graphs,
        availability=availability,
        engine_names=payload["metadata"].get("engine_names", ENGINE_NAMES),
        semantics_fix=semantics_fix,
    )
    updated["metadata"]["positions_note"] = payload["metadata"].get(
        "positions_note",
        "Positions were not rerun for the semantic directedness rescore.",
    )
    write_results(output_dir, updated)
    write_jsonl_rows(output_dir, updated["rows"])
    validate_store(output_dir)
    generate_report(output_dir, updated)
    return updated


def main() -> int:
    """Run the r79 baseline CLI.

    Returns
    -------
    int
        Process exit status.
    """
    args = parse_args()
    set_size_aware_externals(not bool(args.size_blind_externals))
    output_dir: Path = args.output_dir
    start = time.perf_counter()
    graphs = filter_graphs(build_corpus(), args.graphs)
    engine_names = filter_engines(ENGINE_NAMES, args.engines)
    print(f"Corpus <=500 nodes: {len(graphs)}", flush=True)
    if args.rescore_only:
        payload = rescore_existing_store(output_dir, graphs)
        print(f"Rescored rows: {len(payload['rows'])}", flush=True)
        print(f"Wrote {output_dir / 'results.json'}", flush=True)
        print(f"Wrote {output_dir / 'BASELINE.md'}", flush=True)
        return 0
    if args.dagua_only:
        payload, work_dir = run_dagua_only(
            output_dir, graphs, bool(args.resume), fresh=bool(args.fresh)
        )
    else:
        payload, work_dir = run_full(
            output_dir, graphs, engine_names, bool(args.resume), fresh=bool(args.fresh)
        )
    payload["metadata"]["wall_time_s"] = round(time.perf_counter() - start, 3)
    write_results(work_dir, payload)
    validate_store(work_dir)
    generate_report(work_dir, payload)
    validate_store(work_dir)
    publish_store(work_dir, output_dir)
    validate_store(output_dir)
    print(f"Wrote {output_dir / 'results.json'}", flush=True)
    print(f"Wrote {output_dir / 'BASELINE.md'}", flush=True)
    print(f"Wall time: {payload['metadata']['wall_time_s']:.3f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
