"""Evaluate Dagua on held-out standard graph-drawing corpora."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch

from dagua.eval.competitors import get_competitor
from dagua.eval.competitors.base import CompetitorBase, CompetitorResult
from dagua.graph import DaguaGraph
from dagua.metrics import composite_auto, evaluate

OUTPUT_DIR = Path("eval_output/stdcorpora")
CORPUS_DIR = Path("eval_output/stdcorpora")
SEED = 42
TIMEOUT_SECONDS = 120.0
MAX_NODES = 2000
TIE_BAND = 0.5
REPORT_METRICS = (
    "sampled_stress",
    "neighborhood_preservation",
    "crossing_rate",
    "edge_length_cv",
)
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


@dataclass(frozen=True)
class LoadedGraph:
    """Loaded corpus graph with normalized integer edges."""

    name: str
    corpus: str
    graph: DaguaGraph
    source_path: Path
    directed: bool


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-nodes", type=int, default=MAX_NODES)
    parser.add_argument("--dagua-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--engines", nargs="+", default=None)
    parser.add_argument("--graphs", nargs="+", default=None)
    return parser.parse_args()


def git_sha() -> str:
    """Return the current git commit SHA.

    Returns
    -------
    str
        Current commit SHA, or ``unknown`` if git cannot provide one.
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
        Sanitized path component.
    """
    return "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in value)


def position_relpath(graph_name: str, engine_name: str) -> str:
    """Return a relative position tensor path.

    Parameters
    ----------
    graph_name : str
        Corpus graph name.
    engine_name : str
        Layout engine name.

    Returns
    -------
    str
        POSIX-style path below the output directory.
    """
    filename = f"{safe_component(graph_name)}__{safe_component(engine_name)}.pt"
    return str(Path("positions") / filename)


def json_clean(value: Any) -> Any:
    """Convert nested metric values to JSON-safe primitives.

    Parameters
    ----------
    value : Any
        Raw metric, tensor, collection, or scalar.

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


def infer_corpus(path: Path) -> str:
    """Infer a reporting corpus name from a source path.

    Parameters
    ----------
    path : Path
        Source graph path.

    Returns
    -------
    str
        ``rome``, ``north``, ``suitesparse``, or ``misc``.
    """
    lowered = "/".join(part.lower() for part in path.parts)
    if "rome" in lowered:
        return "rome"
    if "north" in lowered or "att" in lowered or "at&t" in lowered:
        return "north"
    if "suitesparse" in lowered or "matrix" in lowered or path.suffix.lower() == ".mtx":
        return "suitesparse"
    return "misc"


def infer_directed(path: Path, format_directed: Optional[bool] = None) -> bool:
    """Infer whether a corpus graph should be scored as directed.

    Parameters
    ----------
    path : Path
        Source graph path.
    format_directed : bool | None, default=None
        Direction reported by the file format, when available.

    Returns
    -------
    bool
        ``True`` only for explicit directed metadata or North DAG names.
    """
    if format_directed is not None:
        return bool(format_directed)
    lowered = "/".join(part.lower() for part in path.parts)
    return ("north" in lowered or "dag" in lowered) and "undirected" not in lowered


def build_graph(
    name: str,
    corpus: str,
    node_count: int,
    edges: Iterable[Tuple[int, int]],
    directed: bool,
    source_path: Path,
) -> LoadedGraph:
    """Build a ``DaguaGraph`` from normalized integer topology.

    Parameters
    ----------
    name : str
        Stable graph name.
    corpus : str
        Reporting corpus name.
    node_count : int
        Number of nodes.
    edges : Iterable[Tuple[int, int]]
        Integer edges using zero-based node IDs.
    directed : bool
        Whether the graph has semantic direction for scoring.
    source_path : Path
        Input file path.

    Returns
    -------
    LoadedGraph
        Loaded graph metadata and ``DaguaGraph`` instance.
    """
    graph = DaguaGraph()
    for node_id in range(node_count):
        graph.add_node(node_id, label=str(node_id))
    seen: Set[Tuple[int, int]] = set()
    for source, target in edges:
        if (
            source == target
            or source < 0
            or target < 0
            or source >= node_count
            or target >= node_count
        ):
            continue
        key = (source, target) if directed else tuple(sorted((source, target)))
        if key in seen:
            continue
        seen.add(key)
        graph.add_edge(source, target)
    graph.is_semantically_directed = directed
    graph.compute_node_sizes()
    return LoadedGraph(
        name=name,
        corpus=corpus,
        graph=graph,
        source_path=source_path,
        directed=directed,
    )


def _numeric_lines(path: Path) -> List[List[int]]:
    """Read whitespace-separated integer rows from a text graph file.

    Parameters
    ----------
    path : Path
        Input ``.graph`` path.

    Returns
    -------
    List[List[int]]
        Parsed integer rows with comments and blank lines removed.
    """
    rows: List[List[int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            body = line.split("#", 1)[0].split("%", 1)[0].strip()
            if not body:
                continue
            try:
                rows.append([int(float(token)) for token in body.split()])
            except ValueError:
                continue
    return rows


def load_graph_file(path: Path) -> LoadedGraph:
    """Load a Rome/North ``.graph`` text file.

    Parameters
    ----------
    path : Path
        Source file path.

    Returns
    -------
    LoadedGraph
        Loaded graph with zero-based node IDs.

    Raises
    ------
    ValueError
        If no usable topology can be parsed.
    """
    rows = _numeric_lines(path)
    if not rows:
        raise ValueError(f"{path} has no numeric graph rows")

    directed = infer_directed(path)
    corpus = infer_corpus(path)
    name = f"{corpus}/{path.stem}"
    header = rows[0]
    edges: List[Tuple[int, int]] = []

    if len(header) == 1 and len(rows) >= header[0] + 1:
        node_count = header[0]
        for index, neighbors in enumerate(rows[1 : node_count + 1]):
            for neighbor in neighbors:
                edges.append((index, neighbor - 1))
        return build_graph(name, corpus, node_count, edges, directed, path)

    if len(header) >= 2 and len(rows) >= header[0] + 1:
        node_count = header[0]
        for index, neighbors in enumerate(rows[1 : node_count + 1]):
            for neighbor in neighbors:
                if neighbor != 0:
                    edges.append((index, neighbor - 1))
        return build_graph(name, corpus, node_count, edges, directed, path)

    edge_rows = [row for row in rows if len(row) >= 2]
    if not edge_rows:
        raise ValueError(f"{path} has no edge rows")
    min_id = min(min(row[0], row[1]) for row in edge_rows)
    offset = 1 if min_id == 1 else 0
    max_id = max(max(row[0], row[1]) for row in edge_rows)
    node_count = max_id - offset + 1
    edges = [(row[0] - offset, row[1] - offset) for row in edge_rows]
    return build_graph(name, corpus, node_count, edges, directed, path)


def load_gml_file(path: Path) -> LoadedGraph:
    """Load a GML file through NetworkX.

    Parameters
    ----------
    path : Path
        Source GML file path.

    Returns
    -------
    LoadedGraph
        Loaded graph with zero-based node IDs.
    """
    try:
        import networkx as nx
    except ImportError as exc:
        raise RuntimeError("networkx is required to read GML files") from exc

    nx_graph = nx.read_gml(path, label=None)
    nodes = list(nx_graph.nodes())
    node_to_index = {node: index for index, node in enumerate(nodes)}
    edges = [(node_to_index[source], node_to_index[target]) for source, target in nx_graph.edges()]
    directed = infer_directed(path, nx_graph.is_directed())
    corpus = infer_corpus(path)
    return build_graph(f"{corpus}/{path.stem}", corpus, len(nodes), edges, directed, path)


def load_graphml_file(path: Path) -> LoadedGraph:
    """Load a GraphML (XML) file through NetworkX.

    Parameters
    ----------
    path : Path
        Source GraphML file path.

    Returns
    -------
    LoadedGraph
        Loaded graph with zero-based node IDs.
    """
    try:
        import networkx as nx
    except ImportError as exc:
        raise RuntimeError("networkx is required to read GraphML files") from exc

    nx_graph = nx.read_graphml(path)
    nodes = list(nx_graph.nodes())
    node_to_index = {node: index for index, node in enumerate(nodes)}
    edges = [(node_to_index[source], node_to_index[target]) for source, target in nx_graph.edges()]
    corpus = infer_corpus(path)
    directed = infer_directed(path, nx_graph.is_directed() if "north" not in corpus else None)
    return build_graph(f"{corpus}/{path.stem}", corpus, len(nodes), edges, directed, path)


def _load_mtx_with_scipy(path: Path) -> Tuple[int, List[Tuple[int, int]]]:
    """Load Matrix Market topology with SciPy.

    Parameters
    ----------
    path : Path
        Source Matrix Market file.

    Returns
    -------
    Tuple[int, List[Tuple[int, int]]]
        Matrix order and zero-based nonzero-coordinate edges.
    """
    try:
        from scipy.io import mmread
    except ImportError as exc:
        raise RuntimeError("scipy is required for this Matrix Market file") from exc

    matrix = mmread(path)
    coo = matrix.tocoo() if hasattr(matrix, "tocoo") else matrix
    row = list(coo.row)
    col = list(coo.col)
    node_count = int(max(coo.shape))
    return node_count, [(int(source), int(target)) for source, target in zip(row, col)]


def _load_mtx_coordinate_fallback(path: Path) -> Tuple[int, List[Tuple[int, int]]]:
    """Load a simple coordinate Matrix Market file without SciPy.

    Parameters
    ----------
    path : Path
        Source Matrix Market file.

    Returns
    -------
    Tuple[int, List[Tuple[int, int]]]
        Matrix order and zero-based nonzero-coordinate edges.
    """
    size_row: Optional[List[int]] = None
    entries: List[Tuple[int, int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue
            parts = stripped.split()
            if size_row is None:
                size_row = [int(parts[0]), int(parts[1])]
                continue
            if len(parts) >= 2:
                entries.append((int(parts[0]) - 1, int(parts[1]) - 1))
    if size_row is None:
        raise ValueError(f"{path} has no Matrix Market size row")
    return max(size_row), entries


def load_mtx_file(path: Path) -> LoadedGraph:
    """Load a SuiteSparse Matrix Market file as an undirected sparsity graph.

    Parameters
    ----------
    path : Path
        Source Matrix Market file.

    Returns
    -------
    LoadedGraph
        Loaded graph with zero-based node IDs.
    """
    try:
        node_count, entries = _load_mtx_with_scipy(path)
    except RuntimeError:
        node_count, entries = _load_mtx_coordinate_fallback(path)
    edges = [(source, target) for source, target in entries if source != target]
    corpus = infer_corpus(path)
    return build_graph(f"{corpus}/{path.stem}", corpus, node_count, edges, False, path)


def load_corpus(corpus_dir: Path, max_nodes: int) -> List[LoadedGraph]:
    """Load all supported graph files under a corpus directory.

    Parameters
    ----------
    corpus_dir : Path
        Directory containing dropped-in corpus files.
    max_nodes : int
        Maximum node count accepted into the measurement run.

    Returns
    -------
    List[LoadedGraph]
        Loaded graphs, sorted by corpus/name.
    """
    loaders = {
        ".graph": load_graph_file,
        ".gml": load_gml_file,
        ".graphml": load_graphml_file,
        ".mtx": load_mtx_file,
    }
    graphs: List[LoadedGraph] = []
    for path in sorted(corpus_dir.rglob("*")):
        loader = loaders.get(path.suffix.lower())
        if loader is None or not path.is_file():
            continue
        loaded = loader(path)
        if loaded.graph.num_nodes <= max_nodes:
            graphs.append(loaded)
    return sorted(graphs, key=lambda item: (item.corpus, item.name))


def load_existing_results(output_dir: Path) -> Dict[str, Any]:
    """Load an existing result payload.

    Parameters
    ----------
    output_dir : Path
        Harness output directory.

    Returns
    -------
    Dict[str, Any]
        Parsed ``results.json`` payload.
    """
    with (output_dir / "results.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_results(output_dir: Path, payload: Dict[str, Any]) -> None:
    """Write the result payload.

    Parameters
    ----------
    output_dir : Path
        Harness output directory.
    payload : Dict[str, Any]
        JSON-serializable payload.

    Returns
    -------
    None
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def rows_path(output_dir: Path) -> Path:
    """Return the resumable row store path.

    Parameters
    ----------
    output_dir : Path
        Harness output directory.

    Returns
    -------
    Path
        JSONL row path.
    """
    return output_dir / "results.rows.jsonl"


def append_row(output_dir: Path, row: Dict[str, Any]) -> None:
    """Append one completed result row.

    Parameters
    ----------
    output_dir : Path
        Harness output directory.
    row : Dict[str, Any]
        Completed result row.

    Returns
    -------
    None
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with rows_path(output_dir).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True))
        handle.write("\n")


def load_jsonl_rows(output_dir: Path) -> List[Dict[str, Any]]:
    """Load resumable rows from disk.

    Parameters
    ----------
    output_dir : Path
        Harness output directory.

    Returns
    -------
    List[Dict[str, Any]]
        Completed rows.
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
    """Rewrite the resumable row store.

    Parameters
    ----------
    output_dir : Path
        Harness output directory.
    rows : Sequence[Dict[str, Any]]
        Rows to write.

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
    """Return the staging directory used before atomic publish.

    Parameters
    ----------
    output_dir : Path
        Final output directory.

    Returns
    -------
    Path
        Sibling staging directory.
    """
    return output_dir.with_name(f"{output_dir.name}.tmp")


def previous_dir(output_dir: Path) -> Path:
    """Return the temporary backup directory used during publish.

    Parameters
    ----------
    output_dir : Path
        Final output directory.

    Returns
    -------
    Path
        Sibling previous-output directory.
    """
    return output_dir.with_name(f"{output_dir.name}.prev")


def _write_worker_message(result_path: str, message: Dict[str, Any]) -> None:
    """Write a child-process result message.

    Parameters
    ----------
    result_path : str
        JSON path used by the child process.
    message : Dict[str, Any]
        Serializable result metadata.

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
    graph: DaguaGraph,
    engine_name: str,
    temp_position_path: str,
    result_path: str,
) -> None:
    """Run one external layout in an isolated child process.

    Parameters
    ----------
    graph : DaguaGraph
        Corpus graph to lay out.
    engine_name : str
        Registered competitor name.
    temp_position_path : str
        Temporary position tensor path.
    result_path : str
        Child status JSON path.

    Returns
    -------
    None
    """
    try:
        competitor = get_competitor(engine_name)
        if competitor is None:
            _write_worker_message(
                result_path,
                {"status": "ERROR", "runtime_s": 0.0, "error": "adapter not registered"},
            )
            os._exit(0)
        result = competitor.layout(graph, timeout=TIMEOUT_SECONDS, seed=SEED)
        if result.pos is None:
            _write_worker_message(
                result_path,
                {
                    "status": "ERROR",
                    "runtime_s": float(result.runtime_seconds),
                    "error": result.error or "adapter returned no positions",
                },
            )
            os._exit(0)
        torch.save(result.pos.detach().cpu().to(dtype=torch.float32), temp_position_path)
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


def make_skip_row(graph: LoadedGraph, engine_name: str, reason: str) -> Dict[str, Any]:
    """Create a skipped result row.

    Parameters
    ----------
    graph : LoadedGraph
        Corpus graph metadata.
    engine_name : str
        Engine name.
    reason : str
        Skip reason.

    Returns
    -------
    Dict[str, Any]
        Result row.
    """
    return {
        "graph": graph.name,
        "corpus": graph.corpus,
        "engine": engine_name,
        "status": "SKIP",
        "runtime_s": 0.0,
        "metrics": {},
        "reported_metrics": {},
        "composite": None,
        "positions_path": None,
        "nodes": graph.graph.num_nodes,
        "edges": int(graph.graph.edge_index.shape[1]),
        "directed": graph.directed,
        "source_path": str(graph.source_path),
        "reason": reason,
    }


def make_error_row(
    graph: LoadedGraph,
    engine_name: str,
    runtime_s: float,
    error: str,
    status_detail: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a failed result row.

    Parameters
    ----------
    graph : LoadedGraph
        Corpus graph metadata.
    engine_name : str
        Engine name.
    runtime_s : float
        Runtime before failure was observed.
    error : str
        Error detail.
    status_detail : str | None, default=None
        Optional machine-readable failure detail.

    Returns
    -------
    Dict[str, Any]
        Result row.
    """
    row = {
        "graph": graph.name,
        "corpus": graph.corpus,
        "engine": engine_name,
        "status": "ERROR",
        "runtime_s": float(runtime_s),
        "metrics": {},
        "reported_metrics": {},
        "composite": None,
        "positions_path": None,
        "nodes": graph.graph.num_nodes,
        "edges": int(graph.graph.edge_index.shape[1]),
        "directed": graph.directed,
        "source_path": str(graph.source_path),
        "error": error,
    }
    if status_detail is not None:
        row["status_detail"] = status_detail
    return row


def external_layout_result(
    graph: LoadedGraph,
    engine_name: str,
    output_dir: Path,
) -> Tuple[Optional[CompetitorResult], Optional[Dict[str, Any]]]:
    """Run one external layout with timeout isolation.

    Parameters
    ----------
    graph : LoadedGraph
        Corpus graph metadata.
    engine_name : str
        Registered external competitor name.
    output_dir : Path
        Output directory for temporary artifacts.

    Returns
    -------
    Tuple[CompetitorResult | None, Dict[str, Any] | None]
        Successful result, or an error row.
    """
    context = mp.get_context("fork")
    temp_path = output_dir / "positions" / f".{safe_component(graph.name)}__{engine_name}.child.pt"
    result_path = output_dir / "positions" / f".{safe_component(graph.name)}__{engine_name}.json"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    for path in (temp_path, result_path):
        if path.exists():
            path.unlink()
    process = context.Process(
        target=_external_layout_worker,
        args=(graph.graph, engine_name, str(temp_path), str(result_path)),
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
        for path in (temp_path, result_path):
            if path.exists():
                path.unlink()
        return None, make_error_row(graph, engine_name, elapsed, "timeout", "timeout")
    exitcode = process.exitcode
    if not result_path.is_file():
        if temp_path.exists():
            temp_path.unlink()
        detail = f"child exited {exitcode}" if exitcode is not None else "child exited"
        return None, make_error_row(graph, engine_name, elapsed, detail)
    with result_path.open("r", encoding="utf-8") as handle:
        message = json.load(handle)
    result_path.unlink()
    if message.get("status") != "OK":
        if temp_path.exists():
            temp_path.unlink()
        return None, make_error_row(
            graph,
            engine_name,
            float(message.get("runtime_s") or elapsed),
            str(message.get("error") or "external child error"),
        )
    position_path = Path(str(message.get("temp_positions_path")))
    if not position_path.is_file():
        return None, make_error_row(
            graph,
            engine_name,
            elapsed,
            "external child returned no positions",
        )
    positions = torch.load(position_path, map_location="cpu")
    position_path.unlink()
    return (
        CompetitorResult(
            name=engine_name,
            pos=positions,
            runtime_seconds=float(message.get("runtime_s") or elapsed),
            error=message.get("error"),
        ),
        None,
    )


def run_engine(graph: LoadedGraph, competitor: CompetitorBase, output_dir: Path) -> Dict[str, Any]:
    """Run one engine on one corpus graph.

    Parameters
    ----------
    graph : LoadedGraph
        Corpus graph metadata.
    competitor : CompetitorBase
        Registered layout engine adapter.
    output_dir : Path
        Output directory for positions.

    Returns
    -------
    Dict[str, Any]
        Completed result row.
    """
    if competitor.name in EXTERNAL_ENGINE_NAMES:
        result, error_row = external_layout_result(graph, competitor.name, output_dir)
        if error_row is not None:
            return error_row
        if result is None:
            return make_error_row(graph, competitor.name, 0.0, "external child failed")
    else:
        try:
            result = competitor.layout(graph.graph, timeout=TIMEOUT_SECONDS, seed=SEED)
        except Exception as exc:  # noqa: BLE001
            return make_error_row(graph, competitor.name, 0.0, f"{type(exc).__name__}: {exc}")

    base_row = {
        "graph": graph.name,
        "corpus": graph.corpus,
        "engine": competitor.name,
        "runtime_s": float(result.runtime_seconds),
        "nodes": graph.graph.num_nodes,
        "edges": int(graph.graph.edge_index.shape[1]),
        "directed": graph.directed,
        "source_path": str(graph.source_path),
    }
    if result.pos is None:
        return {
            **base_row,
            "status": "ERROR",
            "metrics": {},
            "reported_metrics": {},
            "composite": None,
            "positions_path": None,
            "error": result.error or "adapter returned no positions",
        }

    positions = result.pos.detach().cpu().to(dtype=torch.float32)
    try:
        metrics = evaluate(graph.graph, positions, tier="full")
        composite = composite_auto(metrics, graph.directed)
    except Exception as exc:  # noqa: BLE001
        return {
            **base_row,
            "status": "ERROR",
            "metrics": {},
            "reported_metrics": {},
            "composite": None,
            "positions_path": None,
            "error": f"metrics {type(exc).__name__}: {exc}",
        }

    relpath = position_relpath(graph.name, competitor.name)
    position_path = output_dir / relpath
    position_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(positions, position_path)
    clean_metrics = json_clean(metrics)
    return {
        **base_row,
        "status": "OK",
        "metrics": clean_metrics,
        "reported_metrics": {name: clean_metrics.get(name) for name in REPORT_METRICS},
        "composite": float(composite),
        "positions_path": relpath,
        "error": None,
    }


def engine_availability(engine_names: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Inspect availability for requested engines.

    Parameters
    ----------
    engine_names : Sequence[str]
        Engine names to check.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Availability metadata keyed by engine.
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
        availability[engine_name] = {
            "available": bool(available),
            "reason": None if available else "adapter unavailable",
        }
    return availability


def validate_store(output_dir: Path) -> None:
    """Validate that OK rows and position tensors match one-to-one.

    Parameters
    ----------
    output_dir : Path
        Published output directory.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If an OK row is missing its tensor or a tensor has no OK row.
    """
    payload = load_existing_results(output_dir)
    rows = payload.get("rows", [])
    ok_paths = {
        str(row.get("positions_path"))
        for row in rows
        if row.get("status") == "OK" and row.get("positions_path")
    }
    missing = sorted(path for path in ok_paths if not (output_dir / path).is_file())
    position_dir = output_dir / "positions"
    position_paths = (
        {str(path.relative_to(output_dir)) for path in position_dir.glob("*.pt") if path.is_file()}
        if position_dir.is_dir()
        else set()
    )
    orphaned = sorted(position_paths - ok_paths)
    if missing or orphaned:
        details = []
        if missing:
            details.append(f"missing position files for OK rows: {missing}")
        if orphaned:
            details.append(f"orphaned position files without OK rows: {orphaned}")
        raise RuntimeError("; ".join(details))


def graph_best_external(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Return the best external OK row for each graph.

    Parameters
    ----------
    rows : Iterable[Dict[str, Any]]
        Result rows.

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


def summarize_wtl(rows: List[Dict[str, Any]], corpus: Optional[str] = None) -> Tuple[int, int, int]:
    """Summarize Dagua wins, ties, and losses.

    Parameters
    ----------
    rows : List[Dict[str, Any]]
        Result rows.
    corpus : str | None, default=None
        Optional corpus filter.

    Returns
    -------
    Tuple[int, int, int]
        Win, tie, and loss counts.
    """
    filtered = [row for row in rows if corpus is None or row.get("corpus") == corpus]
    best_external = graph_best_external(filtered)
    dagua_rows = {
        str(row["graph"]): row
        for row in filtered
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


def write_markdown(output_dir: Path, payload: Dict[str, Any]) -> None:
    """Write the standard-corpora Markdown report.

    Parameters
    ----------
    output_dir : Path
        Harness output directory.
    payload : Dict[str, Any]
        Published result payload.

    Returns
    -------
    None
    """
    rows = list(payload.get("rows", []))
    corpora = sorted({str(row.get("corpus")) for row in rows if row.get("corpus")})
    lines = [
        "# Standard Corpora Heldout Evaluation",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Git SHA: {payload['git_sha']}",
        f"- Tie band: +/-{TIE_BAND:.1f} composite points",
        f"- Graphs loaded: {payload['graph_count']}",
        "",
        "## Dagua Native vs Best External",
        "",
        "| Corpus | W | T | L |",
        "|---|---:|---:|---:|",
    ]
    overall = summarize_wtl(rows)
    lines.append(f"| all | {overall[0]} | {overall[1]} | {overall[2]} |")
    for corpus in corpora:
        wins, ties, losses = summarize_wtl(rows, corpus)
        lines.append(f"| {corpus} | {wins} | {ties} | {losses} |")
    lines.extend(["", "## Rows", "", "| Graph | Corpus | Dagua | Best external | Delta | Result |"])
    lines.append("|---|---|---:|---:|---:|---|")
    best_external = graph_best_external(rows)
    dagua_rows = {
        str(row["graph"]): row
        for row in rows
        if row.get("engine") == "dagua" and row.get("status") == "OK"
    }
    for graph_name in sorted(dagua_rows):
        dagua_row = dagua_rows[graph_name]
        external_row = best_external.get(graph_name)
        if external_row is None:
            lines.append(
                f"| {graph_name} | {dagua_row.get('corpus')} | "
                f"{float(dagua_row['composite']):.2f} | n/a | n/a | no external OK |"
            )
            continue
        delta = float(dagua_row["composite"]) - float(external_row["composite"])
        label = "W" if delta > TIE_BAND else "T" if delta >= -TIE_BAND else "L"
        lines.append(
            f"| {graph_name} | {dagua_row.get('corpus')} | "
            f"{float(dagua_row['composite']):.2f} | "
            f"{float(external_row['composite']):.2f} ({external_row['engine']}) | "
            f"{delta:.2f} | {label} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Corpus graphs are capped at 2000 nodes by default.",
            "- Rome and SuiteSparse inputs are scored as undirected.",
            "- North DAG inputs are directed only when file metadata or path names say so.",
            "- Missing external engines are recorded as SKIP rows, not as losses.",
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "STDCORPORA.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def publish_results(output_dir: Path, staging: Path, payload: Dict[str, Any]) -> None:
    """Publish a staging directory atomically enough for local benchmark use.

    Parameters
    ----------
    output_dir : Path
        Final output directory.
    staging : Path
        Staging directory containing rows and positions.
    payload : Dict[str, Any]
        Final result payload.

    Returns
    -------
    None
    """
    write_results(staging, payload)
    write_markdown(staging, payload)
    previous = previous_dir(output_dir)
    if previous.exists():
        shutil.rmtree(previous)
    if output_dir.exists():
        output_dir.rename(previous)
    staging.rename(output_dir)
    if previous.exists():
        shutil.rmtree(previous)
    validate_store(output_dir)


def selected_engines(args: argparse.Namespace) -> List[str]:
    """Resolve the engine list for this run.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line options.

    Returns
    -------
    List[str]
        Engine names to run or preserve.
    """
    engines = list(args.engines) if args.engines else list(ENGINE_NAMES)
    if args.dagua_only and "dagua" not in engines:
        engines.insert(0, "dagua")
    return engines


def main() -> int:
    """Run the standard-corpora heldout harness.

    Returns
    -------
    int
        Process exit status.
    """
    args = parse_args()
    graphs = load_corpus(args.corpus_dir, args.max_nodes)
    if args.graphs:
        requested = set(args.graphs)
        graphs = [
            graph
            for graph in graphs
            if graph.name in requested or Path(graph.name).name in requested
        ]
    if not graphs:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        readme = args.output_dir / "README.md"
        if not readme.exists():
            readme.write_text(
                "# Standard Corpora Inputs\n\n"
                "Drop Rome/North `.graph` or `.gml` files and SuiteSparse `.mtx` files here, "
                "then rerun `scripts/r79_stdcorpora_eval.py`.\n",
                encoding="utf-8",
            )
        print(f"No supported graphs found under {args.corpus_dir}", file=sys.stderr)
        return 2

    engines = selected_engines(args)
    availability = engine_availability(engines)
    existing_rows: List[Dict[str, Any]] = []
    if args.dagua_only and (args.output_dir / "results.json").is_file():
        existing_rows = [
            row
            for row in load_existing_results(args.output_dir).get("rows", [])
            if row.get("engine") != "dagua"
        ]

    staging = staging_dir(args.output_dir)
    if staging.exists() and not args.resume:
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    completed = load_jsonl_rows(staging) if args.resume else []
    if not args.resume:
        write_jsonl_rows(staging, existing_rows)
        completed = list(existing_rows)
    completed_keys = {(str(row.get("graph")), str(row.get("engine"))) for row in completed}

    rows = list(completed)
    for graph in graphs:
        for engine_name in engines:
            if args.dagua_only and engine_name != "dagua":
                continue
            key = (graph.name, engine_name)
            if key in completed_keys:
                continue
            available = availability.get(engine_name, {"available": False, "reason": "not checked"})
            competitor = get_competitor(engine_name)
            if competitor is None or not available["available"]:
                reason = str(available.get("reason") or "unavailable")
                row = make_skip_row(graph, engine_name, reason)
            else:
                row = run_engine(graph, competitor, staging)
            append_row(staging, row)
            rows.append(row)
            completed_keys.add(key)
            print(f"{row['status']} {graph.name} {engine_name}")

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": git_sha(),
        "seed": SEED,
        "timeout_seconds": TIMEOUT_SECONDS,
        "max_nodes": args.max_nodes,
        "tie_band": TIE_BAND,
        "graph_count": len(graphs),
        "engines": engines,
        "availability": availability,
        "rows": rows,
    }
    publish_results(args.output_dir, staging, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
