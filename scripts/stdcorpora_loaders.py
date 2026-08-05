"""Standard-corpora graph-file loaders shared by r79 and the GLaDOS runner.

Extracted VERBATIM from ``scripts/r79_stdcorpora_eval.py`` (WP-25 loader
extraction, GLADOS_RUNNER_SPEC.md section 3). The r79 harness re-imports every
name so its pinned CLI behavior is unchanged; the GLaDOS holdout runner imports
the same loaders and passes ``directed_override`` per its explicit
per-corpus directedness policy (never the path-substring heuristic).

The only permitted signature extension over r79 is ``directed_override`` on
each ``load_*_file`` loader: ``None`` (the default, and what r79 passes)
preserves the historical inference exactly; a bool bypasses inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

from dagua.graph import DaguaGraph

MAX_NODES = 2000


@dataclass(frozen=True)
class LoadedGraph:
    """Loaded corpus graph with normalized integer edges."""

    name: str
    corpus: str
    graph: DaguaGraph
    source_path: Path
    directed: bool


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


def load_graph_file(path: Path, directed_override: Optional[bool] = None) -> LoadedGraph:
    """Load a Rome/North ``.graph`` text file.

    Parameters
    ----------
    path : Path
        Source file path.
    directed_override : bool | None, default=None
        When not ``None``, force this directedness instead of the inferred
        value. ``None`` preserves r79's historical path-based inference.

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

    directed = directed_override if directed_override is not None else infer_directed(path)
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


def load_gml_file(path: Path, directed_override: Optional[bool] = None) -> LoadedGraph:
    """Load a GML file through NetworkX.

    Parameters
    ----------
    path : Path
        Source GML file path.
    directed_override : bool | None, default=None
        When not ``None``, force this directedness instead of the inferred
        value. ``None`` preserves r79's historical format-trusting inference.

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
    directed = (
        directed_override
        if directed_override is not None
        else infer_directed(path, nx_graph.is_directed())
    )
    corpus = infer_corpus(path)
    return build_graph(f"{corpus}/{path.stem}", corpus, len(nodes), edges, directed, path)


def load_graphml_file(path: Path, directed_override: Optional[bool] = None) -> LoadedGraph:
    """Load a GraphML (XML) file through NetworkX.

    Parameters
    ----------
    path : Path
        Source GraphML file path.
    directed_override : bool | None, default=None
        When not ``None``, force this directedness instead of the inferred
        value. ``None`` preserves r79's historical inference (including the
        North edgedefault-missing fallback).

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
    directed = (
        directed_override
        if directed_override is not None
        else infer_directed(path, nx_graph.is_directed() if "north" not in corpus else None)
    )
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


def load_mtx_file(path: Path, directed_override: Optional[bool] = None) -> LoadedGraph:
    """Load a SuiteSparse Matrix Market file as an undirected sparsity graph.

    Parameters
    ----------
    path : Path
        Source Matrix Market file.
    directed_override : bool | None, default=None
        When not ``None``, force this directedness instead of the historical
        always-undirected default.

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
    directed = directed_override if directed_override is not None else False
    return build_graph(f"{corpus}/{path.stem}", corpus, node_count, edges, directed, path)


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
