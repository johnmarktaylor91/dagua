"""Verify NetworkX simple-layout source ports against NetworkX 3.6.1."""

# ruff: noqa: E402

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dagua.eval.equivalence_metrics import procrustes_rmsd
from dagua.graph import DaguaGraph
from dagua.layout.ops.networkx_simple import nx_bfs_layers, nx_bipartite_node_set
from dagua.layout.ops.pipelines.arf import layout_arf_pipeline
from dagua.layout.ops.pipelines.bfs import layout_bfs_pipeline
from dagua.layout.ops.pipelines.bipartite import layout_bipartite_pipeline
from dagua.layout.ops.pipelines.circular import layout_circular_pipeline
from dagua.layout.ops.pipelines.multipartite import layout_multipartite_pipeline
from dagua.layout.ops.pipelines.shell import layout_shell_pipeline
from dagua.layout.ops.pipelines.spiral import layout_spiral_pipeline

DEFAULT_REPORT = ROOT / "docs" / "algorithms" / "nx_batch_fidelity.md"
BIT_EXACT_THRESHOLD = 1.0e-12
POSITIONAL_THRESHOLD = 1.0e-6

PipelineFn = Callable[..., torch.Tensor]


def _graph_from_edges(name: str, num_nodes: int, edges: Sequence[Tuple[int, int]]) -> DaguaGraph:
    """Build one deterministic verification graph.

    Parameters
    ----------
    name : str
        Graph name used as a label prefix.
    num_nodes : int
        Number of graph nodes.
    edges : sequence[tuple[int, int]]
        Directed edge pairs.

    Returns
    -------
    DaguaGraph
        Graph with measured node sizes.
    """
    graph = DaguaGraph()
    for node in range(num_nodes):
        graph.add_node(node, label=f"{name}_{node}")
    for source, target in edges:
        graph.add_edge(source, target)
    graph.compute_node_sizes()
    return graph


def _random_dag_edges(num_nodes: int, edge_count: int, seed: int) -> List[Tuple[int, int]]:
    """Generate a deterministic acyclic edge sample.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    edge_count : int
        Requested distinct edge count.
    seed : int
        Python RNG seed.

    Returns
    -------
    list[tuple[int, int]]
        Sorted source-before-target edges.
    """
    candidates = [
        (source, target) for source in range(num_nodes) for target in range(source + 1, num_nodes)
    ]
    rng = random.Random(seed)
    return sorted(rng.sample(candidates, min(edge_count, len(candidates))))


def _verification_graphs() -> List[Tuple[str, DaguaGraph]]:
    """Build the fixed NetworkX-simple verification corpus.

    Returns
    -------
    list[tuple[str, DaguaGraph]]
        The 11 named graphs requested by the megasprint spec.
    """
    binary_tree = [(node, 2 * node + child) for node in range(5) for child in (1, 2)]
    grid = [(row * 5 + column, row * 5 + column + 1) for row in range(5) for column in range(4)] + [
        (row * 5 + column, (row + 1) * 5 + column) for row in range(4) for column in range(5)
    ]
    org_chart = [(0, node) for node in range(1, 6)] + [
        (manager, 6 + (manager - 1) * 2 + child) for manager in range(1, 6) for child in range(2)
    ]
    org_chart_deep: List[Tuple[int, int]] = []
    previous = [0]
    next_node = 1
    for width in (3, 5, 10, 20, 40):
        current = list(range(next_node, next_node + width))
        next_node += width
        for index, child in enumerate(current):
            org_chart_deep.append((previous[index % len(previous)], child))
        previous = current
    cases = [
        ("single_node", 1, []),
        ("small_chain", 6, [(node, node + 1) for node in range(5)]),
        ("binary_tree", 11, binary_tree),
        ("diamond", 4, [(0, 1), (0, 2), (1, 3), (2, 3)]),
        ("grid_5x5", 25, grid),
        ("org_chart_small", 16, org_chart),
        ("long_skip", 5, [(0, 1), (1, 2), (2, 3), (3, 4), (0, 3), (1, 4)]),
        ("disconnected", 5, [(0, 1), (2, 3)]),
        ("cycle_4", 4, [(0, 1), (1, 2), (2, 3), (3, 0)]),
        ("random_dag_50", 50, _random_dag_edges(50, 90, seed=4101)),
        ("org_chart_deep", 79, org_chart_deep),
    ]
    return [(name, _graph_from_edges(name, num_nodes, edges)) for name, num_nodes, edges in cases]


def _graph_to_nx(graph: DaguaGraph) -> Any:
    """Convert a Dagua graph to an integer-node NetworkX DiGraph.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph.

    Returns
    -------
    Any
        ``networkx.DiGraph`` with integer nodes.
    """
    import networkx as nx

    graph_nx = nx.DiGraph()
    graph_nx.add_nodes_from(range(graph.num_nodes))
    if graph.edge_index.numel() > 0:
        graph_nx.add_edges_from((int(s), int(t)) for s, t in graph.edge_index.t().tolist())
    return graph_nx


def _reference_positions(layout_name: str, graph: DaguaGraph) -> np.ndarray:
    """Run the pinned NetworkX reference layout.

    Parameters
    ----------
    layout_name : str
        Layout name under verification.
    graph : DaguaGraph
        Graph to lay out.

    Returns
    -------
    numpy.ndarray
        Reference coordinates with shape ``[N, 2]``.
    """
    import networkx as nx

    graph_nx = _graph_to_nx(graph)
    edge_index = graph.edge_index
    if layout_name == "circular":
        pos = nx.circular_layout(graph_nx)
    elif layout_name == "shell":
        pos = nx.shell_layout(graph_nx)
    elif layout_name == "spiral":
        pos = nx.spiral_layout(graph_nx)
    elif layout_name == "bipartite":
        pos = nx.bipartite_layout(
            graph_nx,
            nodes=nx_bipartite_node_set(edge_index, graph.num_nodes),
        )
    elif layout_name == "multipartite":
        pos = nx.multipartite_layout(
            graph_nx,
            subset_key=nx_bfs_layers(edge_index, graph.num_nodes),
        )
    elif layout_name == "bfs":
        layers = nx_bfs_layers(edge_index, graph.num_nodes)
        pos = nx.multipartite_layout(graph_nx, subset_key=layers)
    elif layout_name == "arf":
        pos = nx.arf_layout(graph_nx, seed=42)
    else:
        raise ValueError(f"Unsupported layout {layout_name!r}.")
    return np.vstack([pos[node] for node in range(graph.num_nodes)])


def _candidate_positions(layout_name: str, graph: DaguaGraph) -> np.ndarray:
    """Run the Dagua source-port layout.

    Parameters
    ----------
    layout_name : str
        Layout name under verification.
    graph : DaguaGraph
        Graph to lay out.

    Returns
    -------
    numpy.ndarray
        Candidate coordinates with shape ``[N, 2]``.
    """
    pipeline_by_name: Dict[str, PipelineFn] = {
        "circular": layout_circular_pipeline,
        "shell": layout_shell_pipeline,
        "spiral": layout_spiral_pipeline,
        "bipartite": layout_bipartite_pipeline,
        "multipartite": layout_multipartite_pipeline,
        "bfs": layout_bfs_pipeline,
        "arf": layout_arf_pipeline,
    }
    kwargs: Dict[str, Any] = {"seed": 42} if layout_name == "arf" else {}
    positions = pipeline_by_name[layout_name](
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        node_sizes=graph.node_sizes,
        **kwargs,
    )
    return positions.detach().cpu().numpy()


def _classify(max_abs: float) -> str:
    """Classify a direct coordinate residual.

    Parameters
    ----------
    max_abs : float
        Maximum absolute coordinate difference.

    Returns
    -------
    str
        ``bit-exact``, ``positional``, or ``divergent``.
    """
    if max_abs <= BIT_EXACT_THRESHOLD:
        return "bit-exact"
    if max_abs <= POSITIONAL_THRESHOLD:
        return "positional"
    return "divergent"


def _compare_layouts() -> List[Dict[str, Any]]:
    """Compare all layouts on the verification corpus.

    Returns
    -------
    list[dict[str, Any]]
        Per-layout, per-graph residual rows.
    """
    rows: List[Dict[str, Any]] = []
    for layout_name in ("circular", "shell", "spiral", "bipartite", "multipartite", "bfs", "arf"):
        for graph_name, graph in _verification_graphs():
            reference = _reference_positions(layout_name, graph)
            candidate = _candidate_positions(layout_name, graph)
            max_abs = float(np.max(np.abs(candidate - reference))) if graph.num_nodes else 0.0
            residual = procrustes_rmsd(candidate, reference)
            rows.append(
                {
                    "layout": layout_name,
                    "graph": graph_name,
                    "max_abs": max_abs,
                    "d_R": residual,
                    "class": _classify(max_abs),
                }
            )
    return rows


def _write_report(rows: List[Dict[str, Any]], path: Path) -> None:
    """Write the markdown fidelity report.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Comparison rows from :func:`_compare_layouts`.
    path : pathlib.Path
        Markdown report path.

    Returns
    -------
    None
        The report is written to disk.
    """
    lines = [
        "# NetworkX simple-layout batch fidelity",
        "",
        "Reference: `networkx.drawing.layout` from NetworkX 3.6.1.",
        "",
        "Pinned fallbacks: bipartite uses BFS parity from node 0 as the explicit node set; "
        "multipartite and bfs use BFS-distance layers from node 0, appending disconnected "
        "components in node order so every Dagua tensor node has a position.",
        "",
        "| Layout | Graph | d_R | max_abs | Class |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['layout']} | {row['graph']} | {row['d_R']:.3e} | "
            f"{row['max_abs']:.3e} | {row['class']} |"
        )
    lines.extend(["", "## Summary", ""])
    for layout_name in ("circular", "shell", "spiral", "bipartite", "multipartite", "bfs", "arf"):
        layout_rows = [row for row in rows if row["layout"] == layout_name]
        bit_exact = sum(row["class"] == "bit-exact" for row in layout_rows)
        positional = sum(row["class"] == "positional" for row in layout_rows)
        divergent = sum(row["class"] == "divergent" for row in layout_rows)
        max_dr = max(float(row["d_R"]) for row in layout_rows)
        lines.append(
            f"- `{layout_name}`: {bit_exact} bit-exact, {positional} positional, "
            f"{divergent} divergent; max d_R={max_dr:.3e}."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    """Run verification and update the markdown report.

    Returns
    -------
    None
        Results are printed and written to ``docs/algorithms``.
    """
    rows = _compare_layouts()
    _write_report(rows, DEFAULT_REPORT)
    for layout_name in ("circular", "shell", "spiral", "bipartite", "multipartite", "bfs", "arf"):
        layout_rows = [row for row in rows if row["layout"] == layout_name]
        bit_exact = sum(row["class"] == "bit-exact" for row in layout_rows)
        positional = sum(row["class"] == "positional" for row in layout_rows)
        divergent = sum(row["class"] == "divergent" for row in layout_rows)
        max_dr = max(float(row["d_R"]) for row in layout_rows)
        print(
            f"{layout_name}: bit-exact={bit_exact}/11 positional={positional}/11 "
            f"divergent={divergent}/11 max_d_R={max_dr:.3e}"
        )
    print(f"wrote {DEFAULT_REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
