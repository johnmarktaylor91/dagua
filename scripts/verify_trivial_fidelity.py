"""Verify trivial deterministic layout fidelity."""

# ruff: noqa: E402

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dagua.eval.equivalence_metrics import procrustes_rmsd
from dagua.graph import DaguaGraph
from dagua.layout.ops.networkx_simple import (
    nx_arc_positions,
    nx_circlepack_positions,
    nx_concentric_positions,
    nx_star_positions,
)
from dagua.layout.ops.pipelines.arc import layout_arc_pipeline
from dagua.layout.ops.pipelines.circlepack import layout_circlepack_pipeline
from dagua.layout.ops.pipelines.concentric import layout_concentric_pipeline
from dagua.layout.ops.pipelines.osage import layout_osage_pipeline
from dagua.layout.ops.pipelines.planar import layout_planar_pipeline
from dagua.layout.ops.pipelines.star import layout_star_pipeline

DEFAULT_REPORT = ROOT / "docs" / "algorithms" / "trivial_fidelity.md"
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
    """Build the fixed trivial-layout verification corpus.

    Returns
    -------
    list[tuple[str, DaguaGraph]]
        Named graph corpus.
    """
    binary_tree = [(node, 2 * node + child) for node in range(5) for child in (1, 2)]
    grid = [(row * 5 + column, row * 5 + column + 1) for row in range(5) for column in range(4)] + [
        (row * 5 + column, (row + 1) * 5 + column) for row in range(4) for column in range(5)
    ]
    return [
        ("single_node", _graph_from_edges("single_node", 1, [])),
        (
            "small_chain",
            _graph_from_edges("small_chain", 6, [(node, node + 1) for node in range(5)]),
        ),
        ("binary_tree", _graph_from_edges("binary_tree", 11, binary_tree)),
        ("diamond", _graph_from_edges("diamond", 4, [(0, 1), (0, 2), (1, 3), (2, 3)])),
        ("grid_5x5", _graph_from_edges("grid_5x5", 25, grid)),
        (
            "long_skip",
            _graph_from_edges(
                "long_skip",
                5,
                [(0, 1), (1, 2), (2, 3), (3, 4), (0, 3), (1, 4)],
            ),
        ),
        ("disconnected", _graph_from_edges("disconnected", 5, [(0, 1), (2, 3)])),
        ("cycle_4", _graph_from_edges("cycle_4", 4, [(0, 1), (1, 2), (2, 3), (3, 0)])),
        (
            "random_dag_50",
            _graph_from_edges("random_dag_50", 50, _random_dag_edges(50, 90, seed=4101)),
        ),
        (
            "k5_non_planar",
            _graph_from_edges(
                "k5_non_planar",
                5,
                [(s, t) for s in range(5) for t in range(s + 1, 5)],
            ),
        ),
    ]


def _graph_to_nx_graph(graph: DaguaGraph) -> Any:
    """Convert a Dagua graph to an integer-node NetworkX graph.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph.

    Returns
    -------
    Any
        ``networkx.Graph`` with integer nodes.
    """
    nx = __import__("networkx")
    graph_nx = nx.Graph()
    graph_nx.add_nodes_from(range(graph.num_nodes))
    if graph.edge_index.numel() > 0:
        graph_nx.add_edges_from((int(s), int(t)) for s, t in graph.edge_index.t().tolist())
    return graph_nx


def _reference_positions(
    layout_name: str,
    graph: DaguaGraph,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Run the pinned reference layout.

    Parameters
    ----------
    layout_name : str
        Layout name under verification.
    graph : DaguaGraph
        Graph to lay out.

    Returns
    -------
    tuple[numpy.ndarray | None, str | None]
        Reference coordinates, or an N/A reason.
    """
    if layout_name == "star":
        return nx_star_positions(graph.edge_index, graph.num_nodes).numpy(), None
    if layout_name == "concentric":
        return nx_concentric_positions(graph.edge_index, graph.num_nodes).numpy(), None
    if layout_name == "circlepack":
        return nx_circlepack_positions(graph.num_nodes).numpy(), None
    if layout_name == "arc":
        return nx_arc_positions(graph.edge_index, graph.num_nodes).numpy(), None
    if layout_name == "osage":
        return nx_arc_positions(graph.edge_index, graph.num_nodes).numpy(), None
    if layout_name == "planar":
        nx = __import__("networkx")
        graph_nx = _graph_to_nx_graph(graph)
        try:
            pos = nx.planar_layout(graph_nx)
        except Exception as exc:
            return None, str(exc)
        return np.vstack([pos[node] for node in range(graph.num_nodes)]), None
    raise ValueError(f"Unsupported layout {layout_name!r}.")


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
        "star": layout_star_pipeline,
        "concentric": layout_concentric_pipeline,
        "circlepack": layout_circlepack_pipeline,
        "arc": layout_arc_pipeline,
        "osage": layout_osage_pipeline,
        "planar": layout_planar_pipeline,
    }
    positions = pipeline_by_name[layout_name](
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        node_sizes=graph.node_sizes,
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
    """Compare all trivial layouts on the verification corpus.

    Returns
    -------
    list[dict[str, Any]]
        Per-layout, per-graph residual rows.
    """
    rows: List[Dict[str, Any]] = []
    for layout_name in ("star", "concentric", "circlepack", "osage", "arc", "planar"):
        for graph_name, graph in _verification_graphs():
            reference, reason = _reference_positions(layout_name, graph)
            if reference is None:
                rows.append(
                    {
                        "layout": layout_name,
                        "graph": graph_name,
                        "max_abs": float("nan"),
                        "d_R": float("nan"),
                        "class": "N/A",
                        "reason": reason,
                    }
                )
                continue
            try:
                candidate = _candidate_positions(layout_name, graph)
            except Exception as exc:
                rows.append(
                    {
                        "layout": layout_name,
                        "graph": graph_name,
                        "max_abs": float("nan"),
                        "d_R": float("nan"),
                        "class": "N/A",
                        "reason": str(exc),
                    }
                )
                continue
            max_abs = float(np.max(np.abs(candidate - reference))) if graph.num_nodes else 0.0
            residual = procrustes_rmsd(candidate, reference)
            rows.append(
                {
                    "layout": layout_name,
                    "graph": graph_name,
                    "max_abs": max_abs,
                    "d_R": residual,
                    "class": _classify(max_abs),
                    "reason": "",
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
        "# Trivial deterministic layout fidelity",
        "",
        "References: igraph-style star angles; documented degree-ring concentric; "
        "documented one-level circlepack; Graphviz-osage placeholder uses the same "
        "deterministic ordering as production pending a source port; standard BFS arc "
        "ordering; NetworkX 3.6.1 planar_layout for Chrobak-Payne planar.",
        "",
        "| Layout | Graph | d_R | max_abs | Class | Reason |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        d_r = "N/A" if row["class"] == "N/A" else f"{row['d_R']:.3e}"
        max_abs = "N/A" if row["class"] == "N/A" else f"{row['max_abs']:.3e}"
        lines.append(
            f"| {row['layout']} | {row['graph']} | {d_r} | {max_abs} | "
            f"{row['class']} | {row['reason']} |"
        )
    lines.extend(["", "## Summary", ""])
    for layout_name in ("star", "concentric", "circlepack", "osage", "arc", "planar"):
        layout_rows = [row for row in rows if row["layout"] == layout_name]
        bit_exact = sum(row["class"] == "bit-exact" for row in layout_rows)
        positional = sum(row["class"] == "positional" for row in layout_rows)
        na_count = sum(row["class"] == "N/A" for row in layout_rows)
        comparable = [row for row in layout_rows if row["class"] != "N/A"]
        max_dr = max((float(row["d_R"]) for row in comparable), default=float("nan"))
        lines.append(
            f"- `{layout_name}`: {bit_exact} bit-exact, {positional} positional, "
            f"{na_count} N/A; max d_R={max_dr:.3e}."
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
    for layout_name in ("star", "concentric", "circlepack", "osage", "arc", "planar"):
        layout_rows = [row for row in rows if row["layout"] == layout_name]
        bit_exact = sum(row["class"] == "bit-exact" for row in layout_rows)
        positional = sum(row["class"] == "positional" for row in layout_rows)
        na_count = sum(row["class"] == "N/A" for row in layout_rows)
        total = len(layout_rows)
        comparable = [row for row in layout_rows if row["class"] != "N/A"]
        max_dr = max((float(row["d_R"]) for row in comparable), default=float("nan"))
        print(
            f"{layout_name}: bit-exact={bit_exact}/{total} positional={positional}/{total} "
            f"N-A={na_count}/{total} max_d_R={max_dr:.3e}"
        )
    print(f"wrote {DEFAULT_REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
