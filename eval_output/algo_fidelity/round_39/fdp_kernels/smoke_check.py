"""Round 39 smoke checks for Graphviz fdp kernel fidelity."""

from __future__ import annotations

import statistics
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from dagua.eval.competitors import get_competitor  # noqa: E402
from dagua.graph import DaguaGraph  # noqa: E402
from scripts.fidelity_analysis import fidelity_procrustes  # noqa: E402

SEEDS: Tuple[int, ...] = (1, 2, 3)
TopologyBuilder = Callable[[], DaguaGraph]


def build_path_graph() -> DaguaGraph:
    """Build an unclustered path topology for flat ``tLayout`` coverage.

    Returns
    -------
    DaguaGraph
        Eight-node directed path with computed node sizes.
    """
    graph = DaguaGraph()
    for index in range(8):
        graph.add_node(f"n{index}")
    for index in range(7):
        graph.add_edge(f"n{index}", f"n{index + 1}")
    graph.compute_node_sizes()
    return graph


def build_clustered_path_graph() -> DaguaGraph:
    """Build a two-cluster path topology for recursive fdp coverage.

    Returns
    -------
    DaguaGraph
        Eight-node path split into two sibling clusters.
    """
    graph = build_path_graph()
    graph.add_cluster("left", [f"n{index}" for index in range(4)])
    graph.add_cluster("right", [f"n{index}" for index in range(4, 8)])
    return graph


def build_multi_cluster_graph() -> DaguaGraph:
    """Build a multi-cluster topology with inter-cluster handoff edges.

    Returns
    -------
    DaguaGraph
        Twelve-node graph arranged as three clustered paths.
    """
    graph = DaguaGraph()
    for index in range(12):
        graph.add_node(f"n{index}")
    for base in (0, 4, 8):
        for offset in range(3):
            graph.add_edge(f"n{base + offset}", f"n{base + offset + 1}")
    graph.add_edge("n3", "n4")
    graph.add_edge("n7", "n8")
    graph.add_edge("n2", "n9")
    graph.add_cluster("alpha", ["n0", "n1", "n2", "n3"])
    graph.add_cluster("beta", ["n4", "n5", "n6", "n7"])
    graph.add_cluster("gamma", ["n8", "n9", "n10", "n11"])
    graph.compute_node_sizes()
    return graph


def smoke_rmsd(graph: DaguaGraph, seed: int) -> float:
    """Compare Dagua fdp fidelity mode with Graphviz fdp for one graph.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to lay out.
    seed : int
        Seed passed to both the Dagua and Graphviz fdp adapters.

    Returns
    -------
    float
        Scale-normalized Procrustes RMSD.

    Raises
    ------
    RuntimeError
        If either competitor adapter is unavailable or fails.
    """
    dagua = get_competitor("classic_fmmm")
    graphviz = get_competitor("graphviz_fdp")
    if dagua is None or graphviz is None:
        raise RuntimeError("Required competitors classic_fmmm and graphviz_fdp are not registered.")

    dagua_result = dagua.layout_with_variant(
        graph,
        timeout=60.0,
        seed=seed,
        variant_params={"steps": 200, "fidelity_mode": True},
    )
    if dagua_result.pos is None:
        raise RuntimeError(f"Dagua fdp fidelity failed: {dagua_result.error}")
    graphviz_result = graphviz.layout(graph, timeout=60.0, seed=seed)
    if graphviz_result.pos is None:
        raise RuntimeError(f"Graphviz fdp failed: {graphviz_result.error}")

    rmsd, _, _, _ = fidelity_procrustes(
        dagua_result.pos.to(dtype=torch.float64),
        graphviz_result.pos.to(dtype=torch.float64),
    )
    return float(rmsd)


def run_smoke_checks() -> Dict[str, List[float]]:
    """Run all Round 39 smoke checks.

    Returns
    -------
    dict[str, list[float]]
        RMSDs keyed by topology name, with one value per seed.
    """
    builders: Tuple[Tuple[str, TopologyBuilder], ...] = (
        ("path", build_path_graph),
        ("clustered", build_clustered_path_graph),
        ("multi_cluster", build_multi_cluster_graph),
    )
    results: Dict[str, List[float]] = {}
    for name, builder in builders:
        results[name] = [smoke_rmsd(builder(), seed) for seed in SEEDS]
    return results


def main() -> None:
    """Print Round 39 smoke RMSDs by topology and seed."""
    results = run_smoke_checks()
    for topology, values in results.items():
        formatted = ", ".join(f"{value:.9f}" for value in values)
        print(
            f"{topology}: {formatted} (mean={statistics.fmean(values):.9f}, max={max(values):.9f})"
        )


if __name__ == "__main__":
    main()
