"""Round 39 neato PCA/CG fidelity smoke and component diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from dagua.eval.competitors import get_competitor  # noqa: E402
from dagua.graph import DaguaGraph  # noqa: E402
from dagua.layout.ops.pipelines.neato import (  # noqa: E402
    layout_neato_pipeline,
    remove_neato_overlap_fidelity,
)
from dagua.layout.ops.pipelines.stress_majorization import (  # noqa: E402
    CURRENT_POSITIONS_KEY,
    GraphvizCgSmacofStep,
    GraphvizPrepareStressMajorizationState,
    _graphviz_normalize_pca_positions,
    _graphviz_pca_project_distances,
    _graphviz_random_initialize_positions,
)
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState  # noqa: E402
from scripts.fidelity_analysis import fidelity_procrustes  # noqa: E402

SEEDS: tuple[int, ...] = (1, 2, 3)
TOPOLOGIES: tuple[str, ...] = ("path", "star", "clustered", "grid")


def build_path_graph(num_nodes: int = 8) -> DaguaGraph:
    """Build a path graph.

    Parameters
    ----------
    num_nodes : int, default=8
        Number of nodes.

    Returns
    -------
    DaguaGraph
        Path graph with computed node sizes.
    """
    graph = DaguaGraph()
    for index in range(num_nodes):
        graph.add_node(f"n{index}")
    for index in range(num_nodes - 1):
        graph.add_edge(f"n{index}", f"n{index + 1}")
    graph.compute_node_sizes()
    return graph


def build_star_graph(num_leaves: int = 9) -> DaguaGraph:
    """Build a hub-and-spoke graph.

    Parameters
    ----------
    num_leaves : int, default=9
        Number of leaves connected to the center.

    Returns
    -------
    DaguaGraph
        Star graph with computed node sizes.
    """
    graph = DaguaGraph()
    graph.add_node("center")
    for index in range(num_leaves):
        graph.add_node(f"leaf{index}")
        graph.add_edge("center", f"leaf{index}")
    graph.compute_node_sizes()
    return graph


def build_clustered_graph(num_nodes: int = 10) -> DaguaGraph:
    """Build a two-cluster path graph.

    Parameters
    ----------
    num_nodes : int, default=10
        Number of path nodes.

    Returns
    -------
    DaguaGraph
        Clustered graph with computed node sizes.
    """
    graph = build_path_graph(num_nodes=num_nodes)
    midpoint = num_nodes // 2
    graph.add_cluster("left", [f"n{index}" for index in range(midpoint)])
    graph.add_cluster("right", [f"n{index}" for index in range(midpoint, num_nodes)])
    graph.compute_node_sizes()
    return graph


def build_grid_graph(width: int = 4, height: int = 3) -> DaguaGraph:
    """Build a rectangular grid graph.

    Parameters
    ----------
    width : int, default=4
        Number of grid columns.
    height : int, default=3
        Number of grid rows.

    Returns
    -------
    DaguaGraph
        Grid graph with computed node sizes.
    """
    graph = DaguaGraph()
    for row in range(height):
        for col in range(width):
            graph.add_node(f"n{row}_{col}")
    for row in range(height):
        for col in range(width):
            if col + 1 < width:
                graph.add_edge(f"n{row}_{col}", f"n{row}_{col + 1}")
            if row + 1 < height:
                graph.add_edge(f"n{row}_{col}", f"n{row + 1}_{col}")
    graph.compute_node_sizes()
    return graph


def build_graphs() -> Dict[str, DaguaGraph]:
    """Build all smoke-check topologies.

    Returns
    -------
    dict[str, DaguaGraph]
        Topology name to graph.
    """
    return {
        "path": build_path_graph(),
        "star": build_star_graph(),
        "clustered": build_clustered_graph(),
        "grid": build_grid_graph(),
    }


def rmsd(left: torch.Tensor, right: torch.Tensor) -> float:
    """Compute scale-normalized Procrustes RMSD.

    Parameters
    ----------
    left : torch.Tensor
        First position tensor with shape ``[N, 2]``.
    right : torch.Tensor
        Second position tensor with shape ``[N, 2]``.

    Returns
    -------
    float
        Procrustes RMSD.
    """
    value, _, _, _ = fidelity_procrustes(
        left.to(dtype=torch.float64),
        right.to(dtype=torch.float64),
    )
    return float(value)


def graphviz_reference(graph: DaguaGraph, seed: int) -> torch.Tensor:
    """Run the Graphviz neato reference.

    Parameters
    ----------
    graph : DaguaGraph
        Input graph.
    seed : int
        Seed passed through Graphviz's ``start`` attribute.

    Returns
    -------
    torch.Tensor
        Reference positions with shape ``[N, 2]``.

    Raises
    ------
    RuntimeError
        If the Graphviz adapter fails.
    """
    competitor = get_competitor("graphviz_neato")
    if competitor is None:
        raise RuntimeError("Missing graphviz_neato competitor.")
    result = competitor.layout(graph, timeout=60.0, seed=seed)
    if result.pos is None:
        raise RuntimeError(f"Graphviz neato failed: {result.error}")
    return result.pos


def run_neato_mode(graph: DaguaGraph, seed: int, fidelity_mode: str) -> torch.Tensor:
    """Run one Dagua neato fidelity mode.

    Parameters
    ----------
    graph : DaguaGraph
        Input graph.
    seed : int
        Solver seed.
    fidelity_mode : str
        Neato fidelity mode.

    Returns
    -------
    torch.Tensor
        Layout positions with shape ``[N, 2]``.
    """
    positions = layout_neato_pipeline(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        node_sizes=graph.node_sizes,
        seed=seed,
        maxiter=200,
        epsilon=0.0001,
        pack=True,
        fidelity_mode=fidelity_mode,
    )
    if not isinstance(positions, torch.Tensor):
        raise RuntimeError("Expected tensor positions.")
    return positions


def run_graphviz_cg_stage(graph: DaguaGraph, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the packed-CG solver before and after overlap removal.

    Parameters
    ----------
    graph : DaguaGraph
        Input graph.
    seed : int
        Solver seed.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Pre-overlap and post-overlap positions.
    """
    problem = LayoutProblem(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        node_sizes=graph.node_sizes,
        edge_weights=None,
        seed=seed,
    )
    state = SolveState()
    ctx = RuntimeContext()
    state = GraphvizPrepareStressMajorizationState().apply(problem, state, ctx)
    state.extras[CURRENT_POSITIONS_KEY] = _graphviz_random_initialize_positions(
        num_nodes=graph.num_nodes,
        dimensions=2,
        seed=seed,
    )
    step = GraphvizCgSmacofStep(epsilon=0.0001)
    for _ in range(200):
        state = step.apply(problem, state, ctx)
        if state.converged:
            break
    current = state.extras[CURRENT_POSITIONS_KEY]
    if not isinstance(current, np.ndarray):
        raise RuntimeError("CG stage did not produce numpy positions.")
    pre_overlap = torch.from_numpy(current.copy()).to(dtype=torch.float32)
    post_overlap = remove_neato_overlap_fidelity(
        positions=pre_overlap,
        node_sizes=graph.node_sizes,
    )
    return pre_overlap, post_overlap


def component_diagnostics(
    graph: DaguaGraph,
    reference: torch.Tensor,
    seed: int,
) -> Dict[str, float]:
    """Compute component-level RMSD diagnostics.

    Parameters
    ----------
    graph : DaguaGraph
        Input graph.
    reference : torch.Tensor
        Graphviz neato reference positions.
    seed : int
        Solver seed.

    Returns
    -------
    dict[str, float]
        RMSD values for initialization, CG, and overlap stages.
    """
    problem = LayoutProblem(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        node_sizes=graph.node_sizes,
        edge_weights=None,
        seed=seed,
    )
    state = GraphvizPrepareStressMajorizationState().apply(problem, SolveState(), RuntimeContext())
    if not isinstance(state.distance_matrix, torch.Tensor):
        raise RuntimeError("Distance preparation failed.")
    distances = state.distance_matrix.to(dtype=torch.float64).numpy()
    pca = torch.from_numpy(
        _graphviz_normalize_pca_positions(_graphviz_pca_project_distances(distances, dimensions=2))
    )
    random_init = torch.from_numpy(_graphviz_random_initialize_positions(graph.num_nodes, 2, seed))
    pre_overlap, post_overlap = run_graphviz_cg_stage(graph=graph, seed=seed)
    return {
        "pca_init_vs_graphviz_final": rmsd(pca, reference),
        "random_init_vs_graphviz_final": rmsd(random_init, reference),
        "cg_pre_overlap_vs_graphviz_final": rmsd(pre_overlap, reference),
        "cg_post_overlap_vs_graphviz_final": rmsd(post_overlap, reference),
    }


def summarize(values: Iterable[float]) -> float:
    """Return the mean of numeric values.

    Parameters
    ----------
    values : Iterable[float]
        Values to average.

    Returns
    -------
    float
        Mean value.
    """
    items = list(values)
    return float(sum(items) / len(items)) if items else float("nan")


def run_smoke() -> List[Dict[str, float | int | str]]:
    """Run the full smoke matrix.

    Returns
    -------
    list[dict[str, float | int | str]]
        Per-topology, per-seed smoke results.
    """
    rows: List[Dict[str, float | int | str]] = []
    for topology, graph in build_graphs().items():
        for seed in SEEDS:
            reference = graphviz_reference(graph=graph, seed=seed)
            graphviz_positions = run_neato_mode(graph=graph, seed=seed, fidelity_mode="graphviz")
            compat_positions = run_neato_mode(
                graph=graph,
                seed=seed,
                fidelity_mode="graphviz_neato",
            )
            row: Dict[str, float | int | str] = {
                "topology": topology,
                "seed": seed,
                "graphviz_rmsd": rmsd(graphviz_positions, reference),
                "compat_rmsd": rmsd(compat_positions, reference),
            }
            row.update(component_diagnostics(graph=graph, reference=reference, seed=seed))
            rows.append(row)
    return rows


def main() -> None:
    """Print the smoke matrix and aggregate RMSDs."""
    rows = run_smoke()
    print("topology seed graphviz compat pca_init random_init cg_pre_overlap cg_post_overlap")
    for row in rows:
        print(
            f"{row['topology']} {row['seed']} "
            f"{row['graphviz_rmsd']:.9f} {row['compat_rmsd']:.9f} "
            f"{row['pca_init_vs_graphviz_final']:.9f} "
            f"{row['random_init_vs_graphviz_final']:.9f} "
            f"{row['cg_pre_overlap_vs_graphviz_final']:.9f} "
            f"{row['cg_post_overlap_vs_graphviz_final']:.9f}"
        )
    print()
    for topology in TOPOLOGIES:
        matching = [row for row in rows if row["topology"] == topology]
        print(
            f"{topology}: graphviz_mean={summarize(row['graphviz_rmsd'] for row in matching):.9f} "
            f"compat_mean={summarize(row['compat_rmsd'] for row in matching):.9f}"
        )
    print(
        f"overall: graphviz_mean={summarize(row['graphviz_rmsd'] for row in rows):.9f} "
        f"compat_mean={summarize(row['compat_rmsd'] for row in rows):.9f}"
    )


if __name__ == "__main__":
    main()
