"""Verify graph-geodesic t-SNE fidelity against the sklearn competitor."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dagua.eval.competitors.tsne_competitor import TSNEGraph  # noqa: E402
from dagua.eval.equivalence_metrics import procrustes_rmsd  # noqa: E402
from dagua.graph import DaguaGraph  # noqa: E402
from dagua.layout.ops.pipelines.tsne_graph import layout_tsne_graph_pipeline  # noqa: E402


@dataclass(frozen=True)
class FidelityCase:
    """One graph case for t-SNE fidelity verification.

    Parameters
    ----------
    name : str
        Case name printed in the report.
    build : Callable[[], DaguaGraph]
        Factory returning the graph to evaluate.
    perplexity : float
        Requested sklearn t-SNE perplexity.
    seed : int
        Random seed for t-SNE initialization.
    """

    name: str
    build: Callable[[], DaguaGraph]
    perplexity: float
    seed: int


@dataclass(frozen=True)
class FidelityResult:
    """One graph t-SNE fidelity result.

    Parameters
    ----------
    name : str
        Case name.
    residual : float
        Rotation/reflection/scale-invariant Procrustes residual.
    max_abs_diff : float
        Maximum coordinate absolute difference before alignment.
    tier : str
        Fidelity tier label.
    """

    name: str
    residual: float
    max_abs_diff: float
    tier: str


def _graph_from_edges(num_nodes: int, edges: list[tuple[int, int]]) -> DaguaGraph:
    """Build a graph with integer node identifiers.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    edges : list[tuple[int, int]]
        Edge pairs.

    Returns
    -------
    DaguaGraph
        Graph with computed node sizes.
    """
    graph = DaguaGraph.from_edge_list(edges, num_nodes=num_nodes)
    graph.compute_node_sizes()
    return graph


def _grid_5x5() -> DaguaGraph:
    """Build the 5x5 grid fidelity graph.

    Returns
    -------
    DaguaGraph
        Undirected grid represented with one directed edge per adjacency.
    """
    edges: list[tuple[int, int]] = []
    width = 5
    for row in range(width):
        for col in range(width):
            node = row * width + col
            if col + 1 < width:
                edges.append((node, node + 1))
            if row + 1 < width:
                edges.append((node, node + width))
    return _graph_from_edges(width * width, edges)


def _random_dag_50() -> DaguaGraph:
    """Build a deterministic sparse random DAG.

    Returns
    -------
    DaguaGraph
        Fifty-node DAG with fixed pseudo-random forward edges.
    """
    rng = np.random.RandomState(123)
    edges: list[tuple[int, int]] = []
    for source in range(50):
        for target in range(source + 1, 50):
            if rng.rand() < 0.045:
                edges.append((source, target))
    return _graph_from_edges(50, edges)


def _cases() -> list[FidelityCase]:
    """Return the small-graph-first verification cases.

    Returns
    -------
    list[FidelityCase]
        Fidelity cases requested by the sprint spec.
    """
    return [
        FidelityCase("single_node", lambda: _graph_from_edges(1, []), 1.0, 42),
        FidelityCase(
            "small_chain",
            lambda: _graph_from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4)]),
            3.0,
            7,
        ),
        FidelityCase(
            "binary_tree",
            lambda: _graph_from_edges(
                7,
                [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)],
            ),
            5.0,
            11,
        ),
        FidelityCase(
            "diamond",
            lambda: _graph_from_edges(4, [(0, 1), (0, 2), (1, 3), (2, 3)]),
            2.0,
            13,
        ),
        FidelityCase("grid_5x5", _grid_5x5, 10.0, 17),
        FidelityCase(
            "org_chart_small",
            lambda: _graph_from_edges(
                8,
                [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 6), (3, 7)],
            ),
            5.0,
            19,
        ),
        FidelityCase(
            "long_skip",
            lambda: _graph_from_edges(
                8,
                [
                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (3, 4),
                    (4, 5),
                    (5, 6),
                    (6, 7),
                    (0, 4),
                    (2, 7),
                ],
            ),
            5.0,
            23,
        ),
        FidelityCase(
            "disconnected",
            lambda: _graph_from_edges(7, [(0, 1), (1, 2), (3, 4), (5, 6)]),
            5.0,
            29,
        ),
        FidelityCase(
            "cycle_4",
            lambda: _graph_from_edges(4, [(0, 1), (1, 2), (2, 3), (3, 0)]),
            2.0,
            31,
        ),
        FidelityCase("random_dag_50", _random_dag_50, 20.0, 37),
        FidelityCase(
            "org_chart_deep",
            lambda: _graph_from_edges(
                10,
                [
                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (3, 4),
                    (2, 5),
                    (5, 6),
                    (6, 7),
                    (1, 8),
                    (8, 9),
                ],
            ),
            7.0,
            41,
        ),
    ]


def _tier(max_abs_diff: float, residual: float) -> str:
    """Classify one fidelity result.

    Parameters
    ----------
    max_abs_diff : float
        Maximum raw coordinate absolute difference.
    residual : float
        Procrustes residual.

    Returns
    -------
    str
        ``bit-exact``, ``positional``, or ``distributional``.
    """
    if max_abs_diff == 0.0:
        return "bit-exact"
    if residual < 1.0e-8:
        return "positional"
    return "distributional"


def _run_case(case: FidelityCase) -> FidelityResult:
    """Run one graph fidelity case.

    Parameters
    ----------
    case : FidelityCase
        Case to execute.

    Returns
    -------
    FidelityResult
        Measured fidelity metrics.

    Raises
    ------
    RuntimeError
        If the sklearn competitor fails.
    """
    graph = case.build()
    reference = TSNEGraph().layout_with_variant(
        graph,
        seed=case.seed,
        variant_params={
            "perplexity": case.perplexity,
            "max_iter": 250,
            "learning_rate": "auto",
        },
    )
    if reference.error is not None or reference.pos is None:
        raise RuntimeError(f"{case.name}: sklearn competitor failed: {reference.error}")

    actual = layout_tsne_graph_pipeline(
        graph.edge_index,
        graph.num_nodes,
        perplexity=case.perplexity,
        max_iter=250,
        seed=case.seed,
    )
    actual_np = actual.detach().cpu().numpy()
    reference_np = reference.pos.detach().cpu().numpy()
    residual = procrustes_rmsd(actual_np, reference_np)
    max_abs_diff = float(np.max(np.abs(actual_np - reference_np))) if actual_np.size else 0.0
    return FidelityResult(
        name=case.name,
        residual=residual,
        max_abs_diff=max_abs_diff,
        tier=_tier(max_abs_diff, residual),
    )


def main() -> None:
    """Print the graph t-SNE fidelity report.

    Returns
    -------
    None
        Results are printed to stdout.
    """
    torch.set_num_threads(1)
    results = [_run_case(case) for case in _cases()]
    counts = {"bit-exact": 0, "positional": 0, "distributional": 0}
    for result in results:
        counts[result.tier] += 1
        print(
            f"{result.name}: residual={result.residual:.12g} "
            f"max_abs_diff={result.max_abs_diff:.12g} tier={result.tier}"
        )
    print(
        "summary: "
        f"bit-exact={counts['bit-exact']} "
        f"positional={counts['positional']} "
        f"distributional={counts['distributional']}"
    )


if __name__ == "__main__":
    main()
