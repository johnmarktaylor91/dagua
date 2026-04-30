"""Regression tests for igraph-compatible LGL fidelity behavior."""

from __future__ import annotations

from typing import List, Tuple

import torch

from dagua.layout.ops.lgl import _build_lgl_bfs_layers, _lgl_updated_maxchange
from dagua.layout.ops.pipelines.lgl import layout_lgl_pipeline


def _adjacency(num_nodes: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
    """Build sorted undirected adjacency for LGL trace tests.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    edges : List[Tuple[int, int]]
        Undirected edge list.

    Returns
    -------
    List[List[int]]
        Sorted adjacency lists.
    """
    neighbor_sets = [set() for _ in range(num_nodes)]
    for source, target in edges:
        neighbor_sets[source].add(target)
        neighbor_sets[target].add(source)
    return [sorted(neighbors) for neighbors in neighbor_sets]


def _igraph_boundaries_from_layers(layers: List[List[int]]) -> List[int]:
    """Convert per-depth layers into igraph BFS boundary-vector form.

    Parameters
    ----------
    layers : List[List[int]]
        Per-depth BFS layers.

    Returns
    -------
    List[int]
        Cumulative boundary vector where depth ``d`` occupies
        ``[boundaries[d], boundaries[d + 1])`` in BFS order.
    """
    boundaries = [0]
    for layer in layers:
        boundaries.append(boundaries[-1] + len(layer))
    return boundaries


def test_lgl_edge_weights_ignored_by_default() -> None:
    """Verify default LGL attraction matches igraph's unweighted behavior."""
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]],
        dtype=torch.long,
    )
    weighted = layout_lgl_pipeline(
        edge_index=edge_index,
        num_nodes=4,
        seed=7,
        root=0,
        maxiter=8,
        edge_weights=torch.tensor([1.0, 1.0, 7.0, 7.0, 0.25, 0.25]),
    )
    unweighted = layout_lgl_pipeline(
        edge_index=edge_index,
        num_nodes=4,
        seed=7,
        root=0,
        maxiter=8,
    )

    torch.testing.assert_close(weighted, unweighted)


def test_lgl_igraph_positive_maxchange_rule() -> None:
    """Verify the convergence helper preserves igraph's signed quirk."""
    movement = torch.tensor([-0.4, -0.2], dtype=torch.float64)

    assert _lgl_updated_maxchange(0.0, movement, igraph_positive_only=True) == 0.0
    assert _lgl_updated_maxchange(0.0, movement, igraph_positive_only=False) == 0.4


def test_lgl_layer_boundaries_match_igraph_assumptions() -> None:
    """Trace path, star, and tree BFS shells against igraph boundaries."""
    graphs = [
        (4, [(0, 1), (1, 2), (2, 3)]),
        (5, [(0, 1), (0, 2), (0, 3), (0, 4)]),
        (7, [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]),
    ]

    for num_nodes, edges in graphs:
        layers, parents, distances = _build_lgl_bfs_layers(
            num_nodes,
            0,
            _adjacency(num_nodes, edges),
        )
        boundaries = _igraph_boundaries_from_layers(layers)
        no_of_layers = len(boundaries) - 1
        processed_shell_sizes = [
            boundaries[layer + 1] - boundaries[layer] for layer in range(1, no_of_layers)
        ]

        assert parents[0] == 0
        assert all(distance >= 0 for distance in distances)
        assert processed_shell_sizes == [len(layer) for layer in layers[1:]]
        if len(boundaries) > 2:
            assert boundaries[2] - 1 == len(layers[1])
