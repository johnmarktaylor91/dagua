"""Regression tests for igraph-compatible LGL fidelity behavior."""

from __future__ import annotations

import math
import random
from typing import List, Tuple

import pytest
import torch

from dagua.layout.ops.lgl import (
    _LGL_BUCKET_NEIGHBOR_OFFSETS,
    _LGL_REPULSION_MIN_DISTANCE,
    LGLInitializePositions,
    _build_lgl_bfs_layers,
    _lgl_updated_maxchange,
)
from dagua.layout.ops.pipelines.lgl import layout_lgl_pipeline
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


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


def _expected_column_major_positions(num_nodes: int, area: float, seed: int) -> torch.Tensor:
    """Rebuild igraph's random-layout draw order for LGL tests.

    Parameters
    ----------
    num_nodes : int
        Number of vertices in the layout.
    area : float
        LGL drawing area.
    seed : int
        Random seed passed to the Python compatibility RNG.

    Returns
    -------
    torch.Tensor
        Position matrix with shape ``[N, 2]`` before root overwrite.
    """
    rng = random.Random(seed)
    radius = math.sqrt(area / math.pi)
    positions = torch.empty((num_nodes, 2), dtype=torch.float64)
    for axis in range(2):
        for node in range(num_nodes):
            positions[node, axis] = rng.uniform(-radius, radius)
    return positions


def _expected_shell_directions(num_children: int, num_nodes: int, seed: int) -> torch.Tensor:
    """Build normalized igraph-style random shell directions.

    Parameters
    ----------
    num_children : int
        Number of shell vertices to place.
    num_nodes : int
        Number of graph vertices whose initial coordinates consume RNG draws.
    seed : int
        Random seed passed to the Python compatibility RNG.

    Returns
    -------
    torch.Tensor
        Direction matrix with shape ``[num_children, 2]``.
    """
    rng = random.Random(seed)
    for _ in range(2 * num_nodes):
        rng.uniform(-1.0, 1.0)
    directions = torch.empty((num_children, 2), dtype=torch.float64)
    for child_index in range(num_children):
        direction = torch.tensor(
            [rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)],
            dtype=torch.float64,
        )
        directions[child_index] = direction / torch.linalg.norm(direction)
    return directions


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


def test_lgl_initial_positions_use_column_major_draw_order() -> None:
    """Verify LGL initialization draws all x values before all y values."""
    area = 64.0
    seed = 13
    root = 2
    problem = LayoutProblem(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=5,
        seed=seed,
    )
    state = SolveState()
    state.extras["lgl_area"] = area
    state.extras["lgl_root"] = root
    state.extras["lgl_root_was_random"] = False
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))

    final_state = LGLInitializePositions().apply(problem, state, ctx)
    assert final_state.pos is not None
    expected = _expected_column_major_positions(num_nodes=5, area=area, seed=seed)
    expected[root] = 0.0

    torch.testing.assert_close(final_state.pos, expected)


def test_lgl_shell_one_uses_random_vectors_like_deeper_shells() -> None:
    """Verify shell-1 placement uses RNG_UNIF vectors instead of equal angles."""
    edge_index = torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=torch.long)
    seed = 19
    area = 16.0
    positions = layout_lgl_pipeline(
        edge_index=edge_index,
        num_nodes=4,
        seed=seed,
        root=0,
        maxiter=0,
        area=area,
    )
    radius = math.sqrt(area / math.pi)
    expected = torch.zeros((4, 2), dtype=torch.float32)
    expected[1:] = (_expected_shell_directions(num_children=3, num_nodes=4, seed=seed) * radius).to(
        dtype=torch.float32
    )

    torch.testing.assert_close(positions, expected)


def test_lgl_grid_and_repulsion_constants_match_igraph() -> None:
    """Verify LGL uses igraph's sparse-grid neighbor set and repulsion epsilon."""
    assert _LGL_BUCKET_NEIGHBOR_OFFSETS == ((0, 0), (1, 0), (0, 1), (1, 1))
    assert _LGL_REPULSION_MIN_DISTANCE == 1.0e-5


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


def test_lgl_rejects_invalid_igraph_scalar_parameters() -> None:
    """Verify LGL validates explicit scalar parameters like igraph."""
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)

    invalid_kwargs = [
        {"maxdelta": 0.0},
        {"area": 0.0},
        {"repulserad": -1.0},
        {"cellsize": 0.0},
        {"root": 2},
    ]
    for kwargs in invalid_kwargs:
        with pytest.raises(ValueError):
            layout_lgl_pipeline(edge_index=edge_index, num_nodes=2, **kwargs)


def test_lgl_warns_on_disconnected_graph_like_igraph() -> None:
    """Verify disconnected LGL inputs surface igraph's warning semantics."""
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)

    with pytest.warns(UserWarning, match="disconnected graphs"):
        layout_lgl_pipeline(edge_index=edge_index, num_nodes=4, seed=5, root=0, maxiter=1)
