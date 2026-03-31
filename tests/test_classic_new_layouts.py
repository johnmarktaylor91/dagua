"""Invariant tests for the newest classic layout implementations."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from scipy.stats import spearmanr

from dagua.layout.classic.classical_mds import _shortest_path_distances, layout_classical_mds
from dagua.layout.classic.reingold_tilford import layout_reingold_tilford
from dagua.layout.classic.spectral import layout_spectral
from dagua.layout.classic.stress_majorization import _stress_value, layout_stress_majorization


def _path_edge_index(num_nodes: int) -> torch.Tensor:
    """Build a directed path graph.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    return torch.tensor(
        [list(range(num_nodes - 1)), list(range(1, num_nodes))],
        dtype=torch.long,
    )


def _tree_edge_index() -> torch.Tensor:
    """Build a binary tree edge list with deterministic node ordering.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, 14]`` for a 15-node binary tree.
    """
    edges: list[tuple[int, int]] = []
    for parent in range(7):
        edges.append((parent, (2 * parent) + 1))
        edges.append((parent, (2 * parent) + 2))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _unbalanced_walker_regression_tree_edge_index() -> torch.Tensor:
    """Build a deep, asymmetric tree that benefits from contour threading.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, 25]`` for a 26-node regression tree.
    """
    return (
        torch.tensor(
            [
                (0, 1),
                (0, 2),
                (2, 3),
                (1, 4),
                (4, 5),
                (4, 6),
                (6, 7),
                (4, 8),
                (4, 9),
                (9, 10),
                (8, 11),
                (11, 12),
                (11, 13),
                (13, 14),
                (12, 15),
                (12, 16),
                (16, 17),
                (12, 18),
                (18, 19),
                (18, 20),
                (20, 21),
                (19, 22),
                (22, 23),
                (22, 24),
                (24, 25),
            ],
            dtype=torch.long,
        )
        .t()
        .contiguous()
    )


def _legacy_reingold_tilford_positions(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Recreate the previous midpoint-only tidy tree layout.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]`` for a rooted tree.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]`` using the pre-fix x assignment.
    """
    children = _children_from_edge_index(edge_index=edge_index, num_nodes=num_nodes)
    depths = [0] * num_nodes
    for source, target in edge_index.t().tolist():
        depths[target] = depths[source] + 1

    preliminary_x = [0.0] * num_nodes
    next_leaf_x = 0.0
    stack: list[tuple[int, bool]] = [(0, False)]
    postorder: list[int] = []
    while stack:
        node_idx, processed = stack.pop()
        if processed:
            postorder.append(node_idx)
            continue
        stack.append((node_idx, True))
        for child_idx in reversed(children[node_idx]):
            stack.append((child_idx, False))

    subtree_spans: dict[int, tuple[float, float]] = {}
    for node_idx in postorder:
        if not children[node_idx]:
            preliminary_x[node_idx] = next_leaf_x
            subtree_spans[node_idx] = (next_leaf_x, next_leaf_x)
            next_leaf_x += 1.0
            continue

        first_child = children[node_idx][0]
        last_child = children[node_idx][-1]
        preliminary_x[node_idx] = 0.5 * (preliminary_x[first_child] + preliminary_x[last_child])
        subtree_spans[node_idx] = (
            subtree_spans[first_child][0],
            subtree_spans[last_child][1],
        )

    positions = torch.zeros((num_nodes, 2), dtype=torch.float32)
    for node_idx in range(num_nodes):
        positions[node_idx, 0] = float(preliminary_x[node_idx])
        positions[node_idx, 1] = float(depths[node_idx])
    positions -= positions.mean(dim=0, keepdim=True)
    return positions


def _layout_width(positions: torch.Tensor) -> float:
    """Measure the horizontal span of a 2D layout.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    float
        Difference between the maximum and minimum x coordinates.
    """
    return float((positions[:, 0].max() - positions[:, 0].min()).item())


def _upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    """Return the strict upper triangle of a square matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
        Square matrix with shape ``[N, N]``.

    Returns
    -------
    numpy.ndarray
        Flat vector of upper-triangle values.
    """
    row_idx, col_idx = np.triu_indices(matrix.shape[0], k=1)
    return matrix[row_idx, col_idx]


def _subtree_nodes(root: int, children: list[list[int]]) -> set[int]:
    """Return the node set spanned by one rooted subtree.

    Parameters
    ----------
    root : int
        Root node index.
    children : list[list[int]]
        Child lists for each node.

    Returns
    -------
    set[int]
        Nodes contained in the rooted subtree.
    """
    subtree = {root}
    for child in children[root]:
        subtree.update(_subtree_nodes(child, children))
    return subtree


def _children_from_edge_index(edge_index: torch.Tensor, num_nodes: int) -> list[list[int]]:
    """Build child lists from a directed tree edge tensor.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    list[list[int]]
        Child lists indexed by parent node.
    """
    children = [[] for _ in range(num_nodes)]
    for source, target in edge_index.t().tolist():
        children[source].append(target)
    return children


def _columnwise_abs_correlation(left: np.ndarray, right: np.ndarray) -> Iterable[float]:
    """Compute absolute correlations between corresponding embedding columns.

    Parameters
    ----------
    left : numpy.ndarray
        First coordinate matrix with shape ``[N, 2]``.
    right : numpy.ndarray
        Second coordinate matrix with shape ``[N, 2]``.

    Returns
    -------
    Iterable[float]
        Absolute correlations for each column pair.
    """
    for column in range(left.shape[1]):
        left_column = left[:, column] - left[:, column].mean()
        right_column = right[:, column] - right[:, column].mean()
        left_norm = np.linalg.norm(left_column)
        right_norm = np.linalg.norm(right_column)
        if left_norm == 0.0 or right_norm == 0.0:
            yield 0.0
            continue
        yield abs(float(np.dot(left_column, right_column) / (left_norm * right_norm)))


def test_classical_mds_preserves_distances() -> None:
    """Classical MDS should preserve shortest-path distances up to rank order."""
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 0, 2], [1, 2, 3, 4, 2, 4]],
        dtype=torch.long,
    )
    num_nodes = 5

    positions = layout_classical_mds(edge_index=edge_index, num_nodes=num_nodes)
    graph_distances = _shortest_path_distances(edge_index, num_nodes, edge_weights=None)
    euclidean = torch.cdist(positions, positions).detach().cpu().numpy()
    correlation = spearmanr(
        _upper_triangle_values(graph_distances),
        _upper_triangle_values(euclidean),
    ).correlation

    assert correlation is not None
    assert float(correlation) > 0.8


def test_stress_maj_monotone() -> None:
    """Stress majorization should not increase stress across trace snapshots."""
    edge_index = _path_edge_index(8)
    result = layout_stress_majorization(
        edge_index=edge_index,
        num_nodes=8,
        iterations=20,
        seed=7,
        trace_every=1,
    )

    assert isinstance(result, tuple)
    positions, traces = result
    assert torch.allclose(positions, traces[-1])

    target_distances = _shortest_path_distances(edge_index, 8, edge_weights=None)
    with np.errstate(divide="ignore"):
        weights = np.where(target_distances > 0.0, 1.0 / np.square(target_distances), 0.0)
    np.fill_diagonal(weights, 0.0)
    stress_values = [
        _stress_value(
            trace.detach().cpu().numpy(),
            target_distances=target_distances,
            weights=weights,
        )
        for trace in traces
    ]

    assert len(stress_values) == 20
    assert all(
        next_value <= current_value + 1.0e-6
        for current_value, next_value in zip(stress_values, stress_values[1:])
    )


def test_reingold_tilford_no_overlap() -> None:
    """Sibling subtrees should occupy disjoint horizontal spans."""
    edge_index = _tree_edge_index()
    positions = layout_reingold_tilford(edge_index=edge_index, num_nodes=15)
    children = _children_from_edge_index(edge_index, num_nodes=15)
    x_coordinates = positions[:, 0].tolist()

    for parent_idx, child_list in enumerate(children):
        if len(child_list) != 2:
            continue
        left_nodes = _subtree_nodes(child_list[0], children)
        right_nodes = _subtree_nodes(child_list[1], children)
        left_max = max(x_coordinates[node_idx] for node_idx in left_nodes)
        right_min = min(x_coordinates[node_idx] for node_idx in right_nodes)
        assert left_max < right_min, parent_idx


def test_reingold_tilford_handles_deep_chain_iteratively() -> None:
    """Deep trees should no longer depend on Python recursion depth."""
    edge_index = _path_edge_index(1000)

    positions = layout_reingold_tilford(edge_index=edge_index, num_nodes=1000)

    assert positions.shape == (1000, 2)
    assert torch.isfinite(positions).all()


def test_reingold_tilford_threads_reduce_unbalanced_tree_width() -> None:
    """Walker contour threading should tighten deeply unbalanced trees."""
    edge_index = _unbalanced_walker_regression_tree_edge_index()
    num_nodes = 26

    legacy_positions = _legacy_reingold_tilford_positions(
        edge_index=edge_index,
        num_nodes=num_nodes,
    )
    walker_positions = layout_reingold_tilford(edge_index=edge_index, num_nodes=num_nodes)

    assert _layout_width(walker_positions) < _layout_width(legacy_positions)


def test_spectral_eigenvector_correctness() -> None:
    """Default spectral output should match the symmetric normalized Laplacian basis."""
    edge_index = _path_edge_index(6)
    positions = layout_spectral(edge_index=edge_index, num_nodes=6, normalization="symmetric")

    adjacency = np.zeros((6, 6), dtype=np.float64)
    for source, target in edge_index.t().tolist():
        adjacency[source, target] = 1.0
        adjacency[target, source] = 1.0
    degrees = adjacency.sum(axis=1)
    inv_sqrt = np.zeros_like(degrees)
    nonzero_mask = degrees > 0.0
    inv_sqrt[nonzero_mask] = 1.0 / np.sqrt(degrees[nonzero_mask])
    normalized_laplacian = np.eye(6) - (np.diag(inv_sqrt) @ adjacency @ np.diag(inv_sqrt))
    eigenvalues, eigenvectors = np.linalg.eigh(normalized_laplacian)
    expected = eigenvectors[:, [1, 2]]
    correlations = list(
        _columnwise_abs_correlation(
            positions.detach().cpu().numpy(),
            expected,
        )
    )

    assert correlations[0] > 0.99
    assert correlations[1] > 0.99


def test_spectral_variant_modes_are_supported() -> None:
    """The spectral normalization variants should all produce finite layouts."""
    edge_index = _path_edge_index(7)
    symmetric = layout_spectral(edge_index=edge_index, num_nodes=7, normalization="symmetric")
    random_walk = layout_spectral(edge_index=edge_index, num_nodes=7, normalization="random_walk")
    unnormalized = layout_spectral(
        edge_index=edge_index,
        num_nodes=7,
        normalization="unnormalized",
    )

    assert torch.isfinite(symmetric).all()
    assert torch.isfinite(random_walk).all()
    assert torch.isfinite(unnormalized).all()
    assert not torch.allclose(symmetric, random_walk)
    assert not torch.allclose(symmetric, unnormalized)
