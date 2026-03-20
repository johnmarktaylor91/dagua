"""Regression tests for the classic SFDP and UMAP graph layouts."""

from __future__ import annotations

import torch

from dagua.layout.classic import layout_sfdp, layout_umap
from dagua.layout.classic.umap_layout import _all_pairs_shortest_paths


def _edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    """Build an edge-index tensor from integer edge pairs.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Directed graph edges.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).transpose(0, 1).contiguous()


def _path_graph(num_nodes: int) -> tuple[torch.Tensor, int]:
    """Create a simple directed path graph.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the path.

    Returns
    -------
    tuple[torch.Tensor, int]
        Edge tensor and node count.
    """
    edges = [(index, index + 1) for index in range(num_nodes - 1)]
    return _edge_index(edges), num_nodes


def _cluster_bridge_graph() -> tuple[torch.Tensor, int]:
    """Create two dense clusters connected by a sparse bridge.

    Parameters
    ----------
    None
        No parameters.

    Returns
    -------
    tuple[torch.Tensor, int]
        Edge tensor and node count.
    """
    edges = [
        (0, 1),
        (1, 2),
        (2, 0),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 3),
        (1, 4),
    ]
    return _edge_index(edges), 6


def _pairwise_distances(pos: torch.Tensor) -> torch.Tensor:
    """Compute all unique pairwise distances for a position tensor.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Pairwise distances for ``i < j``.
    """
    return torch.pdist(pos)


def test_layout_sfdp_returns_finite_positions_with_expected_shape() -> None:
    """SFDP returns a finite ``[N, 2]`` position tensor."""
    edge_index, num_nodes = _cluster_bridge_graph()

    pos = layout_sfdp(edge_index=edge_index, num_nodes=num_nodes, steps=60, seed=7)

    assert isinstance(pos, torch.Tensor)
    assert pos.shape == (num_nodes, 2)
    assert torch.isfinite(pos).all()


def test_layout_sfdp_is_deterministic_for_same_seed() -> None:
    """SFDP is deterministic for repeated seeded runs."""
    edge_index, num_nodes = _cluster_bridge_graph()

    pos_a = layout_sfdp(edge_index=edge_index, num_nodes=num_nodes, steps=50, seed=13)
    pos_b = layout_sfdp(edge_index=edge_index, num_nodes=num_nodes, steps=50, seed=13)

    assert torch.allclose(pos_a, pos_b)


def test_layout_sfdp_connected_nodes_are_closer_than_average_pair() -> None:
    """SFDP keeps path neighbors closer than the average node pair."""
    edge_index, num_nodes = _path_graph(num_nodes=7)

    pos = layout_sfdp(edge_index=edge_index, num_nodes=num_nodes, steps=80, seed=5)
    connected = torch.linalg.vector_norm(pos[edge_index[1]] - pos[edge_index[0]], dim=1)

    assert connected.mean() < _pairwise_distances(pos).mean()


def test_layout_umap_returns_finite_positions_with_expected_shape() -> None:
    """Graph UMAP returns a finite ``[N, 2]`` position tensor."""
    edge_index, num_nodes = _cluster_bridge_graph()

    pos = layout_umap(
        edge_index=edge_index,
        num_nodes=num_nodes,
        n_neighbors=4,
        n_epochs=80,
        seed=11,
    )

    assert isinstance(pos, torch.Tensor)
    assert pos.shape == (num_nodes, 2)
    assert torch.isfinite(pos).all()


def test_layout_umap_is_deterministic_for_same_seed() -> None:
    """Graph UMAP is deterministic for repeated seeded runs."""
    edge_index, num_nodes = _cluster_bridge_graph()

    pos_a = layout_umap(
        edge_index=edge_index,
        num_nodes=num_nodes,
        n_neighbors=4,
        n_epochs=80,
        seed=17,
    )
    pos_b = layout_umap(
        edge_index=edge_index,
        num_nodes=num_nodes,
        n_neighbors=4,
        n_epochs=80,
        seed=17,
    )

    assert torch.allclose(pos_a, pos_b)


def test_layout_umap_disconnected_distances_use_symmetric_fill() -> None:
    """Disconnected graph distances should remain symmetric after finite filling."""
    adjacency = [[1], [0], [3], [2]]

    distances = _all_pairs_shortest_paths(adjacency)

    assert torch.allclose(distances, distances.transpose(0, 1))
    assert float(distances[0, 2].item()) == float(distances[2, 0].item())


def test_layout_umap_connected_nodes_are_closer_than_average_pair() -> None:
    """Graph UMAP preserves local path neighborhoods in the embedding."""
    edge_index, num_nodes = _path_graph(num_nodes=8)

    pos = layout_umap(
        edge_index=edge_index,
        num_nodes=num_nodes,
        n_neighbors=4,
        n_epochs=120,
        seed=3,
    )
    connected = torch.linalg.vector_norm(pos[edge_index[1]] - pos[edge_index[0]], dim=1)

    assert connected.mean() < _pairwise_distances(pos).mean()
