"""Regression tests for the classic SFDP and UMAP graph layouts."""

from __future__ import annotations

import torch

from dagua.layout.classic import layout_sfdp, layout_umap
from dagua.layout.classic.umap_layout import _all_pairs_shortest_paths, _smooth_knn_dist


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


def test_layout_sfdp_edge_weights_pull_heavier_edges_closer() -> None:
    """SFDP should tighten edges with larger spring weights."""
    edge_index, num_nodes = _path_graph(num_nodes=5)
    uniform_weights = torch.ones(edge_index.shape[1], dtype=torch.float32)
    weighted = uniform_weights.clone()
    weighted[1] = 8.0

    uniform_pos = layout_sfdp(
        edge_index=edge_index,
        num_nodes=num_nodes,
        steps=70,
        seed=9,
        edge_weights=uniform_weights,
    )
    weighted_pos = layout_sfdp(
        edge_index=edge_index,
        num_nodes=num_nodes,
        steps=70,
        seed=9,
        edge_weights=weighted,
    )

    uniform_distance = torch.linalg.vector_norm(uniform_pos[1] - uniform_pos[2]).item()
    weighted_distance = torch.linalg.vector_norm(weighted_pos[1] - weighted_pos[2]).item()
    assert weighted_distance < uniform_distance


def test_layout_sfdp_rejects_mismatched_edge_weights() -> None:
    """SFDP should reject edge-weight tensors that do not match ``E``."""
    edge_index, num_nodes = _path_graph(num_nodes=4)

    try:
        layout_sfdp(
            edge_index=edge_index,
            num_nodes=num_nodes,
            edge_weights=torch.ones(1, dtype=torch.float32),
        )
    except ValueError as exc:
        assert "edge_weights length" in str(exc)
    else:
        raise AssertionError("layout_sfdp accepted mismatched edge_weights")


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


def test_layout_umap_edge_weights_shorten_heavier_edges() -> None:
    """UMAP should treat larger edge weights as stronger local connectivity."""
    edge_index, num_nodes = _path_graph(num_nodes=5)
    uniform_weights = torch.ones(edge_index.shape[1], dtype=torch.float32)
    weighted = uniform_weights.clone()
    weighted[1] = 8.0

    uniform_pos = layout_umap(
        edge_index=edge_index,
        num_nodes=num_nodes,
        n_neighbors=3,
        n_epochs=100,
        seed=21,
        edge_weights=uniform_weights,
    )
    weighted_pos = layout_umap(
        edge_index=edge_index,
        num_nodes=num_nodes,
        n_neighbors=3,
        n_epochs=100,
        seed=21,
        edge_weights=weighted,
    )

    uniform_distance = torch.linalg.vector_norm(uniform_pos[1] - uniform_pos[2]).item()
    weighted_distance = torch.linalg.vector_norm(weighted_pos[1] - weighted_pos[2]).item()
    assert weighted_distance < uniform_distance


def test_layout_umap_rejects_mismatched_edge_weights() -> None:
    """UMAP should reject edge-weight tensors that do not match ``E``."""
    edge_index, num_nodes = _path_graph(num_nodes=4)

    try:
        layout_umap(
            edge_index=edge_index,
            num_nodes=num_nodes,
            n_neighbors=2,
            n_epochs=10,
            edge_weights=torch.ones(1, dtype=torch.float32),
        )
    except ValueError as exc:
        assert "edge_weights length" in str(exc)
    else:
        raise AssertionError("layout_umap accepted mismatched edge_weights")


def test_smooth_knn_dist_uses_smallest_positive_rho() -> None:
    """Smooth-kNN should derive ``rho`` from the smallest positive distance."""
    knn_distances = torch.tensor([[0.0, 0.0, 1.0, 2.0]], dtype=torch.float32)

    sigmas, rhos = _smooth_knn_dist(knn_distances=knn_distances, n_neighbors=4)

    assert torch.allclose(rhos, torch.tensor([1.0], dtype=torch.float32))
    assert float(sigmas[0].item()) < 0.1


def test_smooth_knn_dist_skips_self_distance_in_membership_sum() -> None:
    """Smooth-kNN should not include ``j=0`` when solving the bandwidth."""
    knn_distances = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32)

    sigmas, rhos = _smooth_knn_dist(knn_distances=knn_distances, n_neighbors=3)

    assert torch.allclose(rhos, torch.tensor([1.0], dtype=torch.float32))
    assert torch.allclose(sigmas, torch.tensor([1.8653]), atol=5.0e-3)
