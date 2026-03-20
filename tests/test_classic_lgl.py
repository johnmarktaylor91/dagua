"""Tests for the classic LGL layout."""

from __future__ import annotations

import random

import pytest
import torch

from dagua.layout.classic import layout_lgl


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


def _pairwise_distance_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    """Compute Pearson correlation between pairwise distances.

    Parameters
    ----------
    left : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    right : torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    float
        Correlation coefficient in ``[-1, 1]``.
    """
    left_dist = torch.pdist(left.to(dtype=torch.float64))
    right_dist = torch.pdist(right.to(dtype=torch.float64))
    left_centered = left_dist - left_dist.mean()
    right_centered = right_dist - right_dist.mean()
    denom = float(
        torch.linalg.norm(left_centered).item() * torch.linalg.norm(right_centered).item()
    )
    if denom == 0.0:
        return 1.0
    return float(left_centered.dot(right_centered).item() / denom)


def test_layout_lgl_returns_finite_positions() -> None:
    """LGL returns a finite ``[N, 2]`` position tensor."""
    edge_index = _edge_index([(0, 1), (1, 2), (1, 3), (3, 4), (3, 5)])

    pos = layout_lgl(edge_index=edge_index, num_nodes=6, seed=3, root=0, maxiter=20)

    assert pos.shape == (6, 2)
    assert torch.isfinite(pos).all()


def test_layout_lgl_is_deterministic_for_same_seed_and_root() -> None:
    """Fixing both seed and root should make LGL deterministic."""
    edge_index = _edge_index([(0, 1), (1, 2), (1, 3), (3, 4), (3, 5)])

    first = layout_lgl(edge_index=edge_index, num_nodes=6, seed=9, root=0, maxiter=20)
    second = layout_lgl(edge_index=edge_index, num_nodes=6, seed=9, root=0, maxiter=20)

    torch.testing.assert_close(first, second)


def test_layout_lgl_root_override_changes_layout() -> None:
    """Changing the BFS root should change the final layered layout."""
    edge_index = _edge_index([(0, 1), (1, 2), (1, 3), (3, 4), (3, 5)])

    left = layout_lgl(edge_index=edge_index, num_nodes=6, seed=15, root=0, maxiter=20)
    right = layout_lgl(edge_index=edge_index, num_nodes=6, seed=15, root=3, maxiter=20)

    assert not torch.allclose(left, right)


def test_layout_lgl_parallel_edges_change_layout() -> None:
    """Parallel and reciprocal edges should remain distinct springs."""
    single_edge = _edge_index([(0, 1), (1, 2), (1, 3)])
    duplicate_edges = _edge_index([(0, 1), (1, 0), (1, 2), (1, 3)])

    single = layout_lgl(edge_index=single_edge, num_nodes=4, seed=8, root=1, maxiter=20)
    duplicate = layout_lgl(edge_index=duplicate_edges, num_nodes=4, seed=8, root=1, maxiter=20)

    assert not torch.allclose(single, duplicate)


def test_layout_lgl_tracks_igraph_pairwise_distances() -> None:
    """The PyTorch LGL port broadly tracks igraph on a small tree."""
    igraph = pytest.importorskip("igraph")
    edge_index = _edge_index([(0, 1), (1, 2), (1, 3), (3, 4), (3, 5)])
    seed = 4

    ours = layout_lgl(edge_index=edge_index, num_nodes=6, seed=seed, root=0, maxiter=20)
    graph = igraph.Graph(n=6, edges=[(0, 1), (1, 2), (1, 3), (3, 4), (3, 5)], directed=False)
    igraph.set_random_number_generator(random.Random(seed))
    try:
        reference_layout = graph.layout_lgl(root=0, maxiter=20)
    finally:
        igraph.set_random_number_generator(None)
    reference = torch.tensor(reference_layout.coords, dtype=torch.float32)

    correlation = _pairwise_distance_correlation(ours, reference)

    assert correlation > 0.4
