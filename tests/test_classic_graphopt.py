"""Tests for the classic GraphOpt layout."""

from __future__ import annotations

import random

import pytest
import torch

from dagua.layout.classic import layout_graphopt


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


def _graphopt_seed_matrix(num_nodes: int, seed: int) -> list[list[float]]:
    """Recreate the random initial coordinates used by ``layout_graphopt``.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    seed : int
        Random seed.

    Returns
    -------
    list[list[float]]
        Seed matrix suitable for igraph's ``seed=`` parameter.
    """
    rng = random.Random(seed)
    return [[rng.random(), rng.random()] for _ in range(num_nodes)]


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


def test_layout_graphopt_returns_finite_positions() -> None:
    """GraphOpt returns a finite ``[N, 2]`` position tensor."""
    edge_index = _edge_index([(0, 1), (1, 2), (2, 3), (3, 4)])

    pos = layout_graphopt(edge_index=edge_index, num_nodes=5, niter=80, seed=11)

    assert pos.shape == (5, 2)
    assert torch.isfinite(pos).all()


def test_layout_graphopt_is_deterministic_for_same_seed() -> None:
    """Repeated runs with the same seed produce identical layouts."""
    edge_index = _edge_index([(0, 1), (1, 2), (2, 3), (3, 4)])

    first = layout_graphopt(edge_index=edge_index, num_nodes=5, niter=80, seed=17)
    second = layout_graphopt(edge_index=edge_index, num_nodes=5, niter=80, seed=17)

    torch.testing.assert_close(first, second)


def test_layout_graphopt_parallel_edges_change_layout() -> None:
    """Parallel and reciprocal edges should contribute separate springs."""
    single_edge = _edge_index([(0, 1), (1, 2)])
    duplicate_edges = _edge_index([(0, 1), (1, 0), (1, 2)])

    single = layout_graphopt(edge_index=single_edge, num_nodes=3, niter=40, seed=5)
    duplicate = layout_graphopt(edge_index=duplicate_edges, num_nodes=3, niter=40, seed=5)

    assert not torch.allclose(single, duplicate)


def test_layout_graphopt_tracks_igraph_pairwise_distances() -> None:
    """The PyTorch port stays close to igraph on a small connected graph."""
    igraph = pytest.importorskip("igraph")
    edge_index = _edge_index([(0, 1), (1, 2), (2, 3), (1, 4), (2, 5)])
    seed = 13

    ours = layout_graphopt(edge_index=edge_index, num_nodes=6, niter=120, seed=seed)
    graph = igraph.Graph(n=6, edges=[(0, 1), (1, 2), (2, 3), (1, 4), (2, 5)], directed=False)
    reference_layout = graph.layout_graphopt(seed=_graphopt_seed_matrix(6, seed), niter=120)
    reference = torch.tensor(reference_layout.coords, dtype=torch.float32)

    correlation = _pairwise_distance_correlation(ours, reference)

    assert correlation > 0.9
