"""Tests for the classic DrL layout."""

from __future__ import annotations

import random

import pytest
import torch

from dagua.layout.classic import layout_drl


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


def _fast_drl_options() -> dict[str, float]:
    """Return a lightweight DrL schedule for tests.

    Parameters
    ----------
    None
        No parameters.

    Returns
    -------
    dict[str, float]
        Reduced-iteration option mapping.
    """
    return {
        "init_iterations": 0.0,
        "liquid_iterations": 10.0,
        "expansion_iterations": 10.0,
        "cooldown_iterations": 10.0,
        "crunch_iterations": 5.0,
        "simmer_iterations": 0.0,
    }


def _seed_matrix(num_nodes: int, seed: int) -> list[list[float]]:
    """Recreate the default seeded starting positions for the DrL port.

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


def test_layout_drl_returns_finite_positions() -> None:
    """DrL returns a finite ``[N, 2]`` position tensor."""
    edge_index = _edge_index([(0, 1), (1, 2), (2, 3), (1, 4), (4, 5)])

    pos = layout_drl(edge_index=edge_index, num_nodes=6, seed=5, options=_fast_drl_options())

    assert pos.shape == (6, 2)
    assert torch.isfinite(pos).all()


def test_layout_drl_is_deterministic_for_same_seed() -> None:
    """Repeated seeded runs should be deterministic."""
    edge_index = _edge_index([(0, 1), (1, 2), (2, 3), (1, 4), (4, 5)])
    options = _fast_drl_options()

    first = layout_drl(edge_index=edge_index, num_nodes=6, seed=23, options=options)
    second = layout_drl(edge_index=edge_index, num_nodes=6, seed=23, options=options)

    torch.testing.assert_close(first, second)


def test_layout_drl_respects_edge_weights() -> None:
    """Changing edge weights should change the final layout."""
    edge_index = _edge_index([(0, 1), (1, 2), (2, 3), (1, 4), (4, 5)])
    uniform_weights = torch.ones(edge_index.shape[1], dtype=torch.float32)
    weighted = uniform_weights.clone()
    weighted[1] = 8.0

    uniform_pos = layout_drl(
        edge_index=edge_index,
        num_nodes=6,
        seed=31,
        weights=uniform_weights,
        options=_fast_drl_options(),
    )
    weighted_pos = layout_drl(
        edge_index=edge_index,
        num_nodes=6,
        seed=31,
        weights=weighted,
        options=_fast_drl_options(),
    )

    assert not torch.allclose(uniform_pos, weighted_pos)


def test_layout_drl_tracks_igraph_pairwise_distances() -> None:
    """The PyTorch DrL port broadly tracks igraph on a small graph."""
    igraph = pytest.importorskip("igraph")
    edge_index = _edge_index([(0, 1), (1, 2), (2, 3), (1, 4), (4, 5)])
    options = _fast_drl_options()

    seed = 7
    ours = layout_drl(edge_index=edge_index, num_nodes=6, seed=seed, options=options)
    graph = igraph.Graph(n=6, edges=[(0, 1), (1, 2), (2, 3), (1, 4), (4, 5)], directed=False)
    reference_layout = graph.layout_drl(options=options, seed=_seed_matrix(6, seed))
    reference = torch.tensor(reference_layout.coords, dtype=torch.float32)

    correlation = _pairwise_distance_correlation(ours, reference)

    assert correlation > 0.45
