"""Tests for spatial-hash loss helpers."""

from __future__ import annotations

from typing import Set, Tuple

import torch

from dagua.layout.ops.loss_engine import (
    _cell_list_overlap_loss,
    _cell_list_repulsion_loss,
    _exact_overlap_loss,
    _exact_repulsion_loss,
)
from dagua.layout.ops.spatial_hash import UniformSpatialHash


def _pair_set(pairs: torch.Tensor) -> Set[Tuple[int, int]]:
    """Convert a pair tensor to a Python set.

    Parameters
    ----------
    pairs : torch.Tensor
        Long tensor with shape ``[2, M]``.

    Returns
    -------
    set[tuple[int, int]]
        Unique unordered pair set.
    """
    return {
        (int(pairs[0, index].item()), int(pairs[1, index].item()))
        for index in range(int(pairs.shape[1]))
    }


def _true_pairs_within_radius(pos: torch.Tensor, radius: float) -> Set[Tuple[int, int]]:
    """Return exact unordered pairs within an Euclidean radius.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    radius : float
        Radius threshold.

    Returns
    -------
    set[tuple[int, int]]
        Exact pair set.
    """
    dist = torch.cdist(pos, pos)
    pairs: Set[Tuple[int, int]] = set()
    for source in range(int(pos.shape[0])):
        for target in range(source + 1, int(pos.shape[0])):
            if float(dist[source, target].item()) <= radius:
                pairs.add((source, target))
    return pairs


def test_cell_list_candidate_pairs_include_all_true_radius_pairs() -> None:
    """Cell-list candidates should have no false negatives within radius."""
    pos = torch.tensor(
        [
            [0.0, 0.0],
            [0.9, 0.0],
            [1.8, 0.0],
            [0.0, 1.1],
            [10.0, 10.0],
            [10.6, 10.1],
        ],
        dtype=torch.float32,
    )
    radius = 1.25

    candidates = _pair_set(UniformSpatialHash(pos, cutoff_radius=radius).candidate_pairs())
    exact_pairs = _true_pairs_within_radius(pos, radius)

    assert exact_pairs <= candidates


def test_cell_list_losses_match_exact_when_cutoff_covers_all_pairs() -> None:
    """Cell-list loss math should match exact all-pairs math on small inputs."""
    pos = torch.tensor(
        [[0.0, 0.0], [2.0, 0.0], [0.5, 1.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    node_sizes = torch.tensor(
        [[8.0, 10.0], [10.0, 8.0], [9.0, 9.0]],
        dtype=torch.float32,
    )
    cutoff = 100.0

    exact = _exact_repulsion_loss(pos, 1.0e-4, node_sizes) + _exact_overlap_loss(
        pos,
        node_sizes,
        padding=2.0,
    )
    cell_list = _cell_list_repulsion_loss(pos, 1.0e-4, cutoff, node_sizes) + (
        _cell_list_overlap_loss(pos, node_sizes, padding=2.0, cutoff_radius=cutoff)
    )

    assert torch.allclose(cell_list, exact, rtol=1.0e-6, atol=1.0e-6)


def test_cell_list_gradient_matches_exact_when_cutoff_covers_all_pairs() -> None:
    """Cell-list loss gradients should match exact gradients with all pairs included."""
    base_pos = torch.tensor(
        [[0.0, 0.0], [1.0, 0.1], [0.2, 1.2]],
        dtype=torch.float32,
    )
    node_sizes = torch.full((3, 2), 10.0, dtype=torch.float32)
    exact_pos = base_pos.clone().detach().requires_grad_(True)
    cell_pos = base_pos.clone().detach().requires_grad_(True)
    cutoff = 100.0

    exact_loss = _exact_repulsion_loss(exact_pos, 1.0e-4, node_sizes) + _exact_overlap_loss(
        exact_pos,
        node_sizes,
        padding=2.0,
    )
    cell_loss = _cell_list_repulsion_loss(cell_pos, 1.0e-4, cutoff, node_sizes) + (
        _cell_list_overlap_loss(cell_pos, node_sizes, padding=2.0, cutoff_radius=cutoff)
    )
    exact_loss.backward()
    cell_loss.backward()

    assert exact_pos.grad is not None
    assert cell_pos.grad is not None
    assert torch.allclose(cell_pos.grad, exact_pos.grad, rtol=1.0e-5, atol=1.0e-5)
