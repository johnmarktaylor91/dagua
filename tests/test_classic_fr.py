from __future__ import annotations

import torch

from dagua.layout.classic import layout_fr


def _pairwise_distances(pos: torch.Tensor) -> torch.Tensor:
    """Compute all unique pairwise distances for a position tensor.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Pairwise distances for ``i < j``.
    """
    distances = torch.pdist(pos)
    return distances


def test_layout_fr_returns_expected_shape() -> None:
    """FR layout returns an ``[N, 2]`` position tensor."""
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int64)

    pos = layout_fr(edge_index=edge_index, num_nodes=4, steps=50)

    assert isinstance(pos, torch.Tensor)
    assert pos.shape == (4, 2)


def test_layout_fr_is_deterministic_for_same_seed() -> None:
    """FR layout is deterministic for a fixed seed."""
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int64)

    pos_a = layout_fr(edge_index=edge_index, num_nodes=4, steps=80, seed=7)
    pos_b = layout_fr(edge_index=edge_index, num_nodes=4, steps=80, seed=7)

    assert isinstance(pos_a, torch.Tensor)
    assert isinstance(pos_b, torch.Tensor)
    assert torch.allclose(pos_a, pos_b)


def test_layout_fr_changes_with_different_seed() -> None:
    """Different seeds produce different final layouts."""
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int64)

    pos_a = layout_fr(edge_index=edge_index, num_nodes=4, steps=80, seed=7)
    pos_b = layout_fr(edge_index=edge_index, num_nodes=4, steps=80, seed=8)

    assert isinstance(pos_a, torch.Tensor)
    assert isinstance(pos_b, torch.Tensor)
    assert not torch.allclose(pos_a, pos_b)


def test_layout_fr_handles_disconnected_graph() -> None:
    """FR layout handles graphs with no edges."""
    edge_index = torch.zeros((2, 0), dtype=torch.int64)

    pos = layout_fr(edge_index=edge_index, num_nodes=5, steps=40, seed=3)

    assert isinstance(pos, torch.Tensor)
    assert pos.shape == (5, 2)
    assert torch.isfinite(pos).all()


def test_layout_fr_trace_mode_returns_snapshots() -> None:
    """Trace mode returns the final tensor and position snapshots."""
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int64)

    pos, traces = layout_fr(edge_index=edge_index, num_nodes=4, steps=25, trace_every=5)

    assert isinstance(pos, torch.Tensor)
    assert isinstance(traces, list)
    assert pos.shape == (4, 2)
    assert len(traces) == 5
    assert all(trace.shape == (4, 2) for trace in traces)


def test_layout_fr_nodes_do_not_collapse_to_single_point() -> None:
    """Repulsion keeps the final layout spatially spread out."""
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.int64)

    pos = layout_fr(edge_index=edge_index, num_nodes=5, steps=120, seed=11)

    assert isinstance(pos, torch.Tensor)
    assert torch.std(pos) > 0.5


def test_layout_fr_connected_nodes_are_closer_than_average_pair() -> None:
    """Edge-connected nodes should end up closer than the average node pair."""
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int64)

    pos = layout_fr(edge_index=edge_index, num_nodes=4, steps=150, seed=5)

    assert isinstance(pos, torch.Tensor)
    pairwise = _pairwise_distances(pos)
    src = edge_index[0]
    dst = edge_index[1]
    connected = torch.linalg.norm(pos[dst] - pos[src], dim=1)

    assert connected.mean() < pairwise.mean()
