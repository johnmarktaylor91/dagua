"""Tests for W5 differentiable surrogate ops."""

from __future__ import annotations

import torch

from dagua.layout.ops.pipelines.native_surrogates import (
    crossing_angle_loss,
    depth_order_score_surrogate,
    edge_length_cv_loss,
    overlap_hinge_loss,
    path_continuity_loss,
    signed_flow_score_surrogate,
    soft_crossing_loss,
    soft_knn_neighborhood_loss,
)


def test_w5_surrogates_return_finite_on_degenerate_input() -> None:
    """Degenerate coordinates should not emit NaN or inf losses."""
    pos = torch.zeros((4, 2), dtype=torch.float32, requires_grad=True)
    edge_index = torch.tensor([[0, 2, 0], [1, 3, 2]], dtype=torch.long)
    node_sizes = torch.full((4, 2), 10.0)
    depth = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    losses = [
        soft_crossing_loss(pos, edge_index),
        crossing_angle_loss(pos, edge_index),
        path_continuity_loss(pos, edge_index),
        1.0 - signed_flow_score_surrogate(pos, edge_index),
        1.0 - depth_order_score_surrogate(pos, depth),
        edge_length_cv_loss(pos, edge_index),
        overlap_hinge_loss(pos, node_sizes),
        soft_knn_neighborhood_loss(pos, edge_index),
    ]

    for loss in losses:
        assert bool(torch.isfinite(loss).all().item())


def test_soft_crossing_loss_gradient_uncrosses_tiny_case() -> None:
    """A gradient step away from a crossing should reduce soft crossing loss."""
    pos = torch.tensor(
        [[0.0, 0.0], [1.0, 0.9], [0.0, 1.0], [1.0, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    edge_index = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
    before = soft_crossing_loss(pos, edge_index)
    before.backward()

    with torch.no_grad():
        stepped = pos - 0.25 * pos.grad
    after = soft_crossing_loss(stepped, edge_index)

    assert float(after.item()) < float(before.item())


def test_path_continuity_prefers_straight_chain() -> None:
    """The path-continuity surrogate is lower for a straight degree-two chain."""
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    straight = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    bent = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])

    assert float(path_continuity_loss(straight, edge_index).item()) < float(
        path_continuity_loss(bent, edge_index).item()
    )
