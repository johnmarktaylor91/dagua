"""Tests for the (SGD)^2 multicriteria layout."""

from __future__ import annotations

import pytest
import torch

from dagua.layout.classic.sgd2_multi import SmoothSteps, layout_sgd2_multi


def _path_edge_index(num_nodes: int) -> torch.Tensor:
    """Build a directed path edge list.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the path.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    source = torch.arange(0, num_nodes - 1, dtype=torch.long)
    target = source + 1
    return torch.stack([source, target], dim=0)


def test_smooth_steps_interpolates_between_keyframes() -> None:
    """The smooth schedule should interpolate monotonically between values."""
    schedule = SmoothSteps(times=[0, 10, 20], values=[0.0, 1.0, 0.5])

    assert schedule(0) == pytest.approx(0.0)
    assert 0.0 < schedule(5) < 1.0
    assert schedule(10) == pytest.approx(1.0)
    assert 0.5 < schedule(15) < 1.0
    assert schedule(25) == pytest.approx(0.5)


def test_layout_sgd2_multi_returns_expected_shape() -> None:
    """The multicriteria layout should return one 2D point per node."""
    edge_index = _path_edge_index(5)

    positions = layout_sgd2_multi(edge_index=edge_index, num_nodes=5, seed=42, steps=128)

    assert positions.shape == (5, 2)
    assert torch.isfinite(positions).all()


def test_layout_sgd2_multi_is_deterministic_for_same_seed() -> None:
    """Repeated runs with the same seed should match exactly."""
    edge_index = _path_edge_index(6)

    positions_a = layout_sgd2_multi(edge_index=edge_index, num_nodes=6, seed=9, steps=160)
    positions_b = layout_sgd2_multi(edge_index=edge_index, num_nodes=6, seed=9, steps=160)

    assert torch.allclose(positions_a, positions_b)


def test_layout_sgd2_multi_supports_multiple_criteria() -> None:
    """Multiple active criteria and schedules should execute without error."""
    edge_index = _path_edge_index(6)
    schedule = SmoothSteps(times=[0, 50, 100], values=[0.0, 0.5, 1.0])

    positions = layout_sgd2_multi(
        edge_index=edge_index,
        num_nodes=6,
        seed=21,
        steps=120,
        criteria={
            "stress": 1.0,
            "ideal_edge_length": 0.25,
            "aspect_ratio": 0.1,
            "vertex_resolution": 0.2,
        },
        criteria_schedules={"crossings": schedule},
        batch_size=8,
    )

    assert positions.shape == (6, 2)
