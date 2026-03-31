"""Tests for the (SGD)^2 multicriteria layout."""

from __future__ import annotations

import math

import pytest
import torch

from dagua.layout.classic import sgd2_multi as sgd2_multi_module
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


def _cycle_edge_index(num_nodes: int) -> torch.Tensor:
    """Build a directed cycle edge list.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the cycle.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    source = torch.arange(0, num_nodes, dtype=torch.long)
    target = (source + 1) % num_nodes
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


def test_layout_sgd2_multi_does_not_center_returned_positions() -> None:
    """Zero-step runs should return the raw seeded initialization."""
    num_nodes = 4
    edge_index = torch.empty((2, 0), dtype=torch.long)
    torch.manual_seed(17)
    expected = torch.randn((num_nodes, 2), dtype=torch.float32) * math.sqrt(float(num_nodes))

    positions = layout_sgd2_multi(edge_index=edge_index, num_nodes=num_nodes, seed=17, steps=0)

    assert torch.allclose(positions, expected)
    assert not torch.allclose(positions.mean(dim=0), torch.zeros(2))


def test_layout_sgd2_multi_aspect_ratio_uses_sampled_subset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Aspect-ratio batches should respect the shared cyclic sampler."""
    observed: dict[str, int] = {}

    def _fake_aspect_ratio_loss(pos: torch.Tensor, target: float) -> torch.Tensor:
        """Record the sampled batch size while preserving a scalar loss."""
        observed["rows"] = int(pos.shape[0])
        observed["target"] = int(target)
        return pos.sum() * 0.0

    monkeypatch.setattr(sgd2_multi_module, "_aspect_ratio_loss", _fake_aspect_ratio_loss)

    layout_sgd2_multi(
        edge_index=_path_edge_index(10),
        num_nodes=10,
        seed=5,
        steps=1,
        criteria={"aspect_ratio": 1.0},
        batch_size=4,
    )

    assert observed["rows"] == 4
    assert observed["target"] == 1


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


def test_layout_sgd2_multi_supports_crossing_criterion() -> None:
    """The neural crossing criterion should run without producing NaNs."""
    edge_index = _cycle_edge_index(4)

    positions = layout_sgd2_multi(
        edge_index=edge_index,
        num_nodes=4,
        seed=11,
        steps=50,
        criteria={"stress": 0.8, "crossings": 0.2},
        batch_size=4,
    )

    assert positions.shape == (4, 2)
    assert not positions.isnan().any()


def test_crossing_angle_loss_matches_reference_tan_squared_binary_labels() -> None:
    """Crossing-angle loss should use binary labels and the tan-squared penalty."""
    pos = torch.tensor(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [1.0, -1.0],
            [2.0, 1.0],
            [3.0, 0.0],
            [5.0, 0.0],
            [6.0, -1.0],
            [7.0, 1.0],
        ],
        dtype=torch.float32,
    )
    left = torch.tensor([[0, 4], [1, 5]], dtype=torch.long)
    right = torch.tensor([[2, 6], [3, 7]], dtype=torch.long)

    loss = sgd2_multi_module._crossing_angle_loss(pos=pos, left=left, right=right)

    expected = ((1.0 / 5.0) / (4.0 / 5.0 + sgd2_multi_module._EPS)) / 2.0
    assert loss.item() == pytest.approx(expected)


def test_layout_sgd2_multi_supports_aspect_ratio_criterion() -> None:
    """The sampled aspect-ratio criterion should run without producing NaNs."""
    edge_index = _cycle_edge_index(4)

    positions = layout_sgd2_multi(
        edge_index=edge_index,
        num_nodes=4,
        seed=17,
        steps=50,
        criteria={"stress": 0.8, "aspect_ratio": 0.2},
        batch_size=4,
    )

    assert positions.shape == (4, 2)
    assert not positions.isnan().any()


def test_layout_sgd2_multi_supports_bfs_and_incident_edge_sampling() -> None:
    """Neighborhood and angular criteria should execute on finite outputs."""
    edge_index = torch.tensor(
        [[0, 0, 0, 1, 2, 3], [1, 2, 3, 4, 4, 4]],
        dtype=torch.long,
    )

    positions = layout_sgd2_multi(
        edge_index=edge_index,
        num_nodes=5,
        seed=13,
        steps=32,
        criteria={
            "stress": 0.6,
            "neighborhood_preservation": 0.2,
            "angular_resolution": 0.2,
        },
        batch_size=3,
    )

    assert positions.shape == (5, 2)
    assert torch.isfinite(positions).all()
