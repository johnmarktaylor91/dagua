"""Tests for the classic Kamada-Kawai layout implementation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import pytest
import torch

from dagua.layout.classic import kk as kk_module
from dagua.layout.classic import layout_kk


def test_layout_kk_returns_positions_with_expected_shape() -> None:
    """The layout returns a centered ``[N, 2]`` tensor."""
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

    positions = layout_kk(edge_index=edge_index, num_nodes=4, steps=30, seed=7)

    assert positions.shape == (4, 2)
    assert torch.linalg.norm(positions.mean(dim=0)) < 1.0e-4
    assert float(positions.abs().max().item()) <= 1.0 + 1.0e-5


def test_layout_kk_is_deterministic_for_same_seed() -> None:
    """The optimizer run is deterministic when the seed is fixed."""
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

    first = layout_kk(edge_index=edge_index, num_nodes=4, steps=40, seed=11)
    second = layout_kk(edge_index=edge_index, num_nodes=4, steps=40, seed=11)

    assert torch.allclose(first, second)


def test_layout_kk_handles_disconnected_components() -> None:
    """Disconnected components still produce finite coordinates."""
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [1, 2, 0, 4, 5, 3],
        ],
        dtype=torch.long,
    )

    positions = layout_kk(edge_index=edge_index, num_nodes=6, steps=80, seed=3)
    pairwise_distances = torch.cdist(positions, positions)
    component_gap = pairwise_distances[:3, 3:].mean()
    within_component = pairwise_distances[torch.tensor([0, 1, 2]), torch.tensor([1, 2, 0])].mean()

    assert torch.isfinite(positions).all()
    assert component_gap > within_component


def test_layout_kk_trace_mode_returns_snapshots() -> None:
    """Trace mode returns periodic position snapshots."""
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

    positions, traces = layout_kk(
        edge_index=edge_index,
        num_nodes=4,
        steps=12,
        seed=5,
        trace_every=5,
    )

    assert positions.shape == (4, 2)
    assert 0 <= len(traces) <= 2
    assert all(trace.shape == (4, 2) for trace in traces)


def test_layout_kk_keeps_graph_neighbors_closer_than_distant_nodes() -> None:
    """Nodes with graph distance one end up closer than nodes four hops apart."""
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)

    positions = layout_kk(edge_index=edge_index, num_nodes=5, steps=150, seed=17)
    pairwise_distances = torch.cdist(positions, positions)

    assert pairwise_distances[0, 1] < pairwise_distances[0, 4]


def test_layout_kk_supports_single_node_graphs() -> None:
    """A single-node graph returns one finite 2D position."""
    edge_index = torch.empty((2, 0), dtype=torch.long)

    positions = layout_kk(edge_index=edge_index, num_nodes=1, steps=10, seed=13)

    assert positions.shape == (1, 2)
    assert torch.isfinite(positions).all()


@pytest.mark.parametrize("steps", [None, 0])
def test_layout_kk_omits_lbfgsb_maxiter_when_steps_is_unset(
    monkeypatch: pytest.MonkeyPatch,
    steps: Optional[int],
) -> None:
    """The NetworkX-matching KK path should leave ``maxiter`` unset by default."""
    scipy = pytest.importorskip("scipy")
    captured_kwargs: dict[str, Any] = {}

    def fake_minimize(
        objective: Any,
        initial_vector: np.ndarray,
        **kwargs: Any,
    ) -> SimpleNamespace:
        """Capture the SciPy optimizer kwargs without running L-BFGS-B."""
        _ = objective
        captured_kwargs.update(kwargs)
        return SimpleNamespace(x=np.asarray(initial_vector, dtype=np.float64))

    monkeypatch.setattr(scipy.optimize, "minimize", fake_minimize)

    distance_matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    initial_positions = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)

    solved_positions, traces = kk_module._solve_kamada_kawai(
        distance_matrix=distance_matrix,
        initial_positions=initial_positions,
        steps=steps,
        trace_every=0,
    )

    assert "options" not in captured_kwargs
    np.testing.assert_allclose(solved_positions, initial_positions)
    assert traces == []
