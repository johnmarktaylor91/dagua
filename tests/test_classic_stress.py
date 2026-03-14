"""Tests for the classic Stress-SGD layout."""

from __future__ import annotations

import torch

from dagua.layout.classic import layout_stress_sgd


def _path_edge_index(num_nodes: int) -> torch.Tensor:
    """Build a directed path edge index.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the path.

    Returns
    -------
    torch.Tensor
        Edge index with shape ``[2, E]``.
    """
    if num_nodes <= 1:
        return torch.empty((2, 0), dtype=torch.long)
    source = torch.arange(0, num_nodes - 1, dtype=torch.long)
    target = source + 1
    return torch.stack([source, target], dim=0)


def test_layout_stress_sgd_returns_expected_shape() -> None:
    """The layout returns an ``[N, 2]`` position tensor."""
    edge_index = _path_edge_index(4)

    positions = layout_stress_sgd(edge_index=edge_index, num_nodes=4, steps=120, sample_size=64)

    assert positions.shape == (4, 2)


def test_layout_stress_sgd_is_deterministic_for_same_seed() -> None:
    """Repeated runs with the same seed produce identical layouts."""
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)

    positions_a = layout_stress_sgd(
        edge_index=edge_index,
        num_nodes=5,
        steps=150,
        seed=7,
        sample_size=80,
    )
    positions_b = layout_stress_sgd(
        edge_index=edge_index,
        num_nodes=5,
        steps=150,
        seed=7,
        sample_size=80,
    )

    assert torch.allclose(positions_a, positions_b)


def test_layout_stress_sgd_handles_disconnected_components() -> None:
    """Disconnected graphs layout without NaNs and keep components separated."""
    edge_index = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)

    positions = layout_stress_sgd(
        edge_index=edge_index,
        num_nodes=4,
        steps=180,
        seed=11,
        sample_size=64,
    )

    assert torch.isfinite(positions).all()

    centroid_a = positions[torch.tensor([0, 1])].mean(dim=0)
    centroid_b = positions[torch.tensor([2, 3])].mean(dim=0)
    centroid_distance = torch.norm(centroid_a - centroid_b)

    assert centroid_distance.item() > 0.1


def test_layout_stress_sgd_trace_mode_returns_snapshots() -> None:
    """Trace mode returns periodic position snapshots."""
    edge_index = _path_edge_index(5)

    result = layout_stress_sgd(
        edge_index=edge_index,
        num_nodes=5,
        steps=12,
        seed=3,
        sample_size=48,
        trace_every=4,
    )

    assert isinstance(result, tuple)
    positions, traces = result
    assert positions.shape == (5, 2)
    assert len(traces) == 3
    assert all(trace.shape == (5, 2) for trace in traces)
    assert torch.allclose(positions, traces[-1])


def test_layout_stress_sgd_respects_hop_distance_on_average() -> None:
    """Nodes with larger hop distance end up farther apart on average."""
    num_nodes = 6
    edge_index = _path_edge_index(num_nodes)

    positions = layout_stress_sgd(
        edge_index=edge_index,
        num_nodes=num_nodes,
        steps=250,
        seed=19,
        sample_size=96,
    )

    euclidean = torch.cdist(positions, positions)
    hop_one = euclidean[torch.arange(0, 5), torch.arange(1, 6)].mean()
    hop_two = euclidean[torch.arange(0, 4), torch.arange(2, 6)].mean()
    hop_three = euclidean[torch.arange(0, 3), torch.arange(3, 6)].mean()

    assert hop_one < hop_two < hop_three


def test_layout_stress_sgd_path_graph_is_roughly_linear() -> None:
    """A short path graph produces a strongly one-dimensional embedding."""
    edge_index = _path_edge_index(5)

    positions = layout_stress_sgd(
        edge_index=edge_index,
        num_nodes=5,
        steps=300,
        seed=23,
        sample_size=96,
    )

    centered = positions - positions.mean(dim=0, keepdim=True)
    _, singular_values, _ = torch.linalg.svd(centered, full_matrices=False)

    assert singular_values[0] > 2.0 * singular_values[1]
