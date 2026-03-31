"""Tests for the classic ForceAtlas2 layout."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from dagua.layout.classic import layout_fa2
from dagua.layout.classic.fa2 import (
    _adjust_speed_and_apply_forces,
    _attraction_force,
    _build_barnes_hut_tree,
    _compute_degree,
    _gravity_force,
)
from dagua.layout.classic.tsnet import _gradient_descent_step


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
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _cluster_bridge_graph() -> tuple[torch.Tensor, int]:
    """Create a small graph with two dense clusters and a bridge.

    Parameters
    ----------
    None
        No parameters.

    Returns
    -------
    tuple[torch.Tensor, int]
        Edge tensor and node count.
    """
    edges = [
        (0, 1),
        (1, 2),
        (2, 0),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 3),
        (1, 4),
    ]
    return _edge_index(edges), 6


def _star_graph(num_leaves: int) -> tuple[torch.Tensor, int]:
    """Create a directed star centered at node zero.

    Parameters
    ----------
    num_leaves : int
        Number of leaf nodes connected to the hub.

    Returns
    -------
    tuple[torch.Tensor, int]
        Edge tensor and node count.
    """
    edges = [(0, leaf) for leaf in range(1, num_leaves + 1)]
    return _edge_index(edges), num_leaves + 1


def _path_graph(num_nodes: int) -> tuple[torch.Tensor, int]:
    """Create a simple directed path graph.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the chain.

    Returns
    -------
    tuple[torch.Tensor, int]
        Edge tensor and node count.
    """
    edges = [(index, index + 1) for index in range(num_nodes - 1)]
    return _edge_index(edges), num_nodes


def _stability_graph() -> tuple[torch.Tensor, int]:
    """Create the small regression graph used for FA2 stability checks.

    Parameters
    ----------
    None
        No parameters.

    Returns
    -------
    tuple[torch.Tensor, int]
        Edge tensor and node count.
    """
    edges = [
        (0, 1),
        (1, 2),
        (2, 0),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 3),
        (1, 4),
    ]
    return _edge_index(edges), 6


def test_layout_fa2_returns_positions_with_expected_shape() -> None:
    """Return a finite ``[N, 2]`` position tensor."""
    edge_index, num_nodes = _cluster_bridge_graph()

    pos = layout_fa2(edge_index=edge_index, num_nodes=num_nodes, steps=60, seed=7)

    assert isinstance(pos, torch.Tensor)
    assert pos.shape == (num_nodes, 2)
    assert torch.isfinite(pos).all()


def test_layout_fa2_is_deterministic_for_same_seed() -> None:
    """Produce identical layouts for repeated seeded runs."""
    edge_index, num_nodes = _cluster_bridge_graph()

    pos_a = layout_fa2(edge_index=edge_index, num_nodes=num_nodes, steps=80, seed=19)
    pos_b = layout_fa2(edge_index=edge_index, num_nodes=num_nodes, steps=80, seed=19)

    assert torch.allclose(pos_a, pos_b)


def test_layout_fa2_gravity_keeps_nodes_bounded() -> None:
    """Keep disconnected nodes within a reasonable radius of the centroid."""
    edge_index = torch.empty((2, 0), dtype=torch.long)
    pos = layout_fa2(
        edge_index=edge_index,
        num_nodes=24,
        steps=120,
        seed=11,
        gravity=2.5,
        scaling_ratio=0.5,
    )

    centered = pos - pos.mean(dim=0, keepdim=True)
    max_radius = centered.norm(dim=1).max().item()

    assert max_radius < 250.0


def test_layout_fa2_linlog_mode_changes_layout() -> None:
    """Produce a different layout when LinLog attraction is enabled."""
    edge_index, num_nodes = _cluster_bridge_graph()

    default_pos = layout_fa2(edge_index=edge_index, num_nodes=num_nodes, steps=90, seed=5)
    linlog_pos = layout_fa2(
        edge_index=edge_index,
        num_nodes=num_nodes,
        steps=90,
        seed=5,
        linlog=True,
    )

    assert not torch.allclose(default_pos, linlog_pos)


def test_attraction_force_linlog_matches_reference_log_attraction() -> None:
    """Match the reference LinLog attraction scaling and raw-delta direction."""
    pos = torch.tensor([[0.0, 0.0], [3.0, 4.0]], dtype=torch.float32)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    mass = torch.tensor([2.0, 1.0], dtype=torch.float32)

    force = _attraction_force(
        pos=pos,
        edge_index=edge_index,
        mass=mass,
        outbound_att_compensation=3.0,
        outbound_attraction_distribution=True,
        linlog=True,
    )

    delta = pos[0] - pos[1]
    distance = torch.linalg.vector_norm(delta)
    expected_factor = -(3.0 * torch.log1p(distance) / distance) / mass[0]
    expected_attraction = delta * expected_factor
    expected_force = torch.stack((expected_attraction, -expected_attraction))

    torch.testing.assert_close(force, expected_force)


def test_attraction_force_applies_edge_weight_influence() -> None:
    """Raise edge weights to the requested attraction exponent."""
    pos = torch.tensor([[0.0, 0.0], [3.0, 0.0]], dtype=torch.float32)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    mass = torch.tensor([1.0, 1.0], dtype=torch.float32)

    force = _attraction_force(
        pos=pos,
        edge_index=edge_index,
        mass=mass,
        outbound_att_compensation=1.0,
        outbound_attraction_distribution=False,
        edge_weights=torch.tensor([4.0], dtype=torch.float32),
        edge_weight_influence=0.5,
    )

    expected_force = torch.tensor([[6.0, 0.0], [-6.0, 0.0]], dtype=torch.float32)
    torch.testing.assert_close(force, expected_force)


def test_layout_fa2_strong_gravity_changes_layout() -> None:
    """Change the final layout when strong gravity is enabled."""
    edge_index, num_nodes = _cluster_bridge_graph()

    default_pos = layout_fa2(edge_index=edge_index, num_nodes=num_nodes, steps=90, seed=5)
    strong_pos = layout_fa2(
        edge_index=edge_index,
        num_nodes=num_nodes,
        steps=90,
        seed=5,
        strong_gravity=True,
    )

    assert not torch.allclose(default_pos, strong_pos)


def test_gravity_force_strong_mode_skips_axis_aligned_nodes() -> None:
    """Match the reference strong-gravity zero guard on the axes."""
    pos = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 2.0]], dtype=torch.float32)
    mass = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float32)

    force = _gravity_force(
        pos=pos,
        mass=mass,
        gravity=1.5,
        strong_gravity=True,
        scaling_ratio=2.0,
    )

    expected_force = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [-12.0, -24.0]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(force, expected_force)


def test_layout_fa2_trace_mode_returns_snapshots() -> None:
    """Return traced position snapshots at the requested cadence."""
    edge_index, num_nodes = _cluster_bridge_graph()

    pos, traces = layout_fa2(
        edge_index=edge_index,
        num_nodes=num_nodes,
        steps=25,
        seed=3,
        trace_every=10,
    )

    assert pos.shape == (num_nodes, 2)
    assert len(traces) == 3
    assert all(trace.shape == (num_nodes, 2) for trace in traces)
    assert torch.isfinite(torch.stack(traces)).all()


def test_layout_fa2_gives_hubs_more_space_than_a_path() -> None:
    """Place a star hub farther from its neighbors than a path edge length."""
    star_edge_index, star_nodes = _star_graph(num_leaves=10)
    path_edge_index, path_nodes = _path_graph(num_nodes=11)

    star_pos = layout_fa2(
        edge_index=star_edge_index,
        num_nodes=star_nodes,
        steps=140,
        seed=13,
        gravity=1.5,
    )
    path_pos = layout_fa2(
        edge_index=path_edge_index,
        num_nodes=path_nodes,
        steps=140,
        seed=13,
        gravity=1.5,
    )

    hub_distance = (star_pos[1:] - star_pos[0]).norm(dim=1).mean().item()
    path_distance = (path_pos[1:] - path_pos[:-1]).norm(dim=1).mean().item()

    assert hub_distance > path_distance * 1.1


def test_build_barnes_hut_tree_uses_mass_center_diameter() -> None:
    """Size Barnes-Hut regions from the mass-center diameter."""
    pos = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    mass = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    root = _build_barnes_hut_tree(
        pos_np=pos,
        mass_np=mass,
        indices=np.array([0, 1, 2], dtype=np.int64),
    )

    assert root is not None
    assert math.isclose(root.mass_center_x, 4.0 / 3.0)
    assert math.isclose(root.mass_center_y, 1.0 / 3.0)
    expected_size = 2.0 * max(
        math.dist((4.0 / 3.0, 1.0 / 3.0), tuple(position)) for position in pos.tolist()
    )
    assert math.isclose(root.size, expected_size)


def test_build_barnes_hut_tree_splits_degenerate_quadrants_into_singletons() -> None:
    """Break all-in-one-quadrant regions into singleton leaves."""
    pos = np.array([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]], dtype=np.float64)
    mass = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    root = _build_barnes_hut_tree(
        pos_np=pos,
        mass_np=mass,
        indices=np.array([0, 1, 2], dtype=np.int64),
    )

    assert root is not None
    assert root.children is not None
    assert len(root.children) == 3
    assert all(child.indices is not None for child in root.children)
    assert [int(child.indices[0]) for child in root.children] == [0, 1, 2]


def test_adjust_speed_and_apply_forces_matches_reference_math() -> None:
    """Match the reference FA2 speed update and per-node movement factors."""
    pos = torch.tensor([[0.0, 0.0], [1.0, -1.0]], dtype=torch.float32)
    old_force = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
    force = torch.tensor([[0.0, 1.0], [2.0, 0.0]], dtype=torch.float32)
    mass = torch.tensor([2.0, 3.0], dtype=torch.float32)

    updated_pos, speed, speed_efficiency = _adjust_speed_and_apply_forces(
        pos=pos,
        force=force,
        old_force=old_force,
        mass=mass,
        speed=1.0,
        speed_efficiency=1.0,
        jitter_tolerance=1.0,
    )

    total_swinging = 5.0 * math.sqrt(2.0)
    total_effective_traction = 2.5 * math.sqrt(2.0)
    estimated_optimal_jt = 0.05 * math.sqrt(2.0)
    min_jt = math.sqrt(estimated_optimal_jt)
    jt = max(min_jt, estimated_optimal_jt * total_effective_traction / 4.0)
    target_speed = jt * total_effective_traction / total_swinging
    expected_speed = 1.0 + min(target_speed - 1.0, 0.5)
    expected_speed_efficiency = 0.7
    node_swinging = mass * torch.linalg.vector_norm(old_force - force, dim=1)
    expected_factor = expected_speed / (1.0 + torch.sqrt(expected_speed * node_swinging))
    expected_pos = pos + (force * expected_factor.unsqueeze(1))

    torch.testing.assert_close(updated_pos, expected_pos)
    assert speed == pytest.approx(expected_speed)
    assert speed_efficiency == pytest.approx(expected_speed_efficiency)


def test_compute_degree_deduplicates_undirected_edges() -> None:
    """FA2 mass should use deduplicated undirected degree counts."""
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)

    degree = _compute_degree(edge_index=edge_index, num_nodes=3)

    torch.testing.assert_close(degree, torch.tensor([1.0, 2.0, 1.0], dtype=torch.float32))


def test_layout_fa2_remains_finite_for_longer_run() -> None:
    """Keep a representative clustered graph finite during a longer run."""
    edge_index, num_nodes = _stability_graph()

    pos = layout_fa2(edge_index=edge_index, num_nodes=num_nodes, steps=200)

    assert torch.isfinite(pos).all()
    assert pos.abs().max().item() < 1_000.0


def test_tsnet_gradient_descent_step_matches_reference_rule() -> None:
    """tsNET should use the original per-parameter gains update rule."""
    positions = torch.tensor([[1.0, -1.0], [0.5, 0.25]], dtype=torch.float32)
    grad = torch.tensor([[2.0, -3.0], [-4.0, 5.0]], dtype=torch.float32)
    update = torch.tensor([[-0.1, -0.2], [0.3, -0.4]], dtype=torch.float32)
    gains = torch.tensor([[1.0, 2.0], [0.5, 0.25]], dtype=torch.float32)
    expected_positions_input = positions.clone()
    expected_update_input = update.clone()

    next_positions, next_update, next_gains = _gradient_descent_step(
        positions=positions,
        grad=grad,
        update=update,
        gains=gains,
        learning_rate=0.1,
        momentum=0.5,
        min_gain=0.01,
    )

    expected_gains = torch.tensor([[1.2, 1.6], [0.7, 0.45]], dtype=torch.float32)
    expected_grad = grad * expected_gains
    expected_update = (0.5 * expected_update_input) - (0.1 * expected_grad)
    expected_positions = expected_positions_input + expected_update

    torch.testing.assert_close(next_gains, expected_gains)
    torch.testing.assert_close(next_update, expected_update)
    torch.testing.assert_close(next_positions, expected_positions)
