"""Tests for the classic ForceAtlas2 layout."""

from __future__ import annotations

import torch

from dagua.layout.classic import layout_fa2


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
    """Change the final layout when LinLog attraction is enabled."""
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
