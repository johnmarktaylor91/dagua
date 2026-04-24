"""Tests for Sprint 18 ClusterGridArrange op."""

from __future__ import annotations

import torch

from dagua.layout.ops.cluster_arrange import (
    ClusterGridArrange,
    ClusterGridArrangeConfig,
)
from dagua.layout.ops.state import (
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)


def _ctx() -> RuntimeContext:
    return RuntimeContext(plan=ExecutionPlan(device="cpu", optimizer_type="adam"))


def _problem(n: int, clusters: dict | None = None) -> LayoutProblem:
    return LayoutProblem(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=n,
        node_sizes=torch.ones(n, 2) * 20.0,
        seed=42,
        clusters=clusters or {},
    )


def test_cluster_grid_arrange_default_off() -> None:
    """Default config has enabled=False (Sprint 18 decision)."""
    cfg = ClusterGridArrangeConfig()
    assert cfg.enabled is False


def test_cluster_grid_arrange_no_op_when_disabled() -> None:
    """enabled=False -> no-op."""
    n = 30
    pos = torch.zeros(n, 2)
    pos[:, 1] = torch.linspace(0, 100, n)
    state = SolveState(pos=pos.clone())
    op = ClusterGridArrange(ClusterGridArrangeConfig(enabled=False))

    out = op.apply(_problem(n, {"a": list(range(15)), "b": list(range(15, 30))}), state, _ctx())

    assert torch.allclose(out.pos, pos)


def test_cluster_grid_arrange_no_op_without_clusters() -> None:
    """No problem.clusters -> no-op even when enabled."""
    n = 30
    pos = torch.zeros(n, 2)
    pos[:, 1] = torch.linspace(0, 100, n)
    state = SolveState(pos=pos.clone())
    op = ClusterGridArrange(ClusterGridArrangeConfig(enabled=True))

    out = op.apply(_problem(n, clusters=None), state, _ctx())

    assert torch.allclose(out.pos, pos)


def test_cluster_grid_arrange_no_op_when_well_shaped() -> None:
    """Centroids span > threshold of layout -> no-op (already spread)."""
    n = 6
    # Two clusters of 3, well separated horizontally
    pos = torch.tensor(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
            [100.0, 5.0],
            [110.0, 5.0],
            [120.0, 5.0],
        ]
    )
    state = SolveState(pos=pos.clone())
    op = ClusterGridArrange(ClusterGridArrangeConfig(enabled=True))

    out = op.apply(_problem(n, {"a": [0, 1, 2], "b": [3, 4, 5]}), state, _ctx())

    # No movement -- centroids already spread
    assert torch.allclose(out.pos, pos)


def test_cluster_grid_arrange_fires_on_stacked_centroids() -> None:
    """When all clusters share x, op rearranges them on a grid."""
    n = 6
    # Two clusters of 3, all at x=0 but spread vertically (the stacked
    # symptom from clustered_deep)
    pos = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 10.0],
            [0.0, 20.0],
            [0.0, 100.0],
            [0.0, 110.0],
            [0.0, 120.0],
        ]
    )
    state = SolveState(pos=pos.clone())
    op = ClusterGridArrange(ClusterGridArrangeConfig(enabled=True))

    out = op.apply(_problem(n, {"a": [0, 1, 2], "b": [3, 4, 5]}), state, _ctx())

    # X must change for at least one cluster (stacked -> grid)
    assert not torch.allclose(out.pos[:, 0], pos[:, 0])
    # Intra-cluster relative positions preserved (rigid translation)
    a_orig = pos[:3] - pos[:3].mean(dim=0)
    a_new = out.pos[:3] - out.pos[:3].mean(dim=0)
    assert torch.allclose(a_orig, a_new, atol=1e-4)


def test_cluster_grid_arrange_preserves_depth_order() -> None:
    """First cluster (lowest origin y) lands at lowest target y."""
    n = 16
    # 4 vertically stacked clusters of 4 each (gives 2x2 grid)
    pos = torch.zeros(n, 2)
    for cluster_idx in range(4):
        for member_idx in range(4):
            row = cluster_idx * 4 + member_idx
            pos[row, 0] = 0.0  # all at x=0
            pos[row, 1] = cluster_idx * 100.0 + member_idx * 5.0
    state = SolveState(pos=pos.clone())
    op = ClusterGridArrange(ClusterGridArrangeConfig(enabled=True))

    clusters = {
        "stage_3": [12, 13, 14, 15],
        "stage_0": [0, 1, 2, 3],
        "stage_1": [4, 5, 6, 7],
        "stage_2": [8, 9, 10, 11],
    }
    out = op.apply(_problem(n, clusters), state, _ctx())

    # In a 2x2 grid, stage_0 + stage_1 (lowest y origins) are in row 0
    # (lowest target y), stage_2 + stage_3 in row 1 (highest target y).
    centroids_y = {name: out.pos[indices, 1].mean().item() for name, indices in clusters.items()}
    assert centroids_y["stage_0"] < centroids_y["stage_2"]
    assert centroids_y["stage_1"] < centroids_y["stage_3"]
