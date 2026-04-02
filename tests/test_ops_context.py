"""Tests for context ops."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dagua.layout.ops.context import (
    BuildDensityGrid,
    BuildDensityGridConfig,
    BuildEdgeBatchCtx,
    BuildEdgeBatchCtxConfig,
    BuildQuadTree,
    BuildQuadTreeConfig,
    QuadTreeNode,
    RefreshKDTreePairs,
    RefreshKDTreePairsConfig,
    RefreshSampledNodeCtx,
    RefreshSampledNodeCtxConfig,
)
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState


def _collect_quadtree_indices(node: QuadTreeNode | None) -> list[int]:
    """Collect all particle indices stored in a quadtree.

    Parameters
    ----------
    node : QuadTreeNode or None
        Root or subtree node.

    Returns
    -------
    list[int]
        Sorted particle indices reachable from the subtree.
    """
    if node is None:
        return []
    if node.indices is not None:
        return [int(index) for index in node.indices.tolist()]

    indices: list[int] = []
    if node.children is not None:
        for child in node.children:
            indices.extend(_collect_quadtree_indices(child))
    return indices


def _max_quadtree_depth(node: QuadTreeNode | None) -> int:
    """Return the deepest depth present in a quadtree.

    Parameters
    ----------
    node : QuadTreeNode or None
        Root or subtree node.

    Returns
    -------
    int
        Maximum depth in the subtree, or ``-1`` for an empty tree.
    """
    if node is None:
        return -1
    child_depth = max(
        (_max_quadtree_depth(child) for child in node.children or ()),
        default=node.depth,
    )
    return max(node.depth, child_depth)


def test_build_edge_batch_ctx_uses_contiguous_batches_and_filters_self_loops() -> None:
    """Edge-batch context should rotate chunks and drop self-loops."""
    problem = LayoutProblem(
        edge_index=torch.tensor(
            [
                [0, 1, 2, 2],
                [1, 1, 3, 2],
            ],
            dtype=torch.long,
        ),
        num_nodes=4,
    )
    state = SolveState(
        pos=torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 1.0],
                [3.0, 1.5],
            ],
            dtype=torch.float32,
        ),
        step=1,
    )
    op = BuildEdgeBatchCtx(config=BuildEdgeBatchCtxConfig(batch_size=2))

    updated = op.apply(problem, state, RuntimeContext())

    assert updated.edge_batch_context is not None
    assert updated.edge_batch_context.src.tolist() == [2]
    assert updated.edge_batch_context.tgt.tolist() == [3]
    assert torch.allclose(updated.edge_batch_context.dx, torch.tensor([-1.0]))
    assert torch.allclose(updated.edge_batch_context.dy, torch.tensor([-0.5]))
    assert torch.allclose(updated.edge_batch_context.dist_sq, torch.tensor([1.25]))


def test_refresh_sampled_node_ctx_builds_once_then_reuses_until_interval() -> None:
    """Sampled-node refresh should respect its cadence and active cap."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1234)
    problem = LayoutProblem(edge_index=torch.zeros((2, 0), dtype=torch.long), num_nodes=10, seed=11)
    state = SolveState(layers=torch.arange(10, dtype=torch.long) % 3, step=0)
    op = RefreshSampledNodeCtx(config=RefreshSampledNodeCtxConfig(interval=5, active_cap=4))

    first = op.apply(problem, state, RuntimeContext(generator=generator))

    assert first.sampled_node_context is not None
    assert first.sampled_node_context.active_idx.shape == (4,)
    assert first.sampled_node_context.sampled.shape[0] == 4
    assert int(first.sampled_node_context.active_idx.max().item()) < 10

    cached = first.sampled_node_context
    first.step = 1
    second = op.apply(problem, first, RuntimeContext(generator=generator))

    assert second.sampled_node_context is cached


def test_build_quadtree_covers_all_random_nodes_with_bounded_depth() -> None:
    """Barnes-Hut tree build should retain every node exactly once."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(7)
    pos = torch.rand((20, 2), generator=generator, dtype=torch.float32)
    state = SolveState(pos=pos, degree=torch.arange(20, dtype=torch.float32))
    op = BuildQuadTree(config=BuildQuadTreeConfig(max_depth=6))

    updated = op.apply(
        LayoutProblem(edge_index=torch.zeros((2, 0), dtype=torch.long), num_nodes=20),
        state,
        RuntimeContext(),
    )

    root = updated.extras["quadtree"]
    assert root is not None
    assert root.mass > 0.0
    assert sorted(_collect_quadtree_indices(root)) == list(range(20))
    assert _max_quadtree_depth(root) <= 6


def test_build_density_grid_registers_all_nodes() -> None:
    """Density-grid build should record every positioned node."""
    state = SolveState(
        pos=torch.tensor(
            [
                [-1.0, -1.0],
                [0.0, 0.0],
                [0.5, 0.25],
                [1.0, 1.0],
            ],
            dtype=torch.float32,
        )
    )
    op = BuildDensityGrid(config=BuildDensityGridConfig(grid_size=32, view_size=8.0))

    updated = op.apply(
        LayoutProblem(edge_index=torch.zeros((2, 0), dtype=torch.long), num_nodes=4),
        state,
        RuntimeContext(),
    )

    density_grid = updated.extras["density_grid"]
    assert density_grid.density.shape == (32, 32)
    assert len(density_grid.node_cells) == 4
    assert float(density_grid.density.sum().item()) > 0.0


def test_refresh_kdtree_pairs_respects_radius_and_interval() -> None:
    """KD-tree pair refresh should cache results between refresh steps."""
    pytest.importorskip("scipy")

    pos = torch.tensor(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [1.0, 1.0],
            [1.25, 1.1],
        ],
        dtype=torch.float32,
    )
    state = SolveState(pos=pos, step=0)
    op = RefreshKDTreePairs(config=RefreshKDTreePairsConfig(radius=0.3, interval=5))

    updated = op.apply(
        LayoutProblem(edge_index=torch.zeros((2, 0), dtype=torch.long), num_nodes=4),
        state,
        RuntimeContext(),
    )

    expected = np.asarray([[0, 1], [2, 3]], dtype=np.int64)
    assert np.array_equal(updated.extras["kdtree_pairs"], expected)

    cached = updated.extras["kdtree_pairs"].copy()
    updated.pos = updated.pos + 10.0
    updated.step = 1
    reused = op.apply(
        LayoutProblem(edge_index=torch.zeros((2, 0), dtype=torch.long), num_nodes=4),
        updated,
        RuntimeContext(),
    )

    assert np.array_equal(reused.extras["kdtree_pairs"], cached)


def test_build_edge_batch_ctx_returns_empty_context_for_self_loops_only() -> None:
    """Edge-batch context should be empty when the selected batch is all self-loops."""

    problem = LayoutProblem(
        edge_index=torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
        num_nodes=2,
    )
    state = SolveState(pos=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32))

    result = BuildEdgeBatchCtx().apply(problem, state, RuntimeContext())

    assert result.edge_batch_context is not None
    assert result.edge_batch_context.src.numel() == 0
    assert result.edge_batch_context.tgt.numel() == 0


def test_refresh_sampled_node_ctx_refreshes_again_at_interval_boundary() -> None:
    """Sampled-node refresh should rebuild the context once the cadence is hit."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(4321)
    problem = LayoutProblem(edge_index=torch.zeros((2, 0), dtype=torch.long), num_nodes=12, seed=5)
    state = SolveState(layers=torch.arange(12, dtype=torch.long) % 4, step=0)
    op = RefreshSampledNodeCtx(config=RefreshSampledNodeCtxConfig(interval=3, active_cap=4))

    first = op.apply(problem, state, RuntimeContext(generator=generator))
    cached = first.sampled_node_context
    first.step = 3
    refreshed = op.apply(problem, first, RuntimeContext(generator=generator))

    assert refreshed.sampled_node_context is not None
    assert refreshed.sampled_node_context is not cached


def test_build_quadtree_root_mass_matches_degree_augmented_mass_sum() -> None:
    """Barnes-Hut root mass should equal the sum of ``degree + 1`` masses."""

    degree = torch.tensor([0.0, 1.0, 3.0], dtype=torch.float32)
    state = SolveState(
        pos=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=torch.float32),
        degree=degree,
    )

    result = BuildQuadTree().apply(
        LayoutProblem(edge_index=torch.zeros((2, 0), dtype=torch.long), num_nodes=3),
        state,
        RuntimeContext(),
    )

    root = result.extras["quadtree"]
    assert root is not None
    assert root.mass == pytest.approx(float((degree + 1.0).sum().item()), rel=1.0e-6)
