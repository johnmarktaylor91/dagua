"""Tests for Sprint 17 Force2DInitIfFlat op."""

from __future__ import annotations

import torch

from dagua.layout.layers import LayerIndex
from dagua.layout.ops.force_2d_init import (
    Force2DInitIfFlat,
    Force2DInitIfFlatConfig,
)
from dagua.layout.ops.state import (
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)


def _problem(n: int, edges: torch.Tensor) -> LayoutProblem:
    return LayoutProblem(
        edge_index=edges,
        num_nodes=n,
        node_sizes=torch.ones(n, 2) * 20.0,
        seed=42,
    )


def _ctx() -> RuntimeContext:
    return RuntimeContext(plan=ExecutionPlan(device="cpu", optimizer_type="adam"))


def _layer_index_single(n: int) -> LayerIndex:
    """Return a layer_index with all nodes in 1 layer."""
    return LayerIndex(
        node_to_layer=torch.zeros(n, dtype=torch.long),
        layer_offsets=torch.tensor([0, n], dtype=torch.long),
        sorted_nodes=torch.arange(n, dtype=torch.long),
        num_layers=1,
    )


def _layer_index_multi(n: int, n_layers: int) -> LayerIndex:
    """Return a layer_index spreading nodes across multiple layers."""
    per_layer = n // n_layers
    node_to_layer = torch.zeros(n, dtype=torch.long)
    for i in range(n_layers):
        node_to_layer[i * per_layer : (i + 1) * per_layer] = i
    starts = [i * per_layer for i in range(n_layers)] + [n]
    return LayerIndex(
        node_to_layer=node_to_layer,
        layer_offsets=torch.tensor(starts, dtype=torch.long),
        sorted_nodes=torch.arange(n, dtype=torch.long),
        num_layers=n_layers,
    )


def test_force_2d_init_fires_on_single_layer() -> None:
    """When num_layers <= 1 and num_nodes >= min_nodes, randomize y."""
    n = 50
    pos = torch.zeros(n, 2)
    pos[:, 0] = torch.linspace(-100, 100, n)
    state = SolveState(pos=pos, layer_index=_layer_index_single(n))
    op = Force2DInitIfFlat(Force2DInitIfFlatConfig())

    out = op.apply(_problem(n, torch.empty((2, 0), dtype=torch.long)), state, _ctx())

    assert out.pos is not None
    # x should be unchanged
    assert torch.allclose(out.pos[:, 0], pos[:, 0])
    # y should be non-trivial (was all zeros, now spread)
    assert out.pos[:, 1].std().item() > 1.0


def test_force_2d_init_skips_when_disabled() -> None:
    """enabled=False -> no-op even when trigger condition met."""
    n = 50
    pos = torch.zeros(n, 2)
    pos[:, 0] = torch.linspace(-100, 100, n)
    state = SolveState(pos=pos.clone(), layer_index=_layer_index_single(n))
    op = Force2DInitIfFlat(Force2DInitIfFlatConfig(enabled=False))

    out = op.apply(_problem(n, torch.empty((2, 0), dtype=torch.long)), state, _ctx())

    assert torch.allclose(out.pos, pos)


def test_force_2d_init_skips_when_multi_layer() -> None:
    """Multi-layer (acyclic) graphs are unaffected."""
    n = 50
    pos = torch.zeros(n, 2)
    pos[:, 0] = torch.linspace(-100, 100, n)
    state = SolveState(pos=pos.clone(), layer_index=_layer_index_multi(n, 5))
    op = Force2DInitIfFlat(Force2DInitIfFlatConfig())

    out = op.apply(_problem(n, torch.empty((2, 0), dtype=torch.long)), state, _ctx())

    # Y unchanged because trigger didn't fire.
    assert torch.allclose(out.pos[:, 1], pos[:, 1])


def test_force_2d_init_skips_small_graphs() -> None:
    """num_nodes < min_nodes -> no-op even if 1 layer."""
    n = 5
    pos = torch.zeros(n, 2)
    pos[:, 0] = torch.linspace(-100, 100, n)
    state = SolveState(pos=pos.clone(), layer_index=_layer_index_single(n))
    op = Force2DInitIfFlat(Force2DInitIfFlatConfig(min_nodes=10))

    out = op.apply(_problem(n, torch.empty((2, 0), dtype=torch.long)), state, _ctx())

    assert torch.allclose(out.pos, pos)


def test_force_2d_init_deterministic_via_seed() -> None:
    """Same seed -> same y values."""
    n = 50
    pos = torch.zeros(n, 2)
    pos[:, 0] = torch.linspace(-100, 100, n)
    op = Force2DInitIfFlat(Force2DInitIfFlatConfig())

    state_a = SolveState(pos=pos.clone(), layer_index=_layer_index_single(n))
    state_b = SolveState(pos=pos.clone(), layer_index=_layer_index_single(n))
    out_a = op.apply(_problem(n, torch.empty((2, 0), dtype=torch.long)), state_a, _ctx())
    out_b = op.apply(_problem(n, torch.empty((2, 0), dtype=torch.long)), state_b, _ctx())

    assert torch.allclose(out_a.pos, out_b.pos)


def test_force_2d_init_y_extent_proportional_to_x() -> None:
    """y-extent should be ~ x_extent * extent_factor."""
    n = 50
    pos = torch.zeros(n, 2)
    pos[:, 0] = torch.linspace(-100, 100, n)
    state = SolveState(pos=pos.clone(), layer_index=_layer_index_single(n))
    op = Force2DInitIfFlat(Force2DInitIfFlatConfig(extent_factor=1.0))

    out = op.apply(_problem(n, torch.empty((2, 0), dtype=torch.long)), state, _ctx())

    x_extent = float(out.pos[:, 0].max() - out.pos[:, 0].min())
    y_extent = float(out.pos[:, 1].max() - out.pos[:, 1].min())
    # Should be within ~30% of x_extent (rand uniform in [-0.5, 0.5])
    assert abs(y_extent - x_extent) / x_extent < 0.3
