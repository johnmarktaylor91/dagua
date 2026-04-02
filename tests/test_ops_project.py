"""Tests for projection ops."""

from __future__ import annotations

import pytest
import torch

from dagua.layout.ops.project import (
    BoundaryClamp,
    BoundaryClampConfig,
    HardPinProjection,
    MonotoneSafeguard,
    MonotoneSafeguardConfig,
    MovementClamp,
    MovementClampConfig,
    OverlapProjection,
    OverlapProjectionConfig,
)
from dagua.layout.ops.state import FlexConstraints, LayoutProblem, RuntimeContext, SolveState


def _make_problem(
    num_nodes: int = 2,
    node_sizes: torch.Tensor | None = None,
    flex: FlexConstraints | None = None,
) -> LayoutProblem:
    """Create a minimal layout problem for projection-op tests.

    Parameters
    ----------
    num_nodes : int, default=2
        Number of nodes in the synthetic graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]``.
    flex : FlexConstraints, optional
        Optional flex constraints payload.

    Returns
    -------
    LayoutProblem
        Minimal immutable problem instance.
    """
    return LayoutProblem(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        flex=flex,
    )


def test_overlap_projection_resolves_two_overlapping_nodes() -> None:
    """OverlapProjection should separate nodes whose boxes fully overlap."""

    problem = _make_problem(
        node_sizes=torch.tensor([[2.0, 2.0], [2.0, 2.0]], dtype=torch.float32),
    )
    state = SolveState(
        pos=torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32),
    )

    result = OverlapProjection().apply(problem, state, RuntimeContext())

    assert result.pos is not None
    delta = result.pos[0] - result.pos[1]
    min_sep = 2.0
    assert float(delta[0].abs().item()) >= min_sep or float(delta[1].abs().item()) >= min_sep


def test_hard_pin_projection_fixes_pinned_positions() -> None:
    """HardPinProjection should overwrite only hard-pinned axes."""

    flex = FlexConstraints(
        pin_indices=torch.tensor([0, 1], dtype=torch.long),
        pin_targets=torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
        hard_pin_mask=torch.tensor([[True, True], [False, True]], dtype=torch.bool),
    )
    problem = _make_problem(num_nodes=2, flex=flex)
    state = SolveState(
        pos=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
    )

    result = HardPinProjection().apply(problem, state, RuntimeContext())

    assert result.pos is not None
    torch.testing.assert_close(
        result.pos,
        torch.tensor([[5.0, 6.0], [3.0, 8.0]], dtype=torch.float32),
    )


def test_overlap_projection_honors_padding_when_separating_nodes() -> None:
    """OverlapProjection should leave at least the requested padded separation."""

    problem = _make_problem(
        num_nodes=2,
        node_sizes=torch.tensor([[2.0, 2.0], [2.0, 2.0]], dtype=torch.float32),
    )
    state = SolveState(pos=torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32))

    result = OverlapProjection(OverlapProjectionConfig(padding=1.0, iterations=20)).apply(
        problem,
        state,
        RuntimeContext(),
    )

    assert result.pos is not None
    delta = (result.pos[0] - result.pos[1]).abs()
    assert float(delta.max().item()) == pytest.approx(3.0, rel=1.0e-5, abs=1.0e-5)


def test_boundary_clamp_limits_coordinates_to_extent() -> None:
    """BoundaryClamp should clip every coordinate into ``[-extent, extent]``."""

    problem = _make_problem(num_nodes=3)
    state = SolveState(
        pos=torch.tensor([[-5.0, -1.0], [0.5, 9.0], [2.0, -7.5]], dtype=torch.float32)
    )

    result = BoundaryClamp(BoundaryClampConfig(extent=2.5)).apply(problem, state, RuntimeContext())

    assert result.pos is not None
    assert torch.all(result.pos <= 2.5)
    assert torch.all(result.pos >= -2.5)


def test_movement_clamp_fixed_mode_caps_large_displacements() -> None:
    """MovementClamp should rescale forces whose norm exceeds the fixed cap."""

    state = SolveState(forces=torch.tensor([[3.0, 4.0], [0.3, 0.4]], dtype=torch.float32))

    result = MovementClamp(MovementClampConfig(mode="fixed", max_delta=2.0)).apply(
        _make_problem(),
        state,
        RuntimeContext(),
    )

    assert result.forces is not None
    norms = torch.linalg.vector_norm(result.forces, dim=1)
    torch.testing.assert_close(norms, torch.tensor([2.0, 0.5], dtype=torch.float32))


def test_movement_clamp_temperature_mode_uses_global_temperature() -> None:
    """MovementClamp should use ``state.temperature`` as the displacement limit."""

    state = SolveState(
        forces=torch.tensor([[6.0, 8.0], [0.0, 1.0]], dtype=torch.float32),
        temperature=5.0,
    )

    result = MovementClamp().apply(_make_problem(), state, RuntimeContext())

    assert result.forces is not None
    norms = torch.linalg.vector_norm(result.forces, dim=1)
    torch.testing.assert_close(norms, torch.tensor([5.0, 1.0], dtype=torch.float32))


def test_monotone_safeguard_reverts_stress_increasing_move() -> None:
    """MonotoneSafeguard should fall back to the previous accepted positions."""

    problem = _make_problem()

    def objective(pos: torch.Tensor) -> float:
        """Return a simple convex scalar objective for safeguard tests."""

        return float((pos.square()).sum().item())

    op = MonotoneSafeguard(MonotoneSafeguardConfig(max_bisections=4, tolerance=0.0))
    state = SolveState(
        pos=torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
        extras={"monotone_objective": objective},
    )

    state = op.apply(problem, state, RuntimeContext())
    baseline = state.pos.clone()
    state.pos = torch.tensor([[10.0, 0.0], [10.0, 0.0]], dtype=torch.float32)
    state = op.apply(problem, state, RuntimeContext())

    assert state.pos is not None
    torch.testing.assert_close(state.pos, baseline)
    assert state.extras["monotone_previous_value"] == pytest.approx(1.0, rel=1.0e-6)
