"""Tests for the r81-P2 native-stress ``target_unit`` knob.

``NativeStressConfig.target_unit`` defaults to ``"hops"`` everywhere,
preserving today's behavior (stress targets in bare graph-distance units).
``"points"`` is opt-in only: it inserts ``ScaleStressTargetDistances`` passes
that scale every target representation by the mean adjacent summed node radii
so targets and node boxes share one unit (see the op docstring for the
measured defect this fixes).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dagua.layout.ops.native_stress import (
    STRESS_TARGET_UNIT_SCALE_KEY,
    ScaleStressTargetDistances,
    ScaleStressTargetDistancesConfig,
    _mean_adjacent_radii_sum,
)
from dagua.layout.ops.pipelines.native_stress import (
    NativeStressConfig,
    _resolve_native_stress_config,
    build_native_stress_pipeline,
    layout_native_stress_pipeline,
)
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


def _ring_edge_index(num_nodes: int) -> torch.Tensor:
    """Return a simple ring edge index.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the ring.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, num_nodes]``.
    """
    edges = [(i, (i + 1) % num_nodes) for i in range(num_nodes)]
    return torch.tensor(list(zip(*edges)), dtype=torch.long)


def test_target_unit_defaults_to_hops() -> None:
    """The default config keeps today's hop-unit targets."""
    assert NativeStressConfig().target_unit == "hops"


def test_target_unit_rejects_unknown_value() -> None:
    """An unsupported unit name raises during config resolution."""
    with pytest.raises(ValueError, match="target_unit"):
        _resolve_native_stress_config(
            num_nodes=10,
            config=NativeStressConfig(target_unit="furlongs"),
        )


def test_default_pipeline_op_list_has_no_scale_op() -> None:
    """The hop-unit pipeline op list is literally unchanged (default path)."""
    pipeline = build_native_stress_pipeline(NativeStressConfig())
    names = [type(op).__name__ for op in pipeline.ops]
    assert "ScaleStressTargetDistances" not in names


def test_points_pipeline_inserts_three_scale_passes() -> None:
    """Opting in inserts one scale pass per target-materializing stage."""
    pipeline = build_native_stress_pipeline(NativeStressConfig(target_unit="points"))
    scale_ops = [op for op in pipeline.ops if type(op).__name__ == "ScaleStressTargetDistances"]
    assert len(scale_ops) == 3
    assert [op.config.targets for op in scale_ops] == [("pivot",), ("exact",), ("sgd2",)]


def test_target_unit_hops_preserves_default_positions() -> None:
    """Explicit ``"hops"`` must reproduce the no-config call bit-identically."""
    edge_index = _ring_edge_index(10)
    node_sizes = torch.full((10, 2), 20.0)

    default_pos = layout_native_stress_pipeline(
        edge_index=edge_index,
        num_nodes=10,
        node_sizes=node_sizes,
        seed=7,
    )
    explicit_hops_pos = layout_native_stress_pipeline(
        edge_index=edge_index,
        num_nodes=10,
        node_sizes=node_sizes,
        seed=7,
        config=NativeStressConfig(target_unit="hops", seed=7),
    )

    torch.testing.assert_close(default_pos, explicit_hops_pos, rtol=0.0, atol=0.0)


def test_target_unit_points_changes_positions_and_stays_finite() -> None:
    """``"points"`` produces a different, finite, larger-scale layout."""
    edge_index = _ring_edge_index(12)
    node_sizes = torch.full((12, 2), 30.0)

    hops_pos = layout_native_stress_pipeline(
        edge_index=edge_index,
        num_nodes=12,
        node_sizes=node_sizes,
        seed=3,
        config=NativeStressConfig(target_unit="hops", seed=3),
    )
    points_pos = layout_native_stress_pipeline(
        edge_index=edge_index,
        num_nodes=12,
        node_sizes=node_sizes,
        seed=3,
        config=NativeStressConfig(target_unit="points", seed=3),
    )

    assert bool(torch.isfinite(hops_pos).all())
    assert bool(torch.isfinite(points_pos).all())
    assert not torch.allclose(hops_pos, points_pos)


def test_mean_adjacent_radii_sum_matches_manual_value() -> None:
    """Unit scale equals the mean adjacent half-diagonal sum in points."""
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    node_sizes = torch.tensor([[30.0, 40.0], [30.0, 40.0], [60.0, 80.0]])
    problem = LayoutProblem(edge_index=edge_index, num_nodes=3, node_sizes=node_sizes)
    # radii: 25, 25, 50 -> edges (0,1): 50 and (1,2): 75 -> mean 62.5
    assert _mean_adjacent_radii_sum(problem) == pytest.approx(62.5)


def test_scale_op_scales_exact_terms_and_recomputes_weights() -> None:
    """The exact-term pass multiplies distances by K and rebuilds 1/d^2."""
    edge_index = _ring_edge_index(4)
    node_sizes = torch.full((4, 2), 30.0)  # radius 15*sqrt(2); K = 30*sqrt(2)
    problem = LayoutProblem(edge_index=edge_index, num_nodes=4, node_sizes=node_sizes)
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    state = SolveState()
    state.extras["stress_sgd_sources"] = np.array([0, 0], dtype=np.int32)
    state.extras["stress_sgd_targets"] = np.array([1, 2], dtype=np.int32)
    state.extras["stress_sgd_distances"] = np.array([1.0, 2.0], dtype=np.float64)
    state.extras["stress_sgd_weights"] = np.array([1.0, 0.25], dtype=np.float64)

    op = ScaleStressTargetDistances(ScaleStressTargetDistancesConfig(targets=("exact",)))
    state = op.apply(problem, state, ctx)

    expected_scale = float(2.0 * 15.0 * np.sqrt(2.0) * 2.0 / 2.0)  # r_i + r_j for equal nodes
    assert state.extras[STRESS_TARGET_UNIT_SCALE_KEY] == pytest.approx(expected_scale)
    np.testing.assert_allclose(
        state.extras["stress_sgd_distances"],
        np.array([1.0, 2.0]) * expected_scale,
    )
    np.testing.assert_allclose(
        state.extras["stress_sgd_weights"],
        1.0 / np.square(np.array([1.0, 2.0]) * expected_scale),
    )


def test_scale_op_fixed_mode_and_validation() -> None:
    """Fixed mode uses the explicit value; bad configs raise."""
    edge_index = _ring_edge_index(4)
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=4,
        node_sizes=torch.full((4, 2), 10.0),
    )
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    state = SolveState()
    state.pivot_distances = torch.ones((2, 4))

    op = ScaleStressTargetDistances(
        ScaleStressTargetDistancesConfig(mode="fixed", value=3.0, targets=("pivot",))
    )
    state = op.apply(problem, state, ctx)
    torch.testing.assert_close(state.pivot_distances, torch.full((2, 4), 3.0))
    assert state.extras[STRESS_TARGET_UNIT_SCALE_KEY] == pytest.approx(3.0)

    with pytest.raises(ValueError, match="mode"):
        ScaleStressTargetDistances(ScaleStressTargetDistancesConfig(mode="bogus")).apply(
            problem, SolveState(), ctx
        )
    with pytest.raises(ValueError, match="value"):
        ScaleStressTargetDistances(ScaleStressTargetDistancesConfig(mode="fixed", value=0.0)).apply(
            problem, SolveState(), ctx
        )
    with pytest.raises(ValueError, match="targets"):
        ScaleStressTargetDistances(ScaleStressTargetDistancesConfig(targets=("bogus",))).apply(
            problem, SolveState(), ctx
        )


def test_smacof_prep_consumes_unit_scale_from_extras() -> None:
    """SMACOF target rebuild honors the recorded unit scale."""
    from dagua.layout.ops.native_stress import (
        PrepareWarmStartStressMajorization,
        PrepareWarmStartStressMajorizationConfig,
    )

    edge_index = _ring_edge_index(5)
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=5,
        node_sizes=torch.full((5, 2), 20.0),
    )
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    prep = PrepareWarmStartStressMajorization(
        PrepareWarmStartStressMajorizationConfig(size_aware=False)
    )

    baseline_state = SolveState(pos=torch.randn(5, 2))
    baseline_state = prep.apply(problem, baseline_state, ctx)

    scaled_state = SolveState(pos=torch.randn(5, 2))
    scaled_state.extras[STRESS_TARGET_UNIT_SCALE_KEY] = 10.0
    scaled_state = prep.apply(problem, scaled_state, ctx)

    torch.testing.assert_close(
        scaled_state.distance_matrix,
        baseline_state.distance_matrix * 10.0,
    )
