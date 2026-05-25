"""Regression tests for FA2 reference-fidelity controls."""

from __future__ import annotations

import torch

from dagua.eval.competitors.fa2_competitor import _FA2_REFERENCE_PACKAGE_ORDER
from dagua.layout.ops.force import FA2ForceStep, FA2ForceStepConfig
from dagua.layout.ops.pipelines.fa2 import FA2Config, build_fa2_pipeline, layout_fa2_pipeline
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


def _empty_problem(num_nodes: int) -> LayoutProblem:
    """Build an edgeless layout problem for FA2 op tests.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    LayoutProblem
        Problem with no edges and a deterministic seed.
    """
    return LayoutProblem(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=num_nodes,
        seed=42,
    )


def _runtime_context() -> RuntimeContext:
    """Build a CPU runtime context for deterministic FA2 tests.

    Returns
    -------
    RuntimeContext
        Runtime context targeting CPU execution.
    """
    return RuntimeContext(plan=ExecutionPlan(device="cpu"))


def test_fa2_fidelity_mode_uses_float64_internal_output() -> None:
    """The opt-in FA2 fidelity mode should preserve double precision output."""
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

    default_pos = layout_fa2_pipeline(edge_index=edge_index, num_nodes=3, steps=0, seed=7)
    fidelity_pos = layout_fa2_pipeline(
        edge_index=edge_index,
        num_nodes=3,
        steps=0,
        seed=7,
        fidelity_mode=True,
    )

    assert default_pos.dtype == torch.float32
    assert fidelity_pos.dtype == torch.float64
    assert torch.allclose(default_pos.to(dtype=torch.float64), fidelity_pos)


def test_fa2_strong_gravity_moves_axis_aligned_nodes() -> None:
    """Strong gravity should skip only the origin, matching live ``fa2``."""
    state = SolveState(
        pos=torch.tensor([[1.0, 0.0]], dtype=torch.float64),
        old_forces=torch.zeros((1, 2), dtype=torch.float64),
        extras={
            "fa2_undirected_edges": torch.empty((2, 0), dtype=torch.long),
            "fa2_undirected_weights": None,
            "fa2_mass": torch.ones(1, dtype=torch.float64),
            "fa2_outbound_att_compensation": 1.0,
            "fa2_speed": 1.0,
            "fa2_speed_efficiency": 1.0,
        },
    )

    result = FA2ForceStep(
        FA2ForceStepConfig(strong_gravity=True, gravity=1.0, scaling_ratio=2.0)
    ).apply(_empty_problem(num_nodes=1), state, _runtime_context())

    assert result.forces is not None
    assert result.forces[0, 0].item() < 0.0
    assert result.forces[0, 1].item() == 0.0


def test_fa2_reference_prefers_live_fa2_package() -> None:
    """The ``fa2_ref`` comparator target should be explicit and stable."""
    assert _FA2_REFERENCE_PACKAGE_ORDER == ("fa2", "fa2_modified")


def test_fa2_linlog_skips_coincident_edge_attraction() -> None:
    """LinLog attraction should skip zero-distance endpoints like live ``fa2``."""
    state = SolveState(
        pos=torch.zeros((2, 2), dtype=torch.float64),
        old_forces=torch.zeros((2, 2), dtype=torch.float64),
        extras={
            "fa2_undirected_edges": torch.tensor([[0], [1]], dtype=torch.long),
            "fa2_undirected_weights": None,
            "fa2_mass": torch.ones(2, dtype=torch.float64),
            "fa2_outbound_att_compensation": 1.0,
            "fa2_speed": 1.0,
            "fa2_speed_efficiency": 1.0,
        },
    )

    result = FA2ForceStep(FA2ForceStepConfig(linlog=True, gravity=0.0)).apply(
        _empty_problem(num_nodes=2),
        state,
        _runtime_context(),
    )

    assert result.forces is not None
    torch.testing.assert_close(result.forces, torch.zeros((2, 2), dtype=torch.float64))


def test_fa2_dissuade_hubs_aliases_outbound_distribution() -> None:
    """Avoid applying the reference Dissuade Hubs mass divisor twice."""
    state = SolveState(
        pos=torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=torch.float64),
        old_forces=torch.zeros((2, 2), dtype=torch.float64),
        extras={
            "fa2_undirected_edges": torch.tensor([[0], [1]], dtype=torch.long),
            "fa2_undirected_weights": None,
            "fa2_mass": torch.tensor([3.0, 1.0], dtype=torch.float64),
            "fa2_outbound_att_compensation": 1.0,
            "fa2_speed": 1.0,
            "fa2_speed_efficiency": 1.0,
        },
    )

    result = FA2ForceStep(
        FA2ForceStepConfig(
            gravity=0.0,
            scaling_ratio=0.0,
            outbound_attraction_distribution=True,
            dissuade_hubs=True,
        )
    ).apply(_empty_problem(num_nodes=2), state, _runtime_context())

    assert result.forces is not None
    expected = torch.tensor([[2.0 / 3.0, 0.0], [-2.0 / 3.0, 0.0]], dtype=torch.float64)
    torch.testing.assert_close(result.forces, expected)


def test_fa2_fidelity_mode_keeps_last_duplicate_edge_weight() -> None:
    """Fidelity mode should mirror NetworkX's duplicate edge overwrite policy."""
    problem = LayoutProblem(
        edge_index=torch.tensor([[0, 0, 1], [1, 1, 0]], dtype=torch.long),
        num_nodes=2,
        edge_weights=torch.tensor([2.0, 5.0, 7.0], dtype=torch.float64),
        seed=42,
    )
    state = build_fa2_pipeline(FA2Config(steps=0, fidelity_mode=True)).apply(
        problem,
        SolveState(),
        _runtime_context(),
    )

    weights = state.extras["fa2_undirected_weights"]
    assert isinstance(weights, torch.Tensor)
    torch.testing.assert_close(weights, torch.tensor([7.0], dtype=torch.float64))
