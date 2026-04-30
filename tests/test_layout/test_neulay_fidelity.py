"""Regression tests for NeuLay reference-fidelity configuration."""

from __future__ import annotations

import torch

from dagua.layout.ops.base import Repeat
from dagua.layout.ops.neulay import NeuLayPrepareState, NeuLayPrepareStateConfig
from dagua.layout.ops.pipelines.neulay import build_neulay_pipeline, layout_neulay_pipeline
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


def _small_problem() -> LayoutProblem:
    """Create a tiny path-graph layout problem.

    Returns
    -------
    LayoutProblem
        Three-node path graph with a deterministic seed.
    """
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    return LayoutProblem(edge_index=edge_index, num_nodes=3, seed=42)


def _runtime_context() -> RuntimeContext:
    """Create a CPU runtime context for op-level tests.

    Returns
    -------
    RuntimeContext
        Runtime context targeting CPU execution.
    """
    return RuntimeContext(plan=ExecutionPlan(device="cpu"))


def test_old_code_fidelity_resolves_reference_defaults() -> None:
    """Old-code mode should expose the checked-in NeuLay-2.py defaults."""
    state = NeuLayPrepareState(
        NeuLayPrepareStateConfig(total_steps=20_000, fidelity_mode="old_code")
    ).apply(_small_problem(), SolveState(), _runtime_context())

    assert state.extras["neulay_dim"] == 3
    assert state.extras["neulay_gcn_steps"] == 40_000
    assert state.extras["neulay_linear_steps"] == 1_000_000
    assert state.extras["neulay_query_radius"] == 4.0


def test_absolute_query_radius_overrides_scaled_radius() -> None:
    """Explicit absolute query radius should bypass radius-factor scaling."""
    state = NeuLayPrepareState(
        NeuLayPrepareStateConfig(
            radius=0.2,
            query_radius_factor=10.0,
            query_radius=2.5,
            gcn_steps=0,
            total_steps=5,
        )
    ).apply(_small_problem(), SolveState(), _runtime_context())

    assert state.extras["neulay_query_radius"] == 2.5


def test_fdl_steps_are_separate_from_total_steps() -> None:
    """The pipeline should use explicit FDL steps as a post-GCN budget."""
    pipeline = build_neulay_pipeline(steps=7, gcn_steps=3, fdl_steps=5)
    direct_loop = pipeline.ops[5]

    assert isinstance(direct_loop, Repeat)
    assert direct_loop.n == 5


def test_old_code_fidelity_defaults_to_dim_3_output() -> None:
    """Old-code mode should produce 3D coordinates when dim is omitted."""
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

    positions = layout_neulay_pipeline(
        edge_index=edge_index,
        num_nodes=3,
        steps=0,
        gcn_steps=0,
        fdl_steps=0,
        fidelity_mode="old_code",
    )

    assert positions.shape == (3, 3)
