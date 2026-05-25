"""Regression tests for Graphviz SFDP sequential update fidelity."""

from __future__ import annotations

import torch

from dagua.layout.ops.pipelines.sfdp import (
    _sfdp_force_scales,
    _SFDPGraphvizSequentialStep,
    build_sfdp_pipeline,
    layout_sfdp_pipeline,
)
from dagua.layout.ops.sfdp import _SFDP_CURRENT_STEP_KEY, _SFDP_FORCE_NORM_KEY, GraphData
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


def _reference_path_graph() -> GraphData:
    """Build a three-node path in the SFDP graph representation.

    Returns
    -------
    GraphData
        Undirected path graph ``0 -- 1 -- 2`` with unit edge weights.
    """
    return GraphData(
        num_nodes=3,
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_weight=torch.ones((2,), dtype=torch.float32),
        adjacency=[[(1, 1.0)], [(0, 1.0), (2, 1.0)], [(1, 1.0)]],
    )


def test_graphviz_sequential_step_matches_c_update_order_golden_vector() -> None:
    """One fidelity step should match the Graphviz C sequential update order."""
    graph = _reference_path_graph()
    attractive_scale, repulsive_scale = _sfdp_force_scales(
        ideal_length=1.0,
        repulsive_exponent=-1.0,
    )
    state = SolveState(
        pos=torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            dtype=torch.float64,
        )
    )
    state.extras[_SFDP_CURRENT_STEP_KEY] = 0.1
    problem = LayoutProblem(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=3)
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))

    updated = _SFDPGraphvizSequentialStep(
        graph=graph,
        attractive_scale=attractive_scale,
        repulsive_scale=repulsive_scale,
        repulsive_exponent=-1.0,
    ).apply(problem, state, ctx)

    expected = torch.tensor(
        [
            [-0.09333456062030594, -0.03589790793088690],
            [1.06550978295851317, -0.07555440646797849],
            [1.03567930516981765, 1.09341834500032120],
        ],
        dtype=torch.float64,
    )
    assert updated.pos is not None
    assert torch.allclose(updated.pos, expected, rtol=0.0, atol=1.0e-12)
    assert abs(float(updated.extras[_SFDP_FORCE_NORM_KEY]) - 3.6546497740984174) < 1.0e-12


def test_sfdp_graphviz_fidelity_selects_sequential_refinement_ops() -> None:
    """The explicit fidelity selector should swap in sequential refinement ops."""
    default_pipeline = build_sfdp_pipeline(steps=1)
    fidelity_pipeline = build_sfdp_pipeline(steps=1, fidelity_mode="graphviz")

    default_names = [op.name for op in default_pipeline.ops]
    fidelity_names = [op.name for op in fidelity_pipeline.ops]

    assert "sfdp_refine_coarsest" in default_names
    assert "sfdp_graphviz_refine_coarsest" in fidelity_names
    assert "sfdp_graphviz_prolongate_and_refine" in fidelity_names


def test_layout_sfdp_pipeline_accepts_graphviz_fidelity_alias() -> None:
    """The public layout wrapper should run end-to-end with the string alias."""
    edge_index = torch.tensor([[0, 1, 1], [1, 2, 3]], dtype=torch.long)

    positions = layout_sfdp_pipeline(
        edge_index=edge_index,
        num_nodes=4,
        steps=2,
        seed=123,
        fidelity_mode="graphviz",
    )

    assert positions.shape == (4, 2)
    assert positions.dtype == torch.float32
    assert torch.isfinite(positions).all()
