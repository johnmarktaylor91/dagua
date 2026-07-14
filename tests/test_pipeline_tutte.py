"""Pipeline pins for Tutte barycentric embedding."""

from __future__ import annotations

from pathlib import Path

import torch

import dagua
from dagua.config import LayoutConfig
from dagua.layout.ops.pipelines import PIPELINE_REGISTRY, get_pipeline_function
from dagua.layout.ops.pipelines.tutte import build_tutte_pipeline, layout_tutte_pipeline
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import get_op_class


def _edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    """Build a ``[2, E]`` edge tensor.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Source-target edge pairs.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def test_tutte_pipeline_and_ops_are_registered() -> None:
    """Register the public Tutte algorithm and barycentric op.

    Returns
    -------
    None
        Registry lookups must resolve the Tutte entrypoint and op class.
    """
    assert PIPELINE_REGISTRY["tutte"] == (
        "dagua.layout.ops.pipelines.tutte",
        "layout_tutte_pipeline",
    )
    assert get_pipeline_function("TUTTE") is layout_tutte_pipeline
    assert get_op_class("tutte_barycentric_embedding").__name__ == "TutteBarycentricEmbedding"


def test_tutte_pipeline_has_stage_composition() -> None:
    """Pin Tutte as one explicit linear-solve stage.

    Returns
    -------
    None
        The operation sequence must remain visible.
    """
    pipeline = build_tutte_pipeline()
    assert [operation.name for operation in pipeline.ops] == ["tutte_barycentric_embedding"]


def test_tutte_wheel_boundary_and_center_barycenter_are_deterministic() -> None:
    """Pin boundary selection and interior linear solve on a wheel graph.

    Returns
    -------
    None
        The rim is fixed to the convex polygon and the hub solves to its
        neighbor barycenter.
    """
    rim_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    spokes = [(4, node) for node in range(4)]
    edge_index = _edge_index([*rim_edges, *spokes])
    problem = LayoutProblem(edge_index=edge_index, num_nodes=5)
    final_state = build_tutte_pipeline(radius=2.0).apply(
        problem,
        SolveState(),
        RuntimeContext(plan=ExecutionPlan(device="cpu")),
    )

    assert final_state.extras["tutte"]["boundary"] == [0, 1, 2, 3]
    assert final_state.extras["tutte"]["interior"] == [4]
    assert final_state.extras["tutte"]["fallback"] is None
    expected_boundary = torch.tensor(
        [[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0], [0.0, -2.0]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(final_state.pos[:4], expected_boundary, rtol=0.0, atol=1.0e-12)
    torch.testing.assert_close(
        final_state.pos[4],
        final_state.pos[:4].mean(dim=0),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_tutte_tree_uses_documented_convex_fallback() -> None:
    """Trees lack a peripheral cycle and should use the documented fallback.

    Returns
    -------
    None
        The fallback keeps public dispatch finite while naming the N/A reason.
    """
    edge_index = _edge_index([(0, 1), (1, 2)])
    problem = LayoutProblem(edge_index=edge_index, num_nodes=3)
    final_state = build_tutte_pipeline().apply(
        problem,
        SolveState(),
        RuntimeContext(plan=ExecutionPlan(device="cpu")),
    )

    assert final_state.extras["tutte"]["boundary"] == [0, 1, 2]
    assert final_state.extras["tutte"]["fallback"] == (
        "no peripheral cycle; all nodes fixed on convex polygon"
    )
    assert torch.isfinite(final_state.pos).all()


def test_layout_config_algorithm_tutte_dispatches() -> None:
    """Exercise public engine dispatch for ``algorithm='tutte'``.

    Returns
    -------
    None
        Dispatch must return finite coordinates.
    """
    graph = dagua.DaguaGraph.from_edge_list([("a", "b"), ("b", "c"), ("c", "a")])
    positions = dagua.layout(graph, LayoutConfig(algorithm="tutte"))

    assert positions.shape == (3, 2)
    assert torch.isfinite(positions).all()


def test_tutte_production_pipeline_has_no_runtime_delegation() -> None:
    """Guard the implementation against OGDF subprocess delegation.

    Returns
    -------
    None
        Production source must not contain subprocess/reference adapter hooks.
    """
    source_paths = [
        Path(__file__).parents[1] / "dagua" / "layout" / "ops" / "tutte.py",
        Path(__file__).parents[1] / "dagua" / "layout" / "ops" / "pipelines" / "tutte.py",
    ]
    source = "\n".join(path.read_text() for path in source_paths)
    assert "subprocess" not in source
    assert "OGDF" not in source
    assert "ogdf" not in source
