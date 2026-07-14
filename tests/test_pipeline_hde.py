"""Pipeline pins for Harel-Koren high-dimensional embedding."""

from __future__ import annotations

from pathlib import Path

import torch

import dagua
from dagua.config import LayoutConfig
from dagua.layout.ops.hde import hde_project_distances
from dagua.layout.ops.pipelines import PIPELINE_REGISTRY, get_pipeline_function
from dagua.layout.ops.pipelines.hde import build_hde_pipeline, layout_hde_pipeline
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


def test_hde_pipeline_and_ops_are_registered() -> None:
    """Register the public HDE algorithm and reusable PCA init op.

    Returns
    -------
    None
        Registry lookups must resolve the HDE entrypoint and op class.
    """
    assert PIPELINE_REGISTRY["hde"] == (
        "dagua.layout.ops.pipelines.hde",
        "layout_hde_pipeline",
    )
    assert get_pipeline_function("HDE") is layout_hde_pipeline
    assert get_op_class("hde_project_pivot_distances").__name__ == "HDEProjectPivotDistances"


def test_hde_pipeline_has_stage_composition() -> None:
    """Pin HDE as explicit composable stages.

    Returns
    -------
    None
        The operation sequence must remain visible and reusable.
    """
    pipeline = build_hde_pipeline(n_pivots=3)
    assert [operation.name for operation in pipeline.ops] == [
        "build_adjacency",
        "pivot_selection",
        "pivot_distance_queries",
        "hde_project_pivot_distances",
    ]


def test_hde_pivots_distances_and_projection_are_deterministic() -> None:
    """Pin farthest-first pivots and the PCA projection stage.

    Returns
    -------
    None
        Stage snapshots and final coordinates must match the reference helper.
    """
    edge_index = _edge_index([(0, 1), (1, 2), (2, 3), (1, 4), (4, 5)])
    problem = LayoutProblem(edge_index=edge_index, num_nodes=6)
    final_state = build_hde_pipeline(n_pivots=3).apply(
        problem,
        SolveState(),
        RuntimeContext(plan=ExecutionPlan(device="cpu")),
    )

    torch.testing.assert_close(
        final_state.pivot_indices,
        torch.tensor([0, 3, 5], dtype=torch.long),
        rtol=0.0,
        atol=0.0,
    )
    assert final_state.pivot_distances is not None
    torch.testing.assert_close(
        final_state.pivot_distances,
        torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0, 2.0, 3.0],
                [3.0, 2.0, 1.0, 0.0, 3.0, 4.0],
                [3.0, 2.0, 3.0, 4.0, 1.0, 0.0],
            ],
            dtype=torch.float64,
        ),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        final_state.pos,
        hde_project_distances(final_state.pivot_distances),
        rtol=0.0,
        atol=1.0e-12,
    )
    assert final_state.extras["hde"]["reusable_init_op"] == "hde_project_pivot_distances"


def test_layout_config_algorithm_hde_dispatches() -> None:
    """Exercise public engine dispatch for ``algorithm='hde'``.

    Returns
    -------
    None
        Dispatch must return finite coordinates.
    """
    graph = dagua.DaguaGraph.from_edge_list([("a", "b"), ("b", "c"), ("b", "d")])
    positions = dagua.layout(graph, LayoutConfig(algorithm="hde", algorithm_params={"n_pivots": 3}))

    assert positions.shape == (4, 2)
    assert torch.isfinite(positions).all()


def test_hde_production_pipeline_has_no_runtime_delegation() -> None:
    """Guard the implementation against reference-engine delegation.

    Returns
    -------
    None
        Production source must not contain subprocess/reference adapter hooks.
    """
    source_paths = [
        Path(__file__).parents[1] / "dagua" / "layout" / "ops" / "hde.py",
        Path(__file__).parents[1] / "dagua" / "layout" / "ops" / "pipelines" / "hde.py",
    ]
    source = "\n".join(path.read_text() for path in source_paths)
    assert "subprocess" not in source
    assert "Graphlayouts" not in source
    assert "rpy2" not in source
