"""Pipeline pins for the Graphviz twopi-style radial layout."""

from __future__ import annotations

from pathlib import Path

import torch

import dagua
from dagua.config import LayoutConfig
from dagua.layout.ops.graphviz_radial_circular import choose_twopi_root, twopi_ring_levels
from dagua.layout.ops.pipelines import PIPELINE_REGISTRY, get_pipeline_function
from dagua.layout.ops.pipelines.twopi import build_twopi_pipeline, layout_twopi_pipeline
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import get_op_class


def _diamond_edges() -> torch.Tensor:
    """Return the fixed diamond topology.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, 4]``.
    """
    return torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]], dtype=torch.long)


def test_twopi_pipeline_and_ops_are_registered() -> None:
    """Register the public twopi algorithm and radial op.

    Returns
    -------
    None
        Registry lookups must resolve the twopi entrypoint and op class.
    """
    assert PIPELINE_REGISTRY["twopi"] == (
        "dagua.layout.ops.pipelines.twopi",
        "layout_twopi_pipeline",
    )
    assert get_pipeline_function("TWOPI") is layout_twopi_pipeline
    assert get_op_class("twopi_assign_radial_coordinates").__name__ == (
        "TwopiAssignRadialCoordinates"
    )


def test_twopi_pipeline_has_stage_composition() -> None:
    """Pin the twopi algorithm as an explicit operation.

    Returns
    -------
    None
        The pipeline operation sequence must remain explicit.
    """
    pipeline = build_twopi_pipeline()
    assert [operation.name for operation in pipeline.ops] == [
        "twopi_assign_radial_coordinates",
    ]


def test_twopi_diamond_stage_and_position_pins() -> None:
    """Pin root selection, BFS rings, and diamond coordinates.

    Returns
    -------
    None
        Stage snapshots and coordinates must stay deterministic.
    """
    edge_index = _diamond_edges()
    assert choose_twopi_root(edge_index, 4) == 0
    assert twopi_ring_levels(edge_index, 4) == [0, 1, 1, 2]

    problem = LayoutProblem(edge_index=edge_index, num_nodes=4)
    final_state = build_twopi_pipeline(ranksep=72.0).apply(
        problem,
        SolveState(),
        RuntimeContext(plan=ExecutionPlan(device="cpu")),
    )

    assert final_state.extras["twopi"]["root"] == 0
    assert final_state.extras["twopi"]["levels"] == [0, 1, 1, 2]
    torch.testing.assert_close(
        final_state.pos,
        torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 72.0],
                [0.0, -72.0],
                [0.0, 144.0],
            ],
            dtype=torch.float64,
        ),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_layout_config_algorithm_twopi_dispatches() -> None:
    """Exercise public engine dispatch for ``algorithm='twopi'``.

    Returns
    -------
    None
        Dispatch must return finite coordinates.
    """
    graph = dagua.DaguaGraph.from_edge_list([("root", "left"), ("root", "right")])
    positions = dagua.layout(graph, LayoutConfig(algorithm="twopi"))

    assert positions.shape == (3, 2)
    assert torch.isfinite(positions).all()


def test_twopi_production_pipeline_has_no_runtime_delegation() -> None:
    """Guard the implementation against Graphviz subprocess delegation.

    Returns
    -------
    None
        Production source must not contain subprocess/reference adapter hooks.
    """
    source_paths = [
        Path(__file__).parents[1] / "dagua" / "layout" / "ops" / "graphviz_radial_circular.py",
        Path(__file__).parents[1] / "dagua" / "layout" / "ops" / "pipelines" / "twopi.py",
    ]
    source = "\n".join(path.read_text() for path in source_paths)
    assert "subprocess" not in source
    assert "GraphvizTwopi" not in source
    assert "_layout_with_graphviz_engine" not in source
