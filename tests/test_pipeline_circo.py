"""Pipeline pins for the Graphviz circo-style circular layout."""

from __future__ import annotations

from pathlib import Path

import torch

import dagua
from dagua.config import LayoutConfig
from dagua.layout.ops.graphviz_radial_circular import biconnected_components
from dagua.layout.ops.pipelines import PIPELINE_REGISTRY, get_pipeline_function
from dagua.layout.ops.pipelines.circo import build_circo_pipeline, layout_circo_pipeline
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import get_op_class


def _cycle_edges() -> torch.Tensor:
    """Return the fixed four-cycle topology.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, 4]``.
    """
    return torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)


def test_circo_pipeline_and_ops_are_registered() -> None:
    """Register the public circo algorithm and circular op.

    Returns
    -------
    None
        Registry lookups must resolve the circo entrypoint and op class.
    """
    assert PIPELINE_REGISTRY["circo"] == (
        "dagua.layout.ops.pipelines.circo",
        "layout_circo_pipeline",
    )
    assert get_pipeline_function("CIRCO") is layout_circo_pipeline
    assert get_op_class("circo_assign_circular_coordinates").__name__ == (
        "CircoAssignCircularCoordinates"
    )


def test_circo_pipeline_has_stage_composition() -> None:
    """Pin the circo algorithm as an explicit operation.

    Returns
    -------
    None
        The pipeline operation sequence must remain explicit.
    """
    pipeline = build_circo_pipeline()
    assert [operation.name for operation in pipeline.ops] == [
        "circo_assign_circular_coordinates",
    ]


def test_circo_cycle_block_and_position_pins() -> None:
    """Pin biconnected block membership and cycle coordinates.

    Returns
    -------
    None
        Stage snapshots and coordinates must stay deterministic.
    """
    edge_index = _cycle_edges()
    assert biconnected_components(edge_index, 4) == [[0, 1, 2, 3]]

    problem = LayoutProblem(edge_index=edge_index, num_nodes=4)
    final_state = build_circo_pipeline(nodesep=18.0).apply(
        problem,
        SolveState(),
        RuntimeContext(plan=ExecutionPlan(device="cpu")),
    )

    assert final_state.extras["circo"]["blocks"] == [[0, 1, 2, 3]]
    torch.testing.assert_close(
        final_state.pos,
        torch.tensor(
            [
                [18.0, 0.0],
                [0.0, 18.0],
                [-18.0, 0.0],
                [0.0, -18.0],
            ],
            dtype=torch.float64,
        ),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_layout_config_algorithm_circo_dispatches() -> None:
    """Exercise public engine dispatch for ``algorithm='circo'``.

    Returns
    -------
    None
        Dispatch must return finite coordinates.
    """
    graph = dagua.DaguaGraph.from_edge_list([("a", "b"), ("b", "c"), ("c", "a")])
    positions = dagua.layout(graph, LayoutConfig(algorithm="circo"))

    assert positions.shape == (3, 2)
    assert torch.isfinite(positions).all()


def test_circo_production_pipeline_has_no_runtime_delegation() -> None:
    """Guard the implementation against Graphviz subprocess delegation.

    Returns
    -------
    None
        Production source must not contain subprocess/reference adapter hooks.
    """
    source_paths = [
        Path(__file__).parents[1] / "dagua" / "layout" / "ops" / "graphviz_radial_circular.py",
        Path(__file__).parents[1] / "dagua" / "layout" / "ops" / "pipelines" / "circo.py",
    ]
    source = "\n".join(path.read_text() for path in source_paths)
    assert "subprocess" not in source
    assert "GraphvizCirco" not in source
    assert "_layout_with_graphviz_engine" not in source
