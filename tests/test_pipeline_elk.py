"""Pipeline pins and reference checks for the ELK Layered-style pipeline."""

from __future__ import annotations

from pathlib import Path

import torch

from dagua.config import LayoutConfig
from dagua.graph import DaguaGraph
from dagua.layout.engine import layout
from dagua.layout.ops.pipelines import PIPELINE_REGISTRY, get_pipeline_function
from dagua.layout.ops.pipelines.elk import (
    build_elk_pipeline,
    layout_elk_layered_bk_pipeline,
    layout_elk_layered_ns_pipeline,
    layout_elk_lp_pipeline,
    layout_elk_pipeline,
)
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import get_op_class


def _diamond_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """Return the fixed diamond topology and box sizes.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Edge index ``[2, 4]`` and node sizes ``[4, 2]``.
    """
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]], dtype=torch.long)
    node_sizes = torch.tensor([[64.1085968, 34.0]] * 4, dtype=torch.float64)
    return edge_index, node_sizes


def test_elk_pipeline_and_ops_are_registered() -> None:
    """Register ELK algorithms and composable ops.

    Returns
    -------
    None
        Registry lookups must resolve the ELK entrypoints and op classes.
    """
    assert PIPELINE_REGISTRY["elk"] == ("dagua.layout.ops.pipelines.elk", "layout_elk_pipeline")
    assert PIPELINE_REGISTRY["elk_lp"] == (
        "dagua.layout.ops.pipelines.elk",
        "layout_elk_lp_pipeline",
    )
    assert get_pipeline_function("ELK") is layout_elk_pipeline
    assert get_pipeline_function("elk_layered_ns") is layout_elk_layered_ns_pipeline
    assert get_pipeline_function("elk_layered_bk") is layout_elk_layered_bk_pipeline
    assert get_pipeline_function("elk_lp") is layout_elk_lp_pipeline
    assert get_op_class("elk_prepare_graph").__name__ == "ElkPrepareGraph"
    assert get_op_class("elk_place_nodes").__name__ == "ElkPlaceNodes"


def test_elk_pipeline_has_stage_composition() -> None:
    """Pin the ELK pipeline as explicit composable operations.

    Returns
    -------
    None
        Operation sequence must remain phase-structured.
    """
    pipeline = build_elk_pipeline()
    assert [operation.name for operation in pipeline.ops] == [
        "elk_prepare_graph",
        "elk_break_cycles",
        "elk_assign_layers",
        "elk_minimize_crossings",
        "elk_place_nodes",
    ]


def test_elk_diamond_stage_and_position_pins() -> None:
    """Pin layers, order, and top-left positions on a diamond.

    Returns
    -------
    None
        Stage metadata and coordinates must stay deterministic.
    """
    edge_index, node_sizes = _diamond_inputs()
    final_state = build_elk_pipeline().apply(
        LayoutProblem(edge_index=edge_index, num_nodes=4, node_sizes=node_sizes),
        SolveState(),
        RuntimeContext(plan=ExecutionPlan(device="cpu")),
    )

    assert final_state.extras["elk_layers"] == [[0], [1, 2], [3]]
    assert final_state.extras["elk_order"] == {0: 0, 1: 0, 2: 1, 3: 0}
    torch.testing.assert_close(
        final_state.pos,
        torch.tensor(
            [
                [22.684766133333333, 12.0],
                [12.0, 106.0],
                [116.1085968, 106.0],
                [22.684766133333333, 200.0],
            ],
            dtype=torch.float64,
        ),
        rtol=0.0,
        atol=1.0e-7,
    )


def test_elk_variant_position_pins() -> None:
    """Pin direction, spacing, and named variant outputs.

    Returns
    -------
    None
        Public variant options must produce stable coordinates.
    """
    edge_index, node_sizes = _diamond_inputs()
    expected = {
        "UP": [
            [22.684766133333333, 200.0],
            [12.0, 106.0],
            [116.1085968, 106.0],
            [22.684766133333333, 12.0],
        ],
        "RIGHT": [
            [12.0, 22.684766133333333],
            [106.0, 12.0],
            [106.0, 116.1085968],
            [200.0, 22.684766133333333],
        ],
        "spacing": [
            [22.684766133333333, 12.0],
            [12.0, 126.0],
            [76.1085968, 126.0],
            [22.684766133333333, 240.0],
        ],
        "lp": [[12.0, 12.0], [12.0, 106.0], [116.1085968, 106.0], [12.0, 200.0]],
    }
    outputs = {
        "UP": layout_elk_pipeline(edge_index, 4, node_sizes, direction="UP"),
        "RIGHT": layout_elk_pipeline(edge_index, 4, node_sizes, direction="RIGHT"),
        "spacing": layout_elk_pipeline(
            edge_index,
            4,
            node_sizes,
            node_node_spacing=0.0,
            between_layers_spacing=80.0,
        ),
        "lp": layout_elk_lp_pipeline(edge_index, 4, node_sizes),
    }
    for name, positions in outputs.items():
        torch.testing.assert_close(
            positions,
            torch.tensor(expected[name], dtype=torch.float64),
            rtol=0.0,
            atol=1.0e-7,
        )


def test_layout_config_algorithm_elk_dispatches() -> None:
    """Exercise public engine dispatch for ``LayoutConfig(algorithm='elk')``.

    Returns
    -------
    None
        The engine must return one position per graph node.
    """
    graph = DaguaGraph.from_edge_list([("root", "left"), ("root", "right")])
    positions = layout(graph, LayoutConfig(algorithm="elk"))

    assert positions.shape == (3, 2)
    assert torch.isfinite(positions).all()


def test_elk_production_pipeline_has_no_runtime_delegation() -> None:
    """Guard production ELK source against Node or competitor delegation.

    Returns
    -------
    None
        Production source must not contain subprocess/reference adapter hooks.
    """
    source_paths = [
        Path(__file__).parents[1] / "dagua" / "layout" / "ops" / "elk.py",
        Path(__file__).parents[1] / "dagua" / "layout" / "ops" / "pipelines" / "elk.py",
    ]
    source = "\n".join(path.read_text() for path in source_paths)
    assert "subprocess" not in source
    assert "ElkLayered" not in source
    assert "node_modules" not in source
