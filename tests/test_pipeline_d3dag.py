"""Pipeline pins for the d3-dag Sugiyama source port."""

from __future__ import annotations

from pathlib import Path

import torch

from dagua.config import LayoutConfig
from dagua.graph import DaguaGraph
from dagua.layout.engine import layout
from dagua.layout.ops.pipelines import PIPELINE_REGISTRY, get_pipeline_function
from dagua.layout.ops.pipelines.d3dag import build_d3dag_pipeline, layout_d3dag_pipeline
from dagua.layout.ops.taxonomy import get_op_class


def _diamond_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """Return a fixed diamond topology and box sizes.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Edge index ``[2, 4]`` and node sizes ``[4, 2]``.
    """
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]], dtype=torch.long)
    node_sizes = torch.tensor([[40.0, 20.0]] * 4)
    return edge_index, node_sizes


def test_d3dag_pipeline_and_ops_are_registered() -> None:
    """Register the public d3-dag algorithms and reusable ops.

    Returns
    -------
    None
        Registry lookups must resolve d3-dag pipeline and op classes.
    """
    assert PIPELINE_REGISTRY["d3dag"] == (
        "dagua.layout.ops.pipelines.d3dag",
        "layout_d3dag_pipeline",
    )
    assert get_pipeline_function("D3DAG") is layout_d3dag_pipeline
    assert get_op_class("d3dag_coffman_graham_layering").__name__ == ("D3DagCoffmanGrahamLayering")
    assert get_op_class("d3dag_optimal_crossing_order").__name__ == "D3DagOptimalCrossingOrder"


def test_d3dag_pipeline_has_stage_composition() -> None:
    """Pin the implementation as composable operations.

    Returns
    -------
    None
        The operation sequence must remain explicit.
    """
    pipeline = build_d3dag_pipeline()
    assert [operation.name for operation in pipeline.ops] == [
        "d3dag_prepare",
        "d3dag_layering",
        "d3dag_sugify",
        "d3dag_decross",
        "d3dag_coordinate",
    ]


def test_d3dag_diamond_position_pin() -> None:
    """Pin the default d3-dag diamond shape.

    Returns
    -------
    None
        Coordinates match d3-dag's geometry up to the known sibling-order
        residual from ``graphConnect`` node iteration.
    """
    edge_index, node_sizes = _diamond_inputs()
    positions = layout_d3dag_pipeline(edge_index, 4, node_sizes)
    torch.testing.assert_close(
        positions,
        torch.tensor(
            [[40.5, 10.0], [20.0, 31.0], [61.0, 31.0], [40.5, 52.0]],
            dtype=torch.float64,
        ),
        rtol=0.0,
        atol=0.0,
    )


def test_d3dag_variants_run() -> None:
    """Exercise named deterministic variants.

    Returns
    -------
    None
        Longest-path, optimal decross, and greedy coordinate variants produce
        finite ``[N, 2]`` coordinates.
    """
    edge_index, node_sizes = _diamond_inputs()
    for kwargs in (
        {"layering": "longestPath"},
        {"decross": "opt"},
        {"coord": "greedy"},
    ):
        positions = layout_d3dag_pipeline(edge_index, 4, node_sizes, **kwargs)
        assert positions.shape == (4, 2)
        assert torch.isfinite(positions).all()


def test_layout_config_algorithm_d3dag_works() -> None:
    """Exercise public engine dispatch for ``algorithm='d3dag'``.

    Returns
    -------
    None
        Public layout dispatch returns one coordinate per graph node.
    """
    graph = DaguaGraph.from_edge_list([("root", "left"), ("root", "right")])
    positions = layout(graph, LayoutConfig(algorithm="d3dag"))
    assert positions.shape == (3, 2)
    assert torch.isfinite(positions).all()


def test_d3dag_production_pipeline_has_no_runtime_delegation() -> None:
    """Guard the source port against Node/reference delegation.

    Returns
    -------
    None
        Production source must not contain subprocess/reference adapter hooks.
    """
    root = Path(__file__).parents[1]
    source = "\n".join(
        [
            (root / "dagua" / "layout" / "ops" / "d3dag.py").read_text(),
            (root / "dagua" / "layout" / "ops" / "pipelines" / "d3dag.py").read_text(),
        ]
    )
    assert "subprocess" not in source
    assert "D3DagCompetitor" not in source
    assert "node_modules" not in source
