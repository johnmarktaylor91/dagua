"""Pipeline pins for the d3-hierarchy tidy tree source port."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from dagua.config import LayoutConfig
from dagua.eval.competitors.d3hierarchy_competitor import D3HierarchyCompetitor
from dagua.graph import DaguaGraph
from dagua.layout.engine import layout
from dagua.layout.ops.pipelines import PIPELINE_REGISTRY, get_pipeline_function
from dagua.layout.ops.pipelines.d3_tree import (
    build_d3_tree_pipeline,
    layout_d3_tree_pipeline,
    layout_d3_tree_radial_pipeline,
)
from dagua.layout.ops.taxonomy import get_op_class


def _edge_index(edges: Iterable[tuple[int, int]]) -> torch.Tensor:
    """Build an edge-index tensor.

    Parameters
    ----------
    edges : iterable[tuple[int, int]]
        Directed parent-child edges.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    edge_list = list(edges)
    if not edge_list:
        return torch.empty((2, 0), dtype=torch.long)
    sources, targets = zip(*edge_list)
    return torch.tensor([list(sources), list(targets)], dtype=torch.long)


def _graph_from_edges(num_nodes: int, edges: Iterable[tuple[int, int]]) -> DaguaGraph:
    """Build a ``DaguaGraph`` from integer edges.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    edges : iterable[tuple[int, int]]
        Directed parent-child edges.

    Returns
    -------
    DaguaGraph
        Graph using integer node IDs.
    """
    graph = DaguaGraph()
    for node in range(num_nodes):
        graph.add_node(node)
    for source, target in edges:
        graph.add_edge(source, target)
    return graph


def test_d3_tree_pipeline_and_op_are_registered() -> None:
    """Register d3 tree pipelines and coordinate op.

    Returns
    -------
    None
        Registry lookups must resolve the new tree entries.
    """
    assert PIPELINE_REGISTRY["d3_tree"] == (
        "dagua.layout.ops.pipelines.d3_tree",
        "layout_d3_tree_pipeline",
    )
    assert PIPELINE_REGISTRY["d3_tree_radial"] == (
        "dagua.layout.ops.pipelines.d3_tree",
        "layout_d3_tree_radial_pipeline",
    )
    assert get_pipeline_function("D3_TREE") is layout_d3_tree_pipeline
    assert get_op_class("d3_tree_layout").__name__ == "D3TreeLayout"


def test_d3_tree_pipeline_has_stage_composition() -> None:
    """Pin the d3 tree implementation as a composable op pipeline.

    Returns
    -------
    None
        The operation sequence must remain explicit.
    """
    pipeline = build_d3_tree_pipeline()
    assert [operation.name for operation in pipeline.ops] == ["d3_tree_layout"]


def test_d3_tree_binary_position_pin() -> None:
    """Pin d3 tidy tree node-size coordinates for a binary tree.

    Returns
    -------
    None
        Coordinates must match d3 ``tree().nodeSize([1, 1])`` exactly.
    """
    edge_index = _edge_index([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)])
    positions = layout_d3_tree_pipeline(edge_index, 7)
    torch.testing.assert_close(
        positions,
        torch.tensor(
            [
                [0.0, 0.0],
                [-1.5, 1.0],
                [1.5, 1.0],
                [-2.0, 2.0],
                [-1.0, 2.0],
                [1.0, 2.0],
                [2.0, 2.0],
            ],
            dtype=torch.float64,
        ),
        rtol=0.0,
        atol=0.0,
    )


def test_d3_tree_matches_reference_adapter_on_small_trees() -> None:
    """Compare the source port with d3-hierarchy on representative trees.

    Returns
    -------
    None
        All available adapter rows must be bit-exact.
    """
    competitor = D3HierarchyCompetitor()
    assert competitor.available()
    cases = [
        (1, []),
        (4, [(0, 1), (1, 2), (2, 3)]),
        (4, [(0, 1), (0, 2), (0, 3)]),
        (7, [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]),
        (6, [(0, 1), (0, 2), (1, 3), (1, 4), (4, 5)]),
    ]
    for num_nodes, edges in cases:
        graph = _graph_from_edges(num_nodes, edges)
        reference = competitor.layout_with_variant(graph, variant_params={"algorithm": "tree"})
        assert reference.error is None
        assert reference.pos is not None
        actual = layout_d3_tree_pipeline(graph.edge_index, graph.num_nodes)
        torch.testing.assert_close(actual, reference.pos, rtol=0.0, atol=0.0)


def test_layout_config_algorithm_d3_tree_works() -> None:
    """Exercise public engine dispatch for ``algorithm='d3_tree'``.

    Returns
    -------
    None
        Public layout dispatch returns finite coordinates.
    """
    graph = DaguaGraph.from_edge_list([("root", "left"), ("root", "right")])
    positions = layout(graph, LayoutConfig(algorithm="d3_tree"))
    assert positions.shape == (3, 2)
    assert torch.isfinite(positions).all()


def test_d3_tree_radial_variant_runs() -> None:
    """Exercise the named radial tree variant.

    Returns
    -------
    None
        The radial variant returns finite Cartesian coordinates.
    """
    edge_index = _edge_index([(0, 1), (0, 2), (0, 3)])
    positions = layout_d3_tree_radial_pipeline(edge_index, 4)
    assert positions.shape == (4, 2)
    assert torch.isfinite(positions).all()


def test_d3_tree_production_pipeline_has_no_runtime_delegation() -> None:
    """Guard production d3 tree code against Node/reference delegation.

    Returns
    -------
    None
        Production source must not contain subprocess/reference hooks.
    """
    root = Path(__file__).parents[1]
    source = "\n".join(
        [
            (root / "dagua" / "layout" / "ops" / "d3tree.py").read_text(),
            (root / "dagua" / "layout" / "ops" / "pipelines" / "d3_tree.py").read_text(),
        ]
    )
    assert "subprocess" not in source
    assert "D3HierarchyCompetitor" not in source
    assert "node_modules" not in source
