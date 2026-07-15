"""Pipeline tests for the WebCola reimplementation."""

from __future__ import annotations

import inspect
import math

import pytest
import torch

import dagua
from dagua.config import LayoutConfig
from dagua.eval.competitors.webcola_competitor import WebColaCompetitor
from dagua.eval.equivalence_metrics import procrustes_rmsd
from dagua.layout.ops import OP_REGISTRY
from dagua.layout.ops.pipelines import PIPELINE_REGISTRY, get_pipeline_function
from dagua.layout.ops.pipelines.webcola import (
    layout_webcola_constrained_pipeline,
    layout_webcola_pipeline,
)


def _graph(num_nodes: int, edges: list[tuple[int, int]]) -> dagua.DaguaGraph:
    """Build a small indexed Dagua graph.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    edges : list[tuple[int, int]]
        Directed edges.

    Returns
    -------
    dagua.DaguaGraph
        Graph with string IDs matching integer indices.
    """
    graph = dagua.DaguaGraph()
    for index in range(num_nodes):
        graph.add_node(str(index))
    for source, target in edges:
        graph.add_edge(str(source), str(target))
    return graph


def test_webcola_pipeline_is_registered() -> None:
    """Public WebCola algorithm names should resolve to pipeline functions."""
    assert PIPELINE_REGISTRY["webcola"] == (
        "dagua.layout.ops.pipelines.webcola",
        "layout_webcola_pipeline",
    )
    assert PIPELINE_REGISTRY["webcola_constrained"] == (
        "dagua.layout.ops.pipelines.webcola",
        "layout_webcola_constrained_pipeline",
    )
    assert get_pipeline_function("WEBCOLA") is layout_webcola_pipeline
    assert OP_REGISTRY["webcola_run_descent"].__name__ == "RunWebColaDescent"


@pytest.mark.parametrize(
    ("num_nodes", "edges"),
    [
        (1, []),
        (4, [(0, 1), (1, 2), (2, 3)]),
        (4, [(0, 1), (0, 2), (1, 3), (2, 3)]),
        (4, [(0, 1), (1, 2), (2, 3), (3, 0)]),
    ],
)
def test_webcola_unconstrained_matches_node_reference(
    num_nodes: int,
    edges: list[tuple[int, int]],
) -> None:
    """Native WebCola stress should match the Node reference on small graphs."""
    competitor = WebColaCompetitor()
    if not competitor.available():
        pytest.skip("webcola npm package is not available")
    graph = _graph(num_nodes, edges)
    reference = competitor.layout_with_variant(graph, variant_params={"steps": 20})
    assert reference.error is None
    assert reference.pos is not None

    actual = layout_webcola_pipeline(graph.edge_index, graph.num_nodes, steps=20)

    assert torch.max(torch.abs(actual - reference.pos)).item() < 1.0e-12
    assert procrustes_rmsd(actual.numpy(), reference.pos.numpy()) < 1.0e-12


def test_webcola_constrained_matches_node_reference_for_separation() -> None:
    """Native constrained WebCola should match WebCola VPSC projection."""
    competitor = WebColaCompetitor()
    if not competitor.available():
        pytest.skip("webcola npm package is not available")
    graph = _graph(3, [(0, 1), (1, 2)])
    constraints = [{"axis": "x", "left": 0, "right": 2, "gap": 60.0}]
    reference = competitor.layout_with_variant(
        graph,
        variant_params={"steps": 15, "constrained": True, "constraints": constraints},
    )
    assert reference.error is None
    assert reference.pos is not None

    actual = layout_webcola_constrained_pipeline(
        graph.edge_index,
        graph.num_nodes,
        steps=15,
        constraints=constraints,
    )

    assert torch.max(torch.abs(actual - reference.pos)).item() < 1.0e-9
    assert actual[2, 0] - actual[0, 0] >= 60.0 - 1.0e-9


def test_layout_config_algorithm_webcola_dispatches() -> None:
    """Exercise public engine dispatch for ``algorithm='webcola'``."""
    graph = _graph(4, [(0, 1), (1, 2), (2, 3)])
    positions = dagua.layout(graph, LayoutConfig(algorithm="webcola", steps=5))

    assert positions.shape == (4, 2)
    assert torch.isfinite(positions).all()


def test_webcola_pipeline_does_not_delegate_to_node_subprocess() -> None:
    """The native WebCola pipeline must not call the Node reference adapter."""
    source = inspect.getsource(layout_webcola_pipeline)
    module_source = inspect.getsource(inspect.getmodule(layout_webcola_pipeline))

    assert "subprocess" not in source
    assert "WebColaCompetitor" not in module_source


def test_webcola_constrained_separation_is_feasible_without_edges() -> None:
    """Constrained variant should project even edgeless layouts."""
    edge_index = torch.empty((2, 0), dtype=torch.long)
    positions = layout_webcola_constrained_pipeline(
        edge_index,
        2,
        steps=1,
        constraints=[{"axis": "x", "left": 0, "right": 1, "gap": 25.0}],
    )

    assert math.isfinite(float(positions.sum()))
    assert positions[1, 0] - positions[0, 0] >= 25.0 - 1.0e-9
