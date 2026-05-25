"""Tests for the Cytoscape fcose and Gephi YifanHu competitor adapters."""

from __future__ import annotations

import pytest
import torch

from dagua.eval.competitors import get_competitors
from dagua.eval.competitors.cytoscape_fcose_competitor import CytoscapeFcose
from dagua.eval.competitors.gephi_competitor import GephiYifanHu
from dagua.graph import DaguaGraph

pytestmark = pytest.mark.smoke

FCOSE_AVAILABLE = CytoscapeFcose().available()
GEPHI_AVAILABLE = GephiYifanHu().available()


def _small_test_graph() -> DaguaGraph:
    """Return a small evaluation graph used by the external adapter tests.

    Returns
    -------
    DaguaGraph
        Six-node path graph with stable node IDs.
    """
    graph = DaguaGraph()
    for node_idx in range(6):
        graph.add_node(str(node_idx), label=str(node_idx))
    for node_idx in range(5):
        graph.add_edge(str(node_idx), str(node_idx + 1))
    return graph


def test_cytoscape_and_gephi_registered() -> None:
    """The new competitor adapters should register on import.

    Returns
    -------
    None
        This test asserts on the registry contents.
    """
    names = {competitor.name for competitor in get_competitors()}
    assert {"cytoscape_fcose", "gephi_yifanhu"} <= names


def test_cytoscape_fcose_available_returns_bool() -> None:
    """The fcose availability probe should return a boolean.

    Returns
    -------
    None
        This test asserts on the availability probe result type.
    """
    assert isinstance(CytoscapeFcose().available(), bool)


def test_gephi_yifanhu_available_returns_bool() -> None:
    """The Gephi availability probe should return a boolean.

    Returns
    -------
    None
        This test asserts on the availability probe result type.
    """
    assert isinstance(GephiYifanHu().available(), bool)


def test_cytoscape_fcose_variant_params() -> None:
    """The fcose adapter should expose the requested variant parameters.

    Returns
    -------
    None
        This test asserts on the adapter's supported parameter names.
    """
    assert CytoscapeFcose.variant_param_names == frozenset(
        {"quality", "nodeSeparation", "idealEdgeLength", "nodeRepulsion"}
    )


@pytest.mark.skipif(not FCOSE_AVAILABLE, reason="cytoscape-fcose not installed")
def test_cytoscape_fcose_layout() -> None:
    """The fcose adapter should return positions for a small graph.

    Returns
    -------
    None
        This test asserts on the returned position tensor.
    """
    graph = _small_test_graph()
    result = CytoscapeFcose().layout(graph, seed=42)
    assert result.pos is not None, f"fcose failed: {result.error}"
    assert result.pos.shape == (graph.num_nodes, 2)
    assert result.error is None


@pytest.mark.skipif(not FCOSE_AVAILABLE, reason="cytoscape-fcose not installed")
def test_cytoscape_fcose_seed_changes_layout() -> None:
    """The fcose helper should use the benchmark seed to drive random layout.

    Returns
    -------
    None
        Assertions validate same-seed reproducibility and different-seed
        divergence through the Node.js helper.
    """
    graph = _small_test_graph()
    competitor = CytoscapeFcose()

    first = competitor.layout(graph, seed=42)
    second = competitor.layout(graph, seed=42)
    third = competitor.layout(graph, seed=43)

    assert first.pos is not None, first.error
    assert second.pos is not None, second.error
    assert third.pos is not None, third.error
    assert first.error is None
    assert second.error is None
    assert third.error is None
    torch.testing.assert_close(first.pos, second.pos)
    assert not torch.allclose(first.pos, third.pos)


@pytest.mark.skipif(not GEPHI_AVAILABLE, reason="Gephi toolkit not available")
def test_gephi_yifanhu_layout() -> None:
    """The Gephi adapter should return positions for a small graph.

    Returns
    -------
    None
        This test asserts on the returned position tensor.
    """
    graph = _small_test_graph()
    result = GephiYifanHu().layout(graph, seed=42)
    assert result.pos is not None, f"gephi failed: {result.error}"
    assert result.pos.shape == (graph.num_nodes, 2)
    assert result.error is None
