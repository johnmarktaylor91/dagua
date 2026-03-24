"""Tests for the Cytoscape fcose and Gephi YifanHu competitor adapters."""

from __future__ import annotations

import pytest

from dagua.eval.competitors import get_competitors
from dagua.eval.competitors.cytoscape_fcose_competitor import CytoscapeFcose
from dagua.eval.competitors.gephi_competitor import GephiYifanHu
from dagua.eval.graphs import get_test_graphs
from dagua.graph import DaguaGraph

pytestmark = pytest.mark.smoke

FCOSE_AVAILABLE = CytoscapeFcose().available()
GEPHI_AVAILABLE = GephiYifanHu().available()


def _small_test_graph() -> DaguaGraph:
    """Return a small evaluation graph used by the external adapter tests.

    Returns
    -------
    DaguaGraph
        First graph from the small evaluation corpus slice.
    """
    return get_test_graphs(max_nodes=50)[0].graph


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
