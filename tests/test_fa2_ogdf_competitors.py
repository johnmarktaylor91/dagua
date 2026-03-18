"""Smoke tests for the FA2 and OGDF competitor adapters."""

from __future__ import annotations

import pytest

from dagua.eval.competitors import get_competitors
from dagua.eval.competitors.fa2_competitor import FA2Reference
from dagua.eval.competitors.ogdf_competitor import (
    OGDFFMMM,
    OGDFDavidsonHarel,
    OGDFGem,
    OGDFStress,
    OGDFSugiyama,
)
from dagua.graph import DaguaGraph

pytestmark = pytest.mark.smoke

FA2_AVAILABLE = FA2Reference().available()
OGDF_AVAILABLE = OGDFGem().available()


def _make_small_graph() -> DaguaGraph:
    """Create a small connected graph for competitor smoke tests.

    Returns
    -------
    DaguaGraph
        Six-node chain graph.
    """
    graph = DaguaGraph()
    for node_idx in range(6):
        graph.add_node(str(node_idx), label=str(node_idx))
    for node_idx in range(5):
        graph.add_edge(str(node_idx), str(node_idx + 1))
    return graph


def test_fa2_and_ogdf_competitors_registered() -> None:
    """The new competitor adapters should register on import.

    Returns
    -------
    None
        This test asserts on the global competitor registry contents.
    """
    names = {competitor.name for competitor in get_competitors()}
    assert {
        "fa2_ref",
        "ogdf_gem",
        "ogdf_fmmm",
        "ogdf_stress",
        "ogdf_sugiyama",
        "ogdf_davidson_harel",
    } <= names


def test_fa2_available_check_returns_bool() -> None:
    """The FA2 availability probe should return a boolean.

    Returns
    -------
    None
        This test asserts on the availability probe result.
    """
    assert isinstance(FA2Reference().available(), bool)


@pytest.mark.skipif(not FA2_AVAILABLE, reason="ForceAtlas2 reference package not usable")
def test_fa2_layout_returns_positions() -> None:
    """The FA2 adapter should return positions for a small graph.

    Returns
    -------
    None
        This test asserts on the returned position tensor.
    """
    graph = _make_small_graph()
    result = FA2Reference().layout(graph, timeout=30.0)
    assert result.pos is not None
    assert result.pos.shape == (6, 2)
    assert result.error is None


def test_ogdf_available_check_returns_bool() -> None:
    """The OGDF availability probe should return a boolean.

    Returns
    -------
    None
        This test asserts on the availability probe result.
    """
    assert isinstance(OGDFGem().available(), bool)


@pytest.mark.skipif(not OGDF_AVAILABLE, reason="OGDF Python bindings not usable")
def test_ogdf_gem_layout() -> None:
    """The GEM adapter should return positions for a small graph.

    Returns
    -------
    None
        This test asserts on the returned position tensor.
    """
    graph = _make_small_graph()
    result = OGDFGem().layout(graph, timeout=30.0)
    assert result.pos is not None
    assert result.pos.shape == (6, 2)
    assert result.error is None


@pytest.mark.skipif(not OGDF_AVAILABLE, reason="OGDF Python bindings not usable")
def test_ogdf_fmmm_layout() -> None:
    """The FM^3 adapter should return positions for a small graph.

    Returns
    -------
    None
        This test asserts on the returned position tensor.
    """
    graph = _make_small_graph()
    result = OGDFFMMM().layout(graph, timeout=30.0)
    assert result.pos is not None
    assert result.pos.shape == (6, 2)
    assert result.error is None


@pytest.mark.skipif(not OGDF_AVAILABLE, reason="OGDF Python bindings not usable")
def test_ogdf_stress_layout() -> None:
    """The stress adapter should return positions for a small graph.

    Returns
    -------
    None
        This test asserts on the returned position tensor.
    """
    graph = _make_small_graph()
    result = OGDFStress().layout(graph, timeout=30.0)
    assert result.pos is not None
    assert result.pos.shape == (6, 2)
    assert result.error is None


@pytest.mark.skipif(not OGDF_AVAILABLE, reason="OGDF Python bindings not usable")
def test_ogdf_sugiyama_layout() -> None:
    """The Sugiyama adapter should return positions for a small graph.

    Returns
    -------
    None
        This test asserts on the returned position tensor.
    """
    graph = _make_small_graph()
    result = OGDFSugiyama().layout(graph, timeout=30.0)
    assert result.pos is not None
    assert result.pos.shape == (6, 2)
    assert result.error is None


@pytest.mark.skipif(not OGDF_AVAILABLE, reason="OGDF Python bindings not usable")
def test_ogdf_davidson_harel_layout() -> None:
    """The Davidson-Harel adapter should return positions for a small graph.

    Returns
    -------
    None
        This test asserts on the returned position tensor.
    """
    graph = _make_small_graph()
    result = OGDFDavidsonHarel().layout(graph, timeout=30.0)
    assert result.pos is not None
    assert result.pos.shape == (6, 2)
    assert result.error is None
