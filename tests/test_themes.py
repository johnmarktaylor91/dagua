"""Regression tests for built-in theme registration and rendering."""

from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import pytest
import torch

from dagua.graph import DaguaGraph
from dagua.render import render
from dagua.styles import THEME_REGISTRY, Theme

NEW_THEMES: List[str] = [
    "mermaid",
    "d3",
    "cytoscape",
    "gephi",
    "obsidian",
    "yed",
    "drawio",
]


def _make_test_graph(theme_name: str) -> DaguaGraph:
    """Build a minimal graph configured with the requested theme.

    Parameters
    ----------
    theme_name : str
        Name of the built-in theme to attach to the graph.

    Returns
    -------
    DaguaGraph
        Three-node path graph with cached node sizes and positions.
    """
    graph = DaguaGraph()
    graph._theme = THEME_REGISTRY[theme_name]
    graph.add_edge("A", "B")
    graph.add_edge("B", "C")
    graph.compute_node_sizes()
    graph.cache_layout(torch.tensor([[0.0, 40.0], [-30.0, 0.0], [30.0, 0.0]], dtype=torch.float32))
    return graph


@pytest.mark.parametrize("name", NEW_THEMES)
def test_registered(name: str) -> None:
    """Each new competitor theme should be present in the registry.

    Parameters
    ----------
    name : str
        Registry key under test.

    Returns
    -------
    None
    """
    assert name in THEME_REGISTRY


@pytest.mark.parametrize("name", NEW_THEMES)
def test_has_default_node_style(name: str) -> None:
    """Each new theme should expose a default node style.

    Parameters
    ----------
    name : str
        Registry key under test.

    Returns
    -------
    None
    """
    theme: Theme = THEME_REGISTRY[name]
    assert "default" in theme.node_styles


@pytest.mark.parametrize("name", NEW_THEMES)
def test_has_default_edge_style(name: str) -> None:
    """Each new theme should expose a default edge style.

    Parameters
    ----------
    name : str
        Registry key under test.

    Returns
    -------
    None
    """
    theme: Theme = THEME_REGISTRY[name]
    assert "default" in theme.edge_styles


@pytest.mark.parametrize("name", NEW_THEMES)
def test_has_cluster_style(name: str) -> None:
    """Each new theme should expose a cluster style object.

    Parameters
    ----------
    name : str
        Registry key under test.

    Returns
    -------
    None
    """
    theme: Theme = THEME_REGISTRY[name]
    assert theme.cluster_style is not None


@pytest.mark.parametrize("name", NEW_THEMES)
def test_has_graph_style(name: str) -> None:
    """Each new theme should expose a graph style object.

    Parameters
    ----------
    name : str
        Registry key under test.

    Returns
    -------
    None
    """
    theme: Theme = THEME_REGISTRY[name]
    assert theme.graph_style is not None


@pytest.mark.parametrize("name", NEW_THEMES)
def test_render_smoke(name: str) -> None:
    """Each new theme should render a simple graph without error.

    Parameters
    ----------
    name : str
        Registry key under test.

    Returns
    -------
    None
    """
    graph = _make_test_graph(name)
    figure, _axes = render(graph, show=False)
    assert figure is not None
    plt.close(figure)
