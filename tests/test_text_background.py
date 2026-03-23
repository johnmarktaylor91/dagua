"""Tests for text background style exposure and renderer wiring."""

from __future__ import annotations

from typing import Any, List

import matplotlib.pyplot as plt
import pytest
import torch

from dagua.graph import DaguaGraph
from dagua.render import mpl as mpl_renderer
from dagua.render.text import DaguaText
from dagua.styles import EdgeStyle, NodeStyle


def _single_node_graph(style: NodeStyle) -> tuple[DaguaGraph, torch.Tensor]:
    """Build a one-node graph for label rendering assertions.

    Parameters
    ----------
    style : NodeStyle
        Per-node style override.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph and explicit node positions.
    """
    graph = DaguaGraph()
    graph.add_node("node", label="Node", style=style)
    positions = torch.tensor([[0.0, 0.0]])
    return graph, positions


def _single_edge_graph(style: EdgeStyle) -> tuple[DaguaGraph, torch.Tensor]:
    """Build a one-edge graph for edge-label rendering assertions.

    Parameters
    ----------
    style : EdgeStyle
        Per-edge style override.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph and explicit node positions.
    """
    graph = DaguaGraph()
    graph.add_node("a", label="A")
    graph.add_node("b", label="B")
    graph.add_edge("a", "b", label="edge", style=style)
    positions = torch.tensor([[0.0, 0.0], [0.0, 80.0]])
    return graph, positions


def _capture_text_specs(
    monkeypatch: pytest.MonkeyPatch,
    graph: DaguaGraph,
    positions: torch.Tensor,
) -> List[DaguaText]:
    """Render a graph while capturing emitted text specifications.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture.
    graph : DaguaGraph
        Graph to render.
    positions : torch.Tensor
        Explicit node positions with shape ``[N, 2]``.

    Returns
    -------
    list[DaguaText]
        Collected text specs from the matplotlib renderer.
    """
    captured: List[DaguaText] = []

    def _record_specs(
        ax: Any,
        specs: List[DaguaText],
        display_scale: float,
        svg_hover_map: Any = None,
    ) -> list[Any]:
        """Record text specs instead of creating artists.

        Parameters
        ----------
        ax : Any
            Matplotlib axes.
        specs : list[DaguaText]
            Text specs to record.
        display_scale : float
            Display-to-data scale factor.
        svg_hover_map : Any, default=None
            Optional SVG hover map.

        Returns
        -------
        list[Any]
            Empty artist list for renderer compatibility.
        """
        del ax, display_scale, svg_hover_map
        captured.extend(specs)
        return []

    monkeypatch.setattr(mpl_renderer, "render_text", _record_specs)
    fig, _ = mpl_renderer.render(graph, positions=positions)
    plt.close(fig)
    return captured


def _find_spec(specs: List[DaguaText], gid: str) -> DaguaText:
    """Find one captured text spec by gid.

    Parameters
    ----------
    specs : list[DaguaText]
        Captured text specs.
    gid : str
        Target gid.

    Returns
    -------
    DaguaText
        Matching text spec.

    Raises
    ------
    AssertionError
        Raised when no matching spec exists.
    """
    for spec in specs:
        if spec.gid == gid:
            return spec
    raise AssertionError(f"missing text spec {gid!r}")


class TestTextBackgroundFields:
    """Style dataclasses should expose text background controls."""

    def test_node_style_has_text_background(self) -> None:
        """NodeStyle should expose a text background color field.

        Returns
        -------
        None
        """
        style = NodeStyle()

        assert hasattr(style, "text_background")
        assert style.text_background == ""

    def test_node_style_has_text_background_opacity(self) -> None:
        """NodeStyle should expose text background opacity.

        Returns
        -------
        None
        """
        style = NodeStyle()

        assert hasattr(style, "text_background_opacity")
        assert style.text_background_opacity == 0.85

    def test_edge_style_has_label_background_opacity(self) -> None:
        """EdgeStyle should expose label background opacity.

        Returns
        -------
        None
        """
        style = EdgeStyle()

        assert hasattr(style, "label_background_opacity")
        assert style.label_background_opacity == 0.85

    def test_custom_text_background(self) -> None:
        """NodeStyle should preserve custom text background settings.

        Returns
        -------
        None
        """
        style = NodeStyle(text_background="#FFFF00", text_background_opacity=0.5)

        assert style.text_background == "#FFFF00"
        assert style.text_background_opacity == 0.5


class TestTextBackgroundRendering:
    """Renderer should forward style background fields into DaguaText."""

    def test_node_label_background_is_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Node label background fields should be wired into the text spec.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture.

        Returns
        -------
        None
        """
        graph, positions = _single_node_graph(
            NodeStyle(
                text_background="#FFFF00",
                text_background_opacity=0.5,
                text_background_padding=(4.0, 1.5),
                text_background_corner_radius=6.0,
            )
        )

        specs = _capture_text_specs(monkeypatch, graph, positions)
        spec = _find_spec(specs, "dagua-node-label-0")

        assert spec.background == "#FFFF00"
        assert spec.background_alpha == pytest.approx(0.5)
        assert spec.background_padding == pytest.approx((4.0, 1.5))
        assert spec.background_corner_radius == pytest.approx(6.0)

    def test_edge_label_background_is_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Edge label background fields should be wired into the text spec.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture.

        Returns
        -------
        None
        """
        graph, positions = _single_edge_graph(
            EdgeStyle(
                label_background="#ABCDEF",
                label_background_opacity=0.35,
                label_background_padding=(5.0, 6.0),
                label_background_corner_radius=7.0,
            )
        )

        specs = _capture_text_specs(monkeypatch, graph, positions)
        spec = _find_spec(specs, "dagua-edge-label-0")

        assert spec.background == "#ABCDEF"
        assert spec.background_alpha == pytest.approx(0.35)
        assert spec.background_padding == pytest.approx((5.0, 6.0))
        assert spec.background_corner_radius == pytest.approx(7.0)

    def test_gradient_node_labels_use_soft_white_auto_background(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Gradient node labels should use a higher-opacity white auto background.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture.

        Returns
        -------
        None
        """
        graph, positions = _single_node_graph(NodeStyle(gradient="linear"))

        specs = _capture_text_specs(monkeypatch, graph, positions)
        spec = _find_spec(specs, "dagua-node-label-0")

        assert spec.background == "#FFFFFF"
        assert spec.background_alpha == pytest.approx(0.85)

    @pytest.mark.parametrize("fill_pattern", ["pie", "striped"])
    def test_patterned_node_labels_use_opaque_white_auto_background(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fill_pattern: str,
    ) -> None:
        """Pie and striped node labels should ignore the graph background color.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Pytest monkeypatch fixture.
        fill_pattern : str
            Patterned node fill that requires a neutral text pill.

        Returns
        -------
        None
            The generated text background is asserted in place.
        """
        graph, positions = _single_node_graph(NodeStyle(fill_pattern=fill_pattern))
        graph.graph_style.background_color = "#0F172A"

        specs = _capture_text_specs(monkeypatch, graph, positions)
        spec = _find_spec(specs, "dagua-node-label-0")

        assert spec.background == "#FFFFFF"
        assert spec.background_alpha == pytest.approx(0.92)
