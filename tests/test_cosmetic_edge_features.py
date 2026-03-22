"""Tests for cosmetic edge features."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import torch

from dagua.graph import DaguaGraph
from dagua.render import render
from dagua.styles import EdgeStyle

matplotlib.use("Agg")


def _two_node_graph(style: EdgeStyle) -> DaguaGraph:
    """Build a minimal graph for edge-render smoke tests.

    Parameters
    ----------
    style : EdgeStyle
        Edge style applied to the single edge in the graph.

    Returns
    -------
    DaguaGraph
        Graph with one styled edge and cached node positions.
    """
    graph = DaguaGraph()
    graph.add_node("A", label="Source")
    graph.add_node("B", label="Target")
    graph.add_edge("A", "B", style=style, label="edge")
    graph.compute_node_sizes()
    graph.cache_layout(torch.tensor([[0.0, 30.0], [0.0, -30.0]], dtype=torch.float32))
    return graph


class TestTaperedEdge:
    """EdgeStyle tapered-body fields."""

    def test_taper_fields(self) -> None:
        """Taper-specific fields should round-trip through the dataclass."""
        style = EdgeStyle(taper=True, taper_width_start=4.0, taper_width_end=0.5)
        assert style.taper is True
        assert style.taper_width_start == 4.0
        assert style.taper_width_end == 0.5

    def test_default_no_taper(self) -> None:
        """Edges should remain untapered by default."""
        assert EdgeStyle().taper is False


class TestHeadTailLabels:
    """Endpoint-adjacent edge label fields."""

    def test_head_label(self) -> None:
        """Head label text should be stored on the style."""
        style = EdgeStyle(head_label="1")
        assert style.head_label == "1"

    def test_tail_label(self) -> None:
        """Tail label text should be stored on the style."""
        style = EdgeStyle(tail_label="*")
        assert style.tail_label == "*"

    def test_default_empty(self) -> None:
        """Endpoint labels should default to disabled."""
        assert EdgeStyle().head_label == ""
        assert EdgeStyle().tail_label == ""


class TestEdgeGradient:
    """Edge body gradient fields."""

    def test_gradient_fields(self) -> None:
        """Gradient settings should round-trip through the dataclass."""
        style = EdgeStyle(color_gradient="source_to_target", color_gradient_end="#FF0000")
        assert style.color_gradient == "source_to_target"
        assert style.color_gradient_end == "#FF0000"

    def test_default_none(self) -> None:
        """Edges should default to a flat body color."""
        assert EdgeStyle().color_gradient == "none"


class TestEdgeCapJoin:
    """Line cap and join style fields."""

    def test_cap_join_fields(self) -> None:
        """Explicit cap and join settings should be retained."""
        style = EdgeStyle(line_cap="round", line_join="bevel")
        assert style.line_cap == "round"
        assert style.line_join == "bevel"

    def test_cap_join_defaults(self) -> None:
        """Edge cap and join should default to the documented values."""
        style = EdgeStyle()
        assert style.line_cap == "butt"
        assert style.line_join == "miter"


class TestRenderSmoke:
    """Smoke tests for the new cosmetic render paths."""

    def test_render_tapered(self) -> None:
        """Tapered edges should render without raising."""
        graph = _two_node_graph(EdgeStyle(taper=True))
        fig, _ax = render(graph, show=False)
        assert fig is not None
        plt.close(fig)

    def test_render_gradient_edge(self) -> None:
        """Gradient edges should render without raising."""
        graph = _two_node_graph(
            EdgeStyle(
                color="#FF0000",
                color_gradient="source_to_target",
                color_gradient_end="#0000FF",
            )
        )
        fig, _ax = render(graph, show=False)
        assert fig is not None
        plt.close(fig)

    def test_render_head_tail_labels(self) -> None:
        """Head and tail labels should render without raising."""
        graph = _two_node_graph(EdgeStyle(head_label="1", tail_label="*", arrow="none"))
        fig, _ax = render(graph, show=False)
        assert fig is not None
        plt.close(fig)

    def test_render_custom_cap_join(self) -> None:
        """Custom cap and join settings should render through the direct path."""
        graph = _two_node_graph(EdgeStyle(line_cap="round", line_join="round", arrow="none"))
        fig, _ax = render(graph, show=False)
        assert fig is not None
        plt.close(fig)
