"""Tests for pie chart fills and edge crossing jumps."""

from __future__ import annotations

import matplotlib
import pytest
import torch

from dagua.edges import BezierCurve
from dagua.graph import DaguaGraph
from dagua.render.crossings import EdgeCrossing, detect_crossings
from dagua.styles import EdgeStyle, NodeStyle

matplotlib.use("Agg")


def _crossing_curves() -> list[BezierCurve]:
    """Return two straight cubic curves that cross at the origin.

    Returns
    -------
    list[BezierCurve]
        Two routed curves whose interiors cross once.
    """
    return [
        BezierCurve(p0=(-30.0, 30.0), cp1=(-10.0, 10.0), cp2=(10.0, -10.0), p1=(30.0, -30.0)),
        BezierCurve(p0=(30.0, 30.0), cp1=(10.0, 10.0), cp2=(-10.0, -10.0), p1=(-30.0, -30.0)),
    ]


class TestPieChartStyle:
    """Coverage for pie-chart node style fields and rendering."""

    def test_pie_fields(self) -> None:
        """Pie fill style should retain slice colors and values."""
        style = NodeStyle(
            fill_pattern="pie",
            fill_pattern_colors=["#FF0000", "#00FF00", "#0000FF"],
            fill_pattern_values=[1.0, 2.0, 1.0],
        )
        assert style.fill_pattern == "pie"
        assert len(style.fill_pattern_colors or []) == 3
        assert len(style.fill_pattern_values or []) == 3

    def test_donut(self) -> None:
        """Donut charts should retain the configured hole fraction."""
        style = NodeStyle(
            fill_pattern="pie",
            fill_pattern_colors=["#FF0000", "#0000FF"],
            fill_pattern_values=[1.0, 1.0],
            fill_pattern_hole=0.4,
        )
        assert style.fill_pattern_hole == 0.4

    def test_render_pie(self) -> None:
        """Pie chart node rendering should complete without errors."""
        import matplotlib.pyplot as plt

        from dagua.render import render

        graph = DaguaGraph()
        graph.add_node(
            "A",
            label="Pie",
            style=NodeStyle(
                fill_pattern="pie",
                fill_pattern_colors=["#FF6384", "#36A2EB", "#FFCE56"],
                fill_pattern_values=[30.0, 50.0, 20.0],
                shape="circle",
            ),
        )
        graph.add_node("B", label="Normal")
        graph.add_edge("A", "B")
        graph.compute_node_sizes()
        graph.cache_layout(torch.tensor([[0.0, 30.0], [0.0, -30.0]], dtype=torch.float32))

        fig, _ax = render(graph, show=False)
        assert fig is not None
        plt.close(fig)


class TestEdgeCrossingStyle:
    """Coverage for edge crossing style fields."""

    def test_crossing_fields(self) -> None:
        """Crossing jump styles should retain the configured settings."""
        style = EdgeStyle(crossing_style="bridge", crossing_size=8.0)
        assert style.crossing_style == "bridge"
        assert style.crossing_size == 8.0

    def test_default_none(self) -> None:
        """Edges should default to not rendering crossing jumps."""
        assert EdgeStyle().crossing_style == "none"


class TestCrossingDetection:
    """Coverage for curve-curve crossing detection."""

    def test_no_crossings(self) -> None:
        """Parallel curves should not report crossings."""
        curve_a = BezierCurve(p0=(0.0, 0.0), cp1=(10.0, 0.0), cp2=(90.0, 0.0), p1=(100.0, 0.0))
        curve_b = BezierCurve(
            p0=(0.0, 10.0),
            cp1=(10.0, 10.0),
            cp2=(90.0, 10.0),
            p1=(100.0, 10.0),
        )
        crossings = detect_crossings([curve_a, curve_b], edge_count=2)
        assert len(crossings) == 0

    def test_crossing_detected(self) -> None:
        """A simple X pattern should report one central crossing."""
        crossings = detect_crossings(_crossing_curves(), edge_count=2)
        assert len(crossings) == 1
        assert abs(crossings[0].x) < 2.0
        assert abs(crossings[0].y) < 2.0

    def test_crossing_dataclass(self) -> None:
        """The crossing record should expose both edge indices and parameters."""
        crossing = EdgeCrossing(edge_a=0, edge_b=1, x=0.0, y=0.0, t_a=0.5, t_b=0.5)
        assert crossing.edge_a == 0
        assert crossing.edge_b == 1


class TestCrossingRender:
    """Coverage for crossing-jump rendering and the edge view cache."""

    @pytest.mark.parametrize("crossing_style", ["arc", "gap", "sharp", "bridge"])
    def test_render_with_crossings(self, crossing_style: str) -> None:
        """Crossing jump styles should render and populate edge-view crossings."""
        import matplotlib.pyplot as plt

        from dagua.render import render

        graph = DaguaGraph()
        graph.add_node("A", label="A")
        graph.add_node("B", label="B")
        graph.add_node("C", label="C")
        graph.add_node("D", label="D")
        graph.add_edge("A", "D", style=EdgeStyle(crossing_style=crossing_style))
        graph.add_edge("B", "C", style=EdgeStyle(crossing_style=crossing_style))
        graph.compute_node_sizes()
        graph.cache_layout(
            torch.tensor(
                [
                    [-30.0, 30.0],
                    [30.0, 30.0],
                    [-30.0, -30.0],
                    [30.0, -30.0],
                ],
                dtype=torch.float32,
            )
        )

        fig, _ax = render(graph, show=False, curves=_crossing_curves())
        assert fig is not None
        assert len(graph.edge(0).crossings) == 1
        assert len(graph.edge(1).crossings) == 1
        plt.close(fig)
