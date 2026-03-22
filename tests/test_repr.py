"""Tests for compact style repr implementations."""

from __future__ import annotations

from dagua.graph import DaguaGraph
from dagua.styles import ClusterStyle, EdgeStyle, GraphStyle, NodeStyle


class TestNodeStyleRepr:
    """Node style repr tests."""

    def test_default_is_compact(self) -> None:
        """Default node styles render as an empty compact repr."""
        rendered = repr(NodeStyle())
        assert rendered == "NodeStyle()"

    def test_custom_shows_changes(self) -> None:
        """Changed node fields appear in the repr output."""
        rendered = repr(NodeStyle(shape="circle", fill="#FF0000"))
        assert "circle" in rendered
        assert "#FF0000" in rendered
        assert len(rendered) < 100

    def test_many_changes_truncated(self) -> None:
        """Large reprs truncate after the high-priority fields."""
        rendered = repr(
            NodeStyle(
                shape="diamond",
                fill="#FF0000",
                stroke="#00FF00",
                font_size=20.0,
                font_color="#0000FF",
                opacity=0.5,
                shadow=True,
                gradient="linear",
                corner_radius=10.0,
            )
        )
        assert "NodeStyle(" in rendered
        assert "...+" in rendered


class TestEdgeStyleRepr:
    """Edge style repr tests."""

    def test_default_is_compact(self) -> None:
        """Default edge styles render as an empty compact repr."""
        rendered = repr(EdgeStyle())
        assert rendered == "EdgeStyle()"

    def test_custom_shows_changes(self) -> None:
        """Changed edge fields appear in the repr output."""
        rendered = repr(EdgeStyle(color="#FF0000", width=3.0, arrow="diamond"))
        assert "#FF0000" in rendered
        assert "diamond" in rendered


class TestClusterStyleRepr:
    """Cluster style repr tests."""

    def test_default_is_compact(self) -> None:
        """Default cluster styles render as an empty compact repr."""
        rendered = repr(ClusterStyle())
        assert rendered == "ClusterStyle()"

    def test_custom_shows_changes(self) -> None:
        """Changed cluster fields appear in the repr output."""
        rendered = repr(ClusterStyle(fill="#EFEFEF", padding=22.0))
        assert "#EFEFEF" in rendered
        assert "padding=22.0" in rendered


class TestGraphStyleRepr:
    """Graph style repr tests."""

    def test_default_is_compact(self) -> None:
        """Default graph styles render as an empty compact repr."""
        rendered = repr(GraphStyle())
        assert rendered == "GraphStyle()"

    def test_custom_shows_changes(self) -> None:
        """Changed graph fields appear in the repr output."""
        rendered = repr(GraphStyle(background_color="#101010", margin=24.0))
        assert "#101010" in rendered
        assert "margin=24.0" in rendered


class TestGraphRepr:
    """Graph repr regression tests."""

    def test_cyclic_graph_shows_cycle_flag(self) -> None:
        """Cyclic graphs advertise the cycle flag in their compact repr."""
        graph = DaguaGraph.from_edge_list([("a", "b"), ("b", "a")])
        rendered = repr(graph)
        assert "cyclic=True" in rendered
