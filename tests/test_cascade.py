"""Tests for 5-level style cascade resolution."""

import pytest

from dagua.styles import (
    ClusterStyle,
    EdgeStyle,
    NodeStyle,
    resolve_cluster_style,
    resolve_edge_style,
    resolve_node_style,
)


class TestResolveNodeStyle:
    def test_theme_only(self):
        """With no overrides, theme style is returned."""
        theme = NodeStyle(base_color="#0072B2")
        result = resolve_node_style(None, None, theme)
        assert result.base_color == "#0072B2"

    def test_per_element_wins(self):
        """Per-element override takes highest priority."""
        per_el = NodeStyle(font_size=14.0)
        theme = NodeStyle(font_size=8.5)
        result = resolve_node_style(per_el, None, theme)
        assert result.font_size == 14.0

    def test_cluster_member_style(self):
        """Cluster member_node_style overrides theme."""
        cluster_style = NodeStyle(base_color="#D55E00")
        theme = NodeStyle(base_color="#56B4E9")
        result = resolve_node_style(None, [cluster_style], theme)
        assert result.base_color == "#D55E00"

    def test_deepest_cluster_wins(self):
        """Deepest cluster's member style takes priority over shallower."""
        shallow = NodeStyle(font_size=10.0)
        deep = NodeStyle(font_size=16.0)
        theme = NodeStyle()
        # deepest first in the list
        result = resolve_node_style(None, [deep, shallow], theme)
        assert result.font_size == 16.0

    def test_graph_default(self):
        """Graph default fills in when theme has default values."""
        theme = NodeStyle()  # all defaults
        graph_default = NodeStyle(corner_radius=8.0)
        result = resolve_node_style(None, None, theme, graph_default=graph_default)
        assert result.corner_radius == 8.0

    def test_global_default(self):
        """Global default is lowest priority."""
        theme = NodeStyle()
        global_default = NodeStyle(opacity=0.5)
        result = resolve_node_style(None, None, theme, global_default=global_default)
        assert result.opacity == 0.5

    def test_full_cascade(self):
        """All 5 levels: per-element > cluster > theme > graph > global."""
        per_el = NodeStyle(font_size=20.0)
        cluster = NodeStyle(opacity=0.8)
        theme = NodeStyle(corner_radius=6.0)
        graph_default = NodeStyle(stroke_width=2.0)
        global_default = NodeStyle(font_weight="bold")

        result = resolve_node_style(
            per_el, [cluster], theme,
            graph_default=graph_default,
            global_default=global_default,
        )
        assert result.font_size == 20.0      # from per_el
        assert result.opacity == 0.8          # from cluster
        assert result.corner_radius == 6.0    # from theme
        assert result.stroke_width == 2.0     # from graph_default
        assert result.font_weight == "bold"   # from global_default

    def test_none_sources_skipped(self):
        """None sources in cascade are safely skipped."""
        theme = NodeStyle(font_size=10.0)
        result = resolve_node_style(None, [None, None], theme, None, None)
        assert result.font_size == 10.0


class TestResolveEdgeStyle:
    def test_theme_only(self):
        theme = EdgeStyle(color="#FF0000")
        result = resolve_edge_style(None, None, theme)
        assert result.color == "#FF0000"

    def test_per_element_wins(self):
        per_el = EdgeStyle(width=3.0)
        theme = EdgeStyle(width=0.75)
        result = resolve_edge_style(per_el, None, theme)
        assert result.width == 3.0

    def test_cluster_member_edge_style(self):
        cluster = EdgeStyle(color="#00FF00")
        theme = EdgeStyle()
        result = resolve_edge_style(None, [cluster], theme)
        assert result.color == "#00FF00"

    def test_graph_default(self):
        theme = EdgeStyle()
        graph_default = EdgeStyle(opacity=0.3)
        result = resolve_edge_style(None, None, theme, graph_default=graph_default)
        assert result.opacity == 0.3


class TestResolveClusterStyle:
    def test_theme_only(self):
        theme = ClusterStyle(padding=20.0)
        result = resolve_cluster_style(None, theme)
        assert result.padding == 20.0

    def test_per_cluster_wins(self):
        per = ClusterStyle(corner_radius=10.0)
        theme = ClusterStyle()
        result = resolve_cluster_style(per, theme)
        assert result.corner_radius == 10.0

    def test_global_default(self):
        theme = ClusterStyle()
        global_default = ClusterStyle(stroke_width=2.0)
        result = resolve_cluster_style(None, theme, global_default=global_default)
        assert result.stroke_width == 2.0


class TestClusterMemberStyleIntegration:
    def test_cluster_with_member_node_style(self):
        """ClusterStyle.member_node_style affects nodes within the cluster."""
        from dagua.graph import DaguaGraph

        g = DaguaGraph()
        g.add_node("a", label="A")
        g.add_node("b", label="B")
        g.add_node("c", label="C")
        g.add_edge("a", "b")
        g.add_edge("b", "c")

        member_style = NodeStyle(base_color="#D55E00", font_weight="bold")
        g.add_cluster(
            "encoder",
            members=["a", "b"],
            style=ClusterStyle(member_node_style=member_style),
        )

        # Node "a" is in the cluster — should get member style
        style_a = g.get_style_for_node(g._id_to_index["a"])
        assert style_a.base_color == "#D55E00"
        assert style_a.font_weight == "bold"

        # Node "c" is not in the cluster — should get theme default
        style_c = g.get_style_for_node(g._id_to_index["c"])
        assert style_c.base_color != "#D55E00"

    def test_graph_default_node_style(self):
        """Graph.default_node_style fills in as cascade level 4."""
        from dagua.graph import DaguaGraph

        g = DaguaGraph()
        g.default_node_style = NodeStyle(corner_radius=12.0)
        g.add_node("a")

        style = g.get_style_for_node(0)
        assert style.corner_radius == 12.0

    def test_per_element_overrides_cluster_member(self):
        """Per-node style beats cluster member_node_style."""
        from dagua.graph import DaguaGraph

        g = DaguaGraph()
        g.add_node("a", style=NodeStyle(font_size=20.0))
        g.add_node("b")
        g.add_edge("a", "b")

        member_style = NodeStyle(font_size=10.0)
        g.add_cluster("grp", members=["a", "b"],
                       style=ClusterStyle(member_node_style=member_style))

        style = g.get_style_for_node(g._id_to_index["a"])
        assert style.font_size == 20.0  # per-element wins


class TestIOCascade:
    """Test that defaults/flex/member_styles roundtrip through YAML/JSON."""

    def test_defaults_roundtrip_json(self):
        from dagua.graph import DaguaGraph

        g = DaguaGraph()
        g.default_node_style = NodeStyle(font_size=12.0)
        g.add_node("a")
        g.add_edge("a", "a")

        data = g.to_json()
        assert "defaults" in data
        assert data["defaults"]["node_style"]["font_size"] == 12.0

        g2 = DaguaGraph.from_json(data)
        assert g2.default_node_style is not None
        assert g2.default_node_style.font_size == 12.0

    def test_flex_roundtrip_json(self):
        from dagua.flex import Flex, LayoutFlex
        from dagua.graph import DaguaGraph

        g = DaguaGraph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        g.flex = LayoutFlex(
            node_sep=Flex.firm(40),
            pins={"a": (Flex.locked(0), Flex.locked(0))},
        )

        data = g.to_json()
        assert "flex" in data
        assert data["flex"]["node_sep"]["target"] == 40
        assert data["flex"]["node_sep"]["weight"] == 2.0

        g2 = DaguaGraph.from_json(data)
        assert g2.flex is not None
        assert g2.flex.node_sep.target == 40

    def test_member_style_roundtrip_json(self):
        from dagua.graph import DaguaGraph

        g = DaguaGraph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        g.add_cluster(
            "grp", members=["a", "b"],
            style=ClusterStyle(
                member_node_style=NodeStyle(base_color="#D55E00"),
            ),
        )

        data = g.to_json()
        cluster_data = data["clusters"][0]
        assert "style" in cluster_data
        assert "member_node_style" in cluster_data["style"]
