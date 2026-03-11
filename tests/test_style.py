"""Tests for NodeStyle, EdgeStyle, ClusterStyle, GraphStyle, Theme."""

import pytest

from dagua.styles import (
    ClusterStyle, EdgeStyle, GraphStyle, NodeStyle, Theme,
    DEFAULT_THEME_OBJ, DARK_THEME, MINIMAL_THEME,
    DEFAULT_THEME, DEFAULT_NODE_STYLES,
    GRAPHVIZ_MATCH_THEME, GRAPHVIZ_MATCH_NODE_STYLES,
    darken_hex, make_fill, border_from_fill,
    PALETTE, WARM_WHITE, NEAR_BLACK, MEDIUM_GRAY,
)


@pytest.mark.smoke
class TestNodeStyleNewFields:
    """New fields on NodeStyle (font_weight, font_style, shadow, min_width)."""

    def test_defaults(self):
        s = NodeStyle()
        assert s.font_weight == "regular"
        assert s.font_style == "normal"
        assert s.shadow is False
        assert s.shadow_offset == (1.5, -1.5)
        assert s.shadow_color == "#00000020"
        assert s.min_width is None

    def test_custom_values(self):
        s = NodeStyle(font_weight="bold", font_style="italic", shadow=True, min_width=100.0)
        assert s.font_weight == "bold"
        assert s.font_style == "italic"
        assert s.shadow is True
        assert s.min_width == 100.0


@pytest.mark.smoke
class TestEdgeStyleNewFields:
    """New fields on EdgeStyle (routing, label_font_size, etc.)."""

    def test_defaults(self):
        s = EdgeStyle()
        assert s.routing == "bezier"
        assert s.label_font_size == 7.0
        assert s.label_font_color == NEAR_BLACK
        assert s.label_background == WARM_WHITE

    def test_custom_routing(self):
        s = EdgeStyle(routing="straight")
        assert s.routing == "straight"

    def test_custom_label_style(self):
        s = EdgeStyle(label_font_size=9.0, label_font_color="#FF0000")
        assert s.label_font_size == 9.0
        assert s.label_font_color == "#FF0000"


@pytest.mark.smoke
class TestClusterStyleNewFields:
    """New fields on ClusterStyle (font_family, label_offset, depth_*_step)."""

    def test_defaults(self):
        s = ClusterStyle()
        assert s.font_family == ""
        assert s.label_offset == (6.0, 6.0)
        assert s.depth_fill_step == 0.03
        assert s.depth_stroke_step == 0.05

    def test_custom_values(self):
        s = ClusterStyle(font_family="monospace", label_offset=(10.0, 8.0), depth_fill_step=0.05)
        assert s.font_family == "monospace"
        assert s.label_offset == (10.0, 8.0)
        assert s.depth_fill_step == 0.05


@pytest.mark.smoke
class TestGraphStyle:
    """GraphStyle dataclass."""

    def test_defaults(self):
        gs = GraphStyle()
        assert gs.background_color == WARM_WHITE
        assert gs.margin == 30.0
        assert gs.title_font_size == 10.0
        assert gs.max_figsize == (30.0, 40.0)
        assert gs.min_figsize == (4.0, 3.0)
        assert gs.node_label_secondary_scale == 0.85
        assert gs.edge_label_background_opacity == 0.85

    def test_custom_values(self):
        gs = GraphStyle(background_color="#000000", margin=50.0)
        assert gs.background_color == "#000000"
        assert gs.margin == 50.0


@pytest.mark.smoke
class TestTheme:
    """Theme dataclass and lookup methods."""

    def test_defaults(self):
        t = Theme()
        assert t.name == "default"
        assert t.node_styles == {}
        assert t.edge_styles == {}
        assert isinstance(t.cluster_style, ClusterStyle)
        assert isinstance(t.graph_style, GraphStyle)

    def test_get_node_style_found(self):
        t = Theme(node_styles={"input": NodeStyle(base_color="#FF0000")})
        s = t.get_node_style("input")
        assert s.base_color == "#FF0000"

    def test_get_node_style_fallback_to_default(self):
        t = Theme(node_styles={"default": NodeStyle(base_color="#00FF00")})
        s = t.get_node_style("nonexistent")
        assert s.base_color == "#00FF00"

    def test_get_node_style_fallback_to_new(self):
        t = Theme()
        s = t.get_node_style("anything")
        assert isinstance(s, NodeStyle)

    def test_get_edge_style_found(self):
        t = Theme(edge_styles={"if": EdgeStyle(style="dashed")})
        s = t.get_edge_style("if")
        assert s.style == "dashed"

    def test_get_edge_style_fallback(self):
        t = Theme(edge_styles={"default": EdgeStyle(color="#123456")})
        s = t.get_edge_style("nonexistent")
        assert s.color == "#123456"

    def test_copy_is_independent(self):
        t = Theme(node_styles={"default": NodeStyle(base_color="#FF0000")})
        t2 = t.copy()
        t2.node_styles["default"] = NodeStyle(base_color="#00FF00")
        assert t.get_node_style("default").base_color == "#FF0000"


@pytest.mark.smoke
class TestBuiltInThemes:
    """Built-in theme objects exist and are valid."""

    def test_default_theme_obj(self):
        assert DEFAULT_THEME_OBJ.name == "default"
        assert "default" in DEFAULT_THEME_OBJ.node_styles
        assert "input" in DEFAULT_THEME_OBJ.node_styles
        assert "if" in DEFAULT_THEME_OBJ.edge_styles
        assert "then" in DEFAULT_THEME_OBJ.edge_styles

    def test_dark_theme(self):
        assert DARK_THEME.name == "dark"
        assert DARK_THEME.graph_style.background_color == "#1A1E24"
        assert "default" in DARK_THEME.node_styles

    def test_minimal_theme(self):
        assert MINIMAL_THEME.name == "minimal"
        assert MINIMAL_THEME.get_node_style("default").shape == "rect"
        assert MINIMAL_THEME.get_node_style("default").corner_radius == 0.0

    def test_backwards_compat_aliases(self):
        assert DEFAULT_THEME is DEFAULT_NODE_STYLES
        assert GRAPHVIZ_MATCH_THEME is GRAPHVIZ_MATCH_NODE_STYLES
        assert isinstance(DEFAULT_THEME, dict)
        assert "default" in DEFAULT_THEME


@pytest.mark.smoke
class TestDarkenHex:
    """darken_hex utility."""

    def test_darken_white(self):
        result = darken_hex("#FFFFFF", 0.1)
        # Should be slightly darker
        assert result != "#ffffff"

    def test_zero_amount(self):
        result = darken_hex("#FAFAFA", 0.0)
        # Should be unchanged (or very close)
        assert result.lower() == "#fafafa"
