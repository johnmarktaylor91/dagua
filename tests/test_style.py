"""Tests for NodeStyle, EdgeStyle, ClusterStyle, GraphStyle, Theme."""

import dataclasses

import pytest

from dagua.styles import (
    DARK_THEME,
    DEFAULT_NODE_STYLES,
    DEFAULT_THEME,
    DEFAULT_THEME_OBJ,
    GRAPHVIZ_MATCH_DEFAULTS,
    GRAPHVIZ_MATCH_NODE_STYLES,
    GRAPHVIZ_MATCH_THEME,
    MINIMAL_THEME,
    NEAR_BLACK,
    WARM_WHITE,
    ClusterStyle,
    EdgeStyle,
    GraphStyle,
    NodeStyle,
    Theme,
    darken_hex,
)


@pytest.mark.smoke
class TestNodeStyleNewFields:
    """New fields on NodeStyle (font_weight, font_style, shadow, min_*)."""

    def test_defaults(self):
        s = NodeStyle()
        assert s.padding == (14.0, 8.0)
        assert s.font_weight == "regular"
        assert s.font_style == "normal"
        assert s.shadow is False
        assert s.shadow_offset == (1.5, -1.5)
        assert s.shadow_color == "#00000020"
        assert s.min_width is None
        assert s.min_height is None

    def test_custom_values(self):
        s = NodeStyle(
            font_weight="bold",
            font_style="italic",
            shadow=True,
            min_width=100.0,
            min_height=48.0,
        )
        assert s.font_weight == "bold"
        assert s.font_style == "italic"
        assert s.shadow is True
        assert s.min_width == 100.0
        assert s.min_height == 48.0


@pytest.mark.smoke
class TestEdgeStyleNewFields:
    """New fields on EdgeStyle (routing, label_font_size, etc.)."""

    def test_defaults(self):
        s = EdgeStyle()
        assert s.arrow_scale is None
        assert s.arrow_node_fraction == 0.0
        assert s.arrow_width_ratio == 0.7
        assert s.routing == "bezier"
        assert s.label_font_size == 7.0
        assert s.label_font_color == NEAR_BLACK
        assert s.label_background == WARM_WHITE
        assert s.label_offset == 8.0
        assert s.label_side == "auto"

    def test_custom_routing(self):
        s = EdgeStyle(routing="straight")
        assert s.routing == "straight"

    def test_custom_label_style(self):
        s = EdgeStyle(
            arrow_scale=32.0,
            arrow_node_fraction=0.4,
            arrow_width_ratio=0.65,
            label_font_size=9.0,
            label_font_color="#FF0000",
            label_offset=12.0,
            label_side="right",
        )
        assert s.arrow_scale == 32.0
        assert s.arrow_node_fraction == 0.4
        assert s.arrow_width_ratio == 0.65
        assert s.label_font_size == 9.0
        assert s.label_font_color == "#FF0000"
        assert s.label_offset == 12.0
        assert s.label_side == "right"


@pytest.mark.smoke
class TestClusterStyleNewFields:
    """New fields on ClusterStyle (font_family, label_offset, depth_*_step)."""

    def test_defaults(self):
        s = ClusterStyle()
        assert s.padding == 35.0
        assert s.font_family == ""
        assert s.label_offset == (8.0, 20.0)
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
        assert gs.margin == 18.0
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
        assert DEFAULT_THEME_OBJ.get_node_style("default").stroke_width == 0.57
        assert DEFAULT_THEME_OBJ.cluster_style.stroke_width == 0.7
        assert DEFAULT_THEME_OBJ.cluster_style.opacity == 0.32
        assert DEFAULT_THEME_OBJ.graph_style.margin == 18.0

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

    def test_graphviz_match_defaults_are_album_scoped(self) -> None:
        """Graphviz-match defaults should expose the album override values."""

        assert GRAPHVIZ_MATCH_DEFAULTS == {
            "stroke_width": 1.4,
            "padding": (7.0, 4.0),
            "font_size": 12.0,
            "arrow_length": 10.0,
            "arrow_width": 7.0,
            "arrow_scale": 16.0,
            "edge_width": 1.4,
            "edge_opacity": 1.0,
            "min_height": 22.0,
        }


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


def test_graphviz_strict_theme_loads() -> None:
    """graphviz_strict theme is available and returns correct values."""

    from dagua.styles import get_theme

    theme = get_theme("graphviz_strict")
    assert theme.name == "graphviz_strict"
    node_style = theme.get_node_style("default")
    assert node_style.shape == "ellipse"
    assert node_style.fill == "#FFFFFF"
    assert node_style.stroke == "#000000"
    assert node_style.font_size == 14.0
    assert node_style.stroke_width == 1.4
    assert node_style.font_family == "Times New Roman"
    assert node_style.padding == (12.0, 5.0)
    assert node_style.min_width == 48.0
    assert node_style.overflow_policy == "expand_node"

    edge_style = theme.get_edge_style("default")
    assert edge_style.width == 1.3
    assert edge_style.opacity == 1.0
    assert edge_style.arrow_scale is None  # ignored; mpl uses unified point conversion
    assert edge_style.arrow_length == 22.0
    assert edge_style.arrow_width == 15.0
    assert edge_style.arrow_node_fraction == pytest.approx(0.26)
    assert edge_style.arrow_width_ratio == pytest.approx(0.7)
    assert edge_style.label_font_family == "Times New Roman"

    assert theme.cluster_style.fill == "#F8F8F8"
    assert theme.cluster_style.font_size == 12.0
    assert theme.cluster_style.font_weight == "regular"
    assert theme.cluster_style.font_family == "Times New Roman"
    assert theme.cluster_style.opacity == 0.85
    assert theme.graph_style.edge_label_background_opacity == 1.0


def test_graphviz_strict_theme_io_nodes_match_default() -> None:
    """graphviz_strict input and output nodes should exactly match the default node style."""

    from dagua.styles import get_theme

    theme = get_theme("graphviz_strict")
    default_style = dataclasses.asdict(theme.get_node_style("default"))
    input_style = dataclasses.asdict(theme.get_node_style("input"))
    output_style = dataclasses.asdict(theme.get_node_style("output"))

    assert input_style == default_style
    assert output_style == default_style


def test_graphviz_improved_theme_loads() -> None:
    """graphviz improved theme is available and exposes its softer defaults."""

    from dagua.styles import get_theme

    theme = get_theme("graphviz")
    assert theme.name == "graphviz"
    node_style = theme.get_node_style("default")
    assert node_style.shape == "ellipse"
    assert node_style.font_size == 12.0
    assert node_style.stroke != "#000000"
    assert node_style.fill != "#FFFFFF"
    assert node_style.min_width == 44.0
    edge_style = theme.get_edge_style("default")
    assert edge_style.opacity == 0.92
    assert edge_style.arrow_scale is None  # ignored; mpl uses unified point conversion
    assert edge_style.arrow_length == 20.0
    assert edge_style.arrow_width == 14.0
    assert edge_style.arrow_node_fraction == pytest.approx(0.24)
    assert edge_style.arrow_width_ratio == pytest.approx(0.7)
    assert edge_style.arrow_color == "#333333"
    assert theme.get_edge_style("if").opacity == 0.92
    assert theme.get_edge_style("then").opacity == 0.92
    assert theme.get_edge_style("buffer").opacity == 0.7
    assert theme.cluster_style.stroke == "#999999"
    assert theme.cluster_style.opacity == 0.8
    assert theme.graph_style.edge_label_background_opacity == 1.0


def test_graphviz_strict_vs_improved_differences() -> None:
    """Strict and improved Graphviz themes should differ on documented fields."""

    from dagua.styles import get_theme

    strict_theme = get_theme("graphviz_strict")
    improved_theme = get_theme("graphviz")
    strict_node_style = strict_theme.get_node_style("default")
    improved_node_style = improved_theme.get_node_style("default")
    assert strict_node_style.font_family != improved_node_style.font_family
    assert strict_node_style.font_size > improved_node_style.font_size
    assert strict_node_style.stroke != improved_node_style.stroke


def test_all_themes_load() -> None:
    """Every registered theme should load without error."""

    from dagua.styles import THEME_REGISTRY, get_theme

    for name in THEME_REGISTRY:
        theme = get_theme(name)
        assert theme.name == name
