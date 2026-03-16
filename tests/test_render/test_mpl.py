"""Tests for matplotlib renderer."""

import importlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba

import dagua
from dagua.config import LayoutConfig
from dagua.graph import DaguaGraph
from dagua.layout import layout
from dagua.render import render
from dagua.render.mpl import (
    _build_node_patch,
    _cluster_linestyle,
    _draw_clusters,
    _draw_edge_marker,
    _edge_linestyle,
    _node_linestyle,
    _points_to_data_units,
)
from dagua.styles import ClusterStyle, EdgeStyle, NodeStyle

mpl_renderer = importlib.import_module("dagua.render.mpl")


class TestRenderBasic:
    @pytest.mark.slow
    def test_returns_fig_ax(self, simple_chain, fast_config):
        pos = layout(simple_chain, fast_config)
        fig, ax = render(simple_chain, pos)
        assert fig is not None
        assert ax is not None

    @pytest.mark.slow
    def test_empty_graph(self, empty_graph, fast_config):
        pos = layout(empty_graph, fast_config)
        fig, ax = render(empty_graph, pos)
        assert fig is not None

    @pytest.mark.slow
    def test_save_png(self, simple_chain, fast_config, tmp_path):
        pos = layout(simple_chain, fast_config)
        out = str(tmp_path / "test.png")
        render(simple_chain, pos, output=out)
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0

    @pytest.mark.slow
    def test_save_svg(self, simple_chain, fast_config, tmp_path):
        pos = layout(simple_chain, fast_config)
        out = str(tmp_path / "test.svg")
        render(simple_chain, pos, output=out)
        assert Path(out).exists()
        content = Path(out).read_text(encoding="utf-8")
        assert "<title>a</title>" in content or "<title>b</title>" in content

    @pytest.mark.slow
    def test_save_pdf(self, simple_chain, fast_config, tmp_path):
        pos = layout(simple_chain, fast_config)
        out = str(tmp_path / "test.pdf")
        render(simple_chain, pos, output=out)
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0

    @pytest.mark.slow
    def test_save_eps(self, simple_chain, fast_config, tmp_path):
        pos = layout(simple_chain, fast_config)
        out = str(tmp_path / "test.eps")
        render(simple_chain, pos, output=out)
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0

    @pytest.mark.slow
    def test_save_jpeg(self, simple_chain, fast_config, tmp_path):
        pytest.importorskip("PIL")
        pos = layout(simple_chain, fast_config)
        out = str(tmp_path / "test.jpg")
        render(simple_chain, pos, output=out)
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0

    @pytest.mark.slow
    def test_format_override_uses_requested_format(self, simple_chain, fast_config, tmp_path):
        pos = layout(simple_chain, fast_config)
        out = str(tmp_path / "forced.bin")
        render(simple_chain, pos, output=out, format="png")
        assert Path(out).exists()
        with open(out, "rb") as f:
            assert f.read(8) == b"\x89PNG\r\n\x1a\n"

    @pytest.mark.slow
    def test_vector_format_override_uses_requested_format(
        self, simple_chain, fast_config, tmp_path
    ):
        pos = layout(simple_chain, fast_config)
        out = str(tmp_path / "forced-vector.bin")
        render(simple_chain, pos, output=out, format="svg")
        assert Path(out).exists()
        with open(out, "rt", encoding="utf-8") as f:
            content = f.read(256)
        assert "<svg" in content or ":svg" in content

    @pytest.mark.slow
    def test_svg_hover_text_can_be_disabled(self, simple_chain, fast_config, tmp_path):
        pos = layout(simple_chain, fast_config)
        out = str(tmp_path / "no-hover.svg")
        render(simple_chain, pos, output=out, svg_hover_text=False)
        content = Path(out).read_text(encoding="utf-8")
        assert "<title>a</title>" not in content
        assert "<title>b</title>" not in content

    @pytest.mark.slow
    def test_custom_figsize(self, simple_chain, fast_config):
        pos = layout(simple_chain, fast_config)
        fig, ax = render(simple_chain, pos, figsize=(10, 8))
        w, h = fig.get_size_inches()
        assert abs(w - 10) < 0.1
        assert abs(h - 8) < 0.1

    @pytest.mark.slow
    def test_render_can_use_cached_positions(self, simple_chain, fast_config):
        layout(simple_chain, fast_config)
        fig, ax = render(simple_chain)
        assert fig is not None
        assert ax is not None

    @pytest.mark.slow
    def test_draw_relayout_false_requires_fresh_layout(self, simple_chain, fast_config):
        with pytest.raises(ValueError, match="Graph layout is missing"):
            dagua.draw(simple_chain, fast_config, relayout=False)

    @pytest.mark.slow
    def test_draw_uses_graph_direction_when_config_is_implicit(self):
        g = DaguaGraph.from_edge_list([("a", "b"), ("b", "c"), ("c", "d")], direction="LR")
        fig, ax = dagua.draw(g)
        pos = g.last_positions
        assert fig is not None
        assert ax is not None
        assert pos is not None
        x_span = float(pos[:, 0].max().item() - pos[:, 0].min().item())
        y_span = float(pos[:, 1].max().item() - pos[:, 1].min().item())
        assert x_span > y_span

    @pytest.mark.slow
    def test_draw_direction_override_wins(self):
        g = DaguaGraph.from_edge_list([("a", "b"), ("b", "c"), ("c", "d")], direction="TB")
        config = LayoutConfig(steps=60, edge_opt_steps=-1, direction="TB", seed=42)
        fig, ax = dagua.draw(g, config=config, direction="LR")
        pos = g.last_positions
        assert fig is not None
        assert ax is not None
        assert pos is not None
        x_span = float(pos[:, 0].max().item() - pos[:, 0].min().item())
        y_span = float(pos[:, 1].max().item() - pos[:, 1].min().item())
        assert x_span > y_span


class TestRenderWithClusters:
    @pytest.mark.slow
    def test_clustered_graph(self, clustered_graph, fast_config, tmp_path):
        pos = layout(clustered_graph, fast_config)
        out = str(tmp_path / "clustered.png")
        render(clustered_graph, pos, output=out)
        assert Path(out).exists()


class TestRenderEdgeLabels:
    @pytest.mark.slow
    def test_edge_labels(self, fast_config, tmp_path):
        g = DaguaGraph.from_edge_list([("a", "b"), ("b", "c")])
        g.edge_labels = ["first", "second"]
        pos = layout(g, fast_config)
        out = str(tmp_path / "labeled.png")
        render(g, pos, output=out)
        assert Path(out).exists()


class TestRenderStyleFlexibility:
    @pytest.mark.slow
    def test_dotted_border_renders(self, fast_config, tmp_path):
        g = DaguaGraph.from_edge_list([("a", "b")])
        g.node_styles[0] = NodeStyle(stroke_dash="dotted")
        pos = layout(g, fast_config)
        out = str(tmp_path / "dotted.png")
        render(g, pos, output=out)
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "shape",
        [
            "triangle",
            "hexagon",
            "pentagon",
            "octagon",
            "star",
            "parallelogram",
            "trapezoid",
            "cylinder",
        ],
    )
    def test_new_shapes_render(self, fast_config, tmp_path, shape):
        g = DaguaGraph.from_edge_list([("a", "b")])
        g.node_styles[0] = NodeStyle(shape=shape, shadow=True, gradient="linear")
        pos = layout(g, fast_config)
        out = str(tmp_path / f"{shape}.png")
        render(g, pos, output=out)
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0

    @pytest.mark.slow
    def test_rich_label_renders(self, fast_config, tmp_path):
        g = DaguaGraph.from_edge_list([("a", "b")])
        g.node_labels[0] = "**Bold** and *italic*\n{color:#FF0000}red{/color} `mono`"
        g.node_styles[0] = NodeStyle(
            label_format="rich",
            text_align="left",
            text_valign="top",
            text_outline=True,
        )
        pos = layout(g, fast_config)
        out = str(tmp_path / "rich.png")
        render(g, pos, output=out)
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "arrow",
        ["normal", "vee", "dot", "diamond", "tee", "crow", "circle", "open", "none"],
    )
    def test_arrow_types_render(self, fast_config, tmp_path, arrow):
        g = DaguaGraph.from_edge_list([("a", "b")])
        g.edge_styles[0] = EdgeStyle(arrow=arrow, tail_arrow="diamond", arrow_fill="hollow")
        pos = layout(g, fast_config)
        out = str(tmp_path / f"{arrow}.png")
        render(g, pos, output=out)
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0

    @pytest.mark.slow
    def test_edge_label_font_fields_render(self, fast_config, tmp_path):
        g = DaguaGraph.from_edge_list([("a", "b")])
        g.edge_labels = ["edge label"]
        g.edge_styles[0] = EdgeStyle(
            label_font_family="DejaVu Sans",
            label_font_weight="bold",
        )
        pos = layout(g, fast_config)
        out = str(tmp_path / "edge-label-fonts.png")
        render(g, pos, output=out)
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0


def test_graphviz_dash_patterns_are_explicit() -> None:
    """Renderer dash mappings should use Graphviz-like stroke lengths."""

    assert _node_linestyle(NodeStyle(stroke_dash="dashed")) == (0, (6.0, 4.0))
    assert _node_linestyle(NodeStyle(stroke_dash="dotted")) == (0, (1.5, 2.5))
    assert _edge_linestyle(EdgeStyle(style="dashed")) == (0, (6.0, 4.0))
    assert _edge_linestyle(EdgeStyle(style="dotted")) == (0, (1.5, 2.5))
    assert _cluster_linestyle("dotted") == (0, (1.5, 2.5))


def test_triangle_patch_uses_equilateral_proportions() -> None:
    """Triangle nodes should be close to equilateral instead of flat and wide."""

    patch = _build_node_patch(
        x=0.0,
        y=0.0,
        w=120.0,
        h=80.0,
        style=NodeStyle(shape="triangle"),
        facecolor="#ffffff",
        edgecolor="#000000",
        linewidth=1.0,
        linestyle="-",
        zorder=1.0,
    )
    vertices = patch.get_xy()
    content = vertices[:-1]
    width = float(content[:, 0].max() - content[:, 0].min())
    height = float(content[:, 1].max() - content[:, 1].min())

    assert height / width == pytest.approx(3**0.5 / 2.0, rel=0.05)


def test_vee_arrow_marker_is_unfilled() -> None:
    """Vee arrowheads should render as an open marker, not a filled triangle."""

    fig, ax = plt.subplots()
    _draw_edge_marker(
        ax=ax,
        point=(0.0, 0.0),
        direction=(0.0, -1.0),
        marker="vee",
        style=EdgeStyle(arrow="vee", arrow_width=10.0, arrow_length=14.0),
    )

    assert len(ax.patches) == 1
    facecolor = ax.patches[0].get_facecolor()
    plt.close(fig)

    assert facecolor[-1] == pytest.approx(0.0)


def test_filled_arrow_marker_uses_opaque_edge_color() -> None:
    """Filled arrowheads should use the edge color at full opacity."""

    fig, ax = plt.subplots()
    style = EdgeStyle(color="#445566", opacity=0.35, arrow_width=12.0, arrow_length=18.0)
    _draw_edge_marker(
        ax=ax,
        point=(0.0, 0.0),
        direction=(0.0, -1.0),
        marker="normal",
        style=style,
    )

    assert len(ax.patches) == 1
    facecolor = ax.patches[0].get_facecolor()
    edgecolor = ax.patches[0].get_edgecolor()
    plt.close(fig)

    assert facecolor == pytest.approx(to_rgba(style.color, 1.0))
    assert edgecolor == pytest.approx(to_rgba(style.color, 1.0))


def test_normal_arrow_renders_polygon_with_arrow_scale() -> None:
    """Normal arrow markers should render as a filled Polygon, not FancyArrowPatch."""

    fig, ax = plt.subplots()
    style = EdgeStyle(arrow="normal", arrow_width=12.0, arrow_length=18.0, arrow_scale=32.0)
    _draw_edge_marker(
        ax=ax,
        point=(0.0, 0.0),
        direction=(0.0, 1.0),
        marker="normal",
        style=style,
    )

    assert len(ax.patches) == 1
    from matplotlib.patches import Polygon

    assert isinstance(ax.patches[0], Polygon)
    plt.close(fig)


def test_vee_arrow_is_open_polygon() -> None:
    """Vee arrow should render as an open Polygon, not FancyArrowPatch."""

    import torch
    from matplotlib.patches import FancyArrowPatch, Polygon

    graph = DaguaGraph()
    graph.add_node("A", label="From")
    graph.add_node("B", label="To")
    graph.add_edge("A", "B", style=EdgeStyle(arrow="vee"))
    positions = torch.tensor([[0.0, 50.0], [0.0, -50.0]])

    fig, ax = render(graph, positions)
    polygons = [patch for patch in ax.patches if isinstance(patch, Polygon)]
    fancy_arrows = [patch for patch in ax.patches if isinstance(patch, FancyArrowPatch)]
    plt.close(fig)

    assert len(polygons) >= 1, "Vee arrow should be a Polygon"
    assert len(fancy_arrows) == 0, "Vee arrow should not use FancyArrowPatch"


def test_straight_routing_has_arrowhead() -> None:
    """Straight routing must still draw arrow markers when control points collapse."""

    import torch
    from matplotlib.patches import Polygon

    graph = DaguaGraph()
    graph.add_node("A", label="From")
    graph.add_node("B", label="To")
    graph.add_edge("A", "B", style=EdgeStyle(routing="straight", arrow="normal"))
    positions = torch.tensor([[0.0, 50.0], [0.0, -50.0]])

    fig, ax = render(graph, positions)
    polygons = [patch for patch in ax.patches if isinstance(patch, Polygon)]
    plt.close(fig)

    assert polygons, "Straight routing should have at least one arrow polygon"


def test_open_marker_preserves_legacy_data_sizing_without_arrow_scale() -> None:
    """Manual markers should keep legacy data-space sizing when arrow_scale is absent."""

    fig, ax = plt.subplots()
    style = EdgeStyle(arrow="open", arrow_width=12.0, arrow_length=18.0)
    _draw_edge_marker(
        ax=ax,
        point=(0.0, 0.0),
        direction=(0.0, -1.0),
        marker="open",
        style=style,
    )

    assert len(ax.patches) == 1
    vertices = ax.patches[0].get_xy()
    base_center = vertices[1:3].mean(axis=0)
    plt.close(fig)

    assert float(base_center[1]) == pytest.approx(style.arrow_length)
    assert abs(float(vertices[1][0] - vertices[2][0])) == pytest.approx(style.arrow_width)


def test_open_marker_uses_display_scaled_dimensions_with_arrow_scale() -> None:
    """Manual polygon markers should convert display sizing back into data units."""

    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=100)
    ax.set_xlim(-50.0, 50.0)
    ax.set_ylim(-50.0, 50.0)
    style = EdgeStyle(arrow="open", arrow_width=12.0, arrow_length=18.0, arrow_scale=24.0)
    _draw_edge_marker(
        ax=ax,
        point=(0.0, 0.0),
        direction=(0.0, -1.0),
        marker="open",
        style=style,
    )

    assert len(ax.patches) == 1
    vertices = ax.patches[0].get_xy()
    base_center = vertices[1:3].mean(axis=0)
    expected_scale = (
        _points_to_data_units(ax, 1.0, "x") + _points_to_data_units(ax, 1.0, "y")
    ) / 2.0
    plt.close(fig)

    assert float(base_center[1]) == pytest.approx(style.arrow_scale * expected_scale)
    assert abs(float(vertices[1][0] - vertices[2][0])) == pytest.approx(
        style.arrow_width * expected_scale
    )


def test_cluster_labels_expand_bbox_using_measured_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cluster boxes should widen to fit measured label text."""

    graph = DaguaGraph()
    graph.add_node("a")
    graph.add_cluster(
        "outer",
        ["a"],
        label="Wide",
        style=ClusterStyle(
            padding=0.0,
            font_size=20.0,
            label_offset=(0.0, 20.0),
        ),
    )

    monkeypatch.setattr(mpl_renderer, "measure_text", lambda *args, **kwargs: (80.0, 20.0))

    fig, ax = plt.subplots()
    _draw_clusters(
        ax=ax,
        graph=graph,
        pos=np.array([[0.0, 0.0]], dtype=float),
        sizes=np.array([[70.0, 20.0]], dtype=float),
    )

    assert len(ax.patches) == 1
    assert ax.patches[0].get_width() == pytest.approx(80.0)
    assert len(ax.texts) == 1
    assert ax.texts[0].get_clip_on() is False
    plt.close(fig)
