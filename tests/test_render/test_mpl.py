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
    _compute_display_scale,
    _draw_clusters,
    _draw_edge_marker,
    _edge_linestyle,
    _marker_data_size,
    _node_linestyle,
    _star_vertices,
)
from dagua.styles import ClusterStyle, EdgeStyle, NodeStyle
from dagua.utils import compute_node_size

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

    assert _node_linestyle(NodeStyle(stroke_dash="dashed")) == (0, (5.0, 3.0))
    assert _node_linestyle(NodeStyle(stroke_dash="dotted")) == (0, (0.1, 3.0))
    assert _edge_linestyle(EdgeStyle(style="dashed")) == (0, (5.0, 3.0))
    assert _edge_linestyle(EdgeStyle(style="dotted")) == (0, (0.1, 3.0))
    assert _cluster_linestyle("dotted") == (0, (0.1, 3.0))


def test_triangle_patch_uses_graphviz_like_wide_proportions() -> None:
    """Triangle nodes should fill a wide, flat bounding box."""

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

    assert width / height == pytest.approx(120.0 / 80.0)


def test_circle_arrow_marker_is_hollow() -> None:
    """Circle arrowheads should render as outline-only markers."""

    fig, ax = plt.subplots()
    style = EdgeStyle(arrow="circle", arrow_fill="filled")
    fig.canvas.draw()
    _draw_edge_marker(
        ax=ax,
        point=(0.0, 0.0),
        direction=(0.0, -1.0),
        marker="circle",
        style=style,
    )

    assert len(ax.patches) == 1
    facecolor = ax.patches[0].get_facecolor()
    edgecolor = ax.patches[0].get_edgecolor()
    expected_scale = _compute_display_scale(ax)
    plt.close(fig)

    assert facecolor[-1] == pytest.approx(0.0)
    assert edgecolor[-1] == pytest.approx(1.0)
    assert ax.patches[0].radius == pytest.approx(0.85 * style.arrow_width * expected_scale)


def test_dot_arrow_marker_uses_larger_graphviz_like_radius() -> None:
    """Dot markers should use the calibrated radius multiplier."""

    fig, ax = plt.subplots()
    style = EdgeStyle(arrow="dot", arrow_fill="filled", arrow_width=10.0)
    fig.canvas.draw()
    _draw_edge_marker(
        ax=ax,
        point=(0.0, 0.0),
        direction=(0.0, -1.0),
        marker="dot",
        style=style,
    )

    assert len(ax.patches) == 1
    expected_scale = _compute_display_scale(ax)
    plt.close(fig)

    assert ax.patches[0].radius == pytest.approx(0.55 * style.arrow_width * expected_scale)


def test_tee_arrow_marker_uses_visible_bar_offset_and_width() -> None:
    """Tee markers should render as a wide, thin bar set back from the tip."""

    from matplotlib.patches import Polygon

    fig, ax = plt.subplots()
    style = EdgeStyle(arrow="tee", width=1.2, arrow_width=10.0, arrow_length=14.0)
    fig.canvas.draw()
    _draw_edge_marker(
        ax=ax,
        point=(0.0, 0.0),
        direction=(0.0, -1.0),
        marker="tee",
        style=style,
    )

    assert len(ax.lines) == 0
    assert len(ax.patches) == 1
    assert isinstance(ax.patches[0], Polygon)
    vertices = ax.patches[0].get_xy()[:-1]
    x_span = float(vertices[:, 0].max() - vertices[:, 0].min())
    y_span = float(vertices[:, 1].max() - vertices[:, 1].min())
    expected_scale = _compute_display_scale(ax)

    assert float(np.mean(vertices[:, 1])) == pytest.approx(
        (style.arrow_length / 4.0) * expected_scale
    )
    assert x_span == pytest.approx(style.arrow_width * 2.6 * expected_scale)
    assert y_span == pytest.approx((style.arrow_length / 6.0) * 2.0 * expected_scale)
    assert ax.patches[0].get_linewidth() == pytest.approx(0.5)
    plt.close(fig)


def test_star_vertices_use_deeper_concavities() -> None:
    """Star nodes should keep a small inner radius for pronounced points."""

    vertices = _star_vertices(x=0.0, y=0.0, w=100.0, h=100.0)
    radii = np.linalg.norm(vertices, axis=1)
    outer_radius = float(radii[0])
    inner_radius = float(radii[1])

    assert inner_radius / outer_radius == pytest.approx(0.32)


def test_shape_size_adjustments_match_graphviz_calibration() -> None:
    """Graphviz-calibrated shapes should reserve the updated label bounds."""

    triangle_w, triangle_h, _ = compute_node_size("A", shape="triangle")
    star_w, star_h, _ = compute_node_size("A", shape="star")
    diamond_w, diamond_h, _ = compute_node_size("A", shape="diamond")
    ellipse_w, ellipse_h, _ = compute_node_size("", padding=(0.0, 0.0), shape="ellipse")
    # Use taller padding so the shape-specific width floors, not the global
    # minimum width, determine the final aspect ratio.
    hexagon_w, hexagon_h, _ = compute_node_size("", padding=(0.0, 10.0), shape="hexagon")
    pentagon_w, pentagon_h, _ = compute_node_size("", padding=(0.0, 10.0), shape="pentagon")
    octagon_w, octagon_h, _ = compute_node_size("", padding=(0.0, 10.0), shape="octagon")

    assert triangle_w / triangle_h == pytest.approx(3.2)
    assert star_w == pytest.approx(star_h)
    assert diamond_w / diamond_h == pytest.approx(1.4)
    assert ellipse_w == pytest.approx(36.8)
    assert ellipse_h == pytest.approx(18.0)
    assert hexagon_w / hexagon_h == pytest.approx(1.3)
    assert pentagon_w / pentagon_h == pytest.approx(1.2)
    assert octagon_w / octagon_h == pytest.approx(1.15)


def test_compute_node_size_uses_reduced_graphviz_minimums() -> None:
    """Small labels should still respect the updated Graphviz-match minima."""

    width, height, _ = compute_node_size("", padding=(0.0, 0.0))

    assert width == pytest.approx(32.0)
    assert height == pytest.approx(18.0)


def test_vee_arrow_marker_uses_wider_graphviz_spread() -> None:
    """Vee markers should use the widened wing span and heavier outline."""

    fig, ax = plt.subplots()
    style = EdgeStyle(arrow="vee", width=1.1, arrow_width=10.0, arrow_length=14.0)
    fig.canvas.draw()
    _draw_edge_marker(
        ax=ax,
        point=(0.0, 0.0),
        direction=(0.0, -1.0),
        marker="vee",
        style=style,
    )

    vertices = ax.patches[0].get_xy()
    expected_scale = _compute_display_scale(ax)
    assert abs(float(vertices[0][0] - vertices[2][0])) == pytest.approx(
        style.arrow_width * 1.4 * expected_scale
    )
    assert ax.patches[0].get_linewidth() == pytest.approx(max(style.width * 1.8, 2.0))
    plt.close(fig)


def test_crow_arrow_marker_uses_wider_graphviz_spread() -> None:
    """Crow markers should widen their outer tines and use heavier strokes."""

    fig, ax = plt.subplots()
    style = EdgeStyle(arrow="crow", width=1.1, arrow_width=10.0, arrow_length=14.0)
    fig.canvas.draw()
    _draw_edge_marker(
        ax=ax,
        point=(0.0, 0.0),
        direction=(0.0, -1.0),
        marker="crow",
        style=style,
    )

    assert len(ax.lines) == 3
    expected_scale = _compute_display_scale(ax)
    outer_x_offsets = sorted(abs(float(line.get_xdata()[1])) for line in ax.lines[1:])
    assert outer_x_offsets == pytest.approx(
        [style.arrow_width * 0.85 * expected_scale, style.arrow_width * 0.85 * expected_scale]
    )
    for line in ax.lines:
        assert line.get_linewidth() == pytest.approx(max(style.width * 1.8, 2.0))
    plt.close(fig)


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
    """Normal arrow markers should render as a filled Polygon."""

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
    """Vee arrow should render as a stroked custom head, not FancyArrowPatch."""

    import torch
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import FancyArrowPatch, Polygon

    graph = DaguaGraph()
    graph.add_node("A", label="From")
    graph.add_node("B", label="To")
    graph.add_edge("A", "B", style=EdgeStyle(arrow="vee"))
    positions = torch.tensor([[0.0, 50.0], [0.0, -50.0]])

    fig, ax = render(graph, positions)
    polygons = [patch for patch in ax.patches if isinstance(patch, Polygon)]
    fancy_arrows = [patch for patch in ax.patches if isinstance(patch, FancyArrowPatch)]
    head_collections = [
        collection
        for collection in ax.collections
        if isinstance(collection, PatchCollection) and float(collection.get_zorder()) == 2.0
    ]
    assert len(polygons) == 0, "Custom heads should not fall back to standalone Polygon patches"
    assert len(head_collections) >= 1, "Vee arrow should be rendered by the custom head collection"
    assert len(fancy_arrows) == 0, "Vee arrow should not use FancyArrowPatch"
    vee = head_collections[-1]
    assert all(width > 0.0 for width in vee.get_linewidths())
    plt.close(fig)


def test_straight_routing_has_arrowhead() -> None:
    """Straight routing must still draw arrow markers when control points collapse."""

    import torch
    from matplotlib.collections import PatchCollection

    graph = DaguaGraph()
    graph.add_node("A", label="From")
    graph.add_node("B", label="To")
    graph.add_edge("A", "B", style=EdgeStyle(routing="straight", arrow="normal"))
    positions = torch.tensor([[0.0, 50.0], [0.0, -50.0]])

    fig, ax = render(graph, positions)
    head_collections = [
        collection
        for collection in ax.collections
        if isinstance(collection, PatchCollection) and float(collection.get_zorder()) == 2.0
    ]
    assert len(head_collections) >= 1, "Straight routing should produce arrowhead collection"
    verts = head_collections[0].get_paths()[0].vertices
    if np.allclose(verts[0], verts[-1]):
        verts = verts[:-1]
    tip_y = min(vertex[1] for vertex in verts)
    base_ys = sorted(vertex[1] for vertex in verts)[1:]
    assert all(base_y > tip_y for base_y in base_ys), (
        f"Arrow tip (y={tip_y:.1f}) should be closest to target; base vertices at y={base_ys}"
    )
    plt.close(fig)


def test_arrowhead_size_scales_with_graph_range() -> None:
    """Arrowheads should be the same visual size regardless of data range."""

    style = EdgeStyle(arrow_length=10.0, arrow_width=7.0)

    fig1, ax1 = plt.subplots(figsize=(6.0, 6.0))
    ax1.set_xlim(0.0, 100.0)
    ax1.set_ylim(0.0, 100.0)
    ax1.set_aspect("equal")
    fig1.canvas.draw()
    len1, wid1 = _marker_data_size(ax1, style, style.arrow_length, style.arrow_width)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(6.0, 6.0))
    ax2.set_xlim(0.0, 10000.0)
    ax2.set_ylim(0.0, 10000.0)
    ax2.set_aspect("equal")
    fig2.canvas.draw()
    len2, wid2 = _marker_data_size(ax2, style, style.arrow_length, style.arrow_width)
    plt.close(fig2)

    ratio = len2 / len1
    assert 90.0 < ratio < 110.0, f"Expected ~100x ratio, got {ratio:.1f}x"
    assert wid2 / wid1 == pytest.approx(ratio)


def test_arrowhead_scales_with_node_height() -> None:
    """Node-relative arrowheads should be proportional to target node height."""

    style = EdgeStyle(arrow_node_fraction=0.4, arrow_width_ratio=0.7)

    length, width = _marker_data_size(None, style, 10.0, 7.0, node_height=50.0)
    assert length == pytest.approx(20.0)
    assert width == pytest.approx(14.0)

    length2, width2 = _marker_data_size(None, style, 10.0, 7.0, node_height=100.0)
    assert length2 == pytest.approx(40.0)
    assert width2 == pytest.approx(28.0)

    style_fixed = EdgeStyle(arrow_node_fraction=0.0)
    assert style_fixed.arrow_node_fraction == 0.0


def test_open_marker_uses_unified_display_scaled_dimensions() -> None:
    """Manual polygon markers should always convert point sizing into data units."""

    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=100)
    ax.set_xlim(-50.0, 50.0)
    ax.set_ylim(-50.0, 50.0)
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
    expected_scale = _compute_display_scale(ax)
    plt.close(fig)

    assert float(base_center[1]) == pytest.approx(style.arrow_length * expected_scale)
    assert abs(float(vertices[1][0] - vertices[2][0])) == pytest.approx(
        style.arrow_width * expected_scale * 1.2
    )


def test_normal_arrow_marker_uses_wider_graphviz_base() -> None:
    """Normal arrow markers should use the widened triangular base."""

    fig, ax = plt.subplots()
    style = EdgeStyle(arrow="normal", arrow_width=10.0, arrow_length=14.0)
    fig.canvas.draw()
    _draw_edge_marker(
        ax=ax,
        point=(0.0, 0.0),
        direction=(0.0, -1.0),
        marker="normal",
        style=style,
    )

    assert len(ax.patches) == 1
    vertices = ax.patches[0].get_xy()
    expected_scale = _compute_display_scale(ax)
    assert abs(float(vertices[1][0] - vertices[2][0])) == pytest.approx(
        style.arrow_width * 1.2 * expected_scale
    )
    plt.close(fig)


def test_triangle_labels_shift_toward_visual_centroid() -> None:
    """Triangle labels should sit lower than the geometric center."""

    graph = DaguaGraph()
    graph.add_node("a", label="Triangle", style=NodeStyle(shape="triangle"))
    pos = np.array([[10.0, 20.0]])
    sizes = np.array([[120.0, 60.0]])

    fig, ax = plt.subplots()
    mpl_renderer._draw_node_labels(ax, graph, pos, sizes)

    assert len(ax.texts) == 1
    assert float(ax.texts[0].get_position()[1]) == pytest.approx(10.0)
    plt.close(fig)


def test_triangle_rich_labels_shift_toward_visual_centroid() -> None:
    """Rich triangle labels should use the same centroid-aware anchor."""

    graph = DaguaGraph()
    graph.add_node(
        "a",
        label="Triangle",
        style=NodeStyle(shape="triangle", label_format="rich"),
    )
    pos = np.array([[10.0, 20.0]])
    sizes = np.array([[120.0, 60.0]])

    fig, ax = plt.subplots()
    mpl_renderer._draw_node_labels(ax, graph, pos, sizes)

    assert len(ax.texts) == 1
    assert float(ax.texts[0].get_position()[1]) == pytest.approx(10.0)
    plt.close(fig)


def test_parallelogram_patch_uses_stronger_graphviz_skew() -> None:
    """Parallelograms should use the increased shoulder skew."""

    patch = _build_node_patch(
        x=0.0,
        y=0.0,
        w=100.0,
        h=50.0,
        style=NodeStyle(shape="parallelogram"),
        facecolor="#ffffff",
        edgecolor="#000000",
        linewidth=1.0,
        linestyle="-",
        zorder=1.0,
    )
    vertices = patch.get_xy()[:-1]

    assert float(vertices[0][0]) == pytest.approx(-22.0)
    assert float(vertices[2][0]) == pytest.approx(22.0)


def test_trapezoid_patch_uses_stronger_graphviz_taper() -> None:
    """Trapezoids should match Graphviz's narrower-top, wider-bottom shape."""

    patch = _build_node_patch(
        x=0.0,
        y=0.0,
        w=100.0,
        h=50.0,
        style=NodeStyle(shape="trapezoid"),
        facecolor="#ffffff",
        edgecolor="#000000",
        linewidth=1.0,
        linestyle="-",
        zorder=1.0,
    )
    vertices = patch.get_xy()[:-1]

    assert float(vertices[0][0]) == pytest.approx(-22.0)
    assert float(vertices[1][0]) == pytest.approx(22.0)
    assert float(vertices[2][0]) == pytest.approx(50.0)
    assert float(vertices[3][0]) == pytest.approx(-50.0)


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


def test_cluster_offsets_and_corner_radius_use_display_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cluster label offsets and corner radius should be converted from points."""

    graph = DaguaGraph()
    graph.add_node("a")
    graph.add_cluster(
        "outer",
        ["a"],
        label="Cluster",
        style=ClusterStyle(
            padding=0.0,
            corner_radius=6.0,
            label_offset=(8.0, 20.0),
        ),
    )

    monkeypatch.setattr(mpl_renderer, "measure_text", lambda *args, **kwargs: (40.0, 12.0))

    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=100)
    ax.set_xlim(-50.0, 50.0)
    ax.set_ylim(-50.0, 50.0)
    fig.canvas.draw()

    _draw_clusters(
        ax=ax,
        graph=graph,
        pos=np.array([[0.0, 0.0]], dtype=float),
        sizes=np.array([[20.0, 20.0]], dtype=float),
    )

    assert len(ax.patches) == 1
    assert len(ax.texts) == 1
    display_scale = _compute_display_scale(ax)
    label_width = 40.0
    label_height = 12.0
    initial_x_min = -10.0
    initial_x_max = 10.0
    expanded_width = max(label_width + (8.0 * display_scale * 2.0), initial_x_max - initial_x_min)
    x_min = -expanded_width / 2.0
    y_max = 10.0 + max(14.0, label_height)

    assert ax.texts[0].get_position()[0] == pytest.approx(x_min + (8.0 * display_scale))
    assert ax.texts[0].get_position()[1] == pytest.approx(y_max - (20.0 * display_scale))
    assert ax.patches[0].get_boxstyle().rounding_size == pytest.approx(6.0 * display_scale)
    plt.close(fig)
