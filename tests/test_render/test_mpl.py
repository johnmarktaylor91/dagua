"""Tests for matplotlib renderer."""

import colorsys
import importlib
from pathlib import Path
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from matplotlib.colors import to_rgba
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

import dagua
import dagua.utils as dagua_utils
from dagua.config import LayoutConfig
from dagua.edges import BezierCurve, route_edges
from dagua.graph import DaguaGraph
from dagua.layout import layout
from dagua.render import render
from dagua.render.borders import ShapeSpec, build_shape_path, make_clip_proxy
from dagua.render.crossings import EdgeCrossing
from dagua.render.edges.arrowheads import build_arrowhead
from dagua.render.mpl import (
    _build_custom_edge_collection,
    _build_node_patch,
    _cluster_linestyle,
    _compute_display_scale,
    _crossing_span_data_units,
    _draw_clusters,
    _draw_edge_marker,
    _draw_edges_direct,
    _draw_node_fill,
    _draw_pie_fill,
    _draw_sharp_crossing,
    _edge_linestyle,
    _edge_width_data_units,
    _marker_data_size,
    _node_linestyle,
    _star_vertices,
    _trim_curve_for_arrows,
)
from dagua.render.text import layout_plain_text
from dagua.styles import RESOLVED_FONT, ClusterStyle, EdgeStyle, NodeStyle
from dagua.utils import compute_node_size, prepare_label_text
from scripts.generate_cosmetic_album import build_case_catalog

mpl_renderer = importlib.import_module("dagua.render.mpl")


def _rgba_luminance(color: tuple[float, float, float, float]) -> float:
    """Return relative luminance for an RGBA color tuple.

    Parameters
    ----------
    color : tuple[float, float, float, float]
        RGBA color on the ``[0, 1]`` scale.

    Returns
    -------
    float
        Relative luminance on the ``[0, 1]`` scale.
    """
    red, green, blue, _ = color
    return (0.2126 * red) + (0.7152 * green) + (0.0722 * blue)


def _label_bbox(ax: Any, prefix: str) -> tuple[float, float, float, float]:
    """Return the data-coordinate bbox for label patches with the given gid prefix."""
    patches = [
        patch
        for patch in ax.patches
        if isinstance(patch, PathPatch)
        and isinstance(patch.get_gid(), str)
        and patch.get_gid().startswith(prefix)
    ]
    assert patches
    vertices = np.concatenate([patch.get_path().vertices for patch in patches], axis=0)
    return (
        float(vertices[:, 0].min()),
        float(vertices[:, 0].max()),
        float(vertices[:, 1].min()),
        float(vertices[:, 1].max()),
    )


def _expected_plain_label_bbox(
    ax: Any,
    text: str,
    font_size: float,
    font_family: str,
    font_weight: str,
    ha: str,
    va: str,
    anchor_x: float,
    anchor_y: float,
) -> tuple[float, float, float, float]:
    """Return the expected bbox for a plain text block at a given anchor."""
    display_scale = _compute_display_scale(ax)
    block = layout_plain_text(
        text,
        size_data=font_size * display_scale,
        ha=ha,
        va=va,
        font_family=font_family,
        font_weight=font_weight,
        font_style="normal",
        line_spacing=1.2,
        secondary_scale=1.0,
    )
    vertices = []
    for line in block.lines:
        for segment in line.segments:
            if segment.glyph_run.path.vertices.size == 0:
                continue
            shifted = segment.glyph_run.path.vertices + np.array(
                [
                    anchor_x + block.x_offset + segment.x_offset,
                    anchor_y + block.y_offset + line.baseline_y,
                ]
            )
            vertices.append(shifted)
    assert vertices
    merged = np.concatenate(vertices, axis=0)
    return (
        float(merged[:, 0].min()),
        float(merged[:, 0].max()),
        float(merged[:, 1].min()),
        float(merged[:, 1].max()),
    )


def _capture_render_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> List[Tuple[List[Any], float]]:
    """Capture ``render_text`` calls while preserving normal rendering.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture.

    Returns
    -------
    list[tuple[list[Any], float]]
        Captured ``(specs, display_scale)`` pairs for each ``render_text`` call.
    """
    captured: List[Tuple[List[Any], float]] = []
    original_render_text = mpl_renderer.render_text

    def _wrapped_render_text(
        ax: Any,
        specs: list[Any],
        display_scale: float,
        svg_hover_map: Any,
    ) -> Any:
        """Record render-text inputs before delegating to the real renderer."""
        captured.append((list(specs), display_scale))
        return original_render_text(ax, specs, display_scale, svg_hover_map)

    monkeypatch.setattr(mpl_renderer, "render_text", _wrapped_render_text)
    return captured


def test_dark_background_auto_contrasts_builtin_graphviz_defaults() -> None:
    """Renderer should adapt built-in light defaults when only the background turns dark."""

    graph = DaguaGraph.from_edge_list([("a", "b")])
    graph.graph_style.background_color = "#1A1A2E"
    graph.node_types = ["input", "output"]
    positions = torch.tensor([[0.0, 50.0], [0.0, -50.0]], dtype=torch.float32)

    baseline_node = graph.get_style_for_node(0)
    baseline_edge = graph.get_style_for_edge(0)
    render_node = mpl_renderer._node_style_for_render(graph, 0)
    render_edge = mpl_renderer._edge_style_for_render(graph, 0)

    assert render_node.fill == baseline_node.fill
    assert render_node.stroke != baseline_node.stroke
    assert render_edge.color != baseline_edge.color
    assert _rgba_luminance(to_rgba(render_node.fill)) >= 0.85
    assert _rgba_luminance(to_rgba(render_node.stroke)) >= 0.69
    assert _rgba_luminance(to_rgba(render_edge.color)) >= 0.69
    assert not np.allclose(to_rgba(render_node.fill), to_rgba("#FFFFFF"))
    assert not np.allclose(to_rgba(render_node.stroke), to_rgba("#FFFFFF"))
    assert render_node.font_color == "#f5f5f5"
    assert graph.get_style_for_node(0).fill == baseline_node.fill
    assert graph.get_style_for_node(0).stroke == baseline_node.stroke
    assert graph.get_style_for_edge(0).color == baseline_edge.color

    fig, ax = render(graph, positions)
    facecolors = [
        color
        for collection in ax.collections
        if hasattr(collection, "get_facecolors")
        for color in collection.get_facecolors()
    ]
    assert any(np.allclose(color, to_rgba(render_node.fill, 1.0)) for color in facecolors)
    assert any(np.allclose(color, to_rgba(render_node.stroke, 1.0)) for color in facecolors)
    assert any(
        np.allclose(color, to_rgba(render_edge.color, float(render_edge.opacity)))
        for color in facecolors
    )
    plt.close(fig)


def test_adapt_color_for_dark_bg_preserves_hue() -> None:
    """Dark-background adaptation should raise luminance without washing out the tint."""

    original_rgb = to_rgba("#2D6A2D")
    adapted = mpl_renderer._adapt_color_for_dark_bg("#2D6A2D", target_luminance=0.7)
    adapted_rgb = to_rgba(adapted)

    original_hue, _, original_saturation = colorsys.rgb_to_hls(*original_rgb[:3])
    adapted_hue, _, adapted_saturation = colorsys.rgb_to_hls(*adapted_rgb[:3])

    assert _rgba_luminance(adapted_rgb) >= 0.69
    assert adapted_hue == pytest.approx(original_hue, abs=1e-3)
    assert adapted_saturation == pytest.approx(original_saturation, abs=0.05)
    assert not np.allclose(adapted_rgb, to_rgba("#FFFFFF"))


def test_dark_background_keeps_explicit_user_colors() -> None:
    """Renderer-side auto-contrast should not override explicit node or edge colors."""

    graph = DaguaGraph.from_edge_list([("a", "b")])
    graph.graph_style.background_color = "#1A1A2E"
    graph.node_styles = [
        NodeStyle(fill="#FFFFFF", stroke="#FFFFFF", font_color="#FFFFFF"),
        NodeStyle(fill="#FFFFFF", stroke="#FFFFFF", font_color="#FFFFFF"),
    ]
    graph.edge_styles = [
        EdgeStyle(
            color="#FFFFFF",
            arrow_color="#FFFFFF",
            label_font_color="#FFFFFF",
            label_background="#111111",
        )
    ]

    render_node = mpl_renderer._node_style_for_render(graph, 0)
    render_edge = mpl_renderer._edge_style_for_render(graph, 0)

    assert render_node.fill == "#FFFFFF"
    assert render_node.stroke == "#FFFFFF"
    assert render_node.font_color == "#FFFFFF"
    assert render_edge.color == "#FFFFFF"
    assert render_edge.arrow_color == "#FFFFFF"
    assert render_edge.label_font_color == "#FFFFFF"
    assert render_edge.label_background == "#111111"


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


@pytest.mark.parametrize(("shape", "min_vertices"), [("cloud", 18), ("stadium", 14)])
def test_shape_shadows_follow_custom_node_contours(shape: str, min_vertices: int) -> None:
    """Shadow artists should reuse the rendered shape path for non-rect nodes."""

    graph = DaguaGraph()
    graph.add_node("A", label="", style=NodeStyle(shape=shape, shadow=True))
    graph.compute_node_sizes()
    positions = torch.tensor([[0.0, 0.0]], dtype=torch.float32)

    fig, ax = render(graph, positions=positions, show=False)
    shadow_patches = [
        patch
        for patch in ax.patches
        if isinstance(patch, PathPatch) and float(patch.get_zorder()) < 1.5
    ]

    assert shadow_patches
    assert shadow_patches[0].get_path().vertices.shape[0] >= min_vertices
    plt.close(fig)


def test_hatched_nodes_render_visible_overlay() -> None:
    """Hatched fills should add a contrasting hatch overlay on top of the base fill."""

    graph = DaguaGraph()
    graph.add_node("A", label="", style=NodeStyle(fill_pattern="hatched", fill="#E8EEF6"))
    graph.compute_node_sizes()
    positions = torch.tensor([[0.0, 0.0]], dtype=torch.float32)

    fig, ax = render(graph, positions=positions, show=False)
    hatch_patches = [
        patch
        for patch in ax.patches
        if isinstance(patch, PathPatch) and patch.get_hatch() == "////"
    ]

    assert hatch_patches
    hatch_patch = hatch_patches[0]
    assert float(hatch_patch.get_linewidth()) >= 0.8
    assert to_rgba(hatch_patch.get_edgecolor()) != to_rgba(hatch_patch.get_facecolor())
    plt.close(fig)


def test_crossing_span_uses_visible_minimum() -> None:
    """Crossing jumps should keep a large enough data-space span for combo cards."""

    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=100)
    ax.set_xlim(-20.0, 20.0)
    ax.set_ylim(-20.0, 20.0)
    fig.canvas.draw()

    style = EdgeStyle(crossing_style="gap", crossing_size=6.0, width=1.0)
    span = _crossing_span_data_units(ax, style)

    assert span >= 16.0
    plt.close(fig)


def test_sharp_crossing_uses_edge_width_relative_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sharp crossings should stay centered and scale directly with edge width."""

    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=100)
    ax.set_xlim(-25.0, 25.0)
    ax.set_ylim(-25.0, 25.0)
    ax.set_aspect("equal")
    fig.canvas.draw()

    captured: dict[str, Any] = {}

    def _capture_ribbon(
        ax: Any,
        points: np.ndarray,
        width: float,
        color: Any,
        zorder: float,
        cap_start: str = "round",
        cap_end: str = "round",
        join_style: str = "miter",
    ) -> None:
        """Capture the bridge polyline without drawing a real patch."""
        del ax, color, zorder, cap_start, cap_end, join_style
        captured["points"] = points
        captured["width"] = width

    monkeypatch.setattr(mpl_renderer, "_add_filled_ribbon_patch", _capture_ribbon)

    crossing = EdgeCrossing(edge_a=0, edge_b=1, x=0.0, y=0.0, t_a=0.5, t_b=0.5)
    curve = BezierCurve(
        p0=(-20.0, 0.0),
        cp1=(-10.0, 0.0),
        cp2=(10.0, 0.0),
        p1=(20.0, 0.0),
    )
    style = EdgeStyle(crossing_style="sharp", width=2.0)
    _draw_sharp_crossing(ax, crossing, curve, 0.5, style, span=16.0)

    edge_width = _edge_width_data_units(ax, float(style.width))
    expected_half_span = edge_width * 1.5
    expected_height = edge_width * 2.0
    points = captured["points"]

    assert captured["width"] == pytest.approx(edge_width)
    assert points[0][0] == pytest.approx(-expected_half_span)
    assert points[0][1] == pytest.approx(0.0)
    assert points[1][0] == pytest.approx(0.0)
    assert points[1][1] == pytest.approx(expected_height)
    assert points[2][0] == pytest.approx(expected_half_span)
    assert points[2][1] == pytest.approx(0.0)
    plt.close(fig)


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


def test_pie_fill_uses_full_ellipse_bounds() -> None:
    """Pie fills should span the node ellipse instead of the smaller axis."""

    fig, ax = plt.subplots()
    shape_spec = ShapeSpec(
        center_x=15.0,
        center_y=-8.0,
        width=120.0,
        height=60.0,
        shape="ellipse",
    )
    clip_patch = make_clip_proxy(build_shape_path(shape_spec), ax.transData)
    style = NodeStyle(
        fill_pattern="pie",
        fill_pattern_colors=["#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0"],
        fill_pattern_values=[1.0, 1.0, 1.0, 1.0],
    )

    _draw_pie_fill(ax, shape_spec, style, clip_patch)

    pie_patches = [patch for patch in ax.patches if isinstance(patch, PathPatch)]
    assert len(pie_patches) == 4
    assert all(patch.get_clip_path() is not None for patch in pie_patches)

    vertices = np.concatenate([patch.get_path().vertices for patch in pie_patches], axis=0)
    assert float(vertices[:, 0].min()) == pytest.approx(
        shape_spec.center_x - shape_spec.width / 2.0
    )
    assert float(vertices[:, 0].max()) == pytest.approx(
        shape_spec.center_x + shape_spec.width / 2.0
    )
    assert float(vertices[:, 1].min()) == pytest.approx(
        shape_spec.center_y - shape_spec.height / 2.0
    )
    assert float(vertices[:, 1].max()) == pytest.approx(
        shape_spec.center_y + shape_spec.height / 2.0
    )
    assert float(vertices[:, 0].mean()) == pytest.approx(shape_spec.center_x, abs=1.5)
    assert float(vertices[:, 1].mean()) == pytest.approx(shape_spec.center_y, abs=1.5)

    plt.close(fig)


def test_pie_fill_gradient_renders_overlay_after_wedges() -> None:
    """Pie fills should keep a visible gradient overlay when both styles are enabled."""

    fig, ax = plt.subplots()
    shape_spec = ShapeSpec(
        center_x=0.0,
        center_y=0.0,
        width=120.0,
        height=60.0,
        shape="ellipse",
    )
    fill_path = build_shape_path(shape_spec)
    clip_patch = make_clip_proxy(fill_path, ax.transData)
    style = NodeStyle(
        fill="#DCEBFA",
        gradient="linear",
        gradient_color="#FF9800",
        fill_pattern="pie",
        fill_pattern_colors=["#FF6384", "#36A2EB", "#FFCE56"],
        fill_pattern_values=[2.0, 1.0, 1.0],
    )

    _draw_node_fill(
        ax,
        shape_spec,
        fill_path,
        clip_patch,
        0.0,
        0.0,
        120.0,
        60.0,
        style,
        to_rgba(style.fill, style.opacity),
    )

    assert len(ax.images) == 1
    pie_patches = [patch for patch in ax.patches if isinstance(patch, PathPatch)]
    assert len(pie_patches) >= 4
    assert ax.images[0].get_zorder() > max(float(patch.get_zorder()) for patch in pie_patches)
    plt.close(fig)


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
    vertices = ax.patches[0].get_xy()
    x_span = float(vertices[:, 0].max() - vertices[:, 0].min())
    expected_scale = _compute_display_scale(ax)
    plt.close(fig)

    assert ax.patches[0].get_linewidth() == pytest.approx(0.0)
    assert x_span / 2.0 == pytest.approx(0.85 * style.arrow_width * expected_scale, rel=0.05)


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
    assert x_span == pytest.approx(style.arrow_width * 2.9 * expected_scale)
    assert y_span == pytest.approx((style.arrow_length / 5.0) * 2.0 * expected_scale)
    assert ax.patches[0].get_linewidth() == pytest.approx(0.0)
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
    assert star_w == pytest.approx(star_h, rel=0.1)
    assert diamond_w >= 70.4
    assert diamond_w > diamond_h
    assert ellipse_w == pytest.approx(41.6)
    assert ellipse_h == pytest.approx(18.0)
    assert hexagon_w / hexagon_h == pytest.approx(1.3)
    assert pentagon_w / pentagon_h == pytest.approx(1.2)
    assert octagon_w / octagon_h == pytest.approx(1.15)


@pytest.mark.parametrize(
    ("shape", "expected_width"),
    [
        ("hexagon", (80.0 + 22.0) / 0.866),
        ("tab", (80.0 + 22.0) * 1.25),
        ("pentagon", (80.0 + 22.0) * 1.15),
        ("octagon", (80.0 + 22.0) * 1.15),
        ("parallelogram", (80.0 + 22.0) * 1.6),
        ("trapezoid", (80.0 + 22.0) * 1.5),
    ],
)
def test_non_rectangular_shapes_expand_for_inscribed_label_box(
    monkeypatch: pytest.MonkeyPatch,
    shape: str,
    expected_width: float,
) -> None:
    """Non-rectangular shapes should reserve the reduced central label area."""

    monkeypatch.setattr(dagua_utils, "measure_text", lambda *args, **kwargs: (80.0, 20.0))
    dagua_utils._compute_node_size_cached.cache_clear()

    try:
        width, _height, _font_size = compute_node_size("shape-aware", shape=shape)
    finally:
        dagua_utils._compute_node_size_cached.cache_clear()

    assert width == pytest.approx(expected_width)


def test_triangle_shape_reserves_taller_lower_body_for_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Triangle sizing should account for the narrower upper half of the shape."""

    monkeypatch.setattr(dagua_utils, "measure_text", lambda *args, **kwargs: (80.0, 20.0))
    dagua_utils._compute_node_size_cached.cache_clear()

    try:
        width, height, _font_size = compute_node_size("shape-aware", shape="triangle")
    finally:
        dagua_utils._compute_node_size_cached.cache_clear()

    assert height == pytest.approx((20.0 + 18.0) * 2.4)
    assert width == pytest.approx(max((80.0 + 22.0) * 2.8, height * 3.2))


def test_star_shape_reserves_half_size_center_box(monkeypatch: pytest.MonkeyPatch) -> None:
    """Star nodes should keep at least a half-size central box for labels."""

    monkeypatch.setattr(dagua_utils, "measure_text", lambda *args, **kwargs: (80.0, 20.0))
    dagua_utils._compute_node_size_cached.cache_clear()

    try:
        width, height, _font_size = compute_node_size("shape-aware", shape="star")
    finally:
        dagua_utils._compute_node_size_cached.cache_clear()

    assert width >= (80.0 + 22.0) * 3.5
    assert height >= (20.0 + 18.0) * 3.5


def test_box3d_shape_reserves_separate_front_face_label_box(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Box3d sizing should reserve the front face instead of the full silhouette."""

    monkeypatch.setattr(dagua_utils, "measure_text", lambda *args, **kwargs: (80.0, 20.0))
    dagua_utils._compute_node_size_cached.cache_clear()

    try:
        width, height, _font_size = compute_node_size("shape-aware", shape="box3d")
    finally:
        dagua_utils._compute_node_size_cached.cache_clear()

    assert width >= (80.0 + 22.0) * 1.6
    assert height >= (20.0 + 18.0) * 1.5


def test_prepare_label_text_keeps_visible_prefix_for_tight_ellipsis() -> None:
    """Ellipsis truncation should preserve a readable text prefix."""

    assert (
        prepare_label_text(
            "truncate me now please",
            font_size=10.0,
            text_wrap="ellipsis",
            text_max_width=18.0,
        )
        == "trunc..."
    )


def test_diamond_shape_preserves_padded_label_box(monkeypatch: pytest.MonkeyPatch) -> None:
    """Diamond sizing should still exceed the padded label box after calibration."""

    monkeypatch.setattr(dagua_utils, "measure_text", lambda *args, **kwargs: (80.0, 20.0))
    dagua_utils._compute_node_size_cached.cache_clear()

    try:
        width, height, _font_size = compute_node_size("shape-aware", shape="diamond")
    finally:
        dagua_utils._compute_node_size_cached.cache_clear()

    assert width >= 80.0 + 22.0
    assert height >= 20.0 + 18.0
    assert width > height


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

    vertices = np.concatenate([patch.get_path().vertices for patch in ax.patches], axis=0)
    expected_scale = _compute_display_scale(ax)
    expected_span = (style.arrow_width * 1.4 * expected_scale) + _edge_width_data_units(
        ax,
        max(style.width * 1.8, 2.0),
    )
    assert float(vertices[:, 0].max() - vertices[:, 0].min()) == pytest.approx(
        expected_span,
        rel=0.05,
    )
    assert all(patch.get_linewidth() == pytest.approx(0.0) for patch in ax.patches)
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

    assert len(ax.lines) == 0
    expected_scale = _compute_display_scale(ax)
    endpoints = []
    for patch in ax.patches:
        vertices = patch.get_path().vertices
        endpoint = vertices[np.argmax(np.linalg.norm(vertices, axis=1))]
        endpoints.append(endpoint)
    outer_x_offsets = sorted(abs(float(endpoint[0])) for endpoint in endpoints[1:])
    expected_offset = (style.arrow_width * 1.7 * expected_scale) + (
        _edge_width_data_units(ax, max(style.width * 1.8, 2.0)) / 2.0
    )
    assert outer_x_offsets == pytest.approx(
        [expected_offset, expected_offset],
        rel=0.06,
    )
    assert all(patch.get_linewidth() == pytest.approx(0.0) for patch in ax.patches)
    plt.close(fig)


def test_vee_arrow_marker_is_unfilled() -> None:
    """Vee arrowheads should render as ribbon outlines instead of a filled polygon."""

    fig, ax = plt.subplots()
    _draw_edge_marker(
        ax=ax,
        point=(0.0, 0.0),
        direction=(0.0, -1.0),
        marker="vee",
        style=EdgeStyle(arrow="vee", arrow_width=10.0, arrow_length=14.0),
    )

    assert len(ax.patches) == 2
    plt.close(fig)

    assert all(patch.get_facecolor()[-1] == pytest.approx(1.0) for patch in ax.patches)


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
    assert edgecolor[-1] == pytest.approx(0.0)


def test_double_circle_path_returns_outer_ellipse() -> None:
    """Double-circle path should return the outer ellipse only.

    The inner ring is drawn as a separate stroke-only element by the
    renderer so that text is not hidden behind an inner fill.
    """

    shape_spec = ShapeSpec(
        center_x=0.0,
        center_y=0.0,
        width=120.0,
        height=80.0,
        shape="double_circle",
    )
    path = build_shape_path(shape_spec)
    move_indices = np.where(path.codes == MplPath.MOVETO)[0]

    # Only one subpath (the outer ellipse).
    assert move_indices.tolist() == [0]

    half_width = float(np.max(np.abs(path.vertices[:, 0] - shape_spec.center_x)))
    half_height = float(np.max(np.abs(path.vertices[:, 1] - shape_spec.center_y)))
    assert half_width == pytest.approx(shape_spec.width / 2.0)
    assert half_height == pytest.approx(shape_spec.height / 2.0)


def test_box3d_path_keeps_front_face_clear_of_diagonal_overlap() -> None:
    """Box3d outlines should keep the front face as a clean rectangle."""

    shape_spec = ShapeSpec(center_x=0.0, center_y=0.0, width=120.0, height=80.0, shape="box3d")
    path = build_shape_path(shape_spec)
    vertices = np.asarray(path.vertices, dtype=float)
    half_width = shape_spec.width / 2.0
    half_height = shape_spec.height / 2.0
    depth = min(half_width, half_height) * 0.25
    front_right = half_width - depth
    front_top = half_height - (depth * 0.70)

    assert any(np.allclose(vertex, (-half_width, front_top)) for vertex in vertices)
    assert any(np.allclose(vertex, (front_right, -half_height)) for vertex in vertices)
    assert any(np.allclose(vertex, (front_right, front_top)) for vertex in vertices)


def test_render_uses_resolved_font_for_default_italic_node_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Italic node labels should use the installed resolved font family."""

    captured = _capture_render_calls(monkeypatch)
    graph = DaguaGraph()
    graph.add_node("A", label="Italic", style=NodeStyle(font_style="italic"))
    graph.compute_node_sizes()
    graph.cache_layout(torch.tensor([[0.0, 0.0]], dtype=torch.float32))

    fig, _ax = render(graph, show=False)
    plt.close(fig)

    node_label_spec = next(
        spec
        for specs, _display_scale in captured
        for spec in specs
        if spec.gid == "dagua-node-label-0"
    )

    assert node_label_spec.font_style == "italic"
    assert node_label_spec.font_family == RESOLVED_FONT


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


def test_direct_render_trims_edge_body_before_arrowhead() -> None:
    """Direct rendering should stop the body at the arrowhead base."""

    from matplotlib.patches import Polygon

    graph = DaguaGraph.from_edge_list([("A", "B"), ("B", "C")], direction="TB")
    graph.edge_styles = [
        EdgeStyle(arrow="normal", crossing_style="arc"),
        EdgeStyle(arrow="normal", crossing_style="arc"),
    ]
    graph.compute_node_sizes()
    positions = torch.tensor([[0.0, 90.0], [0.0, 0.0], [0.0, -90.0]], dtype=torch.float32)
    half_heights = [float(graph.node_sizes[index, 1]) / 2.0 for index in range(graph.num_nodes)]
    curves = [
        BezierCurve(
            p0=(0.0, float(positions[0, 1]) - half_heights[0]),
            cp1=(0.0, 60.0),
            cp2=(0.0, 30.0),
            p1=(0.0, float(positions[1, 1]) + half_heights[1]),
        ),
        BezierCurve(
            p0=(0.0, float(positions[1, 1]) - half_heights[1]),
            cp1=(0.0, -30.0),
            cp2=(0.0, -60.0),
            p1=(0.0, float(positions[2, 1]) + half_heights[2]),
        ),
    ]

    fig, ax = render(graph, positions=positions, curves=curves)
    body_patches = [
        patch
        for patch in ax.patches
        if isinstance(patch, PathPatch) and abs(float(patch.get_zorder()) - 1.0) < 1e-6
    ]
    marker_patches = [
        patch
        for patch in ax.patches
        if isinstance(patch, Polygon) and abs(float(patch.get_zorder()) - 3.0) < 1e-6
    ]

    assert len(body_patches) == 2
    assert len(marker_patches) == 2

    trimmed_curve = _trim_curve_for_arrows(ax, curves[0], graph.get_style_for_edge(0), graph, 0)
    body_vertices = body_patches[0].get_path().vertices
    assert float(body_vertices[:, 1].max()) == pytest.approx(trimmed_curve.p0[1], abs=0.2)
    assert float(body_vertices[:, 1].min()) == pytest.approx(trimmed_curve.p1[1], abs=0.2)
    # Arrowhead tip should be at the original curve endpoint (node boundary)
    assert tuple(marker_patches[0].get_xy()[0]) == pytest.approx(curves[0].p1, abs=0.5)
    assert not np.allclose(np.asarray(trimmed_curve.p1), np.asarray(curves[0].p1))
    plt.close(fig)


def test_draw_edges_direct_keeps_original_curves_for_crossings_and_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Crossing and marker passes should still receive untrimmed curves."""

    graph = DaguaGraph.from_edge_list([("A", "B"), ("B", "C")], direction="TB")
    graph.edge_styles = [
        EdgeStyle(arrow="normal", crossing_style="arc"),
        EdgeStyle(arrow="normal", crossing_style="arc"),
    ]
    graph.compute_node_sizes()
    curves = [
        BezierCurve(p0=(0.0, 10.0), cp1=(0.0, 7.0), cp2=(0.0, 3.0), p1=(0.0, 0.0)),
        BezierCurve(p0=(1.0, 10.0), cp1=(1.0, 7.0), cp2=(1.0, 3.0), p1=(1.0, 0.0)),
    ]
    crossings = [EdgeCrossing(edge_a=0, edge_b=1, x=0.5, y=5.0, t_a=0.5, t_b=0.5)]
    body_curves: list[BezierCurve] = []
    crossing_curves: list[list[BezierCurve]] = []
    marker_curves: list[list[BezierCurve]] = []

    monkeypatch.setattr(
        mpl_renderer,
        "_draw_direct_edge_body",
        lambda ax, curve, style: body_curves.append(curve) or [],
    )
    monkeypatch.setattr(
        mpl_renderer,
        "_draw_edge_crossings",
        lambda ax, graph, passed_curves, passed_crossings: crossing_curves.append(passed_curves),
    )
    monkeypatch.setattr(
        mpl_renderer,
        "_draw_direct_edge_markers",
        lambda ax, graph, passed_curves: marker_curves.append(passed_curves),
    )

    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=100)
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-5.0, 15.0)
    fig.canvas.draw()

    _draw_edges_direct(ax, graph, curves, crossings=crossings)

    expected_trimmed = _trim_curve_for_arrows(ax, curves[0], graph.get_style_for_edge(0), graph, 0)
    assert body_curves[0].p0 == pytest.approx(expected_trimmed.p0)
    assert body_curves[0].p1 == pytest.approx(expected_trimmed.p1)
    assert crossing_curves == [curves]
    assert marker_curves == [curves]
    plt.close(fig)


def test_trim_curve_for_arrows_preserves_ortho_waypoints() -> None:
    """Orthogonal routes should remain waypoint-backed after arrow trimming."""

    graph = DaguaGraph.from_edge_list([("A", "B")], direction="TB")
    graph.edge_styles = [EdgeStyle(routing="ortho", arrow="normal")]
    graph.compute_node_sizes()
    positions = torch.tensor([[0.0, 90.0], [80.0, -10.0]], dtype=torch.float32)
    curves = route_edges(positions, graph.edge_index, graph.node_sizes, direction="TB", graph=graph)

    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=100)
    ax.set_xlim(-40.0, 120.0)
    ax.set_ylim(-60.0, 120.0)
    fig.canvas.draw()

    trimmed = _trim_curve_for_arrows(ax, curves[0], graph.get_style_for_edge(0), graph, 0)

    assert curves[0].waypoints is not None
    assert trimmed.waypoints is not None
    assert len(trimmed.waypoints) == len(curves[0].waypoints)
    original_points = np.asarray(curves[0].waypoints, dtype=float)
    trimmed_points = np.asarray(trimmed.waypoints, dtype=float)
    assert np.allclose(trimmed_points[0], original_points[0])
    assert not np.allclose(trimmed_points[-1], original_points[-1])
    assert np.allclose(
        trimmed_points[1:-1],
        original_points[1:-1],
    )

    deltas = np.diff(trimmed_points, axis=0)
    assert np.all(np.isclose(deltas[:, 0], 0.0) | np.isclose(deltas[:, 1], 0.0))
    plt.close(fig)


def test_direct_render_uses_filled_patch_geometry_for_edge_artifacts() -> None:
    """Direct-rendered edges should avoid point-stroked line artists."""

    from matplotlib.collections import LineCollection

    graph = DaguaGraph.from_edge_list([("A", "B"), ("C", "D")], direction="TB")
    graph.edge_styles = [
        EdgeStyle(arrow="open", crossing_style="arc", color_gradient="source_to_target"),
        EdgeStyle(arrow="crow", crossing_style="arc"),
    ]
    graph.compute_node_sizes()
    positions = torch.tensor(
        [[-30.0, 80.0], [30.0, -20.0], [30.0, 80.0], [-30.0, -20.0]],
        dtype=torch.float32,
    )

    fig, ax = render(graph, positions=positions)
    direct_artifacts = [
        patch for patch in ax.patches if float(patch.get_zorder()) in {1.0, 1.6, 1.7, 3.0}
    ]

    assert direct_artifacts
    assert len(ax.lines) == 0
    assert not any(isinstance(collection, LineCollection) for collection in ax.collections)
    assert all(patch.get_linewidth() == pytest.approx(0.0) for patch in direct_artifacts)
    plt.close(fig)


def test_direct_edge_markers_place_tip_at_node_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct marker rendering should place the tip at the node boundary."""

    graph = DaguaGraph.from_edge_list([("A", "B")], direction="TB")
    graph.edge_styles[0] = EdgeStyle(arrow="normal")
    graph.node_styles[1] = NodeStyle(stroke_width=4.0, border_position="center")
    graph.compute_node_sizes()

    positions = np.array([[0.0, 90.0], [0.0, 0.0]], dtype=float)
    target_half_height = float(graph.node_sizes[1, 1]) / 2.0
    curve = BezierCurve(
        p0=(0.0, 60.0),
        cp1=(0.0, 45.0),
        cp2=(0.0, 30.0),
        p1=(0.0, target_half_height),
    )

    captured_points: list[tuple[float, float]] = []

    def _capture_marker(
        ax: Any,
        point: tuple[float, float],
        direction: tuple[float, float],
        marker: str,
        style: Any,
        node_width: float = 0.0,
        node_height: float = 0.0,
        is_self_loop: bool = False,
        **kwargs: Any,
    ) -> None:
        del ax, direction, marker, style, node_width, node_height, is_self_loop, kwargs
        captured_points.append(point)

    monkeypatch.setattr(mpl_renderer, "_draw_edge_marker", _capture_marker)

    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=100)
    ax.set_xlim(-80.0, 80.0)
    ax.set_ylim(-20.0, 120.0)
    fig.canvas.draw()

    mpl_renderer._draw_direct_edge_markers(ax, graph, [curve], positions=positions)
    plt.close(fig)

    # Tip should be at the original curve endpoint (node boundary), no offset
    assert captured_points == [pytest.approx((0.0, target_half_height), abs=0.5)]


def test_custom_edge_collection_places_head_at_node_boundary() -> None:
    """Custom rendering should place arrowhead tip at the node boundary."""

    graph = DaguaGraph.from_edge_list([("A", "B")], direction="TB")
    graph.edge_styles[0] = EdgeStyle(arrow="normal")
    graph.node_styles[1] = NodeStyle(stroke_width=4.0, border_position="center")
    graph.compute_node_sizes()

    positions = np.array([[0.0, 90.0], [0.0, 0.0]], dtype=float)
    target_half_height = float(graph.node_sizes[1, 1]) / 2.0
    curve = BezierCurve(
        p0=(0.0, 60.0),
        cp1=(0.0, 45.0),
        cp2=(0.0, 30.0),
        p1=(0.0, target_half_height),
    )

    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=100)
    ax.set_xlim(-80.0, 80.0)
    ax.set_ylim(-20.0, 120.0)
    fig.canvas.draw()

    collection = _build_custom_edge_collection(ax, graph, [curve], positions=positions)
    prepared = collection.prepared_edges[0]
    head_result = prepared.head_result
    assert head_result is not None

    head_vertices = np.concatenate(
        [path.vertices for path in [*head_result.filled_paths, *head_result.stroked_paths]],
        axis=0,
    )
    plt.close(fig)

    # Arrowhead tip should be at the node boundary (no outward offset)
    assert tuple(prepared.lane_curve.p1) == pytest.approx(curve.p1)
    assert float(head_vertices[:, 1].min()) == pytest.approx(target_half_height, abs=1.0)


def test_arrowhead_does_not_extend_into_target_node() -> None:
    """The custom head should extend away from the target-node center."""

    graph = DaguaGraph.from_edge_list([("A", "B")], direction="TB")
    graph.edge_styles[0] = EdgeStyle(arrow="normal")
    graph.compute_node_sizes()

    positions = torch.tensor([[0.0, 40.0], [0.0, 0.0]], dtype=torch.float32)
    curves = route_edges(positions, graph.edge_index, graph.node_sizes, graph.direction, graph)

    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=100)
    ax.set_xlim(-50.0, 50.0)
    ax.set_ylim(-10.0, 50.0)
    fig.canvas.draw()

    collection = _build_custom_edge_collection(ax, graph, curves, positions=positions.numpy())
    prepared = collection.prepared_edges[0]
    head_result = prepared.head_result
    assert head_result is not None

    tip = np.asarray(prepared.lane_curve.p1, dtype=float)
    base = head_result.trim_contour.vertices[:2].mean(axis=0)
    target_center = positions[1].numpy()

    plt.close(fig)

    assert np.linalg.norm(base - target_center) > np.linalg.norm(tip - target_center)


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
        if isinstance(collection, PatchCollection)
        and float(collection.get_zorder()) >= 2.0
        and not (
            len(collection.get_paths()) == graph.num_nodes
            and np.allclose(collection.get_linewidths(), 0.0)
        )
    ]
    assert len(polygons) == 0, "Custom heads should not fall back to standalone Polygon patches"
    assert len(head_collections) >= 1, "Vee arrow should be rendered by the custom head collection"
    assert len(fancy_arrows) == 0, "Vee arrow should not use FancyArrowPatch"
    plt.close(fig)


def test_vee_arrowhead_builder_returns_filled_chevron() -> None:
    """The custom vee head should be a filled chevron shape."""

    result = build_arrowhead(
        "vee",
        tip=(0.0, 0.0),
        tangent=(1.0, 0.0),
        length=14.0,
        width=10.0,
        body_width=2.0,
    )

    assert len(result.filled_paths) == 1, "Vee should have one filled chevron path"
    assert not result.stroked_paths, "Filled vee should have no stroked paths"
    path = result.filled_paths[0]
    assert path.vertices.shape[0] >= 5, (
        "Filled vee needs at least 5 vertices (tip + 2 arms + 2 notch)"
    )


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
        if isinstance(collection, PatchCollection)
        and float(collection.get_zorder()) >= 2.0
        and not (
            len(collection.get_paths()) == graph.num_nodes
            and np.allclose(collection.get_linewidths(), 0.0)
        )
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
    """Manual marker ribbons should always convert point sizing into data units."""

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

    assert len(ax.patches) == 3
    vertices = np.concatenate([patch.get_path().vertices for patch in ax.patches], axis=0)
    expected_scale = _compute_display_scale(ax)
    outline_width = _edge_width_data_units(ax, style.width)
    plt.close(fig)

    assert float(vertices[:, 1].max()) == pytest.approx(
        (style.arrow_length * expected_scale) + (outline_width / 2.0),
        rel=0.05,
    )
    assert float(vertices[:, 0].max() - vertices[:, 0].min()) == pytest.approx(
        style.arrow_width * expected_scale * 1.2,
        rel=0.1,
    )


def test_custom_edge_collection_converts_stroke_width_to_data_units() -> None:
    """Custom edge outlines should use the same data-space width as the body ribbon."""

    graph = DaguaGraph.from_edge_list([("a", "b")])
    graph.edge_styles[0] = EdgeStyle(width=4.0, arrow="none")
    graph.compute_node_sizes()
    curve = dagua.route_edges(
        torch.tensor([[0.0, 60.0], [0.0, -60.0]], dtype=torch.float32),
        graph.edge_index,
        graph.node_sizes,
        graph.direction,
        graph,
    )[0]

    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=100)
    ax.set_xlim(-80.0, 80.0)
    ax.set_ylim(-80.0, 80.0)
    fig.canvas.draw()

    collection = _build_custom_edge_collection(ax, graph, [curve])
    edge = collection.edges[0]
    expected_width = _edge_width_data_units(ax, 4.0)
    plt.close(fig)

    assert edge.width == pytest.approx(expected_width)
    assert edge.stroke_width == pytest.approx(expected_width)


def test_custom_edge_collection_scales_arrowheads_with_edge_width() -> None:
    """Arrowheads should grow sublinearly as the edge stroke gets heavier."""

    graph = DaguaGraph()
    graph.add_node("a")
    graph.add_node("b")
    graph.add_node("c")
    graph.add_edge(
        "a",
        "b",
        style=EdgeStyle(width=1.2, arrow_length=10.0, arrow_width=7.0, arrow_node_fraction=0.0),
    )
    graph.add_edge(
        "a",
        "c",
        style=EdgeStyle(width=4.8, arrow_length=10.0, arrow_width=7.0, arrow_node_fraction=0.0),
    )
    graph.compute_node_sizes()
    positions = torch.tensor([[0.0, 60.0], [-35.0, -60.0], [35.0, -60.0]], dtype=torch.float32)
    curves = dagua.route_edges(
        positions,
        graph.edge_index,
        graph.node_sizes,
        graph.direction,
        graph,
    )

    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=100)
    ax.set_xlim(-80.0, 80.0)
    ax.set_ylim(-80.0, 80.0)
    fig.canvas.draw()

    collection = _build_custom_edge_collection(ax, graph, curves)
    thin_edge, thick_edge = collection.edges
    plt.close(fig)

    # Arrowheads scale sublinearly with edge width (sqrt scaling)
    ratio = thick_edge.arrowhead_length / thin_edge.arrowhead_length
    assert 1.1 < ratio < 2.1, f"Arrow scaling ratio {ratio} out of expected range"


def test_custom_edge_collection_caps_self_loop_arrowheads() -> None:
    """Self-loop arrowheads should stay below the configured node-fraction cap."""

    graph = DaguaGraph()
    graph.add_node(
        "loop",
        style=NodeStyle(shape="star", min_width=120.0, min_height=120.0),
    )
    graph.add_edge(
        "loop",
        "loop",
        style=EdgeStyle(width=2.0, arrow="normal", arrow_length=48.0, arrow_width=36.0),
    )
    graph.compute_node_sizes()
    positions = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    curve = dagua.route_edges(
        positions,
        graph.edge_index,
        graph.node_sizes,
        graph.direction,
        graph,
    )[0]

    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=100)
    ax.set_xlim(-80.0, 80.0)
    ax.set_ylim(-80.0, 80.0)
    fig.canvas.draw()

    collection = _build_custom_edge_collection(ax, graph, [curve])
    edge = collection.edges[0]
    max_length = float(min(graph.node_sizes[0, 0], graph.node_sizes[0, 1])) * 0.25
    plt.close(fig)

    assert edge.arrowhead_length == pytest.approx(max_length)
    assert edge.arrowhead_width == pytest.approx(max_length * 0.7)


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


def test_normal_arrow_marker_scales_with_edge_width() -> None:
    """Direct marker rendering should match the collection's width-aware sizing."""

    fig, ax = plt.subplots()
    thin_style = EdgeStyle(arrow="normal", width=0.5, arrow_width=10.0, arrow_length=14.0)
    thick_style = EdgeStyle(arrow="normal", width=5.0, arrow_width=10.0, arrow_length=14.0)
    fig.canvas.draw()

    _draw_edge_marker(
        ax=ax,
        point=(0.0, 0.0),
        direction=(0.0, -1.0),
        marker="normal",
        style=thin_style,
    )
    _draw_edge_marker(
        ax=ax,
        point=(20.0, 0.0),
        direction=(0.0, -1.0),
        marker="normal",
        style=thick_style,
    )

    thin_vertices = ax.patches[0].get_xy()
    thick_vertices = ax.patches[1].get_xy()
    thin_base_width = abs(float(thin_vertices[1][0] - thin_vertices[2][0]))
    thick_base_width = abs(float(thick_vertices[1][0] - thick_vertices[2][0]))

    assert thick_base_width / thin_base_width == pytest.approx(np.sqrt(10.0), rel=0.05)
    plt.close(fig)


def test_triangle_labels_shift_toward_visual_centroid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Triangle labels should sit lower than the geometric center."""

    captured = _capture_render_calls(monkeypatch)
    graph = DaguaGraph()
    graph.add_node("a", label="Triangle", style=NodeStyle(shape="triangle"))
    pos = np.array([[10.0, 20.0]])
    sizes = np.array([[120.0, 60.0]])

    fig, ax = plt.subplots()
    mpl_renderer._draw_node_labels(ax, graph, pos, sizes)

    bbox = _label_bbox(ax, "dagua-node-label-0")
    specs, _display_scale = captured[0]
    spec = next(item for item in specs if item.gid == "dagua-node-label-0")
    expected_bbox = _expected_plain_label_bbox(
        ax,
        text="Triangle",
        font_size=spec.font_size,
        font_family=spec.font_family,
        font_weight=spec.font_weight,
        ha=spec.ha,
        va=spec.va,
        anchor_x=10.0,
        anchor_y=12.5,
    )
    assert (bbox[2] + bbox[3]) / 2.0 == pytest.approx(
        (expected_bbox[2] + expected_bbox[3]) / 2.0,
        abs=0.75,
    )
    assert (bbox[2] + bbox[3]) / 2.0 < 20.0
    plt.close(fig)


def test_triangle_rich_labels_shift_toward_visual_centroid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rich triangle labels should use the same centroid-aware anchor."""

    captured = _capture_render_calls(monkeypatch)
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

    bbox = _label_bbox(ax, "dagua-node-label-0")
    specs, _display_scale = captured[0]
    spec = next(item for item in specs if item.gid == "dagua-node-label-0")
    expected_bbox = _expected_plain_label_bbox(
        ax,
        text="Triangle",
        font_size=spec.font_size,
        font_family=spec.font_family,
        font_weight=spec.font_weight,
        ha=spec.ha,
        va=spec.va,
        anchor_x=10.0,
        anchor_y=12.5,
    )
    assert (bbox[2] + bbox[3]) / 2.0 == pytest.approx(
        (expected_bbox[2] + expected_bbox[3]) / 2.0,
        abs=0.75,
    )
    assert (bbox[2] + bbox[3]) / 2.0 < 20.0
    plt.close(fig)


@pytest.mark.parametrize(("text_valign", "expected_delta"), [("top", -9.0), ("bottom", 9.0)])
def test_ellipse_labels_inset_top_and_bottom_alignment(
    monkeypatch: pytest.MonkeyPatch,
    text_valign: str,
    expected_delta: float,
) -> None:
    """Ellipse top and bottom labels should move inward relative to polygon shapes."""

    captured = _capture_render_calls(monkeypatch)
    ellipse_graph = DaguaGraph()
    ellipse_graph.add_node(
        "ellipse",
        label="Label",
        style=NodeStyle(shape="ellipse", text_valign=text_valign),
    )
    diamond_graph = DaguaGraph()
    diamond_graph.add_node(
        "diamond",
        label="Label",
        style=NodeStyle(shape="diamond", text_valign=text_valign),
    )
    pos = np.array([[0.0, 0.0]], dtype=float)
    sizes = np.array([[120.0, 60.0]], dtype=float)

    ellipse_fig, ellipse_ax = plt.subplots(figsize=(4.0, 4.0), dpi=100)
    ellipse_ax.set_xlim(-50.0, 50.0)
    ellipse_ax.set_ylim(-50.0, 50.0)
    ellipse_ax.set_aspect("equal")
    ellipse_fig.canvas.draw()

    diamond_fig, diamond_ax = plt.subplots(figsize=(4.0, 4.0), dpi=100)
    diamond_ax.set_xlim(-50.0, 50.0)
    diamond_ax.set_ylim(-50.0, 50.0)
    diamond_ax.set_aspect("equal")
    diamond_fig.canvas.draw()

    mpl_renderer._draw_node_labels(ellipse_ax, ellipse_graph, pos, sizes)
    mpl_renderer._draw_node_labels(diamond_ax, diamond_graph, pos, sizes)

    ellipse_specs, _ellipse_display_scale = captured[0]
    diamond_specs, _diamond_display_scale = captured[1]
    ellipse_spec = next(spec for spec in ellipse_specs if spec.gid == "dagua-node-label-0")
    diamond_spec = next(spec for spec in diamond_specs if spec.gid == "dagua-node-label-0")

    assert ellipse_spec.y - diamond_spec.y == pytest.approx(expected_delta)
    plt.close(ellipse_fig)
    plt.close(diamond_fig)


@pytest.mark.parametrize("text_valign", ["top", "bottom"])
def test_top_and_bottom_label_alignment_enforce_minimum_padding(
    monkeypatch: pytest.MonkeyPatch,
    text_valign: str,
) -> None:
    """Top and bottom node labels should clamp small padding to a 2-point inset."""

    pos = np.array([[0.0, 0.0]], dtype=float)
    sizes = np.array([[120.0, 60.0]], dtype=float)
    graphs = [
        DaguaGraph(),
        DaguaGraph(),
        DaguaGraph(),
    ]
    for graph, pad_y in zip(graphs, [0.0, 2.0, 4.0]):
        graph.add_node(
            "node",
            label="Label",
            style=NodeStyle(shape="roundrect", text_valign=text_valign, padding=(0.0, pad_y)),
        )

    captured = _capture_render_calls(monkeypatch)
    figures_and_axes = [plt.subplots(figsize=(4.0, 4.0), dpi=100) for _ in graphs]
    for fig, ax in figures_and_axes:
        ax.set_xlim(-50.0, 50.0)
        ax.set_ylim(-50.0, 50.0)
        ax.set_aspect("equal")
        fig.canvas.draw()
    for graph, (_fig, ax) in zip(graphs, figures_and_axes):
        mpl_renderer._draw_node_labels(ax, graph, pos, sizes)

    zero_pad_spec = next(spec for spec in captured[0][0] if spec.gid == "dagua-node-label-0")
    min_pad_spec = next(spec for spec in captured[1][0] if spec.gid == "dagua-node-label-0")
    large_pad_spec = next(spec for spec in captured[2][0] if spec.gid == "dagua-node-label-0")

    assert zero_pad_spec.y == pytest.approx(min_pad_spec.y)
    if text_valign == "top":
        assert large_pad_spec.y < zero_pad_spec.y
    else:
        assert large_pad_spec.y > zero_pad_spec.y

    for fig, _ax in figures_and_axes:
        plt.close(fig)


def test_node_and_external_label_font_sizes_use_data_coordinate_scaling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Node-bound labels should encode their target data size before rendering."""

    captured = _capture_render_calls(monkeypatch)
    graph = DaguaGraph()
    graph.add_node(
        "a",
        label="Top\nBottom",
        style=NodeStyle(
            font_size=17.0,
            external_label="Outside",
            external_label_font_size=12.0,
        ),
    )
    pos = np.array([[0.0, 0.0]], dtype=float)
    sizes = np.array([[40.0, 20.0]], dtype=float)

    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=100)
    ax.set_xlim(-50.0, 50.0)
    ax.set_ylim(-50.0, 50.0)
    ax.set_aspect("equal")
    fig.canvas.draw()

    mpl_renderer._draw_node_labels(ax, graph, pos, sizes)
    mpl_renderer._draw_external_labels(ax, graph, pos, sizes)

    node_specs, node_display_scale = captured[0]
    external_specs, external_display_scale = captured[1]
    node_spec = next(spec for spec in node_specs if spec.gid == "dagua-node-label-0")
    external_spec = next(
        spec for spec in external_specs if spec.gid == "dagua-node-external-label-0"
    )

    # Node label font size uses layout-computed data coordinates directly.
    node_expected = 17.0  # font_size from node_font_sizes (data coords)
    external_expected = max(20.0 * 0.35 * (12.0 / 8.0), 20.0 * 0.1)
    external_expected = min(external_expected, 20.0 * 0.6)

    assert node_spec.font_size * node_display_scale == pytest.approx(node_expected)
    assert external_spec.font_size * external_display_scale == pytest.approx(external_expected)
    plt.close(fig)


def test_bold_node_and_external_labels_normalize_weight_and_gain_size_boost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bold node-label paths should request a heavy face and slightly larger size."""

    captured = _capture_render_calls(monkeypatch)
    graph = DaguaGraph()
    graph.add_node(
        "a",
        label="Bold Here",
        style=NodeStyle(
            font_size=16.0,
            font_weight="700",
            external_label="Outside",
            external_label_font_size=10.0,
        ),
    )
    pos = np.array([[0.0, 0.0]], dtype=float)
    sizes = np.array([[40.0, 20.0]], dtype=float)

    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=100)
    ax.set_xlim(-50.0, 50.0)
    ax.set_ylim(-50.0, 50.0)
    ax.set_aspect("equal")
    fig.canvas.draw()

    mpl_renderer._draw_node_labels(ax, graph, pos, sizes)
    mpl_renderer._draw_external_labels(ax, graph, pos, sizes)

    node_specs, node_display_scale = captured[0]
    external_specs, external_display_scale = captured[1]
    node_spec = next(spec for spec in node_specs if spec.gid == "dagua-node-label-0")
    external_spec = next(
        spec for spec in external_specs if spec.gid == "dagua-node-external-label-0"
    )

    baseline_external = max(20.0 * 0.35 * (10.0 / 8.0), 20.0 * 0.1)
    baseline_external = min(baseline_external, 20.0 * 0.6)

    assert node_spec.font_weight == "bold"
    assert external_spec.font_weight == "bold"
    assert node_spec.font_size * node_display_scale == pytest.approx(16.0 * 1.05)
    assert external_spec.font_size * external_display_scale == pytest.approx(
        baseline_external * 1.05
    )
    plt.close(fig)


def test_node_label_wrap_budget_uses_display_scaled_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Node wrap budgets should stay consistent between sizing and render-time layout."""

    captured = _capture_render_calls(monkeypatch)
    graph = DaguaGraph()
    graph.add_node(
        "a",
        label="Wrap this sample label across a few lines",
        style=NodeStyle(shape="rect", text_wrap="wrap", text_max_width=80.0),
    )
    pos = np.array([[0.0, 0.0]], dtype=float)
    sizes = np.array([[120.0, 72.0]], dtype=float)

    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=100)
    ax.set_xlim(-80.0, 80.0)
    ax.set_ylim(-80.0, 80.0)
    ax.set_aspect("equal")
    fig.canvas.draw()

    mpl_renderer._draw_node_labels(ax, graph, pos, sizes)

    node_specs, display_scale = captured[0]
    node_spec = next(spec for spec in node_specs if spec.gid == "dagua-node-label-0")
    prepared = dagua_utils.prepare_label_text(
        node_spec.text,
        font_size=node_spec.font_size * display_scale,
        text_wrap=node_spec.text_wrap,
        text_max_width=node_spec.text_max_width,
        text_transform=node_spec.text_transform,
        label_format="plain",
    )

    assert node_spec.text_max_width == pytest.approx(80.0 * display_scale)
    assert prepared.replace("\n", " ").split() == graph.node_labels[0].split()
    plt.close(fig)


def test_edge_label_font_sizes_use_average_node_height(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Edge labels should scale from the graph's average node height."""

    captured = _capture_render_calls(monkeypatch)
    graph = DaguaGraph()
    graph.add_edge(
        "a",
        "b",
        label="middle",
        style=EdgeStyle(label_font_size=14.0, head_label="H", tail_label="T"),
    )
    sizes = np.array([[20.0, 20.0], [20.0, 40.0]], dtype=float)
    curves = [
        BezierCurve(
            p0=(0.0, 0.0),
            cp1=(10.0, 10.0),
            cp2=(20.0, 10.0),
            p1=(30.0, 0.0),
        )
    ]

    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=100)
    ax.set_xlim(-20.0, 50.0)
    ax.set_ylim(-20.0, 30.0)
    ax.set_aspect("equal")
    fig.canvas.draw()

    mpl_renderer._draw_edge_labels(ax, graph, curves, sizes=sizes)

    specs, display_scale = captured[0]
    main_spec = next(spec for spec in specs if spec.gid == "dagua-edge-label-0")
    head_spec = next(spec for spec in specs if spec.gid == "dagua-edge-head-label-0")
    tail_spec = next(spec for spec in specs if spec.gid == "dagua-edge-tail-label-0")
    avg_node_height = float(sizes[:, 1].mean())

    assert main_spec.font_size * display_scale == pytest.approx(
        avg_node_height * 0.25 * (14.0 / 7.0)
    )
    assert head_spec.font_size * display_scale == pytest.approx(
        avg_node_height * 0.25 * ((14.0 * 0.85) / 7.0)
    )
    assert tail_spec.font_size * display_scale == pytest.approx(
        avg_node_height * 0.25 * ((14.0 * 0.85) / 7.0)
    )
    edge_style = mpl_renderer._edge_style_for_render(graph, 0)
    minimum_requested_offset = (12.0 + ((14.0 * 0.85) / 2.0)) * display_scale
    expected_offset = mpl_renderer._endpoint_label_offset_data(
        edge_style,
        "head",
        avg_node_height,
        display_scale,
    )
    expected_head = mpl_renderer.edge_endpoint_label_position(
        curves[0],
        "head",
        label_offset=expected_offset,
    )
    expected_tail = mpl_renderer.edge_endpoint_label_position(
        curves[0],
        "tail",
        label_offset=expected_offset,
    )
    assert expected_offset >= minimum_requested_offset
    assert (head_spec.x, head_spec.y) == pytest.approx(expected_head)
    assert (tail_spec.x, tail_spec.y) == pytest.approx(expected_tail)
    plt.close(fig)


def test_graph_title_font_size_tracks_graph_height(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Graph titles should be scaled from the rendered graph height."""

    captured = _capture_render_calls(monkeypatch)
    graph = DaguaGraph()
    graph.add_edge("a", "b")
    positions = torch.tensor([[0.0, 0.0], [0.0, 60.0]], dtype=torch.float32)
    graph.graph_style.title_font_size = 15.0

    fig, ax = render(graph, positions=positions, title="Title")
    title_specs = [
        spec
        for specs, _display_scale in captured
        for spec in specs
        if spec.gid == "dagua-graph-title"
    ]
    assert title_specs
    title_spec = title_specs[0]
    title_display_scale = next(
        display_scale
        for specs, display_scale in captured
        if any(spec.gid == "dagua-graph-title" for spec in specs)
    )

    assert graph.node_sizes is not None
    sizes = graph.node_sizes.detach().cpu().numpy()
    margin = float(graph.graph_style.margin)
    y_min = float((positions[:, 1].detach().cpu().numpy() - sizes[:, 1] / 2.0).min() - margin)
    y_max = float((positions[:, 1].detach().cpu().numpy() + sizes[:, 1] / 2.0).max() + margin)
    graph_height = y_max - y_min
    expected = graph_height * 0.03 * (15.0 / 10.0)

    assert title_spec.font_size * title_display_scale == pytest.approx(expected)
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
    from matplotlib.collections import PatchCollection

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
            stroke_width=0.0,
        ),
    )

    monkeypatch.setattr(mpl_renderer, "measure_text_data", lambda *args, **kwargs: (80.0, 20.0))

    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=100)
    ax.set_xlim(-200.0, 200.0)
    ax.set_ylim(-100.0, 100.0)
    fig.canvas.draw()
    _draw_clusters(
        ax=ax,
        graph=graph,
        pos=np.array([[0.0, 0.0]], dtype=float),
        sizes=np.array([[70.0, 20.0]], dtype=float),
    )

    label_patches = [
        patch
        for patch in ax.patches
        if isinstance(patch, PathPatch) and patch.get_gid() is not None
    ]
    assert len(label_patches) == 1
    assert ax.collections
    assert isinstance(ax.collections[0], PatchCollection)
    path = ax.collections[0].get_paths()[0]
    vertices = path.vertices
    width = float(vertices[:, 0].max() - vertices[:, 0].min())
    expected_width = max(70.0, 80.0)
    assert width == pytest.approx(expected_width, abs=0.25)
    assert len(ax.texts) == 0
    assert label_patches[0].get_gid() == "dagua-cluster-label-outer"
    assert label_patches[0].get_clip_on() is False
    plt.close(fig)


def test_cluster_offsets_and_corner_radius_use_display_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cluster label offsets and corner radius should be converted from points."""
    from matplotlib.collections import PatchCollection

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
            stroke_width=0.0,
        ),
    )

    captured = _capture_render_calls(monkeypatch)
    monkeypatch.setattr(mpl_renderer, "measure_text_data", lambda *args, **kwargs: (40.0, 12.0))

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

    label_patches = [
        patch
        for patch in ax.patches
        if isinstance(patch, PathPatch) and patch.get_gid() is not None
    ]
    assert len(label_patches) == 1
    assert ax.collections
    assert isinstance(ax.collections[0], PatchCollection)
    assert len(ax.texts) == 0
    display_scale = _compute_display_scale(ax)
    label_width = 40.0
    label_height = 12.0
    initial_x_min = -10.0
    initial_x_max = 10.0
    expanded_width = max(label_width + (8.0 * display_scale * 2.0), initial_x_max - initial_x_min)
    x_min = -expanded_width / 2.0
    y_max = 10.0 + label_height + mpl_renderer._points_to_data_units(ax, 2.0, "y")
    label_spec = next(spec for spec in captured[0][0] if spec.gid == "dagua-cluster-label-outer")
    path = ax.collections[0].get_paths()[0]
    label_bbox = _label_bbox(ax, "dagua-cluster-label-outer")
    expected_bbox = _expected_plain_label_bbox(
        ax,
        text="Cluster",
        font_size=label_spec.font_size,
        font_family=graph.get_style_for_cluster("outer").font_family or "",
        font_weight=graph.get_style_for_cluster("outer").font_weight,
        ha="left",
        va="top",
        anchor_x=x_min + (8.0 * display_scale),
        anchor_y=y_max - (20.0 * display_scale),
    )
    resolved_style = graph.get_style_for_cluster("outer")
    expected_font_data = (
        20.0
        * 0.3
        * (float(resolved_style.font_size) / mpl_renderer._DEFAULT_CLUSTER_LABEL_FONT_POINTS)
    )

    path_x_min = float(path.vertices[:, 0].min())
    assert label_spec.font_size * display_scale == pytest.approx(expected_font_data)
    assert label_bbox[3] == pytest.approx(expected_bbox[3], abs=0.75)
    assert path.vertices[0][0] == pytest.approx(path_x_min + (6.0 * display_scale), abs=0.05)
    plt.close(fig)


def test_cluster_bottom_left_label_uses_expanded_y_min(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bottom-left cluster labels should anchor from the expanded bottom edge."""

    captured = _capture_render_calls(monkeypatch)
    graph = DaguaGraph()
    graph.add_node("a")
    graph.add_cluster(
        "outer",
        ["a"],
        label="Cluster",
        style=ClusterStyle(
            padding=0.0,
            font_size=12.0,
            label_position="bottom-left",
            label_offset=(8.0, 20.0),
            stroke_width=0.0,
        ),
    )

    monkeypatch.setattr(mpl_renderer, "measure_text_data", lambda *args, **kwargs: (40.0, 12.0))

    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=100)
    ax.set_xlim(-50.0, 50.0)
    ax.set_ylim(-50.0, 50.0)
    ax.set_aspect("equal")
    fig.canvas.draw()

    _draw_clusters(
        ax=ax,
        graph=graph,
        pos=np.array([[0.0, 0.0]], dtype=float),
        sizes=np.array([[20.0, 20.0]], dtype=float),
    )

    display_scale = _compute_display_scale(ax)
    label_spec = next(spec for spec in captured[0][0] if spec.gid == "dagua-cluster-label-outer")
    cluster_path = ax.collections[0].get_paths()[0]
    expected_y_min = (
        -10.0
        - 12.0
        - mpl_renderer._points_to_data_units(
            ax, mpl_renderer._CLUSTER_LABEL_VERTICAL_GAP_POINTS, "y"
        )
    )

    assert label_spec.va == "bottom"
    assert label_spec.y == pytest.approx(expected_y_min + (20.0 * display_scale))
    assert float(cluster_path.vertices[:, 1].min()) == pytest.approx(expected_y_min, abs=0.25)
    plt.close(fig)


def test_cluster_outside_top_label_sits_above_box_without_expanding_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Outside-top cluster labels should live above the box and leave its bounds unchanged."""

    captured = _capture_render_calls(monkeypatch)
    graph = DaguaGraph()
    graph.add_node("a")
    graph.add_cluster(
        "outer",
        ["a"],
        label="Cluster",
        style=ClusterStyle(
            padding=0.0,
            font_size=12.0,
            label_position="outside-top",
            label_offset=(8.0, 20.0),
            stroke_width=0.0,
        ),
    )

    monkeypatch.setattr(mpl_renderer, "measure_text_data", lambda *args, **kwargs: (40.0, 12.0))

    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=100)
    ax.set_xlim(-50.0, 50.0)
    ax.set_ylim(-50.0, 50.0)
    ax.set_aspect("equal")
    fig.canvas.draw()

    _draw_clusters(
        ax=ax,
        graph=graph,
        pos=np.array([[0.0, 0.0]], dtype=float),
        sizes=np.array([[20.0, 20.0]], dtype=float),
    )

    display_scale = _compute_display_scale(ax)
    label_spec = next(spec for spec in captured[0][0] if spec.gid == "dagua-cluster-label-outer")
    cluster_path = ax.collections[0].get_paths()[0]
    raw_y_max = 10.0

    assert label_spec.va == "bottom"
    assert label_spec.y == pytest.approx(raw_y_max + (20.0 * display_scale))
    assert float(cluster_path.vertices[:, 1].max()) == pytest.approx(raw_y_max, abs=0.25)
    assert _label_bbox(ax, "dagua-cluster-label-outer")[2] > raw_y_max
    plt.close(fig)


def test_cluster_label_wrap_budget_uses_display_scaled_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cluster label wrap budgets should match the render-time display scale."""

    captured = _capture_render_calls(monkeypatch)
    graph = DaguaGraph()
    graph.add_node("a")
    graph.add_cluster(
        "outer",
        ["a"],
        label="Wrap this cluster label across multiple lines",
        style=ClusterStyle(
            padding=0.0,
            stroke_width=0.0,
            text_wrap="wrap",
            text_max_width=36.0,
        ),
    )

    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=100)
    ax.set_xlim(-60.0, 60.0)
    ax.set_ylim(-60.0, 60.0)
    ax.set_aspect("equal")
    fig.canvas.draw()

    _draw_clusters(
        ax=ax,
        graph=graph,
        pos=np.array([[0.0, 0.0]], dtype=float),
        sizes=np.array([[20.0, 20.0]], dtype=float),
    )

    cluster_specs, display_scale = captured[0]
    cluster_spec = next(spec for spec in cluster_specs if spec.gid == "dagua-cluster-label-outer")
    prepared = dagua_utils.prepare_label_text(
        cluster_spec.text,
        font_size=cluster_spec.font_size * display_scale,
        text_wrap=cluster_spec.text_wrap,
        text_max_width=cluster_spec.text_max_width,
        text_transform="none",
        label_format="plain",
    )
    line_gids = {
        str(patch.get_gid())
        for patch in ax.patches
        if isinstance(patch, PathPatch)
        and isinstance(patch.get_gid(), str)
        and patch.get_gid().startswith("dagua-cluster-label-outer-")
    }

    assert cluster_spec.text_max_width == pytest.approx(36.0 * display_scale)
    assert "\n" in prepared
    assert "dagua-cluster-label-outer-0" in line_gids
    assert "dagua-cluster-label-outer-1" in line_gids
    plt.close(fig)


def test_cluster_box_expands_for_bottom_labels_but_not_outside_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bottom labels should grow the cluster box while outside labels should not."""

    monkeypatch.setattr(mpl_renderer, "measure_text_data", lambda *args, **kwargs: (40.0, 12.0))

    bottom_graph = DaguaGraph()
    bottom_graph.add_node("a")
    bottom_graph.add_cluster(
        "bottom",
        ["a"],
        label="Bottom",
        style=ClusterStyle(
            padding=0.0,
            label_position="bottom-left",
            label_offset=(8.0, 20.0),
            stroke_width=0.0,
        ),
    )

    outside_graph = DaguaGraph()
    outside_graph.add_node("a")
    outside_graph.add_cluster(
        "outside",
        ["a"],
        label="Outside",
        style=ClusterStyle(
            padding=0.0,
            label_position="outside-top",
            label_offset=(8.0, 20.0),
            stroke_width=0.0,
        ),
    )

    bottom_fig, bottom_ax = plt.subplots(figsize=(4.0, 4.0), dpi=100)
    bottom_ax.set_xlim(-50.0, 50.0)
    bottom_ax.set_ylim(-50.0, 50.0)
    bottom_ax.set_aspect("equal")
    bottom_fig.canvas.draw()
    _draw_clusters(
        ax=bottom_ax,
        graph=bottom_graph,
        pos=np.array([[0.0, 0.0]], dtype=float),
        sizes=np.array([[20.0, 20.0]], dtype=float),
    )

    outside_fig, outside_ax = plt.subplots(figsize=(4.0, 4.0), dpi=100)
    outside_ax.set_xlim(-50.0, 50.0)
    outside_ax.set_ylim(-50.0, 50.0)
    outside_ax.set_aspect("equal")
    outside_fig.canvas.draw()
    _draw_clusters(
        ax=outside_ax,
        graph=outside_graph,
        pos=np.array([[0.0, 0.0]], dtype=float),
        sizes=np.array([[20.0, 20.0]], dtype=float),
    )

    bottom_y_min = float(bottom_ax.collections[0].get_paths()[0].vertices[:, 1].min())
    outside_y_min = float(outside_ax.collections[0].get_paths()[0].vertices[:, 1].min())
    raw_y_min = -10.0

    assert bottom_y_min < raw_y_min
    assert outside_y_min == pytest.approx(raw_y_min, abs=0.25)
    plt.close(bottom_fig)
    plt.close(outside_fig)


def test_sibling_cluster_labels_avoid_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sibling cluster labels should nudge vertically when their boxes collide."""
    graph = DaguaGraph()
    graph.add_node("left_node")
    graph.add_node("right_node")
    graph.add_cluster(
        "root",
        ["left_node", "right_node"],
        label="Root",
        style=ClusterStyle(padding=0.0, stroke_width=0.0),
    )
    graph.add_cluster(
        "left",
        ["left_node"],
        parent="root",
        label="Left Branch",
        style=ClusterStyle(
            padding=0.0,
            font_size=12.0,
            label_position="top-center",
            label_offset=(0.0, 12.0),
            stroke_width=0.0,
        ),
    )
    graph.add_cluster(
        "right",
        ["right_node"],
        parent="root",
        label="Right Branch",
        style=ClusterStyle(
            padding=0.0,
            font_size=12.0,
            label_position="top-center",
            label_offset=(0.0, 12.0),
            stroke_width=0.0,
        ),
    )

    def fake_measure_text_data(*args: Any, **kwargs: Any) -> tuple[float, float]:
        """Return controlled label metrics for deterministic overlap checks."""
        label = str(args[0])
        if label == "Root":
            return (40.0, 14.0)
        return (140.0, 14.0)

    monkeypatch.setattr(mpl_renderer, "measure_text_data", fake_measure_text_data)

    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=100)
    ax.set_xlim(-80.0, 80.0)
    ax.set_ylim(-40.0, 60.0)
    fig.canvas.draw()

    _draw_clusters(
        ax=ax,
        graph=graph,
        pos=np.array([[-12.0, 0.0], [12.0, 0.0]], dtype=float),
        sizes=np.array([[16.0, 16.0], [16.0, 16.0]], dtype=float),
    )

    left_bbox = _label_bbox(ax, "dagua-cluster-label-left")
    right_bbox = _label_bbox(ax, "dagua-cluster-label-right")
    overlaps = (
        left_bbox[0] < right_bbox[1]
        and left_bbox[1] > right_bbox[0]
        and left_bbox[2] < right_bbox[3]
        and left_bbox[3] > right_bbox[2]
    )

    assert not overlaps
    assert right_bbox[3] < left_bbox[3]
    plt.close(fig)


def test_cluster_borders_include_visible_stroke_outline() -> None:
    """Cluster boxes should render borders as filled geometry without outline strokes."""

    graph = DaguaGraph()
    graph.add_node("a")
    graph.add_cluster(
        "outer",
        ["a"],
        label="Cluster",
        style=ClusterStyle(padding=0.0, stroke_width=0.7),
    )

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

    border_patches = [
        patch
        for patch in ax.patches
        if isinstance(patch, PathPatch)
        and patch.get_facecolor()[-1] == pytest.approx(0.0)
        and patch.get_gid() is None
    ]
    assert not border_patches
    assert ax.collections
    assert sum(len(collection.get_paths()) for collection in ax.collections) >= 2
    plt.close(fig)


def test_deep_cluster_bounds_stay_inside_render_axes() -> None:
    """Deep nested cluster boxes should stay inside the computed viewport."""

    case = {case.case_id: case for case in build_case_catalog()}["evil_8_deep_clusters"]
    fig, ax = render(case.graph, case.positions)
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    cluster_collections = [
        collection for collection in ax.collections if collection.get_zorder() < 0.1
    ]
    tolerance = 1e-6

    assert len(cluster_collections) >= 10
    for collection in cluster_collections:
        vertices = np.concatenate([path.vertices for path in collection.get_paths()], axis=0)
        assert float(vertices[:, 0].min()) >= x_min - tolerance
        assert float(vertices[:, 0].max()) <= x_max + tolerance
        assert float(vertices[:, 1].min()) >= y_min - tolerance
        assert float(vertices[:, 1].max()) <= y_max + tolerance
    plt.close(fig)
