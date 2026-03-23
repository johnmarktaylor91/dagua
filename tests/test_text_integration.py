"""Integration tests for TextPath-based rendering inside the matplotlib pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
import torch
from matplotlib.patches import PathPatch
from matplotlib.text import Text

import dagua
from dagua.config import LayoutConfig
from dagua.edges import route_edges
from dagua.graph import DaguaGraph
from dagua.render import render
from dagua.render.mpl import _build_custom_edge_collection, _compute_display_scale
from dagua.render.text import layout_plain_text
from dagua.styles import ClusterStyle, EdgeStyle, NodeStyle


def _positions(points: Sequence[Tuple[float, float]]) -> torch.Tensor:
    """Build a float32 position tensor for render-only tests.

    Parameters
    ----------
    points : sequence[tuple[float, float]]
        Node center coordinates in data units.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.
    """
    return torch.tensor(points, dtype=torch.float32)


def _set_manual_node_sizes(
    graph: DaguaGraph,
    sizes: Sequence[Tuple[float, float]],
    font_sizes: Optional[Sequence[float]] = None,
) -> None:
    """Prime cached node sizes so render-time tests can control node boxes.

    Parameters
    ----------
    graph : DaguaGraph
        Graph whose size cache will be populated.
    sizes : sequence[tuple[float, float]]
        Manual node sizes in data units.
    font_sizes : sequence[float] | None, default=None
        Optional per-node effective font sizes in points.

    Returns
    -------
    None
        Mutates ``graph`` in place.
    """
    graph.node_sizes = torch.tensor(sizes, dtype=graph.size_dtype)
    graph._node_sizes_revision = graph.revision
    if font_sizes is None:
        graph.node_font_sizes = None
    else:
        graph.node_font_sizes = torch.tensor(font_sizes, dtype=torch.float32)


def _single_node_graph(
    label: str,
    style: NodeStyle,
    width: float = 40.0,
    height: float = 20.0,
) -> Tuple[DaguaGraph, torch.Tensor]:
    """Create a single-node graph with a fixed render box.

    Parameters
    ----------
    label : str
        Node label.
    style : NodeStyle
        Node style override.
    width : float, default=40.0
        Manual node width.
    height : float, default=20.0
        Manual node height.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph plus matching single-node position tensor.
    """
    graph = DaguaGraph()
    graph.add_node("a", label=label, style=style)
    _set_manual_node_sizes(graph, [(width, height)], font_sizes=[style.font_size])
    return graph, _positions([(0.0, 0.0)])


def _edge_graph(
    edge_label: str = "edge label",
    edge_style: Optional[EdgeStyle] = None,
) -> Tuple[DaguaGraph, torch.Tensor]:
    """Create a two-node graph with one labeled edge.

    Parameters
    ----------
    edge_label : str, default="edge label"
        Edge label text.
    edge_style : EdgeStyle | None, default=None
        Optional edge style override.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph plus two-node position tensor.
    """
    graph = DaguaGraph()
    graph.add_node("a", label="a")
    graph.add_node("b", label="b")
    graph.add_edge("a", "b", label=edge_label, style=edge_style)
    return graph, _positions([(0.0, 0.0), (60.0, 0.0)])


def _cluster_graph() -> Tuple[DaguaGraph, torch.Tensor]:
    """Create a graph with nested clusters for label layering tests.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph plus four-node position tensor.
    """
    graph = DaguaGraph()
    for node_id in ["a", "b", "c", "d"]:
        graph.add_node(node_id, label=node_id)
    graph.add_cluster(
        "outer",
        ["a", "b", "c", "d"],
        label="Outer",
        style=ClusterStyle(label_position="top-left"),
    )
    graph.add_cluster(
        "inner",
        ["b", "c"],
        parent="outer",
        label="Inner",
        style=ClusterStyle(label_position="top-left"),
    )
    return graph, _positions([(-30.0, 0.0), (-10.0, 0.0), (10.0, 0.0), (30.0, 0.0)])


def _render_graph(
    graph: DaguaGraph,
    positions: torch.Tensor,
    *,
    title: Optional[str] = None,
    output: Optional[str] = None,
    label_positions: Optional[List[Optional[Tuple[float, float]]]] = None,
) -> Tuple[Any, Any]:
    """Render a graph with deterministic manual positions.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to render.
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    title : str | None, default=None
        Optional graph title.
    output : str | None, default=None
        Optional output path.
    label_positions : list[tuple[float, float] | None] | None, default=None
        Optional explicit edge-label positions.

    Returns
    -------
    tuple[Any, Any]
        Matplotlib ``(figure, axes)``.
    """
    fig, ax = render(
        graph,
        positions=positions,
        title=title,
        output=output,
        label_positions=label_positions,
    )
    fig.canvas.draw()
    return fig, ax


def _patch_vertices(ax: Any, patch: PathPatch) -> torch.Tensor:
    """Return patch vertices converted back into data coordinates.

    Parameters
    ----------
    ax : Any
        Axes owning the patch.
    patch : PathPatch
        Patch to inspect.

    Returns
    -------
    torch.Tensor
        Vertex tensor with shape ``[N, 2]`` in data coordinates.
    """
    transformed = patch.get_transform().transform_path(patch.get_path())
    data_path = transformed.transformed(ax.transData.inverted())
    return torch.as_tensor(data_path.vertices, dtype=torch.float32)


def _bbox(ax: Any, patches: Sequence[PathPatch]) -> Tuple[float, float, float, float]:
    """Compute the union bounding box of patches in data coordinates.

    Parameters
    ----------
    ax : Any
        Axes owning the patches.
    patches : sequence[PathPatch]
        Patches to measure.

    Returns
    -------
    tuple[float, float, float, float]
        ``(min_x, max_x, min_y, max_y)``.
    """
    vertices = torch.cat([_patch_vertices(ax, patch) for patch in patches], dim=0)
    return (
        float(vertices[:, 0].min().item()),
        float(vertices[:, 0].max().item()),
        float(vertices[:, 1].min().item()),
        float(vertices[:, 1].max().item()),
    )


def _label_patches(ax: Any, prefix: str) -> List[PathPatch]:
    """Collect label patches whose gid starts with a prefix.

    Parameters
    ----------
    ax : Any
        Rendered axes.
    prefix : str
        Artist gid prefix.

    Returns
    -------
    list[PathPatch]
        Matching path patches.
    """
    patches: List[PathPatch] = []
    for patch in ax.patches:
        gid = patch.get_gid()
        if isinstance(patch, PathPatch) and isinstance(gid, str) and gid.startswith(prefix):
            patches.append(patch)
    return patches


def _fill_patches(ax: Any, prefix: str) -> List[PathPatch]:
    """Collect non-background, non-outline fill patches for one logical label.

    Parameters
    ----------
    ax : Any
        Rendered axes.
    prefix : str
        Artist gid prefix.

    Returns
    -------
    list[PathPatch]
        Matching glyph-fill patches.
    """
    return [
        patch
        for patch in _label_patches(ax, prefix)
        if not any(
            marker in str(patch.get_gid())
            for marker in ("background", "outline", "underline", "strikethrough")
        )
    ]


def _plain_block_size(ax: Any, text: str, style: NodeStyle) -> Tuple[float, float]:
    """Return the laid-out plain-text block size in data coordinates.

    Parameters
    ----------
    ax : Any
        Axes used to compute display scaling.
    text : str
        Label text.
    style : NodeStyle
        Node text style.

    Returns
    -------
    tuple[float, float]
        Block width and height in data units.
    """
    display_scale = _compute_display_scale(ax)
    block = layout_plain_text(
        text,
        size_data=style.font_size * display_scale,
        ha=style.text_align,
        va=style.text_valign,
        font_family=style.font_family_list[0],
        font_weight=style.font_weight,
        font_style=style.font_style,
        font_color=style.font_color,
        line_spacing=1.2,
        secondary_scale=1.0,
    )
    return block.width, block.height


def _expected_plain_bbox(
    ax: Any,
    text: str,
    style: NodeStyle,
    anchor_x: float,
    anchor_y: float,
) -> Tuple[float, float, float, float]:
    """Compute the expected plain-text glyph bbox for a given anchor point.

    Parameters
    ----------
    ax : Any
        Axes used to derive display scaling.
    text : str
        Plain label text.
    style : NodeStyle
        Node text style.
    anchor_x : float
        Text anchor x-coordinate in data units.
    anchor_y : float
        Text anchor y-coordinate in data units.

    Returns
    -------
    tuple[float, float, float, float]
        ``(min_x, max_x, min_y, max_y)`` in data coordinates.
    """
    display_scale = _compute_display_scale(ax)
    block = layout_plain_text(
        text,
        size_data=style.font_size * display_scale,
        ha=style.text_align,
        va=style.text_valign,
        font_family=style.font_family_list[0],
        font_weight=style.font_weight,
        font_style=style.font_style,
        font_color=style.font_color,
        line_spacing=1.2,
        secondary_scale=1.0,
    )
    vertices: List[torch.Tensor] = []
    for line in block.lines:
        for segment in line.segments:
            path_vertices = torch.as_tensor(segment.glyph_run.path.vertices, dtype=torch.float32)
            if path_vertices.numel() == 0:
                continue
            path_vertices = path_vertices + torch.tensor(
                [
                    anchor_x + block.x_offset + segment.x_offset,
                    anchor_y + block.y_offset + line.baseline_y,
                ],
                dtype=torch.float32,
            )
            vertices.append(path_vertices)
    assert vertices
    merged = torch.cat(vertices, dim=0)
    return (
        float(merged[:, 0].min().item()),
        float(merged[:, 0].max().item()),
        float(merged[:, 1].min().item()),
        float(merged[:, 1].max().item()),
    )


def _max_abs_alpha(values: Iterable[float]) -> float:
    """Return the maximum absolute alpha deviation in a sequence.

    Parameters
    ----------
    values : iterable[float]
        Alpha values to inspect.

    Returns
    -------
    float
        Maximum value or ``0.0`` when the iterable is empty.
    """
    collected = list(values)
    return max(collected) if collected else 0.0


def test_draw_no_ax_text(tmp_path: Path) -> None:
    """The renderer should emit no non-title matplotlib Text artists."""
    graph = DaguaGraph.from_edge_list([("a", "b")])
    fig, ax = dagua.draw(
        graph,
        config=LayoutConfig(steps=20, edge_opt_steps=-1, seed=42),
        title="Graph Title",
    )
    nonempty_texts = [
        artist.get_text()
        for artist in ax.get_children()
        if isinstance(artist, Text) and artist.get_text()
    ]

    assert nonempty_texts == ["Graph Title"]
    assert "ax.text(" not in Path("dagua/render/mpl.py").read_text(encoding="utf-8")
    assert "ax.text(" not in Path("dagua/render/edges/collection.py").read_text(encoding="utf-8")
    plt.close(fig)


def test_node_labels_are_pathpatch() -> None:
    """Node labels should render as path patches, not Text artists."""
    graph, positions = _single_node_graph("Node", NodeStyle())
    fig, ax = _render_graph(graph, positions)
    patches = _label_patches(ax, "dagua-node-label-0")

    assert patches
    assert all(isinstance(patch, PathPatch) for patch in patches)
    plt.close(fig)


def test_node_label_left_align() -> None:
    """Left-aligned node labels should move to the padded left anchor."""
    left_style = NodeStyle(text_align="left")
    graph, positions = _single_node_graph("Label", left_style)
    fig, ax = _render_graph(graph, positions)
    display_scale = _compute_display_scale(ax)
    actual_bbox = _bbox(ax, _fill_patches(ax, "dagua-node-label-0"))
    expected_bbox = _expected_plain_bbox(
        ax,
        "Label",
        left_style,
        anchor_x=-20.0 + 10.0 * display_scale,
        anchor_y=0.0,
    )

    assert actual_bbox[0] == pytest.approx(expected_bbox[0], abs=0.75)
    plt.close(fig)


def test_node_label_right_align() -> None:
    """Right-aligned node labels should move to the padded right anchor."""
    right_style = NodeStyle(text_align="right")
    graph, positions = _single_node_graph("Label", right_style)
    fig, ax = _render_graph(graph, positions)
    display_scale = _compute_display_scale(ax)
    actual_bbox = _bbox(ax, _fill_patches(ax, "dagua-node-label-0"))
    expected_bbox = _expected_plain_bbox(
        ax,
        "Label",
        right_style,
        anchor_x=20.0 - 10.0 * display_scale,
        anchor_y=0.0,
    )

    assert actual_bbox[1] == pytest.approx(expected_bbox[1], abs=0.75)
    plt.close(fig)


def test_node_label_top_align() -> None:
    """Top-aligned node labels should move to the padded top anchor."""
    top_style = NodeStyle(text_valign="top")
    graph, positions = _single_node_graph("Label", top_style)
    fig, ax = _render_graph(graph, positions)
    display_scale = _compute_display_scale(ax)
    actual_bbox = _bbox(ax, _fill_patches(ax, "dagua-node-label-0"))
    expected_bbox = _expected_plain_bbox(
        ax,
        "Label",
        top_style,
        anchor_x=0.0,
        anchor_y=10.0 - 6.0 * display_scale,
    )

    assert actual_bbox[3] == pytest.approx(expected_bbox[3], abs=0.75)
    plt.close(fig)


def test_node_label_bottom_align() -> None:
    """Bottom-aligned node labels should move to the padded bottom anchor."""
    bottom_style = NodeStyle(text_valign="bottom")
    graph, positions = _single_node_graph("Label", bottom_style)
    fig, ax = _render_graph(graph, positions)
    display_scale = _compute_display_scale(ax)
    actual_bbox = _bbox(ax, _fill_patches(ax, "dagua-node-label-0"))
    expected_bbox = _expected_plain_bbox(
        ax,
        "Label",
        bottom_style,
        anchor_x=0.0,
        anchor_y=-10.0 + 6.0 * display_scale,
    )

    assert actual_bbox[2] == pytest.approx(expected_bbox[2], abs=0.75)
    plt.close(fig)


def test_triangle_label_y() -> None:
    """Triangle labels should shift downward by one eighth of node height."""
    rect_graph, positions = _single_node_graph("Node", NodeStyle(shape="roundrect"))
    triangle_graph, _ = _single_node_graph("Node", NodeStyle(shape="triangle"))
    rect_fig, rect_ax = _render_graph(rect_graph, positions)
    triangle_fig, triangle_ax = _render_graph(triangle_graph, positions)
    rect_bbox = _bbox(rect_ax, _fill_patches(rect_ax, "dagua-node-label-0"))
    triangle_bbox = _bbox(triangle_ax, _fill_patches(triangle_ax, "dagua-node-label-0"))
    rect_center_y = (rect_bbox[2] + rect_bbox[3]) / 2.0
    triangle_center_y = (triangle_bbox[2] + triangle_bbox[3]) / 2.0

    assert triangle_center_y - rect_center_y == pytest.approx(-(20.0 / 8.0), abs=0.5)
    plt.close(rect_fig)
    plt.close(triangle_fig)


def test_rich_no_secondary_scale() -> None:
    """Rich multiline node labels should keep later lines at full size."""
    graph, positions = _single_node_graph("MMMM\nMMMM", NodeStyle(label_format="rich"))
    fig, ax = _render_graph(graph, positions)
    first_line_bbox = _bbox(ax, _fill_patches(ax, "dagua-node-label-0-0-"))
    second_line_bbox = _bbox(ax, _fill_patches(ax, "dagua-node-label-0-1-"))
    first_height = first_line_bbox[3] - first_line_bbox[2]
    second_height = second_line_bbox[3] - second_line_bbox[2]

    assert second_height / first_height == pytest.approx(1.0, abs=0.12)
    plt.close(fig)


def test_plain_multiline_secondary_scale() -> None:
    """Plain multiline node labels should scale secondary lines down."""
    graph, positions = _single_node_graph("MMMM\nMMMM", NodeStyle(label_format="plain"))
    fig, ax = _render_graph(graph, positions)
    first_line_bbox = _bbox(ax, _fill_patches(ax, "dagua-node-label-0-0"))
    second_line_bbox = _bbox(ax, _fill_patches(ax, "dagua-node-label-0-1"))
    first_height = first_line_bbox[3] - first_line_bbox[2]
    second_height = second_line_bbox[3] - second_line_bbox[2]

    assert second_height < first_height
    assert second_height / first_height == pytest.approx(
        graph.graph_style.node_label_secondary_scale,
        abs=0.15,
    )
    plt.close(fig)


def test_overflow_shrink() -> None:
    """Shrink-to-fit should reduce label width inside a constrained node box."""
    shrink_style = NodeStyle(font_size=18.0, padding=(4.0, 4.0), overflow_policy="shrink_text")
    overflow_style = NodeStyle(font_size=18.0, padding=(4.0, 4.0), overflow_policy="overflow")
    shrink_graph, positions = _single_node_graph(
        "This is a very long label",
        shrink_style,
        width=24.0,
        height=18.0,
    )
    overflow_graph, _ = _single_node_graph(
        "This is a very long label",
        overflow_style,
        width=24.0,
        height=18.0,
    )
    shrink_fig, shrink_ax = _render_graph(shrink_graph, positions)
    overflow_fig, overflow_ax = _render_graph(overflow_graph, positions)
    display_scale = _compute_display_scale(shrink_ax)
    available_width = 24.0 - 2.0 * 4.0 * display_scale
    shrink_bbox = _bbox(shrink_ax, _fill_patches(shrink_ax, "dagua-node-label-0"))
    overflow_bbox = _bbox(overflow_ax, _fill_patches(overflow_ax, "dagua-node-label-0"))

    assert (shrink_bbox[1] - shrink_bbox[0]) < (overflow_bbox[1] - overflow_bbox[0])
    assert (shrink_bbox[1] - shrink_bbox[0]) <= available_width + 0.75
    plt.close(shrink_fig)
    plt.close(overflow_fig)


def test_text_outline_creates_stroke() -> None:
    """Outlined node labels should add separate stroke patches."""
    graph, positions = _single_node_graph("Node", NodeStyle(text_outline=True))
    fig, ax = _render_graph(graph, positions)

    assert any(
        "outline" in str(patch.get_gid()) for patch in _label_patches(ax, "dagua-node-label-0")
    )
    plt.close(fig)


def test_edge_labels_clip_off() -> None:
    """Edge-label patches should preserve matplotlib's unclipped text behavior."""
    graph, positions = _edge_graph()
    fig, ax = _render_graph(graph, positions)

    assert _label_patches(ax, "dagua-edge-label-0")
    assert all(not patch.get_clip_on() for patch in _label_patches(ax, "dagua-edge-label-0"))
    plt.close(fig)


def test_edge_label_background() -> None:
    """Edge labels should render background rectangles via the text module."""
    graph, positions = _edge_graph()
    fig, ax = _render_graph(graph, positions)
    background_patches = [
        patch
        for patch in _label_patches(ax, "dagua-edge-label-0")
        if str(patch.get_gid()).endswith("background")
    ]

    assert background_patches
    plt.close(fig)


def test_cluster_labels_clip_off() -> None:
    """Cluster label patches should render without axes clipping."""
    graph, positions = _cluster_graph()
    fig, ax = _render_graph(graph, positions)
    patches = _label_patches(ax, "dagua-cluster-label-outer")

    assert patches
    assert all(not patch.get_clip_on() for patch in patches)
    plt.close(fig)


def test_cluster_depth_zorder() -> None:
    """Nested cluster labels should increase z-order with depth."""
    graph, positions = _cluster_graph()
    fig, ax = _render_graph(graph, positions)
    outer_zorder = max(
        patch.get_zorder() for patch in _label_patches(ax, "dagua-cluster-label-outer")
    )
    inner_zorder = max(
        patch.get_zorder() for patch in _label_patches(ax, "dagua-cluster-label-inner")
    )

    assert inner_zorder > outer_zorder
    plt.close(fig)


def test_svg_hover_on_labeled_edge(tmp_path: Path) -> None:
    """SVG export should keep hover metadata for edge labels and backgrounds."""
    graph, positions = _edge_graph(edge_label="edge label")
    output = tmp_path / "edge-labels.svg"
    fig, _ = _render_graph(graph, positions, output=str(output))
    content = output.read_text(encoding="utf-8")

    assert "dagua-edge-label-0-background" in content
    assert "<title>a -&gt; b: edge label</title>" in content
    plt.close(fig)


def test_node_label_opacity_independent() -> None:
    """Node text should remain fully opaque when node fills are translucent."""
    graph, positions = _single_node_graph("Node", NodeStyle(opacity=0.5))
    fig, ax = _render_graph(graph, positions)
    alphas: List[float] = []
    for patch in _fill_patches(ax, "dagua-node-label-0"):
        alpha = patch.get_alpha()
        alphas.append(1.0 if alpha is None else float(alpha))

    assert _max_abs_alpha(alphas) == pytest.approx(1.0)
    plt.close(fig)


def test_empty_labels_skipped() -> None:
    """Nodes with empty labels should not create label patches."""
    graph = DaguaGraph()
    graph.add_node("a", label="", style=NodeStyle())
    graph.add_node("b", label="Shown", style=NodeStyle())
    _set_manual_node_sizes(graph, [(30.0, 18.0), (30.0, 18.0)], font_sizes=[9.0, 9.0])
    positions = _positions([(-20.0, 0.0), (20.0, 0.0)])
    fig, ax = _render_graph(graph, positions)

    assert not _label_patches(ax, "dagua-node-label-0")
    assert _label_patches(ax, "dagua-node-label-1")
    plt.close(fig)


def test_edge_collection_labels_textpath() -> None:
    """The edge collection label pass should return path patches, not Text artists."""
    graph, positions = _edge_graph()
    graph.compute_node_sizes()
    assert graph.node_sizes is not None
    curves = route_edges(positions, graph.edge_index, graph.node_sizes, graph.direction, graph)
    fig, ax = plt.subplots()
    ax.set_xlim(-20.0, 80.0)
    ax.set_ylim(-30.0, 30.0)
    ax.set_aspect("equal")
    fig.canvas.draw()
    collection = _build_custom_edge_collection(ax, graph, curves)
    artists = collection.render_labels(ax, display_scale=_compute_display_scale(ax))

    assert artists
    assert all(isinstance(artist, PathPatch) for artist in artists)
    plt.close(fig)
