"""Regression tests for data-coordinate node and cluster borders."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from matplotlib.collections import PatchCollection

from dagua.graph import DaguaGraph
from dagua.render import render
from dagua.render.borders import (
    ShapeSpec,
    annular_path,
    build_shape_path,
    clamp_border_width,
    dash_ribbon_paths,
    inset_shape_path,
    inset_star,
    path_to_closed_vertices,
    reverse_closed_path,
    star_vertices,
)
from dagua.render.mpl import _compute_display_scale, _draw_nodes, _scaled_node_style
from dagua.styles import NodeStyle


def _signed_area(vertices: np.ndarray) -> float:
    """Return the signed polygon area for one closed polyline.

    Parameters
    ----------
    vertices : numpy.ndarray
        Polygon vertices with shape ``[N, 2]``.

    Returns
    -------
    float
        Signed polygon area.
    """

    rolled = np.roll(vertices, -1, axis=0)
    return 0.5 * float(np.sum(vertices[:, 0] * rolled[:, 1] - vertices[:, 1] * rolled[:, 0]))


def test_reverse_closed_path_flips_winding_for_smooth_paths() -> None:
    """Reversing an ellipse path should negate its winding without flattening it."""

    outer = build_shape_path(ShapeSpec(0.0, 0.0, 40.0, 24.0, "ellipse"))
    reversed_path = reverse_closed_path(outer)

    assert _signed_area(path_to_closed_vertices(outer)[:-1]) == pytest.approx(
        -_signed_area(path_to_closed_vertices(reversed_path)[:-1])
    )
    assert set(reversed_path.codes.tolist()) == set(outer.codes.tolist())


def test_annular_path_contains_both_outer_and_inner_subpaths() -> None:
    """Solid borders should be represented as one compound outer-plus-inner ring."""

    spec = ShapeSpec(0.0, 0.0, 60.0, 36.0, "roundrect", corner_radius=6.0)
    outer = build_shape_path(spec)
    inner = inset_shape_path(spec, 4.0)
    ring = annular_path(outer, inner)

    move_count = int(np.count_nonzero(ring.codes == ring.MOVETO))
    close_count = int(np.count_nonzero(ring.codes == ring.CLOSEPOLY))

    assert move_count == 2
    assert close_count == 2


def test_star_inset_uses_uniform_centroid_scaling() -> None:
    """Star inset should shrink every vertex radially toward the centroid."""

    vertices = star_vertices(0.0, 0.0, 100.0, 100.0)
    inset_vertices = inset_star(vertices, 8.0)
    centroid = np.mean(vertices, axis=0)
    original_radii = np.linalg.norm(vertices - centroid, axis=1)
    inset_radii = np.linalg.norm(inset_vertices - centroid, axis=1)

    assert np.all(inset_radii < original_radii)
    assert np.min(inset_radii / original_radii) > 0.1


def test_border_width_clamps_to_forty_percent_fraction() -> None:
    """Border width should clamp silently before it erases the fill area."""

    assert clamp_border_width(100.0, width=20.0, height=10.0) == pytest.approx(2.0)


def test_dash_ribbon_paths_follow_closed_perimeter() -> None:
    """Dashed borders should be emitted as closed filled ribbon segments."""

    spec = ShapeSpec(0.0, 0.0, 80.0, 40.0, "rect")
    centerline = inset_shape_path(spec, 2.0)
    ribbons = dash_ribbon_paths(centerline, "dashed", width=4.0)

    assert ribbons
    assert all(path.codes[0] == path.MOVETO for path in ribbons)
    assert all(path.codes[-1] == path.CLOSEPOLY for path in ribbons)


def test_scaled_node_style_converts_corner_radius_and_shadow_offset() -> None:
    """Node rounded corners and shadows should use the axes display scale."""

    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=100)
    ax.set_xlim(-50.0, 50.0)
    ax.set_ylim(-50.0, 50.0)
    fig.canvas.draw()

    style = NodeStyle(corner_radius=6.0, shadow_offset=(1.5, -1.5))
    display_scale = _compute_display_scale(ax)
    scaled = _scaled_node_style(style, display_scale)
    plt.close(fig)

    assert scaled.corner_radius == pytest.approx(6.0 * display_scale)
    assert scaled.shadow_offset == pytest.approx((1.5 * display_scale, -1.5 * display_scale))


def test_draw_nodes_uses_batched_fill_and_border_collections() -> None:
    """Node rendering should batch fills and border geometry into collections."""

    graph = DaguaGraph()
    graph.add_node("a", style=NodeStyle(shape="roundrect", stroke_width=1.0))
    graph.add_node("b", style=NodeStyle(shape="star", stroke_dash="dashed", stroke_width=1.0))
    pos = np.array([[0.0, 0.0], [80.0, 0.0]], dtype=np.float64)
    sizes = np.array([[40.0, 24.0], [40.0, 40.0]], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(4.0, 3.0), dpi=100)
    ax.set_xlim(-40.0, 120.0)
    ax.set_ylim(-40.0, 40.0)
    ax.set_aspect("equal")
    fig.canvas.draw()
    clip_patches = _draw_nodes(ax=ax, graph=graph, pos=pos, sizes=sizes)

    node_collections = [
        collection for collection in ax.collections if isinstance(collection, PatchCollection)
    ]
    plt.close(fig)

    assert len(clip_patches) == 2
    assert len(node_collections) == 2
    assert sorted(float(collection.get_zorder()) for collection in node_collections) == [2.0, 2.05]


def test_render_keeps_thick_transparent_borders_as_true_rings() -> None:
    """Semi-transparent thick borders should still render through collections."""

    graph = DaguaGraph()
    graph.add_node(
        "a",
        style=NodeStyle(
            shape="ellipse",
            stroke_width=4.0,
            border_opacity=0.4,
            fill="#cfd8dc",
            stroke="#37474f",
        ),
    )
    positions = torch.tensor([[0.0, 0.0]])

    fig, ax = render(graph, positions)
    border_collections = [
        collection
        for collection in ax.collections
        if isinstance(collection, PatchCollection) and float(collection.get_zorder()) == 2.05
    ]
    fig.canvas.draw()
    plt.close(fig)

    assert len(border_collections) == 1
    assert len(border_collections[0].get_paths()) == 1
