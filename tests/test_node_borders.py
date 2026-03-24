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
from dagua.render.borders.dashes import _curvature_scale, _estimate_curvatures, dash_segments
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


def _polyline_length(points: np.ndarray) -> float:
    """Return the cumulative Euclidean length of one sampled polyline.

    Parameters
    ----------
    points : numpy.ndarray
        Ordered polyline vertices with shape ``[N, 2]``.

    Returns
    -------
    float
        Total length of the piecewise-linear path.
    """

    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


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


def test_estimate_curvatures_straight_line_returns_zeros() -> None:
    """Collinear vertices should not report curvature."""

    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
        ],
        dtype=np.float64,
    )

    curvatures = _estimate_curvatures(points)

    assert np.allclose(curvatures, 0.0)


def test_estimate_curvatures_circle_returns_uniform() -> None:
    """A regular polygon approximation of a circle should have uniform curvature."""

    angles = np.linspace(0.0, 2.0 * np.pi, num=32, endpoint=False)
    radius = 10.0
    circle_points = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    closed_points = np.vstack([circle_points, circle_points[0]])

    curvatures = _estimate_curvatures(closed_points)

    assert np.all(curvatures > 0.0)
    assert np.max(curvatures) == pytest.approx(np.min(curvatures), rel=1e-6)


def test_curvature_scale_straight_returns_one() -> None:
    """Zero curvature should leave the dash length unchanged."""

    assert _curvature_scale(0.0) == pytest.approx(1.0)


def test_curvature_scale_tight_curve_reduces() -> None:
    """Tight curves should shorten the visible dash length."""

    assert _curvature_scale(0.5) < 1.0
    assert _curvature_scale(10.0) == pytest.approx(0.4)


def test_dashed_cylinder_cap_has_shorter_segments() -> None:
    """Cylinder caps should shorten visible dashes relative to straight side walls."""

    spec = ShapeSpec(0.0, 0.0, 80.0, 100.0, "cylinder")
    centerline = inset_shape_path(spec, 2.0)
    segments = dash_segments(centerline, "dashed", width=4.0)

    side_lengths = [
        _polyline_length(segment.points)
        for segment in segments
        if abs(float(np.mean(segment.points[:, 0]))) > 35.0
        and abs(float(np.mean(segment.points[:, 1]))) < 25.0
    ]
    cap_lengths = [
        _polyline_length(segment.points)
        for segment in segments
        if abs(float(np.mean(segment.points[:, 1]))) > 30.0
    ]

    assert side_lengths
    assert cap_lengths
    assert np.mean(cap_lengths) < np.mean(side_lengths) * 0.95


def test_inset_shape_path_handles_non_polygon_shapes() -> None:
    """Non-polygon shapes should produce an inset path without crashing."""

    for shape in (
        "cloud",
        "stadium",
        "semicircle",
        "semicircle_down",
        "semicircle_left",
        "semicircle_right",
        "document",
        "tab",
        "note",
        "box3d",
    ):
        spec = ShapeSpec(
            center_x=0.0,
            center_y=0.0,
            width=100.0,
            height=80.0,
            shape=shape,
            corner_radius=0.0,
        )

        path = inset_shape_path(spec, 5.0)

        assert path is not None
        assert len(path.vertices) > 0


@pytest.mark.parametrize(
    ("shape", "flat_points"),
    [
        ("semicircle", np.array([[-20.0, -15.0], [20.0, -15.0]], dtype=np.float64)),
        ("semicircle_down", np.array([[-20.0, 15.0], [20.0, 15.0]], dtype=np.float64)),
        ("semicircle_left", np.array([[20.0, 15.0], [20.0, -15.0]], dtype=np.float64)),
        ("semicircle_right", np.array([[-20.0, 15.0], [-20.0, -15.0]], dtype=np.float64)),
    ],
)
def test_semicircle_paths_place_their_flat_edge_on_the_requested_side(
    shape: str,
    flat_points: np.ndarray,
) -> None:
    """Semicircle variants should orient their flat edge on the requested side."""

    path = build_shape_path(ShapeSpec(0.0, 0.0, 40.0, 30.0, shape))
    observed = np.asarray(path.vertices[:2], dtype=np.float64)

    assert observed == pytest.approx(flat_points)


def test_semicircle_path_uses_aspect_ratio_to_shallow_the_dome() -> None:
    """Larger semicircle aspect ratios should flatten the dome within the bounds."""

    default_path = build_shape_path(
        ShapeSpec(0.0, 0.0, 80.0, 60.0, "semicircle", aspect_ratio=None)
    )
    shallow_path = build_shape_path(ShapeSpec(0.0, 0.0, 80.0, 60.0, "semicircle", aspect_ratio=2.0))

    default_top = float(np.max(default_path.vertices[:, 1]))
    shallow_top = float(np.max(shallow_path.vertices[:, 1]))

    assert shallow_top < default_top


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
