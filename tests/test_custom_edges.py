"""Tests for the custom edge rendering package."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from matplotlib.collections import LineCollection, PatchCollection

from dagua.graph import DaguaGraph
from dagua.render.borders.shapes import NOTE_FOLD_SIZE_RATIO, ShapeSpec, note_path
from dagua.render.edges import available_arrowheads, build_arrowhead
from dagua.render.edges.collection import (
    MIN_TAPER_WIDTH,
    DaguaEdge,
    DaguaEdgeCollection,
    _head_body_direction,
    _redistribute_face_angles,
    _stroked_head_linewidth,
    _terminal_angle,
    _terminal_face,
    _trimmed_body_curve,
    choose_rendering_tier,
)
from dagua.render.edges.dashes import dash_curve, parse_dash_pattern
from dagua.render.edges.geometry import (
    CubicBezier,
    adaptive_subdivide,
    build_arc_length_table,
    offset_cubic_control_points,
    validate_lane_separation,
)
from dagua.render.edges.intersection import intersect_node_boundary
from dagua.render.edges.labels import place_edge_label
from dagua.render.edges.ribbon import curve_ribbon_path
from dagua.render.mpl import _build_custom_edge_collection, render
from dagua.styles import EdgeStyle


def _curve() -> CubicBezier:
    """Return a representative curved cubic bezier."""
    return CubicBezier.from_points((0.0, 0.0), (20.0, 25.0), (40.0, -25.0), (60.0, 0.0))


def _minimum_angular_gap(angles: list[float]) -> float:
    """Return the tightest circular gap between sorted angles.

    Parameters
    ----------
    angles : list[float]
        Angles in degrees on ``[0, 360)``.

    Returns
    -------
    float
        Smallest wrapped gap in degrees.
    """
    wrapped = np.sort(np.mod(np.asarray(angles, dtype=np.float64), 360.0))
    extended = np.concatenate([wrapped, wrapped[:1] + 360.0])
    return float(np.min(np.diff(extended)))


def test_adaptive_subdivision_refines_curved_edges() -> None:
    """Curved edges should subdivide more than straight edges."""
    straight = CubicBezier.from_points((0.0, 0.0), (20.0, 0.0), (40.0, 0.0), (60.0, 0.0))
    curved_samples = adaptive_subdivide(_curve(), flatness=0.5)
    straight_samples = adaptive_subdivide(straight, flatness=0.5)

    assert len(curved_samples) > len(straight_samples)
    assert len(curved_samples) >= 40
    assert len(straight_samples) == 2


def test_ribbon_path_is_closed_and_round_caps_add_vertices() -> None:
    """Round-capped ribbons should produce a closed polygon with extra detail."""
    butt = curve_ribbon_path(_curve(), width=4.0, cap_start="butt", cap_end="butt")
    rounded = curve_ribbon_path(_curve(), width=4.0, cap_start="round", cap_end="round")

    assert butt.codes is not None
    assert rounded.codes is not None
    assert butt.codes[-1] == rounded.codes[-1]
    assert butt.codes[-1] == 79  # Path.CLOSEPOLY
    assert rounded.vertices.shape[0] > butt.vertices.shape[0]


def test_curve_ribbon_path_uses_dense_curved_sampling() -> None:
    """Curved ribbon paths should keep enough vertices to hide the polyline scaffold."""
    refined = curve_ribbon_path(_curve(), width=4.0)

    assert refined.vertices.shape[0] >= 81


def test_tapered_edge_collection_keeps_visible_endpoint_width() -> None:
    """Tapered collection ribbons should floor the thin endpoint width."""

    edge = DaguaEdge(
        curve=_curve(),
        tapered=True,
        taper_width_start=4.0,
        taper_width_end=0.05,
        color="#336699",
        alpha=1.0,
    )
    collection = DaguaEdgeCollection([edge], tier="full")
    fig, ax = plt.subplots()

    artists = collection.render_bodies(ax)

    assert len(artists) == 1
    patch_collection = artists[0]
    path = patch_collection.get_paths()[0]
    point_count = (path.vertices.shape[0] - 1) // 2
    tip_width = float(np.linalg.norm(path.vertices[point_count - 1] - path.vertices[point_count]))

    assert tip_width == pytest.approx(MIN_TAPER_WIDTH, abs=1e-6)
    plt.close(fig)


def test_dash_curve_uses_round_caps_per_segment() -> None:
    """Dashed segments should use butt caps to avoid pill-shaped terminals."""
    segments = dash_curve(_curve(), "dashed", width=3.0)

    assert len(segments) >= 2
    assert all(segment.cap_start == "butt" for segment in segments)
    assert all(segment.cap_end == "butt" for segment in segments)


def test_dotted_dash_curve_uses_short_round_segments() -> None:
    """Dotted edges should render as circular dots instead of short capsules."""
    curve = CubicBezier.from_points((0.0, 0.0), (6.0, 0.0), (12.0, 0.0), (18.0, 0.0))
    width = 3.0

    segments = dash_curve(curve, "dotted", width=width)

    assert segments
    first_length = build_arc_length_table(segments[0].curve).total_length
    dot_path = curve_ribbon_path(
        segments[0].curve,
        width=width,
        cap_start=segments[0].cap_start,
        cap_end=segments[0].cap_end,
    )
    dot_vertices = dot_path.vertices
    dot_width = float(dot_vertices[:, 0].max() - dot_vertices[:, 0].min())
    dot_height = float(dot_vertices[:, 1].max() - dot_vertices[:, 1].min())

    assert first_length < width * 0.25
    assert dot_width == pytest.approx(width, rel=0.20)
    assert dot_height == pytest.approx(width, rel=0.20)
    assert dot_width / dot_height == pytest.approx(1.0, rel=0.20)


def test_dash_curve_drops_truncated_terminal_dash() -> None:
    """A final short dash should be omitted instead of ending awkwardly at the head."""
    curve = CubicBezier.from_points((0.0, 0.0), (6.0, 0.0), (12.0, 0.0), (18.0, 0.0))

    segments = dash_curve(curve, "dashed", width=2.0)

    assert len(segments) == 1


def test_dash_curve_aligns_terminal_dash_with_arrowhead_join() -> None:
    """Aligned dash patterns should finish on a full dash, not a final gap or dot."""
    curve = CubicBezier.from_points((0.0, 0.0), (20.0, 12.0), (40.0, -12.0), (60.0, 0.0))

    segments = dash_curve(curve, "dashdot", width=3.0, align_to_end=True)
    lengths = [build_arc_length_table(segment.curve).total_length for segment in segments]

    assert segments
    assert np.allclose(segments[-1].curve.p1, curve.p1)
    assert segments[-1].cap_end == "butt"
    assert lengths[-1] == pytest.approx(max(lengths), rel=0.05)
    assert min(lengths) < max(lengths) * 0.2


def test_dashdot_uses_distinct_dash_and_dot_caps() -> None:
    """Dashdot should keep circular dots clearly separated from longer dashes."""
    curve = CubicBezier.from_points((0.0, 0.0), (18.0, 10.0), (36.0, -10.0), (54.0, 0.0))
    width = 2.5

    segments = dash_curve(curve, "dashdot", width=width)
    lengths = [build_arc_length_table(segment.curve).total_length for segment in segments]
    dash_lengths = [
        length
        for length, segment in zip(lengths, segments)
        if segment.cap_start == "butt" and segment.cap_end == "butt"
    ]
    dot_lengths = [
        length
        for length, segment in zip(lengths, segments)
        if segment.cap_start == "round" and segment.cap_end == "round"
    ]
    dot_paths = [
        curve_ribbon_path(
            segment.curve,
            width=width,
            cap_start=segment.cap_start,
            cap_end=segment.cap_end,
        )
        for segment in segments
        if segment.cap_start == "round" and segment.cap_end == "round"
    ]

    assert segments
    assert dash_lengths
    assert dot_lengths
    assert dot_paths
    assert max(dash_lengths) == pytest.approx(width * 5.0, rel=0.08)
    assert max(dot_lengths) < width * 0.05
    dot_vertices = dot_paths[0].vertices
    dot_width = float(dot_vertices[:, 0].max() - dot_vertices[:, 0].min())
    dot_height = float(dot_vertices[:, 1].max() - dot_vertices[:, 1].min())
    assert dot_width == pytest.approx(width, rel=0.20)
    assert dot_height == pytest.approx(width, rel=0.20)
    assert dot_width / dot_height == pytest.approx(1.0, rel=0.20)


def test_thick_dash_patterns_expand_gaps_for_readability() -> None:
    """Dash patterns should preserve the intended line-style rhythm at thick widths."""
    dash_on, dash_off = parse_dash_pattern("dashed", width=6.0)
    dashdot_on, dashdot_gap, dot_on, dot_gap = parse_dash_pattern("dashdot", width=6.0)

    assert dash_on < 24.0
    assert dash_off > 16.5
    assert dashdot_on == pytest.approx(30.0)
    assert dashdot_gap == pytest.approx(18.0)
    assert dot_on < 0.5
    assert dot_gap == pytest.approx(18.0)


@pytest.mark.parametrize(
    ("angle_degrees", "expected_face"),
    [
        (0.0, "east"),
        (45.0, "northeast"),
        (90.0, "north"),
        (135.0, "northwest"),
        (180.0, "west"),
        (225.0, "southwest"),
        (270.0, "south"),
        (315.0, "southeast"),
    ],
)
def test_terminal_face_returns_8_directions(angle_degrees: float, expected_face: str) -> None:
    """Terminal faces should resolve to all eight directional sectors."""
    angle_radians = np.deg2rad(angle_degrees)
    direction = np.array([np.cos(angle_radians), np.sin(angle_radians)], dtype=np.float64)

    assert _terminal_face(direction) == expected_face


def test_redistribute_face_angles_spreads_evenly() -> None:
    """Crowded faces should spread members across the interior of one sector."""
    redistributed = _redistribute_face_angles(
        [(0, 170.0), (1, 175.0), (2, 180.0), (3, 185.0), (4, 190.0)],
        face_center_angle=180.0,
    )

    assert [edge_index for edge_index, _ in redistributed] == [0, 1, 2, 3, 4]
    assert [angle for _, angle in redistributed] == pytest.approx(
        [160.0, 170.0, 180.0, 190.0, 200.0]
    )


@pytest.mark.parametrize("spec", ["normal", "dot", "diamond", "vee", "crow", "box", "simple"])
def test_arrowhead_result_separates_filled_and_stroked_geometry(spec: str) -> None:
    """Arrowheads should report fill geometry separately from stroke geometry."""
    result = build_arrowhead(spec, tip=(0.0, 0.0), tangent=(-1.0, 0.0), length=8.0, width=5.0)

    assert result.trim_contour.vertices.shape[0] >= 2
    # All standard arrowheads produce filled geometry.
    assert len(result.filled_paths) >= 1


def test_open_arrowhead_becomes_stroked() -> None:
    """Open modifiers should route fill geometry into the stroked pass."""
    result = build_arrowhead("onormal", tip=(0.0, 0.0), tangent=(-1.0, 0.0), length=8.0, width=5.0)

    assert result.filled_paths == []
    assert len(result.stroked_paths) >= 1


def test_graphviz_open_arrowhead_is_a_stroked_v_shape() -> None:
    """Graphviz's ``open`` head should render as two stroked tines with no fill.

    Returns
    -------
    None
        This test only performs assertions.
    """
    result = build_arrowhead("open", tip=(0.0, 0.0), tangent=(-1.0, 0.0), length=8.0, width=5.0)

    assert result.filled_paths == []
    assert len(result.stroked_paths) == 2
    for path in result.stroked_paths:
        assert path.vertices.shape == (2, 2)
        assert np.allclose(path.vertices[0], np.array([0.0, 0.0]))
        assert path.vertices[1, 0] < 0.0


def test_hollow_arrowheads_gain_extra_size_for_visual_weight() -> None:
    """Hollow heads should scale up so they do not look undersized beside filled ones."""
    filled = build_arrowhead(
        "normal",
        tip=(0.0, 0.0),
        tangent=(1.0, 0.0),
        length=8.0,
        width=5.0,
        body_width=3.0,
        fill_mode="filled",
    )
    hollow = build_arrowhead(
        "normal",
        tip=(0.0, 0.0),
        tangent=(1.0, 0.0),
        length=8.0,
        width=5.0,
        body_width=3.0,
        fill_mode="hollow",
    )

    filled_extent = float(np.max(filled.filled_paths[0].vertices[:, 0]))
    hollow_extent = float(np.max(hollow.stroked_paths[0].vertices[:, 0]))

    assert hollow_extent > filled_extent * 1.25


def test_arrowhead_neck_matches_body_width_and_overlaps_body() -> None:
    """Filled heads should trim at the ribbon width and overlap slightly into the body."""
    body_direction = np.array([-1.0, 0.0], dtype=np.float64)
    tip = np.array([0.0, 0.0], dtype=np.float64)
    result = build_arrowhead(
        "normal",
        tip=tip,
        tangent=body_direction,
        length=8.0,
        width=6.0,
        body_width=4.0,
    )

    trim_vertices = result.trim_contour.vertices[:2]
    trim_width = np.linalg.norm(trim_vertices[0] - trim_vertices[1])
    trim_depth = np.dot(trim_vertices.mean(axis=0) - tip, body_direction)
    max_depth = max(
        np.dot(vertex - tip, body_direction) for vertex in result.filled_paths[0].vertices
    )

    assert trim_width == pytest.approx(4.0)
    assert max_depth > trim_depth


def test_open_and_hollow_arrowheads_increase_stroke_weight() -> None:
    """Open and hollow heads should request heavier outline strokes."""
    vee = build_arrowhead(
        "vee",
        tip=(0.0, 0.0),
        tangent=(-1.0, 0.0),
        length=8.0,
        width=5.0,
        body_width=3.0,
    )
    hollow = build_arrowhead(
        "onormal",
        tip=(0.0, 0.0),
        tangent=(-1.0, 0.0),
        length=8.0,
        width=5.0,
        body_width=3.0,
    )

    assert vee.stroke_width_scale > 1.0
    assert hollow.stroke_width_scale > 1.0


def test_vee_arrowhead_seats_on_full_body_width() -> None:
    """Filled vee heads should still anchor on the full ribbon width."""
    result = build_arrowhead(
        "vee",
        tip=(0.0, 0.0),
        tangent=(1.0, 0.0),
        length=10.0,
        width=8.0,
        body_width=6.0,
    )

    vertices = result.filled_paths[0].vertices
    max_x = float(np.max(vertices[:, 0]))
    anchor_values = sorted(
        float(vertex[1]) for vertex in vertices if vertex[0] == pytest.approx(max_x)
    )

    assert anchor_values[0] == pytest.approx(-anchor_values[1])
    assert abs(anchor_values[0]) > 3.0


def test_tee_arrowhead_uses_bolder_crossbar_than_the_ribbon_body() -> None:
    """Tee heads should read as a wide bar instead of disappearing into the edge."""
    result = build_arrowhead(
        "tee",
        tip=(0.0, 0.0),
        tangent=(1.0, 0.0),
        length=8.0,
        width=10.0,
        body_width=6.0,
    )

    bar = result.stroked_paths[0].vertices
    trim_vertices = result.trim_contour.vertices[:2]

    assert np.allclose(bar[:, 0], trim_vertices[0, 0])
    assert np.linalg.norm(bar[0] - bar[1]) == pytest.approx(11.0)


def test_note_shape_fold_is_large_enough_to_read_after_downscaling() -> None:
    """Note cards should keep a visible fold line and clipped corner.

    Returns
    -------
    None
        The folded-corner geometry is asserted in place.
    """

    spec = ShapeSpec(center_x=0.0, center_y=0.0, width=20.0, height=10.0, shape="note")

    path = note_path(spec)

    assert NOTE_FOLD_SIZE_RATIO == pytest.approx(0.45)
    assert [7.75, 5.0] in path.vertices.tolist()
    assert [7.75, 2.75] in path.vertices.tolist()
    assert [10.0, 2.75] in path.vertices.tolist()


def test_crow_arrowhead_tines_merge_at_the_neck() -> None:
    """Crow heads should read as one forked marker instead of three detached tines."""
    result = build_arrowhead(
        "crow",
        tip=(0.0, 0.0),
        tangent=(1.0, 0.0),
        length=8.0,
        width=10.0,
        body_width=6.0,
    )

    assert len(result.filled_paths) == 3

    neck_intervals = []
    for path in result.filled_paths:
        max_x = float(np.max(path.vertices[:, 0]))
        y_values = sorted(
            float(vertex[1]) for vertex in path.vertices if vertex[0] == pytest.approx(max_x)
        )
        neck_intervals.append((y_values[0], y_values[-1]))

    lower_interval, center_interval, upper_interval = sorted(
        neck_intervals,
        key=lambda item: item[0],
    )

    assert center_interval[1] - center_interval[0] > 6.0
    assert center_interval[1] >= upper_interval[0]
    assert center_interval[0] <= lower_interval[1]


@pytest.mark.parametrize(
    ("spec", "expected_anchors"),
    [
        # crow uses filled tines (not stroked), tested separately.
        ("bracket", np.array([-3.0, 3.0], dtype=np.float64)),
        ("curve", np.array([3.0], dtype=np.float64)),
        ("icurve", np.array([-3.0], dtype=np.float64)),
    ],
)
def test_ornamental_heads_anchor_from_body_edges(
    spec: str,
    expected_anchors: np.ndarray,
) -> None:
    """Open and ornamental heads should start on the ribbon edge geometry."""
    result = build_arrowhead(
        spec,
        tip=(0.0, 0.0),
        tangent=(1.0, 0.0),
        length=9.0,
        width=8.0,
        body_width=6.0,
    )

    anchor_values = []
    for path in result.stroked_paths:
        max_x = float(np.max(path.vertices[:, 0]))
        for vertex in path.vertices:
            if vertex[0] == pytest.approx(max_x):
                anchor_values.append(float(vertex[1]))
    observed = np.array(sorted(anchor_values), dtype=np.float64)

    assert np.allclose(observed, expected_anchors)


def test_odot_overlaps_into_body_instead_of_sitting_tangent() -> None:
    """Open dots should trim inside the circle so the body runs into the marker."""
    result = build_arrowhead(
        "odot",
        tip=(0.0, 0.0),
        tangent=(1.0, 0.0),
        length=8.0,
        width=6.0,
        body_width=4.0,
    )

    trim_midpoint_x = float(result.trim_contour.vertices[:2, 0].mean())
    circle_back_x = float(np.max(result.stroked_paths[0].vertices[:, 0]))

    assert trim_midpoint_x < circle_back_x


def test_stroked_head_linewidth_grows_with_thick_edges() -> None:
    """Open and hollow heads should gain outline weight as the body gets thicker."""
    outline_result = build_arrowhead(
        "vee",
        tip=(0.0, 0.0),
        tangent=(-1.0, 0.0),
        length=8.0,
        width=5.0,
        body_width=3.0,
    )
    hollow_result = build_arrowhead(
        "onormal",
        tip=(0.0, 0.0),
        tangent=(-1.0, 0.0),
        length=8.0,
        width=5.0,
        body_width=3.0,
    )
    thin_edge = DaguaEdge(curve=_curve(), stroke_width=1.0)
    thick_edge = DaguaEdge(curve=_curve(), stroke_width=8.0)

    assert (
        _stroked_head_linewidth(thick_edge, outline_result)
        > _stroked_head_linewidth(thin_edge, outline_result) * 8.0
    )
    assert (
        _stroked_head_linewidth(thick_edge, hollow_result)
        > _stroked_head_linewidth(thin_edge, hollow_result) * 8.0
    )


def test_trimmed_head_preserves_stroke_scale() -> None:
    """Prepared head results should keep their requested stroke-weight multiplier."""
    edge = DaguaEdge(curve=_curve(), width=3.0, stroke_width=3.0, arrowhead="vee")

    _, head_result, _ = _trimmed_body_curve(edge, edge.curve)

    assert head_result is not None
    assert head_result.stroke_width_scale > 1.0


def test_compound_arrowheads_insert_tangent_spacing_between_primitives() -> None:
    """Compound arrowheads should leave visible gaps between successive markers."""
    result = build_arrowhead(
        "dotnormaldiamond",
        tip=(0.0, 0.0),
        tangent=(1.0, 0.0),
        length=10.0,
        width=7.0,
        body_width=4.0,
    )

    extents = sorted(
        (float(np.min(path.vertices[:, 0])), float(np.max(path.vertices[:, 0])))
        for path in result.filled_paths
    )

    assert len(extents) == 3
    assert extents[1][0] > extents[0][1]
    assert extents[2][0] > extents[1][1]


def test_available_arrowheads_include_required_builtins() -> None:
    """The registry should expose the requested Graphviz and matplotlib heads."""
    names = available_arrowheads()

    assert "normal" in names
    assert "odot" in names
    assert "obox" in names
    assert "simple" in names
    assert "wedge" in names
    assert len(names) >= 18


def test_dense_cross_sampling_detects_insufficient_lane_gap() -> None:
    """Lane validation should fail when two offset curves are too close."""
    curve = _curve()
    near = offset_cubic_control_points(curve, 0.5)
    far = offset_cubic_control_points(curve, 6.0)

    valid_near, observed = validate_lane_separation([curve, near], min_gap=2.0, n_samples=50)
    valid_far, _ = validate_lane_separation([curve, far], min_gap=2.0, n_samples=50)

    assert not valid_near
    assert observed is not None and observed < 2.0
    assert valid_far


def test_roundrect_intersection_hits_visible_corner_arc() -> None:
    """Rounded-rectangle intersections should land on the visible arc, not the bbox corner."""
    hit = intersect_node_boundary(
        center=(0.0, 0.0),
        half_size=(10.0, 6.0),
        shape="roundrect",
        corner_radius=3.0,
        ray_origin=(0.0, 0.0),
        ray_direction=(10.0, 6.0),
    )

    assert hit[0] < 10.0
    assert hit[1] < 6.0
    assert hit[0] > 6.0
    assert hit[1] > 2.0


def test_semicircle_intersection_hits_the_flat_edge_when_cast_straight_down() -> None:
    """Semicircle intersections should return the flat diameter for downward rays."""

    hit = intersect_node_boundary(
        center=(0.0, 0.0),
        half_size=(20.0, 15.0),
        shape="semicircle",
        corner_radius=0.0,
        ray_origin=(0.0, 0.0),
        ray_direction=(0.0, -10.0),
    )

    assert tuple(float(value) for value in hit) == pytest.approx((0.0, -15.0))


def test_semicircle_intersection_hits_the_curved_dome_for_upward_rays() -> None:
    """Semicircle intersections should stay on the dome when casting upward rays."""

    hit = intersect_node_boundary(
        center=(0.0, 0.0),
        half_size=(20.0, 15.0),
        shape="semicircle",
        corner_radius=0.0,
        ray_origin=(0.0, -10.0),
        ray_direction=(0.0, 10.0),
        aspect_ratio=2.0,
    )

    assert float(hit[0]) == pytest.approx(0.0)
    assert float(hit[1]) == pytest.approx(-5.0)


def test_label_rotation_follows_curve_tangent() -> None:
    """Rotated labels should inherit a non-zero upright tangent angle."""
    placement = place_edge_label(_curve(), label_position=0.5, label_offset=4.0, label_rotate=True)

    assert placement.t > 0.0
    assert placement.t < 1.0
    assert placement.angle_degrees != 0.0


def test_rendering_tier_thresholds_match_spec() -> None:
    """Tier selection should follow the approved edge-count breakpoints."""
    assert choose_rendering_tier(10) == "full"
    assert choose_rendering_tier(5000) == "simplified"
    assert choose_rendering_tier(50000) == "lines"
    assert choose_rendering_tier(150000) == "bundled"


def test_collection_renders_bodies_and_heads_in_two_passes() -> None:
    """The collection should render bodies first and heads second."""
    fig, ax = plt.subplots()
    collection = DaguaEdgeCollection(
        [
            DaguaEdge(curve=_curve(), width=2.0, color="#224466", arrowhead="normal"),
            DaguaEdge(curve=offset_cubic_control_points(_curve(), 5.0), width=2.0, color="#446688"),
        ]
    )

    body_artists = collection.render_bodies(ax)
    head_artists = collection.render_heads(ax)

    assert any(isinstance(artist, PatchCollection) for artist in body_artists)
    assert any(isinstance(artist, PatchCollection) for artist in head_artists)
    assert all(float(artist.get_zorder()) == 1.0 for artist in body_artists)
    assert all(float(artist.get_zorder()) == pytest.approx(2.1) for artist in head_artists)
    plt.close(fig)


def test_collection_scales_dense_head_sizes() -> None:
    """Crowded terminals should shrink arrowheads before rendering."""
    edges = [
        DaguaEdge(
            curve=CubicBezier.from_points(
                (-40.0, float(offset)),
                (-24.0, float(offset)),
                (-8.0, float(offset) * 0.15),
                (0.0, 0.0),
            ),
            width=2.0,
            arrowhead="normal",
            target_node=1,
        )
        for offset in (-6.0, -3.0, 0.0, 3.0)
    ]

    collection = DaguaEdgeCollection(edges)

    assert any(edge.resolved_arrow_length() < 8.0 for edge in collection.edges)


def test_collection_replaces_zero_length_edges_with_visible_micro_loops() -> None:
    """Coincident endpoints should render via a synthetic loop instead of disappearing."""
    zero_curve = CubicBezier.from_points((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    collection = DaguaEdgeCollection([DaguaEdge(curve=zero_curve, arrowhead="normal", width=1.0)])

    assert build_arc_length_table(collection.prepared_edges[0].lane_curve).total_length > 0.0
    assert collection.prepared_edges[0].head_result is not None


def test_collection_simplifies_very_dense_terminal_heads() -> None:
    """Very dense fan-in should fall back to tee or no head."""
    edges = [
        DaguaEdge(
            curve=CubicBezier.from_points(
                (-40.0, float(offset)),
                (-24.0, float(offset)),
                (-8.0, float(offset) * 0.1),
                (0.0, 0.0),
            ),
            width=2.0,
            arrowhead="normal",
            target_node=1,
        )
        for offset in range(-11, 12, 2)
    ]

    collection = DaguaEdgeCollection(edges)

    assert all(edge.arrowhead in {"tee", "none"} for edge in collection.edges)


def test_hub_node_arrowheads_dont_overlap() -> None:
    """Dense hub heads should redistribute their terminal approach angles."""
    tip = np.array([0.0, 0.0], dtype=np.float64)
    base_angles = np.linspace(172.0, 188.0, num=8)
    edges = []
    for angle_degrees in base_angles:
        angle_radians = np.deg2rad(angle_degrees)
        direction = np.array([np.cos(angle_radians), np.sin(angle_radians)], dtype=np.float64)
        edges.append(
            DaguaEdge(
                curve=CubicBezier.from_points(
                    tip + (direction * 36.0),
                    tip + (direction * 24.0),
                    tip + (direction * 8.0),
                    tip,
                ),
                width=2.0,
                arrowhead="normal",
                target_node=1,
            )
        )

    collection = DaguaEdgeCollection(edges)
    redistributed_angles = [
        _terminal_angle(_head_body_direction(edge.curve)) for edge in collection.edges
    ]

    assert _minimum_angular_gap(redistributed_angles) >= (40.0 / 7.0) - 1e-6


def test_custom_arrow_overrides_still_respect_thick_edge_head_floors() -> None:
    """Explicit head dimensions should not undersize thick-edge arrowheads."""
    edge = DaguaEdge(
        curve=_curve(),
        width=5.0,
        arrowhead_length=8.0,
        arrowhead_width=8.0,
        tail_arrow_length=9.0,
        tail_arrow_width=9.0,
    )

    assert edge.resolved_arrow_length() == pytest.approx(15.0)
    assert edge.resolved_arrow_width() == pytest.approx(12.5)
    assert edge.resolved_tail_arrow_length() == pytest.approx(15.0)
    assert edge.resolved_tail_arrow_width() == pytest.approx(12.5)


def test_line_tier_uses_line_collection() -> None:
    """High edge counts should fall back to the line-based tier."""
    fig, ax = plt.subplots()
    edges = [
        DaguaEdge(curve=_curve(), width=1.0, color="#333333", group_key=(index, index + 1))
        for index in range(10001)
    ]
    collection = DaguaEdgeCollection(edges)
    artists = collection.render_bodies(ax)

    assert collection.tier == "lines"
    assert any(isinstance(artist, LineCollection) for artist in artists)
    plt.close(fig)


def test_mpl_translation_builds_custom_collection() -> None:
    """The matplotlib renderer should translate graph edges into the custom collection."""
    graph = DaguaGraph.from_edge_list([("a", "b"), ("a", "b")])
    graph.edge_styles[0] = EdgeStyle(style="dashed", arrow="normal")
    graph.edge_styles[1] = EdgeStyle(style="solid", arrow="vee", tail_arrow="dot")
    graph.compute_node_sizes()
    curves = [
        type(
            "Curve",
            (),
            {"p0": (0.0, 0.0), "cp1": (10.0, 20.0), "cp2": (30.0, -20.0), "p1": (40.0, 0.0)},
        )(),
        type(
            "Curve",
            (),
            {"p0": (0.0, 4.0), "cp1": (10.0, 24.0), "cp2": (30.0, -16.0), "p1": (40.0, 4.0)},
        )(),
    ]
    fig, ax = plt.subplots()
    ax.set_xlim(-10.0, 50.0)
    ax.set_ylim(-30.0, 30.0)
    ax.set_aspect("equal")

    collection = _build_custom_edge_collection(ax, graph, curves)  # type: ignore[arg-type]

    assert isinstance(collection, DaguaEdgeCollection)
    assert len(collection.edges) == 2
    assert collection.edges[0].linestyle == "dashed"
    assert collection.edges[1].tail_arrow == "dot"
    plt.close(fig)


def test_render_uses_custom_edge_collections(tmp_path) -> None:
    """Full renders should emit patch collections for custom edge geometry."""
    graph = DaguaGraph.from_edge_list([("a", "b"), ("b", "c")])
    graph.edge_styles[0] = EdgeStyle(arrow="normal", style="dashed")
    graph.edge_styles[1] = EdgeStyle(arrow="diamond", tail_arrow="dot")
    positions = torch.tensor([[0.0, 0.0], [50.0, -40.0], [100.0, 0.0]], dtype=torch.float32)
    graph._layout_positions = positions
    graph._layout_revision = graph.revision
    out = tmp_path / "custom-edge.png"

    fig, ax = render(graph, positions=positions, output=str(out))

    assert out.exists()
    assert any(isinstance(collection, PatchCollection) for collection in ax.collections)
    plt.close(fig)
