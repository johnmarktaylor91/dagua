"""Tests for the custom edge rendering package."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from matplotlib.collections import LineCollection, PatchCollection

from dagua.graph import DaguaGraph
from dagua.render.edges import available_arrowheads, build_arrowhead
from dagua.render.edges.collection import DaguaEdge, DaguaEdgeCollection, choose_rendering_tier
from dagua.render.edges.dashes import dash_curve
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


def test_adaptive_subdivision_refines_curved_edges() -> None:
    """Curved edges should subdivide more than straight edges."""
    straight = CubicBezier.from_points((0.0, 0.0), (20.0, 0.0), (40.0, 0.0), (60.0, 0.0))
    curved_samples = adaptive_subdivide(_curve(), flatness=0.5)
    straight_samples = adaptive_subdivide(straight, flatness=0.5)

    assert len(curved_samples) > len(straight_samples)
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


def test_dash_curve_uses_round_caps_per_segment() -> None:
    """Visible dash segments should carry round caps on both ends."""
    segments = dash_curve(_curve(), "dashed", width=3.0)

    assert len(segments) >= 2
    assert all(segment.cap_start == "round" for segment in segments)
    assert all(segment.cap_end == "round" for segment in segments)


def test_dotted_dash_curve_uses_short_round_segments() -> None:
    """Dotted edges should render as small round marks rather than chunky dashes."""
    curve = CubicBezier.from_points((0.0, 0.0), (6.0, 0.0), (12.0, 0.0), (18.0, 0.0))

    segments = dash_curve(curve, "dotted", width=3.0)

    assert segments
    first_length = build_arc_length_table(segments[0].curve).total_length
    assert first_length < 1.0


def test_dash_curve_drops_truncated_terminal_dash() -> None:
    """A final short dash should be omitted instead of ending awkwardly at the head."""
    curve = CubicBezier.from_points((0.0, 0.0), (6.0, 0.0), (12.0, 0.0), (18.0, 0.0))

    segments = dash_curve(curve, "dashed", width=2.0)

    assert len(segments) == 1


@pytest.mark.parametrize("spec", ["normal", "dot", "diamond", "vee", "crow", "box", "simple"])
def test_arrowhead_result_separates_filled_and_stroked_geometry(spec: str) -> None:
    """Arrowheads should report fill geometry separately from stroke geometry."""
    result = build_arrowhead(spec, tip=(0.0, 0.0), tangent=(-1.0, 0.0), length=8.0, width=5.0)

    assert result.trim_contour.vertices.shape[0] >= 2
    if spec in {"vee", "crow"}:
        assert result.filled_paths == []
        assert len(result.stroked_paths) >= 1
    else:
        assert len(result.filled_paths) >= 1


def test_open_arrowhead_becomes_stroked() -> None:
    """Open modifiers should route fill geometry into the stroked pass."""
    result = build_arrowhead("onormal", tip=(0.0, 0.0), tangent=(-1.0, 0.0), length=8.0, width=5.0)

    assert result.filled_paths == []
    assert len(result.stroked_paths) >= 1


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
    assert all(float(artist.get_zorder()) == 2.0 for artist in head_artists)
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
