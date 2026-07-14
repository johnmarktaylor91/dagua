"""Tests for visual parity v2 SVG extractors."""

from __future__ import annotations

import pytest

from scripts.visual_parity import extractors

SVG_SNIPPET = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 160 120">
<g class="graph" id="graph0">
<g class="cluster" id="clust1"><title>cluster_a</title>
<polygon fill="none" stroke="black" points="5,5 155,5 155,115 5,115"/>
</g>
<g class="node" id="node1"><title>box</title>
<polygon fill="white" stroke="black" points="10,10 50,10 50,40 10,40"/>
</g>
<g class="node" id="node2"><title>diamond</title>
<polygon fill="white" stroke="black" points="85,10 110,25 85,40 60,25"/>
</g>
<g class="node" id="node3"><title>doublecircle</title>
<path fill="white" stroke="black" d="M 115 10 L 145 10 L 145 40 L 115 40 Z"/>
</g>
</g>
</svg>"""


def test_extract_shape_paths_covers_box_diamond_doublecircle() -> None:
    """Assert canned non-ellipse SVG outlines produce path records."""

    records = extractors.extract_shape_paths(SVG_SNIPPET)
    by_id = {record["element_id"]: record for record in records}
    assert set(by_id) == {"box", "diamond", "doublecircle"}
    assert by_id["box"]["bbox"] == [10.0, 10.0, 50.0, 40.0]
    assert by_id["diamond"]["area"] == pytest.approx(750.0)
    assert by_id["doublecircle"]["command_inventory"]["M"] == 1
    assert by_id["doublecircle"]["path_iou"] == pytest.approx(1.0)


def test_arrow_metric_family_reports_all_required_axes() -> None:
    """Assert arrow metrics are a family, not union IoU only."""

    ref = [(0.0, 0.0), (10.0, 3.0), (10.0, -3.0)]
    cand = [(0.0, 0.0), (9.0, 3.0), (9.0, -3.0)]
    metrics = extractors.arrow_metric_family(
        ref,
        cand,
        tangent=(1.0, 0.0),
        reference_fill="black",
        candidate_fill="none",
        reference_arrow="lnormalvee",
        candidate_arrow="rveenormal",
    )
    assert {
        "arrow_polygon_iou",
        "arrow_len_pct",
        "arrow_width_pct",
        "arrow_fill_mode",
        "arrow_compound_order",
        "arrow_side_clip",
    } <= set(metrics)
    assert metrics["arrow_fill_mode"]["match"] is False
    assert metrics["arrow_compound_order"]["match"] is False
    assert metrics["arrow_side_clip"]["match"] is False


def test_label_glyph_extent_carries_provenance() -> None:
    """Assert label glyph extent includes the required provenance fields."""

    extent = extractors.label_glyph_extent(
        "long label",
        12.0,
        target_kind="svg_declared",
        font_resolver="matplotlib",
        resolved_font_file="/tmp/font.ttf",
    )
    assert extent["width_pt"] > 0.0
    assert extent["font_resolver"] == "matplotlib"
    assert extent["resolved_font_file"] == "/tmp/font.ttf"
    assert extent["target_kind"] == "svg_declared"


def test_cluster_and_edge_trim_extractors() -> None:
    """Assert cluster and edge-trim helper records are measurable."""

    clusters = extractors.cluster_rect_features(SVG_SNIPPET)
    assert clusters[0]["cluster_id"] == "cluster_a"
    assert clusters[0]["border_segments"] == 4
    assert extractors.edge_trim_distance((20.0, 20.0), (10.0, 10.0, 50.0, 40.0)) == 0.0
    assert extractors.edge_trim_distance((0.0, 0.0), (10.0, 10.0, 50.0, 40.0)) > 0.0


def test_spline_polyline_empty_inputs_are_numeric() -> None:
    """Assert E1 spline inputs produce numeric distances even when empty."""

    assert extractors.symmetric_mean_point_to_polyline([], []) == 0.0
    assert extractors.symmetric_mean_point_to_polyline([], [(0.0, 0.0)]) == float("inf")
