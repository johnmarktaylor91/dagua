"""Tests for the gallery audit builder script."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from PIL import Image, ImageColor

from scripts.build_gallery_audit import (
    ARROW_DEMO_EDGE_WIDTH,
    ARROW_DEMO_NODE_FRACTION,
    BOARD_GRID_TOP,
    CURVATURE_CARD_MARGIN,
    DEFAULT_COMPARISON_STROKE,
    PAIR_ARROW_GAP,
    PAIR_SCALAR_COMPARISON_GAP,
    _apply_reference_params,
    _build_fixture,
    _combo_flow_positions,
    _combo_params,
    _compose_board,
    _prepare_reference_render,
    _render_reference_canvas,
    build_combo_items,
    build_evil_items,
    build_gallery_audit,
    build_reference_items,
)
from scripts.generate_cosmetic_album import VARIED_EXTERNAL_LABELS


def test_gallery_audit_inventory_matches_expected_counts() -> None:
    """The resolved audit inventories should expose the requested card counts.

    Returns
    -------
    None
        The manifest counts are asserted in place.
    """

    reference_items = build_reference_items()
    combo_items = build_combo_items()
    evil_items = build_evil_items()

    assert len(reference_items) == 133
    assert len(combo_items) == 79
    assert len(evil_items) == 35
    assert {
        "nodes_shapes_rect",
        "nodes_shapes_roundrect",
        "nodes_fills_gradient_linear",
        "nodes_borders_corner_radius_12",
        "nodes_borders_border_count_1_vs_2",
        "edges_arrows_normal",
        "graph_direction_lr",
    }.issubset({item.card_id for item in reference_items})
    assert "nodes_borders_corner_radius_8" not in {item.card_id for item in reference_items}
    assert {
        "combo_shadow_gradient",
        "combo_bold_shadow_gradient",
        "combo_kitchen_sink_3",
        "combo_pie_bold",
        "combo_bt_cluster_rounded",
        "combo_kitchen_sink_6",
    }.issubset({item.card_id for item in combo_items})
    assert {
        "evil_huge_arrows",
        "evil_self_loop_star",
        "evil_all_arrows_gradient",
        "evil_contradictory_styles",
    }.issubset({item.card_id for item in evil_items})


def test_build_gallery_audit_writes_subset(tmp_path: Path) -> None:
    """A reduced audit build should emit cards, comparisons, boards, and index data.

    Parameters
    ----------
    tmp_path : Path
        Temporary pytest directory.

    Returns
    -------
    None
        The generated artifact tree is asserted in place.
    """

    if shutil.which("dot") is None:
        pytest.skip("Graphviz dot is not installed")

    output_dir = tmp_path / "gallery_audit"
    result = build_gallery_audit(
        output_dir=str(output_dir),
        cards=True,
        boards=True,
        comparisons=True,
        combos=True,
        evil=True,
        index=True,
        reference_card_ids=[
            "nodes_shapes_rect",
            "nodes_shapes_roundrect",
            "nodes_shapes_ellipse",
            "nodes_shapes_diamond",
        ],
        combo_card_ids=[
            "combo_shadow_gradient",
            "combo_dashed_border_arrow",
            "combo_bold_italic",
            "combo_opacity_shadow",
        ],
        evil_card_ids=["evil_self_loop_star"],
    )

    assert result.reference_count == 4
    assert result.comparison_count == 4
    assert result.combo_count == 4
    assert result.evil_count == 1
    assert result.board_count == 2

    expected_paths = [
        output_dir / "cards/reference/nodes/shapes/rect.png",
        output_dir / "cards/reference/nodes/shapes/roundrect.png",
        output_dir / "cards/comparisons/nodes/shapes/rect_vs_graphviz.png",
        output_dir / "cards/combos/2way/shadow_gradient.png",
        output_dir / "cards/evil/evil_self_loop_star.png",
        output_dir / "boards/reference/nodes_shapes_01.png",
        output_dir / "boards/combos/2way_01.png",
        output_dir / "index.jsonl",
    ]
    for path in expected_paths:
        assert path.exists()
        assert path.stat().st_size > 0

    for image_path in expected_paths[:-1]:
        with Image.open(image_path) as image:
            assert image.width <= 1600
            assert image.height <= 1200

    reference_sidecar = output_dir / "cards/reference/nodes/shapes/rect.json"
    combo_sidecar = output_dir / "cards/combos/2way/shadow_gradient.json"
    evil_sidecar = output_dir / "cards/evil/evil_self_loop_star.json"
    assert reference_sidecar.exists()
    assert combo_sidecar.exists()
    assert evil_sidecar.exists()

    index_rows = [
        json.loads(line)
        for line in (output_dir / "index.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(index_rows) == 9
    rect_row = next(row for row in index_rows if row["id"] == "nodes_shapes_rect")
    combo_row = next(row for row in index_rows if row["id"] == "combo_shadow_gradient")
    evil_row = next(row for row in index_rows if row["id"] == "evil_self_loop_star")
    assert rect_row["comparison_path"] == "cards/comparisons/nodes/shapes/rect_vs_graphviz.png"
    assert rect_row["has_comparison"] is True
    assert combo_row["category"] == "combos/2way"
    assert combo_row["has_comparison"] is False
    assert evil_row["category"] == "evil"
    assert evil_row["has_comparison"] is False


def test_compose_board_reserves_gap_below_header(tmp_path: Path) -> None:
    """Board composition should leave white space below the shared header.

    Parameters
    ----------
    tmp_path : Path
        Temporary pytest directory.

    Returns
    -------
    None
        The board gap and tile placement are asserted in place.
    """

    image_paths = []
    for index in range(4):
        image_path = tmp_path / f"tile_{index}.png"
        Image.new("RGB", (1600, 1200), "#FF0000").save(image_path)
        image_paths.append(image_path)

    output_path = tmp_path / "board.png"
    _compose_board(image_paths, output_path, "Nodes / Shapes (1)")

    with Image.open(output_path) as board:
        assert board.getpixel((64, 110)) == (255, 255, 255)
        assert board.getpixel((64, BOARD_GRID_TOP + 16)) == (255, 0, 0)


def test_prepare_reference_render_scales_arrow_demo_cards() -> None:
    """Arrow demo cards should use larger arrowheads and tighter pair spacing.

    Returns
    -------
    None
        The prepared arrow styles and positions are asserted in place.
    """

    item = next(item for item in build_reference_items() if item.card_id == "edges_arrows_normal")
    graph, positions = _prepare_reference_render(item)
    edge_style = graph.edge_styles[0]

    assert edge_style is not None
    assert edge_style.width == pytest.approx(ARROW_DEMO_EDGE_WIDTH)
    assert edge_style.arrow_node_fraction == pytest.approx(ARROW_DEMO_NODE_FRACTION)
    assert float(positions[0, 1] - positions[1, 1]) == pytest.approx(PAIR_ARROW_GAP)


def test_prepare_reference_render_adds_scalar_default_context() -> None:
    """Scalar node sweeps should compare the default node against the swept value.

    Returns
    -------
    None
        The comparison labels, node styles, and side-by-side positions are asserted.
    """

    item = next(
        item for item in build_reference_items() if item.card_id == "nodes_borders_stroke_width_3_0"
    )
    graph, positions = _prepare_reference_render(item)
    left_style = graph.node_styles[0]
    right_style = graph.node_styles[1]
    edge_style = graph.edge_styles[0]

    assert left_style is not None
    assert right_style is not None
    assert edge_style is not None
    assert graph.node_labels == ["Default", "3.0"]
    assert left_style.stroke_width == pytest.approx(1.5)
    assert right_style.stroke_width == pytest.approx(3.0)
    assert edge_style.arrow == "none"
    assert edge_style.width == pytest.approx(0.0)
    assert edge_style.opacity == pytest.approx(0.0)
    assert float(positions[1, 0] - positions[0, 0]) == pytest.approx(PAIR_SCALAR_COMPARISON_GAP)
    assert float(positions[0, 1]) == pytest.approx(0.0)
    assert float(positions[1, 1]) == pytest.approx(0.0)


def test_prepare_reference_render_strengthens_subtle_border_comparisons() -> None:
    """Border comparison cards should use explicit baseline styling for contrast.

    Returns
    -------
    None
        The comparison-node overrides are asserted in place.
    """

    item = next(
        item
        for item in build_reference_items()
        if item.card_id == "nodes_borders_border_opacity_0_2"
    )
    graph, _ = _prepare_reference_render(item)
    left_style = graph.node_styles[0]
    right_style = graph.node_styles[1]

    assert left_style is not None
    assert right_style is not None
    assert left_style.stroke_width == pytest.approx(3.0)
    assert left_style.stroke == DEFAULT_COMPARISON_STROKE
    assert right_style.border_opacity == pytest.approx(0.2)

    item = next(
        item for item in build_reference_items() if item.card_id == "nodes_borders_stroke_width_0_5"
    )
    graph, _ = _prepare_reference_render(item)
    left_style = graph.node_styles[0]
    right_style = graph.node_styles[1]

    assert left_style is not None
    assert right_style is not None
    # stroke_width is now a scalar comparison feature: left is default (1.5),
    # right is the swept value (0.5).
    assert left_style.stroke_width == pytest.approx(1.5)
    assert right_style.stroke_width == pytest.approx(0.5)

    item = next(
        item
        for item in build_reference_items()
        if item.card_id == "nodes_borders_border_position_outside"
    )
    graph, _ = _prepare_reference_render(item)
    left_style = graph.node_styles[0]
    right_style = graph.node_styles[1]

    assert left_style is not None
    assert right_style is not None
    assert left_style.stroke_width == pytest.approx(50.0)
    assert right_style.stroke_width == pytest.approx(50.0)

    item = next(
        item
        for item in build_reference_items()
        if item.card_id == "nodes_borders_border_count_1_vs_2"
    )
    graph, _ = _prepare_reference_render(item)
    left_style = graph.node_styles[0]
    right_style = graph.node_styles[1]

    assert left_style is not None
    assert right_style is not None
    assert graph.node_labels == ["1", "2"]
    assert left_style.border_count == 1
    assert right_style.border_count == 2
    assert right_style.stroke_width == pytest.approx(3.0)


def test_prepare_reference_render_applies_gallery_demo_tweaks() -> None:
    """Card-specific gallery tweaks should produce the intended demo setup.

    Returns
    -------
    None
        The tailored fixture and style adjustments are asserted in place.
    """

    item = next(
        item for item in build_reference_items() if item.card_id == "nodes_text_external_label_top"
    )
    graph, _ = _prepare_reference_render(item)
    top_style = graph.node_styles[0]
    bottom_style = graph.node_styles[1]

    assert top_style is not None
    assert bottom_style is not None
    assert top_style.external_label == "ID 42"
    assert bottom_style.external_label == ""

    item = next(
        item for item in build_reference_items() if item.card_id == "nodes_fills_fill_pattern_pie"
    )
    graph, _ = _prepare_reference_render(item)
    assert graph.node_labels == ["A", "B"]

    item = next(
        item for item in build_reference_items() if item.card_id == "edges_routing_curvature_0_8"
    )
    graph, _ = _prepare_reference_render(item)
    assert graph.graph_style.margin == pytest.approx(CURVATURE_CARD_MARGIN)

    item = next(
        item
        for item in build_reference_items()
        if item.card_id == "edges_routing_port_style_center"
    )
    graph, positions = _prepare_reference_render(item)
    assert graph.num_nodes == 7
    assert graph.edge_index.shape[1] == 6
    assert positions.shape == (7, 2)


def test_combo_params_adds_new_combo_feature_defaults() -> None:
    """Combo parameter translation should add defaults for new style features.

    Returns
    -------
    None
        The translated combo defaults are asserted in place.
    """

    hatched_params = _combo_params(
        {
            "fill_pattern": "hatched",
            "shape": "cloud",
            "external_label": "Queue",
            "text_outline": True,
        },
        "combo_flow",
    )
    hatched_node = hatched_params["node"]

    assert isinstance(hatched_node, dict)
    assert hatched_node["text_background"] == "#FFFFFF"
    assert hatched_node["text_background_opacity"] == pytest.approx(0.92)
    assert hatched_node["text_background_padding"] == (6.0, 3.0)
    assert hatched_node["text_background_corner_radius"] == pytest.approx(4.0)
    assert hatched_node["external_label_font_size"] == pytest.approx(10.0)
    assert hatched_node["external_label_offset"] == pytest.approx(8.0)
    assert hatched_node["text_outline_color"] == "#FFFFFF"
    assert hatched_node["text_outline_width"] == pytest.approx(2.0)
    assert hatched_params["node_labels"] == ["Svc", "API", "Hub", "Log", "Out"]

    pie_params = _combo_params({"fill_pattern": "pie", "shape": "box3d"}, "combo_flow")
    pie_node = pie_params["node"]

    assert isinstance(pie_node, dict)
    assert pie_node["min_width"] == pytest.approx(120.0)
    assert pie_node["min_height"] == pytest.approx(80.0)
    assert pie_params["node_labels"] == ["DB", "Srv", "App", "Web", "Out"]

    crossing_params = _combo_params({"crossing_style": "gap"}, "crossing")
    crossing_edge = crossing_params["edge"]

    assert isinstance(crossing_edge, dict)
    assert crossing_edge["crossing_size"] == pytest.approx(20.0)
    assert crossing_edge["width"] == pytest.approx(4.0)

    outlined_params = _combo_params(
        {
            "text_outline": True,
            "text_outline_color": "#333333",
            "shadow": True,
        },
        "combo_flow",
    )
    outlined_node = outlined_params["node"]

    assert isinstance(outlined_node, dict)
    assert outlined_node["text_outline_color"] == "#333333"

    endpoint_params = _combo_params(
        {
            "head_label": "H",
            "tail_label": "T",
            "head_label_offset": 12.0,
            "tail_label_offset": 12.0,
        },
        "combo_flow",
    )
    endpoint_edge = endpoint_params["edge"]

    assert isinstance(endpoint_edge, dict)
    assert endpoint_edge["head_label_offset"] == pytest.approx(12.0)
    assert endpoint_edge["tail_label_offset"] == pytest.approx(12.0)


def test_apply_reference_params_varies_external_labels_per_node() -> None:
    """Varied external-label params should assign distinct labels across nodes."""

    params = _combo_params(
        {
            "external_label_varied": True,
            "external_label_position": "bottom",
            "corner_radius": 12.0,
        },
        "combo_flow",
    )
    graph, positions = _build_fixture("combo_flow")
    _apply_reference_params(graph, positions, params, "combo_flow")

    styles = graph.node_styles
    assert styles
    assert [style.external_label for style in styles if style is not None] == list(
        VARIED_EXTERNAL_LABELS
    )


def test_combo_flow_positions_support_bt_and_rl_directions() -> None:
    """Combo flow positions should produce stable layouts for BT and RL directions.

    Returns
    -------
    None
        The direction-specific node coordinates are asserted in place.
    """

    bt_positions = _combo_flow_positions("BT")
    rl_positions = _combo_flow_positions("RL")

    assert tuple(float(value) for value in bt_positions[0]) == pytest.approx((0.0, -210.0))
    assert tuple(float(value) for value in bt_positions[-1]) == pytest.approx((170.0, 200.0))
    assert tuple(float(value) for value in rl_positions[0]) == pytest.approx((280.0, 0.0))
    assert tuple(float(value) for value in rl_positions[1]) == pytest.approx((120.0, 120.0))
    assert tuple(float(value) for value in rl_positions[2]) == pytest.approx((120.0, -120.0))
    assert tuple(float(value) for value in rl_positions[-1]) == pytest.approx((-280.0, -120.0))


def test_render_reference_canvas_preserves_dark_graph_background() -> None:
    """Graph background cards should fill the normalized canvas, not just the graph bbox.

    Returns
    -------
    None
        The normalized canvas background color is asserted in place.
    """

    item = next(item for item in build_reference_items() if item.card_id == "graph_background_dark")
    canvas = _render_reference_canvas(item, (800, 600), (24, 24, 24, 24))

    assert canvas.getpixel((8, 8)) == ImageColor.getrgb("#1F2937")
