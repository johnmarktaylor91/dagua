"""Tests for the cosmetic album generator script."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch

from scripts.generate_cosmetic_album import (
    COMBO_DIAMOND_PAIR_VERTICAL_GAP,
    GRAPHVIZ_PAIR_VERTICAL_GAP,
    VARIED_EXTERNAL_LABELS,
    _base_cluster_style,
    build_case_catalog,
    build_cosmetic_album,
)


def test_build_case_catalog_covers_expected_counts() -> None:
    """The case catalog should cover every requested cosmetic bucket."""

    cases = build_case_catalog()
    category_counts: dict[str, int] = {}
    for case in cases:
        category_counts[case.category] = category_counts.get(case.category, 0) + 1

    assert len({case.case_id for case in cases}) == len(cases)
    assert category_counts == {
        "arrow_types": 18,
        "border_styles": 3,
        "clusters": 3,
        "combo_2way": 43,
        "combo_3way": 20,
        "combo_4way": 10,
        "combo_5way": 6,
        "corner_radius": 3,
        "direction": 4,
        "edge_routing": 3,
        "edge_styles": 3,
        "evil_combos": 35,
        "gradients": 3,
        "node_shapes": 13,
        "opacity": 4,
        "rich_labels": 2,
        "shadows": 2,
        "text_formatting": 7,
    }


def test_build_case_catalog_contains_required_evil_case_ids() -> None:
    """The catalog should include every requested evil stress case."""

    case_ids = {case.case_id for case in build_case_catalog()}

    assert {
        "evil_self_loop_star",
        "evil_self_loop_diamond",
        "evil_self_loop_triangle",
        "evil_long_wrap_star",
        "evil_long_wrap_triangle",
        "evil_mega_hub",
        "evil_zero_width_big_arrow",
        "evil_mixed_overflow",
        "evil_empty_labels",
        "evil_unicode_labels",
        "evil_negative_curvature",
        "evil_hundred_nodes",
        "evil_8_deep_clusters",
        "evil_pie_star",
        "evil_donut_diamond",
        "evil_taxi_self_loop",
        "evil_all_arrows_gradient",
        "evil_white_gradient",
        "evil_extreme_taper_crossing",
        "evil_contradictory_styles",
    }.issubset(case_ids)


def test_build_case_catalog_contains_required_combo_case_ids() -> None:
    """The catalog should include every requested 2-way through 5-way combo case."""

    case_ids = {case.case_id for case in build_case_catalog()}

    assert {
        "combo_shadow_gradient",
        "combo_dashed_border_arrow",
        "combo_bold_italic",
        "combo_opacity_shadow",
        "combo_gradient_rounded",
        "combo_dotted_edge_vee",
        "combo_cluster_gradient",
        "combo_double_border_shadow",
        "combo_text_wrap_bold",
        "combo_striped_fill_dashed",
        "combo_taper_gradient_edge",
        "combo_crossing_thick_edge",
        "combo_crossing_sharp_dashed",
        "combo_hexagon_gradient",
        "combo_diamond_shadow",
        "combo_star_dotted",
        "combo_circle_double_border",
        "combo_lr_direction_ortho",
        "combo_opacity_gradient",
        "combo_italic_large_font",
        "combo_rounded_dashed_shadow",
        "combo_taxi_gradient_edge",
        "combo_straight_shadow",
        "combo_pie_bold",
        "combo_donut_shadow",
        "combo_external_label_rounded",
        "combo_text_outline_gradient",
        "combo_cylinder_dashed",
        "combo_cloud_shadow",
        "combo_stadium_gradient",
        "combo_tab_bold",
        "combo_note_italic",
        "combo_document_shadow",
        "combo_box3d_gradient",
        "combo_parallelogram_dotted",
        "combo_trapezoid_gradient",
        "combo_pentagon_shadow",
        "combo_octagon_double_border",
        "combo_crossing_gap_thick",
        "combo_hatched_gradient",
        "combo_bt_direction_shadow",
        "combo_rl_direction_gradient",
        "combo_crow_arrow_bold",
        "combo_bold_shadow_gradient",
        "combo_dashed_diamond_opacity",
        "combo_cluster_rounded_gradient",
        "combo_taper_crossing_thick",
        "combo_italic_hexagon_shadow",
        "combo_double_border_gradient_rounded",
        "combo_striped_shadow_dashed_edge",
        "combo_text_bg_bold_rounded",
        "combo_lr_bezier_vee",
        "combo_opacity_cluster_dashed",
        "combo_taxi_shadow_gradient",
        "combo_pie_gradient_bold",
        "combo_external_label_diamond_shadow",
        "combo_cloud_gradient_italic",
        "combo_stadium_striped_shadow",
        "combo_hatched_shadow_bold",
        "combo_head_tail_labels_ortho",
        "combo_bt_cluster_rounded",
        "combo_text_outline_shadow_bold",
        "combo_color_gradient_taper_thick",
        "combo_bold_shadow_gradient_rounded",
        "combo_diamond_dashed_opacity_italic",
        "combo_cluster_gradient_shadow_double_border",
        "combo_taper_crossing_gradient_thick",
        "combo_hexagon_striped_shadow_bold",
        "combo_pie_shadow_gradient_bold",
        "combo_cylinder_dashed_shadow_gradient",
        "combo_taxi_crossing_gap_gradient",
        "combo_cloud_striped_shadow_italic",
        "combo_ext_label_hexagon_gradient_bold",
        "combo_kitchen_sink_1",
        "combo_kitchen_sink_2",
        "combo_kitchen_sink_3",
        "combo_kitchen_sink_4",
        "combo_kitchen_sink_5",
        "combo_kitchen_sink_6",
    }.issubset(case_ids)


def test_combo_cases_use_graphs_that_exercise_their_features() -> None:
    """Affected combo demos should use real graphs and valid rounded-node shapes."""

    cases = {case.case_id: case for case in build_case_catalog()}

    for case_id in ("combo_bold_shadow_gradient", "combo_bold_shadow_gradient_rounded"):
        graph = cases[case_id].graph
        assert graph.num_nodes == 2
        assert graph.num_edges == 1

    rounded_styles = cases["combo_bold_shadow_gradient_rounded"].graph.node_styles
    assert rounded_styles
    assert all(style is not None and style.shape == "roundrect" for style in rounded_styles)
    assert all(style is not None and style.corner_radius == 12.0 for style in rounded_styles)

    kitchen_sink_styles = cases["combo_kitchen_sink_2"].graph.node_styles
    assert kitchen_sink_styles
    assert all(style is not None and style.shape == "roundrect" for style in kitchen_sink_styles)

    for case_id in ("combo_kitchen_sink_1", "combo_kitchen_sink_2", "combo_kitchen_sink_3"):
        graph = cases[case_id].graph
        assert graph.num_nodes >= 4
        assert graph.num_edges >= 2


def test_new_combo_cases_preserve_requested_settings_and_fixtures() -> None:
    """New combo cases should keep the specified metadata and graph fixtures."""

    cases = {case.case_id: case for case in build_case_catalog()}

    pie_case = cases["combo_pie_bold"]
    assert pie_case.settings == {
        "combo": "2way",
        "fill_pattern": "pie",
        "fill_pattern_colors": ["#56B4E9", "#D55E00", "#009E73"],
        "fill_pattern_values": [3, 2, 1],
        "font_weight": "bold",
    }
    assert float(pie_case.positions[0, 1].item() - pie_case.positions[1, 1].item()) == (
        pytest.approx(COMBO_DIAMOND_PAIR_VERTICAL_GAP)
    )

    hatched_case = cases["combo_hatched_gradient"]
    assert hatched_case.graph.num_nodes == 2
    assert hatched_case.graph.num_edges == 1
    assert hatched_case.settings == {
        "combo": "2way",
        "fill_pattern": "hatched",
        "gradient": "linear",
    }
    assert all(
        style is not None and style.fill_pattern == "hatched" and style.gradient == "linear"
        for style in hatched_case.graph.node_styles
    )

    gap_case = cases["combo_crossing_gap_thick"]
    assert gap_case.settings == {
        "combo": "2way",
        "routing": "straight",
        "crossing_style": "gap",
        "crossing_size": 20.0,
        "width": 3.5,
    }
    assert all(
        style is not None
        and style.routing == "straight"
        and style.crossing_style == "gap"
        and style.crossing_size == pytest.approx(20.0)
        for style in gap_case.graph.edge_styles
    )

    taxi_gradient_case = cases["combo_taxi_gradient_edge"]
    assert taxi_gradient_case.settings == {
        "combo": "2way",
        "routing": "bezier",
        "color_gradient": "source_to_target",
    }
    assert all(
        style is not None and style.routing == "bezier"
        for style in taxi_gradient_case.graph.edge_styles
    )

    crow_case = cases["combo_crow_arrow_bold"]
    assert crow_case.settings == {
        "combo": "2way",
        "arrow": "crow",
        "font_weight": "bold",
        "font_size": 12.0,
    }
    assert all(
        style is not None and style.font_weight == "bold" and style.font_size == pytest.approx(12.0)
        for style in crow_case.graph.node_styles
    )

    note_case = cases["combo_note_italic"]
    assert note_case.settings == {
        "combo": "2way",
        "shape": "note",
        "font_style": "italic",
    }
    assert all(
        style is not None and style.font_style == "italic" for style in note_case.graph.node_styles
    )

    diamond_case = cases["combo_external_label_diamond_shadow"]
    diamond_styles = diamond_case.graph.node_styles
    assert diamond_styles
    assert all(style is not None and style.shape == "diamond" for style in diamond_styles)
    assert [style.external_label for style in diamond_styles if style is not None] == list(
        VARIED_EXTERNAL_LABELS[: len(diamond_styles)]
    )
    assert float(
        diamond_case.positions[0, 1].item() - diamond_case.positions[1, 1].item()
    ) == pytest.approx(COMBO_DIAMOND_PAIR_VERTICAL_GAP)
    assert diamond_case.settings["external_label_varied"] is True

    rounded_case = cases["combo_external_label_rounded"]
    rounded_styles = rounded_case.graph.node_styles
    assert rounded_styles
    assert [style.external_label for style in rounded_styles if style is not None] == list(
        VARIED_EXTERNAL_LABELS[: len(rounded_styles)]
    )
    assert rounded_case.settings["external_label_varied"] is True

    hexagon_case = cases["combo_ext_label_hexagon_gradient_bold"]
    hexagon_styles = hexagon_case.graph.node_styles
    assert hexagon_styles
    assert [style.external_label for style in hexagon_styles if style is not None] == list(
        VARIED_EXTERNAL_LABELS[: len(hexagon_styles)]
    )
    assert hexagon_case.settings["external_label_varied"] is True

    cluster_case = cases["combo_bt_cluster_rounded"]
    assert cluster_case.graph.direction == "BT"
    assert "group" in cluster_case.graph.cluster_styles
    assert cluster_case.settings == {
        "combo": "3way",
        "direction": "BT",
        "cluster": True,
        "corner_radius": 10,
    }

    kitchen_sink_case = cases["combo_kitchen_sink_6"]
    assert kitchen_sink_case.graph.num_nodes == 4
    assert kitchen_sink_case.graph.num_edges == 2
    assert kitchen_sink_case.settings == {
        "combo": "5way",
        "shape": "stadium",
        "routing": "taxi",
        "shadow": True,
        "gradient": "linear",
        "crossing_style": "gap",
    }

    kitchen_sink_label_case = cases["combo_kitchen_sink_5"]
    kitchen_sink_label_styles = kitchen_sink_label_case.graph.node_styles
    assert kitchen_sink_label_styles
    assert [
        style.external_label for style in kitchen_sink_label_styles if style is not None
    ] == list(
        VARIED_EXTERNAL_LABELS[: len(kitchen_sink_label_styles)],
    )
    assert kitchen_sink_label_case.settings["external_label_varied"] is True

    endpoint_label_case = cases["combo_head_tail_labels_ortho"]
    assert endpoint_label_case.settings == {
        "combo": "3way",
        "head_label": "H",
        "tail_label": "T",
        "head_label_offset": 18.0,
        "tail_label_offset": 18.0,
        "routing": "ortho",
        "arrow": "none",
    }
    assert all(
        style is not None
        and style.head_label_offset == pytest.approx(18.0)
        and style.tail_label_offset == pytest.approx(18.0)
        and style.arrow == "none"
        for style in endpoint_label_case.graph.edge_styles
    )

    sharp_crossing_case = cases["combo_crossing_sharp_dashed"]
    assert sharp_crossing_case.settings == {
        "combo": "2way",
        "crossing_style": "sharp",
        "edge_style": "dashed",
    }
    assert all(
        style is not None and style.crossing_style == "sharp" and style.style == "dashed"
        for style in sharp_crossing_case.graph.edge_styles
    )

    outline_case = cases["combo_text_outline_shadow_bold"]
    assert outline_case.settings == {
        "combo": "3way",
        "text_outline": True,
        "text_outline_color": "#333333",
        "text_outline_width": 2.0,
        "shadow": True,
        "font_weight": "bold",
    }
    assert all(
        style is not None and style.text_outline is True and style.text_outline_color == "#333333"
        for style in outline_case.graph.node_styles
    )

    cloud_case = cases["combo_cloud_striped_shadow_italic"]
    assert cloud_case.settings == {
        "combo": "4way",
        "shape": "cloud",
        "fill_pattern": "striped",
        "shadow": True,
        "font_style": "italic",
    }
    assert all(
        style is not None and style.font_style == "italic" for style in cloud_case.graph.node_styles
    )


def test_evil_cases_apply_requested_viewport_and_arrowhead_fixes() -> None:
    """Evil combo fixtures should keep labels visible and self-loop arrows proportional.

    Returns
    -------
    None
        The updated node sizing, spacing, and arrow settings are asserted in place.
    """

    cases = {case.case_id: case for case in build_case_catalog()}

    for case_id in ("evil_long_wrap_star", "evil_long_wrap_triangle"):
        case = cases[case_id]
        style = case.graph.node_styles[0]
        assert style is not None
        assert style.min_width == pytest.approx(100.0)
        assert style.min_height == pytest.approx(100.0)
        assert style.font_size == pytest.approx(7.0)
        assert style.overflow_policy == "shrink_text"
        assert style.min_font_size == pytest.approx(4.0)
        assert case.graph.node_labels == ["Long text wrapping inside concave shape"]

    negative_style = cases["evil_negative_curvature"].graph.node_styles[0]
    assert negative_style is not None
    assert negative_style.min_width == pytest.approx(120.0)

    mixed_case = cases["evil_mixed_overflow"]
    mixed_styles = mixed_case.graph.node_styles
    assert mixed_styles
    assert mixed_case.graph.num_nodes == 3
    assert mixed_case.graph.num_edges == 2
    assert mixed_case.graph.node_labels == [
        "Overflow text here",
        "Shrink me down",
        "Expand to fit",
    ]
    assert all(
        style is not None
        and style.shape == "rect"
        and style.min_width == pytest.approx(180.0)
        and style.font_size == pytest.approx(10.0)
        and style.text_max_width == pytest.approx(150.0)
        for style in mixed_styles
    )
    assert tuple(float(value) for value in mixed_case.positions[0]) == pytest.approx((0.0, 150.0))
    assert tuple(float(value) for value in mixed_case.positions[-1]) == pytest.approx((0.0, -150.0))

    for case_id in ("evil_self_loop_star", "evil_self_loop_diamond", "evil_self_loop_triangle"):
        edge_style = cases[case_id].graph.edge_styles[0]
        assert edge_style is not None
        assert edge_style.arrow_length == pytest.approx(5.0)
        assert edge_style.arrow_width == pytest.approx(3.5)

    cluster_case = cases["evil_8_deep_clusters"]
    assert cluster_case.graph.num_nodes == 8
    assert cluster_case.graph.num_edges == 8 - 1
    assert cluster_case.settings["cluster_depth"] == 5
    assert all(
        style is not None
        and style.stroke_width == pytest.approx(2.0)
        and style.font_size == pytest.approx(10.0)
        and style.padding == pytest.approx(35.0)
        for style in cluster_case.graph.cluster_styles.values()
    )
    assert tuple(float(value) for value in cluster_case.positions[0]) == pytest.approx(
        (-120.0, 60.0)
    )
    assert tuple(float(value) for value in cluster_case.positions[-1]) == pytest.approx(
        (240.0, -60.0)
    )


def test_build_cosmetic_album_renders_dagua_only_subset(tmp_path: Path) -> None:
    """Dagua-only cases should render without Graphviz."""

    output_dir = tmp_path / "album"
    result = build_cosmetic_album(
        output_dir=str(output_dir),
        case_ids=["corner_radius_6", "rich_label_bold_mixed"],
    )

    assert Path(result.manifest_path).exists()
    assert output_dir.joinpath("corner_radius", "corner_radius_6_dagua.png").exists()
    assert output_dir.joinpath("rich_labels", "bold_mixed_dagua.png").exists()

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["total_images"] == 2
    assert {row["case_id"] for row in manifest["cases"]} == {
        "corner_radius_6",
        "rich_label_bold_mixed",
    }


def test_build_cosmetic_album_renders_graphviz_subset(tmp_path: Path) -> None:
    """A Graphviz comparison case should emit an image and manifest row."""

    if shutil.which("dot") is None:
        pytest.skip("Graphviz dot is not installed")

    output_dir = tmp_path / "album"
    result = build_cosmetic_album(
        output_dir=str(output_dir),
        case_ids=["node_shape_rectangle"],
    )

    image_path = output_dir / "node_shapes" / "rectangle_dagua_vs_graphviz.png"
    assert Path(result.manifest_path).exists()
    assert image_path.exists()
    assert image_path.stat().st_size > 0

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["total_images"] == 1
    assert manifest["cases"][0]["case_id"] == "node_shape_rectangle"
    assert manifest["cases"][0]["competitor"] == "Graphviz dot"


def test_pairwise_comparison_cases_use_top_to_bottom_positions() -> None:
    """Shared comparison cases should place the source node above the target."""

    cases = {case.case_id: case for case in build_case_catalog()}

    for case_id in [
        "node_shape_rectangle",
        "arrow_head_normal",
        "border_style_dashed",
        "edge_style_dotted",
        "edge_routing_bezier",
    ]:
        positions = cases[case_id].positions
        assert float(positions[0, 1].item()) > float(positions[1, 1].item())


def test_pairwise_comparison_cases_use_compact_vertical_gap() -> None:
    """Two-node comparison cases should use the tighter Graphviz-matched gap."""

    cases = {case.case_id: case for case in build_case_catalog()}

    for case_id in [
        "arrow_head_normal",
        "border_style_dashed",
        "edge_style_dotted",
        "edge_routing_bezier",
    ]:
        positions = cases[case_id].positions
        vertical_gap = float(positions[0, 1].item() - positions[1, 1].item())
        assert vertical_gap == GRAPHVIZ_PAIR_VERTICAL_GAP


def test_direction_cases_use_wider_horizontal_spacing() -> None:
    """LR and RL direction demos should keep nodes visibly separated."""

    cases = {case.case_id: case for case in build_case_catalog()}

    assert torch.equal(
        cases["direction_lr"].positions,
        torch.tensor(
            [[0.0, 0.0], [160.0, 0.0], [320.0, 0.0]],
            dtype=torch.float32,
        ),
    )
    assert torch.equal(
        cases["direction_rl"].positions,
        torch.tensor(
            [[320.0, 0.0], [160.0, 0.0], [0.0, 0.0]],
            dtype=torch.float32,
        ),
    )


def test_cluster_cases_use_vertical_chain_positions() -> None:
    """Flat cluster demos should show a top-to-bottom chain through the cluster."""

    cases = {case.case_id: case for case in build_case_catalog()}
    expected_positions = torch.tensor(
        [[0.0, 80.0], [0.0, 0.0], [0.0, -80.0]],
        dtype=torch.float32,
    )

    assert torch.equal(cases["cluster_fill"].positions, expected_positions)
    assert torch.equal(cases["cluster_border"].positions, expected_positions)


def test_base_cluster_style_uses_visible_graphviz_matched_defaults() -> None:
    """Cluster defaults should keep borders visible with labels inside the box."""

    style = _base_cluster_style()

    assert style.stroke_width == 2.0
    assert style.padding == 50.0
    assert style.font_size == 13.0
    assert style.opacity == 0.9
    assert style.label_position == "top-left"
    assert style.label_offset == (12.0, 10.0)


def test_cluster_border_case_keeps_heavier_base_border_defaults() -> None:
    """The dashed cluster demo should only override fill and dash pattern."""

    cases = {case.case_id: case for case in build_case_catalog()}
    style = cases["cluster_border"].graph.cluster_styles["group"]

    assert style.fill == "#FFFFFF"
    assert style.stroke_dash == "dashed"
    assert style.stroke_width == 2.0
    assert style.opacity == 0.9
    assert style.label_position == "top-left"
    assert style.label_offset == (12.0, 10.0)


def test_ortho_routing_case_uses_offset_positions() -> None:
    """The orthogonal routing demo should offset x to expose the elbow segment."""

    cases = {case.case_id: case for case in build_case_catalog()}
    positions = cases["edge_routing_ortho"].positions

    assert torch.equal(
        positions,
        torch.tensor(
            [[-40.0, 55.0], [40.0, -55.0]],
            dtype=torch.float32,
        ),
    )
    assert not torch.isclose(positions[0, 0], positions[1, 0])


def test_build_cosmetic_album_dagua_only_requires_cached_competitors(tmp_path: Path) -> None:
    """Comparison-only iteration should fail fast when the competitor cache is absent."""

    with pytest.raises(ValueError, match="--dagua-only requires --cache-competitor"):
        build_cosmetic_album(
            output_dir=str(tmp_path / "album"),
            case_ids=["node_shape_rectangle"],
            dagua_only=True,
        )


def test_build_cosmetic_album_reports_missing_competitor_cache(tmp_path: Path) -> None:
    """Missing competitor cache entries should emit a placeholder manifest entry."""

    result = build_cosmetic_album(
        output_dir=str(tmp_path / "album"),
        case_ids=["node_shape_rectangle"],
        dagua_only=True,
        cache_competitor=True,
    )

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["total_images"] == 1
    assert manifest["cases"][0]["placeholder"] is True
    assert "Missing cached competitor render" in str(manifest["cases"][0]["render_error"])


def test_build_cosmetic_album_accepts_comma_delimited_category_filters(tmp_path: Path) -> None:
    """Comma-delimited category and case-id filters should be accepted."""

    result = build_cosmetic_album(
        output_dir=str(tmp_path / "album"),
        categories=["combo_2way,combo_3way"],
        case_ids=["combo_shadow_gradient,combo_bold_shadow_gradient"],
    )

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["total_images"] == 2
    assert {row["case_id"] for row in manifest["cases"]} == {
        "combo_shadow_gradient",
        "combo_bold_shadow_gradient",
    }
