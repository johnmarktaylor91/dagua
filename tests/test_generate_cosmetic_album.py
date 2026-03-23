"""Tests for the cosmetic album generator script."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch

from scripts.generate_cosmetic_album import (
    GRAPHVIZ_PAIR_VERTICAL_GAP,
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

    assert len(cases) == 121
    assert category_counts == {
        "arrow_types": 18,
        "border_styles": 3,
        "clusters": 3,
        "combo_2way": 20,
        "combo_3way": 10,
        "combo_4way": 5,
        "combo_5way": 3,
        "corner_radius": 3,
        "direction": 4,
        "edge_routing": 3,
        "edge_styles": 3,
        "evil_combos": 15,
        "gradients": 3,
        "node_shapes": 13,
        "opacity": 4,
        "rich_labels": 2,
        "shadows": 2,
        "text_formatting": 7,
    }


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
        "combo_hexagon_gradient",
        "combo_diamond_shadow",
        "combo_star_dotted",
        "combo_circle_double_border",
        "combo_lr_direction_ortho",
        "combo_opacity_gradient",
        "combo_italic_large_font",
        "combo_rounded_dashed_shadow",
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
        "combo_bold_shadow_gradient_rounded",
        "combo_diamond_dashed_opacity_italic",
        "combo_cluster_gradient_shadow_double_border",
        "combo_taper_crossing_gradient_thick",
        "combo_hexagon_striped_shadow_bold",
        "combo_kitchen_sink_1",
        "combo_kitchen_sink_2",
        "combo_kitchen_sink_3",
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
