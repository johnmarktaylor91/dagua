"""Tests for the comprehensive cosmetic gallery generator."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_comprehensive_gallery import (
    EXTERNAL_LABEL_PANEL_SIZE,
    WIDE_DIRECTION_PANEL_SIZE,
    SweepConfig,
    _panel_size_for_sweep,
    apply_sweep_value,
    build_comprehensive_gallery,
    build_graph,
    build_sweep_catalog,
)


def _sweep_by_name(name: str) -> SweepConfig:
    """Return a sweep config by its stable name.

    Parameters
    ----------
    name : str
        Sweep identifier.

    Returns
    -------
    SweepConfig
        Matching sweep configuration.
    """

    return next(entry for entry in build_sweep_catalog() if entry.name == name)


def test_build_sweep_catalog_covers_expected_counts() -> None:
    """The sweep catalog should cover the requested cosmetic buckets."""

    sweeps = build_sweep_catalog()
    category_counts: dict[str, int] = {}
    for sweep in sweeps:
        category_counts[sweep.category] = category_counts.get(sweep.category, 0) + 1

    assert len(sweeps) == 55
    assert category_counts == {
        "clusters": 6,
        "edges/advanced": 4,
        "edges/arrows": 4,
        "edges/labels": 3,
        "edges/routing": 1,
        "edges/styles": 4,
        "graph": 3,
        "nodes/borders": 8,
        "nodes/effects": 4,
        "nodes/fills": 6,
        "nodes/shapes": 1,
        "nodes/text": 11,
    }

    taper_sweep = _sweep_by_name("edge_taper")
    assert taper_sweep.field == "taper_width_end"
    assert taper_sweep.values == [3.0, 2.0, 1.0, 0.5, 0.1]

    arrow_length_sweep = _sweep_by_name("edge_arrow_length")
    assert arrow_length_sweep.values == [5.0, 10.0, 20.0, 35.0]

    arrow_width_sweep = _sweep_by_name("edge_arrow_width")
    assert arrow_width_sweep.values == [3.0, 7.0, 14.0, 25.0]

    arrow_types_sweep = _sweep_by_name("edge_arrow_types")
    assert arrow_types_sweep.values == [
        "normal",
        "vee",
        "dot",
        "diamond",
        "tee",
        "crow",
        "circle",
        "open",
    ]

    head_tail_sweep = _sweep_by_name("edge_head_tail_labels")
    assert head_tail_sweep.values == ["none", "head_tail", "in_out", "src_dst"]


def test_apply_sweep_value_sets_node_special_cases() -> None:
    """Node sweep helpers should apply their required companion fields."""

    gradient_mode_sweep = _sweep_by_name("node_gradient")
    graph, _ = build_graph(gradient_mode_sweep, "radial")
    apply_sweep_value(graph, gradient_mode_sweep, "radial")
    node_style = graph.node_styles[0]

    assert node_style is not None
    assert node_style.gradient == "radial"
    assert node_style.fill == "#4C7AE6"
    assert node_style.gradient_color == "#FF9A4A"
    assert node_style.font_color == "#F7F8FA"
    assert node_style.min_width == 100.0
    assert node_style.min_height == 60.0

    gradient_sweep = _sweep_by_name("node_gradient_angle")
    graph, _ = build_graph(gradient_sweep, 45.0)
    apply_sweep_value(graph, gradient_sweep, 45.0)
    node_style = graph.node_styles[0]

    assert node_style is not None
    assert node_style.gradient == "linear"
    assert node_style.gradient_angle == 45.0
    assert node_style.fill == "#4C7AE6"
    assert node_style.gradient_color == "#FF9A4A"

    outline_sweep = _sweep_by_name("node_text_outline")
    graph, _ = build_graph(outline_sweep, True)
    apply_sweep_value(graph, outline_sweep, True)
    node_style = graph.node_styles[0]

    assert node_style is not None
    assert node_style.text_outline is True
    assert node_style.text_outline_color == "#0072B2"
    assert node_style.text_outline_width == 2.0

    shape_sweep = _sweep_by_name("node_shape")
    graph, _ = build_graph(shape_sweep, "double_circle")
    apply_sweep_value(graph, shape_sweep, "double_circle")
    node_style = graph.node_styles[0]

    assert node_style is not None
    assert node_style.shape == "double_circle"
    assert node_style.stroke_width == 0.0

    graph, _ = build_graph(shape_sweep, "star")
    apply_sweep_value(graph, shape_sweep, "star")
    node_style = graph.node_styles[0]

    assert node_style is not None
    assert node_style.shape == "star"
    assert node_style.min_width == 80.0
    assert node_style.min_height == 60.0
    assert graph.node_labels[0] == "star"

    pie_chart_sweep = _sweep_by_name("node_pie_chart")
    graph, _ = build_graph(pie_chart_sweep, [3.0, 2.0, 1.0])
    apply_sweep_value(graph, pie_chart_sweep, [3.0, 2.0, 1.0])
    node_style = graph.node_styles[0]

    assert node_style is not None
    assert node_style.fill_pattern == "pie"
    assert node_style.shape == "circle"
    assert "shape" in node_style._set_fields
    assert node_style.min_width == 96.0
    assert node_style.min_height == 96.0
    assert graph.node_labels[0] == ""

    donut_sweep = _sweep_by_name("node_donut")
    graph, _ = build_graph(donut_sweep, 0.4)
    apply_sweep_value(graph, donut_sweep, 0.4)
    node_style = graph.node_styles[0]

    assert node_style is not None
    assert node_style.fill_pattern == "pie"
    assert node_style.shape == "circle"
    assert node_style.min_width == 96.0
    assert node_style.min_height == 96.0

    stroke_width_sweep = _sweep_by_name("node_stroke_width")
    graph, _ = build_graph(stroke_width_sweep, 5.0)
    apply_sweep_value(graph, stroke_width_sweep, 5.0)
    node_style = graph.node_styles[0]

    assert node_style is not None
    assert node_style.min_width == 100.0
    assert node_style.min_height == 60.0

    corner_radius_sweep = _sweep_by_name("node_corner_radius")
    assert corner_radius_sweep.values == [0.0, 4.0, 8.0, 12.0, 20.0]

    graph, _ = build_graph(corner_radius_sweep, 20.0)
    apply_sweep_value(graph, corner_radius_sweep, 20.0)
    node_style = graph.node_styles[0]

    assert node_style is not None
    assert node_style.shape == "roundrect"
    assert node_style.corner_radius == 20.0
    assert node_style.min_width == 108.0

    border_position_sweep = _sweep_by_name("node_border_position")
    graph, _ = build_graph(border_position_sweep, "outside")
    apply_sweep_value(graph, border_position_sweep, "outside")
    node_style = graph.node_styles[0]

    assert node_style is not None
    assert node_style.shape == "roundrect"
    assert node_style.border_position == "outside"
    assert node_style.stroke_width == 4.0

    stroke_cap_sweep = _sweep_by_name("node_stroke_cap")
    graph, _ = build_graph(stroke_cap_sweep, "round")
    apply_sweep_value(graph, stroke_cap_sweep, "round")
    node_style = graph.node_styles[0]

    assert node_style is not None
    assert node_style.shape == "rect"
    assert node_style.stroke_dash == "dashed"
    assert node_style.stroke_width == 4.0

    stroke_join_sweep = _sweep_by_name("node_stroke_join")
    graph, _ = build_graph(stroke_join_sweep, "bevel")
    apply_sweep_value(graph, stroke_join_sweep, "bevel")
    node_style = graph.node_styles[0]

    assert node_style is not None
    assert node_style.shape == "star"
    assert node_style.stroke_join == "bevel"
    assert node_style.stroke_width == 5.0
    assert node_style.min_width == 120.0
    assert node_style.min_height == 120.0
    assert graph.node_labels[0] == ""

    text_valign_sweep = _sweep_by_name("node_text_valign")
    graph, _ = build_graph(text_valign_sweep, "top")
    apply_sweep_value(graph, text_valign_sweep, "top")
    node_style = graph.node_styles[0]

    assert node_style is not None
    assert node_style.shape == "rect"
    assert node_style.corner_radius == 0.0
    assert node_style.min_height == 100.0

    text_rotation_sweep = _sweep_by_name("node_text_rotation")
    graph, _ = build_graph(text_rotation_sweep, 90.0)
    apply_sweep_value(graph, text_rotation_sweep, 90.0)
    node_style = graph.node_styles[0]

    assert node_style is not None
    assert node_style.shape == "rect"
    assert node_style.min_width == 100.0
    assert node_style.min_height == 100.0
    assert graph.node_labels[0] == "Rotate"

    text_align_sweep = _sweep_by_name("node_text_align")
    graph, _ = build_graph(text_align_sweep, "left")
    apply_sweep_value(graph, text_align_sweep, "left")
    node_style = graph.node_styles[0]

    assert node_style is not None
    assert node_style.shape == "rect"
    assert node_style.corner_radius == 0.0
    assert node_style.min_width == 120.0
    assert node_style.min_height == 90.0
    assert "shape" in node_style._set_fields
    assert graph.node_labels[0] == "First Line\nSecond Line\nThird Line"

    text_wrap_sweep = _sweep_by_name("node_text_wrap")
    graph, _ = build_graph(text_wrap_sweep, "wrap")
    apply_sweep_value(graph, text_wrap_sweep, "wrap")
    node_style = graph.node_styles[0]

    assert node_style is not None
    assert node_style.shape == "rect"
    assert node_style.text_max_width == 80.0
    assert node_style.min_width == 110.0
    assert node_style.min_height == 72.0
    assert graph.node_labels[0] == "Wrap this sample label across a few lines"

    external_label_sweep = _sweep_by_name("node_external_label")
    graph, _ = build_graph(external_label_sweep, "right")
    apply_sweep_value(graph, external_label_sweep, "right")
    node_style = graph.node_styles[0]

    assert node_style is not None
    assert node_style.external_label == "ID 42"
    assert graph._theme.graph_style.margin == 20.0
    assert _panel_size_for_sweep(external_label_sweep) == EXTERNAL_LABEL_PANEL_SIZE

    overflow_sweep = _sweep_by_name("node_overflow_policy")
    graph, _ = build_graph(overflow_sweep, "expand_node")
    apply_sweep_value(graph, overflow_sweep, "expand_node")
    node_style = graph.node_styles[0]

    assert node_style is not None
    assert node_style.shape == "rect"
    assert node_style.font_size == 5.5
    assert node_style.min_width == 84.0
    assert node_style.min_height == 40.0
    assert graph.node_labels[0] == "Overflow demo"


def test_apply_sweep_value_sets_edge_and_graph_special_cases() -> None:
    """Edge and graph sweep helpers should set linked fields consistently."""

    taper_sweep = _sweep_by_name("edge_taper")
    graph, _ = build_graph(taper_sweep, 3.0)
    apply_sweep_value(graph, taper_sweep, 3.0)
    edge_style = graph.edge_styles[0]

    assert edge_style is not None
    assert edge_style.taper is False
    assert edge_style.taper_width_start == 3.0
    assert edge_style.taper_width_end == 3.0

    graph, _ = build_graph(taper_sweep, 0.1)
    apply_sweep_value(graph, taper_sweep, 0.1)
    edge_style = graph.edge_styles[0]

    assert edge_style is not None
    assert edge_style.taper is True
    assert edge_style.taper_width_end == 0.1

    gradient_sweep = _sweep_by_name("edge_color_gradient")
    graph, _ = build_graph(gradient_sweep, "source_to_target")
    apply_sweep_value(graph, gradient_sweep, "source_to_target")
    edge_style = graph.edge_styles[0]
    node_style = graph.node_styles[0]

    assert edge_style is not None
    assert edge_style.color == "#0057FF"
    assert edge_style.color_gradient_end == "#FF6A00"
    assert edge_style.width == 3.0
    assert node_style is not None
    assert node_style.shape == "rect"
    assert graph.node_labels == ["Start\n#0057FF", "End\n#FF6A00"]

    arrow_length_sweep = _sweep_by_name("edge_arrow_length")
    graph, _ = build_graph(arrow_length_sweep, 35.0)
    apply_sweep_value(graph, arrow_length_sweep, 35.0)
    edge_style = graph.edge_styles[0]

    assert edge_style is not None
    assert edge_style.width == 2.5
    assert edge_style.arrow_node_fraction == 0.0

    labels_sweep = _sweep_by_name("edge_head_tail_labels")
    graph, _ = build_graph(labels_sweep, "head_tail")
    apply_sweep_value(graph, labels_sweep, "head_tail")
    edge_style = graph.edge_styles[0]
    node_style = graph.node_styles[0]

    assert edge_style is not None
    assert node_style is not None
    assert edge_style.head_label == "Head"
    assert edge_style.tail_label == "Tail"
    assert edge_style.label_font_size == 9.0
    assert edge_style.head_label_offset == 12.0
    assert node_style.shape == "rect"

    arrow_types_sweep = _sweep_by_name("edge_arrow_types")
    graph, _ = build_graph(arrow_types_sweep, "diamond")
    apply_sweep_value(graph, arrow_types_sweep, "diamond")
    node_styles = [style for style in graph.node_styles if style is not None]

    assert len(node_styles) == 2
    assert all(style.shape == "rect" for style in node_styles)
    assert all("shape" in style._set_fields for style in node_styles)
    assert graph.node_labels == ["A", "B"]

    crossing_style_sweep = _sweep_by_name("edge_crossing_style")
    graph, _ = build_graph(crossing_style_sweep, "arc")
    apply_sweep_value(graph, crossing_style_sweep, "arc")
    edge_styles = [style for style in graph.edge_styles if style is not None]

    assert len(edge_styles) == 2
    assert all(style.crossing_style == "arc" for style in edge_styles)
    assert all(style.crossing_size == 12.0 for style in edge_styles)
    assert all(style.width == 3.0 for style in edge_styles)

    shadow_sweep = _sweep_by_name("node_shadow")
    graph, _ = build_graph(shadow_sweep, False)
    apply_sweep_value(graph, shadow_sweep, False)
    shadow_off_style = graph.node_styles[0]

    graph, _ = build_graph(shadow_sweep, True)
    apply_sweep_value(graph, shadow_sweep, True)
    shadow_on_style = graph.node_styles[0]

    assert shadow_off_style is not None
    assert shadow_on_style is not None
    assert shadow_off_style.fill == shadow_on_style.fill == "#DCEBFA"
    assert shadow_off_style.stroke == shadow_on_style.stroke == "#4C77A3"

    background_sweep = _sweep_by_name("graph_background")
    graph, _ = build_graph(background_sweep, "#0F0F10")
    apply_sweep_value(graph, background_sweep, "#0F0F10")
    node_style = graph.node_styles[0]
    edge_style = graph.edge_styles[0]

    assert node_style is not None
    assert edge_style is not None
    assert node_style.font_color == "#F7F8FA"
    assert node_style.fill == "#24303B"
    assert edge_style.color == "#9FB0C0"
    assert graph._theme.graph_style.background_color == "#0F0F10"


def test_build_graph_uses_expected_showcase_positions() -> None:
    """Direction and edge showcase sweeps should use fixed diagnostic positions."""

    sweep = _sweep_by_name("graph_direction")
    _, positions_lr = build_graph(sweep, "LR")
    _, positions_rl = build_graph(sweep, "RL")

    assert positions_lr.tolist() == [[-110.0, 0.0], [0.0, 0.0], [110.0, 0.0]]
    assert positions_rl.tolist() == [[110.0, 0.0], [0.0, 0.0], [-110.0, 0.0]]

    routing_sweep = _sweep_by_name("edge_routing")
    _, routing_positions = build_graph(routing_sweep, "ortho")
    assert routing_positions.tolist() == [[-92.0, 62.0], [92.0, -62.0]]

    arrow_sweep = _sweep_by_name("edge_arrow_length")
    _, arrow_positions = build_graph(arrow_sweep, 35.0)
    assert arrow_positions.tolist() == [[0.0, 82.0], [0.0, -82.0]]

    curvature_sweep = _sweep_by_name("edge_curvature")
    _, curvature_positions = build_graph(curvature_sweep, 1.0)
    assert curvature_positions.tolist() == [[-90.0, 55.0], [90.0, -55.0]]


def test_graph_direction_uses_compact_labels_and_wider_panels() -> None:
    """Direction sweeps should reserve more horizontal space for LR and RL layouts."""

    sweep = _sweep_by_name("graph_direction")
    graph, _ = build_graph(sweep, "LR")
    apply_sweep_value(graph, sweep, "LR")
    node_styles = [style for style in graph.node_styles if style is not None]

    assert graph.node_labels == ["A", "B", "C"]
    assert all(style.min_width == 72.0 for style in node_styles)
    assert all(style.min_height == 42.0 for style in node_styles)
    assert _panel_size_for_sweep(sweep) == WIDE_DIRECTION_PANEL_SIZE


def test_build_comprehensive_gallery_renders_filtered_subset(tmp_path: Path) -> None:
    """Filtered rendering should emit the selected sweep and manifest files."""

    output_dir = tmp_path / "gallery"
    result = build_comprehensive_gallery(
        output_dir=str(output_dir),
        sweep="node_gradient",
    )

    manifest_path = Path(result.manifest_path)
    assert manifest_path.exists()
    assert (output_dir / "summary.md").exists()
    assert (output_dir / "nodes" / "fills" / "sweep_node_gradient.png").exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["total_images"] == 1
    assert manifest["total_sweeps"] == 1
    assert manifest["sweeps"][0]["name"] == "node_gradient"
    assert manifest["sweeps"][0]["image_path"] == "nodes/fills/sweep_node_gradient.png"
