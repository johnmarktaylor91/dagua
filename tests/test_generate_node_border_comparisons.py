"""Tests for the node border comparison image generator."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable, Tuple

import pytest
from PIL import Image, ImageDraw

import scripts.generate_node_border_comparisons as generator
from scripts.generate_node_border_comparisons import (
    REQUESTED_FILENAMES,
    _cluster_graph,
    _default_dag,
    _graphviz_comparison_dag,
    _mixed_shapes_graph,
    _render_generator_map,
    _shape_demo_specs,
    generate_node_border_comparisons,
)


def test_requested_filenames_match_generator_map() -> None:
    """The generator map should cover the full requested artifact set."""

    generator_map = _render_generator_map()

    assert len(REQUESTED_FILENAMES) == 16
    assert tuple(generator_map.keys()) == REQUESTED_FILENAMES


def test_generate_node_border_comparisons_renders_subset(tmp_path: Path) -> None:
    """A mixed subset should render manual, graph, and comparison panels."""

    filenames = [
        "border_weight_ladder.png",
        "default_nodes.png",
        "mpl_native_comparison.png",
    ]

    rendered_paths = generate_node_border_comparisons(str(tmp_path), filenames=filenames)

    assert [Path(path).name for path in rendered_paths] == filenames
    for path_str in rendered_paths:
        path = Path(path_str)
        assert path.exists()
        assert path.stat().st_size > 0


def test_graphviz_comparison_graph_matches_requested_dag() -> None:
    """The Graphviz comparison should use the requested five-node DAG."""

    graph, positions = _graphviz_comparison_dag()
    edge_pairs = {
        (int(graph.edge_index[0, index].item()), int(graph.edge_index[1, index].item()))
        for index in range(graph.edge_index.shape[1])
    }

    assert graph.node_labels == ["Input", "Process", "Transform", "Validate", "Output"]
    assert positions.shape == (5, 2)
    assert edge_pairs == {(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)}


def test_default_and_graphviz_core_dags_share_the_same_showcase_scene() -> None:
    """The two core images should compare the same labeled DAG."""

    default_graph, default_positions = _default_dag()
    comparison_graph, comparison_positions = _graphviz_comparison_dag()

    assert default_graph.direction == "TB"
    assert comparison_graph.direction == "TB"
    assert default_graph.node_labels == comparison_graph.node_labels
    assert tuple(default_positions.flatten().tolist()) == tuple(
        comparison_positions.flatten().tolist()
    )


@pytest.mark.parametrize(
    "builder",
    [_default_dag, _graphviz_comparison_dag, _mixed_shapes_graph, _cluster_graph],
)
def test_core_graph_scenes_use_straight_edge_routing(
    builder: Callable[[], Tuple[Any, Any]],
) -> None:
    """The fixed-position graph scenes should avoid bezier loop artifacts."""

    graph, _ = builder()

    for edge_index in range(graph.edge_index.shape[1]):
        assert graph.get_style_for_edge(edge_index).routing == "straight"


def test_solid_shape_showcase_uses_shared_fill_and_stroke() -> None:
    """The solid gallery should compare geometry with one consistent style."""

    specs = _shape_demo_specs("solid")

    assert len(specs) == 13
    assert {spec.style.fill for spec in specs} == {specs[0].style.fill}
    assert {spec.style.stroke for spec in specs} == {specs[0].style.stroke}
    assert {spec.style.stroke_dash for spec in specs} == {"solid"}


def test_render_graphviz_comparison_uses_shared_positions_for_graphviz_native(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The comparison should render Graphviz natively from the shared fixed scene."""

    captured: dict[str, object] = {}

    def fake_render_dagua_graph(
        output_path: Path,
        title: str | None,
        graph: Any,
        positions: Any,
        *,
        figure_size: Tuple[float, float],
        margin: float = 26.0,
    ) -> None:
        """Record Dagua panel render arguments without rendering an image."""

        captured["dagua_title"] = title
        captured["dagua_labels"] = list(graph.node_labels)
        captured["dagua_positions_shape"] = tuple(positions.shape)
        captured["dagua_margin"] = margin
        output_path.write_bytes(b"dagua")

    def fake_render_graphviz_native(
        graph: Any,
        output_path: Path,
        *,
        engine: str,
        positions: Any = None,
    ) -> None:
        """Record Graphviz render arguments without invoking Graphviz."""

        captured["engine"] = engine
        captured["graphviz_positions"] = positions
        captured["graphviz_labels"] = list(graph.node_labels)
        output_path.write_bytes(b"graphviz")

    def fake_compose_image_panels(
        output_path: Path,
        title: str,
        panels: Any,
        *,
        figure_size: Tuple[float, float],
    ) -> None:
        """Record panel composition arguments without opening raster inputs."""

        captured["composed_title"] = title
        captured["panel_titles"] = [panel_title for panel_title, _ in panels]
        output_path.write_bytes(b"comparison")

    monkeypatch.setattr(generator, "_render_dagua_graph", fake_render_dagua_graph)
    monkeypatch.setattr(generator, "_render_graphviz_native", fake_render_graphviz_native)
    monkeypatch.setattr(generator, "_compose_image_panels", fake_compose_image_panels)

    generator._render_graphviz_comparison(tmp_path / "graphviz_comparison.png")

    assert captured["engine"] == "neato"
    assert captured["graphviz_positions"] is not None
    assert tuple(captured["graphviz_positions"].shape) == (5, 2)
    assert captured["dagua_title"] is None
    assert captured["dagua_labels"] == ["Input", "Process", "Transform", "Validate", "Output"]
    assert captured["graphviz_labels"] == ["Input", "Process", "Transform", "Validate", "Output"]
    assert captured["panel_titles"] == ["Graphviz native", "Dagua"]


def test_normalize_panel_raster_matches_content_scale_across_sources(tmp_path: Path) -> None:
    """Normalization should remove extra whitespace before panel composition."""

    image_specs = [
        ("tight.png", (240, 160), (50, 45, 190, 115)),
        ("loose.png", (420, 280), (140, 105, 280, 175)),
    ]
    output_size = (600, 420)
    normalized_images: list[Any] = []

    for filename, size, rectangle in image_specs:
        image = Image.new("RGB", size, "white")
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle(rectangle, radius=16, fill="#DCEEFF", outline="#4B6E88", width=4)
        path = tmp_path / filename
        image.save(path)
        normalized_images.append(generator._normalize_panel_raster(path, canvas_size=output_size))

    def _content_bbox(image_array: Any) -> tuple[int, int, int, int]:
        """Return the non-white bounding box for a normalized panel raster."""

        mask = (255 - image_array[..., :3]).sum(axis=2) > 18
        y_indices, x_indices = mask.nonzero()
        return (
            int(x_indices.min()),
            int(y_indices.min()),
            int(x_indices.max()) + 1,
            int(y_indices.max()) + 1,
        )

    first_bbox = _content_bbox(normalized_images[0])
    second_bbox = _content_bbox(normalized_images[1])

    assert normalized_images[0].shape == (output_size[1], output_size[0], 3)
    assert normalized_images[1].shape == (output_size[1], output_size[0], 3)
    assert first_bbox == second_bbox


@pytest.mark.skipif(shutil.which("neato") is None, reason="Graphviz neato is not installed")
def test_generate_node_border_comparisons_renders_graphviz_panel(tmp_path: Path) -> None:
    """Graphviz comparison generation should emit the requested output file."""

    rendered_paths = generate_node_border_comparisons(
        str(tmp_path),
        filenames=["graphviz_comparison.png"],
    )

    graphviz_path = Path(rendered_paths[0])
    assert graphviz_path.name == "graphviz_comparison.png"
    assert graphviz_path.exists()
    assert graphviz_path.stat().st_size > 0
