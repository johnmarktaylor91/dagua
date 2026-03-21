"""Tests for the node border comparison image generator."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable, Tuple

import pytest

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

    assert graph.node_labels == ["A", "B", "C", "D", "E"]
    assert positions.shape == (5, 2)
    assert edge_pairs == {(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)}


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


def test_render_graphviz_comparison_uses_dot_engine(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The Graphviz comparison should render Graphviz via dot without fixed positions."""

    captured: dict[str, object] = {}

    def fake_render_dagua_graph(
        output_path: Path,
        title: str | None,
        graph: Any,
        positions: Any,
        *,
        figure_size: Tuple[float, float],
    ) -> None:
        """Record Dagua panel render arguments without rendering an image."""

        captured["dagua_title"] = title
        captured["dagua_labels"] = list(graph.node_labels)
        captured["dagua_positions_shape"] = tuple(positions.shape)
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

    assert captured["engine"] == "dot"
    assert captured["graphviz_positions"] is None
    assert captured["dagua_title"] is None
    assert captured["dagua_labels"] == ["A", "B", "C", "D", "E"]
    assert captured["graphviz_labels"] == ["A", "B", "C", "D", "E"]
    assert captured["panel_titles"] == ["Graphviz native", "Dagua"]


@pytest.mark.skipif(shutil.which("dot") is None, reason="Graphviz dot is not installed")
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
