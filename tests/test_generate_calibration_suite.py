"""Tests for the rendering calibration suite generator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

import scripts.generate_calibration_suite as calibration


def _write_placeholder_png(output_path: Path) -> None:
    """Write a small placeholder PNG used by cache-behavior tests.

    Parameters
    ----------
    output_path : Path
        Destination PNG path.

    Returns
    -------
    None
        Writes the placeholder image to disk.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (120, 90), "#FFFFFF")
    image.save(output_path)


def test_build_case_catalog_covers_expected_categories() -> None:
    """The catalog should cover every requested output category."""

    cases = calibration.build_case_catalog()
    category_counts: dict[str, int] = {}
    case_ids = {case.case_id for case in cases}
    for case in cases:
        category_counts[case.category] = category_counts.get(case.category, 0) + 1

    assert len(cases) == 113
    assert category_counts == {
        "cluster_options": 7,
        "combinations_2way": 5,
        "combinations_3way": 3,
        "edge_options": 45,
        "extreme_values": 7,
        "node_options": 29,
        "scaling": 4,
        "text_options": 13,
    }
    assert {
        "lineweight_0.25",
        "arrowhead_normal",
        "shape_rect",
        "rich_text",
        "nested_3",
        "weight_x_style",
        "parallel_edges",
        "graph_32in",
    }.issubset(case_ids)


def test_build_calibration_suite_renders_subset(tmp_path: Path, monkeypatch) -> None:
    """A filtered subset should render images and a manifest."""

    output_dir = tmp_path / "calibration"

    monkeypatch.setattr(calibration, "_graphviz_available", lambda: True)
    monkeypatch.setattr(
        calibration,
        "_render_graphviz_png",
        lambda dot_source, output_path, engine: _write_placeholder_png(output_path),
    )
    monkeypatch.setattr(
        calibration,
        "_render_matplotlib_png",
        lambda scene, output_path: _write_placeholder_png(output_path),
    )

    result = calibration.build_calibration_suite(
        output_dir=str(output_dir),
        case_ids=["lineweight_0.25"],
    )

    image_path = output_dir / "edge_options" / "lineweight_0.25.png"
    assert Path(result.manifest_path).exists()
    assert image_path.exists()
    assert image_path.stat().st_size > 0

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["total_images"] == 1
    assert manifest["cases"][0]["case_id"] == "lineweight_0.25"
    assert Path(manifest["cases"][0]["graphviz_cache"]).exists()
    assert Path(manifest["cases"][0]["matplotlib_cache"]).exists()


def test_build_calibration_suite_reuses_cached_references(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Cached Graphviz and matplotlib references should be reused on rerun."""

    output_dir = tmp_path / "calibration"
    calls = {"graphviz": 0, "matplotlib": 0}

    def fake_graphviz(dot_source: str, output_path: Path, engine: str) -> None:
        """Record a Graphviz render and emit a placeholder PNG.

        Parameters
        ----------
        dot_source : str
            DOT source string.
        output_path : Path
            Destination image path.
        engine : str
            Graphviz engine name.

        Returns
        -------
        None
            Writes the placeholder PNG.
        """

        del dot_source, engine
        calls["graphviz"] += 1
        _write_placeholder_png(output_path)

    def fake_matplotlib(scene: calibration.CalibrationScene, output_path: Path) -> None:
        """Record a matplotlib reference render and emit a placeholder PNG.

        Parameters
        ----------
        scene : calibration.CalibrationScene
            Render scene.
        output_path : Path
            Destination image path.

        Returns
        -------
        None
            Writes the placeholder PNG.
        """

        del scene
        calls["matplotlib"] += 1
        _write_placeholder_png(output_path)

    monkeypatch.setattr(calibration, "_graphviz_available", lambda: True)
    monkeypatch.setattr(calibration, "_render_graphviz_png", fake_graphviz)
    monkeypatch.setattr(calibration, "_render_matplotlib_png", fake_matplotlib)

    calibration.build_calibration_suite(
        output_dir=str(output_dir),
        case_ids=["lineweight_0.25"],
    )
    calibration.build_calibration_suite(
        output_dir=str(output_dir),
        case_ids=["lineweight_0.25"],
    )

    assert calls == {"graphviz": 1, "matplotlib": 1}


def test_build_graphviz_dot_emits_positions_for_all_nodes() -> None:
    """Every emitted node should carry a fixed-position ``pos=`` attribute."""

    scene = calibration._edge_scene(
        [("0.5 pt", calibration._base_edge_style(width=0.5))],
        columns=1,
    )

    dot_source = calibration._build_graphviz_dot(scene)

    assert calibration._dot_has_pos_for_all_nodes(dot_source) is True


def test_render_graphviz_png_uses_dot_without_n2_when_requested(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Native ``dot`` renders should not force fixed-position mode."""

    calls: list[list[str]] = []

    class _CompletedProcess:
        """Minimal subprocess result used by the render-command test."""

        returncode = 0
        stderr = ""

    def fake_run(
        command: list[str],
        *,
        input: str,
        text: bool,
        capture_output: bool,
        check: bool,
    ) -> _CompletedProcess:
        """Capture the Graphviz command and emulate a successful render."""

        del input, text, capture_output, check
        calls.append(command)
        output_path = Path(command[-1])
        _write_placeholder_png(output_path)
        return _CompletedProcess()

    monkeypatch.setattr(calibration.subprocess, "run", fake_run)

    output_path = tmp_path / "graphviz.png"
    calibration._render_graphviz_png(
        'digraph G {\n  n0 [label="A", pos="0,0!"];\n}\n',
        output_path,
        "dot",
    )

    assert output_path.exists()
    assert calls == [
        ["dot", f"-Gdpi={calibration.RAW_RENDER_DPI}", "-Tpng", "-o", str(output_path)]
    ]


def test_render_graphviz_png_uses_n2_for_fixed_position_neato(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Explicit non-``dot`` engines should keep ``-n2`` when positions exist."""

    calls: list[list[str]] = []

    class _CompletedProcess:
        """Minimal subprocess result used by the render-command test."""

        returncode = 0
        stderr = ""

    def fake_run(
        command: list[str],
        *,
        input: str,
        text: bool,
        capture_output: bool,
        check: bool,
    ) -> _CompletedProcess:
        """Capture the Graphviz command and emulate a successful render."""

        del input, text, capture_output, check
        calls.append(command)
        output_path = Path(command[-1])
        _write_placeholder_png(output_path)
        return _CompletedProcess()

    monkeypatch.setattr(calibration.subprocess, "run", fake_run)

    output_path = tmp_path / "graphviz.png"
    calibration._render_graphviz_png(
        'digraph G {\n  n0 [label="A", pos="0,0!"];\n}\n',
        output_path,
        "neato",
    )

    assert output_path.exists()
    assert calls == [
        ["neato", "-n2", f"-Gdpi={calibration.RAW_RENDER_DPI}", "-Tpng", "-o", str(output_path)]
    ]


def test_resolved_scene_figsize_tracks_scene_bounds() -> None:
    """Auto-sized figures should respond to the graph's actual bounds."""

    compact_scene = calibration._edge_scene(
        [("0.5 pt", calibration._base_edge_style(width=0.5))],
        columns=1,
    )
    wide_scene = calibration._scaling_scene()

    compact_width, compact_height = calibration._resolved_scene_figsize(compact_scene)
    wide_width, wide_height = calibration._resolved_scene_figsize(wide_scene)

    assert compact_width < 6.0
    assert compact_height < 6.0
    assert wide_width > compact_width
    assert wide_width > wide_height


def test_resolved_scene_figsize_ignores_scene_figsize_override() -> None:
    """Raw render sizing should ignore per-scene composed-size metadata."""

    uncapped_scene = calibration._edge_scene(
        [("0.5 pt", calibration._base_edge_style(width=0.5))],
        columns=1,
    )
    capped_scene = calibration._edge_scene(
        [("0.5 pt", calibration._base_edge_style(width=0.5))],
        columns=1,
        figsize=(1.4, 1.4),
    )

    assert calibration._resolved_scene_figsize(capped_scene) == pytest.approx(
        calibration._resolved_scene_figsize(uncapped_scene)
    )


def test_scaling_cases_reuse_one_content_matched_raw_figsize() -> None:
    """Scaling cases should vary only the composed comparison size."""

    cases = calibration._scaling_cases()
    baseline_scene = calibration._scaling_scene()
    baseline_positions = baseline_scene.positions.detach().clone()
    baseline_raw_figsize = calibration._resolved_scene_figsize(baseline_scene)

    previous_height = 0.0
    for case, expected_width in zip(cases, [4.0, 8.0, 16.0, 32.0]):
        scene = case.build_scene()
        assert torch.equal(scene.positions, baseline_positions)
        assert scene.figsize is None
        assert calibration._resolved_scene_figsize(scene) == pytest.approx(baseline_raw_figsize)
        width, height = case.comparison_figsize
        assert width == pytest.approx(expected_width)
        assert height >= previous_height
        previous_height = height


def test_base_cluster_style_uses_visible_border_defaults() -> None:
    """Cluster defaults should keep borders dark enough and wide enough to see."""

    style = calibration._base_cluster_style()

    assert style.stroke == "#64748B"
    assert style.stroke_width >= 1.0


def test_build_graphviz_dot_emits_cluster_penwidth() -> None:
    """Cluster DOT output should forward the configured border width."""

    style = calibration._base_cluster_style()
    scene = calibration._cluster_chain_scene(style, depth=1)

    dot_source = calibration._build_graphviz_dot(scene)

    assert f'penwidth="{style.stroke_width:.2f}"' in dot_source


def test_edge_scene_preserves_requested_edge_width() -> None:
    """Edge width overrides should survive scene construction unchanged."""

    thin_scene = calibration._edge_scene(
        [("0.5 pt", calibration._base_edge_style(width=0.5))],
        columns=1,
    )
    thick_scene = calibration._edge_scene(
        [("4.0 pt", calibration._base_edge_style(width=4.0))],
        columns=1,
    )

    assert thin_scene.graph.get_style_for_edge(0).width == 0.5
    assert thick_scene.graph.get_style_for_edge(0).width == 4.0


def test_scene_bounds_include_self_loop_extent() -> None:
    """Self-loop bounds should extend above the node boxes."""

    scene = calibration._self_loop_scene()
    scene.graph.compute_node_sizes()
    positions = scene.positions.detach().cpu().numpy()
    sizes = scene.graph.node_sizes.detach().cpu().numpy()
    node_top = float((positions[:, 1] + sizes[:, 1] / 2.0).max())

    _, _, _, y_max = calibration._scene_content_bounds(scene)

    assert y_max > node_top + 20.0
