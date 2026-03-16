"""Tests for the cosmetic combo album generator."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from scripts.generate_combo_album import build_case_catalog, build_combo_album, build_combo_catalog


def test_build_case_catalog_covers_expected_counts() -> None:
    """The combo catalog should cover every requested category and case count."""

    cases = build_combo_catalog()
    category_counts: dict[str, int] = {}
    for case in cases:
        category_counts[case.category] = category_counts.get(case.category, 0) + 1

    assert len(cases) == 176
    assert category_counts == {
        "01_shape_x_border": 12,
        "02_shape_x_gradient": 8,
        "03_arrow_x_edgestyle": 12,
        "04_arrow_x_routing": 9,
        "05_arrow_proportions": 9,
        "06_arrow_head_tail": 8,
        "07_text_overflow": 12,
        "08_edge_labels": 10,
        "09_short_edges": 8,
        "10_self_loops_parallel": 8,
        "11_opacity_interactions": 6,
        "12_shadow_interactions": 6,
        "13_direction_x_routing": 8,
        "14_cluster_combos": 10,
        "15_color_contrast": 8,
        "16_dark_mode": 6,
        "17_extreme_params": 10,
        "18_dense_mixed": 8,
        "19_real_world_patterns": 6,
        "20_kitchen_sink": 12,
    }


def test_build_case_catalog_alias_matches_combo_catalog() -> None:
    """The legacy and new catalog entry points should return the same case IDs."""

    assert [case.case_id for case in build_case_catalog()] == [
        case.case_id for case in build_combo_catalog()
    ]


def test_build_combo_catalog_contains_part_two_cases() -> None:
    """Part 2 case IDs should be present in the catalog."""

    case_ids = {case.case_id for case in build_combo_catalog()}
    assert {
        "both_faded",
        "shadow_large_radius",
        "lr_dashed",
        "dashed_cluster_dashed_nodes",
        "black_gradient",
        "pipeline",
        "parallel_mixed_all",
    }.issubset(case_ids)


def test_build_combo_album_renders_dagua_only_subset(tmp_path: Path) -> None:
    """A Dagua-only category should render without Graphviz."""

    output_dir = tmp_path / "combo_album"
    result = build_combo_album(
        output_dir=str(output_dir),
        categories=["02_shape_x_gradient"],
    )

    assert Path(result.manifest_path).exists()
    assert (output_dir / "summary.md").exists()
    assert len(list((output_dir / "02_shape_x_gradient").glob("*.png"))) == 8

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["total_images"] == 8
    assert manifest["category_counts"] == {"02_shape_x_gradient": 8}
    assert all(row["comparison"] is False for row in manifest["cases"])


def test_build_combo_album_can_force_solo_panels(tmp_path: Path) -> None:
    """Comparison-capable cases should still render in Dagua-only mode."""

    output_dir = tmp_path / "combo_album"
    result = build_combo_album(
        output_dir=str(output_dir),
        categories=["01_shape_x_border"],
        dagua_only=True,
    )

    assert Path(result.manifest_path).exists()
    assert len(list((output_dir / "01_shape_x_border").glob("*.png"))) == 12

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["total_images"] == 12
    assert manifest["dagua_only"] is True
    assert all(row["render_mode"] == "dagua_only" for row in manifest["cases"])
    assert all(row["comparison"] is True for row in manifest["cases"])


def test_build_combo_album_renders_cluster_combo_subset(tmp_path: Path) -> None:
    """A Part 2 cluster category should render in Dagua-only mode."""

    output_dir = tmp_path / "combo_album"
    result = build_combo_album(
        output_dir=str(output_dir),
        categories=["14_cluster_combos"],
        dagua_only=True,
    )

    assert Path(result.manifest_path).exists()
    assert len(list((output_dir / "14_cluster_combos").glob("*.png"))) == 10

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["total_images"] == 10
    assert manifest["dagua_only"] is True
    assert all(row["render_mode"] == "dagua_only" for row in manifest["cases"])


def test_build_combo_album_renders_kitchen_sink_subset(tmp_path: Path) -> None:
    """Kitchen-sink Dagua-only cases should render without shadow/dash regressions."""

    output_dir = tmp_path / "combo_album"
    result = build_combo_album(
        output_dir=str(output_dir),
        categories=["20_kitchen_sink"],
        case_ids=["diamond_dashed_gradient_shadow", "selfloop_dashed_vee_label_shadow"],
        dagua_only=True,
    )

    assert Path(result.manifest_path).exists()
    assert len(list((output_dir / "20_kitchen_sink").glob("*.png"))) == 2

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["total_images"] == 2
    assert manifest["dagua_only"] is True
    assert {row["case_id"] for row in manifest["cases"]} == {
        "diamond_dashed_gradient_shadow",
        "selfloop_dashed_vee_label_shadow",
    }


def test_build_combo_album_renders_graphviz_category(tmp_path: Path) -> None:
    """A comparison category should emit Graphviz-backed images when available."""

    if shutil.which("dot") is None:
        pytest.skip("Graphviz dot is not installed")

    output_dir = tmp_path / "combo_album"
    result = build_combo_album(
        output_dir=str(output_dir),
        categories=["01_shape_x_border"],
    )

    assert Path(result.manifest_path).exists()
    assert len(list((output_dir / "01_shape_x_border").glob("*.png"))) == 12

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["total_images"] == 12
    assert manifest["dagua_only"] is False
    assert manifest["cases"][0]["comparison"] is True
    assert manifest["cases"][0]["render_mode"] == "comparison"


def test_build_combo_album_renders_cluster_graphviz_category(tmp_path: Path) -> None:
    """Cluster comparison cases should render with Graphviz when available."""

    if shutil.which("dot") is None:
        pytest.skip("Graphviz dot is not installed")

    output_dir = tmp_path / "combo_album"
    result = build_combo_album(
        output_dir=str(output_dir),
        categories=["14_cluster_combos"],
    )

    assert Path(result.manifest_path).exists()
    assert len(list((output_dir / "14_cluster_combos").glob("*.png"))) == 10

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["total_images"] == 10
    assert manifest["dagua_only"] is False
    assert any(row["comparison"] is True for row in manifest["cases"])
    assert any(row["render_mode"] == "comparison" for row in manifest["cases"])
