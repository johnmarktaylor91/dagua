"""Tests for the cosmetic combo album generator script."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from scripts.generate_combo_album import build_case_catalog, build_combo_album


def test_build_case_catalog_covers_expected_counts() -> None:
    """The combo catalog should cover every requested category and case count."""

    cases = build_case_catalog()
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


def test_build_combo_album_renders_dagua_only_subset(tmp_path: Path) -> None:
    """Dagua-only iteration mode should render comparison-capable cases as solo panels."""

    output_dir = tmp_path / "combo_album"
    result = build_combo_album(
        output_dir=str(output_dir),
        case_ids=["star_dashed", "dark_baseline"],
        dagua_only=True,
    )

    manifest_path = Path(result.manifest_path)
    summary_path = output_dir / "summary.md"
    assert manifest_path.exists()
    assert summary_path.exists()
    assert output_dir.joinpath("01_shape_x_border", "star_dashed.png").exists()
    assert output_dir.joinpath("16_dark_mode", "dark_baseline.png").exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["total_images"] == 2
    assert manifest["dagua_only"] is True
    assert manifest["category_counts"] == {
        "01_shape_x_border": 1,
        "16_dark_mode": 1,
    }
    assert {row["case_id"] for row in manifest["cases"]} == {"star_dashed", "dark_baseline"}
    assert {row["render_mode"] for row in manifest["cases"]} == {"dagua_only"}
    assert "## Category Index" in summary_path.read_text(encoding="utf-8")


def test_build_combo_album_renders_graphviz_subset(tmp_path: Path) -> None:
    """A comparison-capable case should render a Graphviz comparison when Graphviz is installed."""

    if shutil.which("dot") is None or shutil.which("neato") is None:
        pytest.skip("Graphviz executables are not installed")

    output_dir = tmp_path / "combo_album"
    result = build_combo_album(
        output_dir=str(output_dir),
        case_ids=["star_dashed"],
    )

    image_path = output_dir / "01_shape_x_border" / "star_dashed.png"
    manifest_path = Path(result.manifest_path)
    summary_path = output_dir / "summary.md"

    assert image_path.exists()
    assert image_path.stat().st_size > 0
    assert manifest_path.exists()
    assert summary_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["total_images"] == 1
    assert manifest["cases"][0]["case_id"] == "star_dashed"
    assert manifest["cases"][0]["comparison"] is True
    assert manifest["cases"][0]["render_mode"] == "comparison"
