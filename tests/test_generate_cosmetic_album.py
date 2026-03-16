"""Tests for the cosmetic album generator script."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch

from scripts.generate_cosmetic_album import build_case_catalog, build_cosmetic_album


def test_build_case_catalog_covers_expected_counts() -> None:
    """The case catalog should cover every requested cosmetic bucket."""

    cases = build_case_catalog()
    category_counts: dict[str, int] = {}
    for case in cases:
        category_counts[case.category] = category_counts.get(case.category, 0) + 1

    assert len(cases) == 68
    assert category_counts == {
        "arrow_types": 18,
        "border_styles": 3,
        "clusters": 3,
        "corner_radius": 3,
        "direction": 4,
        "edge_routing": 3,
        "edge_styles": 3,
        "gradients": 3,
        "node_shapes": 13,
        "opacity": 4,
        "rich_labels": 2,
        "shadows": 2,
        "text_formatting": 7,
    }


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
        [[0.0, 120.0], [0.0, 0.0], [0.0, -120.0]],
        dtype=torch.float32,
    )

    assert torch.equal(cases["cluster_fill"].positions, expected_positions)
    assert torch.equal(cases["cluster_border"].positions, expected_positions)


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
    """Missing competitor cache entries should surface a clear error."""

    with pytest.raises(RuntimeError, match="Missing cached competitor render"):
        build_cosmetic_album(
            output_dir=str(tmp_path / "album"),
            case_ids=["node_shape_rectangle"],
            dagua_only=True,
            cache_competitor=True,
        )
