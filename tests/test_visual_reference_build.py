"""Tests for the v2 visual reference guide builder (Lane D item 5)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from scripts.build_feature_reference import build_feature_reference_v2
from scripts.visual_parity.io import write_card_manifest, write_coverage_matrix, write_ledger
from scripts.visual_parity.types import (
    CARD_MANIFEST_SCHEMA_VERSION,
    COVERAGE_MATRIX_SCHEMA_VERSION,
    LEDGER_SCHEMA_VERSION,
)

MAX_PAGE_BYTES = 5 * 1024 * 1024


@pytest.fixture
def _stub_stores(tmp_path: Path) -> dict[str, Path]:
    """Write a minimal card manifest, coverage matrix, ledger, and refcache.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory.

    Returns
    -------
    dict[str, Path]
        Paths for ``card_manifest``, ``coverage_matrix``, ``ledger``, and
        ``refcache``.
    """

    card_manifest_path = tmp_path / "card_manifest.json"
    coverage_matrix_path = tmp_path / "coverage_matrix.json"
    ledger_path = tmp_path / "ledger.json"
    refcache_root = tmp_path / "refcache"
    (refcache_root / "graphviz").mkdir(parents=True)
    Image.new("RGB", (100, 80), "white").save(refcache_root / "graphviz" / "shape_ellipse.png")

    write_card_manifest(
        card_manifest_path,
        {
            "schema_version": CARD_MANIFEST_SCHEMA_VERSION,
            "generated_at": "ISO-8601",
            "cards": [
                {
                    "case_id": "shape_ellipse",
                    "category": "node_options",
                    "description": "ellipse shape",
                    "source": "v2_calibration_suite",
                    "kind": None,
                    "tier": None,
                    "reference_tool": "graphviz",
                    "reference_attr": "shape",
                    "reference_value": "ellipse",
                    "coverage_cell_id": "graphviz.node.shape.ellipse",
                    "target_kind": "svg_declared",
                    "size_policy": "auto",
                },
                {
                    "case_id": "arrowhead_normal",
                    "category": "edge_options",
                    "description": "normal arrowhead",
                    "source": "v2_calibration_suite",
                    "kind": None,
                    "tier": None,
                    "reference_tool": "graphviz",
                    "reference_attr": "arrowhead",
                    "reference_value": "normal",
                    "coverage_cell_id": None,
                    "target_kind": "svg_declared",
                    "size_policy": "auto",
                },
            ],
        },
    )
    write_coverage_matrix(
        coverage_matrix_path,
        {
            "schema_version": COVERAGE_MATRIX_SCHEMA_VERSION,
            "generated_at": "ISO-8601",
            "reference_pins": {},
            "source_snapshots": [],
            "adapter_capabilities": [],
            "cells": [
                {
                    "cell_id": "graphviz.node.shape.ellipse",
                    "tool": "graphviz",
                    "object": "node",
                    "attribute": "shape",
                    "value": "ellipse",
                    "value_group": "shape",
                    "source": "gv-shapes",
                    "dagua_field": "NodeStyle.shape",
                    "dagua_value": "ellipse",
                    "support_status": "supported",
                    "parity_status": "in_tolerance",
                    "priority": "P0",
                    "target_kind": "svg_declared",
                    "geometry_mode": "injected",
                }
            ],
        },
    )
    write_ledger(
        ledger_path,
        {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "run_id": "visual_parity_v2_2026_07",
            "created_at": "ISO-8601",
            "updated_at": "ISO-8601",
            "environment": {},
            "ratchets": {"global_in_tol_floor_pct": 85.0},
            "auditor": {},
            "rows": [],
            "knobs": [],
            "rounds": [],
            "residuals": [],
        },
    )
    return {
        "card_manifest": card_manifest_path,
        "coverage_matrix": coverage_matrix_path,
        "ledger": ledger_path,
        "refcache": refcache_root,
    }


def test_v2_build_produces_index_and_shapes_page_with_filled_slot(
    tmp_path: Path,
    _stub_stores: dict[str, Path],
) -> None:
    """The v2 guide should produce an index and a shapes page with a filled slot."""

    output_dir = tmp_path / "guide"
    markdown_index = tmp_path / "VISUAL_REFERENCE.md"

    result = build_feature_reference_v2(
        output_dir=output_dir,
        card_manifest_path=_stub_stores["card_manifest"],
        coverage_matrix_path=_stub_stores["coverage_matrix"],
        ledger_path=_stub_stores["ledger"],
        refcache_root=_stub_stores["refcache"],
        markdown_index_path=markdown_index,
    )

    assert Path(result.index_path).exists()
    assert "shapes" in result.page_paths
    shapes_page = Path(result.page_paths["shapes"])
    assert shapes_page.exists()
    assert result.filled_competitor_slots >= 1
    assert markdown_index.exists()

    shapes_html = shapes_page.read_text(encoding="utf-8")
    assert "shape_ellipse" in shapes_html
    assert "img/shape_ellipse.png" in shapes_html
    assert (output_dir / "img" / "shape_ellipse.png").exists()


def test_v2_build_pages_stay_under_size_cap(
    tmp_path: Path,
    _stub_stores: dict[str, Path],
) -> None:
    """Every generated HTML page should stay well under the 5 MB guide cap."""

    output_dir = tmp_path / "guide"
    markdown_index = tmp_path / "VISUAL_REFERENCE.md"

    result = build_feature_reference_v2(
        output_dir=output_dir,
        card_manifest_path=_stub_stores["card_manifest"],
        coverage_matrix_path=_stub_stores["coverage_matrix"],
        ledger_path=_stub_stores["ledger"],
        refcache_root=_stub_stores["refcache"],
        markdown_index_path=markdown_index,
    )

    all_pages = [Path(result.index_path)] + [Path(p) for p in result.page_paths.values()]
    for page in all_pages:
        assert page.stat().st_size < MAX_PAGE_BYTES


def test_v2_domain_classification_covers_shapes_and_arrowheads(
    tmp_path: Path,
    _stub_stores: dict[str, Path],
) -> None:
    """Cards should classify into their expected domain pages."""

    output_dir = tmp_path / "guide"
    markdown_index = tmp_path / "VISUAL_REFERENCE.md"

    result = build_feature_reference_v2(
        output_dir=output_dir,
        card_manifest_path=_stub_stores["card_manifest"],
        coverage_matrix_path=_stub_stores["coverage_matrix"],
        ledger_path=_stub_stores["ledger"],
        refcache_root=_stub_stores["refcache"],
        markdown_index_path=markdown_index,
    )

    assert result.domain_counts["shapes"] == 1
    assert result.domain_counts["arrowheads"] == 1


def test_v2_build_reads_status_badge_from_coverage_matrix(
    tmp_path: Path,
    _stub_stores: dict[str, Path],
) -> None:
    """A card linked to a coverage cell should show that cell's parity status."""

    output_dir = tmp_path / "guide"
    markdown_index = tmp_path / "VISUAL_REFERENCE.md"

    result = build_feature_reference_v2(
        output_dir=output_dir,
        card_manifest_path=_stub_stores["card_manifest"],
        coverage_matrix_path=_stub_stores["coverage_matrix"],
        ledger_path=_stub_stores["ledger"],
        refcache_root=_stub_stores["refcache"],
        markdown_index_path=markdown_index,
    )

    shapes_html = Path(result.page_paths["shapes"]).read_text(encoding="utf-8")
    assert "in_tolerance" in shapes_html
