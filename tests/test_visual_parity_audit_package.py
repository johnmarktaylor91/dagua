"""Tests for visual parity v2 audit-package bundling and scorer fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from PIL import Image

from scripts.visual_parity.audit_package import build_audit_package
from scripts.visual_parity.select_vision_model import select_vision_models


def _write_image(path: Path, size: tuple[int, int], color: str = "white") -> None:
    """Write a simple RGB test image.

    Parameters
    ----------
    path
        Destination PNG path.
    size
        Image size in pixels.
    color
        Pillow color name.

    Returns
    -------
    None
        The image is written to disk.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _write_probe_manifest(path: Path) -> Dict[str, Any]:
    """Write a minimal 8-panel probe defect manifest.

    Parameters
    ----------
    path
        Destination manifest path.

    Returns
    -------
    dict[str, Any]
        Manifest payload written to disk.
    """

    defect_ids = [
        "wrong_font",
        "invisible_edge_stem",
        "truncated_label",
        "arrow_fill_mismatch",
        "cluster_border_missing_one_side",
        "size_normalization_artifact",
    ]
    control_ids = ["antialiased_residual", "true_noop_match"]
    payload: Dict[str, Any] = {
        "panel_count": 8,
        "defect_count": 6,
        "control_count": 2,
        "panels": [
            {"panel_id": panel_id, "is_control": False, "image_path": f"{panel_id}.png"}
            for panel_id in defect_ids
        ]
        + [
            {"panel_id": panel_id, "is_control": True, "image_path": f"{panel_id}.png"}
            for panel_id in control_ids
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def test_audit_package_bundles_images_metrics_and_prompt_variables(tmp_path: Path) -> None:
    """Audit package should copy round artifacts and expose prompt variables."""

    round_image = tmp_path / "round" / "pair.png"
    metric_summary = tmp_path / "round" / "metrics.json"
    pixel_summary = tmp_path / "round" / "pixel.json"
    prompt = tmp_path / "prompts" / "audit_v2.md"
    _write_image(round_image, (120, 80))
    metric_summary.write_text('{"metric": "ok"}\n', encoding="utf-8")
    pixel_summary.write_text('{"pixel": "ok"}\n', encoding="utf-8")
    prompt.parent.mkdir(parents=True)
    prompt.write_text("ROLE: test auditor\n", encoding="utf-8")

    result = build_audit_package(
        round_image_paths=[round_image],
        output_dir=tmp_path / "package",
        metric_summary_paths=[metric_summary],
        prompt_path=prompt,
        round_id="g005",
        pixel_triage_summary_path=pixel_summary,
        prior_findings=["old_gap"],
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["round_id"] == "g005"
    assert manifest["prompt"]["variables"]["N"] == 1
    assert manifest["prompt"]["variables"]["pair_image_paths"] == ["images/pair_001.png"]
    assert manifest["prompt"]["variables"]["metric_summaries"] == ["metrics/metrics.json"]
    assert manifest["prompt"]["variables"]["pixel_triage_summary"] == "metrics/pixel.json"
    assert manifest["prompt"]["variables"]["prior_findings"] == ["old_gap"]
    assert (result.output_dir / "images" / "pair_001.png").exists()
    assert (result.output_dir / "metrics" / "metrics.json").exists()
    assert (result.output_dir / "prompts" / "audit_v2.md").exists()


def test_canary_identity_stays_only_in_harness_sidecar(tmp_path: Path) -> None:
    """Audit-visible manifest must not reveal the corrupted canary panel."""

    round_image = tmp_path / "round" / "pair.png"
    canary_image = tmp_path / "tripwires" / "tw_stem.png"
    metric_summary = tmp_path / "round" / "metrics.json"
    prompt = tmp_path / "prompts" / "audit_v2.md"
    _write_image(round_image, (100, 100))
    _write_image(canary_image, (100, 100), color="black")
    metric_summary.write_text('{"metric": "ok"}\n', encoding="utf-8")
    prompt.parent.mkdir(parents=True)
    prompt.write_text("ROLE: test auditor\n", encoding="utf-8")

    result = build_audit_package(
        round_image_paths=[round_image],
        output_dir=tmp_path / "package",
        metric_summary_paths=[metric_summary],
        prompt_path=prompt,
        canary_image_path=canary_image,
        canary_tripwire_id="tw_stem",
    )

    manifest_text = result.manifest_path.read_text(encoding="utf-8")
    sidecar = json.loads(result.sidecar_path.read_text(encoding="utf-8"))
    assert "canary" not in manifest_text
    assert "tw_stem" not in manifest_text
    assert "tripwire" not in manifest_text
    assert sidecar["canary_panel_id"] == "pair_002"
    assert sidecar["canary_tripwire_id"] == "tw_stem"


def test_audit_package_downscales_images_to_cap(tmp_path: Path) -> None:
    """All packaged images should respect the configured side-length cap."""

    large_image = tmp_path / "round" / "large.png"
    prompt = tmp_path / "prompts" / "audit_v2.md"
    _write_image(large_image, (320, 120))
    prompt.parent.mkdir(parents=True)
    prompt.write_text("ROLE: test auditor\n", encoding="utf-8")

    result = build_audit_package(
        round_image_paths=[large_image],
        output_dir=tmp_path / "package",
        prompt_path=prompt,
        image_cap_px=100,
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    packaged_image = result.output_dir / manifest["images"][0]["image_path"]
    with Image.open(packaged_image) as image:
        assert max(image.size) <= 100
    assert manifest["images"][0]["max_side_px"] <= 100


def test_select_vision_model_scores_canned_response_fixture(tmp_path: Path) -> None:
    """The canned scorer fixture should produce primary, ceiling, fallback, and scores."""

    probe_manifest = tmp_path / "defect_manifest.json"
    _write_probe_manifest(probe_manifest)
    responses = Path("tests/fixtures/visual_parity/canned_model_responses.json")

    result = select_vision_models(responses, probe_manifest)

    assert result["primary_auditor"] == "candidate-primary"
    assert result["ceiling_auditor"] == "candidate-primary"
    assert result["fallback_auditor"] == "candidate-fallback"
    assert len(result["scores"]) == 3
    rejected = {row["model"]: row["rejected"] for row in result["scores"]}
    assert rejected["candidate-rejected"] is True
