"""Tests for per-card pixel diff perceptual metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.per_card_pixel_diff import (
    _compute_perceptual_metrics,
    _default_backend_for_gallery,
)


def test_compute_perceptual_metrics_identical_images() -> None:
    """Identical grayscale-converted RGB images should have perfect SSIM.

    Returns
    -------
    None
        Assertions validate the perceptual metric payload.
    """

    image = np.full((16, 16, 3), 255, dtype=np.uint8)

    metrics = _compute_perceptual_metrics(image, image.copy())

    assert metrics["ssim"] == 1.0
    assert metrics["ssim_loss"] == 0.0


def test_compute_perceptual_metrics_detects_structural_change() -> None:
    """A one-pixel structural stroke change should produce nonzero SSIM loss.

    Returns
    -------
    None
        Assertions validate that thin-feature changes are visible to SSIM.
    """

    reference = np.full((32, 32, 3), 255, dtype=np.uint8)
    changed = reference.copy()
    changed[:, 8, :] = 0

    metrics = _compute_perceptual_metrics(reference, changed)

    assert 0.0 <= metrics["ssim"] < 1.0
    assert metrics["ssim_loss"] > 0.0


def test_default_backend_for_gallery_uses_cairo_suffix() -> None:
    """Cairo gallery roots should render with cairo by default.

    Returns
    -------
    None
        Assertions validate audit-gallery backend inference.
    """

    assert _default_backend_for_gallery(Path("eval_output/gallery_audit")) == "agg"
    assert _default_backend_for_gallery(Path("eval_output/gallery_audit_cairo")) == "cairo"
