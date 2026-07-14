"""Tests for the visual parity two-panel compositor."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from PIL import Image, ImageDraw

from scripts.visual_parity.compose import MAX_OUTPUT_SIDE_PX, compose_pair
from scripts.visual_parity.types import GeometryMode


def _write_panel(path: Path, size: Tuple[int, int], ink_box: Tuple[int, int, int, int]) -> None:
    """Write a simple test panel.

    Parameters
    ----------
    path
        Destination image path.
    size
        Image size.
    ink_box
        Rectangle ink bounds.

    Returns
    -------
    None
        The image is written.
    """

    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle(ink_box, fill="black")
    image.save(path)


def test_compose_pair_preserves_deliberate_size_mismatch(tmp_path: Path) -> None:
    """Shared crop must not normalize one side independently."""

    ref = tmp_path / "ref.png"
    dagua = tmp_path / "dagua.png"
    out = tmp_path / "pair.png"
    _write_panel(ref, (120, 80), (10, 10, 30, 30))
    _write_panel(dagua, (180, 120), (140, 90, 170, 110))

    manifest = compose_pair(
        ref,
        dagua,
        out,
        case_id="case",
        round_id="r001",
        reference_label="graphviz svg-cairo",
        geometry_mode=GeometryMode.INJECTED,
        dpi=100,
    )

    assert out.exists()
    assert manifest.crop_box_px == (0, 0, 180, 120)
    assert manifest.pixel_size == (180, 120)
    assert manifest.metric_uses_crop is False


def test_compose_pair_asserts_output_under_cap(tmp_path: Path) -> None:
    """The compositor should cap every emitted comparison image."""

    ref = tmp_path / "ref.png"
    dagua = tmp_path / "dagua.png"
    out = tmp_path / "pair.png"
    _write_panel(ref, (2600, 400), (10, 10, 2580, 390))
    _write_panel(dagua, (2600, 400), (20, 20, 2570, 380))

    compose_pair(
        ref,
        dagua,
        out,
        case_id="wide",
        round_id="r001",
        reference_label="graphviz svg-cairo",
        geometry_mode=GeometryMode.NATIVE,
    )

    with Image.open(out) as image:
        assert max(image.size) <= MAX_OUTPUT_SIDE_PX
