"""Visual parity v2 tests for the pixel diff runner."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import pytest
from PIL import Image


def _load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON object from disk.

    Parameters
    ----------
    path
        JSON path.

    Returns
    -------
    dict[str, Any]
        Parsed JSON object.
    """

    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.skipif(shutil.which("dot") is None, reason="Graphviz dot unavailable")
def test_quick_svg_cairo_run_emits_v2_summary(tmp_path: Path) -> None:
    """Quick svg-cairo run should emit v2 metrics and alignment manifests."""

    out_dir = tmp_path / "svg"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/parity_pixel_diff.py",
            "--quick",
            "--cases",
            "tiny_graph",
            "--reference",
            "svg-cairo",
            "--bit-equivalent",
            "--inject-splines",
            "--out",
            str(out_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, result.stderr
    payload = _load_json(out_dir / "summary.json")
    assert payload["summary"]["total_panels"] > 0
    assert payload["png_raster"] is None
    first = payload["panels"][0]
    assert first["edges"]
    assert "corridor_ink_ratio" in first["edges"][0]
    assert "edge_centerline_dist_px" in first["edges"][0]
    assert first["alignment_manifest"]["metric_uses_crop"] is False
    with Image.open(first["paths"]["composite"]) as composite:
        assert max(composite.size) <= 2000


@pytest.mark.skipif(shutil.which("dot") is None, reason="Graphviz dot unavailable")
def test_dot_png_reference_uses_png_raster_lane(tmp_path: Path) -> None:
    """dot-png runs should report only in the separate png_raster section."""

    out_dir = tmp_path / "png"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/parity_pixel_diff.py",
            "--quick",
            "--cases",
            "tiny_graph",
            "--reference",
            "dot-png",
            "--inject-splines",
            "--out",
            str(out_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, result.stderr
    payload = _load_json(out_dir / "summary.json")
    assert payload["summary"] == {}
    assert payload["panels"] == []
    assert payload["png_raster"]["summary"]["total_panels"] == 1
    assert payload["png_raster"]["panels"][0]["reference_kind"] == "dot-png"
