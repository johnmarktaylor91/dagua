"""Tests for visual parity v2 tripwire interlocks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.visual_parity import tripwires

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_all_tripwires_present_and_passing() -> None:
    """Assert all 11 tripwires exist and pass."""

    results = tripwires.run_all()
    by_id = {result.tripwire_id: result for result in results}
    assert set(by_id) == {
        "tw_font",
        "tw_stem",
        "tw_trunc",
        "tw_kern",
        "tw_arrowfill",
        "tw_arroworder",
        "tw_size",
        "tw_color",
        "tw_spline",
        "tw_cluster",
        "tw_scalehide",
    }
    for tripwire_id, result in by_id.items():
        assert result.status == "pass"
        assert result.observed_effect["clean_fired"] is False
        assert result.observed_effect["injected_fired"] is True


def test_kern_and_trunc_move_label_glyph_extent_in_predicted_direction() -> None:
    """Assert clip and kern defects move label_glyph_extent_pt enough."""

    by_id = {result.tripwire_id: result for result in tripwires.run_all()}
    trunc = by_id["tw_trunc"].observed_effect
    kern = by_id["tw_kern"].observed_effect
    assert trunc["injected"] > trunc["clean"]
    assert kern["injected"] > kern["clean"]


def test_tripwire_cli_writes_canonical_and_requested_reports(tmp_path: Path) -> None:
    """Assert ``--all`` exits zero and writes both report locations."""

    requested = tmp_path / "tripwires.json"
    canonical = REPO_ROOT / tripwires.CANONICAL_STATUS_PATH
    if canonical.exists():
        canonical.unlink()
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.visual_parity.tripwires",
            "--all",
            "--out",
            str(requested),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(requested.read_text())
    assert report["status"] == "pass"
    assert canonical.exists()
    assert json.loads(canonical.read_text())["status"] == "pass"


def test_threshold_weakening_makes_all_exit_nonzero(tmp_path: Path) -> None:
    """Assert deliberately impossible thresholds fail the suite."""

    requested = tmp_path / "tripwires_fail.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.visual_parity.tripwires",
            "--all",
            "--threshold-scale",
            "1000",
            "--out",
            str(requested),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 1
    report = json.loads(requested.read_text())
    assert report["status"] == "fail"
    assert report["failed_metric_ids"]


def test_spline_tripwire_fires_after_e1_wiring() -> None:
    """Assert the spline tripwire fires on a flattened spline."""

    by_id = {result.tripwire_id: result for result in tripwires.run_all()}
    spline = by_id["tw_spline"]
    assert spline.status == "pass"
    assert spline.observed_effect["clean"] == pytest.approx(0.0)
    assert spline.observed_effect["injected"] > 3.0
