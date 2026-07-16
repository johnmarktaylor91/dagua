"""Regression tests for visual parity metric profiles and locks."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import pytest
import torch

from dagua.render.mpl import _density_scaled_node_sizes, density_aware_size_factor
from dagua.styles import GRAPHVIZ_STRICT_THEME
from scripts.visual_parity.io import read_ledger

REPO_ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = REPO_ROOT / ".project-context/research/sprint_visual_parity_v2/ledger.json"
DEFAULT_GLOBAL_FLOOR_PCT = 85.0


def _metric_payload(by_feature: Mapping[str, float], global_pct: float = 100.0) -> Dict[str, Any]:
    """Build a small parity metric payload for lock assertions.

    Parameters
    ----------
    by_feature
        Mapping from feature id to percent in tolerance.
    global_pct
        Global in-tolerance percent.

    Returns
    -------
    dict[str, Any]
        Minimal metrics payload.
    """

    return {
        "summary": {
            "in_tolerance_pct": global_pct,
            "by_feature_type": {
                feature: {"pct": pct, "compared": 1, "in_tolerance": int(pct >= 99.5)}
                for feature, pct in by_feature.items()
            },
        }
    }


def _ledger_lock_config(ledger_path: Path = LEDGER_PATH) -> Dict[str, Any]:
    """Read the visual parity v2 ledger lock configuration.

    Parameters
    ----------
    ledger_path
        Ledger JSON path.

    Returns
    -------
    dict[str, Any]
        Global floor and locked feature floors.
    """

    ledger = read_ledger(ledger_path)
    ratchets = ledger.get("ratchets", {})
    locked: Dict[str, float] = {}
    for row in ledger.get("rows", []):
        if not row.get("locked"):
            continue
        for metric in row.get("metrics", []):
            metric_id = str(metric.get("metric_id", ""))
            if metric_id:
                locked[metric_id] = float(metric.get("lock_floor_pct", 99.5))
    return {
        "global_floor_pct": float(
            ratchets.get("global_in_tol_floor_pct", DEFAULT_GLOBAL_FLOOR_PCT)
        ),
        "locked_features": locked,
    }


def _assert_ledger_locks(payload: Mapping[str, Any], lock_config: Mapping[str, Any]) -> None:
    """Assert a metrics payload satisfies ledger-driven floors.

    Parameters
    ----------
    payload
        Metrics payload from ``scripts/parity_metrics.py``.
    lock_config
        Lock configuration from :func:`_ledger_lock_config`.

    Returns
    -------
    None
        Raises an assertion on failure.
    """

    summary = payload["summary"]
    assert float(summary["in_tolerance_pct"]) >= float(lock_config["global_floor_pct"])
    by_feature = summary["by_feature_type"]
    for feature, floor in lock_config["locked_features"].items():
        assert feature in by_feature, f"locked feature {feature!r} missing from metrics"
        assert float(by_feature[feature]["pct"]) >= float(floor)


def test_density_aware_size_factor_matches_graphviz_fixture_density() -> None:
    """Assert sparse pair fixtures stay fixed while dense fixtures shrink."""

    assert density_aware_size_factor(2, 400.0) == pytest.approx(1.0)
    assert density_aware_size_factor(5, 400.0) < 1.0
    assert density_aware_size_factor(20, 400.0) == pytest.approx(0.25)


def test_graphviz_strict_render_preserves_computed_node_floor() -> None:
    """Graphviz-strict rendering should not shrink computed node dimensions."""

    computed_sizes = torch.tensor([[54.0, 36.0]] * 3, dtype=torch.float32)

    rendered_sizes, factor = _density_scaled_node_sizes(
        computed_sizes,
        node_count=3,
        layout_extent_pt=144.0,
        enabled=GRAPHVIZ_STRICT_THEME.graph_style.density_aware_node_shrink,
    )

    assert factor == pytest.approx(1.0)
    assert torch.equal(rendered_sizes, computed_sizes)


def test_ledger_locked_feature_missing_is_failure(tmp_path: Path) -> None:
    """Assert ledger locks fail when a locked feature is absent."""

    ledger = {
        "schema_version": 2,
        "rows": [
            {
                "locked": True,
                "metrics": [{"metric_id": "node_fill", "lock_floor_pct": 99.5}],
            }
        ],
        "knobs": [],
        "rounds": [],
        "residuals": [],
        "ratchets": {"global_in_tol_floor_pct": 85.0},
    }
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    lock_config = _ledger_lock_config(ledger_path)
    with pytest.raises(AssertionError, match="locked feature"):
        _assert_ledger_locks(_metric_payload({}), lock_config)


def test_current_ledger_global_floor_is_enforced() -> None:
    """Assert the current v2 ledger floor is consumed by the parity test."""

    lock_config = _ledger_lock_config()
    locked_payload = {
        feature: float(floor) for feature, floor in lock_config["locked_features"].items()
    }
    _assert_ledger_locks(
        _metric_payload(locked_payload, global_pct=float(lock_config["global_floor_pct"])),
        lock_config,
    )


@pytest.mark.skipif(shutil.which("dot") is None, reason="graphviz dot binary not available")
def test_v1_profile_output_is_byte_compatible(tmp_path: Path) -> None:
    """Assert explicit v1 emits the same JSON bytes as the default profile."""

    default_out = tmp_path / "default.json"
    v1_out = tmp_path / "v1.json"
    base_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "parity_metrics.py"),
        "--quick",
        "--cases",
        "tiny_graph",
    ]
    default_proc = subprocess.run(
        [*base_cmd, "--out", str(default_out)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert default_proc.returncode == 0, default_proc.stderr[-500:]
    v1_proc = subprocess.run(
        [*base_cmd, "--profile", "v1", "--out", str(v1_out)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert v1_proc.returncode == 0, v1_proc.stderr[-500:]
    assert default_out.read_bytes() == v1_out.read_bytes()


@pytest.mark.skipif(shutil.which("dot") is None, reason="graphviz dot binary not available")
def test_v2_profile_quick_emits_new_feature_rows(tmp_path: Path) -> None:
    """Assert the v2 profile emits provenance and additive feature families."""

    out_path = tmp_path / "v2.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "parity_metrics.py"),
            "--quick",
            "--cases",
            "tiny_graph",
            "--profile",
            "v2",
            "--out",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-500:]
    data = json.loads(out_path.read_text())
    assert data["provenance"]["reference_kind"] == "svg_declared"
    by_feature = data["summary"]["by_feature_type"]
    assert "node_autosize_w_pt" in by_feature
    assert "node_autosize_h_pt" in by_feature
    assert "label_glyph_extent_pt" in by_feature
    assert "spline_path_dist_pt" in by_feature
