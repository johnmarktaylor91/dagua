"""Regression tests for graphviz_strict cosmetic parity.

Locks in the per-feature in-tolerance gains achieved by metric-driven
optimization. Future theme work cannot regress the locked features.

See ``.project-context/knowledge/visual_tuning_workflow.md`` for the
postmortem driving this approach (replacing qualitative VLM audits with
quantitative metric loss).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


# Features that are 100% in tolerance as of round 19. Future commits cannot
# regress these without explicitly updating this list (failing CI is the
# signal that the regression happened).
LOCKED_FEATURES_100PCT = {
    "font_size_pt",
    "font_family",
    "node_fill",
    "node_stroke",
    "node_stroke_width_pt",
    "edge_stroke_color",
    "edge_stroke_width_pt",
    "bg_color",
    "margin_pt",
    "cluster_fill",
    "cluster_stroke",
    "cluster_stroke_width_pt",
    "cluster_label_font_size_pt",
}

# Features at >= 99% but with documented residual.
NEAR_LOCKED_FEATURES = {
    "arrow_length_pt": 99.0,
    "arrow_filled": 99.0,
    "arrow_width_pt": 98.0,  # 99.38% with principal-axis extraction; allow drift to 98%
}

# Global in-tolerance gate.
GLOBAL_IN_TOLERANCE_FLOOR_PCT = 85.0  # round-6 75x50pt theme floor leaves ellipse size strict


@pytest.mark.slow
def test_graphviz_strict_parity_metrics_lock() -> None:
    """Run parity_metrics.py and assert the locked features stay locked.

    This is the regression gate for graphviz_strict cosmetic parity. It
    requires the ``dot`` binary; skips otherwise.
    """

    if shutil.which("dot") is None:
        pytest.skip("graphviz dot binary not available")

    out_path = Path("/tmp/test_parity_metrics_lock.json")
    if out_path.exists():
        out_path.unlink()

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "parity_metrics.py"),
            "--out",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, f"parity_metrics.py failed: {proc.stderr[-500:]}"
    assert out_path.exists()

    data = json.loads(out_path.read_text())
    summary = data["summary"]
    by_feat = summary["by_feature_type"]

    # Global in-tolerance floor.
    pct = summary["in_tolerance_pct"]
    assert pct >= GLOBAL_IN_TOLERANCE_FLOOR_PCT, (
        f"global in-tolerance dropped to {pct:.2f}%, "
        f"below floor of {GLOBAL_IN_TOLERANCE_FLOOR_PCT:.2f}%. "
        f"see eval_output/parity_metrics_lock.json for failing features."
    )

    # Per-feature 100% lock.
    for feat in LOCKED_FEATURES_100PCT:
        if feat not in by_feat:
            continue
        feat_pct = by_feat[feat]["pct"]
        floor = 99.0 if feat == "ellipse_ry_pt" else 99.5
        assert feat_pct >= floor, (
            f"locked feature '{feat}' dropped to {feat_pct:.2f}%, "
            f"below {floor}% floor. graphviz_strict regressed."
        )

    # Near-locked feature gates.
    for feat, floor in NEAR_LOCKED_FEATURES.items():
        if feat not in by_feat:
            continue
        feat_pct = by_feat[feat]["pct"]
        assert feat_pct >= floor, (
            f"near-locked feature '{feat}' dropped to {feat_pct:.2f}%, below {floor}% floor."
        )
