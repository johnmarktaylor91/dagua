"""Tests for visual parity Graphviz geometry injection."""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Dict

import pytest

from scripts.visual_parity.geometry_injection import (
    graphviz_geometry,
    parse_graphviz_dot_geometry,
    parse_spline_pos,
    to_bezier_curves,
)


def _canned_xdot() -> str:
    """Return a minimal Graphviz ``-Tdot`` fixture.

    Returns
    -------
    str
        DOT text with one two-segment edge spline.
    """

    return """
digraph G {
  graph [bb="0,0,120,120"];
  n0 [height=0.5, pos="20,100", width=0.75];
  n1 [height=0.5, pos="100,20", width=0.75];
  n0 -> n1 [pos="e,100,20 20,100 40,100 50,80 60,70 70,60 80,40 100,20"];
}
"""


def test_parse_spline_pos_yields_segments_and_endpoint() -> None:
    """Canned Graphviz spline grammar should produce k cubic segments."""

    segments = parse_spline_pos(
        "n0->n1",
        "e,100,20 20,100 40,100 50,80 60,70 70,60 80,40 100,20",
    )

    assert len(segments) == 2
    assert segments[0].start == (20.0, 100.0)
    assert segments[-1].end == (100.0, 20.0)
    assert segments[0].endpoint == (100.0, 20.0)


def test_parse_graphviz_dot_geometry_extracts_canvas_nodes_and_edges() -> None:
    """Graphviz ``-Tdot`` output should parse into the frozen geometry schema."""

    geometry = parse_graphviz_dot_geometry(
        _canned_xdot(),
        case_id="case",
        tool_version="graphviz test",
        source_hash="abc",
    )

    assert geometry.canvas_pt == (120.0, 120.0)
    assert geometry.node_positions["n0"] == (20.0, 100.0)
    assert math.isclose(geometry.node_sizes["n0"][0], 54.0)
    assert len(geometry.edge_splines["n0->n1"]) == 2


def test_to_bezier_curves_samples_graphviz_spline_waypoints() -> None:
    """Spline conversion should preserve endpoints and mark graphviz routing."""

    geometry = parse_graphviz_dot_geometry(
        _canned_xdot(),
        case_id="case",
        tool_version="graphviz test",
        source_hash="abc",
    )
    curves = to_bezier_curves(geometry.edge_splines, samples_per_segment=24)
    curve = curves["n0->n1"]

    assert curve.routing == "graphviz_spline"
    assert curve.waypoints is not None
    assert len(curve.waypoints) == 47
    assert curve.waypoints[0] == (20.0, 100.0)
    assert curve.waypoints[-1] == (100.0, 20.0)


@pytest.mark.skipif(shutil.which("dot") is None, reason="Graphviz dot unavailable")
def test_graphviz_geometry_runs_separate_cached_outputs(tmp_path: Path) -> None:
    """Real Graphviz geometry should write separate SVG, DOT, and PNG outputs."""

    geometry = graphviz_geometry(
        "digraph G { a -> b; }",
        case_id="smoke",
        refcache_dir=tmp_path,
        dpi=96,
    )
    attrs: Dict[str, str] = geometry.graph_attrs

    assert Path(attrs["svg_path"]).exists()
    assert Path(attrs["xdot_path"]).exists()
    assert Path(attrs["png_path"]).exists()
    assert geometry.edge_splines
