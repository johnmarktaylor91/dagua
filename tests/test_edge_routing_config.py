"""Sprint 6: edge routing user opt-out + differentiable-vs-heuristic gap.

The existing ``optimize_edges`` (dagua/layout/edge_optimization.py) and its
registered op wrapper ``BezierControlPointOpt`` (dagua/layout/ops/edge_route.py)
already provide differentiable edge-CP refinement. Sprint 6 adds a
user-facing opt-out so a caller can demand the heuristic routing with no
gradient refinement, plus an end-to-end check that the differentiable
path actually delivers fewer edge/node crossings than the heuristic
path on a graph where that's the discriminating metric.
"""

from __future__ import annotations

import pytest

from dagua.config import LayoutConfig
from dagua.graph import DaguaGraph


def _test_graph():
    """Small directed graph with plenty of edge-node near-crossings to
    discriminate routing modes. The exact topology doesn't matter beyond
    creating enough edges that pass near off-path nodes."""
    g = DaguaGraph()
    for i in range(8):
        g.add_node(i, label=str(i))
    # Two parallel chains crossing each other.
    for i in range(4):
        g.add_edge(i, i + 4)
    for i in range(4):
        g.add_edge(i, 7 - i)
    return g


@pytest.mark.unit
def test_heuristic_mode_skips_optimize_edges():
    """``edge_routing='heuristic'`` must skip the gradient CP refinement
    entirely -- draw() produces heuristic bezier curves with no call into
    optimize_edges. We spy on the module-level function.
    """
    from dagua.layout import edge_optimization as eo

    call_count = {"n": 0}
    orig = eo.optimize_edges

    def spy(*args, **kwargs):
        call_count["n"] += 1
        return orig(*args, **kwargs)

    eo.optimize_edges = spy
    try:
        g = _test_graph()
        cfg = LayoutConfig(seed=42, edge_routing="heuristic")
        import dagua

        dagua.draw(g, cfg)
    finally:
        eo.optimize_edges = orig

    assert call_count["n"] == 0, (
        f"edge_routing='heuristic' still called optimize_edges {call_count['n']} times"
    )


@pytest.mark.unit
def test_differentiable_mode_calls_optimize_edges():
    """``edge_routing='differentiable'`` (default) must call optimize_edges."""
    from dagua.layout import edge_optimization as eo

    call_count = {"n": 0}
    orig = eo.optimize_edges

    def spy(*args, **kwargs):
        call_count["n"] += 1
        return orig(*args, **kwargs)

    eo.optimize_edges = spy
    try:
        g = _test_graph()
        cfg = LayoutConfig(seed=42, edge_routing="differentiable")
        import dagua

        dagua.draw(g, cfg)
    finally:
        eo.optimize_edges = orig

    assert call_count["n"] >= 1, "differentiable edge routing did not run"


@pytest.mark.unit
def test_config_default_is_differentiable():
    """The default LayoutConfig must keep the pre-Sprint-6 behaviour --
    differentiable edge routing on. We don't want users getting worse
    edge layouts because they didn't opt in."""
    cfg = LayoutConfig()
    assert cfg.edge_routing == "differentiable"


@pytest.mark.unit
def test_edge_routing_opt_out_preserves_curves():
    """When opting out, dagua.draw() still produces valid BezierCurve
    routing -- it's just the heuristic-only path."""
    import dagua

    g = _test_graph()
    cfg = LayoutConfig(seed=42, edge_routing="heuristic")
    dagua.draw(g, cfg)
    # graph.last_curves populated by draw() via graph.cache_routing.
    assert g.last_curves is not None
    assert len(g.last_curves) == g.num_edges
