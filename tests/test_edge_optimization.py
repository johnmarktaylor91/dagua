"""Smoke tests for edge optimization, label placement, and new metrics."""

import pytest
import torch

from dagua.edges import BezierCurve, route_edges, place_edge_labels, evaluate_bezier, bezier_tangent
from dagua.graph import DaguaGraph
from dagua.config import LayoutConfig
from dagua.styles import EdgeStyle, NodeStyle


# --- Fixtures ---

@pytest.fixture
def simple_graph():
    """4-node DAG: 0->1, 0->2, 1->3, 2->3."""
    g = DaguaGraph()
    for i in range(4):
        g.add_node(i, label=f"node_{i}")
    g.add_edge(0, 1, label="e01")
    g.add_edge(0, 2, label="e02")
    g.add_edge(1, 3, label="e13")
    g.add_edge(2, 3, label="e23")
    g.compute_node_sizes()
    return g


@pytest.fixture
def simple_positions():
    return torch.tensor([
        [0.0, 0.0],
        [-30.0, 60.0],
        [30.0, 60.0],
        [0.0, 120.0],
    ])


# --- Phase 1: Style fields ---

@pytest.mark.smoke
def test_edge_style_new_fields():
    """EdgeStyle has curvature, label_position, port_style, label_avoidance."""
    es = EdgeStyle()
    assert es.label_position == 0.5
    assert es.curvature == 0.4
    assert es.port_style == "distributed"
    assert es.label_avoidance is True


@pytest.mark.smoke
def test_node_style_overflow_fields():
    """NodeStyle has overflow_policy and min_font_size."""
    ns = NodeStyle()
    assert ns.overflow_policy == "shrink_text"
    assert ns.min_font_size == 5.0


@pytest.mark.smoke
def test_minimal_theme_straight_edges():
    """MINIMAL_THEME uses curvature=0.0."""
    from dagua.styles import MINIMAL_THEME
    es = MINIMAL_THEME.get_edge_style("default")
    assert es.curvature == 0.0


# --- Phase 2: Node sizing ---

@pytest.mark.smoke
def test_compute_node_size_returns_3tuple():
    """compute_node_size returns (w, h, font_size)."""
    from dagua.utils import compute_node_size
    result = compute_node_size("hello")
    assert len(result) == 3
    w, h, fs = result
    assert w > 0 and h > 0 and fs > 0


@pytest.mark.smoke
def test_shrink_text_policy():
    """shrink_text policy reduces font size for long labels."""
    from dagua.utils import compute_node_size
    long_label = "a" * 200
    w, h, fs = compute_node_size(long_label, overflow_policy="shrink_text", min_font_size=5.0)
    assert fs <= 8.5  # should have shrunk from default


@pytest.mark.smoke
def test_expand_node_policy():
    """expand_node policy allows wider aspect ratio."""
    from dagua.utils import compute_node_size
    long_label = "a" * 100
    w_expand, _, _ = compute_node_size(long_label, overflow_policy="expand_node")
    w_overflow, _, _ = compute_node_size(long_label, overflow_policy="overflow")
    # expand_node allows up to 10:1 vs 6:1
    assert w_expand >= w_overflow


@pytest.mark.smoke
def test_node_font_sizes_tensor(simple_graph):
    """compute_node_sizes populates node_font_sizes tensor."""
    assert simple_graph.node_font_sizes is not None
    assert simple_graph.node_font_sizes.shape[0] == simple_graph.num_nodes


# --- Phase 3: Curvature threading ---

@pytest.mark.smoke
def test_curvature_zero_gives_straight(simple_graph, simple_positions):
    """Curvature=0 produces straight-line edges (cp == endpoints)."""
    # Override edge style to curvature=0
    for i in range(len(simple_graph.edge_styles)):
        simple_graph.edge_styles[i] = EdgeStyle(curvature=0.0)

    curves = route_edges(simple_positions, simple_graph.edge_index,
                         simple_graph.node_sizes, "TB", simple_graph)
    for c in curves:
        # Control points should equal endpoints (degenerate bezier)
        assert c.cp1 == c.p0 or (abs(c.cp1[0] - c.p0[0]) < 1e-6 and abs(c.cp1[1] - c.p0[1]) < 1e-6)


@pytest.mark.smoke
def test_center_port_style(simple_graph, simple_positions):
    """port_style='center' puts all ports at node center x."""
    for i in range(len(simple_graph.edge_styles)):
        simple_graph.edge_styles[i] = EdgeStyle(port_style="center")

    curves = route_edges(simple_positions, simple_graph.edge_index,
                         simple_graph.node_sizes, "TB", simple_graph)
    # All source ports should be at node center x
    src_indices = simple_graph.edge_index[0].tolist()
    for e_idx, c in enumerate(curves):
        node_x = simple_positions[src_indices[e_idx], 0].item()
        assert abs(c.p0[0] - node_x) < 1e-6


# --- Phase 4: Edge optimization ---

@pytest.mark.smoke
def test_optimize_edges_runs(simple_graph, simple_positions):
    """optimize_edges runs without error and returns same number of curves."""
    from dagua.layout.edge_optimization import optimize_edges

    curves = route_edges(simple_positions, simple_graph.edge_index,
                         simple_graph.node_sizes, "TB", simple_graph)
    config = LayoutConfig()
    config.edge_opt_steps = 10  # fast
    config.edge_opt_lr = 0.1

    optimized = optimize_edges(curves, simple_positions, simple_graph.edge_index,
                               simple_graph.node_sizes, config, simple_graph)
    assert len(optimized) == len(curves)
    # Each curve should still be a BezierCurve
    for c in optimized:
        assert isinstance(c, BezierCurve)


@pytest.mark.smoke
def test_optimize_edges_zero_steps(simple_graph, simple_positions):
    """edge_opt_steps=0 returns original curves unchanged."""
    from dagua.layout.edge_optimization import optimize_edges

    curves = route_edges(simple_positions, simple_graph.edge_index,
                         simple_graph.node_sizes, "TB", simple_graph)
    config = LayoutConfig()
    config.edge_opt_steps = 0

    result = optimize_edges(curves, simple_positions, simple_graph.edge_index,
                            simple_graph.node_sizes, config, simple_graph)
    assert result is curves


# --- Phase 5: Edge label placement ---

@pytest.mark.smoke
def test_place_edge_labels(simple_graph, simple_positions):
    """place_edge_labels returns positions for labeled edges, None for unlabeled."""
    curves = route_edges(simple_positions, simple_graph.edge_index,
                         simple_graph.node_sizes, "TB", simple_graph)
    positions = place_edge_labels(curves, simple_positions, simple_graph.node_sizes,
                                  simple_graph.edge_labels, simple_graph)
    assert len(positions) == len(curves)
    for i, p in enumerate(positions):
        if simple_graph.edge_labels[i]:
            assert p is not None
            assert len(p) == 2  # (x, y)
        else:
            assert p is None


@pytest.mark.smoke
def test_place_edge_labels_no_labels():
    """place_edge_labels handles graphs with no labels."""
    g = DaguaGraph()
    g.add_node(0)
    g.add_node(1)
    g.add_edge(0, 1)
    g.compute_node_sizes()
    pos = torch.tensor([[0.0, 0.0], [0.0, 60.0]])
    curves = route_edges(pos, g.edge_index, g.node_sizes, "TB", g)
    positions = place_edge_labels(curves, pos, g.node_sizes, g.edge_labels, g)
    assert len(positions) == 1
    assert positions[0] is None


# --- Phase 6: New metrics ---

@pytest.mark.smoke
def test_edge_curvature_consistency_metric(simple_graph, simple_positions):
    """edge_curvature_consistency returns cv and mean."""
    from dagua.metrics import edge_curvature_consistency
    curves = route_edges(simple_positions, simple_graph.edge_index,
                         simple_graph.node_sizes, "TB", simple_graph)
    result = edge_curvature_consistency(curves)
    assert "edge_curvature_cv" in result
    assert "edge_curvature_mean" in result


@pytest.mark.smoke
def test_port_angular_resolution_metric(simple_graph, simple_positions):
    """port_angular_resolution returns mean angle in degrees."""
    from dagua.metrics import port_angular_resolution
    curves = route_edges(simple_positions, simple_graph.edge_index,
                         simple_graph.node_sizes, "TB", simple_graph)
    result = port_angular_resolution(curves, simple_graph.edge_index)
    assert "port_angular_res_mean_deg" in result


@pytest.mark.smoke
def test_edge_node_crossing_count_metric(simple_graph, simple_positions):
    """edge_node_crossing_count returns count and rate."""
    from dagua.metrics import edge_node_crossing_count
    curves = route_edges(simple_positions, simple_graph.edge_index,
                         simple_graph.node_sizes, "TB", simple_graph)
    result = edge_node_crossing_count(curves, simple_positions,
                                      simple_graph.node_sizes, simple_graph.edge_index)
    assert "edge_node_crossings" in result
    assert "edge_node_crossing_rate" in result


@pytest.mark.smoke
def test_label_overlap_count_metric(simple_graph, simple_positions):
    """label_overlap_count returns overlap counts."""
    from dagua.metrics import label_overlap_count
    curves = route_edges(simple_positions, simple_graph.edge_index,
                         simple_graph.node_sizes, "TB", simple_graph)
    lps = place_edge_labels(curves, simple_positions, simple_graph.node_sizes,
                            simple_graph.edge_labels, simple_graph)
    result = label_overlap_count(lps, simple_graph.edge_labels,
                                 simple_positions, simple_graph.node_sizes)
    assert "label_overlaps" in result
    assert "label_node_overlaps" in result


# --- Phase 7: Pipeline integration ---

@pytest.mark.smoke
def test_draw_full_pipeline(simple_graph, simple_positions):
    """draw() runs the full pipeline with edge optimization."""
    import dagua
    config = LayoutConfig()
    config.edge_opt_steps = 5  # fast
    fig, ax = dagua.draw(simple_graph, config=config)
    assert fig is not None


@pytest.mark.smoke
def test_draw_no_edge_opt(simple_graph, simple_positions):
    """draw() works with edge_opt_steps=0."""
    import dagua
    config = LayoutConfig()
    config.edge_opt_steps = 0
    fig, ax = dagua.draw(simple_graph, config=config)
    assert fig is not None


@pytest.mark.smoke
def test_render_accepts_precomputed(simple_graph, simple_positions):
    """render() accepts pre-computed curves and label_positions."""
    from dagua.render import render
    curves = route_edges(simple_positions, simple_graph.edge_index,
                         simple_graph.node_sizes, "TB", simple_graph)
    lps = place_edge_labels(curves, simple_positions, simple_graph.node_sizes,
                            simple_graph.edge_labels, simple_graph)
    fig, ax = render(simple_graph, simple_positions, curves=curves, label_positions=lps)
    assert fig is not None


@pytest.mark.smoke
def test_full_metrics_with_curves(simple_graph, simple_positions):
    """full() includes new edge-aware metrics when curves are provided."""
    from dagua.metrics import full
    from dagua.layout import layout
    config = LayoutConfig()
    pos = layout(simple_graph, config)
    curves = route_edges(pos, simple_graph.edge_index,
                         simple_graph.node_sizes, "TB", simple_graph)
    lps = place_edge_labels(curves, pos, simple_graph.node_sizes,
                            simple_graph.edge_labels, simple_graph)
    result = full(pos, simple_graph.edge_index, node_sizes=simple_graph.node_sizes,
                  curves=curves, label_positions=lps, edge_labels=simple_graph.edge_labels)
    assert "edge_curvature_cv" in result
    assert "port_angular_res_mean_deg" in result
    assert "edge_node_crossings" in result
    assert "label_overlaps" in result
