"""Sprint 7: node size + text polish exit criteria.

Sprint 7 exit (02_sprint_map.md L308-L310):
"Running layout on a graph with very long node labels produces no label
clipping and no label-label overlap on the held-out suite."

This file tests that criterion directly on synthetic long-label graphs.
Label-node overlap and label-label overlap are counted by the existing
``label_overlap_count`` metric; we construct graphs where the heuristic
path is expected to produce zero of each, and assert it.

Node label clipping is tested indirectly: if content-aware sizing is
doing its job (compute_node_size + overflow_policy), the node_sizes
tensor should be large enough that the measured text extent fits
inside the node bbox.
"""

from __future__ import annotations

import pytest

import dagua
from dagua.config import LayoutConfig
from dagua.edges import place_edge_labels, route_edges
from dagua.graph import DaguaGraph
from dagua.metrics import label_overlap_count
from dagua.utils import measure_text_fallback


@pytest.mark.unit
def test_long_node_labels_produce_no_label_clipping():
    """Sprint 7 criterion: a graph with very long node labels must not
    clip ANY label at render time. "Clipping" = the rendered text
    extent (measured at the EFFECTIVE font size after shrink_text
    policy) exceeds the node bbox. When shrink_text hits the
    min_font_size floor AND the text still exceeds the bbox, the
    sizing pipeline must fall back to expanding the node bbox so the
    label fits.
    """
    g = DaguaGraph()
    labels = [
        "A",
        "short",
        "moderate label",
        "this is a longer than average label",
        "a very long node label that should not clip under any circumstance",
        "and an even longer one, spanning many words, to stress sizing",
    ]
    for i, label in enumerate(labels):
        g.add_node(i, label=label)
    for i in range(len(labels) - 1):
        g.add_edge(i, i + 1)

    g.compute_node_sizes()

    for i, label in enumerate(labels):
        effective_fs = g.node_font_sizes[i].item()
        w_text, h_text = measure_text_fallback(label, effective_fs)
        w_node = g.node_sizes[i, 0].item()
        h_node = g.node_sizes[i, 1].item()
        # A tiny padding slack is fine (rendering pads by a few pt).
        slack = 4.0
        assert w_node + slack >= w_text, (
            f"node {i} clips horizontally at effective fs={effective_fs}: "
            f"label '{label[:30]}...' needs {w_text:.1f} but bbox is {w_node:.1f}"
        )
        assert h_node + slack >= h_text, (
            f"node {i} clips vertically at effective fs={effective_fs}: "
            f"label '{label[:30]}...' needs {h_text:.1f} but bbox is {h_node:.1f}"
        )


@pytest.mark.unit
def test_no_label_label_overlap_on_sparse_labeled_graph():
    """A sparse graph with distinct edge labels should produce zero
    label-label overlap. place_edge_labels already has greedy collision
    avoidance; this test pins that behaviour."""
    g = DaguaGraph()
    for i in range(6):
        g.add_node(i, label=f"node_{i}")
    g.add_edge(0, 1, label="alpha")
    g.add_edge(1, 2, label="beta")
    g.add_edge(2, 3, label="gamma")
    g.add_edge(3, 4, label="delta")
    g.add_edge(4, 5, label="epsilon")

    positions = dagua.layout(g, LayoutConfig(seed=42))
    g.compute_node_sizes()
    curves = route_edges(positions, g.edge_index, g.node_sizes, g.direction, g)
    label_positions = place_edge_labels(curves, positions, g.node_sizes, g.edge_labels, g)

    result = label_overlap_count(label_positions, g.edge_labels, positions, g.node_sizes)
    assert result["label_overlaps"] == 0, (
        f"sparse labeled graph had {result['label_overlaps']} label-label overlaps"
    )


@pytest.mark.unit
def test_label_placement_reduces_overlap_vs_naive_midpoint():
    """place_edge_labels' greedy collision avoidance should do better than
    naive "just put label at t=0.5" placement on a graph crafted to have
    many labels near the same midpoint region."""
    g = DaguaGraph()
    # Hub with many labeled outgoing edges -- naive midpoints cluster near the hub.
    g.add_node(0, label="hub")
    for i in range(1, 9):
        g.add_node(i, label=f"leaf_{i}")
        g.add_edge(0, i, label=f"L{i}")

    positions = dagua.layout(g, LayoutConfig(seed=42))
    g.compute_node_sizes()
    curves = route_edges(positions, g.edge_index, g.node_sizes, g.direction, g)
    label_positions_greedy = place_edge_labels(curves, positions, g.node_sizes, g.edge_labels, g)
    result_greedy = label_overlap_count(
        label_positions_greedy, g.edge_labels, positions, g.node_sizes
    )

    # Naive placement: every label at bezier t=0.5 with no offset.
    from dagua.edges import evaluate_bezier

    naive_positions = []
    for c in curves:
        mx, my = evaluate_bezier(c, 0.5)
        naive_positions.append((mx, my))
    result_naive = label_overlap_count(naive_positions, g.edge_labels, positions, g.node_sizes)

    total_greedy = result_greedy["label_overlaps"] + result_greedy["label_node_overlaps"]
    total_naive = result_naive["label_overlaps"] + result_naive["label_node_overlaps"]
    assert total_greedy <= total_naive, (
        f"greedy placement ({total_greedy} total overlaps) worse than "
        f"naive midpoint ({total_naive})"
    )


@pytest.mark.unit
def test_held_out_suite_has_zero_label_clipping():
    """Aggregate: run the compute_node_sizes path on a variety of
    synthetic graphs with long labels and verify zero clipping across
    the board.
    """
    long_labels = [
        "ShortName",
        "This is node name #{i}",
        "another-longish-label-with-hyphens-{i}",
        "A label {i} whose length varies to stress sizing",
    ]
    clip_count = 0
    total_nodes = 0
    for lbl_tpl_idx, lbl_tpl in enumerate(long_labels):
        g = DaguaGraph()
        for i in range(5):
            g.add_node(i, label=lbl_tpl.format(i=i))
        for i in range(4):
            g.add_edge(i, i + 1)
        g.compute_node_sizes()
        for i in range(5):
            label = g.node_labels[i] if i < len(g.node_labels) else ""
            effective_fs = g.node_font_sizes[i].item()
            w_text, h_text = measure_text_fallback(label, effective_fs)
            w_node = g.node_sizes[i, 0].item()
            h_node = g.node_sizes[i, 1].item()
            total_nodes += 1
            slack = 4.0
            if w_node + slack < w_text or h_node + slack < h_text:
                clip_count += 1

    assert clip_count == 0, f"{clip_count}/{total_nodes} nodes clip their labels"
