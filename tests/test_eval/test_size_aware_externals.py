"""Tests for the r80-P6 size-aware external adapter policy.

External layout engines (Graphviz dot/sfdp/neato, ELK layered, dagre) were
historically laid out size-blind while dagua's own composite score always
scores overlaps against the real label-measured node boxes. These tests
cover the fix: size-capable adapters now emit real per-node sizes by
default, with ``--size-blind-externals`` (``set_size_aware_externals``)
restoring the old placeholder behavior for store-compatibility experiments.
"""

from __future__ import annotations

import torch

from dagua.eval.competitors.dagre_competitor import (
    _DEFAULT_NODE_HEIGHT as DAGRE_DEFAULT_HEIGHT,
)
from dagua.eval.competitors.dagre_competitor import (
    _DEFAULT_NODE_WIDTH as DAGRE_DEFAULT_WIDTH,
)
from dagua.eval.competitors.dagre_competitor import _build_dagre_input
from dagua.eval.competitors.dagre_competitor import _node_wh as dagre_node_wh
from dagua.eval.competitors.elk_competitor import _build_elk_children, _cluster_children
from dagua.eval.competitors.elk_competitor import _node_wh as elk_node_wh
from dagua.eval.competitors.graphviz_competitor import _graph_to_dot
from dagua.eval.size_policy import set_size_aware_externals, size_aware_externals
from dagua.graph import DaguaGraph
from dagua.graphviz_utils import to_dot


def _sized_graph() -> DaguaGraph:
    """Build a two-node graph with distinct, non-default node sizes.

    Returns
    -------
    DaguaGraph
        Graph with ``node_sizes`` populated to a shape unlikely to collide
        with any adapter's hardcoded placeholder default (120x40).
    """
    graph = DaguaGraph()
    graph.add_node("a")
    graph.add_node("b")
    graph.add_edge("a", "b")
    graph.node_sizes = torch.tensor([[200.0, 60.0], [90.0, 30.0]])
    return graph


def test_size_aware_externals_defaults_true_and_round_trips() -> None:
    """The toggle defaults to size-aware and round-trips cleanly.

    Returns
    -------
    None
    """
    assert size_aware_externals() is True
    try:
        set_size_aware_externals(False)
        assert size_aware_externals() is False
        set_size_aware_externals(True)
        assert size_aware_externals() is True
    finally:
        set_size_aware_externals(True)


def test_graphviz_dot_statement_emits_real_size_when_aware() -> None:
    """``_graph_to_dot`` emits width/height/fixedsize from real node sizes.

    Returns
    -------
    None
    """
    graph = _sized_graph()
    try:
        set_size_aware_externals(True)
        dot_src = _graph_to_dot(graph)
    finally:
        set_size_aware_externals(True)

    assert "fixedsize=true" in dot_src
    # 200pt / 72 = 2.7778in
    assert "width=2.7778" in dot_src
    assert "height=0.8333" in dot_src


def test_graphviz_dot_statement_size_blind_omits_size_attrs() -> None:
    """``--size-blind-externals`` restores the old size-blind DOT output.

    Returns
    -------
    None
    """
    graph = _sized_graph()
    try:
        set_size_aware_externals(False)
        dot_src = _graph_to_dot(graph)
    finally:
        set_size_aware_externals(True)

    assert "fixedsize" not in dot_src
    assert "width=" not in dot_src
    assert "height=" not in dot_src


def test_to_dot_default_behavior_is_unchanged_without_node_sizes() -> None:
    """``to_dot`` without ``node_sizes`` never emits size attributes.

    This guards the many other callers of ``to_dot`` (benchmarks/, scripts/,
    render-comparison helpers) that never pass ``node_sizes`` and must keep
    their exact prior DOT output.

    Returns
    -------
    None
    """
    graph = _sized_graph()
    dot_src = to_dot(graph)

    assert "fixedsize" not in dot_src
    assert "width=" not in dot_src


def test_to_dot_with_node_sizes_emits_fixedsize() -> None:
    """``to_dot(graph, node_sizes=...)`` opts into size-aware DOT output.

    Returns
    -------
    None
    """
    graph = _sized_graph()
    dot_src = to_dot(graph, node_sizes=graph.node_sizes)

    assert "fixedsize=true" in dot_src
    assert "width=2.7778" in dot_src


def test_elk_node_wh_uses_real_sizes_when_aware() -> None:
    """ELK's per-node width/height helper returns real sizes when aware.

    Returns
    -------
    None
    """
    graph = _sized_graph()
    try:
        set_size_aware_externals(True)
        w, h = elk_node_wh(graph, 0)
    finally:
        set_size_aware_externals(True)

    assert w == 200.0
    assert h == 60.0


def test_elk_node_wh_falls_back_to_placeholder_when_blind() -> None:
    """ELK's helper falls back to the historical 120x40 when size-blind.

    Returns
    -------
    None
    """
    graph = _sized_graph()
    try:
        set_size_aware_externals(False)
        w, h = elk_node_wh(graph, 0)
    finally:
        set_size_aware_externals(True)

    assert (w, h) == (120.0, 40.0)


def test_elk_children_carry_real_sizes() -> None:
    """The ELK JSON request's node children carry real per-node sizes.

    Returns
    -------
    None
    """
    graph = _sized_graph()
    try:
        set_size_aware_externals(True)
        children = _build_elk_children(graph, None, _cluster_children(graph), set())
    finally:
        set_size_aware_externals(True)

    by_id = {child["id"]: child for child in children}
    assert by_id["0"]["width"] == 200.0
    assert by_id["0"]["height"] == 60.0
    assert by_id["1"]["width"] == 90.0
    assert by_id["1"]["height"] == 30.0


def test_dagre_node_wh_uses_real_sizes_when_aware() -> None:
    """dagre's per-node width/height helper returns real sizes when aware.

    Returns
    -------
    None
    """
    graph = _sized_graph()
    try:
        set_size_aware_externals(True)
        w, h = dagre_node_wh(graph, 1)
    finally:
        set_size_aware_externals(True)

    assert w == 90.0
    assert h == 30.0


def test_dagre_node_wh_falls_back_to_placeholder_when_blind() -> None:
    """dagre's helper falls back to the historical 120x40 when size-blind.

    Returns
    -------
    None
    """
    graph = _sized_graph()
    try:
        set_size_aware_externals(False)
        w, h = dagre_node_wh(graph, 1)
        assert (w, h) == (DAGRE_DEFAULT_WIDTH, DAGRE_DEFAULT_HEIGHT)
    finally:
        set_size_aware_externals(True)


def test_dagre_input_nodes_carry_real_sizes() -> None:
    """The dagre JSON payload's nodes list carries real per-node sizes.

    Returns
    -------
    None
    """
    graph = _sized_graph()
    try:
        set_size_aware_externals(True)
        payload = _build_dagre_input(graph)
    finally:
        set_size_aware_externals(True)

    by_id = {node["id"]: node for node in payload["nodes"]}
    assert by_id["0"]["width"] == 200.0
    assert by_id["0"]["height"] == 60.0
    assert by_id["1"]["width"] == 90.0
    assert by_id["1"]["height"] == 30.0


def test_dagre_input_nodes_use_placeholder_when_size_blind() -> None:
    """The dagre JSON payload falls back to 120x40 when size-blind.

    Returns
    -------
    None
    """
    graph = _sized_graph()
    try:
        set_size_aware_externals(False)
        payload = _build_dagre_input(graph)
    finally:
        set_size_aware_externals(True)

    by_id = {node["id"]: node for node in payload["nodes"]}
    assert by_id["0"]["width"] == DAGRE_DEFAULT_WIDTH
    assert by_id["0"]["height"] == DAGRE_DEFAULT_HEIGHT
