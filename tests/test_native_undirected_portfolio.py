"""Tests for the r80-S4 undirected-portfolio native route."""

from __future__ import annotations

import torch

from dagua.config import LayoutConfig
from dagua.graph import DaguaGraph
from dagua.layout.graph_classify import classify_graph
from dagua.layout.ops.pipelines.dagua_native import (
    _choose_native_pipeline,
    _choose_native_pipeline_baseline,
)
from dagua.layout.ops.pipelines.native_undirected import (
    NEATO_QUALITY_THRESHOLD,
    _candidate_is_degenerate,
    _neato_in_contest,
    _score_undirected_candidate,
)
from dagua.layout.ops.state import LayoutProblem


def _ring_with_chords(num_nodes: int = 10) -> DaguaGraph:
    """Return a small cyclic undirected-declared graph."""
    edges = [(i, (i + 1) % num_nodes) for i in range(num_nodes)]
    edges += [(i, (i + num_nodes // 2) % num_nodes) for i in range(num_nodes // 2)]
    graph = DaguaGraph.from_edge_list(
        edges,
        num_nodes=num_nodes,
        is_semantically_directed=False,
    )
    graph.compute_node_sizes()
    return graph


def test_declared_undirected_routes_to_portfolio() -> None:
    """A declared-undirected graph selects the portfolio route."""
    graph = _ring_with_chords()
    structure = classify_graph(graph.edge_index, graph.num_nodes, graph=graph)
    config = LayoutConfig(seed=42, device="cpu")

    assert structure.is_semantically_directed is False
    assert _choose_native_pipeline(structure=structure, config=config) == "undirected_portfolio"


def test_directed_graph_does_not_route_to_portfolio() -> None:
    """A directed DAG keeps its baseline route."""
    edges = [(i, i + 1) for i in range(9)] + [(0, 5), (2, 7)]
    graph = DaguaGraph.from_edge_list(edges, num_nodes=10)
    graph.compute_node_sizes()
    structure = classify_graph(graph.edge_index, graph.num_nodes, graph=graph)
    config = LayoutConfig(seed=42, device="cpu")

    selected = _choose_native_pipeline(structure=structure, config=config)

    assert structure.is_semantically_directed is True
    assert selected != "undirected_portfolio"


def test_baseline_helper_matches_prior_routing_for_directed() -> None:
    """The factored baseline helper returns the same route the full chooser picks."""
    edges = [(i, i + 1) for i in range(9)] + [(0, 5), (2, 7)]
    graph = DaguaGraph.from_edge_list(edges, num_nodes=10)
    structure = classify_graph(graph.edge_index, graph.num_nodes, graph=graph)
    config = LayoutConfig(seed=42, device="cpu")

    assert _choose_native_pipeline(
        structure=structure, config=config
    ) == _choose_native_pipeline_baseline(structure=structure, config=config)


def test_forced_pipeline_beats_portfolio_branch() -> None:
    """force_pipeline overrides the undirected-portfolio branch."""
    graph = _ring_with_chords()
    structure = classify_graph(graph.edge_index, graph.num_nodes, graph=graph)
    config = LayoutConfig(seed=42, device="cpu")
    config.force_pipeline = "stress"

    assert _choose_native_pipeline(structure=structure, config=config) == "stress"


def test_degenerate_collapsed_candidate_is_rejected() -> None:
    """A fully-collapsed candidate trips the degeneracy guard."""
    num_nodes = 8
    edges = [(i, (i + 1) % num_nodes) for i in range(num_nodes)]
    edge_index = torch.tensor(list(zip(*edges)), dtype=torch.long)
    node_sizes = torch.full((num_nodes, 2), 40.0)
    collapsed_pos = torch.zeros((num_nodes, 2)) + torch.rand((num_nodes, 2)) * 0.5

    degenerate, reason = _candidate_is_degenerate(collapsed_pos, node_sizes, edge_index)

    assert degenerate is True
    assert reason != ""


def test_healthy_spread_candidate_passes_guard() -> None:
    """A well-spread candidate passes the degeneracy guard."""
    num_nodes = 8
    edges = [(i, (i + 1) % num_nodes) for i in range(num_nodes)]
    edge_index = torch.tensor(list(zip(*edges)), dtype=torch.long)
    node_sizes = torch.full((num_nodes, 2), 40.0)
    angles = torch.arange(num_nodes, dtype=torch.float32) * (2 * torch.pi / num_nodes)
    spread_pos = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * 200.0

    degenerate, reason = _candidate_is_degenerate(spread_pos, node_sizes, edge_index)

    assert degenerate is False
    assert reason == ""


def test_collapsed_challenger_loses_to_sane_incumbent() -> None:
    """Contest semantics: a collapsed challenger is rejected BEFORE scoring.

    Simulates the guard + selection flow directly: the collapsed candidate is
    filtered by the degeneracy guard, so the sane incumbent wins regardless
    of what composite score the collapsed layout would have received.
    """
    num_nodes = 8
    edges = [(i, (i + 1) % num_nodes) for i in range(num_nodes)]
    edge_index = torch.tensor(list(zip(*edges)), dtype=torch.long)
    node_sizes = torch.full((num_nodes, 2), 40.0)
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        direction="TB",
    )

    angles = torch.arange(num_nodes, dtype=torch.float32) * (2 * torch.pi / num_nodes)
    incumbent_pos = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * 200.0
    collapsed_pos = torch.rand((num_nodes, 2)) * 0.5

    # Mirror layout_native_undirected_portfolio's challenger flow.
    scores = {"incumbent": _score_undirected_candidate(incumbent_pos, problem, None)}
    positions = {"incumbent": incumbent_pos}
    degenerate, _ = _candidate_is_degenerate(collapsed_pos, node_sizes, edge_index)
    if not degenerate:
        positions["challenger"] = collapsed_pos
        scores["challenger"] = _score_undirected_candidate(collapsed_pos, problem, None)

    best = "incumbent"
    for name, score in scores.items():
        if name != "incumbent" and score > scores[best]:
            best = name

    assert degenerate is True
    assert best == "incumbent"
    assert torch.equal(positions[best], incumbent_pos)


def test_neato_contest_quality_gate() -> None:
    """Candidate C joins at quality >= high, or at balanced for small graphs.

    The balanced-quality node cap is probe-derived: every balanced-quality
    contest win for neato in P8_PORTFOLIO_PROBE.md sits at n <= 80 where its
    SMACOF loop converges in seconds; above the cap it is slow and never won.
    """
    from dagua.layout.ops.pipelines.native_undirected import NEATO_BALANCED_NODE_CAP

    balanced = LayoutConfig(seed=42, quality="balanced")
    high = LayoutConfig(seed=42, quality="high")

    assert _neato_in_contest(balanced, NEATO_BALANCED_NODE_CAP + 1) is False
    assert _neato_in_contest(balanced, NEATO_BALANCED_NODE_CAP) is True
    assert _neato_in_contest(high, NEATO_BALANCED_NODE_CAP + 1) is True
    assert NEATO_QUALITY_THRESHOLD == 0.75
    assert NEATO_BALANCED_NODE_CAP == 80


def test_portfolio_layout_end_to_end_produces_finite_positions() -> None:
    """Full layout() on a declared-undirected graph goes through the route."""
    from dagua.layout import layout

    graph = _ring_with_chords()
    pos = layout(graph, LayoutConfig(seed=42, device="cpu"))

    assert pos.shape == (graph.num_nodes, 2)
    assert bool(torch.isfinite(pos).all())
