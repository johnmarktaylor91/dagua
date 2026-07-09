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


def _dense_clique_problem() -> tuple[torch.Tensor, torch.Tensor, LayoutProblem]:
    """Build a 30-node dense-overlap-clique candidate and its problem.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, LayoutProblem]
        Near-coincident positions, 60x20 label-size boxes, and a ring
        problem carrying those sizes.
    """
    num_nodes = 30
    generator = torch.Generator().manual_seed(0)
    pos = torch.randn(num_nodes, 2, generator=generator) * 0.5
    node_sizes = torch.full((num_nodes, 2), 60.0)
    node_sizes[:, 1] = 20.0
    edges = [(i, (i + 1) % num_nodes) for i in range(num_nodes)]
    problem = LayoutProblem(
        edge_index=torch.tensor(list(zip(*edges)), dtype=torch.long),
        num_nodes=num_nodes,
        node_sizes=node_sizes,
    )
    return pos, node_sizes, problem


def test_convergent_cleanup_variant_resolves_dense_overlaps() -> None:
    """The convergent cleanup variant fully resolves a dense overlap clique.

    A near-coincident 30-node candidate with real label boxes is a dense
    overlap clique the legacy projector provably stalls on (P3B2
    forensics). ``convergent=True`` must resolve it to zero -- that is the
    variant's reason to exist in the contest. Trajectory risk is
    referee-protected: the contest scores each variant against the
    incumbent with the honest composite.
    """
    from dagua.layout.ops.pipelines.native_undirected import _project_candidate
    from dagua.metrics import count_overlaps

    pos, node_sizes, problem = _dense_clique_problem()
    assert count_overlaps(pos, node_sizes) > 400  # sanity: dense clique

    projected = _project_candidate(pos, problem, convergent=True)

    assert count_overlaps(projected, node_sizes) == 0
    # Input candidate tensor must not be mutated (contest reuses it for the
    # other cleanup variant).
    assert count_overlaps(pos, node_sizes) > 400


def test_legacy_cleanup_variant_matches_trunk_call() -> None:
    """The default cleanup variant reproduces the S4 trunk projection call.

    The r80-S2b bisect proved the trunk's flagship portfolio wins
    (petersen_10 79.0 etc.) are legacy-cleaned candidates; replacing that
    cleanup silently removed them from the pool. The default
    ``_project_candidate`` call must therefore stay bit-identical to the
    trunk's ``project_overlaps(pos, node_sizes)``.
    """
    from dagua.layout.ops.pipelines.native_undirected import _project_candidate
    from dagua.layout.projection import project_overlaps

    pos, node_sizes, problem = _dense_clique_problem()

    projected = _project_candidate(pos, problem)

    trunk_call = pos.detach().clone().to(dtype=torch.float32)
    project_overlaps(trunk_call, node_sizes.to(dtype=torch.float32))
    torch.testing.assert_close(projected, trunk_call, rtol=0.0, atol=0.0)


def test_contest_registers_both_cleanup_variants() -> None:
    """Each challenger contributes BOTH cleanup variants to the contest.

    Never replace a candidate -- add both and let the referee choose
    (r80-S2b). Verified by spying on the contest's scoring calls during a
    real portfolio run on a declared-undirected graph.
    """
    from dagua.layout.ops.pipelines import native_undirected as nu

    graph = _ring_with_chords()
    scored_positions: list[torch.Tensor] = []
    original_score = nu._score_undirected_candidate

    def spy(pos, problem, cluster_ids):
        scored_positions.append(pos.detach().clone())
        return original_score(pos, problem, cluster_ids)

    original_project = nu._project_candidate
    project_calls: list[bool] = []

    def spy_project(pos, problem, convergent=False):
        project_calls.append(bool(convergent))
        return original_project(pos, problem, convergent=convergent)

    nu._score_undirected_candidate = spy
    nu._project_candidate = spy_project
    try:
        from dagua.layout import layout

        layout(graph, LayoutConfig(seed=42, device="cpu"))
    finally:
        nu._score_undirected_candidate = original_score
        nu._project_candidate = original_project

    # Every challenger cleanup ran both variants (False and True in pairs).
    assert project_calls, "portfolio contest never cleaned a challenger"
    assert project_calls.count(False) == project_calls.count(True)
    # And the contest scored more candidates than incumbent + one-per-
    # challenger (i.e., variants were ADDED, not substituted).
    n_challengers = project_calls.count(False)
    assert len(scored_positions) > 1 + n_challengers
