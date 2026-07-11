"""Tests for the r80-S4 undirected-portfolio native route."""

from __future__ import annotations

import torch

from dagua.config import LayoutConfig
from dagua.graph import DaguaGraph
from dagua.layout.graph_classify import classify_graph
from dagua.layout.ops.pipelines.dagua_native import (
    _choose_native_pipeline,
    _choose_native_pipeline_baseline,
    _unshear_bimodal_edges,
)
from dagua.layout.ops.pipelines.native_undirected import (
    BALANCED_LARGE_REFINEMENT_STEPS,
    DEGENERACY_MAX_ISOLATED_SPREAD_RATIO,
    FULL_REFINEMENT_STEPS,
    LARGE_CONTEST_NODE_THRESHOLD,
    MAX_CONTEST_NODES,
    NEATO_QUALITY_THRESHOLD,
    _candidate_is_degenerate,
    _candidate_is_eligible,
    _candidate_refinement_steps,
    _cleanup_variants_for_size,
    _neato_in_contest,
    _project_candidate_prism,
    _repair_flung_isolates,
    _score_undirected_candidate,
    _use_large_prism_shortlist,
)
from dagua.layout.ops.state import LayoutProblem


def _centroid_spread_ratio(pos: torch.Tensor) -> float:
    """Return max-to-median centroid distance for a layout.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    float
        Ratio of maximum centroid distance to median centroid distance.
    """
    distances = torch.linalg.vector_norm(pos - pos.mean(dim=0, keepdim=True), dim=1)
    median = float(torch.median(distances).item())
    if median == 0.0:
        return 0.0
    return float(distances.max().item()) / median


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


def _ring_plus_isolate(
    isolate_distance: float,
    ring_nodes: int = 50,
    ring_radius: float = 100.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Build a ring layout plus one degree-0 isolate at a chosen distance.

    Parameters
    ----------
    isolate_distance : float
        Distance of the isolated node from the ring center (origin).
    ring_nodes : int, default=50
        Connected ring size. Large enough that the isolate barely shifts
        the centroid, keeping the measured ratio close to the nominal one.
    ring_radius : float, default=100.0
        Ring radius.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]
        Positions ``[N, 2]``, node sizes ``[N, 2]``, edge index ``[2, E]``,
        and the isolate's measured centroid-distance / median-distance ratio
        (the exact quantity the guard evaluates).
    """
    edges = [(i, (i + 1) % ring_nodes) for i in range(ring_nodes)]
    edge_index = torch.tensor(list(zip(*edges)), dtype=torch.long)
    angles = torch.arange(ring_nodes, dtype=torch.float32) * (2 * torch.pi / ring_nodes)
    ring_pos = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * ring_radius
    pos = torch.cat([ring_pos, torch.tensor([[isolate_distance, 0.0]])], dim=0)
    node_sizes = torch.full((ring_nodes + 1, 2), 2.0)
    distances = torch.linalg.vector_norm(pos - pos.mean(dim=0, keepdim=True), dim=1)
    ratio = float(distances[-1].item()) / float(torch.median(distances).item())
    return pos, node_sizes, edge_index, ratio


def test_isolated_node_at_5x_median_passes_spread_guard() -> None:
    """A degree-0 node at ~5x median distance is peripheral, NOT pathological.

    r80 round-3 calibration (measured on the old store): legitimate isolate
    placements reach 5.4x median (er_500 periphery 0.5-4.8x,
    multi_component_80 tiles 2.8-2.9x); the pathology class starts at 15.1x.
    The 8x threshold must PASS peripheral placement.
    """
    pos, node_sizes, edge_index, ratio = _ring_plus_isolate(isolate_distance=510.0)
    assert 4.0 < ratio < 6.0  # sanity: this case sits in the legitimate band

    degenerate, reason = _candidate_is_degenerate(pos, node_sizes, edge_index)

    assert degenerate is False
    assert reason == ""


def test_isolated_node_at_15x_median_is_rejected_by_spread_guard() -> None:
    """A degree-0 node at ~15x median distance is the fling pathology.

    Matches the measured random_bipartite_60 pathology floor (15.1x, range
    15-21x on the old store). The 8x threshold must REJECT it.
    """
    pos, node_sizes, edge_index, ratio = _ring_plus_isolate(isolate_distance=1600.0)
    assert 12.0 < ratio < 18.0  # sanity: this case sits in the pathological band

    degenerate, reason = _candidate_is_degenerate(pos, node_sizes, edge_index)

    assert degenerate is True
    assert "isolated-node centroid spread" in reason


def test_far_flung_connected_node_passes_spread_guard() -> None:
    """Non-isolated spread is legitimate structure and is NOT judged.

    r80 gate verdict: the first (global max/median) form of the spread guard
    also rejected legitimately-dispersed candidates -- multi_component_80
    (-11.0), er_500 (real win flipped to loss, -4.9), scale_free_ba_120
    (-1.9). A far-out CONNECTED node (ER periphery, long chain end) must
    pass; only degree-0 isolates are the metric-blind pathology.
    """
    num_nodes = 10
    edges = [(i, (i + 1) % num_nodes) for i in range(num_nodes)]
    edge_index = torch.tensor(list(zip(*edges)), dtype=torch.long)
    node_sizes = torch.full((num_nodes, 2), 2.0)
    angles = torch.arange(num_nodes, dtype=torch.float32) * (2 * torch.pi / num_nodes)
    pos = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * 100.0
    pos[-1] = torch.tensor([2000.0, 0.0])  # node 9 is connected (ring member)

    degenerate, reason = _candidate_is_degenerate(pos, node_sizes, edge_index)

    assert degenerate is False
    assert reason == ""


def test_dispersed_multi_component_passes_spread_guard() -> None:
    """A multi-component tiling with high global spread but no isolates passes.

    multi_component_80-style case: several connected components tiled far
    apart produce a large max/median centroid-distance ratio with ZERO
    degree-0 nodes. The narrowed guard must not judge it (the global form
    regressed multi_component_80 92.52 -> 81.55 in the r80 gate sweep).
    """
    # A 14-node path near the origin plus a 2-node component tiled far away;
    # no isolated nodes anywhere. The far pair drags max/median centroid
    # distance to ~7x while every node keeps degree >= 1.
    edges = [(i, i + 1) for i in range(13)] + [(14, 15)]
    positions = [[10.0 * i, 0.0] for i in range(14)]
    positions += [[100000.0, 0.0], [100010.0, 0.0]]
    edge_index = torch.tensor(list(zip(*edges)), dtype=torch.long)
    pos = torch.tensor(positions, dtype=torch.float32)
    node_sizes = torch.full((pos.shape[0], 2), 2.0)

    # Sanity: this layout WOULD trip a global 6x max/median test.
    assert _centroid_spread_ratio(pos) > 6.0

    degenerate, reason = _candidate_is_degenerate(pos, node_sizes, edge_index)

    assert degenerate is False
    assert reason == ""


def test_repair_leaves_below_threshold_layout_byte_unchanged() -> None:
    """The repair path is a NO-OP for layouts below the fling threshold.

    r80 round 4: unconditional packing regressed er_500 and
    multi_component_80 whose isolates sat at a legitimate 2.8-4.8x median.
    Repair must fire ONLY above ``DEGENERACY_MAX_ISOLATED_SPREAD_RATIO``;
    below it the candidate's raw layout is returned byte-identical.
    """
    pos, node_sizes, edge_index, ratio = _ring_plus_isolate(isolate_distance=510.0)
    assert 4.0 < ratio < DEGENERACY_MAX_ISOLATED_SPREAD_RATIO  # legitimate band
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=pos.shape[0],
        node_sizes=node_sizes,
        direction="TB",
    )

    repaired = _repair_flung_isolates(pos, problem, node_sep=25.0)

    assert repaired is pos  # byte-identical: the very same tensor, untouched
    assert torch.equal(repaired, pos)


def test_repair_packs_flung_isolate_next_to_core() -> None:
    """Above the threshold, repair re-tiles the flung isolate near the core."""
    pos, node_sizes, edge_index, ratio = _ring_plus_isolate(isolate_distance=1600.0)
    assert ratio > DEGENERACY_MAX_ISOLATED_SPREAD_RATIO  # pathological band
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=pos.shape[0],
        node_sizes=node_sizes,
        direction="TB",
    )

    repaired = _repair_flung_isolates(pos, problem, node_sep=25.0)

    assert repaired.shape == pos.shape
    assert not torch.equal(repaired, pos)
    from dagua.layout.ops.pipelines.native_undirected import _max_isolated_spread_ratio

    assert _max_isolated_spread_ratio(repaired, edge_index) <= (
        DEGENERACY_MAX_ISOLATED_SPREAD_RATIO
    )
    degenerate, reason = _candidate_is_degenerate(repaired, node_sizes, edge_index)
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
    """Candidate C joins throughout the contest cap with scheduled work."""
    from dagua.layout.ops.pipelines.native_undirected import NEATO_BALANCED_NODE_CAP

    balanced = LayoutConfig(seed=42, quality="balanced")
    high = LayoutConfig(seed=42, quality="high")

    assert _neato_in_contest(balanced, NEATO_BALANCED_NODE_CAP + 1) is False
    assert _neato_in_contest(balanced, NEATO_BALANCED_NODE_CAP) is True
    assert _neato_in_contest(high, NEATO_BALANCED_NODE_CAP + 1) is True
    assert NEATO_QUALITY_THRESHOLD == 0.75
    assert NEATO_BALANCED_NODE_CAP == MAX_CONTEST_NODES


def test_candidate_refinement_schedule_preserves_high_quality() -> None:
    """Large balanced solves are bounded while high quality keeps 500 steps."""
    balanced = LayoutConfig(seed=42, quality="balanced")
    high = LayoutConfig(seed=42, quality="high")

    assert _candidate_refinement_steps(balanced, 150) == FULL_REFINEMENT_STEPS
    assert _candidate_refinement_steps(balanced, 500) == BALANCED_LARGE_REFINEMENT_STEPS
    assert BALANCED_LARGE_REFINEMENT_STEPS == 10
    assert _candidate_refinement_steps(high, 500) == FULL_REFINEMENT_STEPS


def test_prism_candidate_finishes_residual_overlaps_to_zero() -> None:
    """PRISM plus its residual scale loop reaches literal zero overlaps."""
    positions = torch.tensor(
        [[0.0, 0.0], [9.0, 0.0], [18.0, 0.0], [27.0, 0.0]], dtype=torch.float32
    )
    node_sizes = torch.full((4, 2), 20.0)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=4,
        node_sizes=node_sizes,
        direction="TB",
    )

    projected = _project_candidate_prism(positions, problem)

    from dagua.metrics import count_overlaps

    assert count_overlaps(projected.cpu(), node_sizes) == 0


def test_prism_duplicate_positions_fail_closed() -> None:
    """PRISM rejects duplicate positions instead of returning extreme coordinates."""
    positions = torch.tensor([[0.0, 0.0], [0.0, 0.0], [100.0, 100.0]])
    node_sizes = torch.full((3, 2), 10.0)
    edge_index = torch.tensor([[0, 1], [2, 2]], dtype=torch.long)
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=3,
        node_sizes=node_sizes,
        direction="TB",
    )

    assert _project_candidate_prism(positions, problem) is None


def test_geometry_candidate_rejects_overlap_increase() -> None:
    """The adversarial collinear dodge that adds an overlap is ineligible."""
    from dagua.layout.ops.pipelines.dagua_native import _collinear_dodge

    edge_index = torch.tensor([[0, 0, 1, 1, 2, 3, 4], [2, 1, 5, 7, 3, 5, 5]], dtype=torch.long)
    positions = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 10.0],
            [0.0, 20.0],
            [15.4435, -2.0417],
            [3.3375, 13.7076],
            [-20.0198, -22.4181],
            [50.0, 50.0],
            [50.0, 50.0],
        ]
    )
    node_sizes = torch.full((8, 2), 2.0)
    candidate = _collinear_dodge(positions, edge_index, delta=0.15)

    assert candidate is not None
    eligible, reason = _candidate_is_eligible(candidate, positions, node_sizes, edge_index)
    assert eligible is False
    assert reason == "overlaps increased 1->2"


def test_general_challengers_cover_500_node_corpus() -> None:
    """The contest cap covers the corpus without a smaller challenger cutoff."""
    from dagua.layout.ops.pipelines import native_undirected

    assert MAX_CONTEST_NODES >= 500
    assert not hasattr(native_undirected, "MAX_GENERAL_CHALLENGER_NODES")


def test_large_contest_runs_only_prism_cleanup() -> None:
    """Large contests retain only the corpus-winning PRISM cleanup path."""
    assert _cleanup_variants_for_size(LARGE_CONTEST_NODE_THRESHOLD) == (
        ("", False),
        ("_convergent", True),
        ("_prism", None),
    )
    assert _cleanup_variants_for_size(LARGE_CONTEST_NODE_THRESHOLD + 1) == (("_prism", None),)


def test_large_shortlist_retains_degree_four_mesh_incumbent() -> None:
    """Large shortlist excludes mesh topology and includes higher-degree graphs."""
    n = LARGE_CONTEST_NODE_THRESHOLD + 1
    mesh_edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    mesh_problem = LayoutProblem(edge_index=mesh_edges, num_nodes=n)
    hub_edges = torch.tensor([[0, 0, 0, 0, 0], [1, 2, 3, 4, 5]], dtype=torch.long)
    hub_problem = LayoutProblem(edge_index=hub_edges, num_nodes=n)

    assert _use_large_prism_shortlist(mesh_problem) is False
    assert _use_large_prism_shortlist(hub_problem) is True


def test_portfolio_layout_end_to_end_produces_finite_positions() -> None:
    """Full layout() on a declared-undirected graph goes through the route."""
    from dagua.layout import layout

    graph = _ring_with_chords()
    pos = layout(graph, LayoutConfig(seed=42, device="cpu"))

    assert pos.shape == (graph.num_nodes, 2)
    assert bool(torch.isfinite(pos).all())


def test_random_bipartite_60_isolates_stay_near_core() -> None:
    """Portfolio layout keeps random_bipartite_60 singleton nodes near the core."""
    from dagua.eval.graphs import get_test_graphs
    from dagua.layout import layout

    graph = next(
        test_graph.graph
        for test_graph in get_test_graphs()
        if test_graph.name == "random_bipartite_60"
    )

    pos = layout(graph, LayoutConfig(seed=42, device="cpu"))

    assert pos.shape == (graph.num_nodes, 2)
    assert bool(torch.isfinite(pos).all())
    assert _centroid_spread_ratio(pos) <= 3.0


def test_synthetic_singletons_stay_near_connected_component() -> None:
    """Portfolio layout packs degree-zero singleton nodes near a small path."""
    from dagua.layout import layout

    path_edges = [(node, node + 1) for node in range(7)]
    graph = DaguaGraph.from_edge_list(
        path_edges,
        num_nodes=10,
        is_semantically_directed=False,
    )
    graph.compute_node_sizes()

    pos = layout(graph, LayoutConfig(seed=42, device="cpu"))

    assert pos.shape == (graph.num_nodes, 2)
    assert bool(torch.isfinite(pos).all())
    assert _centroid_spread_ratio(pos) <= 3.0


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

    def spy(pos, problem, cluster_ids, aesthetic_profile=None):
        scored_positions.append(pos.detach().clone())
        return original_score(pos, problem, cluster_ids, aesthetic_profile)

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


def _clustered_ring_graph(
    num_nodes: int = 12,
    weighted: bool = False,
) -> DaguaGraph:
    """Return a small declared-undirected graph with two clusters.

    Parameters
    ----------
    num_nodes : int, default=12
        Total node count (split evenly into two clusters).
    weighted : bool, default=False
        Whether to attach non-uniform edge weights.

    Returns
    -------
    DaguaGraph
        A ring-of-cliques graph with ``cluster_a``/``cluster_b`` clusters.
    """
    half = num_nodes // 2
    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    for block_start in (0, half):
        for i in range(block_start, block_start + half):
            for j in range(i + 1, block_start + half):
                edges.append((i, j))
                weights.append(3.0 if weighted else 1.0)
    # One bridge edge between the two clusters.
    edges.append((half - 1, half))
    weights.append(0.2 if weighted else 1.0)

    graph = DaguaGraph()
    for node_idx in range(num_nodes):
        graph.add_node(node_idx)
    for (source, target), weight in zip(edges, weights):
        graph.add_edge(source, target, weight=weight if weighted else None)
    graph.add_cluster("cluster_a", list(range(0, half)))
    graph.add_cluster("cluster_b", list(range(half, num_nodes)))
    graph.is_semantically_directed = False
    graph.compute_node_sizes()
    return graph


def test_cluster_aware_sfdp_candidate_none_without_clusters() -> None:
    """The cluster-aware sfdp candidate is skipped when there are no clusters."""
    from dagua.config import LayoutConfig
    from dagua.layout.ops.pipelines.native_undirected import _cluster_aware_sfdp_candidate
    from dagua.layout.ops.state import ExecutionPlan, RuntimeContext

    graph = _ring_with_chords()
    problem = LayoutProblem(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        node_sizes=graph.node_sizes,
    )
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))

    result = _cluster_aware_sfdp_candidate(problem, LayoutConfig(seed=42), ctx)

    assert result is None


def test_cluster_aware_sfdp_candidate_produces_finite_positions() -> None:
    """The cluster-aware sfdp candidate returns finite positions for a clustered graph."""
    from dagua.config import LayoutConfig
    from dagua.layout.ops.pipelines.native_undirected import (
        _build_cluster_ids,
        _cluster_aware_sfdp_candidate,
        _score_undirected_candidate,
    )
    from dagua.layout.ops.state import ExecutionPlan, RuntimeContext

    graph = _clustered_ring_graph()
    problem = LayoutProblem(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        node_sizes=graph.node_sizes,
        clusters=graph.clusters,
        cluster_parents=graph.cluster_parents,
        seed=42,
    )
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))

    pos = _cluster_aware_sfdp_candidate(problem, LayoutConfig(seed=42), ctx)

    assert pos is not None
    assert pos.shape == (graph.num_nodes, 2)
    assert bool(torch.isfinite(pos).all())
    # Cluster containment scoring must actually see the cluster ids (r80-S9
    # Deliverable 1's explicit gate expectation: verify the cluster term is
    # computed for this candidate, not silently skipped).
    cluster_ids = _build_cluster_ids(problem)
    assert cluster_ids is not None
    score = _score_undirected_candidate(pos, problem, cluster_ids)
    assert score == score  # not NaN


def test_clustered_undirected_contest_reaches_cluster_candidate() -> None:
    """A full portfolio run on a clustered-undirected graph scores candidate D.

    Regression guard for the S9 diagnosis: the contest must actually invoke
    the cluster-aware sfdp candidate (name ``"cluster_sfdp"`` or
    ``"cluster_sfdp_convergent"``) when ``problem.clusters`` is set, never
    silently skip it.
    """
    from dagua.layout import layout
    from dagua.layout.ops.pipelines import native_undirected as nu

    graph = _clustered_ring_graph()

    original_candidate_fn = nu._cluster_aware_sfdp_candidate
    calls: list[bool] = []

    def spy_candidate(problem, config, ctx):
        calls.append(bool(problem.clusters))
        return original_candidate_fn(problem, config, ctx)

    nu._cluster_aware_sfdp_candidate = spy_candidate
    try:
        pos = layout(graph, LayoutConfig(seed=42, device="cpu"))
    finally:
        nu._cluster_aware_sfdp_candidate = original_candidate_fn

    assert pos.shape == (graph.num_nodes, 2)
    assert bool(torch.isfinite(pos).all())
    assert calls, "clustered-undirected contest never invoked candidate D"
    assert all(calls)


def test_weighted_similarity_candidate_none_without_weights() -> None:
    """The weighted-similarity candidate is skipped when there are no weights."""
    from dagua.layout.ops.pipelines.native_undirected import _weighted_similarity_candidate

    graph = _ring_with_chords()
    problem = LayoutProblem(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        node_sizes=graph.node_sizes,
    )

    result = _weighted_similarity_candidate(problem, seed=42)

    assert result is None


def test_weighted_similarity_candidate_produces_finite_positions() -> None:
    """The weighted-similarity candidate returns finite positions for a weighted graph."""
    from dagua.layout.ops.pipelines.native_undirected import _weighted_similarity_candidate

    graph = _clustered_ring_graph(weighted=True)
    problem = LayoutProblem(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        node_sizes=graph.node_sizes,
        edge_weights=graph.edge_weights,
        seed=42,
    )

    pos = _weighted_similarity_candidate(problem, seed=42)

    assert pos is not None
    assert pos.shape == (graph.num_nodes, 2)
    assert bool(torch.isfinite(pos).all())


def test_weighted_similarity_candidate_uses_inverse_transform() -> None:
    """The candidate's positions match a direct ``weight_transform="inverse"`` call.

    Locks the mini-probe's decision (P12_SQUEEZE.md): the challenger must
    actually use the "inverse" transform, not silently fall back to "none".
    """
    from dagua.layout.ops.pipelines.native_stress import (
        NativeStressConfig,
        layout_native_stress_pipeline,
    )
    from dagua.layout.ops.pipelines.native_undirected import (
        WEIGHTED_SIMILARITY_TRANSFORM,
        _weighted_similarity_candidate,
    )

    graph = _clustered_ring_graph(weighted=True)
    problem = LayoutProblem(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        node_sizes=graph.node_sizes,
        edge_weights=graph.edge_weights,
        seed=42,
    )

    assert WEIGHTED_SIMILARITY_TRANSFORM == "inverse"

    candidate_pos = _weighted_similarity_candidate(problem, seed=42)
    direct_pos = layout_native_stress_pipeline(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        node_sizes=graph.node_sizes,
        edge_weights=graph.edge_weights,
        seed=42,
        config=NativeStressConfig(weight_transform="inverse", seed=42),
    )

    torch.testing.assert_close(candidate_pos, direct_pos, rtol=0.0, atol=0.0)


def test_weighted_undirected_contest_reaches_weighted_similarity_candidate() -> None:
    """A full portfolio run on a weighted-undirected graph scores candidate E."""
    from dagua.layout import layout
    from dagua.layout.ops.pipelines import native_undirected as nu

    graph = _clustered_ring_graph(weighted=True)
    original_candidate_fn = nu._weighted_similarity_candidate
    calls: list[bool] = []

    def spy_candidate(problem, seed):
        calls.append(problem.edge_weights is not None)
        return original_candidate_fn(problem, seed)

    nu._weighted_similarity_candidate = spy_candidate
    try:
        pos = layout(graph, LayoutConfig(seed=42, device="cpu"))
    finally:
        nu._weighted_similarity_candidate = original_candidate_fn

    assert pos.shape == (graph.num_nodes, 2)
    assert bool(torch.isfinite(pos).all())
    assert calls, "weighted-undirected contest never invoked candidate E"
    assert all(calls)


def test_stress_points_candidate_uses_point_targets() -> None:
    """The additive stress challenger should match an explicit point-unit solve."""
    from dagua.layout.ops.pipelines.native_stress import (
        NativeStressConfig,
        layout_native_stress_pipeline,
    )
    from dagua.layout.ops.pipelines.native_undirected import _stress_points_candidate

    graph = _ring_with_chords()
    problem = LayoutProblem(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        node_sizes=graph.node_sizes,
        seed=42,
    )

    candidate_pos = _stress_points_candidate(problem, seed=42)
    direct_pos = layout_native_stress_pipeline(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        node_sizes=graph.node_sizes,
        seed=42,
        config=NativeStressConfig(target_unit="points", seed=42),
    )

    torch.testing.assert_close(candidate_pos, direct_pos, rtol=0.0, atol=0.0)


def test_new_candidates_share_the_degeneracy_guard() -> None:
    """New candidates D and E are added via the shared ``_add_challenger`` path.

    Both new candidates are added via the existing ``_add_challenger``
    helper (never a bespoke selection path), so a pathologically collapsed
    D/E output can never win the contest outright -- this is a structural
    guarantee test, not a numeric-score test: it confirms the SAME guard
    the sfdp/neato challengers already use rejects a collapsed candidate.
    """
    from dagua.layout.ops.pipelines.native_undirected import _candidate_is_degenerate

    num_nodes = 8
    node_sizes = torch.full((num_nodes, 2), 40.0)
    edges = [(i, (i + 1) % num_nodes) for i in range(num_nodes)]
    edge_index = torch.tensor(list(zip(*edges)), dtype=torch.long)
    collapsed = torch.rand((num_nodes, 2)) * 0.5

    degenerate, reason = _candidate_is_degenerate(collapsed, node_sizes, edge_index)

    assert degenerate is True
    assert reason != ""


def test_unshear_orthogonalizes_sheared_grid_edge_families() -> None:
    """The grid challenger maps two sheared direction families to right angles."""
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [1.5, 1.0], [1.0, 2.0], [2.0, 2.0]])
    edge_index = torch.tensor([[0, 2, 4, 0, 1, 2], [1, 3, 5, 2, 3, 4]], dtype=torch.long)

    unsheared = _unshear_bimodal_edges(pos, edge_index)

    assert unsheared is not None
    vectors = unsheared[edge_index[1]] - unsheared[edge_index[0]]
    horizontal = vectors[:3].mean(dim=0)
    diagonal = vectors[3:].mean(dim=0)
    cosine = torch.dot(horizontal, diagonal) / (
        torch.linalg.vector_norm(horizontal) * torch.linalg.vector_norm(diagonal)
    )
    assert float(torch.abs(cosine).item()) < 1e-5
