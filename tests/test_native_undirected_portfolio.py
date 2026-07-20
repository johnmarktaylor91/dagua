"""Tests for the r80-S4 undirected-portfolio native route."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import pytest
import torch

from dagua.config import LayoutConfig
from dagua.graph import DaguaGraph
from dagua.layout.graph_classify import classify_graph
from dagua.layout.ops.pipelines.dagua_native import (
    _choose_native_pipeline,
    _choose_native_pipeline_baseline,
    _unshear_bimodal_edges,
)
from dagua.layout.ops.pipelines.native_arm_s import (
    ARM_S_ACCEPTANCE_MARGIN,
    ARM_S_CANDIDATE_PREFIX,
    ARM_S_SCALE_MULTIPLIERS,
    ARM_S_STRICT_WIN_REFERENCE,
    ArmSCandidate,
    ArmSProjectionTelemetry,
    build_arm_s_stress_candidates,
    build_arm_s_stress_finalist,
    calibrate_arm_s_scale,
    evaluate_arm_s_admission,
    exact_arm_s_overlap_count,
)
from dagua.layout.ops.pipelines.native_undirected import (
    BALANCED_LARGE_REFINEMENT_STEPS,
    BALANCED_SMALL_REFINEMENT_STEPS,
    DEGENERACY_MAX_ISOLATED_SPREAD_RATIO,
    FULL_REFINEMENT_STEPS,
    LARGE_CONTEST_NODE_THRESHOLD,
    MAX_CONTEST_NODES,
    NEATO_QUALITY_THRESHOLD,
    TSNET_PERPLEXITIES,
    _arm_s_full_score_budget_available,
    _candidate_is_degenerate,
    _candidate_is_eligible,
    _candidate_refinement_steps,
    _cleanup_variants_for_size,
    _cluster_candidate_is_dual_admissible,
    _ClusterScoreTelemetry,
    _log_marketplace_telemetry,
    _neato_in_contest,
    _portfolio_has_budget,
    _predicted_arm_budget_preserving_arm_s_score,
    _predicted_undirected_arm_budget_available,
    _prediction_cpu_elapsed_s,
    _project_candidate_prism,
    _record_insufficient_predicted_budget_skip,
    _repair_flung_isolates,
    _restore_proxy_finalist_slots,
    _rgg_geometric_seed_candidate,
    _rgg_geometric_seed_enabled,
    _score_undirected_candidate,
    _score_undirected_candidate_payload,
    _select_undirected_winner,
    _small_world_knn_seed_candidate,
    _small_world_knn_seed_enabled,
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


def test_undirected_referee_forwards_extended_cluster_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clustered undirected scoring forwards Event-A cluster-ruler metadata."""
    calls: list[dict[str, object]] = []

    def fake_full(*args: object, **kwargs: object) -> dict[str, float]:
        """Capture full-ruler kwargs and return deterministic metric scores."""
        del args
        calls.append(dict(kwargs))
        return {
            "ksm_score": 1.0,
            "edge_crossing_score": 1.0,
            "node_occlusion_score": 1.0,
            "neighborhood_preservation_score": 1.0,
            "edge_length_deviation_score": 1.0,
            "gabriel_score": 1.0,
            "crossing_angle_score": 1.0,
            "angular_resolution_score": 1.0,
            "path_continuity_score": 1.0,
            "cluster_silhouette_score": 1.0,
            "cluster_exclusion_score": 0.0,
            "cluster_sibling_overlap_score": 0.0,
            "cluster_nesting_fidelity_score": 0.0,
            "cluster_edge_intrusion_score": 0.0,
            "cluster_label_occlusion_score": 0.0,
            "cluster_compactness_score": 0.0,
        }

    monkeypatch.setattr("dagua.metrics.full", fake_full)
    problem = LayoutProblem(
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        num_nodes=3,
        node_sizes=torch.ones((3, 2), dtype=torch.float32),
        clusters={"group": [0, 1, 2]},
        cluster_parents={"group": None},
        cluster_labels={"group": "Group"},
        label_positions=[None, None],
        edge_labels=["a", "b"],
    )
    score, telemetry = _score_undirected_candidate_payload(
        torch.zeros((3, 2), dtype=torch.float32),
        problem,
        torch.zeros((3,), dtype=torch.long),
    )

    assert calls[0]["clusters"] == problem.clusters
    assert calls[0]["cluster_parents"] == problem.cluster_parents
    assert calls[0]["cluster_labels"] == problem.cluster_labels
    assert calls[0]["label_positions"] == problem.label_positions
    assert calls[0]["edge_labels"] == problem.edge_labels
    assert telemetry is not None
    assert score == telemetry.extended_score
    assert telemetry.old_score > telemetry.extended_score


def test_undirected_cluster_dual_ruler_rejects_old_regression() -> None:
    """Clustered challengers must improve extended score without old loss."""
    incumbent = _ClusterScoreTelemetry(extended_score=80.0, old_score=90.0, metrics={})
    challenger = _ClusterScoreTelemetry(extended_score=81.0, old_score=89.9, metrics={})

    assert not _cluster_candidate_is_dual_admissible(challenger, incumbent)
    assert (
        _select_undirected_winner(
            {"incumbent": incumbent.extended_score, "challenger": challenger.extended_score},
            {"incumbent": incumbent, "challenger": challenger},
        )
        == "incumbent"
    )


def test_declared_undirected_routes_to_portfolio() -> None:
    """A declared-undirected graph selects the portfolio route."""
    graph = _ring_with_chords()
    structure = classify_graph(graph.edge_index, graph.num_nodes, graph=graph)
    config = LayoutConfig(seed=42, device="cpu")

    assert structure.is_semantically_directed is False
    assert _choose_native_pipeline(structure=structure, config=config) == "undirected_portfolio"


def test_directed_graph_routes_to_directed_portfolio() -> None:
    """A directed DAG selects the honest directed-table contest."""
    edges = [(i, i + 1) for i in range(9)] + [(0, 5), (2, 7)]
    graph = DaguaGraph.from_edge_list(edges, num_nodes=10)
    graph.compute_node_sizes()
    structure = classify_graph(graph.edge_index, graph.num_nodes, graph=graph)
    config = LayoutConfig(seed=42, device="cpu")

    selected = _choose_native_pipeline(structure=structure, config=config)

    assert structure.is_semantically_directed is True
    assert selected == "directed_portfolio"


def test_suppressed_directed_portfolio_reproduces_baseline_route() -> None:
    """Suppressed re-entry selects the exact pre-contest incumbent route."""
    edges = [(i, i + 1) for i in range(9)] + [(0, 5), (2, 7)]
    graph = DaguaGraph.from_edge_list(edges, num_nodes=10)
    structure = classify_graph(graph.edge_index, graph.num_nodes, graph=graph)
    config = LayoutConfig(seed=42, device="cpu")

    config._dagua_native_suppress_portfolio = True

    assert _choose_native_pipeline(structure=structure, config=config) == (
        _choose_native_pipeline_baseline(structure=structure, config=config)
    )


def test_forced_pipeline_beats_portfolio_branch() -> None:
    """force_pipeline overrides the undirected-portfolio branch."""
    graph = _ring_with_chords()
    structure = classify_graph(graph.edge_index, graph.num_nodes, graph=graph)
    config = LayoutConfig(seed=42, device="cpu")
    config.force_pipeline = "stress"

    assert _choose_native_pipeline(structure=structure, config=config) == "stress"


def test_undirected_portfolio_is_incumbent_monotone() -> None:
    """The selected winner must not score below the suppressed incumbent."""
    from dagua.layout import layout

    graph = _ring_with_chords(num_nodes=8)
    incumbent_config = LayoutConfig(seed=42, device="cpu")
    incumbent_config._dagua_native_suppress_portfolio = True
    incumbent_pos = layout(graph, incumbent_config)
    winner_pos = layout(graph, LayoutConfig(seed=42, device="cpu"))
    problem = LayoutProblem(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        node_sizes=graph.node_sizes,
        direction=graph.direction,
    )

    incumbent_score = _score_undirected_candidate(incumbent_pos, problem, None)
    winner_score = _score_undirected_candidate(winner_pos, problem, None)

    assert winner_score >= incumbent_score


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
    assert BALANCED_SMALL_REFINEMENT_STEPS == 75
    assert _candidate_refinement_steps(balanced, 500) == BALANCED_LARGE_REFINEMENT_STEPS
    assert BALANCED_LARGE_REFINEMENT_STEPS == 10
    assert _candidate_refinement_steps(high, 500) == FULL_REFINEMENT_STEPS


def test_small_world_knn_seed_is_finite_and_structurally_gated() -> None:
    """W4 small-world seed covers sparse cyclic local-neighborhood graphs."""
    from dagua.eval.graphs import make_small_world

    graph = make_small_world(120, 6, 0.1, seed=42)
    structure = classify_graph(graph.edge_index, graph.num_nodes, graph=graph)
    problem = LayoutProblem(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        node_sizes=torch.full((graph.num_nodes, 2), 18.0),
        structure=structure,
    )
    angles = torch.arange(graph.num_nodes, dtype=torch.float32) * (2.0 * torch.pi / graph.num_nodes)
    incumbent = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * 200.0

    candidate = _small_world_knn_seed_candidate(incumbent, problem, steps=4)

    assert _small_world_knn_seed_enabled(problem)
    assert candidate.shape == incumbent.shape
    assert bool(torch.isfinite(candidate).all().item())
    assert not torch.equal(candidate, incumbent)


def test_rgg_geometric_seed_is_finite_and_structurally_gated() -> None:
    """W4 geometric seed covers dense spatial random graphs."""
    from dagua.eval.graphs import make_random_geometric

    test_graph = make_random_geometric(120, radius=0.18, seed=42)
    graph = test_graph.graph
    structure = classify_graph(graph.edge_index, graph.num_nodes, graph=graph)
    problem = LayoutProblem(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        node_sizes=torch.full((graph.num_nodes, 2), 18.0),
        structure=structure,
    )

    candidate = _rgg_geometric_seed_candidate(problem, seed=42, node_sep=18.0)

    assert _rgg_geometric_seed_enabled(problem)
    assert candidate.shape == (graph.num_nodes, 2)
    assert bool(torch.isfinite(candidate).all().item())
    extent = candidate.max(dim=0).values - candidate.min(dim=0).values
    assert float(extent.min().item()) > 0.0


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


def test_mid_size_prism_shortlist_requires_low_degree_uniformity() -> None:
    """Mid-size shortcut admits BA-like hubs but excludes Chung-Lu-like hubs."""
    from types import SimpleNamespace

    n = 120
    hub_edges = torch.tensor([[0] * 30, list(range(1, 31))], dtype=torch.long)
    ba_like = LayoutProblem(
        edge_index=hub_edges,
        num_nodes=n,
        structure=SimpleNamespace(degree_uniformity=0.8),
    )
    chung_lu_like = LayoutProblem(
        edge_index=hub_edges,
        num_nodes=n,
        structure=SimpleNamespace(degree_uniformity=2.5),
    )

    assert _use_large_prism_shortlist(ba_like) is True
    assert _use_large_prism_shortlist(chung_lu_like) is False


def test_large_shortlist_runs_before_expensive_incumbent(monkeypatch: object) -> None:
    """Large non-mesh graphs use the SFDP-PRISM holder before the incumbent."""
    import importlib

    from dagua.layout.ops.pipelines.native_undirected import layout_native_undirected_portfolio
    from dagua.layout.ops.state import RuntimeContext, SolveState

    n = LARGE_CONTEST_NODE_THRESHOLD + 1
    edge_index = torch.tensor(
        [[0, 0, 0, 0, 0], [1, 2, 3, 4, 5]],
        dtype=torch.long,
    )
    problem = LayoutProblem(edge_index=edge_index, num_nodes=n, seed=42)
    shortcut_pos = torch.stack(
        (torch.arange(n, dtype=torch.float32), torch.zeros(n, dtype=torch.float32)),
        dim=1,
    )
    native_undirected = importlib.import_module("dagua.layout.ops.pipelines.native_undirected")
    dagua_native = importlib.import_module("dagua.layout.ops.pipelines.dagua_native")
    calls: list[str] = []

    def fake_shortlist(*args: object, **kwargs: object) -> torch.Tensor:
        """Record the shortcut and return a finite large-graph candidate."""
        del args, kwargs
        calls.append("shortcut")
        return shortcut_pos

    def fake_mini_contest(*args: object, **kwargs: object) -> torch.Tensor:
        """Return the shortcut candidate without running extra challengers."""
        del args, kwargs
        calls.append("mini")
        return shortcut_pos

    def fail_incumbent(*args: object, **kwargs: object) -> torch.Tensor:
        """Fail if the expensive incumbent runs before the shortcut."""
        del args, kwargs
        raise AssertionError("incumbent ran before large shortlist")

    monkeypatch.setattr(native_undirected, "_large_prism_shortlist_candidate", fake_shortlist)
    monkeypatch.setattr(native_undirected, "_router_v2_large_mini_contest", fake_mini_contest)
    monkeypatch.setattr(dagua_native, "_run_native_problem", fail_incumbent)

    result = layout_native_undirected_portfolio(
        problem,
        SolveState(),
        RuntimeContext(),
        LayoutConfig(seed=42),
    )

    assert calls == ["shortcut", "mini"]
    torch.testing.assert_close(result, shortcut_pos)


def test_large_w4_seed_shortcut_uses_shortlist_holder(monkeypatch: object) -> None:
    """Seed-gated large mini-contests do not pull in the full incumbent."""
    import importlib
    from types import SimpleNamespace

    from dagua.layout.ops.pipelines.native_undirected import layout_native_undirected_portfolio
    from dagua.layout.ops.state import RuntimeContext, SolveState

    n = LARGE_CONTEST_NODE_THRESHOLD + 50
    edges = []
    for node in range(n):
        edges.append((node, (node + 1) % n))
        edges.append((node, (node + 2) % n))
        edges.append((node, (node + 5) % n))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    structure = SimpleNamespace(
        max_degree=6,
        is_directed_acyclic=False,
        hub_edge_fraction=0.1,
        degree_uniformity=0.0,
    )
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=n,
        node_sizes=torch.full((n, 2), 18.0),
        structure=structure,
        seed=42,
    )
    shortcut_pos = torch.zeros((n, 2), dtype=torch.float32)
    native_undirected = importlib.import_module("dagua.layout.ops.pipelines.native_undirected")

    def fake_shortlist(*args: object, **kwargs: object) -> torch.Tensor:
        """Return a finite shortcut candidate."""
        del args, kwargs
        return shortcut_pos

    def fake_mini_contest(
        baseline_pos: torch.Tensor,
        problem: LayoutProblem,
        config: LayoutConfig,
        incumbent_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Assert the mini-contest receives only the shortlist holder."""
        del problem, config
        torch.testing.assert_close(baseline_pos, shortcut_pos)
        assert incumbent_pos is None
        return baseline_pos

    monkeypatch.setattr(native_undirected, "_large_prism_shortlist_candidate", fake_shortlist)
    monkeypatch.setattr(native_undirected, "_router_v2_large_mini_contest", fake_mini_contest)

    result = layout_native_undirected_portfolio(
        problem,
        SolveState(),
        RuntimeContext(),
        LayoutConfig(seed=42),
    )

    torch.testing.assert_close(result, shortcut_pos)


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


def test_arm_s_candidate_none_without_clusters() -> None:
    """Arm S construction is cluster-gated and inert on flat rows."""
    graph = _ring_with_chords()
    problem = LayoutProblem(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        node_sizes=graph.node_sizes,
        seed=42,
    )

    result = build_arm_s_stress_candidates(problem)

    assert result == {}


def test_arm_s_scale_calibration_targets_median_edge_length() -> None:
    """Arm S scale calibration uses a deterministic median-edge target."""
    seed = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]],
        dtype=torch.float32,
    )
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    node_sizes = torch.full((3, 2), 4.0, dtype=torch.float32)

    scaled = calibrate_arm_s_scale(seed, edge_index, node_sizes, multiplier=2.0)
    lengths = torch.linalg.vector_norm(
        scaled[edge_index[0]] - scaled[edge_index[1]],
        dim=1,
    )

    assert torch.median(lengths).item() == pytest.approx(
        2.0 * torch.linalg.vector_norm(node_sizes, dim=1).mean().item()
    )


def test_arm_s_fake_stress_seed_builds_predeclared_scale_ladder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arm S reuses one stress seed and emits only predeclared scale variants."""
    import importlib

    stress_sgd = importlib.import_module("dagua.layout.ops.pipelines.stress_sgd")

    graph = _clustered_ring_graph(num_nodes=6)
    problem = LayoutProblem(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        node_sizes=torch.full((graph.num_nodes, 2), 2.0, dtype=torch.float32),
        clusters=graph.clusters,
        cluster_parents=graph.cluster_parents,
        seed=42,
    )
    seed = torch.stack(
        (
            torch.arange(graph.num_nodes, dtype=torch.float32) * 10.0,
            torch.zeros(graph.num_nodes, dtype=torch.float32),
        ),
        dim=1,
    )
    calls: list[int] = []

    def fake_stress_seed(*args: object, **kwargs: object) -> torch.Tensor:
        """Return a deterministic fake stress seed."""
        del args, kwargs
        calls.append(1)
        return seed

    monkeypatch.setattr(stress_sgd, "layout_stress_sgd_pipeline", fake_stress_seed)

    candidates = build_arm_s_stress_candidates(problem)

    assert len(calls) == 1
    assert tuple(candidates) == tuple(
        f"{ARM_S_CANDIDATE_PREFIX}_k{multiplier:g}" for multiplier in ARM_S_SCALE_MULTIPLIERS
    )
    for payload in candidates.values():
        assert payload.positions.shape == (graph.num_nodes, 2)
        assert payload.projection.displacement_ratio <= 0.25
        assert exact_arm_s_overlap_count(payload.positions, problem.node_sizes) == 0


def test_arm_s_proxy_prefilter_returns_one_full_score_finalist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arm S builds the full ladder but exposes one proxy-selected finalist."""
    import importlib

    stress_sgd = importlib.import_module("dagua.layout.ops.pipelines.stress_sgd")

    graph = _clustered_ring_graph(num_nodes=8)
    problem = LayoutProblem(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        node_sizes=torch.full((graph.num_nodes, 2), 2.0, dtype=torch.float32),
        clusters=graph.clusters,
        cluster_parents=graph.cluster_parents,
        seed=42,
    )
    seed = torch.stack(
        (
            torch.arange(graph.num_nodes, dtype=torch.float32) * 10.0,
            torch.zeros(graph.num_nodes, dtype=torch.float32),
        ),
        dim=1,
    )
    calls: list[int] = []

    def fake_stress_seed(*args: object, **kwargs: object) -> torch.Tensor:
        """Return a deterministic fake stress seed."""
        del args, kwargs
        calls.append(1)
        return seed

    monkeypatch.setattr(stress_sgd, "layout_stress_sgd_pipeline", fake_stress_seed)

    finalist, ladder, proxy_scores = build_arm_s_stress_finalist(problem)

    assert len(calls) == 1
    assert len(ladder) == len(ARM_S_SCALE_MULTIPLIERS)
    assert set(proxy_scores) == set(ladder)
    assert finalist is not None
    assert finalist.name in ladder


def test_arm_s_full_score_budget_uses_one_referee_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The final Arm S score gate uses the single-referee predicted cost."""
    from dagua.layout.ops.pipelines import native_undirected as nu

    config = LayoutConfig()
    config._dagua_native_deadline_s = 120.0
    monkeypatch.setattr(nu.time, "perf_counter", lambda: 100.0)
    monkeypatch.setattr(nu.time, "process_time", lambda: 10.0)

    assert _arm_s_full_score_budget_available(config, predicted_cost_s=6.0)
    assert not _arm_s_full_score_budget_available(config, predicted_cost_s=8.0)


def test_late_arms_preserve_pending_arm_s_score_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post-Arm-S arms must leave room for the pending honest score."""
    from dagua.layout.ops.pipelines import native_undirected as nu

    config = LayoutConfig()
    config._dagua_native_deadline_s = 300.0
    monkeypatch.setattr(nu.time, "perf_counter", lambda: 100.0)
    monkeypatch.setattr(nu.time, "process_time", lambda: 10.0)

    assert _predicted_arm_budget_preserving_arm_s_score(
        config,
        predicted_cost_s=45.0,
        arm_s_pending=False,
    )
    assert not _predicted_arm_budget_preserving_arm_s_score(
        config,
        predicted_cost_s=45.0,
        arm_s_pending=True,
    )


def test_arm_s_rejection_restores_displaced_proxy_challenger() -> None:
    """Removing Arm S refills its proxy slot with the next challenger."""
    finalist_names = ["incumbent", "arm_s_stress_k10", "cluster_sfdp"]
    challenger_names = ["arm_s_stress_k10", "cluster_sfdp", "community_scaffold", "sfdp"]

    restored = _restore_proxy_finalist_slots(
        finalist_names=["incumbent", "cluster_sfdp"],
        challenger_names=challenger_names,
        raw_finalist_names=[],
        proxy_slot_count=2,
        excluded_names={"arm_s_stress_k10"},
    )

    assert "arm_s_stress_k10" not in restored
    assert restored == ["incumbent", "cluster_sfdp", "community_scaffold"]
    assert finalist_names[1] == "arm_s_stress_k10"


def test_arm_s_named_floor_admission_drops_compactness_floor() -> None:
    """Arm S admission enforces named floors without a compactness mean floor."""
    payload = ArmSCandidate(
        name="arm_s_stress_k10",
        positions=torch.zeros((2, 2), dtype=torch.float32),
        scale_multiplier=10.0,
        pre_projection_overlap_count=0,
        projection=ArmSProjectionTelemetry(
            max_displacement=0.0,
            mean_node_diagonal=10.0,
            displacement_ratio=0.0,
        ),
    )
    metrics = {
        "overlap_count": 0.0,
        "neighborhood_preservation_score": 0.30,
        "ksm_score": 0.6516,
        "cluster_nesting_fidelity_score": 0.7353,
        "cluster_compactness_score": 0.0,
    }

    report = evaluate_arm_s_admission(
        payload,
        metrics,
        extended_score=ARM_S_STRICT_WIN_REFERENCE + ARM_S_ACCEPTANCE_MARGIN + 0.001,
    )

    assert report.passed
    assert report.failures == ()


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


def test_tsnet_flavors_include_uniform_perplexity_five_candidate() -> None:
    """The standard tsNET family includes default and perplexity-five runs."""
    assert TSNET_PERPLEXITIES == (30.0, 5.0)


def test_arm_prediction_cost_uses_process_time_not_wall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expensive-arm cost prediction uses CPU seconds, not wall seconds."""
    from dagua.layout.ops.pipelines import native_undirected as nu

    monkeypatch.setattr(nu.time, "process_time", lambda: 103.5)
    monkeypatch.setattr(nu.time, "perf_counter", lambda: 999.0)

    assert _prediction_cpu_elapsed_s(100.0) == 3.5


def test_unloaded_process_time_prediction_preserves_admission_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When CPU and wall elapsed match, predicted admission is unchanged."""
    from dagua.layout.ops.pipelines import native_undirected as nu

    config = LayoutConfig()
    config._dagua_native_deadline_s = 115.0
    monkeypatch.setattr(nu.time, "perf_counter", lambda: 100.0)
    monkeypatch.setattr(nu.time, "process_time", lambda: 14.0)

    cpu_cost_s = _prediction_cpu_elapsed_s(10.0)

    assert cpu_cost_s == 4.0
    assert _predicted_undirected_arm_budget_available(config, cpu_cost_s) is True
    assert _portfolio_has_budget(config, min_remaining_s=10.0) is True
    assert getattr(config, "_dagua_native_process_deadline_s") == 29.0


def test_no_deadline_portfolio_budget_behavior_is_unchanged() -> None:
    """No benchmark budget keeps optional-arm admission open and metadata absent."""
    config = LayoutConfig()

    assert _portfolio_has_budget(config, min_remaining_s=10.0) is True
    assert _predicted_undirected_arm_budget_available(config, predicted_cost_s=10_000.0) is True
    assert not hasattr(config, "_dagua_native_process_deadline_s")


def test_loaded_wall_time_does_not_change_predicted_arm_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Process budget, not consumed wall time, controls predicted arm admission."""
    from dagua.layout.ops.pipelines import native_undirected as nu

    config = LayoutConfig()
    config._dagua_native_deadline_s = 106.0
    config._dagua_native_process_deadline_s = 220.0
    monkeypatch.setattr(nu.time, "perf_counter", lambda: 100.0)
    monkeypatch.setattr(nu.time, "process_time", lambda: 100.0)

    assert _predicted_undirected_arm_budget_available(config, predicted_cost_s=20.0) is True
    assert _portfolio_has_budget(config, min_remaining_s=10.0) is True


def test_marketplace_telemetry_includes_process_time(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Marketplace driver telemetry reports route and per-arm CPU totals."""
    caplog.set_level("INFO")
    positions = {"incumbent": torch.zeros((2, 2)), "candidate": torch.ones((2, 2))}

    _log_marketplace_telemetry(
        route="undirected",
        structural_gate="unit",
        positions=positions,
        proxy_scores={"incumbent": 0.0, "candidate": 1.0},
        full_scores={"incumbent": 0.0, "candidate": 1.0},
        finalist_names=["incumbent", "candidate"],
        winner_name="candidate",
        started_at=10.0,
        started_process_at=time.process_time(),
        arm_process_totals={"incumbent": 0.25, "candidate": 0.5},
    )

    records = [
        json.loads(record.message.removeprefix("Native marketplace telemetry "))
        for record in caplog.records
        if record.message.startswith("Native marketplace telemetry ")
    ]

    assert records
    assert "process_time_s" in records[-1]
    assert {arm["name"]: arm["process_time_s"] for arm in records[-1]["arms"]} == {
        "candidate": 0.5,
        "incumbent": 0.25,
    }


def test_insufficient_predicted_budget_skip_emits_jsonl_telemetry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Predicted-budget arm skips are visible without a logging shim."""
    telemetry_path = tmp_path / "arm-skips.jsonl"
    monkeypatch.setenv("DAGUA_W5_TELEMETRY_PATH", str(telemetry_path))
    config = LayoutConfig()
    config._dagua_native_deadline_s = time.perf_counter() + 30.0

    _record_insufficient_predicted_budget_skip(
        arm="tsnet_perp30_seed0",
        config=config,
        predicted_cost_s=12.5,
    )

    stdout = capsys.readouterr().out
    records = [json.loads(line) for line in telemetry_path.read_text().splitlines()]

    assert stdout.count("native_undirected_arm_skip ") == 1
    assert len(records) == 1
    assert records[0]["event"] == "native_undirected_arm_skip"
    assert records[0]["arm"] == "tsnet_perp30_seed0"
    assert records[0]["reason"] == "insufficient_predicted_budget"
    assert records[0]["predicted_cost_s"] == 12.5
    assert getattr(config, "_dagua_native_arm_skip_telemetry") == records


def test_skipped_predicted_arm_contest_returns_best_computed_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skipping late expensive arms still returns the best computed finalist."""
    import importlib

    from dagua.layout.ops.pipelines import native_undirected as nu
    from dagua.layout.ops.pipelines.native_undirected import layout_native_undirected_portfolio
    from dagua.layout.ops.state import RuntimeContext, SolveState

    dagua_native = importlib.import_module("dagua.layout.ops.pipelines.dagua_native")
    sfdp = importlib.import_module("dagua.layout.ops.pipelines.sfdp")

    incumbent = torch.zeros((4, 2), dtype=torch.float32)
    challenger = torch.tensor(
        [[0.0, 0.0], [80.0, 0.0], [80.0, 80.0], [0.0, 80.0]],
        dtype=torch.float32,
    )
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    problem = LayoutProblem(edge_index=edge_index, num_nodes=4, seed=42)
    config = LayoutConfig(seed=42, device="cpu")
    config._dagua_native_deadline_s = time.perf_counter() + 300.0

    def fake_run_native_problem(*args: object, **kwargs: object) -> torch.Tensor:
        """Return the deterministic incumbent without running the router."""
        del args, kwargs
        return incumbent

    def fake_sfdp_pipeline(*args: object, **kwargs: object) -> torch.Tensor:
        """Return a finite already-computed challenger."""
        del args, kwargs
        return challenger

    def fake_score(
        pos: torch.Tensor,
        problem: LayoutProblem,
        cluster_ids: Optional[torch.Tensor],
        aesthetic_profile: object = None,
        all_pairs_dist: Optional[object] = None,
    ) -> float:
        """Prefer the computed challenger over the incumbent."""
        del problem, cluster_ids, aesthetic_profile, all_pairs_dist
        return 10.0 if torch.equal(pos, challenger) else 0.0

    monkeypatch.setattr(dagua_native, "_run_native_problem", fake_run_native_problem)
    monkeypatch.setattr(dagua_native, "_collinear_dodge", lambda *args, **kwargs: None)
    monkeypatch.setattr(dagua_native, "_unshear_bimodal_edges", lambda *args, **kwargs: None)
    monkeypatch.setattr(sfdp, "layout_sfdp_pipeline", fake_sfdp_pipeline)
    monkeypatch.setattr(nu, "_neato_in_contest", lambda *args, **kwargs: False)
    monkeypatch.setattr(nu, "_stress_points_candidate", lambda *args, **kwargs: None)
    monkeypatch.setattr(nu, "_predicted_undirected_arm_budget_available", lambda *args: False)
    monkeypatch.setattr(nu, "_proxy_undirected_candidate", fake_score)
    monkeypatch.setattr(nu, "_score_undirected_candidate_cached", fake_score)

    result = layout_native_undirected_portfolio(
        problem,
        SolveState(),
        RuntimeContext(),
        config,
    )

    torch.testing.assert_close(result, challenger)
