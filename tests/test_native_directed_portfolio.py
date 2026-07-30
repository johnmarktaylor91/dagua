"""Tests for the r83 directed-table native portfolio."""

from __future__ import annotations

import importlib
import inspect
import signal
import time
from types import SimpleNamespace
from typing import Callable, Optional, TypeVar

import pytest
import torch

from dagua.config import LayoutConfig
from dagua.eval.graphs import _make_r8_lr_direction, get_test_graphs
from dagua.graph import DaguaGraph
from dagua.layout import layout
from dagua.layout.graph_classify import classify_graph
from dagua.layout.ops.ordering import _expanded_layered_graph
from dagua.layout.ops.pipelines.dagua_native import _choose_native_pipeline
from dagua.layout.ops.pipelines.native_directed import (
    DIRECTED_FULL_REFEREE_TOP_K,
    DIRECTED_NESTED_STRESS_EDGE_NODE_RATIO_MAX,
    DIRECTED_NESTED_STRESS_MAX_CLUSTER_DEPTH,
    DIRECTED_NESTED_STRESS_MAX_NODES,
    DIRECTED_NESTED_STRESS_PARETO_KEYS,
    IGRAPH_OUTPUT_SCALE,
    SUGIYAMA_FIDELITY_MODES,
    SUGIYAMA_NODE_SEP_GRID,
    SUGIYAMA_RANK_SEP_GRID,
    _assign_recombinant_x_coordinates,
    _bounded_connected_nested_dag_for_stress,
    _build_dot_order_candidate,
    _build_fan_compaction_candidate,
    _build_nested_stress_candidate,
    _clean_fan_bundle_for_compaction,
    _crossing_edge_pairs,
    _directed_cluster_candidate_is_dual_admissible,
    _directed_dot_order_candidates,
    _directed_dot_order_enabled,
    _directed_mrtree_enabled,
    _directed_ordering_candidate_dual_dominates,
    _directed_pivot_mds_candidates,
    _directed_recombinant_layered_candidates,
    _directed_recombinant_layered_enabled,
    _directed_stress_blend_candidates,
    _DirectedClusterScoreTelemetry,
    _DotOrderSpec,
    _exact_crossing_count,
    _exact_crossing_count_loop,
    _fan_compaction_candidate_is_accepted,
    _force_challengers_enabled,
    _full_sugiyama_grid_enabled,
    _lever2_expanded_x_assignment_enabled,
    _maybe_accept_fan_compaction_arm,
    _maybe_accept_nested_stress_arm,
    _nested_stress_candidate_pareto_admissible,
    _ordering_cost_admissible,
    _rank_local_zero_crossing_swap_candidate,
    _rank_to_nodes_from_incumbent_y,
    _recombinant_rank_values,
    _register_challenger_variants,
    _restore_projected_rank_order,
    _runtime_referee_telemetry,
    _score_directed_candidate,
    _score_directed_candidate_pair,
    _score_directed_candidate_referee_payload,
    _select_directed_winner,
    layout_native_directed_portfolio,
)
from dagua.layout.ops.pipelines.native_finisher import W5ScorePair, w5_dominates
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState

_T = TypeVar("_T")


def _fan_bundle_problem() -> LayoutProblem:
    """Return a clean multi-hub fan-bundle problem.

    Returns
    -------
    LayoutProblem
        Directed hub-spoke DAG with two dominant fan hubs.
    """
    edges: list[tuple[int, int]] = []
    entry = 0
    exit_node = 1
    next_node = 2
    hubs: list[int] = []
    for _hub_index in range(2):
        hub = next_node
        next_node += 1
        hubs.append(hub)
        edges.append((entry, hub))
        for _spoke_index in range(5):
            spoke = next_node
            next_node += 1
            edges.append((hub, spoke))
            edges.append((spoke, exit_node))
    edges.append((hubs[0], hubs[1]))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return LayoutProblem(
        edge_index=edge_index,
        num_nodes=next_node,
        node_sizes=torch.ones((next_node, 2), dtype=torch.float32),
        seed=7,
    )


def _wide_single_layer_problem() -> LayoutProblem:
    """Return the plain wide-layer canary shape.

    Returns
    -------
    LayoutProblem
        Single source and sink around a wide middle layer.
    """
    edges: list[tuple[int, int]] = []
    source = 0
    sink = 1
    for node in range(2, 12):
        edges.append((source, node))
        edges.append((node, sink))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return LayoutProblem(
        edge_index=edge_index,
        num_nodes=12,
        node_sizes=torch.ones((12, 2), dtype=torch.float32),
        seed=7,
    )


def _random_bipartite_problem() -> LayoutProblem:
    """Return a deterministic random-bipartite canary shape.

    Returns
    -------
    LayoutProblem
        Bipartite DAG whose middle nodes do not reconverge as fan spokes.
    """
    edges = [
        (left, 10 + ((left * 7 + offset * 3) % 10)) for left in range(10) for offset in range(3)
    ]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return LayoutProblem(
        edge_index=edge_index,
        num_nodes=20,
        node_sizes=torch.ones((20, 2), dtype=torch.float32),
        seed=7,
    )


def _nested_dag_problem() -> LayoutProblem:
    """Return a connected compound DAG for nested-stress arm tests.

    Returns
    -------
    LayoutProblem
        Runtime-declared nested DAG with two child clusters under a parent.
    """
    edge_index = torch.tensor(
        [[0, 0, 1, 2, 3, 4, 5], [1, 2, 3, 3, 4, 5, 6]],
        dtype=torch.long,
    )
    return LayoutProblem(
        edge_index=edge_index,
        num_nodes=7,
        node_sizes=torch.full((7, 2), 10.0, dtype=torch.float32),
        seed=11,
        clusters={"root": list(range(7)), "left": [0, 1, 3], "right": [2, 4, 5, 6]},
        cluster_parents={"root": None, "left": "root", "right": "root"},
        direction="TB",
    )


def test_nested_stress_prefilter_builds_only_runtime_nested_connected_dag() -> None:
    """The nested-stress arm opens only for bounded connected compound DAGs."""
    nested = _nested_dag_problem()
    plain = LayoutProblem(
        edge_index=nested.edge_index,
        num_nodes=nested.num_nodes,
        node_sizes=nested.node_sizes,
    )
    cyclic = LayoutProblem(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        num_nodes=3,
        node_sizes=torch.ones((3, 2), dtype=torch.float32),
        clusters={"root": [0, 1, 2], "child": [0, 1]},
        cluster_parents={"root": None, "child": "root"},
    )
    disconnected = LayoutProblem(
        edge_index=torch.tensor([[0, 2], [1, 3]], dtype=torch.long),
        num_nodes=4,
        node_sizes=torch.ones((4, 2), dtype=torch.float32),
        clusters={"root": [0, 1, 2, 3], "child": [0, 1]},
        cluster_parents={"root": None, "child": "root"},
    )

    assert _bounded_connected_nested_dag_for_stress(nested)
    assert not _bounded_connected_nested_dag_for_stress(plain)
    assert not _bounded_connected_nested_dag_for_stress(cyclic)
    assert not _bounded_connected_nested_dag_for_stress(disconnected)


def test_nested_stress_prefilter_enforces_cosigned_runtime_caps() -> None:
    """The nested-stress guard rejects oversized, dense, and over-deep DAGs."""
    nested = _nested_dag_problem()
    oversized = LayoutProblem(
        edge_index=torch.stack(
            [
                torch.arange(DIRECTED_NESTED_STRESS_MAX_NODES, dtype=torch.long),
                torch.arange(1, DIRECTED_NESTED_STRESS_MAX_NODES + 1, dtype=torch.long),
            ]
        ),
        num_nodes=DIRECTED_NESTED_STRESS_MAX_NODES + 1,
        node_sizes=torch.ones((DIRECTED_NESTED_STRESS_MAX_NODES + 1, 2), dtype=torch.float32),
        clusters={"root": list(range(DIRECTED_NESTED_STRESS_MAX_NODES + 1)), "child": [0, 1]},
        cluster_parents={"root": None, "child": "root"},
    )
    dense_edges = [(source, target) for source in range(8) for target in range(source + 1, 8)]
    dense = LayoutProblem(
        edge_index=torch.tensor(dense_edges, dtype=torch.long).t().contiguous(),
        num_nodes=8,
        node_sizes=torch.ones((8, 2), dtype=torch.float32),
        clusters={"root": list(range(8)), "child": [0, 1]},
        cluster_parents={"root": None, "child": "root"},
    )
    deep_parents: dict[str, Optional[str]] = {"root": None}
    parent = "root"
    for depth in range(DIRECTED_NESTED_STRESS_MAX_CLUSTER_DEPTH + 1):
        child = f"child_{depth}"
        deep_parents[child] = parent
        parent = child
    over_deep = LayoutProblem(
        edge_index=nested.edge_index,
        num_nodes=nested.num_nodes,
        node_sizes=nested.node_sizes,
        clusters={"root": list(range(nested.num_nodes)), **{name: [0] for name in deep_parents}},
        cluster_parents=deep_parents,
    )

    assert _bounded_connected_nested_dag_for_stress(nested)
    assert not _bounded_connected_nested_dag_for_stress(oversized)
    assert not _bounded_connected_nested_dag_for_stress(dense)
    assert len(dense_edges) / dense.num_nodes > DIRECTED_NESTED_STRESS_EDGE_NODE_RATIO_MAX
    assert not _bounded_connected_nested_dag_for_stress(over_deep)


def test_nested_stress_strict_pareto_rejects_nondominating_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-dominating nested-stress candidate keeps the incumbent unchanged."""
    problem = _nested_dag_problem()
    incumbent = torch.arange(problem.num_nodes * 2, dtype=torch.float32).reshape(
        problem.num_nodes,
        2,
    )
    challenger = incumbent + 10.0
    native_directed = importlib.import_module("dagua.layout.ops.pipelines.native_directed")

    metrics = [
        {
            "dag_consistency": 1.0,
            "directed_flow_score": 0.8,
            "neighborhood_preservation_score": 0.9,
            "edge_length_deviation_score": 0.7,
        },
        {
            "dag_consistency": 1.0,
            "directed_flow_score": 0.9,
            "neighborhood_preservation_score": 0.89,
            "edge_length_deviation_score": 0.8,
        },
    ]

    def fake_metrics(*args: object, **kwargs: object) -> dict[str, float]:
        """Return incumbent metrics first, then candidate metrics."""
        del args, kwargs
        return metrics.pop(0)

    monkeypatch.setattr(native_directed, "_build_nested_stress_candidate", lambda *args: challenger)
    monkeypatch.setattr(native_directed, "_nested_stress_raw_metrics", fake_metrics)

    returned = _maybe_accept_nested_stress_arm(
        problem,
        incumbent,
        LayoutConfig(),
        cluster_ids=torch.zeros((problem.num_nodes,), dtype=torch.long),
        all_pairs_dist=None,
        seed=11,
    )

    assert returned.data_ptr() == incumbent.data_ptr()
    assert torch.equal(returned, incumbent)


def test_nested_stress_comparator_requires_dag_floor_and_strict_pareto() -> None:
    """Nested-stress admission has no tolerance and enforces the DAG floor."""
    assert "cluster_sibling_overlap_score" in DIRECTED_NESTED_STRESS_PARETO_KEYS
    assert "cluster_nesting_fidelity_score" in DIRECTED_NESTED_STRESS_PARETO_KEYS
    incumbent = {
        "dag_consistency": 0.9,
        "ksm_score": 0.7,
        "neighborhood_preservation_score": 0.6,
        "cluster_sibling_overlap_score": 0.9,
        "cluster_nesting_fidelity_score": 0.9,
    }
    equal = dict(incumbent)
    below_floor = {
        "dag_consistency": 0.49,
        "ksm_score": 1.0,
        "neighborhood_preservation_score": 1.0,
        "cluster_sibling_overlap_score": 1.0,
        "cluster_nesting_fidelity_score": 1.0,
    }
    dominating = {
        "dag_consistency": 0.9,
        "ksm_score": 0.8,
        "neighborhood_preservation_score": 0.6,
        "cluster_sibling_overlap_score": 0.9,
        "cluster_nesting_fidelity_score": 0.9,
    }
    degraded_sibling = {
        "dag_consistency": 0.9,
        "ksm_score": 0.8,
        "neighborhood_preservation_score": 0.6,
        "cluster_sibling_overlap_score": 0.89,
        "cluster_nesting_fidelity_score": 0.9,
    }

    assert not _nested_stress_candidate_pareto_admissible(equal, incumbent)
    assert not _nested_stress_candidate_pareto_admissible(below_floor, incumbent)
    assert not _nested_stress_candidate_pareto_admissible(degraded_sibling, incumbent)
    assert _nested_stress_candidate_pareto_admissible(dominating, incumbent)


def test_nested_stress_warm_starts_from_live_incumbent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Stress-SGD arm receives the live incumbent as ``init_pos``."""
    problem = _nested_dag_problem()
    incumbent = torch.arange(problem.num_nodes * 2, dtype=torch.float32).reshape(
        problem.num_nodes,
        2,
    )
    captured: dict[str, torch.Tensor] = {}
    stress_sgd = importlib.import_module("dagua.layout.ops.pipelines.stress_sgd")

    def fake_stress_sgd(**kwargs: object) -> torch.Tensor:
        """Capture warm-start coordinates and return a finite candidate."""
        init_pos = kwargs["init_pos"]
        assert isinstance(init_pos, torch.Tensor)
        captured["init_pos"] = init_pos.clone()
        return init_pos + 1.0

    monkeypatch.setattr(stress_sgd, "layout_stress_sgd_pipeline", fake_stress_sgd)

    _build_nested_stress_candidate(problem, incumbent, LayoutConfig(), seed=19)

    assert torch.equal(captured["init_pos"], incumbent)


def test_nested_stress_builder_is_deterministic_without_competitor_import() -> None:
    """The real nested-stress builder is deterministic and in-house only."""
    problem = _nested_dag_problem()
    incumbent = torch.stack(
        [torch.arange(problem.num_nodes, dtype=torch.float32), torch.zeros(problem.num_nodes)],
        dim=1,
    )
    first = _build_nested_stress_candidate(problem, incumbent, LayoutConfig(), seed=23)
    second = _build_nested_stress_candidate(problem, incumbent, LayoutConfig(), seed=23)
    native_directed = importlib.import_module("dagua.layout.ops.pipelines.native_directed")

    assert torch.equal(first, second)
    assert "Competitor" not in inspect.getsource(native_directed._build_nested_stress_candidate)


def test_directed_fan_compaction_prefilter_builds_only_clean_fan_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fan arm builds only after the clean fan-bundle pre-filter opens."""
    built = 0
    incumbent = torch.zeros((_fan_bundle_problem().num_nodes, 2), dtype=torch.float32)

    def fake_candidate(
        problem: LayoutProblem,
        incumbent_pos: torch.Tensor,
        config: LayoutConfig,
    ) -> torch.Tensor:
        """Record fan-arm construction and return a rejected finite candidate."""
        nonlocal built
        del problem, config
        built += 1
        return incumbent_pos + 1.0

    native_directed = importlib.import_module("dagua.layout.ops.pipelines.native_directed")
    monkeypatch.setattr(native_directed, "_build_fan_compaction_candidate", fake_candidate)
    monkeypatch.setattr(
        native_directed,
        "_fan_compaction_candidate_is_accepted",
        lambda *args: False,
    )

    fan_problem = _fan_bundle_problem()
    assert _clean_fan_bundle_for_compaction(fan_problem)
    assert not _clean_fan_bundle_for_compaction(_wide_single_layer_problem())
    assert not _clean_fan_bundle_for_compaction(_random_bipartite_problem())

    _maybe_accept_fan_compaction_arm(fan_problem, incumbent, LayoutConfig())
    _maybe_accept_fan_compaction_arm(
        _wide_single_layer_problem(),
        torch.zeros((12, 2), dtype=torch.float32),
        LayoutConfig(),
    )
    _maybe_accept_fan_compaction_arm(
        _random_bipartite_problem(),
        torch.zeros((20, 2), dtype=torch.float32),
        LayoutConfig(),
    )

    assert built == 1


def test_directed_fan_compaction_comparator_requires_halved_area_and_no_debt() -> None:
    """Fan-arm acceptance uses visual area, crossings, and overlaps only."""
    problem = LayoutProblem(
        edge_index=torch.tensor([[0, 2], [1, 3]], dtype=torch.long),
        num_nodes=4,
        node_sizes=torch.ones((4, 2), dtype=torch.float32),
    )
    incumbent = torch.tensor(
        [[-10.0, 0.0], [10.0, 0.0], [-10.0, 5.0], [10.0, 5.0]],
        dtype=torch.float32,
    )
    compact = incumbent * 0.25
    not_compact_enough = incumbent * 0.75
    crossing_candidate = torch.tensor(
        [[-2.5, 0.0], [2.5, 5.0], [-2.5, 5.0], [2.5, 0.0]],
        dtype=torch.float32,
    )
    overlapping_candidate = torch.zeros((4, 2), dtype=torch.float32)

    assert _fan_compaction_candidate_is_accepted(incumbent, compact, problem)
    assert not _fan_compaction_candidate_is_accepted(incumbent, not_compact_enough, problem)
    assert not _fan_compaction_candidate_is_accepted(incumbent, crossing_candidate, problem)
    assert not _fan_compaction_candidate_is_accepted(incumbent, overlapping_candidate, problem)


def test_directed_fan_compaction_reject_keeps_incumbent_bit_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-improving fan candidate returns the exact incumbent tensor."""
    problem = _fan_bundle_problem()
    incumbent = torch.arange(problem.num_nodes * 2, dtype=torch.float32).reshape(
        problem.num_nodes,
        2,
    )
    challenger = incumbent + 100.0

    native_directed = importlib.import_module("dagua.layout.ops.pipelines.native_directed")
    monkeypatch.setattr(
        native_directed,
        "_build_fan_compaction_candidate",
        lambda *args: challenger,
    )
    monkeypatch.setattr(
        native_directed,
        "_fan_compaction_candidate_is_accepted",
        lambda *args: False,
    )

    returned = _maybe_accept_fan_compaction_arm(problem, incumbent, LayoutConfig())

    assert returned.data_ptr() == incumbent.data_ptr()
    assert torch.equal(returned, incumbent)


def test_directed_fan_compaction_acceptance_is_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An accepted fan arm returns before scorer-selected arms can replace it."""
    problem = _fan_bundle_problem()
    incumbent = torch.zeros((problem.num_nodes, 2), dtype=torch.float32)
    challenger = torch.ones((problem.num_nodes, 2), dtype=torch.float32)

    def fake_native_problem(*args: object, **kwargs: object) -> torch.Tensor:
        """Return a deterministic incumbent for the directed portfolio."""
        del args, kwargs
        return incumbent

    def fake_accept(
        accepted_problem: LayoutProblem,
        incumbent_pos: torch.Tensor,
        config: LayoutConfig,
    ) -> torch.Tensor:
        """Mark the fan arm accepted and return the compact challenger."""
        del accepted_problem, incumbent_pos
        config._dagua_native_fan_compaction_accepted = True
        return challenger

    def fail_score(*args: object, **kwargs: object) -> float:
        """Fail if terminal fan acceptance falls through to scoring."""
        del args, kwargs
        raise AssertionError("accepted fan arm must be terminal")

    dagua_native = importlib.import_module("dagua.layout.ops.pipelines.dagua_native")
    native_directed = importlib.import_module("dagua.layout.ops.pipelines.native_directed")
    monkeypatch.setattr(dagua_native, "_run_native_problem", fake_native_problem)
    monkeypatch.setattr(native_directed, "_maybe_accept_fan_compaction_arm", fake_accept)
    monkeypatch.setattr(native_directed, "_score_directed_candidate_cached", fail_score)

    returned = layout_native_directed_portfolio(
        problem,
        SolveState(),
        RuntimeContext(),
        LayoutConfig(),
    )

    assert torch.equal(returned, challenger)


def test_directed_fan_compaction_builder_is_deterministic_without_competitor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real fan arm is deterministic and does not call competitor pipelines."""
    problem = _fan_bundle_problem()
    incumbent = torch.zeros((problem.num_nodes, 2), dtype=torch.float32)

    def fail_competitor(*args: object, **kwargs: object) -> torch.Tensor:
        """Fail if the fan arm delegates to the Sugiyama competitor pipeline."""
        del args, kwargs
        raise AssertionError("competitor pipeline must not be called")

    sugiyama = importlib.import_module("dagua.layout.ops.pipelines.sugiyama")
    monkeypatch.setattr(sugiyama, "layout_sugiyama_pipeline", fail_competitor)

    first = _build_fan_compaction_candidate(problem, incumbent, LayoutConfig())
    second = _build_fan_compaction_candidate(problem, incumbent, LayoutConfig())

    assert first is not None
    assert second is not None
    assert torch.equal(first, second)


def test_directed_referee_forwards_extended_cluster_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clustered directed scoring forwards Event-A cluster-ruler metadata."""
    calls: list[dict[str, object]] = []

    def fake_full(*args: object, **kwargs: object) -> dict[str, float]:
        """Capture full-ruler kwargs and return deterministic directed scores."""
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
            "directed_flow_score": 1.0,
            "depth_order_score": 1.0,
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
    score, telemetry = _score_directed_candidate_referee_payload(
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


def test_directed_cluster_dual_ruler_uses_v3_with_flag_guard() -> None:
    """Clustered directed challengers use V3 plus frozen degeneracy flags."""
    incumbent = _DirectedClusterScoreTelemetry(
        extended_score=80.0,
        old_score=90.0,
        metrics={},
        v3_tiered=75.0,
        champion_ineligibility_flags=frozenset(),
    )
    challenger = _DirectedClusterScoreTelemetry(
        extended_score=81.0,
        old_score=10.0,
        metrics={},
        v3_tiered=75.1,
        champion_ineligibility_flags=frozenset(),
    )
    regressor = _DirectedClusterScoreTelemetry(
        extended_score=82.0,
        old_score=91.0,
        metrics={},
        v3_tiered=75.2,
        champion_ineligibility_flags=frozenset({"DEGENERATE_SCALE"}),
    )

    assert _directed_cluster_candidate_is_dual_admissible(challenger, incumbent)
    assert not _directed_cluster_candidate_is_dual_admissible(regressor, incumbent)
    assert (
        _select_directed_winner(
            {"incumbent": incumbent.extended_score, "challenger": challenger.extended_score},
            {"incumbent": incumbent, "challenger": challenger},
        )
        == "challenger"
    )


def test_directed_runtime_referee_preserves_non_weighted_selection() -> None:
    """Neutral severe-G6 keys leave non-weighted directed selection unchanged."""
    incumbent = _DirectedClusterScoreTelemetry(
        extended_score=10.0,
        old_score=10.0,
        metrics={},
        v3_referee_eligibility_key=(1, -0.0),
    )
    challenger = _DirectedClusterScoreTelemetry(
        extended_score=11.0,
        old_score=11.0,
        metrics={},
        v3_referee_eligibility_key=(1, -0.0),
    )

    assert (
        _select_directed_winner(
            {"incumbent": incumbent.extended_score, "challenger": challenger.extended_score},
            {"incumbent": incumbent, "challenger": challenger},
        )
        == "challenger"
    )


def test_directed_runtime_referee_demotes_weighted_severe_g6_breach() -> None:
    """Weighted directed selection ranks eligibility before composite score."""
    incumbent = _DirectedClusterScoreTelemetry(
        extended_score=10.0,
        old_score=10.0,
        metrics={},
        v3_referee_eligibility_key=(1, -0.0),
    )
    challenger = _DirectedClusterScoreTelemetry(
        extended_score=99.0,
        old_score=99.0,
        metrics={},
        v3_referee_eligibility_key=(0, -0.20),
        v3_severe_g6_breach=True,
        v3_referee_ineligibility_reason="severe_g6_breach",
    )

    assert (
        _select_directed_winner(
            {"incumbent": incumbent.extended_score, "challenger": challenger.extended_score},
            {"incumbent": incumbent, "challenger": challenger},
        )
        == "incumbent"
    )


def _run_with_watchdog(func: Callable[[], _T], timeout_s: float) -> _T:
    """Run a callable with a wall-clock alarm in the current process.

    Parameters
    ----------
    func : Callable[[], _T]
        Zero-argument callable to execute.
    timeout_s : float
        Maximum runtime in seconds before raising ``TimeoutError``.

    Returns
    -------
    _T
        Value returned by ``func`` before the alarm fires.
    """
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0.0)

    def handle_timeout(signum: int, frame: object) -> None:
        """Raise a Python exception when the watchdog alarm fires."""
        del signum, frame
        raise TimeoutError(f"operation exceeded {timeout_s:.1f}s watchdog")

    signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        return func()
    finally:
        signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])
        signal.signal(signal.SIGALRM, previous_handler)


def test_r8_nested_lr_direction_native_layout_terminates() -> None:
    """Native directed portfolio returns finite R8 LR positions promptly."""
    graph = _make_r8_lr_direction().graph
    graph.compute_node_sizes()
    config = LayoutConfig(algorithm="dagua_native", seed=42, device="cpu")

    started = time.perf_counter()
    positions = _run_with_watchdog(lambda: layout(graph, config), timeout_s=20.0)
    runtime_s = time.perf_counter() - started

    assert positions.shape == (30, 2)
    assert torch.isfinite(positions).all()
    assert runtime_s < 20.0


def test_semantic_cyclic_graph_routes_to_common_contest() -> None:
    """A semantic digraph with a cycle follows the ruler's common table."""
    edges = [(0, 1), (1, 2), (2, 0), (2, 3)]
    graph = DaguaGraph.from_edge_list(
        edges,
        num_nodes=4,
        is_semantically_directed=True,
    )
    structure = classify_graph(graph.edge_index, graph.num_nodes, graph=graph)

    assert structure.is_directed_acyclic is False
    assert _choose_native_pipeline(structure, LayoutConfig()) == "undirected_portfolio"


def test_force_gate_accepts_skip_dense_dag_and_multiedges() -> None:
    """R7 force challengers open for long skips or duplicate directed edges."""
    skip_edges = torch.tensor(
        [[0, 1, 2, 0, 0, 1], [1, 2, 3, 2, 3, 3]],
        dtype=torch.long,
    )
    multiedges = torch.tensor([[0, 0, 1], [1, 1, 2]], dtype=torch.long)
    chain = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

    assert _force_challengers_enabled(skip_edges, 4)
    assert _force_challengers_enabled(multiedges, 3)
    assert not _force_challengers_enabled(chain, 4)


def _target_recombinant_structure() -> SimpleNamespace:
    """Return classifier metadata representative of the item-3 target rows.

    Returns
    -------
    SimpleNamespace
        Structural object with the attributes consumed by the recombinant
        layered gate.
    """
    return SimpleNamespace(
        is_directed_acyclic=True,
        is_acyclic=True,
        is_semantically_directed=True,
        topology_tags=(),
        num_layers_effective=18,
        num_layers=19,
        edge_to_node_ratio=2.5,
        hub_edge_fraction=0.35,
        diameter_estimate=5,
    )


def test_recombinant_layered_gate_is_targeted_and_off_class_noop() -> None:
    """Recombinant candidates are not constructed for off-class graphs."""
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    target = LayoutProblem(
        edge_index=edge_index,
        num_nodes=100,
        node_sizes=torch.full((100, 2), 20.0),
        structure=_target_recombinant_structure(),
    )
    undirected = LayoutProblem(
        edge_index=edge_index,
        num_nodes=100,
        node_sizes=torch.full((100, 2), 20.0),
        structure=SimpleNamespace(
            **{
                **vars(_target_recombinant_structure()),
                "is_semantically_directed": False,
            }
        ),
    )
    broad_random_dag = LayoutProblem(
        edge_index=edge_index,
        num_nodes=200,
        node_sizes=torch.full((200, 2), 20.0),
        structure=SimpleNamespace(
            **{
                **vars(_target_recombinant_structure()),
                "diameter_estimate": 10,
            }
        ),
    )

    assert _directed_recombinant_layered_enabled(target)
    assert not _directed_recombinant_layered_enabled(undirected)
    assert not _directed_recombinant_layered_enabled(broad_random_dag)


def test_recombinant_layered_budget_gate_skips_when_tight() -> None:
    """A tight benchmark deadline prevents recombinant candidate construction."""
    edge_index = torch.tensor([[0, 0, 1], [1, 2, 3]], dtype=torch.long)
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=100,
        node_sizes=torch.full((100, 2), 20.0),
        structure=_target_recombinant_structure(),
    )
    config = LayoutConfig()
    config._dagua_native_deadline_s = time.perf_counter() + 0.01
    incumbent = torch.zeros((100, 2), dtype=torch.float32)

    assert _directed_recombinant_layered_candidates(problem, incumbent, config) == {}


def _dot_order_structure(**overrides: object) -> SimpleNamespace:
    """Return classifier metadata representative of dot-order target DAGs.

    Parameters
    ----------
    overrides : object
        Structural fields that should override the default target metadata.

    Returns
    -------
    SimpleNamespace
        Structural object consumed by the dot-order gate.
    """
    values = {
        "is_directed_acyclic": True,
        "is_acyclic": True,
        "is_semantically_directed": True,
        "topology_tags": (),
        "num_layers_effective": 3,
        "num_layers": 3,
        "edge_to_node_ratio": 1.0,
        "hub_edge_fraction": 0.2,
        "diameter_estimate": 3,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _skip_edge_dot_order_problem() -> LayoutProblem:
    """Return a small DAG where expanded mincross can improve chord crossings.

    Returns
    -------
    LayoutProblem
        Directed acyclic graph with rank-span-two crossing skip edges.
    """
    edge_index = torch.tensor(
        [
            [0, 1, 0, 1, 2, 3],
            [5, 4, 2, 3, 5, 4],
        ],
        dtype=torch.long,
    )
    return LayoutProblem(
        edge_index=edge_index,
        num_nodes=6,
        node_sizes=torch.full((6, 2), 10.0),
        structure=_dot_order_structure(),
    )


def test_dot_order_gate_structural_and_off_class_noop() -> None:
    """Default dot-order calls only mark the fired telemetry false."""
    problem = LayoutProblem(
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        num_nodes=3,
        node_sizes=torch.full((3, 2), 10.0),
        structure=_dot_order_structure(),
    )
    config = LayoutConfig()
    incumbent = torch.arange(6, dtype=torch.float32).reshape(3, 2)

    candidates = _directed_dot_order_candidates(problem, incumbent, config)

    assert candidates == {}
    assert getattr(config, "_dagua_native_dot_order_fired") is False
    assert torch.equal(incumbent, torch.arange(6, dtype=torch.float32).reshape(3, 2))
    assert not hasattr(config, "_dagua_native_dot_order_telemetry")


def test_dot_order_default_gate_is_disabled_for_target_shape() -> None:
    """Target-shaped dot-order graphs still cannot build default candidates."""
    problem = _skip_edge_dot_order_problem()
    incumbent = torch.tensor(
        [
            [-20.0, 0.0],
            [20.0, 0.0],
            [-20.0, 40.0],
            [20.0, 40.0],
            [-20.0, 80.0],
            [20.0, 80.0],
        ],
        dtype=torch.float32,
    )
    config = LayoutConfig()

    candidates = _directed_dot_order_candidates(problem, incumbent, config)

    assert not _directed_dot_order_enabled(problem)
    assert candidates == {}
    assert getattr(config, "_dagua_native_dot_order_fired") is False


def test_dot_order_direct_builder_remains_byte_deterministic() -> None:
    """Two direct dot-order builds return byte-identical candidate tensors."""
    problem = _skip_edge_dot_order_problem()
    incumbent = torch.tensor(
        [
            [-20.0, 0.0],
            [20.0, 0.0],
            [-20.0, 40.0],
            [20.0, 40.0],
            [-20.0, 80.0],
            [20.0, 80.0],
        ],
        dtype=torch.float32,
    )
    spec = _DotOrderSpec(
        name="dot_ns_dotx",
        layering="network_simplex_tightened",
        xcoord="dot_lp",
        warm_start=False,
    )

    first, first_expanded_n = _build_dot_order_candidate(spec, problem, incumbent, LayoutConfig())
    second, second_expanded_n = _build_dot_order_candidate(spec, problem, incumbent, LayoutConfig())

    assert first is not None
    assert second is not None
    assert first_expanded_n == second_expanded_n
    assert torch.equal(first, second)


def test_dot_order_candidate_places_all_real_nodes_with_rank_consistent_y() -> None:
    """The expanded dot-order candidate maps every real node back to finite coordinates."""
    problem = _skip_edge_dot_order_problem()
    incumbent = torch.zeros((6, 2), dtype=torch.float32)
    candidate, _expanded_n = _build_dot_order_candidate(
        _DotOrderSpec(
            name="dot_ns_dotx",
            layering="network_simplex_tightened",
            xcoord="dot_lp",
            warm_start=False,
        ),
        problem,
        incumbent,
        LayoutConfig(),
    )

    assert candidate is not None
    assert candidate.shape == (6, 2)
    assert torch.isfinite(candidate).all()
    for src, dst in problem.edge_index.t().tolist():
        assert float(candidate[int(dst), 1].item()) > float(candidate[int(src), 1].item())


def test_dot_ns_dotx_constructive_win_on_skip_edge_dag() -> None:
    """The dot mincross arm reduces exact chord crossings on a skip-edge DAG."""
    problem = _skip_edge_dot_order_problem()
    incumbent = torch.tensor(
        [
            [-20.0, 0.0],
            [20.0, 0.0],
            [-20.0, 40.0],
            [20.0, 40.0],
            [-20.0, 80.0],
            [20.0, 80.0],
        ],
        dtype=torch.float32,
    )
    candidate, expanded_n = _build_dot_order_candidate(
        _DotOrderSpec(
            name="dot_ns_dotx",
            layering="network_simplex_tightened",
            xcoord="dot_lp",
            warm_start=False,
        ),
        problem,
        incumbent,
        LayoutConfig(),
    )

    assert candidate is not None
    assert expanded_n <= 8 * int(problem.num_nodes)
    assert _exact_crossing_count(candidate, problem.edge_index) < _exact_crossing_count(
        incumbent,
        problem.edge_index,
    )


def test_expanded_virtual_chain_crossings_match_original_chords() -> None:
    """Expanded adjacent-rank chains preserve the chord crossing count."""
    rank_values = [0, 0, 2, 2]
    edge_index = torch.tensor([[0, 1], [3, 2]], dtype=torch.long)
    expanded_ranks, expanded_edges, _virtual_ids, _penalties = _expanded_layered_graph(
        rank_values,
        edge_index,
        None,
    )
    chord_pos = torch.tensor(
        [[-10.0, 0.0], [10.0, 0.0], [-10.0, 20.0], [10.0, 20.0]],
        dtype=torch.float32,
    )
    expanded_pos = torch.zeros((len(expanded_ranks), 2), dtype=torch.float32)
    expanded_pos[:4] = chord_pos
    expanded_pos[4] = torch.tensor([-10.0, 10.0])
    expanded_pos[5] = torch.tensor([10.0, 10.0])

    assert _exact_crossing_count(expanded_pos, expanded_edges) == _exact_crossing_count(
        chord_pos,
        edge_index,
    )


def test_recombinant_bk_uses_span_two_virtual_chain_edges() -> None:
    """A rank-span-two edge influences BK after virtual-chain expansion."""
    problem = LayoutProblem(
        edge_index=torch.tensor([[0], [2]], dtype=torch.long),
        num_nodes=3,
        node_sizes=torch.full((3, 2), 10.0),
    )
    ordered_layers = [[0], [1], [2]]
    spec = type(
        "Spec",
        (),
        {"xcoord": "brandes_koepf"},
    )()

    x_values = _assign_recombinant_x_coordinates(spec, ordered_layers, problem, node_sep=10.0)

    assert x_values is not None
    assert abs(float(x_values[0].item()) - float(x_values[2].item())) < 1.0e-5


@pytest.mark.parametrize("xcoord", ["dot_lp", "brandes_koepf"])
def test_recombinant_x_assignment_preserves_given_real_layer_order(
    xcoord: str,
) -> None:
    """Span-one layers keep the ordering stage order during x assignment."""
    problem = LayoutProblem(
        edge_index=torch.tensor([[2, 0, 1], [5, 3, 4]], dtype=torch.long),
        num_nodes=6,
        node_sizes=torch.full((6, 2), 10.0),
    )
    ordered_layers = [[2, 0, 1], [5, 3, 4]]
    spec = type("Spec", (), {"xcoord": xcoord})()

    x_values = _assign_recombinant_x_coordinates(spec, ordered_layers, problem, node_sep=10.0)

    assert x_values is not None
    assert float(x_values[2].item()) < float(x_values[0].item()) < float(x_values[1].item())
    assert float(x_values[5].item()) < float(x_values[3].item()) < float(x_values[4].item())


def test_recombinant_dot_lp_expansion_preserves_real_edge_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expanded recombinant dot-x edges inherit original edge weights."""
    dagua_native = importlib.import_module("dagua.layout.ops.pipelines.dagua_native")
    captured: dict[str, torch.Tensor] = {}

    def fake_dot_x(
        rank_ordering: list[list[int]],
        node_widths: torch.Tensor,
        edge_index: torch.Tensor,
        node_sep: float = 18.0,
        edge_weights: Optional[torch.Tensor] = None,
        center: bool = True,
    ) -> torch.Tensor:
        """Capture expanded weights and return monotone coordinates."""
        del rank_ordering, edge_index, node_sep, center
        assert edge_weights is not None
        captured["edge_weights"] = edge_weights.detach().clone()
        return torch.arange(int(node_widths.numel()), dtype=torch.float32)

    monkeypatch.setattr(dagua_native, "_graphviz_dot_x_position_network_simplex", fake_dot_x)
    problem = LayoutProblem(
        edge_index=torch.tensor([[0], [2]], dtype=torch.long),
        edge_weights=torch.tensor([7.0], dtype=torch.float32),
        num_nodes=3,
        node_sizes=torch.full((3, 2), 10.0),
    )
    spec = type("Spec", (), {"xcoord": "dot_lp"})()

    x_values = _assign_recombinant_x_coordinates(spec, [[0], [1], [2]], problem, node_sep=10.0)

    assert x_values is not None
    assert torch.equal(captured["edge_weights"], torch.tensor([7.0, 7.0], dtype=torch.float32))


def test_recombinant_ns_ranker_falls_back_before_oversized_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Oversized Lever-2 rows avoid the network-simplex ranker entirely."""
    elk = importlib.import_module("dagua.layout.ops.elk")

    def fail_network_simplex(*args: object, **kwargs: object) -> list[int]:
        """Fail if the oversized row still enters the network-simplex ranker."""
        del args, kwargs
        raise AssertionError("network simplex should be capped before ranking")

    monkeypatch.setattr(elk, "_network_simplex_layers", fail_network_simplex)
    chain_edges = [(node, node + 1) for node in range(19)]
    skip_edges = [(node, 19) for node in range(11)]
    edge_index = torch.tensor(chain_edges + skip_edges, dtype=torch.long).t().contiguous()
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=20,
        node_sizes=torch.full((20, 2), 10.0),
    )
    spec = type("Spec", (), {"layering": "network_simplex_tightened"})()

    ranks = _recombinant_rank_values(spec, problem, torch.zeros((20, 2), dtype=torch.float32))

    assert ranks == list(range(20))


def test_recombinant_dot_lp_uses_raw_x_assignment_above_expansion_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Oversized Lever-2 expansions fall back to raw-graph dot-x assignment."""
    dagua_native = importlib.import_module("dagua.layout.ops.pipelines.dagua_native")
    captured: dict[str, object] = {}

    def fake_dot_x(
        rank_ordering: list[list[int]],
        node_widths: torch.Tensor,
        edge_index: torch.Tensor,
        node_sep: float = 18.0,
        edge_weights: Optional[torch.Tensor] = None,
        center: bool = True,
    ) -> torch.Tensor:
        """Capture raw fallback payload and return monotone coordinates."""
        del node_sep, center
        captured["rank_ordering"] = [list(layer) for layer in rank_ordering]
        captured["node_width_count"] = int(node_widths.numel())
        captured["edge_index"] = edge_index.detach().clone()
        captured["edge_weights"] = None if edge_weights is None else edge_weights.detach().clone()
        return torch.arange(int(node_widths.numel()), dtype=torch.float32)

    monkeypatch.setattr(dagua_native, "_graphviz_dot_x_position_network_simplex", fake_dot_x)
    sources = list(range(5))
    targets = list(range(5, 10))
    edges = [(src, dst) for src in sources for dst in targets]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weights = torch.arange(1, len(edges) + 1, dtype=torch.float32)
    problem = LayoutProblem(
        edge_index=edge_index,
        edge_weights=edge_weights,
        num_nodes=10,
        node_sizes=torch.full((10, 2), 10.0),
    )
    ordered_layers = [sources] + [[] for _ in range(8)] + [targets]
    spec = type("Spec", (), {"xcoord": "dot_lp"})()

    x_values = _assign_recombinant_x_coordinates(spec, ordered_layers, problem, node_sep=10.0)

    assert x_values is not None
    assert not _lever2_expanded_x_assignment_enabled(210, 10)
    assert captured["rank_ordering"] == [sources, targets]
    assert captured["node_width_count"] == 10
    assert torch.equal(captured["edge_index"], edge_index)
    assert torch.equal(captured["edge_weights"], edge_weights)


def test_directed_portfolio_dot_order_is_not_registered_for_clusters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clustered directed portfolios must not admit dot-order candidates."""
    dagua_native = importlib.import_module("dagua.layout.ops.pipelines.dagua_native")
    native_directed = importlib.import_module("dagua.layout.ops.pipelines.native_directed")
    incumbent = torch.zeros((4, 2), dtype=torch.float32)
    calls: list[str] = []

    def fake_native_problem(*args: object, **kwargs: object) -> torch.Tensor:
        """Return a finite incumbent quickly."""
        del args, kwargs
        return incumbent.clone()

    def fake_register_dot(*args: object, **kwargs: object) -> object:
        """Record any forbidden dot registration."""
        del args, kwargs
        calls.append("dot")
        return None

    monkeypatch.setattr(dagua_native, "_run_native_problem", fake_native_problem)
    monkeypatch.setattr(native_directed, "_directed_dot_order_enabled", lambda *args: True)
    monkeypatch.setattr(native_directed, "_register_dot_order_candidates", fake_register_dot)
    monkeypatch.setattr(native_directed, "_directed_pivot_mds_candidates", lambda *args: {})
    monkeypatch.setattr(native_directed, "_directed_stress_blend_candidates", lambda *args: {})
    monkeypatch.setattr(native_directed, "_directed_mrtree_enabled", lambda *args: False)
    monkeypatch.setattr(native_directed, "_force_challengers_enabled", lambda *args: False)
    monkeypatch.setattr(
        native_directed,
        "_directed_recombinant_layered_enabled",
        lambda *args: False,
    )
    monkeypatch.setattr(native_directed, "_directed_wide_dag_ordering_enabled", lambda *args: False)
    monkeypatch.setattr(native_directed, "_ordering_cost_admissible", lambda *args, **kwargs: False)
    problem = LayoutProblem(
        edge_index=torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
        num_nodes=4,
        node_sizes=torch.full((4, 2), 2.0),
        clusters={"root": [0, 1, 2, 3], "child": [0, 1]},
        cluster_parents={"root": None, "child": "root"},
    )
    config = LayoutConfig()

    layout_native_directed_portfolio(problem, SolveState(), RuntimeContext(), config)

    assert calls == []
    assert not hasattr(config, "_dagua_native_dot_order_fired")


def test_challenger_registration_includes_guarded_raw_variant() -> None:
    """Parity candidates expose raw positions alongside cleanup variants."""
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    raw = torch.tensor([[0.0, 0.0], [40.0, 30.0], [80.0, 0.0], [120.0, 30.0]])
    problem = LayoutProblem(edge_index=edge_index, num_nodes=4, node_sizes=None)
    positions: dict[str, torch.Tensor] = {}

    _register_challenger_variants("dot", raw, problem, LayoutConfig(), positions)

    assert torch.equal(positions["dot_raw"], raw)
    assert {"dot_raw", "dot", "dot_convergent"} <= positions.keys()


def test_projected_rank_order_restores_dot_tie_without_losing_separation() -> None:
    """The dot-x projector should retain mincross order and separated x values."""
    raw = torch.tensor([[10.0, 0.0], [-10.0, 0.0], [0.0, 50.0]])
    projected = torch.tensor([[-30.0, 0.0], [30.0, 0.0], [0.0, 50.0]])

    restored = _restore_projected_rank_order(raw, projected)

    assert restored[:, 0].tolist() == [30.0, -30.0, 0.0]
    assert abs(float(restored[0, 0] - restored[1, 0])) == 60.0


def test_directed_scorer_sets_declared_hierarchy(monkeypatch: object) -> None:
    """The referee receives the same directed-table gate as the benchmark."""
    captured: dict[str, object] = {}

    def fake_full(*args: object, **kwargs: object) -> dict[str, float]:
        """Return a minimal numeric metric payload."""
        return {"node_occlusion": 1.0}

    def fake_composite(metrics: dict[str, float], is_semantically_directed: bool) -> float:
        """Capture directed routing inputs and return a stable score."""
        captured["metrics"] = metrics
        captured["directed"] = is_semantically_directed
        return 7.0

    monkeypatch.setattr("dagua.metrics.full", fake_full)  # type: ignore[attr-defined]
    monkeypatch.setattr("dagua.metrics.composite_auto", fake_composite)  # type: ignore[attr-defined]
    problem = LayoutProblem(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=1)

    score = _score_directed_candidate(torch.zeros((1, 2)), problem, None)

    assert score == 7.0
    assert captured["directed"] is True
    assert captured["metrics"] == {"node_occlusion": 1.0, "declared_hierarchical": True}


def test_directed_portfolio_is_incumbent_monotone() -> None:
    """The selected directed winner must not score below the incumbent."""
    from dagua.layout import layout

    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]
    graph = DaguaGraph.from_edge_list(edges, num_nodes=5)
    graph.compute_node_sizes()
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

    incumbent_score = _score_directed_candidate(incumbent_pos, problem, None)
    winner_score = _score_directed_candidate(winner_pos, problem, None)

    assert winner_score >= incumbent_score


def test_directed_narrow_seed_candidates_are_finite() -> None:
    """W3 narrow directed seeds produce finite non-degenerate layouts."""
    edge_index = torch.tensor(
        [[0, 0, 1, 2, 3, 4, 2, 5, 6, 7, 1], [1, 2, 3, 3, 4, 6, 5, 7, 7, 8, 8]],
        dtype=torch.long,
    )
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=9,
        node_sizes=torch.full((9, 2), 30.0),
    )
    incumbent = torch.stack(
        [torch.arange(9, dtype=torch.float32) * 40.0, torch.arange(9, dtype=torch.float32) * 8.0],
        dim=1,
    )

    pivot_candidates = _directed_pivot_mds_candidates(problem, incumbent, node_sep=30.0, seed=42)
    stress_candidates = _directed_stress_blend_candidates(problem, incumbent, seed=42)

    assert {"pivot_mds", "pivot_mds_rot90", "pivot_mds_flow_blend"} <= set(pivot_candidates)
    assert {"stress_blend_0.2", "stress_blend_0.4"} == set(stress_candidates)
    for candidate in [*pivot_candidates.values(), *stress_candidates.values()]:
        assert candidate.shape == (9, 2)
        assert bool(torch.isfinite(candidate).all().item())
        extent = candidate.max(dim=0).values - candidate.min(dim=0).values
        assert float(extent.max().item()) > 0.0


def test_directed_mrtree_and_rank_swap_targets_are_structurally_gated() -> None:
    """W3 MrTree and rank-local swap candidates cover long-skip DAGs."""
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 8],
        ],
        dtype=torch.long,
    )
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=9,
        node_sizes=torch.full((9, 2), 20.0),
    )
    incumbent = torch.tensor(
        [
            [0.0, 0.0],
            [80.0, 40.0],
            [20.0, 80.0],
            [60.0, 120.0],
            [10.0, 160.0],
            [90.0, 200.0],
            [40.0, 240.0],
            [70.0, 280.0],
            [30.0, 320.0],
        ]
    )

    swapped = _rank_local_zero_crossing_swap_candidate(incumbent, edge_index)

    assert _directed_mrtree_enabled(problem)
    assert swapped.shape == incumbent.shape
    assert bool(torch.isfinite(swapped).all().item())


def test_exact_crossing_count_vectorized_matches_loop() -> None:
    """The vectorized crossing count matches the old strict-crossing loop."""
    pos = torch.tensor(
        [
            [0.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
            [10.0, 0.0],
            [5.0, 12.0],
            [12.0, 5.0],
        ],
        dtype=torch.float32,
    )
    edge_index = torch.tensor(
        [
            [0, 2, 4, 0, 1],
            [1, 3, 5, 2, 3],
        ],
        dtype=torch.long,
    )

    assert _exact_crossing_count(pos, edge_index) == _exact_crossing_count_loop(pos, edge_index)
    assert _exact_crossing_count(pos, edge_index) == 3


def test_rank_swap_respects_exhausted_deadline() -> None:
    """The rank-local swap arm exits before trials when no budget remains."""
    edge_index = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
    incumbent = torch.tensor(
        [
            [0.0, 0.0],
            [10.0, 10.0],
            [10.0, 0.0],
            [0.0, 10.0],
        ],
        dtype=torch.float32,
    )
    config = LayoutConfig()
    config._dagua_native_deadline_s = time.perf_counter() - 1.0

    swapped = _rank_local_zero_crossing_swap_candidate(incumbent, edge_index, config=config)

    assert torch.equal(swapped, incumbent)


def test_rank_ordering_exhaustive_finds_tiny_optimum() -> None:
    """Width-two exhaustive rank ordering reaches the zero-crossing optimum."""
    edge_index = torch.tensor([[0, 1], [3, 2]], dtype=torch.long)
    incumbent = torch.tensor(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
        ],
        dtype=torch.float32,
    )

    ordered = _rank_local_zero_crossing_swap_candidate(incumbent, edge_index)

    assert _exact_crossing_count(incumbent, edge_index) == 1
    assert _exact_crossing_count(ordered, edge_index) == 0


def test_rank_ordering_uses_drawn_y_layers_not_longest_path_ranks() -> None:
    """The ordering arm permutes incumbent y-layers when graph ranks disagree."""
    edge_index = torch.tensor([[0, 1, 0, 4], [3, 2, 4, 2]], dtype=torch.long)
    incumbent = torch.tensor(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
            [5.0, 20.0],
        ],
        dtype=torch.float32,
    )

    drawn_layers = _rank_to_nodes_from_incumbent_y(incumbent, edge_index, 5)
    ordered = _rank_local_zero_crossing_swap_candidate(incumbent, edge_index)

    assert sorted(drawn_layers[1]) == [2, 3]
    assert _exact_crossing_count(ordered, edge_index) < _exact_crossing_count(
        incumbent,
        edge_index,
    )


def test_rank_ordering_non_adjacent_reinsert_reduces_crossings() -> None:
    """The small-graph ordering pass accepts only fewer-crossing layouts."""
    edge_index = torch.tensor([[0, 1, 2], [5, 4, 3]], dtype=torch.long)
    incumbent = torch.tensor(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
            [20.0, 10.0],
        ],
        dtype=torch.float32,
    )

    ordered = _rank_local_zero_crossing_swap_candidate(incumbent, edge_index)

    assert _exact_crossing_count(ordered, edge_index) < _exact_crossing_count(
        incumbent,
        edge_index,
    )


def test_rank_ordering_noop_when_crossings_cannot_improve() -> None:
    """A rank ordering with no crossing improvement returns byte-identical positions."""
    edge_index = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    incumbent = torch.tensor(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
        ],
        dtype=torch.float32,
    )

    ordered = _rank_local_zero_crossing_swap_candidate(incumbent, edge_index)

    assert torch.equal(ordered, incumbent)


def test_rank_ordering_library_mode_wall_clock_cap() -> None:
    """Width-eight ranks return promptly without benchmark deadline metadata."""
    sources = []
    targets = []
    for src in range(8):
        for dst in range(8, 16):
            sources.append(src)
            targets.append(dst)
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    x_values = torch.arange(8, dtype=torch.float32) * 10.0
    incumbent = torch.cat(
        [
            torch.stack([x_values, torch.zeros(8)], dim=1),
            torch.stack([torch.flip(x_values, dims=(0,)), torch.full((8,), 10.0)], dim=1),
            torch.stack([x_values, torch.full((8,), 20.0)], dim=1),
        ],
        dim=0,
    )

    started = time.perf_counter()
    ordered = _rank_local_zero_crossing_swap_candidate(incumbent, edge_index, config=None)
    elapsed_s = time.perf_counter() - started

    assert ordered.shape == incumbent.shape
    assert elapsed_s < 3.0


def test_ordering_cost_gate_blocks_dense_medium_graph() -> None:
    """Medium DAGs with too many edge pairs are not admitted to ordering."""
    rank_to_nodes = {0: list(range(65)), 1: list(range(65, 130))}

    assert not _ordering_cost_admissible(
        num_nodes=130,
        edge_count=900,
        rank_to_nodes=rank_to_nodes,
        max_passes=3,
    )


def test_ordering_cost_gate_excludes_nudges_from_trial_pair_product() -> None:
    """Medium cost gating estimates permutation/search work, not nudge trials."""
    rank_to_nodes = {rank: [rank] for rank in range(130)}
    rank_to_nodes[0] = [0, 1]

    assert _ordering_cost_admissible(
        num_nodes=130,
        edge_count=199,
        rank_to_nodes=rank_to_nodes,
        max_passes=3,
    )
    assert not _ordering_cost_admissible(
        num_nodes=130,
        edge_count=700,
        rank_to_nodes={0: list(range(65)), 1: list(range(65, 130))},
        max_passes=3,
    )


def test_ordering_pair_sweep_checks_budget_internally(monkeypatch: object) -> None:
    """Crossing pair collection exits during large scans when budget expires."""
    native_directed = importlib.import_module("dagua.layout.ops.pipelines.native_directed")
    edge_count = 100
    sources = torch.arange(0, edge_count * 2, 2, dtype=torch.long)
    targets = sources + 1
    edge_index = torch.stack([sources, targets])
    x_values = torch.arange(edge_count * 2, dtype=torch.float32)
    pos = torch.stack([x_values, torch.zeros_like(x_values)], dim=1)
    calls = 0

    def fake_segments_cross(*args: object, **kwargs: object) -> bool:
        """Count segment tests and report no crossings."""
        nonlocal calls
        del args, kwargs
        calls += 1
        return False

    config = LayoutConfig()
    config._dagua_native_deadline_s = time.perf_counter() - 1.0
    monkeypatch.setattr(native_directed, "_segments_cross", fake_segments_cross)

    crossings = _crossing_edge_pairs(
        pos,
        edge_index,
        max_pairs=64,
        config=config,
        started_at=time.perf_counter(),
        wall_time_cap_s=10.0,
    )

    assert crossings == []
    assert calls < edge_count * (edge_count - 1) // 2


def test_directed_ordering_reachable_for_medium_small_band_once(monkeypatch: object) -> None:
    """A 65..128 node portfolio can reach ordering without a duplicate late pass."""
    dagua_native = importlib.import_module("dagua.layout.ops.pipelines.dagua_native")
    native_directed = importlib.import_module("dagua.layout.ops.pipelines.native_directed")
    sugiyama = importlib.import_module("dagua.layout.ops.pipelines.sugiyama")
    num_nodes = 100
    sources: list[int] = []
    targets: list[int] = []
    for src in range(num_nodes):
        for delta in range(1, 4):
            dst = src + delta
            if dst < num_nodes and len(sources) < 285:
                sources.append(src)
                targets.append(dst)
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    y_values = torch.arange(num_nodes, dtype=torch.float32) // 8
    incumbent = torch.stack([torch.arange(num_nodes, dtype=torch.float32), y_values], dim=1)
    ordering_calls = 0
    ordering_passes: list[int] = []

    def fake_native_problem(*args: object, **kwargs: object) -> torch.Tensor:
        """Return a finite incumbent with repeated drawn ranks."""
        del args, kwargs
        return incumbent.clone()

    def fake_sugiyama(**kwargs: object) -> torch.Tensor:
        """Return incumbent-identical candidates without external solver cost."""
        del kwargs
        return incumbent.clone()

    def fake_register(
        name: str,
        raw_pos: torch.Tensor,
        problem: LayoutProblem,
        config: LayoutConfig,
        positions: dict[str, torch.Tensor],
        preserve_rank_order: bool = False,
        arm_timings: Optional[dict[str, tuple[float, float]]] = None,
        timing_span: Optional[tuple[float, float]] = None,
    ) -> None:
        """Register cheap incumbent-identical challengers."""
        del problem, config, preserve_rank_order, arm_timings, timing_span
        positions[name] = raw_pos

    def fake_rank_ordering(
        incumbent_pos: torch.Tensor,
        edge_index_arg: torch.Tensor,
        max_passes: int = 3,
        config: Optional[LayoutConfig] = None,
    ) -> torch.Tensor:
        """Record that the portfolio reached the ordering arm."""
        nonlocal ordering_calls
        del edge_index_arg, config
        ordering_calls += 1
        ordering_passes.append(max_passes)
        assert int(incumbent_pos.shape[0]) == num_nodes
        return incumbent_pos.clone()

    def fake_score_payload(
        pos: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> tuple[float, _DirectedClusterScoreTelemetry]:
        """Keep the incumbent ahead at the V3 payload scorer seam."""
        del args, kwargs
        score = 2.0 if torch.equal(pos, incumbent) else 1.0
        return (
            score,
            _DirectedClusterScoreTelemetry(
                extended_score=score,
                old_score=score,
                metrics={},
                v3_tiered=score,
            ),
        )

    monkeypatch.setattr(dagua_native, "_run_native_problem", fake_native_problem)
    monkeypatch.setattr(sugiyama, "layout_sugiyama_pipeline", fake_sugiyama)
    monkeypatch.setattr(native_directed, "_register_challenger_variants", fake_register)
    monkeypatch.setattr(native_directed, "_directed_pivot_mds_candidates", lambda *args: {})
    monkeypatch.setattr(native_directed, "_directed_stress_blend_candidates", lambda *args: {})
    monkeypatch.setattr(native_directed, "_directed_mrtree_enabled", lambda *args: False)
    monkeypatch.setattr(native_directed, "_force_challengers_enabled", lambda *args: False)
    monkeypatch.setattr(
        native_directed,
        "_rank_local_zero_crossing_swap_candidate",
        fake_rank_ordering,
    )
    monkeypatch.setattr(
        native_directed,
        "_score_directed_candidate_referee_payload",
        fake_score_payload,
    )
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=torch.full((num_nodes, 2), 60.0),
    )

    returned = layout_native_directed_portfolio(
        problem,
        SolveState(),
        RuntimeContext(),
        LayoutConfig(),
    )

    assert torch.equal(returned, incumbent)
    assert ordering_calls == 1
    assert ordering_passes == [0]


def test_directed_portfolio_rejects_crossing_win_that_dual_gate_rejects(
    monkeypatch: object,
) -> None:
    """A crossing-only ordering win cannot alter the portfolio output."""
    from dagua.layout.ops.pipelines.native_finisher import W5ScorePair

    dagua_native = importlib.import_module("dagua.layout.ops.pipelines.dagua_native")
    native_directed = importlib.import_module("dagua.layout.ops.pipelines.native_directed")
    sugiyama = importlib.import_module("dagua.layout.ops.pipelines.sugiyama")
    incumbent = torch.tensor(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
        ],
        dtype=torch.float32,
    )
    edge_index = torch.tensor([[0, 1], [3, 2]], dtype=torch.long)
    captured: dict[str, torch.Tensor] = {}

    def fake_native_problem(
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
        config: LayoutConfig,
    ) -> torch.Tensor:
        """Return the incumbent that has one fixable crossing."""
        del problem, state, ctx, config
        return incumbent.clone()

    def fake_score(*args: object, **kwargs: object) -> float:
        """Keep all non-ordering candidates tied with the incumbent."""
        del args, kwargs
        return 10.0

    def fake_dual_gate(
        candidate: torch.Tensor,
        incumbent_pair: W5ScorePair,
        problem: LayoutProblem,
        cluster_ids: Optional[torch.Tensor],
        all_pairs_dist: Optional[object],
    ) -> tuple[bool, W5ScorePair, tuple[int, float]]:
        """Reject the crossing-improving candidate under the frozen dual gate."""
        del incumbent_pair, problem, cluster_ids, all_pairs_dist
        captured["candidate"] = candidate
        return False, W5ScorePair(directed=11.0, undirected=9.0), (1, -0.0)

    def fake_sugiyama(**kwargs: object) -> torch.Tensor:
        """Return a tied non-ordering challenger without external solver cost."""
        del kwargs
        return incumbent.clone()

    def fake_register(
        name: str,
        raw_pos: torch.Tensor,
        problem: LayoutProblem,
        config: LayoutConfig,
        positions: dict[str, torch.Tensor],
        preserve_rank_order: bool = False,
        arm_timings: Optional[dict[str, tuple[float, float]]] = None,
        timing_span: Optional[tuple[float, float]] = None,
    ) -> None:
        """Register only incumbent-identical challengers."""
        del problem, config, preserve_rank_order, arm_timings, timing_span
        positions[name] = raw_pos

    monkeypatch.setattr(dagua_native, "_run_native_problem", fake_native_problem)
    monkeypatch.setattr(native_directed, "_directed_pivot_mds_candidates", lambda *args: {})
    monkeypatch.setattr(native_directed, "_directed_stress_blend_candidates", lambda *args: {})
    monkeypatch.setattr(native_directed, "_directed_mrtree_enabled", lambda *args: False)
    monkeypatch.setattr(native_directed, "_force_challengers_enabled", lambda *args: False)
    monkeypatch.setattr(native_directed, "_score_directed_candidate", fake_score)
    monkeypatch.setattr(native_directed, "_score_directed_candidate_pair", fake_score)
    monkeypatch.setattr(
        native_directed,
        "_directed_ordering_candidate_dual_dominates",
        fake_dual_gate,
    )
    monkeypatch.setattr(native_directed, "_register_challenger_variants", fake_register)
    monkeypatch.setattr(sugiyama, "layout_sugiyama_pipeline", fake_sugiyama)
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=4,
        node_sizes=torch.full((4, 2), 2.0),
    )

    returned = layout_native_directed_portfolio(
        problem,
        SolveState(),
        RuntimeContext(),
        LayoutConfig(),
    )

    assert _exact_crossing_count(captured["candidate"], edge_index) == 0
    assert torch.equal(returned, incumbent)


def test_directed_w5_incumbent_uses_same_payload_pair_and_axes(monkeypatch: object) -> None:
    """The directed W5 incumbent route passes pair and axes from one payload."""
    from dagua.layout.ops.pipelines.native_finisher import (
        W5FinisherResult,
        W5HonestAxes,
        W5ScorePair,
        make_w5_skip_result,
    )

    dagua_native = importlib.import_module("dagua.layout.ops.pipelines.dagua_native")
    native_directed = importlib.import_module("dagua.layout.ops.pipelines.native_directed")
    native_finisher = importlib.import_module("dagua.layout.ops.pipelines.native_finisher")
    sugiyama = importlib.import_module("dagua.layout.ops.pipelines.sugiyama")
    incumbent = torch.tensor(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
        ],
        dtype=torch.float32,
    )
    ordering_seed = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 10.0],
            [10.0, 0.0],
            [10.0, 10.0],
        ],
        dtype=torch.float32,
    )
    edge_index = torch.tensor([[0, 1], [3, 2]], dtype=torch.long)
    stale_pair = W5ScorePair(directed=10.0, undirected=10.0)
    payload_pair = W5ScorePair(directed=20.0, undirected=20.0)
    payload_axes = W5HonestAxes(flow=0.42, depth=0.84, ksm=0.9, edge_length=0.8)
    captured: dict[str, object] = {}

    def fake_native_problem(*args: object, **kwargs: object) -> torch.Tensor:
        """Return the incumbent for the directed contest."""
        del args, kwargs
        return incumbent.clone()

    def fake_score(*args: object, **kwargs: object) -> float:
        """Keep the scalar incumbent winner despite an ordering W5 seed."""
        del args, kwargs
        return 100.0

    def fake_pair(*args: object, **kwargs: object) -> W5ScorePair:
        """Return the older pair that must not be split from fresh axes."""
        del args, kwargs
        return stale_pair

    def fake_payload(*args: object, **kwargs: object) -> tuple[W5ScorePair, W5HonestAxes]:
        """Return the pair and axes that must travel together into W5."""
        del args, kwargs
        return payload_pair, payload_axes

    def fake_dual_gate(
        *args: object, **kwargs: object
    ) -> tuple[bool, W5ScorePair, tuple[int, float]]:
        """Admit the ordering seed while keeping scalar best_name incumbent."""
        del args, kwargs
        return True, W5ScorePair(directed=11.0, undirected=11.0), (1, -0.0)

    def fake_rank_swap(*args: object, **kwargs: object) -> torch.Tensor:
        """Return a distinct zero-crossing ordering seed."""
        del args, kwargs
        return ordering_seed.clone()

    def fake_crossing_count(pos: torch.Tensor, edges: torch.Tensor) -> int:
        """Report the ordering seed as crossing-improving."""
        del edges
        return 1 if torch.equal(pos, incumbent) else 0

    def fake_sugiyama(**kwargs: object) -> torch.Tensor:
        """Return an incumbent-identical challenger."""
        del kwargs
        return incumbent.clone()

    def fake_register(
        name: str,
        raw_pos: torch.Tensor,
        problem: LayoutProblem,
        config: LayoutConfig,
        positions: dict[str, torch.Tensor],
        preserve_rank_order: bool = False,
        arm_timings: Optional[dict[str, tuple[float, float]]] = None,
        timing_span: Optional[tuple[float, float]] = None,
    ) -> None:
        """Register only no-op challengers."""
        del name, problem, config, preserve_rank_order, arm_timings, timing_span
        positions["noop"] = raw_pos

    def fake_run_w5_finisher(
        *,
        incumbent_pos: torch.Tensor,
        incumbent_score_pair: W5ScorePair,
        seeds: object,
        edge_index: torch.Tensor,
        node_sizes: torch.Tensor,
        score_fn: object,
        is_semantically_directed: bool,
        declared_hierarchical: bool,
        direction_is_declared: bool = False,
        config: Optional[LayoutConfig] = None,
        accept_margin: float = 0.05,
        incumbent_axes: Optional[W5HonestAxes] = None,
        referee_key_fn: Optional[object] = None,
    ) -> W5FinisherResult:
        """Capture the W5 incumbent payload and return a no-op result."""
        del seeds, node_sizes, score_fn, accept_margin, referee_key_fn
        captured["pair"] = incumbent_score_pair
        captured["axes"] = incumbent_axes
        return make_w5_skip_result(
            incumbent_pos=incumbent_pos,
            incumbent_score_pair=incumbent_score_pair,
            reason="unit_noop",
            edge_index=edge_index,
            config=config,
            is_semantically_directed=is_semantically_directed,
            declared_hierarchical=declared_hierarchical,
            direction_is_declared=direction_is_declared,
        )

    monkeypatch.setattr(dagua_native, "_run_native_problem", fake_native_problem)
    monkeypatch.setattr(native_directed, "_directed_pivot_mds_candidates", lambda *args: {})
    monkeypatch.setattr(native_directed, "_directed_stress_blend_candidates", lambda *args: {})
    monkeypatch.setattr(native_directed, "_directed_mrtree_enabled", lambda *args: False)
    monkeypatch.setattr(native_directed, "_force_challengers_enabled", lambda *args: False)
    monkeypatch.setattr(native_directed, "_score_directed_candidate", fake_score)
    monkeypatch.setattr(native_directed, "_score_directed_candidate_pair", fake_pair)
    monkeypatch.setattr(native_directed, "_score_directed_candidate_payload", fake_payload)
    monkeypatch.setattr(
        native_directed,
        "_directed_ordering_candidate_dual_dominates",
        fake_dual_gate,
    )
    monkeypatch.setattr(native_directed, "_rank_local_zero_crossing_swap_candidate", fake_rank_swap)
    monkeypatch.setattr(native_directed, "_exact_crossing_count", fake_crossing_count)
    monkeypatch.setattr(native_directed, "_register_challenger_variants", fake_register)
    monkeypatch.setattr(native_finisher, "run_w5_finisher", fake_run_w5_finisher)
    monkeypatch.setattr(native_finisher, "log_w5_telemetry", lambda *args: None)
    monkeypatch.setattr(sugiyama, "layout_sugiyama_pipeline", fake_sugiyama)
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=4,
        node_sizes=torch.full((4, 2), 2.0),
    )

    layout_native_directed_portfolio(problem, SolveState(), RuntimeContext(), LayoutConfig())

    assert captured["pair"] == payload_pair
    assert captured["pair"] != stale_pair
    assert captured["axes"] == payload_axes

    deferred_config = LayoutConfig()
    deferred_config._dagua_native_defer_w5 = True
    captured.clear()

    layout_native_directed_portfolio(problem, SolveState(), RuntimeContext(), deferred_config)

    assert captured == {}


def test_directed_portfolio_rejects_recombinant_without_dual_dominance(
    monkeypatch: object,
) -> None:
    """A recombinant candidate that fails the dual gate cannot replace incumbent."""
    from dagua.layout.ops.pipelines.native_finisher import W5ScorePair

    dagua_native = importlib.import_module("dagua.layout.ops.pipelines.dagua_native")
    native_directed = importlib.import_module("dagua.layout.ops.pipelines.native_directed")
    sugiyama = importlib.import_module("dagua.layout.ops.pipelines.sugiyama")
    incumbent = torch.zeros((100, 2), dtype=torch.float32)
    challenger = torch.stack(
        [torch.arange(100, dtype=torch.float32), torch.arange(100, dtype=torch.float32)],
        dim=1,
    )
    captured: dict[str, torch.Tensor] = {}

    def fake_native_problem(*args: object, **kwargs: object) -> torch.Tensor:
        """Return the incumbent for the directed contest."""
        del args, kwargs
        return incumbent.clone()

    def fake_score(*args: object, **kwargs: object) -> float:
        """Keep non-recombinant candidates tied with the incumbent."""
        del args, kwargs
        return 10.0

    def fake_recombinant_candidates(*args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        """Return one recombinant candidate that must be dual-gated."""
        del args, kwargs
        return {"recomb_test": challenger.clone()}

    def fake_dual_gate(
        candidate: torch.Tensor,
        incumbent_pair: W5ScorePair,
        problem: LayoutProblem,
        cluster_ids: Optional[torch.Tensor],
        all_pairs_dist: Optional[object],
    ) -> tuple[bool, W5ScorePair, tuple[int, float]]:
        """Reject the recombinant candidate under the dual frozen rulers."""
        del incumbent_pair, problem, cluster_ids, all_pairs_dist
        captured["candidate"] = candidate
        return False, W5ScorePair(directed=11.0, undirected=9.0), (1, -0.0)

    def fake_sugiyama(**kwargs: object) -> torch.Tensor:
        """Return tied non-recombinant challengers cheaply."""
        del kwargs
        return incumbent.clone()

    def fake_register(
        name: str,
        raw_pos: torch.Tensor,
        problem: LayoutProblem,
        config: LayoutConfig,
        positions: dict[str, torch.Tensor],
        preserve_rank_order: bool = False,
        arm_timings: Optional[dict[str, tuple[float, float]]] = None,
        timing_span: Optional[tuple[float, float]] = None,
    ) -> None:
        """Register variants without projection cost."""
        del problem, config, preserve_rank_order, arm_timings, timing_span
        positions[name] = raw_pos

    monkeypatch.setattr(dagua_native, "_run_native_problem", fake_native_problem)
    monkeypatch.setattr(native_directed, "_directed_pivot_mds_candidates", lambda *args: {})
    monkeypatch.setattr(native_directed, "_directed_stress_blend_candidates", lambda *args: {})
    monkeypatch.setattr(native_directed, "_directed_mrtree_enabled", lambda *args: False)
    monkeypatch.setattr(native_directed, "_force_challengers_enabled", lambda *args: False)
    monkeypatch.setattr(native_directed, "_score_directed_candidate", fake_score)
    monkeypatch.setattr(native_directed, "_score_directed_candidate_pair", fake_score)
    monkeypatch.setattr(native_directed, "_register_challenger_variants", fake_register)
    monkeypatch.setattr(
        native_directed,
        "_directed_recombinant_layered_candidates",
        fake_recombinant_candidates,
    )
    monkeypatch.setattr(
        native_directed,
        "_directed_ordering_candidate_dual_dominates",
        fake_dual_gate,
    )
    monkeypatch.setattr(sugiyama, "layout_sugiyama_pipeline", fake_sugiyama)
    problem = LayoutProblem(
        edge_index=torch.tensor([[0, 0, 1], [1, 2, 3]], dtype=torch.long),
        num_nodes=100,
        node_sizes=torch.full((100, 2), 20.0),
        structure=_target_recombinant_structure(),
    )

    returned = layout_native_directed_portfolio(
        problem,
        SolveState(),
        RuntimeContext(),
        LayoutConfig(),
    )

    assert torch.equal(captured["candidate"], challenger)
    assert torch.equal(returned, incumbent)


def test_directed_portfolio_full_path_noop_keeps_incumbent(monkeypatch: object) -> None:
    """The complete portfolio path returns the incumbent when ordering is no-op."""
    dagua_native = importlib.import_module("dagua.layout.ops.pipelines.dagua_native")
    native_directed = importlib.import_module("dagua.layout.ops.pipelines.native_directed")
    sugiyama = importlib.import_module("dagua.layout.ops.pipelines.sugiyama")
    incumbent = torch.tensor(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
        ],
        dtype=torch.float32,
    )

    def fake_native_problem(*args: object, **kwargs: object) -> torch.Tensor:
        """Return a zero-crossing incumbent."""
        del args, kwargs
        return incumbent.clone()

    def fake_score(*args: object, **kwargs: object) -> float:
        """Keep all candidates tied so the incumbent wins ties."""
        del args, kwargs
        return 10.0

    def fake_sugiyama(**kwargs: object) -> torch.Tensor:
        """Return an incumbent-identical challenger."""
        del kwargs
        return incumbent.clone()

    def fake_register(
        name: str,
        raw_pos: torch.Tensor,
        problem: LayoutProblem,
        config: LayoutConfig,
        positions: dict[str, torch.Tensor],
        preserve_rank_order: bool = False,
        arm_timings: Optional[dict[str, tuple[float, float]]] = None,
        timing_span: Optional[tuple[float, float]] = None,
    ) -> None:
        """Register only no-op candidates."""
        del problem, config, preserve_rank_order, arm_timings, timing_span
        positions[name] = raw_pos

    monkeypatch.setattr(dagua_native, "_run_native_problem", fake_native_problem)
    monkeypatch.setattr(native_directed, "_directed_pivot_mds_candidates", lambda *args: {})
    monkeypatch.setattr(native_directed, "_directed_stress_blend_candidates", lambda *args: {})
    monkeypatch.setattr(native_directed, "_directed_mrtree_enabled", lambda *args: False)
    monkeypatch.setattr(native_directed, "_force_challengers_enabled", lambda *args: False)
    monkeypatch.setattr(native_directed, "_score_directed_candidate", fake_score)
    monkeypatch.setattr(native_directed, "_register_challenger_variants", fake_register)
    monkeypatch.setattr(sugiyama, "layout_sugiyama_pipeline", fake_sugiyama)
    problem = LayoutProblem(
        edge_index=torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
        num_nodes=4,
        node_sizes=torch.full((4, 2), 2.0),
    )

    returned = layout_native_directed_portfolio(
        problem,
        SolveState(),
        RuntimeContext(),
        LayoutConfig(),
    )

    assert torch.equal(returned, incumbent)


def test_directed_ordering_dual_gate_rejects_single_ruler_win(
    monkeypatch: object,
) -> None:
    """Ordering candidates must beat both frozen rulers before contest admission."""
    from dagua.layout.ops.pipelines.native_finisher import W5ScorePair

    native_directed = importlib.import_module("dagua.layout.ops.pipelines.native_directed")
    edge_index = torch.tensor([[0, 1], [3, 2]], dtype=torch.long)
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=4,
        node_sizes=torch.full((4, 2), 2.0),
    )
    incumbent_pair = W5ScorePair(directed=10.0, undirected=10.0)
    candidate = torch.tensor(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
        ],
        dtype=torch.float32,
    )

    def fake_pair(*args: object, **kwargs: object) -> W5ScorePair:
        """Return a directed-only improvement for the candidate."""
        del args, kwargs
        return W5ScorePair(directed=11.0, undirected=9.0)

    monkeypatch.setattr(native_directed, "_score_directed_candidate_pair", fake_pair)

    dominates, pair, _candidate_referee_key = _directed_ordering_candidate_dual_dominates(
        candidate,
        incumbent_pair,
        problem,
        None,
        None,
    )

    assert not dominates
    assert pair == W5ScorePair(directed=11.0, undirected=9.0)


def test_directed_ordering_dual_gate_demotes_referee_breacher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A severe-G6-breaching ordering candidate cannot replace a compliant winner."""
    from dagua.layout.ops.pipelines.native_finisher import W5ScorePair

    native_directed = importlib.import_module("dagua.layout.ops.pipelines.native_directed")
    edge_index = torch.tensor([[0, 1], [3, 2]], dtype=torch.long)
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=4,
        node_sizes=torch.full((4, 2), 2.0),
        edge_weights=torch.tensor([1.0, 3.0], dtype=torch.float32),
    )
    incumbent_pair = W5ScorePair(directed=10.0, undirected=10.0)
    candidate = torch.tensor(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
        ],
        dtype=torch.float32,
    )

    def fake_pair(*args: object, **kwargs: object) -> W5ScorePair:
        """Return a dual-ruler improvement that only the referee can reject."""
        del args, kwargs
        return W5ScorePair(directed=100.0, undirected=100.0)

    def fake_referee(
        pos: torch.Tensor,
        problem: LayoutProblem,
    ) -> tuple[tuple[int, float], bool, str]:
        """Mark the ordering candidate as the only severe-G6 breacher."""
        del problem
        if torch.equal(pos, candidate):
            return (0, -0.50), True, "severe_g6_breach"
        return (1, -0.0), False, "compliant"

    monkeypatch.setattr(native_directed, "_score_directed_candidate_pair", fake_pair)
    monkeypatch.setattr(native_directed, "_runtime_referee_telemetry", fake_referee)

    dominates, pair, candidate_referee_key = _directed_ordering_candidate_dual_dominates(
        candidate,
        incumbent_pair,
        problem,
        None,
        None,
        (1, -0.0),
    )

    assert not dominates
    assert pair == W5ScorePair(directed=100.0, undirected=100.0)
    assert candidate_referee_key == (0, -0.50)


def test_r79_weighted_skew_dag_severe_g6_candidate_rejected_by_referee_key() -> None:
    """A real r79 weighted-DAG candidate is rejected by the severe-G6 prefix."""
    test_graph = next(
        graph for graph in get_test_graphs() if graph.name == "r79_weighted_skew_dag_6x10"
    )
    graph = test_graph.graph
    problem = LayoutProblem(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        node_sizes=graph.compute_node_sizes(),
        edge_weights=graph.edge_weights,
        direction="directed",
    )
    candidate = torch.stack(
        (torch.arange(graph.num_nodes, dtype=torch.float32), torch.zeros(graph.num_nodes)),
        dim=1,
    )
    candidate_pair = _score_directed_candidate_pair(candidate, problem, None, None)
    incumbent_pair = W5ScorePair(
        directed=candidate_pair.directed - 1.0,
        undirected=candidate_pair.undirected - 1.0,
    )

    dominates, admitted_pair, candidate_referee_key = _directed_ordering_candidate_dual_dominates(
        candidate,
        incumbent_pair,
        problem,
        None,
        None,
        (1, -0.0),
    )
    telemetry_key, breached, reason = _runtime_referee_telemetry(candidate, problem)

    assert not dominates
    assert admitted_pair == candidate_pair
    assert candidate_referee_key == telemetry_key
    assert candidate_referee_key[0] == 0
    assert breached
    assert reason == "severe_g6_breach"
    assert candidate_pair.champion_ineligibility_flags == frozenset()
    assert w5_dominates(
        candidate_pair,
        incumbent_pair,
        candidate_referee_key=(1, -0.0),
        incumbent_referee_key=(1, -0.0),
        tallied_axis="directed",
    )


def test_directed_incumbent_config_is_not_deadline_weakened(monkeypatch: object) -> None:
    """A benchmark deadline must not alter the exact incumbent solve config."""
    captured: list[LayoutConfig] = []

    def fake_native_problem(
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
        config: LayoutConfig,
    ) -> torch.Tensor:
        """Capture the incumbent config and return finite positions."""
        del state, ctx
        captured.append(config)
        return torch.zeros((problem.num_nodes, 2), dtype=torch.float32)

    def fake_score(*args: object, **kwargs: object) -> float:
        """Return a tied score so the incumbent remains selected."""
        del args, kwargs
        return 1.0

    dagua_native = importlib.import_module("dagua.layout.ops.pipelines.dagua_native")
    native_directed = importlib.import_module("dagua.layout.ops.pipelines.native_directed")
    monkeypatch.setattr(dagua_native, "_run_native_problem", fake_native_problem)
    monkeypatch.setattr(native_directed, "_score_directed_candidate", fake_score)
    config = LayoutConfig(time_budget_s=123.0, multi_start_k=4)
    config._dagua_native_deadline_s = time.perf_counter() - 1.0
    problem = LayoutProblem(
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        num_nodes=2,
        node_sizes=torch.full((2, 2), 60.0),
    )

    layout_native_directed_portfolio(problem, SolveState(), RuntimeContext(), config)

    assert len(captured) == 1
    assert captured[0].time_budget_s == 123.0
    assert captured[0].multi_start_k == 4
    assert getattr(captured[0], "_dagua_native_suppress_portfolio") is True
    assert not hasattr(captured[0], "_dagua_native_polish_battery")
    assert not hasattr(captured[0], "_dagua_native_final_projection_iterations")


def test_directed_portfolio_adds_uniform_sugiyama_grid(monkeypatch: object) -> None:
    """Every directed graph receives the same mode and spacing grid."""
    calls: list[dict[str, object]] = []

    def fake_native_problem(*args: object, **kwargs: object) -> torch.Tensor:
        """Return a stable incumbent for the portfolio contest."""
        return torch.zeros((2, 2), dtype=torch.float32)

    def fake_sugiyama(**kwargs: object) -> torch.Tensor:
        """Capture Sugiyama spacing and return non-degenerate positions."""
        calls.append(kwargs)
        return torch.tensor([[0.0, 0.0], [0.0, 100.0]], dtype=torch.float32)

    def fake_register(
        name: str,
        raw_pos: torch.Tensor,
        problem: LayoutProblem,
        config: LayoutConfig,
        positions: dict[str, torch.Tensor],
        preserve_rank_order: bool = False,
        arm_timings: Optional[dict[str, tuple[float, float]]] = None,
        timing_span: Optional[tuple[float, float]] = None,
    ) -> None:
        """Record each challenger without invoking overlap projection."""
        del preserve_rank_order, arm_timings, timing_span
        positions[name] = raw_pos

    def fake_score(*args: object) -> float:
        """Return tied scores so the incumbent remains selected."""
        return 0.0

    dagua_native = importlib.import_module("dagua.layout.ops.pipelines.dagua_native")
    sugiyama = importlib.import_module("dagua.layout.ops.pipelines.sugiyama")
    native_directed = importlib.import_module("dagua.layout.ops.pipelines.native_directed")
    monkeypatch.setattr(dagua_native, "_run_native_problem", fake_native_problem)
    monkeypatch.setattr(sugiyama, "layout_sugiyama_pipeline", fake_sugiyama)
    monkeypatch.setattr(native_directed, "_register_challenger_variants", fake_register)
    monkeypatch.setattr(native_directed, "_score_directed_candidate", fake_score)
    problem = LayoutProblem(
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        num_nodes=2,
        node_sizes=torch.full((2, 2), 60.0),
    )

    layout_native_directed_portfolio(
        problem,
        SolveState(),
        RuntimeContext(),
        LayoutConfig(),
    )

    grid_size = (
        len(SUGIYAMA_FIDELITY_MODES) * len(SUGIYAMA_RANK_SEP_GRID) * len(SUGIYAMA_NODE_SEP_GRID)
    )
    assert len(calls) == 4 + grid_size
    assert calls[0]["graphviz_corrected_dot_x"] is True
    assert calls[1]["graphviz_preserve_point_units"] is True
    assert all("rank_sep" not in call and "node_sep" not in call for call in calls[2:4])
    observed = {(call["fidelity_mode"], call["rank_sep"], call["node_sep"]) for call in calls[4:]}
    expected = {
        (mode, rank_sep, node_sep)
        for mode in SUGIYAMA_FIDELITY_MODES
        for rank_sep in SUGIYAMA_RANK_SEP_GRID
        for node_sep in SUGIYAMA_NODE_SEP_GRID
    }
    assert observed == expected
    assert IGRAPH_OUTPUT_SCALE == 50.0


def test_directed_grid_gate_keeps_small_wide_dags() -> None:
    """Width and dummy structural limits apply only to n>=250 DAGs."""
    config = LayoutConfig()
    config._dagua_native_deadline_s = time.perf_counter() + 1.0
    problem = LayoutProblem(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=102,
        node_sizes=torch.full((102, 2), 60.0),
    )

    assert _full_sugiyama_grid_enabled(problem, config)


def test_directed_large_deadline_skips_cartesian_sugiyama_grid(monkeypatch: object) -> None:
    """Large DAGs under a hard deadline keep only fast Sugiyama arms."""
    calls: list[dict[str, object]] = []
    num_nodes = 300

    def fake_native_problem(*args: object, **kwargs: object) -> torch.Tensor:
        """Return a finite incumbent quickly."""
        del args, kwargs
        return torch.zeros((num_nodes, 2), dtype=torch.float32)

    def fake_sugiyama(**kwargs: object) -> torch.Tensor:
        """Record each Sugiyama solve and return finite positions."""
        calls.append(kwargs)
        y = torch.arange(num_nodes, dtype=torch.float32)
        return torch.stack([torch.zeros_like(y), y * 100.0], dim=1)

    def fake_register(
        name: str,
        raw_pos: torch.Tensor,
        problem: LayoutProblem,
        config: LayoutConfig,
        positions: dict[str, torch.Tensor],
        preserve_rank_order: bool = False,
        arm_timings: Optional[dict[str, tuple[float, float]]] = None,
        timing_span: Optional[tuple[float, float]] = None,
    ) -> None:
        """Record one candidate without projection cost."""
        del problem, config, preserve_rank_order, arm_timings, timing_span
        positions[name] = raw_pos

    def fake_proxy(*args: object, **kwargs: object) -> float:
        """Return a tied proxy score."""
        del args, kwargs
        return 0.0

    def fake_score(*args: object, **kwargs: object) -> float:
        """Return a tied full score."""
        del args, kwargs
        return 0.0

    dagua_native = importlib.import_module("dagua.layout.ops.pipelines.dagua_native")
    sugiyama = importlib.import_module("dagua.layout.ops.pipelines.sugiyama")
    native_directed = importlib.import_module("dagua.layout.ops.pipelines.native_directed")
    monkeypatch.setattr(dagua_native, "_run_native_problem", fake_native_problem)
    monkeypatch.setattr(sugiyama, "layout_sugiyama_pipeline", fake_sugiyama)
    monkeypatch.setattr(native_directed, "_register_challenger_variants", fake_register)
    monkeypatch.setattr(native_directed, "_proxy_directed_candidate", fake_proxy)
    monkeypatch.setattr(native_directed, "_score_directed_candidate", fake_score)
    edge_index = torch.stack(
        [
            torch.arange(num_nodes - 1, dtype=torch.long),
            torch.arange(1, num_nodes, dtype=torch.long),
        ]
    )
    config = LayoutConfig()
    config._dagua_native_deadline_s = time.perf_counter() + 130.0

    layout_native_directed_portfolio(
        LayoutProblem(
            edge_index=edge_index,
            num_nodes=num_nodes,
            node_sizes=torch.full((num_nodes, 2), 60.0),
        ),
        SolveState(),
        RuntimeContext(),
        config,
    )

    assert len(calls) == 4
    assert all("rank_sep" not in call and "node_sep" not in call for call in calls[2:])


def test_directed_predicted_cost_skips_second_dotx_arm(monkeypatch: object) -> None:
    """The point-unit dot-x arm does not start when sibling cost predicts risk."""
    calls: list[dict[str, object]] = []
    predictions: list[float] = []
    num_nodes = 250

    def fake_native_problem(*args: object, **kwargs: object) -> torch.Tensor:
        """Return a finite incumbent quickly."""
        del args, kwargs
        return torch.zeros((num_nodes, 2), dtype=torch.float32)

    def fake_sugiyama(**kwargs: object) -> torch.Tensor:
        """Record each Sugiyama solve and return finite positions."""
        calls.append(kwargs)
        y = torch.arange(num_nodes, dtype=torch.float32)
        return torch.stack([torch.zeros_like(y), y * 100.0], dim=1)

    def fake_register(
        name: str,
        raw_pos: torch.Tensor,
        problem: LayoutProblem,
        config: LayoutConfig,
        positions: dict[str, torch.Tensor],
        preserve_rank_order: bool = False,
        arm_timings: Optional[dict[str, tuple[float, float]]] = None,
        timing_span: Optional[tuple[float, float]] = None,
    ) -> None:
        """Record one candidate without projection cost."""
        del problem, config, preserve_rank_order, arm_timings, timing_span
        positions[name] = raw_pos

    def fake_predicted(config: LayoutConfig, predicted_cost_s: float) -> bool:
        """Allow the first dot-x arm and reject the measured sibling follow-up."""
        del config
        predictions.append(predicted_cost_s)
        return len(calls) == 0

    def fake_proxy(*args: object, **kwargs: object) -> float:
        """Return a tied proxy score."""
        del args, kwargs
        return 0.0

    def fake_score(*args: object, **kwargs: object) -> float:
        """Return a tied full score."""
        del args, kwargs
        return 0.0

    dagua_native = importlib.import_module("dagua.layout.ops.pipelines.dagua_native")
    sugiyama = importlib.import_module("dagua.layout.ops.pipelines.sugiyama")
    native_directed = importlib.import_module("dagua.layout.ops.pipelines.native_directed")
    monkeypatch.setattr(dagua_native, "_run_native_problem", fake_native_problem)
    monkeypatch.setattr(sugiyama, "layout_sugiyama_pipeline", fake_sugiyama)
    monkeypatch.setattr(native_directed, "_register_challenger_variants", fake_register)
    monkeypatch.setattr(native_directed, "_predicted_arm_budget_available", fake_predicted)
    monkeypatch.setattr(native_directed, "_proxy_directed_candidate", fake_proxy)
    monkeypatch.setattr(native_directed, "_score_directed_candidate", fake_score)
    monkeypatch.setattr(native_directed, "_directed_pivot_mds_candidates", lambda *args: {})
    monkeypatch.setattr(native_directed, "_directed_stress_blend_candidates", lambda *args: {})
    monkeypatch.setattr(native_directed, "_directed_mrtree_enabled", lambda *args: False)
    monkeypatch.setattr(
        native_directed,
        "_directed_recombinant_layered_enabled",
        lambda *args: False,
    )
    monkeypatch.setattr(native_directed, "_ordering_cost_admissible", lambda *args, **kwargs: False)
    edge_index = torch.stack(
        [
            torch.arange(num_nodes - 1, dtype=torch.long),
            torch.arange(1, num_nodes, dtype=torch.long),
        ]
    )

    layout_native_directed_portfolio(
        LayoutProblem(
            edge_index=edge_index,
            num_nodes=num_nodes,
            node_sizes=torch.full((num_nodes, 2), 60.0),
        ),
        SolveState(),
        RuntimeContext(),
        LayoutConfig(),
    )

    assert len(calls) == 1
    assert calls[0]["graphviz_corrected_dot_x"] is True
    assert predictions[-2:] == [pytest.approx(2.2), pytest.approx(2.2)]


def test_directed_sugiyama_ledger_admission_skips_before_run(monkeypatch: object) -> None:
    """Directed Sugiyama arms use ledger admission instead of measured sibling runtime."""
    from dagua.layout.ops.pipelines.native_budget import DECISION_LOG_ATTR, install_budget_ledger

    calls: list[dict[str, object]] = []
    num_nodes = 250

    def fake_native_problem(*args: object, **kwargs: object) -> torch.Tensor:
        """Return a finite incumbent quickly."""
        del args, kwargs
        return torch.zeros((num_nodes, 2), dtype=torch.float32)

    def fake_sugiyama(**kwargs: object) -> torch.Tensor:
        """Record unexpected Sugiyama execution."""
        calls.append(kwargs)
        y = torch.arange(num_nodes, dtype=torch.float32)
        return torch.stack([torch.zeros_like(y), y * 100.0], dim=1)

    def fake_proxy(*args: object, **kwargs: object) -> float:
        """Return tied proxy scores."""
        del args, kwargs
        return 0.0

    def fake_score(*args: object, **kwargs: object) -> float:
        """Return tied full scores."""
        del args, kwargs
        return 0.0

    dagua_native = importlib.import_module("dagua.layout.ops.pipelines.dagua_native")
    native_undirected = importlib.import_module("dagua.layout.ops.pipelines.native_undirected")
    sugiyama = importlib.import_module("dagua.layout.ops.pipelines.sugiyama")
    native_directed = importlib.import_module("dagua.layout.ops.pipelines.native_directed")
    monkeypatch.setattr(dagua_native, "_run_native_problem", fake_native_problem)
    monkeypatch.setattr(sugiyama, "layout_sugiyama_pipeline", fake_sugiyama)
    monkeypatch.setattr(native_directed, "_predicted_arm_budget_available", lambda *args: True)
    monkeypatch.setattr(native_undirected, "_portfolio_has_budget", lambda *args, **kwargs: True)
    monkeypatch.setattr(native_directed, "_proxy_directed_candidate", fake_proxy)
    monkeypatch.setattr(native_directed, "_score_directed_candidate", fake_score)
    monkeypatch.setattr(native_directed, "_directed_pivot_mds_candidates", lambda *args: {})
    monkeypatch.setattr(native_directed, "_directed_stress_blend_candidates", lambda *args: {})
    monkeypatch.setattr(native_directed, "_directed_mrtree_enabled", lambda *args: False)
    monkeypatch.setattr(native_directed, "_force_challengers_enabled", lambda *args: False)
    edge_index = torch.stack(
        [
            torch.arange(num_nodes - 1, dtype=torch.long),
            torch.arange(1, num_nodes, dtype=torch.long),
        ]
    )
    config = LayoutConfig()
    install_budget_ledger(config, timeout_s=6.0, return_reserve_dwu=5.0)

    layout_native_directed_portfolio(
        LayoutProblem(
            edge_index=edge_index,
            num_nodes=num_nodes,
            node_sizes=torch.full((num_nodes, 2), 60.0),
        ),
        SolveState(),
        RuntimeContext(),
        config,
    )

    assert calls == []
    assert any(
        record["event"] == "skip" and record["reason"] == "optional_directed_sugiyama_cluster_dotx"
        for record in getattr(config, DECISION_LOG_ATTR)
    )


def test_directed_referee_full_scores_only_proxy_finalists(monkeypatch: object) -> None:
    """Directed contests quick-score all arms but full-score only challenger finalists."""
    from dagua.layout.ops.pipelines.native_budget import DECISION_LOG_ATTR, install_budget_ledger

    full_scored: list[float] = []
    proxy_scored: list[float] = []
    num_nodes = 250

    def fake_native_problem(*args: object, **kwargs: object) -> torch.Tensor:
        """Return the incumbent position."""
        del args, kwargs
        return torch.zeros((num_nodes, 2), dtype=torch.float32)

    def fake_sugiyama(**kwargs: object) -> torch.Tensor:
        """Return distinct x coordinates so proxy order is deterministic."""
        rank_sep = float(kwargs.get("rank_sep", len(proxy_scored) + 1.0))
        node_sep = float(kwargs.get("node_sep", 0.0))
        value = rank_sep + node_sep * 0.01
        y = torch.arange(num_nodes, dtype=torch.float32)
        return torch.stack([torch.full_like(y, value), y * 100.0], dim=1)

    def fake_register(
        name: str,
        raw_pos: torch.Tensor,
        problem: LayoutProblem,
        config: LayoutConfig,
        positions: dict[str, torch.Tensor],
        preserve_rank_order: bool = False,
        arm_timings: Optional[dict[str, tuple[float, float]]] = None,
        timing_span: Optional[tuple[float, float]] = None,
    ) -> None:
        """Register one variant per candidate family."""
        del problem, config, preserve_rank_order, arm_timings, timing_span
        positions[name] = raw_pos

    def fake_proxy(
        pos: torch.Tensor,
        problem: LayoutProblem,
        cluster_ids: Optional[torch.Tensor],
        all_pairs_dist: object = None,
    ) -> float:
        """Use x coordinate as the proxy score."""
        del problem, cluster_ids, all_pairs_dist
        score = float(pos[0, 0].item())
        proxy_scored.append(score)
        return score

    def fake_score_payload(
        pos: torch.Tensor,
        problem: LayoutProblem,
        cluster_ids: Optional[torch.Tensor],
        all_pairs_dist: object = None,
    ) -> tuple[float, _DirectedClusterScoreTelemetry]:
        """Use x coordinate as the full score."""
        del problem, cluster_ids, all_pairs_dist
        score = float(pos[0, 0].item())
        full_scored.append(score)
        return (
            score,
            _DirectedClusterScoreTelemetry(
                extended_score=score,
                old_score=score,
                metrics={},
                v3_tiered=score,
            ),
        )

    def fake_grid_enabled(problem: LayoutProblem, config: LayoutConfig) -> bool:
        """Force the large-graph test to build enough candidates."""
        del problem, config
        return True

    dagua_native = importlib.import_module("dagua.layout.ops.pipelines.dagua_native")
    sugiyama = importlib.import_module("dagua.layout.ops.pipelines.sugiyama")
    native_directed = importlib.import_module("dagua.layout.ops.pipelines.native_directed")
    monkeypatch.setattr(dagua_native, "_run_native_problem", fake_native_problem)
    monkeypatch.setattr(sugiyama, "layout_sugiyama_pipeline", fake_sugiyama)
    monkeypatch.setattr(native_directed, "_register_challenger_variants", fake_register)
    monkeypatch.setattr(native_directed, "_proxy_directed_candidate", fake_proxy)
    monkeypatch.setattr(
        native_directed,
        "_score_directed_candidate_referee_payload",
        fake_score_payload,
    )
    monkeypatch.setattr(native_directed, "_full_sugiyama_grid_enabled", fake_grid_enabled)
    edge_index = torch.stack(
        [
            torch.arange(num_nodes - 1, dtype=torch.long),
            torch.arange(1, num_nodes, dtype=torch.long),
        ]
    )
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=torch.full((num_nodes, 2), 60.0),
    )

    config = LayoutConfig()
    install_budget_ledger(config, timeout_s=300.0, return_reserve_dwu=5.0)

    layout_native_directed_portfolio(problem, SolveState(), RuntimeContext(), config)

    expected_candidates = 4 + len(SUGIYAMA_FIDELITY_MODES) * len(SUGIYAMA_RANK_SEP_GRID) * len(
        SUGIYAMA_NODE_SEP_GRID
    )
    assert len(proxy_scored) == expected_candidates + 1
    assert len(full_scored) == DIRECTED_FULL_REFEREE_TOP_K + 1
    decision_log = getattr(config, DECISION_LOG_ATTR)
    admitted_sugiyama = [
        record["reason"]
        for record in decision_log
        if record["event"] == "admit"
        and str(record["reason"]).startswith("optional_directed_sugiyama")
    ]
    skipped_sugiyama = [
        record["reason"]
        for record in decision_log
        if record["event"] == "skip"
        and str(record["reason"]).startswith("optional_directed_sugiyama")
    ]

    assert len(admitted_sugiyama) == expected_candidates
    assert skipped_sugiyama == []
