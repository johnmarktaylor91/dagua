"""Tests for native scale guardrail substrate."""

from __future__ import annotations

import time

import pytest
import torch

from dagua.config import LayoutConfig
from dagua.eval.graphs import TestGraph
from dagua.graph import DaguaGraph
from dagua.layout.ops.pipelines.native_arm_s import ARM_S_PRIOR_S
from dagua.layout.ops.pipelines.native_guardrails import (
    CLUSTER_SKELETON_ONLY,
    INCUMBENT_ONLY,
    SAMPLED_POLISH,
    SMALL_EXACT,
    NativeGuardrailCaps,
    NativeGuardrailPlan,
    build_native_guardrail_cost_inputs,
    build_native_guardrail_plan,
    build_native_guardrail_samples,
    register_native_guardrail_observation,
    run_guarded_native_cluster_candidate,
)
from dagua.layout.ops.state import LayoutProblem
from scripts import native_sprint_score as scorer


def _problem_from_test_graph(graph: TestGraph) -> LayoutProblem:
    """Build a layout problem from one scorer test graph.

    Parameters
    ----------
    graph : TestGraph
        Scorer test graph wrapper.

    Returns
    -------
    LayoutProblem
        Prepared problem carrying topology and cluster metadata.
    """
    dagua_graph = graph.graph
    return LayoutProblem(
        edge_index=dagua_graph.edge_index.detach().to(device="cpu", dtype=torch.long),
        num_nodes=dagua_graph.num_nodes,
        node_sizes=torch.ones((dagua_graph.num_nodes, 2), dtype=torch.float32),
        clusters=dagua_graph.clusters,
        cluster_parents=dagua_graph.cluster_parents,
        seed=42,
    )


def _plain_graph() -> DaguaGraph:
    """Return a small non-cluster control graph.

    Returns
    -------
    DaguaGraph
        Two-node directed graph with no cluster metadata.
    """
    graph = DaguaGraph()
    graph.add_node("a")
    graph.add_node("b")
    graph.add_edge("a", "b")
    graph.compute_node_sizes()
    return graph


def _plain_problem() -> LayoutProblem:
    """Return a non-cluster layout problem.

    Returns
    -------
    LayoutProblem
        Small problem without clusters.
    """
    return LayoutProblem(
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        num_nodes=2,
        node_sizes=torch.ones((2, 2), dtype=torch.float32),
        seed=42,
    )


def _finite_incumbent(num_nodes: int) -> torch.Tensor:
    """Return deterministic full incumbent positions.

    Parameters
    ----------
    num_nodes : int
        Node count.

    Returns
    -------
    torch.Tensor
        Full finite positions with shape ``[N, 2]``.
    """
    values = torch.arange(num_nodes * 2, dtype=torch.float32)
    return values.reshape(num_nodes, 2)


def test_scale_guardrail_cost_inputs_cover_required_dimensions() -> None:
    """The 1k R8 row exposes every required structural cost dimension."""
    graph = scorer.build_graph_map(["r8_nested_scale_1k_budget"])["r8_nested_scale_1k_budget"]
    problem = _problem_from_test_graph(graph)

    cost = build_native_guardrail_cost_inputs(problem)

    assert cost.num_nodes == 1000
    assert cost.num_edges == int(problem.edge_index.shape[1])
    assert cost.cluster_count == len(problem.clusters or {})
    assert cost.max_depth >= 1
    assert cost.total_ancestor_memberships >= cost.cluster_count
    assert cost.sibling_pair_count >= 0
    assert cost.edge_cluster_obstacle_count == cost.num_edges * cost.cluster_count
    assert cost.max_rank_width >= 1
    assert cost.estimated_dummy_expansion >= 0


def test_guardrail_plan_and_samples_are_deterministic_on_scale_1k() -> None:
    """Repeated scale-row planning produces identical modes, samples, and caps."""
    graph = scorer.build_graph_map(["r8_nested_scale_1k_budget"])["r8_nested_scale_1k_budget"]
    problem = _problem_from_test_graph(graph)
    caps = NativeGuardrailCaps(
        sibling_pair_sample_cap=11,
        exclusion_pair_sample_cap=13,
        edge_cluster_obstacle_sample_cap=17,
    )

    first = build_native_guardrail_plan(problem, LayoutConfig(seed=42), caps=caps)
    second = build_native_guardrail_plan(problem, LayoutConfig(seed=42), caps=caps)

    assert first == second
    assert len(first.samples.sibling_pairs) <= caps.sibling_pair_sample_cap
    assert len(first.samples.exclusion_pairs) <= caps.exclusion_pair_sample_cap
    assert len(first.samples.edge_cluster_obstacles) <= caps.edge_cluster_obstacle_sample_cap


def test_guardrail_caps_bound_large_pair_populations() -> None:
    """Sample caps bound sibling, exclusion, and obstacle work."""
    clusters = {f"c{index:03d}": [index] for index in range(80)}
    parents: dict[str, str | None] = {name: None for name in clusters}
    edge_index = torch.stack(
        (
            torch.arange(0, 199, dtype=torch.long),
            torch.arange(1, 200, dtype=torch.long),
        )
    )
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=200,
        node_sizes=torch.ones((200, 2), dtype=torch.float32),
        clusters=clusters,
        cluster_parents=parents,
    )
    caps = NativeGuardrailCaps(
        sibling_pair_sample_cap=7,
        exclusion_pair_sample_cap=9,
        edge_cluster_obstacle_sample_cap=5,
    )

    cost = build_native_guardrail_cost_inputs(problem)
    samples = build_native_guardrail_samples(problem, caps)

    assert cost.sibling_pair_count == 80 * 79 // 2
    assert cost.edge_cluster_obstacle_count == 199 * 80
    assert len(samples.sibling_pairs) <= 7
    assert len(samples.exclusion_pairs) <= 9
    assert len(samples.edge_cluster_obstacles) <= 5
    assert len(set(samples.sibling_pairs)) == len(samples.sibling_pairs)


@pytest.mark.parametrize(
    "mode",
    [SMALL_EXACT, SAMPLED_POLISH, CLUSTER_SKELETON_ONLY, INCUMBENT_ONLY],
)
def test_degrade_tiers_return_full_finite_positions(mode: str) -> None:
    """Every degrade tier returns a finite full-position tensor."""
    problem = _plain_problem()
    cost = build_native_guardrail_cost_inputs(problem)
    incumbent = _finite_incumbent(problem.num_nodes)
    plan = NativeGuardrailPlan(
        mode=mode,
        admitted=mode != INCUMBENT_ONLY,
        cost_inputs=cost,
        caps=NativeGuardrailCaps(),
        samples=build_native_guardrail_samples(problem),
        predicted_cost_s=0.0,
        available_work_s=None,
    )

    def candidate_fn(_plan: NativeGuardrailPlan) -> torch.Tensor:
        """Return a full finite challenger.

        Parameters
        ----------
        _plan : NativeGuardrailPlan
            Active guardrail plan.

        Returns
        -------
        torch.Tensor
            Full finite positions.
        """
        return incumbent + 1.0

    result = run_guarded_native_cluster_candidate(
        plan=plan,
        incumbent_pos=incumbent,
        candidate_fn=candidate_fn,
    )

    assert result.shape == incumbent.shape
    assert bool(torch.isfinite(result).all().item())
    if mode == INCUMBENT_ONLY:
        assert torch.equal(result, incumbent)


def test_deadline_admission_uses_incumbent_only_before_reserve() -> None:
    """Deadline pressure degrades candidate admission to incumbent-only."""
    config = LayoutConfig(seed=42)
    setattr(config, "_dagua_native_deadline_s", time.perf_counter() + 1.0)
    plan = build_native_guardrail_plan(_plain_problem(), config, prior_cost_s=0.1)

    assert plan.mode == INCUMBENT_ONLY
    assert not plan.admitted
    assert plan.skip_reason == "insufficient_predicted_budget"


def test_arm_s_prior_uses_existing_guardrail_admission() -> None:
    """Arm S's measured prior is admitted by the existing guardrail substrate."""
    graph = scorer.build_graph_map(["r8_nested_scale_1k_budget"])["r8_nested_scale_1k_budget"]
    problem = _problem_from_test_graph(graph)

    plan = build_native_guardrail_plan(problem, LayoutConfig(seed=42), prior_cost_s=ARM_S_PRIOR_S)

    assert plan.admitted
    assert plan.predicted_cost_s == pytest.approx(ARM_S_PRIOR_S * 2.0)
    assert plan.skip_reason is None


def test_observed_cost_updates_only_config_telemetry() -> None:
    """Observed runs append telemetry without changing planning constants."""
    config = LayoutConfig(seed=42)

    register_native_guardrail_observation(
        config,
        candidate="recursive_cluster",
        mode=SAMPLED_POLISH,
        elapsed_s=1.25,
    )

    assert getattr(config, "_dagua_native_guardrail_observations") == [
        {"candidate": "recursive_cluster", "mode": SAMPLED_POLISH, "elapsed_s": 1.25}
    ]


def test_guardrail_evaluation_is_noop_for_current_outputs() -> None:
    """Planning does not mutate current scale-row or non-cluster outputs."""
    scale_graph = scorer.build_graph_map(["r8_nested_scale_1k_budget"])["r8_nested_scale_1k_budget"]
    scale_problem = _problem_from_test_graph(scale_graph)
    plain_graph = _plain_graph()
    plain_problem = _plain_problem()
    positions: list[tuple[LayoutProblem, torch.Tensor]] = [
        (scale_problem, _finite_incumbent(scale_problem.num_nodes)),
        (plain_problem, _finite_incumbent(plain_graph.num_nodes)),
    ]

    for problem, incumbent in positions:
        before = incumbent.detach().clone()
        plan = build_native_guardrail_plan(problem, LayoutConfig(seed=42))
        result = run_guarded_native_cluster_candidate(
            plan=plan,
            incumbent_pos=incumbent,
            candidate_fn=lambda _plan: incumbent,
        )

        assert torch.equal(incumbent, before)
        assert torch.equal(result, before)
