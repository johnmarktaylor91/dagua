"""Tests for the r83 directed-table native portfolio."""

from __future__ import annotations

import importlib
import time
from typing import Optional

import torch

from dagua.config import LayoutConfig
from dagua.graph import DaguaGraph
from dagua.layout.graph_classify import classify_graph
from dagua.layout.ops.pipelines.dagua_native import _choose_native_pipeline
from dagua.layout.ops.pipelines.native_directed import (
    DIRECTED_FULL_REFEREE_TOP_K,
    IGRAPH_OUTPUT_SCALE,
    SUGIYAMA_FIDELITY_MODES,
    SUGIYAMA_NODE_SEP_GRID,
    SUGIYAMA_RANK_SEP_GRID,
    _directed_mrtree_enabled,
    _directed_pivot_mds_candidates,
    _directed_stress_blend_candidates,
    _exact_crossing_count,
    _exact_crossing_count_loop,
    _force_challengers_enabled,
    _full_sugiyama_grid_enabled,
    _rank_local_zero_crossing_swap_candidate,
    _register_challenger_variants,
    _restore_projected_rank_order,
    _score_directed_candidate,
    layout_native_directed_portfolio,
)
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState


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
        return len(predictions) == 1

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
    assert len(predictions) == 2


def test_directed_referee_full_scores_only_proxy_finalists(monkeypatch: object) -> None:
    """Directed contests quick-score all arms but full-score only challenger finalists."""
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

    def fake_score(
        pos: torch.Tensor,
        problem: LayoutProblem,
        cluster_ids: Optional[torch.Tensor],
        all_pairs_dist: object = None,
    ) -> float:
        """Use x coordinate as the full score."""
        del problem, cluster_ids, all_pairs_dist
        score = float(pos[0, 0].item())
        full_scored.append(score)
        return score

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
    monkeypatch.setattr(native_directed, "_score_directed_candidate", fake_score)
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

    layout_native_directed_portfolio(problem, SolveState(), RuntimeContext(), LayoutConfig())

    expected_candidates = 4 + len(SUGIYAMA_FIDELITY_MODES) * len(SUGIYAMA_RANK_SEP_GRID) * len(
        SUGIYAMA_NODE_SEP_GRID
    )
    assert len(proxy_scored) == expected_candidates + 1
    assert len(full_scored) == DIRECTED_FULL_REFEREE_TOP_K + 1
