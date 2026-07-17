"""Tests for the r83 directed-table native portfolio."""

from __future__ import annotations

import importlib
import time
from types import SimpleNamespace
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
    _crossing_edge_pairs,
    _directed_mrtree_enabled,
    _directed_ordering_candidate_dual_dominates,
    _directed_pivot_mds_candidates,
    _directed_recombinant_layered_candidates,
    _directed_recombinant_layered_enabled,
    _directed_stress_blend_candidates,
    _exact_crossing_count,
    _exact_crossing_count_loop,
    _force_challengers_enabled,
    _full_sugiyama_grid_enabled,
    _ordering_cost_admissible,
    _rank_local_zero_crossing_swap_candidate,
    _rank_to_nodes_from_incumbent_y,
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

    def fake_score(*args: object, **kwargs: object) -> float:
        """Keep all candidates tied so the incumbent remains selected."""
        del args, kwargs
        return 1.0

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
    monkeypatch.setattr(native_directed, "_score_directed_candidate", fake_score)
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
    ) -> tuple[bool, W5ScorePair]:
        """Reject the crossing-improving candidate under the frozen dual gate."""
        del incumbent_pair, problem, cluster_ids, all_pairs_dist
        captured["candidate"] = candidate
        return False, W5ScorePair(directed=11.0, undirected=9.0)

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

    def fake_dual_gate(*args: object, **kwargs: object) -> tuple[bool, W5ScorePair]:
        """Admit the ordering seed while keeping scalar best_name incumbent."""
        del args, kwargs
        return True, W5ScorePair(directed=11.0, undirected=11.0)

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
    ) -> W5FinisherResult:
        """Capture the W5 incumbent payload and return a no-op result."""
        del seeds, node_sizes, score_fn, accept_margin
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
    ) -> tuple[bool, W5ScorePair]:
        """Reject the recombinant candidate under the dual frozen rulers."""
        del incumbent_pair, problem, cluster_ids, all_pairs_dist
        captured["candidate"] = candidate
        return False, W5ScorePair(directed=11.0, undirected=9.0)

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

    dominates, pair = _directed_ordering_candidate_dual_dominates(
        candidate,
        incumbent_pair,
        problem,
        None,
        None,
    )

    assert not dominates
    assert pair == W5ScorePair(directed=11.0, undirected=9.0)


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
