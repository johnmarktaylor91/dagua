"""Tests for the r83 directed-table native portfolio."""

from __future__ import annotations

import importlib

import torch

from dagua.config import LayoutConfig
from dagua.graph import DaguaGraph
from dagua.layout.graph_classify import classify_graph
from dagua.layout.ops.pipelines.dagua_native import _choose_native_pipeline
from dagua.layout.ops.pipelines.native_directed import (
    IGRAPH_OUTPUT_SCALE,
    SUGIYAMA_FIDELITY_MODES,
    SUGIYAMA_NODE_SEP_GRID,
    SUGIYAMA_RANK_SEP_GRID,
    _force_challengers_enabled,
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
    ) -> None:
        """Record each challenger without invoking overlap projection."""
        del preserve_rank_order
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
