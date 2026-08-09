"""Tests for the planar-certificate contest arm (sprint2 W1-B)."""

from __future__ import annotations

import importlib

import torch

from dagua.config import LayoutConfig
from dagua.layout.graph_classify import classify_graph
from dagua.layout.ops.pipelines.native_planar_arm import (
    PLANAR_ARM_MAX_NODES,
    build_planar_arm_candidates,
    planar_arm_admitted,
)
from dagua.layout.ops.planar_polish import exact_crossing_count
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState


def _wheel_problem(spokes: int = 8) -> LayoutProblem:
    """Return a wheel-graph problem (planar, 3-connected, single component)."""
    edges = []
    for spoke in range(1, spokes + 1):
        edges.append((0, spoke))
        edges.append((spoke, 1 + spoke % spokes))
    edge_index = torch.tensor(sorted(set(edges)), dtype=torch.long).t().contiguous()
    num_nodes = spokes + 1
    structure = classify_graph(edge_index, num_nodes)
    return LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=torch.full((num_nodes, 2), 20.0),
        structure=structure,
        seed=42,
    )


def _k5_problem() -> LayoutProblem:
    """Return a K5 problem (non-planar: the gate must stay closed)."""
    edges = [(u, v) for u in range(5) for v in range(u + 1, 5)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    structure = classify_graph(edge_index, 5)
    return LayoutProblem(
        edge_index=edge_index,
        num_nodes=5,
        node_sizes=torch.full((5, 2), 20.0),
        structure=structure,
        seed=42,
    )


def test_gate_opens_on_exact_planar_and_closes_on_k5() -> None:
    """Gate requires exact planarity with a cached embedding."""
    wheel = _wheel_problem()
    assert wheel.structure is not None and wheel.structure.is_planar is True
    assert planar_arm_admitted(wheel)
    k5 = _k5_problem()
    assert k5.structure is not None and k5.structure.is_planar is False
    assert not planar_arm_admitted(k5)


def test_gate_closes_without_embedding_or_above_cap() -> None:
    """Euler-hint-only planarity (no embedding) and oversize rows fail closed."""
    from types import SimpleNamespace

    wheel = _wheel_problem()
    hint_only = SimpleNamespace(is_planar=True, planar_embedding=None, num_components=1)
    problem = LayoutProblem(
        edge_index=wheel.edge_index,
        num_nodes=wheel.num_nodes,
        structure=hint_only,  # type: ignore[arg-type]
        seed=42,
    )
    assert not planar_arm_admitted(problem)
    oversized = LayoutProblem(
        edge_index=wheel.edge_index,
        num_nodes=PLANAR_ARM_MAX_NODES + 1,
        structure=wheel.structure,
        seed=42,
    )
    assert not planar_arm_admitted(oversized)


def test_builder_emits_parity_floor_and_zero_crossing_polish() -> None:
    """The arm yields FPP raw + polished variants with exact zero crossings."""
    problem = _wheel_problem()
    candidates = build_planar_arm_candidates(problem, node_sep=40.0)
    assert "planar_fpp" in candidates
    assert "planar_fpp_polished" in candidates
    for name, pos in candidates.items():
        assert bool(torch.isfinite(pos).all().item()), name
        if name.endswith("_polished") or name == "planar_fpp":
            assert exact_crossing_count(pos, problem.edge_index) == 0, name


def test_builder_is_deterministic() -> None:
    """Two builder runs produce byte-identical candidates."""
    problem = _wheel_problem()
    first = build_planar_arm_candidates(problem, node_sep=40.0)
    second = build_planar_arm_candidates(problem, node_sep=40.0)
    assert sorted(first) == sorted(second)
    for name in first:
        assert torch.equal(first[name], second[name]), name


def test_gate_closed_row_never_builds_candidates(monkeypatch: object) -> None:
    """On a non-planar row the arm code never runs (byte-inert contract)."""
    from dagua.layout.ops.pipelines.native_undirected import (
        layout_native_undirected_portfolio,
    )

    arm_module = importlib.import_module("dagua.layout.ops.pipelines.native_planar_arm")

    def _must_not_run(*args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        raise AssertionError("planar arm built candidates on a gate-closed row")

    monkeypatch.setattr(arm_module, "build_planar_arm_candidates", _must_not_run)
    problem = _k5_problem()
    result = layout_native_undirected_portfolio(
        problem,
        SolveState(),
        RuntimeContext(),
        LayoutConfig(seed=42),
    )
    assert bool(torch.isfinite(result).all().item())


def test_planar_arm_fires_inside_undirected_contest(monkeypatch: object) -> None:
    """On a gated planar row the contest builds and registers the arm."""
    from dagua.layout.ops.pipelines.native_undirected import (
        layout_native_undirected_portfolio,
    )

    arm_module = importlib.import_module("dagua.layout.ops.pipelines.native_planar_arm")
    real_builder = arm_module.build_planar_arm_candidates
    seen: list[str] = []

    def _recording_builder(*args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        candidates = real_builder(*args, **kwargs)
        seen.extend(sorted(candidates))
        return candidates

    monkeypatch.setattr(arm_module, "build_planar_arm_candidates", _recording_builder)
    problem = _wheel_problem()
    result = layout_native_undirected_portfolio(
        problem,
        SolveState(),
        RuntimeContext(),
        LayoutConfig(seed=42),
    )
    assert bool(torch.isfinite(result).all().item())
    assert "planar_fpp" in seen
    assert any(name.endswith("_polished") for name in seen)
