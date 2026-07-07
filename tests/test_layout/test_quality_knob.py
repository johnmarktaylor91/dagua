"""Tests for the public native layout quality knob."""

from __future__ import annotations

import time

import torch

from dagua.config import LayoutConfig
from dagua.graph import DaguaGraph
from dagua.layout.engine import layout
from dagua.layout.resolve import prepare_pipeline_config, resolve_quality_budgets
from dagua.metrics import composite, full


def _budget_tuple(
    quality: float,
    num_nodes: int = 1000,
) -> tuple[float, int, int, int, float, float]:
    """Return ordered scalar budgets for monotonicity assertions.

    Parameters
    ----------
    quality : float
        Normalized quality value.
    num_nodes : int, default=1000
        Graph size used to cap pivot budgets.

    Returns
    -------
    tuple[float, int, int, int, float, float]
        Step multiplier, multi-start count, pivots, SMACOF iterations,
        multilevel refine multiplier, and sampling rate.
    """
    budget = resolve_quality_budgets(quality, num_nodes=num_nodes)
    return (
        budget.step_multiplier,
        budget.multi_start_k,
        budget.stress_n_pivots,
        budget.smacof_iters,
        budget.ml_refine_multiplier,
        budget.sampling_rate,
    )


def _small_graph(seed: int) -> DaguaGraph:
    """Build a deterministic small DAG with mild crossing pressure.

    Parameters
    ----------
    seed : int
        Accepted for seeded-test symmetry. The graph shape is fixed.

    Returns
    -------
    DaguaGraph
        Graph with 6 nodes and deterministic directed edges.
    """
    del seed
    graph = DaguaGraph()
    for node_id in range(6):
        graph.add_node(node_id)
    for node_id in range(5):
        graph.add_edge(node_id, node_id + 1)
    for source in range(4):
        graph.add_edge(source, source + 2)
    graph.compute_node_sizes()
    return graph


def _large_dag(num_nodes: int) -> DaguaGraph:
    """Build a connected chain DAG for wall-clock budget smoke tests.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.

    Returns
    -------
    DaguaGraph
        Deterministic chain graph.
    """
    graph = DaguaGraph()
    for node_id in range(num_nodes):
        graph.add_node(node_id)
    for node_id in range(num_nodes - 1):
        graph.add_edge(node_id, node_id + 1)
    graph.compute_node_sizes()
    return graph


def _score(graph: DaguaGraph, pos: torch.Tensor) -> float:
    """Return deterministic composite for a graph layout.

    Parameters
    ----------
    graph : DaguaGraph
        Scored graph.
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    float
        Composite score.
    """
    torch.manual_seed(0)
    return float(composite(full(pos, graph.edge_index, node_sizes=graph.node_sizes)))


def test_quality_budget_mapping_is_monotonic() -> None:
    """Verify higher quality never lowers scalar budgets."""
    qualities = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
    budget_rows = [_budget_tuple(quality) for quality in qualities]
    for previous, current in zip(budget_rows, budget_rows[1:]):
        assert all(
            current_value >= previous_value
            for previous_value, current_value in zip(previous, current)
        )


def test_quality_names_match_float_aliases() -> None:
    """Verify named quality aliases normalize to their documented floats."""
    aliases = {
        "draft": 0.25,
        "balanced": 0.5,
        "high": 0.75,
        "max": 1.0,
    }
    for name, value in aliases.items():
        assert LayoutConfig(quality=name).quality == LayoutConfig(quality=value).quality
        assert resolve_quality_budgets(
            float(LayoutConfig(quality=name).quality)
        ) == resolve_quality_budgets(value)


def test_explicit_steps_override_quality_knob() -> None:
    """Verify explicit ``steps`` is preserved over quality-derived auto steps."""
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    config = LayoutConfig(quality="max", steps=7)
    prepared = prepare_pipeline_config(
        config=config,
        num_nodes=4,
        edge_index=edge_index,
        device="cpu",
        layer_assignments=None,
        prebuilt_layer_index=None,
        graph_structure=None,
        skip_classification=False,
    )
    assert getattr(prepared, "_dagua_native_steps") == 7


def test_quality_high_smoke_spends_more_and_scores_near_draft() -> None:
    """Compare draft and high-quality layouts on seeded small graphs."""
    warmup = _small_graph(5)
    layout(warmup, LayoutConfig(seed=5, quality=0.5, algorithm="native_stress"))

    low_total = 0.0
    high_total = 0.0
    for seed in (11, 17, 23):
        graph = _small_graph(seed)
        low_start = time.perf_counter()
        low_pos = layout(graph, LayoutConfig(seed=seed, quality=0.1, algorithm="native_stress"))
        low_total += time.perf_counter() - low_start

        high_start = time.perf_counter()
        high_pos = layout(graph, LayoutConfig(seed=seed, quality=0.9, algorithm="native_stress"))
        high_total += time.perf_counter() - high_start

        assert high_pos.shape == low_pos.shape
        assert _score(graph, high_pos) >= _score(graph, low_pos) - 0.5
    assert high_total >= low_total


def test_time_budget_returns_finite_positions_under_wall_cap() -> None:
    """Verify a large native run respects a small wall-clock cap."""
    graph = _large_dag(2000)
    start = time.perf_counter()
    pos = layout(graph, LayoutConfig(seed=42, quality="max", time_budget_s=2.0))
    elapsed = time.perf_counter() - start
    assert elapsed < 6.0
    assert pos.shape == (graph.num_nodes, 2)
    assert bool(torch.isfinite(pos).all().item())
