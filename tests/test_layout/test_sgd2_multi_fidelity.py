"""Regression tests for Round 23 sgd2_multi fidelity fixes."""

from __future__ import annotations

import random
import sys
import types
from pathlib import Path

import networkx as nx
import pytest
import torch

from dagua.eval.competitors import sgd2_multi_competitor
from dagua.eval.competitors.sgd2_multi_competitor import SGD2MultiRef
from dagua.graph import DaguaGraph
from dagua.layout.ops.pipelines.sgd2_multi import layout_sgd2_multi_pipeline
from dagua.layout.ops.sgd2_multi import _build_adjacency


def _make_small_graph() -> DaguaGraph:
    """Create a connected three-node graph.

    Returns
    -------
    DaguaGraph
        Three-node path graph.
    """
    graph = DaguaGraph()
    for node_idx in range(3):
        graph.add_node(str(node_idx), label=str(node_idx))
    graph.add_edge("0", "1")
    graph.add_edge("1", "2")
    return graph


def _install_fake_sgd2_modules(
    monkeypatch: pytest.MonkeyPatch,
    random_values: list[float],
) -> None:
    """Install in-memory upstream modules for adapter RNG tests.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture.
    random_values : list[float]
        Mutable list that receives one Python-random sample per optimize call.

    Returns
    -------
    None
        ``sys.modules`` is patched in-place by pytest.
    """
    criteria_module = types.ModuleType("criteria")
    criteria_module.ideal_edge_length = lambda *args, **kwargs: None
    criteria_module.stress = lambda *args, **kwargs: None
    criteria_module.aspect_ratio = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "criteria", criteria_module)

    class FakeGD2:
        """Minimal upstream GD2 stub that observes Python random state."""

        def __init__(self, graph: nx.Graph) -> None:
            self.non_incident_edge_pairs = [(0, 1, 1, 2)]
            self.pos = torch.zeros((graph.number_of_nodes(), 2), dtype=torch.float32)

        def optimize(self, **kwargs: object) -> None:
            """Record one random value after adapter seeding."""
            del kwargs
            random_values.append(random.random())

    gd2_module = types.ModuleType("gd2")
    gd2_module.GD2 = FakeGD2
    monkeypatch.setitem(sys.modules, "gd2", gd2_module)


def test_sgd2_multi_available_requires_criteria_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The adapter should not claim availability without ``criteria.py``."""
    (tmp_path / "gd2.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(sgd2_multi_competitor, "_SGD2_REPO", tmp_path)
    monkeypatch.delitem(sys.modules, "gd2", raising=False)
    monkeypatch.delitem(sys.modules, "criteria", raising=False)

    assert not sgd2_multi_competitor._sgd2_multi_available()
    result = SGD2MultiRef().layout(_make_small_graph(), seed=7)
    assert result.pos is None
    assert "criteria.py" in str(result.error)


def test_sgd2_multi_adapter_seeds_python_random(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated adapter runs with the same seed should reset Python random."""
    random_values: list[float] = []
    _install_fake_sgd2_modules(monkeypatch=monkeypatch, random_values=random_values)

    first = SGD2MultiRef().layout(_make_small_graph(), seed=19)
    random.seed(999)
    second = SGD2MultiRef().layout(_make_small_graph(), seed=19)

    assert first.error is None
    assert second.error is None
    assert random_values[0] == pytest.approx(random_values[1])


def test_sgd2_multi_weighted_parallel_edges_use_min_distance() -> None:
    """Weighted duplicate edges should keep the shortest parallel weight."""
    edge_index = torch.tensor([[0, 0, 1], [1, 1, 2]], dtype=torch.long)
    edge_weights = torch.tensor([5.0, 2.0, 3.0], dtype=torch.float32)

    adjacency = _build_adjacency(edge_index=edge_index, num_nodes=3, edge_weights=edge_weights)

    assert adjacency[0] == [(1, pytest.approx(2.0))]
    assert adjacency[1] == [(0, pytest.approx(2.0)), (2, pytest.approx(3.0))]


def test_sgd2_multi_crossing_only_no_pairs_matches_stress_fallback() -> None:
    """Crossing-only tiny graphs should use the reference stress fallback."""
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

    crossing_only = layout_sgd2_multi_pipeline(
        edge_index=edge_index,
        num_nodes=3,
        seed=23,
        steps=2,
        criteria={"crossings": 1.0},
        lr=0.01,
        grad_clamp=5.0,
        batch_size=8,
    )
    stress_only = layout_sgd2_multi_pipeline(
        edge_index=edge_index,
        num_nodes=3,
        seed=23,
        steps=2,
        criteria={"stress": 1.0},
        lr=0.01,
        grad_clamp=5.0,
        batch_size=8,
    )

    assert torch.allclose(crossing_only, stress_only)
