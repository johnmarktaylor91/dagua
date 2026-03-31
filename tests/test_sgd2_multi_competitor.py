"""Regression tests for the reference (SGD)^2 multicriteria adapter."""

from __future__ import annotations

import sys
import types

import networkx as nx
import pytest
import torch

from dagua.eval.competitors.sgd2_multi_competitor import SGD2MultiRef, _compat_criteria_patches
from dagua.graph import DaguaGraph


def _make_small_graph() -> DaguaGraph:
    """Create a connected graph for multicriteria adapter tests.

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


def test_compat_criteria_patch_handles_empty_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sampling zero edges should return a scalar loss instead of crashing."""
    criteria_module = types.ModuleType("criteria")
    criteria_module.ideal_edge_length = lambda *args, **kwargs: None
    criteria_module.stress = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "criteria", criteria_module)

    with _compat_criteria_patches():
        pos = torch.zeros((2, 2), dtype=torch.float32)
        graph = nx.Graph()
        graph.add_edge(0, 1)
        loss = criteria_module.ideal_edge_length(pos, graph, {0: 0, 1: 1}, sampleSize=0)

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])
    assert loss.item() == 0.0


def test_sgd2_multi_reports_nan_divergence(monkeypatch: pytest.MonkeyPatch) -> None:
    """NaN positions from the upstream optimizer should become a clean error."""
    criteria_module = types.ModuleType("criteria")
    criteria_module.ideal_edge_length = lambda *args, **kwargs: None
    criteria_module.stress = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "criteria", criteria_module)

    observed: dict[str, object] = {}

    class FakeGD2:
        """Minimal upstream GD2 stub for adapter regression coverage."""

        def __init__(self, graph: nx.Graph) -> None:
            del graph
            self.pos = torch.full((3, 2), float("nan"), dtype=torch.float32)

        def optimize(self, **kwargs: object) -> None:
            """Capture optimize kwargs without doing any work."""
            observed.update(kwargs)

    gd2_module = types.ModuleType("gd2")
    gd2_module.GD2 = FakeGD2
    monkeypatch.setitem(sys.modules, "gd2", gd2_module)

    result = SGD2MultiRef().layout(_make_small_graph(), seed=7)

    assert result.pos is None
    assert result.error == "optimization diverged (NaN positions)"
    assert observed["grad_clamp"] == 5.0


def test_sgd2_multi_forwards_variant_grad_clamp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Variant-provided grad clamp should reach the upstream optimizer."""
    criteria_module = types.ModuleType("criteria")
    criteria_module.ideal_edge_length = lambda *args, **kwargs: None
    criteria_module.stress = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "criteria", criteria_module)

    observed: dict[str, object] = {}

    class FakeGD2:
        """Minimal upstream GD2 stub that captures optimize kwargs."""

        def __init__(self, graph: nx.Graph) -> None:
            del graph
            self.pos = torch.zeros((3, 2), dtype=torch.float32)

        def optimize(self, **kwargs: object) -> None:
            """Record forwarded optimize kwargs."""
            observed.update(kwargs)

    gd2_module = types.ModuleType("gd2")
    gd2_module.GD2 = FakeGD2
    monkeypatch.setitem(sys.modules, "gd2", gd2_module)

    result = SGD2MultiRef().layout_with_variant(
        _make_small_graph(),
        seed=7,
        variant_params={"grad_clamp": 7.0},
    )

    assert result.error is None
    assert result.pos is not None
    assert observed["grad_clamp"] == 7.0
