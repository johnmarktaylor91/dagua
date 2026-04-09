"""Regression tests for the reference (SGD)^2 multicriteria adapter."""

from __future__ import annotations

import sys
import types

import networkx as nx
import pytest
import torch

from dagua.eval.competitors.sgd2_multi_competitor import (
    SGD2MultiRef,
    _compat_criteria_patches,
    _sgd2_multi_available,
)
from dagua.eval.graphs import TestGraph, get_test_graphs
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
            self.non_incident_edge_pairs = [(0, 1, 1, 2)]
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
            self.non_incident_edge_pairs = [(0, 1, 1, 2)]
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


@pytest.fixture(scope="module")
def test_graphs() -> dict[str, TestGraph]:
    """Return evaluation graphs keyed by name for SGD2 wrapper regressions.

    Returns
    -------
    dict[str, TestGraph]
        Cached evaluation graphs keyed by graph name.
    """
    return {graph.name: graph for graph in get_test_graphs()}


requires_sgd2_repo = pytest.mark.skipif(
    not _sgd2_multi_available(),
    reason="SGD2 reference repo not cloned at /tmp/graph-drawing",
)


@requires_sgd2_repo
class TestSGD2WrapperBugA:
    """Regression coverage for the undirected edge construction fix."""

    @pytest.mark.parametrize(
        "graph_name",
        [
            "recurrent_feedback_cell",
            "bipartite_4_3_4",
            "hub_and_spoke_3x20",
            "hub_spoke_5x50",
        ],
    )
    def test_small_graph_does_not_nan(
        self,
        test_graphs: dict[str, TestGraph],
        graph_name: str,
    ) -> None:
        """Stress-only SGD2 should not diverge on small graphs with reverse edges.

        Parameters
        ----------
        test_graphs : dict[str, TestGraph]
            Evaluation graphs keyed by name.
        graph_name : str
            Graph expected to remain connected after NetworkX construction.

        Returns
        -------
        None
            Pytest assertion helper.
        """
        graph = test_graphs[graph_name]
        result = SGD2MultiRef().layout_with_variant(
            graph.graph,
            timeout=60.0,
            seed=42,
            variant_params={
                "criteria_weights": {"stress": 1.0},
                "max_iter": 100,
                "optimizer_kwargs": {"lr": 0.01},
                "sample_sizes": {"stress": 16},
            },
        )
        assert result.pos is not None, f"Layout failed: {result.error}"
        assert not torch.isnan(result.pos).any(), "Layout produced NaN positions"
        assert result.pos.shape == (graph.graph.num_nodes, 2)


@requires_sgd2_repo
class TestSGD2WrapperBugB:
    """Regression coverage for empty crossing-sample handling."""

    def test_parallel_multiedge_bundle_with_crossings_variant(
        self,
        test_graphs: dict[str, TestGraph],
    ) -> None:
        """Crossing criteria should be stripped when no non-incident edges exist.

        Parameters
        ----------
        test_graphs : dict[str, TestGraph]
            Evaluation graphs keyed by name.

        Returns
        -------
        None
            Pytest assertion helper.
        """
        graph = test_graphs["parallel_multiedge_bundle"]
        result = SGD2MultiRef().layout_with_variant(
            graph.graph,
            timeout=60.0,
            seed=42,
            variant_params={
                "criteria_weights": {"stress": 1.0, "crossings": 0.5},
                "max_iter": 10,
                "optimizer_kwargs": {"lr": 0.01},
                "sample_sizes": {"stress": 128, "crossings": 128},
            },
        )
        assert result.error is None, (
            f"Expected success after stripping crossings, got error: {result.error}"
        )
        assert result.pos is not None
        assert result.pos.shape == (graph.graph.num_nodes, 2)

    def test_only_crossings_on_tiny_graph_falls_back_to_stress(
        self,
        test_graphs: dict[str, TestGraph],
    ) -> None:
        """Crossings-only requests should fall back to stress on tiny graphs.

        Parameters
        ----------
        test_graphs : dict[str, TestGraph]
            Evaluation graphs keyed by name.

        Returns
        -------
        None
            Pytest assertion helper.
        """
        graph = test_graphs["parallel_multiedge_bundle"]
        result = SGD2MultiRef().layout_with_variant(
            graph.graph,
            timeout=60.0,
            seed=42,
            variant_params={
                "criteria_weights": {"crossings": 1.0},
                "max_iter": 10,
                "optimizer_kwargs": {"lr": 0.01},
                "sample_sizes": {"crossings": 128},
            },
        )
        assert result.error is None, f"Expected stress fallback, got error: {result.error}"
        assert result.pos is not None
        assert result.pos.shape == (graph.graph.num_nodes, 2)
