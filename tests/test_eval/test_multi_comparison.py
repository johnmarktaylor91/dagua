"""Tests for multi-engine comparison infrastructure."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import dagua.eval.competitors as competitor_registry
from dagua.eval.compare import MultiComparisonResult, compare_engines, layout_all
from dagua.eval.graphs import TestGraph, get_test_graphs
from dagua.graph import DaguaGraph


def _make_small_graph() -> DaguaGraph:
    """Create a small graph that all lightweight engines can handle.

    Returns
    -------
    DaguaGraph
        Graph with computed node sizes.
    """
    graph = DaguaGraph.from_edge_list(
        [
            ("a", "b"),
            ("b", "c"),
            ("c", "d"),
            ("a", "d"),
        ]
    )
    graph.compute_node_sizes()
    if graph.node_sizes is None:
        raise AssertionError("node sizes should be available after compute_node_sizes()")
    return graph


class TestMultiComparisonResult:
    def test_winner_auto_selected(self):
        r = MultiComparisonResult(
            graph_name="test",
            engine_metrics={
                "dagua": {"overall_quality": 80.0},
                "graphviz": {"overall_quality": 70.0},
            },
            engine_positions={
                "dagua": torch.randn(5, 2),
                "graphviz": torch.randn(5, 2),
            },
        )
        assert r.winner == "dagua"

    def test_winner_with_single_engine(self):
        r = MultiComparisonResult(
            graph_name="test",
            engine_metrics={"dagua": {"overall_quality": 50.0}},
            engine_positions={"dagua": torch.randn(5, 2)},
        )
        assert r.winner == "dagua"

    def test_empty_metrics(self):
        r = MultiComparisonResult(
            graph_name="test",
            engine_metrics={},
            engine_positions={},
        )
        assert r.winner == ""


class TestCompareEngines:
    def test_layout_all_returns_positions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """layout_all should return an engine-to-positions mapping.

        Returns
        -------
        None
            This test asserts on the returned mapping contents.
        """
        graph = _make_small_graph()
        monkeypatch.setattr(
            competitor_registry,
            "get_available_competitors",
            lambda: [
                SimpleNamespace(
                    name="dagua",
                    max_nodes=100,
                    layout=lambda candidate_graph, timeout=300.0: SimpleNamespace(
                        pos=torch.zeros((candidate_graph.num_nodes, 2))
                    ),
                )
            ],
        )
        results = layout_all(graph)
        assert "dagua" in results
        assert results["dagua"] is not None
        assert tuple(results["dagua"].shape) == (graph.num_nodes, 2)

    def test_compare_returns_pairwise_similarity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """compare_engines should populate pairwise similarity for valid outputs.

        Returns
        -------
        None
            This test asserts on the pairwise similarity payload.
        """
        graph = _make_small_graph()
        test_graph = TestGraph(name="stub_graph", graph=graph)
        pos_a = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5], [3.0, 1.5]], dtype=torch.float32)
        pos_b = torch.tensor([[0.1, 0.0], [1.1, 1.1], [1.9, 0.4], [2.9, 1.4]], dtype=torch.float32)
        monkeypatch.setattr(
            competitor_registry,
            "get_available_competitors",
            lambda: [
                SimpleNamespace(
                    name="dagua",
                    max_nodes=100,
                    layout=lambda candidate_graph, timeout=300.0: SimpleNamespace(
                        pos=pos_a.clone()
                    ),
                ),
                SimpleNamespace(
                    name="graphviz_dot",
                    max_nodes=100,
                    layout=lambda candidate_graph, timeout=300.0: SimpleNamespace(
                        pos=pos_b.clone()
                    ),
                ),
            ],
        )

        results = compare_engines(graphs=[test_graph], max_nodes=100)
        assert len(results) == 1
        assert ("dagua", "graphviz_dot") in results[0].pairwise_similarity
        assert (
            results[0].pairwise_similarity[("dagua", "graphviz_dot")]["procrustes_similarity"] > 0.9
        )

    def test_compare_returns_results(self) -> None:
        """compare_engines should return structured per-graph results.

        Returns
        -------
        None
            This test asserts on the returned comparison payload.
        """
        graphs = get_test_graphs(max_nodes=50)[:2]
        if not graphs:
            pytest.skip("No small test graphs available")
        results = compare_engines(graphs=graphs, max_nodes=50)
        assert isinstance(results, list)
        # Should have at least 1 result if dagua competitor is available
        if results:
            r = results[0]
            assert isinstance(r, MultiComparisonResult)
            assert r.graph_name
            assert len(r.engine_metrics) >= 1
            valid_positions = [pos for pos in r.engine_positions.values() if pos is not None]
            if len(valid_positions) >= 2:
                assert r.pairwise_similarity
                first_similarity = next(iter(r.pairwise_similarity.values()))
                assert "procrustes_similarity" in first_similarity

    def test_compare_with_output_dir(self):
        graphs = get_test_graphs(max_nodes=30)[:1]
        if not graphs:
            pytest.skip("No small test graphs available")
        with tempfile.TemporaryDirectory() as tmpdir:
            results = compare_engines(graphs=graphs, output_dir=tmpdir, max_nodes=30)
            # Check that image files were created (if >1 engine available)
            if results and len(results[0].engine_positions) > 1:
                img_files = list(Path(tmpdir).glob("multi_*.png"))
                assert len(img_files) >= 1

    def test_compare_with_engine_filter(self):
        graphs = get_test_graphs(max_nodes=30)[:1]
        if not graphs:
            pytest.skip("No small test graphs available")
        results = compare_engines(graphs=graphs, engines=["dagua"], max_nodes=30)
        if results:
            assert all("dagua" in r.engine_metrics for r in results if r.engine_metrics)


class TestRenderMultiComparison:
    def test_render_two_engines(self):
        pytest.importorskip("matplotlib")
        from dagua.graph import DaguaGraph
        from dagua.graphviz_utils import render_multi_comparison

        g = DaguaGraph()
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.compute_node_sizes()

        pos1 = torch.tensor([[0.0, 0.0], [50.0, 50.0], [100.0, 100.0]])
        pos2 = torch.tensor([[10.0, 0.0], [40.0, 60.0], [90.0, 110.0]])

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name

        try:
            result = render_multi_comparison(g, {"Engine A": pos1, "Engine B": pos2}, path)
            assert Path(result).exists()
            assert Path(result).stat().st_size > 0
        finally:
            Path(path).unlink(missing_ok=True)


class TestPrintMultiComparisonTable:
    def test_print_table(self, capsys):
        from dagua.eval.compare import print_multi_comparison_table

        results = [
            MultiComparisonResult(
                graph_name="diamond",
                engine_metrics={
                    "dagua": {"overall_quality": 80.0},
                    "graphviz": {"overall_quality": 70.0},
                },
                engine_positions={},
            ),
        ]
        print_multi_comparison_table(results)
        captured = capsys.readouterr()
        assert "diamond" in captured.out
        assert "dagua" in captured.out

    def test_empty_results(self, capsys):
        from dagua.eval.compare import print_multi_comparison_table

        print_multi_comparison_table([])
        captured = capsys.readouterr()
        assert "No comparison results" in captured.out
