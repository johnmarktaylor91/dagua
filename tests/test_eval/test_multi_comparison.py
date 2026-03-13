"""Tests for multi-engine comparison infrastructure."""

import tempfile
from pathlib import Path

import pytest
import torch

from dagua.eval.compare import MultiComparisonResult, compare_engines
from dagua.eval.graphs import get_test_graphs


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
    def test_compare_returns_results(self):
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

    def test_compare_with_output_dir(self):
        graphs = get_test_graphs(max_nodes=30)[:1]
        if not graphs:
            pytest.skip("No small test graphs available")
        with tempfile.TemporaryDirectory() as tmpdir:
            results = compare_engines(
                graphs=graphs, output_dir=tmpdir, max_nodes=30
            )
            # Check that image files were created (if >1 engine available)
            if results and len(results[0].engine_positions) > 1:
                img_files = list(Path(tmpdir).glob("multi_*.png"))
                assert len(img_files) >= 1

    def test_compare_with_engine_filter(self):
        graphs = get_test_graphs(max_nodes=30)[:1]
        if not graphs:
            pytest.skip("No small test graphs available")
        results = compare_engines(
            graphs=graphs, engines=["dagua"], max_nodes=30
        )
        if results:
            assert all(
                "dagua" in r.engine_metrics
                for r in results
                if r.engine_metrics
            )


class TestRenderMultiComparison:
    def test_render_two_engines(self):
        pytest.importorskip("matplotlib")
        from dagua.graphviz_utils import render_multi_comparison
        from dagua.graph import DaguaGraph

        g = DaguaGraph()
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.compute_node_sizes()

        pos1 = torch.tensor([[0.0, 0.0], [50.0, 50.0], [100.0, 100.0]])
        pos2 = torch.tensor([[10.0, 0.0], [40.0, 60.0], [90.0, 110.0]])

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name

        try:
            result = render_multi_comparison(
                g, {"Engine A": pos1, "Engine B": pos2}, path
            )
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
