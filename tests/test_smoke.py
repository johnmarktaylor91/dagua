"""Smoke tests for recent core changes.

Covers: ProgressContext, verbose output, num_workers config, from_edge_list
with num_nodes, _longest_path_layering_vectorized on deep DAGs, and basic
multilevel layout with verbose=True.

All tests marked @pytest.mark.smoke — run with: pytest tests/test_smoke.py -m smoke
"""

import time

import pytest
import torch

import dagua
from dagua.config import LayoutConfig
from dagua.graph import DaguaGraph
from dagua.layout.engine import ProgressContext, layout
from dagua.utils import _longest_path_layering_vectorized, longest_path_layering


@pytest.mark.smoke
class TestProgressContext:
    """ProgressContext import and basic usage."""

    def test_import(self):
        assert ProgressContext is not None

    def test_default_indent(self):
        ctx = ProgressContext()
        assert ctx.indent == "  "

    def test_custom_indent(self):
        ctx = ProgressContext(indent="    ")
        assert ctx.indent == "    "


@pytest.mark.smoke
class TestVerboseOutput:
    """Verbose output emits [dagua] prefix in both direct and multilevel paths."""

    def test_direct_layout_verbose(self, capsys):
        g = DaguaGraph.from_edge_list([("a", "b"), ("b", "c"), ("c", "d")])
        config = LayoutConfig(steps=10, verbose=True)
        dagua.layout(g, config)
        captured = capsys.readouterr()
        assert "[dagua]" in captured.out

    def test_direct_layout_verbose_reports_node_count(self, capsys):
        g = DaguaGraph.from_edge_list([("a", "b"), ("b", "c")])
        config = LayoutConfig(steps=10, verbose=True)
        dagua.layout(g, config)
        captured = capsys.readouterr()
        assert "3" in captured.out  # 3 nodes

    def test_verbose_off_is_silent(self, capsys):
        g = DaguaGraph.from_edge_list([("a", "b"), ("b", "c")])
        config = LayoutConfig(steps=10, verbose=False)
        dagua.layout(g, config)
        captured = capsys.readouterr()
        assert captured.out == ""


@pytest.mark.smoke
class TestNumWorkersConfig:
    """num_workers config field exists and defaults correctly."""

    def test_default_value(self):
        config = LayoutConfig()
        assert config.num_workers == 0

    def test_custom_value(self):
        config = LayoutConfig(num_workers=4)
        assert config.num_workers == 4

    def test_field_is_int(self):
        config = LayoutConfig()
        assert isinstance(config.num_workers, int)


@pytest.mark.smoke
class TestFromEdgeListNumNodes:
    """from_edge_list with num_nodes pre-creates nodes correctly."""

    def test_num_nodes_creates_expected_count(self):
        edges = [(0, 1), (1, 2)]
        g = DaguaGraph.from_edge_list(edges, num_nodes=5)
        # Should have 5 nodes even though edges only reference 0, 1, 2
        assert g.num_nodes == 5

    def test_num_nodes_without_edges(self):
        g = DaguaGraph.from_edge_list([], num_nodes=3)
        assert g.num_nodes == 3

    def test_num_nodes_preserves_edges(self):
        edges = [(0, 1), (1, 2), (2, 3)]
        g = DaguaGraph.from_edge_list(edges, num_nodes=4)
        assert g.num_nodes == 4
        assert g.edge_index.shape == (2, 3)

    def test_without_num_nodes_only_referenced(self):
        edges = [("x", "y")]
        g = DaguaGraph.from_edge_list(edges)
        assert g.num_nodes == 2


@pytest.mark.smoke
class TestLongestPathLayeringVectorized:
    """_longest_path_layering_vectorized handles chain graphs efficiently."""

    def test_small_chain_correct(self):
        """Verify correctness on a small chain: 0->1->2->3->4."""
        n = 5
        src = list(range(n - 1))
        tgt = list(range(1, n))
        edge_index = torch.tensor([src, tgt], dtype=torch.long)
        layers = _longest_path_layering_vectorized(edge_index, n)
        assert layers.tolist() == [0, 1, 2, 3, 4]

    def test_agrees_with_scalar_version(self):
        """Vectorized result should match the scalar BFS version."""
        n = 100
        src = list(range(n - 1))
        tgt = list(range(1, n))
        edge_index = torch.tensor([src, tgt], dtype=torch.long)
        scalar_result = longest_path_layering(edge_index, n)
        vector_result = _longest_path_layering_vectorized(edge_index, n)
        scalar_list = scalar_result.tolist() if isinstance(scalar_result, torch.Tensor) else scalar_result
        vector_list = vector_result.tolist() if isinstance(vector_result, torch.Tensor) else vector_result
        assert scalar_list == vector_list

    def test_100k_chain_completes_in_time(self):
        """Deep chain of 100K nodes must complete in <5s."""
        n = 100_000
        src = torch.arange(n - 1, dtype=torch.long)
        tgt = torch.arange(1, n, dtype=torch.long)
        edge_index = torch.stack([src, tgt])

        t0 = time.perf_counter()
        layers = _longest_path_layering_vectorized(edge_index, n)
        elapsed = time.perf_counter() - t0

        assert elapsed < 5.0, f"Took {elapsed:.2f}s, expected <5s"
        assert layers.shape[0] == n
        assert layers[0].item() == 0
        assert layers[-1].item() == n - 1

    def test_diamond_dag(self):
        """Diamond: 0->1, 0->2, 1->3, 2->3 — node 3 should be at layer 2."""
        edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]], dtype=torch.long)
        layers = _longest_path_layering_vectorized(edge_index, 4)
        assert layers[0] == 0
        assert layers[1] == 1
        assert layers[2] == 1
        assert layers[3] == 2


@pytest.mark.smoke
class TestMultilevelVerbose:
    """Basic multilevel layout works with verbose=True."""

    def test_multilevel_verbose_output(self, capsys):
        """A graph above multilevel_threshold triggers multilevel path and verbose output."""
        # Use a small threshold to force multilevel without creating a huge graph
        n = 200
        edges = [(f"n{i}", f"n{i+1}") for i in range(n - 1)]
        g = DaguaGraph.from_edge_list(edges)
        config = LayoutConfig(
            steps=10,
            verbose=True,
            multilevel_threshold=100,  # force multilevel
            multilevel_min_nodes=50,
            multilevel_coarse_steps=10,
            multilevel_refine_steps=5,
        )
        pos = dagua.layout(g, config)
        captured = capsys.readouterr()

        assert pos.shape == (n, 2)
        # Multilevel verbose should mention hierarchy and phases
        assert "[dagua]" in captured.out
        assert "hierarchy" in captured.out.lower() or "Phase" in captured.out

    def test_multilevel_produces_valid_positions(self):
        """Multilevel layout should produce finite, non-NaN positions."""
        n = 150
        edges = [(f"n{i}", f"n{i+1}") for i in range(n - 1)]
        g = DaguaGraph.from_edge_list(edges)
        config = LayoutConfig(
            steps=10,
            multilevel_threshold=100,
            multilevel_min_nodes=50,
            multilevel_coarse_steps=10,
            multilevel_refine_steps=5,
        )
        pos = dagua.layout(g, config)

        assert pos.shape == (n, 2)
        assert torch.isfinite(pos).all(), "Positions contain NaN or Inf"
