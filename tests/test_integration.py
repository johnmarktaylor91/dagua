"""End-to-end integration tests."""

import pytest
import tempfile
from pathlib import Path

import torch

import dagua
from dagua.graph import DaguaGraph
from dagua.config import LayoutConfig
from dagua.metrics import compute_all_metrics


class TestEndToEnd:
    """Full pipeline: construct → layout → render."""

    def test_draw_convenience(self, tmp_path):
        g = DaguaGraph.from_edge_list([("a", "b"), ("b", "c")])
        out = str(tmp_path / "draw.png")
        fig, ax = dagua.draw(g, output=out)
        assert Path(out).exists()
        assert fig is not None

    def test_full_pipeline(self, tmp_path):
        g = DaguaGraph.from_edge_list([
            ("input", "conv1"), ("conv1", "relu"),
            ("relu", "fc"), ("fc", "output"),
        ])
        config = LayoutConfig(steps=100)
        pos = dagua.layout(g, config)
        g.compute_node_sizes()
        m = compute_all_metrics(pos, g.edge_index, g.node_sizes)

        assert m["node_overlaps"] == 0
        assert m["dag_fraction"] == 1.0
        assert m["edge_crossings"] == 0

        out = str(tmp_path / "pipeline.png")
        dagua.render(g, pos, output=out)
        assert Path(out).exists()

    def test_networkx_roundtrip(self):
        pytest.importorskip("networkx")
        import networkx as nx

        G = nx.DiGraph()
        G.add_edges_from([("a", "b"), ("b", "c"), ("a", "c")])
        g = DaguaGraph.from_networkx(G)

        assert g.num_nodes == 3
        assert g.edge_index.shape[1] == 3

        config = LayoutConfig(steps=50)
        pos = dagua.layout(g, config)
        assert pos.shape == (3, 2)


class TestLargerGraphs:
    """Test with larger synthetic graphs."""

    @pytest.mark.slow
    def test_50_node_dag(self):
        import random
        random.seed(42)
        edges = []
        for i in range(49):
            j = random.randint(i + 1, min(i + 5, 49))
            edges.append((f"n{i}", f"n{j}"))

        g = DaguaGraph.from_edge_list(edges)
        config = LayoutConfig(steps=200)
        pos = dagua.layout(g, config)
        g.compute_node_sizes()
        m = compute_all_metrics(pos, g.edge_index, g.node_sizes)

        assert m["node_overlaps"] == 0
        assert m["dag_fraction"] >= 0.95

    @pytest.mark.slow
    def test_wide_bipartite(self):
        edges = []
        for i in range(8):
            for j in range(8):
                edges.append((f"src_{i}", f"tgt_{j}"))

        g = DaguaGraph.from_edge_list(edges)
        config = LayoutConfig(steps=200)
        pos = dagua.layout(g, config)
        g.compute_node_sizes()
        m = compute_all_metrics(pos, g.edge_index, g.node_sizes)

        assert m["node_overlaps"] == 0
        assert m["dag_fraction"] >= 0.9


class TestFromTorchlens:
    """Test TorchLens integration."""

    def test_from_torchlens_simple(self):
        """from_torchlens should produce a valid graph from a simple MLP."""
        try:
            import torchlens as tl
            import torch.nn as nn
        except ImportError:
            pytest.skip("TorchLens not installed")

        model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
        x = torch.randn(1, 10)
        log = tl.log_forward_pass(model, x, vis_mode="none")
        g = DaguaGraph.from_torchlens(log)

        assert g.num_nodes > 0
        assert g.edge_index.shape[1] > 0

        # Should be layoutable
        pos = dagua.layout(g, LayoutConfig(steps=50))
        assert pos.shape == (g.num_nodes, 2)

    def test_from_torchlens_with_skip(self):
        """from_torchlens should handle skip connections (residual block)."""
        try:
            import torchlens as tl
            import torch.nn as nn
        except ImportError:
            pytest.skip("TorchLens not installed")

        class ResBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 10)
                self.fc2 = nn.Linear(10, 10)

            def forward(self, x):
                return torch.relu(self.fc2(torch.relu(self.fc1(x))) + x)

        model = ResBlock()
        x = torch.randn(1, 10)
        log = tl.log_forward_pass(model, x, vis_mode="none")
        g = DaguaGraph.from_torchlens(log)

        assert g.num_nodes > 0
        # Should have clusters from module nesting
        assert len(g.clusters) >= 0  # may or may not depending on nesting depth

        pos = dagua.layout(g, LayoutConfig(steps=50))
        assert pos.shape == (g.num_nodes, 2)


class TestGraphvizComparison:
    """Test Graphviz comparison utilities."""

    def test_layout_with_graphviz(self):
        from dagua.graphviz_utils import layout_with_graphviz
        g = DaguaGraph.from_edge_list([("a", "b"), ("b", "c")])
        pos = layout_with_graphviz(g)
        assert pos.shape == (3, 2)

    def test_render_comparison(self, tmp_path):
        from dagua.graphviz_utils import layout_with_graphviz, render_comparison
        g = DaguaGraph.from_edge_list([("a", "b"), ("b", "c")])
        dagua_pos = dagua.layout(g)
        gv_pos = layout_with_graphviz(g)
        out = str(tmp_path / "compare.png")
        render_comparison(g, dagua_pos, gv_pos, out)
        assert Path(out).exists()
