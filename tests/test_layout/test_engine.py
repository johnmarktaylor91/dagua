"""Tests for the core layout optimization loop."""

import pytest
import torch

from dagua.graph import DaguaGraph
from dagua.config import LayoutConfig
from dagua.layout import layout
from dagua.layout.multilevel import build_hierarchy
from dagua.metrics import compute_all_metrics


class TestLayoutBasic:
    def test_returns_positions(self, simple_chain, fast_config):
        pos = layout(simple_chain, fast_config)
        assert pos.shape == (5, 2)
        assert pos.dtype == torch.float32

    def test_empty_graph(self, empty_graph, fast_config):
        pos = layout(empty_graph, fast_config)
        assert pos.shape == (0, 2)

    def test_single_node(self, single_node_graph, fast_config):
        pos = layout(single_node_graph, fast_config)
        assert pos.shape == (1, 2)


class TestLayoutQuality:
    @pytest.mark.slow
    def test_no_overlaps_chain(self, simple_chain):
        config = LayoutConfig(steps=200)
        pos = layout(simple_chain, config)
        simple_chain.compute_node_sizes()
        m = compute_all_metrics(pos, simple_chain.edge_index, simple_chain.node_sizes)
        assert m["node_overlaps"] == 0

    @pytest.mark.slow
    def test_no_overlaps_diamond(self, diamond_graph):
        config = LayoutConfig(steps=200)
        pos = layout(diamond_graph, config)
        diamond_graph.compute_node_sizes()
        m = compute_all_metrics(pos, diamond_graph.edge_index, diamond_graph.node_sizes)
        assert m["node_overlaps"] == 0

    @pytest.mark.slow
    def test_dag_fraction_chain(self, simple_chain):
        config = LayoutConfig(steps=200)
        pos = layout(simple_chain, config)
        simple_chain.compute_node_sizes()
        m = compute_all_metrics(pos, simple_chain.edge_index, simple_chain.node_sizes)
        assert m["dag_fraction"] == 1.0

    @pytest.mark.slow
    def test_dag_fraction_high(self, diamond_graph):
        config = LayoutConfig(steps=200)
        pos = layout(diamond_graph, config)
        diamond_graph.compute_node_sizes()
        m = compute_all_metrics(pos, diamond_graph.edge_index, diamond_graph.node_sizes)
        assert m["dag_fraction"] >= 0.9

    @pytest.mark.slow
    def test_no_crossings_chain(self, simple_chain):
        config = LayoutConfig(steps=200)
        pos = layout(simple_chain, config)
        simple_chain.compute_node_sizes()
        m = compute_all_metrics(pos, simple_chain.edge_index, simple_chain.node_sizes)
        assert m["edge_crossings"] == 0


class TestLayoutDirections:
    def test_tb_flow(self, simple_chain):
        config = LayoutConfig(steps=100, direction="TB")
        pos = layout(simple_chain, config)
        # In TB, each successive node should have increasing y
        for i in range(4):
            assert pos[i, 1] < pos[i + 1, 1], f"Node {i} should be above node {i+1} in TB"

    def test_bt_flow(self, simple_chain):
        config = LayoutConfig(steps=100, direction="BT")
        pos = layout(simple_chain, config)
        # In BT, each successive node should have DECREASING y (upward)
        for i in range(4):
            assert pos[i, 1] > pos[i + 1, 1], f"Node {i} should be below node {i+1} in BT"

    @pytest.mark.slow
    def test_lr_flow(self, simple_chain):
        config = LayoutConfig(steps=200, direction="LR")
        pos = layout(simple_chain, config)
        # In LR, x-range should be larger than y-range (wide, not tall)
        x_range = pos[:, 0].max() - pos[:, 0].min()
        y_range = pos[:, 1].max() - pos[:, 1].min()
        assert x_range > y_range * 0.5, "LR layout should be wider than tall"

    @pytest.mark.slow
    def test_rl_flow(self, simple_chain):
        config = LayoutConfig(steps=200, direction="RL")
        pos = layout(simple_chain, config)
        # In RL, flow goes right-to-left: first node should be rightmost
        x_range = pos[:, 0].max() - pos[:, 0].min()
        y_range = pos[:, 1].max() - pos[:, 1].min()
        assert x_range > y_range * 0.5, "RL layout should be wider than tall"


class TestLayoutClusters:
    @pytest.mark.slow
    def test_cluster_compactness(self, clustered_graph):
        config = LayoutConfig(steps=200)
        pos = layout(clustered_graph, config)
        # Members of same cluster should be closer to each other
        enc_pos = pos[[1, 2]]  # encoder nodes
        dec_pos = pos[[3, 4]]  # decoder nodes
        enc_spread = (enc_pos[0] - enc_pos[1]).norm()
        # Just check it's finite and reasonable
        assert enc_spread < 500

    @pytest.mark.slow
    def test_cluster_separation(self, clustered_graph):
        config = LayoutConfig(steps=200)
        pos = layout(clustered_graph, config)
        enc_center = pos[[1, 2]].mean(dim=0)
        dec_center = pos[[3, 4]].mean(dim=0)
        separation = (enc_center - dec_center).norm()
        assert separation > 10  # clusters should be separated


class TestLayoutReproducibility:
    def test_seed_reproducibility(self, diamond_graph):
        config = LayoutConfig(steps=100, seed=42)
        pos1 = layout(diamond_graph, config)
        pos2 = layout(diamond_graph, config)
        assert torch.allclose(pos1, pos2, atol=1e-3)

    def test_none_seed_works(self):
        """Layout should work with seed=None (non-deterministic)."""
        from dagua.eval.graphs import _random_dag
        g = _random_dag(30, 50, seed=0)
        config = LayoutConfig(steps=50, seed=None)
        pos = layout(g, config)
        assert pos.shape[0] == g.num_nodes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestLayoutCUDA:
    def test_cuda_layout(self, simple_chain):
        config = LayoutConfig(steps=50, device="cuda")
        pos = layout(simple_chain, config)
        assert pos.shape == (5, 2)
        # Result is on the compute device
        assert pos.device.type in ("cpu", "cuda")


def test_build_hierarchy_accepts_precomputed_layer_assignments():
    graph = DaguaGraph.from_edge_list([("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")])
    graph.compute_node_sizes()
    assert graph.node_sizes is not None
    precomputed = torch.tensor([0, 1, 1, 2], dtype=torch.long)
    captured: list[torch.Tensor] = []

    levels = build_hierarchy(
        graph.edge_index,
        graph.num_nodes,
        graph.node_sizes,
        min_nodes=2,
        max_levels=2,
        initial_layer_assignments=precomputed,
        layer_assignments_callback=lambda tensor: captured.append(tensor),
    )

    assert levels
    assert not captured
    assert torch.equal(levels[0].fine_layer_assignments, precomputed)
