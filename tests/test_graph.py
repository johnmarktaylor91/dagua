"""Tests for Graph class — construction, ID mapping, properties."""

import pytest
import torch

from dagua.graph import DaguaGraph
from dagua.styles import NodeStyle, EdgeStyle


class TestGraphConstruction:
    def test_empty_graph(self):
        g = DaguaGraph()
        assert g.num_nodes == 0
        assert g.edge_index.numel() == 0

    def test_add_node(self):
        g = DaguaGraph()
        g.add_node("a")
        g.add_node("b", label="Node B")
        assert g.num_nodes == 2
        assert g.node_labels[0] == "a"
        assert g.node_labels[1] == "Node B"

    def test_add_edge(self):
        g = DaguaGraph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b")
        assert g.edge_index.shape == (2, 1)
        assert g.edge_index[0, 0].item() == 0
        assert g.edge_index[1, 0].item() == 1

    def test_add_edge_auto_creates_nodes(self):
        g = DaguaGraph()
        g.add_edge("x", "y")
        assert g.num_nodes == 2
        assert "x" in g._id_to_index
        assert "y" in g._id_to_index

    def test_add_cluster(self):
        g = DaguaGraph.from_edge_list([("a", "b"), ("b", "c")])
        g.add_cluster("group1", [0, 1], label="My Group")
        assert "group1" in g.clusters
        assert g.clusters["group1"] == [0, 1]
        assert g.cluster_labels["group1"] == "My Group"


class TestFromEdgeList:
    def test_basic(self):
        g = DaguaGraph.from_edge_list([("a", "b"), ("b", "c")])
        assert g.num_nodes == 3
        assert g.edge_index.shape == (2, 2)

    def test_preserves_order(self):
        edges = [("x", "y"), ("y", "z"), ("x", "z")]
        g = DaguaGraph.from_edge_list(edges)
        assert g.node_labels[:3] == ["x", "y", "z"]

    def test_no_duplicate_nodes(self):
        g = DaguaGraph.from_edge_list([("a", "b"), ("b", "c"), ("a", "c")])
        assert g.num_nodes == 3


class TestFromEdgeIndex:
    def test_basic(self):
        ei = torch.tensor([[0, 1], [1, 2]])
        g = DaguaGraph.from_edge_index(ei, num_nodes=3)
        assert g.num_nodes == 3
        assert g.edge_index.shape == (2, 2)

    def test_with_labels(self):
        ei = torch.tensor([[0, 1], [1, 2]])
        g = DaguaGraph.from_edge_index(ei, num_nodes=3)
        g.node_labels = ["A", "B", "C"]
        assert g.node_labels == ["A", "B", "C"]


class TestNodeSizes:
    def test_compute_node_sizes(self, simple_chain):
        simple_chain.compute_node_sizes()
        assert simple_chain.node_sizes.shape == (5, 2)
        assert (simple_chain.node_sizes > 0).all()

    def test_sizes_reflect_label_width(self):
        g = DaguaGraph.from_edge_list([("short", "a_very_long_label_here")])
        g.compute_node_sizes()
        # Longer label should produce wider node
        assert g.node_sizes[1, 0] > g.node_sizes[0, 0]


class TestStyles:
    def test_default_style(self, simple_chain):
        style = simple_chain.get_style_for_node(0)
        assert isinstance(style, NodeStyle)
        assert style.fill is not None

    def test_custom_node_style(self):
        g = DaguaGraph()
        g.add_node("a")
        g.node_styles[0] = NodeStyle(fill="#ff0000")
        style = g.get_style_for_node(0)
        assert style.fill == "#ff0000"

    def test_node_type_styling(self):
        g = DaguaGraph()
        g.add_node("inp")
        g.node_types[-1] = "input"  # override the default type
        style = g.get_style_for_node(0)
        assert style.fill == "#98FB98"  # default input theme color


class TestDeviceTransfer:
    def test_to_cpu(self, simple_chain):
        simple_chain.compute_node_sizes()
        g = simple_chain.to("cpu")
        assert g.edge_index.device.type == "cpu"
        assert g.node_sizes.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_cuda(self, simple_chain):
        simple_chain.compute_node_sizes()
        g = simple_chain.to("cuda")
        assert g.edge_index.device.type == "cuda"
