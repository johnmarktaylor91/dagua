"""Tests for Graph class — construction, ID mapping, properties."""

import pytest
import torch
import dagua

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

    def test_graph_uses_configured_storage_dtypes(self):
        import dagua

        dagua.configure(index_dtype="int32", size_dtype="float16")
        g = DaguaGraph()
        g.add_edge("a", "b")
        g.compute_node_sizes()
        assert g.edge_index.dtype == torch.int32
        assert g.node_sizes.dtype == torch.float16

    def test_per_graph_storage_dtype_override(self):
        g = DaguaGraph(index_dtype="int32", size_dtype="float64")
        g.add_edge("a", "b")
        g.compute_node_sizes()
        assert g.edge_index.dtype == torch.int32
        assert g.node_sizes.dtype == torch.float64

    def test_add_cluster(self):
        g = DaguaGraph.from_edge_list([("a", "b"), ("b", "c")])
        g.add_cluster("group1", [0, 1], label="My Group")
        assert "group1" in g.clusters
        assert g.clusters["group1"] == [0, 1]
        assert g.cluster_labels["group1"] == "My Group"

    def test_add_cluster_unknown_member_raises_by_default(self):
        g = DaguaGraph()
        with pytest.raises(KeyError, match="Unknown cluster member"):
            g.add_cluster("group1", ["missing"])

    def test_add_cluster_can_be_non_strict(self):
        g = DaguaGraph()
        g.add_cluster("group1", ["missing"], strict=False)
        assert g.clusters["group1"] == []

    def test_mutation_invalidates_cached_layout(self, fast_config):
        g = DaguaGraph.from_edge_list([("a", "b"), ("b", "c")])
        pos = dagua.layout(g, fast_config)
        assert g.has_fresh_layout
        assert g.layout_status == "fresh"
        assert g.last_positions is not None

        g.add_edge("c", "d")
        assert not g.has_fresh_layout
        assert g.layout_status == "missing"
        assert g.last_positions is None


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

    def test_respects_index_dtype_override(self):
        ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64)
        g = DaguaGraph.from_edge_index(ei, num_nodes=3, index_dtype=torch.int32)
        assert g.edge_index.dtype == torch.int32

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

    def test_size_cache_invalidates_when_label_changes(self):
        g = DaguaGraph.from_edge_list([("a", "b")])
        g.compute_node_sizes()
        width_before = g.node_sizes[0, 0].item()
        g.node_labels[0] = "a much longer label"
        g.invalidate_layout()
        g._touch()
        g.compute_node_sizes()
        assert g.node_sizes[0, 0].item() > width_before


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
        # Input type uses bluish green from Wong palette (muted fill)
        from dagua.styles import PALETTE, make_fill
        expected_fill = make_fill(PALETTE["bluish_green"])
        assert style.fill == expected_fill


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
