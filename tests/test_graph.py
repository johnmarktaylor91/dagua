"""Tests for Graph class — construction, ID mapping, properties."""

import pytest
import torch

import dagua
from dagua.graph import DaguaGraph
from dagua.styles import NodeStyle


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

    def test_edge_weights_via_add_edge(self):
        """Edge weights accumulate correctly via add_edge."""
        g = DaguaGraph()
        g.add_edge("a", "b", weight=2.0)
        g.add_edge("b", "c", weight=3.0)
        g.add_edge("c", "d")
        assert g.edge_weights is None
        assert g.edge_index.shape == (2, 3)
        assert g.edge_weights is not None
        assert g.edge_weights.shape == (3,)
        assert g.edge_weights.tolist() == [2.0, 3.0, 1.0]

    def test_edge_weights_none_when_no_weights(self):
        """edge_weights stays None when no weights are provided."""
        g = DaguaGraph()
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        _ = g.edge_index
        assert g.edge_weights is None

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
        _pos = dagua.layout(g, fast_config)
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

    def test_edge_weights_from_edge_list_3tuples(self):
        """from_edge_list with (src, tgt, weight) triples."""
        g = DaguaGraph.from_edge_list([(0, 1, 2.5), (1, 2, 0.5)], num_nodes=3)
        assert g.edge_weights is not None
        assert g.edge_weights.tolist() == [2.5, 0.5]

    def test_edge_weights_from_edge_list_mixed(self):
        """from_edge_list with mixed 2-tuples and 3-tuples."""
        g = DaguaGraph.from_edge_list([(0, 1), (1, 2, 3.0)], num_nodes=3)
        assert g.edge_weights is not None
        assert g.edge_weights.tolist() == [1.0, 3.0]

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

    def test_edge_weights_from_edge_index(self):
        """from_edge_index with explicit edge_weights tensor."""
        ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        ew = torch.tensor([2.0, 3.0])
        g = DaguaGraph.from_edge_index(ei, num_nodes=3, edge_weights=ew)
        assert g.edge_weights is not None
        assert g.edge_weights.tolist() == [2.0, 3.0]

    def test_edge_weights_from_edge_index_validation(self):
        """from_edge_index rejects mismatched edge_weights."""
        ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        ew = torch.tensor([2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match="edge_weights length"):
            DaguaGraph.from_edge_index(ei, num_nodes=3, edge_weights=ew)

    def test_respects_index_dtype_override(self):
        ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64)
        g = DaguaGraph.from_edge_index(ei, num_nodes=3, index_dtype=torch.int32)
        assert g.edge_index.dtype == torch.int32

    def test_with_labels(self):
        ei = torch.tensor([[0, 1], [1, 2]])
        g = DaguaGraph.from_edge_index(ei, num_nodes=3)
        g.node_labels = ["A", "B", "C"]
        assert g.node_labels == ["A", "B", "C"]


def test_edge_weights_from_networkx():
    """from_networkx extracts weight edge attribute."""
    import networkx as nx

    graph = nx.Graph()
    graph.add_edge(0, 1, weight=2.0)
    graph.add_edge(1, 2, weight=5.0)
    graph.add_edge(2, 3)
    g = DaguaGraph.from_networkx(graph)
    assert g.edge_weights is not None
    assert g.edge_weights.shape[0] == g.edge_index.shape[1]
    weights_set = set(g.edge_weights.tolist())
    assert 2.0 in weights_set
    assert 5.0 in weights_set


def test_edge_weights_backfill():
    """Weights backfill to 1.0 for pre-existing unweighted edges."""
    g = DaguaGraph()
    g.add_edge("a", "b")
    _ = g.edge_index
    g.add_edge("b", "c", weight=5.0)
    _ = g.edge_index
    assert g.edge_weights is not None
    assert g.edge_weights.shape[0] == 2
    assert g.edge_weights[0].item() == 1.0
    assert g.edge_weights[1].item() == 5.0


class TestNodeSizes:
    def test_compute_node_sizes(self, simple_chain):
        simple_chain.compute_node_sizes()
        assert simple_chain.node_sizes.shape == (5, 2)
        assert (simple_chain.node_sizes > 0).all()

    def test_compute_node_sizes_no_labels(self):
        """Regression: compute_node_sizes must not produce 1D tensor for labelless graphs."""
        g = dagua.DaguaGraph()
        g.num_nodes = 100
        g._edge_index_tensor = torch.zeros(2, 0, dtype=torch.long)
        g.node_sizes = torch.full((100, 2), 20.0)
        g.compute_node_sizes()
        assert g.node_sizes.ndim == 2
        assert g.node_sizes.shape == (100, 2)

    def test_compute_node_sizes_partial_labels(self):
        """compute_node_sizes handles graphs with fewer labels than nodes."""
        g = dagua.DaguaGraph()
        g.add_node("a", label="hello")
        g.add_node("b", label="world")
        g.num_nodes = 5
        g.compute_node_sizes()
        assert g.node_sizes.ndim == 2
        assert g.node_sizes.shape == (5, 2)

    def test_sizes_reflect_label_width(self):
        g = DaguaGraph.from_edge_list([("short", "a_very_long_label_here")])
        g.node_styles = [
            NodeStyle(overflow_policy="expand_node"),
            NodeStyle(overflow_policy="expand_node"),
        ]
        g.compute_node_sizes()
        # Longer label should produce wider node
        assert g.node_sizes[1, 0] > g.node_sizes[0, 0]

    def test_size_cache_invalidates_when_label_changes(self):
        g = DaguaGraph.from_edge_list([("a", "b")])
        g.node_styles = [
            NodeStyle(overflow_policy="expand_node"),
            NodeStyle(overflow_policy="expand_node"),
        ]
        g.compute_node_sizes()
        width_before = g.node_sizes[0, 0].item()
        g.node_labels[0] = "a much longer label"
        g.invalidate_layout()
        g._touch()
        g.compute_node_sizes()
        assert g.node_sizes[0, 0].item() > width_before

    def test_min_height_style_floors_computed_height(self):
        g = DaguaGraph.from_edge_list([("a", "b")])
        g.node_styles[0] = NodeStyle(min_height=48.0)
        g.compute_node_sizes()
        assert g.node_sizes[0, 1].item() >= 48.0

    def test_shrink_text_fits_constrained_ellipse(self) -> None:
        """shrink_text should reduce the font until ellipse text fits the fixed node."""
        from dagua.utils import CURVED_SHAPE_INSCRIBE_FACTOR, measure_text

        graph = DaguaGraph()
        graph.add_node(
            "a",
            label="Shrink this ellipse label aggressively",
            style=NodeStyle(
                shape="ellipse",
                font_size=18.0,
                padding=(6.0, 6.0),
                min_width=150.0,
                min_height=56.0,
                overflow_policy="shrink_text",
                min_font_size=3.0,
            ),
        )

        graph.compute_node_sizes()

        style = graph.get_style_for_node(0)
        node_width = float(graph.node_sizes[0, 0].item())
        node_height = float(graph.node_sizes[0, 1].item())
        fitted_font_size = float(graph.node_font_sizes[0].item())
        text_width, text_height = measure_text(
            graph.node_labels[0],
            style.font_family,
            fitted_font_size,
            style.font_weight,
            style.font_style,
        )

        assert fitted_font_size < style.font_size
        assert (text_width + style.padding[0] * 2.0) * CURVED_SHAPE_INSCRIBE_FACTOR <= (
            node_width + 1e-6
        )
        assert (text_height + style.padding[1] * 2.0) * CURVED_SHAPE_INSCRIBE_FACTOR <= (
            node_height + 1e-6
        )

    def test_expand_node_grows_ellipse_for_full_size_text(self) -> None:
        """expand_node should enlarge ellipse bounds instead of shrinking the label."""
        from dagua.utils import CURVED_SHAPE_INSCRIBE_FACTOR, measure_text

        graph = DaguaGraph()
        graph.add_node(
            "a",
            label="Expand this ellipse label without shrinking",
            style=NodeStyle(
                shape="ellipse",
                font_size=16.0,
                padding=(6.0, 6.0),
                min_width=90.0,
                min_height=42.0,
                overflow_policy="expand_node",
            ),
        )

        graph.compute_node_sizes()

        style = graph.get_style_for_node(0)
        node_width = float(graph.node_sizes[0, 0].item())
        node_height = float(graph.node_sizes[0, 1].item())
        fitted_font_size = float(graph.node_font_sizes[0].item())
        text_width, text_height = measure_text(
            graph.node_labels[0],
            style.font_family,
            fitted_font_size,
            style.font_weight,
            style.font_style,
        )

        assert fitted_font_size == pytest.approx(style.font_size)
        assert node_width > float(style.min_width or 0.0)
        assert (text_width + style.padding[0] * 2.0) * CURVED_SHAPE_INSCRIBE_FACTOR <= (
            node_width + 1e-6
        )
        assert (text_height + style.padding[1] * 2.0) * CURVED_SHAPE_INSCRIBE_FACTOR <= (
            node_height + 1e-6
        )


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
        from dagua.styles import GRAPHVIZ_THEME

        expected_fill = GRAPHVIZ_THEME.get_node_style("input").fill
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
