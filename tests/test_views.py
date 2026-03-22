"""Tests for lightweight graph view objects and navigation helpers."""

from __future__ import annotations

import pytest
import torch

from dagua.graph import DaguaGraph
from dagua.styles import ClusterStyle, EdgeStyle, NodeStyle
from dagua.views import ClusterView, EdgeView, NodeView


@pytest.fixture
def sample_graph() -> DaguaGraph:
    """Build a small graph with cached sizes, layout, and a cluster.

    Returns
    -------
    DaguaGraph
        Sample graph used by the view tests.
    """
    graph = DaguaGraph()
    graph.add_node("a", label="Node A")
    graph.add_node("b", label="Node B")
    graph.add_node("c", label="Node C")
    graph.add_edge("a", "b", weight=2.0)
    graph.add_edge("b", "c")
    graph.add_edge("a", "c", label="skip")
    graph.add_cluster("group1", ["a", "b"])
    graph.compute_node_sizes()
    positions = torch.tensor([[0.0, 40.0], [-30.0, 0.0], [30.0, 0.0]], dtype=torch.float32)
    graph.cache_layout(positions)
    return graph


class TestNodeView:
    """Node view behavior and navigation tests."""

    def test_basic_properties(self, sample_graph: DaguaGraph) -> None:
        """Node views expose identity, index, and label data."""
        node = sample_graph.node("a")
        assert isinstance(node, NodeView)
        assert node.label == "Node A"
        assert node.index == 0
        assert node.id == "a"

    def test_degree(self, sample_graph: DaguaGraph) -> None:
        """Degree helpers reflect the finalized edge index."""
        node = sample_graph.node("a")
        assert node.out_degree == 2
        assert node.in_degree == 0
        assert node.degree == 2

    def test_edges(self, sample_graph: DaguaGraph) -> None:
        """Connected edge lists split incoming and outgoing directions."""
        node = sample_graph.node("a")
        assert len(node.outgoing_edges) == 2
        assert len(node.incoming_edges) == 0
        assert len(node.edges) == 2

    def test_neighbors(self, sample_graph: DaguaGraph) -> None:
        """Neighbor lookup ignores direction and removes duplicates."""
        node = sample_graph.node("a")
        neighbor_labels = {neighbor.label for neighbor in node.neighbors}
        assert neighbor_labels == {"Node B", "Node C"}

    def test_position(self, sample_graph: DaguaGraph) -> None:
        """Cached layout positions are exposed as Python tuples."""
        node = sample_graph.node("a")
        position = node.position
        assert position is not None
        assert abs(position[0] - 0.0) < 0.1

    def test_clusters(self, sample_graph: DaguaGraph) -> None:
        """Node views can navigate to containing clusters."""
        node = sample_graph.node("a")
        assert len(node.clusters) == 1
        assert node.clusters[0].name == "group1"

    def test_repr(self, sample_graph: DaguaGraph) -> None:
        """The node repr surfaces high-signal fields only."""
        rendered = repr(sample_graph.node("a"))
        assert "Node A" in rendered
        assert "degree=2" in rendered

    def test_equality(self, sample_graph: DaguaGraph) -> None:
        """Views compare equal when they reference the same graph slot."""
        first = sample_graph.node("a")
        second = sample_graph.node("a")
        assert first == second
        assert first != sample_graph.node("b")


class TestEdgeView:
    """Edge view behavior tests."""

    def test_source_target(self, sample_graph: DaguaGraph) -> None:
        """Edge views navigate to source and target nodes."""
        edge = sample_graph.edge(0)
        assert edge.source.label == "Node A"
        assert edge.target.label == "Node B"

    def test_weight(self, sample_graph: DaguaGraph) -> None:
        """Missing edge weights backfill to 1.0 once weights are present."""
        edge = sample_graph.edge(0)
        assert edge.weight == 2.0
        second_edge = sample_graph.edge(1)
        assert second_edge.weight == 1.0

    def test_label(self, sample_graph: DaguaGraph) -> None:
        """Edge labels are exposed when present."""
        edge = sample_graph.edge(2)
        assert edge.label == "skip"

    def test_repr(self, sample_graph: DaguaGraph) -> None:
        """The edge repr includes endpoints and notable metadata."""
        rendered = repr(sample_graph.edge(0))
        assert "Node A" in rendered
        assert "Node B" in rendered
        assert "weight" in rendered


class TestClusterView:
    """Cluster view behavior tests."""

    def test_members(self, sample_graph: DaguaGraph) -> None:
        """Cluster views expose member nodes."""
        cluster = sample_graph.cluster("group1")
        assert isinstance(cluster, ClusterView)
        assert cluster.member_count == 2
        member_labels = {member.label for member in cluster.members}
        assert member_labels == {"Node A", "Node B"}

    def test_parent_and_children(self) -> None:
        """Cluster views follow the hierarchy helpers on the graph."""
        graph = DaguaGraph.from_edge_list([("a", "b"), ("b", "c")])
        graph.add_cluster("outer", ["a", "b", "c"])
        graph.add_cluster("inner", ["b", "c"], parent="outer")

        outer = graph.cluster("outer")
        inner = graph.cluster("inner")

        assert outer.children == [inner]
        assert inner.parent == outer
        assert inner.depth == 1

    def test_repr(self, sample_graph: DaguaGraph) -> None:
        """The cluster repr reports member counts compactly."""
        rendered = repr(sample_graph.cluster("group1"))
        assert "group1" in rendered
        assert "members=2" in rendered


class TestGraphNavigation:
    """Graph navigation helper tests."""

    def test_getitem(self, sample_graph: DaguaGraph) -> None:
        """Indexing by node ID returns a node view."""
        node = sample_graph["a"]
        assert isinstance(node, NodeView)
        assert node.label == "Node A"

    def test_node_by_index(self, sample_graph: DaguaGraph) -> None:
        """Integer indices are accepted when no integer node ID matches."""
        node = sample_graph.node(0)
        assert node.label == "Node A"

    def test_num_edges(self, sample_graph: DaguaGraph) -> None:
        """The edge count includes pending edges without finalization."""
        assert sample_graph.num_edges == 3

    def test_contains(self, sample_graph: DaguaGraph) -> None:
        """Membership checks use user-facing node IDs."""
        assert "a" in sample_graph
        assert "nonexistent" not in sample_graph

    def test_len(self, sample_graph: DaguaGraph) -> None:
        """The graph length is its node count."""
        assert len(sample_graph) == 3

    def test_node_id_reverse_o1(self, sample_graph: DaguaGraph) -> None:
        """Reverse node ID lookups use the O(1) index-to-ID mapping."""
        assert sample_graph.node_id(0) == "a"
        assert sample_graph.node_id(1) == "b"

    def test_nodes_property(self, sample_graph: DaguaGraph) -> None:
        """The primary node iterator lives on ``graph.nodes``."""
        nodes = list(sample_graph.nodes)
        assert len(nodes) == 3
        assert all(isinstance(node, NodeView) for node in nodes)

    def test_edges_property(self, sample_graph: DaguaGraph) -> None:
        """The primary edge iterator lives on ``graph.edges``."""
        edges = list(sample_graph.edges)
        assert len(edges) == 3
        assert all(isinstance(edge, EdgeView) for edge in edges)

    def test_clusters_view(self, sample_graph: DaguaGraph) -> None:
        """The cluster iterator lives on ``graph.clusters_view``."""
        clusters = list(sample_graph.clusters_view)
        assert len(clusters) == 1
        assert all(isinstance(cluster, ClusterView) for cluster in clusters)

    def test_iterator_aliases(self, sample_graph: DaguaGraph) -> None:
        """Deprecated iterator aliases still resolve to the new iterators."""
        assert len(list(sample_graph.nodes_iter)) == 3
        assert len(list(sample_graph.edges_iter)) == 3
        assert len(list(sample_graph.edges_view)) == 3
        assert len(list(sample_graph.clusters_iter)) == 1

    def test_edges_for_node(self, sample_graph: DaguaGraph) -> None:
        """Graph helpers delegate edge queries through node views."""
        assert len(sample_graph.edges_for_node("a")) == 2

    def test_edges_between(self, sample_graph: DaguaGraph) -> None:
        """Directed edge lookup returns matching edge views only."""
        edges = sample_graph.edges_between("a", "b")
        assert len(edges) == 1
        assert edges[0].source.id == "a"

    def test_clusters_for_node(self, sample_graph: DaguaGraph) -> None:
        """Graph helpers delegate cluster queries through node views."""
        clusters = sample_graph.clusters_for_node("a")
        assert [cluster.name for cluster in clusters] == ["group1"]

    def test_unknown_node_raises(self, sample_graph: DaguaGraph) -> None:
        """Unknown node IDs raise ``KeyError``."""
        with pytest.raises(KeyError):
            sample_graph.node("nonexistent")

    def test_edge_index_out_of_range(self, sample_graph: DaguaGraph) -> None:
        """Invalid edge indices raise ``IndexError``."""
        with pytest.raises(IndexError):
            sample_graph.edge(99)


class TestGraphRepr:
    """Graph repr and summary tests."""

    def test_repr_compact(self, sample_graph: DaguaGraph) -> None:
        """The graph repr summarizes counts and state compactly."""
        rendered = repr(sample_graph)
        assert "3 nodes" in rendered
        assert "3 edges" in rendered
        assert "1 cluster" in rendered

    def test_summary(self, sample_graph: DaguaGraph) -> None:
        """The graph summary reports layout state and header information."""
        summary = sample_graph.summary
        assert "Layout:" in summary
        assert "DaguaGraph" in summary


class TestCriticViewFixes:
    """Regression tests for the adversarial API critic fixes."""

    def test_successors(self, sample_graph: DaguaGraph) -> None:
        """Successor lookup follows outgoing edge direction."""
        node = sample_graph["a"]
        successors = node.successors
        labels = {successor.label for successor in successors}
        assert labels == {"Node B", "Node C"}

    def test_predecessors(self, sample_graph: DaguaGraph) -> None:
        """Predecessor lookup follows incoming edge direction."""
        node = sample_graph["c"]
        predecessors = node.predecessors
        labels = {predecessor.label for predecessor in predecessors}
        assert labels == {"Node A", "Node B"}

    def test_style_override(self, sample_graph: DaguaGraph) -> None:
        """Views expose only the explicit per-element override style."""
        node = sample_graph["a"]
        edge = sample_graph.edge(0)
        cluster = sample_graph.cluster("group1")
        assert node.style_override is None or isinstance(node.style_override, NodeStyle)
        assert edge.style_override is None or isinstance(edge.style_override, EdgeStyle)
        assert cluster.style_override is None or isinstance(cluster.style_override, ClusterStyle)
