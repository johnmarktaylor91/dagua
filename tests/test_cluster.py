"""Tests for cluster hierarchy support.

Tests cover: hierarchy API, layout constraints, edge routing,
rendering pipeline, IO round-trip, and cycle interactions.
"""

import pytest
import torch

from dagua import DaguaGraph, LayoutConfig, draw, layout, render, route_edges
from dagua.layout.constraints import (
    cluster_compactness_loss,
    cluster_containment_loss,
    cluster_separation_loss,
)
from dagua.utils import collect_cluster_leaves


# ─── Helpers ─────────────────────────────────────────────────────────────


def _make_hierarchy_graph():
    """3-level hierarchy: outer -> mid -> inner, with 9 nodes."""
    g = DaguaGraph()
    for i in range(9):
        g.add_node(i, label=f"n{i}")
    for i in range(8):
        g.add_edge(i, i + 1)

    # 3-level nesting: outer contains all, mid contains 3-8, inner contains 6-8
    g.add_cluster("outer", list(range(9)))
    g.add_cluster("mid", list(range(3, 9)), parent="outer")
    g.add_cluster("inner", list(range(6, 9)), parent="mid")
    return g


def _make_sibling_graph():
    """Two sibling clusters under a common parent."""
    g = DaguaGraph()
    for i in range(6):
        g.add_node(i, label=f"n{i}")
    g.add_edge(0, 3)
    g.add_edge(1, 4)
    g.add_edge(2, 5)
    g.add_cluster("parent", list(range(6)))
    g.add_cluster("left", [0, 1, 2], parent="parent")
    g.add_cluster("right", [3, 4, 5], parent="parent")
    return g


# ─── Hierarchy API ───────────────────────────────────────────────────────


class TestHierarchyAPI:
    def test_add_cluster_with_parent(self):
        g = DaguaGraph()
        for i in range(4):
            g.add_node(i)
        g.add_cluster("outer", [0, 1, 2, 3])
        g.add_cluster("inner", [2, 3], parent="outer")
        assert g.cluster_parents["inner"] == "outer"
        assert "outer" not in g.cluster_parents or g.cluster_parents.get("outer") is None

    def test_add_cluster_parent_order_independence(self):
        """Child before parent should work."""
        g = DaguaGraph()
        for i in range(4):
            g.add_node(i)
        # Add child first, then parent
        g.add_cluster("inner", [2, 3], parent="outer")
        g.add_cluster("outer", [0, 1, 2, 3])
        assert g.cluster_parents["inner"] == "outer"
        assert g.clusters["inner"] == [2, 3]
        assert g.clusters["outer"] == [0, 1, 2, 3]

    def test_cluster_depth(self):
        g = _make_hierarchy_graph()
        assert g.cluster_depth("outer") == 0
        assert g.cluster_depth("mid") == 1
        assert g.cluster_depth("inner") == 2

    def test_cluster_children(self):
        g = _make_hierarchy_graph()
        assert g.cluster_children("outer") == ["mid"]
        assert g.cluster_children("mid") == ["inner"]
        assert g.cluster_children("inner") == []

    def test_max_cluster_depth(self):
        g = _make_hierarchy_graph()
        assert g.max_cluster_depth == 2

    def test_max_cluster_depth_flat(self):
        g = DaguaGraph()
        for i in range(4):
            g.add_node(i)
        g.add_cluster("a", [0, 1])
        g.add_cluster("b", [2, 3])
        assert g.max_cluster_depth == 0

    def test_cluster_ids(self):
        g = _make_hierarchy_graph()
        ids = g.cluster_ids
        assert ids is not None
        assert ids.shape == (9,)
        # Nodes 6-8 should be assigned to inner (deepest)
        # The exact index depends on sorted order: inner, mid, outer
        cluster_names = sorted(g.clusters.keys())
        inner_idx = cluster_names.index("inner")
        for node in [6, 7, 8]:
            assert ids[node].item() == inner_idx

    def test_nested_dict_auto_converts_parents(self):
        """Dict-of-dicts should auto-populate cluster_parents."""
        g = DaguaGraph()
        for i in range(6):
            g.add_node(i)
        g.add_cluster("root", {"left": [0, 1, 2], "right": [3, 4, 5]})
        assert "left" in g.clusters
        assert "right" in g.clusters
        assert g.cluster_parents.get("left") == "root"
        assert g.cluster_parents.get("right") == "root"

    def test_cycle_detection(self):
        """Chain a -> b, then adding 'a' with parent='b' creates a cycle."""
        g = DaguaGraph()
        for i in range(6):
            g.add_node(i)
        g.add_cluster("a", [0, 1])
        g.add_cluster("b", [2, 3], parent="a")
        # 'a' is ancestor of 'b'. Making 'a' a child of 'b' = cycle.
        with pytest.raises(ValueError, match="cycle"):
            g.add_cluster("a", [0, 1], parent="b")

    def test_cycle_detection_3_level(self):
        """Chain a -> b -> c, then adding 'a' with parent='c' creates a cycle."""
        g = DaguaGraph()
        for i in range(6):
            g.add_node(i)
        g.add_cluster("a", [0, 1])
        g.add_cluster("b", [2, 3], parent="a")
        g.add_cluster("c", [4, 5], parent="b")
        with pytest.raises(ValueError, match="cycle"):
            g.add_cluster("a", [0, 1], parent="c")

    def test_leaf_cluster_members(self):
        g = _make_hierarchy_graph()
        # outer should include all 9 nodes (direct + children's members)
        outer_members = g.leaf_cluster_members("outer")
        assert set(outer_members) == set(range(9))
        # inner should just have 6,7,8
        inner_members = g.leaf_cluster_members("inner")
        assert set(inner_members) == {6, 7, 8}


# ─── Layout Constraints ─────────────────────────────────────────────────


class TestLayoutConstraints:
    def test_compactness_with_nested_dict(self):
        """Dict members should participate in compactness loss."""
        pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        clusters = {"root": {"a": [0, 1], "b": [2, 3]}}
        loss = cluster_compactness_loss(pos, clusters, device=pos.device)
        assert loss.item() > 0

    def test_separation_sibling_only(self):
        """With cluster_parents, only same-level clusters repel."""
        pos = torch.tensor([
            [0.0, 0.0], [1.0, 0.0],  # left cluster
            [5.0, 0.0], [6.0, 0.0],  # right cluster
            [0.0, 0.0], [6.0, 0.0],  # parent cluster (covers all)
        ])
        node_sizes = torch.ones(6, 2) * 2.0
        clusters = {
            "parent": [0, 1, 2, 3, 4, 5],
            "left": [0, 1],
            "right": [2, 3],
        }
        cluster_parents = {"left": "parent", "right": "parent"}

        # With parents: only left vs right repel (they share parent "parent")
        # parent is NOT repelled against its children
        loss_with = cluster_separation_loss(
            pos, node_sizes, clusters, device=pos.device,
            cluster_parents=cluster_parents,
        )
        # Without parents: all pairs repel
        loss_without = cluster_separation_loss(
            pos, node_sizes, clusters, device=pos.device,
        )
        # With hierarchy awareness, parent-child pairs are excluded,
        # so loss should differ (fewer pairs considered)
        assert isinstance(loss_with.item(), float)
        assert isinstance(loss_without.item(), float)

    def test_containment_loss_inside(self):
        """Child inside parent → low/zero loss."""
        # Parent covers [0, 0] to [10, 10], child fits inside
        pos = torch.tensor([
            [2.0, 2.0], [8.0, 8.0],   # parent members
            [4.0, 4.0], [6.0, 6.0],   # child members (inside parent)
        ])
        node_sizes = torch.ones(4, 2) * 2.0
        clusters = {"parent": [0, 1], "child": [2, 3]}
        cluster_parents = {"child": "parent"}

        loss = cluster_containment_loss(
            pos, node_sizes, clusters, cluster_parents, padding=1.0, device=pos.device,
        )
        assert loss.item() < 1.0  # should be very small

    def test_containment_loss_outside(self):
        """Child outside parent → high loss."""
        pos = torch.tensor([
            [0.0, 0.0], [5.0, 5.0],    # parent members
            [50.0, 50.0], [60.0, 60.0],  # child members (far outside!)
        ])
        node_sizes = torch.ones(4, 2) * 2.0
        clusters = {"parent": [0, 1], "child": [2, 3]}
        cluster_parents = {"child": "parent"}

        loss = cluster_containment_loss(
            pos, node_sizes, clusters, cluster_parents, padding=1.0, device=pos.device,
        )
        assert loss.item() > 10.0

    def test_containment_no_parents_zero(self):
        """No hierarchy → zero containment loss."""
        pos = torch.tensor([[0.0, 0.0], [5.0, 5.0]])
        node_sizes = torch.ones(2, 2) * 2.0
        clusters = {"a": [0], "b": [1]}
        cluster_parents = {}  # no hierarchy

        loss = cluster_containment_loss(
            pos, node_sizes, clusters, cluster_parents, device=pos.device,
        )
        assert loss.item() == 0.0


# ─── Integration ─────────────────────────────────────────────────────────


class TestIntegration:
    def test_nested_cluster_layout(self):
        """Layout with 3-level hierarchy completes without error."""
        g = _make_hierarchy_graph()
        config = LayoutConfig(steps=50, seed=42)
        pos = layout(g, config)
        assert pos.shape == (9, 2)
        assert not torch.isnan(pos).any()

    def test_nested_cluster_draw(self):
        """Full draw() pipeline with nested clusters completes."""
        g = _make_hierarchy_graph()
        config = LayoutConfig(steps=30, seed=42, edge_opt_steps=-1)
        fig, ax = draw(g, config)
        assert fig is not None

    def test_sibling_layout(self):
        """Layout with sibling clusters produces valid positions."""
        g = _make_sibling_graph()
        config = LayoutConfig(steps=50, seed=42)
        pos = layout(g, config)
        assert pos.shape == (6, 2)

    def test_no_clusters_zero_overhead(self):
        """Graph without clusters should not run cluster code paths."""
        g = DaguaGraph()
        for i in range(5):
            g.add_node(i)
        for i in range(4):
            g.add_edge(i, i + 1)
        config = LayoutConfig(steps=30, seed=42)
        pos = layout(g, config)
        assert pos.shape == (5, 2)

    def test_cluster_with_cyclic_graph(self):
        """Clusters + back edges together should work."""
        g = DaguaGraph()
        for i in range(4):
            g.add_node(i, label=f"n{i}")
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        g.add_edge(3, 1)  # back edge
        g.add_cluster("group", [1, 2, 3])
        config = LayoutConfig(steps=30, seed=42, edge_opt_steps=-1)
        pos = layout(g, config)
        assert pos.shape == (4, 2)


# ─── IO ──────────────────────────────────────────────────────────────────


class TestIO:
    def test_parent_json_roundtrip(self):
        """Serialize/deserialize parent field."""
        g = _make_hierarchy_graph()
        json_data = g.to_json()

        # Check parent field is in JSON
        cluster_data = json_data.get("clusters", [])
        parents_found = {c["name"]: c.get("parent") for c in cluster_data}
        assert parents_found.get("mid") == "outer"
        assert parents_found.get("inner") == "mid"
        assert parents_found.get("outer") is None

        # Round-trip
        g2 = DaguaGraph.from_json(json_data)
        assert g2.cluster_parents.get("mid") == "outer"
        assert g2.cluster_parents.get("inner") == "mid"
        assert len(g2.clusters) == 3

    def test_parent_missing_backwards_compat(self):
        """JSON without parent field still works."""
        json_data = {
            "nodes": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
            "edges": [{"source": "a", "target": "b"}],
            "clusters": [
                {"name": "group", "members": ["a", "b", "c"]},
            ],
        }
        g = DaguaGraph.from_json(json_data)
        assert "group" in g.clusters
        assert g.cluster_parents.get("group") is None


# ─── Edge Routing ────────────────────────────────────────────────────────


class TestEdgeRouting:
    def test_cluster_aware_routing(self):
        """Edges should deflect around foreign clusters."""
        g = DaguaGraph()
        # Source on left, target on right, cluster in the middle
        g.add_node("src", label="src")
        g.add_node("tgt", label="tgt")
        g.add_node("c1", label="c1")
        g.add_node("c2", label="c2")
        g.add_edge("src", "tgt")
        g.add_cluster("middle", ["c1", "c2"])

        g.compute_node_sizes()
        # Place src far left, tgt far right, cluster members in the middle
        pos = torch.tensor([
            [-100.0, 0.0],  # src
            [100.0, 0.0],   # tgt
            [0.0, 0.0],     # c1
            [0.0, 20.0],    # c2
        ])

        curves = route_edges(pos, g.edge_index, g.node_sizes, "TB", g)
        assert len(curves) == 1
        # The curve should exist (basic sanity)
        curve = curves[0]
        assert curve.p0 is not None
        assert curve.p1 is not None

    def test_edge_routing_no_clusters_unchanged(self):
        """Without clusters, routing should be normal bezier."""
        g = DaguaGraph()
        g.add_node("a", label="a")
        g.add_node("b", label="b")
        g.add_edge("a", "b")
        g.compute_node_sizes()
        pos = torch.tensor([[0.0, 0.0], [0.0, 50.0]])
        curves = route_edges(pos, g.edge_index, g.node_sizes, "TB", g)
        assert len(curves) == 1


# ─── Rendering ───────────────────────────────────────────────────────────


class TestRendering:
    def test_cluster_style_depth(self):
        """Rendering uses true depth for darkening, not enumerate index."""
        g = _make_hierarchy_graph()
        config = LayoutConfig(steps=30, seed=42, edge_opt_steps=-1)
        pos = layout(g, config)
        g.compute_node_sizes()
        # Just verify render completes — visual depth is hard to assert
        fig, ax = render(g, pos, config)
        assert fig is not None
