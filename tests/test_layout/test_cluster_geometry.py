"""Focused regression checks for cluster-aware layout geometry."""

from __future__ import annotations

import torch

from dagua.config import LayoutConfig
from dagua.graph import DaguaGraph
from dagua.layout import layout
from dagua.layout.ops.cluster_geometry import (
    ClusterLabelMetrics,
    ClusterTree,
    cluster_descendants,
    cluster_leaves_only_at_level,
    cluster_subtree,
    compute_cluster_placement_bbox,
)
from dagua.layout.ops.state import LayoutProblem


def _cluster_bbox(graph: DaguaGraph, positions: torch.Tensor, cluster_name: str) -> torch.Tensor:
    """Return the flat leaf bbox for an existing layout cluster.

    Parameters
    ----------
    graph : DaguaGraph
        Graph with computed cluster membership.
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    cluster_name : str
        Cluster whose flat bbox should be measured.

    Returns
    -------
    torch.Tensor
        Bounding box as ``[x_min, y_min, x_max, y_max]``.
    """
    graph.compute_node_sizes()
    assert graph.node_sizes is not None
    members = graph.clusters[cluster_name]
    member_pos = positions[members]
    member_sizes = graph.node_sizes[members]
    x0 = torch.min(member_pos[:, 0] - member_sizes[:, 0] / 2)
    x1 = torch.max(member_pos[:, 0] + member_sizes[:, 0] / 2)
    y0 = torch.min(member_pos[:, 1] - member_sizes[:, 1] / 2)
    y1 = torch.max(member_pos[:, 1] + member_sizes[:, 1] / 2)
    return torch.tensor([x0, y0, x1, y1], dtype=torch.float32)


def _bbox_overlap_area(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return the overlap area for two axis-aligned bboxes.

    Parameters
    ----------
    a : torch.Tensor
        First bbox as ``[x_min, y_min, x_max, y_max]``.
    b : torch.Tensor
        Second bbox as ``[x_min, y_min, x_max, y_max]``.

    Returns
    -------
    float
        Positive overlap area, or zero when the boxes do not overlap.
    """
    x_overlap = max(0.0, min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0])))
    y_overlap = max(0.0, min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1])))
    return x_overlap * y_overlap


def test_cluster_tree_single_cluster_round_trip() -> None:
    """A single flat cluster should be both root and direct leaf owner."""
    tree = ClusterTree.from_flat_membership({"root": [0, 1, 2]}, {"root": None})

    assert tree.roots == ("root",)
    assert tree.children_per_cluster["root"] == ()
    assert cluster_descendants(tree, "root") == frozenset({0, 1, 2})
    assert cluster_leaves_only_at_level(tree, "root") == frozenset({0, 1, 2})
    assert cluster_subtree(tree, "root") == ("root",)


def test_cluster_tree_nested_cluster_removes_child_leaves() -> None:
    """Parent direct leaves should exclude all immediate child descendants."""
    tree = ClusterTree.from_flat_membership(
        {"outer": [0, 1, 2, 3], "inner": [2, 3]},
        {"outer": None, "inner": "outer"},
    )

    assert tree.roots == ("outer",)
    assert tree.children_per_cluster["outer"] == ("inner",)
    assert cluster_descendants(tree, "outer") == frozenset({0, 1, 2, 3})
    assert cluster_leaves_only_at_level(tree, "outer") == frozenset({0, 1})
    assert cluster_leaves_only_at_level(tree, "inner") == frozenset({2, 3})
    assert cluster_subtree(tree, "outer") == ("outer", "inner")


def test_cluster_tree_three_siblings_keep_members_separate() -> None:
    """Sibling clusters should share the root parent without losing leaves."""
    tree = ClusterTree.from_flat_membership(
        {"a": [0], "b": [1, 2], "c": [3]},
        {"a": None, "b": None, "c": None},
    )

    assert tree.roots == ("a", "b", "c")
    assert cluster_leaves_only_at_level(tree, "a") == frozenset({0})
    assert cluster_leaves_only_at_level(tree, "b") == frozenset({1, 2})
    assert cluster_leaves_only_at_level(tree, "c") == frozenset({3})


def test_cluster_tree_deep_nesting_four_levels() -> None:
    """A four-level chain should produce direct leaves at every level."""
    tree = ClusterTree.from_flat_membership(
        {
            "l0": [0, 1, 2, 3, 4],
            "l1": [1, 2, 3, 4],
            "l2": [2, 3, 4],
            "l3": [3, 4],
        },
        {"l0": None, "l1": "l0", "l2": "l1", "l3": "l2"},
    )

    assert cluster_subtree(tree, "l0") == ("l0", "l1", "l2", "l3")
    assert cluster_leaves_only_at_level(tree, "l0") == frozenset({0})
    assert cluster_leaves_only_at_level(tree, "l1") == frozenset({1})
    assert cluster_leaves_only_at_level(tree, "l2") == frozenset({2})
    assert cluster_leaves_only_at_level(tree, "l3") == frozenset({3, 4})


def test_compute_cluster_placement_bbox_formula_sanity() -> None:
    """The placement bbox should include member sizes, padding, and label band."""
    box = compute_cluster_placement_bbox(
        inner_positions=torch.tensor([[0.0, 0.0], [10.0, 4.0]]),
        inner_sizes=torch.tensor([[2.0, 4.0], [6.0, 2.0]]),
        label_metrics=ClusterLabelMetrics(label_width_pt=20.0, label_height_pt=6.0),
        side_padding_pt=2.0,
        label_band_pt=8.0,
    )

    assert box.inner_bbox == (-1.0, -2.0, 13.0, 5.0)
    assert box.width == 24.0
    assert box.height == 19.0
    assert box.anchor_offset == (1.0, 3.5)
    assert box.label_band_y_extent == (15.0, 7.0)


def test_layout_problem_lazily_memoizes_cluster_tree() -> None:
    """LayoutProblem should build and reuse a cluster tree on demand."""
    problem = LayoutProblem(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=3,
        clusters={"outer": [0, 1, 2], "inner": [1, 2]},
        cluster_parents={"outer": None, "inner": "outer"},
    )

    tree = problem.get_cluster_tree()
    assert tree is not None
    assert tree is problem.get_cluster_tree()
    assert cluster_leaves_only_at_level(tree, "outer") == frozenset({0})


def test_sibling_clusters_do_not_overlap_badly(clustered_graph: DaguaGraph) -> None:
    """Sibling cluster leaf bboxes should remain non-overlapping."""
    pos = layout(clustered_graph, LayoutConfig(steps=80, edge_opt_steps=-1, seed=42))
    enc = _cluster_bbox(clustered_graph, pos, "encoder")
    dec = _cluster_bbox(clustered_graph, pos, "decoder")

    assert _bbox_overlap_area(enc, dec) == 0.0


def test_parent_cluster_contains_child_cluster() -> None:
    """Parent cluster leaf bbox should contain the child leaf bbox."""
    g = DaguaGraph.from_edge_list(
        [
            ("input", "enc1"),
            ("enc1", "enc2"),
            ("enc2", "mid"),
            ("mid", "dec1"),
            ("dec1", "out"),
        ]
    )
    g.add_cluster("outer", ["enc1", "enc2", "mid", "dec1"], label="Outer")
    g.add_cluster("inner", ["enc1", "enc2"], parent="outer", label="Inner")

    pos = layout(g, LayoutConfig(steps=80, edge_opt_steps=-1, seed=42))
    outer = _cluster_bbox(g, pos, "outer")
    inner = _cluster_bbox(g, pos, "inner")

    assert float(outer[0]) <= float(inner[0])
    assert float(outer[1]) <= float(inner[1])
    assert float(outer[2]) >= float(inner[2])
    assert float(outer[3]) >= float(inner[3])
