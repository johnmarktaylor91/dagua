"""Regression tests for recursive cluster-aware placement."""

from __future__ import annotations

from typing import Tuple

import torch

from dagua.config import LayoutConfig
from dagua.graph import DaguaGraph
from dagua.layout import layout
from dagua.layout.ops.cluster_geometry import ClusterLabelMetrics, compute_cluster_placement_bbox


def _make_nested_clusters_graph() -> DaguaGraph:
    """Build a graph with an outer cluster and two sibling child clusters.

    Returns
    -------
    DaguaGraph
        Test graph with one external root node above the clustered region.
    """
    graph = DaguaGraph(direction="TB")
    for source, target in [
        ("external", "left_a"),
        ("external", "right_a"),
        ("left_a", "left_b"),
        ("right_a", "right_b"),
        ("left_b", "outer_mid"),
        ("right_b", "outer_mid"),
    ]:
        graph.add_edge(source, target)
    graph.add_cluster("outer", ["left_a", "left_b", "right_a", "right_b", "outer_mid"])
    graph.add_cluster("left", ["left_a", "left_b"], parent="outer")
    graph.add_cluster("right", ["right_a", "right_b"], parent="outer")
    return graph


def _make_cluster_showcase_graph() -> DaguaGraph:
    """Build a multi-cluster showcase graph.

    Returns
    -------
    DaguaGraph
        Graph with one nested cluster and two root sibling clusters.
    """
    graph = DaguaGraph(direction="TB")
    for source, target in [
        ("entry", "a1"),
        ("entry", "b1"),
        ("entry", "c1"),
        ("a1", "a2"),
        ("a2", "a_inner"),
        ("b1", "b2"),
        ("c1", "c2"),
        ("a_inner", "exit"),
        ("b2", "exit"),
        ("c2", "exit"),
    ]:
        graph.add_edge(source, target)
    graph.add_cluster("outer", ["a1", "a2", "a_inner"])
    graph.add_cluster("inner", ["a_inner"], parent="outer")
    graph.add_cluster("medium", ["b1", "b2"])
    graph.add_cluster("small", ["c1", "c2"])
    return graph


def _cluster_bbox(
    graph: DaguaGraph,
    positions: torch.Tensor,
    cluster_name: str,
    config: LayoutConfig,
) -> torch.Tensor:
    """Compute the placement bbox for a laid-out cluster.

    Parameters
    ----------
    graph : DaguaGraph
        Graph containing cluster membership and node sizes.
    positions : torch.Tensor
        Layout positions with shape ``[N, 2]``.
    cluster_name : str
        Cluster name to measure.
    config : LayoutConfig
        Cluster padding configuration.

    Returns
    -------
    torch.Tensor
        Bounding box as ``[x_min, y_min, x_max, y_max]``.
    """
    graph.compute_node_sizes()
    assert graph.node_sizes is not None
    members = graph.clusters[cluster_name]
    box = compute_cluster_placement_bbox(
        inner_positions=positions[members].to(dtype=torch.float32),
        inner_sizes=graph.node_sizes[members].to(dtype=torch.float32),
        label_metrics=ClusterLabelMetrics(label_width_pt=0.0, label_height_pt=0.0),
        side_padding_pt=config.cluster_side_padding_pt + config.cluster_external_clearance_pt,
        label_band_pt=config.cluster_label_band_pt,
    )
    anchor = positions[members].to(dtype=torch.float32).mean(dim=0)
    center = anchor + torch.tensor(box.anchor_offset, dtype=torch.float32)
    half = torch.tensor([box.width / 2.0, box.height / 2.0], dtype=torch.float32)
    return torch.cat((center - half, center + half))


def _node_bbox(graph: DaguaGraph, positions: torch.Tensor, node_name: str) -> torch.Tensor:
    """Compute a node bbox from rendered size metadata.

    Parameters
    ----------
    graph : DaguaGraph
        Graph containing node sizes.
    positions : torch.Tensor
        Layout positions with shape ``[N, 2]``.
    node_name : str
        Node ID to measure.

    Returns
    -------
    torch.Tensor
        Bounding box as ``[x_min, y_min, x_max, y_max]``.
    """
    graph.compute_node_sizes()
    assert graph.node_sizes is not None
    index = graph._id_to_index[node_name]
    half = graph.node_sizes[index].to(dtype=torch.float32) / 2.0
    center = positions[index].to(dtype=torch.float32)
    return torch.cat((center - half, center + half))


def _overlap_area(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return area overlap between two bboxes.

    Parameters
    ----------
    left : torch.Tensor
        First bbox as ``[x_min, y_min, x_max, y_max]``.
    right : torch.Tensor
        Second bbox as ``[x_min, y_min, x_max, y_max]``.

    Returns
    -------
    float
        Positive overlap area, or zero when disjoint.
    """
    x_overlap = max(
        0.0,
        min(float(left[2]), float(right[2])) - max(float(left[0]), float(right[0])),
    )
    y_overlap = max(
        0.0,
        min(float(left[3]), float(right[3])) - max(float(left[1]), float(right[1])),
    )
    return x_overlap * y_overlap


def _separation(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return the axis-aligned gap between two disjoint bboxes.

    Parameters
    ----------
    left : torch.Tensor
        First bbox as ``[x_min, y_min, x_max, y_max]``.
    right : torch.Tensor
        Second bbox as ``[x_min, y_min, x_max, y_max]``.

    Returns
    -------
    float
        Maximum separating-axis gap, or zero when boxes overlap.
    """
    gaps = (
        float(right[0] - left[2]),
        float(left[0] - right[2]),
        float(right[1] - left[3]),
        float(left[1] - right[3]),
    )
    return max(0.0, max(gaps))


def _layout_for_driver(graph: DaguaGraph) -> Tuple[torch.Tensor, LayoutConfig]:
    """Run the FR cluster-aware driver with deterministic settings.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to lay out.

    Returns
    -------
    tuple[torch.Tensor, LayoutConfig]
        Final positions and the config used to compute them.
    """
    config = LayoutConfig(algorithm="fr", cluster_aware=True, steps=60, seed=42)
    return layout(graph, config), config


def test_nested_clusters_have_disjoint_siblings_and_containment() -> None:
    """Nested siblings should be disjoint and structurally contained."""
    graph = _make_nested_clusters_graph()
    positions, config = _layout_for_driver(graph)

    outer = _cluster_bbox(graph, positions, "outer", config)
    left = _cluster_bbox(graph, positions, "left", config)
    right = _cluster_bbox(graph, positions, "right", config)
    external = _node_bbox(graph, positions, "external")

    assert _overlap_area(left, right) == 0.0
    assert float(outer[0]) < float(left[0])
    assert float(outer[1]) < float(left[1])
    assert float(outer[2]) > float(right[2])
    assert float(outer[3]) > float(right[3])
    assert _overlap_area(external, outer) == 0.0
    assert _separation(external, outer) >= 0.0


def test_cluster_showcase_root_sibling_clusters_are_disjoint() -> None:
    """Representative showcase root clusters should not overlap."""
    graph = _make_cluster_showcase_graph()
    positions, config = _layout_for_driver(graph)

    outer = _cluster_bbox(graph, positions, "outer", config)
    medium = _cluster_bbox(graph, positions, "medium", config)
    small = _cluster_bbox(graph, positions, "small", config)
    inner = _cluster_bbox(graph, positions, "inner", config)

    assert _overlap_area(outer, medium) == 0.0
    assert _overlap_area(medium, small) == 0.0
    assert float(outer[0]) <= float(inner[0])
    assert float(outer[1]) <= float(inner[1])
    assert float(outer[2]) >= float(inner[2])
    assert float(outer[3]) >= float(inner[3])
    assert float(outer[3] - outer[1]) > float(inner[3] - inner[1])


def test_deep_nested_parent_bbox_updates_after_child_geometry_changes() -> None:
    """Changing a deep child size should expand every parent bbox."""
    graph = _make_nested_clusters_graph()
    positions, config = _layout_for_driver(graph)
    before_outer = _cluster_bbox(graph, positions, "outer", config)

    graph.compute_node_sizes()
    assert graph.node_sizes is not None
    graph.node_sizes = graph.node_sizes.clone()
    graph.node_sizes[graph._id_to_index["left_a"], 0] = 220.0
    graph._node_sizes_revision = graph.revision
    updated_positions, updated_config = _layout_for_driver(graph)
    after_outer = _cluster_bbox(graph, updated_positions, "outer", updated_config)

    assert float(after_outer[2] - after_outer[0]) > float(before_outer[2] - before_outer[0])


def test_cluster_aware_false_keeps_legacy_flat_algorithm_path() -> None:
    """Disabling cluster-aware placement should keep the flat FR path usable."""
    graph = _make_nested_clusters_graph()
    config = LayoutConfig(algorithm="fr", cluster_aware=False, steps=20, seed=42)

    positions = layout(graph, config)

    assert positions.shape == (graph.num_nodes, 2)
    assert torch.isfinite(positions).all()
