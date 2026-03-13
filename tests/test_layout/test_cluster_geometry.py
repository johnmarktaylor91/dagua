"""Focused regression checks for cluster-aware layout geometry."""

from __future__ import annotations

import torch

from dagua.config import LayoutConfig
from dagua.graph import DaguaGraph
from dagua.layout import layout


def _cluster_bbox(graph: DaguaGraph, positions: torch.Tensor, cluster_name: str) -> torch.Tensor:
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
    x_overlap = max(0.0, min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0])))
    y_overlap = max(0.0, min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1])))
    return x_overlap * y_overlap


def test_sibling_clusters_do_not_overlap_badly(clustered_graph: DaguaGraph) -> None:
    pos = layout(clustered_graph, LayoutConfig(steps=80, edge_opt_steps=-1, seed=42))
    enc = _cluster_bbox(clustered_graph, pos, "encoder")
    dec = _cluster_bbox(clustered_graph, pos, "decoder")

    assert _bbox_overlap_area(enc, dec) == 0.0


def test_parent_cluster_contains_child_cluster() -> None:
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
