"""Tests for fidelity-mode dagua_native Graphviz-dot sub-components."""

from __future__ import annotations

import torch

from dagua.config import LayoutConfig
from dagua.layout.ops.pipelines.dagua_native import (
    _apply_dot_cluster_fidelity_layout,
    _build_dot_cluster_skeletons,
    _dot_rank_assignment,
    _is_graphviz_dot_cluster_fidelity_mode,
    layout_dagua_native_pipeline,
)


def test_dot_cluster_skeleton_counts_match_cluster_c_build_skeleton() -> None:
    """Golden vector for Graphviz ``cluster.c:build_skeleton`` counters."""
    edge_index = torch.tensor(
        [
            [0, 0, 1, 3],
            [1, 2, 2, 4],
        ],
        dtype=torch.long,
    )
    ranks = (0, 1, 2, 0, 1)
    clusters = {"cluster_a": (0, 1, 2), "cluster_b": (3, 4)}

    skeletons = _build_dot_cluster_skeletons(
        clusters=clusters,
        cluster_parents=None,
        ranks=ranks,
        edge_index=edge_index,
    )

    by_name = {skeleton.name: skeleton for skeleton in skeletons}
    assert by_name["cluster_a"].rankleader_ranks == (0, 1, 2)
    assert by_name["cluster_a"].rankleader_uf_sizes == (1, 1, 1)
    assert by_name["cluster_a"].skeleton_edge_counts == (2, 2)
    assert by_name["cluster_b"].rankleader_ranks == (0, 1)
    assert by_name["cluster_b"].rankleader_uf_sizes == (1, 1)
    assert by_name["cluster_b"].skeleton_edge_counts == (1,)


def test_dot_cluster_skeleton_collapses_multi_node_rank_uf_size() -> None:
    """Graphviz decrements rankleader UF size when a rank has multiple nodes."""
    edge_index = torch.tensor([[0, 1], [2, 2]], dtype=torch.long)
    ranks = (0, 0, 1)

    (skeleton,) = _build_dot_cluster_skeletons(
        clusters={"cluster_a": (0, 1, 2)},
        cluster_parents=None,
        ranks=ranks,
        edge_index=edge_index,
    )

    assert skeleton.rankleader_ranks == (0, 1)
    assert skeleton.rankleader_uf_sizes == (1, 1)
    assert skeleton.skeleton_edge_counts == (2,)


def test_dot_cluster_fidelity_layout_separates_sibling_cluster_boxes() -> None:
    """Fidelity cluster layout should reserve non-overlapping sibling blocks."""
    edge_index = torch.tensor(
        [
            [0, 1, 3, 4, 2],
            [1, 2, 4, 5, 3],
        ],
        dtype=torch.long,
    )
    node_sizes = torch.full((6, 2), 20.0, dtype=torch.float32)
    base_pos = torch.zeros((6, 2), dtype=torch.float32)
    clusters = {"left": (0, 1, 2), "right": (3, 4, 5)}
    ranks = _dot_rank_assignment(edge_index, 6)

    out = _apply_dot_cluster_fidelity_layout(
        base_pos,
        edge_index,
        node_sizes,
        clusters,
        cluster_parents=None,
    )

    left_max = float((out[[0, 1, 2], 0] + 10.0).max().item())
    right_min = float((out[[3, 4, 5], 0] - 10.0).min().item())
    assert right_min > left_max
    rank_mean = sum(ranks) / len(ranks)
    for node, rank in enumerate(ranks):
        assert float(out[node, 1].item()) == float((rank - rank_mean) * 72.0)


def test_dagua_native_pipeline_cluster_fidelity_mode_is_invokable() -> None:
    """The public native pipeline should accept the narrow cluster fidelity mode."""
    edge_index = torch.tensor(
        [
            [0, 0, 1, 3],
            [1, 3, 2, 4],
        ],
        dtype=torch.long,
    )
    node_sizes = torch.full((5, 2), 24.0, dtype=torch.float32)
    config = LayoutConfig(
        algorithm="dagua_native",
        steps=2,
        edge_equalize_polish=False,
        force_pipeline="layered_dag",
    )

    out = layout_dagua_native_pipeline(
        edge_index=edge_index,
        num_nodes=5,
        node_sizes=node_sizes,
        config=config,
        clusters={"cluster_left": (0, 1, 2), "cluster_right": (3, 4)},
        fidelity_mode="dot_clusters",
    )

    assert _is_graphviz_dot_cluster_fidelity_mode("dot_clusters")
    assert out.shape == (5, 2)
    assert torch.isfinite(out).all()
