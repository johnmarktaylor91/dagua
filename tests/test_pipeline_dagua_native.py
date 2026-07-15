"""Tests for fidelity-mode dagua_native Graphviz-dot sub-components."""

from __future__ import annotations

import torch

from dagua.config import LayoutConfig
from dagua.layout.ops.pipelines.dagua_native import (
    _apply_dot_cluster_fidelity_layout,
    _best_of_polish,
    _build_dot_cluster_skeletons,
    _collinear_dodge,
    _dot_rank_assignment,
    _is_graphviz_dot_cluster_fidelity_mode,
    layout_dagua_native_pipeline,
)


def test_collinear_dodge_moves_blocker_off_skip_edge() -> None:
    """A node centered on a non-incident skip edge is shifted perpendicular."""
    pos = torch.tensor([[0.0, 0.0], [0.0, 10.0], [0.0, 20.0]])
    edge_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)

    dodged = _collinear_dodge(pos, edge_index)

    assert dodged is not None
    assert torch.equal(dodged[[0, 2]], pos[[0, 2]])
    assert float(torch.abs(dodged[1, 0]).item()) > 0.0
    assert float(dodged[1, 1].item()) == float(pos[1, 1].item())


def test_directed_polish_rejects_degenerate_geometry_candidate(monkeypatch) -> None:
    """Shared directed polish routes geometry through the degeneracy guard."""
    import importlib

    native = importlib.import_module("dagua.layout.ops.pipelines.dagua_native")

    pos = torch.tensor([[0.0, 0.0], [0.0, 100.0], [0.0, 200.0]])
    edge_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
    node_sizes = torch.full((3, 2), 10.0)

    monkeypatch.setattr(native, "_POLISH_SETTINGS", ())
    monkeypatch.setattr(native, "_collinear_dodge", lambda *args, **kwargs: torch.zeros_like(pos))
    polished = _best_of_polish(
        pos,
        edge_index,
        node_sizes,
        is_semantically_directed=True,
        declared_hierarchical=True,
    )

    assert torch.equal(polished, pos)


def test_polish_scores_cyclic_digraph_with_common_ruler(monkeypatch) -> None:
    """Cyclic directed polish candidates use the benchmark's common table."""
    import dagua.metrics as metrics

    pos = torch.tensor([[0.0, 0.0], [10.0, 0.0], [5.0, 10.0]])
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    node_sizes = torch.full((3, 2), 2.0)
    observed: list[tuple[bool, bool]] = []

    def fake_full(*args: object, **kwargs: object) -> dict[str, float]:
        """Return minimal numeric metrics for selector-routing inspection."""
        del args, kwargs
        return {"neighborhood_preservation_score": 1.0}

    def fake_composite_auto(numeric: dict[str, float], directed: bool) -> float:
        """Record the semantic and hierarchy flags passed by the selector."""
        observed.append((directed, bool(numeric["declared_hierarchical"])))
        return 50.0

    monkeypatch.setattr(metrics, "full", fake_full)
    monkeypatch.setattr(metrics, "composite_auto", fake_composite_auto)

    _best_of_polish(
        pos,
        edge_index,
        node_sizes,
        is_semantically_directed=True,
        declared_hierarchical=False,
        polish_battery="default",
    )

    assert observed
    assert set(observed) == {(True, False)}


def test_dense_collinear_dodge_is_skipped_before_blocker_scan() -> None:
    """Dense O(E*N) blocker detection is capped on a 300-node graph."""
    n = 300
    pos = torch.stack((torch.arange(n, dtype=torch.float32), torch.zeros(n)), dim=1)
    edge_index = torch.triu_indices(n, n, offset=1)

    assert _collinear_dodge(pos, edge_index) is None


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
