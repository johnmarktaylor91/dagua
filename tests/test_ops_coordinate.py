"""Tests for coordinate assignment layout ops."""

from __future__ import annotations

import torch

from dagua.layout.ops.coordinate import (
    BrandesKopf4Pass,
    BrandesKopf4PassConfig,
    BucheimWalkerTree,
    BucheimWalkerTreeConfig,
    ClusterAwareXCompaction,
    ClusterAwareXCompactionConfig,
    ComponentTilingCrossingRisk,
    ComponentTilingCrossingRiskConfig,
    RankRowSnap,
    RankRowSnapConfig,
    _enforce_row_adjacent_min_spacing,
)
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState


def _edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    """Build a ``[2, E]`` edge tensor from a Python edge list.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Source-target edge tuples.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)
    sources, targets = zip(*edges)
    return torch.tensor([list(sources), list(targets)], dtype=torch.long)


def test_brandes_kopf_4pass_produces_valid_x_coordinates() -> None:
    """BrandesKopf4Pass should respect the supplied ordering with finite coordinates."""
    problem = LayoutProblem(
        edge_index=_edge_index([(0, 3), (1, 2), (3, 4), (2, 5)]),
        num_nodes=6,
        node_sizes=torch.ones((6, 2), dtype=torch.float32),
    )
    state = SolveState(
        layers=torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long),
        ordering=torch.tensor([0, 1, 1, 0, 0, 1], dtype=torch.long),
    )

    result = BrandesKopf4Pass(BrandesKopf4PassConfig(node_sep=1.0, rank_sep=2.0)).apply(
        problem, state, RuntimeContext()
    )

    assert result.pos is not None
    assert result.pos.shape == (6, 2)
    assert torch.isfinite(result.pos).all()
    assert torch.equal(result.pos[:, 1].cpu(), torch.tensor([0.0, 0.0, 2.0, 2.0, 4.0, 4.0]))
    assert result.pos[0, 0].item() < result.pos[1, 0].item()
    assert result.pos[3, 0].item() < result.pos[2, 0].item()
    assert result.pos[4, 0].item() < result.pos[5, 0].item()


def test_bucheim_walker_tree_places_simple_tree_by_depth_and_sibling_order() -> None:
    """BucheimWalkerTree should place a simple rooted tree with tidy ordering."""
    problem = LayoutProblem(
        edge_index=_edge_index([(0, 1), (0, 2), (1, 3), (1, 4)]),
        num_nodes=5,
    )

    result = BucheimWalkerTree(
        BucheimWalkerTreeConfig(sibling_sep=1.0, layer_sep=1.5, component_gap=2.0)
    ).apply(problem, SolveState(), RuntimeContext())

    assert result.pos is not None
    assert result.pos.shape == (5, 2)
    assert torch.isfinite(result.pos).all()
    assert result.pos[0, 1].item() < result.pos[1, 1].item()
    assert result.pos[1, 1].item() == result.pos[2, 1].item()
    assert result.pos[3, 1].item() == result.pos[4, 1].item()
    assert result.pos[1, 1].item() < result.pos[3, 1].item()
    assert result.pos[1, 0].item() < result.pos[2, 0].item()
    assert result.pos[3, 0].item() < result.pos[4, 0].item()
    assert result.pos[1, 0].item() < result.pos[0, 0].item() < result.pos[2, 0].item()


def test_brandes_kopf_4pass_respects_node_separation_within_layers() -> None:
    """BrandesKopf4Pass should keep same-layer nodes at least ``node_sep`` apart."""

    problem = LayoutProblem(
        edge_index=_edge_index([(0, 3), (1, 3), (2, 4), (3, 5), (4, 5)]),
        num_nodes=6,
        node_sizes=torch.ones((6, 2), dtype=torch.float32),
    )
    state = SolveState(
        layers=torch.tensor([0, 0, 0, 1, 1, 2], dtype=torch.long),
        ordering=torch.tensor([0, 1, 2, 0, 1, 0], dtype=torch.long),
    )

    result = BrandesKopf4Pass(BrandesKopf4PassConfig(node_sep=1.5, rank_sep=2.0)).apply(
        problem,
        state,
        RuntimeContext(),
    )

    assert result.pos is not None
    for layer_nodes in ([0, 1, 2], [3, 4]):
        x_coords = result.pos[layer_nodes, 0]
        assert torch.all(x_coords[1:] - x_coords[:-1] >= 1.5)


def test_bucheim_walker_tree_keeps_siblings_non_overlapping_and_layers_aligned() -> None:
    """BucheimWalkerTree should align depths and separate sibling subtrees."""

    problem = LayoutProblem(
        edge_index=_edge_index([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5)]),
        num_nodes=6,
    )

    result = BucheimWalkerTree(
        BucheimWalkerTreeConfig(sibling_sep=1.5, layer_sep=2.0, component_gap=3.0)
    ).apply(problem, SolveState(), RuntimeContext())

    assert result.pos is not None
    torch.testing.assert_close(
        result.pos[[1, 2], 1],
        torch.tensor([-0.6667, -0.6667]),
        atol=1.0e-4,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result.pos[[3, 4, 5], 1],
        torch.tensor([1.3333, 1.3333, 1.3333]),
        atol=1.0e-4,
        rtol=0.0,
    )
    assert float(result.pos[2, 0].item() - result.pos[1, 0].item()) >= 1.5
    assert float(result.pos[4, 0].item() - result.pos[3, 0].item()) >= 1.5


def test_rank_row_snap_collapses_y_to_layer_medians() -> None:
    """RankRowSnap should remove within-rank y jitter for acyclic layered graphs."""
    problem = LayoutProblem(edge_index=_edge_index([(0, 2), (1, 3)]), num_nodes=4)
    state = SolveState(
        pos=torch.tensor([[0.0, 0.0], [1.0, 2.0], [0.0, 99.0], [1.0, 101.0]]),
        layers=torch.tensor([0, 0, 1, 1], dtype=torch.long),
    )

    result = RankRowSnap(RankRowSnapConfig(min_layers=1)).apply(problem, state, RuntimeContext())

    assert result.pos is not None
    assert result.pos[0, 1].item() == result.pos[1, 1].item()
    assert result.pos[2, 1].item() == result.pos[3, 1].item()
    assert result.extras["rank_row_snap_applied"] is True


def test_rank_row_snap_skips_cyclic_graphs() -> None:
    """RankRowSnap should skip when the acyclic predicate is false."""
    problem = LayoutProblem(edge_index=_edge_index([(0, 1), (1, 0)]), num_nodes=2)
    pos = torch.tensor([[0.0, 0.0], [1.0, 10.0]])
    state = SolveState(pos=pos.clone(), layers=torch.tensor([0, 1], dtype=torch.long))

    result = RankRowSnap(RankRowSnapConfig(is_acyclic=False)).apply(
        problem,
        state,
        RuntimeContext(),
    )

    assert result.pos is not None
    assert torch.equal(result.pos, pos)
    assert result.extras["rank_row_snap_applied"] is False


def test_rank_row_snap_skips_too_few_layers_by_default() -> None:
    """RankRowSnap should preserve useful shallow-DAG vertical separation."""
    problem = LayoutProblem(edge_index=_edge_index([(0, 2), (1, 3)]), num_nodes=4)
    pos = torch.tensor([[0.0, 0.0], [1.0, 2.0], [0.0, 99.0], [1.0, 101.0]])
    state = SolveState(pos=pos.clone(), layers=torch.tensor([0, 0, 1, 1], dtype=torch.long))

    result = RankRowSnap(RankRowSnapConfig()).apply(problem, state, RuntimeContext())

    assert result.pos is not None
    assert torch.equal(result.pos, pos)
    assert result.extras["rank_row_snap_applied"] is False


def test_rank_row_snap_enforces_min_row_spacing_after_y_snap() -> None:
    """RankRowSnap should repair row-adjacent x gaps created before snapping."""
    problem = LayoutProblem(
        edge_index=_edge_index([(0, 2), (1, 3)]),
        num_nodes=4,
        node_sizes=torch.full((4, 2), 10.0, dtype=torch.float32),
    )
    state = SolveState(
        pos=torch.tensor([[0.0, 0.0], [5.0, 1.0], [0.0, 99.0], [5.0, 100.0]]),
        layers=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        ordering=torch.tensor([0, 1, 0, 1], dtype=torch.long),
    )

    result = RankRowSnap(RankRowSnapConfig(min_layers=1, node_sep=0.0, row_min_gap=2.0)).apply(
        problem, state, RuntimeContext()
    )

    assert result.pos is not None
    assert float(result.pos[1, 0].item() - result.pos[0, 0].item()) == 12.0
    assert float(result.pos[3, 0].item() - result.pos[2, 0].item()) == 12.0
    assert result.pos[0, 1].item() == result.pos[1, 1].item()
    assert result.pos[2, 1].item() == result.pos[3, 1].item()


def test_cluster_aware_x_compaction_adds_sibling_cluster_gap() -> None:
    """ClusterAwareXCompaction should separate sibling cluster x intervals."""
    problem = LayoutProblem(
        edge_index=_edge_index([(0, 4), (1, 5), (2, 6), (3, 7)]),
        num_nodes=8,
        node_sizes=torch.ones((8, 2), dtype=torch.float32),
        clusters={
            "a": [0, 2, 4, 6],
            "b": [1, 3, 5, 7],
            "a_inner": [2, 6],
            "b_inner": [3, 7],
            "a_single": [0],
        },
        cluster_parents={
            "a": None,
            "b": None,
            "a_inner": "a",
            "b_inner": "b",
            "a_single": "a",
        },
    )
    state = SolveState(
        pos=torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 1.0],
                [3.0, 1.0],
            ]
        ),
        layers=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long),
        ordering=torch.tensor([0, 2, 1, 3, 0, 2, 1, 3], dtype=torch.long),
    )

    result = ClusterAwareXCompaction(
        ClusterAwareXCompactionConfig(
            node_sep=1.0,
            cluster_gap_multiplier=2.0,
            min_long_edge_fraction=0.0,
        )
    ).apply(problem, state, RuntimeContext())

    assert result.pos is not None
    a_x = result.pos[[0, 2, 4, 6], 0]
    b_x = result.pos[[1, 3, 5, 7], 0]
    assert float(b_x.min().item() - a_x.max().item()) >= 2.0


def test_cluster_aware_x_compaction_skips_without_clusters() -> None:
    """ClusterAwareXCompaction should leave non-clustered graphs unchanged."""
    problem = LayoutProblem(edge_index=_edge_index([(0, 2), (1, 3)]), num_nodes=4)
    pos = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    state = SolveState(pos=pos.clone(), layers=torch.tensor([0, 0, 1, 1], dtype=torch.long))

    result = ClusterAwareXCompaction(ClusterAwareXCompactionConfig()).apply(
        problem,
        state,
        RuntimeContext(),
    )

    assert result.pos is not None
    assert torch.equal(result.pos, pos)


def test_cluster_aware_x_compaction_skips_low_long_edge_fraction() -> None:
    """ClusterAwareXCompaction should skip dense sibling cross-talk shapes."""
    problem = LayoutProblem(
        edge_index=_edge_index([(0, 4), (1, 5), (2, 6), (3, 7)]),
        num_nodes=8,
        node_sizes=torch.ones((8, 2), dtype=torch.float32),
        clusters={"a": [0, 2, 4, 6], "b": [1, 3, 5, 7]},
        cluster_parents={"a": None, "b": None},
    )
    pos = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    state = SolveState(
        pos=pos.clone(),
        layers=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long),
        ordering=torch.tensor([0, 2, 1, 3, 0, 2, 1, 3], dtype=torch.long),
    )

    result = ClusterAwareXCompaction(ClusterAwareXCompactionConfig()).apply(
        problem,
        state,
        RuntimeContext(),
    )

    assert result.pos is not None
    assert torch.equal(result.pos, pos)
    assert result.extras["cluster_aware_x_compaction_applied"] is False


def test_row_adjacent_min_spacing_expands_crowded_row_exactly() -> None:
    """Same-row spacing repair should add only the required extra width."""
    pos = torch.tensor([[0.0, 0.0], [5.0, 0.0], [20.0, 1.0]])
    layers = torch.tensor([0, 0, 1], dtype=torch.long)
    ordering = torch.tensor([0, 1, 0], dtype=torch.long)
    node_sizes = torch.tensor([[10.0, 2.0], [8.0, 2.0], [1.0, 1.0]])

    result = _enforce_row_adjacent_min_spacing(
        pos=pos,
        layers=layers,
        ordering=ordering,
        node_sizes=node_sizes,
        min_gap=2.0,
    )

    assert float(result[1, 0].item() - result[0, 0].item()) == 12.0
    assert float(result[1, 0].item() + result[0, 0].item()) == 5.0
    torch.testing.assert_close(result[2], pos[2])


def test_row_adjacent_min_spacing_leaves_uncrowded_row_untouched() -> None:
    """Same-row spacing repair should no-op when all adjacent gaps pass."""
    pos = torch.tensor([[0.0, 0.0], [12.5, 0.0], [30.0, 1.0]])
    layers = torch.tensor([0, 0, 1], dtype=torch.long)
    ordering = torch.tensor([0, 1, 0], dtype=torch.long)
    node_sizes = torch.tensor([[10.0, 2.0], [8.0, 2.0], [1.0, 1.0]])

    result = _enforce_row_adjacent_min_spacing(
        pos=pos,
        layers=layers,
        ordering=ordering,
        node_sizes=node_sizes,
        min_gap=2.0,
    )

    torch.testing.assert_close(result, pos)


def test_cluster_aware_x_compaction_preserves_cluster_contiguity_after_row_spacing() -> None:
    """Cluster compaction should keep sibling clusters ordered after row repair."""
    problem = LayoutProblem(
        edge_index=_edge_index([(0, 4), (1, 5), (2, 6), (3, 7)]),
        num_nodes=8,
        node_sizes=torch.full((8, 2), 10.0, dtype=torch.float32),
        clusters={"a": [0, 1, 4, 5]},
        cluster_parents={"a": None},
    )
    state = SolveState(
        pos=torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 1.0],
                [3.0, 1.0],
            ]
        ),
        layers=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long),
        ordering=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long),
    )

    result = ClusterAwareXCompaction(
        ClusterAwareXCompactionConfig(
            node_sep=0.0,
            row_min_gap=2.0,
            cluster_gap_multiplier=0.0,
            min_long_edge_fraction=0.0,
        )
    ).apply(problem, state, RuntimeContext())

    assert result.pos is not None
    a_x = result.pos[[0, 1, 4, 5], 0]
    b_x = result.pos[[2, 3, 6, 7], 0]
    assert float(a_x.max().item()) < float(b_x.min().item())
    for layer_nodes in ([0, 1, 2, 3], [4, 5, 6, 7]):
        ordered = sorted(layer_nodes, key=lambda node: float(result.pos[node, 0].item()))
        distances = result.pos[ordered[1:], 0] - result.pos[ordered[:-1], 0]
        assert torch.all(distances >= 12.0)


def test_component_tiling_crossing_risk_skips_connected_graph() -> None:
    """ComponentTilingCrossingRisk should no-op for connected graphs."""
    problem = LayoutProblem(edge_index=_edge_index([(0, 1), (1, 2)]), num_nodes=3)
    pos = torch.tensor([[0.0, 0.0], [0.0, 10.0], [0.0, 20.0]])
    state = SolveState(pos=pos.clone(), layers=torch.tensor([0, 1, 2], dtype=torch.long))

    result = ComponentTilingCrossingRisk(ComponentTilingCrossingRiskConfig()).apply(
        problem,
        state,
        RuntimeContext(),
    )

    assert result.pos is not None
    assert torch.equal(result.pos, pos)
