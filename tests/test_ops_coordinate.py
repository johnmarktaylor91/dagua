"""Tests for coordinate assignment layout ops."""

from __future__ import annotations

import torch

from dagua.layout.ops.coordinate import (
    BrandesKopf4Pass,
    BrandesKopf4PassConfig,
    BucheimWalkerTree,
    BucheimWalkerTreeConfig,
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
