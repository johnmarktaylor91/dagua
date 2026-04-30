"""Regression tests for FR NetworkX fidelity mode."""

from __future__ import annotations

import networkx as nx
import torch

from dagua.eval.competitors.classic_competitor import ClassicFR
from dagua.graph import DaguaGraph
from dagua.layout.ops.pipelines.fr import layout_fr_pipeline
from dagua.layout.ops.preprocess import _build_fr_adjacency_matrix


def _edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    """Build an edge-index tensor from integer edge tuples.

    Parameters
    ----------
    edges : list of tuple[int, int]
        Directed edges as ``(source, target)`` pairs.

    Returns
    -------
    torch.Tensor
        Edge-index tensor with shape ``[2, E]``.
    """
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _networkx_spring_tensor(
    edges: list[tuple[int, int]],
    num_nodes: int,
    seed: int,
) -> torch.Tensor:
    """Return adapter-scaled NetworkX spring positions for a small directed graph.

    Parameters
    ----------
    edges : list of tuple[int, int]
        Directed edges as ``(source, target)`` pairs.
    num_nodes : int
        Number of graph nodes.
    seed : int
        Random seed passed to ``networkx.spring_layout``.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]`` scaled like the NetworkX
        competitor adapter.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(edges)
    nx_pos = nx.spring_layout(graph, seed=seed, iterations=50, method="force")
    pos = torch.zeros((num_nodes, 2), dtype=torch.float32)
    for node_id, coordinates in nx_pos.items():
        pos[node_id] = torch.as_tensor(coordinates, dtype=torch.float32) * 500.0
    return pos


def test_fr_networkx_compat_matches_dense_spring_layout() -> None:
    """The strict FR mode should match NetworkX dense force output."""
    edges = [(0, 1), (1, 2)]
    edge_index = _edge_index(edges)

    actual = layout_fr_pipeline(
        edge_index=edge_index,
        num_nodes=3,
        steps=50,
        seed=42,
        networkx_compat=True,
    )
    expected = _networkx_spring_tensor(edges=edges, num_nodes=3, seed=42)

    torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-4)


def test_classic_fr_competitor_uses_strict_networkx_path() -> None:
    """The fidelity competitor should bypass the legacy FR default selector."""
    edges = [(0, 1), (1, 2)]
    graph = DaguaGraph.from_edge_index(_edge_index(edges), num_nodes=3)

    result = ClassicFR().layout(graph, seed=42)

    assert result.error is None
    assert result.pos is not None
    expected = _networkx_spring_tensor(edges=edges, num_nodes=3, seed=42)
    torch.testing.assert_close(result.pos, expected, rtol=1.0e-5, atol=1.0e-4)


def test_fr_weighted_duplicate_edges_keep_last_networkx_weight() -> None:
    """FR adjacency should match ``DiGraph.add_edge`` last-write semantics."""
    edge_index = _edge_index([(0, 1), (0, 1), (1, 2)])
    edge_weights = torch.tensor([2.0, 5.0, 7.0], dtype=torch.float64)

    adjacency = _build_fr_adjacency_matrix(
        edge_index=edge_index,
        num_nodes=3,
        edge_weights=edge_weights,
    )

    assert float(adjacency[0, 1].item()) == 5.0
    assert float(adjacency[1, 2].item()) == 7.0
