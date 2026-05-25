"""Regression tests for FR NetworkX fidelity mode."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
import torch
from scipy.spatial import procrustes

from dagua.eval.competitors.classic_competitor import ClassicFR
from dagua.eval.competitors.igraph_competitor import IgraphFR
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


def _procrustes_rmsd(left: torch.Tensor, right: torch.Tensor) -> float:
    """Compute Procrustes-aligned RMSD for two layouts.

    Parameters
    ----------
    left : torch.Tensor
        First position tensor with shape ``[N, 2]``.
    right : torch.Tensor
        Second position tensor with shape ``[N, 2]``.

    Returns
    -------
    float
        Root mean squared aligned distance.
    """
    left_array = left.detach().cpu().numpy()
    right_array = right.detach().cpu().numpy()
    _, _, disparity = procrustes(left_array, right_array)
    return float(np.sqrt(disparity / float(left_array.shape[0])))


def _networkx_spring_tensor(
    edges: list[tuple[int, int]],
    num_nodes: int,
    seed: int,
    edge_weights: torch.Tensor | None = None,
    k: float | None = None,
    pos: torch.Tensor | None = None,
    fixed: list[int] | None = None,
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
    edge_weights : torch.Tensor, optional
        Optional directed edge weights with shape ``[E]``.
    k : float, optional
        Explicit optimal node spacing for NetworkX.
    pos : torch.Tensor, optional
        Optional full initial positions with shape ``[N, 2]``.
    fixed : list of int, optional
        Optional fixed node indices. NetworkX skips final rescale when set.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]`` scaled like the NetworkX
        competitor adapter.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_nodes))
    if edge_weights is None:
        graph.add_edges_from(edges)
    else:
        for (source, target), weight in zip(edges, edge_weights.tolist()):
            graph.add_edge(source, target, weight=float(weight))

    nx_pos_input = None
    if pos is not None:
        nx_pos_input = {
            node_index: pos[node_index].detach().cpu().numpy() for node_index in range(num_nodes)
        }
    nx_pos = nx.spring_layout(
        graph,
        seed=seed,
        iterations=50,
        method="force",
        k=k,
        pos=nx_pos_input,
        fixed=fixed,
    )
    pos = torch.zeros((num_nodes, 2), dtype=torch.float32)
    for node_id, coordinates in nx_pos.items():
        scale = 1.0 if fixed is not None else 500.0
        pos[node_id] = torch.as_tensor(coordinates, dtype=torch.float32) * scale
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


def test_fr_networkx_compat_accepts_explicit_k() -> None:
    """The strict FR mode should forward explicit NetworkX ``k`` spacing."""
    edges = [(0, 1), (1, 2), (2, 3)]
    edge_index = _edge_index(edges)

    actual = layout_fr_pipeline(
        edge_index=edge_index,
        num_nodes=4,
        steps=50,
        seed=7,
        networkx_compat=True,
        k=0.75,
    )
    expected = _networkx_spring_tensor(edges=edges, num_nodes=4, seed=7, k=0.75)

    torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-4)


def test_fr_networkx_compat_honors_fixed_nodes_without_rescale() -> None:
    """Fixed nodes should stay unmoved and disable NetworkX final rescale."""
    edges = [(0, 1), (1, 2)]
    edge_index = _edge_index(edges)
    initial = torch.tensor(
        [[0.0, 0.0], [0.25, 0.75], [1.0, 0.0]],
        dtype=torch.float64,
    )

    actual = layout_fr_pipeline(
        edge_index=edge_index,
        num_nodes=3,
        steps=50,
        seed=42,
        pos=initial,
        fixed=[0],
        networkx_compat=True,
    )
    expected = _networkx_spring_tensor(
        edges=edges,
        num_nodes=3,
        seed=42,
        pos=initial,
        fixed=[0],
    )

    torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-4)
    torch.testing.assert_close(actual[0], initial[0].to(dtype=torch.float32))


def test_fr_igraph_fidelity_matches_reference_adapter() -> None:
    """The igraph fidelity path should match python-igraph's FR adapter."""
    pytest.importorskip("igraph")
    edges = [(index, index + 1) for index in range(7)] + [(0, 4), (2, 6)]
    edge_index = _edge_index(edges)
    graph = DaguaGraph.from_edge_index(edge_index, num_nodes=8)

    actual = layout_fr_pipeline(
        edge_index=edge_index,
        num_nodes=8,
        seed=3,
        fidelity_mode="igraph",
    )
    expected = IgraphFR().layout(graph, seed=3).pos

    assert expected is not None
    assert _procrustes_rmsd(actual, expected) < 1.0e-3
