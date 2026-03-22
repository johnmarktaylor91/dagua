"""Tests for shared graph-distance utilities."""

from __future__ import annotations

import numpy as np
import torch

from dagua.layout.classic._graph_distances import (
    UNREACHABLE,
    all_pairs_shortest_paths,
    bfs_distances,
    build_directed_adjacency,
    build_undirected_adjacency,
    dijkstra_distances,
    is_connected,
)


def _make_edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    """Build a ``[2, E]`` edge-index tensor from edge tuples.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Source-target edge tuples.

    Returns
    -------
    torch.Tensor
        Edge-index tensor with shape ``[2, E]``.
    """
    if not edges:
        return torch.zeros(2, 0, dtype=torch.long)
    src, tgt = zip(*edges)
    return torch.tensor([list(src), list(tgt)], dtype=torch.long)


class TestBuildUndirectedAdjacency:
    """Coverage for undirected adjacency construction."""

    def test_empty_graph(self) -> None:
        """Return one empty neighbor list per node for an empty graph."""
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        adjacency = build_undirected_adjacency(edge_index, 3)
        assert len(adjacency) == 3
        assert all(len(neighbors) == 0 for neighbors in adjacency)

    def test_simple_chain(self) -> None:
        """Populate both directions for each undirected edge."""
        edge_index = _make_edge_index([(0, 1), (1, 2)])
        adjacency = build_undirected_adjacency(edge_index, 3)
        neighbors_1 = [neighbor for neighbor, _ in adjacency[1]]
        assert 0 in neighbors_1
        assert 2 in neighbors_1

    def test_self_loops_ignored(self) -> None:
        """Ignore self-loops when building undirected neighbors."""
        edge_index = _make_edge_index([(0, 0), (0, 1)])
        adjacency = build_undirected_adjacency(edge_index, 2)
        assert len(adjacency[0]) == 1

    def test_weighted(self) -> None:
        """Retain explicit per-edge weights in the output adjacency."""
        edge_index = _make_edge_index([(0, 1), (1, 2)])
        edge_weights = torch.tensor([2.0, 3.0])
        adjacency = build_undirected_adjacency(edge_index, 3, edge_weights=edge_weights)
        assert adjacency[0] == [(1, 2.0)]
        assert adjacency[1] == [(0, 2.0), (2, 3.0)]

    def test_duplicate_edges_keep_min_weight(self) -> None:
        """Keep the minimum weight when duplicate edges appear."""
        edge_index = _make_edge_index([(0, 1), (0, 1)])
        edge_weights = torch.tensor([5.0, 2.0])
        adjacency = build_undirected_adjacency(edge_index, 2, edge_weights=edge_weights)
        assert adjacency[0] == [(1, 2.0)]


class TestBuildDirectedAdjacency:
    """Coverage for directed adjacency construction."""

    def test_direction_preserved(self) -> None:
        """Only add the forward edge for directed adjacency."""
        edge_index = _make_edge_index([(0, 1), (1, 2)])
        adjacency = build_directed_adjacency(edge_index, 3)
        assert adjacency[0] == [(1, 1.0)]
        assert adjacency[1] == [(2, 1.0)]
        assert adjacency[2] == []

    def test_duplicate_edges_keep_min_weight(self) -> None:
        """Use the minimum weight for duplicate directed edges."""
        edge_index = _make_edge_index([(0, 1), (0, 1)])
        edge_weights = torch.tensor([4.0, 1.5])
        adjacency = build_directed_adjacency(edge_index, 2, edge_weights=edge_weights)
        assert adjacency[0] == [(1, 1.5)]


class TestBFSDistances:
    """Coverage for unweighted shortest-path traversal."""

    def test_chain(self) -> None:
        """Compute hop counts on a simple chain."""
        edge_index = _make_edge_index([(0, 1), (1, 2), (2, 3)])
        adjacency = build_undirected_adjacency(edge_index, 4)
        distances = bfs_distances(adjacency, 0)
        assert distances.tolist() == [0, 1, 2, 3]

    def test_cycle(self) -> None:
        """Take the shorter branch around a cycle."""
        edge_index = _make_edge_index([(0, 1), (1, 2), (2, 3), (3, 0)])
        adjacency = build_undirected_adjacency(edge_index, 4)
        distances = bfs_distances(adjacency, 0)
        assert distances.tolist() == [0, 1, 2, 1]

    def test_disconnected(self) -> None:
        """Mark unreachable nodes with ``UNREACHABLE``."""
        edge_index = _make_edge_index([(0, 1)])
        adjacency = build_undirected_adjacency(edge_index, 3)
        distances = bfs_distances(adjacency, 0)
        assert distances[2] == UNREACHABLE


class TestDijkstraDistances:
    """Coverage for weighted shortest-path traversal."""

    def test_unweighted_matches_bfs(self) -> None:
        """Match BFS on a unit-weight graph."""
        edge_index = _make_edge_index([(0, 1), (1, 2), (2, 3)])
        adjacency = build_undirected_adjacency(edge_index, 4)
        bfs_data = bfs_distances(adjacency, 0)
        dijkstra_data = dijkstra_distances(adjacency, 0)
        for index in range(4):
            assert abs(dijkstra_data[index] - bfs_data[index]) < 1e-6

    def test_weighted_shortcut(self) -> None:
        """Prefer the lower-weight multi-hop route over a heavier direct edge."""
        edge_index = _make_edge_index([(0, 1), (1, 2), (0, 2)])
        edge_weights = torch.tensor([1.0, 1.0, 5.0])
        adjacency = build_undirected_adjacency(edge_index, 3, edge_weights=edge_weights)
        distances = dijkstra_distances(adjacency, 0)
        assert abs(distances[2] - 2.0) < 1e-6

    def test_cycle(self) -> None:
        """Handle cycles without revisiting finalized nodes."""
        edge_index = _make_edge_index([(0, 1), (1, 2), (2, 3), (3, 0)])
        edge_weights = torch.tensor([2.0, 2.0, 2.0, 1.0])
        adjacency = build_undirected_adjacency(edge_index, 4, edge_weights=edge_weights)
        distances = dijkstra_distances(adjacency, 0)
        assert np.allclose(distances, np.asarray([0.0, 2.0, 3.0, 1.0]))

    def test_disconnected(self) -> None:
        """Use ``inf`` for unreachable weighted distances."""
        edge_index = _make_edge_index([(0, 1)])
        adjacency = build_undirected_adjacency(edge_index, 3)
        distances = dijkstra_distances(adjacency, 0)
        assert np.isinf(distances[2])


class TestAllPairsSP:
    """Coverage for repeated-source shortest-path helpers."""

    def test_unweighted(self) -> None:
        """Build an integer all-pairs matrix with BFS."""
        edge_index = _make_edge_index([(0, 1), (1, 2)])
        adjacency = build_undirected_adjacency(edge_index, 3)
        distances = all_pairs_shortest_paths(adjacency, weighted=False)
        assert distances[0, 2] == 2
        assert distances[1, 0] == 1

    def test_weighted(self) -> None:
        """Build a float all-pairs matrix with Dijkstra."""
        edge_index = _make_edge_index([(0, 1), (1, 2)])
        edge_weights = torch.tensor([2.0, 3.0])
        adjacency = build_undirected_adjacency(edge_index, 3, edge_weights=edge_weights)
        distances = all_pairs_shortest_paths(adjacency, weighted=True)
        assert abs(distances[0, 2] - 5.0) < 1e-6


class TestIsConnected:
    """Coverage for graph connectivity checks."""

    def test_connected(self) -> None:
        """Return ``True`` for a connected graph."""
        edge_index = _make_edge_index([(0, 1), (1, 2)])
        adjacency = build_undirected_adjacency(edge_index, 3)
        assert is_connected(adjacency)

    def test_disconnected(self) -> None:
        """Return ``False`` for a disconnected graph."""
        edge_index = _make_edge_index([(0, 1)])
        adjacency = build_undirected_adjacency(edge_index, 3)
        assert not is_connected(adjacency)

    def test_single_node(self) -> None:
        """Treat a single-node graph as connected."""
        adjacency = build_undirected_adjacency(torch.zeros(2, 0, dtype=torch.long), 1)
        assert is_connected(adjacency)
