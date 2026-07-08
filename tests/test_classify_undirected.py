"""Semantic direction inference tests for graph classification."""

from __future__ import annotations

import torch

from dagua.graph import DaguaGraph
from dagua.layout.graph_classify import _infer_semantically_directed, classify_graph


def test_infer_semantically_directed_chain_returns_true() -> None:
    """Linear chains remain semantically directed despite one node per layer."""
    edge_index = torch.stack(
        [
            torch.arange(0, 9, dtype=torch.long),
            torch.arange(1, 10, dtype=torch.long),
        ]
    )

    assert _infer_semantically_directed(edge_index, 10) is True


def test_infer_semantically_directed_oriented_ring_returns_false() -> None:
    """Ascending-index oriented rings look like spurious undirected DAGs.

    r80 revision: a ring visited in natural node order and then mechanically
    oriented ascending (0->1->...->11, wrap 0->11) is almost entirely
    adjacent-layer edges (11/12) and is now correctly treated as a directed
    chain-with-one-skip-edge (a common real pattern, e.g. residual
    connections) rather than a spurious undirected ring -- see
    ``test_infer_semantically_directed_deep_chain_of_blocks_returns_true``.
    This test instead exercises a ring visited in a *shuffled* node order,
    which is what mechanically-oriented corpus graphs (karate, sbm_4x30,
    etc.) actually look like: scattered, mostly non-adjacent layer spans.
    """
    num_nodes = 12
    # Ring visited in permuted order so consecutive ring neighbors are NOT
    # adjacent in node index / layer -- unlike a naturally-numbered ring.
    order = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
    ring_edges = [(order[i], order[(i + 1) % num_nodes]) for i in range(num_nodes)]
    # Mechanical ascending orientation (min -> max) mirrors how an
    # undirected graph gets force-DAG'd by node-index tiebreaking.
    sources = torch.tensor([min(u, v) for u, v in ring_edges], dtype=torch.long)
    targets = torch.tensor([max(u, v) for u, v in ring_edges], dtype=torch.long)
    layer_assignments = torch.arange(num_nodes, dtype=torch.long)

    assert (
        _infer_semantically_directed(
            torch.stack([sources, targets]),
            num_nodes,
            layer_assignments=layer_assignments,
        )
        is False
    )


def test_infer_semantically_directed_layered_random_dag_returns_true() -> None:
    """Wide layered DAGs keep semantic direction when many nodes share layers."""
    num_nodes = 200
    layer_assignments = torch.arange(num_nodes, dtype=torch.long) // 20
    sources = torch.arange(0, 180, dtype=torch.long)
    targets = sources + 20

    assert (
        _infer_semantically_directed(
            torch.stack([sources, targets]),
            num_nodes,
            layer_assignments=layer_assignments,
        )
        is True
    )


def test_classify_graph_respects_explicit_semantic_direction_override() -> None:
    """Explicit graph hints override the structural heuristic."""
    graph = DaguaGraph.from_edge_list(
        [(index, index + 1) for index in range(9)],
        num_nodes=10,
        is_semantically_directed=False,
    )

    structure = classify_graph(graph.edge_index, graph.num_nodes, graph=graph)

    assert structure.is_semantically_directed is False


def test_classify_graph_populates_semantic_direction() -> None:
    """GraphStructure includes inferred semantic direction."""
    edge_index = torch.stack(
        [
            torch.arange(0, 9, dtype=torch.long),
            torch.arange(1, 10, dtype=torch.long),
        ]
    )

    structure = classify_graph(edge_index, 10)

    assert structure.is_semantically_directed is True


def test_infer_semantically_directed_deep_chain_of_blocks_returns_true() -> None:
    """Deep chains of meaningful stages (e.g. transformer blocks) stay directed.

    Each "block" has 2 nodes (one own layer each), and edges connect
    consecutive layers almost exclusively (mostly adjacent-layer spans), with
    num_layers/num_nodes >= 0.4. This must NOT be reclassified as undirected
    just because the layering is deep.
    """
    num_blocks = 30
    num_nodes = num_blocks * 2  # 60 nodes, 60 layers (one node per layer)
    layer_assignments = torch.arange(num_nodes, dtype=torch.long)
    # Chain: node i -> node i+1 for all consecutive nodes (adjacent-layer).
    sources = torch.arange(0, num_nodes - 1, dtype=torch.long)
    targets = torch.arange(1, num_nodes, dtype=torch.long)
    edge_index = torch.stack([sources, targets])

    assert (
        _infer_semantically_directed(
            edge_index,
            num_nodes,
            layer_assignments=layer_assignments,
        )
        is True
    )


def test_infer_semantically_directed_reciprocal_pair_graph_returns_false() -> None:
    """Reciprocal-edge graphs (undirected origin) are never semantically directed."""
    num_nodes = 12
    forward = torch.arange(0, num_nodes - 1, dtype=torch.long)
    forward_targets = torch.arange(1, num_nodes, dtype=torch.long)
    # Store every edge in both directions (reciprocal ratio well above 0.3).
    sources = torch.cat([forward, forward_targets])
    targets = torch.cat([forward_targets, forward])
    edge_index = torch.stack([sources, targets])

    assert _infer_semantically_directed(edge_index, num_nodes) is False


def test_infer_semantically_directed_mechanically_oriented_dense_graph_returns_false() -> None:
    """Deep layering with mostly non-adjacent edge spans stays undirected.

    Mirrors a mechanically index-oriented dense graph: layering is deep
    (num_layers/num_nodes >= 0.4) but most edges skip multiple layers rather
    than connecting consecutive stages, so the previous 0.4 rule's verdict
    (undirected) must be preserved.
    """
    num_nodes = 20
    layer_assignments = torch.arange(num_nodes, dtype=torch.long)
    # A few adjacent-layer edges (span=1)...
    adjacent_sources = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    adjacent_targets = adjacent_sources + 1
    # ...outnumbered by skip edges with larger, mixed spans.
    skip_sources = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], dtype=torch.long
    )
    skip_targets = skip_sources + 5
    skip_targets = torch.clamp(skip_targets, max=num_nodes - 1)
    sources = torch.cat([adjacent_sources, skip_sources])
    targets = torch.cat([adjacent_targets, skip_targets])
    edge_index = torch.stack([sources, targets])

    assert (
        _infer_semantically_directed(
            edge_index,
            num_nodes,
            layer_assignments=layer_assignments,
        )
        is False
    )
