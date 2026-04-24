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
    """Ascending-index oriented rings look like spurious undirected DAGs."""
    num_nodes = 12
    sources = torch.tensor(list(range(num_nodes - 1)) + [0], dtype=torch.long)
    targets = torch.tensor(list(range(1, num_nodes)) + [num_nodes - 1], dtype=torch.long)
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
