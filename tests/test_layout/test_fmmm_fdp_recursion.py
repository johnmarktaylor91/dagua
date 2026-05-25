"""Regression tests for Graphviz fdp derived-graph recursion fidelity."""

from __future__ import annotations

import math

import torch

from dagua.layout.ops.cluster_geometry import ClusterTree
from dagua.layout.ops.pipelines.fmmm import (
    _fdp_recursion_components,
    _fdp_recursion_derive_graph,
    _fdp_recursion_expand_cluster_ports,
    _FdpRecursionPort,
    graphviz_fdp_fidelity,
    layout_fmmm_pipeline,
)


def _edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    """Build an edge tensor from edge tuples.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Edge tuples in original graph order.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    sources, targets = zip(*edges)
    return torch.tensor([list(sources), list(targets)], dtype=torch.long)


def _cluster_tree() -> ClusterTree:
    """Build a tiny cluster tree used by the fdp recursion tests.

    Returns
    -------
    ClusterTree
        Tree with one root cluster containing nodes ``0`` and ``1``.
    """
    return ClusterTree.from_flat_membership({"cluster_A": [0, 1]}, {"cluster_A": None})


def test_fdp_derive_graph_collapses_clusters_and_groups_real_edges() -> None:
    """Derived graph should match Graphviz ``deriveGraph`` collapse semantics.

    Returns
    -------
    None
        Assertions validate node creation order and collapsed edge backrefs.
    """
    edge_index = _edge_index([(0, 2), (1, 2), (1, 3)])

    derived = _fdp_recursion_derive_graph(edge_index, 4, _cluster_tree(), None)

    assert [node.kind for node in derived.nodes] == ["cluster", "leaf", "leaf"]
    assert [node.key for node in derived.nodes] == ["cluster_A", 2, 3]
    assert [(edge.source, edge.target, edge.real_edges) for edge in derived.edges] == [
        (0, 1, (0, 1)),
        (0, 2, (2,)),
    ]


def test_fdp_components_merge_all_port_components_first() -> None:
    """Port-bearing components should be merged into the first component.

    Returns
    -------
    None
        Assertions validate the ``findCComp`` ordering used by fdp.
    """
    edge_index = _edge_index([(0, 2)])
    ports = (_FdpRecursionPort(edge_id=0, node=0, alpha=0.0),)

    derived = _fdp_recursion_derive_graph(edge_index, 4, _cluster_tree(), "cluster_A", ports)
    components = _fdp_recursion_components(derived)

    assert [node.kind for node in derived.nodes] == ["leaf", "leaf", "port"]
    assert components == ((2, 0), (1,))


def test_fdp_expand_cluster_ports_matches_reference_edge_order_and_angles() -> None:
    """Cluster expansion should preserve Graphviz port order and angle spacing.

    Returns
    -------
    None
        Assertions use hand-captured C behavior for two same-side real edges:
        edges are emitted in stored order with a maximum two-degree delta.
    """
    edge_index = _edge_index([(0, 2), (1, 2), (1, 3)])
    derived = _fdp_recursion_derive_graph(edge_index, 4, _cluster_tree(), None)
    positions = {
        0: torch.tensor([0.0, 0.0], dtype=torch.float32),
        1: torch.tensor([1.0, 0.0], dtype=torch.float32),
        2: torch.tensor([0.0, 1.0], dtype=torch.float32),
    }

    ports = _fdp_recursion_expand_cluster_ports(derived, positions, 0, edge_index)

    assert [(port.edge_id, port.node) for port in ports] == [(0, 0), (1, 1), (2, 1)]
    assert math.isclose(ports[0].alpha, 0.0, abs_tol=1.0e-12)
    assert math.isclose(ports[1].alpha, math.pi / 90.0, abs_tol=1.0e-12)
    assert math.isclose(ports[2].alpha, math.pi / 2.0, abs_tol=1.0e-12)


def test_fdp_recursion_guard_keeps_default_fmmm_behavior_unchanged() -> None:
    """Cluster metadata should be ignored unless ``fidelity_mode`` is enabled.

    Returns
    -------
    None
        Assertion validates the public behavior guard for existing callers.
    """
    edge_index = _edge_index([(0, 2), (1, 2), (1, 3)])
    node_sizes = torch.ones((4, 2), dtype=torch.float32)

    without_clusters = layout_fmmm_pipeline(edge_index, 4, node_sizes=node_sizes, steps=4, seed=9)
    with_clusters = layout_fmmm_pipeline(
        edge_index,
        4,
        node_sizes=node_sizes,
        steps=4,
        seed=9,
        clusters={"cluster_A": [0, 1]},
        cluster_parents={"cluster_A": None},
    )

    assert torch.equal(without_clusters, with_clusters)


def test_graphviz_fdp_fidelity_returns_finite_clustered_positions() -> None:
    """The public fdp recursion entrypoint should produce finite positions.

    Returns
    -------
    None
        Assertion validates the component is invokable end-to-end.
    """
    edge_index = _edge_index([(0, 2), (1, 2), (1, 3)])
    node_sizes = torch.ones((4, 2), dtype=torch.float32)

    positions = graphviz_fdp_fidelity(
        edge_index=edge_index,
        num_nodes=4,
        node_sizes=node_sizes,
        steps=4,
        seed=5,
        clusters={"cluster_A": [0, 1]},
        cluster_parents={"cluster_A": None},
    )

    assert positions.shape == (4, 2)
    assert torch.isfinite(positions).all()
