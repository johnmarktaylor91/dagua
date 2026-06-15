"""Regression tests for Graphviz fdp derived-graph recursion fidelity."""

from __future__ import annotations

import importlib
import math
from typing import Optional

import pytest
import torch

from dagua.layout.ops.cluster_geometry import ClusterTree
from dagua.layout.ops.pipelines.fmmm import (
    _fdp_recursion_component_offsets,
    _fdp_recursion_components,
    _fdp_recursion_derive_graph,
    _fdp_recursion_expand_cluster_ports,
    _FdpRecursionPort,
    _graphviz_tile_pack_offsets,
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
    """Port-bearing components should merge first in derived-node order.

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
    assert components == ((0, 2), (1,))


def test_fdp_recursion_component_offsets_use_graphviz_tile_pack() -> None:
    """Recursive fdp levels should reuse the Graphviz component packer.

    Returns
    -------
    None
        Assertion validates that recursion-level sibling component offsets stay
        wired to the R36 tile-packing port instead of the old row packer.
    """
    component_boxes = [(10.0, 4.0), (6.0, 6.0), (3.0, 12.0)]
    expected = _graphviz_tile_pack_offsets(
        [(0.0, 0.0, width, height) for width, height in component_boxes],
    )

    offsets = _fdp_recursion_component_offsets(component_boxes)

    assert [tuple(float(value) for value in offset.tolist()) for offset in offsets] == expected


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
    """Cluster metadata should be ignored unless Graphviz FDP fidelity is requested.

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


def test_ogdf_fmmm_fidelity_ignores_cluster_metadata() -> None:
    """OGDF FMMM fidelity should match OGDF's plain-graph cluster behavior.

    Returns
    -------
    None
        Assertion validates that clustered benchmark graphs no longer take the
        Graphviz FDP recursion route when targeting OGDF FMMM.
    """
    edge_index = _edge_index([(0, 2), (1, 2), (1, 3)])
    node_sizes = torch.ones((4, 2), dtype=torch.float32)

    without_clusters = layout_fmmm_pipeline(
        edge_index,
        4,
        node_sizes=node_sizes,
        steps=4,
        seed=9,
        fidelity_mode=True,
    )
    with_clusters = layout_fmmm_pipeline(
        edge_index,
        4,
        node_sizes=node_sizes,
        steps=4,
        seed=9,
        fidelity_mode=True,
        clusters={"cluster_A": [0, 1]},
        cluster_parents={"cluster_A": None},
    )

    assert torch.equal(without_clusters, with_clusters)


def test_fmmm_graphviz_fdp_mode_accepts_cluster_metadata() -> None:
    """Explicit Graphviz FDP fidelity should keep the cluster recursion route.

    Returns
    -------
    None
        Assertion validates that the separate Graphviz-FDP target remains
        available after OGDF FMMM cluster metadata is ignored.
    """
    edge_index = _edge_index([(0, 2), (1, 2), (1, 3)])
    node_sizes = torch.ones((4, 2), dtype=torch.float32)

    positions = layout_fmmm_pipeline(
        edge_index,
        4,
        node_sizes=node_sizes,
        steps=4,
        seed=9,
        fidelity_mode="graphviz_fdp",
        clusters={"cluster_A": [0, 1]},
        cluster_parents={"cluster_A": None},
    )

    assert positions.shape == (4, 2)
    assert torch.isfinite(positions).all()


def test_fmmm_graphviz_fdp_mode_routes_unclustered_graphs_to_fdp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plain Graphviz FDP fidelity should use the FDP component emulator.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest patch helper.

    Returns
    -------
    None
        Assertion validates that unclustered ``fidelity_mode="graphviz_fdp"``
        does not fall through to the OGDF FMMM fidelity path.
    """
    edge_index = _edge_index([(0, 1), (1, 2)])
    fmmm = importlib.import_module("dagua.layout.ops.pipelines.fmmm")
    calls: list[int] = []

    def fake_component_layout(
        edge_index: torch.Tensor,
        num_nodes: int,
        node_sizes: Optional[torch.Tensor],
        seed: int,
        edge_weights: Optional[torch.Tensor] = None,
        max_iters: int = 600,
        flip_y: bool = True,
    ) -> torch.Tensor:
        """Return a sentinel component layout while recording ``maxiter``.

        Parameters
        ----------
        edge_index : torch.Tensor
            Local edge tensor with shape ``[2, E]``.
        num_nodes : int
            Number of local component nodes.
        node_sizes : torch.Tensor, optional
            Optional node sizes with shape ``[N, 2]``.
        seed : int
            Graphviz seed value.
        edge_weights : torch.Tensor, optional
            Optional edge weights with shape ``[E]``.
        max_iters : int, default=600
            Graphviz ``maxiter`` budget.
        flip_y : bool, default=True
            Whether the component helper should flip y.

        Returns
        -------
        torch.Tensor
            Sentinel positions with shape ``[N, 2]``.
        """
        del edge_index, node_sizes, seed, edge_weights, flip_y
        calls.append(max_iters)
        return torch.arange(num_nodes * 2, dtype=torch.float32).reshape(num_nodes, 2)

    monkeypatch.setattr(fmmm, "_graphviz_fdp_component_layout", fake_component_layout)

    positions = layout_fmmm_pipeline(
        edge_index,
        3,
        node_sizes=torch.full((3, 2), 10.0),
        steps=123,
        seed=9,
        fidelity_mode="graphviz_fdp",
    )

    assert calls == [123]
    assert positions.shape == (3, 2)
    assert torch.isfinite(positions).all()


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
