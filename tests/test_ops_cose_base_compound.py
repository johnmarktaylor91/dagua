"""Unit checks for the shared cose-base compound core."""

from __future__ import annotations

import torch

from dagua.layout.ops.cose_base_compound import (
    CoSECompoundOptions,
    _calc_gravitational_force,
    _calc_repulsion_force,
    _child_maps,
    _Graph,
    _Node,
    _Rect,
    _tile_nodes,
    build_cose_compound_state,
    layout_cose_base_compound,
)


def test_cose_base_child_maps_remove_nested_members_from_direct_parent() -> None:
    """Nested clusters should expose only direct leaves to each LGraph.

    Returns
    -------
    None
        The parent graph must contain the child cluster plus only non-child
        leaves, matching Cytoscape's nested parent-node construction.
    """
    members = {"outer": [0, 1, 2, 3], "inner": [1, 2]}
    child_clusters, direct_nodes = _child_maps(members, {"inner": "outer"}, 4)

    assert child_clusters[None] == ["outer"]
    assert child_clusters["outer"] == ["inner"]
    assert direct_nodes["outer"] == [0, 3]
    assert direct_nodes["inner"] == [1, 2]


def test_cose_base_tile_nodes_matches_reference_strip_packing() -> None:
    """The tiler should use area-descending row-filling dimensions.

    Returns
    -------
    None
        Organization rows and dimensions should match the cose-base
        ``tileNodes`` arithmetic for this hand-computed case.
    """
    nodes = [
        _Node("small", _Rect(0.0, 0.0, 10.0, 10.0)),
        _Node("wide", _Rect(0.0, 0.0, 30.0, 10.0)),
        _Node("tall", _Rect(0.0, 0.0, 10.0, 40.0)),
    ]
    organization = _tile_nodes(nodes, min_width=0.0, vertical_padding=10.0, horizontal_padding=10.0)

    assert [[node.id for node in row] for row in organization.rows] == [["tall", "small"], ["wide"]]
    assert organization.row_width == [30.0, 30.0]
    assert organization.row_height == [40.0, 20.0]
    assert organization.width == 30.0
    assert organization.height == 60.0


def test_cose_base_compound_gravity_formula_uses_owner_scope() -> None:
    """Compound gravity should use compound range and strength multipliers.

    Returns
    -------
    None
        A non-root-owned node outside its owner graph range should receive
        ``-gravity * distance * gravityCompound``.
    """
    root = _Graph(parent=None)
    compound = _Node("c", _Rect(0.0, 0.0, 100.0, 100.0), owner=root)
    child = _Graph(
        parent=compound,
        left=0.0,
        right=100.0,
        top=0.0,
        bottom=100.0,
        estimated_size=10.0,
    )
    leaf = _Node("0", _Rect(180.0, 50.0, 10.0, 10.0), owner=child)
    state = build_cose_compound_state(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=0,
        node_sizes=None,
        clusters=None,
        cluster_parents=None,
        options=CoSECompoundOptions(gravity=0.25, gravity_compound=2.0, gravity_range_compound=1.5),
    )
    state.root = root

    _calc_gravitational_force(state, leaf)

    assert leaf.gravitation_force_x == -67.5
    assert leaf.gravitation_force_y == -2.5


def test_cose_base_repulsion_is_sibling_scoped() -> None:
    """Repulsion should be skipped for nodes with different owner graphs.

    Returns
    -------
    None
        Nodes in separate LGraphs must not accumulate repulsion force.
    """
    root = _Graph(parent=None)
    graph_a = _Graph(parent=_Node("a", _Rect()), nodes=[])
    graph_b = _Graph(parent=_Node("b", _Rect()), nodes=[])
    node_a = _Node("0", _Rect(0.0, 0.0, 10.0, 10.0), owner=graph_a)
    node_b = _Node("1", _Rect(100.0, 0.0, 10.0, 10.0), owner=graph_b)
    state = build_cose_compound_state(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=0,
        node_sizes=None,
        clusters=None,
        cluster_parents=None,
        options=CoSECompoundOptions(),
    )
    state.root = root

    _calc_repulsion_force(state, node_a, node_b)

    assert node_a.repulsion_force_x == 0.0
    assert node_b.repulsion_force_x == 0.0


def test_cose_base_compound_layout_is_deterministic() -> None:
    """The compound core should be deterministic for a fixed seed and options.

    Returns
    -------
    None
        Two runs with identical inputs must produce identical tensors.
    """
    edge_index = torch.tensor([[0, 1, 2, 3, 1], [1, 2, 3, 0, 4]], dtype=torch.long)
    clusters = {"outer": [0, 1, 2, 3, 4], "inner": [1, 2]}
    parents = {"inner": "outer"}
    first = layout_cose_base_compound(
        edge_index=edge_index,
        num_nodes=5,
        node_sizes=None,
        clusters=clusters,
        cluster_parents=parents,
        options=CoSECompoundOptions(steps=4, seed=11),
    )
    second = layout_cose_base_compound(
        edge_index=edge_index,
        num_nodes=5,
        node_sizes=None,
        clusters=clusters,
        cluster_parents=parents,
        options=CoSECompoundOptions(steps=4, seed=11),
    )

    assert torch.equal(first, second)
    assert first.shape == (5, 2)
    assert torch.isfinite(first).all()
