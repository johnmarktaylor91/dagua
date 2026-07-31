"""Unit checks for the shared cose-base compound core."""

from __future__ import annotations

import math

import pytest
import torch

from dagua.layout.ops.cose_base_compound import (
    CoSECompoundOptions,
    _calc_estimated_size_graph,
    _calc_gravitational_force,
    _calc_ideal_edge_lengths,
    _calc_inclusion_depths,
    _calc_lowest_common_ancestors,
    _calc_repulsion_force,
    _calc_repulsion_forces,
    _child_maps,
    _Graph,
    _Node,
    _Rect,
    _scatter_node,
    _tile_nodes,
    _update_graph_bounds,
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


def test_cose_base_repulsion_uses_grid_surrounding_range() -> None:
    """FR-grid repulsion should only include Cytoscape surrounding siblings.

    Returns
    -------
    None
        The first tick should refresh grid neighborhoods and ignore siblings
        outside the CoSE repulsion range.
    """
    state = build_cose_compound_state(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=3,
        node_sizes=torch.full((3, 2), 30.0),
        clusters=None,
        cluster_parents=None,
        options=CoSECompoundOptions(),
    )
    near_left = state.nodes_by_leaf[0]
    near_right = state.nodes_by_leaf[1]
    far = state.nodes_by_leaf[2]
    near_left.rect.x = 0.0
    near_left.rect.y = 0.0
    near_right.rect.x = 40.0
    near_right.rect.y = 0.0
    far.rect.x = 500.0
    far.rect.y = 0.0
    state.total_iterations = 1
    state.repulsion_range = 100.0
    _update_graph_bounds(state.root, True)

    _calc_repulsion_forces(state, grid_update_allowed=True, force_surrounding_update=False)

    assert near_left.surrounding == [near_right]
    assert far not in near_left.surrounding
    assert near_left.repulsion_force_x != 0.0
    assert far.repulsion_force_x == 0.0


def test_cose_base_inter_graph_ideal_length_uses_layout_base_depths() -> None:
    """Inter-graph ideal lengths should use layout-base node depths.

    Returns
    -------
    None
        Leaves under one compound should have depth two, producing the
        Cytoscape 1.0.3 ideal length for this micro compound edge.
    """
    edge_index = torch.tensor([[2], [3]], dtype=torch.long)
    state = build_cose_compound_state(
        edge_index=edge_index,
        num_nodes=6,
        node_sizes=torch.full((6, 2), 30.0),
        clusters={"a": [0, 1, 2], "b": [3, 4, 5]},
        cluster_parents=None,
        options=CoSECompoundOptions(),
    )

    _calc_lowest_common_ancestors(state)
    _calc_inclusion_depths(state)
    _calc_estimated_size_graph(state.root)
    _calc_ideal_edge_lengths(state)

    edge = state.edges[0]
    assert state.nodes_by_leaf[2].inclusion_tree_depth == 2
    assert state.nodes_by_leaf[3].inclusion_tree_depth == 2
    assert edge.ideal_length == pytest.approx(83.92304845413264)


def test_cose_base_random_scatter_matches_layout_base_world_frame() -> None:
    """Random scatter should use layout-base's world-centered top-left frame.

    Returns
    -------
    None
        The first sine-RNG draw pair should match Cytoscape layout-base.
    """
    state = build_cose_compound_state(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=1,
        node_sizes=torch.full((1, 2), 30.0),
        clusters=None,
        cluster_parents=None,
        options=CoSECompoundOptions(),
    )
    node = state.nodes_by_leaf[0]
    expected_x = 1200.0 + ((math.sin(1) * 10000.0) % 1.0) * 2000.0 - 1000.0
    expected_y = 900.0 + ((math.sin(2) * 10000.0) % 1.0) * 2000.0 - 1000.0

    _scatter_node(state, node)

    assert node.rect.x == pytest.approx(expected_x)
    assert node.rect.y == pytest.approx(expected_y)


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
