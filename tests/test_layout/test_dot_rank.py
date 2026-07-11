"""Regression tests for the Graphviz dot rank-assignment port."""

from __future__ import annotations

import random

import torch

from dagua.layout.ops.pipelines.dot_rank import graphviz_rank_assignment
from dagua.layout.ops.pipelines.sugiyama import layout_sugiyama_pipeline
from dagua.layout.ops.sugiyama import (
    _build_graphviz_x_aux_edges,
    _expand_long_edges_with_dummy_nodes,
    _graphviz_cluster_rank_assignments,
)


def _virtual_factory(tail: object, head: object, rank: int, edge_index: int) -> str:
    """Return a deterministic virtual node id for assertions.

    Parameters
    ----------
    tail : object
        Original edge tail.
    head : object
        Original edge head.
    rank : int
        Intermediate rank.
    edge_index : int
        Original edge index.

    Returns
    -------
    str
        Stable virtual node id.
    """
    return f"v_{tail}_{head}_{rank}_{edge_index}"


def test_graphviz_rank_assignment_matches_dot_phase1_diamond() -> None:
    """Match Graphviz 7.0.5 ``dot -Tdot`` with ``graph [phase=1]``.

    The captured reference graph is:
    ``a -> b; a -> c; b -> d; c -> d``.
    """
    ranks, virtual_edges = graphviz_rank_assignment(
        edges=[("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")],
        virtual_node_factory=_virtual_factory,
    )

    assert ranks == {"a": 0, "b": 1, "c": 1, "d": 2}
    assert virtual_edges == []


def test_graphviz_rank_assignment_matches_dot_phase1_weighted_skip() -> None:
    """Match a Graphviz phase-1 golden vector with weighted and long edges.

    The captured reference graph is:
    ``a -> b [weight=5]; a -> c; b -> d; c -> d; b -> e; e -> d``.
    """
    ranks, virtual_edges = graphviz_rank_assignment(
        edges=[
            ("a", "b", 1, 5),
            ("a", "c"),
            ("b", "d"),
            ("c", "d"),
            ("b", "e"),
            ("e", "d"),
        ],
        virtual_node_factory=_virtual_factory,
    )

    assert ranks == {"a": 0, "b": 1, "c": 2, "d": 3, "e": 2}
    assert [edge.chain for edge in virtual_edges] == [
        ("a", "v_a_c_1_1", "c"),
        ("b", "v_b_d_2_2", "d"),
    ]


def test_graphviz_rank_assignment_splits_minlen_long_edge() -> None:
    """Create virtual nodes for a Graphviz ``minlen=3`` phase-1 edge."""
    ranks, virtual_edges = graphviz_rank_assignment(
        edges=[("a", "d", 3)],
        virtual_node_factory=_virtual_factory,
    )

    assert ranks == {"a": 0, "d": 3}
    assert len(virtual_edges) == 1
    assert virtual_edges[0].virtual_nodes == ("v_a_d_1_0", "v_a_d_2_0")
    assert virtual_edges[0].chain == ("a", "v_a_d_1_0", "v_a_d_2_0", "d")


def test_graphviz_rank_assignment_handles_tensor_components() -> None:
    """Match Graphviz phase-1 ranks for disconnected tensor input."""
    edge_index = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
    ranks, virtual_edges = graphviz_rank_assignment(
        edges=edge_index,
        virtual_node_factory=_virtual_factory,
        num_nodes=4,
        edge_minlens=[1, 2],
    )

    assert ranks == {0: 0, 1: 1, 2: 0, 3: 2}
    assert [edge.chain for edge in virtual_edges] == [(2, "v_2_3_1_1", 3)]


def test_graphviz_rank_assignment_matches_random_dag_200_phase1() -> None:
    """Match the Graphviz 7.0.5 phase-1 golden ranks for ``random_dag_200``."""
    rng = random.Random(42)
    named_edges: set[tuple[str, str]] = set()
    while len(named_edges) < 300:
        tail = rng.randint(0, 198)
        head = rng.randint(tail + 1, 199)
        named_edges.add((f"n{tail}", f"n{head}"))
    edges = [(int(tail[1:]), int(head[1:])) for tail, head in sorted(named_edges)]

    ranks, _ = graphviz_rank_assignment(
        edges=edges,
        virtual_node_factory=_virtual_factory,
        num_nodes=200,
    )

    assert [ranks[node] for node in range(200)] == [
        0,
        1,
        1,
        2,
        0,
        5,
        2,
        0,
        3,
        2,
        3,
        3,
        4,
        0,
        2,
        5,
        4,
        2,
        1,
        5,
        2,
        0,
        0,
        5,
        0,
        2,
        2,
        3,
        2,
        2,
        0,
        2,
        4,
        2,
        3,
        3,
        0,
        4,
        4,
        3,
        3,
        3,
        1,
        6,
        6,
        3,
        0,
        1,
        2,
        1,
        3,
        5,
        3,
        0,
        3,
        4,
        1,
        3,
        1,
        5,
        3,
        3,
        4,
        2,
        0,
        2,
        0,
        2,
        3,
        0,
        5,
        5,
        1,
        0,
        0,
        2,
        3,
        3,
        1,
        6,
        3,
        0,
        5,
        3,
        3,
        4,
        0,
        2,
        4,
        3,
        2,
        4,
        2,
        4,
        2,
        4,
        3,
        3,
        2,
        0,
        1,
        0,
        4,
        5,
        3,
        2,
        0,
        0,
        3,
        4,
        5,
        5,
        4,
        3,
        4,
        2,
        2,
        4,
        1,
        6,
        4,
        3,
        0,
        3,
        3,
        2,
        4,
        0,
        3,
        4,
        0,
        3,
        4,
        6,
        1,
        4,
        5,
        3,
        5,
        2,
        2,
        1,
        6,
        5,
        4,
        3,
        1,
        3,
        5,
        5,
        2,
        4,
        3,
        4,
        5,
        4,
        6,
        4,
        6,
        5,
        4,
        4,
        5,
        5,
        0,
        3,
        4,
        4,
        5,
        6,
        4,
        6,
        4,
        7,
        5,
        6,
        7,
        2,
        5,
        5,
        7,
        5,
        6,
        8,
        5,
        6,
        6,
        7,
        6,
        6,
        7,
        8,
        7,
        8,
        9,
        8,
        6,
        9,
        9,
        9,
    ]


def test_sugiyama_graphviz_fidelity_uses_network_simplex_layers() -> None:
    """Use graphviz rank assignment only when graphviz fidelity is enabled."""
    edge_index = torch.tensor(
        [
            [0, 0, 2, 3],
            [1, 4, 3, 4],
        ],
        dtype=torch.long,
    )

    default_pos = layout_sugiyama_pipeline(edge_index=edge_index, num_nodes=5)
    graphviz_pos = layout_sugiyama_pipeline(
        edge_index=edge_index,
        num_nodes=5,
        fidelity_mode="graphviz",
    )

    assert torch.unique(default_pos[:, 1]).numel() == 3
    assert torch.unique(graphviz_pos[:, 1]).numel() == 3
    assert graphviz_pos[0, 1] == graphviz_pos[3, 1]
    assert default_pos[0, 1] != default_pos[3, 1]


def test_graphviz_cluster_rank_assignment_uses_member_local_offsets() -> None:
    """Collapse a ranked cluster through its leader before parent ranking."""
    edge_index = torch.tensor(
        [
            [0, 1, 2],
            [1, 3, 0],
        ],
        dtype=torch.long,
    )

    layers, _, cluster_bounds = _graphviz_cluster_rank_assignments(
        edge_index=edge_index,
        edge_weights=None,
        num_nodes=4,
        clusters={"cluster_c": [0, 1]},
        cluster_parents={"cluster_c": None},
    )

    assert layers.tolist() == [1, 2, 0, 3]
    assert cluster_bounds == {"cluster_c": (1, 2)}


def test_sugiyama_graphviz_fidelity_uses_dot_x_simplex() -> None:
    """Use Graphviz dot x-network-simplex only for graphviz fidelity mode."""
    edge_index = torch.tensor(
        [
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ],
        dtype=torch.long,
    )
    node_sizes = torch.full((11, 2), 44.0, dtype=torch.float32)

    graphviz_pos = layout_sugiyama_pipeline(
        edge_index=edge_index,
        num_nodes=11,
        node_sizes=node_sizes,
        rank_sep=1.0,
        node_sep=1.0,
        fidelity_mode="graphviz",
    )
    graphviz_dot_alias_pos = layout_sugiyama_pipeline(
        edge_index=edge_index,
        num_nodes=11,
        node_sizes=node_sizes,
        rank_sep=1.0,
        node_sep=1.0,
        fidelity_mode="graphviz_dot",
    )

    assert graphviz_pos[:, 0].tolist() == [
        0.5,
        0.0,
        1.0,
        -1.0,
        0.0,
        1.0,
        2.0,
        -2.0,
        -1.0,
        0.0,
        1.0,
    ]
    assert not torch.equal(graphviz_pos[:, 0], graphviz_dot_alias_pos[:, 0])


def test_graphviz_virtual_node_width_seed_matches_lr_minlen() -> None:
    """Seed Graphviz dummy-node half-widths before LR minlen construction."""
    edge_index = torch.tensor(
        [
            [0, 1],
            [2, 3],
        ],
        dtype=torch.long,
    )
    layer_assignments = torch.tensor([0, 0, 2, 2], dtype=torch.long)
    node_sizes = torch.full((4, 2), 54.0, dtype=torch.float32)
    expanded_graph, expanded_weights = _expand_long_edges_with_dummy_nodes(
        edge_index=edge_index,
        layer_assignments=layer_assignments,
        node_sizes=node_sizes,
        num_original_nodes=4,
        edge_weights=None,
        use_graphviz_edge_order=True,
        graphviz_virtual_node_sep=72.0,
    )
    aux_edges, _ = _build_graphviz_x_aux_edges(
        layers=expanded_graph.layers,
        edge_index=expanded_graph.edge_index,
        edge_weights=expanded_weights,
        node_sizes=expanded_graph.node_sizes,
        num_nodes=expanded_graph.num_nodes,
        num_original_nodes=4,
        node_sep=72.0,
    )

    assert expanded_graph.node_sizes[4, 0].item() == 74.0
    assert 146 in [edge[2] for edge in aux_edges if edge[3] == 0]


def test_graphviz_cluster_x_aux_edges_use_boundary_nodes() -> None:
    """Add cluster containment and sibling separation inside the x aux graph."""
    node_sizes = torch.full((4, 2), 54.0, dtype=torch.float32)

    aux_edges, initial_ranks = _build_graphviz_x_aux_edges(
        layers=[[0, 1], [2, 3]],
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_weights=None,
        node_sizes=node_sizes,
        num_nodes=4,
        num_original_nodes=4,
        node_sep=72.0,
        graphviz_cluster_members={"left": (0, 2), "right": (1, 3)},
        graphviz_cluster_parents={"left": None, "right": None},
    )

    assert initial_ranks[4] < initial_ranks[0]
    assert initial_ranks[5] > initial_ranks[2]
    assert (4, 0, 35, 0) in aux_edges
    assert (2, 5, 35, 0) in aux_edges
    assert (5, 6, 8, 0) in aux_edges
    assert (4, 5, 1, 128) in aux_edges
