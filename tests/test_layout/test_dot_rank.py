"""Regression tests for the Graphviz dot rank-assignment port."""

from __future__ import annotations

import torch

from dagua.layout.ops.pipelines.dot_rank import graphviz_rank_assignment
from dagua.layout.ops.pipelines.sugiyama import layout_sugiyama_pipeline


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
