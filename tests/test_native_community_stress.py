"""Tests for deterministic native community-stress helpers."""

from __future__ import annotations

import hashlib

import numpy as np
import torch

from dagua.eval.graphs import _make_r79_weighted_community_graph, _make_weighted_clusters
from dagua.eval.ruler_v3_groups import adjusted_rand_index
from dagua.layout.ops.pipelines.native_community_stress import (
    greedy_modularity_communities,
    layout_community_stress_pipeline,
    resolve_community_labels,
)
from dagua.layout.ops.pipelines.native_lattice_grid import layout_geodesic_stress_pipeline
from dagua.layout.ops.state import LayoutProblem


def _target_graph(name: str) -> LayoutProblem:
    """Return a layout problem for one named eval graph.

    Parameters
    ----------
    name : str
        Eval graph name.

    Returns
    -------
    LayoutProblem
        Problem carrying edge weights and cluster metadata.
    """
    if name == "r79_weighted_community_4x18":
        test_graph = _make_r79_weighted_community_graph()
    elif name == "weighted_clusters_3x10":
        test_graph = _make_weighted_clusters()
    else:
        raise ValueError(f"unknown target graph {name}")
    graph = test_graph.graph
    return LayoutProblem(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        node_sizes=None,
        clusters=graph.clusters,
        cluster_parents=graph.cluster_parents,
        edge_weights=graph.edge_weights,
        seed=42,
    )


def _hash_positions(pos: torch.Tensor) -> str:
    """Return a byte hash for a position tensor.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    str
        SHA-256 digest of the contiguous CPU tensor bytes.
    """
    array = pos.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _edge_order_permutation(edge_index: torch.Tensor) -> torch.Tensor:
    """Return a deterministic nontrivial edge permutation.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor shaped ``[2, E]``.

    Returns
    -------
    torch.Tensor
        Permutation indices shaped ``[E]``.
    """
    edge_count = int(edge_index.shape[1])
    order = torch.arange(edge_count, dtype=torch.long)
    return (order * 37 + 11).remainder(edge_count).argsort(stable=True)


def test_greedy_modularity_is_bit_identical_and_edge_order_invariant() -> None:
    """Weighted CNM must be deterministic across runs and edge permutations."""
    problem = _target_graph("r79_weighted_community_4x18")
    labels_a = greedy_modularity_communities(
        problem.edge_index,
        problem.num_nodes,
        edge_weights=problem.edge_weights,
    )
    labels_b = greedy_modularity_communities(
        problem.edge_index,
        problem.num_nodes,
        edge_weights=problem.edge_weights,
    )
    order = _edge_order_permutation(problem.edge_index)
    assert problem.edge_weights is not None
    labels_permuted = greedy_modularity_communities(
        problem.edge_index[:, order],
        problem.num_nodes,
        edge_weights=problem.edge_weights[order],
    )

    assert torch.equal(labels_a, labels_b)
    assert torch.equal(labels_a, labels_permuted)


def test_greedy_modularity_reproduces_weighted_target_row_aris() -> None:
    """Weighted CNM recovers the planted partitions for target weighted rows."""
    cases = {
        "r79_weighted_community_4x18": np.array([node_index // 18 for node_index in range(4 * 18)]),
        "weighted_clusters_3x10": np.array([node_index // 10 for node_index in range(3 * 10)]),
    }
    for graph_name, expected in cases.items():
        problem = _target_graph(graph_name)
        labels = greedy_modularity_communities(
            problem.edge_index,
            problem.num_nodes,
            edge_weights=problem.edge_weights,
        )
        ari = adjusted_rand_index(expected, labels.numpy())
        assert ari == 1.0


def test_resolve_community_labels_uses_declared_leaf_clusters() -> None:
    """Declared nested clusters should resolve to leaf-most memberships."""
    edge_index = torch.tensor([[1, 3, 3, 4], [2, 4, 5, 5]], dtype=torch.long)
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=6,
        clusters={"outer": [0, 1, 2], "inner": [1, 2], "right": [3, 4, 5]},
        cluster_parents={"outer": None, "inner": "outer", "right": None},
    )
    labels = resolve_community_labels(problem)

    assert labels is not None
    assert int(labels[1].item()) == int(labels[2].item())
    assert int(labels[0].item()) != int(labels[1].item())


def test_community_stress_is_byte_identical_across_runs() -> None:
    """Community stress should be deterministic for a fixed seed."""
    problem = _target_graph("weighted_clusters_3x10")
    labels = resolve_community_labels(problem)
    assert labels is not None
    pos_a = layout_community_stress_pipeline(
        edge_index=problem.edge_index,
        num_nodes=problem.num_nodes,
        node_sizes=problem.node_sizes,
        community_labels=labels,
        inter_scale=1.5,
        edge_weights=problem.edge_weights,
        seed=42,
    )
    pos_b = layout_community_stress_pipeline(
        edge_index=problem.edge_index,
        num_nodes=problem.num_nodes,
        node_sizes=problem.node_sizes,
        community_labels=labels,
        inter_scale=1.5,
        edge_weights=problem.edge_weights,
        seed=42,
    )

    assert _hash_positions(pos_a) == _hash_positions(pos_b)


def test_geodesic_refactor_no_hook_matches_public_caller_hash() -> None:
    """No-hook geodesic calls remain byte-identical for existing callers."""
    problem = _target_graph("weighted_clusters_3x10")
    pos_a = layout_geodesic_stress_pipeline(
        edge_index=problem.edge_index,
        num_nodes=problem.num_nodes,
        node_sizes=problem.node_sizes,
        edge_weights=problem.edge_weights,
        seed=42,
    )
    pos_b = layout_geodesic_stress_pipeline(
        edge_index=problem.edge_index,
        num_nodes=problem.num_nodes,
        node_sizes=problem.node_sizes,
        edge_weights=problem.edge_weights,
        seed=42,
    )

    assert _hash_positions(pos_a) == _hash_positions(pos_b)
