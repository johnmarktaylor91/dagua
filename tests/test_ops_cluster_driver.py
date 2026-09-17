"""Tests for the recursive-to-iterative ClusterAwareDriver placement (drywell R2-B1 F-2)."""

from __future__ import annotations

import sys

import torch

import dagua
from dagua.config import LayoutConfig
from dagua.layout.ops.pipelines.native_undirected import _cluster_aware_sfdp_candidate
from dagua.layout.ops.state import (
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)


def _deep_chain_clusters(depth: int, num_nodes: int) -> tuple[dict, dict]:
    """Build a ``depth``-deep linear singleton cluster chain over ``num_nodes``."""
    names = [f"c{index:06d}" for index in range(depth)]
    clusters = {name: [index % num_nodes] for index, name in enumerate(names)}
    parent_of = {names[index]: names[index - 1] for index in range(1, depth)}
    return clusters, parent_of


def test_engine_dispatch_fr_completes_on_1200_deep_cluster_chain() -> None:
    """Drywell R2-B1 F-2 regression: deep valid nesting must not crash dispatch.

    ``ClusterAwareDriver._place_level`` recursed one frame per nesting level,
    so a 6-node path with a 1200-deep singleton cluster chain crashed
    ``dagua.layout(..., algorithm="fr")`` with RecursionError (997
    cluster_driver frames) in EVERY name order once the tree build was made
    depth-safe. The iterative post-order placement must complete at Python's
    default recursion limit.
    """
    graph = dagua.DaguaGraph()
    for index in range(6):
        graph.add_node(f"n{index}")
    for index in range(5):
        graph.add_edge(f"n{index}", f"n{index + 1}")
    clusters, parent_of = _deep_chain_clusters(1200, 6)
    graph.clusters.update(clusters)
    graph.cluster_parents.update(parent_of)

    previous_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(1000)
    try:
        pos = dagua.layout(graph, LayoutConfig(algorithm="fr", seed=42))
    finally:
        sys.setrecursionlimit(previous_limit)

    assert pos is not None and pos.shape == (6, 2)
    assert torch.isfinite(pos).all()


def test_cluster_sfdp_challenger_fires_on_deep_nested_row() -> None:
    """Drywell R2-B1 F-2 regression: the cluster-SFDP arm must fire when deep.

    On the default native path the challenger ran through the same recursive
    driver inside a family ``except Exception``, so deep nesting silently
    disarmed the one structural cluster-placement candidate (r80-S9). Calling
    the candidate directly (no swallow in the way) must return finite
    positions at the default recursion limit.
    """
    num_nodes = 8
    edges = [(index, index + 1) for index in range(num_nodes - 1)] + [(0, 4)]
    clusters, parent_of = _deep_chain_clusters(1200, num_nodes)
    clusters["real_a"] = [0, 1, 2, 3]
    clusters["real_b"] = [4, 5, 6, 7]
    problem = LayoutProblem(
        edge_index=torch.tensor(list(zip(*edges)), dtype=torch.long),
        num_nodes=num_nodes,
        node_sizes=torch.ones((num_nodes, 2), dtype=torch.float32) * 20.0,
        seed=42,
        clusters=clusters,
        cluster_parents=parent_of,
    )
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu", optimizer_type="adam"))

    previous_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(1000)
    try:
        candidate = _cluster_aware_sfdp_candidate(problem, LayoutConfig(seed=42), ctx)
    finally:
        sys.setrecursionlimit(previous_limit)

    assert candidate is not None
    assert candidate.shape == (num_nodes, 2)
    assert torch.isfinite(candidate).all()


def test_cluster_driver_shallow_hierarchy_placements_intact() -> None:
    """The iterative rewrite keeps ordinary nested placement behavior.

    A small two-level hierarchy through the public driver entry must still
    produce one placement per cluster (children before parents in the
    placements insertion order) and finite positions.
    """
    from dagua.layout.engine import _build_cluster_inner_pipeline
    from dagua.layout.ops.cluster_driver import ClusterAwareDriver

    problem = LayoutProblem(
        edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long),
        num_nodes=6,
        node_sizes=torch.ones((6, 2), dtype=torch.float32) * 20.0,
        seed=7,
        clusters={"outer": [0, 1, 2, 3], "inner": [1, 2]},
        cluster_parents={"inner": "outer"},
    )
    inner_pipeline = _build_cluster_inner_pipeline("fr", LayoutConfig(seed=7))
    driver = ClusterAwareDriver(inner_pipeline=inner_pipeline.ops)
    state = driver.apply(
        problem,
        SolveState(),
        RuntimeContext(plan=ExecutionPlan(device="cpu", optimizer_type="adam")),
    )

    assert state.pos is not None and torch.isfinite(state.pos).all()
    placements = state.extras["cluster_inner_layout"]
    assert list(placements) == ["inner", "outer"]
