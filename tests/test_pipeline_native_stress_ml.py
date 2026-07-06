"""Tests for the native stress multilevel scale path."""

from __future__ import annotations

import torch

import dagua
from dagua.config import LayoutConfig
from dagua.graph import DaguaGraph
from dagua.layout.ops.pipelines.native_stress import layout_native_stress_pipeline
from dagua.layout.ops.pipelines.native_stress_ml import (
    NativeStressMLConfig,
    layout_native_stress_ml_pipeline,
    should_use_native_stress_ml,
)


def _path_edge_index(num_nodes: int) -> torch.Tensor:
    """Create a path edge tensor.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the path.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, max(num_nodes - 1, 0)]``.
    """
    if num_nodes <= 1:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(
        [list(range(num_nodes - 1)), list(range(1, num_nodes))],
        dtype=torch.long,
    )


def test_native_stress_ml_sketch_gate_node_boundary() -> None:
    """The node threshold should fire exactly at ``ml_min_nodes``."""
    config = NativeStressMLConfig(ml_min_nodes=10, ml_min_edges=100)

    assert not should_use_native_stress_ml(9, 99, config)
    assert should_use_native_stress_ml(10, 0, config)


def test_native_stress_ml_sketch_gate_edge_boundary() -> None:
    """The edge threshold should fire independently of the node threshold."""
    config = NativeStressMLConfig(ml_min_nodes=100, ml_min_edges=10)

    assert not should_use_native_stress_ml(99, 9, config)
    assert should_use_native_stress_ml(1, 10, config)


def test_native_stress_ml_below_gate_matches_plain_native_stress() -> None:
    """Below the sketch gate, ``native_stress_ml`` should be the plain core."""
    edge_index = _path_edge_index(8)
    node_sizes = torch.ones((8, 2), dtype=torch.float32)
    config = LayoutConfig(
        algorithm_params={
            "ml_min_nodes": 100,
            "ml_min_edges": 100,
            "steps": 4,
            "late_steps": 0,
            "smacof_iters": 0,
        },
        seed=13,
        steps=4,
    )

    plain = layout_native_stress_pipeline(
        edge_index=edge_index,
        num_nodes=8,
        node_sizes=node_sizes,
        config=config,
        seed=13,
    )
    multilevel = layout_native_stress_ml_pipeline(
        edge_index=edge_index,
        num_nodes=8,
        node_sizes=node_sizes,
        config=config,
        seed=13,
    )

    assert torch.equal(multilevel, plain)


def test_native_stress_ml_forced_dispatch_returns_finite_positions() -> None:
    """Public algorithm dispatch should resolve ``native_stress_ml``."""
    graph = DaguaGraph.from_edge_list([(f"n{i}", f"n{i + 1}") for i in range(11)])
    config = LayoutConfig(
        algorithm="native_stress_ml",
        algorithm_params={
            "ml_min_nodes": 8,
            "coarsest_nodes": 6,
            "max_levels": 1,
            "coarse_steps": 2,
            "refine_steps": 2,
            "overlap_max_nodes": 0,
        },
        seed=5,
        steps=2,
    )

    pos = dagua.layout(graph, config)

    assert pos.shape == (graph.num_nodes, 2)
    assert torch.isfinite(pos).all()


def test_native_stress_ml_is_seed_deterministic() -> None:
    """The multilevel path should be deterministic for a fixed seed."""
    edge_index = _path_edge_index(12)
    node_sizes = torch.ones((12, 2), dtype=torch.float32)
    config = LayoutConfig(
        algorithm_params={
            "ml_min_nodes": 8,
            "coarsest_nodes": 6,
            "max_levels": 1,
            "coarse_steps": 2,
            "refine_steps": 2,
            "overlap_max_nodes": 0,
        },
        seed=23,
        steps=2,
    )

    first = layout_native_stress_ml_pipeline(edge_index, 12, node_sizes, config=config, seed=23)
    second = layout_native_stress_ml_pipeline(edge_index, 12, node_sizes, config=config, seed=23)

    assert torch.equal(first, second)


def test_native_stress_ml_sampled_coarsest_fallback_is_deterministic() -> None:
    """The sampled coarsest fallback should repeat exactly for a fixed seed."""
    edge_index = _path_edge_index(80)
    node_sizes = torch.ones((80, 2), dtype=torch.float32)
    config = LayoutConfig(
        algorithm_params={
            "ml_min_nodes": 8,
            "coarsest_nodes": 8,
            "max_levels": 1,
            "coarse_steps": 1,
            "refine_steps": 1,
            "overlap_max_nodes": 0,
        },
        seed=31,
        steps=1,
    )

    first = layout_native_stress_ml_pipeline(edge_index, 80, node_sizes, config=config, seed=31)
    second = layout_native_stress_ml_pipeline(edge_index, 80, node_sizes, config=config, seed=31)

    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
