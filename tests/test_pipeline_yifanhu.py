"""Smoke tests for the native YifanHu pipeline."""

from __future__ import annotations

import torch

import dagua
from dagua import LayoutConfig
from dagua.layout.ops.pipelines import PIPELINE_REGISTRY
from dagua.layout.ops.pipelines.yifanhu import build_yifanhu_pipeline, layout_yifanhu_pipeline
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


def _edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    """Build a standard ``[2, E]`` edge tensor.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Directed edges as ``(source, target)`` pairs.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    sources, targets = zip(*edges)
    return torch.tensor([list(sources), list(targets)], dtype=torch.long)


def _assert_valid_positions(positions: torch.Tensor, num_nodes: int) -> None:
    """Assert that a coordinate tensor is finite and has expected shape.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor to validate.
    num_nodes : int
        Expected number of rows.

    Returns
    -------
    None
        The function asserts on invalid positions.
    """
    assert positions.shape == (num_nodes, 2)
    assert positions.dtype == torch.float32
    assert torch.isfinite(positions).all()


def test_yifanhu_pipeline_registered() -> None:
    """The YifanHu algorithm should be available through pipeline dispatch.

    Returns
    -------
    None
        The test asserts registry membership.
    """
    assert PIPELINE_REGISTRY["yifanhu"] == (
        "dagua.layout.ops.pipelines.yifanhu",
        "layout_yifanhu_pipeline",
    )


def test_layout_yifanhu_pipeline_smoke_path_graph() -> None:
    """The YifanHu pipeline should produce finite coordinates for a path graph.

    Returns
    -------
    None
        The test asserts finite output coordinates.
    """
    edge_index = _edge_index([(0, 1), (1, 2), (2, 3), (3, 4)])
    positions = layout_yifanhu_pipeline(edge_index=edge_index, num_nodes=5, steps=20, seed=42)

    _assert_valid_positions(positions, num_nodes=5)


def test_layout_yifanhu_pipeline_smoke_disconnected_weighted_graph() -> None:
    """The YifanHu pipeline should handle disconnected weighted inputs.

    Returns
    -------
    None
        The test asserts finite output coordinates.
    """
    edge_index = _edge_index([(0, 1), (1, 2), (3, 4), (5, 6)])
    edge_weights = torch.tensor([1.0, 2.0, 0.5, 3.0], dtype=torch.float32)
    positions = layout_yifanhu_pipeline(
        edge_index=edge_index,
        num_nodes=8,
        edge_weights=edge_weights,
        steps=15,
        seed=7,
    )

    _assert_valid_positions(positions, num_nodes=8)


def test_build_yifanhu_pipeline_direct_execution() -> None:
    """The composed YifanHu pipeline object should execute directly.

    Returns
    -------
    None
        The test asserts finite output coordinates from direct pipeline use.
    """
    edge_index = _edge_index([(0, 1), (0, 2), (2, 3), (2, 4), (4, 5)])
    problem = LayoutProblem(edge_index=edge_index, num_nodes=6, seed=99)
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))

    final_state = build_yifanhu_pipeline(steps=12).apply(problem, state, ctx)

    assert final_state.pos is not None
    _assert_valid_positions(final_state.pos, num_nodes=6)


def test_layout_dispatch_accepts_yifanhu_algorithm() -> None:
    """Public layout dispatch should accept ``algorithm='yifanhu'``.

    Returns
    -------
    None
        The test asserts finite public API output coordinates.
    """
    graph = dagua.DaguaGraph()
    for node in range(6):
        graph.add_node(str(node))
    for source, target in [(0, 1), (1, 2), (1, 3), (3, 4), (3, 5)]:
        graph.add_edge(str(source), str(target))

    positions = dagua.layout(graph, LayoutConfig(algorithm="yifanhu", steps=10, seed=11))

    _assert_valid_positions(positions, num_nodes=6)
