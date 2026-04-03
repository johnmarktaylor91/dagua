"""Exact-fidelity tests for the composable GEM pipeline."""

from __future__ import annotations

from typing import Iterable

import pytest
import torch

from dagua.layout.classic.gem import layout_gem
from dagua.layout.ops.pipelines.gem import build_gem_pipeline, layout_gem_pipeline
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


def _edge_index_from_edges(edges: Iterable[tuple[int, int]]) -> torch.Tensor:
    """Build a standard ``[2, E]`` edge tensor from edge tuples.

    Parameters
    ----------
    edges : Iterable[tuple[int, int]]
        Directed edges as ``(source, target)`` pairs.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    edge_list = list(edges)
    if not edge_list:
        return torch.empty((2, 0), dtype=torch.long)
    sources, targets = zip(*edge_list)
    return torch.tensor([list(sources), list(targets)], dtype=torch.long)


def _path_edge_index(num_nodes: int) -> torch.Tensor:
    """Build a directed path graph edge tensor.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    torch.Tensor
        Directed path graph edge tensor.
    """
    return _edge_index_from_edges((index, index + 1) for index in range(max(num_nodes - 1, 0)))


def _disconnected_edge_index() -> torch.Tensor:
    """Build a small disconnected graph with two components and isolates.

    Returns
    -------
    torch.Tensor
        Directed edge tensor for the disconnected graph.
    """
    return _edge_index_from_edges([(0, 1), (1, 2), (3, 4)])


def _complete_edge_index(num_nodes: int) -> torch.Tensor:
    """Build a directed complete graph without self-loops.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    torch.Tensor
        Dense directed complete graph edge tensor.
    """
    return _edge_index_from_edges(
        (source, target)
        for source in range(num_nodes)
        for target in range(num_nodes)
        if source != target
    )


def _assert_exact_match(classic: torch.Tensor, pipeline: torch.Tensor) -> None:
    """Assert that two GEM outputs match exactly.

    Parameters
    ----------
    classic : torch.Tensor
        Reference output from classic GEM.
    pipeline : torch.Tensor
        Output from the composable pipeline.

    Returns
    -------
    None
        This helper asserts exact equality.
    """
    assert classic.dtype == pipeline.dtype
    assert classic.device == pipeline.device
    assert torch.equal(classic, pipeline)


def _run_pipeline_direct(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    max_iters: int,
    seed: int,
    node_sizes: torch.Tensor | None = None,
    edge_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute ``build_gem_pipeline`` directly on a fresh solve state.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    max_iters : int
        Maximum number of OGDF node updates.
    seed : int
        Random seed used for initialization.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Final position tensor produced by the pipeline.
    """
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_gem_pipeline(max_iters=max_iters).apply(problem, state, ctx)
    assert final_state.pos is not None
    return final_state.pos


class TestGEMPipelineFidelity:
    """Bit-exact regression coverage for the GEM pipeline."""

    @pytest.mark.parametrize(
        ("num_nodes", "seed"),
        [(0, 42), (1, 42), (2, 42), (5, 42), (5, 99), (20, 42), (50, 7)],
    )
    def test_layout_gem_pipeline_matches_classic_for_requested_sizes(
        self,
        num_nodes: int,
        seed: int,
    ) -> None:
        """The adapter should match classic GEM exactly for the requested cases."""
        edge_index = _path_edge_index(num_nodes)

        classic = layout_gem(edge_index=edge_index, num_nodes=num_nodes, max_iters=500, seed=seed)
        pipeline = layout_gem_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            max_iters=500,
            seed=seed,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_gem_pipeline_matches_classic_with_edge_weights(self) -> None:
        """Weighted GEM attraction should remain bit-identical in the pipeline."""
        edge_index = _edge_index_from_edges([(0, 1), (0, 2), (1, 3), (2, 4), (4, 5), (3, 5)])
        edge_weights = torch.tensor([0.25, 1.5, 2.0, 0.75, 1.25, 3.0], dtype=torch.float64)

        classic = layout_gem(
            edge_index=edge_index,
            num_nodes=6,
            max_iters=500,
            seed=17,
            edge_weights=edge_weights,
        )
        pipeline = layout_gem_pipeline(
            edge_index=edge_index,
            num_nodes=6,
            max_iters=500,
            seed=17,
            edge_weights=edge_weights,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_gem_pipeline_matches_classic_on_disconnected_graph(self) -> None:
        """Disconnected components and isolated nodes should match exactly."""
        edge_index = _disconnected_edge_index()

        classic = layout_gem(edge_index=edge_index, num_nodes=7, max_iters=500, seed=99)
        pipeline = layout_gem_pipeline(edge_index=edge_index, num_nodes=7, max_iters=500, seed=99)

        _assert_exact_match(classic, pipeline)

    def test_build_gem_pipeline_matches_classic_on_complete_graph(self) -> None:
        """The raw pipeline object should match classic GEM on a dense graph."""
        edge_index = _complete_edge_index(5)

        classic = layout_gem(edge_index=edge_index, num_nodes=5, max_iters=500, seed=7)
        pipeline = _run_pipeline_direct(edge_index=edge_index, num_nodes=5, max_iters=500, seed=7)

        _assert_exact_match(classic, pipeline)

    def test_layout_gem_pipeline_matches_classic_with_node_sizes(self) -> None:
        """Node sizes affect extent and desired lengths -- must remain identical."""
        edge_index = _path_edge_index(10)
        node_sizes = torch.rand((10, 2), dtype=torch.float32) * 20.0 + 5.0

        classic = layout_gem(
            edge_index=edge_index,
            num_nodes=10,
            node_sizes=node_sizes,
            max_iters=500,
            seed=42,
        )
        pipeline = layout_gem_pipeline(
            edge_index=edge_index,
            num_nodes=10,
            node_sizes=node_sizes,
            max_iters=500,
            seed=42,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_gem_pipeline_matches_classic_low_iters(self) -> None:
        """Low iteration count should still be bit-identical."""
        edge_index = _path_edge_index(8)

        classic = layout_gem(edge_index=edge_index, num_nodes=8, max_iters=10, seed=42)
        pipeline = layout_gem_pipeline(edge_index=edge_index, num_nodes=8, max_iters=10, seed=42)

        _assert_exact_match(classic, pipeline)

    def test_layout_gem_pipeline_matches_classic_zero_iters(self) -> None:
        """Zero iterations should produce identical output."""
        edge_index = _path_edge_index(5)

        classic = layout_gem(edge_index=edge_index, num_nodes=5, max_iters=0, seed=42)
        pipeline = layout_gem_pipeline(edge_index=edge_index, num_nodes=5, max_iters=0, seed=42)

        _assert_exact_match(classic, pipeline)

    def test_build_gem_pipeline_matches_classic_with_edge_weights_direct(self) -> None:
        """Direct pipeline invocation with edge weights should match."""
        edge_index = _edge_index_from_edges([(0, 1), (1, 2), (2, 3)])
        edge_weights = torch.tensor([2.0, 0.5, 1.5], dtype=torch.float32)

        classic = layout_gem(
            edge_index=edge_index,
            num_nodes=4,
            max_iters=200,
            seed=77,
            edge_weights=edge_weights,
        )
        pipeline = _run_pipeline_direct(
            edge_index=edge_index,
            num_nodes=4,
            max_iters=200,
            seed=77,
            edge_weights=edge_weights,
        )

        _assert_exact_match(classic, pipeline)
