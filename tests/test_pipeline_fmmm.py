"""Exact-fidelity tests for the composable FM^3 pipeline."""

from __future__ import annotations

from typing import Iterable

import pytest
import torch

from dagua.layout.classic.fmmm import layout_fmmm
from dagua.layout.ops.pipelines.fmmm import build_fmmm_pipeline, layout_fmmm_pipeline
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
    """Assert that two FM^3 outputs match exactly.

    Parameters
    ----------
    classic : torch.Tensor
        Reference output from classic FM^3.
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
    steps: int,
    seed: int,
    node_sizes: torch.Tensor | None = None,
    edge_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute ``build_fmmm_pipeline`` directly on a fresh solve state.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    steps : int
        Total refinement budget.
    seed : int
        Random seed.
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
    final_state = build_fmmm_pipeline(steps=steps).apply(problem, state, ctx)
    assert final_state.pos is not None
    return final_state.pos


class TestFMMMPipelineFidelity:
    """Bit-exact regression coverage for the FM^3 pipeline."""

    def test_empty_graph(self) -> None:
        """Zero-node graph returns an empty tensor."""
        edge_index = torch.empty((2, 0), dtype=torch.long)
        classic = layout_fmmm(edge_index=edge_index, num_nodes=0, steps=100, seed=42)
        pipeline = layout_fmmm_pipeline(edge_index=edge_index, num_nodes=0, steps=100, seed=42)
        _assert_exact_match(classic, pipeline)

    def test_single_node(self) -> None:
        """Single-node graph returns a zero tensor."""
        edge_index = torch.empty((2, 0), dtype=torch.long)
        classic = layout_fmmm(edge_index=edge_index, num_nodes=1, steps=100, seed=42)
        pipeline = layout_fmmm_pipeline(edge_index=edge_index, num_nodes=1, steps=100, seed=42)
        _assert_exact_match(classic, pipeline)

    @pytest.mark.parametrize(
        ("num_nodes", "seed"),
        [(2, 42), (5, 42), (5, 99), (10, 42), (20, 42), (50, 7)],
    )
    def test_layout_fmmm_pipeline_matches_classic_for_path_graphs(
        self,
        num_nodes: int,
        seed: int,
    ) -> None:
        """The adapter should match classic FM^3 exactly for path graphs."""
        edge_index = _path_edge_index(num_nodes)

        classic = layout_fmmm(edge_index=edge_index, num_nodes=num_nodes, steps=100, seed=seed)
        pipeline = layout_fmmm_pipeline(
            edge_index=edge_index, num_nodes=num_nodes, steps=100, seed=seed
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_fmmm_pipeline_matches_classic_with_edge_weights(self) -> None:
        """Weighted FM^3 attraction should remain bit-identical in the pipeline."""
        edge_index = _edge_index_from_edges([(0, 1), (0, 2), (1, 3), (2, 4), (4, 5), (3, 5)])
        edge_weights = torch.tensor([0.25, 1.5, 2.0, 0.75, 1.25, 3.0], dtype=torch.float64)

        classic = layout_fmmm(
            edge_index=edge_index,
            num_nodes=6,
            steps=100,
            seed=17,
            edge_weights=edge_weights,
        )
        pipeline = layout_fmmm_pipeline(
            edge_index=edge_index,
            num_nodes=6,
            steps=100,
            seed=17,
            edge_weights=edge_weights,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_fmmm_pipeline_matches_classic_on_disconnected_graph(self) -> None:
        """Disconnected components and isolated nodes should match exactly."""
        edge_index = _disconnected_edge_index()

        classic = layout_fmmm(edge_index=edge_index, num_nodes=7, steps=100, seed=99)
        pipeline = layout_fmmm_pipeline(edge_index=edge_index, num_nodes=7, steps=100, seed=99)

        _assert_exact_match(classic, pipeline)

    def test_build_fmmm_pipeline_matches_classic_on_complete_graph(self) -> None:
        """The raw pipeline object should match classic FM^3 on a dense graph."""
        edge_index = _complete_edge_index(5)

        classic = layout_fmmm(edge_index=edge_index, num_nodes=5, steps=100, seed=7)
        pipeline = _run_pipeline_direct(edge_index=edge_index, num_nodes=5, steps=100, seed=7)

        _assert_exact_match(classic, pipeline)

    def test_layout_fmmm_pipeline_matches_classic_with_node_sizes(self) -> None:
        """Node sizes affect extent calculation and should match exactly."""
        edge_index = _path_edge_index(10)
        node_sizes = torch.rand(10, 2, dtype=torch.float32) * 5.0 + 1.0

        classic = layout_fmmm(
            edge_index=edge_index,
            num_nodes=10,
            node_sizes=node_sizes,
            steps=100,
            seed=42,
        )
        pipeline = layout_fmmm_pipeline(
            edge_index=edge_index,
            num_nodes=10,
            node_sizes=node_sizes,
            steps=100,
            seed=42,
        )

        _assert_exact_match(classic, pipeline)

    @pytest.mark.parametrize("steps", [0, 10, 50, 200])
    def test_layout_fmmm_pipeline_matches_classic_for_various_step_counts(
        self,
        steps: int,
    ) -> None:
        """Different step budgets should all produce bit-identical results."""
        edge_index = _path_edge_index(15)

        classic = layout_fmmm(edge_index=edge_index, num_nodes=15, steps=steps, seed=42)
        pipeline = layout_fmmm_pipeline(edge_index=edge_index, num_nodes=15, steps=steps, seed=42)

        _assert_exact_match(classic, pipeline)

    def test_layout_fmmm_pipeline_matches_classic_large_graph_triggers_hierarchy(
        self,
    ) -> None:
        """A graph larger than the coarse target should trigger multilevel coarsening."""
        num_nodes = 80
        edge_index = _path_edge_index(num_nodes)

        classic = layout_fmmm(edge_index=edge_index, num_nodes=num_nodes, steps=100, seed=42)
        pipeline = layout_fmmm_pipeline(
            edge_index=edge_index, num_nodes=num_nodes, steps=100, seed=42
        )

        _assert_exact_match(classic, pipeline)

    def test_build_fmmm_pipeline_matches_classic_large_complete(self) -> None:
        """Direct pipeline invocation on a larger dense graph should match."""
        edge_index = _complete_edge_index(8)

        classic = layout_fmmm(edge_index=edge_index, num_nodes=8, steps=100, seed=99)
        pipeline = _run_pipeline_direct(edge_index=edge_index, num_nodes=8, steps=100, seed=99)

        _assert_exact_match(classic, pipeline)
