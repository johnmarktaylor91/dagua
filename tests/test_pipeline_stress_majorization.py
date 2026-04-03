"""Exact-fidelity tests for the composable stress majorization pipeline."""

from __future__ import annotations

from typing import Iterable

import pytest
import torch

from dagua.layout.classic.stress_majorization import layout_stress_majorization
from dagua.layout.ops.pipelines.stress_majorization import (
    build_stress_majorization_pipeline,
    layout_stress_majorization_pipeline,
)
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
    """Assert that two stress majorization outputs match exactly.

    Parameters
    ----------
    classic : torch.Tensor
        Reference output from classic stress majorization.
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
    iterations: int,
    seed: int,
    edge_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute ``build_stress_majorization_pipeline`` directly on a fresh state.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    iterations : int
        Number of SMACOF majorization steps.
    seed : int
        Random seed used for initialization.
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
        edge_weights=edge_weights,
        seed=seed,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_stress_majorization_pipeline(
        iterations=iterations,
    ).apply(problem, state, ctx)
    assert final_state.pos is not None
    return final_state.pos


class TestStressMajorizationPipelineFidelity:
    """Bit-exact regression coverage for the stress majorization pipeline."""

    @pytest.mark.parametrize(
        ("num_nodes", "seed"),
        [(0, 42), (1, 42), (2, 42), (5, 42), (5, 99), (20, 42), (50, 7)],
    )
    def test_pipeline_matches_classic_for_requested_sizes(
        self,
        num_nodes: int,
        seed: int,
    ) -> None:
        """The adapter should match classic stress majorization exactly."""
        edge_index = _path_edge_index(num_nodes)

        classic = layout_stress_majorization(
            edge_index=edge_index,
            num_nodes=num_nodes,
            iterations=200,
            seed=seed,
        )
        pipeline = layout_stress_majorization_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            iterations=200,
            seed=seed,
        )

        _assert_exact_match(classic, pipeline)

    def test_pipeline_matches_classic_with_edge_weights(self) -> None:
        """Weighted stress majorization should remain bit-identical."""
        edge_index = _edge_index_from_edges([(0, 1), (0, 2), (1, 3), (2, 4), (4, 5), (3, 5)])
        edge_weights = torch.tensor([0.25, 1.5, 2.0, 0.75, 1.25, 3.0], dtype=torch.float64)

        classic = layout_stress_majorization(
            edge_index=edge_index,
            num_nodes=6,
            iterations=200,
            seed=17,
            edge_weights=edge_weights,
        )
        pipeline = layout_stress_majorization_pipeline(
            edge_index=edge_index,
            num_nodes=6,
            iterations=200,
            seed=17,
            edge_weights=edge_weights,
        )

        _assert_exact_match(classic, pipeline)

    def test_pipeline_matches_classic_on_disconnected_graph(self) -> None:
        """Disconnected components and isolated nodes should match exactly."""
        edge_index = _disconnected_edge_index()

        classic = layout_stress_majorization(
            edge_index=edge_index, num_nodes=7, iterations=200, seed=99
        )
        pipeline = layout_stress_majorization_pipeline(
            edge_index=edge_index, num_nodes=7, iterations=200, seed=99
        )

        _assert_exact_match(classic, pipeline)

    def test_build_pipeline_matches_classic_on_complete_graph(self) -> None:
        """The raw pipeline object should match classic on a dense graph."""
        edge_index = _complete_edge_index(5)

        classic = layout_stress_majorization(
            edge_index=edge_index, num_nodes=5, iterations=200, seed=7
        )
        pipeline = _run_pipeline_direct(edge_index=edge_index, num_nodes=5, iterations=200, seed=7)

        _assert_exact_match(classic, pipeline)

    def test_pipeline_matches_classic_with_trace(self) -> None:
        """Trace output should match classic trace output exactly."""
        edge_index = _path_edge_index(5)

        classic_result = layout_stress_majorization(
            edge_index=edge_index,
            num_nodes=5,
            iterations=50,
            seed=42,
            trace_every=10,
        )
        pipeline_result = layout_stress_majorization_pipeline(
            edge_index=edge_index,
            num_nodes=5,
            iterations=50,
            seed=42,
            trace_every=10,
        )

        assert isinstance(classic_result, tuple)
        assert isinstance(pipeline_result, tuple)
        classic_pos, classic_traces = classic_result
        pipeline_pos, pipeline_traces = pipeline_result

        _assert_exact_match(classic_pos, pipeline_pos)
        assert len(classic_traces) == len(pipeline_traces)
        for i, (ct, pt) in enumerate(zip(classic_traces, pipeline_traces)):
            assert torch.equal(ct, pt), f"Trace {i} differs"

    def test_pipeline_matches_classic_zero_iterations(self) -> None:
        """Zero iterations should match classic (just init and finalize)."""
        edge_index = _path_edge_index(5)

        classic = layout_stress_majorization(
            edge_index=edge_index, num_nodes=5, iterations=0, seed=42
        )
        pipeline = layout_stress_majorization_pipeline(
            edge_index=edge_index, num_nodes=5, iterations=0, seed=42
        )

        _assert_exact_match(classic, pipeline)

    def test_pipeline_matches_classic_few_iterations(self) -> None:
        """Small iteration counts should still be bit-identical."""
        edge_index = _path_edge_index(10)

        for iters in [1, 3, 10]:
            classic = layout_stress_majorization(
                edge_index=edge_index, num_nodes=10, iterations=iters, seed=42
            )
            pipeline = layout_stress_majorization_pipeline(
                edge_index=edge_index, num_nodes=10, iterations=iters, seed=42
            )

            _assert_exact_match(classic, pipeline)
