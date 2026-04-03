"""Exact-fidelity tests for the composable stress-SGD pipeline."""

from __future__ import annotations

from typing import Iterable

import pytest
import torch

from dagua.layout.classic.stress_sgd import layout_stress_sgd
from dagua.layout.ops.pipelines.stress_sgd import (
    build_stress_sgd_pipeline,
    layout_stress_sgd_pipeline,
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
    """Assert that two stress-SGD outputs match exactly.

    Parameters
    ----------
    classic : torch.Tensor
        Reference output from classic stress-SGD.
    pipeline : torch.Tensor
        Output from the composable pipeline.

    Returns
    -------
    None
        This helper asserts exact equality.
    """
    assert classic.dtype == pipeline.dtype, (
        f"dtype mismatch: classic={classic.dtype}, pipeline={pipeline.dtype}"
    )
    assert classic.device == pipeline.device, (
        f"device mismatch: classic={classic.device}, pipeline={pipeline.device}"
    )
    assert torch.equal(classic, pipeline), (
        f"Tensor values differ. Max abs diff: {(classic - pipeline).abs().max().item():.2e}"
    )


def _run_pipeline_direct(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    steps: int,
    seed: int,
    edge_weights: torch.Tensor | None = None,
    trace_every: int = 0,
    eps: float = 0.01,
    max_exact_nodes: int = 10_000,
    sample_size: int | str = "auto",
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    """Execute ``build_stress_sgd_pipeline`` directly on a fresh solve state.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    steps : int
        Number of SGD epochs.
    seed : int
        Random seed used for initialization.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.
    trace_every : int, default=0
        Trace snapshot interval.
    eps : float, default=0.01
        Final schedule scale factor.
    max_exact_nodes : int, default=10000
        Exact code path threshold.
    sample_size : int or str, default="auto"
        Large-graph sampling budget.

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, list[torch.Tensor]]
        Final positions or (positions, traces).
    """
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weights=edge_weights,
        seed=seed,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_stress_sgd_pipeline(
        steps=steps,
        eps=eps,
        max_exact_nodes=max_exact_nodes,
        sample_size=sample_size,
        trace_every=trace_every,
    ).apply(problem, state, ctx)
    assert final_state.pos is not None
    traces = final_state.extras.get("stress_sgd_traces", [])
    if trace_every > 0:
        return final_state.pos, traces
    return final_state.pos


class TestStressSGDPipelineFidelity:
    """Bit-exact regression coverage for the stress-SGD pipeline."""

    @pytest.mark.parametrize(
        ("num_nodes", "seed"),
        [(0, 42), (1, 42), (2, 42), (5, 42), (5, 99), (10, 42), (20, 7)],
    )
    def test_pipeline_matches_classic_for_various_sizes(
        self,
        num_nodes: int,
        seed: int,
    ) -> None:
        """The adapter should match classic stress-SGD exactly for path graphs."""
        edge_index = _path_edge_index(num_nodes)

        classic = layout_stress_sgd(edge_index=edge_index, num_nodes=num_nodes, steps=30, seed=seed)
        pipeline = layout_stress_sgd_pipeline(
            edge_index=edge_index, num_nodes=num_nodes, steps=30, seed=seed
        )

        _assert_exact_match(classic, pipeline)

    def test_pipeline_matches_classic_on_disconnected_graph(self) -> None:
        """Disconnected graphs should fall back identically."""
        edge_index = _disconnected_edge_index()

        classic = layout_stress_sgd(edge_index=edge_index, num_nodes=7, steps=30, seed=99)
        pipeline = layout_stress_sgd_pipeline(edge_index=edge_index, num_nodes=7, steps=30, seed=99)

        _assert_exact_match(classic, pipeline)

    def test_pipeline_matches_classic_on_complete_graph(self) -> None:
        """Dense complete graphs should match exactly via the raw pipeline."""
        edge_index = _complete_edge_index(5)

        classic = layout_stress_sgd(edge_index=edge_index, num_nodes=5, steps=30, seed=7)
        pipeline = _run_pipeline_direct(edge_index=edge_index, num_nodes=5, steps=30, seed=7)

        _assert_exact_match(classic, pipeline)

    def test_pipeline_matches_classic_with_edge_weights(self) -> None:
        """Weighted stress-SGD should remain bit-identical in the pipeline."""
        edge_index = _edge_index_from_edges([(0, 1), (0, 2), (1, 3), (2, 4), (4, 5), (3, 5)])
        edge_weights = torch.tensor([0.25, 1.5, 2.0, 0.75, 1.25, 3.0], dtype=torch.float64)

        classic = layout_stress_sgd(
            edge_index=edge_index,
            num_nodes=6,
            steps=30,
            seed=17,
            edge_weights=edge_weights,
        )
        pipeline = layout_stress_sgd_pipeline(
            edge_index=edge_index,
            num_nodes=6,
            steps=30,
            seed=17,
            edge_weights=edge_weights,
        )

        _assert_exact_match(classic, pipeline)

    def test_pipeline_matches_classic_zero_steps(self) -> None:
        """Zero steps should produce valid output matching classic."""
        edge_index = _path_edge_index(5)

        classic = layout_stress_sgd(edge_index=edge_index, num_nodes=5, steps=0, seed=42)
        pipeline = layout_stress_sgd_pipeline(edge_index=edge_index, num_nodes=5, steps=0, seed=42)

        _assert_exact_match(classic, pipeline)

    def test_pipeline_matches_classic_with_tracing(self) -> None:
        """Trace output should be identical between classic and pipeline."""
        edge_index = _path_edge_index(5)

        classic_result = layout_stress_sgd(
            edge_index=edge_index,
            num_nodes=5,
            steps=10,
            seed=42,
            trace_every=3,
        )
        pipeline_result = layout_stress_sgd_pipeline(
            edge_index=edge_index,
            num_nodes=5,
            steps=10,
            seed=42,
            trace_every=3,
        )

        assert isinstance(classic_result, tuple)
        assert isinstance(pipeline_result, tuple)
        classic_pos, classic_traces = classic_result
        pipeline_pos, pipeline_traces = pipeline_result

        _assert_exact_match(classic_pos, pipeline_pos)
        assert len(classic_traces) == len(pipeline_traces), (
            f"Trace count mismatch: {len(classic_traces)} vs {len(pipeline_traces)}"
        )
        for i, (ct, pt) in enumerate(zip(classic_traces, pipeline_traces)):
            assert torch.equal(ct, pt), f"Trace {i} differs"

    def test_pipeline_matches_classic_different_eps(self) -> None:
        """Custom eps should produce identical results."""
        edge_index = _path_edge_index(8)

        classic = layout_stress_sgd(edge_index=edge_index, num_nodes=8, steps=15, seed=42, eps=0.1)
        pipeline = layout_stress_sgd_pipeline(
            edge_index=edge_index, num_nodes=8, steps=15, seed=42, eps=0.1
        )

        _assert_exact_match(classic, pipeline)

    def test_pipeline_matches_classic_approximate_path(self) -> None:
        """Force the approximate code path and verify bit-identity.

        Uses max_exact_nodes=0 to push even a small graph through the
        pivot-based approximation.
        """
        edge_index = _path_edge_index(10)

        classic = layout_stress_sgd(
            edge_index=edge_index,
            num_nodes=10,
            steps=15,
            seed=42,
            max_exact_nodes=0,
        )
        pipeline = layout_stress_sgd_pipeline(
            edge_index=edge_index,
            num_nodes=10,
            steps=15,
            seed=42,
            max_exact_nodes=0,
        )

        _assert_exact_match(classic, pipeline)

    def test_build_pipeline_matches_classic_on_star_graph(self) -> None:
        """Star graph via the raw build_stress_sgd_pipeline."""
        edges = [(0, i) for i in range(1, 8)]
        edge_index = _edge_index_from_edges(edges)

        classic = layout_stress_sgd(edge_index=edge_index, num_nodes=8, steps=20, seed=11)
        pipeline = _run_pipeline_direct(edge_index=edge_index, num_nodes=8, steps=20, seed=11)

        _assert_exact_match(classic, pipeline)

    def test_pipeline_matches_classic_cycle_graph(self) -> None:
        """Cycle graph should match exactly."""
        n = 6
        edges = [(i, (i + 1) % n) for i in range(n)]
        edge_index = _edge_index_from_edges(edges)

        classic = layout_stress_sgd(edge_index=edge_index, num_nodes=n, steps=30, seed=42)
        pipeline = layout_stress_sgd_pipeline(edge_index=edge_index, num_nodes=n, steps=30, seed=42)

        _assert_exact_match(classic, pipeline)


class TestStressSGDPipelineValidation:
    """Input validation tests for the pipeline adapter."""

    def test_negative_num_nodes_raises(self) -> None:
        """Negative num_nodes should raise ValueError."""
        edge_index = torch.empty((2, 0), dtype=torch.long)
        with pytest.raises(ValueError, match="num_nodes"):
            layout_stress_sgd_pipeline(edge_index=edge_index, num_nodes=-1, steps=10, seed=42)

    def test_negative_steps_raises(self) -> None:
        """Negative steps should raise ValueError."""
        edge_index = torch.empty((2, 0), dtype=torch.long)
        with pytest.raises(ValueError, match="steps"):
            layout_stress_sgd_pipeline(edge_index=edge_index, num_nodes=5, steps=-1, seed=42)

    def test_negative_trace_every_raises(self) -> None:
        """Negative trace_every should raise ValueError."""
        edge_index = _path_edge_index(5)
        with pytest.raises(ValueError, match="trace_every"):
            layout_stress_sgd_pipeline(
                edge_index=edge_index,
                num_nodes=5,
                steps=10,
                seed=42,
                trace_every=-1,
            )

    def test_nonpositive_eps_raises(self) -> None:
        """Non-positive eps should raise ValueError."""
        edge_index = _path_edge_index(5)
        with pytest.raises(ValueError, match="eps"):
            layout_stress_sgd_pipeline(
                edge_index=edge_index, num_nodes=5, steps=10, seed=42, eps=0.0
            )

    def test_negative_max_exact_nodes_raises(self) -> None:
        """Negative max_exact_nodes should raise ValueError."""
        edge_index = _path_edge_index(5)
        with pytest.raises(ValueError, match="max_exact_nodes"):
            layout_stress_sgd_pipeline(
                edge_index=edge_index,
                num_nodes=5,
                steps=10,
                seed=42,
                max_exact_nodes=-1,
            )

    def test_negative_steps_in_build_raises(self) -> None:
        """Negative steps in build_stress_sgd_pipeline should raise."""
        with pytest.raises(ValueError, match="steps"):
            build_stress_sgd_pipeline(steps=-1)
