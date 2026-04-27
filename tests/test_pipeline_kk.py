"""Exact-fidelity tests for the composable Kamada-Kawai pipeline."""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import pytest
import torch

from dagua.layout.classic.kk import layout_kk
from dagua.layout.ops.pipelines.kk import build_kk_pipeline, layout_kk_pipeline
from dagua.layout.ops.state import (
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)


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
    """Build a disconnected graph with multiple components.

    Returns
    -------
    torch.Tensor
        Directed edge tensor for the disconnected graph.
    """
    return _edge_index_from_edges([(0, 1), (1, 2), (3, 4), (4, 5), (6, 7)])


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
    """Assert that two KK outputs match exactly.

    Parameters
    ----------
    classic : torch.Tensor
        Reference output from classic KK.
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


def _assert_trace_match(
    classic: Tuple[torch.Tensor, List[torch.Tensor]],
    pipeline: Tuple[torch.Tensor, List[torch.Tensor]],
) -> None:
    """Assert that the final positions and trace snapshots match exactly.

    Parameters
    ----------
    classic : tuple[torch.Tensor, list[torch.Tensor]]
        Reference output from classic KK trace mode.
    pipeline : tuple[torch.Tensor, list[torch.Tensor]]
        Output from the composable pipeline trace mode.

    Returns
    -------
    None
        This helper asserts exact equality.
    """
    classic_pos, classic_traces = classic
    pipeline_pos, pipeline_traces = pipeline

    _assert_exact_match(classic_pos, pipeline_pos)
    assert len(classic_traces) == len(pipeline_traces)
    for classic_trace, pipeline_trace in zip(classic_traces, pipeline_traces):
        assert classic_trace.dtype == pipeline_trace.dtype
        assert classic_trace.device == pipeline_trace.device
        assert torch.equal(classic_trace, pipeline_trace)


def _top_to_bottom_fraction(positions: torch.Tensor, edge_index: torch.Tensor) -> float:
    """Return the share of edges whose target is below the source.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.

    Returns
    -------
    float
        Fraction of edges satisfying top-to-bottom direction.
    """
    if edge_index.numel() == 0:
        return 1.0
    source = edge_index[0]
    target = edge_index[1]
    aligned = positions[target, 1] >= positions[source, 1]
    return float(aligned.to(dtype=torch.float32).mean().item())


def _run_pipeline_direct(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    steps: Optional[int],
    trace_every: int,
    pos: Optional[torch.Tensor] = None,
    edge_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Execute ``build_kk_pipeline`` directly on a fresh solve state.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    steps : int, optional
        Maximum L-BFGS-B iterations.
    trace_every : int
        Callback snapshot cadence.
    pos : torch.Tensor, optional
        Optional initial positions with shape ``[N, 2]``.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.

    Returns
    -------
    tuple[torch.Tensor, list[torch.Tensor]]
        Final positions and trace snapshots produced by the pipeline.
    """
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weights=edge_weights,
    )
    state = SolveState()
    if pos is not None:
        state.extras["kk_initial_pos"] = pos
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_kk_pipeline(steps=steps, trace_every=trace_every).apply(problem, state, ctx)
    assert final_state.pos is not None
    return final_state.pos, final_state.extras.get("kk_traces", [])


class TestKKPipelineFidelity:
    """Bit-exact regression coverage for the KK pipeline."""

    @pytest.mark.parametrize(
        ("num_nodes", "seed"),
        [(0, 42), (1, 42), (2, 42), (5, 42), (5, 99), (20, 42)],
    )
    def test_layout_kk_pipeline_matches_classic_for_requested_sizes(
        self,
        num_nodes: int,
        seed: int,
    ) -> None:
        """The adapter should match classic KK exactly for these cases."""
        pytest.importorskip("scipy")
        edge_index = _path_edge_index(num_nodes)

        classic = layout_kk(edge_index=edge_index, num_nodes=num_nodes, steps=40, seed=seed)
        pipeline = layout_kk_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            steps=40,
            seed=seed,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_kk_pipeline_matches_classic_with_edge_weights(self) -> None:
        """Weighted shortest-path targets should remain bit-identical."""
        pytest.importorskip("scipy")
        edge_index = _edge_index_from_edges([(0, 1), (0, 2), (1, 3), (2, 4), (4, 5), (3, 5)])
        edge_weights = torch.tensor([0.25, 1.5, 2.0, 0.75, 1.25, 3.0], dtype=torch.float64)

        classic = layout_kk(
            edge_index=edge_index,
            num_nodes=6,
            steps=60,
            seed=17,
            edge_weights=edge_weights,
        )
        pipeline = layout_kk_pipeline(
            edge_index=edge_index,
            num_nodes=6,
            steps=60,
            seed=17,
            edge_weights=edge_weights,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_kk_pipeline_can_orient_to_graph_direction(self) -> None:
        """Direction orientation should fix KK's arbitrary vertical sign."""
        pytest.importorskip("scipy")
        edge_index = _path_edge_index(6)

        raw = layout_kk_pipeline(edge_index=edge_index, num_nodes=6, steps=40)
        oriented = layout_kk_pipeline(
            edge_index=edge_index,
            num_nodes=6,
            steps=40,
            direction="TB",
            orient_to_direction=True,
        )

        assert _top_to_bottom_fraction(raw, edge_index) == 0.0
        assert _top_to_bottom_fraction(oriented, edge_index) == 1.0
        assert torch.equal(oriented[:, 0], raw[:, 0])
        assert torch.equal(oriented[:, 1], -raw[:, 1])

    def test_layout_kk_pipeline_matches_classic_on_disconnected_graph(self) -> None:
        """Disconnected components and isolates should match exactly."""
        pytest.importorskip("scipy")
        edge_index = _disconnected_edge_index()

        classic = layout_kk(edge_index=edge_index, num_nodes=9, steps=60, seed=99)
        pipeline = layout_kk_pipeline(edge_index=edge_index, num_nodes=9, steps=60, seed=99)

        _assert_exact_match(classic, pipeline)

    def test_build_kk_pipeline_matches_classic_on_complete_graph(self) -> None:
        """The raw pipeline object should match classic KK on a dense graph."""
        pytest.importorskip("scipy")
        edge_index = _complete_edge_index(5)

        classic = layout_kk(edge_index=edge_index, num_nodes=5, steps=40, seed=7)
        pipeline, traces = _run_pipeline_direct(
            edge_index=edge_index,
            num_nodes=5,
            steps=40,
            trace_every=0,
        )

        _assert_exact_match(classic, pipeline)
        assert traces == []

    def test_layout_kk_pipeline_matches_classic_with_trace_and_custom_pos(self) -> None:
        """Trace mode and explicit initial positions should remain bit-identical."""
        pytest.importorskip("scipy")
        edge_index = _edge_index_from_edges([(0, 1), (1, 2), (2, 3), (1, 4), (2, 5)])
        initial_pos = torch.tensor(
            [
                [-0.5, 0.1],
                [-0.1, 0.6],
                [0.2, -0.4],
                [0.8, 0.3],
                [-0.7, -0.2],
                [0.5, 0.9],
            ],
            dtype=torch.float32,
        )

        classic = layout_kk(
            edge_index=edge_index,
            num_nodes=6,
            steps=35,
            seed=13,
            trace_every=5,
            pos=initial_pos,
        )
        pipeline = layout_kk_pipeline(
            edge_index=edge_index,
            num_nodes=6,
            steps=35,
            seed=13,
            trace_every=5,
            pos=initial_pos,
        )

        _assert_trace_match(classic, pipeline)
