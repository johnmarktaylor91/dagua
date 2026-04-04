"""Exact-fidelity tests for the composable tsNET pipeline."""

from __future__ import annotations

from typing import Iterable

import pytest
import torch

from dagua.layout.classic.tsnet import layout_tsnet
from dagua.layout.ops.pipelines.tsnet import build_tsnet_pipeline, layout_tsnet_pipeline
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.tsnet import TsnetPrepareState


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
    """Assert that two tsNET outputs match exactly.

    Parameters
    ----------
    classic : torch.Tensor
        Reference output from classic tsNET.
    pipeline : torch.Tensor
        Output from the composable pipeline.

    Returns
    -------
    None
        This helper asserts exact equality.
    """
    assert classic.dtype == pipeline.dtype
    assert classic.device == pipeline.device
    assert torch.equal(classic, pipeline), (
        f"Outputs differ.\n"
        f"  max abs diff: {(classic - pipeline).abs().max().item()}\n"
        f"  classic[:3]:  {classic[:3].tolist()}\n"
        f"  pipeline[:3]: {pipeline[:3].tolist()}"
    )


def _run_pipeline_direct(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    steps: int,
    seed: int,
    edge_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute ``build_tsnet_pipeline`` directly on a fresh solve state.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    steps : int
        Number of optimization updates.
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
    final_state = build_tsnet_pipeline(steps=steps).apply(problem, state, ctx)
    assert final_state.pos is not None
    return final_state.pos


def test_tsnet_prepare_state_populates_typed_distance_matrix() -> None:
    """tsNET preprocessing should cache all-pairs distances on the typed field."""
    problem = LayoutProblem(
        edge_index=_path_edge_index(4),
        num_nodes=4,
        seed=42,
    )
    state = SolveState(extras={"tsnet_perplexity": 3.0})

    prepared = TsnetPrepareState().apply(problem, state, RuntimeContext())

    assert prepared.distance_matrix is not None
    assert prepared.distance_matrix.tolist() == [
        [0.0, 1.0, 2.0, 3.0],
        [1.0, 0.0, 1.0, 2.0],
        [2.0, 1.0, 0.0, 1.0],
        [3.0, 2.0, 1.0, 0.0],
    ]
    assert "tsnet_probabilities" in prepared.extras


class TestTsnetPipelineFidelity:
    """Bit-exact regression coverage for the tsNET pipeline."""

    @pytest.mark.parametrize(
        ("num_nodes", "seed"),
        [(0, 42), (1, 42), (2, 42), (5, 42), (5, 99), (10, 42)],
    )
    def test_layout_tsnet_pipeline_matches_classic_for_requested_sizes(
        self,
        num_nodes: int,
        seed: int,
    ) -> None:
        """The adapter should match classic tsNET exactly for the requested cases."""
        edge_index = _path_edge_index(num_nodes)
        # Use fewer steps for test speed -- fidelity is independent of step count.
        steps = 50

        classic = layout_tsnet(edge_index=edge_index, num_nodes=num_nodes, steps=steps, seed=seed)
        pipeline = layout_tsnet_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            steps=steps,
            seed=seed,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_tsnet_pipeline_matches_classic_with_edge_weights(self) -> None:
        """Weighted tsNET distances should remain bit-identical in the pipeline."""
        edge_index = _edge_index_from_edges([(0, 1), (0, 2), (1, 3), (2, 4), (4, 5), (3, 5)])
        edge_weights = torch.tensor([0.25, 1.5, 2.0, 0.75, 1.25, 3.0], dtype=torch.float64)

        classic = layout_tsnet(
            edge_index=edge_index,
            num_nodes=6,
            steps=50,
            seed=17,
            edge_weights=edge_weights,
        )
        pipeline = layout_tsnet_pipeline(
            edge_index=edge_index,
            num_nodes=6,
            steps=50,
            seed=17,
            edge_weights=edge_weights,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_tsnet_pipeline_matches_classic_on_disconnected_graph(self) -> None:
        """Disconnected components and isolated nodes should match exactly."""
        edge_index = _disconnected_edge_index()

        classic = layout_tsnet(edge_index=edge_index, num_nodes=7, steps=50, seed=99)
        pipeline = layout_tsnet_pipeline(edge_index=edge_index, num_nodes=7, steps=50, seed=99)

        _assert_exact_match(classic, pipeline)

    def test_build_tsnet_pipeline_matches_classic_on_complete_graph(self) -> None:
        """The raw pipeline object should match classic tsNET on a dense graph."""
        edge_index = _complete_edge_index(5)

        classic = layout_tsnet(edge_index=edge_index, num_nodes=5, steps=50, seed=7)
        pipeline = _run_pipeline_direct(edge_index=edge_index, num_nodes=5, steps=50, seed=7)

        _assert_exact_match(classic, pipeline)

    def test_layout_tsnet_pipeline_zero_steps(self) -> None:
        """Zero steps should still produce valid output matching classic."""
        edge_index = _path_edge_index(5)

        classic = layout_tsnet(edge_index=edge_index, num_nodes=5, steps=0, seed=42)
        pipeline = layout_tsnet_pipeline(edge_index=edge_index, num_nodes=5, steps=0, seed=42)

        _assert_exact_match(classic, pipeline)
