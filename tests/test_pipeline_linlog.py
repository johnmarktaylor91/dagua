"""Exact-fidelity tests for the composable LinLog pipeline."""

from __future__ import annotations

from typing import Iterable, Optional

import pytest
import torch

from dagua.layout.classic.linlog import layout_linlog
from dagua.layout.ops.pipelines.linlog import build_linlog_pipeline, layout_linlog_pipeline
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
    """Build a disconnected graph with multiple components and isolates.

    Returns
    -------
    torch.Tensor
        Directed edge tensor for the disconnected graph.
    """
    return _edge_index_from_edges([(0, 1), (1, 2), (3, 4), (6, 7)])


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
    """Assert that two LinLog outputs match exactly.

    Parameters
    ----------
    classic : torch.Tensor
        Reference output from classic LinLog.
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
    seed: int,
    steps: int,
    a: float = 1.0,
    r: float = 0.0,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Execute ``build_linlog_pipeline`` directly on a fresh solve state.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    seed : int
        Random seed for initialization and sampled repulsion.
    steps : int
        Number of Adam updates.
    a : float, default=1.0
        Attraction exponent.
    r : float, default=0.0
        Repulsion exponent.
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
    final_state = build_linlog_pipeline(steps=steps, a=a, r=r).apply(problem, state, ctx)
    assert final_state.pos is not None
    return final_state.pos


class TestLinLogPipelineFidelity:
    """Bit-exact regression coverage for the LinLog pipeline."""

    @pytest.mark.parametrize(
        ("num_nodes", "seed", "steps"),
        [(0, 42, 10), (1, 42, 10), (2, 42, 10), (5, 42, 30), (5, 99, 30), (20, 42, 40)],
    )
    def test_layout_linlog_pipeline_matches_classic_for_requested_sizes(
        self,
        num_nodes: int,
        seed: int,
        steps: int,
    ) -> None:
        """The adapter should match classic LinLog exactly."""
        edge_index = _path_edge_index(num_nodes)

        classic = layout_linlog(edge_index=edge_index, num_nodes=num_nodes, steps=steps, seed=seed)
        pipeline = layout_linlog_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            steps=steps,
            seed=seed,
            fidelity_mode=False,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_linlog_pipeline_matches_classic_with_edge_weights(self) -> None:
        """Weighted attraction should remain bit-identical."""
        edge_index = _edge_index_from_edges([(0, 1), (0, 2), (1, 3), (2, 4), (4, 5), (3, 5)])
        edge_weights = torch.tensor([0.25, 1.5, 2.0, 0.75, 1.25, 3.0], dtype=torch.float64)

        classic = layout_linlog(
            edge_index=edge_index,
            num_nodes=6,
            steps=35,
            seed=17,
            edge_weights=edge_weights,
        )
        pipeline = layout_linlog_pipeline(
            edge_index=edge_index,
            num_nodes=6,
            steps=35,
            seed=17,
            edge_weights=edge_weights,
            fidelity_mode=False,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_linlog_pipeline_matches_classic_on_disconnected_graph(self) -> None:
        """Disconnected components and isolates should match exactly."""
        edge_index = _disconnected_edge_index()

        classic = layout_linlog(edge_index=edge_index, num_nodes=9, steps=35, seed=99)
        pipeline = layout_linlog_pipeline(
            edge_index=edge_index,
            num_nodes=9,
            steps=35,
            seed=99,
            fidelity_mode=False,
        )

        _assert_exact_match(classic, pipeline)

    def test_build_linlog_pipeline_matches_classic_on_complete_graph(self) -> None:
        """The raw pipeline object should match classic LinLog on a dense graph."""
        edge_index = _complete_edge_index(5)

        classic = layout_linlog(edge_index=edge_index, num_nodes=5, steps=25, seed=7)
        pipeline = _run_pipeline_direct(edge_index=edge_index, num_nodes=5, steps=25, seed=7)

        _assert_exact_match(classic, pipeline)

    def test_layout_linlog_pipeline_matches_classic_with_node_sizes(self) -> None:
        """Node-size-driven output extent should remain bit-identical."""
        edge_index = _edge_index_from_edges([(0, 1), (1, 2), (2, 3), (1, 4), (2, 5)])
        node_sizes = torch.tensor(
            [
                [10.0, 12.0],
                [11.0, 8.0],
                [7.0, 9.0],
                [6.0, 6.0],
                [9.0, 10.0],
                [5.0, 7.0],
            ],
            dtype=torch.float32,
        )

        classic = layout_linlog(
            edge_index=edge_index,
            num_nodes=6,
            node_sizes=node_sizes,
            steps=35,
            seed=13,
        )
        pipeline = layout_linlog_pipeline(
            edge_index=edge_index,
            num_nodes=6,
            node_sizes=node_sizes,
            steps=35,
            seed=13,
            fidelity_mode=False,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_linlog_pipeline_matches_classic_with_custom_exponents(self) -> None:
        """Non-default attraction and repulsion exponents should stay exact."""
        edge_index = _edge_index_from_edges([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 4)])

        classic = layout_linlog(edge_index=edge_index, num_nodes=5, steps=30, seed=23, a=2.0, r=1.0)
        pipeline = layout_linlog_pipeline(
            edge_index=edge_index,
            num_nodes=5,
            steps=30,
            seed=23,
            a=2.0,
            r=1.0,
            fidelity_mode=False,
        )

        _assert_exact_match(classic, pipeline)
