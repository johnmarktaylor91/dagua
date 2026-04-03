"""Exact-fidelity tests for the composable classical MDS pipeline."""

from __future__ import annotations

from typing import Iterable, Optional

import pytest
import torch

from dagua.layout.classic.classical_mds import layout_classical_mds
from dagua.layout.ops.pipelines.classical_mds import (
    build_classical_mds_pipeline,
    layout_classical_mds_pipeline,
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
    """Assert that two classical MDS outputs match exactly.

    Parameters
    ----------
    classic : torch.Tensor
        Reference output from classic classical MDS.
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
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Execute ``build_classical_mds_pipeline`` directly on a fresh solve state.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
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
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_classical_mds_pipeline().apply(problem, state, ctx)
    assert final_state.pos is not None
    return final_state.pos


class TestClassicalMDSPipelineFidelity:
    """Bit-exact regression coverage for the classical MDS pipeline."""

    @pytest.mark.parametrize(
        ("num_nodes", "seed"),
        [(0, 42), (1, 42), (2, 42), (5, 42), (5, 99), (20, 42), (50, 7)],
    )
    def test_layout_classical_mds_pipeline_matches_classic_for_requested_sizes(
        self,
        num_nodes: int,
        seed: int,
    ) -> None:
        """The adapter should match classic classical MDS exactly."""
        edge_index = _path_edge_index(num_nodes)

        classic = layout_classical_mds(edge_index=edge_index, num_nodes=num_nodes, seed=seed)
        pipeline = layout_classical_mds_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            seed=seed,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_classical_mds_pipeline_matches_classic_with_edge_weights(self) -> None:
        """Weighted shortest-path targets should remain bit-identical."""
        edge_index = _edge_index_from_edges([(0, 1), (0, 2), (1, 3), (2, 4), (4, 5), (3, 5)])
        edge_weights = torch.tensor([0.25, 1.5, 2.0, 0.75, 1.25, 3.0], dtype=torch.float64)

        classic = layout_classical_mds(
            edge_index=edge_index,
            num_nodes=6,
            seed=17,
            edge_weights=edge_weights,
        )
        pipeline = layout_classical_mds_pipeline(
            edge_index=edge_index,
            num_nodes=6,
            seed=17,
            edge_weights=edge_weights,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_classical_mds_pipeline_matches_classic_on_disconnected_graph(self) -> None:
        """Disconnected components and isolated nodes should match exactly."""
        edge_index = _disconnected_edge_index()

        classic = layout_classical_mds(edge_index=edge_index, num_nodes=9, seed=99)
        pipeline = layout_classical_mds_pipeline(edge_index=edge_index, num_nodes=9, seed=99)

        _assert_exact_match(classic, pipeline)

    def test_build_classical_mds_pipeline_matches_classic_on_complete_graph(self) -> None:
        """The raw pipeline object should match classic classical MDS on a dense graph."""
        edge_index = _complete_edge_index(5)

        classic = layout_classical_mds(edge_index=edge_index, num_nodes=5, seed=7)
        pipeline = _run_pipeline_direct(edge_index=edge_index, num_nodes=5)

        _assert_exact_match(classic, pipeline)

    def test_layout_classical_mds_pipeline_matches_classic_with_node_sizes(self) -> None:
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

        classic = layout_classical_mds(
            edge_index=edge_index,
            num_nodes=6,
            seed=13,
            node_sizes=node_sizes,
        )
        pipeline = layout_classical_mds_pipeline(
            edge_index=edge_index,
            num_nodes=6,
            seed=13,
            node_sizes=node_sizes,
        )

        _assert_exact_match(classic, pipeline)
