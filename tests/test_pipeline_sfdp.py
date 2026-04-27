"""Exact-fidelity tests for the composable SFDP pipeline."""

from __future__ import annotations

from typing import Iterable

import pytest
import torch

from dagua.layout.classic.sfdp import layout_sfdp
from dagua.layout.ops.pipelines.sfdp import build_sfdp_pipeline, layout_sfdp_pipeline
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
    """Assert that two SFDP outputs match exactly.

    Parameters
    ----------
    classic : torch.Tensor
        Reference output from classic SFDP.
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
    theta: float = 0.6,
    repulsive_exponent: float = -1.0,
    edge_weights: torch.Tensor | None = None,
    direction: str = "TB",
) -> torch.Tensor:
    """Execute ``build_sfdp_pipeline`` directly on a fresh solve state.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    steps : int
        Maximum number of iterations per level.
    seed : int
        Random seed.
    theta : float
        Barnes-Hut opening angle.
    repulsive_exponent : float
        SFDP repulsion exponent.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.
    direction : str, default="TB"
        Requested layout flow direction.

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
        direction=direction,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_sfdp_pipeline(
        steps=steps,
        theta=theta,
        repulsive_exponent=repulsive_exponent,
    ).apply(problem, state, ctx)
    assert final_state.pos is not None
    return final_state.pos


class TestSFDPPipelineFidelity:
    """Bit-exact regression coverage for the SFDP pipeline."""

    @pytest.mark.parametrize(
        ("num_nodes", "seed"),
        [(0, 123), (1, 123), (2, 123), (5, 123), (5, 99), (20, 123), (50, 7)],
    )
    def test_layout_sfdp_pipeline_matches_classic_for_requested_sizes(
        self,
        num_nodes: int,
        seed: int,
    ) -> None:
        """The adapter should match classic SFDP exactly for the requested cases."""
        edge_index = _path_edge_index(num_nodes)

        classic = layout_sfdp(edge_index=edge_index, num_nodes=num_nodes, steps=500, seed=seed)
        pipeline = layout_sfdp_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            steps=500,
            seed=seed,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_sfdp_pipeline_matches_classic_with_edge_weights(self) -> None:
        """Weighted SFDP should remain bit-identical in the pipeline."""
        edge_index = _edge_index_from_edges([(0, 1), (0, 2), (1, 3), (2, 4), (4, 5), (3, 5)])
        edge_weights = torch.tensor([0.25, 1.5, 2.0, 0.75, 1.25, 3.0], dtype=torch.float64)

        classic = layout_sfdp(
            edge_index=edge_index,
            num_nodes=6,
            steps=500,
            seed=17,
            edge_weights=edge_weights,
        )
        pipeline = layout_sfdp_pipeline(
            edge_index=edge_index,
            num_nodes=6,
            steps=500,
            seed=17,
            edge_weights=edge_weights,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_sfdp_pipeline_matches_classic_on_disconnected_graph(self) -> None:
        """Disconnected components and isolated nodes should match exactly."""
        edge_index = _disconnected_edge_index()

        classic = layout_sfdp(edge_index=edge_index, num_nodes=7, steps=500, seed=99)
        pipeline = layout_sfdp_pipeline(edge_index=edge_index, num_nodes=7, steps=500, seed=99)

        _assert_exact_match(classic, pipeline)

    def test_build_sfdp_pipeline_matches_classic_on_complete_graph(self) -> None:
        """The raw pipeline object should match classic SFDP on a dense graph."""
        edge_index = _complete_edge_index(5)

        classic = layout_sfdp(edge_index=edge_index, num_nodes=5, steps=500, seed=7)
        pipeline = _run_pipeline_direct(edge_index=edge_index, num_nodes=5, steps=500, seed=7)

        _assert_exact_match(classic, pipeline)

    def test_layout_sfdp_pipeline_matches_classic_with_node_sizes(self) -> None:
        """Node sizes should affect the extent scaling identically."""
        edge_index = _path_edge_index(10)
        node_sizes = torch.rand(10, 2, dtype=torch.float32) * 20.0 + 5.0

        classic = layout_sfdp(
            edge_index=edge_index,
            num_nodes=10,
            node_sizes=node_sizes,
            steps=500,
            seed=42,
        )
        pipeline = layout_sfdp_pipeline(
            edge_index=edge_index,
            num_nodes=10,
            node_sizes=node_sizes,
            steps=500,
            seed=42,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_sfdp_pipeline_matches_classic_with_custom_theta(self) -> None:
        """Custom theta should propagate identically."""
        edge_index = _path_edge_index(15)

        classic = layout_sfdp(
            edge_index=edge_index,
            num_nodes=15,
            steps=500,
            seed=42,
            theta=0.8,
        )
        pipeline = layout_sfdp_pipeline(
            edge_index=edge_index,
            num_nodes=15,
            steps=500,
            seed=42,
            theta=0.8,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_sfdp_pipeline_matches_classic_with_custom_repulsive_exponent(
        self,
    ) -> None:
        """Custom repulsive exponent should propagate identically."""
        edge_index = _path_edge_index(10)

        classic = layout_sfdp(
            edge_index=edge_index,
            num_nodes=10,
            steps=500,
            seed=42,
            repulsive_exponent=-2.0,
        )
        pipeline = layout_sfdp_pipeline(
            edge_index=edge_index,
            num_nodes=10,
            steps=500,
            seed=42,
            repulsive_exponent=-2.0,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_sfdp_pipeline_matches_classic_zero_steps(self) -> None:
        """Zero steps should produce the same output as classic."""
        edge_index = _path_edge_index(8)

        classic = layout_sfdp(edge_index=edge_index, num_nodes=8, steps=0, seed=42)
        pipeline = layout_sfdp_pipeline(edge_index=edge_index, num_nodes=8, steps=0, seed=42)

        _assert_exact_match(classic, pipeline)

    def test_layout_sfdp_pipeline_matches_classic_single_edge(self) -> None:
        """A single edge between two nodes should match exactly."""
        edge_index = _edge_index_from_edges([(0, 1)])

        classic = layout_sfdp(edge_index=edge_index, num_nodes=2, steps=500, seed=42)
        pipeline = layout_sfdp_pipeline(edge_index=edge_index, num_nodes=2, steps=500, seed=42)

        _assert_exact_match(classic, pipeline)

    def test_layout_sfdp_pipeline_matches_classic_star_graph(self) -> None:
        """Star graph topology should match exactly."""
        edges = [(0, i) for i in range(1, 8)]
        edge_index = _edge_index_from_edges(edges)

        classic = layout_sfdp(edge_index=edge_index, num_nodes=8, steps=500, seed=42)
        pipeline = layout_sfdp_pipeline(edge_index=edge_index, num_nodes=8, steps=500, seed=42)

        _assert_exact_match(classic, pipeline)

    def test_layout_sfdp_pipeline_orients_directed_path_to_requested_direction(
        self,
    ) -> None:
        """Final SFDP orientation should respect the requested directed flow."""
        edge_index = _path_edge_index(14)

        top_to_bottom = layout_sfdp_pipeline(
            edge_index=edge_index,
            num_nodes=14,
            steps=500,
            seed=42,
            direction="TB",
        )
        bottom_to_top = layout_sfdp_pipeline(
            edge_index=edge_index,
            num_nodes=14,
            steps=500,
            seed=42,
            direction="BT",
        )

        source = edge_index[0]
        target = edge_index[1]
        assert torch.all(top_to_bottom[target, 1] > top_to_bottom[source, 1])
        assert torch.all(bottom_to_top[target, 1] < bottom_to_top[source, 1])
