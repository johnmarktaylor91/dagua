"""Exact-fidelity tests for the composable FA2 pipeline."""

from __future__ import annotations

from typing import Iterable

import pytest
import torch

from dagua.layout.classic.fa2 import layout_fa2
from dagua.layout.ops.pipelines.fa2 import (
    FA2Config,
    build_fa2_pipeline,
    layout_fa2_pipeline,
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
    """Assert that two FA2 outputs match exactly.

    Parameters
    ----------
    classic : torch.Tensor
        Reference output from classic FA2.
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
        f"Mismatch: max abs diff = {(classic - pipeline).abs().max().item()}"
    )


def _run_pipeline_direct(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    config: FA2Config,
    seed: int,
    edge_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute ``build_fa2_pipeline`` directly on a fresh solve state.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    config : FA2Config
        FA2 configuration.
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
    final_state = build_fa2_pipeline(config=config).apply(problem, state, ctx)
    assert final_state.pos is not None
    return final_state.pos


class TestFA2PipelineFidelity:
    """Bit-exact regression coverage for the FA2 pipeline."""

    @pytest.mark.parametrize(
        ("num_nodes", "seed"),
        [(0, 42), (1, 42), (2, 42), (5, 42), (5, 99), (20, 42), (50, 7)],
    )
    def test_layout_fa2_pipeline_matches_classic_for_requested_sizes(
        self,
        num_nodes: int,
        seed: int,
    ) -> None:
        """The adapter should match classic FA2 exactly for the requested cases."""
        edge_index = _path_edge_index(num_nodes)

        classic = layout_fa2(edge_index=edge_index, num_nodes=num_nodes, steps=50, seed=seed)
        pipeline = layout_fa2_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            steps=50,
            seed=seed,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_fa2_pipeline_matches_classic_with_edge_weights(self) -> None:
        """Weighted FA2 attraction should remain bit-identical in the pipeline."""
        edge_index = _edge_index_from_edges([(0, 1), (0, 2), (1, 3), (2, 4), (4, 5), (3, 5)])
        edge_weights = torch.tensor([0.25, 1.5, 2.0, 0.75, 1.25, 3.0], dtype=torch.float64)

        classic = layout_fa2(
            edge_index=edge_index,
            num_nodes=6,
            steps=50,
            seed=17,
            edge_weights=edge_weights,
        )
        pipeline = layout_fa2_pipeline(
            edge_index=edge_index,
            num_nodes=6,
            steps=50,
            seed=17,
            edge_weights=edge_weights,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_fa2_pipeline_matches_classic_on_disconnected_graph(self) -> None:
        """Disconnected components and isolated nodes should match exactly."""
        edge_index = _disconnected_edge_index()

        classic = layout_fa2(edge_index=edge_index, num_nodes=7, steps=50, seed=99)
        pipeline = layout_fa2_pipeline(edge_index=edge_index, num_nodes=7, steps=50, seed=99)

        _assert_exact_match(classic, pipeline)

    def test_build_fa2_pipeline_matches_classic_on_complete_graph(self) -> None:
        """The raw pipeline object should match classic FA2 on a dense graph."""
        edge_index = _complete_edge_index(5)
        config = FA2Config(steps=50)

        classic = layout_fa2(edge_index=edge_index, num_nodes=5, steps=50, seed=7)
        pipeline = _run_pipeline_direct(edge_index=edge_index, num_nodes=5, config=config, seed=7)

        _assert_exact_match(classic, pipeline)

    def test_layout_fa2_pipeline_matches_classic_linlog(self) -> None:
        """LinLog attraction mode should remain bit-identical."""
        edge_index = _path_edge_index(10)

        classic = layout_fa2(edge_index=edge_index, num_nodes=10, steps=30, seed=42, linlog=True)
        pipeline = layout_fa2_pipeline(
            edge_index=edge_index, num_nodes=10, steps=30, seed=42, linlog=True
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_fa2_pipeline_matches_classic_strong_gravity(self) -> None:
        """Strong gravity mode should remain bit-identical."""
        edge_index = _path_edge_index(10)

        classic = layout_fa2(
            edge_index=edge_index, num_nodes=10, steps=30, seed=42, strong_gravity=True
        )
        pipeline = layout_fa2_pipeline(
            edge_index=edge_index, num_nodes=10, steps=30, seed=42, strong_gravity=True
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_fa2_pipeline_matches_classic_dissuade_hubs(self) -> None:
        """Dissuade hubs mode should remain bit-identical."""
        edge_index = _complete_edge_index(6)

        classic = layout_fa2(
            edge_index=edge_index, num_nodes=6, steps=30, seed=42, dissuade_hubs=True
        )
        pipeline = layout_fa2_pipeline(
            edge_index=edge_index, num_nodes=6, steps=30, seed=42, dissuade_hubs=True
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_fa2_pipeline_matches_classic_no_outbound_att(self) -> None:
        """Disabling outbound attraction distribution should remain bit-identical."""
        edge_index = _path_edge_index(8)

        classic = layout_fa2(
            edge_index=edge_index,
            num_nodes=8,
            steps=30,
            seed=42,
            outbound_attraction_distribution=False,
        )
        pipeline = layout_fa2_pipeline(
            edge_index=edge_index,
            num_nodes=8,
            steps=30,
            seed=42,
            outbound_attraction_distribution=False,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_fa2_pipeline_matches_classic_edge_weight_influence(self) -> None:
        """Non-default edge_weight_influence should remain bit-identical."""
        edge_index = _edge_index_from_edges([(0, 1), (1, 2), (2, 3), (3, 4)])
        edge_weights = torch.tensor([2.0, 0.5, 1.5, 3.0], dtype=torch.float32)

        classic = layout_fa2(
            edge_index=edge_index,
            num_nodes=5,
            steps=30,
            seed=42,
            edge_weights=edge_weights,
            edge_weight_influence=0.5,
        )
        pipeline = layout_fa2_pipeline(
            edge_index=edge_index,
            num_nodes=5,
            steps=30,
            seed=42,
            edge_weights=edge_weights,
            edge_weight_influence=0.5,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_fa2_pipeline_matches_classic_barnes_hut(self) -> None:
        """Barnes-Hut approximation mode should remain bit-identical."""
        edge_index = _path_edge_index(15)

        classic = layout_fa2(
            edge_index=edge_index,
            num_nodes=15,
            steps=30,
            seed=42,
            barnes_hut=True,
            barnes_hut_theta=1.2,
        )
        pipeline = layout_fa2_pipeline(
            edge_index=edge_index,
            num_nodes=15,
            steps=30,
            seed=42,
            barnes_hut=True,
            barnes_hut_theta=1.2,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_fa2_pipeline_matches_classic_custom_gravity_scaling(self) -> None:
        """Custom gravity and scaling_ratio should remain bit-identical."""
        edge_index = _path_edge_index(10)

        classic = layout_fa2(
            edge_index=edge_index,
            num_nodes=10,
            steps=30,
            seed=42,
            gravity=5.0,
            scaling_ratio=10.0,
        )
        pipeline = layout_fa2_pipeline(
            edge_index=edge_index,
            num_nodes=10,
            steps=30,
            seed=42,
            gravity=5.0,
            scaling_ratio=10.0,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_fa2_pipeline_matches_classic_zero_steps(self) -> None:
        """Zero steps should produce only initialized positions."""
        edge_index = _path_edge_index(5)

        classic = layout_fa2(edge_index=edge_index, num_nodes=5, steps=0, seed=42)
        pipeline = layout_fa2_pipeline(edge_index=edge_index, num_nodes=5, steps=0, seed=42)

        _assert_exact_match(classic, pipeline)

    def test_layout_fa2_pipeline_matches_classic_combined_flags(self) -> None:
        """Multiple flags combined should remain bit-identical."""
        edge_index = _complete_edge_index(8)
        edge_weights = torch.ones(edge_index.shape[1], dtype=torch.float32) * 2.0

        classic = layout_fa2(
            edge_index=edge_index,
            num_nodes=8,
            steps=25,
            seed=7,
            linlog=True,
            strong_gravity=True,
            dissuade_hubs=True,
            outbound_attraction_distribution=True,
            edge_weights=edge_weights,
            edge_weight_influence=0.0,
            gravity=3.0,
            scaling_ratio=5.0,
        )
        pipeline = layout_fa2_pipeline(
            edge_index=edge_index,
            num_nodes=8,
            steps=25,
            seed=7,
            linlog=True,
            strong_gravity=True,
            dissuade_hubs=True,
            outbound_attraction_distribution=True,
            edge_weights=edge_weights,
            edge_weight_influence=0.0,
            gravity=3.0,
            scaling_ratio=5.0,
        )

        _assert_exact_match(classic, pipeline)
