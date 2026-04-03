"""Exact-fidelity tests for the composable DrL pipeline."""

from __future__ import annotations

from typing import Iterable

import pytest
import torch

from dagua.layout.classic.drl import layout_drl
from dagua.layout.ops.pipelines.drl import build_drl_pipeline, layout_drl_pipeline
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


def _star_edge_index(num_nodes: int) -> torch.Tensor:
    """Build a star graph with node 0 as the hub.

    Parameters
    ----------
    num_nodes : int
        Total number of nodes including the hub.

    Returns
    -------
    torch.Tensor
        Star graph edge tensor.
    """
    return _edge_index_from_edges((0, target) for target in range(1, num_nodes))


def _assert_exact_match(classic: torch.Tensor, pipeline: torch.Tensor) -> None:
    """Assert that two DrL outputs match exactly.

    Parameters
    ----------
    classic : torch.Tensor
        Reference output from classic DrL.
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
        f"Max abs diff: {(classic - pipeline).abs().max().item()}"
    )


def _run_pipeline_direct(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    seed: int,
    options: str = "default",
    edge_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute ``build_drl_pipeline`` directly on a fresh solve state.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    seed : int
        Random seed used for initialization.
    options : str, default="default"
        DrL preset name.
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
    final_state = build_drl_pipeline(options=options).apply(problem, state, ctx)
    assert final_state.pos is not None
    return final_state.pos


class TestDRLPipelineFidelity:
    """Bit-exact regression coverage for the DrL pipeline."""

    @pytest.mark.parametrize(
        ("num_nodes", "seed"),
        [(0, 42), (1, 42), (2, 42), (5, 42), (5, 99), (10, 42), (20, 7)],
    )
    def test_layout_drl_pipeline_matches_classic_for_requested_sizes(
        self,
        num_nodes: int,
        seed: int,
    ) -> None:
        """The adapter should match classic DrL exactly for path graphs."""
        edge_index = _path_edge_index(num_nodes)

        classic = layout_drl(
            edge_index=edge_index,
            num_nodes=num_nodes,
            seed=seed,
        )
        pipeline = layout_drl_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            seed=seed,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_drl_pipeline_matches_classic_with_edge_weights(self) -> None:
        """Weighted DrL should remain bit-identical in the pipeline."""
        edge_index = _edge_index_from_edges([(0, 1), (0, 2), (1, 3), (2, 4), (4, 5), (3, 5)])
        edge_weights = torch.tensor([0.25, 1.5, 2.0, 0.75, 1.25, 3.0], dtype=torch.float64)

        classic = layout_drl(
            edge_index=edge_index,
            num_nodes=6,
            seed=17,
            edge_weights=edge_weights,
        )
        pipeline = layout_drl_pipeline(
            edge_index=edge_index,
            num_nodes=6,
            seed=17,
            edge_weights=edge_weights,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_drl_pipeline_matches_classic_on_disconnected_graph(self) -> None:
        """Disconnected components and isolated nodes should match exactly."""
        edge_index = _disconnected_edge_index()

        classic = layout_drl(
            edge_index=edge_index,
            num_nodes=7,
            seed=99,
        )
        pipeline = layout_drl_pipeline(
            edge_index=edge_index,
            num_nodes=7,
            seed=99,
        )

        _assert_exact_match(classic, pipeline)

    def test_build_drl_pipeline_matches_classic_on_complete_graph(self) -> None:
        """The raw pipeline object should match classic DrL on a dense graph."""
        edge_index = _complete_edge_index(5)

        classic = layout_drl(
            edge_index=edge_index,
            num_nodes=5,
            seed=7,
        )
        pipeline = _run_pipeline_direct(
            edge_index=edge_index,
            num_nodes=5,
            seed=7,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_drl_pipeline_matches_classic_on_star_graph(self) -> None:
        """Star topology exercises hub-spoke attraction dynamics."""
        edge_index = _star_edge_index(8)

        classic = layout_drl(
            edge_index=edge_index,
            num_nodes=8,
            seed=42,
        )
        pipeline = layout_drl_pipeline(
            edge_index=edge_index,
            num_nodes=8,
            seed=42,
        )

        _assert_exact_match(classic, pipeline)

    @pytest.mark.parametrize("preset", ["default", "coarsen", "coarsest", "refine", "final"])
    def test_layout_drl_pipeline_matches_classic_for_presets(self, preset: str) -> None:
        """All 5 DrL presets should produce bit-identical output."""
        edge_index = _path_edge_index(8)

        classic = layout_drl(
            edge_index=edge_index,
            num_nodes=8,
            seed=42,
            options=preset,
        )
        pipeline = layout_drl_pipeline(
            edge_index=edge_index,
            num_nodes=8,
            seed=42,
            options=preset,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_drl_pipeline_matches_classic_with_custom_options(self) -> None:
        """Custom per-phase overrides via mapping should match exactly."""
        edge_index = _path_edge_index(6)
        custom_options = {
            "liquid_iterations": 100,
            "expansion_iterations": 50,
            "cooldown_temperature": 500.0,
        }

        classic = layout_drl(
            edge_index=edge_index,
            num_nodes=6,
            seed=42,
            options=custom_options,
        )
        pipeline = layout_drl_pipeline(
            edge_index=edge_index,
            num_nodes=6,
            seed=42,
            options=custom_options,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_drl_pipeline_empty_graph(self) -> None:
        """Zero-node graph should return empty tensor with correct shape."""
        edge_index = torch.empty((2, 0), dtype=torch.long)

        classic = layout_drl(edge_index=edge_index, num_nodes=0)
        pipeline = layout_drl_pipeline(edge_index=edge_index, num_nodes=0)

        assert classic.shape == (0, 2)
        assert pipeline.shape == (0, 2)
        assert classic.dtype == pipeline.dtype

    def test_layout_drl_pipeline_single_node_no_edges(self) -> None:
        """Single isolated node should match classic exactly."""
        edge_index = torch.empty((2, 0), dtype=torch.long)

        classic = layout_drl(edge_index=edge_index, num_nodes=1, seed=42)
        pipeline = layout_drl_pipeline(edge_index=edge_index, num_nodes=1, seed=42)

        _assert_exact_match(classic, pipeline)

    def test_layout_drl_pipeline_self_loop_filtered(self) -> None:
        """Self-loops should be filtered identically by both paths."""
        edge_index = _edge_index_from_edges([(0, 0), (0, 1), (1, 2)])

        classic = layout_drl(edge_index=edge_index, num_nodes=3, seed=42)
        pipeline = layout_drl_pipeline(edge_index=edge_index, num_nodes=3, seed=42)

        _assert_exact_match(classic, pipeline)

    def test_build_drl_pipeline_rejects_unknown_preset(self) -> None:
        """Unknown preset name should raise ValueError from the resolver."""
        edge_index = _path_edge_index(3)
        with pytest.raises(ValueError, match="unknown DrL preset"):
            layout_drl_pipeline(
                edge_index=edge_index,
                num_nodes=3,
                options="nonexistent_preset",
            )
