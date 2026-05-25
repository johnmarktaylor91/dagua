"""Exact-fidelity tests for the composable GraphOpt pipeline."""

from __future__ import annotations

from typing import Iterable, Optional

import pytest
import torch

from dagua.layout.classic.graphopt import layout_graphopt
from dagua.layout.ops.pipelines.graphopt import build_graphopt_pipeline, layout_graphopt_pipeline
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
    """Build a disconnected graph with two components and isolates.

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
    """Assert that two GraphOpt outputs match exactly.

    Parameters
    ----------
    classic : torch.Tensor
        Reference output from classic GraphOpt.
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
    niter: int,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Execute ``build_graphopt_pipeline`` directly on a fresh solve state.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    seed : int
        Random seed used for initialization.
    niter : int
        Number of GraphOpt iterations.
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
    final_state = build_graphopt_pipeline(niter=niter).apply(problem, state, ctx)
    assert final_state.pos is not None
    return final_state.pos


class TestGraphOptPipelineFidelity:
    """Bit-exact regression coverage for the GraphOpt pipeline."""

    def test_graphopt_fidelity_fallback_uses_igraph_rng(self) -> None:
        """GraphOpt fidelity fallback initialization should use igraph's RNG."""
        edge_index = _path_edge_index(2)

        pos = layout_graphopt_pipeline(
            edge_index=edge_index,
            num_nodes=2,
            niter=0,
            seed=42,
            fidelity_mode=True,
        )

        expected = torch.tensor(
            [
                [-0.725433349609375, 0.6807889938354492],
                [0.9683611392974854, 0.8450748324394226],
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(pos, expected, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize(
        ("num_nodes", "seed"),
        [(0, 42), (1, 42), (2, 42), (5, 42), (5, 99), (20, 42), (50, 7)],
    )
    def test_layout_graphopt_pipeline_matches_classic_for_requested_sizes(
        self,
        num_nodes: int,
        seed: int,
    ) -> None:
        """The adapter should match classic GraphOpt exactly for these cases."""
        edge_index = _path_edge_index(num_nodes)

        classic = layout_graphopt(edge_index=edge_index, num_nodes=num_nodes, niter=80, seed=seed)
        pipeline = layout_graphopt_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            niter=80,
            seed=seed,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_graphopt_pipeline_matches_classic_with_edge_weights(self) -> None:
        """Weighted GraphOpt springs should remain bit-identical."""
        edge_index = _edge_index_from_edges([(0, 1), (0, 2), (1, 3), (2, 4), (4, 5), (3, 5)])
        edge_weights = torch.tensor([0.25, 1.5, 2.0, 0.75, 1.25, 3.0], dtype=torch.float64)

        classic = layout_graphopt(
            edge_index=edge_index,
            num_nodes=6,
            niter=80,
            seed=17,
            edge_weights=edge_weights,
        )
        pipeline = layout_graphopt_pipeline(
            edge_index=edge_index,
            num_nodes=6,
            niter=80,
            seed=17,
            edge_weights=edge_weights,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_graphopt_pipeline_matches_classic_on_disconnected_graph(self) -> None:
        """Disconnected components and isolated nodes should match exactly."""
        edge_index = _disconnected_edge_index()

        classic = layout_graphopt(edge_index=edge_index, num_nodes=7, niter=80, seed=99)
        pipeline = layout_graphopt_pipeline(edge_index=edge_index, num_nodes=7, niter=80, seed=99)

        _assert_exact_match(classic, pipeline)

    def test_build_graphopt_pipeline_matches_classic_on_complete_graph(self) -> None:
        """The raw pipeline object should match classic GraphOpt on a dense graph."""
        edge_index = _complete_edge_index(5)

        classic = layout_graphopt(edge_index=edge_index, num_nodes=5, niter=80, seed=7)
        pipeline = _run_pipeline_direct(edge_index=edge_index, num_nodes=5, niter=80, seed=7)

        _assert_exact_match(classic, pipeline)

    def test_layout_graphopt_pipeline_matches_classic_with_parallel_edges(self) -> None:
        """Parallel and reciprocal edges should preserve exact spring multiplicity."""
        edge_index = _edge_index_from_edges([(0, 1), (1, 0), (0, 1), (1, 2)])

        classic = layout_graphopt(edge_index=edge_index, num_nodes=3, niter=60, seed=5)
        pipeline = layout_graphopt_pipeline(edge_index=edge_index, num_nodes=3, niter=60, seed=5)

        _assert_exact_match(classic, pipeline)

    def test_layout_graphopt_pipeline_accepts_initial_positions(self) -> None:
        """The public pipeline should use a supplied GraphOpt seed matrix."""
        edge_index = _path_edge_index(3)
        initial_pos = torch.tensor(
            [[-1.0, 0.5], [0.0, 0.0], [1.0, -0.5]],
            dtype=torch.float64,
        )

        pos = layout_graphopt_pipeline(
            edge_index=edge_index,
            num_nodes=3,
            niter=0,
            seed=99,
            initial_pos=initial_pos,
            fidelity_mode=True,
        )

        torch.testing.assert_close(pos, initial_pos.to(dtype=torch.float32))
