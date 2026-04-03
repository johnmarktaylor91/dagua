"""Exact-fidelity tests for the composable spectral pipeline."""

from __future__ import annotations

from typing import Iterable, Optional

import pytest
import torch

from dagua.layout.classic.spectral import layout_spectral
from dagua.layout.ops.pipelines.spectral import build_spectral_pipeline, layout_spectral_pipeline
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
    """Assert that two spectral outputs match exactly.

    Parameters
    ----------
    classic : torch.Tensor
        Reference output from classic spectral layout.
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
    normalization: str,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Execute ``build_spectral_pipeline`` directly on a fresh solve state.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    normalization : str
        Laplacian normalization mode.
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
    final_state = build_spectral_pipeline(normalization=normalization).apply(problem, state, ctx)
    assert final_state.pos is not None
    return final_state.pos


class TestSpectralPipelineFidelity:
    """Bit-exact regression coverage for the spectral pipeline."""

    @pytest.mark.parametrize(
        ("num_nodes", "seed", "normalization"),
        [
            (0, 42, "symmetric"),
            (1, 42, "symmetric"),
            (2, 42, "symmetric"),
            (5, 42, "symmetric"),
            (5, 99, "random_walk"),
            (20, 42, "unnormalized"),
        ],
    )
    def test_layout_spectral_pipeline_matches_classic_for_requested_sizes(
        self,
        num_nodes: int,
        seed: int,
        normalization: str,
    ) -> None:
        """The adapter should match classic spectral layout exactly."""
        edge_index = _path_edge_index(num_nodes)

        classic = layout_spectral(
            edge_index=edge_index,
            num_nodes=num_nodes,
            seed=seed,
            normalization=normalization,
        )
        pipeline = layout_spectral_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            seed=seed,
            normalization=normalization,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_spectral_pipeline_matches_classic_with_edge_weights(self) -> None:
        """Weighted adjacency should remain bit-identical."""
        edge_index = _edge_index_from_edges([(0, 1), (0, 2), (1, 3), (2, 4), (4, 5), (3, 5)])
        edge_weights = torch.tensor([0.25, 1.5, 2.0, 0.75, 1.25, 3.0], dtype=torch.float64)

        classic = layout_spectral(
            edge_index=edge_index,
            num_nodes=6,
            seed=17,
            edge_weights=edge_weights,
            normalization="symmetric",
        )
        pipeline = layout_spectral_pipeline(
            edge_index=edge_index,
            num_nodes=6,
            seed=17,
            edge_weights=edge_weights,
            normalization="symmetric",
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_spectral_pipeline_matches_classic_on_disconnected_graph(self) -> None:
        """Disconnected components and isolates should match exactly."""
        edge_index = _disconnected_edge_index()

        classic = layout_spectral(
            edge_index=edge_index,
            num_nodes=9,
            normalization="random_walk",
        )
        pipeline = layout_spectral_pipeline(
            edge_index=edge_index,
            num_nodes=9,
            normalization="random_walk",
        )

        _assert_exact_match(classic, pipeline)

    def test_build_spectral_pipeline_matches_classic_on_complete_graph(self) -> None:
        """The raw pipeline object should match classic spectral layout on a dense graph."""
        edge_index = _complete_edge_index(5)

        classic = layout_spectral(
            edge_index=edge_index,
            num_nodes=5,
            normalization="unnormalized",
        )
        pipeline = _run_pipeline_direct(
            edge_index=edge_index,
            num_nodes=5,
            normalization="unnormalized",
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_spectral_pipeline_matches_classic_with_node_sizes(self) -> None:
        """Node-size passthrough should preserve the classic output exactly."""
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

        classic = layout_spectral(
            edge_index=edge_index,
            num_nodes=6,
            node_sizes=node_sizes,
            normalization="symmetric",
        )
        pipeline = layout_spectral_pipeline(
            edge_index=edge_index,
            num_nodes=6,
            node_sizes=node_sizes,
            normalization="symmetric",
        )

        _assert_exact_match(classic, pipeline)
