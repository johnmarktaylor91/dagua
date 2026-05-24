"""Exact-fidelity tests for the composable LGL pipeline."""

from __future__ import annotations

from typing import Iterable

import pytest
import torch

from dagua.layout.classic.lgl import layout_lgl
from dagua.layout.ops.pipelines.lgl import build_lgl_pipeline, layout_lgl_pipeline
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
    """Assert that two LGL outputs match exactly.

    Parameters
    ----------
    classic : torch.Tensor
        Reference output from classic LGL.
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
    maxiter: int = 150,
    root: int | None = None,
    edge_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute ``build_lgl_pipeline`` directly on a fresh solve state.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    seed : int
        Random seed used for initialization.
    maxiter : int, default=150
        Maximum refinement iterations per BFS layer.
    root : int, optional
        BFS root vertex.
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
    state.extras["lgl_root_was_random"] = root is None
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_lgl_pipeline(maxiter=maxiter, root=root).apply(problem, state, ctx)
    assert final_state.pos is not None
    return final_state.pos


class TestLGLPipelineFidelity:
    """Bit-exact regression coverage for the LGL pipeline."""

    @pytest.mark.parametrize(
        ("num_nodes", "seed"),
        [(0, 42), (1, 42), (2, 42), (5, 42), (5, 99), (10, 42), (20, 7)],
    )
    def test_layout_lgl_pipeline_matches_classic_for_requested_sizes(
        self,
        num_nodes: int,
        seed: int,
    ) -> None:
        """The adapter should match classic LGL exactly for path graphs."""
        edge_index = _path_edge_index(num_nodes)

        classic = layout_lgl(
            edge_index=edge_index,
            num_nodes=num_nodes,
            seed=seed,
        )
        pipeline = layout_lgl_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            seed=seed,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_lgl_pipeline_matches_classic_with_edge_weights(self) -> None:
        """Weighted LGL should remain bit-identical in the pipeline."""
        edge_index = _edge_index_from_edges([(0, 1), (0, 2), (1, 3), (2, 4), (4, 5), (3, 5)])
        edge_weights = torch.tensor([0.25, 1.5, 2.0, 0.75, 1.25, 3.0], dtype=torch.float64)

        classic = layout_lgl(
            edge_index=edge_index,
            num_nodes=6,
            seed=17,
            edge_weights=edge_weights,
        )
        pipeline = layout_lgl_pipeline(
            edge_index=edge_index,
            num_nodes=6,
            seed=17,
            edge_weights=edge_weights,
            use_edge_weights=True,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_lgl_pipeline_matches_classic_on_disconnected_graph(self) -> None:
        """Disconnected components and isolated nodes should match exactly."""
        edge_index = _disconnected_edge_index()

        classic = layout_lgl(
            edge_index=edge_index,
            num_nodes=7,
            seed=99,
        )
        pipeline = layout_lgl_pipeline(
            edge_index=edge_index,
            num_nodes=7,
            seed=99,
        )

        _assert_exact_match(classic, pipeline)

    def test_build_lgl_pipeline_matches_classic_on_complete_graph(self) -> None:
        """The raw pipeline object should match classic LGL on a dense graph."""
        edge_index = _complete_edge_index(5)

        classic = layout_lgl(
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

    def test_layout_lgl_pipeline_matches_classic_on_star_graph(self) -> None:
        """Star topology exercises the BFS shell depth-1 angular placement."""
        edge_index = _star_edge_index(8)

        classic = layout_lgl(
            edge_index=edge_index,
            num_nodes=8,
            seed=42,
        )
        pipeline = layout_lgl_pipeline(
            edge_index=edge_index,
            num_nodes=8,
            seed=42,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_lgl_pipeline_matches_classic_with_explicit_root(self) -> None:
        """Explicit root should match classic LGL exactly."""
        edge_index = _path_edge_index(10)

        classic = layout_lgl(
            edge_index=edge_index,
            num_nodes=10,
            seed=42,
            root=3,
        )
        pipeline = layout_lgl_pipeline(
            edge_index=edge_index,
            num_nodes=10,
            seed=42,
            root=3,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_lgl_pipeline_matches_classic_with_custom_params(self) -> None:
        """Non-default coolexp, maxiter, and area should match exactly."""
        edge_index = _path_edge_index(8)

        classic = layout_lgl(
            edge_index=edge_index,
            num_nodes=8,
            seed=42,
            maxiter=50,
            coolexp=2.0,
            area=200.0,
        )
        pipeline = layout_lgl_pipeline(
            edge_index=edge_index,
            num_nodes=8,
            seed=42,
            maxiter=50,
            coolexp=2.0,
            area=200.0,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_lgl_pipeline_empty_graph(self) -> None:
        """Zero-node graph should return empty tensor with correct shape."""
        edge_index = torch.empty((2, 0), dtype=torch.long)

        classic = layout_lgl(edge_index=edge_index, num_nodes=0)
        pipeline = layout_lgl_pipeline(edge_index=edge_index, num_nodes=0)

        assert classic.shape == (0, 2)
        assert pipeline.shape == (0, 2)
        assert classic.dtype == pipeline.dtype

    def test_layout_lgl_pipeline_single_node_no_edges(self) -> None:
        """Single isolated node should match classic exactly."""
        edge_index = torch.empty((2, 0), dtype=torch.long)

        classic = layout_lgl(edge_index=edge_index, num_nodes=1, seed=42)
        pipeline = layout_lgl_pipeline(edge_index=edge_index, num_nodes=1, seed=42)

        _assert_exact_match(classic, pipeline)

    def test_build_lgl_pipeline_rejects_negative_maxiter(self) -> None:
        """Negative maxiter should raise ValueError."""
        with pytest.raises(ValueError, match="maxiter"):
            build_lgl_pipeline(maxiter=-1)

    def test_build_lgl_pipeline_rejects_non_positive_coolexp(self) -> None:
        """Non-positive coolexp should raise ValueError."""
        with pytest.raises(ValueError, match="coolexp"):
            build_lgl_pipeline(coolexp=0.0)
