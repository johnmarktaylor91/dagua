"""Exact-fidelity tests for the composable Sugiyama pipeline."""

from __future__ import annotations

from typing import Iterable

import pytest
import torch

from dagua.layout.classic.sugiyama import layout_sugiyama
from dagua.layout.ops.pipelines.sugiyama import (
    build_sugiyama_pipeline,
    layout_sugiyama_pipeline,
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


def _diamond_dag_edge_index() -> torch.Tensor:
    """Build a diamond-shaped DAG: 0->1, 0->2, 1->3, 2->3.

    Returns
    -------
    torch.Tensor
        Diamond DAG edge tensor.
    """
    return _edge_index_from_edges([(0, 1), (0, 2), (1, 3), (2, 3)])


def _wide_dag_edge_index() -> torch.Tensor:
    """Build a wider DAG with multiple layers and fan-out.

    Returns
    -------
    torch.Tensor
        Wide DAG edge tensor with 8 nodes.
    """
    return _edge_index_from_edges(
        [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 4),
            (2, 4),
            (2, 5),
            (3, 5),
            (3, 6),
            (4, 7),
            (5, 7),
            (6, 7),
        ]
    )


def _disconnected_dag_edge_index() -> torch.Tensor:
    """Build a small disconnected DAG with two components and isolates.

    Returns
    -------
    torch.Tensor
        Disconnected DAG edge tensor with 7 nodes total.
    """
    return _edge_index_from_edges([(0, 1), (1, 2), (3, 4)])


def _long_edge_dag() -> torch.Tensor:
    """Build a DAG with edges that skip layers (requiring dummy nodes).

    Returns
    -------
    torch.Tensor
        DAG with long edges spanning multiple layers.
    """
    return _edge_index_from_edges(
        [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 3),
            (0, 3),
        ]
    )


def _cyclic_edge_index() -> torch.Tensor:
    """Build a graph with a cycle to exercise cycle removal.

    Returns
    -------
    torch.Tensor
        Edge tensor with a cycle: 0->1->2->0, plus 2->3.
    """
    return _edge_index_from_edges([(0, 1), (1, 2), (2, 0), (2, 3)])


def _assert_exact_match(classic: torch.Tensor, pipeline: torch.Tensor) -> None:
    """Assert that two Sugiyama outputs match exactly.

    Parameters
    ----------
    classic : torch.Tensor
        Reference output from classic Sugiyama.
    pipeline : torch.Tensor
        Output from the composable pipeline.
    """
    assert classic.dtype == pipeline.dtype, (
        f"dtype mismatch: classic={classic.dtype}, pipeline={pipeline.dtype}"
    )
    assert classic.device == pipeline.device, (
        f"device mismatch: classic={classic.device}, pipeline={pipeline.device}"
    )
    assert torch.equal(classic, pipeline), (
        f"Positions differ.\nClassic:\n{classic}\nPipeline:\n{pipeline}"
    )


def _run_pipeline_direct(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    node_sizes: torch.Tensor | None = None,
    rank_sep: float = 1.0,
    node_sep: float = 1.0,
    seed: int = 42,
    barycenter_passes: int = 24,
    edge_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute ``build_sugiyama_pipeline`` directly on a fresh solve state.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Node sizes for spacing.
    rank_sep : float, default=1.0
        Vertical layer spacing.
    node_sep : float, default=1.0
        Horizontal node spacing.
    seed : int, default=42
        Random seed for API compatibility.
    barycenter_passes : int, default=24
        Number of crossing minimization sweeps.
    edge_weights : torch.Tensor, optional
        Optional edge weights.

    Returns
    -------
    torch.Tensor
        Final position tensor produced by the pipeline.
    """
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    pipeline = build_sugiyama_pipeline(
        rank_sep=rank_sep,
        node_sep=node_sep,
        barycenter_passes=barycenter_passes,
        seed=seed,
    )
    final_state = pipeline.apply(problem, state, ctx)
    assert final_state.pos is not None
    return final_state.pos


class TestSugiyamaPipelineFidelity:
    """Bit-exact regression coverage for the Sugiyama pipeline."""

    @pytest.mark.parametrize(
        ("num_nodes", "seed"),
        [(0, 42), (1, 42), (2, 42), (5, 42), (5, 99), (10, 42), (20, 7)],
    )
    def test_path_graphs(self, num_nodes: int, seed: int) -> None:
        """Path graphs of various sizes should match classic exactly."""
        edge_index = _path_edge_index(num_nodes)

        classic = layout_sugiyama(edge_index=edge_index, num_nodes=num_nodes, seed=seed)
        pipeline = layout_sugiyama_pipeline(edge_index=edge_index, num_nodes=num_nodes, seed=seed)

        _assert_exact_match(classic, pipeline)

    def test_diamond_dag(self) -> None:
        """Diamond DAG should match classic exactly."""
        edge_index = _diamond_dag_edge_index()

        classic = layout_sugiyama(edge_index=edge_index, num_nodes=4)
        pipeline = layout_sugiyama_pipeline(edge_index=edge_index, num_nodes=4)

        _assert_exact_match(classic, pipeline)

    def test_wide_dag(self) -> None:
        """Wide DAG with fan-out should match classic exactly."""
        edge_index = _wide_dag_edge_index()

        classic = layout_sugiyama(edge_index=edge_index, num_nodes=8)
        pipeline = layout_sugiyama_pipeline(edge_index=edge_index, num_nodes=8)

        _assert_exact_match(classic, pipeline)

    def test_disconnected_dag(self) -> None:
        """Disconnected components and isolated nodes should match exactly."""
        edge_index = _disconnected_dag_edge_index()

        classic = layout_sugiyama(edge_index=edge_index, num_nodes=7, seed=99)
        pipeline = layout_sugiyama_pipeline(edge_index=edge_index, num_nodes=7, seed=99)

        _assert_exact_match(classic, pipeline)

    def test_long_edges_with_dummy_nodes(self) -> None:
        """Edges spanning multiple layers (requiring dummies) should match."""
        edge_index = _long_edge_dag()

        classic = layout_sugiyama(edge_index=edge_index, num_nodes=4)
        pipeline = layout_sugiyama_pipeline(edge_index=edge_index, num_nodes=4)

        _assert_exact_match(classic, pipeline)

    def test_cyclic_graph(self) -> None:
        """Graphs with cycles should match after cycle removal."""
        edge_index = _cyclic_edge_index()

        classic = layout_sugiyama(edge_index=edge_index, num_nodes=4)
        pipeline = layout_sugiyama_pipeline(edge_index=edge_index, num_nodes=4)

        _assert_exact_match(classic, pipeline)

    def test_custom_spacing(self) -> None:
        """Custom rank_sep and node_sep should match classic."""
        edge_index = _diamond_dag_edge_index()

        classic = layout_sugiyama(edge_index=edge_index, num_nodes=4, rank_sep=2.5, node_sep=1.5)
        pipeline = layout_sugiyama_pipeline(
            edge_index=edge_index, num_nodes=4, rank_sep=2.5, node_sep=1.5
        )

        _assert_exact_match(classic, pipeline)

    def test_layer_sep_alias(self) -> None:
        """The layer_sep alias should override rank_sep identically."""
        edge_index = _diamond_dag_edge_index()

        classic = layout_sugiyama(edge_index=edge_index, num_nodes=4, layer_sep=3.0)
        pipeline = layout_sugiyama_pipeline(edge_index=edge_index, num_nodes=4, layer_sep=3.0)

        _assert_exact_match(classic, pipeline)

    def test_with_node_sizes(self) -> None:
        """Node sizes should affect spacing identically."""
        edge_index = _wide_dag_edge_index()
        node_sizes = torch.tensor([[20, 10]] * 8, dtype=torch.float32)

        classic = layout_sugiyama(edge_index=edge_index, num_nodes=8, node_sizes=node_sizes)
        pipeline = layout_sugiyama_pipeline(
            edge_index=edge_index, num_nodes=8, node_sizes=node_sizes
        )

        _assert_exact_match(classic, pipeline)

    def test_with_edge_weights(self) -> None:
        """Edge weights should affect barycenter ordering identically."""
        edge_index = _wide_dag_edge_index()
        edge_weights = torch.tensor(
            [0.5, 1.0, 2.0, 1.5, 0.3, 1.0, 0.8, 1.2, 2.5, 0.7, 1.0],
            dtype=torch.float32,
        )

        classic = layout_sugiyama(
            edge_index=edge_index,
            num_nodes=8,
            edge_weights=edge_weights,
        )
        pipeline = layout_sugiyama_pipeline(
            edge_index=edge_index,
            num_nodes=8,
            edge_weights=edge_weights,
        )

        _assert_exact_match(classic, pipeline)

    def test_build_pipeline_direct_on_diamond(self) -> None:
        """The raw pipeline object should match classic on a diamond DAG."""
        edge_index = _diamond_dag_edge_index()

        classic = layout_sugiyama(edge_index=edge_index, num_nodes=4, seed=7)
        pipeline = _run_pipeline_direct(edge_index=edge_index, num_nodes=4, seed=7)

        _assert_exact_match(classic, pipeline)

    def test_empty_graph(self) -> None:
        """Empty graph (0 nodes) should produce an empty tensor."""
        edge_index = torch.empty((2, 0), dtype=torch.long)

        classic = layout_sugiyama(edge_index=edge_index, num_nodes=0)
        pipeline = layout_sugiyama_pipeline(edge_index=edge_index, num_nodes=0)

        _assert_exact_match(classic, pipeline)

    def test_single_node_no_edges(self) -> None:
        """Single node with no edges should match classic."""
        edge_index = torch.empty((2, 0), dtype=torch.long)

        classic = layout_sugiyama(edge_index=edge_index, num_nodes=1)
        pipeline = layout_sugiyama_pipeline(edge_index=edge_index, num_nodes=1)

        _assert_exact_match(classic, pipeline)

    def test_many_barycenter_passes(self) -> None:
        """Different barycenter pass counts should match classic exactly."""
        edge_index = _wide_dag_edge_index()

        for passes in [0, 1, 4, 12, 24, 48]:
            classic = layout_sugiyama(
                edge_index=edge_index,
                num_nodes=8,
                barycenter_passes=passes,
            )
            pipeline = layout_sugiyama_pipeline(
                edge_index=edge_index,
                num_nodes=8,
                barycenter_passes=passes,
            )
            _assert_exact_match(classic, pipeline)


class TestSugiyamaPipelineTraces:
    """Tests for trace snapshot fidelity."""

    def test_trace_snapshots_match(self) -> None:
        """Trace snapshots should match classic exactly."""
        edge_index = _wide_dag_edge_index()

        classic_result = layout_sugiyama(
            edge_index=edge_index,
            num_nodes=8,
            trace_every=4,
        )
        pipeline_result = layout_sugiyama_pipeline(
            edge_index=edge_index,
            num_nodes=8,
            trace_every=4,
        )

        assert isinstance(classic_result, tuple)
        assert isinstance(pipeline_result, tuple)

        classic_pos, classic_traces = classic_result
        pipeline_pos, pipeline_traces = pipeline_result

        _assert_exact_match(classic_pos, pipeline_pos)
        assert len(classic_traces) == len(pipeline_traces)
        for idx, (ct, pt) in enumerate(zip(classic_traces, pipeline_traces)):
            assert torch.equal(ct, pt), f"Trace {idx} differs"


class TestSugiyamaPipelineEdgeRoutes:
    """Tests for edge route fidelity."""

    def test_edge_routes_match(self) -> None:
        """Edge routes should match classic exactly."""
        edge_index = _long_edge_dag()

        classic_result = layout_sugiyama(
            edge_index=edge_index,
            num_nodes=4,
            return_edge_routes=True,
        )
        pipeline_result = layout_sugiyama_pipeline(
            edge_index=edge_index,
            num_nodes=4,
            return_edge_routes=True,
        )

        assert isinstance(classic_result, tuple)
        assert isinstance(pipeline_result, tuple)

        classic_pos, classic_routes = classic_result
        pipeline_pos, pipeline_routes = pipeline_result

        _assert_exact_match(classic_pos, pipeline_pos)
        assert len(classic_routes) == len(pipeline_routes)
        for idx, (cr, pr) in enumerate(zip(classic_routes, pipeline_routes)):
            assert torch.equal(cr, pr), f"Route {idx} differs"

    def test_edge_routes_with_traces(self) -> None:
        """Edge routes + traces should match classic exactly."""
        edge_index = _long_edge_dag()

        classic_result = layout_sugiyama(
            edge_index=edge_index,
            num_nodes=4,
            return_edge_routes=True,
            trace_every=6,
        )
        pipeline_result = layout_sugiyama_pipeline(
            edge_index=edge_index,
            num_nodes=4,
            return_edge_routes=True,
            trace_every=6,
        )

        assert isinstance(classic_result, tuple)
        assert isinstance(pipeline_result, tuple)

        classic_pos, classic_traces, classic_routes = classic_result
        pipeline_pos, pipeline_traces, pipeline_routes = pipeline_result

        _assert_exact_match(classic_pos, pipeline_pos)
        assert len(classic_traces) == len(pipeline_traces)
        for idx, (ct, pt) in enumerate(zip(classic_traces, pipeline_traces)):
            assert torch.equal(ct, pt), f"Trace {idx} differs"
        assert len(classic_routes) == len(pipeline_routes)
        for idx, (cr, pr) in enumerate(zip(classic_routes, pipeline_routes)):
            assert torch.equal(cr, pr), f"Route {idx} differs"

    def test_edge_routes_on_cyclic_graph(self) -> None:
        """Edge routes on cyclic graphs (after cycle removal) should match."""
        edge_index = _cyclic_edge_index()

        classic_result = layout_sugiyama(
            edge_index=edge_index,
            num_nodes=4,
            return_edge_routes=True,
        )
        pipeline_result = layout_sugiyama_pipeline(
            edge_index=edge_index,
            num_nodes=4,
            return_edge_routes=True,
        )

        assert isinstance(classic_result, tuple)
        assert isinstance(pipeline_result, tuple)

        classic_pos, classic_routes = classic_result
        pipeline_pos, pipeline_routes = pipeline_result

        _assert_exact_match(classic_pos, pipeline_pos)
        assert len(classic_routes) == len(pipeline_routes)
        for idx, (cr, pr) in enumerate(zip(classic_routes, pipeline_routes)):
            assert torch.equal(cr, pr), f"Route {idx} differs"
