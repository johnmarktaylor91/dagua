"""Exact-fidelity tests for the composable Reingold-Tilford pipeline."""

from __future__ import annotations

from typing import Iterable

import pytest
import torch

from dagua.layout.classic.reingold_tilford import layout_reingold_tilford
from dagua.layout.ops.pipelines.reingold_tilford import (
    build_reingold_tilford_pipeline,
    layout_reingold_tilford_pipeline,
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


def _binary_tree_edge_index(depth: int) -> tuple[torch.Tensor, int]:
    """Build a perfect binary tree edge tensor.

    Parameters
    ----------
    depth : int
        Depth of the binary tree (0 = just root).

    Returns
    -------
    tuple[torch.Tensor, int]
        Edge tensor and total node count.
    """
    num_nodes = (1 << (depth + 1)) - 1
    edges = []
    for node in range(num_nodes):
        left = 2 * node + 1
        right = 2 * node + 2
        if left < num_nodes:
            edges.append((node, left))
        if right < num_nodes:
            edges.append((node, right))
    return _edge_index_from_edges(edges), num_nodes


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
    """Assert that two RT outputs match exactly.

    Parameters
    ----------
    classic : torch.Tensor
        Reference output from classic Reingold-Tilford.
    pipeline : torch.Tensor
        Output from the composable pipeline.

    Returns
    -------
    None
        This helper asserts exact equality.
    """
    assert classic.dtype == pipeline.dtype, (
        f"dtype mismatch: classic={classic.dtype}, pipeline={pipeline.dtype}"
    )
    assert classic.device == pipeline.device, (
        f"device mismatch: classic={classic.device}, pipeline={pipeline.device}"
    )
    assert torch.equal(classic, pipeline), (
        f"value mismatch: max abs diff={(classic - pipeline).abs().max().item()}"
    )


def _run_pipeline_direct(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    node_sizes: torch.Tensor | None = None,
    horizontal: bool = False,
    edge_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute ``build_reingold_tilford_pipeline`` directly on a fresh state.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]``.
    horizontal : bool, default=False
        Whether to rotate the layout.
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
        node_sizes=node_sizes,
        edge_weights=edge_weights,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_reingold_tilford_pipeline(horizontal=horizontal).apply(problem, state, ctx)
    assert final_state.pos is not None
    return final_state.pos


class TestReingoldTilfordPipelineFidelity:
    """Bit-exact regression coverage for the Reingold-Tilford pipeline."""

    @pytest.mark.parametrize(
        "num_nodes",
        [0, 1, 2, 5, 10, 20, 50],
    )
    def test_path_graphs(self, num_nodes: int) -> None:
        """Pipeline should match classic RT exactly on path graphs."""
        edge_index = _path_edge_index(num_nodes)

        classic = layout_reingold_tilford(edge_index=edge_index, num_nodes=num_nodes)
        pipeline = layout_reingold_tilford_pipeline(edge_index=edge_index, num_nodes=num_nodes)

        _assert_exact_match(classic, pipeline)

    @pytest.mark.parametrize("depth", [0, 1, 2, 3, 4])
    def test_binary_trees(self, depth: int) -> None:
        """Pipeline should match classic RT exactly on binary trees."""
        edge_index, num_nodes = _binary_tree_edge_index(depth)

        classic = layout_reingold_tilford(edge_index=edge_index, num_nodes=num_nodes)
        pipeline = layout_reingold_tilford_pipeline(edge_index=edge_index, num_nodes=num_nodes)

        _assert_exact_match(classic, pipeline)

    def test_disconnected_graph(self) -> None:
        """Disconnected components and isolated nodes should match exactly."""
        edge_index = _disconnected_edge_index()

        classic = layout_reingold_tilford(edge_index=edge_index, num_nodes=7)
        pipeline = layout_reingold_tilford_pipeline(edge_index=edge_index, num_nodes=7)

        _assert_exact_match(classic, pipeline)

    def test_complete_graph(self) -> None:
        """The raw pipeline object should match classic RT on a dense graph."""
        edge_index = _complete_edge_index(5)

        classic = layout_reingold_tilford(edge_index=edge_index, num_nodes=5)
        pipeline = _run_pipeline_direct(edge_index=edge_index, num_nodes=5)

        _assert_exact_match(classic, pipeline)

    def test_single_isolated_node(self) -> None:
        """A single node with no edges should produce a centered position."""
        edge_index = torch.empty((2, 0), dtype=torch.long)

        classic = layout_reingold_tilford(edge_index=edge_index, num_nodes=1)
        pipeline = layout_reingold_tilford_pipeline(edge_index=edge_index, num_nodes=1)

        _assert_exact_match(classic, pipeline)

    def test_horizontal_mode(self) -> None:
        """Horizontal flag should produce identical rotation to classic."""
        edge_index, num_nodes = _binary_tree_edge_index(3)

        classic = layout_reingold_tilford(
            edge_index=edge_index, num_nodes=num_nodes, horizontal=True
        )
        pipeline = layout_reingold_tilford_pipeline(
            edge_index=edge_index, num_nodes=num_nodes, horizontal=True
        )

        _assert_exact_match(classic, pipeline)

    def test_with_node_sizes(self) -> None:
        """Node sizes should affect spacing identically in pipeline and classic."""
        edge_index, num_nodes = _binary_tree_edge_index(2)
        node_sizes = torch.rand((num_nodes, 2), dtype=torch.float32) * 5.0

        classic = layout_reingold_tilford(
            edge_index=edge_index,
            num_nodes=num_nodes,
            node_sizes=node_sizes,
        )
        pipeline = layout_reingold_tilford_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            node_sizes=node_sizes,
        )

        _assert_exact_match(classic, pipeline)

    def test_with_edge_weights(self) -> None:
        """Edge weights should be accepted without breaking fidelity."""
        edge_index = _edge_index_from_edges([(0, 1), (0, 2), (1, 3), (2, 4), (4, 5), (3, 5)])
        edge_weights = torch.tensor([0.25, 1.5, 2.0, 0.75, 1.25, 3.0], dtype=torch.float64)

        classic = layout_reingold_tilford(
            edge_index=edge_index,
            num_nodes=6,
            edge_weights=edge_weights,
        )
        pipeline = layout_reingold_tilford_pipeline(
            edge_index=edge_index,
            num_nodes=6,
            edge_weights=edge_weights,
        )

        _assert_exact_match(classic, pipeline)

    def test_build_pipeline_direct_on_binary_tree(self) -> None:
        """The raw Pipeline object should match classic RT on a binary tree."""
        edge_index, num_nodes = _binary_tree_edge_index(3)

        classic = layout_reingold_tilford(edge_index=edge_index, num_nodes=num_nodes)
        pipeline = _run_pipeline_direct(edge_index=edge_index, num_nodes=num_nodes)

        _assert_exact_match(classic, pipeline)

    def test_star_graph(self) -> None:
        """Star topology (hub with many leaves) should match exactly."""
        hub = 0
        num_leaves = 10
        num_nodes = 1 + num_leaves
        edge_index = _edge_index_from_edges((hub, leaf) for leaf in range(1, num_nodes))

        classic = layout_reingold_tilford(edge_index=edge_index, num_nodes=num_nodes)
        pipeline = layout_reingold_tilford_pipeline(edge_index=edge_index, num_nodes=num_nodes)

        _assert_exact_match(classic, pipeline)

    def test_forest_multiple_roots(self) -> None:
        """Multiple disconnected trees should match classic exactly."""
        # Two separate trees: 0->1->2 and 3->4->5, plus isolated node 6
        edge_index = _edge_index_from_edges([(0, 1), (1, 2), (3, 4), (4, 5)])

        classic = layout_reingold_tilford(edge_index=edge_index, num_nodes=7)
        pipeline = layout_reingold_tilford_pipeline(edge_index=edge_index, num_nodes=7)

        _assert_exact_match(classic, pipeline)
