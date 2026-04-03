"""Exact-fidelity tests for the composable NeuLay pipeline."""

from __future__ import annotations

from typing import Iterable

import pytest
import torch

from dagua.layout.classic.neulay import layout_neulay
from dagua.layout.ops.pipelines.neulay import build_neulay_pipeline, layout_neulay_pipeline
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
    return _edge_index_from_edges((i, i + 1) for i in range(max(num_nodes - 1, 0)))


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
        (s, t) for s in range(num_nodes) for t in range(num_nodes) if s != t
    )


def _assert_exact_match(classic: torch.Tensor, pipeline: torch.Tensor) -> None:
    """Assert that two NeuLay outputs match exactly.

    Parameters
    ----------
    classic : torch.Tensor
        Reference output from classic NeuLay.
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
    assert classic.shape == pipeline.shape, (
        f"shape mismatch: classic={classic.shape}, pipeline={pipeline.shape}"
    )
    assert torch.equal(classic, pipeline), (
        f"values differ: max abs diff = {(classic - pipeline).abs().max().item()}"
    )


def _run_pipeline_direct(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    steps: int,
    seed: int,
    gcn_steps: int = 2_000,
    use_gcn: bool = True,
    dim: int = 2,
    lr: float = 0.01,
    radius: float = 0.4,
    magnitude: float | None = None,
) -> torch.Tensor:
    """Execute ``build_neulay_pipeline`` directly on a fresh solve state.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    steps : int
        Total optimization budget.
    seed : int
        Random seed.
    gcn_steps : int
        Number of GCN reparameterization steps.
    use_gcn : bool
        Whether to run the GCN phase.
    dim : int
        Embedding dimensionality.
    lr : float
        RMSprop learning rate.
    radius : float
        Gaussian repulsion radius.
    magnitude : float or None
        Repulsion magnitude.

    Returns
    -------
    torch.Tensor
        Final position tensor produced by the pipeline.
    """
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        seed=seed,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    pipeline = build_neulay_pipeline(
        steps=steps,
        gcn_steps=gcn_steps,
        use_gcn=use_gcn,
        dim=dim,
        lr=lr,
        radius=radius,
        magnitude=magnitude,
    )
    final_state = pipeline.apply(problem, state, ctx)
    assert final_state.pos is not None
    return final_state.pos


class TestNeuLayPipelineFidelity:
    """Bit-exact regression coverage for the NeuLay pipeline."""

    def test_empty_graph(self) -> None:
        """Zero-node graph should match classic exactly."""
        edge_index = torch.empty((2, 0), dtype=torch.long)

        classic = layout_neulay(edge_index=edge_index, num_nodes=0, seed=42)
        pipeline = layout_neulay_pipeline(edge_index=edge_index, num_nodes=0, seed=42)

        _assert_exact_match(classic, pipeline)

    def test_single_node(self) -> None:
        """Single-node graph should return zeros like classic."""
        edge_index = torch.empty((2, 0), dtype=torch.long)

        classic = layout_neulay(edge_index=edge_index, num_nodes=1, seed=42)
        pipeline = layout_neulay_pipeline(edge_index=edge_index, num_nodes=1, seed=42)

        _assert_exact_match(classic, pipeline)

    @pytest.mark.parametrize("seed", [42, 99, 7])
    def test_path_graph_no_gcn(self, seed: int) -> None:
        """Direct-only mode on a path graph should match classic exactly."""
        num_nodes = 5
        edge_index = _path_edge_index(num_nodes)
        steps = 100

        classic = layout_neulay(
            edge_index=edge_index,
            num_nodes=num_nodes,
            seed=seed,
            steps=steps,
            gcn_steps=0,
            use_gcn=False,
        )
        pipeline = layout_neulay_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            seed=seed,
            steps=steps,
            gcn_steps=0,
            use_gcn=False,
        )

        _assert_exact_match(classic, pipeline)

    def test_path_graph_with_gcn(self) -> None:
        """Two-phase mode on a path graph should match classic exactly."""
        num_nodes = 8
        edge_index = _path_edge_index(num_nodes)
        steps = 200
        gcn_steps = 50

        classic = layout_neulay(
            edge_index=edge_index,
            num_nodes=num_nodes,
            seed=42,
            steps=steps,
            gcn_steps=gcn_steps,
            use_gcn=True,
        )
        pipeline = layout_neulay_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            seed=42,
            steps=steps,
            gcn_steps=gcn_steps,
            use_gcn=True,
        )

        _assert_exact_match(classic, pipeline)

    def test_disconnected_graph_no_gcn(self) -> None:
        """Disconnected graph should match classic in direct-only mode."""
        edge_index = _disconnected_edge_index()
        num_nodes = 7

        classic = layout_neulay(
            edge_index=edge_index,
            num_nodes=num_nodes,
            seed=99,
            steps=80,
            gcn_steps=0,
            use_gcn=False,
        )
        pipeline = layout_neulay_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            seed=99,
            steps=80,
            gcn_steps=0,
            use_gcn=False,
        )

        _assert_exact_match(classic, pipeline)

    def test_complete_graph_no_gcn(self) -> None:
        """Dense complete graph should match classic in direct-only mode."""
        num_nodes = 5
        edge_index = _complete_edge_index(num_nodes)

        classic = layout_neulay(
            edge_index=edge_index,
            num_nodes=num_nodes,
            seed=7,
            steps=60,
            gcn_steps=0,
            use_gcn=False,
        )
        pipeline = layout_neulay_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            seed=7,
            steps=60,
            gcn_steps=0,
            use_gcn=False,
        )

        _assert_exact_match(classic, pipeline)

    def test_build_pipeline_direct_no_gcn(self) -> None:
        """The raw pipeline object should match classic via direct invocation."""
        num_nodes = 6
        edge_index = _edge_index_from_edges([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)])

        classic = layout_neulay(
            edge_index=edge_index,
            num_nodes=num_nodes,
            seed=42,
            steps=50,
            gcn_steps=0,
            use_gcn=False,
        )
        pipeline = _run_pipeline_direct(
            edge_index=edge_index,
            num_nodes=num_nodes,
            steps=50,
            seed=42,
            gcn_steps=0,
            use_gcn=False,
        )

        _assert_exact_match(classic, pipeline)

    def test_explicit_magnitude(self) -> None:
        """Explicit magnitude parameter should match classic exactly."""
        num_nodes = 5
        edge_index = _path_edge_index(num_nodes)

        classic = layout_neulay(
            edge_index=edge_index,
            num_nodes=num_nodes,
            seed=42,
            steps=50,
            gcn_steps=0,
            use_gcn=False,
            magnitude=10.0,
        )
        pipeline = layout_neulay_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            seed=42,
            steps=50,
            gcn_steps=0,
            use_gcn=False,
            magnitude=10.0,
        )

        _assert_exact_match(classic, pipeline)

    def test_custom_lr_and_radius(self) -> None:
        """Non-default lr and radius should match classic exactly."""
        num_nodes = 5
        edge_index = _path_edge_index(num_nodes)

        classic = layout_neulay(
            edge_index=edge_index,
            num_nodes=num_nodes,
            seed=42,
            steps=50,
            gcn_steps=0,
            use_gcn=False,
            lr=0.005,
            radius=0.8,
        )
        pipeline = layout_neulay_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            seed=42,
            steps=50,
            gcn_steps=0,
            use_gcn=False,
            lr=0.005,
            radius=0.8,
        )

        _assert_exact_match(classic, pipeline)

    def test_gcn_only_no_linear_steps(self) -> None:
        """When gcn_steps >= steps, classic returns GCN output directly."""
        num_nodes = 5
        edge_index = _path_edge_index(num_nodes)

        classic = layout_neulay(
            edge_index=edge_index,
            num_nodes=num_nodes,
            seed=42,
            steps=50,
            gcn_steps=50,
            use_gcn=True,
        )
        pipeline = layout_neulay_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            seed=42,
            steps=50,
            gcn_steps=50,
            use_gcn=True,
        )

        _assert_exact_match(classic, pipeline)

    def test_dim_3(self) -> None:
        """3D output should match classic exactly."""
        num_nodes = 4
        edge_index = _path_edge_index(num_nodes)

        classic = layout_neulay(
            edge_index=edge_index,
            num_nodes=num_nodes,
            seed=42,
            steps=30,
            gcn_steps=0,
            use_gcn=False,
            dim=3,
        )
        pipeline = layout_neulay_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            seed=42,
            steps=30,
            gcn_steps=0,
            use_gcn=False,
            dim=3,
        )

        _assert_exact_match(classic, pipeline)

    def test_two_node_edge(self) -> None:
        """Minimal two-node graph should match classic."""
        edge_index = _edge_index_from_edges([(0, 1)])

        classic = layout_neulay(
            edge_index=edge_index,
            num_nodes=2,
            seed=42,
            steps=30,
            gcn_steps=0,
            use_gcn=False,
        )
        pipeline = layout_neulay_pipeline(
            edge_index=edge_index,
            num_nodes=2,
            seed=42,
            steps=30,
            gcn_steps=0,
            use_gcn=False,
        )

        _assert_exact_match(classic, pipeline)
