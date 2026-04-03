"""Exact-fidelity tests for the composable MaxEnt-Stress pipeline."""

from __future__ import annotations

from typing import Iterable

import pytest
import torch

from dagua.layout.classic.maxent_stress import layout_maxent_stress
from dagua.layout.ops.pipelines.maxent_stress import (
    build_maxent_stress_pipeline,
    layout_maxent_stress_pipeline,
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
    """Assert that two maxent-stress outputs match exactly.

    Parameters
    ----------
    classic : torch.Tensor
        Reference output from classic maxent-stress.
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
        f"Output mismatch. Max abs diff: {(classic - pipeline).abs().max().item():.2e}"
    )


def _run_pipeline_direct(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    steps: int,
    seed: int,
    alpha: float = 1.0,
    use_entropy: bool = False,
    use_majorization: bool = True,
    edge_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute ``build_maxent_stress_pipeline`` directly on a fresh state.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    steps : int
        Number of optimization iterations.
    seed : int
        Random seed.
    alpha : float, default=1.0
        Entropy repulsion weight.
    use_entropy : bool, default=False
        Include the entropy repulsion term.
    use_majorization : bool, default=True
        Prefer majorization for eligible problems.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Final position tensor produced by the pipeline.
    """
    pipeline = build_maxent_stress_pipeline(
        steps=steps,
        alpha=alpha,
        use_entropy=use_entropy,
        use_majorization=use_majorization,
        num_nodes=num_nodes,
    )
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weights=edge_weights,
        seed=seed,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = pipeline.apply(problem, state, ctx)
    assert final_state.pos is not None
    return final_state.pos


# ===================================================================
# Majorization branch (default: use_majorization=True, steps=200)
# ===================================================================


class TestMaxentStressMajorizationFidelity:
    """Bit-exact tests for the majorization branch."""

    @pytest.mark.parametrize(
        ("num_nodes", "seed"),
        [(0, 42), (1, 42), (2, 42), (5, 42), (5, 99), (10, 42), (20, 7)],
    )
    def test_majorization_matches_classic_path_graphs(
        self,
        num_nodes: int,
        seed: int,
    ) -> None:
        """Majorization pipeline matches classic for path graphs."""
        edge_index = _path_edge_index(num_nodes)

        classic = layout_maxent_stress(
            edge_index=edge_index,
            num_nodes=num_nodes,
            steps=200,
            seed=seed,
            use_entropy=False,
            use_majorization=True,
        )
        pipeline = layout_maxent_stress_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            steps=200,
            seed=seed,
            use_entropy=False,
            use_majorization=True,
        )

        _assert_exact_match(classic, pipeline)

    def test_majorization_matches_classic_disconnected_graph(self) -> None:
        """Majorization pipeline matches classic on disconnected components."""
        edge_index = _disconnected_edge_index()

        classic = layout_maxent_stress(
            edge_index=edge_index,
            num_nodes=7,
            steps=200,
            seed=99,
            use_entropy=False,
            use_majorization=True,
        )
        pipeline = layout_maxent_stress_pipeline(
            edge_index=edge_index,
            num_nodes=7,
            steps=200,
            seed=99,
            use_entropy=False,
            use_majorization=True,
        )

        _assert_exact_match(classic, pipeline)

    def test_majorization_matches_classic_complete_graph(self) -> None:
        """Majorization pipeline matches classic on a complete graph."""
        edge_index = _complete_edge_index(6)

        classic = layout_maxent_stress(
            edge_index=edge_index,
            num_nodes=6,
            steps=200,
            seed=7,
            use_entropy=False,
            use_majorization=True,
        )
        pipeline = layout_maxent_stress_pipeline(
            edge_index=edge_index,
            num_nodes=6,
            steps=200,
            seed=7,
            use_entropy=False,
            use_majorization=True,
        )

        _assert_exact_match(classic, pipeline)

    def test_majorization_matches_classic_with_edge_weights(self) -> None:
        """Majorization pipeline matches classic with weighted edges."""
        edge_index = _edge_index_from_edges([(0, 1), (0, 2), (1, 3), (2, 4), (4, 5), (3, 5)])
        edge_weights = torch.tensor([0.25, 1.5, 2.0, 0.75, 1.25, 3.0], dtype=torch.float64)

        classic = layout_maxent_stress(
            edge_index=edge_index,
            num_nodes=6,
            steps=200,
            seed=17,
            use_entropy=False,
            use_majorization=True,
            edge_weights=edge_weights,
        )
        pipeline = layout_maxent_stress_pipeline(
            edge_index=edge_index,
            num_nodes=6,
            steps=200,
            seed=17,
            use_entropy=False,
            use_majorization=True,
            edge_weights=edge_weights,
        )

        _assert_exact_match(classic, pipeline)

    def test_majorization_build_direct_matches_classic(self) -> None:
        """The raw majorization pipeline object matches classic."""
        edge_index = _path_edge_index(8)

        classic = layout_maxent_stress(
            edge_index=edge_index,
            num_nodes=8,
            steps=200,
            seed=42,
            use_entropy=False,
            use_majorization=True,
        )
        pipeline = _run_pipeline_direct(
            edge_index=edge_index,
            num_nodes=8,
            steps=200,
            seed=42,
            use_entropy=False,
            use_majorization=True,
        )

        _assert_exact_match(classic, pipeline)


# ===================================================================
# Gradient branch (non-default steps, use_entropy, or large graphs)
# ===================================================================


class TestMaxentStressGradientFidelity:
    """Bit-exact tests for the gradient branch."""

    @pytest.mark.parametrize(
        ("num_nodes", "seed", "steps"),
        [(2, 42, 50), (5, 42, 100), (5, 99, 30), (10, 42, 80)],
    )
    def test_gradient_matches_classic_non_default_steps(
        self,
        num_nodes: int,
        seed: int,
        steps: int,
    ) -> None:
        """Gradient pipeline matches classic when steps != 200."""
        edge_index = _path_edge_index(num_nodes)

        classic = layout_maxent_stress(
            edge_index=edge_index,
            num_nodes=num_nodes,
            steps=steps,
            seed=seed,
            use_entropy=False,
            use_majorization=True,
        )
        pipeline = layout_maxent_stress_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            steps=steps,
            seed=seed,
            use_entropy=False,
            use_majorization=True,
        )

        _assert_exact_match(classic, pipeline)

    @pytest.mark.parametrize(
        ("num_nodes", "seed"),
        [(3, 42), (5, 42), (8, 99)],
    )
    def test_gradient_matches_classic_with_entropy(
        self,
        num_nodes: int,
        seed: int,
    ) -> None:
        """Gradient pipeline matches classic when use_entropy=True."""
        edge_index = _path_edge_index(num_nodes)

        classic = layout_maxent_stress(
            edge_index=edge_index,
            num_nodes=num_nodes,
            steps=50,
            seed=seed,
            alpha=1.0,
            use_entropy=True,
            use_majorization=True,
        )
        pipeline = layout_maxent_stress_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            steps=50,
            seed=seed,
            alpha=1.0,
            use_entropy=True,
            use_majorization=True,
        )

        _assert_exact_match(classic, pipeline)

    def test_gradient_matches_classic_with_majorization_disabled(self) -> None:
        """Gradient pipeline matches classic when use_majorization=False."""
        edge_index = _path_edge_index(6)

        classic = layout_maxent_stress(
            edge_index=edge_index,
            num_nodes=6,
            steps=200,
            seed=42,
            use_entropy=False,
            use_majorization=False,
        )
        pipeline = layout_maxent_stress_pipeline(
            edge_index=edge_index,
            num_nodes=6,
            steps=200,
            seed=42,
            use_entropy=False,
            use_majorization=False,
        )

        _assert_exact_match(classic, pipeline)

    def test_gradient_matches_classic_with_edge_weights_and_entropy(
        self,
    ) -> None:
        """Gradient pipeline matches classic with weights + entropy."""
        edge_index = _edge_index_from_edges([(0, 1), (1, 2), (2, 3), (3, 4)])
        edge_weights = torch.tensor([1.5, 0.5, 2.0, 0.75], dtype=torch.float64)

        classic = layout_maxent_stress(
            edge_index=edge_index,
            num_nodes=5,
            steps=40,
            seed=7,
            alpha=0.5,
            use_entropy=True,
            edge_weights=edge_weights,
        )
        pipeline = layout_maxent_stress_pipeline(
            edge_index=edge_index,
            num_nodes=5,
            steps=40,
            seed=7,
            alpha=0.5,
            use_entropy=True,
            edge_weights=edge_weights,
        )

        _assert_exact_match(classic, pipeline)

    def test_gradient_matches_classic_disconnected_non_default(self) -> None:
        """Gradient pipeline matches classic on disconnected + non-default steps."""
        edge_index = _disconnected_edge_index()

        classic = layout_maxent_stress(
            edge_index=edge_index,
            num_nodes=7,
            steps=100,
            seed=42,
            use_entropy=False,
            use_majorization=True,
        )
        pipeline = layout_maxent_stress_pipeline(
            edge_index=edge_index,
            num_nodes=7,
            steps=100,
            seed=42,
            use_entropy=False,
            use_majorization=True,
        )

        _assert_exact_match(classic, pipeline)

    def test_gradient_build_direct_matches_classic(self) -> None:
        """The raw gradient pipeline object matches classic."""
        edge_index = _complete_edge_index(5)

        classic = layout_maxent_stress(
            edge_index=edge_index,
            num_nodes=5,
            steps=60,
            seed=7,
            use_entropy=False,
            use_majorization=True,
        )
        pipeline = _run_pipeline_direct(
            edge_index=edge_index,
            num_nodes=5,
            steps=60,
            seed=7,
            use_entropy=False,
            use_majorization=True,
        )

        _assert_exact_match(classic, pipeline)


# ===================================================================
# Edge cases
# ===================================================================


class TestMaxentStressEdgeCases:
    """Edge cases: empty, single-node, zero-step graphs."""

    def test_empty_graph(self) -> None:
        """Empty graph produces empty output."""
        edge_index = torch.empty((2, 0), dtype=torch.long)

        classic = layout_maxent_stress(edge_index=edge_index, num_nodes=0, steps=200)
        pipeline = layout_maxent_stress_pipeline(edge_index=edge_index, num_nodes=0, steps=200)

        _assert_exact_match(classic, pipeline)

    def test_single_node(self) -> None:
        """Single node produces a zero position."""
        edge_index = torch.empty((2, 0), dtype=torch.long)

        classic = layout_maxent_stress(edge_index=edge_index, num_nodes=1, steps=200)
        pipeline = layout_maxent_stress_pipeline(edge_index=edge_index, num_nodes=1, steps=200)

        _assert_exact_match(classic, pipeline)

    def test_two_nodes_majorization(self) -> None:
        """Two connected nodes through majorization."""
        edge_index = _edge_index_from_edges([(0, 1)])

        classic = layout_maxent_stress(
            edge_index=edge_index,
            num_nodes=2,
            steps=200,
            seed=42,
        )
        pipeline = layout_maxent_stress_pipeline(
            edge_index=edge_index,
            num_nodes=2,
            steps=200,
            seed=42,
        )

        _assert_exact_match(classic, pipeline)

    def test_two_nodes_gradient(self) -> None:
        """Two connected nodes through gradient path."""
        edge_index = _edge_index_from_edges([(0, 1)])

        classic = layout_maxent_stress(
            edge_index=edge_index,
            num_nodes=2,
            steps=50,
            seed=42,
        )
        pipeline = layout_maxent_stress_pipeline(
            edge_index=edge_index,
            num_nodes=2,
            steps=50,
            seed=42,
        )

        _assert_exact_match(classic, pipeline)
