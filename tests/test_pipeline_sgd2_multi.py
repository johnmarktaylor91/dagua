"""Exact-fidelity tests for the composable (SGD)^2 multicriteria pipeline."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import pytest
import torch

from dagua.layout.classic.sgd2_multi import SmoothSteps, layout_sgd2_multi
from dagua.layout.ops.pipelines.sgd2_multi import (
    build_sgd2_multi_pipeline,
    layout_sgd2_multi_pipeline,
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
    """Assert that two (SGD)^2 outputs match exactly.

    Parameters
    ----------
    classic : torch.Tensor
        Reference output from classic (SGD)^2.
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
        f"Tensor values differ. Max abs diff: {(classic - pipeline).abs().max().item():.2e}"
    )


def _run_pipeline_direct(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    steps: int,
    seed: int,
    criteria: Optional[Dict[str, float]] = None,
    criteria_schedules: Optional[Dict[str, SmoothSteps]] = None,
    lr: float = 1.0,
    momentum: float = 0.7,
    grad_clamp: float = 4.0,
    batch_size: int = 16,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Execute ``build_sgd2_multi_pipeline`` directly on a fresh solve state.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    steps : int
        Maximum SGD iterations.
    seed : int
        Random seed.
    criteria : dict[str, float] | None
        Static criterion weights.
    criteria_schedules : dict[str, SmoothSteps] | None
        Per-criterion schedules.
    lr : float
        Learning rate.
    momentum : float
        SGD momentum.
    grad_clamp : float
        Symmetric gradient clamp.
    batch_size : int
        Mini-batch size.
    edge_weights : torch.Tensor | None
        Optional edge weights.

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
    final_state = build_sgd2_multi_pipeline(
        steps=steps,
        criteria=criteria,
        criteria_schedules=criteria_schedules,
        lr=lr,
        momentum=momentum,
        grad_clamp=grad_clamp,
        batch_size=batch_size,
    ).apply(problem, state, ctx)
    assert final_state.pos is not None
    return final_state.pos


class TestSGD2MultiPipelineFidelity:
    """Bit-exact regression coverage for the (SGD)^2 multicriteria pipeline."""

    @pytest.mark.parametrize(
        ("num_nodes", "seed"),
        [(0, 42), (1, 42), (2, 42), (5, 42), (5, 99), (10, 42)],
    )
    def test_pipeline_matches_classic_stress_default(
        self,
        num_nodes: int,
        seed: int,
    ) -> None:
        """Default stress criterion should match classic exactly."""
        edge_index = _path_edge_index(num_nodes)
        # Use fewer steps for test speed -- bit-identity holds at any step count
        steps = 50

        classic = layout_sgd2_multi(
            edge_index=edge_index, num_nodes=num_nodes, steps=steps, seed=seed
        )
        pipeline = layout_sgd2_multi_pipeline(
            edge_index=edge_index, num_nodes=num_nodes, steps=steps, seed=seed
        )

        _assert_exact_match(classic, pipeline)

    def test_pipeline_matches_classic_with_ideal_edge_length(self) -> None:
        """Ideal edge length criterion should match classic exactly."""
        edge_index = _edge_index_from_edges([(0, 1), (0, 2), (1, 3), (2, 4), (4, 5), (3, 5)])
        criteria = {"ideal_edge_length": 1.0}

        classic = layout_sgd2_multi(
            edge_index=edge_index,
            num_nodes=6,
            steps=50,
            seed=17,
            criteria=criteria,
        )
        pipeline = layout_sgd2_multi_pipeline(
            edge_index=edge_index,
            num_nodes=6,
            steps=50,
            seed=17,
            criteria=criteria,
        )

        _assert_exact_match(classic, pipeline)

    def test_pipeline_matches_classic_multicriteria(self) -> None:
        """Multiple criteria combined should match classic exactly."""
        edge_index = _path_edge_index(8)
        criteria = {"stress": 1.0, "ideal_edge_length": 0.5}

        classic = layout_sgd2_multi(
            edge_index=edge_index,
            num_nodes=8,
            steps=50,
            seed=42,
            criteria=criteria,
        )
        pipeline = layout_sgd2_multi_pipeline(
            edge_index=edge_index,
            num_nodes=8,
            steps=50,
            seed=42,
            criteria=criteria,
        )

        _assert_exact_match(classic, pipeline)

    def test_pipeline_matches_classic_with_schedules(self) -> None:
        """Smooth-step schedules should match classic exactly."""
        edge_index = _path_edge_index(6)
        criteria_schedules = {
            "stress": SmoothSteps(times=[0, 25, 50], values=[1.0, 0.5, 0.1]),
        }

        classic = layout_sgd2_multi(
            edge_index=edge_index,
            num_nodes=6,
            steps=50,
            seed=42,
            criteria_schedules=criteria_schedules,
        )
        pipeline = layout_sgd2_multi_pipeline(
            edge_index=edge_index,
            num_nodes=6,
            steps=50,
            seed=42,
            criteria_schedules=criteria_schedules,
        )

        _assert_exact_match(classic, pipeline)

    def test_pipeline_matches_classic_aspect_ratio(self) -> None:
        """Aspect ratio criterion should match classic exactly."""
        edge_index = _path_edge_index(8)
        criteria = {"aspect_ratio": 1.0}

        classic = layout_sgd2_multi(
            edge_index=edge_index,
            num_nodes=8,
            steps=50,
            seed=42,
            criteria=criteria,
        )
        pipeline = layout_sgd2_multi_pipeline(
            edge_index=edge_index,
            num_nodes=8,
            steps=50,
            seed=42,
            criteria=criteria,
        )

        _assert_exact_match(classic, pipeline)

    def test_pipeline_matches_classic_neighborhood_preservation(self) -> None:
        """Neighborhood preservation criterion should match classic exactly."""
        edge_index = _path_edge_index(8)
        criteria = {"neighborhood_preservation": 1.0}

        classic = layout_sgd2_multi(
            edge_index=edge_index,
            num_nodes=8,
            steps=50,
            seed=42,
            criteria=criteria,
        )
        pipeline = layout_sgd2_multi_pipeline(
            edge_index=edge_index,
            num_nodes=8,
            steps=50,
            seed=42,
            criteria=criteria,
        )

        _assert_exact_match(classic, pipeline)

    def test_pipeline_matches_classic_crossing_angle(self) -> None:
        """Crossing angle maximization should match classic exactly."""
        edge_index = _complete_edge_index(5)
        criteria = {"crossing_angle_maximization": 1.0}

        classic = layout_sgd2_multi(
            edge_index=edge_index,
            num_nodes=5,
            steps=50,
            seed=42,
            criteria=criteria,
        )
        pipeline = layout_sgd2_multi_pipeline(
            edge_index=edge_index,
            num_nodes=5,
            steps=50,
            seed=42,
            criteria=criteria,
        )

        _assert_exact_match(classic, pipeline)

    def test_pipeline_matches_classic_angular_resolution(self) -> None:
        """Angular resolution criterion should match classic exactly."""
        edge_index = _complete_edge_index(5)
        criteria = {"angular_resolution": 1.0}

        classic = layout_sgd2_multi(
            edge_index=edge_index,
            num_nodes=5,
            steps=50,
            seed=42,
            criteria=criteria,
        )
        pipeline = layout_sgd2_multi_pipeline(
            edge_index=edge_index,
            num_nodes=5,
            steps=50,
            seed=42,
            criteria=criteria,
        )

        _assert_exact_match(classic, pipeline)

    def test_pipeline_matches_classic_vertex_resolution(self) -> None:
        """Vertex resolution criterion should match classic exactly."""
        edge_index = _path_edge_index(8)
        criteria = {"vertex_resolution": 1.0}

        classic = layout_sgd2_multi(
            edge_index=edge_index,
            num_nodes=8,
            steps=50,
            seed=42,
            criteria=criteria,
        )
        pipeline = layout_sgd2_multi_pipeline(
            edge_index=edge_index,
            num_nodes=8,
            steps=50,
            seed=42,
            criteria=criteria,
        )

        _assert_exact_match(classic, pipeline)

    def test_pipeline_matches_classic_crossings_neural(self) -> None:
        """Neural crossing detector should match classic exactly."""
        edge_index = _complete_edge_index(5)
        criteria = {"crossings": 1.0}

        classic = layout_sgd2_multi(
            edge_index=edge_index,
            num_nodes=5,
            steps=20,
            seed=42,
            criteria=criteria,
        )
        pipeline = layout_sgd2_multi_pipeline(
            edge_index=edge_index,
            num_nodes=5,
            steps=20,
            seed=42,
            criteria=criteria,
        )

        _assert_exact_match(classic, pipeline)

    def test_pipeline_matches_classic_with_edge_weights(self) -> None:
        """Weighted edges should produce bit-identical results."""
        edge_index = _edge_index_from_edges([(0, 1), (0, 2), (1, 3), (2, 4), (4, 5), (3, 5)])
        edge_weights = torch.tensor([0.25, 1.5, 2.0, 0.75, 1.25, 3.0], dtype=torch.float64)

        classic = layout_sgd2_multi(
            edge_index=edge_index,
            num_nodes=6,
            steps=50,
            seed=17,
            edge_weights=edge_weights,
        )
        pipeline = layout_sgd2_multi_pipeline(
            edge_index=edge_index,
            num_nodes=6,
            steps=50,
            seed=17,
            edge_weights=edge_weights,
        )

        _assert_exact_match(classic, pipeline)

    def test_pipeline_matches_classic_on_disconnected_graph(self) -> None:
        """Disconnected graphs should match classic exactly."""
        edge_index = _disconnected_edge_index()

        classic = layout_sgd2_multi(edge_index=edge_index, num_nodes=7, steps=50, seed=99)
        pipeline = layout_sgd2_multi_pipeline(edge_index=edge_index, num_nodes=7, steps=50, seed=99)

        _assert_exact_match(classic, pipeline)

    def test_build_pipeline_matches_classic_on_complete_graph(self) -> None:
        """The raw pipeline object should match classic on a dense graph."""
        edge_index = _complete_edge_index(5)

        classic = layout_sgd2_multi(edge_index=edge_index, num_nodes=5, steps=50, seed=7)
        pipeline = _run_pipeline_direct(edge_index=edge_index, num_nodes=5, steps=50, seed=7)

        _assert_exact_match(classic, pipeline)

    def test_pipeline_matches_classic_zero_steps(self) -> None:
        """Zero steps should produce valid output matching classic."""
        edge_index = _path_edge_index(5)

        classic = layout_sgd2_multi(edge_index=edge_index, num_nodes=5, steps=0, seed=42)
        pipeline = layout_sgd2_multi_pipeline(edge_index=edge_index, num_nodes=5, steps=0, seed=42)

        _assert_exact_match(classic, pipeline)

    def test_pipeline_matches_classic_custom_lr_momentum(self) -> None:
        """Custom optimizer hyperparameters should match classic."""
        edge_index = _path_edge_index(6)

        classic = layout_sgd2_multi(
            edge_index=edge_index,
            num_nodes=6,
            steps=50,
            seed=42,
            lr=0.5,
            momentum=0.9,
        )
        pipeline = layout_sgd2_multi_pipeline(
            edge_index=edge_index,
            num_nodes=6,
            steps=50,
            seed=42,
            lr=0.5,
            momentum=0.9,
        )

        _assert_exact_match(classic, pipeline)

    def test_pipeline_matches_classic_custom_batch_size(self) -> None:
        """Custom batch size should match classic exactly."""
        edge_index = _path_edge_index(8)

        classic = layout_sgd2_multi(
            edge_index=edge_index,
            num_nodes=8,
            steps=50,
            seed=42,
            batch_size=8,
        )
        pipeline = layout_sgd2_multi_pipeline(
            edge_index=edge_index,
            num_nodes=8,
            steps=50,
            seed=42,
            batch_size=8,
        )

        _assert_exact_match(classic, pipeline)

    def test_pipeline_matches_classic_star_graph(self) -> None:
        """Star graph via the raw build_sgd2_multi_pipeline."""
        edges = [(0, i) for i in range(1, 8)]
        edge_index = _edge_index_from_edges(edges)

        classic = layout_sgd2_multi(edge_index=edge_index, num_nodes=8, steps=50, seed=11)
        pipeline = _run_pipeline_direct(edge_index=edge_index, num_nodes=8, steps=50, seed=11)

        _assert_exact_match(classic, pipeline)

    def test_pipeline_matches_classic_cycle_graph(self) -> None:
        """Cycle graph should match exactly."""
        n = 6
        edges = [(i, (i + 1) % n) for i in range(n)]
        edge_index = _edge_index_from_edges(edges)

        classic = layout_sgd2_multi(edge_index=edge_index, num_nodes=n, steps=50, seed=42)
        pipeline = layout_sgd2_multi_pipeline(edge_index=edge_index, num_nodes=n, steps=50, seed=42)

        _assert_exact_match(classic, pipeline)

    def test_pipeline_matches_classic_full_multicriteria_mix(self) -> None:
        """Stress + ideal_edge_length + aspect_ratio combined should match."""
        edge_index = _complete_edge_index(5)
        criteria = {
            "stress": 1.0,
            "ideal_edge_length": 0.5,
            "aspect_ratio": 0.3,
        }

        classic = layout_sgd2_multi(
            edge_index=edge_index,
            num_nodes=5,
            steps=50,
            seed=42,
            criteria=criteria,
        )
        pipeline = layout_sgd2_multi_pipeline(
            edge_index=edge_index,
            num_nodes=5,
            steps=50,
            seed=42,
            criteria=criteria,
        )

        _assert_exact_match(classic, pipeline)


class TestSGD2MultiPipelineValidation:
    """Input validation tests for the pipeline adapter."""

    def test_negative_num_nodes_raises(self) -> None:
        """Negative num_nodes should raise ValueError."""
        edge_index = torch.empty((2, 0), dtype=torch.long)
        with pytest.raises(ValueError, match="num_nodes"):
            layout_sgd2_multi_pipeline(edge_index=edge_index, num_nodes=-1, steps=10, seed=42)

    def test_negative_steps_raises(self) -> None:
        """Negative steps should raise ValueError."""
        edge_index = torch.empty((2, 0), dtype=torch.long)
        with pytest.raises(ValueError, match="steps"):
            layout_sgd2_multi_pipeline(edge_index=edge_index, num_nodes=5, steps=-1, seed=42)

    def test_nonpositive_lr_raises(self) -> None:
        """Non-positive lr should raise ValueError."""
        edge_index = _path_edge_index(5)
        with pytest.raises(ValueError, match="lr"):
            layout_sgd2_multi_pipeline(
                edge_index=edge_index, num_nodes=5, steps=10, seed=42, lr=0.0
            )

    def test_invalid_momentum_raises(self) -> None:
        """Momentum outside [0, 1) should raise ValueError."""
        edge_index = _path_edge_index(5)
        with pytest.raises(ValueError, match="momentum"):
            layout_sgd2_multi_pipeline(
                edge_index=edge_index,
                num_nodes=5,
                steps=10,
                seed=42,
                momentum=1.0,
            )

    def test_nonpositive_grad_clamp_raises(self) -> None:
        """Non-positive grad_clamp should raise ValueError."""
        edge_index = _path_edge_index(5)
        with pytest.raises(ValueError, match="grad_clamp"):
            layout_sgd2_multi_pipeline(
                edge_index=edge_index,
                num_nodes=5,
                steps=10,
                seed=42,
                grad_clamp=0.0,
            )

    def test_nonpositive_batch_size_raises(self) -> None:
        """Non-positive batch_size should raise ValueError."""
        edge_index = _path_edge_index(5)
        with pytest.raises(ValueError, match="batch_size"):
            layout_sgd2_multi_pipeline(
                edge_index=edge_index,
                num_nodes=5,
                steps=10,
                seed=42,
                batch_size=0,
            )

    def test_negative_steps_in_build_raises(self) -> None:
        """Negative steps in build_sgd2_multi_pipeline should raise."""
        with pytest.raises(ValueError, match="steps"):
            build_sgd2_multi_pipeline(steps=-1)

    def test_edge_weights_wrong_ndim_raises(self) -> None:
        """Edge weights with wrong ndim should raise ValueError."""
        edge_index = _path_edge_index(3)
        edge_weights = torch.ones((2, 2), dtype=torch.float64)
        with pytest.raises(ValueError, match="edge_weights"):
            layout_sgd2_multi_pipeline(
                edge_index=edge_index,
                num_nodes=3,
                steps=10,
                seed=42,
                edge_weights=edge_weights,
            )

    def test_edge_weights_wrong_length_raises(self) -> None:
        """Edge weights with mismatched length should raise ValueError."""
        edge_index = _path_edge_index(3)
        edge_weights = torch.ones((5,), dtype=torch.float64)
        with pytest.raises(ValueError, match="edge_weights"):
            layout_sgd2_multi_pipeline(
                edge_index=edge_index,
                num_nodes=3,
                steps=10,
                seed=42,
                edge_weights=edge_weights,
            )
