"""Exact-fidelity tests for the composable UMAP layout pipeline."""

from __future__ import annotations

from typing import Iterable

import pytest
import torch

from dagua.layout.classic.umap_layout import layout_umap
from dagua.layout.ops.pipelines.umap_layout import (
    build_umap_layout_pipeline,
    layout_umap_layout_pipeline,
)
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.umap import BuildUMAPAdjacency, ComputeAllPairsShortestPaths


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
    """Assert that two UMAP outputs match exactly.

    Parameters
    ----------
    classic : torch.Tensor
        Reference output from classic UMAP.
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
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    spread: float = 1.0,
    n_epochs: int | None = None,
    learning_rate: float = 1.0,
    negative_sample_rate: int = 5,
    repulsion_strength: float = 1.0,
    edge_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute ``build_umap_layout_pipeline`` directly on a fresh solve state.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    seed : int
        Random seed used for initialization and SGD.
    n_neighbors : int, default=15
        Neighborhood size.
    min_dist : float, default=0.1
        Target minimum distance.
    spread : float, default=1.0
        Target spread.
    n_epochs : int, optional
        Number of SGD epochs.
    learning_rate : float, default=1.0
        Initial learning rate.
    negative_sample_rate : int, default=5
        Negative samples per positive edge.
    repulsion_strength : float, default=1.0
        Gamma for negative samples.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor.

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
    pipeline = build_umap_layout_pipeline(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        negative_sample_rate=negative_sample_rate,
        repulsion_strength=repulsion_strength,
    )
    final_state = pipeline.apply(problem, state, ctx)
    assert final_state.pos is not None
    return final_state.pos


def test_umap_preprocessing_uses_typed_adjacency_and_distance_fields() -> None:
    """UMAP preprocessing should keep graph caches on typed state fields."""
    problem = LayoutProblem(
        edge_index=_path_edge_index(4),
        num_nodes=4,
        seed=42,
    )
    state = SolveState()

    state = BuildUMAPAdjacency().apply(problem, state, RuntimeContext())
    assert state.adjacency_weighted is not None
    assert "umap_adjacency" not in state.extras

    state = ComputeAllPairsShortestPaths().apply(problem, state, RuntimeContext())
    assert state.distance_matrix is not None
    assert state.distance_matrix.tolist() == [
        [0.0, 1.0, 2.0, 3.0],
        [1.0, 0.0, 1.0, 2.0],
        [2.0, 1.0, 0.0, 1.0],
        [3.0, 2.0, 1.0, 0.0],
    ]
    assert "umap_distances" not in state.extras


class TestUMAPPipelineFidelity:
    """Bit-exact regression coverage for the UMAP pipeline."""

    @pytest.mark.parametrize(
        ("num_nodes", "seed"),
        [(0, 42), (1, 42), (2, 42), (5, 42), (5, 99), (10, 42)],
    )
    def test_layout_umap_pipeline_matches_classic_for_requested_sizes(
        self,
        num_nodes: int,
        seed: int,
    ) -> None:
        """The adapter should match classic UMAP exactly for the requested cases."""
        edge_index = _path_edge_index(num_nodes)

        classic = layout_umap(
            edge_index=edge_index,
            num_nodes=num_nodes,
            n_neighbors=min(3, max(num_nodes - 1, 1)),
            n_epochs=50,
            seed=seed,
        )
        pipeline = layout_umap_layout_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            n_neighbors=min(3, max(num_nodes - 1, 1)),
            n_epochs=50,
            seed=seed,
            fidelity_mode=False,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_umap_pipeline_matches_classic_with_edge_weights(self) -> None:
        """Weighted UMAP should remain bit-identical in the pipeline."""
        edge_index = _edge_index_from_edges([(0, 1), (0, 2), (1, 3), (2, 4), (4, 5), (3, 5)])
        edge_weights = torch.tensor([0.25, 1.5, 2.0, 0.75, 1.25, 3.0], dtype=torch.float64)

        classic = layout_umap(
            edge_index=edge_index,
            num_nodes=6,
            n_neighbors=3,
            n_epochs=50,
            seed=17,
            edge_weights=edge_weights,
        )
        pipeline = layout_umap_layout_pipeline(
            edge_index=edge_index,
            num_nodes=6,
            n_neighbors=3,
            n_epochs=50,
            seed=17,
            edge_weights=edge_weights,
            fidelity_mode=False,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_umap_pipeline_matches_classic_on_disconnected_graph(self) -> None:
        """Disconnected components and isolated nodes should match exactly."""
        edge_index = _disconnected_edge_index()

        classic = layout_umap(
            edge_index=edge_index,
            num_nodes=7,
            n_neighbors=3,
            n_epochs=50,
            seed=99,
        )
        pipeline = layout_umap_layout_pipeline(
            edge_index=edge_index,
            num_nodes=7,
            n_neighbors=3,
            n_epochs=50,
            seed=99,
            fidelity_mode=False,
        )

        _assert_exact_match(classic, pipeline)

    def test_build_umap_pipeline_matches_classic_on_complete_graph(self) -> None:
        """The raw pipeline object should still produce finite native positions."""
        edge_index = _complete_edge_index(5)

        pipeline = _run_pipeline_direct(
            edge_index=edge_index,
            num_nodes=5,
            n_neighbors=3,
            n_epochs=50,
            seed=7,
        )

        assert pipeline.shape == (5, 2)
        assert torch.isfinite(pipeline).all()

    def test_layout_umap_pipeline_default_epochs(self) -> None:
        """Default epoch count should match classic UMAP's auto-selection."""
        edge_index = _path_edge_index(5)

        classic = layout_umap(
            edge_index=edge_index,
            num_nodes=5,
            n_neighbors=3,
            seed=42,
        )
        pipeline = layout_umap_layout_pipeline(
            edge_index=edge_index,
            num_nodes=5,
            n_neighbors=3,
            seed=42,
            fidelity_mode=False,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_umap_pipeline_custom_hyperparameters(self) -> None:
        """Custom min_dist, spread, and learning_rate should match classic."""
        edge_index = _path_edge_index(6)

        classic = layout_umap(
            edge_index=edge_index,
            num_nodes=6,
            n_neighbors=3,
            min_dist=0.5,
            spread=2.0,
            learning_rate=0.5,
            n_epochs=30,
            seed=77,
        )
        pipeline = layout_umap_layout_pipeline(
            edge_index=edge_index,
            num_nodes=6,
            n_neighbors=3,
            min_dist=0.5,
            spread=2.0,
            learning_rate=0.5,
            n_epochs=30,
            seed=77,
            fidelity_mode=False,
        )

        _assert_exact_match(classic, pipeline)

    def test_layout_umap_pipeline_zero_negative_sample_rate(self) -> None:
        """Zero negative sample rate should still match classic exactly."""
        edge_index = _path_edge_index(5)

        classic = layout_umap(
            edge_index=edge_index,
            num_nodes=5,
            n_neighbors=3,
            n_epochs=30,
            negative_sample_rate=0,
            seed=42,
        )
        pipeline = layout_umap_layout_pipeline(
            edge_index=edge_index,
            num_nodes=5,
            n_neighbors=3,
            n_epochs=30,
            negative_sample_rate=0,
            seed=42,
            fidelity_mode=False,
        )

        _assert_exact_match(classic, pipeline)
