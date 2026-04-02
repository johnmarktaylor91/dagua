"""Tests for ordering layout ops."""

from __future__ import annotations

import pytest
import torch

from dagua.layout.ops.ordering import (
    BarycenterSweep,
    BarycenterSweepConfig,
    MedianSweep,
    MedianSweepConfig,
    SpectralOrder,
    TransposeHeuristic,
    TransposeHeuristicConfig,
)
from dagua.layout.ops.preprocess import BuildAdjacency
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState


def _edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    """Build a ``[2, E]`` edge tensor from a Python edge list.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Source-target edge tuples.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)
    sources, targets = zip(*edges)
    return torch.tensor([list(sources), list(targets)], dtype=torch.long)


def _layered_problem() -> LayoutProblem:
    """Create a small three-layer DAG used across ordering tests.

    Returns
    -------
    LayoutProblem
        Layout problem with six nodes and four directed edges.
    """
    return LayoutProblem(
        edge_index=_edge_index([(0, 3), (1, 2), (3, 4), (2, 5)]),
        num_nodes=6,
        seed=13,
    )


def _layered_state() -> SolveState:
    """Create the minimal layered state for ordering-op tests.

    Returns
    -------
    SolveState
        Solve state with a fixed three-layer assignment.
    """
    return SolveState(layers=torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long))


def _crossings_between_adjacent_layers(
    edge_index: torch.Tensor,
    layers: torch.Tensor,
    ordering: torch.Tensor,
) -> int:
    """Count crossings between edges whose endpoints lie on adjacent layers.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    layers : torch.Tensor
        Layer assignment tensor with shape ``[N]``.
    ordering : torch.Tensor
        Per-node in-layer ranks with shape ``[N]``.

    Returns
    -------
    int
        Number of pairwise edge crossings across adjacent layers.
    """
    edges = []
    for source, target in edge_index.t().tolist():
        source_layer = int(layers[source].item())
        target_layer = int(layers[target].item())
        if abs(source_layer - target_layer) != 1:
            continue
        if source_layer < target_layer:
            upper_node, lower_node = source, target
            upper_layer = source_layer
        else:
            upper_node, lower_node = target, source
            upper_layer = target_layer
        edges.append((upper_layer, upper_node, lower_node))

    crossings = 0
    for index, first in enumerate(edges):
        first_layer, first_upper, first_lower = first
        for second in edges[index + 1 :]:
            second_layer, second_upper, second_lower = second
            if first_layer != second_layer:
                continue
            upper_cross = int(ordering[first_upper].item()) - int(ordering[second_upper].item())
            lower_cross = int(ordering[first_lower].item()) - int(ordering[second_lower].item())
            if upper_cross * lower_cross < 0:
                crossings += 1
    return crossings


def _assert_layerwise_permutation(layers: torch.Tensor, ordering: torch.Tensor) -> None:
    """Assert that each layer stores a valid contiguous rank permutation.

    Parameters
    ----------
    layers : torch.Tensor
        Layer assignment tensor with shape ``[N]``.
    ordering : torch.Tensor
        Per-node in-layer ordering tensor with shape ``[N]``.

    Returns
    -------
    None
        Assertions fire on invalid permutations.
    """
    for layer_id in torch.unique(layers).tolist():
        layer_nodes = torch.nonzero(layers == int(layer_id), as_tuple=False).flatten()
        expected = torch.arange(layer_nodes.numel(), dtype=torch.long)
        actual = torch.sort(ordering[layer_nodes]).values.cpu()
        assert torch.equal(actual, expected)


def test_barycenter_sweep_reorders_middle_layer_on_three_layer_dag() -> None:
    """BarycenterSweep should move the middle layer toward parent barycenters."""
    problem = _layered_problem()
    state = _layered_state()

    BuildAdjacency().apply(problem, state, RuntimeContext())
    result = BarycenterSweep(
        BarycenterSweepConfig(passes=4, direction="both", use_weights=True)
    ).apply(problem, state, RuntimeContext())

    assert result.ordering is not None
    assert result.ordering.tolist() == [0, 1, 1, 0, 0, 1]


def test_median_sweep_matches_expected_middle_layer_order() -> None:
    """MedianSweep should produce the same stable order on the reference DAG."""
    problem = _layered_problem()
    state = _layered_state()

    BuildAdjacency().apply(problem, state, RuntimeContext())
    result = MedianSweep(MedianSweepConfig(passes=4)).apply(problem, state, RuntimeContext())

    assert result.ordering is not None
    assert result.ordering.tolist() == [0, 1, 1, 0, 0, 1]


def test_transpose_heuristic_swaps_crossing_pair() -> None:
    """TransposeHeuristic should reduce crossings while keeping per-layer permutations."""
    problem = _layered_problem()
    state = SolveState(
        layers=torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long),
        ordering=torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long),
    )
    initial_crossings = _crossings_between_adjacent_layers(
        problem.edge_index,
        state.layers,
        state.ordering,
    )

    result = TransposeHeuristic(TransposeHeuristicConfig(passes=4)).apply(
        problem,
        state,
        RuntimeContext(),
    )

    assert result.ordering is not None
    assert _crossings_between_adjacent_layers(problem.edge_index, state.layers, result.ordering) < (
        initial_crossings
    )
    for layer_id in range(3):
        layer_nodes = torch.nonzero(state.layers == layer_id, as_tuple=False).flatten()
        layer_order = result.ordering[layer_nodes]
        expected = torch.arange(layer_nodes.numel(), dtype=torch.long)
        assert torch.equal(torch.sort(layer_order).values.cpu(), expected)


def test_spectral_order_returns_per_layer_permutations() -> None:
    """SpectralOrder should emit a valid permutation of ranks inside each layer."""
    problem = LayoutProblem(
        edge_index=_edge_index([(0, 2), (1, 2), (2, 3), (2, 4)]),
        num_nodes=5,
        seed=19,
    )
    state = SolveState(layers=torch.tensor([0, 0, 1, 2, 2], dtype=torch.long))

    result = SpectralOrder().apply(problem, state, RuntimeContext())

    assert result.ordering is not None
    for layer_id in range(3):
        layer_nodes = torch.nonzero(state.layers == layer_id, as_tuple=False).flatten()
        layer_order = result.ordering[layer_nodes]
        expected = torch.arange(layer_nodes.numel(), dtype=torch.long)
        assert torch.equal(torch.sort(layer_order).values.cpu(), expected)


def test_barycenter_sweep_reduces_crossings_on_known_crossing_dag() -> None:
    """BarycenterSweep should reduce crossings from the default layer order."""

    problem = LayoutProblem(
        edge_index=_edge_index([(0, 3), (1, 2), (0, 5), (1, 4)]),
        num_nodes=6,
    )
    state = SolveState(layers=torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.long))
    BuildAdjacency().apply(problem, state, RuntimeContext())
    initial_ordering = torch.tensor([0, 1, 0, 1, 2, 3], dtype=torch.long)

    result = BarycenterSweep(BarycenterSweepConfig(passes=6)).apply(
        problem,
        state,
        RuntimeContext(),
    )

    assert result.ordering is not None
    assert _crossings_between_adjacent_layers(problem.edge_index, state.layers, result.ordering) < (
        _crossings_between_adjacent_layers(problem.edge_index, state.layers, initial_ordering)
    )


def test_transpose_heuristic_keeps_crossing_count_when_order_is_already_optimal() -> None:
    """TransposeHeuristic should avoid introducing new crossings on an optimal order."""

    problem = LayoutProblem(
        edge_index=_edge_index([(0, 2), (1, 3), (2, 4), (3, 5)]),
        num_nodes=6,
    )
    state = SolveState(
        layers=torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long),
        ordering=torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long),
    )
    initial_crossings = _crossings_between_adjacent_layers(
        problem.edge_index,
        state.layers,
        state.ordering,
    )

    result = TransposeHeuristic(TransposeHeuristicConfig(passes=4)).apply(
        problem,
        state,
        RuntimeContext(),
    )

    assert result.ordering is not None
    assert (
        _crossings_between_adjacent_layers(problem.edge_index, state.layers, result.ordering)
        <= initial_crossings
    )


def test_barycenter_sweep_zero_passes_leaves_the_default_layer_order() -> None:
    """With zero passes, BarycenterSweep should keep the initial per-layer order."""

    problem = _layered_problem()
    state = _layered_state()
    BuildAdjacency().apply(problem, state, RuntimeContext())

    result = BarycenterSweep(BarycenterSweepConfig(passes=0)).apply(
        problem,
        state,
        RuntimeContext(),
    )

    assert result.ordering is not None
    assert result.ordering.tolist() == [0, 1, 0, 1, 0, 1]


@pytest.mark.parametrize("direction", ["down", "up", "both"])
def test_barycenter_sweep_direction_modes_return_valid_permutations(direction: str) -> None:
    """Every BarycenterSweep direction should produce valid layerwise permutations."""

    problem = _layered_problem()
    state = _layered_state()
    BuildAdjacency().apply(problem, state, RuntimeContext())

    result = BarycenterSweep(BarycenterSweepConfig(passes=3, direction=direction)).apply(
        problem,
        state,
        RuntimeContext(),
    )

    assert result.ordering is not None
    _assert_layerwise_permutation(state.layers, result.ordering)


def test_barycenter_sweep_weighted_and_unweighted_modes_can_diverge() -> None:
    """Edge weights should be able to change the barycenter ordering."""

    problem = LayoutProblem(edge_index=_edge_index([]), num_nodes=5)
    state = SolveState(
        layers=torch.tensor([0, 0, 0, 1, 1], dtype=torch.long),
        adjacency=[
            [(3, 10.0), (4, 1.0)],
            [(4, 1.0)],
            [(3, 1.0)],
            [(0, 10.0), (2, 1.0)],
            [(0, 1.0), (1, 1.0)],
        ],
    )

    weighted = BarycenterSweep(BarycenterSweepConfig(passes=2, use_weights=True)).apply(
        problem,
        SolveState(layers=state.layers.clone(), adjacency=state.adjacency),
        RuntimeContext(),
    )
    unweighted = BarycenterSweep(BarycenterSweepConfig(passes=2, use_weights=False)).apply(
        problem,
        SolveState(layers=state.layers.clone(), adjacency=state.adjacency),
        RuntimeContext(),
    )

    assert weighted.ordering is not None
    assert unweighted.ordering is not None
    assert weighted.ordering.tolist() != unweighted.ordering.tolist()


def test_barycenter_sweep_handles_dense_adjacency_inputs() -> None:
    """BarycenterSweep should accept dense adjacency tensors as input."""

    problem = _layered_problem()
    state = _layered_state()
    BuildAdjacency().apply(problem, state, RuntimeContext())
    dense = torch.zeros((problem.num_nodes, problem.num_nodes), dtype=torch.float32)
    for source, target in problem.edge_index.t().tolist():
        dense[source, target] = 1.0
        dense[target, source] = 1.0
    state.adjacency = dense

    result = BarycenterSweep(BarycenterSweepConfig(passes=3)).apply(
        problem,
        state,
        RuntimeContext(),
    )

    assert result.ordering is not None
    _assert_layerwise_permutation(state.layers, result.ordering)


def test_median_sweep_produces_valid_permutations_on_weighted_adjacency() -> None:
    """MedianSweep should ignore weights but still accept weighted adjacency lists."""

    problem = LayoutProblem(edge_index=_edge_index([]), num_nodes=5)
    state = SolveState(
        layers=torch.tensor([0, 0, 0, 1, 1], dtype=torch.long),
        adjacency=[
            [(3, 10.0), (4, 1.0)],
            [(4, 1.0)],
            [(3, 1.0)],
            [(0, 10.0), (2, 1.0)],
            [(0, 1.0), (1, 1.0)],
        ],
    )

    result = MedianSweep(MedianSweepConfig(passes=3)).apply(problem, state, RuntimeContext())

    assert result.ordering is not None
    _assert_layerwise_permutation(state.layers, result.ordering)


def test_median_sweep_differs_from_weighted_barycenter_on_asymmetric_graph() -> None:
    """Median and weighted barycenter sweeps should diverge when weights dominate one side."""

    problem = LayoutProblem(edge_index=_edge_index([]), num_nodes=5)
    adjacency = [
        [(3, 10.0), (4, 1.0)],
        [(4, 1.0)],
        [(3, 1.0)],
        [(0, 10.0), (2, 1.0)],
        [(0, 1.0), (1, 1.0)],
    ]
    layers = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)

    barycenter = BarycenterSweep(BarycenterSweepConfig(passes=3, use_weights=True)).apply(
        problem,
        SolveState(layers=layers.clone(), adjacency=adjacency),
        RuntimeContext(),
    )
    median = MedianSweep(MedianSweepConfig(passes=3)).apply(
        problem,
        SolveState(layers=layers.clone(), adjacency=adjacency),
        RuntimeContext(),
    )

    assert barycenter.ordering is not None
    assert median.ordering is not None
    assert barycenter.ordering.tolist() != median.ordering.tolist()


def test_median_sweep_handles_dense_adjacency_inputs() -> None:
    """MedianSweep should accept dense adjacency tensors."""

    problem = _layered_problem()
    state = _layered_state()
    BuildAdjacency().apply(problem, state, RuntimeContext())
    dense = torch.zeros((problem.num_nodes, problem.num_nodes), dtype=torch.float32)
    for source, target in problem.edge_index.t().tolist():
        dense[source, target] = 1.0
        dense[target, source] = 1.0
    state.adjacency = dense

    result = MedianSweep(MedianSweepConfig(passes=3)).apply(problem, state, RuntimeContext())

    assert result.ordering is not None
    _assert_layerwise_permutation(state.layers, result.ordering)


def test_transpose_heuristic_zero_passes_leaves_ordering_unchanged() -> None:
    """Setting zero transpose passes should be a no-op."""

    problem = _layered_problem()
    state = SolveState(
        layers=torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long),
        ordering=torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long),
    )

    result = TransposeHeuristic(TransposeHeuristicConfig(passes=0)).apply(
        problem,
        state,
        RuntimeContext(),
    )

    assert result.ordering is not None
    assert result.ordering.tolist() == [0, 1, 0, 1, 0, 1]


def test_transpose_heuristic_preserves_valid_permutations_after_swaps() -> None:
    """TransposeHeuristic should keep contiguous ranks inside each layer."""

    problem = _layered_problem()
    state = SolveState(
        layers=torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long),
        ordering=torch.tensor([0, 1, 1, 0, 1, 0], dtype=torch.long),
    )

    result = TransposeHeuristic(TransposeHeuristicConfig(passes=4)).apply(
        problem,
        state,
        RuntimeContext(),
    )

    assert result.ordering is not None
    _assert_layerwise_permutation(state.layers, result.ordering)


def test_spectral_order_handles_disconnected_graphs() -> None:
    """SpectralOrder should still produce valid per-layer permutations when components split."""

    problem = LayoutProblem(
        edge_index=_edge_index([(0, 2), (1, 3)]),
        num_nodes=6,
        seed=31,
    )
    state = SolveState(layers=torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long))

    result = SpectralOrder().apply(problem, state, RuntimeContext())

    assert result.ordering is not None
    _assert_layerwise_permutation(state.layers, result.ordering)


def test_spectral_order_handles_graphs_without_edges() -> None:
    """SpectralOrder should fall back to the stable layer order on edgeless graphs."""

    problem = LayoutProblem(edge_index=_edge_index([]), num_nodes=4, seed=7)
    state = SolveState(layers=torch.tensor([0, 0, 1, 1], dtype=torch.long))

    result = SpectralOrder().apply(problem, state, RuntimeContext())

    assert result.ordering is not None
    assert result.ordering.tolist() == [0, 1, 0, 1]
