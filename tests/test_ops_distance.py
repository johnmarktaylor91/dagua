"""Tests for distance layout ops."""

from __future__ import annotations

import numpy as np
import torch

from dagua.layout.classic._graph_distances import (
    all_pairs_shortest_paths,
    build_undirected_adjacency,
)
from dagua.layout.classic.pivot_mds import _select_pivots
from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.distance import (
    AllPairsShortestPaths,
    BFSDistances,
    BFSDistancesConfig,
    ConnectivityCheck,
    DijkstraDistances,
    PivotDistanceQueries,
    PivotSelection,
    PivotSelectionConfig,
)
from dagua.layout.ops.native_stress import (
    InflateStressTargetDistances,
    PrepareWarmStartStressMajorization,
    RunWarmStartStressSGDApproximateSchedule,
    RunWarmStartStressSGDApproximateScheduleConfig,
)
from dagua.layout.ops.preprocess import BuildAdjacency, BuildAdjacencyConfig
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, get_op_class


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
    src, tgt = zip(*edges)
    return torch.tensor([list(src), list(tgt)], dtype=torch.long)


def _problem(
    edges: list[tuple[int, int]],
    num_nodes: int,
    edge_weights: torch.Tensor | None = None,
) -> LayoutProblem:
    """Create a layout problem for distance-op tests.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Source-target edge tuples.
    num_nodes : int
        Number of graph nodes.
    edge_weights : torch.Tensor | None, optional
        Optional per-edge weights with shape ``[E]``.

    Returns
    -------
    LayoutProblem
        Test problem instance.
    """
    return LayoutProblem(
        edge_index=_edge_index(edges),
        num_nodes=num_nodes,
        edge_weights=edge_weights,
        seed=23,
    )


def _build_state(problem: LayoutProblem, weighted: bool = False) -> SolveState:
    """Build adjacency into a fresh solve state.

    Parameters
    ----------
    problem : LayoutProblem
        Layout problem whose topology should be cached.
    weighted : bool, default=False
        Whether to use ``problem.edge_weights``.

    Returns
    -------
    SolveState
        State with ``adjacency`` populated.
    """
    state = SolveState()
    BuildAdjacency(BuildAdjacencyConfig(weighted=weighted)).apply(
        problem,
        state,
        RuntimeContext(),
    )
    return state


def test_bfs_distances_match_reference_and_handle_disconnected_graphs() -> None:
    """BFSDistances should match the classic BFS matrix on a disconnected graph."""
    problem = _problem([(0, 1), (1, 2), (3, 4)], num_nodes=5)
    state = _build_state(problem)

    BFSDistances().apply(problem, state, RuntimeContext())

    expected = torch.from_numpy(
        all_pairs_shortest_paths(build_undirected_adjacency(problem.edge_index, 5), weighted=False)
    )
    assert state.distance_matrix is not None
    assert torch.equal(state.distance_matrix, expected)
    assert state.distance_matrix[0].tolist() == [0, 1, 2, -1, -1]


def test_inflate_stress_target_distances_updates_adjacent_exact_terms() -> None:
    """Size-aware stress inflation should only update adjacent pair targets."""
    problem = _problem([(0, 1), (1, 2)], num_nodes=3)
    problem.node_sizes = torch.tensor(
        [[6.0, 8.0], [0.0, 10.0], [2.0, 0.0]],
        dtype=torch.float32,
    )
    state = SolveState()
    state.extras["stress_sgd_sources"] = torch.tensor([0, 0, 1]).numpy()
    state.extras["stress_sgd_targets"] = torch.tensor([1, 2, 2]).numpy()
    state.extras["stress_sgd_distances"] = torch.tensor([1.0, 2.0, 1.0]).numpy()

    InflateStressTargetDistances().apply(problem, state, RuntimeContext())

    distances = torch.as_tensor(state.extras["stress_sgd_distances"])
    weights = torch.as_tensor(state.extras["stress_sgd_weights"])
    assert torch.allclose(distances, torch.tensor([11.0, 2.0, 7.0]))
    assert torch.allclose(weights, 1.0 / distances.square())


def test_prepare_warm_start_stress_majorization_inflates_dense_targets() -> None:
    """Warm-start SMACOF preparation should reuse positions and size-aware targets."""
    problem = _problem([(0, 1)], num_nodes=2)
    problem.node_sizes = torch.tensor([[2.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
    state = SolveState(pos=torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32))

    PrepareWarmStartStressMajorization().apply(problem, state, RuntimeContext())

    assert state.distance_matrix is not None
    assert torch.allclose(
        state.distance_matrix,
        torch.tensor([[0.0, 3.0], [3.0, 0.0]], dtype=state.distance_matrix.dtype),
    )
    assert "sm_current_positions" in state.extras


def test_warm_start_approximate_stress_sgd_keeps_existing_positions() -> None:
    """Approximate native-stress SGD should refine from ``state.pos``."""
    problem = _problem([(0, 1), (1, 2), (2, 3)], num_nodes=4)
    warm_start = torch.tensor(
        [[0.0, 0.0], [1.0, 0.2], [2.0, 0.1], [3.0, 0.0]],
        dtype=torch.float32,
    )
    state = SolveState(
        pos=warm_start.clone(),
        distance_matrix=torch.tensor(
            [[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]],
            dtype=torch.float32,
        ),
    )
    state.extras["stress_sgd_exact_mode"] = False
    state.extras["stress_sgd_num_nodes"] = 4
    state.extras["stress_sgd_device"] = torch.device("cpu")
    state.extras["stress_sgd_rng"] = np.random.RandomState(23)

    RunWarmStartStressSGDApproximateSchedule(
        RunWarmStartStressSGDApproximateScheduleConfig(steps=0)
    ).apply(problem, state, RuntimeContext())

    assert state.pos is not None
    assert torch.equal(state.pos, warm_start)


def test_native_stress_warm_start_approximate_op_is_registered() -> None:
    """Native-stress support ops should be discoverable in the op registry."""
    op_class = get_op_class("native_stress_warm_start_approximate_schedule")

    assert op_class.category == OpCategory.OPTIMIZE


def test_bfs_distances_support_per_source_queries() -> None:
    """BFSDistances should optionally emit per-source rows into extras."""
    problem = _problem([(0, 1), (1, 2), (2, 3)], num_nodes=4)
    state = _build_state(problem)
    state.extras["distance_sources"] = [2]

    BFSDistances(BFSDistancesConfig(unreachable=-9)).apply(problem, state, RuntimeContext())

    assert state.distance_matrix is None
    assert 2 in state.extras["distance_rows"]
    assert state.extras["distance_rows"][2].tolist() == [2, 1, 0, 1]


def test_dijkstra_distances_match_reference_weighted_paths() -> None:
    """DijkstraDistances should match the classic weighted all-pairs matrix."""
    weights = torch.tensor([1.0, 1.0, 5.0], dtype=torch.float32)
    problem = _problem([(0, 1), (1, 2), (0, 2)], num_nodes=3, edge_weights=weights)
    state = _build_state(problem, weighted=True)

    DijkstraDistances().apply(problem, state, RuntimeContext())

    expected = torch.from_numpy(
        all_pairs_shortest_paths(
            build_undirected_adjacency(problem.edge_index, 3, edge_weights=weights),
            weighted=True,
        )
    )
    assert state.distance_matrix is not None
    assert torch.equal(state.distance_matrix, expected)
    assert torch.allclose(
        state.distance_matrix[0],
        torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64),
    )


def test_all_pairs_shortest_paths_matches_reference_exactly_on_connected_graph() -> None:
    """APSP should match the shared classic helper exactly on connected graphs."""
    weights = torch.tensor([2.0, 3.0, 1.0], dtype=torch.float32)
    problem = _problem([(0, 1), (1, 2), (2, 3)], num_nodes=4, edge_weights=weights)
    state = _build_state(problem, weighted=True)

    AllPairsShortestPaths().apply(problem, state, RuntimeContext())

    expected = torch.from_numpy(
        all_pairs_shortest_paths(
            build_undirected_adjacency(problem.edge_index, 4, edge_weights=weights),
            weighted=True,
        )
    )
    assert state.distance_matrix is not None
    assert torch.equal(state.distance_matrix, expected)


def test_all_pairs_shortest_paths_fills_disconnected_pairs_with_max_plus_one() -> None:
    """Disconnected APSP should replace unreachable pairs with ``max + 1``."""
    problem = _problem([(0, 1), (2, 3)], num_nodes=4)
    state = _build_state(problem)

    AllPairsShortestPaths().apply(problem, state, RuntimeContext())

    assert state.distance_matrix is not None
    assert state.distance_matrix.dtype == torch.float64
    assert state.distance_matrix.tolist() == [
        [0.0, 1.0, 2.0, 2.0],
        [1.0, 0.0, 2.0, 2.0],
        [2.0, 2.0, 0.0, 1.0],
        [2.0, 2.0, 1.0, 0.0],
    ]


def test_pivot_selection_matches_reference_maxmin_heuristic() -> None:
    """PivotSelection should reproduce the reference Pivot-MDS pivot order."""
    problem = _problem([(0, 1), (1, 2), (2, 3), (2, 4), (4, 5)], num_nodes=6)
    state = _build_state(problem)
    ctx = RuntimeContext()

    PivotSelection(PivotSelectionConfig(n_pivots=3)).apply(problem, state, ctx)

    expected_indices, _ = _select_pivots(
        build_undirected_adjacency(problem.edge_index, problem.num_nodes),
        n_pivots=3,
        seed=problem.seed,
        weighted=False,
    )
    assert state.pivot_indices is not None
    assert torch.equal(state.pivot_indices, expected_indices)


def test_pivot_distance_queries_match_reference_distances() -> None:
    """PivotDistanceQueries should reproduce the reference pivot distance rows."""
    weights = torch.tensor([1.0, 2.0, 1.0, 3.0], dtype=torch.float32)
    problem = _problem([(0, 1), (1, 2), (2, 3), (1, 4)], num_nodes=5, edge_weights=weights)
    state = _build_state(problem, weighted=True)

    PivotSelection(PivotSelectionConfig(n_pivots=2)).apply(problem, state, RuntimeContext())
    PivotDistanceQueries().apply(problem, state, RuntimeContext())

    _, expected_distances = _select_pivots(
        build_undirected_adjacency(problem.edge_index, problem.num_nodes, edge_weights=weights),
        n_pivots=2,
        seed=problem.seed,
        weighted=True,
    )
    assert state.pivot_distances is not None
    assert torch.equal(state.pivot_distances, expected_distances)


def test_connectivity_check_sets_boolean_flag() -> None:
    """ConnectivityCheck should report graph connectivity in extras."""
    problem = _problem([(0, 1), (2, 3)], num_nodes=4)
    state = _build_state(problem)

    ConnectivityCheck().apply(problem, state, RuntimeContext())

    assert state.extras["is_connected"] is False


def test_bfs_distances_handles_empty_graph() -> None:
    """BFS distance computation should return an empty matrix for ``N=0``."""

    problem = _problem([], num_nodes=0)
    state = _build_state(problem)

    BFSDistances().apply(problem, state, RuntimeContext())

    assert state.distance_matrix is not None
    assert state.distance_matrix.shape == (0, 0)
    assert state.distance_matrix.dtype == torch.int32


def test_all_pairs_shortest_paths_handles_empty_graph() -> None:
    """APSP should return an empty matrix for ``N=0`` without crashing."""

    problem = _problem([], num_nodes=0)
    state = _build_state(problem)

    AllPairsShortestPaths().apply(problem, state, RuntimeContext())

    assert state.distance_matrix is not None
    assert state.distance_matrix.shape == (0, 0)
    assert state.distance_matrix.dtype == torch.float64


def test_pivot_selection_and_queries_handle_empty_graph() -> None:
    """Pivot ops should emit empty tensors for an empty graph."""

    problem = _problem([], num_nodes=0)
    state = _build_state(problem)

    PivotSelection(PivotSelectionConfig(n_pivots=3)).apply(problem, state, RuntimeContext())
    PivotDistanceQueries().apply(problem, state, RuntimeContext())

    assert state.pivot_indices is not None
    assert state.pivot_indices.shape == (0,)
    assert state.pivot_distances is not None
    assert state.pivot_distances.shape == (0, 0)


def test_bfs_distances_handles_disconnected_pair_without_edges() -> None:
    """Two isolated nodes should keep the unreachable BFS sentinel."""

    problem = _problem([], num_nodes=2)
    state = _build_state(problem)

    BFSDistances().apply(problem, state, RuntimeContext())

    assert state.distance_matrix is not None
    assert state.distance_matrix.tolist() == [[0, -1], [-1, 0]]


def test_dijkstra_distances_support_per_source_queries_with_parallel_edges() -> None:
    """Weighted per-source queries should respect duplicate-edge aggregation."""

    weights = torch.tensor([5.0, 1.0, 2.0], dtype=torch.float32)
    problem = _problem([(0, 1), (0, 1), (1, 2)], num_nodes=3, edge_weights=weights)
    state = SolveState()
    state.extras["distance_sources"] = [0]

    BuildAdjacency(BuildAdjacencyConfig(weighted=True)).apply(problem, state, RuntimeContext())
    DijkstraDistances().apply(problem, state, RuntimeContext())

    assert state.distance_matrix is None
    assert state.extras["distance_rows"][0].tolist() == [0.0, 1.0, 3.0]


def test_all_pairs_shortest_paths_ignores_self_loops_and_uses_min_parallel_edge() -> None:
    """Weighted APSP should ignore loops and keep the minimum duplicate cost."""

    weights = torch.tensor([7.0, 5.0, 1.0, 2.0], dtype=torch.float32)
    problem = _problem([(0, 0), (0, 1), (0, 1), (1, 2)], num_nodes=3, edge_weights=weights)
    state = SolveState()

    BuildAdjacency(BuildAdjacencyConfig(weighted=True)).apply(problem, state, RuntimeContext())
    AllPairsShortestPaths().apply(problem, state, RuntimeContext())

    assert state.distance_matrix is not None
    assert state.distance_matrix.tolist() == [
        [0.0, 1.0, 3.0],
        [1.0, 0.0, 2.0],
        [3.0, 2.0, 0.0],
    ]


def test_distance_pipeline_builds_apsp_for_large_fan_out_star() -> None:
    """BuildAdjacency -> APSP should produce the expected star-graph distances."""

    edges = [(0, leaf) for leaf in range(1, 100)]
    problem = _problem(edges, num_nodes=100)
    pipeline = Pipeline([BuildAdjacency(), AllPairsShortestPaths()])

    result = pipeline.apply(problem, SolveState(), RuntimeContext())

    assert result.distance_matrix is not None
    assert result.distance_matrix.shape == (100, 100)
    assert float(result.distance_matrix[0, 99].item()) == 1.0
    assert float(result.distance_matrix[1, 99].item()) == 2.0


def test_bfs_distances_clears_stale_distance_rows_after_full_matrix_run() -> None:
    """A full BFS matrix run should clear stale per-source rows from extras."""

    problem = _problem([(0, 1), (1, 2)], num_nodes=3)
    state = _build_state(problem)
    state.extras["distance_sources"] = [0]

    BFSDistances().apply(problem, state, RuntimeContext())
    assert "distance_rows" in state.extras

    state.extras.pop("distance_sources")
    BFSDistances().apply(problem, state, RuntimeContext())

    assert state.distance_matrix is not None
    assert state.distance_matrix.tolist() == [
        [0.0, 1.0, 2.0],
        [1.0, 0.0, 1.0],
        [2.0, 1.0, 0.0],
    ]
    assert "distance_rows" not in state.extras


def test_distance_op_metadata_declares_expected_targets() -> None:
    """Distance ops covered here should advertise their primary write targets."""

    assert "distance_matrix" in BFSDistances.writes
    assert "distance_matrix" in DijkstraDistances.writes
    assert AllPairsShortestPaths.writes == ("distance_matrix",)
    assert PivotSelection.writes == ("pivot_indices",)
    assert PivotDistanceQueries.writes == ("pivot_distances",)
    assert ConnectivityCheck.writes == ("extras.is_connected",)
