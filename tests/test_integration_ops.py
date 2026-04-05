"""Integration tests for algorithm dispatch and composable ops pipelines."""

from __future__ import annotations

import torch

import dagua
from dagua.config import LayoutConfig
from dagua.graph import DaguaGraph
from dagua.layout.ops.anneal import InitTemperatureFromExtent, LinearCool
from dagua.layout.ops.base import Pipeline, Repeat
from dagua.layout.ops.force import ApplyDisplacement, FRCombinedForce
from dagua.layout.ops.init import RandomUniformInit, RandomUniformInitConfig
from dagua.layout.ops.postprocess import CenterPositions, ScalePositions, ScalePositionsConfig
from dagua.layout.ops.preprocess import FRPrepareAdjacency
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.stress import (
    FinalizeStressMajorizationPositions,
    InitializeStressMajorizationPositions,
    PrepareStressMajorizationState,
    SmacofStep,
)

_TEST_SEED = 42
_TEST_NODE_COUNT = 10


def _build_test_graph() -> DaguaGraph:
    """Construct a small connected graph used across dispatch tests.

    Returns
    -------
    DaguaGraph
        Ten-node directed graph with a path backbone and a few shortcuts.
    """

    edges = [
        ("n0", "n1"),
        ("n1", "n2"),
        ("n2", "n3"),
        ("n3", "n4"),
        ("n4", "n5"),
        ("n5", "n6"),
        ("n6", "n7"),
        ("n7", "n8"),
        ("n8", "n9"),
        ("n0", "n5"),
        ("n2", "n7"),
        ("n4", "n9"),
    ]
    return DaguaGraph.from_edge_list(edges)


def _layout_problem_from_graph(graph: DaguaGraph, *, seed: int = _TEST_SEED) -> LayoutProblem:
    """Create a layout problem from a ``DaguaGraph``.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to convert into an immutable layout problem.
    seed : int, default=42
        Seed forwarded to pipeline ops that require deterministic randomness.

    Returns
    -------
    LayoutProblem
        Problem carrying the graph topology and resolved node sizes.
    """

    graph.compute_node_sizes()
    assert graph.node_sizes is not None
    return LayoutProblem(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        node_sizes=graph.node_sizes,
        seed=seed,
    )


def _runtime_context() -> RuntimeContext:
    """Build a CPU runtime context for direct pipeline execution.

    Returns
    -------
    RuntimeContext
        Runtime context with a CPU execution plan.
    """

    return RuntimeContext(plan=ExecutionPlan(device="cpu"))


def _assert_valid_positions(
    positions: torch.Tensor,
    *,
    num_nodes: int = _TEST_NODE_COUNT,
    dtype: torch.dtype | None = None,
) -> None:
    """Assert that a position tensor is finite and has the expected shape.

    Parameters
    ----------
    positions : torch.Tensor
        Candidate positions to validate.
    num_nodes : int, default=10
        Expected row count.
    dtype : torch.dtype | None, optional
        Expected dtype when provided.

    Returns
    -------
    None
        The assertions validate the tensor in-place.
    """

    assert positions.shape == (num_nodes, 2)
    assert torch.isfinite(positions).all()
    if dtype is not None:
        assert positions.dtype == dtype


def _assert_positions_differ(first: torch.Tensor, second: torch.Tensor) -> None:
    """Assert that two layouts differ by more than floating-point noise.

    Parameters
    ----------
    first : torch.Tensor
        First position tensor.
    second : torch.Tensor
        Second position tensor.

    Returns
    -------
    None
        The assertion ensures the compared layouts are not effectively equal.
    """

    max_difference = float(torch.max(torch.abs(first - second)).item())
    assert max_difference > 1.0e-5


def _run_pipeline(pipeline: Pipeline, problem: LayoutProblem) -> torch.Tensor:
    """Execute a composed ops pipeline and return its final coordinates.

    Parameters
    ----------
    pipeline : Pipeline
        Pipeline under test.
    problem : LayoutProblem
        Immutable layout problem consumed by the pipeline.

    Returns
    -------
    torch.Tensor
        Final position tensor produced by the pipeline.
    """

    final_state = pipeline.apply(problem, SolveState(), _runtime_context())
    assert final_state.pos is not None
    return final_state.pos


def test_layout_with_algorithm_fr() -> None:
    """FR algorithm dispatch should return a float32 ``[10, 2]`` tensor."""

    graph = _build_test_graph()
    positions = dagua.layout(graph, LayoutConfig(algorithm="fr", steps=50, seed=_TEST_SEED))

    _assert_valid_positions(positions, dtype=torch.float32)


def test_layout_with_algorithm_kk() -> None:
    """KK algorithm dispatch should return a float32 ``[10, 2]`` tensor."""

    graph = _build_test_graph()
    positions = dagua.layout(graph, LayoutConfig(algorithm="kk", steps=50, seed=_TEST_SEED))

    _assert_valid_positions(positions, dtype=torch.float32)


def test_layout_with_algorithm_fa2_params() -> None:
    """FA2 algorithm params should change the resulting layout."""

    graph = _build_test_graph()
    baseline = dagua.layout(graph, LayoutConfig(algorithm="fa2", steps=50, seed=_TEST_SEED))
    tuned = dagua.layout(
        graph,
        LayoutConfig(
            algorithm="fa2",
            steps=50,
            seed=_TEST_SEED,
            algorithm_params={"gravity": 2.0, "strong_gravity": True},
        ),
    )

    _assert_valid_positions(baseline, dtype=torch.float32)
    _assert_valid_positions(tuned, dtype=torch.float32)
    _assert_positions_differ(baseline, tuned)


def test_layout_with_algorithm_stress_maj() -> None:
    """Stress majorization dispatch should accept per-algorithm iterations."""

    graph = _build_test_graph()
    positions = dagua.layout(
        graph,
        LayoutConfig(
            algorithm="stress_majorization",
            seed=_TEST_SEED,
            algorithm_params={"iterations": 10},
        ),
    )

    _assert_valid_positions(positions, dtype=torch.float32)


def test_config_override_changes_output() -> None:
    """Changing FR step count should affect the resulting positions."""

    graph = _build_test_graph()
    short_run = dagua.layout(graph, LayoutConfig(algorithm="fr", steps=5, seed=_TEST_SEED))
    long_run = dagua.layout(graph, LayoutConfig(algorithm="fr", steps=50, seed=_TEST_SEED))

    _assert_valid_positions(short_run, dtype=torch.float32)
    _assert_valid_positions(long_run, dtype=torch.float32)
    _assert_positions_differ(short_run, long_run)


def test_cross_algorithm_composition() -> None:
    """Cross-family FR pipeline composition should produce valid positions."""

    graph = _build_test_graph()
    problem = _layout_problem_from_graph(graph)
    pipeline = Pipeline(
        [
            RandomUniformInit(RandomUniformInitConfig(rng_backend="numpy", scale="none")),
            FRPrepareAdjacency(),
            InitTemperatureFromExtent(),
            Repeat(
                n=20,
                ops=[
                    FRCombinedForce(),
                    ApplyDisplacement(),
                    LinearCool(),
                ],
            ),
            CenterPositions(),
            ScalePositions(ScalePositionsConfig(method="max_abs", factor=50.0)),
        ]
    )

    positions = _run_pipeline(pipeline, problem)
    _assert_valid_positions(positions)


def test_novel_hybrid_pipeline() -> None:
    """A stress-seeded FR refinement pipeline should execute without error."""

    graph = _build_test_graph()
    problem = _layout_problem_from_graph(graph)
    pipeline = Pipeline(
        [
            PrepareStressMajorizationState(),
            InitializeStressMajorizationPositions(),
            Repeat(n=5, ops=[SmacofStep()]),
            FinalizeStressMajorizationPositions(),
            FRPrepareAdjacency(),
            InitTemperatureFromExtent(),
            Repeat(
                n=10,
                ops=[
                    FRCombinedForce(),
                    ApplyDisplacement(),
                    LinearCool(),
                ],
            ),
            CenterPositions(),
            ScalePositions(ScalePositionsConfig(method="max_abs", factor=40.0)),
        ]
    )

    positions = _run_pipeline(pipeline, problem)
    _assert_valid_positions(positions, dtype=torch.float32)
