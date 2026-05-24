"""Tests for primitive initialization ops."""

from __future__ import annotations

import random
from typing import Callable

import numpy as np
import pytest
import torch

from dagua.layout.classic.fr import layout_fr
from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.init import (
    GRAPHOPT_INITIAL_POS_KEY,
    CircularInit,
    CircularInitConfig,
    ClassicalMDSInit,
    ClassicalMDSInitConfig,
    DeterministicInit,
    DeterministicInitConfig,
    FromAlgorithmInit,
    FromAlgorithmInitConfig,
    GraphOptInitializePositions,
    GraphOptInitializePositionsConfig,
    PivotMDSInit,
    PivotMDSInitConfig,
    RandomNormalInit,
    RandomNormalInitConfig,
    RandomUniformInit,
    RandomUniformInitConfig,
    SpectralInit,
    SpectralInitConfig,
    XavierInit,
    XavierInitConfig,
)
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


def _make_problem(
    num_nodes: int = 10,
    seed: int = 42,
    device: str = "cpu",
) -> LayoutProblem:
    """Create a small layered test graph.

    Parameters
    ----------
    num_nodes : int, default=10
        Number of nodes in the graph.
    seed : int, default=42
        Problem seed forwarded to initialization ops.
    device : str, default="cpu"
        Device for the edge tensor.

    Returns
    -------
    LayoutProblem
        Layout problem with 15 edges when ``num_nodes == 10``.
    """

    if num_nodes == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    elif num_nodes == 1:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    else:
        edge_index = torch.tensor(
            [
                [2, 0, 1, 0, 1, 2, 3, 4, 5, 5, 4, 3, 5, 4, 3],
                [3, 4, 5, 5, 4, 5, 6, 7, 8, 9, 6, 7, 7, 8, 9],
            ],
            dtype=torch.long,
            device=device,
        )
    return LayoutProblem(edge_index=edge_index, num_nodes=num_nodes, seed=seed)


def _make_layers(num_nodes: int) -> torch.Tensor:
    """Create deterministic layer assignments for the small test graph.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.

    Returns
    -------
    torch.Tensor
        Layer tensor with shape ``[N]``.
    """

    if num_nodes == 0:
        return torch.empty((0,), dtype=torch.long)
    if num_nodes == 1:
        return torch.zeros((1,), dtype=torch.long)
    return torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2], dtype=torch.long)


def _make_path_problem(
    num_nodes: int,
    seed: int = 42,
    device: str = "cpu",
) -> LayoutProblem:
    """Create a connected path graph with ``num_nodes`` nodes.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    seed : int, default=42
        Problem seed forwarded to initialization ops.
    device : str, default="cpu"
        Device for the edge tensor.

    Returns
    -------
    LayoutProblem
        Path-graph layout problem.
    """
    if num_nodes <= 1:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    else:
        sources = torch.arange(0, num_nodes - 1, dtype=torch.long, device=device)
        targets = torch.arange(1, num_nodes, dtype=torch.long, device=device)
        edge_index = torch.stack((sources, targets), dim=0)
    return LayoutProblem(edge_index=edge_index, num_nodes=num_nodes, seed=seed)


def _make_context(
    device: str = "cpu",
    generator_seed: int | None = None,
) -> RuntimeContext:
    """Create a runtime context for init-op tests.

    Parameters
    ----------
    device : str, default="cpu"
        Output device requested by the execution plan.
    generator_seed : int | None, default=None
        Optional seed for ``ctx.generator``.

    Returns
    -------
    RuntimeContext
        Runtime context with a seeded generator when requested.
    """

    generator = None
    if generator_seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(generator_seed)
    return RuntimeContext(plan=ExecutionPlan(device=device), generator=generator)


def _state_factory(op: object, num_nodes: int) -> SolveState:
    """Build the minimal solve state required by an op.

    Parameters
    ----------
    op : object
        Op instance under test.
    num_nodes : int
        Number of nodes in the corresponding problem.

    Returns
    -------
    SolveState
        Solve state with layers populated for deterministic initialization.
    """

    if isinstance(op, DeterministicInit):
        return SolveState(layers=_make_layers(num_nodes))
    return SolveState()


def test_random_uniform_init_torch_backend_matches_seeded_generator() -> None:
    """Torch backend should match a single seeded ``torch.rand`` call exactly."""

    problem = _make_problem(seed=7)
    ctx = _make_context(generator_seed=7)

    result = RandomUniformInit(RandomUniformInitConfig(scale="unit", rng_backend="torch")).apply(
        problem, SolveState(), ctx
    )

    expected_generator = torch.Generator(device="cpu")
    expected_generator.manual_seed(7)
    expected = torch.rand((10, 2), generator=expected_generator, dtype=torch.float32)

    assert result.pos is not None
    assert torch.equal(result.pos.cpu(), expected)


def test_random_uniform_init_python_backend_matches_private_random_sequence() -> None:
    """Python backend should match row-major ``random.Random.random`` draws."""

    problem = _make_problem(seed=11)

    result = RandomUniformInit(RandomUniformInitConfig(scale="unit", rng_backend="python")).apply(
        problem, SolveState(), _make_context()
    )

    rng = random.Random(11)
    expected = torch.tensor(
        [rng.random() for _ in range(20)],
        dtype=torch.float32,
    ).reshape(10, 2)

    assert result.pos is not None
    assert torch.equal(result.pos.cpu(), expected)


def test_random_uniform_init_numpy_backend_matches_randomstate_exactly() -> None:
    """NumPy backend should exactly reproduce ``RandomState.rand`` output."""

    problem = _make_problem(seed=42)

    result = RandomUniformInit(RandomUniformInitConfig(scale="unit", rng_backend="numpy")).apply(
        problem, SolveState(), _make_context()
    )

    expected = torch.from_numpy(np.random.RandomState(42).rand(10, 2))

    assert result.pos is not None
    assert torch.equal(result.pos.cpu(), expected)


def test_graphopt_fidelity_init_matches_igraph_adapter_seed_matrix() -> None:
    """GraphOpt fidelity init should match the igraph benchmark seed matrix."""

    problem = _make_problem(num_nodes=6, seed=13)

    result = GraphOptInitializePositions(
        GraphOptInitializePositionsConfig(fidelity_mode=True)
    ).apply(
        problem,
        SolveState(),
        _make_context(),
    )

    expected = torch.from_numpy(np.random.RandomState(13).uniform(-1.0, 1.0, size=(6, 2)))

    assert result.pos is not None
    torch.testing.assert_close(result.pos.cpu(), expected)


def test_graphopt_init_uses_supplied_matrix_before_rng() -> None:
    """GraphOpt init should honor an explicit ``graphopt_initial_pos`` matrix."""

    problem = _make_problem(num_nodes=3, seed=13)
    initial_pos = torch.tensor(
        [[4.0, 3.0], [2.0, 1.0], [-1.0, -2.0]],
        dtype=torch.float32,
    )
    state = SolveState(extras={GRAPHOPT_INITIAL_POS_KEY: initial_pos})

    result = GraphOptInitializePositions(
        GraphOptInitializePositionsConfig(fidelity_mode=True)
    ).apply(
        problem,
        state,
        _make_context(),
    )

    assert result.pos is not None
    torch.testing.assert_close(result.pos.cpu(), initial_pos.to(dtype=torch.float64))


def test_random_normal_init_matches_seeded_torch_randn_exactly() -> None:
    """Normal init should match a single seeded ``torch.randn`` call exactly."""

    problem = _make_problem(seed=42)

    result = RandomNormalInit(RandomNormalInitConfig()).apply(
        problem,
        SolveState(),
        _make_context(),
    )

    expected_generator = torch.Generator(device="cpu")
    expected_generator.manual_seed(42)
    expected = torch.randn((10, 2), generator=expected_generator, dtype=torch.float32) * 1.0e-4

    assert result.pos is not None
    assert torch.equal(result.pos.cpu(), expected)


def test_circular_init_places_nodes_on_scaled_circle() -> None:
    """Circular init should place every node on the requested-radius circle."""

    problem = _make_problem(seed=3)

    result = CircularInit(CircularInitConfig(scale=2.5)).apply(
        problem,
        SolveState(),
        _make_context(),
    )

    assert result.pos is not None
    radii = torch.linalg.norm(result.pos.cpu(), dim=1)
    assert torch.allclose(radii, torch.full((10,), 2.5), atol=1.0e-6)
    assert torch.allclose(result.pos.cpu()[0], torch.tensor([2.5, 0.0]), atol=1.0e-6)


@pytest.mark.parametrize("num_nodes", [0, 1])
@pytest.mark.parametrize(
    "factory",
    [
        lambda: RandomUniformInit(RandomUniformInitConfig(scale="unit", rng_backend="numpy")),
        lambda: RandomNormalInit(RandomNormalInitConfig()),
        lambda: CircularInit(CircularInitConfig(scale=3.0)),
        lambda: XavierInit(XavierInitConfig(dim=2)),
        lambda: DeterministicInit(DeterministicInitConfig(method="input")),
    ],
)
def test_basic_init_ops_handle_empty_and_single_node_graphs(
    num_nodes: int,
    factory: Callable[[], object],
) -> None:
    """Core init ops should handle ``N=0`` and ``N=1`` consistently."""

    problem = _make_problem(num_nodes=num_nodes, seed=19)
    op = factory()

    result = op.apply(problem, _state_factory(op, num_nodes), _make_context())

    if num_nodes == 0:
        assert result.pos is None
    else:
        assert result.pos is not None
        assert result.pos.shape == (1, 2)
        torch.testing.assert_close(result.pos.cpu(), torch.zeros((1, 2), dtype=torch.float32))


def test_random_uniform_init_handles_disconnected_pair_without_crashing() -> None:
    """Uniform init should still allocate finite coordinates for an edgeless pair."""

    problem = _make_problem(num_nodes=2, seed=31)

    result = RandomUniformInit(RandomUniformInitConfig(scale="unit", rng_backend="numpy")).apply(
        problem, SolveState(), _make_context()
    )

    assert result.pos is not None
    assert result.pos.shape == (2, 2)
    assert torch.isfinite(result.pos).all()


def test_random_uniform_init_numpy_backend_is_reproducible_for_same_seed() -> None:
    """The NumPy backend should be bitwise reproducible for identical seeds."""

    first = RandomUniformInit(RandomUniformInitConfig(scale="unit", rng_backend="numpy")).apply(
        _make_problem(seed=42), SolveState(), _make_context()
    )
    second = RandomUniformInit(RandomUniformInitConfig(scale="unit", rng_backend="numpy")).apply(
        _make_problem(seed=42), SolveState(), _make_context()
    )

    assert first.pos is not None
    assert second.pos is not None
    assert torch.equal(first.pos.cpu(), second.pos.cpu())


def test_random_uniform_init_numpy_backend_changes_with_different_seed() -> None:
    """The NumPy backend should change coordinates when the seed changes."""

    first = RandomUniformInit(RandomUniformInitConfig(scale="unit", rng_backend="numpy")).apply(
        _make_problem(seed=42), SolveState(), _make_context()
    )
    second = RandomUniformInit(RandomUniformInitConfig(scale="unit", rng_backend="numpy")).apply(
        _make_problem(seed=43), SolveState(), _make_context()
    )

    assert first.pos is not None
    assert second.pos is not None
    assert not torch.equal(first.pos.cpu(), second.pos.cpu())


def test_random_normal_init_is_reproducible_for_same_seed() -> None:
    """Normal init should be bitwise reproducible for identical seeds."""

    first = RandomNormalInit(RandomNormalInitConfig()).apply(
        _make_problem(seed=42),
        SolveState(),
        _make_context(),
    )
    second = RandomNormalInit(RandomNormalInitConfig()).apply(
        _make_problem(seed=42),
        SolveState(),
        _make_context(),
    )

    assert first.pos is not None
    assert second.pos is not None
    assert torch.equal(first.pos.cpu(), second.pos.cpu())


def test_random_normal_init_changes_with_different_seed() -> None:
    """Normal init should change coordinates when the seed changes."""

    first = RandomNormalInit(RandomNormalInitConfig()).apply(
        _make_problem(seed=42),
        SolveState(),
        _make_context(),
    )
    second = RandomNormalInit(RandomNormalInitConfig()).apply(
        _make_problem(seed=43),
        SolveState(),
        _make_context(),
    )

    assert first.pos is not None
    assert second.pos is not None
    assert not torch.equal(first.pos.cpu(), second.pos.cpu())


@pytest.mark.parametrize(
    ("factory", "writes"),
    [
        (
            lambda: RandomUniformInit(RandomUniformInitConfig(scale="unit", rng_backend="numpy")),
            ("pos",),
        ),
        (lambda: RandomNormalInit(RandomNormalInitConfig()), ("pos",)),
        (lambda: CircularInit(CircularInitConfig()), ("pos",)),
        (lambda: XavierInit(XavierInitConfig(dim=2)), ("pos",)),
        (lambda: DeterministicInit(DeterministicInitConfig(method="input")), ("pos",)),
    ],
)
def test_init_op_metadata_declares_position_writes(
    factory: Callable[[], object],
    writes: tuple[str, ...],
) -> None:
    """Init ops covered here should declare position writes and nothing else."""

    op = factory()

    assert op.writes == writes


def test_xavier_init_matches_seeded_reference_sequence() -> None:
    """Xavier init should match the seeded NeuLay-style reference sequence."""

    problem = _make_problem(seed=42)
    cpu_rng_state = torch.random.get_rng_state()
    try:
        result = XavierInit(XavierInitConfig(dim=2)).apply(
            problem,
            SolveState(),
            _make_context(),
        )

        torch.manual_seed(42)
        expected = torch.empty((10, 2), dtype=torch.float32)
        torch.nn.init.xavier_uniform_(expected, gain=float(10) ** 0.5)
    finally:
        torch.random.set_rng_state(cpu_rng_state)

    assert result.pos is not None
    assert torch.equal(result.pos.cpu(), expected)


def test_deterministic_init_uses_barycenter_order_before_assigning_grid() -> None:
    """Deterministic init should reorder layer-1 nodes by barycenter."""

    problem = _make_problem(seed=5)
    state = SolveState(layers=_make_layers(problem.num_nodes))

    result = DeterministicInit(
        DeterministicInitConfig(method="barycenter", node_sep=25.0, rank_sep=50.0)
    ).apply(problem, state, _make_context())

    assert result.pos is not None
    assert torch.equal(result.pos.cpu()[:, 1], state.layers.to(dtype=torch.float32) * 50.0)
    assert result.pos[4, 0].item() == pytest.approx(0.0)
    assert result.pos[5, 0].item() == pytest.approx(25.0)
    assert result.pos[3, 0].item() == pytest.approx(50.0)


def test_spectral_init_produces_valid_positions_on_connected_graph() -> None:
    """Spectral init should return finite 2D positions for a connected graph."""

    problem = _make_path_problem(num_nodes=20, seed=13)

    result = SpectralInit(SpectralInitConfig()).apply(problem, SolveState(), _make_context())

    assert result.pos is not None
    assert result.pos.shape == (20, 2)
    assert torch.isfinite(result.pos).all()
    assert result.laplacian is not None


def test_classical_mds_init_produces_valid_positions() -> None:
    """Classical MDS init should produce finite 2D positions from APSP."""

    problem = _make_path_problem(num_nodes=10, seed=17)

    result = ClassicalMDSInit(ClassicalMDSInitConfig()).apply(
        problem,
        SolveState(),
        _make_context(),
    )

    assert result.pos is not None
    assert result.pos.shape == (10, 2)
    assert torch.isfinite(result.pos).all()
    assert result.distance_matrix is not None
    assert result.distance_matrix.shape == (10, 10)


def test_pivot_mds_init_respects_seeded_first_pivot_and_repeats_exactly() -> None:
    """Pivot MDS init should honor the seeded first pivot and be repeatable."""

    seed = 23
    problem = _make_path_problem(num_nodes=15, seed=seed)
    op = PivotMDSInit(PivotMDSInitConfig(n_pivots=6))

    first_result = op.apply(problem, SolveState(), _make_context())
    second_result = op.apply(problem, SolveState(), _make_context())

    expected_generator = torch.Generator(device="cpu")
    expected_generator.manual_seed(seed)
    expected_first_pivot = int(
        torch.randint(0, problem.num_nodes, (1,), generator=expected_generator).item()
    )

    assert first_result.pos is not None
    assert first_result.pivot_indices is not None
    assert first_result.pivot_distances is not None
    assert first_result.pivot_indices[0].item() == expected_first_pivot
    assert torch.equal(first_result.pos.cpu(), second_result.pos.cpu())
    assert torch.equal(first_result.pivot_indices.cpu(), second_result.pivot_indices.cpu())


def test_from_algorithm_init_delegates_to_fr() -> None:
    """Delegated FR init should match the direct FR classic implementation."""

    problem = _make_path_problem(num_nodes=12, seed=31)

    result = FromAlgorithmInit(FromAlgorithmInitConfig(algorithm="fr", inner_steps=50)).apply(
        problem, SolveState(), _make_context()
    )

    expected = layout_fr(
        edge_index=problem.edge_index,
        num_nodes=problem.num_nodes,
        node_sizes=problem.node_sizes,
        steps=50,
        seed=problem.seed,
        edge_weights=problem.edge_weights,
    )

    assert result.pos is not None
    assert torch.equal(result.pos.cpu(), expected.cpu())


@pytest.mark.parametrize(
    ("op_factory", "dim"),
    [
        (lambda: RandomUniformInit(), 2),
        (lambda: RandomNormalInit(), 2),
        (lambda: CircularInit(), 2),
        (lambda: XavierInit(), 2),
        (lambda: DeterministicInit(), 2),
    ],
)
def test_init_ops_handle_zero_and_single_node_edge_cases(
    op_factory: Callable[[], object],
    dim: int,
) -> None:
    """All init ops should leave ``N=0`` untouched and place ``N=1`` at the origin."""

    empty_problem = _make_problem(num_nodes=0)
    single_problem = _make_problem(num_nodes=1)
    empty_op = op_factory()
    single_op = op_factory()

    empty_state = _state_factory(empty_op, num_nodes=0)
    single_state = _state_factory(single_op, num_nodes=1)

    empty_result = empty_op.apply(empty_problem, empty_state, _make_context())
    single_result = single_op.apply(single_problem, single_state, _make_context())

    assert empty_result is empty_state
    assert empty_result.pos is None
    assert single_result.pos is not None
    assert torch.equal(single_result.pos.cpu(), torch.zeros((1, dim), dtype=torch.float32))


def test_pipeline_runs_random_uniform_init() -> None:
    """Pipeline composition should work with a single init op."""

    problem = _make_problem(seed=21)
    pipeline = Pipeline([RandomUniformInit()], name="init_only")

    result = pipeline.apply(problem, SolveState(), _make_context())

    assert result.pos is not None
    assert result.pos.shape == (10, 2)
    assert result.ops_applied == ["random_uniform_init"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for this test.")
@pytest.mark.parametrize(
    "op_factory",
    [
        lambda: RandomUniformInit(RandomUniformInitConfig(rng_backend="numpy", scale="unit")),
        lambda: RandomNormalInit(),
        lambda: CircularInit(),
        lambda: XavierInit(),
        lambda: DeterministicInit(),
    ],
)
def test_init_ops_respect_cuda_device_placement(
    op_factory: Callable[[], object],
) -> None:
    """Each init op should place its output tensor on ``ctx.plan.device``."""

    problem = _make_problem(device="cpu")
    op = op_factory()
    state = _state_factory(op, num_nodes=problem.num_nodes)

    result = op.apply(problem, state, _make_context(device="cuda"))

    assert result.pos is not None
    assert result.pos.device.type == "cuda"
