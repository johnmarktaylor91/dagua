"""Pipeline-composition tests for complete layout algorithm workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pytest
import torch

from dagua.layout.ops.anneal import LinearCool, LRDecay, LRDecayConfig
from dagua.layout.ops.base import Conditional, EarlyBreak, LossGroup, LossOp, Op, Pipeline, Repeat
from dagua.layout.ops.coarsen import HeavyEdgeMatching
from dagua.layout.ops.converge import (
    DisplacementThreshold,
    DisplacementThresholdConfig,
    StallCount,
    StallCountConfig,
)
from dagua.layout.ops.coordinate import BrandesKopf4Pass
from dagua.layout.ops.distance import AllPairsShortestPaths
from dagua.layout.ops.force import (
    ApplyDisplacement,
    InverseDistanceRepulsion,
    StressSGDPairUpdate,
    UniformSpringAttraction,
    ZeroForces,
)
from dagua.layout.ops.init import (
    RandomNormalInit,
    RandomNormalInitConfig,
    RandomUniformInit,
    SpectralInit,
)
from dagua.layout.ops.layering import InsertDummyNodes, LayerPromotion, LongestPathLayering
from dagua.layout.ops.loss_classic import (
    ExactPairStressLoss,
    LinLogAttractionLoss,
    LinLogRepulsionLoss,
)
from dagua.layout.ops.loss_engine import DagOrderingLoss, EdgeAttractionLoss, RepulsionLoss
from dagua.layout.ops.optimize import (
    ClipGradNorm,
    ClipGradNormConfig,
    CreateOptimizer,
    CreateOptimizerConfig,
    OptimizerStep,
)
from dagua.layout.ops.ordering import BarycenterSweep
from dagua.layout.ops.postprocess import (
    CenterPositions,
    DirectionTransform,
    DirectionTransformConfig,
    ScalePositions,
    ScalePositionsConfig,
    StripDummyNodes,
)
from dagua.layout.ops.preprocess import BuildAdjacency, DetectCycles, MakeAcyclic
from dagua.layout.ops.prolong import DirectMapping, DirectMappingConfig
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


def _edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    """Build a standard ``[2, E]`` edge tensor.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Directed edges as ``(source, target)`` pairs.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """

    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)
    sources, targets = zip(*edges)
    return torch.tensor([list(sources), list(targets)], dtype=torch.long)


def _path_problem(
    num_nodes: int,
    *,
    seed: int = 0,
    node_size: float = 5.0,
) -> LayoutProblem:
    """Create a directed path graph with uniform node sizes.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    seed : int, default=0
        Problem seed forwarded to random ops.
    node_size : float, default=5.0
        Width and height assigned to every node.

    Returns
    -------
    LayoutProblem
        Path-graph layout problem.
    """

    edges = [(index, index + 1) for index in range(num_nodes - 1)]
    return LayoutProblem(
        edge_index=_edge_index(edges),
        num_nodes=num_nodes,
        node_sizes=torch.full((num_nodes, 2), node_size, dtype=torch.float32),
        seed=seed,
    )


def _runtime_context(seed: int = 0) -> RuntimeContext:
    """Create a deterministic CPU runtime context.

    Parameters
    ----------
    seed : int, default=0
        Seed for the shared torch generator.

    Returns
    -------
    RuntimeContext
        Runtime context with a seeded generator.
    """

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return RuntimeContext(plan=ExecutionPlan(device="cpu"), generator=generator)


def _assert_valid_positions(state: SolveState, num_nodes: int) -> torch.Tensor:
    """Assert that a solve state carries finite 2D coordinates.

    Parameters
    ----------
    state : SolveState
        Solve state under test.
    num_nodes : int
        Expected number of visible nodes.

    Returns
    -------
    torch.Tensor
        Validated position tensor.
    """

    assert state.pos is not None
    assert state.pos.shape == (num_nodes, 2)
    assert torch.isfinite(state.pos).all()
    return state.pos


def _clone_state_with_grad(state: SolveState) -> SolveState:
    """Clone the differentiable fields needed for scalar loss evaluation.

    Parameters
    ----------
    state : SolveState
        Source state to clone.

    Returns
    -------
    SolveState
        State with a fresh differentiable ``pos`` tensor.
    """

    assert state.pos is not None
    cloned = SolveState(
        pos=state.pos.detach().clone().requires_grad_(True),
        layers=None if state.layers is None else state.layers.clone(),
        distance_matrix=None if state.distance_matrix is None else state.distance_matrix.clone(),
        extras=dict(state.extras),
    )
    return cloned


def _engine_total_loss(problem: LayoutProblem, state: SolveState) -> float:
    """Evaluate the requested engine-loss bundle without backpropagation.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable graph inputs.
    state : SolveState
        Solve state containing differentiable positions.

    Returns
    -------
    float
        Sum of DAG, attraction, and repulsion losses.
    """

    eval_state = _clone_state_with_grad(state)
    return float(
        (
            DagOrderingLoss().evaluate(problem, eval_state, RuntimeContext())
            + EdgeAttractionLoss().evaluate(problem, eval_state, RuntimeContext())
            + RepulsionLoss().evaluate(problem, eval_state, RuntimeContext())
        ).item()
    )


def _linlog_total_loss(problem: LayoutProblem, state: SolveState) -> float:
    """Evaluate the LinLog attraction-plus-repulsion objective.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable graph inputs.
    state : SolveState
        Solve state containing differentiable positions.

    Returns
    -------
    float
        Sum of LinLog attraction and repulsion losses.
    """

    eval_state = _clone_state_with_grad(state)
    return float(
        (
            LinLogAttractionLoss().evaluate(problem, eval_state, RuntimeContext())
            + LinLogRepulsionLoss().evaluate(problem, eval_state, RuntimeContext())
        ).item()
    )


def _exact_stress(problem: LayoutProblem, state: SolveState) -> float:
    """Evaluate exact stress for the current embedding.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable graph inputs.
    state : SolveState
        Solve state containing positions and APSP distances.

    Returns
    -------
    float
        Exact stress value.
    """

    eval_state = _clone_state_with_grad(state)
    return float(ExactPairStressLoss().evaluate(problem, eval_state, RuntimeContext()).item())


def _state_is_converged(
    problem: LayoutProblem,
    state: SolveState,
    ctx: RuntimeContext,
) -> bool:
    """Return the current convergence flag.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable graph inputs.
    state : SolveState
        Solve state under test.
    ctx : RuntimeContext
        Runtime context. Unused.

    Returns
    -------
    bool
        Whether the state has converged.
    """

    del problem, ctx
    return state.converged


def _repeat_counter_converged(
    problem: LayoutProblem,
    state: SolveState,
    ctx: RuntimeContext,
) -> bool:
    """Read the convergence latch set by the test compute op.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable graph inputs.
    state : SolveState
        Solve state under test.
    ctx : RuntimeContext
        Runtime context. Unused.

    Returns
    -------
    bool
        ``True`` when the test repeat loop should stop.
    """

    del problem, ctx
    return bool(state.extras.get("repeat_counter_converged", False))


@dataclass
class _RecordDisplacement(Op):
    """Record per-iteration mean displacement for FR convergence assertions."""

    name: str = "record_displacement"
    category: str = "test"

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Append the mean displacement since the previous call.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state carrying positions.
        ctx : RuntimeContext
            Runtime context. Unused.

        Returns
        -------
        SolveState
            State with updated displacement history in ``extras``.
        """

        del problem, ctx
        if state.pos is None:
            return state
        previous = state.extras.get("fr_record_prev_pos")
        state.extras["fr_record_prev_pos"] = state.pos.detach().clone()
        if not isinstance(previous, torch.Tensor) or (
            tuple(previous.shape) != tuple(state.pos.shape)
        ):
            return state
        displacement = (state.pos.detach() - previous.to(device=state.pos.device)).norm(dim=1)
        history = state.extras.setdefault("fr_displacement_history", [])
        history.append(float(displacement.mean().item()))
        return state


@dataclass
class _MarkBranch(Op):
    """Write the executed conditional branch name into ``state.extras``."""

    branch: str
    name: str = "mark_branch"
    category: str = "test"

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Record the branch name.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context. Unused.

        Returns
        -------
        SolveState
            State with ``extras['branch']`` updated.
        """

        del problem, ctx
        state.extras["branch"] = self.branch
        return state


@dataclass
class _ConvergingCounterOp(Op):
    """Increment a repeat counter and flip a test convergence latch."""

    limit: int
    name: str = "compute_op"
    category: str = "test"

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Update the loop counter stored in ``state.extras``.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context. Unused.

        Returns
        -------
        SolveState
            State with updated repeat bookkeeping.
        """

        del problem, ctx
        iterations = int(state.extras.get("repeat_iterations", 0)) + 1
        state.extras["repeat_iterations"] = iterations
        state.extras["repeat_counter_converged"] = iterations >= self.limit
        return state


@dataclass
class _QuadraticLoss(LossOp):
    """Simple scalar loss used to observe LossGroup backward behavior."""

    scale: float
    label: str

    name: str = "quadratic_loss"
    category: str = "test"
    weight_key: str = ""
    default_weight: float = 1.0

    def evaluate(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> torch.Tensor:
        """Return a scaled quadratic energy on the active positions.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state with differentiable positions.
        ctx : RuntimeContext
            Runtime context. Unused.

        Returns
        -------
        torch.Tensor
            Scalar differentiable loss.
        """

        del problem, ctx
        assert state.pos is not None
        return state.pos.square().sum() * self.scale

    def __repr__(self) -> str:
        """Return a descriptive debug representation.

        Returns
        -------
        str
            Loss label and scale.
        """

        return f"_QuadraticLoss(label={self.label!r}, scale={self.scale})"


def test_fruchterman_reingold_pipeline_produces_positions_consumes_forces_and_cools() -> None:
    """The FR-style pipeline should move nodes, accumulate forces, and cool down."""

    problem = LayoutProblem(
        edge_index=_edge_index([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5), (1, 4)]),
        num_nodes=6,
        node_sizes=torch.full((6, 2), 4.0, dtype=torch.float32),
        seed=7,
    )
    initial_state = SolveState(temperature=1.0, total_steps=10, extras={"force_area": 1.0})
    init_only = RandomUniformInit().apply(problem, SolveState(), _runtime_context(seed=7))
    pipeline = Pipeline(
        [
            RandomUniformInit(),
            Repeat(
                10,
                [
                    ZeroForces(),
                    InverseDistanceRepulsion(),
                    UniformSpringAttraction(),
                    ApplyDisplacement(),
                    LinearCool(),
                    DisplacementThreshold(DisplacementThresholdConfig(threshold=0.05)),
                    EarlyBreak(predicate=_state_is_converged),
                ],
            ),
        ]
    )

    result = pipeline.apply(problem, initial_state, _runtime_context(seed=7))

    positions = _assert_valid_positions(result, problem.num_nodes)
    assert init_only.pos is not None
    assert not torch.allclose(positions, init_only.pos)
    assert result.forces is not None
    assert torch.linalg.norm(result.forces).item() > 0.0
    assert result.temperature is not None
    assert result.temperature < 1.0


def test_sugiyama_pipeline_produces_positions_assigns_layers_and_strips_dummy_nodes() -> None:
    """The layered Sugiyama pipeline should assign layers and preserve original node count."""

    problem = LayoutProblem(
        edge_index=_edge_index([(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 4)]),
        num_nodes=6,
        node_sizes=torch.ones((6, 2), dtype=torch.float32),
        seed=11,
    )
    pipeline = Pipeline(
        [
            DetectCycles(),
            MakeAcyclic(),
            BuildAdjacency(),
            LongestPathLayering(),
            LayerPromotion(),
            InsertDummyNodes(),
            BarycenterSweep(),
            BrandesKopf4Pass(),
            StripDummyNodes(),
        ]
    )

    result = pipeline.apply(problem, SolveState(), RuntimeContext())

    _assert_valid_positions(result, problem.num_nodes)
    assert result.layers is not None
    assert result.layers.shape == (problem.num_nodes,)
    expanded_graph = result.extras["expanded_graph"]
    assert expanded_graph.num_nodes > problem.num_nodes
    assert result.pos is not None
    assert result.pos.shape[0] == problem.num_nodes


def test_gradient_based_engine_loop_pipeline_decreases_loss_updates_positions_and_converges() -> (
    None
):
    """The optimizer-driven pipeline should reduce loss and stop through convergence logic."""

    problem = LayoutProblem(
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        num_nodes=5,
        node_sizes=torch.full((5, 2), 10.0, dtype=torch.float32),
        seed=13,
    )
    prefix = Pipeline(
        [
            RandomUniformInit(),
            CreateOptimizer(CreateOptimizerConfig(optimizer_type="adam", lr=0.1)),
        ]
    )
    initial_state = prefix.apply(problem, SolveState(total_steps=20), _runtime_context(seed=13))
    initial_loss = _engine_total_loss(problem, initial_state)
    initial_pos = initial_state.pos.detach().clone() if initial_state.pos is not None else None
    pipeline = Pipeline(
        [
            RandomUniformInit(),
            CreateOptimizer(CreateOptimizerConfig(optimizer_type="adam", lr=0.1)),
            Repeat(
                20,
                [
                    LossGroup([DagOrderingLoss(), EdgeAttractionLoss(), RepulsionLoss()]),
                    ClipGradNorm(ClipGradNormConfig(max_norm=10.0)),
                    OptimizerStep(),
                    StallCount(StallCountConfig(limit=2, rel_threshold=0.2)),
                    EarlyBreak(predicate=_state_is_converged),
                ],
            ),
        ]
    )

    result = pipeline.apply(problem, SolveState(total_steps=20), _runtime_context(seed=13))

    _assert_valid_positions(result, problem.num_nodes)
    assert result.prev_loss < initial_loss
    assert initial_pos is not None
    assert result.pos is not None
    assert not torch.allclose(result.pos.detach(), initial_pos)
    assert result.converged is True
    assert result.step < 20


def test_spectral_layout_pipeline_returns_two_dimensional_positions() -> None:
    """The spectral pipeline should return a finite two-eigenvector embedding."""

    problem = LayoutProblem(
        edge_index=_edge_index([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]),
        num_nodes=6,
        node_sizes=torch.ones((6, 2), dtype=torch.float32),
        seed=17,
    )
    pipeline = Pipeline([BuildAdjacency(), SpectralInit()])

    result = pipeline.apply(problem, SolveState(), RuntimeContext())

    positions = _assert_valid_positions(result, problem.num_nodes)
    assert result.laplacian is not None
    assert positions.shape[1] == 2
    assert float(positions[:, 0].std().item()) > 0.0
    assert float(positions[:, 1].std().item()) > 0.0


def test_stress_sgd_pipeline_reduces_stress_over_iterations() -> None:
    """The Stress-SGD pipeline should reduce exact stress on a small path graph."""

    problem = _path_problem(5, seed=19, node_size=2.0)
    initial_state = Pipeline(
        [BuildAdjacency(), AllPairsShortestPaths(), RandomUniformInit()]
    ).apply(
        problem,
        SolveState(total_steps=5, extras={"stress_sgd_pair": (3, 4), "stress_sgd_eta": 1.0}),
        _runtime_context(seed=19),
    )
    initial_stress = _exact_stress(problem, initial_state)
    pipeline = Pipeline(
        [
            BuildAdjacency(),
            AllPairsShortestPaths(),
            RandomUniformInit(),
            Repeat(5, [StressSGDPairUpdate()]),
        ]
    )

    result = pipeline.apply(
        problem,
        SolveState(total_steps=5, extras={"stress_sgd_pair": (3, 4), "stress_sgd_eta": 1.0}),
        _runtime_context(seed=19),
    )

    _assert_valid_positions(result, problem.num_nodes)
    assert _exact_stress(problem, result) < initial_stress


def test_multilevel_skeleton_round_trip_produces_valid_positions() -> None:
    """Heavy-edge coarsening followed by direct prolongation should round-trip cleanly."""

    problem = _path_problem(8, seed=1, node_size=1.0)
    coarse_pos = torch.tensor(
        [[-4.0, 0.0], [-2.0, 1.0], [0.0, 0.0], [2.0, -1.0], [4.0, 0.0]],
        dtype=torch.float32,
    )
    pipeline = Pipeline([HeavyEdgeMatching(), DirectMapping(DirectMappingConfig(jitter_scale=0.0))])

    result = pipeline.apply(problem, SolveState(pos=coarse_pos.clone()), _runtime_context(seed=1))

    positions = _assert_valid_positions(result, problem.num_nodes)
    assert result.hierarchy is not None
    assert result.hierarchy
    fine_to_coarse = result.hierarchy[0].fine_to_coarse
    assert fine_to_coarse is not None
    torch.testing.assert_close(positions, coarse_pos[fine_to_coarse])


def test_linlog_pipeline_decreases_loss_and_decays_learning_rate() -> None:
    """The LinLog optimizer pipeline should reduce objective value and decay LR."""

    problem = LayoutProblem(
        edge_index=_edge_index([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]),
        num_nodes=5,
        node_sizes=torch.ones((5, 2), dtype=torch.float32),
        seed=29,
    )
    prefix = Pipeline(
        [
            RandomNormalInit(RandomNormalInitConfig(std=0.1)),
            CreateOptimizer(CreateOptimizerConfig(optimizer_type="adam", lr=0.2)),
        ]
    )
    initial_state = prefix.apply(problem, SolveState(total_steps=10), _runtime_context(seed=29))
    initial_loss = _linlog_total_loss(problem, initial_state)
    pipeline = Pipeline(
        [
            RandomNormalInit(RandomNormalInitConfig(std=0.1)),
            CreateOptimizer(CreateOptimizerConfig(optimizer_type="adam", lr=0.2)),
            Repeat(
                10,
                [
                    LossGroup([LinLogAttractionLoss(), LinLogRepulsionLoss()]),
                    OptimizerStep(),
                    LRDecay(LRDecayConfig(mode="linear", start_lr=0.2, end_lr=0.02)),
                ],
            ),
        ]
    )

    result = pipeline.apply(problem, SolveState(total_steps=10), _runtime_context(seed=29))

    _assert_valid_positions(result, problem.num_nodes)
    assert result.prev_loss < initial_loss
    assert result.optimizer is not None
    assert result.optimizer.param_groups[0]["lr"] == pytest.approx(0.02, rel=1.0e-6)


def test_conditional_branching_selects_small_and_large_pipelines() -> None:
    """Conditional should route to the correct branch for small and large graphs."""

    small_problem = _path_problem(8, seed=31, node_size=1.0)
    large_problem = _path_problem(101, seed=31, node_size=1.0)
    small_pipeline = Pipeline([RandomUniformInit(), _MarkBranch("small")])
    big_pipeline = Pipeline([RandomUniformInit(), _MarkBranch("big")])
    conditional = Conditional(
        predicate=lambda problem, state, ctx: problem.num_nodes > 100,
        op=big_pipeline,
        else_op=small_pipeline,
    )

    small_result = conditional.apply(small_problem, SolveState(), _runtime_context(seed=31))
    large_result = conditional.apply(large_problem, SolveState(), _runtime_context(seed=31))

    _assert_valid_positions(small_result, small_problem.num_nodes)
    _assert_valid_positions(large_result, large_problem.num_nodes)
    assert small_result.extras["branch"] == "small"
    assert large_result.extras["branch"] == "big"


def test_repeat_with_early_break_stops_before_iteration_budget() -> None:
    """Repeat should stop once EarlyBreak observes the convergence predicate."""

    problem = _path_problem(5, seed=37, node_size=1.0)
    repeat = Repeat(
        1000,
        [
            _ConvergingCounterOp(limit=7),
            EarlyBreak(predicate=_repeat_counter_converged),
        ],
    )

    result = repeat.apply(problem, SolveState(), RuntimeContext())

    assert result.converged is True
    assert result.step == 7
    assert result.extras["repeat_iterations"] == 7


def test_loss_group_modes_produce_gradients_and_combined_uses_single_backward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Combined mode should backward once while per-loss mode backprops once per loss."""

    problem = _path_problem(5, seed=41, node_size=1.0)
    losses: List[LossOp] = [
        _QuadraticLoss(scale=1.0, label="loss1"),
        _QuadraticLoss(scale=0.5, label="loss2"),
        _QuadraticLoss(scale=0.25, label="loss3"),
    ]
    call_counter = {"count": 0}
    original_backward = torch.Tensor.backward

    def _counted_backward(self: torch.Tensor, *args: object, **kwargs: object) -> None:
        """Count tensor backward invocations before delegating to PyTorch.

        Parameters
        ----------
        self : torch.Tensor
            Tensor whose gradient is being propagated.
        *args : object
            Positional arguments forwarded to the original method.
        **kwargs : object
            Keyword arguments forwarded to the original method.

        Returns
        -------
        None
            The method delegates to the original tensor backward call.
        """

        call_counter["count"] += 1
        original_backward(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "backward", _counted_backward, raising=False)

    combined_state = SolveState(
        pos=torch.randn((5, 2), dtype=torch.float32, requires_grad=True),
    )
    LossGroup(losses=losses, backward_mode="combined").apply(
        problem,
        combined_state,
        RuntimeContext(),
    )
    combined_count = call_counter["count"]

    call_counter["count"] = 0
    per_loss_state = SolveState(
        pos=torch.randn((5, 2), dtype=torch.float32, requires_grad=True),
    )
    LossGroup(losses=losses, backward_mode="per_loss").apply(
        problem,
        per_loss_state,
        RuntimeContext(),
    )
    per_loss_count = call_counter["count"]

    assert combined_state.pos is not None
    assert combined_state.pos.grad is not None
    assert torch.linalg.norm(combined_state.pos.grad).item() > 0.0
    assert per_loss_state.pos is not None
    assert per_loss_state.pos.grad is not None
    assert torch.linalg.norm(per_loss_state.pos.grad).item() > 0.0
    assert combined_count == 1
    assert per_loss_count == 3


def test_postprocess_chain_centers_scales_and_swaps_axes() -> None:
    """Center, scale, and LR direction-transform should compose predictably."""

    problem = _path_problem(6, seed=43, node_size=1.0)
    base_pipeline = Pipeline(
        [
            RandomUniformInit(),
            CenterPositions(),
            ScalePositions(ScalePositionsConfig(method="max_abs", factor=1.0)),
        ]
    )
    transform_pipeline = Pipeline(
        [
            RandomUniformInit(),
            CenterPositions(),
            ScalePositions(ScalePositionsConfig(method="max_abs", factor=1.0)),
            DirectionTransform(DirectionTransformConfig(direction="LR")),
        ]
    )

    base_result = base_pipeline.apply(problem, SolveState(), _runtime_context(seed=43))
    transformed_result = transform_pipeline.apply(problem, SolveState(), _runtime_context(seed=43))

    assert base_result.pos is not None
    positions = _assert_valid_positions(transformed_result, problem.num_nodes)
    torch.testing.assert_close(positions.mean(dim=0), torch.zeros(2), atol=1.0e-5, rtol=0.0)
    assert float(positions.abs().max().item()) == pytest.approx(1.0, rel=1.0e-6)
    torch.testing.assert_close(positions, base_result.pos[:, [1, 0]])


def test_full_fr_pipeline_converges_on_a_five_node_graph() -> None:
    """A full FR pipeline should reach a sub-threshold final displacement."""

    threshold = 0.03
    problem = LayoutProblem(
        edge_index=_edge_index([(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]),
        num_nodes=5,
        node_sizes=torch.full((5, 2), 2.0, dtype=torch.float32),
        seed=47,
    )
    pipeline = Pipeline(
        [
            RandomUniformInit(),
            Repeat(
                80,
                [
                    ZeroForces(),
                    InverseDistanceRepulsion(),
                    UniformSpringAttraction(),
                    ApplyDisplacement(),
                    LinearCool(),
                    _RecordDisplacement(),
                    DisplacementThreshold(DisplacementThresholdConfig(threshold=threshold)),
                    EarlyBreak(predicate=_state_is_converged),
                ],
            ),
        ]
    )

    result = pipeline.apply(
        problem,
        SolveState(temperature=1.0, total_steps=80, extras={"force_area": 1.0}),
        _runtime_context(seed=47),
    )

    _assert_valid_positions(result, problem.num_nodes)
    history = result.extras["fr_displacement_history"]
    assert history
    assert result.converged is True
    assert history[-1] <= threshold
