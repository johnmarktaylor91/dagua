"""Tests for optimization ops."""

from __future__ import annotations

import math

import pytest
import torch

from dagua.layout.ops import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.base import EarlyBreak, LossGroup, Op, Pipeline, Repeat
from dagua.layout.ops.converge import StallCount, StallCountConfig
from dagua.layout.ops.loss_engine import DagOrderingLoss, RepulsionLoss
from dagua.layout.ops.optimize import (
    ClipGradNorm,
    ClipGradNormConfig,
    ClipGradValue,
    ClipGradValueConfig,
    CreateOptimizer,
    CreateOptimizerConfig,
    LBFGSStep,
    LBFGSStepConfig,
    OptimizerStep,
    OptimizerStepConfig,
    TSNEGainsMomentumStep,
    TSNEGainsMomentumStepConfig,
    UMAPPairSGD,
    UMAPPairSGDConfig,
)


class _RecordLoss(Op):
    """Append the current scalar loss to ``state.extras['loss_history']``."""

    name = "record_loss"

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Store the latest ``prev_loss`` in the state's history buffer.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this helper op.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution context. Unused by this helper op.

        Returns
        -------
        SolveState
            State with an appended loss history entry.
        """
        del problem, ctx

        history = state.extras.setdefault("loss_history", [])
        history.append(float(state.prev_loss))
        return state


def _state_is_converged(
    problem: LayoutProblem,
    state: SolveState,
    ctx: RuntimeContext,
) -> bool:
    """Mirror the solve state's convergence flag for ``EarlyBreak`` tests.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout inputs. Unused by this helper predicate.
    state : SolveState
        Mutable solve state.
    ctx : RuntimeContext
        Execution context. Unused by this helper predicate.

    Returns
    -------
    bool
        ``True`` when the current solve state is already marked converged.
    """
    del problem, ctx
    return state.converged


def _make_problem(num_nodes: int = 3) -> LayoutProblem:
    """Create a minimal layout problem for optimization-op tests.

    Parameters
    ----------
    num_nodes : int, default=3
        Number of graph nodes in the synthetic test problem.

    Returns
    -------
    LayoutProblem
        Minimal problem instance.
    """
    edge_count = max(num_nodes - 1, 0)
    if edge_count == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        sources = torch.arange(0, edge_count, dtype=torch.long)
        targets = torch.arange(1, num_nodes, dtype=torch.long)
        edge_index = torch.stack([sources, targets], dim=0)
    return LayoutProblem(edge_index=edge_index, num_nodes=num_nodes, seed=7)


@pytest.mark.parametrize(
    ("optimizer_type", "expected_type"),
    [
        ("adam", torch.optim.Adam),
        ("sgd", torch.optim.SGD),
        ("sgd_nesterov", torch.optim.SGD),
        ("rmsprop", torch.optim.RMSprop),
    ],
)
def test_create_optimizer_supports_each_requested_type(
    optimizer_type: str,
    expected_type: type[torch.optim.Optimizer],
) -> None:
    """CreateOptimizer should instantiate every supported optimizer type."""

    state = SolveState(pos=torch.zeros((2, 2), dtype=torch.float32))
    op = CreateOptimizer(CreateOptimizerConfig(optimizer_type=optimizer_type, lr=0.05))

    result = op.apply(_make_problem(2), state, RuntimeContext())

    assert isinstance(result.optimizer, expected_type)
    if optimizer_type == "sgd_nesterov":
        assert bool(result.optimizer.defaults["nesterov"]) is True


def test_create_optimizer_creates_adam_with_correct_lr() -> None:
    """CreateOptimizer should attach an Adam optimizer with the requested LR."""

    state = SolveState(pos=torch.zeros((3, 2), dtype=torch.float32))
    op = CreateOptimizer(CreateOptimizerConfig(optimizer_type="adam", lr=0.05))

    result = op.apply(_make_problem(), state, RuntimeContext())

    assert isinstance(result.optimizer, torch.optim.Adam)
    assert result.pos is not None
    assert result.pos.requires_grad
    assert math.isclose(float(result.optimizer.param_groups[0]["lr"]), 0.05, rel_tol=1.0e-6)


def test_create_optimizer_supports_extras_target_and_named_storage() -> None:
    """CreateOptimizer should support extras targets and non-default storage."""

    extra_tensor = torch.ones((2, 2), dtype=torch.float32)
    state = SolveState(extras={"aux": extra_tensor})
    op = CreateOptimizer(
        CreateOptimizerConfig(
            optimizer_type="rmsprop",
            lr=0.01,
            target="extras.aux",
            key="aux",
        )
    )

    result = op.apply(_make_problem(2), state, RuntimeContext())

    assert result.optimizer is None
    assert isinstance(result.extras["optimizer_aux"], torch.optim.RMSprop)
    assert isinstance(result.extras["aux"], torch.Tensor)
    assert result.extras["aux"].requires_grad


@pytest.mark.parametrize("optimizer_type", ["adam", "sgd", "sgd_nesterov", "rmsprop"])
def test_optimizer_step_reduces_quadratic_loss_over_ten_steps(optimizer_type: str) -> None:
    """Repeated optimizer steps should lower a simple quadratic objective."""

    state = SolveState(pos=torch.tensor([[4.0, -3.0]], dtype=torch.float32))
    problem = _make_problem(1)
    CreateOptimizer(CreateOptimizerConfig(optimizer_type=optimizer_type, lr=0.05)).apply(
        problem,
        state,
        RuntimeContext(),
    )

    assert state.pos is not None
    assert state.optimizer is not None
    initial_loss = float(state.pos.square().sum().item())

    for _ in range(10):
        state.optimizer.zero_grad(set_to_none=True)
        loss = state.pos.square().sum()
        loss.backward()
        OptimizerStep().apply(problem, state, RuntimeContext())

    final_loss = float(state.pos.square().sum().item())

    assert final_loss < initial_loss


def test_optimizer_step_actually_updates_pos_given_a_gradient() -> None:
    """OptimizerStep should move positions when gradients are present."""

    state = SolveState(pos=torch.tensor([[1.0, -1.0]], dtype=torch.float32, requires_grad=True))
    create = CreateOptimizer(CreateOptimizerConfig(optimizer_type="sgd", lr=0.1))
    create.apply(_make_problem(1), state, RuntimeContext())
    assert state.pos is not None
    state.pos.grad = torch.tensor([[2.0, -4.0]], dtype=torch.float32)
    before = state.pos.detach().clone()

    result = OptimizerStep(OptimizerStepConfig()).apply(_make_problem(1), state, RuntimeContext())

    assert result.pos is not None
    assert not torch.allclose(result.pos.detach(), before)
    assert torch.allclose(result.pos.detach(), before - (0.1 * torch.tensor([[2.0, -4.0]])))


def test_clip_grad_norm_clips_large_gradients() -> None:
    """ClipGradNorm should enforce the requested total gradient norm."""

    pos = torch.zeros((2, 2), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.SGD([pos], lr=0.1)
    pos.grad = torch.full_like(pos, 10.0)
    state = SolveState(pos=pos, optimizer=optimizer)

    ClipGradNorm(ClipGradNormConfig(max_norm=1.5)).apply(_make_problem(2), state, RuntimeContext())

    assert pos.grad is not None
    assert float(torch.linalg.norm(pos.grad).item()) <= 1.5 * (1.0 + 1.0e-5)


def test_create_optimizer_and_step_support_multiple_named_optimizers() -> None:
    """Multiple optimizers should update their own targets independently."""

    state = SolveState(
        pos=torch.tensor([[2.0, -1.0]], dtype=torch.float32),
        extras={"aux": torch.tensor([[1.5, -0.5]], dtype=torch.float32)},
    )
    problem = _make_problem(1)

    CreateOptimizer(CreateOptimizerConfig(optimizer_type="adam", lr=0.05)).apply(
        problem,
        state,
        RuntimeContext(),
    )
    CreateOptimizer(
        CreateOptimizerConfig(
            optimizer_type="sgd",
            lr=0.1,
            target="extras.aux",
            key="aux",
        )
    ).apply(problem, state, RuntimeContext())

    assert state.pos is not None
    assert state.optimizer is not None
    assert isinstance(state.extras["optimizer_aux"], torch.optim.SGD)
    aux_tensor = state.extras["aux"]
    assert isinstance(aux_tensor, torch.Tensor)

    state.optimizer.zero_grad(set_to_none=True)
    aux_optimizer = state.extras["optimizer_aux"]
    assert isinstance(aux_optimizer, torch.optim.Optimizer)
    aux_optimizer.zero_grad(set_to_none=True)

    pos_before = state.pos.detach().clone()
    aux_before = aux_tensor.detach().clone()
    pos_loss = state.pos.square().sum()
    aux_loss = (aux_tensor - 2.0).square().sum()
    pos_loss.backward()
    aux_loss.backward()

    OptimizerStep().apply(problem, state, RuntimeContext())
    OptimizerStep(OptimizerStepConfig(key="aux")).apply(problem, state, RuntimeContext())

    aux_after = state.extras["aux"]
    assert isinstance(aux_after, torch.Tensor)
    assert not torch.allclose(state.pos.detach(), pos_before)
    assert not torch.allclose(aux_after.detach(), aux_before)


def test_clip_grad_value_clamps_gradient_entries() -> None:
    """ClipGradValue should clamp each gradient component symmetrically."""

    pos = torch.zeros((1, 2), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.SGD([pos], lr=0.1)
    pos.grad = torch.tensor([[10.0, -10.0]], dtype=torch.float32)
    state = SolveState(pos=pos, optimizer=optimizer)

    ClipGradValue(ClipGradValueConfig(max_value=2.5)).apply(
        _make_problem(1),
        state,
        RuntimeContext(),
    )

    assert pos.grad is not None
    assert torch.allclose(pos.grad, torch.tensor([[2.5, -2.5]], dtype=torch.float32))


def test_lbfgs_step_runs_kk_solver_and_updates_loss() -> None:
    """LBFGSStep should run the SciPy KK solve and record the objective value."""

    pos = torch.tensor([[0.0, 0.0], [0.1, 0.0]], dtype=torch.float32, requires_grad=True)
    state = SolveState(
        pos=pos,
        distance_matrix=torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32),
    )

    result = LBFGSStep(LBFGSStepConfig(maxiter=5)).apply(
        _make_problem(2),
        state,
        RuntimeContext(),
    )

    assert result.pos is not None
    assert result.pos.shape == (2, 2)
    assert math.isfinite(result.prev_loss)


def test_lbfgs_step_reduces_two_node_distance_error() -> None:
    """LBFGSStep should move a simple two-node layout toward its target distance."""

    pos = torch.tensor([[0.0, 0.0], [0.1, 0.0]], dtype=torch.float32, requires_grad=True)
    state = SolveState(
        pos=pos,
        distance_matrix=torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32),
    )
    before_error = abs(float(torch.linalg.norm(pos[0] - pos[1]).item()) - 1.0)

    result = LBFGSStep(LBFGSStepConfig(maxiter=20)).apply(
        _make_problem(2),
        state,
        RuntimeContext(),
    )

    assert result.pos is not None
    after_error = abs(float(torch.linalg.norm(result.pos[0] - result.pos[1]).item()) - 1.0)
    assert after_error < before_error


def test_tsne_gains_momentum_step_updates_positions_and_state() -> None:
    """TSNEGainsMomentumStep should initialize and persist gains and momentum."""

    pos = torch.tensor([[0.1, -0.2], [0.3, 0.4]], dtype=torch.float32, requires_grad=True)
    pos.grad = torch.tensor([[1.0, -2.0], [-0.5, 0.25]], dtype=torch.float32)
    state = SolveState(pos=pos, step=0)
    before = pos.detach().clone()

    result = TSNEGainsMomentumStep(TSNEGainsMomentumStepConfig()).apply(
        _make_problem(2),
        state,
        RuntimeContext(),
    )

    assert result.pos is not None
    assert not torch.allclose(result.pos.detach(), before)
    assert "tsne_update" in result.extras
    assert "tsne_gains" in result.extras
    assert result.pos.grad is not None
    assert torch.allclose(result.pos.grad, torch.zeros_like(result.pos.grad))


def test_umap_pair_sgd_updates_positions_with_precomputed_pairs() -> None:
    """UMAPPairSGD should update positions and sampling counters."""

    pos = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    state = SolveState(
        pos=pos,
        step=0,
        total_steps=10,
        extras={
            "umap_head": torch.tensor([0], dtype=torch.long),
            "umap_tail": torch.tensor([1], dtype=torch.long),
            "umap_epochs_per_sample": torch.tensor([1.0], dtype=torch.float32),
            "umap_a": 1.0,
            "umap_b": 1.0,
            "umap_seed": 3,
        },
    )
    before = pos.detach().clone()

    result = UMAPPairSGD(UMAPPairSGDConfig()).apply(_make_problem(3), state, RuntimeContext())

    assert result.pos is not None
    assert not torch.allclose(result.pos.detach(), before)
    assert "umap_next_sample_epoch" in result.extras
    assert "umap_next_negative_epoch" in result.extras


def test_full_optimization_loop_reduces_engine_loss_over_iterations() -> None:
    """The requested composed engine loop should decrease its recorded loss."""

    problem = LayoutProblem(
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        num_nodes=2,
        node_sizes=torch.full((2, 2), 10.0, dtype=torch.float32),
    )
    state = SolveState(
        pos=torch.tensor([[0.0, 40.0], [0.0, 0.0]], dtype=torch.float32),
        extras={},
    )
    pipeline = Pipeline(
        [
            CreateOptimizer(CreateOptimizerConfig(optimizer_type="adam", lr=0.1)),
            Repeat(
                20,
                [
                    LossGroup([DagOrderingLoss(), RepulsionLoss()]),
                    _RecordLoss(),
                    ClipGradNorm(ClipGradNormConfig(max_norm=10.0)),
                    OptimizerStep(),
                    StallCount(StallCountConfig(limit=50, rel_threshold=1.0e-8)),
                ],
            ),
        ]
    )

    result = pipeline.apply(problem, state, RuntimeContext())

    history = result.extras["loss_history"]
    assert isinstance(history, list)
    assert len(history) == 20
    assert history[-1] < history[0]


def test_full_optimization_loop_can_break_early_from_stall_count() -> None:
    """EarlyBreak should stop the requested loop once StallCount reports convergence."""

    problem = LayoutProblem(
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        num_nodes=2,
        node_sizes=torch.full((2, 2), 10.0, dtype=torch.float32),
    )
    state = SolveState(
        pos=torch.tensor([[0.0, 40.0], [0.0, 0.0]], dtype=torch.float32),
        extras={},
    )
    pipeline = Pipeline(
        [
            CreateOptimizer(CreateOptimizerConfig(optimizer_type="adam", lr=0.1)),
            Repeat(
                20,
                [
                    LossGroup([DagOrderingLoss(), RepulsionLoss()]),
                    _RecordLoss(),
                    ClipGradNorm(ClipGradNormConfig(max_norm=10.0)),
                    OptimizerStep(),
                    StallCount(StallCountConfig(limit=1, rel_threshold=1.0)),
                    EarlyBreak(predicate=_state_is_converged),
                ],
            ),
        ]
    )

    result = pipeline.apply(problem, state, RuntimeContext())

    history = result.extras["loss_history"]
    assert isinstance(history, list)
    assert result.converged is True
    assert result.step < 20
    assert len(history) == result.step
