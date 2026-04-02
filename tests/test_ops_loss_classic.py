"""Tests for classic loss ops in ``dagua.layout.ops``."""

from __future__ import annotations

from typing import Tuple

import pytest
import torch

from dagua.layout.ops import (
    DavidsonHarelEnergyLoss,
    DavidsonHarelEnergyLossConfig,
    ExactPairStressLoss,
    KLDivergenceLoss,
    KLDivergenceLossConfig,
    LayoutProblem,
    LinLogAttractionLoss,
    LinLogRepulsionLoss,
    LossGroup,
    LossOp,
    RuntimeContext,
    SolveState,
)


def _path_graph_problem(num_nodes: int = 5) -> LayoutProblem:
    """Build a simple path-graph layout problem.

    Parameters
    ----------
    num_nodes : int, default=5
        Number of nodes in the path.

    Returns
    -------
    LayoutProblem
        Path-graph layout problem.
    """
    source = torch.arange(0, num_nodes - 1, dtype=torch.long)
    target = torch.arange(1, num_nodes, dtype=torch.long)
    edge_index = torch.stack([source, target], dim=0)
    return LayoutProblem(edge_index=edge_index, num_nodes=num_nodes)


def _path_distance_matrix(num_nodes: int = 5) -> torch.Tensor:
    """Build the exact shortest-path matrix for a path graph.

    Parameters
    ----------
    num_nodes : int, default=5
        Number of nodes in the path.

    Returns
    -------
    torch.Tensor
        Distance matrix with shape ``[N, N]``.
    """
    index = torch.arange(num_nodes, dtype=torch.float32)
    return (index[:, None] - index[None, :]).abs()


def _classic_loss_case(loss_name: str) -> Tuple[LossOp, LayoutProblem, SolveState]:
    """Build one classic loss scenario for scalar and gradient assertions.

    Parameters
    ----------
    loss_name : str
        Registered classic loss name under test.

    Returns
    -------
    tuple[LossOp, LayoutProblem, SolveState]
        Loss op plus a fresh problem/state pair.

    Raises
    ------
    ValueError
        If the loss name is unsupported by this test module.
    """
    if loss_name == ExactPairStressLoss.name:
        num_nodes = 4
        problem = _path_graph_problem(num_nodes)
        positions = torch.tensor(
            [
                [0.0, 0.0],
                [1.2, 0.0],
                [2.6, 0.0],
                [3.8, 0.0],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )
        state = SolveState(pos=positions, distance_matrix=_path_distance_matrix(num_nodes))
        return ExactPairStressLoss(), problem, state

    if loss_name == KLDivergenceLoss.name:
        problem = LayoutProblem(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=3)
        positions = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 2.0],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )
        affinity = torch.tensor(
            [
                [1.0e-12, 0.20, 0.05],
                [0.20, 1.0e-12, 0.25],
                [0.05, 0.25, 1.0e-12],
            ],
            dtype=torch.float32,
        )
        affinity = affinity / affinity.sum()
        state = SolveState(pos=positions, affinity_matrix=affinity, step=500)
        return KLDivergenceLoss(KLDivergenceLossConfig(exaggeration_steps=0)), problem, state

    if loss_name == LinLogAttractionLoss.name:
        problem = LayoutProblem(
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
            num_nodes=4,
        )
        positions = torch.tensor(
            [
                [-1.2, 0.0],
                [-0.1, 0.7],
                [0.8, -0.4],
                [1.3, 0.6],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )
        state = SolveState(pos=positions)
        return LinLogAttractionLoss(), problem, state

    if loss_name == LinLogRepulsionLoss.name:
        problem = LayoutProblem(
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
            num_nodes=4,
        )
        positions = torch.tensor(
            [
                [-1.0, 0.0],
                [-0.2, 0.6],
                [0.5, -0.3],
                [1.1, 0.5],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )
        state = SolveState(pos=positions)
        return LinLogRepulsionLoss(), problem, state

    if loss_name == DavidsonHarelEnergyLoss.name:
        problem = LayoutProblem(
            edge_index=torch.tensor([[0, 0, 1, 3], [1, 2, 3, 4]], dtype=torch.long),
            num_nodes=5,
        )
        positions = torch.tensor(
            [
                [0.0, 0.0],
                [0.4, 0.0],
                [1.2, 0.1],
                [0.7, 0.05],
                [0.8, 0.8],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )
        state = SolveState(pos=positions)
        return DavidsonHarelEnergyLoss(), problem, state

    raise ValueError(f"Unsupported classic loss {loss_name!r}.")


@pytest.mark.parametrize(
    "loss_name",
    [
        ExactPairStressLoss.name,
        KLDivergenceLoss.name,
        LinLogAttractionLoss.name,
        LinLogRepulsionLoss.name,
        DavidsonHarelEnergyLoss.name,
    ],
)
def test_classic_losses_return_scalar_tensors_with_gradients(loss_name: str) -> None:
    """Each selected classic loss should return a differentiable scalar."""

    loss_op, problem, state = _classic_loss_case(loss_name)

    result = loss_op.evaluate(problem, state, RuntimeContext())

    assert result.shape == ()
    assert result.requires_grad


def test_exact_pair_stress_loss_matches_known_path_value() -> None:
    """Exact stress should match the analytic value on a stretched path."""

    problem = _path_graph_problem()
    positions = torch.tensor(
        [
            [0.0, 0.0],
            [1.5, 0.0],
            [3.0, 0.0],
            [4.5, 0.0],
            [6.0, 0.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    state = SolveState(pos=positions, distance_matrix=_path_distance_matrix())

    loss = ExactPairStressLoss().evaluate(problem, state, RuntimeContext())

    assert torch.isclose(loss, torch.tensor(2.5, dtype=torch.float32))


def test_exact_pair_stress_loss_is_near_zero_for_matching_distances() -> None:
    """Exact stress should vanish when Euclidean and graph distances match."""

    problem = _path_graph_problem(4)
    positions = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    state = SolveState(pos=positions, distance_matrix=_path_distance_matrix(4))

    loss = ExactPairStressLoss().evaluate(problem, state, RuntimeContext())

    assert loss.item() <= 1.0e-6


def test_exact_pair_stress_loss_is_positive_for_distance_mismatch() -> None:
    """Exact stress should increase once the embedding stretches graph distances."""

    problem = _path_graph_problem(4)
    positions = torch.tensor(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [4.0, 0.0],
            [8.0, 0.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    state = SolveState(pos=positions, distance_matrix=_path_distance_matrix(4))

    loss = ExactPairStressLoss().evaluate(problem, state, RuntimeContext())

    assert loss.item() > 0.0


def test_kl_divergence_loss_returns_scalar_with_gradient() -> None:
    """The t-SNE KL op should produce a scalar that backpropagates."""

    problem = LayoutProblem(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=4)
    positions = torch.tensor(
        [
            [-1.0, 0.5],
            [-0.2, -0.4],
            [0.8, 0.1],
            [0.4, -0.9],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    affinity = torch.tensor(
        [
            [1.0e-12, 0.08, 0.04, 0.03],
            [0.08, 1.0e-12, 0.05, 0.02],
            [0.04, 0.05, 1.0e-12, 0.06],
            [0.03, 0.02, 0.06, 1.0e-12],
        ],
        dtype=torch.float32,
    )
    affinity = affinity / affinity.sum()
    state = SolveState(pos=positions, affinity_matrix=affinity)

    loss = KLDivergenceLoss().evaluate(problem, state, RuntimeContext())
    loss.backward()

    assert loss.ndim == 0
    assert positions.grad is not None
    assert torch.isfinite(positions.grad).all()


def test_kl_divergence_loss_matches_manual_probability_matrix() -> None:
    """KL divergence should match the exact Student-t normalization formula."""

    problem = LayoutProblem(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=3)
    positions = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 2.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    affinity = torch.tensor(
        [
            [1.0e-12, 0.20, 0.05],
            [0.20, 1.0e-12, 0.25],
            [0.05, 0.25, 1.0e-12],
        ],
        dtype=torch.float32,
    )
    affinity = affinity / affinity.sum()
    state = SolveState(pos=positions, affinity_matrix=affinity, step=500)

    loss = KLDivergenceLoss(KLDivergenceLossConfig(exaggeration_steps=0)).evaluate(
        problem,
        state,
        RuntimeContext(),
    )

    pairwise_sq_dist = (
        (positions.detach()[:, None, :] - positions.detach()[None, :, :]).square().sum(dim=2)
    )
    q_numerators = (1.0 + pairwise_sq_dist).reciprocal()
    q_numerators.fill_diagonal_(0.0)
    q_matrix = q_numerators / q_numerators.sum()
    expected = (
        affinity * (affinity.clamp_min(1.0e-12).log() - q_matrix.clamp_min(1.0e-12).log())
    ).sum()

    assert torch.isclose(loss, expected, atol=1.0e-6)


@pytest.mark.parametrize(
    ("weight_name", "minimum"),
    [
        ("w_distribution", 0.1),
        ("w_border", 1.0e-4),
        ("w_edge_length", 0.1),
        ("w_crossing", 1.0e-4),
        ("w_node_edge", 1.0),
    ],
)
def test_davidson_harel_energy_all_five_terms_contribute(
    weight_name: str,
    minimum: float,
) -> None:
    """Each Davidson-Harel term should produce a positive energy on a bad layout."""

    problem = LayoutProblem(
        edge_index=torch.tensor([[0, 0, 1, 3], [1, 2, 3, 4]], dtype=torch.long),
        num_nodes=5,
    )
    positions = torch.tensor(
        [
            [0.0, 0.0],
            [0.4, 0.0],
            [1.2, 0.1],
            [0.7, 0.05],
            [0.8, 0.8],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    config_kwargs = {
        "w_distribution": 0.0,
        "w_border": 0.0,
        "w_edge_length": 0.0,
        "w_crossing": 0.0,
        "w_node_edge": 0.0,
    }
    config_kwargs[weight_name] = 1.0
    state = SolveState(pos=positions)

    loss = DavidsonHarelEnergyLoss(DavidsonHarelEnergyLossConfig(**config_kwargs)).evaluate(
        problem,
        state,
        RuntimeContext(),
    )

    assert loss.item() > minimum


def test_linlog_losses_work_inside_loss_group() -> None:
    """LinLog attraction and repulsion should compose inside ``LossGroup``."""

    problem = LayoutProblem(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        num_nodes=4,
    )
    positions = torch.tensor(
        [
            [-1.2, 0.0],
            [-0.1, 0.7],
            [0.8, -0.4],
            [1.3, 0.6],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    state = SolveState(pos=positions)
    group = LossGroup([LinLogAttractionLoss(), LinLogRepulsionLoss()])

    group.apply(problem, state, RuntimeContext())

    assert state.prev_loss != 0.0
    assert positions.grad is not None
    assert torch.isfinite(positions.grad).all()


def test_davidson_harel_energy_loss_is_nonzero_for_random_positions() -> None:
    """Davidson-Harel energy should be positive on a non-degenerate layout."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(7)
    problem = LayoutProblem(
        edge_index=torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]], dtype=torch.long),
        num_nodes=4,
    )
    positions = torch.randn((4, 2), generator=generator, dtype=torch.float32)
    state = SolveState(pos=positions)

    loss = DavidsonHarelEnergyLoss().evaluate(problem, state, RuntimeContext())

    assert loss.item() > 0.0
