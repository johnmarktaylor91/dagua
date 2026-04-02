"""Tests for classic loss ops in ``dagua.layout.ops``."""

from __future__ import annotations

from typing import Tuple

import pytest
import torch

from dagua.layout.ops import (
    CyclicSampler,
    CyclicSamplerConfig,
    DavidsonHarelEnergyLoss,
    DavidsonHarelEnergyLossConfig,
    ElasticLoss,
    EntropyLoss,
    EntropyLossConfig,
    ExactPairStressLoss,
    ExactPairStressLossConfig,
    KDTreeRepulsionLoss,
    KDTreeRepulsionLossConfig,
    KLDivergenceLoss,
    KLDivergenceLossConfig,
    LayoutProblem,
    LinLogAttractionLoss,
    LinLogAttractionLossConfig,
    LinLogRepulsionLoss,
    LinLogRepulsionLossConfig,
    LossGroup,
    LossOp,
    PivotApproxStressLoss,
    RuntimeContext,
    SGD2CriterionLoss,
    SGD2CriterionLossConfig,
    SGD2CrossingDetectorStep,
    SGD2CrossingDetectorStepConfig,
    SolveState,
    UMAPCrossEntropyLoss,
    UMAPCrossEntropyLossConfig,
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


def _make_generator(seed: int) -> torch.Generator:
    """Build a deterministic CPU generator for sampled classic losses.

    Parameters
    ----------
    seed : int
        Random seed value.

    Returns
    -------
    torch.Generator
        Seeded generator on CPU.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _sgd2_stress_mean_for_path(positions: torch.Tensor, distance_matrix: torch.Tensor) -> float:
    """Compute the exact mean stress batch used by the SGD2 stress criterion.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    distance_matrix : torch.Tensor
        Graph distance matrix with shape ``[N, N]``.

    Returns
    -------
    float
        Mean weighted stress over all finite upper-triangle pairs.
    """
    upper = torch.triu_indices(distance_matrix.shape[0], distance_matrix.shape[1], offset=1)
    targets = distance_matrix[upper[0], upper[1]]
    mask = torch.isfinite(targets) & (targets > 0)
    pair_index = upper[:, mask]
    pair_targets = targets[mask]
    lengths = torch.linalg.norm(positions[pair_index[0]] - positions[pair_index[1]], dim=1)
    weights = pair_targets.reciprocal().square()
    return float((weights * (lengths - pair_targets).square()).mean().item())


def test_exact_pair_stress_loss_weight_fn_config_changes_weighting() -> None:
    """Stress weight functions should produce distinct losses for the same stretched path."""

    problem = _path_graph_problem(4)
    positions = torch.tensor(
        [
            [0.0, 0.0],
            [1.5, 0.0],
            [3.0, 0.0],
            [4.5, 0.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    state = SolveState(pos=positions, distance_matrix=_path_distance_matrix(4))
    inverse_sq_loss = ExactPairStressLoss(
        ExactPairStressLossConfig(weight_fn="inverse_sq"),
    ).evaluate(problem, state, RuntimeContext())
    inverse_loss = ExactPairStressLoss(ExactPairStressLossConfig(weight_fn="inverse")).evaluate(
        problem,
        SolveState(pos=positions, distance_matrix=_path_distance_matrix(4)),
        RuntimeContext(),
    )
    uniform_loss = ExactPairStressLoss(ExactPairStressLossConfig(weight_fn="uniform")).evaluate(
        problem,
        SolveState(pos=positions, distance_matrix=_path_distance_matrix(4)),
        RuntimeContext(),
    )

    assert inverse_sq_loss.item() < inverse_loss.item() < uniform_loss.item()


def test_pivot_approx_stress_loss_is_zero_for_exact_embedding() -> None:
    """Pivot stress should vanish when pivot distances exactly match Euclidean distances."""

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
    state = SolveState(
        pos=positions,
        pivot_indices=torch.arange(4, dtype=torch.long),
        pivot_distances=_path_distance_matrix(4),
    )

    loss = PivotApproxStressLoss().evaluate(problem, state, RuntimeContext())

    assert loss.item() <= 1.0e-6


def test_pivot_approx_stress_loss_matches_twice_exact_stress_with_all_nodes_as_pivots() -> None:
    """Using every node as a pivot should yield the ordered-pair form of exact stress."""

    problem = _path_graph_problem(4)
    positions = torch.tensor(
        [
            [0.0, 0.0],
            [1.3, 0.0],
            [2.7, 0.0],
            [4.2, 0.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    distance_matrix = _path_distance_matrix(4)
    exact_loss = ExactPairStressLoss().evaluate(
        problem,
        SolveState(pos=positions, distance_matrix=distance_matrix),
        RuntimeContext(),
    )
    pivot_loss = PivotApproxStressLoss().evaluate(
        problem,
        SolveState(
            pos=positions,
            pivot_indices=torch.arange(4, dtype=torch.long),
            pivot_distances=distance_matrix,
        ),
        RuntimeContext(),
    )

    assert pivot_loss.item() == pytest.approx(2.0 * exact_loss.item(), rel=1.0e-6)


def test_pivot_approx_stress_loss_is_positive_for_distorted_embedding() -> None:
    """Pivot stress should be nonzero once the embedding departs from graph distances."""

    problem = _path_graph_problem(4)
    state = SolveState(
        pos=torch.tensor(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [4.0, 0.0],
                [8.0, 0.0],
            ],
            dtype=torch.float32,
            requires_grad=True,
        ),
        pivot_indices=torch.arange(4, dtype=torch.long),
        pivot_distances=_path_distance_matrix(4),
    )

    loss = PivotApproxStressLoss().evaluate(problem, state, RuntimeContext())

    assert loss.item() > 0.0


def test_kl_divergence_loss_early_exaggeration_multiplier_changes_loss() -> None:
    """Early exaggeration should change the KL value before the configured step cutoff."""

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
    early_loss = KLDivergenceLoss(
        KLDivergenceLossConfig(exaggeration=4.0, exaggeration_steps=10),
    ).evaluate(
        problem,
        SolveState(pos=positions, affinity_matrix=affinity, step=0),
        RuntimeContext(),
    )
    late_loss = KLDivergenceLoss(
        KLDivergenceLossConfig(exaggeration=4.0, exaggeration_steps=10),
    ).evaluate(
        problem,
        SolveState(pos=positions, affinity_matrix=affinity, step=50),
        RuntimeContext(),
    )

    assert early_loss.item() != pytest.approx(late_loss.item(), rel=1.0e-6)


def test_umap_cross_entropy_loss_neg_rate_increases_repulsion_term() -> None:
    """Increasing the negative-sample rate should increase UMAP cross entropy for the same seed."""

    problem = _path_graph_problem(4)
    pos = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    low_neg_loss = UMAPCrossEntropyLoss(
        UMAPCrossEntropyLossConfig(neg_rate=1, repulsion_strength=1.0),
    ).evaluate(
        problem,
        SolveState(pos=pos),
        RuntimeContext(generator=_make_generator(57)),
    )
    high_neg_loss = UMAPCrossEntropyLoss(
        UMAPCrossEntropyLossConfig(neg_rate=4, repulsion_strength=1.0),
    ).evaluate(
        problem,
        SolveState(pos=pos),
        RuntimeContext(generator=_make_generator(57)),
    )

    assert high_neg_loss.item() > low_neg_loss.item()


def test_umap_cross_entropy_loss_repulsion_strength_scales_negative_term() -> None:
    """Repulsion strength should scale the negative-sample contribution."""

    problem = _path_graph_problem(4)
    pos = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    no_repulsion_loss = UMAPCrossEntropyLoss(
        UMAPCrossEntropyLossConfig(neg_rate=4, repulsion_strength=0.0),
    ).evaluate(
        problem,
        SolveState(pos=pos),
        RuntimeContext(generator=_make_generator(57)),
    )
    strong_repulsion_loss = UMAPCrossEntropyLoss(
        UMAPCrossEntropyLossConfig(neg_rate=4, repulsion_strength=2.0),
    ).evaluate(
        problem,
        SolveState(pos=pos),
        RuntimeContext(generator=_make_generator(57)),
    )

    assert strong_repulsion_loss.item() > no_repulsion_loss.item()


def test_umap_cross_entropy_loss_backpropagates() -> None:
    """UMAP cross entropy should produce finite gradients with a deterministic sampler seed."""

    problem = _path_graph_problem(4)
    positions = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    state = SolveState(pos=positions)

    loss = UMAPCrossEntropyLoss(
        UMAPCrossEntropyLossConfig(neg_rate=4, repulsion_strength=1.0),
    ).evaluate(problem, state, RuntimeContext(generator=_make_generator(57)))
    loss.backward()

    assert positions.grad is not None
    assert torch.isfinite(positions.grad).all()


def test_linlog_attraction_loss_is_zero_without_edges() -> None:
    """LinLog attraction should vanish on an edgeless graph."""

    problem = LayoutProblem(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=2)

    loss = LinLogAttractionLoss().evaluate(
        problem,
        SolveState(
            pos=torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32, requires_grad=True),
        ),
        RuntimeContext(),
    )

    assert loss.item() == pytest.approx(0.0, abs=1.0e-6)


def test_linlog_attraction_loss_is_positive_for_connected_nodes() -> None:
    """LinLog attraction should be positive when connected endpoints are separated."""

    problem = _path_graph_problem(4)
    loss = LinLogAttractionLoss().evaluate(
        problem,
        SolveState(
            pos=torch.tensor(
                [
                    [0.0, 0.0],
                    [1.5, 0.0],
                    [3.0, 0.0],
                    [4.5, 0.0],
                ],
                dtype=torch.float32,
                requires_grad=True,
            ),
        ),
        RuntimeContext(),
    )

    assert loss.item() > 0.0


def test_linlog_attraction_loss_exponent_changes_result() -> None:
    """The attraction exponent should change the objective value."""

    problem = _path_graph_problem(4)
    pos = torch.tensor(
        [
            [0.0, 0.0],
            [1.5, 0.0],
            [3.0, 0.0],
            [4.5, 0.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    exponent_one_loss = LinLogAttractionLoss(
        LinLogAttractionLossConfig(exponent_a=1.0),
    ).evaluate(problem, SolveState(pos=pos), RuntimeContext())
    exponent_two_loss = LinLogAttractionLoss(
        LinLogAttractionLossConfig(exponent_a=2.0),
    ).evaluate(problem, SolveState(pos=pos), RuntimeContext())

    assert exponent_two_loss.item() > exponent_one_loss.item()


def test_linlog_repulsion_loss_is_zero_for_single_node() -> None:
    """LinLog repulsion should vanish when there are no node pairs."""

    problem = LayoutProblem(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=1)

    loss = LinLogRepulsionLoss().evaluate(
        problem,
        SolveState(pos=torch.tensor([[0.0, 0.0]], dtype=torch.float32, requires_grad=True)),
        RuntimeContext(),
    )

    assert loss.item() == pytest.approx(0.0, abs=1.0e-6)


def test_linlog_repulsion_loss_matches_log_form_at_zero_exponent() -> None:
    """Zero repulsion exponent should recover the logarithmic LinLog form."""

    problem = _path_graph_problem(4)
    positions = torch.tensor(
        [
            [0.0, 0.0],
            [1.5, 0.0],
            [3.0, 0.0],
            [4.5, 0.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    loss = LinLogRepulsionLoss(LinLogRepulsionLossConfig(exponent_r=0.0)).evaluate(
        problem,
        SolveState(pos=positions),
        RuntimeContext(),
    )
    expected = 0.0
    for src in range(4):
        for dst in range(src + 1, 4):
            expected -= torch.log(torch.tensor(float(dst - src) * 1.5)).item()

    assert loss.item() == pytest.approx(expected, rel=1.0e-6)


def test_linlog_repulsion_loss_exponent_changes_result() -> None:
    """Changing the repulsion exponent should change the all-pairs repulsion energy."""

    problem = _path_graph_problem(4)
    positions = torch.tensor(
        [
            [0.0, 0.0],
            [1.5, 0.0],
            [3.0, 0.0],
            [4.5, 0.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    exponent_zero_loss = LinLogRepulsionLoss(
        LinLogRepulsionLossConfig(exponent_r=0.0),
    ).evaluate(problem, SolveState(pos=positions), RuntimeContext())
    exponent_one_loss = LinLogRepulsionLoss(
        LinLogRepulsionLossConfig(exponent_r=1.0),
    ).evaluate(problem, SolveState(pos=positions), RuntimeContext())

    assert exponent_one_loss.item() != pytest.approx(exponent_zero_loss.item(), rel=1.0e-6)


def test_entropy_loss_is_nonzero_for_non_edge_pairs() -> None:
    """Entropy loss should account for exact non-edge pairs."""

    problem = LayoutProblem(
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        num_nodes=3,
    )

    loss = EntropyLoss().evaluate(
        problem,
        SolveState(
            pos=torch.tensor(
                [[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]], dtype=torch.float32, requires_grad=True
            ),
        ),
        RuntimeContext(),
    )

    assert loss.item() != pytest.approx(0.0, abs=1.0e-6)


def test_entropy_loss_alpha_scales_result() -> None:
    """Entropy alpha should scale the exact non-edge term linearly."""

    problem = LayoutProblem(
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        num_nodes=3,
    )
    positions = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]], dtype=torch.float32, requires_grad=True
    )
    base_loss = EntropyLoss(EntropyLossConfig(alpha=1.0)).evaluate(
        problem,
        SolveState(pos=positions),
        RuntimeContext(),
    )
    doubled_loss = EntropyLoss(EntropyLossConfig(alpha=2.0)).evaluate(
        problem,
        SolveState(pos=positions),
        RuntimeContext(),
    )

    assert doubled_loss.item() == pytest.approx(2.0 * base_loss.item(), rel=1.0e-6)


def test_entropy_loss_is_zero_for_complete_graph() -> None:
    """Entropy loss should vanish when there are no non-edge pairs."""

    problem = LayoutProblem(
        edge_index=torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long),
        num_nodes=3,
    )

    loss = EntropyLoss().evaluate(
        problem,
        SolveState(
            pos=torch.tensor(
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=torch.float32, requires_grad=True
            ),
        ),
        RuntimeContext(),
    )

    assert loss.item() == pytest.approx(0.0, abs=1.0e-6)


def test_davidson_harel_energy_zeroing_one_weight_removes_that_term() -> None:
    """Removing one Davidson-Harel term weight should lower the total energy."""

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
    )
    full_loss = DavidsonHarelEnergyLoss().evaluate(
        problem, SolveState(pos=positions), RuntimeContext()
    )
    no_crossing_loss = DavidsonHarelEnergyLoss(
        DavidsonHarelEnergyLossConfig(w_crossing=0.0),
    ).evaluate(problem, SolveState(pos=positions), RuntimeContext())

    assert full_loss.item() > no_crossing_loss.item()


def test_elastic_loss_is_positive_for_connected_nodes() -> None:
    """Elastic loss should be positive when edges connect distinct positions."""

    problem = LayoutProblem(
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        num_nodes=3,
    )

    loss = ElasticLoss().evaluate(
        problem,
        SolveState(
            pos=torch.tensor(
                [[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]], dtype=torch.float32, requires_grad=True
            ),
        ),
        RuntimeContext(),
    )

    assert loss.item() > 0.0


def test_elastic_loss_is_zero_without_edges() -> None:
    """Elastic loss should vanish on an edgeless graph."""

    problem = LayoutProblem(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=3)

    loss = ElasticLoss().evaluate(
        problem,
        SolveState(
            pos=torch.tensor(
                [[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]], dtype=torch.float32, requires_grad=True
            ),
        ),
        RuntimeContext(),
    )

    assert loss.item() == pytest.approx(0.0, abs=1.0e-6)


def test_elastic_loss_collapses_directed_duplicate_edges() -> None:
    """Elastic loss should treat opposite directed duplicates as one undirected spring."""

    unique_problem = LayoutProblem(
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        num_nodes=3,
    )
    duplicated_problem = LayoutProblem(
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
        num_nodes=3,
    )
    positions = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]], dtype=torch.float32, requires_grad=True
    )
    unique_loss = ElasticLoss().evaluate(
        unique_problem,
        SolveState(pos=positions),
        RuntimeContext(),
    )
    duplicated_loss = ElasticLoss().evaluate(
        duplicated_problem,
        SolveState(pos=positions),
        RuntimeContext(),
    )

    assert duplicated_loss.item() == pytest.approx(unique_loss.item(), rel=1.0e-6)


def test_kdtree_repulsion_loss_penalizes_close_nodes_more_than_far_nodes() -> None:
    """KD-tree repulsion should drop toward zero as all nearby pairs disappear."""

    problem = LayoutProblem(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=3)
    close_loss = KDTreeRepulsionLoss().evaluate(
        problem,
        SolveState(
            pos=torch.tensor(
                [[0.0, 0.0], [0.1, 0.0], [2.0, 0.0]], dtype=torch.float32, requires_grad=True
            ),
        ),
        RuntimeContext(),
    )
    far_loss = KDTreeRepulsionLoss().evaluate(
        problem,
        SolveState(
            pos=torch.tensor(
                [[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]], dtype=torch.float32, requires_grad=True
            ),
        ),
        RuntimeContext(),
    )

    assert close_loss.item() > far_loss.item()


def test_kdtree_repulsion_loss_radius_config_changes_neighbor_query() -> None:
    """Larger KD-tree radii should include more or stronger nearby-pair penalties."""

    problem = LayoutProblem(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=3)
    positions = torch.tensor(
        [[0.0, 0.0], [0.5, 0.0], [2.0, 0.0]], dtype=torch.float32, requires_grad=True
    )
    small_radius_loss = KDTreeRepulsionLoss(KDTreeRepulsionLossConfig(radius=0.4)).evaluate(
        problem,
        SolveState(pos=positions),
        RuntimeContext(),
    )
    large_radius_loss = KDTreeRepulsionLoss(KDTreeRepulsionLossConfig(radius=1.0)).evaluate(
        problem,
        SolveState(pos=positions),
        RuntimeContext(),
    )

    assert large_radius_loss.item() > small_radius_loss.item()


def test_kdtree_repulsion_loss_caches_pair_queries_in_state() -> None:
    """KD-tree repulsion should cache the SciPy pair query for reuse."""

    problem = LayoutProblem(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=3)
    state = SolveState(
        pos=torch.tensor(
            [[0.0, 0.0], [0.1, 0.0], [2.0, 0.0]], dtype=torch.float32, requires_grad=True
        ),
    )

    KDTreeRepulsionLoss().evaluate(problem, state, RuntimeContext())

    assert "neulay_kdtree_pairs" in state.extras
    assert "neulay_kdtree_query_radius" in state.extras


def test_sgd2_criterion_loss_stress_returns_finite_scalar_and_gradient() -> None:
    """The SGD2 stress criterion should return a finite scalar and backpropagate."""

    problem = _path_graph_problem(4)
    positions = torch.tensor(
        [
            [0.0, 0.0],
            [1.4, 0.0],
            [2.8, 0.0],
            [4.5, 0.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    state = SolveState(pos=positions)

    loss = SGD2CriterionLoss(SGD2CriterionLossConfig(criterion="stress", batch_size=6)).evaluate(
        problem,
        state,
        RuntimeContext(),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert positions.grad is not None
    assert torch.isfinite(positions.grad).all()


def test_sgd2_criterion_loss_stores_batch_size_and_sampler_metadata() -> None:
    """Evaluating an SGD2 criterion should populate its shared sampler metadata in state."""

    problem = _path_graph_problem(4)
    state = SolveState(
        pos=torch.tensor(
            [
                [0.0, 0.0],
                [1.4, 0.0],
                [2.8, 0.0],
                [4.5, 0.0],
            ],
            dtype=torch.float32,
            requires_grad=True,
        ),
    )

    SGD2CriterionLoss(SGD2CriterionLossConfig(criterion="stress", batch_size=5)).evaluate(
        problem,
        state,
        RuntimeContext(),
    )

    assert state.extras["sgd2_active_criterion"] == "stress"
    assert state.extras["sgd2_batch_size"] == 5
    assert "stress" in state.extras["sgd2_samplers"]


def test_sgd2_criterion_loss_matches_full_stress_mean_when_batch_covers_all_pairs() -> None:
    """A full-batch SGD2 stress step should match the exact sampled-stress mean."""

    problem = _path_graph_problem(4)
    positions = torch.tensor(
        [
            [0.0, 0.0],
            [1.4, 0.0],
            [2.8, 0.0],
            [4.5, 0.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    distance_matrix = _path_distance_matrix(4)
    state = SolveState(pos=positions, distance_matrix=distance_matrix)

    loss = SGD2CriterionLoss(SGD2CriterionLossConfig(criterion="stress", batch_size=99)).evaluate(
        problem,
        state,
        RuntimeContext(),
    )

    assert loss.item() == pytest.approx(
        _sgd2_stress_mean_for_path(positions.detach(), distance_matrix),
        rel=1.0e-6,
    )


def test_sgd2_crossing_detector_step_updates_detector_parameters() -> None:
    """Running crossing-detector inner steps should update detector weights."""

    problem = LayoutProblem(
        edge_index=torch.tensor([[0, 1], [3, 2]], dtype=torch.long),
        num_nodes=4,
    )
    state = SolveState(
        pos=torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=torch.float32,
            requires_grad=True,
        ),
    )
    op = SGD2CrossingDetectorStep(SGD2CrossingDetectorStepConfig(inner_steps=2, detector_lr=0.05))
    op.apply(problem, state, RuntimeContext())
    detector = state.extras["sgd2_crossing_state"].detector
    before = [param.detach().clone() for param in detector.parameters()]
    op.apply(problem, state, RuntimeContext())
    after = [param.detach().clone() for param in detector.parameters()]

    assert any(
        not torch.allclose(before_param, after_param)
        for before_param, after_param in zip(before, after)
    )


def test_sgd2_crossing_detector_step_zero_inner_steps_leaves_detector_unchanged() -> None:
    """With zero inner steps, the helper should skip detector training updates."""

    problem = LayoutProblem(
        edge_index=torch.tensor([[0, 1], [3, 2]], dtype=torch.long),
        num_nodes=4,
    )
    state = SolveState(
        pos=torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=torch.float32,
            requires_grad=True,
        ),
    )
    op = SGD2CrossingDetectorStep(SGD2CrossingDetectorStepConfig(inner_steps=0, detector_lr=0.05))
    op.apply(problem, state, RuntimeContext())
    detector = state.extras["sgd2_crossing_state"].detector
    before = [param.detach().clone() for param in detector.parameters()]
    op.apply(problem, state, RuntimeContext())
    after = [param.detach().clone() for param in detector.parameters()]

    assert all(
        torch.allclose(before_param, after_param)
        for before_param, after_param in zip(before, after)
    )


def test_sgd2_crossing_detector_step_sets_prev_loss_and_crossing_cache() -> None:
    """The crossing-detector helper should report a loss and cache it in state extras."""

    problem = LayoutProblem(
        edge_index=torch.tensor([[0, 1], [3, 2]], dtype=torch.long),
        num_nodes=4,
    )
    positions = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    state = SolveState(pos=positions)

    SGD2CrossingDetectorStep(
        SGD2CrossingDetectorStepConfig(inner_steps=2, detector_lr=0.05),
    ).apply(problem, state, RuntimeContext())

    assert state.prev_loss > 0.0
    assert "sgd2_crossing_loss" in state.extras
    assert positions.grad is not None


def test_cyclic_sampler_covers_each_index_once_when_batches_tile_the_pool() -> None:
    """The cyclic sampler should cover every index exactly once across one full epoch."""

    problem = _path_graph_problem(4)
    state = SolveState(
        pos=torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
            ],
            dtype=torch.float32,
            requires_grad=True,
        ),
        extras={"sgd2_active_criterion": "stress"},
    )
    CyclicSampler(CyclicSamplerConfig(pool_size=4)).apply(problem, state, RuntimeContext())
    sampler = state.extras["sgd2_samplers"]["stress"]
    first_batch = sampler.sample(2)
    second_batch = sampler.sample(2)

    assert set(torch.cat([first_batch, second_batch]).tolist()) == {0, 1, 2, 3}


def test_cyclic_sampler_refreshes_after_pool_exhaustion() -> None:
    """Sampling past the end of the pool should reshuffle and still return a valid batch."""

    problem = _path_graph_problem(4)
    state = SolveState(
        pos=torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
            ],
            dtype=torch.float32,
            requires_grad=True,
        ),
        extras={"sgd2_active_criterion": "stress"},
    )
    CyclicSampler(CyclicSamplerConfig(pool_size=5)).apply(problem, state, RuntimeContext())
    sampler = state.extras["sgd2_samplers"]["stress"]
    sampler.sample(3)
    refreshed_batch = sampler.sample(3)

    assert refreshed_batch.shape == (3,)
    assert set(refreshed_batch.tolist()).issubset({0, 1, 2, 3, 4})


def test_cyclic_sampler_infers_pool_size_from_active_sgd2_criterion() -> None:
    """When pool size is zero, CyclicSampler should infer the stress-pair pool."""

    problem = _path_graph_problem(4)
    state = SolveState(
        pos=torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
            ],
            dtype=torch.float32,
            requires_grad=True,
        ),
        extras={"sgd2_active_criterion": "stress"},
    )

    CyclicSampler(CyclicSamplerConfig(pool_size=0)).apply(problem, state, RuntimeContext())
    sampler = state.extras["sgd2_samplers"]["stress"]
    inferred_epoch = sampler.sample(6)

    assert set(inferred_epoch.tolist()) == {0, 1, 2, 3, 4, 5}
