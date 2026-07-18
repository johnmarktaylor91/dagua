"""Tests for engine loss operations."""

from __future__ import annotations

from typing import List, Optional, Tuple

import pytest
import torch

from dagua.layout.layers import build_layer_index
from dagua.layout.losses import flex_spacing_loss as reference_flex_spacing_loss
from dagua.layout.ops.base import LossGroup, LossOp
from dagua.layout.ops.loss_engine import (
    AlignmentLoss,
    BackEdgeCompactnessLoss,
    ClusterCompactnessLoss,
    ClusterContainmentLoss,
    ClusterSeparationLoss,
    CrossingLoss,
    CrossingLossConfig,
    DagOrderingLoss,
    EdgeAttractionLoss,
    EdgeAttractionLossConfig,
    EdgeLengthVarianceLoss,
    EdgeStraightnessLoss,
    FanoutDistributionLoss,
    FanoutDistributionLossConfig,
    FlexSpacingLoss,
    OverlapAvoidanceLoss,
    OverlapAvoidanceLossConfig,
    PositionPinLoss,
    RepulsionLoss,
    RepulsionLossConfig,
    SpacingConsistencyLoss,
    SpacingConsistencyLossConfig,
)
from dagua.layout.ops.state import (
    AnnealingSchedule,
    FlexConstraints,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)


def _make_state_from_pos(
    pos: torch.Tensor,
    layers: Optional[torch.Tensor] = None,
    step: int = 0,
) -> SolveState:
    """Build a solve state from raw position and layer tensors.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    layers : torch.Tensor | None, default=None
        Optional layer assignment tensor with shape ``[N]``.
    step : int, default=0
        Current optimizer step for losses that gate behavior on iteration.

    Returns
    -------
    SolveState
        Fresh solve state with differentiable positions.
    """
    layer_index = build_layer_index(layers) if layers is not None else None
    return SolveState(
        pos=pos.clone().detach().requires_grad_(True),
        layers=layers,
        layer_index=layer_index,
        step=step,
    )


def _engine_loss_case(
    loss_name: str,
    quality: str,
) -> Tuple[LossOp, LayoutProblem, SolveState, float]:
    """Build one engine-loss scenario for perfect or bad layout assertions.

    Parameters
    ----------
    loss_name : str
        Registered engine loss name.
    quality : str
        Either ``"perfect"`` or ``"bad"``.

    Returns
    -------
    tuple[LossOp, LayoutProblem, SolveState, float]
        Loss op, problem, state, and the comparison threshold for the case.

    Raises
    ------
    ValueError
        If the requested case is unknown.
    """
    if quality not in {"perfect", "bad"}:
        raise ValueError(f"Unsupported quality {quality!r}.")

    is_perfect = quality == "perfect"

    if loss_name == DagOrderingLoss.name:
        problem = LayoutProblem(
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            num_nodes=2,
            node_sizes=torch.full((2, 2), 10.0, dtype=torch.float32),
        )
        pos = (
            torch.tensor([[0.0, 0.0], [0.0, 40.0]], dtype=torch.float32)
            if is_perfect
            else torch.tensor([[0.0, 40.0], [0.0, 0.0]], dtype=torch.float32)
        )
        return (
            DagOrderingLoss(),
            problem,
            _make_state_from_pos(pos),
            (1.0e-6 if is_perfect else 1.0),
        )

    if loss_name == EdgeAttractionLoss.name:
        problem = LayoutProblem(
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            num_nodes=2,
        )
        pos = (
            torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
            if is_perfect
            else torch.tensor([[10.0, 0.0], [0.0, 20.0]], dtype=torch.float32)
        )
        return (
            EdgeAttractionLoss(),
            problem,
            _make_state_from_pos(pos),
            (1.0e-6 if is_perfect else 1.0),
        )

    if loss_name == EdgeStraightnessLoss.name:
        problem = LayoutProblem(
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            num_nodes=2,
        )
        pos = (
            torch.tensor([[0.0, 0.0], [0.0, 10.0]], dtype=torch.float32)
            if is_perfect
            else torch.tensor([[5.0, 0.0], [-5.0, 10.0]], dtype=torch.float32)
        )
        return (
            EdgeStraightnessLoss(),
            problem,
            _make_state_from_pos(pos),
            (1.0e-6 if is_perfect else 1.0),
        )

    if loss_name == EdgeLengthVarianceLoss.name:
        problem = LayoutProblem(
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            num_nodes=3,
        )
        pos = (
            torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
            if is_perfect
            else torch.tensor([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]], dtype=torch.float32)
        )
        return (
            EdgeLengthVarianceLoss(),
            problem,
            _make_state_from_pos(pos),
            (1.0e-6 if is_perfect else 0.1),
        )

    if loss_name == RepulsionLoss.name:
        problem = LayoutProblem(
            edge_index=torch.empty((2, 0), dtype=torch.long),
            num_nodes=2,
            node_sizes=torch.full((2, 2), 10.0, dtype=torch.float32),
        )
        pos = (
            torch.tensor([[0.0, 0.0], [1000.0, 0.0]], dtype=torch.float32)
            if is_perfect
            else torch.tensor([[0.0, 0.0], [0.1, 0.0]], dtype=torch.float32)
        )
        return RepulsionLoss(), problem, _make_state_from_pos(pos), (1.0e-5 if is_perfect else 1.0)

    if loss_name == OverlapAvoidanceLoss.name:
        problem = LayoutProblem(
            edge_index=torch.empty((2, 0), dtype=torch.long),
            num_nodes=2,
            node_sizes=torch.full((2, 2), 10.0, dtype=torch.float32),
        )
        pos = (
            torch.tensor([[0.0, 0.0], [30.0, 0.0]], dtype=torch.float32)
            if is_perfect
            else torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        )
        return (
            OverlapAvoidanceLoss(),
            problem,
            _make_state_from_pos(pos),
            (1.0e-6 if is_perfect else 1.0),
        )

    if loss_name == CrossingLoss.name:
        problem = LayoutProblem(
            edge_index=torch.tensor([[0, 1], [3, 2]], dtype=torch.long),
            num_nodes=4,
        )
        pos = (
            torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float32)
            if is_perfect
            else torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
        )
        return CrossingLoss(), problem, _make_state_from_pos(pos), (1.0e-2 if is_perfect else 0.1)

    if loss_name == ClusterCompactnessLoss.name:
        problem = LayoutProblem(
            edge_index=torch.empty((2, 0), dtype=torch.long),
            num_nodes=4,
            clusters={"a": [0, 1], "b": [2, 3]},
        )
        pos = (
            torch.tensor([[0.0, 0.0], [0.0, 0.0], [10.0, 0.0], [10.0, 0.0]], dtype=torch.float32)
            if is_perfect
            else torch.tensor(
                [[0.0, 0.0], [4.0, 0.0], [10.0, 0.0], [14.0, 0.0]],
                dtype=torch.float32,
            )
        )
        return (
            ClusterCompactnessLoss(),
            problem,
            _make_state_from_pos(pos),
            (1.0e-6 if is_perfect else 0.1),
        )

    if loss_name == ClusterSeparationLoss.name:
        problem = LayoutProblem(
            edge_index=torch.empty((2, 0), dtype=torch.long),
            num_nodes=4,
            node_sizes=torch.full((4, 2), 2.0, dtype=torch.float32),
            clusters={"a": [0, 1], "b": [2, 3]},
            cluster_parents={"a": None, "b": None},
        )
        pos = (
            torch.tensor([[0.0, 0.0], [0.0, 1.0], [30.0, 0.0], [30.0, 1.0]], dtype=torch.float32)
            if is_perfect
            else torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
        )
        return (
            ClusterSeparationLoss(),
            problem,
            _make_state_from_pos(pos),
            (1.0e-6 if is_perfect else 1.0),
        )

    if loss_name == ClusterContainmentLoss.name:
        problem = LayoutProblem(
            edge_index=torch.empty((2, 0), dtype=torch.long),
            num_nodes=4,
            node_sizes=torch.full((4, 2), 2.0, dtype=torch.float32),
            clusters={"parent": [0, 3], "child": [1, 2]},
            cluster_parents={"parent": None, "child": "parent"},
        )
        pos = (
            torch.tensor([[-1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
            if is_perfect
            else torch.tensor(
                [[-1.0, 0.0], [0.0, 0.0], [20.0, 0.0], [1.0, 0.0]],
                dtype=torch.float32,
            )
        )
        return (
            ClusterContainmentLoss(),
            problem,
            _make_state_from_pos(pos),
            (1.0e-6 if is_perfect else 0.1),
        )

    if loss_name == SpacingConsistencyLoss.name:
        layers = torch.tensor([0, 0, 0], dtype=torch.long)
        problem = LayoutProblem(
            edge_index=torch.empty((2, 0), dtype=torch.long),
            num_nodes=3,
            node_sizes=torch.full((3, 2), 10.0, dtype=torch.float32),
        )
        pos = (
            torch.tensor([[0.0, 0.0], [35.0, 0.0], [70.0, 0.0]], dtype=torch.float32)
            if is_perfect
            else torch.tensor([[0.0, 0.0], [20.0, 0.0], [70.0, 0.0]], dtype=torch.float32)
        )
        return (
            SpacingConsistencyLoss(),
            problem,
            _make_state_from_pos(pos, layers=layers),
            (1.0e-6 if is_perfect else 1.0),
        )

    if loss_name == FanoutDistributionLoss.name:
        problem = LayoutProblem(
            edge_index=torch.tensor([[0, 0, 0, 0, 0], [1, 2, 3, 4, 5]], dtype=torch.long),
            num_nodes=6,
        )
        pos = (
            torch.tensor(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.3090, 0.9511],
                    [-0.8090, 0.5878],
                    [-0.8090, -0.5878],
                    [0.3090, -0.9511],
                ],
                dtype=torch.float32,
            )
            if is_perfect
            else torch.tensor(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [2.0, 0.0],
                    [3.0, 0.0],
                    [4.0, 0.0],
                    [5.0, 0.0],
                ],
                dtype=torch.float32,
            )
        )
        return (
            FanoutDistributionLoss(),
            problem,
            _make_state_from_pos(pos),
            (1.0e-5 if is_perfect else 0.1),
        )

    if loss_name == BackEdgeCompactnessLoss.name:
        problem = LayoutProblem(
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            num_nodes=2,
        )
        pos = (
            torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.float32)
            if is_perfect
            else torch.tensor([[10.0, 1.0], [0.0, 0.0]], dtype=torch.float32)
        )
        return (
            BackEdgeCompactnessLoss(),
            problem,
            _make_state_from_pos(pos),
            (1.0e-6 if is_perfect else 1.0),
        )

    if loss_name == PositionPinLoss.name:
        flex = FlexConstraints(
            pin_indices=torch.tensor([0], dtype=torch.long),
            pin_targets=torch.tensor([[5.0, 5.0]], dtype=torch.float32),
            pin_weights=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
            soft_pin_mask=torch.tensor([[True, True]], dtype=torch.bool),
            hard_pin_mask=torch.zeros((1, 2), dtype=torch.bool),
        )
        problem = LayoutProblem(
            edge_index=torch.empty((2, 0), dtype=torch.long),
            num_nodes=1,
            flex=flex,
        )
        pos = (
            torch.tensor([[5.0, 5.0]], dtype=torch.float32)
            if is_perfect
            else torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        )
        return (
            PositionPinLoss(),
            problem,
            _make_state_from_pos(pos),
            (1.0e-6 if is_perfect else 1.0),
        )

    if loss_name == AlignmentLoss.name:
        flex = FlexConstraints(
            align_groups=[(torch.tensor([0, 1, 2], dtype=torch.long), 2.0, 0)],
        )
        problem = LayoutProblem(
            edge_index=torch.empty((2, 0), dtype=torch.long),
            num_nodes=3,
            flex=flex,
        )
        pos = (
            torch.tensor([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]], dtype=torch.float32)
            if is_perfect
            else torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=torch.float32)
        )
        return AlignmentLoss(), problem, _make_state_from_pos(pos), (1.0e-6 if is_perfect else 0.1)

    if loss_name == FlexSpacingLoss.name:
        layers = torch.tensor([0, 0, 0], dtype=torch.long)
        problem = LayoutProblem(
            edge_index=torch.empty((2, 0), dtype=torch.long),
            num_nodes=3,
            node_sizes=torch.full((3, 2), 10.0, dtype=torch.float32),
            flex=FlexConstraints(flex_node_sep=25.0, flex_node_sep_weight=1.5),
        )
        pos = (
            torch.tensor([[0.0, 0.0], [35.0, 0.0], [70.0, 0.0]], dtype=torch.float32)
            if is_perfect
            else torch.tensor([[0.0, 0.0], [20.0, 0.0], [70.0, 0.0]], dtype=torch.float32)
        )
        return (
            FlexSpacingLoss(),
            problem,
            _make_state_from_pos(pos, layers=layers),
            (1.0e-6 if is_perfect else 1.0),
        )

    raise ValueError(f"Unsupported engine loss {loss_name!r}.")


def _make_problem() -> LayoutProblem:
    """Build a graph problem that exercises every engine loss.

    Returns
    -------
    LayoutProblem
        Problem with edges, node sizes, clusters, layers, and flex data.
    """
    edge_index = torch.tensor(
        [
            [0, 0, 0, 0, 0, 5],
            [1, 2, 3, 4, 5, 1],
        ],
        dtype=torch.long,
    )
    node_sizes = torch.full((6, 2), 10.0, dtype=torch.float32)
    flex = FlexConstraints(
        pin_indices=torch.tensor([0, 3], dtype=torch.long),
        pin_targets=torch.tensor([[5.0, 45.0], [5.0, 10.0]], dtype=torch.float32),
        pin_weights=torch.tensor([[1.0, 1.0], [2.0, 0.0]], dtype=torch.float32),
        soft_pin_mask=torch.tensor([[True, True], [True, False]], dtype=torch.bool),
        hard_pin_mask=torch.zeros((2, 2), dtype=torch.bool),
        align_groups=[(torch.tensor([1, 2, 3], dtype=torch.long), 2.0, 0)],
        flex_node_sep=25.0,
        flex_node_sep_weight=1.5,
    )
    return LayoutProblem(
        edge_index=edge_index,
        num_nodes=6,
        node_sizes=node_sizes,
        clusters={
            "parent": [0, 1, 2, 3],
            "child": [1, 2],
            "other": [4, 5],
        },
        cluster_parents={
            "parent": None,
            "child": "parent",
            "other": None,
        },
        flex=flex,
    )


def _make_state() -> SolveState:
    """Build a solve state with positions and layer metadata.

    Returns
    -------
    SolveState
        State containing differentiable positions and a layer index.
    """
    pos = torch.tensor(
        [
            [0.0, 50.0],
            [-20.0, 0.0],
            [-10.0, 0.0],
            [0.0, 0.0],
            [10.0, 0.0],
            [20.0, 100.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    layers = torch.tensor([0, 1, 1, 1, 1, 1], dtype=torch.long)
    return SolveState(
        pos=pos,
        layers=layers,
        layer_index=build_layer_index(layers),
        step=0,
    )


def _all_losses() -> List[LossOp]:
    """Return one instance of each engine loss op.

    Returns
    -------
    list[LossOp]
        Every engine loss op requested by the task.
    """
    return [
        DagOrderingLoss(),
        EdgeAttractionLoss(),
        EdgeStraightnessLoss(),
        EdgeLengthVarianceLoss(),
        RepulsionLoss(),
        OverlapAvoidanceLoss(),
        CrossingLoss(),
        ClusterCompactnessLoss(),
        ClusterSeparationLoss(),
        ClusterContainmentLoss(),
        SpacingConsistencyLoss(),
        FanoutDistributionLoss(),
        BackEdgeCompactnessLoss(),
        PositionPinLoss(),
        AlignmentLoss(),
        FlexSpacingLoss(),
    ]


@pytest.mark.parametrize("loss_op", _all_losses(), ids=lambda op: op.name)
def test_each_loss_returns_scalar_tensor_with_requires_grad(loss_op: LossOp) -> None:
    """Each engine loss should return a differentiable scalar."""

    problem = _make_problem()
    state = _make_state()

    result = loss_op.evaluate(problem, state, RuntimeContext())

    assert result.shape == ()
    assert result.requires_grad


@pytest.mark.parametrize("loss_name", [loss_op.name for loss_op in _all_losses()])
def test_each_engine_loss_is_near_zero_for_perfect_layout(loss_name: str) -> None:
    """Each engine loss should vanish or nearly vanish on its ideal layout."""

    loss_op, problem, state, max_value = _engine_loss_case(loss_name, "perfect")

    result = loss_op.evaluate(problem, state, RuntimeContext())

    assert result.item() <= max_value


@pytest.mark.parametrize("loss_name", [loss_op.name for loss_op in _all_losses()])
def test_each_engine_loss_is_nonzero_for_bad_layout(loss_name: str) -> None:
    """Each engine loss should report a positive penalty for a bad layout."""

    loss_op, problem, state, min_value = _engine_loss_case(loss_name, "bad")

    result = loss_op.evaluate(problem, state, RuntimeContext())

    assert result.item() >= min_value


def test_dag_ordering_loss_is_nonzero_for_wrong_way_edges() -> None:
    """DAG ordering loss should penalize edges whose targets sit above sources."""

    edge_index = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 4],
        ],
        dtype=torch.long,
    )
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=5,
        node_sizes=torch.full((5, 2), 10.0, dtype=torch.float32),
    )
    state = SolveState(
        pos=torch.tensor(
            [
                [0.0, 0.0],
                [0.0, -10.0],
                [0.0, -20.0],
                [0.0, -30.0],
                [0.0, -40.0],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )
    )

    result = DagOrderingLoss().evaluate(problem, state, RuntimeContext())

    assert result.item() > 0.0


def test_dag_ordering_loss_is_near_zero_for_forward_edges() -> None:
    """DAG ordering loss should disappear when all edges point downward."""

    loss_op, problem, state, max_value = _engine_loss_case(DagOrderingLoss.name, "perfect")

    result = loss_op.evaluate(problem, state, RuntimeContext())

    assert result.item() <= max_value


def test_repulsion_loss_is_nonzero_for_overlapping_nodes() -> None:
    """Repulsion should be positive when nodes are nearly coincident."""

    problem = LayoutProblem(
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        num_nodes=3,
        node_sizes=torch.full((3, 2), 10.0, dtype=torch.float32),
    )
    state = SolveState(
        pos=torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.0, 0.1],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )
    )

    result = RepulsionLoss().evaluate(problem, state, RuntimeContext())

    assert result.item() > 0.0


def test_repulsion_loss_is_near_zero_for_well_separated_nodes() -> None:
    """Repulsion should decay toward zero as inter-node distance grows."""

    problem = LayoutProblem(
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        num_nodes=3,
        node_sizes=torch.full((3, 2), 10.0, dtype=torch.float32),
    )
    state = SolveState(
        pos=torch.tensor(
            [
                [0.0, 0.0],
                [10_000.0, 0.0],
                [0.0, 10_000.0],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )
    )

    result = RepulsionLoss().evaluate(problem, state, RuntimeContext())

    assert result.item() < 1e-6


def test_crossing_loss_distinguishes_crossing_and_non_crossing_layouts() -> None:
    """Crossing loss should be materially larger for a known crossing pair."""

    loss_op, problem, good_state, max_value = _engine_loss_case(CrossingLoss.name, "perfect")
    bad_op, _same_problem, bad_state, min_value = _engine_loss_case(CrossingLoss.name, "bad")

    good_loss = loss_op.evaluate(problem, good_state, RuntimeContext())
    bad_loss = bad_op.evaluate(problem, bad_state, RuntimeContext())

    assert good_loss.item() <= max_value
    assert bad_loss.item() >= min_value
    assert bad_loss.item() > good_loss.item()


def test_loss_group_composes_three_engine_losses() -> None:
    """LossGroup should combine multiple engine losses and backpropagate once."""

    problem = _make_problem()
    state = _make_state()
    group = LossGroup(
        losses=[
            DagOrderingLoss(),
            EdgeAttractionLoss(),
            RepulsionLoss(),
        ],
        backward_mode="combined",
    )

    result = group.apply(problem, state, RuntimeContext())

    assert result.prev_loss > 0.0
    assert state.pos is not None
    assert state.pos.grad is not None


def test_loss_group_per_loss_mode_updates_gradients() -> None:
    """LossGroup should backpropagate in per-loss mode with three losses."""

    problem = _make_problem()
    state = _make_state()
    group = LossGroup(
        losses=[
            DagOrderingLoss(),
            EdgeAttractionLoss(),
            RepulsionLoss(),
        ],
        backward_mode="per_loss",
    )

    result = group.apply(problem, state, RuntimeContext())

    assert result.prev_loss > 0.0
    assert state.pos is not None
    assert state.pos.grad is not None
    assert torch.linalg.norm(state.pos.grad).item() > 0.0


def test_loss_group_combined_and_per_loss_modes_both_backpropagate() -> None:
    """Combined and per-loss modes should both populate usable gradients."""

    problem = _make_problem()
    combined_state = _make_state()
    per_loss_state = _make_state()
    losses = [DagOrderingLoss(), EdgeAttractionLoss(), RepulsionLoss()]

    LossGroup(losses=losses, backward_mode="combined").apply(
        problem,
        combined_state,
        RuntimeContext(),
    )
    LossGroup(losses=losses, backward_mode="per_loss").apply(
        problem,
        per_loss_state,
        RuntimeContext(),
    )

    assert combined_state.pos is not None
    assert combined_state.pos.grad is not None
    assert torch.linalg.norm(combined_state.pos.grad).item() > 0.0
    assert per_loss_state.pos is not None
    assert per_loss_state.pos.grad is not None
    assert torch.linalg.norm(per_loss_state.pos.grad).item() > 0.0


@pytest.mark.parametrize(
    ("edge_index", "pos", "expected"),
    [
        (
            torch.tensor([[0], [1]], dtype=torch.long),
            torch.tensor([[0.0, 0.0], [0.0, 40.0]], dtype=torch.float32),
            pytest.approx(0.0, abs=1.0e-6),
        ),
        (
            torch.tensor([[0], [1]], dtype=torch.long),
            torch.tensor([[0.0, 40.0], [0.0, 0.0]], dtype=torch.float32),
            pytest.approx(75.0, rel=1.0e-5),
        ),
        (
            torch.empty((2, 0), dtype=torch.long),
            torch.tensor([[0.0, 0.0], [0.0, 40.0]], dtype=torch.float32),
            pytest.approx(0.0, abs=1.0e-6),
        ),
    ],
    ids=["correct_order", "reversed_order", "empty_edges"],
)
def test_dag_ordering_loss_cases(
    edge_index: torch.Tensor,
    pos: torch.Tensor,
    expected: pytest.ApproxBase,
) -> None:
    """Dag ordering should respect forward, reversed, and empty edge sets."""

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=2,
        node_sizes=torch.full((2, 2), 10.0, dtype=torch.float32),
    )

    loss = DagOrderingLoss().evaluate(problem, _make_state_from_pos(pos), RuntimeContext())

    assert loss.item() == expected


def test_dag_ordering_loss_single_edge_matches_margin_violation() -> None:
    """A single wrong-way edge should incur exactly one margin violation."""

    problem = LayoutProblem(
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        num_nodes=2,
        node_sizes=torch.full((2, 2), 10.0, dtype=torch.float32),
    )
    state = _make_state_from_pos(
        torch.tensor([[0.0, 10.0], [0.0, 5.0]], dtype=torch.float32),
    )

    loss = DagOrderingLoss().evaluate(problem, state, RuntimeContext())

    assert loss.item() == pytest.approx(40.0, rel=1.0e-5)


def test_edge_attraction_loss_prefers_close_neighbors() -> None:
    """Edge attraction should increase as connected endpoints separate."""

    problem = LayoutProblem(
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        num_nodes=2,
    )
    close_loss = EdgeAttractionLoss().evaluate(
        problem,
        _make_state_from_pos(torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)),
        RuntimeContext(),
    )
    far_loss = EdgeAttractionLoss().evaluate(
        problem,
        _make_state_from_pos(torch.tensor([[0.0, 0.0], [4.0, 0.0]], dtype=torch.float32)),
        RuntimeContext(),
    )

    assert close_loss.item() == pytest.approx(0.0, abs=1.0e-6)
    assert far_loss.item() > close_loss.item()


def test_edge_attraction_loss_x_bias_changes_horizontal_penalty() -> None:
    """Higher x-bias should amplify purely horizontal edge displacement."""

    problem = LayoutProblem(
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        num_nodes=2,
    )
    pos = torch.tensor([[0.0, 0.0], [4.0, 0.0]], dtype=torch.float32)
    low_bias_loss = EdgeAttractionLoss(EdgeAttractionLossConfig(x_bias=1.0)).evaluate(
        problem,
        _make_state_from_pos(pos),
        RuntimeContext(),
    )
    high_bias_loss = EdgeAttractionLoss(EdgeAttractionLossConfig(x_bias=8.0)).evaluate(
        problem,
        _make_state_from_pos(pos),
        RuntimeContext(),
    )

    assert high_bias_loss.item() > low_bias_loss.item()


def test_edge_attraction_loss_is_zero_without_edges() -> None:
    """Edge attraction should vanish when the graph has no edges."""

    problem = LayoutProblem(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=2)
    loss = EdgeAttractionLoss().evaluate(
        problem,
        _make_state_from_pos(torch.tensor([[0.0, 0.0], [4.0, 0.0]], dtype=torch.float32)),
        RuntimeContext(),
    )

    assert loss.item() == pytest.approx(0.0, abs=1.0e-6)


@pytest.mark.parametrize(
    ("edge_index", "pos", "expected"),
    [
        (
            torch.tensor([[0], [1]], dtype=torch.long),
            torch.tensor([[0.0, 0.0], [0.0, 10.0]], dtype=torch.float32),
            pytest.approx(0.0, abs=1.0e-6),
        ),
        (
            torch.tensor([[0], [1]], dtype=torch.long),
            torch.tensor([[5.0, 0.0], [-5.0, 10.0]], dtype=torch.float32),
            pytest.approx(100.0, rel=1.0e-5),
        ),
        (
            torch.empty((2, 0), dtype=torch.long),
            torch.tensor([[0.0, 0.0], [3.0, 4.0]], dtype=torch.float32),
            pytest.approx(0.0, abs=1.0e-6),
        ),
    ],
    ids=["vertical", "angled", "empty_edges"],
)
def test_edge_straightness_loss_cases(
    edge_index: torch.Tensor,
    pos: torch.Tensor,
    expected: pytest.ApproxBase,
) -> None:
    """Edge straightness should only penalize horizontal displacement."""

    problem = LayoutProblem(edge_index=edge_index, num_nodes=2)

    loss = EdgeStraightnessLoss().evaluate(problem, _make_state_from_pos(pos), RuntimeContext())

    assert loss.item() == expected


@pytest.mark.parametrize(
    ("edge_index", "pos", "expected"),
    [
        (
            torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=torch.float32),
            pytest.approx(0.0, abs=1.0e-6),
        ),
        (
            torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            torch.tensor([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]], dtype=torch.float32),
            None,
        ),
        (
            torch.tensor([[0], [1]], dtype=torch.long),
            torch.tensor([[0.0, 0.0], [3.0, 4.0]], dtype=torch.float32),
            pytest.approx(0.0, abs=1.0e-6),
        ),
    ],
    ids=["uniform_lengths", "varied_lengths", "single_edge"],
)
def test_edge_length_variance_loss_cases(
    edge_index: torch.Tensor,
    pos: torch.Tensor,
    expected: pytest.ApproxBase | None,
) -> None:
    """Edge length variance should vanish for uniform or underspecified edge sets."""

    problem = LayoutProblem(edge_index=edge_index, num_nodes=3)

    loss = EdgeLengthVarianceLoss().evaluate(
        problem,
        _make_state_from_pos(pos),
        RuntimeContext(),
    )

    if expected is None:
        assert loss.item() > 0.0
    else:
        assert loss.item() == expected


def test_edge_length_variance_loss_is_zero_without_edges() -> None:
    """Edge length variance should be zero when no edges are present."""

    problem = LayoutProblem(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=3)

    loss = EdgeLengthVarianceLoss().evaluate(
        problem,
        _make_state_from_pos(
            torch.tensor([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]], dtype=torch.float32)
        ),
        RuntimeContext(),
    )

    assert loss.item() == pytest.approx(0.0, abs=1.0e-6)


def test_repulsion_loss_penalizes_overlapping_nodes_more_than_separated_nodes() -> None:
    """Repulsion should be strongest when nodes are nearly coincident."""

    problem = LayoutProblem(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=3,
        node_sizes=torch.full((3, 2), 10.0, dtype=torch.float32),
    )
    overlapping_loss = RepulsionLoss().evaluate(
        problem,
        _make_state_from_pos(
            torch.tensor([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]], dtype=torch.float32)
        ),
        RuntimeContext(),
    )
    separated_loss = RepulsionLoss().evaluate(
        problem,
        _make_state_from_pos(
            torch.tensor([[0.0, 0.0], [10_000.0, 0.0], [0.0, 10_000.0]], dtype=torch.float32),
        ),
        RuntimeContext(),
    )

    assert overlapping_loss.item() > 1.0
    assert separated_loss.item() < 1.0e-6


def test_repulsion_loss_matches_two_node_exact_formula() -> None:
    """The current engine repulsion op should use the exact two-node inverse-distance term."""

    problem = LayoutProblem(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=2)
    state = _make_state_from_pos(torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=torch.float32))

    loss = RepulsionLoss().evaluate(problem, state, RuntimeContext())

    assert loss.item() == pytest.approx(1.0 / 4.0001, rel=1.0e-5)


def test_repulsion_loss_threshold_config_is_currently_a_no_op_for_exact_engine_path() -> None:
    """Repulsion config thresholds should not affect the current exact-only engine op path."""

    problem = LayoutProblem(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=2)
    pos = torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
    low_threshold_loss = RepulsionLoss(RepulsionLossConfig(threshold=1)).evaluate(
        problem,
        _make_state_from_pos(pos),
        RuntimeContext(),
    )
    high_threshold_loss = RepulsionLoss(RepulsionLossConfig(threshold=10_000)).evaluate(
        problem,
        _make_state_from_pos(pos),
        RuntimeContext(),
    )

    assert low_threshold_loss.item() == pytest.approx(high_threshold_loss.item(), rel=1.0e-6)


def test_overlap_avoidance_loss_is_zero_for_non_overlapping_boxes() -> None:
    """Overlap avoidance should vanish once node boxes are disjoint."""

    problem = LayoutProblem(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=2,
        node_sizes=torch.full((2, 2), 10.0, dtype=torch.float32),
    )

    loss = OverlapAvoidanceLoss().evaluate(
        problem,
        _make_state_from_pos(torch.tensor([[0.0, 0.0], [30.0, 0.0]], dtype=torch.float32)),
        RuntimeContext(),
    )

    assert loss.item() == pytest.approx(0.0, abs=1.0e-6)


def test_overlap_avoidance_loss_is_positive_for_overlapping_boxes() -> None:
    """Overlap avoidance should increase when boxes intersect."""

    problem = LayoutProblem(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=2,
        node_sizes=torch.full((2, 2), 10.0, dtype=torch.float32),
    )

    loss = OverlapAvoidanceLoss().evaluate(
        problem,
        _make_state_from_pos(torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)),
        RuntimeContext(),
    )

    assert loss.item() > 0.0


def test_overlap_avoidance_loss_padding_increases_penalty() -> None:
    """More padding should require more separation between node boxes."""

    problem = LayoutProblem(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=2,
        node_sizes=torch.full((2, 2), 10.0, dtype=torch.float32),
    )
    pos = torch.tensor([[0.0, 0.0], [12.0, 0.0]], dtype=torch.float32)
    small_padding_loss = OverlapAvoidanceLoss(OverlapAvoidanceLossConfig(padding=0.0)).evaluate(
        problem,
        _make_state_from_pos(pos),
        RuntimeContext(),
    )
    large_padding_loss = OverlapAvoidanceLoss(OverlapAvoidanceLossConfig(padding=4.0)).evaluate(
        problem,
        _make_state_from_pos(pos),
        RuntimeContext(),
    )

    assert large_padding_loss.item() > small_padding_loss.item()


def test_crossing_loss_detects_known_crossings() -> None:
    """Crossing loss should be nonzero for a simple two-edge crossing."""

    problem = LayoutProblem(
        edge_index=torch.tensor([[0, 1], [3, 2]], dtype=torch.long),
        num_nodes=4,
    )

    loss = CrossingLoss().evaluate(
        problem,
        _make_state_from_pos(
            torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
        ),
        RuntimeContext(),
    )

    assert loss.item() > 0.9


def test_crossing_loss_is_near_zero_for_parallel_edges() -> None:
    """Crossing loss should stay near zero when edge endpoint order is preserved."""

    problem = LayoutProblem(
        edge_index=torch.tensor([[0, 1], [3, 2]], dtype=torch.long),
        num_nodes=4,
    )

    loss = CrossingLoss(CrossingLossConfig(alpha=20.0)).evaluate(
        problem,
        _make_state_from_pos(
            torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float32)
        ),
        RuntimeContext(),
    )

    assert loss.item() < 1.0e-6


def test_crossing_loss_alpha_config_changes_crossing_sharpness() -> None:
    """Higher alpha should sharpen the crossing proxy for the same geometry."""

    problem = LayoutProblem(
        edge_index=torch.tensor([[0, 1], [3, 2]], dtype=torch.long),
        num_nodes=4,
    )
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    low_alpha_loss = CrossingLoss(CrossingLossConfig(alpha=1.0)).evaluate(
        problem,
        _make_state_from_pos(pos),
        RuntimeContext(),
    )
    high_alpha_loss = CrossingLoss(CrossingLossConfig(alpha=20.0)).evaluate(
        problem,
        _make_state_from_pos(pos),
        RuntimeContext(),
    )

    assert high_alpha_loss.item() > low_alpha_loss.item()


def test_cluster_compactness_loss_prefers_tight_clusters() -> None:
    """Cluster compactness should be smaller for tightly packed members."""

    problem = LayoutProblem(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=4,
        clusters={"a": [0, 1], "b": [2, 3]},
    )
    tight_loss = ClusterCompactnessLoss().evaluate(
        problem,
        _make_state_from_pos(
            torch.tensor([[0.0, 0.0], [0.1, 0.0], [10.0, 0.0], [10.1, 0.0]], dtype=torch.float32),
        ),
        RuntimeContext(),
    )
    spread_loss = ClusterCompactnessLoss().evaluate(
        problem,
        _make_state_from_pos(
            torch.tensor([[0.0, 0.0], [4.0, 0.0], [10.0, 0.0], [14.0, 0.0]], dtype=torch.float32),
        ),
        RuntimeContext(),
    )

    assert tight_loss.item() < spread_loss.item()


def test_cluster_compactness_loss_is_zero_for_singleton_clusters() -> None:
    """Singleton clusters should not contribute compactness energy."""

    problem = LayoutProblem(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=2,
        clusters={"a": [0], "b": [1]},
    )

    loss = ClusterCompactnessLoss().evaluate(
        problem,
        _make_state_from_pos(torch.tensor([[0.0, 0.0], [10.0, 0.0]], dtype=torch.float32)),
        RuntimeContext(),
    )

    assert loss.item() == pytest.approx(0.0, abs=1.0e-6)


def test_cluster_compactness_loss_is_zero_without_clusters() -> None:
    """Cluster compactness should vanish when no cluster metadata is present."""

    problem = LayoutProblem(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=2)

    loss = ClusterCompactnessLoss().evaluate(
        problem,
        _make_state_from_pos(torch.tensor([[0.0, 0.0], [10.0, 0.0]], dtype=torch.float32)),
        RuntimeContext(),
    )

    assert loss.item() == pytest.approx(0.0, abs=1.0e-6)


def test_cluster_separation_loss_penalizes_overlapping_sibling_clusters() -> None:
    """Sibling cluster bboxes should repel when they overlap."""

    problem = LayoutProblem(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=4,
        node_sizes=torch.full((4, 2), 2.0, dtype=torch.float32),
        clusters={"a": [0, 1], "b": [2, 3]},
        cluster_parents={"a": None, "b": None},
    )
    overlapping_loss = ClusterSeparationLoss().evaluate(
        problem,
        _make_state_from_pos(
            torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
        ),
        RuntimeContext(),
    )
    separated_loss = ClusterSeparationLoss().evaluate(
        problem,
        _make_state_from_pos(
            torch.tensor([[0.0, 0.0], [0.0, 1.0], [40.0, 0.0], [40.0, 1.0]], dtype=torch.float32)
        ),
        RuntimeContext(),
    )

    assert overlapping_loss.item() > 0.0
    assert separated_loss.item() == pytest.approx(0.0, abs=1.0e-6)


def test_cluster_separation_loss_ignores_parent_child_pairs() -> None:
    """Parent and child clusters should not repel each other in the separation term."""

    problem = LayoutProblem(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=4,
        node_sizes=torch.full((4, 2), 2.0, dtype=torch.float32),
        clusters={"parent": [0, 3], "child": [1, 2]},
        cluster_parents={"parent": None, "child": "parent"},
    )

    loss = ClusterSeparationLoss().evaluate(
        problem,
        _make_state_from_pos(
            torch.tensor([[-1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
        ),
        RuntimeContext(),
    )

    assert loss.item() == pytest.approx(0.0, abs=1.0e-6)


def test_cluster_containment_loss_requires_child_bbox_inside_parent_bbox() -> None:
    """Cluster containment should distinguish contained and escaped child clusters."""

    problem = LayoutProblem(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=4,
        node_sizes=torch.full((4, 2), 2.0, dtype=torch.float32),
        clusters={"parent": [0, 3], "child": [1, 2]},
        cluster_parents={"parent": None, "child": "parent"},
    )
    inside_loss = ClusterContainmentLoss().evaluate(
        problem,
        _make_state_from_pos(
            torch.tensor([[-20.0, 0.0], [0.0, 0.0], [0.0, 1.0], [20.0, 0.0]], dtype=torch.float32)
        ),
        RuntimeContext(),
    )
    outside_loss = ClusterContainmentLoss().evaluate(
        problem,
        _make_state_from_pos(
            torch.tensor([[-1.0, 0.0], [0.0, 0.0], [20.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        ),
        RuntimeContext(),
    )

    assert inside_loss.item() == pytest.approx(0.0, abs=1.0e-6)
    assert outside_loss.item() > 0.0


def test_cluster_containment_loss_is_zero_without_hierarchy() -> None:
    """Containment should be disabled when parent metadata is absent."""

    problem = LayoutProblem(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=4,
        node_sizes=torch.full((4, 2), 2.0, dtype=torch.float32),
        clusters={"parent": [0, 3], "child": [1, 2]},
    )

    loss = ClusterContainmentLoss().evaluate(
        problem,
        _make_state_from_pos(
            torch.tensor([[-1.0, 0.0], [0.0, 0.0], [20.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        ),
        RuntimeContext(),
    )

    assert loss.item() == pytest.approx(0.0, abs=1.0e-6)


def test_cluster_separation_loss_grad_routes_to_one_row_per_cluster() -> None:
    """With multiple cluster members tied at the bbox boundary, the gradient
    must flow to exactly ONE row per cluster per axis -- matching the legacy
    ``pos[idx].min(dim=0)`` first-occurrence semantics. Splitting gradient
    across tied rows would make the cluster losses behave subtly differently
    from the pre-vectorization implementation under highly symmetric inputs.
    """

    node_sizes = torch.full((12, 2), 4.0, dtype=torch.float32)
    # Two clusters whose bboxes overlap (-> nonzero separation loss) AND
    # whose min/max corners are duplicated within the cluster.
    pos_tensor = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [10.0, 10.0],
            [10.0, 10.0],
            [5.0, 5.0],
            [5.0, 5.0],
            [5.0, 5.0],
            [5.0, 5.0],
            [15.0, 15.0],
            [15.0, 15.0],
            [10.0, 10.0],
            [10.0, 10.0],
        ],
        dtype=torch.float32,
    )

    state = _make_state_from_pos(pos_tensor)
    problem = LayoutProblem(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=12,
        node_sizes=node_sizes,
        clusters={"a": [0, 1, 2, 3, 4, 5], "b": [6, 7, 8, 9, 10, 11]},
        cluster_parents={"a": None, "b": None},
    )
    loss = ClusterSeparationLoss().evaluate(problem, state, RuntimeContext())
    loss.backward()

    # Tie semantics: lowest row id wins. Cluster A's max-x and max-y
    # boundary is held by rows 2 and 3 (tied at [10, 10]); first-match
    # rule should put gradient on row 2 only. Cluster B's min-x and
    # min-y boundary is held by rows 6 and 7 (tied at [5, 5]); first
    # match should put gradient on row 6 only.
    grad_per_row = state.pos.grad.abs().sum(dim=1)
    nonzero_rows = grad_per_row.nonzero().squeeze(-1).tolist()
    assert nonzero_rows == [2, 6], f"expected gradient on rows [2, 6] only, got {nonzero_rows}"


def test_cluster_loss_cache_is_reused_across_steps_and_invalidated_on_change() -> None:
    """The per-pipeline-call cluster cache should be built ONCE per problem
    and reused across many ``evaluate`` calls (the optimizer steps), and it
    must be invalidated when the cluster dict identity changes (e.g. when
    a new layout problem starts in the same SolveState container).
    """
    from dagua.layout.losses import _ClusterCache

    node_sizes = torch.full((4, 2), 2.0, dtype=torch.float32)
    pos = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32
    ).requires_grad_(True)

    problem = LayoutProblem(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=4,
        node_sizes=node_sizes,
        clusters={"a": [0, 1], "b": [2, 3]},
        cluster_parents={"a": None, "b": None},
    )
    state = _make_state_from_pos(pos)
    op = ClusterSeparationLoss()

    op.evaluate(problem, state, RuntimeContext())
    cached_first = state.extras.get("_cluster_cache")
    assert cached_first is not None
    assert isinstance(cached_first[1], _ClusterCache)

    # Re-evaluate -- cache identity must NOT change between steps.
    op.evaluate(problem, state, RuntimeContext())
    cached_second = state.extras.get("_cluster_cache")
    assert cached_second[1] is cached_first[1], (
        "cluster cache rebuilt on every step -- the per-call cache is not being reused"
    )

    # Hand the same SolveState to a NEW problem (simulating a new layout()
    # call that didn't reset extras). Cache MUST rebuild.
    problem_b = LayoutProblem(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=4,
        node_sizes=node_sizes,
        clusters={"x": [0, 2], "y": [1, 3]},
        cluster_parents={"x": None, "y": None},
    )
    op.evaluate(problem_b, state, RuntimeContext())
    cached_after = state.extras.get("_cluster_cache")
    assert cached_after[1] is not cached_first[1], (
        "cluster cache survived a problem.clusters identity change -- stale cache risk"
    )
    assert cached_after[1].num_clusters == 2


def test_spacing_consistency_loss_prefers_uniform_gaps() -> None:
    """Spacing consistency should be near zero for uniform same-layer gaps."""

    layers = torch.tensor([0, 0, 0], dtype=torch.long)
    problem = LayoutProblem(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=3,
        node_sizes=torch.full((3, 2), 10.0, dtype=torch.float32),
    )
    uniform_loss = SpacingConsistencyLoss().evaluate(
        problem,
        _make_state_from_pos(
            torch.tensor([[0.0, 0.0], [35.0, 0.0], [70.0, 0.0]], dtype=torch.float32),
            layers=layers,
        ),
        RuntimeContext(),
    )
    irregular_loss = SpacingConsistencyLoss().evaluate(
        problem,
        _make_state_from_pos(
            torch.tensor([[0.0, 0.0], [20.0, 0.0], [70.0, 0.0]], dtype=torch.float32),
            layers=layers,
        ),
        RuntimeContext(),
    )

    assert uniform_loss.item() == pytest.approx(0.0, abs=1.0e-6)
    assert irregular_loss.item() > 0.0


def test_spacing_consistency_loss_target_gap_config_changes_penalty() -> None:
    """Changing the target gap should change the spacing penalty for the same geometry."""

    layers = torch.tensor([0, 0, 0], dtype=torch.long)
    problem = LayoutProblem(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=3,
        node_sizes=torch.full((3, 2), 10.0, dtype=torch.float32),
    )
    state = _make_state_from_pos(
        torch.tensor([[0.0, 0.0], [35.0, 0.0], [70.0, 0.0]], dtype=torch.float32),
        layers=layers,
    )
    matched_loss = SpacingConsistencyLoss().evaluate(problem, state, RuntimeContext())
    mismatched_loss = SpacingConsistencyLoss(
        SpacingConsistencyLossConfig(target_gap=10.0),
    ).evaluate(
        problem,
        _make_state_from_pos(
            torch.tensor([[0.0, 0.0], [35.0, 0.0], [70.0, 0.0]], dtype=torch.float32),
            layers=layers,
        ),
        RuntimeContext(),
    )

    assert matched_loss.item() < mismatched_loss.item()


def test_spacing_consistency_loss_is_zero_without_layer_index() -> None:
    """Spacing consistency needs a layer index and should otherwise return zero."""

    problem = LayoutProblem(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=3,
        node_sizes=torch.full((3, 2), 10.0, dtype=torch.float32),
    )

    loss = SpacingConsistencyLoss().evaluate(
        problem,
        SolveState(
            pos=torch.tensor(
                [[0.0, 0.0], [20.0, 0.0], [70.0, 0.0]], dtype=torch.float32, requires_grad=True
            )
        ),
        RuntimeContext(),
    )

    assert loss.item() == pytest.approx(0.0, abs=1.0e-6)


def test_fanout_distribution_loss_prefers_even_angular_fans() -> None:
    """Evenly spaced fanout should score better than a skewed fan."""

    problem = LayoutProblem(
        edge_index=torch.tensor([[0, 0, 0, 0, 0], [1, 2, 3, 4, 5]], dtype=torch.long),
        num_nodes=6,
    )
    even_loss = FanoutDistributionLoss().evaluate(
        problem,
        _make_state_from_pos(
            torch.tensor(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.3090, 0.9511],
                    [-0.8090, 0.5878],
                    [-0.8090, -0.5878],
                    [0.3090, -0.9511],
                ],
                dtype=torch.float32,
            ),
        ),
        RuntimeContext(),
    )
    skewed_loss = FanoutDistributionLoss().evaluate(
        problem,
        _make_state_from_pos(
            torch.tensor(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [2.0, 0.0],
                    [3.0, 0.0],
                    [4.0, 0.0],
                    [5.0, 0.0],
                ],
                dtype=torch.float32,
            ),
        ),
        RuntimeContext(),
    )

    assert even_loss.item() < 1.0e-4
    assert skewed_loss.item() > even_loss.item()


def test_fanout_distribution_loss_degree_threshold_controls_activation() -> None:
    """Raising the degree threshold above the hub degree should disable the loss."""

    problem = LayoutProblem(
        edge_index=torch.tensor([[0, 0, 0, 0, 0], [1, 2, 3, 4, 5]], dtype=torch.long),
        num_nodes=6,
    )
    pos = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [5.0, 0.0],
        ],
        dtype=torch.float32,
    )
    active_loss = FanoutDistributionLoss(FanoutDistributionLossConfig(degree_threshold=5)).evaluate(
        problem,
        _make_state_from_pos(pos),
        RuntimeContext(),
    )
    disabled_loss = FanoutDistributionLoss(
        FanoutDistributionLossConfig(degree_threshold=6)
    ).evaluate(
        problem,
        _make_state_from_pos(pos),
        RuntimeContext(),
    )

    assert active_loss.item() > 0.0
    assert disabled_loss.item() == pytest.approx(0.0, abs=1.0e-6)


def test_fanout_distribution_loss_is_zero_without_edges() -> None:
    """Fanout distribution should vanish when there are no outgoing edges to examine."""

    problem = LayoutProblem(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=2)

    loss = FanoutDistributionLoss().evaluate(
        problem,
        _make_state_from_pos(torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)),
        RuntimeContext(),
    )

    assert loss.item() == pytest.approx(0.0, abs=1.0e-6)


@pytest.mark.parametrize(
    ("pos", "expected"),
    [
        (
            torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.float32),
            pytest.approx(0.0, abs=1.0e-6),
        ),
        (
            torch.tensor([[10.0, 1.0], [0.0, 0.0]], dtype=torch.float32),
            pytest.approx(100.0, rel=1.0e-5),
        ),
        (
            torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
            pytest.approx(0.0, abs=1.0e-6),
        ),
    ],
    ids=["vertical_back_edge", "wide_back_edge", "forward_edge"],
)
def test_back_edge_compactness_loss_cases(
    pos: torch.Tensor,
    expected: pytest.ApproxBase,
) -> None:
    """Back-edge compactness should only penalize horizontal span on actual back edges."""

    problem = LayoutProblem(edge_index=torch.tensor([[0], [1]], dtype=torch.long), num_nodes=2)

    loss = BackEdgeCompactnessLoss().evaluate(problem, _make_state_from_pos(pos), RuntimeContext())

    assert loss.item() == expected


def test_position_pin_loss_is_zero_at_target() -> None:
    """Position pinning should vanish once the pinned node reaches its target."""

    flex = FlexConstraints(
        pin_indices=torch.tensor([0], dtype=torch.long),
        pin_targets=torch.tensor([[5.0, 5.0]], dtype=torch.float32),
        pin_weights=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        soft_pin_mask=torch.tensor([[True, True]], dtype=torch.bool),
        hard_pin_mask=torch.zeros((1, 2), dtype=torch.bool),
    )
    problem = LayoutProblem(
        edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=1, flex=flex
    )

    loss = PositionPinLoss().evaluate(
        problem,
        _make_state_from_pos(torch.tensor([[5.0, 5.0]], dtype=torch.float32)),
        RuntimeContext(),
    )

    assert loss.item() == pytest.approx(0.0, abs=1.0e-6)


def test_position_pin_loss_is_positive_away_from_target() -> None:
    """Position pinning should penalize displacement from the requested pin target."""

    flex = FlexConstraints(
        pin_indices=torch.tensor([0], dtype=torch.long),
        pin_targets=torch.tensor([[5.0, 5.0]], dtype=torch.float32),
        pin_weights=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        soft_pin_mask=torch.tensor([[True, True]], dtype=torch.bool),
        hard_pin_mask=torch.zeros((1, 2), dtype=torch.bool),
    )
    problem = LayoutProblem(
        edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=1, flex=flex
    )

    loss = PositionPinLoss().evaluate(
        problem,
        _make_state_from_pos(torch.tensor([[0.0, 0.0]], dtype=torch.float32)),
        RuntimeContext(),
    )

    assert loss.item() > 0.0


def test_position_pin_loss_is_zero_without_pin_constraints() -> None:
    """Position pinning should be disabled when flex pin metadata is missing."""

    problem = LayoutProblem(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=1)

    loss = PositionPinLoss().evaluate(
        problem,
        _make_state_from_pos(torch.tensor([[0.0, 0.0]], dtype=torch.float32)),
        RuntimeContext(),
    )

    assert loss.item() == pytest.approx(0.0, abs=1.0e-6)


def test_alignment_loss_is_zero_for_aligned_nodes() -> None:
    """Alignment should vanish when all nodes share the constrained coordinate."""

    problem = LayoutProblem(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=3,
        flex=FlexConstraints(align_groups=[(torch.tensor([0, 1, 2], dtype=torch.long), 2.0, 0)]),
    )

    loss = AlignmentLoss().evaluate(
        problem,
        _make_state_from_pos(
            torch.tensor([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]], dtype=torch.float32)
        ),
        RuntimeContext(),
    )

    assert loss.item() == pytest.approx(0.0, abs=1.0e-6)


def test_alignment_loss_is_positive_for_misaligned_nodes() -> None:
    """Alignment should penalize variance along the constrained axis."""

    problem = LayoutProblem(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=3,
        flex=FlexConstraints(align_groups=[(torch.tensor([0, 1, 2], dtype=torch.long), 2.0, 0)]),
    )

    loss = AlignmentLoss().evaluate(
        problem,
        _make_state_from_pos(
            torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=torch.float32)
        ),
        RuntimeContext(),
    )

    assert loss.item() > 0.0


def test_alignment_loss_is_zero_without_groups() -> None:
    """Alignment should vanish when no alignment groups are configured."""

    problem = LayoutProblem(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=3)

    loss = AlignmentLoss().evaluate(
        problem,
        _make_state_from_pos(
            torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=torch.float32)
        ),
        RuntimeContext(),
    )

    assert loss.item() == pytest.approx(0.0, abs=1.0e-6)


def test_flex_spacing_loss_matches_reference_delegate() -> None:
    """Flex spacing should exactly delegate to the reference flex-spacing helper."""

    layers = torch.tensor([0, 0, 0], dtype=torch.long)
    layer_index = build_layer_index(layers)
    node_sizes = torch.full((3, 2), 10.0, dtype=torch.float32)
    pos = torch.tensor([[0.0, 0.0], [20.0, 0.0], [70.0, 0.0]], dtype=torch.float32)
    problem = LayoutProblem(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=3,
        node_sizes=node_sizes,
        flex=FlexConstraints(flex_node_sep=25.0, flex_node_sep_weight=1.5),
    )
    state = _make_state_from_pos(pos, layers=layers)

    loss = FlexSpacingLoss().evaluate(problem, state, RuntimeContext())
    reference = reference_flex_spacing_loss(
        state.pos,
        node_sizes,
        layer_index,
        target_sep=25.0,
        weight=1.5,
    )

    assert loss.item() == pytest.approx(reference.item(), rel=1.0e-6)


def test_flex_spacing_loss_weight_changes_result() -> None:
    """Flex spacing should scale with the configured flex weight."""

    layers = torch.tensor([0, 0, 0], dtype=torch.long)
    pos = torch.tensor([[0.0, 0.0], [20.0, 0.0], [70.0, 0.0]], dtype=torch.float32)
    node_sizes = torch.full((3, 2), 10.0, dtype=torch.float32)
    low_weight_problem = LayoutProblem(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=3,
        node_sizes=node_sizes,
        flex=FlexConstraints(flex_node_sep=25.0, flex_node_sep_weight=0.5),
    )
    high_weight_problem = LayoutProblem(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=3,
        node_sizes=node_sizes,
        flex=FlexConstraints(flex_node_sep=25.0, flex_node_sep_weight=2.0),
    )
    low_weight_loss = FlexSpacingLoss().evaluate(
        low_weight_problem,
        _make_state_from_pos(pos, layers=layers),
        RuntimeContext(),
    )
    high_weight_loss = FlexSpacingLoss().evaluate(
        high_weight_problem,
        _make_state_from_pos(pos, layers=layers),
        RuntimeContext(),
    )

    assert high_weight_loss.item() > low_weight_loss.item()


def test_flex_spacing_loss_is_zero_without_flex_data() -> None:
    """Flex spacing should return zero when the problem does not define flex spacing."""

    layers = torch.tensor([0, 0, 0], dtype=torch.long)
    problem = LayoutProblem(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=3,
        node_sizes=torch.full((3, 2), 10.0, dtype=torch.float32),
    )

    loss = FlexSpacingLoss().evaluate(
        problem,
        _make_state_from_pos(
            torch.tensor([[0.0, 0.0], [20.0, 0.0], [70.0, 0.0]], dtype=torch.float32), layers=layers
        ),
        RuntimeContext(),
    )

    assert loss.item() == pytest.approx(0.0, abs=1.0e-6)


def test_loss_group_combined_mode_supports_all_engine_losses() -> None:
    """A combined-mode loss group should evaluate and backpropagate all engine losses together."""

    problem = _make_problem()
    state = _make_state()

    LossGroup(losses=_all_losses(), backward_mode="combined").apply(
        problem, state, RuntimeContext()
    )

    assert state.prev_loss > 0.0
    assert state.pos is not None
    assert state.pos.grad is not None
    assert torch.isfinite(state.pos.grad).all()


def test_loss_group_combined_and_per_loss_modes_share_gradient_direction() -> None:
    """Combined and per-loss backward modes should point in the same descent direction."""

    problem = _make_problem()
    combined_state = _make_state()
    per_loss_state = _make_state()
    losses = _all_losses()

    LossGroup(losses=losses, backward_mode="combined").apply(
        problem, combined_state, RuntimeContext()
    )
    LossGroup(losses=losses, backward_mode="per_loss").apply(
        problem, per_loss_state, RuntimeContext()
    )

    assert combined_state.pos is not None
    assert combined_state.pos.grad is not None
    assert per_loss_state.pos is not None
    assert per_loss_state.pos.grad is not None
    combined_grad = combined_state.pos.grad.flatten()
    per_loss_grad = per_loss_state.pos.grad.flatten()
    cosine = torch.dot(combined_grad, per_loss_grad) / (
        torch.linalg.norm(combined_grad) * torch.linalg.norm(per_loss_grad)
    )

    assert cosine.item() > 0.9999


def test_loss_group_uses_annealing_schedule_weights() -> None:
    """LossGroup should prefer annealed weights over loss default weights."""

    problem = LayoutProblem(
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        num_nodes=2,
        node_sizes=torch.full((2, 2), 10.0, dtype=torch.float32),
    )
    pos = torch.tensor([[0.0, 40.0], [4.0, 0.0]], dtype=torch.float32)
    dag_loss = DagOrderingLoss()
    attraction_loss = EdgeAttractionLoss()
    expected_dag = dag_loss.evaluate(problem, _make_state_from_pos(pos), RuntimeContext()).item()
    expected_attraction = attraction_loss.evaluate(
        problem,
        _make_state_from_pos(pos),
        RuntimeContext(),
    ).item()
    state = _make_state_from_pos(pos)
    state.annealing = AnnealingSchedule(
        current_weights={
            dag_loss.weight_key: 0.0,
            attraction_loss.weight_key: 7.0,
        },
    )

    LossGroup(losses=[dag_loss, attraction_loss], backward_mode="combined").apply(
        problem,
        state,
        RuntimeContext(),
    )

    assert state.prev_loss == pytest.approx(7.0 * expected_attraction, rel=1.0e-6)
    assert state.prev_loss != pytest.approx(
        dag_loss.default_weight * expected_dag
        + attraction_loss.default_weight * expected_attraction,
        rel=1.0e-6,
    )
