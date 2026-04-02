"""Tests for engine loss operations."""

from __future__ import annotations

from typing import List, Optional, Tuple

import pytest
import torch

from dagua.layout.layers import build_layer_index
from dagua.layout.ops.base import LossGroup, LossOp
from dagua.layout.ops.loss_engine import (
    AlignmentLoss,
    BackEdgeCompactnessLoss,
    ClusterCompactnessLoss,
    ClusterContainmentLoss,
    ClusterSeparationLoss,
    CrossingLoss,
    DagOrderingLoss,
    EdgeAttractionLoss,
    EdgeLengthVarianceLoss,
    EdgeStraightnessLoss,
    FanoutDistributionLoss,
    FlexSpacingLoss,
    OverlapAvoidanceLoss,
    PositionPinLoss,
    RepulsionLoss,
    SpacingConsistencyLoss,
)
from dagua.layout.ops.state import FlexConstraints, LayoutProblem, RuntimeContext, SolveState


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
