"""Tests for post-layout coordinate operations."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from dagua.layout.ops import LayoutProblem, Pipeline, Repeat, RuntimeContext, SolveState
from dagua.layout.ops.anneal import LinearCool, LinearCoolConfig
from dagua.layout.ops.force import (
    ApplyDisplacement,
    InverseDistanceRepulsion,
    UniformSpringAttraction,
    ZeroForces,
)
from dagua.layout.ops.init import RandomUniformInit
from dagua.layout.ops.postprocess import (
    CenterPositions,
    DirectionTransform,
    DirectionTransformConfig,
    NormalizePositions,
    NormalizePositionsConfig,
    ScalePositions,
    ScalePositionsConfig,
    SpreadFanoutChildren,
    SpreadFanoutChildrenConfig,
    StripDummyNodes,
)


@dataclass(frozen=True)
class _ExpandedGraphStub:
    """Minimal expanded graph payload for dummy-node stripping tests."""

    num_nodes: int


def _make_problem(
    num_nodes: int,
    edge_index: torch.Tensor | None = None,
    node_sizes: torch.Tensor | None = None,
) -> LayoutProblem:
    """Create a minimal layout problem for postprocess op tests.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    edge_index : torch.Tensor, optional
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]``.

    Returns
    -------
    LayoutProblem
        Minimal immutable layout problem.
    """
    if edge_index is None:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return LayoutProblem(edge_index=edge_index, num_nodes=num_nodes, node_sizes=node_sizes)


def test_center_positions_produces_zero_mean_output() -> None:
    """CenterPositions should subtract the coordinate-wise mean."""

    problem = _make_problem(num_nodes=3)
    state = SolveState(pos=torch.tensor([[1.0, 4.0], [3.0, 2.0], [5.0, 8.0]], dtype=torch.float32))

    result = CenterPositions().apply(problem, state, RuntimeContext())

    torch.testing.assert_close(result.pos.mean(dim=0), torch.zeros(2))


def test_scale_positions_max_abs_produces_unit_extent() -> None:
    """ScalePositions should normalize by the maximum absolute coordinate."""

    problem = _make_problem(num_nodes=3)
    state = SolveState(
        pos=torch.tensor([[2.0, -4.0], [1.0, 0.5], [-3.0, 2.0]], dtype=torch.float32)
    )

    result = ScalePositions(ScalePositionsConfig(method="max_abs", factor=1.0)).apply(
        problem,
        state,
        RuntimeContext(),
    )

    assert result.pos is not None
    assert pytest.approx(float(result.pos.abs().max().item()), rel=1e-6, abs=1e-6) == 1.0


def test_normalize_positions_on_ten_node_graph() -> None:
    """NormalizePositions should center and scale to the requested extent."""

    num_nodes = 10
    problem = _make_problem(num_nodes=num_nodes)
    state = SolveState(
        pos=torch.stack(
            [
                torch.arange(num_nodes, dtype=torch.float32),
                torch.linspace(-2.0, 3.0, steps=num_nodes),
            ],
            dim=1,
        )
    )

    result = NormalizePositions(NormalizePositionsConfig(extent_fn="sqrt_n_times_5")).apply(
        problem,
        state,
        RuntimeContext(),
    )

    assert result.pos is not None
    expected_extent = (num_nodes**0.5) * 5.0
    torch.testing.assert_close(result.pos.mean(dim=0), torch.zeros(2), atol=1e-5, rtol=0.0)
    assert (
        pytest.approx(float(result.pos.abs().max().item()), rel=1e-6, abs=1e-6) == expected_extent
    )


@pytest.mark.parametrize(
    ("direction", "expected"),
    [
        ("TB", torch.tensor([[1.0, 2.0], [-3.0, 4.0]], dtype=torch.float32)),
        ("BT", torch.tensor([[1.0, -2.0], [-3.0, -4.0]], dtype=torch.float32)),
        ("LR", torch.tensor([[2.0, 1.0], [4.0, -3.0]], dtype=torch.float32)),
        ("RL", torch.tensor([[-2.0, 1.0], [-4.0, -3.0]], dtype=torch.float32)),
    ],
)
def test_direction_transform_supports_all_four_directions(
    direction: str,
    expected: torch.Tensor,
) -> None:
    """DirectionTransform should match the documented TB/BT/LR/RL mapping."""

    problem = _make_problem(num_nodes=2)
    state = SolveState(pos=torch.tensor([[1.0, 2.0], [-3.0, 4.0]], dtype=torch.float32))

    result = DirectionTransform(DirectionTransformConfig(direction=direction)).apply(
        problem,
        state,
        RuntimeContext(),
    )

    assert result.pos is not None
    torch.testing.assert_close(result.pos, expected)


def test_strip_dummy_nodes_removes_extra_nodes() -> None:
    """StripDummyNodes should drop dummy-node coordinates from the tail."""

    problem = _make_problem(num_nodes=3)
    state = SolveState(
        pos=torch.tensor(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [9.0, 9.0], [10.0, 10.0]],
            dtype=torch.float32,
        ),
        extras={"expanded_graph": _ExpandedGraphStub(num_nodes=5)},
    )

    result = StripDummyNodes().apply(problem, state, RuntimeContext())

    assert result.pos is not None
    assert result.pos.shape == (3, 2)
    torch.testing.assert_close(
        result.pos,
        torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=torch.float32),
    )


def test_spread_fanout_children_redistributes_hub_children() -> None:
    """SpreadFanoutChildren should widen child x-positions around the hub."""

    edge_index = torch.tensor(
        [[0, 0, 0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6, 7, 8]],
        dtype=torch.long,
    )
    problem = _make_problem(num_nodes=9, edge_index=edge_index)
    initial_pos = torch.tensor(
        [
            [10.0, 0.0],
            [9.8, 1.0],
            [9.9, 1.0],
            [10.0, 1.0],
            [10.1, 1.0],
            [10.2, 1.0],
            [10.3, 1.0],
            [10.4, 1.0],
            [10.5, 1.0],
        ],
        dtype=torch.float32,
    )
    state = SolveState(
        pos=initial_pos.clone(),
        layers=torch.tensor([0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long),
    )

    result = SpreadFanoutChildren(SpreadFanoutChildrenConfig(hub_threshold=8, widening=1.5)).apply(
        problem, state, RuntimeContext()
    )

    assert result.pos is not None
    child_x = result.pos[1:, 0]
    assert pytest.approx(float(child_x.mean().item()), rel=1e-6, abs=1e-6) == 10.0
    assert torch.all(child_x[1:] > child_x[:-1])
    assert float(child_x[-1].item() - child_x[0].item()) > float(
        initial_pos[8, 0].item() - initial_pos[1, 0].item()
    )


def test_pipeline_random_init_center_and_scale() -> None:
    """A simple pipeline should produce centered, unit-extent positions."""

    problem = _make_problem(num_nodes=6)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1234)
    pipeline = Pipeline([RandomUniformInit(), CenterPositions(), ScalePositions()])

    result = pipeline.apply(problem, SolveState(), RuntimeContext(generator=generator))

    assert result.pos is not None
    torch.testing.assert_close(result.pos.mean(dim=0), torch.zeros(2), atol=1e-5, rtol=0.0)
    assert pytest.approx(float(result.pos.abs().max().item()), rel=1e-6, abs=1e-6) == 1.0
    assert result.ops_applied == [
        "random_uniform_init",
        "center_positions",
        "scale_positions",
    ]


def test_normalize_positions_uses_node_size_extent_fallback() -> None:
    """NormalizePositions should expand to the node-size-derived extent when larger."""

    problem = _make_problem(
        num_nodes=4,
        node_sizes=torch.full((4, 2), 10.0, dtype=torch.float32),
    )
    state = SolveState(
        pos=torch.tensor(
            [[0.0, 0.0], [1.0, 2.0], [3.0, 1.0], [2.0, 4.0]],
            dtype=torch.float32,
        )
    )

    result = NormalizePositions(
        NormalizePositionsConfig(extent_fn="sqrt_n_times_5", node_size_scale=2.0)
    ).apply(problem, state, RuntimeContext())

    assert result.pos is not None
    assert float(result.pos.abs().max().item()) == pytest.approx(40.0, rel=1.0e-6)


def test_normalize_positions_degenerate_layout_uses_deterministic_fallback() -> None:
    """NormalizePositions should generate a stable fallback when the span collapses."""

    problem = _make_problem(num_nodes=3)
    state = SolveState(pos=torch.ones((3, 2), dtype=torch.float32))

    result = NormalizePositions().apply(problem, state, RuntimeContext())

    assert result.pos is not None
    torch.testing.assert_close(result.pos[:, 1], torch.zeros(3), atol=1.0e-6, rtol=0.0)
    assert result.pos[:, 0].tolist() == pytest.approx([-8.6602545, 0.0, 8.6602545], rel=1.0e-6)


def test_full_fr_pipeline_centers_positions_after_repeat_loop() -> None:
    """A short FR-style pipeline should finish centered with the expected step count."""

    problem = _make_problem(
        num_nodes=3,
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(123)
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
                    LinearCool(LinearCoolConfig(rate=0.1)),
                ],
            ),
            CenterPositions(),
        ]
    )

    result = pipeline.apply(
        problem,
        SolveState(temperature=1.0),
        RuntimeContext(generator=generator),
    )

    assert result.pos is not None
    assert result.step == 10
    torch.testing.assert_close(result.pos.mean(dim=0), torch.zeros(2), atol=1.0e-5, rtol=0.0)
    assert float(result.temperature) == pytest.approx(0.0, abs=1.0e-12)
