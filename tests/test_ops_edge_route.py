"""Tests for edge-routing ops."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from dagua.edges import BezierCurve
from dagua.layout.ops.edge_route import (
    BezierControlPointOpt,
    BezierControlPointOptConfig,
    ReconstructEdgeRoutes,
)
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState


@dataclass(frozen=True)
class _ExpandedGraphStub:
    """Minimal expanded-graph payload for edge-route reconstruction tests."""

    edge_paths: list[list[int]]
    num_nodes: int


def _make_problem() -> LayoutProblem:
    """Create a small graph with three routed edges.

    Returns
    -------
    LayoutProblem
        Minimal layout problem with unit node sizes.
    """
    edge_index = torch.tensor([[0, 2, 0], [1, 3, 3]], dtype=torch.long)
    node_sizes = torch.ones((4, 2), dtype=torch.float32)
    return LayoutProblem(edge_index=edge_index, num_nodes=4, node_sizes=node_sizes)


def test_bezier_control_point_opt_optimizes_simple_three_edge_graph() -> None:
    """BezierControlPointOpt should return finite Bezier routes with moved controls."""

    problem = _make_problem()
    state = SolveState(
        pos=torch.tensor(
            [
                [0.0, 0.0],
                [3.0, 3.0],
                [0.0, 3.0],
                [3.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        edge_routes=[
            torch.tensor([[0.0, 0.0], [3.0, 3.0]], dtype=torch.float32),
            torch.tensor([[0.0, 3.0], [3.0, 0.0]], dtype=torch.float32),
            torch.tensor([[0.0, 0.0], [3.0, 0.0]], dtype=torch.float32),
        ],
    )

    result = BezierControlPointOpt(
        BezierControlPointOptConfig(
            lr=0.1,
            steps=12,
            w_edge_crossing=8.0,
            w_edge_node_crossing=5.0,
            w_edge_angular_res=1.0,
            w_edge_curvature_consistency=1.0,
            w_edge_curvature_penalty=0.25,
            w_edge_cluster_crossing=0.0,
        )
    ).apply(problem, state, RuntimeContext())

    assert isinstance(result.edge_routes, list)
    assert len(result.edge_routes) == 3
    assert all(isinstance(route, BezierCurve) for route in result.edge_routes)

    first_route = result.edge_routes[0]
    assert first_route is not None
    assert first_route.p0 == (0.0, 0.0)
    assert first_route.p1 == (3.0, 3.0)

    linear_cp1 = (1.0, 1.0)
    linear_cp2 = (2.0, 2.0)
    assert (
        abs(first_route.cp1[0] - linear_cp1[0]) > 1.0e-3
        or abs(first_route.cp1[1] - linear_cp1[1]) > 1.0e-3
    )
    assert (
        abs(first_route.cp2[0] - linear_cp2[0]) > 1.0e-3
        or abs(first_route.cp2[1] - linear_cp2[1]) > 1.0e-3
    )


def test_reconstruct_edge_routes_builds_polylines_from_dummy_node_chain() -> None:
    """ReconstructEdgeRoutes should rebuild ordered routes from edge_paths."""

    problem = LayoutProblem(
        edge_index=torch.tensor([[0, 1], [2, 2]], dtype=torch.long), num_nodes=3
    )
    expanded_positions = torch.tensor(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [4.0, 0.0],
            [1.0, 1.0],
            [3.0, 1.0],
        ],
        dtype=torch.float32,
    )
    state = SolveState(
        pos=expanded_positions,
        back_edge_mask=torch.tensor([False, True]),
        extras={
            "expanded_graph": _ExpandedGraphStub(
                edge_paths=[[0, 3, 4, 2], [1, 2]],
                num_nodes=5,
            )
        },
    )

    result = ReconstructEdgeRoutes().apply(problem, state, RuntimeContext())

    assert isinstance(result.edge_routes, list)
    assert len(result.edge_routes) == 2
    torch.testing.assert_close(result.edge_routes[0], expanded_positions[[0, 3, 4, 2]])
    torch.testing.assert_close(result.edge_routes[1], expanded_positions[[2, 1]])
