"""Tests for edge-routing ops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pytest
import torch

from dagua.edges import BezierCurve
from dagua.layout import edge_optimization as edge_opt
from dagua.layout.ops import edge_route as edge_route_ops
from dagua.layout.ops.edge_route import (
    BezierControlPointOpt,
    BezierControlPointOptConfig,
    ReconstructEdgeRoutes,
    _route_to_bezier,
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


def _quality_loss(
    routes: Sequence[torch.Tensor | BezierCurve],
    problem: LayoutProblem,
    positions: torch.Tensor,
    config: BezierControlPointOptConfig,
) -> float:
    """Evaluate the edge-optimization objective for a set of routes.

    Parameters
    ----------
    routes : sequence[torch.Tensor | BezierCurve]
        Route descriptions to score.
    problem : LayoutProblem
        Graph topology and node sizes.
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    config : BezierControlPointOptConfig
        Objective weights.

    Returns
    -------
    float
        Weighted scalar objective value.
    """
    curves = [_route_to_bezier(route) for route in routes]
    edge_count = len(curves)
    endpoints = torch.zeros((edge_count, 2, 2), dtype=torch.float32)
    control_points = torch.zeros((edge_count, 2, 2), dtype=torch.float32)
    for index, curve in enumerate(curves):
        endpoints[index, 0] = torch.tensor(curve.p0, dtype=torch.float32)
        endpoints[index, 1] = torch.tensor(curve.p1, dtype=torch.float32)
        control_points[index, 0] = torch.tensor(curve.cp1, dtype=torch.float32)
        control_points[index, 1] = torch.tensor(curve.cp2, dtype=torch.float32)

    total_loss = torch.tensor(0.0, dtype=torch.float32)
    if edge_count == 0:
        return 0.0

    t_samples = torch.linspace(0.0, 1.0, 10).unsqueeze(0)
    points = edge_opt._evaluate_bezier_batch(endpoints, control_points, t_samples)
    sources = problem.edge_index[0].tolist()
    targets = problem.edge_index[1].tolist()

    if config.w_edge_crossing > 0.0 and edge_count > 1:
        total_loss = total_loss + config.w_edge_crossing * edge_opt._edge_crossing_loss(
            points,
            edge_count,
        )
    if config.w_edge_node_crossing > 0.0:
        total_loss = total_loss + config.w_edge_node_crossing * edge_opt._edge_node_crossing_loss(
            points,
            positions.float(),
            problem.node_sizes.float(),
            sources,
            targets,
            edge_count,
        )
    if config.w_edge_angular_res > 0.0:
        angular_loss = edge_opt._port_angular_resolution_loss(
            endpoints,
            control_points,
            sources,
            targets,
        )
        total_loss = total_loss + (config.w_edge_angular_res * angular_loss)
    if config.w_edge_curvature_consistency > 0.0:
        curvature_consistency_loss = edge_opt._curvature_consistency_loss(
            endpoints,
            control_points,
            t_samples,
        )
        total_loss = total_loss + (config.w_edge_curvature_consistency * curvature_consistency_loss)
    if config.w_edge_curvature_penalty > 0.0:
        curvature_penalty_loss = edge_opt._curvature_penalty_loss(
            endpoints,
            control_points,
            t_samples,
        )
        total_loss = total_loss + (config.w_edge_curvature_penalty * curvature_penalty_loss)
    return float(total_loss.item())


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


def test_bezier_control_point_opt_improves_edge_quality_over_iterations() -> None:
    """BezierControlPointOpt should reduce the weighted routing objective on a crossing graph."""
    problem = _make_problem()
    positions = torch.tensor(
        [
            [0.0, 0.0],
            [3.0, 3.0],
            [0.0, 3.0],
            [3.0, 0.0],
        ],
        dtype=torch.float32,
    )
    routes = [
        torch.tensor([[0.0, 0.0], [3.0, 3.0]], dtype=torch.float32),
        torch.tensor([[0.0, 3.0], [3.0, 0.0]], dtype=torch.float32),
        torch.tensor([[0.0, 0.0], [3.0, 0.0]], dtype=torch.float32),
    ]
    config = BezierControlPointOptConfig(steps=12, w_edge_cluster_crossing=0.0)

    before = _quality_loss(routes, problem, positions, config)
    result = BezierControlPointOpt(config).apply(
        problem,
        SolveState(pos=positions, edge_routes=routes),
        RuntimeContext(),
    )
    after = _quality_loss(result.edge_routes, problem, positions, config)

    assert after < before


def test_bezier_control_point_opt_handles_zero_edges() -> None:
    """BezierControlPointOpt should return an empty route list for edgeless graphs."""
    problem = LayoutProblem(
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        num_nodes=2,
        node_sizes=torch.ones((2, 2), dtype=torch.float32),
    )
    state = SolveState(
        pos=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32),
        edge_routes=[],
    )

    result = BezierControlPointOpt(BezierControlPointOptConfig(steps=5)).apply(
        problem,
        state,
        RuntimeContext(),
    )

    assert result.edge_routes == []


def test_bezier_control_point_opt_handles_single_edge() -> None:
    """BezierControlPointOpt should preserve endpoints on single-edge inputs."""
    problem = LayoutProblem(
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        num_nodes=2,
        node_sizes=torch.ones((2, 2), dtype=torch.float32),
    )
    state = SolveState(
        pos=torch.tensor([[0.0, 0.0], [4.0, 2.0]], dtype=torch.float32),
        edge_routes=[torch.tensor([[0.0, 0.0], [4.0, 2.0]], dtype=torch.float32)],
    )

    result = BezierControlPointOpt(BezierControlPointOptConfig(steps=8)).apply(
        problem,
        state,
        RuntimeContext(),
    )

    assert isinstance(result.edge_routes[0], BezierCurve)
    assert result.edge_routes[0].p0 == (0.0, 0.0)
    assert result.edge_routes[0].p1 == (4.0, 2.0)


def test_bezier_control_point_opt_weight_configuration_affects_result() -> None:
    """BezierControlPointOpt should produce different controls under different weights."""
    problem = _make_problem()
    positions = torch.tensor(
        [
            [0.0, 0.0],
            [3.0, 3.0],
            [0.0, 3.0],
            [3.0, 0.0],
        ],
        dtype=torch.float32,
    )
    routes = [
        torch.tensor([[0.0, 0.0], [3.0, 3.0]], dtype=torch.float32),
        torch.tensor([[0.0, 3.0], [3.0, 0.0]], dtype=torch.float32),
        torch.tensor([[0.0, 0.0], [3.0, 0.0]], dtype=torch.float32),
    ]

    low_cross = BezierControlPointOpt(
        BezierControlPointOptConfig(steps=12, w_edge_crossing=0.0, w_edge_cluster_crossing=0.0)
    ).apply(problem, SolveState(pos=positions, edge_routes=routes), RuntimeContext())
    high_cross = BezierControlPointOpt(
        BezierControlPointOptConfig(steps=12, w_edge_crossing=20.0, w_edge_cluster_crossing=0.0)
    ).apply(problem, SolveState(pos=positions, edge_routes=routes), RuntimeContext())

    assert low_cross.edge_routes[0].cp1 != high_cross.edge_routes[0].cp1
    assert low_cross.edge_routes[0].cp2 != high_cross.edge_routes[0].cp2


def test_bezier_control_point_opt_passes_grad_clip_to_optimizer_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BezierControlPointOpt should forward the configured grad-clip value to optimize_edges."""
    observed_grad_clip: list[float] = []

    def _fake_optimize_edges(
        curves: list[BezierCurve],
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        node_sizes: torch.Tensor,
        config: BezierControlPointOptConfig,
        graph: object | None = None,
        trace: object | None = None,
    ) -> list[BezierCurve]:
        """Capture the forwarded config and return the input curves unchanged."""
        del positions, edge_index, node_sizes, graph, trace
        observed_grad_clip.append(config.grad_clip)
        return curves

    monkeypatch.setattr(edge_route_ops, "optimize_edges", _fake_optimize_edges)
    problem = LayoutProblem(
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        num_nodes=2,
        node_sizes=torch.ones((2, 2), dtype=torch.float32),
    )
    state = SolveState(
        pos=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32),
        edge_routes=[torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)],
    )

    BezierControlPointOpt(BezierControlPointOptConfig(steps=3, grad_clip=7.5)).apply(
        problem,
        state,
        RuntimeContext(),
    )

    assert observed_grad_clip == [7.5]


def test_reconstruct_edge_routes_handles_straight_edges_without_dummies() -> None:
    """ReconstructEdgeRoutes should handle edge paths that contain only endpoints."""
    problem = LayoutProblem(edge_index=torch.tensor([[0], [1]], dtype=torch.long), num_nodes=2)
    positions = torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
    state = SolveState(
        pos=positions,
        extras={"expanded_graph": _ExpandedGraphStub(edge_paths=[[0, 1]], num_nodes=2)},
    )

    result = ReconstructEdgeRoutes().apply(problem, state, RuntimeContext())

    torch.testing.assert_close(result.edge_routes[0], positions[[0, 1]])


def test_reconstruct_edge_routes_handles_reversed_edges() -> None:
    """ReconstructEdgeRoutes should flip routes flagged as reversed back-edges."""
    problem = LayoutProblem(edge_index=torch.tensor([[0], [1]], dtype=torch.long), num_nodes=2)
    positions = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=torch.float32)
    state = SolveState(
        pos=positions,
        back_edge_mask=torch.tensor([True]),
        extras={"expanded_graph": _ExpandedGraphStub(edge_paths=[[0, 2]], num_nodes=3)},
    )

    result = ReconstructEdgeRoutes().apply(problem, state, RuntimeContext())

    torch.testing.assert_close(result.edge_routes[0], positions[[2, 0]])


def test_reconstruct_edge_routes_requires_expanded_graph_metadata() -> None:
    """ReconstructEdgeRoutes should fail fast when expanded-graph metadata is missing."""
    with pytest.raises(ValueError, match="expanded_graph"):
        ReconstructEdgeRoutes().apply(
            LayoutProblem(edge_index=torch.tensor([[0], [1]], dtype=torch.long), num_nodes=2),
            SolveState(pos=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)),
            RuntimeContext(),
        )


def test_reconstruct_edge_routes_requires_expanded_graph_edge_paths() -> None:
    """ReconstructEdgeRoutes should reject expanded-graph objects without edge_paths."""
    state = SolveState(
        pos=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32),
        extras={"expanded_graph": object()},
    )

    with pytest.raises(ValueError, match="edge_paths"):
        ReconstructEdgeRoutes().apply(
            LayoutProblem(edge_index=torch.tensor([[0], [1]], dtype=torch.long), num_nodes=2),
            state,
            RuntimeContext(),
        )


def test_reconstruct_edge_routes_requires_positions_for_all_expanded_nodes() -> None:
    """ReconstructEdgeRoutes should reject incomplete expanded-node position tensors."""
    state = SolveState(
        pos=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32),
        extras={"expanded_graph": _ExpandedGraphStub(edge_paths=[[0, 1, 2]], num_nodes=3)},
    )

    with pytest.raises(ValueError, match="requires expanded positions"):
        ReconstructEdgeRoutes().apply(
            LayoutProblem(edge_index=torch.tensor([[0], [1]], dtype=torch.long), num_nodes=2),
            state,
            RuntimeContext(),
        )
