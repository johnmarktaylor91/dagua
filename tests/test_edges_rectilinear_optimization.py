"""Tests for differentiable rectilinear edge-route optimization."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch

from dagua.config import LayoutConfig
from dagua.edges import BezierCurve, _compute_curve, route_edges
from dagua.eval.graphs import get_test_graphs
from dagua.layout.edge_optimization import (
    _manhattan_axis_penalty,
    _optimize_bezier_edges,
    optimize_edges,
)
from dagua.render.crossings import detect_crossings


def _curve(points: Sequence[Tuple[float, float]], routing: str = "ortho") -> BezierCurve:
    """Build a waypoint-backed test curve.

    Parameters
    ----------
    points : sequence[tuple[float, float]]
        Polyline points in draw order.
    routing : str, default="ortho"
        Routing mode to store on the curve.

    Returns
    -------
    BezierCurve
        Curve with explicit waypoint geometry.
    """
    return BezierCurve(
        p0=points[0],
        cp1=points[1] if len(points) > 2 else points[0],
        cp2=points[-2] if len(points) > 2 else points[-1],
        p1=points[-1],
        waypoints=tuple(points),
        routing=routing,
        direction="TB",
    )


def _segments_from_curves(curves: Sequence[BezierCurve]) -> torch.Tensor:
    """Convert waypoint curves to a segment tensor.

    Parameters
    ----------
    curves : sequence[BezierCurve]
        Curves with explicit waypoints.

    Returns
    -------
    torch.Tensor
        Segment tensor with shape ``[R, 2, 2]``.
    """
    segments = []
    for curve in curves:
        assert curve.waypoints is not None
        points = torch.tensor(curve.waypoints, dtype=torch.float32)
        segments.append(torch.stack([points[:-1], points[1:]], dim=1))
    return torch.cat(segments, dim=0)


def _rectilinear_config() -> LayoutConfig:
    """Return a fast config that isolates rectilinear crossing optimization.

    Returns
    -------
    LayoutConfig
        Configuration for deterministic unit tests.
    """
    return LayoutConfig(
        edge_opt_steps=100,
        edge_opt_lr=2.0,
        w_edge_node_crossing=0.0,
        w_edge_angular_res=0.0,
        w_edge_curvature_consistency=0.0,
        w_edge_curvature_penalty=0.0,
        w_edge_cluster_crossing=0.0,
    )


def _crossing_fixture(y_min: float, y_max: float) -> tuple[list[BezierCurve], torch.Tensor]:
    """Build a two-edge rectilinear crossing fixture.

    Parameters
    ----------
    y_min : float
        Lower y-coordinate of the vertical crossing edge.
    y_max : float
        Upper y-coordinate of the vertical crossing edge.

    Returns
    -------
    tuple[list[BezierCurve], torch.Tensor]
        Curves plus node positions matching their endpoints.
    """
    curves = [
        _curve([(0.0, 0.0), (0.0, 50.0), (100.0, 50.0), (100.0, 100.0)]),
        _curve([(50.0, y_min), (50.0, (y_min + y_max) / 2.0), (50.0, y_max)]),
    ]
    positions = torch.tensor(
        [[0.0, 0.0], [100.0, 100.0], [50.0, y_min], [50.0, y_max]],
        dtype=torch.float32,
    )
    return curves, positions


def test_manhattan_axis_penalty_zero_for_axis_aligned_segments() -> None:
    """Axis-aligned segments should have zero Manhattan-axis penalty."""
    segments = torch.tensor(
        [
            [[0.0, 0.0], [10.0, 0.0]],
            [[10.0, 0.0], [10.0, 20.0]],
        ],
        dtype=torch.float32,
    )

    assert _manhattan_axis_penalty(segments).item() == 0.0


def test_manhattan_axis_penalty_positive_for_diagonal_segments() -> None:
    """Diagonal segments should have positive Manhattan-axis penalty."""
    segments = torch.tensor([[[0.0, 0.0], [10.0, 10.0]]], dtype=torch.float32)

    assert _manhattan_axis_penalty(segments).item() > 0.0


def test_optimize_edges_ortho_keeps_axis_aligned_segments() -> None:
    """Orthogonal optimization should converge to axis-aligned segments."""
    curves, positions = _crossing_fixture(45.0, 55.0)
    edge_index = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
    node_sizes = torch.zeros((4, 2), dtype=torch.float32)

    optimized = optimize_edges(curves, positions, edge_index, node_sizes, _rectilinear_config())

    assert _manhattan_axis_penalty(_segments_from_curves(optimized)).item() < 1e-8


def test_optimize_edges_taxi_step_fraction_stays_clamped() -> None:
    """Taxi optimization should return valid learned step fractions."""
    curves = [
        _compute_curve(0.0, 0.0, 100.0, 100.0, "TB", "taxi", 0.4),
        _compute_curve(100.0, 0.0, 0.0, 100.0, "TB", "taxi", 0.4),
    ]
    positions = torch.tensor(
        [[0.0, 0.0], [100.0, 100.0], [100.0, 0.0], [0.0, 100.0]],
        dtype=torch.float32,
    )
    edge_index = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
    node_sizes = torch.zeros((4, 2), dtype=torch.float32)

    optimized = optimize_edges(curves, positions, edge_index, node_sizes, _rectilinear_config())

    for curve in optimized:
        assert curve.step_fraction is not None
        assert 0.1 <= float(curve.step_fraction) <= 0.9


def test_rectilinear_optimizer_reduces_crossings_on_two_synthetic_graphs() -> None:
    """Rectilinear optimization should reduce crossings on synthetic fixtures."""
    edge_index = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
    node_sizes = torch.zeros((4, 2), dtype=torch.float32)

    for y_min, y_max in ((45.0, 55.0), (47.0, 53.0)):
        curves, positions = _crossing_fixture(y_min, y_max)
        before = len(detect_crossings(curves, len(curves), min_distance=0.0))
        optimized = optimize_edges(curves, positions, edge_index, node_sizes, _rectilinear_config())
        after = len(detect_crossings(optimized, len(optimized), min_distance=0.0))

        assert after < before


def test_bezier_dispatch_matches_legacy_optimizer_on_random_dag_200() -> None:
    """All-bezier public dispatch should remain byte-identical to the legacy path."""
    graph = next(item.graph for item in get_test_graphs() if item.name == "random_dag_200")
    graph.compute_node_sizes()
    generator = torch.Generator().manual_seed(42)
    positions = torch.randn((graph.num_nodes, 2), generator=generator) * 100.0
    curves = route_edges(positions, graph.edge_index, graph.node_sizes, graph.direction, graph)
    config = LayoutConfig(edge_opt_steps=2, edge_opt_lr=0.1)

    torch.manual_seed(42)
    legacy = _optimize_bezier_edges(
        curves, positions, graph.edge_index, graph.node_sizes, config, graph
    )
    torch.manual_seed(42)
    dispatched = optimize_edges(
        curves, positions, graph.edge_index, graph.node_sizes, config, graph
    )

    legacy_controls = [(curve.p0, curve.cp1, curve.cp2, curve.p1) for curve in legacy]
    dispatched_controls = [(curve.p0, curve.cp1, curve.cp2, curve.p1) for curve in dispatched]
    assert dispatched_controls == legacy_controls
