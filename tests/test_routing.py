"""Tests for bezier edge routing."""

import pytest
import torch

from dagua.edges import route_edges, evaluate_bezier, BezierCurve


class TestRouteEdges:
    def test_basic_routing(self):
        pos = torch.tensor([[0.0, 0.0], [0.0, 100.0]])
        ei = torch.tensor([[0], [1]])
        ns = torch.tensor([[40.0, 20.0], [40.0, 20.0]])
        curves = route_edges(pos, ei, ns)
        assert len(curves) == 1
        assert isinstance(curves[0], BezierCurve)

    def test_empty_edges(self):
        pos = torch.tensor([[0.0, 0.0], [0.0, 100.0]])
        ei = torch.zeros(2, 0, dtype=torch.long)
        ns = torch.tensor([[40.0, 20.0], [40.0, 20.0]])
        curves = route_edges(pos, ei, ns)
        assert len(curves) == 0

    def test_multiple_edges(self):
        pos = torch.tensor([[0.0, 0.0], [0.0, 100.0], [50.0, 100.0]])
        ei = torch.tensor([[0, 0], [1, 2]])
        ns = torch.tensor([[40.0, 20.0]] * 3)
        curves = route_edges(pos, ei, ns)
        assert len(curves) == 2

    def test_curve_endpoints(self):
        pos = torch.tensor([[0.0, 0.0], [0.0, 100.0]])
        ei = torch.tensor([[0], [1]])
        ns = torch.tensor([[40.0, 20.0], [40.0, 20.0]])
        curves = route_edges(pos, ei, ns)
        c = curves[0]
        # Start should be near bottom of source node
        assert abs(c.p0[1] - 10.0) < 1.0  # sy + sh/2
        # End should be near top of target node
        assert abs(c.p1[1] - 90.0) < 1.0  # ty - th/2


class TestEvaluateBezier:
    def test_endpoints(self):
        curve = BezierCurve((0.0, 0.0), (0.0, 33.0), (0.0, 66.0), (0.0, 100.0))
        start = evaluate_bezier(curve, 0.0)
        end = evaluate_bezier(curve, 1.0)
        assert abs(start[0]) < 0.01
        assert abs(start[1]) < 0.01
        assert abs(end[0]) < 0.01
        assert abs(end[1] - 100.0) < 0.01

    def test_midpoint(self):
        curve = BezierCurve((0.0, 0.0), (0.0, 33.0), (0.0, 66.0), (0.0, 100.0))
        mid = evaluate_bezier(curve, 0.5)
        assert abs(mid[1] - 50.0) < 5.0  # approximately at midpoint
