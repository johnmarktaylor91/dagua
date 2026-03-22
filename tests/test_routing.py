"""Tests for bezier edge routing."""

import pytest
import torch

from dagua.edges import BezierCurve, evaluate_bezier, route_edges
from dagua.graph import DaguaGraph
from dagua.styles import EdgeStyle, NodeStyle


class TestRouteEdges:
    def test_basic_routing(self) -> None:
        """A single edge should produce one bezier curve."""
        pos = torch.tensor([[0.0, 0.0], [0.0, 100.0]])
        ei = torch.tensor([[0], [1]])
        ns = torch.tensor([[40.0, 20.0], [40.0, 20.0]])
        curves = route_edges(pos, ei, ns)
        assert len(curves) == 1
        assert isinstance(curves[0], BezierCurve)

    def test_empty_edges(self) -> None:
        """Routing should return an empty list when there are no edges."""
        pos = torch.tensor([[0.0, 0.0], [0.0, 100.0]])
        ei = torch.zeros(2, 0, dtype=torch.long)
        ns = torch.tensor([[40.0, 20.0], [40.0, 20.0]])
        curves = route_edges(pos, ei, ns)
        assert len(curves) == 0

    def test_multiple_edges(self) -> None:
        """Multiple edges should preserve one curve per edge."""
        pos = torch.tensor([[0.0, 0.0], [0.0, 100.0], [50.0, 100.0]])
        ei = torch.tensor([[0, 0], [1, 2]])
        ns = torch.tensor([[40.0, 20.0]] * 3)
        curves = route_edges(pos, ei, ns)
        assert len(curves) == 2

    def test_curve_endpoints(self) -> None:
        """Default TB routing should reverse ports for upward back-edges."""
        pos = torch.tensor([[0.0, 0.0], [0.0, 100.0]])
        ei = torch.tensor([[0], [1]])
        ns = torch.tensor([[40.0, 20.0], [40.0, 20.0]])
        curves = route_edges(pos, ei, ns)
        c = curves[0]
        # Start should be near bottom of source node
        assert abs(c.p0[1] - 10.0) < 1.0  # sy + sh/2
        # End should be near top of target node
        assert abs(c.p1[1] - 90.0) < 1.0  # ty - th/2

    def test_self_loop_routing(self) -> None:
        """Default TB self-loops should sit above the node with no NaN values."""
        pos = torch.tensor([[50.0, 50.0]])
        ei = torch.tensor([[0], [0]])  # self-loop
        ns = torch.tensor([[40.0, 20.0]])
        curves = route_edges(pos, ei, ns)
        assert len(curves) == 1
        c = curves[0]
        # No NaN in any control point
        for pt in [c.p0, c.cp1, c.cp2, c.p1]:
            assert not any(v != v for v in pt), f"NaN in control point: {pt}"
        # Start and end should be the same (closed loop)
        assert c.p0 == c.p1
        # TB loops anchor on the top edge and bulge upward (+y in matplotlib).
        assert c.p0 == pytest.approx((50.0, 60.0))
        assert c.cp1[1] > c.p0[1]
        assert c.cp2[1] > c.p0[1]

    def test_self_loop_evaluate_no_nan(self) -> None:
        """Evaluating a self-loop curve at various t should produce no NaN."""
        pos = torch.tensor([[50.0, 50.0]])
        ei = torch.tensor([[0], [0]])
        ns = torch.tensor([[40.0, 20.0]])
        curves = route_edges(pos, ei, ns)
        c = curves[0]
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            pt = evaluate_bezier(c, t)
            assert not any(v != v for v in pt), f"NaN at t={t}: {pt}"

    def test_cluster_routing_ignores_invalid_members(self) -> None:
        """Cluster-aware routing should ignore out-of-range cluster members."""
        g = DaguaGraph.from_edge_list([(0, 1)])
        g.node_styles[0] = NodeStyle(shape="ellipse")
        g.node_styles[1] = NodeStyle(shape="diamond")
        g.add_cluster("mixed", [0, 99], label="mixed", strict=False)
        g.compute_node_sizes()

        pos = torch.tensor([[0.0, 0.0], [40.0, 100.0]])
        curves = route_edges(pos, g.edge_index, g.node_sizes, graph=g)

        assert len(curves) == 1
        curve = curves[0]
        for pt in [curve.p0, curve.cp1, curve.cp2, curve.p1]:
            assert all(torch.isfinite(torch.tensor(pt))), f"non-finite control point: {pt}"

    @pytest.mark.parametrize(
        ("direction", "positions", "expected_start", "expected_end"),
        [
            ("TB", [[0.0, 100.0], [0.0, 0.0]], (0.0, 90.0), (0.0, 10.0)),
            ("BT", [[0.0, 0.0], [0.0, 100.0]], (0.0, 10.0), (0.0, 90.0)),
            ("LR", [[0.0, 0.0], [100.0, 0.0]], (20.0, 0.0), (80.0, 0.0)),
            ("RL", [[100.0, 0.0], [0.0, 0.0]], (80.0, 0.0), (20.0, 0.0)),
        ],
    )
    def test_ports_follow_layout_direction(
        self,
        direction: str,
        positions: list[list[float]],
        expected_start: tuple[float, float],
        expected_end: tuple[float, float],
    ) -> None:
        """Ports should attach to the node side that faces the flow direction."""
        pos = torch.tensor(positions)
        ei = torch.tensor([[0], [1]])
        ns = torch.tensor([[40.0, 20.0], [40.0, 20.0]])

        curve = route_edges(pos, ei, ns, direction=direction)[0]

        assert curve.p0 == pytest.approx(expected_start)
        assert curve.p1 == pytest.approx(expected_end)

    @pytest.mark.parametrize(
        ("direction", "positions", "expected_start", "expected_end"),
        [
            ("TB", [[0.0, 0.0], [0.0, 100.0]], (0.0, 10.0), (0.0, 90.0)),
            ("BT", [[0.0, 100.0], [0.0, 0.0]], (0.0, 90.0), (0.0, 10.0)),
            ("LR", [[100.0, 0.0], [0.0, 0.0]], (80.0, 0.0), (20.0, 0.0)),
            ("RL", [[0.0, 0.0], [100.0, 0.0]], (20.0, 0.0), (80.0, 0.0)),
        ],
    )
    def test_back_edges_reverse_ports(
        self,
        direction: str,
        positions: list[list[float]],
        expected_start: tuple[float, float],
        expected_end: tuple[float, float],
    ) -> None:
        """Back-edges should flip to the opposite node sides before routing."""
        pos = torch.tensor(positions)
        ei = torch.tensor([[0], [1]])
        ns = torch.tensor([[40.0, 20.0], [40.0, 20.0]])

        curve = route_edges(pos, ei, ns, direction=direction)[0]

        assert curve.p0 == pytest.approx(expected_start)
        assert curve.p1 == pytest.approx(expected_end)

    def test_center_port_style_uses_node_centers_for_vertical_flow(self) -> None:
        """Center ports should ignore distributed ranks on vertical layouts."""
        g = DaguaGraph.from_edge_list([(0, 1), (0, 2)])
        g.edge_styles[0] = EdgeStyle(port_style="center", routing="straight")
        g.edge_styles[1] = EdgeStyle(port_style="center", routing="straight")

        pos = torch.tensor([[50.0, 100.0], [0.0, 0.0], [100.0, 0.0]])
        ns = torch.tensor([[40.0, 20.0], [40.0, 20.0], [40.0, 20.0]])

        curves = route_edges(pos, g.edge_index, ns, direction="TB", graph=g)

        assert curves[0].p0 == pytest.approx((50.0, 90.0))
        assert curves[1].p0 == pytest.approx((50.0, 90.0))
        assert curves[0].p1 == pytest.approx((0.0, 10.0))
        assert curves[1].p1 == pytest.approx((100.0, 10.0))

    def test_center_port_style_uses_node_centers_for_horizontal_flow(self) -> None:
        """Center ports should ignore distributed ranks on horizontal layouts."""
        g = DaguaGraph.from_edge_list([(0, 1), (0, 2)])
        g.edge_styles[0] = EdgeStyle(port_style="center", routing="straight")
        g.edge_styles[1] = EdgeStyle(port_style="center", routing="straight")

        pos = torch.tensor([[0.0, 50.0], [100.0, 0.0], [100.0, 100.0]])
        ns = torch.tensor([[40.0, 20.0], [40.0, 20.0], [40.0, 20.0]])

        curves = route_edges(pos, g.edge_index, ns, direction="LR", graph=g)

        assert curves[0].p0 == pytest.approx((20.0, 50.0))
        assert curves[1].p0 == pytest.approx((20.0, 50.0))
        assert curves[0].p1 == pytest.approx((80.0, 0.0))
        assert curves[1].p1 == pytest.approx((80.0, 100.0))

    @pytest.mark.parametrize(
        ("direction", "expected_anchor", "expected_outward"),
        [
            ("TB", (50.0, 60.0), (True, True)),
            ("BT", (50.0, 40.0), (False, False)),
            ("LR", (30.0, 50.0), (False, False)),
            ("RL", (70.0, 50.0), (True, True)),
        ],
    )
    def test_self_loops_follow_layout_direction(
        self,
        direction: str,
        expected_anchor: tuple[float, float],
        expected_outward: tuple[bool, bool],
    ) -> None:
        """Self-loops should anchor on the layout-facing side of the node."""
        pos = torch.tensor([[50.0, 50.0]])
        ei = torch.tensor([[0], [0]])
        ns = torch.tensor([[40.0, 20.0]])

        curve = route_edges(pos, ei, ns, direction=direction)[0]

        assert curve.p0 == pytest.approx(expected_anchor)
        if direction in ("TB", "BT"):
            assert (curve.cp1[1] > curve.p0[1], curve.cp2[1] > curve.p0[1]) == expected_outward
        else:
            assert (curve.cp1[0] > curve.p0[0], curve.cp2[0] > curve.p0[0]) == expected_outward

    def test_tb_self_loop_is_taller_than_it_is_wide(self) -> None:
        """Top-bottom self-loops should read as node-owned top loops, not side arcs."""
        pos = torch.tensor([[50.0, 50.0]])
        ei = torch.tensor([[0], [0]])
        ns = torch.tensor([[40.0, 20.0]])

        curve = route_edges(pos, ei, ns, direction="TB")[0]
        vertical_rise = curve.cp1[1] - curve.p0[1]
        horizontal_offset = abs(curve.cp1[0] - curve.p0[0])

        assert curve.p0 == pytest.approx((50.0, 60.0))
        assert curve.cp1[1] == pytest.approx(curve.cp2[1])
        assert vertical_rise > horizontal_offset


class TestEvaluateBezier:
    def test_endpoints(self) -> None:
        """Bezier evaluation should match the endpoints at t=0 and t=1."""
        curve = BezierCurve((0.0, 0.0), (0.0, 33.0), (0.0, 66.0), (0.0, 100.0))
        start = evaluate_bezier(curve, 0.0)
        end = evaluate_bezier(curve, 1.0)
        assert abs(start[0]) < 0.01
        assert abs(start[1]) < 0.01
        assert abs(end[0]) < 0.01
        assert abs(end[1] - 100.0) < 0.01

    def test_midpoint(self) -> None:
        """Bezier midpoint should remain near the geometric midpoint."""
        curve = BezierCurve((0.0, 0.0), (0.0, 33.0), (0.0, 66.0), (0.0, 100.0))
        mid = evaluate_bezier(curve, 0.5)
        assert abs(mid[1] - 50.0) < 5.0  # approximately at midpoint
