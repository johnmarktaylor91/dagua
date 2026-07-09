"""Tests for r80-S7 routing-quality deliverables: node-bbox avoidance,
port angular spread, and label-vs-edge-path avoidance."""

import pytest
import torch

from dagua.edges import (
    BezierCurve,
    _build_node_grid,
    _curve_polyline_samples,
    _curve_samples_hit_rect,
    _deflect_around_nodes,
    _label_path_crossings,
    _port_spread_bias_deg,
    _rotate_point_around,
    place_edge_labels,
    route_edges,
)
from dagua.graph import DaguaGraph
from dagua.metrics import edge_node_crossing_count, port_angular_resolution
from dagua.styles import EdgeStyle


class TestNodeBboxAvoidance:
    def test_route_edges_deflects_around_blocking_node(self) -> None:
        """A node sitting on a would-be-straight chord should get dodged."""
        pos = torch.tensor([[0.0, 0.0], [0.0, 150.0], [10.0, 70.0]])
        ei = torch.tensor([[0], [1]])
        ns = torch.tensor([[40.0, 20.0], [40.0, 20.0], [50.0, 30.0]])

        curves = route_edges(pos, ei, ns)
        rect = (10.0 - 25.0 - 4.0, 70.0 - 15.0 - 4.0, 10.0 + 25.0 + 4.0, 70.0 + 15.0 + 4.0)
        assert not _curve_samples_hit_rect(curves[0], rect)

    def test_avoid_nodes_default_true_reduces_edge_node_crossings(self) -> None:
        """Default routing should not regress edge-node crossings vs a naive
        straight chord through a densely-packed row of nodes."""
        # Source/target far apart vertically; several nodes dotted along the
        # straight path between them at different x offsets.
        positions = [[0.0, 0.0], [0.0, 300.0]]
        sizes = [[40.0, 20.0], [40.0, 20.0]]
        for i in range(1, 6):
            positions.append([5.0 * (i % 2), 50.0 * i])
            sizes.append([40.0, 24.0])
        pos = torch.tensor(positions)
        ns = torch.tensor(sizes)
        ei = torch.tensor([[0], [1]])

        curves = route_edges(pos, ei, ns)
        result = edge_node_crossing_count(curves, pos, ns, ei)
        assert result["edge_node_crossings"] == 0

    def test_avoid_nodes_opt_out_preserves_prior_behavior(self) -> None:
        """EdgeStyle.avoid_nodes=False should skip node deflection entirely."""
        g = DaguaGraph.from_edge_list([(0, 1)])
        g.add_node("blocker", label="blocker")
        g.edge_styles[0] = EdgeStyle(avoid_nodes=False)

        pos = torch.tensor([[0.0, 0.0], [0.0, 150.0], [10.0, 70.0]])
        ns = torch.tensor([[40.0, 20.0], [40.0, 20.0], [50.0, 30.0]])

        curves_off = route_edges(pos, g.edge_index, ns, graph=g)
        curves_on = route_edges(pos, g.edge_index, ns)  # avoid_nodes defaults True, no graph

        rect = (10.0 - 25.0 - 4.0, 70.0 - 15.0 - 4.0, 10.0 + 25.0 + 4.0, 70.0 + 15.0 + 4.0)
        # Opted out: deflection must be skipped, so the naive curve still
        # cuts through the blocker's box.
        assert _curve_samples_hit_rect(curves_off[0], rect)
        # Default (avoid_nodes=True): deflection clears it.
        assert not _curve_samples_hit_rect(curves_on[0], rect)

    def test_dense_neighborhood_fallback_never_hangs_and_returns_curve(self) -> None:
        """When no bounded attempt can clear a giant blocking node, the
        deflection helper must fall back to leaving the curve as-is rather
        than looping forever."""
        curve = BezierCurve((0.0, 10.0), (0.0, 34.0), (0.0, 66.0), (0.0, 90.0))
        grid = _build_node_grid([0.0, 0.0, 0.0], [0.0, 100.0, 50.0], 47.14)
        out = _deflect_around_nodes(
            curve,
            0,
            1,
            grid,
            47.14,
            [0.0, 0.0, 0.0],
            [0.0, 100.0, 50.0],
            [20.0, 20.0, 600.0],  # absurdly wide blocker spanning the whole chord
            [20.0, 20.0, 600.0],
            max_attempts=2,
        )
        assert isinstance(out, BezierCurve)  # returned, did not hang

    def test_endpoints_never_treated_as_blockers(self) -> None:
        """Source/target nodes themselves must never trigger self-deflection."""
        pos = torch.tensor([[0.0, 0.0], [0.0, 100.0]])
        ei = torch.tensor([[0], [1]])
        ns = torch.tensor([[200.0, 200.0], [200.0, 200.0]])
        curves = route_edges(pos, ei, ns)
        assert len(curves) == 1
        assert isinstance(curves[0], BezierCurve)

    def test_deflection_offset_never_exceeds_chord_fraction(self) -> None:
        """r80-S7b#1: the applied control-point push must stay within
        0.6x the chord length -- offsets comparable to the chord make the
        curve loop back on itself (lasso curls)."""
        curve = BezierCurve((0.0, 10.0), (0.0, 49.0), (0.0, 101.0), (0.0, 140.0))
        grid = _build_node_grid([0.0, 0.0, 10.0], [0.0, 150.0, 70.0], 47.14)
        out = _deflect_around_nodes(
            curve,
            0,
            1,
            grid,
            47.14,
            [0.0, 0.0, 10.0],
            [0.0, 150.0, 70.0],
            [40.0, 40.0, 50.0],
            [20.0, 20.0, 30.0],
        )
        chord = 130.0
        for orig, new in [(curve.cp1, out.cp1), (curve.cp2, out.cp2)]:
            push = ((new[0] - orig[0]) ** 2 + (new[1] - orig[1]) ** 2) ** 0.5
            assert push <= chord * 0.6 + 1e-9

    def test_short_edge_uncleariable_blocker_left_unchanged_no_lasso(self) -> None:
        """r80-S7b#1: a SHORT edge whose blocker cannot be cleared within
        the chord-scaled cap must be left unchanged (previously the 16x
        ladder blew the control points out to several times the chord,
        producing loop-back curls)."""
        # Chord length 40; blocker 60x60 dead-center: clearing needs a
        # push far beyond 0.6*40=24, so the deflector must give up.
        curve = BezierCurve((0.0, 0.0), (0.0, 12.0), (0.0, 28.0), (0.0, 40.0))
        grid = _build_node_grid([0.0, 0.0, 0.0], [0.0, 40.0, 20.0], 40.0)
        out = _deflect_around_nodes(
            curve,
            0,
            1,
            grid,
            40.0,
            [0.0, 0.0, 0.0],
            [0.0, 40.0, 20.0],
            [10.0, 10.0, 60.0],
            [10.0, 10.0, 60.0],
        )
        assert out.cp1 == curve.cp1
        assert out.cp2 == curve.cp2


class TestPortAngularSpread:
    def test_port_spread_bias_zero_for_single_port(self) -> None:
        assert _port_spread_bias_deg(0, 1) == 0.0

    def test_port_spread_bias_symmetric_and_ordered(self) -> None:
        low = _port_spread_bias_deg(0, 3)
        mid = _port_spread_bias_deg(1, 3)
        high = _port_spread_bias_deg(2, 3)
        assert low < mid < high
        assert low == pytest.approx(-high)
        assert mid == pytest.approx(0.0)

    def test_hub_node_out_edges_gain_angular_separation(self) -> None:
        """A node with several outgoing edges to distinct targets should
        show a nonzero minimum port angle (the pre-r80-S7 implementation
        always gave TB out-edges a purely-vertical initial tangent, which
        collapses this metric to 0 deg for any fan-out >= 2)."""
        g = DaguaGraph.from_edge_list([(0, 1), (0, 2), (0, 3), (0, 4)])
        pos = torch.tensor(
            [
                [100.0, 0.0],
                [0.0, 100.0],
                [70.0, 100.0],
                [130.0, 100.0],
                [200.0, 100.0],
            ]
        )
        ns = torch.tensor([[40.0, 20.0]] * 5)

        curves = route_edges(pos, g.edge_index, ns, direction="TB", graph=g)
        result = port_angular_resolution(curves, g.edge_index)
        assert result["port_angular_res_mean_deg"] > 0.0

    def test_bias_rotation_matches_manual_rotation(self) -> None:
        rotated = _rotate_point_around((10.0, 0.0), (0.0, 0.0), 90.0)
        assert rotated[0] == pytest.approx(0.0, abs=1e-6)
        assert rotated[1] == pytest.approx(10.0, abs=1e-6)

    def test_single_out_edge_keeps_straight_down_tangent(self) -> None:
        """A node with exactly one outgoing edge (total=1) must be
        unaffected -- there is nothing to spread against."""
        pos = torch.tensor([[0.0, 0.0], [0.0, 100.0]])
        ei = torch.tensor([[0], [1]])
        ns = torch.tensor([[40.0, 20.0], [40.0, 20.0]])
        curve = route_edges(pos, ei, ns, direction="TB")[0]
        dx, dy = curve.cp1[0] - curve.p0[0], curve.cp1[1] - curve.p0[1]
        assert dx == pytest.approx(0.0, abs=1e-6)
        assert dy > 0.0


class TestLabelPathAvoidance:
    def test_curve_polyline_samples_bezier(self) -> None:
        curve = BezierCurve((0.0, 0.0), (0.0, 33.0), (0.0, 66.0), (0.0, 100.0))
        samples = _curve_polyline_samples(curve, sample_count=5)
        assert len(samples) == 5
        assert samples[0] == pytest.approx((0.0, 0.0))
        assert samples[-1] == pytest.approx((0.0, 100.0))

    def test_curve_polyline_samples_waypoints(self) -> None:
        curve = BezierCurve(
            (0.0, 0.0),
            (0.0, 10.0),
            (10.0, 10.0),
            (10.0, 20.0),
            waypoints=((0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 20.0)),
        )
        samples = _curve_polyline_samples(curve)
        assert samples == [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 20.0)]

    def test_label_path_crossings_detects_other_edge(self) -> None:
        crossing_curve = BezierCurve((-50.0, 5.0), (-10.0, 5.0), (10.0, 5.0), (50.0, 5.0))
        owner_curve = BezierCurve((0.0, -50.0), (0.0, -10.0), (0.0, 10.0), (0.0, 50.0))
        # Dense sampling so the coarse-grid crossing check can't straddle a
        # thin label box between two consecutive samples.
        polylines = [
            _curve_polyline_samples(crossing_curve, sample_count=40),
            _curve_polyline_samples(owner_curve, sample_count=40),
        ]
        bboxes = []
        for poly in polylines:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            bboxes.append((min(xs), min(ys), max(xs), max(ys)))

        label_bbox = (-5.0, -5.0, 5.0, 5.0)
        # Owner is edge index 1 (the vertical curve); the OTHER edge (index
        # 0, horizontal) crosses the label box and must be counted.
        assert _label_path_crossings(label_bbox, 1, polylines, bboxes) == 1
        # A curve must never count itself as a crossing of its own label.
        self_only = [polylines[0]]
        self_bbox = [bboxes[0]]
        assert _label_path_crossings(label_bbox, 0, self_only, self_bbox) == 0

    def test_place_edge_labels_avoids_crossing_edge_path(self) -> None:
        """A label's naive anchor sitting on another edge's path should be
        nudged to a candidate that avoids the crossing where possible."""
        g = DaguaGraph.from_edge_list([("a", "b"), ("c", "d")])
        g.edge_labels[0] = "label"

        # Edge a->b is horizontal at y=0 (label anchors near its midpoint,
        # y=0). Edge c->d is vertical, crossing straight through x=0 -- if
        # the naive t=0.5 anchor for a->b sits at (0, 0) it is exactly on
        # c->d's path.
        pos = torch.tensor([[-60.0, 0.0], [60.0, 0.0], [0.0, -60.0], [0.0, 60.0]])
        ns = torch.tensor([[20.0, 10.0]] * 4)

        curves = route_edges(pos, g.edge_index, ns, direction="LR", graph=g)
        label_positions = place_edge_labels(curves, pos, ns, g.edge_labels, graph=g)

        assert label_positions[0] is not None
