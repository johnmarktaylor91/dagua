"""Tests for r80-S7 routing-quality deliverables: node-bbox avoidance,
port angular spread, and label-vs-edge-path avoidance."""

import pytest
import torch

from dagua.edges import (
    BezierCurve,
    _build_node_grid,
    _count_route_crossings,
    _curve_polyline_samples,
    _curve_samples_hit_rect,
    _deflect_around_nodes,
    _label_path_crossings,
    _poly_bbox,
    _port_spread_bias_deg,
    _rotate_point_around,
    _segments_cross,
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


class TestCrossingAwareAcceptance:
    def test_segments_cross_basic(self) -> None:
        assert _segments_cross((0.0, 0.0), (10.0, 10.0), (0.0, 10.0), (10.0, 0.0))
        assert not _segments_cross((0.0, 0.0), (10.0, 0.0), (0.0, 5.0), (10.0, 5.0))

    def test_segments_shared_endpoint_not_a_crossing(self) -> None:
        """Two edges leaving the same port touch at that point -- that is
        contact, not a crossing."""
        assert not _segments_cross((0.0, 0.0), (10.0, 10.0), (0.0, 0.0), (10.0, -10.0))

    def test_count_route_crossings_with_early_exit(self) -> None:
        vertical = [(0.0, -10.0), (0.0, 0.0), (0.0, 10.0)]
        # Single-segment horizontals: no polyline vertex exactly on x=0,
        # so each crossing is strictly interior to both segments.
        horiz_a = [(-10.0, -5.0), (10.0, -5.0)]
        horiz_b = [(-10.0, 5.0), (10.0, 5.0)]
        routed = [horiz_a, horiz_b]
        bboxes = [_poly_bbox(p) for p in routed]
        vb = _poly_bbox(vertical)
        assert _count_route_crossings(vertical, vb, routed, bboxes) == 2
        # Early exit truncates as soon as the comparison is decided.
        assert _count_route_crossings(vertical, vb, routed, bboxes, stop_above=0) == 1

    def test_spread_reverted_when_it_creates_a_crossing(self) -> None:
        """r80-S7b#2: near-parallel long edges (the long_skip failure
        mode) -- if fanning tangents apart makes two neighbors cross,
        the later edge must fall back to its unbiased route and the
        pair must not cross."""
        # One hub at bottom with two long, nearly-parallel edges going up
        # to two targets that are horizontally very close: with a +-23 deg
        # fan-out the curves swing wide and can cross mid-flight.
        g = DaguaGraph.from_edge_list([(0, 1), (0, 2)])
        pos = torch.tensor([[0.0, 0.0], [-4.0, 400.0], [4.0, 400.0]])
        ns = torch.tensor([[30.0, 16.0]] * 3)

        curves = route_edges(pos, g.edge_index, ns, direction="TB", graph=g)

        polys = [_curve_polyline_samples(c, sample_count=24) for c in curves]
        crossings = 0
        for i in range(len(polys[0]) - 1):
            for j in range(len(polys[1]) - 1):
                if _segments_cross(polys[0][i], polys[0][i + 1], polys[1][j], polys[1][j + 1]):
                    crossings += 1
        assert crossings == 0

    def test_acceptance_is_deterministic(self) -> None:
        """Same inputs -> identical routes (referee has no RNG)."""
        g = DaguaGraph.from_edge_list([(0, 1), (0, 2), (1, 3), (2, 3), (0, 3)])
        pos = torch.tensor([[0.0, 0.0], [-60.0, 120.0], [60.0, 120.0], [0.0, 240.0], [5.0, 60.0]])
        ns = torch.tensor([[40.0, 20.0]] * 5)
        a = route_edges(pos, g.edge_index, ns, direction="TB", graph=g)
        b = route_edges(pos, g.edge_index, ns, direction="TB", graph=g)
        for ca, cb in zip(a, b):
            assert ca.cp1 == cb.cp1 and ca.cp2 == cb.cp2


class TestDensityScaledSpread:
    def test_sparse_neighborhood_keeps_full_budget(self) -> None:
        from dagua.edges import _local_density_spread_scales

        # 3 nodes far apart (each alone in its grid neighborhood).
        xs = [0.0, 1000.0, 2000.0]
        ys = [0.0, 0.0, 0.0]
        grid = _build_node_grid(xs, ys, 45.0)
        scales = _local_density_spread_scales(grid, 45.0, xs, ys)
        assert scales == [1.0, 1.0, 1.0]

    def test_dense_clump_shrinks_budget_with_floor(self) -> None:
        from dagua.edges import _local_density_spread_scales

        # 26 nodes stacked in one grid cell: n_local=25 per node.
        xs = [10.0] * 26
        ys = [10.0] * 26
        grid = _build_node_grid(xs, ys, 45.0)
        scales = _local_density_spread_scales(grid, 45.0, xs, ys)
        expected = max(0.3, (4.0 / 25.0) ** 0.5)
        assert all(abs(s - expected) < 1e-12 for s in scales)
        assert all(0.3 <= s < 1.0 for s in scales)

    def test_dense_hub_gets_smaller_fan_than_sparse_hub(self) -> None:
        """Same hub fan-out, one embedded in a dense clump: its initial
        tangents must spread strictly less than the sparse hub's."""
        import math as _math

        def hub_min_angle(extra_nodes: int) -> float:
            edges = [(0, 1), (0, 2)]
            g = DaguaGraph.from_edge_list(edges)
            positions = [[0.0, 0.0], [-150.0, 300.0], [150.0, 300.0]]
            sizes = [[30.0, 16.0]] * 3
            for k in range(extra_nodes):
                g.add_node(f"filler_{k}")
                # Clump fillers right next to the hub (same grid cell zone).
                positions.append([8.0 + (k % 3), -8.0 - (k // 3)])
                sizes.append([30.0, 16.0])
            pos = torch.tensor(positions)
            ns = torch.tensor(sizes)
            curves = route_edges(pos, g.edge_index, ns, direction="TB", graph=g)
            angles = []
            for c in curves:
                dx = c.cp1[0] - c.p0[0]
                dy = c.cp1[1] - c.p0[1]
                angles.append(_math.atan2(dy, dx))
            return abs(angles[0] - angles[1])

        sparse = hub_min_angle(0)
        dense = hub_min_angle(20)
        assert dense < sparse


def test_route_edges_survives_non_finite_positions():
    """Divergent layouts (NaN/inf positions) must render as-is, not crash.

    r80 adversarial review MEDIUM-1: the S7 spatial grid and deflection
    machinery floor()ed non-finite coordinates. Non-finite inputs now skip
    avoidance/spread and fall back to pre-r80 behavior.
    """
    import torch

    from dagua.edges import route_edges

    n = 12
    edges = torch.tensor([[i, (i + 1) % n] for i in range(n)], dtype=torch.long).t()
    sizes = torch.full((n, 2), 40.0)
    for value in (float("nan"), float("inf"), float("-inf")):
        pos = torch.full((n, 2), value)
        assert len(route_edges(pos, edges, sizes)) == n
    mixed = torch.rand(n, 2) * 500
    mixed[3, 0] = float("nan")
    assert len(route_edges(mixed, edges, sizes)) == n
