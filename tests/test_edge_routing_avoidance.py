"""Tests for r80-S7 routing-quality deliverables: node-bbox avoidance,
port angular spread, and label-vs-edge-path avoidance."""

import torch

from dagua.edges import (
    BezierCurve,
    _build_node_grid,
    _curve_samples_hit_rect,
    _deflect_around_nodes,
    route_edges,
)
from dagua.graph import DaguaGraph
from dagua.metrics import edge_node_crossing_count
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
