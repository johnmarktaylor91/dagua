"""Tests for r80-S6 external drawing capture: adapters, wrappers, store blob."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from dagua.eval.competitors.base import CompetitorResult
from dagua.eval.competitors.elk_competitor import _collect_elk_routes
from dagua.eval.competitors.graphviz_competitor import (
    _coerce_layout_capture,
    _parse_graphviz_json_drawing,
    _parse_xdot_spline,
)
from dagua.eval.drawing import native_route_coverage, polyline_to_curve, routes_to_curves
from dagua.graph import DaguaGraph

FIXTURE = Path(__file__).parent / "fixtures" / "r80_dot_json_small.json"


def _fixture_graph() -> DaguaGraph:
    """Rebuild the graph the checked-in dot -Tjson fixture was produced from."""
    edges = [
        ("n0", "n1"),
        ("n0", "n2"),
        ("n1", "n3"),
        ("n2", "n3"),
        ("n3", "n4"),
        ("n3", "n5"),
        ("n0", "n3"),
    ]
    graph = DaguaGraph.from_edge_list(edges)
    graph.edge_labels = ["L1", None, "L2", None, None, "L3", None]
    graph.compute_node_sizes()
    return graph


class TestXdotSplineParsing:
    def test_parses_endpoint_prefix_spline(self) -> None:
        polyline = _parse_xdot_spline("e,39.8,121.1 84.4,173.8 73.7,161.1 59.0,143.7 47.1,129.8")
        assert polyline is not None
        # Starts at the first control point, ends at the arrow tip.
        assert polyline[0] == (84.4, 173.8)
        assert polyline[-1] == (39.8, 121.1)
        assert len(polyline) > 4  # sampled, not just control points

    def test_parses_start_prefix(self) -> None:
        polyline = _parse_xdot_spline("s,1.0,2.0 0.0,0.0 10.0,0.0 20.0,0.0 30.0,0.0")
        assert polyline is not None
        assert polyline[0] == (1.0, 2.0)

    def test_rejects_garbage(self) -> None:
        assert _parse_xdot_spline("100,200") is None
        assert _parse_xdot_spline("") is None
        assert _parse_xdot_spline("not,a x,spline") is None

    def test_rejects_non_bezier_count(self) -> None:
        # 3 points cannot form a piecewise cubic (needs 3k+1 >= 4).
        assert _parse_xdot_spline("0,0 1,1 2,2") is None


class TestGraphvizJsonDrawing:
    def test_fixture_routes_align_with_edges(self) -> None:
        graph = _fixture_graph()
        data = json.loads(FIXTURE.read_text())
        routes, labels = _parse_graphviz_json_drawing(data, graph)
        assert routes is not None
        assert len(routes) == graph.edge_index.shape[1]
        assert all(route is not None and len(route) >= 2 for route in routes)

        # Route endpoints must sit near their edge's node centers (port
        # offsets are bounded by the node box, ~40pt for these labels).
        positions = torch.zeros(graph.num_nodes, 2)
        for obj in data["objects"]:
            index = int(obj["name"][1:])
            x_str, y_str = obj["pos"].split(",")
            positions[index, 0] = float(x_str)
            positions[index, 1] = -float(y_str)
        for e_idx, route in enumerate(routes):
            s = int(graph.edge_index[0, e_idx].item())
            t = int(graph.edge_index[1, e_idx].item())
            start_gap = math.dist(route[0], tuple(positions[s].tolist()))
            end_gap = math.dist(route[-1], tuple(positions[t].tolist()))
            assert start_gap < 60.0, f"edge {e_idx} start too far from source"
            assert end_gap < 60.0, f"edge {e_idx} end too far from target"

    def test_fixture_labels_only_on_labeled_edges(self) -> None:
        graph = _fixture_graph()
        data = json.loads(FIXTURE.read_text())
        _, labels = _parse_graphviz_json_drawing(data, graph)
        assert labels is not None
        for e_idx, label in enumerate(graph.edge_labels):
            if label:
                assert labels[e_idx] is not None, f"labeled edge {e_idx} missing anchor"
            else:
                assert labels[e_idx] is None

    def test_y_axis_flipped_to_dagua_convention(self) -> None:
        graph = _fixture_graph()
        data = json.loads(FIXTURE.read_text())
        routes, _ = _parse_graphviz_json_drawing(data, graph)
        # Graphviz y-up coords are positive; dagua stores negated y.
        all_y = [y for route in routes for _, y in route]
        assert min(all_y) < 0.0

    def test_empty_payload_returns_none(self) -> None:
        graph = _fixture_graph()
        assert _parse_graphviz_json_drawing({}, graph) == (None, None)
        assert _parse_graphviz_json_drawing({"objects": [], "edges": []}, graph) == (
            None,
            None,
        )


class TestCoerceLayoutCapture:
    def test_bare_tensor_passthrough(self) -> None:
        pos = torch.zeros(3, 2)
        out_pos, routes, labels = _coerce_layout_capture(pos)
        assert out_pos is pos
        assert routes is None
        assert labels is None

    def test_triple_passthrough(self) -> None:
        pos = torch.zeros(3, 2)
        triple = (pos, [[(0.0, 0.0), (1.0, 1.0)]], [None])
        assert _coerce_layout_capture(triple) == triple


class TestElkRouteParsing:
    def test_bend_points_parsed(self) -> None:
        data = {
            "edges": [
                {
                    "id": "e0",
                    "sections": [
                        {
                            "startPoint": {"x": 0.0, "y": 0.0},
                            "bendPoints": [{"x": 0.0, "y": 50.0}, {"x": 100.0, "y": 50.0}],
                            "endPoint": {"x": 100.0, "y": 100.0},
                        }
                    ],
                },
                {
                    "id": "e1",
                    "sections": [
                        {"startPoint": {"x": 5.0, "y": 5.0}, "endPoint": {"x": 6.0, "y": 6.0}}
                    ],
                },
            ]
        }
        routes = _collect_elk_routes(data, 2)
        assert routes is not None
        assert routes[0] == [(0.0, 0.0), (0.0, 50.0), (100.0, 50.0), (100.0, 100.0)]
        assert routes[1] == [(5.0, 5.0), (6.0, 6.0)]

    def test_no_sections_returns_none(self) -> None:
        assert _collect_elk_routes({"edges": [{"id": "e0"}]}, 1) is None
        assert _collect_elk_routes({}, 1) is None

    def test_unknown_ids_ignored(self) -> None:
        data = {
            "edges": [
                {
                    "id": "weird",
                    "sections": [{"startPoint": {"x": 0, "y": 0}, "endPoint": {"x": 1, "y": 1}}],
                },
                {
                    "id": "e9",
                    "sections": [{"startPoint": {"x": 0, "y": 0}, "endPoint": {"x": 1, "y": 1}}],
                },
            ]
        }
        assert _collect_elk_routes(data, 2) is None


class TestDrawingWrappers:
    def test_polyline_to_curve_waypoints(self) -> None:
        curve = polyline_to_curve([(0.0, 0.0), (5.0, 5.0), (10.0, 0.0)])
        assert curve.waypoints == ((0.0, 0.0), (5.0, 5.0), (10.0, 0.0))
        assert curve.p0 == (0.0, 0.0)
        assert curve.p1 == (10.0, 0.0)

    def test_routes_to_curves_with_fallback(self) -> None:
        pos = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
        ei = torch.tensor([[0, 2], [1, 3]])
        routes = [[(0.0, 0.0), (4.0, 3.0), (10.0, 0.0)], None]
        curves = routes_to_curves(routes, pos, ei)
        assert curves is not None
        assert curves[0].waypoints is not None
        # Fallback: straight node-center segment.
        assert curves[1].waypoints is None
        assert curves[1].p0 == (0.0, 10.0)
        assert curves[1].p1 == (10.0, 10.0)

    def test_routes_none_passthrough(self) -> None:
        pos = torch.zeros(2, 2)
        ei = torch.tensor([[0], [1]])
        assert routes_to_curves(None, pos, ei) is None

    def test_native_route_coverage(self) -> None:
        assert native_route_coverage(None, 4) == 0.0
        assert native_route_coverage([[(0, 0), (1, 1)], None, [(0, 0)], [(2, 2), (3, 3)]], 4) == 0.5


class TestCompetitorResultDefaults:
    def test_optional_fields_default_none(self) -> None:
        result = CompetitorResult(name="x", pos=torch.zeros(1, 2), runtime_seconds=0.1)
        assert result.routes is None
        assert result.edge_label_positions is None


class TestRoutesBlobRoundtrip:
    def test_roundtrip_with_routes(self, tmp_path: Path) -> None:
        blob = {
            "routes": [[(0.0, 0.0), (1.0, 2.0)], None],
            "edge_label_positions": [None, (5.0, 6.0)],
        }
        path = tmp_path / "routes" / "g__eng.pt"
        path.parent.mkdir(parents=True)
        torch.save(blob, path)
        loaded = torch.load(path, weights_only=False)
        assert loaded == blob

    def test_absent_blob_means_none(self, tmp_path: Path) -> None:
        """Old stores have no routes/ dir; consumers must treat that as None."""
        row = {"positions_path": "positions/g__eng.pt"}  # no routes_path key
        assert row.get("routes_path") is None
        assert not (tmp_path / "routes").exists()
