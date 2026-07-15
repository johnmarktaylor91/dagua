"""Tests for the r80-S6 full-drawing metrics (routed crossings, bends, composite)."""

from __future__ import annotations

import torch

from dagua.edges import BezierCurve, route_edges
from dagua.metrics import (
    bend_count,
    composite_drawing,
    routed_crossing_rate,
    sampled_crossing_rate,
)


def _straight_curves(pos: torch.Tensor, edge_index: torch.Tensor) -> list:
    """Build degenerate-straight node-center curves for every edge."""
    curves = []
    for e in range(edge_index.shape[1]):
        s = int(edge_index[0, e].item())
        t = int(edge_index[1, e].item())
        p0 = (float(pos[s, 0]), float(pos[s, 1]))
        p1 = (float(pos[t, 0]), float(pos[t, 1]))
        curves.append(BezierCurve(p0, p0, p1, p1, routing="straight"))
    return curves


def _random_graph(seed: int = 123, n: int = 60, e: int = 150):
    gen = torch.Generator().manual_seed(seed)
    pos = torch.rand(n, 2, generator=gen) * 500
    src = torch.randint(0, n, (e,), generator=gen)
    tgt = torch.randint(0, n, (e,), generator=gen)
    mask = src != tgt
    edge_index = torch.stack([src[mask], tgt[mask]])
    return pos, edge_index


class TestRoutedCrossingRate:
    def test_straight_curves_match_sampled_crossing_rate_exhaustive(self) -> None:
        """On straight center-segment curves the routed metric is bit-identical
        to sampled_crossing_rate (pair space exhausted branch)."""
        pos, ei = _random_graph()
        curves = _straight_curves(pos, ei)
        for seed in (0, 7, 42):
            straight = sampled_crossing_rate(pos, ei, 50_000, seed=seed)
            routed = routed_crossing_rate(curves, ei, 50_000, seed=seed)
            assert routed["routed_crossing_rate"] == straight["crossing_rate"]
            assert routed["routed_crossing_n_samples"] == straight["crossing_n_samples"]
            assert routed["routed_crossing_estimated_total"] == straight["crossing_estimated_total"]

    def test_straight_curves_match_sampled_crossing_rate_subsampled(self) -> None:
        """Same equivalence on the without-replacement subsampling branch."""
        pos, ei = _random_graph()
        curves = _straight_curves(pos, ei)
        for seed in (0, 7):
            straight = sampled_crossing_rate(pos, ei, 500, seed=seed)
            routed = routed_crossing_rate(curves, ei, 500, seed=seed)
            assert routed["routed_crossing_rate"] == straight["crossing_rate"]
            assert routed["routed_crossing_n_samples"] == straight["crossing_n_samples"]

    def test_curved_route_differs_from_straight(self) -> None:
        """A bulging bezier crosses a neighbor the straight chord misses."""
        # Two vertical edges side by side: straight chords never cross.
        pos = torch.tensor([[0.0, 0.0], [0.0, 100.0], [30.0, 0.0], [30.0, 100.0]])
        ei = torch.tensor([[0, 2], [1, 3]])
        straight = sampled_crossing_rate(pos, ei, 1000, seed=0)
        assert straight["crossing_rate"] == 0.0

        # Bulge edge 0 rightward across edge 1's column and back.
        bulged = BezierCurve((0.0, 0.0), (120.0, 25.0), (120.0, 75.0), (0.0, 100.0))
        curves = [bulged, _straight_curves(pos, ei)[1]]
        routed = routed_crossing_rate(curves, ei, 1000, seed=0)
        assert routed["routed_crossing_rate"] > 0.0

    def test_empty_and_single_edge(self) -> None:
        empty = routed_crossing_rate([], torch.zeros((2, 0), dtype=torch.long), 100)
        assert empty["routed_crossing_rate"] == 0.0
        one = routed_crossing_rate(
            [BezierCurve((0, 0), (0, 0), (1, 1), (1, 1))],
            torch.tensor([[0], [1]]),
            100,
        )
        assert one["routed_crossing_rate"] == 0.0

    def test_curve_count_mismatch_raises(self) -> None:
        pos, ei = _random_graph(n=10, e=12)
        curves = _straight_curves(pos, ei)[:-1]
        try:
            routed_crossing_rate(curves, ei, 100, seed=0)
        except ValueError as exc:
            assert "align" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected ValueError on curve/edge mismatch")


class TestBendCount:
    def test_straight_has_no_bends(self) -> None:
        curve = BezierCurve((0.0, 0.0), (0.0, 0.0), (100.0, 0.0), (100.0, 0.0))
        result = bend_count([curve])
        assert result["bend_total"] == 0
        assert result["bend_mean_per_edge"] == 0.0

    def test_ortho_elbow_counts_bends(self) -> None:
        # One Manhattan elbow corridor: two 90-degree bends.
        elbow = BezierCurve(
            (0.0, 0.0),
            (0.0, 50.0),
            (100.0, 50.0),
            (100.0, 100.0),
            waypoints=((0.0, 0.0), (0.0, 50.0), (100.0, 50.0), (100.0, 100.0)),
            routing="ortho",
        )
        result = bend_count([elbow])
        assert result["bend_total"] == 2
        assert result["bend_edges_with_bends"] == 1

    def test_taxi_z_route_counts_bends(self) -> None:
        z = BezierCurve(
            (0.0, 0.0),
            (0.0, 35.0),
            (50.0, 65.0),
            (50.0, 100.0),
            waypoints=(
                (0.0, 0.0),
                (0.0, 35.0),
                (50.0, 35.0),
                (50.0, 65.0),
                (100.0, 65.0),
                (100.0, 100.0),
            ),
            routing="taxi",
        )
        result = bend_count([z])
        assert result["bend_total"] == 4

    def test_gentle_bezier_below_threshold(self) -> None:
        # A mild arc sampled finely: every inter-segment turn stays small.
        gentle = BezierCurve((0.0, 0.0), (30.0, 10.0), (70.0, 10.0), (100.0, 0.0))
        result = bend_count([gentle], angle_threshold_deg=15.0)
        assert result["bend_total"] == 0

    def test_empty(self) -> None:
        assert bend_count([])["bend_total"] == 0


class TestCompositeDrawing:
    def _setup(self):
        gen = torch.Generator().manual_seed(9)
        pos = torch.rand(20, 2, generator=gen) * 400
        src = torch.arange(19)
        tgt = torch.arange(1, 20)
        extra_s = torch.randint(0, 20, (15,), generator=gen)
        extra_t = torch.randint(0, 20, (15,), generator=gen)
        mask = extra_s != extra_t
        ei = torch.cat(
            [torch.stack([src, tgt]), torch.stack([extra_s[mask], extra_t[mask]])],
            dim=1,
        )
        sizes = torch.full((20, 2), 24.0)
        curves = route_edges(pos, ei, sizes, "TB", None)
        return pos, ei, sizes, curves

    def test_deterministic_given_seed(self) -> None:
        pos, ei, sizes, curves = self._setup()
        a = composite_drawing(pos, ei, sizes, curves, seed=5)
        b = composite_drawing(pos, ei, sizes, curves, seed=5)
        assert a == b

    def test_score_in_range_and_components_present(self) -> None:
        pos, ei, sizes, curves = self._setup()
        result = composite_drawing(pos, ei, sizes, curves, seed=0)
        assert 0.0 <= result["composite_drawing"] <= 100.0
        for key in (
            "drawing_crossing_rate",
            "drawing_edge_node_crossing_rate",
            "drawing_port_angular_deg",
            "drawing_curvature_cv",
            "drawing_bend_mean_per_edge",
            "drawing_node_overlaps",
            "drawing_has_labels",
        ):
            assert key in result

    def test_no_labels_drops_inapplicable_terms(self) -> None:
        pos, ei, sizes, curves = self._setup()
        result = composite_drawing(pos, ei, sizes, curves, seed=0)
        assert result["drawing_has_labels"] is False
        assert result["drawing_term_label_node"] is None
        assert result["drawing_term_label_label"] is None

    def test_labels_scored_when_present(self) -> None:
        pos, ei, sizes, curves = self._setup()
        edge_labels = ["lbl"] + [None] * (ei.shape[1] - 1)
        # Anchor the label directly on node 0's center: guaranteed overlap.
        label_positions = [(float(pos[0, 0]), float(pos[0, 1]))] + [None] * (ei.shape[1] - 1)
        result = composite_drawing(
            pos,
            ei,
            sizes,
            curves,
            label_positions=label_positions,
            edge_labels=edge_labels,
            seed=0,
        )
        assert result["drawing_has_labels"] is True
        assert result["drawing_label_node_overlaps"] >= 1
        assert result["drawing_term_label_node"] < 1.0

    def test_computable_without_layout_run(self) -> None:
        """Constructed purely from (positions, sizes, curves): no engine call."""
        pos = torch.tensor([[0.0, 0.0], [200.0, 0.0], [100.0, 150.0]])
        ei = torch.tensor([[0, 1], [1, 2]])
        sizes = torch.full((3, 2), 20.0)
        curves = _straight_curves(pos, ei)
        result = composite_drawing(pos, ei, sizes, curves, seed=0)
        assert 0.0 <= result["composite_drawing"] <= 100.0
