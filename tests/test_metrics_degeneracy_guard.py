"""Tests for the r80-P6 composite degeneracy guard.

The S1 HIGH-3 audit finding: a point-collapsed layout (every node stacked
near the same location) trivially maximizes edge-length uniformity (CV of an
all-equal tiny-length distribution is 0) and crossing rate (zero-length
segments never register as crossing), even though the layout is a fully
overlapping unreadable mess. These tests confirm the guard zeroes both
terms when the layout is at a degenerate scale, and leaves normal (and
missing-provenance) layouts unaffected.
"""

from __future__ import annotations

import pytest

from dagua.metrics import (
    DEGENERATE_SCALE_RATIO,
    _is_degenerate_scale,
    composite,
    composite_undirected,
)


def _directed_metrics(**overrides: float) -> dict[str, float]:
    """Build a complete directed-composite metrics dict with overrides.

    Parameters
    ----------
    **overrides : float
        Fields to override on top of the baseline metrics.

    Returns
    -------
    dict[str, float]
        Metrics dict accepted by ``composite()``.
    """
    base = {
        "dag_consistency": 0.8,
        "edge_length_cv": 0.2,
        "depth_spearman_rho": 0.5,
        "overlap_count": 0,
        "edge_straightness_mean_deg": 10.0,
        "crossing_rate": 0.02,
        "sampled_stress": 0.3,
        "angular_res_mean_deg": 30.0,
        "cluster_mean_sep_ratio": 3.0,
    }
    base.update(overrides)
    return base


def _undirected_metrics(**overrides: float) -> dict[str, float]:
    """Build a complete undirected-composite metrics dict with overrides.

    Parameters
    ----------
    **overrides : float
        Fields to override on top of the baseline metrics.

    Returns
    -------
    dict[str, float]
        Metrics dict accepted by ``composite_undirected()``.
    """
    base = {
        "edge_length_cv": 0.2,
        "overlap_count": 0,
        "crossing_rate": 0.02,
        "angular_res_mean_deg": 30.0,
        "cluster_mean_sep_ratio": 3.0,
    }
    base.update(overrides)
    return base


class TestIsDegenerateScale:
    def test_missing_fields_is_not_degenerate(self) -> None:
        """Metrics without edge_length_mean/node_diag_mean never trigger the guard."""
        assert _is_degenerate_scale({}) is False
        assert _is_degenerate_scale({"edge_length_mean": 1.0}) is False
        assert _is_degenerate_scale({"node_diag_mean": 1.0}) is False

    def test_zero_node_diag_is_not_degenerate(self) -> None:
        """A zero (or near-zero) node diagonal cannot divide meaningfully."""
        metrics = {"edge_length_mean": 0.0, "node_diag_mean": 0.0}
        assert _is_degenerate_scale(metrics) is False

    def test_below_threshold_is_degenerate(self) -> None:
        """Edge length below the ratio threshold flags as degenerate."""
        metrics = {"edge_length_mean": 1.0, "node_diag_mean": 10.0}
        assert 1.0 < DEGENERATE_SCALE_RATIO * 10.0
        assert _is_degenerate_scale(metrics) is True

    def test_above_threshold_is_not_degenerate(self) -> None:
        """Edge length well above the ratio threshold is a normal layout."""
        metrics = {"edge_length_mean": 50.0, "node_diag_mean": 10.0}
        assert _is_degenerate_scale(metrics) is False

    def test_exactly_at_threshold_is_not_degenerate(self) -> None:
        """The comparison is strict-less-than at the boundary."""
        metrics = {"edge_length_mean": 2.5, "node_diag_mean": 10.0}  # exactly 0.25x
        assert _is_degenerate_scale(metrics) is False


class TestCompositeDirectedGuard:
    def test_collapsed_layout_scores_zero_on_guarded_terms(self) -> None:
        """A degenerate layout gets zero length-uniformity and crossing credit."""
        metrics = _directed_metrics(
            edge_length_cv=0.0,  # would vacuously max length-uniformity
            crossing_rate=0.0,  # would vacuously max crossing credit
            edge_length_mean=0.1,
            node_diag_mean=10.0,  # 0.1 < 0.25 * 10 -> degenerate
        )
        guarded = composite(metrics)

        ungated = dict(metrics)
        del ungated["edge_length_mean"]
        del ungated["node_diag_mean"]
        unguarded_score = composite(ungated)

        # The guard must have removed exactly the 18 (length) + 9 (crossing) points
        # that a vacuous CV=0/crossing=0 would otherwise award.
        assert guarded == pytest.approx(unguarded_score - 18.0 - 9.0)

    def test_normal_layout_is_unaffected_by_guard(self) -> None:
        """A non-degenerate layout scores identically whether or not the
        scale-provenance fields are present."""
        metrics_with_scale = _directed_metrics(edge_length_mean=50.0, node_diag_mean=10.0)
        metrics_without_scale = _directed_metrics()

        assert composite(metrics_with_scale) == pytest.approx(composite(metrics_without_scale))

    def test_missing_scale_provenance_preserves_prior_behavior(self) -> None:
        """Metrics predating node_diag_mean score exactly as before (no guard)."""
        metrics = _directed_metrics(edge_length_cv=0.0, crossing_rate=0.0)
        assert "node_diag_mean" not in metrics

        expected = (
            22 * 0.8
            + 18 * 1.0  # edge_length_cv=0.0 -> full credit, guard cannot fire
            + 13 * 0.5
            + 8
            + 9 * (1.0 - 10.0 / 45.0)
            + 9 * 1.0  # crossing_rate=0.0 -> full credit, guard cannot fire
            + 10 * (1.0 - 0.3)
            + 5 * (30.0 / 40.0)
            + 6 * (3.0 / 5.0)
        )
        assert composite(metrics) == pytest.approx(expected)


class TestCompositeUndirectedGuard:
    def test_collapsed_layout_scores_zero_on_guarded_terms(self) -> None:
        """A degenerate undirected layout gets zero length/crossing credit."""
        metrics = _undirected_metrics(
            edge_length_cv=0.0,
            crossing_rate=0.0,
            edge_length_mean=0.1,
            node_diag_mean=10.0,
        )
        guarded = composite_undirected(metrics)

        ungated = dict(metrics)
        del ungated["edge_length_mean"]
        del ungated["node_diag_mean"]
        unguarded_score = composite_undirected(ungated)

        # 40 (length uniformity) + 20 (crossing) points removed by the guard.
        assert guarded == pytest.approx(unguarded_score - 40.0 - 20.0)

    def test_fully_collapsed_layout_no_longer_beats_a_normal_layout(self) -> None:
        """Regression for S1 HIGH-3: collapse must not outscore a decent layout.

        Before the guard, an all-collapsed-to-a-point layout with overlap
        scored 65/100 -- HIGHER than a normal, non-degenerate random layout
        at 29.3/100 (see P8C_HARNESS_AUDIT.md repro). After the guard, the
        collapsed layout must score at most the binary overlap credit (0,
        since a collapsed layout always overlaps) plus any neutral terms.
        """
        collapsed = _undirected_metrics(
            edge_length_cv=0.0,
            overlap_count=300,  # collapsed onto each other -> overlaps
            crossing_rate=0.0,
            edge_length_mean=0.0,
            node_diag_mean=10.0,
        )
        normal_random = _undirected_metrics(
            edge_length_cv=0.5,
            overlap_count=7,
            crossing_rate=0.372,
            edge_length_mean=50.0,
            node_diag_mean=10.0,
        )

        assert composite_undirected(collapsed) < composite_undirected(normal_random)
