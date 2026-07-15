"""Tests for the honest-ruler point-collapse degeneracy guard."""

from __future__ import annotations

import pytest

from dagua.metrics import (
    _COMMON_WEIGHTS,
    DEGENERATE_SCALE_RATIO,
    _is_degenerate_scale,
    composite,
    composite_undirected,
)


def _profile(**overrides: float) -> dict[str, float]:
    """Build a complete common profile with optional overrides."""
    profile = {name: 0.8 for name in _COMMON_WEIGHTS}
    profile.update(overrides)
    return profile


class TestIsDegenerateScale:
    """Exercise the scale provenance predicate boundaries."""

    def test_missing_fields_is_not_degenerate(self) -> None:
        """Missing scale provenance never triggers the guard."""
        assert _is_degenerate_scale({}) is False
        assert _is_degenerate_scale({"edge_length_mean": 1.0}) is False
        assert _is_degenerate_scale({"node_diag_mean": 1.0}) is False

    def test_zero_node_diag_is_not_degenerate(self) -> None:
        """A zero node diagonal cannot define a meaningful scale ratio."""
        assert _is_degenerate_scale({"edge_length_mean": 0.0, "node_diag_mean": 0.0}) is False

    def test_below_threshold_is_degenerate(self) -> None:
        """Edge length below the frozen scale ratio triggers the guard."""
        metrics = {"edge_length_mean": 1.0, "node_diag_mean": 10.0}
        assert 1.0 < DEGENERATE_SCALE_RATIO * 10.0
        assert _is_degenerate_scale(metrics) is True

    def test_at_or_above_threshold_is_not_degenerate(self) -> None:
        """The scale predicate is strict at the boundary."""
        assert _is_degenerate_scale({"edge_length_mean": 2.5, "node_diag_mean": 10.0}) is False
        assert _is_degenerate_scale({"edge_length_mean": 50.0, "node_diag_mean": 10.0}) is False


def test_common_collapse_zeroes_vacuous_geometry_terms() -> None:
    """A point collapse cannot retain maxima from geometry-free terms."""
    normal = _profile(edge_length_mean=50.0, node_diag_mean=10.0)
    collapsed = _profile(
        edge_length_mean=0.1,
        node_diag_mean=10.0,
        node_occlusion_score=0.05,
    )
    assert composite_undirected(collapsed) < composite_undirected(normal)
    assert composite_undirected(collapsed) == pytest.approx(
        100.0 * 13.0 * 0.05 / sum(_COMMON_WEIGHTS.values())
    )


def test_directed_collapse_guard_preserves_only_semantic_terms() -> None:
    """DF and DO remain meaningful while collapsed common geometry is zeroed."""
    collapsed = {
        **_profile(edge_length_mean=0.1, node_diag_mean=10.0),
        "declared_hierarchical": True,
        "directed_flow_score": 0.4,
        "depth_order_score": 0.6,
    }
    assert composite(collapsed) == pytest.approx(16 * 0.4 + 9 * 0.6)


def test_missing_provenance_does_not_change_complete_profile() -> None:
    """Normal and provenance-free profiles have identical common scores."""
    without = _profile()
    with_scale = _profile(edge_length_mean=50.0, node_diag_mean=10.0)
    assert composite_undirected(without) == pytest.approx(composite_undirected(with_scale))
