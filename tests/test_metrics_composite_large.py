"""Tests for the large-graph (quick-tier-only) composite variants.

r80-P6 (S1 MEDIUM-1): ``composite_large`` hardcoded the DIRECTED weight
scheme with no undirected counterpart -- a latent landmine for any
undirected N>2000 graph scored through the large-graph path. These tests
cover both flavors (``composite_large`` and the new
``composite_large_undirected``) plus the ``composite_large_auto``
dispatcher.
"""

from __future__ import annotations

import pytest

from dagua.metrics import (
    composite_large,
    composite_large_auto,
    composite_large_undirected,
)


def _quick_directed_metrics(**overrides: float) -> dict[str, float]:
    """Build a complete quick-tier directed metrics dict.

    Parameters
    ----------
    **overrides : float
        Fields to override on top of the baseline metrics.

    Returns
    -------
    dict[str, float]
        Metrics dict accepted by ``composite_large()``.
    """
    base = {
        "dag_consistency": 0.8,
        "edge_length_cv": 0.2,
        "depth_spearman_rho": 0.5,
        "overlap_count": 0,
        "edge_straightness_mean_deg": 10.0,
    }
    base.update(overrides)
    return base


def _quick_undirected_metrics(**overrides: float) -> dict[str, float]:
    """Build a complete quick-tier undirected metrics dict.

    Parameters
    ----------
    **overrides : float
        Fields to override on top of the baseline metrics.

    Returns
    -------
    dict[str, float]
        Metrics dict accepted by ``composite_large_undirected()``.
    """
    base = {
        "edge_length_cv": 0.2,
        "overlap_count": 0,
    }
    base.update(overrides)
    return base


class TestCompositeLarge:
    def test_perfect_metrics_score_100(self) -> None:
        """Composite large returns 100 for a perfect quick-tier metrics dict."""
        metrics = _quick_directed_metrics(
            dag_consistency=1.0,
            edge_length_cv=0.0,
            depth_spearman_rho=1.0,
            overlap_count=0,
            edge_straightness_mean_deg=0.0,
        )
        assert composite_large(metrics) == pytest.approx(100.0)

    def test_missing_field_raises(self) -> None:
        """A missing quick-mode field raises rather than silently defaulting."""
        metrics = _quick_directed_metrics()
        del metrics["dag_consistency"]
        with pytest.raises(ValueError, match="missing required quick-mode fields"):
            composite_large(metrics)

    def test_weights_sum_formula(self) -> None:
        """Composite large applies its documented 30/25/20/15/10 weights."""
        metrics = _quick_directed_metrics(
            dag_consistency=0.5,
            edge_length_cv=0.4,
            depth_spearman_rho=0.6,
            overlap_count=1,
            edge_straightness_mean_deg=9.0,
        )
        expected = 30 * 0.5 + 25 * 0.6 + 20 * 0.6 + 15 * 0.0 + 10 * (1.0 - 9.0 / 45.0)
        assert composite_large(metrics) == pytest.approx(expected)


class TestCompositeLargeUndirected:
    def test_perfect_metrics_score_100(self) -> None:
        """Composite large undirected returns 100 for perfect retained metrics."""
        metrics = _quick_undirected_metrics(edge_length_cv=0.0, overlap_count=0)
        assert composite_large_undirected(metrics) == pytest.approx(100.0)

    def test_worst_case_scores_zero(self) -> None:
        """Maximal CV and any overlap scores 0."""
        metrics = _quick_undirected_metrics(edge_length_cv=1.0, overlap_count=5)
        assert composite_large_undirected(metrics) == pytest.approx(0.0)

    def test_missing_field_raises(self) -> None:
        """A missing quick-mode field raises rather than silently defaulting."""
        metrics = _quick_undirected_metrics()
        del metrics["edge_length_cv"]
        with pytest.raises(ValueError, match="missing required quick-mode fields"):
            composite_large_undirected(metrics)

    def test_ignores_direction_sensitive_fields(self) -> None:
        """Direction-sensitive fields (even if present) do not affect the score.

        Mirrors composite_undirected dropping dag_consistency/depth/
        straightness: composite_large_undirected must not silently pick
        them up even if a caller's metrics dict happens to carry them.
        """
        metrics = _quick_undirected_metrics(edge_length_cv=0.3, overlap_count=0)
        metrics_with_directed_noise = dict(metrics)
        metrics_with_directed_noise.update(
            {
                "dag_consistency": 0.0,  # would tank a directed score
                "depth_spearman_rho": -1.0,
                "edge_straightness_mean_deg": 45.0,
            }
        )

        assert composite_large_undirected(metrics) == pytest.approx(
            composite_large_undirected(metrics_with_directed_noise)
        )

    def test_weights_sum_formula(self) -> None:
        """Composite large undirected applies its documented 65/35 weights."""
        metrics = _quick_undirected_metrics(edge_length_cv=0.4, overlap_count=0)
        expected = 65 * 0.6 + 35 * 1.0
        assert composite_large_undirected(metrics) == pytest.approx(expected)


class TestCompositeLargeAuto:
    def test_directed_dispatches_to_composite_large(self) -> None:
        """composite_large_auto(directed=True) matches composite_large."""
        metrics = _quick_directed_metrics()
        assert composite_large_auto(metrics, is_semantically_directed=True) == pytest.approx(
            composite_large(metrics)
        )

    def test_undirected_dispatches_to_composite_large_undirected(self) -> None:
        """composite_large_auto(directed=False) matches composite_large_undirected."""
        metrics = _quick_undirected_metrics()
        assert composite_large_auto(metrics, is_semantically_directed=False) == pytest.approx(
            composite_large_undirected(metrics)
        )

    def test_none_defaults_to_directed(self) -> None:
        """composite_large_auto(None) conservatively defaults to directed."""
        metrics = _quick_directed_metrics()
        assert composite_large_auto(metrics, is_semantically_directed=None) == pytest.approx(
            composite_large(metrics)
        )
