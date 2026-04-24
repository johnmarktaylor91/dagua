"""Tests for undirected metric composite variants."""

from __future__ import annotations

import pytest

from dagua.metrics import composite, composite_auto, composite_undirected


def _sample_metrics() -> dict[str, float]:
    """Create a complete metrics dictionary with non-trivial values.

    Returns
    -------
    dict[str, float]
        Metrics with all keys used by directed and undirected composites.
    """
    return {
        "dag_consistency": 0.8,
        "edge_length_cv": 0.2,
        "depth_spearman_rho": 0.5,
        "overlap_count": 0,
        "edge_straightness_mean_deg": 10.0,
        "crossing_rate": 0.02,
        "angular_res_mean_deg": 30.0,
        "cluster_mean_sep_ratio": 3.0,
    }


def test_composite_undirected_rescales_retained_metrics() -> None:
    """Composite undirected returns 100 for perfect retained metrics."""
    metrics = {
        "edge_length_cv": 0.0,
        "overlap_count": 0,
        "crossing_rate": 0.0,
        "angular_res_mean_deg": 180.0,
        "cluster_sep": 1.0,
    }

    assert composite_undirected(metrics) == pytest.approx(100.0)


def test_composite_auto_directed_matches_composite() -> None:
    """Composite auto uses the directed score for directed graphs."""
    metrics = _sample_metrics()

    assert composite_auto(metrics, is_semantically_directed=True) == pytest.approx(
        composite(metrics)
    )


def test_composite_auto_undirected_matches_composite_undirected() -> None:
    """Composite auto uses the undirected score for undirected graphs."""
    metrics = _sample_metrics()

    assert composite_auto(metrics, is_semantically_directed=False) == pytest.approx(
        composite_undirected(metrics)
    )


def test_composite_auto_none_matches_composite() -> None:
    """Composite auto defaults to the directed score conservatively."""
    metrics = _sample_metrics()

    assert composite_auto(metrics, is_semantically_directed=None) == pytest.approx(
        composite(metrics)
    )
