"""Tests for undirected metric composite variants."""

from __future__ import annotations

import pytest

from dagua.metrics import _COMMON_WEIGHTS, composite, composite_auto, composite_undirected


def _sample_metrics() -> dict[str, float]:
    """Create a complete metrics dictionary with non-trivial values.

    Returns
    -------
    dict[str, float]
        Metrics with all keys used by directed and undirected composites.
    """
    metrics = {name: 0.6 for name in _COMMON_WEIGHTS}
    metrics.update(
        {
            "declared_hierarchical": True,
            "directed_flow_score": 0.8,
            "depth_order_score": 0.7,
        }
    )
    return metrics


def test_composite_undirected_rescales_retained_metrics() -> None:
    """Composite undirected returns 100 for perfect retained metrics."""
    metrics = {name: 1.0 for name in _COMMON_WEIGHTS}

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


def test_composite_auto_none_matches_common() -> None:
    """Absent semantic direction routes to the common ruler."""
    metrics = _sample_metrics()

    assert composite_auto(metrics, is_semantically_directed=None) == pytest.approx(
        composite_undirected(metrics)
    )
