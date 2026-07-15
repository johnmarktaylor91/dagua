"""Tests that large-graph composites preserve the shared ruler identity."""

from __future__ import annotations

import pytest

from dagua.metrics import (
    _COMMON_WEIGHTS,
    composite,
    composite_auto,
    composite_large,
    composite_large_auto,
    composite_large_undirected,
    composite_undirected,
)


def _sampled_profile() -> dict[str, float]:
    """Return a partial profile representing scale-unavailable terms."""
    return {
        "ksm_score": 0.8,
        "node_occlusion_score": 0.6,
        "edge_length_deviation_score": 0.7,
    }


def test_large_undirected_is_thin_shared_wrapper() -> None:
    """The undirected large API forwards to the common frozen table."""
    metrics = _sampled_profile()
    assert composite_large_undirected(metrics) == pytest.approx(composite_undirected(metrics))
    expected = 100.0 * (25 * 0.8 + 13 * 0.6 + 7 * 0.7) / (25 + 13 + 7)
    assert composite_large_undirected(metrics) == pytest.approx(expected)


def test_large_directed_is_thin_shared_wrapper() -> None:
    """The directed large API forwards to the hierarchy-gated composite."""
    metrics = {
        **_sampled_profile(),
        "declared_hierarchical": True,
        "directed_flow_score": 0.9,
        "depth_order_score": 0.7,
    }
    assert composite_large(metrics) == pytest.approx(composite(metrics))


def test_large_auto_uses_same_semantic_routing() -> None:
    """Large auto requires both semantic direction and hierarchy metadata."""
    hierarchical = {
        **_sampled_profile(),
        "declared_hierarchical": True,
        "directed_flow_score": 0.9,
        "depth_order_score": 0.7,
    }
    assert composite_large_auto(hierarchical, True) == pytest.approx(
        composite_auto(hierarchical, True)
    )
    assert composite_large_auto(hierarchical, False) == pytest.approx(
        composite_undirected(hierarchical)
    )
    assert composite_large_auto(_sampled_profile(), True) == pytest.approx(
        composite_undirected(_sampled_profile())
    )


def test_large_complete_profile_uses_frozen_weights() -> None:
    """A complete sampled profile uses all ten common weights without divergence."""
    metrics = {name: 1.0 for name in _COMMON_WEIGHTS}
    assert composite_large_undirected(metrics) == pytest.approx(100.0)
