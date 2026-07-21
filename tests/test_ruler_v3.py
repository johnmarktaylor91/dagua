"""Regression tests for the scoring-only V3 ruler core."""

from __future__ import annotations

from typing import Tuple

import pytest
import torch

from dagua.eval.ruler_v3 import (
    CORE_TIERS,
    PURE_GEOMETRY_FACETS,
    WHITESPACE_RATIO_HI,
    crossing_weight_multiplier,
    renormalized_score,
    score_core_v3,
)


def _probe_layout() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a compact graph that exercises all V3 CORE facets.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Positions ``[N, 2]``, edge index ``[2, E]``, and node sizes ``[N, 2]``.
    """
    pos = torch.tensor(
        [
            [0.0, 0.0],
            [4.0, 0.0],
            [4.0, 4.0],
            [0.0, 4.0],
            [8.0, 4.0],
            [12.0, 4.0],
        ],
        dtype=torch.float64,
    )
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 0, 1, 3, 4],
            [1, 2, 3, 0, 2, 3, 4, 5],
        ],
        dtype=torch.long,
    )
    node_sizes = torch.full((6, 2), 0.5, dtype=torch.float64)
    return pos, edge_index, node_sizes


def test_unit_invariance_for_all_core_facets() -> None:
    """Joint scaling of positions and node sizes must preserve every facet."""
    pos, edge_index, node_sizes = _probe_layout()
    baseline = score_core_v3(pos, edge_index, node_sizes)

    for alpha in (0.02, 0.1, 1.0, 10.0, 50.0):
        scaled = score_core_v3(alpha * pos, edge_index, alpha * node_sizes)
        for code, facet in baseline.facets.items():
            assert scaled.facets[code].score == pytest.approx(facet.score, abs=1e-9)
        assert scaled.scores["tiered"] == pytest.approx(baseline.scores["tiered"], abs=1e-9)


def test_position_only_scale_invariance_for_pure_geometry_facets() -> None:
    """Pure-geometry facets must ignore position-only scale changes."""
    pos, edge_index, node_sizes = _probe_layout()
    baseline = score_core_v3(pos, edge_index, node_sizes)

    for alpha in (0.02, 0.1, 1.0, 10.0, 50.0):
        scaled = score_core_v3(alpha * pos, edge_index, node_sizes)
        for code in PURE_GEOMETRY_FACETS:
            assert scaled.facets[code].score == pytest.approx(
                baseline.facets[code].score,
                abs=1e-9,
            )


def test_c5_whitespace_band_flags_crater_and_stays_flat_on_legitimate_chain() -> None:
    """The C5 band should tolerate normal chain extent but penalize crater sprawl."""
    node_count = 10
    pos = torch.stack(
        (
            torch.arange(node_count, dtype=torch.float64) * 3.0,
            torch.zeros(node_count, dtype=torch.float64),
        ),
        dim=1,
    )
    edge_index = torch.tensor(
        [
            list(range(node_count - 1)),
            list(range(1, node_count)),
        ],
        dtype=torch.long,
    )
    node_sizes = torch.ones((node_count, 2), dtype=torch.float64)

    legitimate = score_core_v3(pos, edge_index, node_sizes)
    crater = score_core_v3(1000.0 * pos, edge_index, node_sizes)

    assert legitimate.facets["C5"].score == pytest.approx(1.0)
    assert crater.facets["C5"].score is not None
    assert crater.facets["C5"].score < 0.01
    assert crater.facets["C5"].metadata["whitespace_ratio"] > WHITESPACE_RATIO_HI
    assert "SPRAWL" in crater.flags


def test_applicability_renormalization_excludes_missing_facets() -> None:
    """Missing facets should leave the denominator instead of receiving neutral fill."""
    score = renormalized_score(
        {"present": 0.5, "missing": None},
        {"present": 2.0, "missing": 100.0},
    )

    assert score == pytest.approx(50.0)


def test_triple_view_composite_and_tier_weights_are_published() -> None:
    """V3 should publish equal, tiered, and Tier-1-only scoring views."""
    pos, edge_index, node_sizes = _probe_layout()
    result = score_core_v3(pos, edge_index, node_sizes)

    assert set(result.scores) == {"tiered", "equal", "tier1_only"}
    assert result.facets["C1"].base_weight == pytest.approx(4.0)
    assert result.facets["C5"].base_weight == pytest.approx(2.0)
    assert result.facets["C9"].base_weight == pytest.approx(1.0)
    assert result.facets["C2"].effective_weight == pytest.approx(
        4.0 * crossing_weight_multiplier(pos.shape[0])
    )
    assert sum(1 for tier in CORE_TIERS.values() if tier == 1) == 3
    assert result.coverage["applicable_facets"] == 10
    assert 0.0 <= result.scores["equal"] <= 100.0
    assert 0.0 <= result.scores["tiered"] <= 100.0
    assert 0.0 <= result.scores["tier1_only"] <= 100.0
