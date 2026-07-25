"""Degeneracy consumption tests for the frozen V3 ruler."""

from __future__ import annotations

from typing import List, Tuple

import pytest
import torch

from dagua.eval.ruler_v3 import (
    RulerV3Result,
    _headline_degeneracy_fold,
    _triple_view_scores,
    score_core_v3,
)
from scripts import freeze_v3_rebaseline, native_sprint_score

_FAST_SCORE_KWARGS = {
    "crossing_samples": 1000,
    "neighborhood_samples": 100,
    "stress_sources": 20,
    "stress_targets": 50,
}


def _sizes(count: int) -> torch.Tensor:
    """Return unit node boxes for synthetic V3 tests.

    Parameters
    ----------
    count : int
        Number of node-size rows.

    Returns
    -------
    torch.Tensor
        Node box sizes with shape ``[count, 2]``.
    """
    return torch.ones((count, 2), dtype=torch.float64)


def _score(
    positions: List[Tuple[float, float]],
    edges: List[Tuple[int, int]],
) -> RulerV3Result:
    """Score a synthetic layout with reduced deterministic budgets.

    Parameters
    ----------
    positions : List[Tuple[float, float]]
        Node center positions.
    edges : List[Tuple[int, int]]
        Directed edge pairs.

    Returns
    -------
    RulerV3Result
        V3 ruler result.
    """
    pos = torch.tensor(positions, dtype=torch.float64)
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return score_core_v3(pos, edge_index, _sizes(len(positions)), **_FAST_SCORE_KWARGS)


def _raw_scores(result: RulerV3Result) -> dict[str, float]:
    """Recompute the unfolded composite views from result facets.

    Parameters
    ----------
    result : RulerV3Result
        Folded V3 result.

    Returns
    -------
    dict[str, float]
        Unfolded composite views.
    """
    return _triple_view_scores(result.facets)


def test_degenerate_scale_folds_headline_but_not_diagnostics() -> None:
    """DEGENERATE_SCALE folds headline views by k_DS only."""
    result = _score(
        [(0.0, 0.0), (0.1, 0.0), (0.2, 0.0), (0.3, 0.0)],
        [(0, 1), (1, 2), (2, 3)],
    )
    raw = _raw_scores(result)
    k = float(result.metadata["headline_degeneracy_fold"])

    assert "DEGENERATE_SCALE" in result.flags
    assert 0.25 <= k <= 0.5
    assert result.scores["tiered"] == pytest.approx(raw["tiered"] * k)
    assert result.scores["tiered_linear"] == pytest.approx(raw["tiered_linear"] * k)
    assert result.scores["tiered_capped"] == pytest.approx(raw["tiered_capped"] * k)
    assert result.scores["tiered_hold_instrument"] == pytest.approx(
        raw["tiered_hold_instrument"] * k
    )
    assert result.scores["equal"] == pytest.approx(raw["equal"])
    assert result.scores["tier1_only"] == pytest.approx(raw["tier1_only"])


def test_ds_fold_curve_is_sevhalf() -> None:
    """Pin the DS fold to the round-1-adopted sevhalf curve at points where curves diverge.

    Guards against regressing to the rejected ``sev`` curve or the reverse-engineered
    ``clamp(elr/.25, .25, .5)`` third curve (both give k=0.4357 at the sparse_pair point,
    vs sevhalf's 0.25).
    """

    def k_ds(elr: float) -> float:
        return _headline_degeneracy_fold(
            edge_length_mean=elr,
            node_diag_mean=1.0,
            degenerate_scale=True,
            sprawl_collapse=False,
            occlusion_score=None,
            whitespace_ratio=1.0,
            overlap_count=0,
            overlap_area_severity=0.0,
            clearance_penalty=0.0,
            clearance_contact_pairs=0,
            visual_packing_fill=1.0,
            num_nodes=4,
        )

    # sparse_pair operating point: sevhalf floors to 0.25 (rejected/accidental curves -> 0.4357).
    assert k_ds(0.1089) == pytest.approx(0.25)
    # mid-band divergence: sevhalf ramps to 0.4 (accidental clamp(.,.25,.5) curve -> 0.5).
    assert k_ds(0.2) == pytest.approx(0.4)
    # DS-flag boundary: half fold.
    assert k_ds(0.25) == pytest.approx(0.5)


def test_sprawl_collapse_folds_and_pure_sprawl_releases() -> None:
    """SPRAWL rows fold only when C4 is below the pure-sprawl boundary."""
    sprawl_collapse = _score(
        [(0.0, 0.0), (0.0, 0.0), (1000.0, 0.0), (0.0, 1000.0)],
        [(0, 1), (1, 2), (2, 3)],
    )
    pure_sprawl = _score(
        [(0.0, 0.0), (1000.0, 0.0), (0.0, 1000.0), (1000.0, 1000.0)],
        [(0, 1), (1, 2), (2, 3)],
    )
    sprawl_raw = _raw_scores(sprawl_collapse)
    pure_raw = _raw_scores(pure_sprawl)

    assert {"SPRAWL", "SPRAWL_COLLAPSE"}.issubset(set(sprawl_collapse.flags))
    assert sprawl_collapse.facets["C4"].score < 1.0
    assert sprawl_collapse.scores["tiered"] < sprawl_raw["tiered"]
    assert "SPRAWL" in pure_sprawl.flags
    assert "SPRAWL_COLLAPSE" not in pure_sprawl.flags
    assert pure_sprawl.facets["C4"].score == pytest.approx(1.0)
    assert pure_sprawl.scores["tiered"] == pytest.approx(pure_raw["tiered"])


def test_coincident_collapse_detector_flags_without_headline_fold() -> None:
    """cf publishes COINCIDENT_COLLAPSE but is not a headline fold input."""
    collapsed = _score(
        [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
        [(0, 1), (1, 2), (2, 3)],
    )
    spread = _score(
        [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0)],
        [(0, 1), (1, 2), (2, 3)],
    )
    cf_only_consumed = _score(
        [(0.0, 0.0), (0.2, 0.0), (10.0, 0.0), (20.0, 0.0)],
        [(0, 2), (1, 3), (2, 3)],
    )
    cf_raw = _raw_scores(cf_only_consumed)

    assert collapsed.metadata["coincident_collapse_fraction"] == pytest.approx(1.0)
    assert "COINCIDENT_COLLAPSE" in collapsed.flags
    assert spread.metadata["coincident_collapse_fraction"] == pytest.approx(0.0)
    assert "COINCIDENT_COLLAPSE" not in spread.flags
    assert "COINCIDENT_COLLAPSE" in cf_only_consumed.flags
    assert "DEGENERATE_SCALE" not in cf_only_consumed.flags
    assert "SPRAWL_COLLAPSE" not in cf_only_consumed.flags
    assert cf_only_consumed.facets["C4"].metadata["overlap_count"] == 1
    assert cf_only_consumed.metadata["headline_degeneracy_fold"] == pytest.approx(0.744)
    assert cf_only_consumed.scores["tiered"] == pytest.approx(cf_raw["tiered"] * 0.744)


def test_occlusion_floor_flag_is_not_fold_input_but_contacts_can_fold() -> None:
    """OCCLUSION_FLOOR remains diagnostic while zero-overlap contacts can fold."""
    result = _score(
        [(0.0, 0.0), (1.1, 0.0), (3.0, 0.0), (5.0, 0.0)],
        [(0, 1), (1, 2), (2, 3)],
    )
    raw = _raw_scores(result)

    assert result.flags == ("OCCLUSION_FLOOR",)
    assert result.facets["C4"].metadata["overlap_count"] == 0
    assert result.facets["C4"].metadata["clearance_contact_pairs"] == 1
    assert result.metadata["headline_degeneracy_fold"] == pytest.approx(0.9583208215578847)
    assert result.scores["tiered"] == pytest.approx(
        raw["tiered"] * result.metadata["headline_degeneracy_fold"]
    )
    assert result.scores["equal"] == pytest.approx(raw["equal"])


def test_degeneracy_eligibility_precedes_referee_and_score_keys() -> None:
    """Champion selection ranks eligible rows above DS, SPCV, and cf rows."""
    eligible_high = {"v3_row_flags": [], "v3_referee_eligibility_key": [1, -0.0], "v3_tiered": 10.0}
    of_only = {
        "v3_row_flags": ["OCCLUSION_FLOOR"],
        "v3_referee_eligibility_key": [1, -0.0],
        "v3_tiered": 10.0,
    }
    ds_high = {
        "v3_row_flags": ["DEGENERATE_SCALE"],
        "v3_referee_eligibility_key": [1, -0.0],
        "v3_tiered": 99.0,
    }
    spcv_high = {
        "v3_row_flags": ["SPRAWL_COLLAPSE"],
        "v3_referee_eligibility_key": [1, -0.0],
        "v3_tiered": 99.0,
    }
    cf_high = {
        "v3_row_flags": ["COINCIDENT_COLLAPSE"],
        "v3_referee_eligibility_key": [1, -0.0],
        "v3_tiered": 99.0,
    }

    native_key = native_sprint_score._selection_key
    freeze_key = freeze_v3_rebaseline._selection_key
    assert native_key(eligible_high, "v3_tiered", native_selection=False) > native_key(
        ds_high, "v3_tiered", native_selection=False
    )
    assert native_key(of_only, "v3_tiered", native_selection=False) > native_key(
        spcv_high, "v3_tiered", native_selection=False
    )
    assert freeze_key(eligible_high, native=False) > freeze_key(cf_high, native=False)
    assert freeze_key(of_only, native=True) > freeze_key(ds_high, native=True)
