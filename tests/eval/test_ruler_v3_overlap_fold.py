"""Overlap and continuity pins for the V3 headline fold."""

from __future__ import annotations

from typing import Optional

import pytest

from dagua.eval.ruler_v3 import _headline_degeneracy_fold


def _fold(
    *,
    edge_length_mean: float = 1.0,
    node_diag_mean: float = 1.0,
    degenerate_scale: bool = False,
    sprawl_collapse: bool = False,
    occlusion_score: Optional[float] = None,
    whitespace_ratio: float = 1.0,
    overlap_count: int = 0,
    num_nodes: int = 100,
) -> float:
    """Evaluate the headline fold with explicit defaults for isolated leg tests.

    Parameters
    ----------
    edge_length_mean : float, optional
        Mean center-to-center edge length.
    node_diag_mean : float, optional
        Mean node-box diagonal.
    degenerate_scale : bool, optional
        Published degenerate-scale flag value; retained for compatibility.
    sprawl_collapse : bool, optional
        Whether the frozen sprawl predicate is active.
    occlusion_score : float | None, optional
        C4 node-occlusion score.
    whitespace_ratio : float, optional
        C5 ``visual_area / area_floor`` ratio.
    overlap_count : int, optional
        Count of overlapping node visual-box pairs from C4 metadata.
    num_nodes : int, optional
        Number of graph nodes used to normalize overlap density.

    Returns
    -------
    float
        Headline fold multiplier.
    """
    return _headline_degeneracy_fold(
        edge_length_mean=edge_length_mean,
        node_diag_mean=node_diag_mean,
        degenerate_scale=degenerate_scale,
        sprawl_collapse=sprawl_collapse,
        occlusion_score=occlusion_score,
        whitespace_ratio=whitespace_ratio,
        overlap_count=overlap_count,
        num_nodes=num_nodes,
    )


@pytest.mark.parametrize("num_nodes", [0, 1, 7, 100])
def test_zero_overlap_leg_is_exact_identity(num_nodes: int) -> None:
    """Pin ``overlap_count == 0`` to an exact identity multiplier."""
    assert (
        _fold(
            edge_length_mean=1.0,
            node_diag_mean=2.0,
            overlap_count=0,
            num_nodes=num_nodes,
        )
        == 1.0
    )


@pytest.mark.parametrize(
    ("overlap_count", "expected"),
    [
        (0, 1.0),
        (25, 0.9375),
        (50, 0.5),
        (100, 0.5),
    ],
)
def test_overlap_fold_curve_values(overlap_count: int, expected: float) -> None:
    """Pin the co-signed cubic overlap leg at q = 0, 0.25, 0.5, and 1.0."""
    assert _fold(overlap_count=overlap_count, num_nodes=100) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("elr", "expected"),
    [
        (0.10, 0.25),
        (0.125, 0.25),
        (0.2497, 0.4994),
        (0.2504, 0.5008),
        (0.30, 0.60),
        (0.50, 1.0),
        (0.60, 1.0),
    ],
)
def test_ungated_ds_fold_curve_is_continuous(elr: float, expected: float) -> None:
    """Pin the ungated DS continuity leg across the old flag boundary."""
    assert _fold(edge_length_mean=elr, node_diag_mean=1.0) == pytest.approx(expected)


def test_old_ds_flag_no_longer_gates_scale_fold() -> None:
    """Verify the same elr receives the same fold on both sides of the old flag value."""
    flagged = _fold(edge_length_mean=0.2504, node_diag_mean=1.0, degenerate_scale=True)
    unflagged = _fold(edge_length_mean=0.2504, node_diag_mean=1.0, degenerate_scale=False)

    assert flagged == pytest.approx(0.5008)
    assert unflagged == pytest.approx(flagged)


def test_min_composition_selects_smallest_leg() -> None:
    """Pin ``min(k_OV, k_DS, k_SPRAWL)`` composition across the three legs."""
    assert _fold(edge_length_mean=1.0, node_diag_mean=1.0, overlap_count=50) == pytest.approx(0.5)
    assert _fold(edge_length_mean=0.10, node_diag_mean=1.0, overlap_count=0) == pytest.approx(0.25)
    assert _fold(
        edge_length_mean=1.0,
        node_diag_mean=1.0,
        sprawl_collapse=True,
        occlusion_score=0.90,
        whitespace_ratio=1000.0,
        overlap_count=0,
    ) == pytest.approx(0.25)
