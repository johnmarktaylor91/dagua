"""Overlap and continuity pins for the V3 headline fold."""

from __future__ import annotations

from typing import Optional

import pytest
import torch

from dagua.eval.ruler_v3 import _headline_degeneracy_fold, _smooth_clearance_occlusion_score


def _fold(
    *,
    edge_length_mean: float = 1.0,
    node_diag_mean: float = 1.0,
    degenerate_scale: bool = False,
    sprawl_collapse: bool = False,
    occlusion_score: Optional[float] = None,
    whitespace_ratio: float = 1.0,
    overlap_count: int = 0,
    overlap_area_severity: float = 0.0,
    clearance_penalty: float = 0.0,
    clearance_contact_pairs: int = 0,
    visual_packing_fill: float = 1.0,
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
    overlap_area_severity : float, optional
        Exact strict-overlap area severity harvested from C4 metadata.
    clearance_penalty : float, optional
        Total sub-band contact penalty harvested from C4 metadata.
    clearance_contact_pairs : int, optional
        Number of sub-band contact pairs that contributed to ``clearance_penalty``.
    visual_packing_fill : float, optional
        Sum of visual-box area divided by union bounding-box area.
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
        overlap_area_severity=overlap_area_severity,
        clearance_penalty=clearance_penalty,
        clearance_contact_pairs=clearance_contact_pairs,
        visual_packing_fill=visual_packing_fill,
        num_nodes=num_nodes,
    )


def _c4_probe(centers: torch.Tensor, sizes: torch.Tensor) -> dict[str, float]:
    """Evaluate C4 metadata for synthetic visual boxes.

    Parameters
    ----------
    centers : torch.Tensor
        Visual box centers with shape ``[N, 2]``.
    sizes : torch.Tensor
        Visual box sizes with shape ``[N, 2]``.

    Returns
    -------
    dict[str, float]
        C4 score metadata from the production clearance loop.
    """
    return _smooth_clearance_occlusion_score(
        centers.to(dtype=torch.float64),
        sizes.to(dtype=torch.float64),
        label_inclusive=True,
        seed=0,
    )


@pytest.mark.parametrize("fill", [0.0, 0.55, 0.60])
def test_zero_overlap_low_fill_is_exact_identity_by_construction(fill: float) -> None:
    """Pin ``ov == 0`` and ``F <= 0.60`` to an exact identity multiplier."""
    assert (
        _fold(
            edge_length_mean=1.0,
            node_diag_mean=2.0,
            overlap_count=0,
            overlap_area_severity=0.0,
            clearance_penalty=0.0,
            clearance_contact_pairs=0,
            visual_packing_fill=fill,
        )
        == 1.0
    )


def test_zero_overlap_high_fill_can_penalize_clearance_contacts() -> None:
    """Pin removal of the old ``overlap_count == 0`` identity branch."""
    assert _fold(
        overlap_count=0,
        overlap_area_severity=0.0,
        clearance_penalty=0.25,
        clearance_contact_pairs=1,
        visual_packing_fill=0.75,
    ) == pytest.approx(0.9814814814814815)


def test_exact_intersection_bounds_nested_box_severity() -> None:
    """Pin nested-box intersection to the smaller box area, not gap product."""
    result = _c4_probe(
        torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        torch.tensor([[100.0, 100.0], [10.0, 10.0]]),
    )

    assert result["overlap_count"] == 1
    assert result["clearance_contact_pairs"] == 1
    assert result["overlap_area_severity"] * 2.0 == pytest.approx(1.0)
    assert result["overlap_area_severity"] * 2.0 != pytest.approx(30.25)


@pytest.mark.parametrize(
    ("center", "size"),
    [
        ((0.0, 0.0), (1.0, 1.0)),
        ((0.25, 0.0), (1.0, 1.0)),
        ((0.25, 0.25), (2.0, 1.0)),
        ((0.0, 0.0), (10.0, 10.0)),
        ((0.3, -0.4), (0.5, 4.0)),
    ],
)
def test_pair_overlap_fraction_is_bounded_by_construction(
    center: tuple[float, float],
    size: tuple[float, float],
) -> None:
    """Pin ``f_ij <= 1`` across representative overlap geometries."""
    result = _c4_probe(
        torch.tensor([[0.0, 0.0], center]),
        torch.tensor([[1.0, 1.0], size]),
    )

    assert result["overlap_count"] == 1
    assert result["clearance_contact_pairs"] == 1
    assert 0.0 < result["overlap_area_severity"] * 2.0 <= 1.0


@pytest.mark.parametrize(
    ("fill", "expected"),
    [
        (0.55, 1.0),
        (0.60, 1.0),
        (0.675, 0.9375),
        (0.75, 0.5),
        (0.90, 0.5),
    ],
)
def test_overlap_packing_fill_gate(fill: float, expected: float) -> None:
    """Pin the co-signed P(F) gate at the frozen fill landmarks."""
    assert _fold(
        overlap_count=1,
        clearance_penalty=1.0,
        clearance_contact_pairs=2,
        visual_packing_fill=fill,
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("severity", "expected"),
    [
        (0.0, 1.0),
        (0.125, 0.9375),
        (0.25, 0.5),
        (0.5, 0.5),
    ],
)
def test_overlap_severity_curve_values(severity: float, expected: float) -> None:
    """Pin S composition and k_OV saturation at S = 0, 0.125, 0.25, and 0.5."""
    assert _fold(
        overlap_count=1,
        overlap_area_severity=severity,
        visual_packing_fill=1.0,
    ) == pytest.approx(expected)


def test_clearance_penalty_decomposition_invariant() -> None:
    """Pin ``clearance_penalty == J*n + overlap_count + n_abut``."""
    result = _c4_probe(
        torch.tensor(
            [
                [0.0, 0.0],
                [0.5, 0.0],
                [1.5, 0.0],
                [2.4, 1.4],
            ]
        ),
        torch.ones((4, 2)),
    )

    assert result["overlap_count"] == 1
    assert result["clearance_contact_pairs"] == 4
    assert result["clearance_abut_count"] == 1
    assert result["clearance_penalty"] == pytest.approx(
        result["packed_seam_severity"] * 4.0
        + result["overlap_count"]
        + result["clearance_abut_count"]
    )


def test_overlap_contact_shrinkage_kappa_is_two() -> None:
    """Pin ``Cbar = clearance_penalty / (clearance_contact_pairs + 2)``."""
    one_pair = _fold(
        overlap_count=1,
        clearance_penalty=0.25,
        clearance_contact_pairs=1,
        visual_packing_fill=0.75,
    )
    ten_pairs = _fold(
        overlap_count=1,
        clearance_penalty=0.25,
        clearance_contact_pairs=10,
        visual_packing_fill=0.75,
    )

    assert one_pair == pytest.approx(0.9814814814814815)
    assert ten_pairs == pytest.approx(0.9997106481481481)
    assert one_pair != pytest.approx(ten_pairs)


def test_seam_clearance_is_l2_positive_part_norm() -> None:
    """Pin the diagonal-gap seam debt to the L2 positive-part norm."""
    out_of_band = _c4_probe(
        torch.tensor([[0.0, 0.0], [1.6, 1.6]]),
        torch.ones((2, 2)),
    )
    in_band = _c4_probe(
        torch.tensor([[0.0, 0.0], [1.3, 1.3]]),
        torch.ones((2, 2)),
    )

    assert out_of_band["packed_seam_severity"] == 0.0
    assert out_of_band["clearance_penalty"] == 0.0
    assert out_of_band["clearance_contact_pairs"] == 0
    assert in_band["packed_seam_severity"] == pytest.approx(0.07999999999999996)
    assert in_band["clearance_penalty"] == pytest.approx(0.15999999999999992)
    assert in_band["clearance_contact_pairs"] == 1


@pytest.mark.parametrize(
    ("overlap_area_severity", "clearance_penalty", "visual_packing_fill"),
    [
        (float("nan"), 0.0, 1.0),
        (0.0, float("nan"), 1.0),
        (0.0, 0.0, float("nan")),
        (float("inf"), 0.0, 1.0),
        (0.0, float("-inf"), 1.0),
        (0.0, 0.0, float("inf")),
    ],
)
def test_nonfinite_overlap_severity_saturates_nonzero_ov_fold(
    overlap_area_severity: float,
    clearance_penalty: float,
    visual_packing_fill: float,
) -> None:
    """Pin non-finite OV telemetry to the saturated penalty, not identity."""
    assert _fold(
        overlap_count=1,
        overlap_area_severity=overlap_area_severity,
        clearance_penalty=clearance_penalty,
        clearance_contact_pairs=1,
        visual_packing_fill=visual_packing_fill,
    ) == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("centers", "sizes", "match"),
    [
        (
            torch.tensor([[0.0, 0.0], [float("nan"), 0.0]]),
            torch.ones((2, 2)),
            "centers",
        ),
        (
            torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
            torch.tensor([[1.0, 1.0], [float("inf"), 1.0]]),
            "sizes",
        ),
        (
            torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
            torch.tensor([[1.0, 1.0], [0.0, 1.0]]),
            "sizes",
        ),
        (
            torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
            torch.full((2, 2), 1.0e-200, dtype=torch.float64),
            "areas",
        ),
    ],
)
def test_c4_rejects_unusable_visual_box_geometry(
    centers: torch.Tensor,
    sizes: torch.Tensor,
    match: str,
) -> None:
    """Pin invalid visual boxes to fail closed before severity accumulation."""
    with pytest.raises(ValueError, match=match):
        _c4_probe(centers, sizes)


def test_c4_empty_geometry_remains_identity() -> None:
    """Pin the n=0 C4 branch to an identity score with no box-area work."""
    result = _c4_probe(torch.empty((0, 2)), torch.empty((0, 2)))

    assert result["overlap_count"] == 0
    assert result["node_occlusion_score"] == 1.0
    assert result["clearance_contact_pairs"] == 0
    assert result["visual_packing_fill"] == 0.0


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
    assert _fold(
        edge_length_mean=1.0,
        node_diag_mean=1.0,
        overlap_count=1,
        overlap_area_severity=0.25,
    ) == pytest.approx(0.5)
    assert _fold(edge_length_mean=0.10, node_diag_mean=1.0, overlap_count=0) == pytest.approx(0.25)
    assert _fold(
        edge_length_mean=1.0,
        node_diag_mean=1.0,
        sprawl_collapse=True,
        occlusion_score=0.90,
        whitespace_ratio=1000.0,
        overlap_count=0,
    ) == pytest.approx(0.25)
