"""Exact numeric goldens for shared RULER V4 contract mathematics."""

from __future__ import annotations

import math

import pytest
import torch

from dagua.eval.ruler_v4._util import global_blend, mean_result
from dagua.eval.ruler_v4.edges import _segment_box_deficit_integral
from dagua.eval.ruler_v4.scene import BoxGeometry, ResultState


def test_global_blend_one_sacrificed_object_exact() -> None:
    """Pin U11 section 17's fractional trim, CVaR, and smooth maximum."""

    defects = [0.0] * 9 + [1.0]
    trimmed = 1.0 / 18.0
    smooth_maximum = 1.0 + 0.05 * math.log((1.0 + 9.0 * math.exp(-20.0)) / 10.0)
    expected = 0.65 * trimmed + 0.25 + 0.10 * smooth_maximum
    assert global_blend(defects) == pytest.approx(expected, abs=1e-15)


@pytest.mark.parametrize("defect", [0.0, 1.0])
def test_global_blend_singleton_identity_exact(defect: float) -> None:
    """Pin the blend's exact singleton limits.

    Parameters
    ----------
    defect : float
        Endpoint defect supplied by pytest.
    """

    assert global_blend([defect]) == pytest.approx(defect, abs=0.0)


def test_frozen_ratio_and_noisy_or_composition_exact() -> None:
    """Pin manifest row ratios and the fixed noisy-or family."""

    weighted = mean_result("U38", {"U38.L_clear": 0.2, "U38.L_pack": 0.4, "U38.L_prop": 0.8})
    assert weighted.state is ResultState.VALUE
    assert weighted.value == pytest.approx(0.55 * 0.2 + 0.30 * 0.4 + 0.15 * 0.8)

    noisy = mean_result("U20a", {"U20a.i": 0.2, "U20a.ii": 0.4, "U20a.iii": 0.8})
    expected = 1.0 - (1.0 - 0.2) ** 0.45 * (1.0 - 0.4) ** 0.25 * (1.0 - 0.8) ** 0.30
    assert noisy.state is ResultState.VALUE
    assert noisy.value == pytest.approx(expected, abs=1e-15)


def test_u10_band_boundary_has_exact_zero_integral() -> None:
    """Pin U10 golden 2 at the exact clearance-band boundary."""

    box = BoxGeometry(
        center=torch.zeros(2, dtype=torch.float64),
        half_extents=torch.ones(2, dtype=torch.float64),
        owner=2,
    )
    integral, penetrates = _segment_box_deficit_integral(
        torch.tensor([-2.0, 1.5], dtype=torch.float64),
        torch.tensor([2.0, 1.5], dtype=torch.float64),
        box,
        0.0,
        1.0,
    )
    assert integral == pytest.approx(0.0, abs=0.0)
    assert penetrates is False


def test_u10_center_pierce_closed_form_exact() -> None:
    """Pin one U10 section-5a face/interior closed-form integral."""

    box = BoxGeometry(
        center=torch.zeros(2, dtype=torch.float64),
        half_extents=torch.ones(2, dtype=torch.float64),
        owner=2,
    )
    integral, penetrates = _segment_box_deficit_integral(
        torch.tensor([-2.0, 0.0], dtype=torch.float64),
        torch.tensor([2.0, 0.0], dtype=torch.float64),
        box,
        0.0,
        1.0,
    )
    assert integral == pytest.approx(9.0, abs=1e-14)
    assert penetrates is True
