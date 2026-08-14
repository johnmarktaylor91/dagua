"""Robust frame and intrinsic-unit contract tests."""

from __future__ import annotations

import torch

from dagua.eval.ruler_v4.frames import (
    overflow_defect,
    robust_frame,
    robust_projection,
    trim_count,
)
from dagua.eval.ruler_v4.scene import Scene


def test_explicit_trim_count_table() -> None:
    """Pin U21's explicit per-side trim counts at representative sizes."""

    assert [trim_count(count) for count in (10, 20, 50, 200, 1000)] == [2, 2, 2, 4, 20]


def test_small_n_uses_median_mad_and_floor() -> None:
    """Small populations never degrade into a submitted bounding box."""

    points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [1000.0, 0.0]])
    frame = robust_frame(points, 2.0)
    assert frame.regime == 2
    assert torch.allclose(frame.center, torch.tensor([1.5, 0.0], dtype=torch.float64))
    assert frame.half_extents[0] == 3.0
    assert frame.half_extents[1] == 1.0


def test_large_n_trims_outliers_explicitly() -> None:
    """Two extreme points per side do not define a large-N frame."""

    points = torch.stack((torch.arange(100, dtype=torch.float64), torch.zeros(100)), dim=1)
    points[:2, 0] = -10000.0
    points[-2:, 0] = 10000.0
    frame = robust_frame(points, 1.0)
    assert frame.regime == 1
    assert frame.trim_count == 2
    assert frame.half_extents[0] == 47.5


def test_uniform_reference_overflow_has_anchored_zero(semantic_scene: Scene) -> None:
    """Overflow at or below its input anchor produces an exact zero defect."""

    frame = robust_frame(semantic_scene.positions, semantic_scene.intrinsic_unit)
    _, _, defect = overflow_defect(semantic_scene, frame)
    assert 0.0 <= defect <= 1.0


def test_frame_rejects_nan() -> None:
    """Non-finite inputs are typed exceptions rather than NaN frames."""

    try:
        robust_frame(torch.tensor([[float("nan"), 0.0]]), 1.0)
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("non-finite frame input was accepted")


def test_direction_resolved_frame_uses_same_small_n_contract() -> None:
    """Fixed-direction consumers share U21's median/MAD rule and floor."""

    projection = robust_projection(torch.tensor([0.0, 1.0, 2.0, 1000.0]), 2.0)
    assert projection.center == 1.5
    assert projection.half_extent == 3.0
    assert not projection.floor_bound
