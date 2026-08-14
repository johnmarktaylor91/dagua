"""Robust trimmed-core frames and intrinsic-unit frame measurements."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch

from dagua.eval.ruler_v4.scene import Scene

N_SMALL = 50
N_FLOOR = 4
TRIM_RATE = 0.02
MIN_TRIM = 2
MIN_CORE = 4
CORE_FRACTION = 0.5
MAD_MULTIPLIER = 3.0
HALF_EXTENT_FLOOR = 0.5


@dataclass(frozen=True)
class RobustFrame:
    """Robust axis-aligned scene frame.

    Parameters
    ----------
    center : torch.Tensor
        Frame center with shape ``[2]``.
    half_extents : torch.Tensor
        Floored robust half extents with shape ``[2]``.
    trim_count : int
        Explicit per-side trim count.
    regime : int
        U21 frame regime, one through three.
    floor_bound : tuple[bool, bool]
        Whether the universal floor bound on each axis.
    """

    center: torch.Tensor
    half_extents: torch.Tensor
    trim_count: int
    regime: int
    floor_bound: Tuple[bool, bool]

    @property
    def area(self) -> float:
        """Return full frame area.

        Returns
        -------
        float
            ``4 * half_width * half_height``.
        """

        return float(4.0 * torch.prod(self.half_extents).item())


def minimum_core_count(node_count: int) -> int:
    """Return the normative minimum retained core population.

    Parameters
    ----------
    node_count : int
        Number of primitive centers.

    Returns
    -------
    int
        At least half the population and at least four objects.
    """

    return max(MIN_CORE, math.ceil(CORE_FRACTION * node_count))


def trim_count(node_count: int) -> int:
    """Return the explicit U21 trim count per side and direction.

    Parameters
    ----------
    node_count : int
        Number of primitive centers.

    Returns
    -------
    int
        Clamped trim count. Small-N regimes report the same input-only count even
        though their spread estimator does not discard objects.
    """

    if node_count < 0:
        raise ValueError("node_count must be nonnegative")
    maximum = max(MIN_TRIM, math.floor((node_count - minimum_core_count(node_count)) / 2.0))
    return min(max(math.ceil(TRIM_RATE * node_count), MIN_TRIM), maximum)


def intrinsic_unit(scene: Scene) -> float:
    """Return the median declared primitive diagonal.

    Parameters
    ----------
    scene : Scene
        Validated scene with StyleContract-derived node boxes.

    Returns
    -------
    float
        Positive intrinsic unit in scene coordinates.
    """

    diagonals = torch.stack(
        [
            2.0 * torch.linalg.vector_norm(box.half_extents.to(torch.float64))
            for box in scene.node_boxes
        ]
    )
    if diagonals.numel() == 0:
        raise ValueError("intrinsic unit requires at least one primitive")
    value = float(torch.median(diagonals).item())
    if value <= 0.0 or not math.isfinite(value):
        raise ValueError("primitive diagonals must be finite and positive")
    return value


def _median(values: torch.Tensor) -> torch.Tensor:
    """Compute the conventional midpoint median along axis zero.

    Parameters
    ----------
    values : torch.Tensor
        Float64 tensor with shape ``[N, D]``.

    Returns
    -------
    torch.Tensor
        Median vector with shape ``[D]``.
    """

    ordered, _ = torch.sort(values, dim=0)
    count = ordered.shape[0]
    if count % 2:
        return ordered[count // 2]
    return (ordered[count // 2 - 1] + ordered[count // 2]) / 2.0


def robust_frame(positions: torch.Tensor, unit: float) -> RobustFrame:
    """Construct the U21 robust frame without any bounding-box extrema.

    Parameters
    ----------
    positions : torch.Tensor
        Finite node centers with shape ``[N, 2]``.
    unit : float
        Positive median primitive diagonal.

    Returns
    -------
    RobustFrame
        Exact input-count-selected trimmed-core or median/MAD frame.

    Raises
    ------
    ValueError
        If geometry is empty, malformed, non-finite, or has a nonpositive unit.
    """

    if positions.ndim != 2 or positions.shape[1] != 2 or positions.shape[0] == 0:
        raise ValueError("positions must have shape [N, 2] with N >= 1")
    points = positions.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(points).all()) or unit <= 0.0 or not math.isfinite(unit):
        raise ValueError("frame inputs must be finite and unit must be positive")
    count = points.shape[0]
    trim = trim_count(count)
    if count >= N_SMALL:
        ordered, _ = torch.sort(points, dim=0)
        lower = ordered[trim]
        upper = ordered[count - trim - 1]
        center = (lower + upper) / 2.0
        raw_half = (upper - lower) / 2.0
        regime = 1
    else:
        center = _median(points)
        deviation = torch.abs(points - center)
        raw_half = MAD_MULTIPLIER * _median(deviation)
        regime = 2 if count >= N_FLOOR else 3
    floor = torch.full((2,), HALF_EXTENT_FLOOR * unit, dtype=torch.float64)
    bound = raw_half <= floor
    half_extents = torch.maximum(raw_half, floor)
    return RobustFrame(
        center=center.clone(),
        half_extents=half_extents.clone(),
        trim_count=trim,
        regime=regime,
        floor_bound=(bool(bound[0]), bool(bound[1])),
    )


def box_outside_area(
    box_center: torch.Tensor,
    box_half_extents: torch.Tensor,
    frame: RobustFrame,
) -> float:
    """Compute exact axis-aligned primitive area outside a robust frame.

    Parameters
    ----------
    box_center : torch.Tensor
        Primitive center with shape ``[2]``.
    box_half_extents : torch.Tensor
        Primitive half extents with shape ``[2]``.
    frame : RobustFrame
        Input robust frame.

    Returns
    -------
    float
        Nonnegative escaped primitive area.
    """

    box_low = box_center - box_half_extents
    box_high = box_center + box_half_extents
    frame_low = frame.center - frame.half_extents
    frame_high = frame.center + frame.half_extents
    overlap_extent = torch.clamp(
        torch.minimum(box_high, frame_high) - torch.maximum(box_low, frame_low), min=0.0
    )
    total = 4.0 * torch.prod(box_half_extents)
    inside = torch.prod(overlap_extent)
    return float(torch.clamp(total - inside, min=0.0).item())


def overflow_anchor(scene: Scene, frame: RobustFrame) -> float:
    """Compute the input-anchored achievable minimum overflow from U21.

    Parameters
    ----------
    scene : Scene
        Validated scene and StyleContract-derived primitives.
    frame : RobustFrame
        Robust frame whose explicit trim count selects the anchor.

    Returns
    -------
    float
        Closed-form escaped-mass anchor normalized by reference core area.
    """

    count = scene.node_count
    grid_side = math.ceil(math.sqrt(count))
    trim = frame.trim_count
    surviving_side = max(0, grid_side - 2 * trim)
    inside_count = min(count, surviving_side**2)
    outside_count = count - inside_count
    primitive_area = sum(
        float(4.0 * torch.prod(box.half_extents).item()) for box in scene.node_boxes
    )
    mean_area = primitive_area / count
    max_diagonal = max(
        float(2.0 * torch.linalg.vector_norm(box.half_extents).item()) for box in scene.node_boxes
    )
    core_steps = grid_side - 1 - 2 * trim
    if core_steps <= 0:
        # U21 defines the below-floor value by continuity at N_small.  Reusing the
        # N_small grid geometry keeps the anchor input-only and avoids a zero area.
        grid_side = math.ceil(math.sqrt(N_SMALL))
        reference_trim = trim_count(N_SMALL)
        surviving_side = max(0, grid_side - 2 * reference_trim)
        inside_count = min(N_SMALL, surviving_side**2)
        outside_count = N_SMALL - inside_count
        core_steps = grid_side - 1 - 2 * reference_trim
    core_area = (core_steps * max_diagonal) ** 2
    return float((outside_count * mean_area) / core_area)


def overflow_defect(scene: Scene, frame: RobustFrame) -> Tuple[float, float, float]:
    """Compute U21 escaped mass and its input-anchored smooth defect.

    Parameters
    ----------
    scene : Scene
        Validated scene.
    frame : RobustFrame
        Robust frame.

    Returns
    -------
    tuple[float, float, float]
        Raw escaped mass per frame area, anchor, and bounded defect.
    """

    escaped_area = sum(
        box_outside_area(box.center, box.half_extents, frame) for box in scene.node_boxes
    )
    mass_out = escaped_area / frame.area
    anchor = overflow_anchor(scene, frame)
    excess = max(0.0, mass_out - anchor)
    smooth = 0.0 if excess <= 0.0 else excess**2 / (excess + 0.05)
    return mass_out, anchor, smooth / (1.0 + smooth)
