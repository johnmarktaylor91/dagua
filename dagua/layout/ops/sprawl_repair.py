"""Candidate-only radial repair for layouts with outlier-driven sprawl.

The implementation deliberately mirrors FIELD's two-pass radial winsorize
recipe without importing or changing the scale strategy.  It changes relative
geometry only; this module contains no uniform scale-band calibration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple

import torch

from dagua.layout.ops.base import Op
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

# A two-RMS cap has a corpus-independent tail interpretation: by the second-
# moment bound, no more than one quarter of mass can lie beyond it. FIELD's
# two-pass update then prevents a single extreme from defining its own cap.
SPRAWL_REPAIR_CAP_RMS = 2.0
SPRAWL_REPAIR_ROBUST_QUANTILE = 0.90
SPRAWL_REPAIR_EXTENT_RATIO = 1.25
_MIN_RADIUS = 1.0e-9


def radial_winsorize_positions(
    pos: torch.Tensor,
    *,
    cap_multiple: float = SPRAWL_REPAIR_CAP_RMS,
) -> torch.Tensor:
    """Pull radial outliers onto a deterministic RMS-radius cap.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    cap_multiple : float, default=SPRAWL_REPAIR_CAP_RMS
        Positive radius cap expressed as a multiple of RMS radius.

    Returns
    -------
    torch.Tensor
        Two-pass winsorized positions with shape ``[N, 2]``. Degenerate,
        non-finite, and non-positive-cap inputs are returned byte-identically.

    Notes
    -----
    Two fixed passes match FIELD's recipe: the first extreme can inflate its
    own RMS estimate, while the second pass measures the partially repaired
    mass. No RNG, graph identity, or corpus metadata enters the transform.
    """
    if (
        int(pos.shape[0]) < 2
        or not math.isfinite(float(cap_multiple))
        or float(cap_multiple) <= 0.0
        or not bool(torch.isfinite(pos).all().item())
    ):
        return pos
    work = pos.detach()
    for _pass_index in range(2):
        center = work.mean(dim=0, keepdim=True)
        delta = work - center
        radius_squared = (delta * delta).sum(dim=1)
        rms_radius = torch.sqrt(radius_squared.mean()).clamp_min(_MIN_RADIUS)
        radius = torch.sqrt(radius_squared).clamp_min(_MIN_RADIUS)
        scale = torch.clamp(float(cap_multiple) * rms_radius / radius, max=1.0)
        work = center + delta * scale.unsqueeze(1)
    return work


def robust_full_extent_ratio(
    pos: torch.Tensor,
    *,
    robust_quantile: float = SPRAWL_REPAIR_ROBUST_QUANTILE,
) -> float:
    """Return full radial extent divided by a robust radial extent.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    robust_quantile : float, default=SPRAWL_REPAIR_ROBUST_QUANTILE
        Quantile in ``(0, 1)`` used as the robust occupied-radius boundary.

    Returns
    -------
    float
        Finite full-to-robust radius ratio. ``1.0`` is returned when the
        geometry cannot provide a meaningful positive extent.
    """
    if (
        int(pos.shape[0]) < 2
        or not 0.0 < float(robust_quantile) < 1.0
        or not bool(torch.isfinite(pos).all().item())
    ):
        return 1.0
    work = pos.detach()
    center = torch.quantile(work, 0.5, dim=0)
    radii = torch.linalg.norm(work - center, dim=1)
    robust_radius = float(torch.quantile(radii, float(robust_quantile)).item())
    full_radius = float(torch.max(radii).item())
    if (
        not math.isfinite(robust_radius)
        or not math.isfinite(full_radius)
        or robust_radius <= _MIN_RADIUS
    ):
        return 1.0
    return max(1.0, full_radius / robust_radius)


def sprawl_repair_gate(
    pos: torch.Tensor,
    *,
    c5_whitespace_ratio: Optional[float],
) -> bool:
    """Return whether input-only geometry admits the repair candidate.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    c5_whitespace_ratio : float or None
        Runtime C5 whitespace ratio, when the restricted referee exposes it.

    Returns
    -------
    bool
        ``True`` when C5 exceeds its frozen upper band of ``16`` or the full
        radius extends at least 25 percent past the radius containing 90
        percent of nodes. The latter threshold denotes visible tail leverage,
        rather than a graph- or corpus-fitted identity rule.
    """
    if c5_whitespace_ratio is not None and math.isfinite(float(c5_whitespace_ratio)):
        if float(c5_whitespace_ratio) > 16.0:
            return True
    return robust_full_extent_ratio(pos) > SPRAWL_REPAIR_EXTENT_RATIO


@register_op
@dataclass(frozen=True)
class RadialWinsorize(Op):
    """Apply FIELD-compatible radial winsorization to current positions."""

    cap_multiple: float = SPRAWL_REPAIR_CAP_RMS

    name: ClassVar[str] = "radial_winsorize"
    category: ClassVar[OpCategory] = OpCategory.POSTPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Winsorize the state's positions and preserve all other state.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs; unused by this geometry-only transform.
        state : SolveState
            Mutable solve state carrying positions with shape ``[N, 2]``.
        ctx : RuntimeContext
            Runtime infrastructure; unused because the op is deterministic.

        Returns
        -------
        SolveState
            The same state object with winsorized positions when available.
        """
        del problem, ctx
        if state.pos is not None:
            state.pos = radial_winsorize_positions(
                state.pos,
                cap_multiple=float(self.cap_multiple),
            )
        return state


__all__ = [
    "RadialWinsorize",
    "SPRAWL_REPAIR_CAP_RMS",
    "SPRAWL_REPAIR_EXTENT_RATIO",
    "SPRAWL_REPAIR_ROBUST_QUANTILE",
    "radial_winsorize_positions",
    "robust_full_extent_ratio",
    "sprawl_repair_gate",
]
