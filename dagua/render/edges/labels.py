"""Edge label placement helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

from dagua.render.edges.geometry import CubicBezier, point_tangent_at_fraction, vector_norm


@dataclass(frozen=True)
class EdgeLabelPlacement:
    """Resolved edge label anchor.

    Parameters
    ----------
    x : float
        Anchor x-coordinate in data space.
    y : float
        Anchor y-coordinate in data space.
    angle_degrees : float
        Counterclockwise text rotation angle.
    t : float
        Arc-length parameter location used for placement.
    """

    x: float
    y: float
    angle_degrees: float
    t: float


def _upright_angle(angle_degrees: float) -> float:
    """Flip text angles so labels remain readable.

    Parameters
    ----------
    angle_degrees : float
        Raw tangent angle.

    Returns
    -------
    float
        Upright angle in degrees.
    """
    if angle_degrees > 90.0:
        return angle_degrees - 180.0
    if angle_degrees < -90.0:
        return angle_degrees + 180.0
    return angle_degrees


def place_edge_label(
    curve: CubicBezier,
    label_position: float = 0.5,
    label_offset: float = 8.0,
    label_rotate: bool = False,
    label_side: str = "auto",
) -> EdgeLabelPlacement:
    """Place a label along a curve with optional tangent rotation.

    Parameters
    ----------
    curve : CubicBezier
        Edge centerline.
    label_position : float, default=0.5
        Arc-length fraction where the label is anchored.
    label_offset : float, default=8.0
        Perpendicular offset in data units.
    label_rotate : bool, default=False
        Whether to rotate the text to follow the curve.
    label_side : str, default="auto"
        ``"left"``, ``"right"``, or ``"auto"``.

    Returns
    -------
    EdgeLabelPlacement
        Placement result in data coordinates.
    """
    point, tangent, t = point_tangent_at_fraction(curve, label_position)
    magnitude = vector_norm(tangent)
    if magnitude <= 1e-9:
        normal_x = 0.0
        normal_y = 1.0
        angle = 0.0
    else:
        normal_x = -float(tangent[1]) / magnitude
        normal_y = float(tangent[0]) / magnitude
        angle = math.degrees(math.atan2(float(tangent[1]), float(tangent[0])))

    if label_side == "right":
        normal_x *= -1.0
        normal_y *= -1.0
    elif label_side == "auto" and normal_y < 0.0:
        normal_x *= -1.0
        normal_y *= -1.0

    final_angle = _upright_angle(angle) if label_rotate else 0.0
    return EdgeLabelPlacement(
        x=float(point[0] + normal_x * label_offset),
        y=float(point[1] + normal_y * label_offset),
        angle_degrees=final_angle,
        t=t,
    )
