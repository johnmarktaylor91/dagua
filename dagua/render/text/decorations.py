"""Decoration path helpers for data-coordinate text rendering."""

from __future__ import annotations

import math

import numpy as np
from matplotlib.path import Path

_KAPPA = 4.0 * (math.sqrt(2.0) - 1.0) / 3.0


def _rectangle_path(left: float, bottom: float, right: float, top: float) -> Path:
    """Build a closed rectangular path.

    Parameters
    ----------
    left : float
        Left edge.
    bottom : float
        Bottom edge.
    right : float
        Right edge.
    top : float
        Top edge.

    Returns
    -------
    Path
        Closed counter-clockwise rectangle path.
    """
    vertices = np.array(
        [
            [left, bottom],
            [right, bottom],
            [right, top],
            [left, top],
            [left, bottom],
        ],
        dtype=float,
    )
    codes = np.array(
        [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY],
        dtype=np.uint8,
    )
    return Path(vertices, codes)


def background_rect_path(
    cx: float,
    cy: float,
    content_width: float,
    content_height: float,
    padding_x: float,
    padding_y: float,
    corner_radius: float = 0.0,
) -> Path:
    """Create a rounded background rectangle path.

    Parameters
    ----------
    cx : float
        Center x-coordinate of the logical text block.
    cy : float
        Center y-coordinate of the logical text block.
    content_width : float
        Text block width in data units.
    content_height : float
        Text block height in data units.
    padding_x : float
        Horizontal padding in data units.
    padding_y : float
        Vertical padding in data units.
    corner_radius : float, default=0.0
        Corner radius in data units.

    Returns
    -------
    Path
        Closed path suitable for filling.
    """
    total_width = max(content_width + padding_x * 2.0, 0.0)
    total_height = max(content_height + padding_y * 2.0, 0.0)
    left = cx - total_width / 2.0
    right = cx + total_width / 2.0
    bottom = cy - total_height / 2.0
    top = cy + total_height / 2.0
    radius = min(max(corner_radius, 0.0), total_width / 2.0, total_height / 2.0)
    if radius <= 0.0:
        return _rectangle_path(left, bottom, right, top)

    control = radius * _KAPPA
    vertices = np.array(
        [
            [left + radius, top],
            [right - radius, top],
            [right - radius + control, top],
            [right, top - radius + control],
            [right, top - radius],
            [right, bottom + radius],
            [right, bottom + radius - control],
            [right - radius + control, bottom],
            [right - radius, bottom],
            [left + radius, bottom],
            [left + radius - control, bottom],
            [left, bottom + radius - control],
            [left, bottom + radius],
            [left, top - radius],
            [left, top - radius + control],
            [left + radius - control, top],
            [left + radius, top],
            [left + radius, top],
        ],
        dtype=float,
    )
    codes = np.array(
        [
            Path.MOVETO,
            Path.LINETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.LINETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.LINETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.LINETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.CLOSEPOLY,
        ],
        dtype=np.uint8,
    )
    return Path(vertices, codes)


def underline_path(
    x_start: float,
    x_end: float,
    y: float,
    thickness: float,
) -> Path:
    """Create a filled underline rectangle.

    Parameters
    ----------
    x_start : float
        Underline start x-coordinate.
    x_end : float
        Underline end x-coordinate.
    y : float
        Center y-coordinate of the underline bar.
    thickness : float
        Underline thickness in data units.

    Returns
    -------
    Path
        Closed rectangular path.
    """
    half_thickness = max(thickness, 0.0) / 2.0
    return _rectangle_path(x_start, y - half_thickness, x_end, y + half_thickness)


def strikethrough_path(
    x_start: float,
    x_end: float,
    y: float,
    thickness: float,
) -> Path:
    """Create a filled strikethrough rectangle.

    Parameters
    ----------
    x_start : float
        Strikethrough start x-coordinate.
    x_end : float
        Strikethrough end x-coordinate.
    y : float
        Center y-coordinate of the strike bar.
    thickness : float
        Strikethrough thickness in data units.

    Returns
    -------
    Path
        Closed rectangular path.
    """
    half_thickness = max(thickness, 0.0) / 2.0
    return _rectangle_path(x_start, y - half_thickness, x_end, y + half_thickness)
