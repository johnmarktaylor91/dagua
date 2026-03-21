"""Compound-path helpers for solid node and cluster borders."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from matplotlib.path import Path
from numpy.typing import NDArray

from dagua.render.borders.inset import polygon_signed_area
from dagua.render.borders.shapes import path_to_closed_vertices

FloatArray = NDArray[np.float64]


def _subpath_segments(
    path: Path,
) -> List[Tuple[FloatArray, List[Tuple[int, FloatArray, FloatArray]]]]:
    """Split a path into subpaths and decode each segment.

    Parameters
    ----------
    path : matplotlib.path.Path
        Closed source path.

    Returns
    -------
    list[tuple[numpy.ndarray, list[tuple[int, numpy.ndarray, numpy.ndarray]]]]
        One entry per subpath. Each entry stores the start point and a list of
        ``(code, start, vertices)`` segment payloads.
    """

    if path.codes is None:
        raise ValueError("Annular path construction requires explicit path codes.")

    subpaths: List[Tuple[FloatArray, List[Tuple[int, FloatArray, FloatArray]]]] = []
    codes = np.asarray(path.codes, dtype=np.uint8)
    vertices = np.asarray(path.vertices, dtype=np.float64)
    index = 0
    current_start: FloatArray | None = None
    current_segments: List[Tuple[int, FloatArray, FloatArray]] = []
    current_point: FloatArray | None = None

    while index < len(codes):
        code = int(codes[index])
        if code == Path.MOVETO:
            if current_start is not None:
                subpaths.append((current_start, current_segments))
            current_start = vertices[index].copy()
            current_point = current_start.copy()
            current_segments = []
            index += 1
            continue
        if current_start is None or current_point is None:
            raise ValueError("Encountered a path segment before MOVETO.")
        if code == Path.LINETO:
            current_segments.append(
                (Path.LINETO, current_point.copy(), vertices[index : index + 1].copy())
            )
            current_point = vertices[index].copy()
            index += 1
            continue
        if code == Path.CURVE3:
            payload = vertices[index : index + 2].copy()
            current_segments.append((Path.CURVE3, current_point.copy(), payload))
            current_point = payload[-1].copy()
            index += 2
            continue
        if code == Path.CURVE4:
            payload = vertices[index : index + 3].copy()
            current_segments.append((Path.CURVE4, current_point.copy(), payload))
            current_point = payload[-1].copy()
            index += 3
            continue
        if code == Path.CLOSEPOLY:
            if current_point is not None and not np.allclose(current_point, current_start):
                current_segments.append(
                    (Path.LINETO, current_point.copy(), current_start[None, :].copy())
                )
            current_segments.append(
                (Path.CLOSEPOLY, current_start.copy(), current_start[None, :].copy())
            )
            current_point = current_start.copy()
            index += 1
            continue
        raise ValueError(f"Unsupported path code: {code}.")

    if current_start is not None:
        subpaths.append((current_start, current_segments))
    return subpaths


def reverse_closed_path(path: Path) -> Path:
    """Reverse the winding direction of a closed path without flattening curves.

    Parameters
    ----------
    path : matplotlib.path.Path
        Closed path to reverse.

    Returns
    -------
    matplotlib.path.Path
        Reversed path.
    """

    reversed_vertices: List[FloatArray] = []
    reversed_codes: List[int] = []

    for start_point, segments in _subpath_segments(path):
        if not segments:
            continue
        reversed_vertices.append(start_point.copy())
        reversed_codes.append(Path.MOVETO)
        for code, segment_start, payload in reversed(segments):
            if code == Path.CLOSEPOLY:
                continue
            if code == Path.LINETO:
                reversed_vertices.append(segment_start.copy())
                reversed_codes.append(Path.LINETO)
                continue
            if code == Path.CURVE3:
                reversed_vertices.extend([payload[0].copy(), segment_start.copy()])
                reversed_codes.extend([Path.CURVE3, Path.CURVE3])
                continue
            if code == Path.CURVE4:
                reversed_vertices.extend(
                    [payload[1].copy(), payload[0].copy(), segment_start.copy()]
                )
                reversed_codes.extend([Path.CURVE4, Path.CURVE4, Path.CURVE4])
                continue
        reversed_vertices.append(start_point.copy())
        reversed_codes.append(Path.CLOSEPOLY)

    return Path(np.vstack(reversed_vertices), reversed_codes)


def annular_path(outer_path: Path, inner_path: Path) -> Path:
    """Build a smooth annular border path from outer and inner closed outlines.

    Parameters
    ----------
    outer_path : matplotlib.path.Path
        Outer border outline.
    inner_path : matplotlib.path.Path
        Inner fill outline.

    Returns
    -------
    matplotlib.path.Path
        Compound ring path with the inner winding reversed when needed.
    """

    outer_area = polygon_signed_area(path_to_closed_vertices(outer_path)[:-1])
    inner_area = polygon_signed_area(path_to_closed_vertices(inner_path)[:-1])
    corrected_inner = (
        reverse_closed_path(inner_path) if outer_area * inner_area > 0.0 else inner_path
    )
    vertices = np.vstack([outer_path.vertices, corrected_inner.vertices])
    codes = np.concatenate([outer_path.codes, corrected_inner.codes]).tolist()
    return Path(vertices, codes)
