"""Tests for newly added node border shapes."""

from __future__ import annotations

import numpy as np
import pytest

from dagua.render.borders.shapes import ShapeSpec, build_shape_path

NEW_SHAPES = ["double_circle", "cloud", "stadium", "tab", "note", "document", "box3d"]
EXISTING_SHAPES = [
    "rect",
    "roundrect",
    "ellipse",
    "circle",
    "diamond",
    "triangle",
    "hexagon",
    "pentagon",
    "octagon",
    "star",
    "cylinder",
    "parallelogram",
    "trapezoid",
]


@pytest.mark.parametrize("shape", NEW_SHAPES)
def test_new_shapes_return_valid_paths(shape: str) -> None:
    """Verify each new shape produces a populated matplotlib path.

    Parameters
    ----------
    shape : str
        Node shape name under test.

    Returns
    -------
    None
        This test asserts path validity.
    """

    spec = ShapeSpec(
        center_x=100.0,
        center_y=100.0,
        width=80.0,
        height=60.0,
        shape=shape,
    )
    path = build_shape_path(spec)

    assert path is not None
    assert len(path.vertices) >= 3


@pytest.mark.parametrize("shape", NEW_SHAPES)
def test_new_shape_vertices_stay_near_their_center(shape: str) -> None:
    """Verify new shape vertices remain within generous bounds.

    Parameters
    ----------
    shape : str
        Node shape name under test.

    Returns
    -------
    None
        This test asserts vertex positions stay close to the requested center.
    """

    spec = ShapeSpec(
        center_x=50.0,
        center_y=50.0,
        width=40.0,
        height=30.0,
        shape=shape,
    )
    path = build_shape_path(spec)
    vertices = np.asarray(path.vertices, dtype=np.float64)

    assert np.all(np.abs(vertices[:, 0] - 50.0) < 80.0)
    assert np.all(np.abs(vertices[:, 1] - 50.0) < 60.0)


@pytest.mark.parametrize("shape", NEW_SHAPES)
def test_new_shapes_work_across_multiple_sizes(shape: str) -> None:
    """Verify each new shape builds at several aspect ratios and scales.

    Parameters
    ----------
    shape : str
        Node shape name under test.

    Returns
    -------
    None
        This test asserts path generation across multiple sizes.
    """

    for width, height in [(10.0, 10.0), (100.0, 50.0), (50.0, 100.0), (200.0, 200.0)]:
        spec = ShapeSpec(center_x=0.0, center_y=0.0, width=width, height=height, shape=shape)
        path = build_shape_path(spec)

        assert path is not None
        assert len(path.vertices) >= 3


@pytest.mark.parametrize("shape", EXISTING_SHAPES)
def test_existing_shapes_still_build(shape: str) -> None:
    """Verify the pre-existing shape set still dispatches successfully.

    Parameters
    ----------
    shape : str
        Existing node shape name under test.

    Returns
    -------
    None
        This test asserts existing dispatch behavior is unchanged.
    """

    spec = ShapeSpec(center_x=0.0, center_y=0.0, width=50.0, height=40.0, shape=shape)
    path = build_shape_path(spec)

    assert path is not None
