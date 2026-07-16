"""Tests for newly added node border shapes."""

from __future__ import annotations

import numpy as np
import pytest

from dagua.render.borders.shapes import ShapeSpec, build_shape_path, polygon_vertices
from dagua.render.edges.intersection import ray_polygon_intersection
from dagua.utils import compute_node_size

NEW_SHAPES = [
    "arrow",
    "double_circle",
    "cloud",
    "stadium",
    "semicircle",
    "semicircle_up",
    "semicircle_down",
    "semicircle_left",
    "semicircle_right",
    "tab",
    "note",
    "house",
    "invhouse",
    "folder",
    "component",
    "Msquare",
    "Mdiamond",
    "Mcircle",
    "doubleoctagon",
    "tripleoctagon",
    "document",
    "box3d",
    "promoter",
    "cds",
    "terminator",
    "ribosite",
    "proteasesite",
    "rpromoter",
    "rarrow",
    "larrow",
    "assembly",
    "insulator",
    "signature",
    "invtrapezium",
]
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
ROUNDED_POLYGON_VERTEX_COUNTS = {
    "round_triangle": 3,
    "round_diamond": 4,
    "round_pentagon": 5,
    "round_hexagon": 6,
    "round_octagon": 8,
}


def test_custom_polygon_scales_centered_unit_points_to_node_box() -> None:
    """Scale Cytoscape ``-1..1`` polygon points to the requested node bounds.

    Returns
    -------
    None
        The closed path vertices are asserted in data coordinates.
    """

    path = build_shape_path(
        ShapeSpec(
            center_x=10.0,
            center_y=-5.0,
            width=20.0,
            height=10.0,
            shape="polygon",
            polygon_points=[(-1.0, -1.0), (1.0, -1.0), (0.0, 1.0)],
        )
    )

    np.testing.assert_allclose(
        path.vertices,
        np.array([[0.0, -10.0], [20.0, -10.0], [10.0, 0.0], [0.0, -10.0]]),
    )


@pytest.mark.parametrize(("shape", "vertex_count"), ROUNDED_POLYGON_VERTEX_COUNTS.items())
def test_rounded_polygon_paths_curve_every_corner(shape: str, vertex_count: int) -> None:
    """Verify rounded polygons preserve corner count and replace sharp joins.

    Parameters
    ----------
    shape : str
        Rounded-polygon shape name.
    vertex_count : int
        Expected number of base polygon corners.

    Returns
    -------
    None
        This test asserts quadratic path-code structure and nonzero rounding.
    """

    path = build_shape_path(
        ShapeSpec(center_x=0.0, center_y=0.0, width=100.0, height=80.0, shape=shape)
    )

    assert path.codes is not None
    assert np.count_nonzero(path.codes == path.CURVE3) == vertex_count * 2
    assert np.count_nonzero(path.codes == path.LINETO) == vertex_count - 1
    assert len(path.vertices) == vertex_count * 3 + 1

    base_shape = shape.removeprefix("round_")
    corners = polygon_vertices(
        ShapeSpec(center_x=0.0, center_y=0.0, width=100.0, height=80.0, shape=base_shape)
    )
    linear_vertices = path.vertices[path.codes == path.LINETO]
    assert all(
        not np.any(np.all(np.isclose(linear_vertices, corner), axis=1)) for corner in corners
    ), "Base-polygon corners must only act as curve controls, never sharp linear joins"


@pytest.mark.parametrize(("shape", "vertex_count"), ROUNDED_POLYGON_VERTEX_COUNTS.items())
def test_rounded_polygons_share_base_shape_layout_geometry(shape: str, vertex_count: int) -> None:
    """Verify rounded variants retain base sizing and edge-intersection geometry.

    Parameters
    ----------
    shape : str
        Rounded-polygon shape name.
    vertex_count : int
        Base polygon corner count used to select the corresponding name.

    Returns
    -------
    None
        This test asserts layout sizing and routed boundary intersections.
    """

    base_shapes = {3: "triangle", 4: "diamond", 5: "pentagon", 6: "hexagon", 8: "octagon"}
    base_shape = base_shapes[vertex_count]
    rounded_size = compute_node_size("rounded label", shape=shape)
    base_size = compute_node_size("rounded label", shape=base_shape)
    rounded_hit = ray_polygon_intersection(
        center=[0.0, 0.0],
        half_size=[50.0, 40.0],
        shape=shape,
        ray_origin=[0.0, 0.0],
        ray_direction=[1.0, 0.35],
    )
    base_hit = ray_polygon_intersection(
        center=[0.0, 0.0],
        half_size=[50.0, 40.0],
        shape=base_shape,
        ray_origin=[0.0, 0.0],
        ray_direction=[1.0, 0.35],
    )

    assert rounded_size == pytest.approx(base_size)
    assert rounded_hit == pytest.approx(base_hit)


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


def test_semicircle_smoke_paths_build_with_multiple_orientations() -> None:
    """Semicircle variants should all build populated paths.

    Returns
    -------
    None
        The requested paths are asserted in place.
    """

    for orient in [
        "semicircle",
        "semicircle_up",
        "semicircle_down",
        "semicircle_left",
        "semicircle_right",
    ]:
        spec = ShapeSpec(center_x=0.0, center_y=0.0, width=40.0, height=30.0, shape=orient)
        path = build_shape_path(spec)

        assert path is not None
        assert len(path.vertices) > 3


def test_arrow_shape_uses_expected_vertices_and_closes() -> None:
    """Arrow nodes should use a clearly notched chevron outline and close cleanly.

    Returns
    -------
    None
        Assertions run in place.
    """

    spec = ShapeSpec(center_x=0.0, center_y=0.0, width=10.0, height=10.0, shape="arrow")

    path = build_shape_path(spec)

    assert path.codes[0] == path.MOVETO
    assert path.codes[-1] == path.CLOSEPOLY
    assert path.vertices.shape[0] == 7
    np.testing.assert_allclose(path.vertices[0], np.array([-5.0, 5.0]))
    np.testing.assert_allclose(path.vertices[1], np.array([2.5, 5.0]))
    np.testing.assert_allclose(path.vertices[2], np.array([5.0, 0.0]))
    np.testing.assert_allclose(path.vertices[3], np.array([2.5, -5.0]))
    np.testing.assert_allclose(path.vertices[4], np.array([-5.0, -5.0]))
    np.testing.assert_allclose(path.vertices[5], np.array([-1.5, 0.0]))
    np.testing.assert_allclose(path.vertices[0], path.vertices[-1])


@pytest.mark.parametrize(
    ("shape", "expected_bounds"),
    [
        ("house", (-72.0, -29.12, 72.0, 36.0)),
        ("invhouse", (-72.0, -36.0, 72.0, 29.12)),
        ("folder", (-72.0, -36.0, 72.0, 40.0)),
        ("tab", (-72.0, -36.0, 72.0, 40.0)),
        ("component", (-76.0, -36.0, 72.0, 36.0)),
        ("note", (-72.0, -36.0, 72.0, 36.0)),
        ("Msquare", (-72.0, -72.0, 72.0, 72.0)),
        ("Mdiamond", (-72.0, -36.0, 72.0, 36.0)),
        ("Mcircle", (-72.0, -72.0, 72.0, 72.0)),
        ("doubleoctagon", (-76.0, -40.0, 76.0, 40.0)),
        ("tripleoctagon", (-80.0, -44.0, 80.0, 44.0)),
    ],
)
def test_graphviz_shape_silhouette_bounds(
    shape: str,
    expected_bounds: tuple[float, float, float, float],
) -> None:
    """Match Graphviz's fixed-size 144-by-72 point silhouette bounds.

    Parameters
    ----------
    shape : str
        Graphviz-compatible Dagua shape name.
    expected_bounds : tuple[float, float, float, float]
        Expected ``(left, bottom, right, top)`` bounds in points.

    Returns
    -------
    None
        The emitted path bounds are asserted in place.
    """

    path = build_shape_path(
        ShapeSpec(center_x=0.0, center_y=0.0, width=144.0, height=72.0, shape=shape)
    )
    vertices = np.asarray(path.vertices, dtype=np.float64)
    actual_bounds = (
        float(vertices[:, 0].min()),
        float(vertices[:, 1].min()),
        float(vertices[:, 0].max()),
        float(vertices[:, 1].max()),
    )

    np.testing.assert_allclose(actual_bounds, expected_bounds, atol=0.01)


@pytest.mark.parametrize(
    ("shape", "expected_vertices"),
    [
        (
            "promoter",
            [
                [18.0, 18.0],
                [-36.0, 18.0],
                [-36.0, 0.0],
                [-30.0, 0.0],
                [-30.0, 12.0],
                [18.0, 12.0],
                [18.0, 9.0],
                [30.0, 15.0],
                [18.0, 21.0],
                [18.0, 18.0],
            ],
        ),
        (
            "cds",
            [[60.0, 30.0], [-72.0, 30.0], [-72.0, -30.0], [60.0, -30.0], [72.0, 0.0], [60.0, 30.0]],
        ),
        (
            "terminator",
            [
                [3.0, 0.0],
                [3.0, 6.0],
                [9.0, 6.0],
                [9.0, 12.0],
                [-9.0, 12.0],
                [-9.0, 6.0],
                [-3.0, 6.0],
                [-3.0, 0.0],
                [3.0, 0.0],
            ],
        ),
        (
            "ribosite",
            [
                [3.0, 6.0],
                [3.0, 7.5],
                [1.5, 9.0],
                [3.0, 10.5],
                [3.0, 12.0],
                [1.5, 12.0],
                [0.0, 10.5],
                [-1.5, 12.0],
                [-3.0, 12.0],
                [-3.0, 10.5],
                [-1.5, 9.0],
                [-3.0, 7.5],
                [-3.0, 6.0],
                [-1.5, 6.0],
                [0.0, 7.5],
                [1.5, 6.0],
                [3.0, 6.0],
            ],
        ),
        (
            "proteasesite",
            [
                [3.0, 6.0],
                [3.0, 7.5],
                [1.5, 9.0],
                [3.0, 10.5],
                [3.0, 12.0],
                [1.5, 12.0],
                [0.0, 10.5],
                [-1.5, 12.0],
                [-3.0, 12.0],
                [-3.0, 10.5],
                [-1.5, 9.0],
                [-3.0, 7.5],
                [-3.0, 6.0],
                [-1.5, 6.0],
                [0.0, 7.5],
                [1.5, 6.0],
                [3.0, 6.0],
            ],
        ),
        (
            "rpromoter",
            [
                [54.0, 30.0],
                [-72.0, 30.0],
                [-72.0, -36.0],
                [-54.0, -36.0],
                [-54.0, -30.0],
                [54.0, -30.0],
                [54.0, -36.0],
                [72.0, 0.0],
                [54.0, 36.0],
                [54.0, 30.0],
            ],
        ),
        (
            "rarrow",
            [
                [54.0, 30.0],
                [-72.0, 30.0],
                [-72.0, -30.0],
                [54.0, -30.0],
                [54.0, -36.0],
                [72.0, 0.0],
                [54.0, 36.0],
                [54.0, 30.0],
            ],
        ),
        (
            "larrow",
            [
                [72.0, 30.0],
                [-54.0, 30.0],
                [-54.0, 36.0],
                [-72.0, 0.0],
                [-54.0, -36.0],
                [-54.0, -30.0],
                [72.0, -30.0],
                [72.0, 30.0],
            ],
        ),
        (
            "signature",
            [[72.0, 30.0], [-72.0, 30.0], [-72.0, -30.0], [72.0, -30.0], [72.0, 30.0]],
        ),
        (
            "insulator",
            [[6.0, 6.0], [6.0, -6.0], [-6.0, -6.0], [-6.0, 6.0], [6.0, 6.0]],
        ),
        (
            "invtrapezium",
            [[-42.05, -36.0], [42.05, -36.0], [72.0, 36.0], [-72.0, 36.0], [-42.05, -36.0]],
        ),
    ],
)
def test_graphviz_705_special_shape_vertices(
    shape: str,
    expected_vertices: list[list[float]],
) -> None:
    """Match Graphviz 7.0.5 fixed-size special-shape polygon coordinates.

    Parameters
    ----------
    shape : str
        Graphviz-compatible Dagua shape name.
    expected_vertices : list[list[float]]
        Expected centered vertices for a 144-by-72 point node.

    Returns
    -------
    None
        The emitted vertices are asserted in place.
    """

    path = build_shape_path(
        ShapeSpec(center_x=0.0, center_y=0.0, width=144.0, height=72.0, shape=shape)
    )

    np.testing.assert_allclose(path.vertices, np.asarray(expected_vertices), atol=0.01)


def test_graphviz_705_assembly_uses_two_exact_bars() -> None:
    """Match Graphviz 7.0.5's two closed assembly rectangles.

    Returns
    -------
    None
        Both compound subpaths are asserted in place.
    """

    path = build_shape_path(
        ShapeSpec(center_x=0.0, center_y=0.0, width=144.0, height=72.0, shape="assembly")
    )
    expected = np.array(
        [
            [-12.0, 1.5],
            [12.0, 1.5],
            [12.0, 7.5],
            [-12.0, 7.5],
            [-12.0, 1.5],
            [-12.0, -7.5],
            [12.0, -7.5],
            [12.0, -1.5],
            [-12.0, -1.5],
            [-12.0, -7.5],
        ]
    )

    np.testing.assert_allclose(path.vertices, expected)
