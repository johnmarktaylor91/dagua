"""Tests for the new arrowhead primitives."""

from __future__ import annotations

import numpy as np
import pytest

from dagua.render.edges.arrowheads import ARROWHEAD_REGISTRY, ArrowheadResult
from dagua.styles import EdgeStyle, get_theme

NEW_ARROWS = [
    "crows_foot_one",
    "crows_foot_many",
    "crows_foot_one_mandatory",
    "crows_foot_many_mandatory",
    "crows_foot_many_optional",
    "triangle_tee",
]

EXISTING_ARROWS = [
    "normal",
    "inv",
    "dot",
    "box",
    "vee",
    "tee",
    "crow",
    "diamond",
    "curve",
    "icurve",
    "simple",
    "fancy",
    "wedge",
    "bracket",
    "none",
]


def _build_local_arrow(
    name: str,
    length: float,
    width: float,
    body_width: float,
) -> ArrowheadResult:
    """Build a registered primitive in local coordinates.

    Parameters
    ----------
    name : str
        Registered arrowhead name.
    length : float
        Primitive length in local coordinates.
    width : float
        Primitive width in local coordinates.
    body_width : float
        Ribbon body width in local coordinates.

    Returns
    -------
    ArrowheadResult
        Local arrowhead geometry returned by the primitive builder.
    """
    return ARROWHEAD_REGISTRY[name].builder(length, width, body_width)


@pytest.mark.parametrize("name", NEW_ARROWS)
def test_new_arrowheads_are_registered(name: str) -> None:
    """Each requested arrowhead should be present in the registry.

    Parameters
    ----------
    name : str
        Registered arrowhead name under test.

    Returns
    -------
    None
        This test only performs assertions.
    """
    assert name in ARROWHEAD_REGISTRY


@pytest.mark.parametrize("name", NEW_ARROWS)
def test_new_arrowheads_build_geometry(name: str) -> None:
    """Each new builder should return usable geometry.

    Parameters
    ----------
    name : str
        Registered arrowhead name under test.

    Returns
    -------
    None
        This test only performs assertions.
    """
    result = _build_local_arrow(name, 10.0, 6.0, 2.0)

    assert result.trim_contour is not None
    assert result.filled_paths or result.stroked_paths


@pytest.mark.parametrize("name", NEW_ARROWS)
def test_new_arrowheads_scale_across_sizes(name: str) -> None:
    """New builders should work across the expected size range.

    Parameters
    ----------
    name : str
        Registered arrowhead name under test.

    Returns
    -------
    None
        This test only performs assertions.
    """
    for length in [5.0, 10.0, 20.0]:
        result = _build_local_arrow(name, length, length * 0.6, length * 0.2)
        assert result.trim_contour.vertices.shape[0] >= 2
        assert result.filled_paths or result.stroked_paths


@pytest.mark.parametrize(
    ("name", "expected_stroke_only"),
    [
        ("crows_foot_one", True),
        ("crows_foot_many", True),
        ("crows_foot_one_mandatory", True),
        ("crows_foot_many_mandatory", True),
        ("crows_foot_many_optional", True),
        ("triangle_tee", False),
    ],
)
def test_new_arrowhead_registry_flags_match_paint_mode(
    name: str,
    expected_stroke_only: bool,
) -> None:
    """Registry metadata should match each primitive's paint model.

    Parameters
    ----------
    name : str
        Registered arrowhead name under test.
    expected_stroke_only : bool
        Expected registry ``stroke_only`` flag.

    Returns
    -------
    None
        This test only performs assertions.
    """
    assert ARROWHEAD_REGISTRY[name].stroke_only is expected_stroke_only


def test_triangle_tee_combines_filled_triangle_and_stroked_bar() -> None:
    """The Cytoscape-inspired head should contain both fill and stroke geometry.

    Returns
    -------
    None
        This test only performs assertions.
    """
    result = _build_local_arrow("triangle_tee", 10.0, 6.0, 2.0)

    assert len(result.filled_paths) == 1
    assert len(result.stroked_paths) == 1


@pytest.mark.parametrize(
    "name",
    [
        "crows_foot_one",
        "crows_foot_many",
        "crows_foot_one_mandatory",
        "crows_foot_many_mandatory",
        "crows_foot_many_optional",
    ],
)
def test_crows_foot_heads_are_stroked_only(name: str) -> None:
    """ER heads should remain line-based primitives.

    Parameters
    ----------
    name : str
        Registered crow's foot variant under test.

    Returns
    -------
    None
        This test only performs assertions.
    """
    result = _build_local_arrow(name, 10.0, 6.0, 2.0)

    assert result.filled_paths == []
    assert len(result.stroked_paths) >= 1


@pytest.mark.parametrize("name", EXISTING_ARROWS)
def test_existing_arrowheads_still_build(name: str) -> None:
    """Adding the new primitives should not break existing built-ins.

    Parameters
    ----------
    name : str
        Existing registered arrowhead name under test.

    Returns
    -------
    None
        This test only performs assertions.
    """
    assert name in ARROWHEAD_REGISTRY
    result = _build_local_arrow(name, 10.0, 6.0, 2.0)
    assert result.trim_contour is not None
    assert result is not None


def test_normal_arrowhead_uses_simple_triangle_for_thin_edges() -> None:
    """Thin normal arrows should fall back to a plain three-point triangle.

    Returns
    -------
    None
        This test only performs assertions.
    """
    result = _build_local_arrow("normal", 10.0, 6.0, 0.8)

    assert len(result.filled_paths) == 1
    assert result.stroked_paths == []
    assert np.allclose(
        result.filled_paths[0].vertices,
        np.array(
            [
                [0.0, 0.0],
                [10.0, 3.0],
                [10.0, -3.0],
                [0.0, 0.0],
            ]
        ),
    )
    assert np.allclose(
        result.trim_contour.vertices,
        np.array(
            [
                [10.0, 0.4],
                [10.0, -0.4],
            ]
        ),
    )


def test_graphviz_theme_keeps_node_relative_arrowheads() -> None:
    """The improved Graphviz theme should not shrink node-relative arrowheads.

    Returns
    -------
    None
        This test only performs assertions.
    """
    theme = get_theme("graphviz")
    edge_style = theme.get_edge_style("default")

    assert EdgeStyle().arrow_node_fraction == pytest.approx(0.35)
    assert edge_style.arrow_node_fraction >= 0.3
    assert edge_style.arrow_node_fraction == pytest.approx(0.35)
