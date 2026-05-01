"""Tests for Matplotlib render backend selection."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.image as mpimg
import pytest
import torch

import dagua
from dagua import DaguaGraph
from dagua.render._backend import (
    _cairo_available,
    _resolve_backend,
    set_default_backend,
)
from dagua.styles import GRAPHVIZ_STRICT_THEME, NodeStyle


def _two_node_graph() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build a minimal graph with fixed positions.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Two-node graph and ``[2, 2]`` node position tensor.
    """
    graph = DaguaGraph()
    graph.add_node("left", label="Left")
    graph.add_node("right", label="Right")
    graph.add_edge("left", "right", label="edge")
    positions = torch.tensor([[-60.0, 0.0], [60.0, 0.0]], dtype=torch.float32)
    return graph, positions


def _assert_png_has_left_and_right_content(output: Path) -> None:
    """Assert a rendered PNG contains visible content on both sides.

    Parameters
    ----------
    output : Path
        Path to a PNG render.

    Returns
    -------
    None
        Raises an assertion error if the render looks blank or one-sided.
    """
    image = mpimg.imread(output)
    rgb = image[..., :3]
    content = (rgb < 0.92).any(axis=2)
    width_midpoint = content.shape[1] // 2
    assert content.any()
    assert content[:, :width_midpoint].any()
    assert content[:, width_midpoint:].any()


def test_resolve_backend_none_auto_detects() -> None:
    """None -> cairo if mplcairo installed, else Agg."""
    cls, name = _resolve_backend(None)
    if _cairo_available():
        assert "cairo" in cls.__module__.lower() or "mplcairo" in cls.__module__.lower()
        assert name == "cairo"
    else:
        assert "agg" in cls.__module__.lower()
        assert name == "agg"


def test_resolve_backend_agg_explicit() -> None:
    """Explicit 'agg' always returns Agg, regardless of mplcairo."""
    cls, name = _resolve_backend("agg")
    assert "agg" in cls.__module__.lower()
    assert name == "agg"


def test_resolve_backend_cairo_explicit_raises_when_missing() -> None:
    """Explicit 'cairo' without mplcairo raises ImportError with install message."""
    if _cairo_available():
        pytest.skip("mplcairo is installed; can't test missing path")
    with pytest.raises(ImportError, match="dagua\\[cairo\\]"):
        _resolve_backend("cairo")


def test_resolve_backend_cairo_explicit_works_when_installed() -> None:
    """Explicit 'cairo' returns mplcairo canvas when installed."""
    if not _cairo_available():
        pytest.skip("mplcairo is not importable")
    cls, name = _resolve_backend("cairo")
    assert "cairo" in cls.__module__.lower() or "mplcairo" in cls.__module__.lower()
    assert name == "cairo"


def test_resolve_backend_unknown_raises_value_error() -> None:
    """Unknown backend names raise ValueError."""
    with pytest.raises(ValueError):
        _resolve_backend("unknown_backend_name")


def test_set_default_backend_overrides_auto_detect() -> None:
    """Global default backend override controls None resolution."""
    set_default_backend("agg")
    try:
        _cls, name = _resolve_backend(None)
        assert name == "agg"
    finally:
        set_default_backend(None)


def test_auto_size_to_label_expands_fixed_overflow_node() -> None:
    """auto_size_to_label should turn min dimensions into floors."""
    fixed_graph = DaguaGraph()
    fixed_graph.add_node(
        "n",
        label="A very long node label",
        style=NodeStyle(
            shape="rect",
            font_size=14.0,
            padding=(4.0, 2.0),
            min_width=30.0,
            min_height=18.0,
            overflow_policy="shrink_text",
            auto_expand_on_floor_overflow=False,
        ),
    )
    fixed_graph.compute_node_sizes()

    auto_graph = DaguaGraph()
    auto_graph.add_node(
        "n",
        label="A very long node label",
        style=NodeStyle(
            shape="rect",
            font_size=14.0,
            padding=(4.0, 2.0),
            min_width=30.0,
            min_height=18.0,
            overflow_policy="shrink_text",
            auto_size_to_label=True,
        ),
    )
    auto_graph.compute_node_sizes()

    assert fixed_graph.node_sizes is not None
    assert auto_graph.node_sizes is not None
    assert float(fixed_graph.node_sizes[0, 0]) == 30.0
    assert float(fixed_graph.node_sizes[0, 1]) == 18.0
    assert float(auto_graph.node_sizes[0, 0]) > 30.0
    assert float(auto_graph.node_sizes[0, 1]) >= 18.0


def test_graphviz_strict_theme_enables_compact_auto_sizing() -> None:
    """Graphviz strict nodes should use dot-style auto-sized floors."""
    graph = DaguaGraph(_theme=GRAPHVIZ_STRICT_THEME.copy())
    graph.add_node("source", label="Source")
    graph.compute_node_sizes()

    assert graph.node_sizes is not None
    style = graph.get_style_for_node(0)
    width = float(graph.node_sizes[0, 0])
    height = float(graph.node_sizes[0, 1])

    assert style.auto_size_to_label is True
    assert style.min_width == 54.0
    assert style.min_height == 36.0
    assert 54.0 <= width < 100.0
    assert 36.0 <= height < 70.0


@pytest.mark.parametrize(
    "backend",
    [
        "agg",
        pytest.param(
            "cairo",
            marks=pytest.mark.skipif(not _cairo_available(), reason="mplcairo not installed"),
        ),
    ],
)
def test_render_basic_under_both_backends(backend: str, tmp_path: Path) -> None:
    """Both backends should produce valid output for a minimal graph."""
    graph, positions = _two_node_graph()
    output = tmp_path / f"test_{backend}.png"

    dagua.render(graph, positions, backend=backend, output=output)

    assert output.exists()
    assert output.stat().st_size > 1000
    _assert_png_has_left_and_right_content(output)
