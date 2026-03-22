"""Tests for the final cosmetic node-rendering batch."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from matplotlib.collections import PatchCollection

from dagua.graph import DaguaGraph
from dagua.render import mpl as mpl_renderer
from dagua.render import render
from dagua.render.text import DaguaText
from dagua.styles import NodeStyle

matplotlib.use("Agg")


def _two_node_graph(
    primary_style: NodeStyle,
    primary_label: str = "Primary",
) -> tuple[DaguaGraph, torch.Tensor]:
    """Build a small two-node graph for renderer assertions.

    Parameters
    ----------
    primary_style : NodeStyle
        Style override applied to the first node.
    primary_label : str, default="Primary"
        Label text for the first node.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph plus explicit node positions with shape ``[2, 2]``.
    """
    graph = DaguaGraph()
    graph.add_node("A", label=primary_label, style=primary_style)
    graph.add_node("B", label="Other")
    graph.add_edge("A", "B")
    graph.compute_node_sizes()
    positions = torch.tensor([[0.0, 30.0], [0.0, -30.0]], dtype=torch.float32)
    graph.cache_layout(positions)
    return graph, positions


def _capture_text_specs(
    monkeypatch: pytest.MonkeyPatch,
    graph: DaguaGraph,
    positions: torch.Tensor,
) -> List[DaguaText]:
    """Capture text specs emitted by the matplotlib renderer.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture.
    graph : DaguaGraph
        Graph to render.
    positions : torch.Tensor
        Explicit node positions with shape ``[N, 2]``.

    Returns
    -------
    list[DaguaText]
        Text specifications handed to ``render_text``.
    """
    captured: List[DaguaText] = []

    def _record_specs(
        ax: Any,
        specs: List[DaguaText],
        display_scale: float,
        svg_hover_map: Any = None,
    ) -> list[Any]:
        """Record emitted text specs without creating artists.

        Parameters
        ----------
        ax : Any
            Matplotlib axes.
        specs : list[DaguaText]
            Specs passed into the renderer.
        display_scale : float
            Point-to-data conversion factor.
        svg_hover_map : Any, default=None
            Optional hover-text accumulator.

        Returns
        -------
        list[Any]
            Empty placeholder list for renderer compatibility.
        """
        del ax, display_scale, svg_hover_map
        captured.extend(specs)
        return []

    monkeypatch.setattr(mpl_renderer, "render_text", _record_specs)
    fig, _ax = mpl_renderer.render(graph, positions=positions, show=False)
    plt.close(fig)
    return captured


def _find_text_spec(specs: List[DaguaText], gid: str) -> DaguaText:
    """Return one captured text spec by gid.

    Parameters
    ----------
    specs : list[DaguaText]
        Captured specs.
    gid : str
        Artist gid to find.

    Returns
    -------
    DaguaText
        Matching spec.

    Raises
    ------
    AssertionError
        Raised when the gid was not emitted.
    """
    for spec in specs:
        if spec.gid == gid:
            return spec
    raise AssertionError(f"missing text spec {gid!r}")


def _border_collection_extent_width(style: NodeStyle) -> float:
    """Render one node and return the border-collection width.

    Parameters
    ----------
    style : NodeStyle
        Node style to render.

    Returns
    -------
    float
        Width of the border path's bounding box in data coordinates.
    """
    graph = DaguaGraph()
    graph.add_node("A", label="Border", style=style)
    graph.compute_node_sizes()
    pos = np.array([[0.0, 0.0]], dtype=np.float64)
    sizes = graph.node_sizes.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(4.0, 3.0), dpi=100)
    ax.set_xlim(-80.0, 80.0)
    ax.set_ylim(-60.0, 60.0)
    ax.set_aspect("equal")
    fig.canvas.draw()
    mpl_renderer._draw_nodes(ax=ax, graph=graph, pos=pos, sizes=sizes)
    border_collections = [
        collection
        for collection in ax.collections
        if isinstance(collection, PatchCollection) and float(collection.get_zorder()) == 2.05
    ]
    plt.close(fig)

    assert len(border_collections) == 1
    border_paths = border_collections[0].get_paths()
    assert len(border_paths) == 1
    return float(border_paths[0].get_extents().width)


class TestTextRotation:
    """Cover the text-rotation node-style field and renderer wiring."""

    def test_field_exists(self) -> None:
        """NodeStyle should store explicit label rotation."""
        style = NodeStyle(text_rotation=45.0)
        assert style.text_rotation == 45.0

    def test_default_zero(self) -> None:
        """Node labels should default to zero rotation."""
        assert NodeStyle().text_rotation == 0.0

    def test_render_forwards_rotation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Renderer should forward node text rotation into DaguaText specs."""
        graph, positions = _two_node_graph(NodeStyle(text_rotation=30.0), primary_label="Rotated")
        specs = _capture_text_specs(monkeypatch, graph, positions)

        label_spec = _find_text_spec(specs, "dagua-node-label-0")
        assert label_spec.rotation == pytest.approx(30.0)


class TestExternalLabels:
    """Cover external node labels and their render-time placement."""

    def test_field_exists(self) -> None:
        """NodeStyle should expose external-label fields."""
        style = NodeStyle(external_label="ID: 42", external_label_position="bottom")
        assert style.external_label == "ID: 42"
        assert style.external_label_position == "bottom"

    def test_default_empty(self) -> None:
        """External labels should default to disabled."""
        assert NodeStyle().external_label == ""

    @pytest.mark.parametrize("position", ["top", "bottom", "left", "right"])
    def test_positions(self, position: str) -> None:
        """NodeStyle should accept each supported external-label position."""
        style = NodeStyle(external_label="x", external_label_position=position)
        assert style.external_label_position == position

    def test_render_emits_external_label_spec(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Renderer should emit a separate text spec for external labels."""
        graph, positions = _two_node_graph(
            NodeStyle(external_label="ID: 1", external_label_position="bottom"),
            primary_label="Main",
        )
        specs = _capture_text_specs(monkeypatch, graph, positions)

        external_spec = _find_text_spec(specs, "dagua-node-external-label-0")
        assert external_spec.text == "ID: 1"
        assert external_spec.va == "top"
        assert external_spec.clip_on is False

    def test_render_expands_bounds_for_wide_external_label(self) -> None:
        """Wide external labels should expand figure bounds instead of clipping."""
        graph, positions = _two_node_graph(
            NodeStyle(
                external_label="external label that is intentionally very wide",
                external_label_position="right",
            ),
            primary_label="Main",
        )
        fig, ax = render(graph, positions=positions, show=False)
        x_limits = ax.get_xlim()
        plt.close(fig)

        assert x_limits[1] > 110.0


class TestBorderPosition:
    """Cover inside, center, and outside node border placement."""

    def test_field_exists(self) -> None:
        """NodeStyle should expose border-position selection."""
        style = NodeStyle(border_position="inside")
        assert style.border_position == "inside"

    def test_default_center(self) -> None:
        """Border placement should default to center."""
        assert NodeStyle().border_position == "center"

    def test_outside(self) -> None:
        """NodeStyle should store the outside border mode."""
        style = NodeStyle(border_position="outside")
        assert style.border_position == "outside"

    def test_outside_border_expands_geometry(self) -> None:
        """Outside borders should occupy more width than inside borders."""
        inside_width = _border_collection_extent_width(
            NodeStyle(border_position="inside", stroke_width=3.0)
        )
        outside_width = _border_collection_extent_width(
            NodeStyle(border_position="outside", stroke_width=3.0)
        )

        assert outside_width > inside_width

    def test_render_inside_border(self) -> None:
        """Inside-border nodes should render without error."""
        graph, positions = _two_node_graph(
            NodeStyle(border_position="inside", stroke_width=3.0),
            primary_label="Inside",
        )
        fig, _ax = render(graph, positions=positions, show=False)
        assert fig is not None
        plt.close(fig)


class TestImageNodes:
    """Cover image-backed node rendering and graceful fallback."""

    def test_field_exists(self) -> None:
        """NodeStyle should expose image fields."""
        style = NodeStyle(image="/path/to/image.png", image_fit="cover")
        assert style.image == "/path/to/image.png"
        assert style.image_fit == "cover"

    def test_default_empty(self) -> None:
        """Image nodes should default to disabled."""
        assert NodeStyle().image == ""

    def test_image_opacity(self) -> None:
        """NodeStyle should store image opacity."""
        style = NodeStyle(image="x.png", image_opacity=0.5)
        assert style.image_opacity == 0.5

    def test_render_image_node(self, tmp_path: Path) -> None:
        """Valid node images should render as clipped AxesImage artists."""
        from PIL import Image

        image_path = tmp_path / "node.png"
        Image.new("RGBA", (4, 2), (255, 0, 0, 255)).save(image_path)

        graph, positions = _two_node_graph(
            NodeStyle(image=str(image_path), image_fit="cover"),
            primary_label="Image",
        )
        fig, ax = render(graph, positions=positions, show=False)

        assert len(ax.images) == 1
        plt.close(fig)

    def test_render_without_image_file(self) -> None:
        """Missing image files should fall back to the normal fill without error."""
        graph, positions = _two_node_graph(
            NodeStyle(image="/nonexistent/path.png"),
            primary_label="Image",
        )
        fig, ax = render(graph, positions=positions, show=False)

        assert fig is not None
        assert len(ax.images) == 0
        plt.close(fig)
