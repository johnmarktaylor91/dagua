"""Regression tests for opt-in display-point render overrides."""

import io
from dataclasses import fields
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from dagua.graph import DaguaGraph
from dagua.render.mpl import render
from dagua.styles import ClusterStyle, EdgeStyle, NodeStyle


def _saved_rgb(fig: Any, dpi: int) -> np.ndarray:
    """Save a figure at a DPI and return the rasterized RGB pixels.

    Parameters
    ----------
    fig : Any
        Matplotlib figure to save.
    dpi : int
        Rasterization DPI for the saved image.

    Returns
    -------
    numpy.ndarray
        RGB image with shape ``[H, W, 3]``.
    """
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi)
    buffer.seek(0)
    with Image.open(buffer) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _build_pair_graph(
    node_style: NodeStyle,
    edge_style: EdgeStyle,
    cluster_style: ClusterStyle,
) -> Tuple[DaguaGraph, torch.Tensor]:
    """Build the canonical pair fixture used for override checks.

    Parameters
    ----------
    node_style : NodeStyle
        Style applied to both nodes.
    edge_style : EdgeStyle
        Style applied to the connecting edge.
    cluster_style : ClusterStyle
        Style applied to a cluster around both nodes.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph and fixed positions with shape ``[2, 2]``.
    """
    graph = DaguaGraph()
    graph.add_node("source", label="H", style=node_style)
    graph.add_node("target", label="I", style=node_style)
    graph.add_edge("source", "target", label="edge", style=edge_style)
    graph.add_cluster("pair", ["source", "target"], label="Pair", style=cluster_style)
    positions = torch.tensor([[0.0, 0.0], [180.0, 0.0]], dtype=torch.float32)
    return graph, positions


def _render_pair_image(
    node_style: NodeStyle,
    edge_style: EdgeStyle,
    cluster_style: ClusterStyle,
    dpi: int = 100,
) -> np.ndarray:
    """Render the pair fixture and return its RGB pixels.

    Parameters
    ----------
    node_style : NodeStyle
        Node style for the render.
    edge_style : EdgeStyle
        Edge style for the render.
    cluster_style : ClusterStyle
        Cluster style for the render.
    dpi : int, default=100
        Rasterization DPI.

    Returns
    -------
    numpy.ndarray
        RGB image with shape ``[H, W, 3]``.
    """
    graph, positions = _build_pair_graph(node_style, edge_style, cluster_style)
    fig, _ = render(graph, positions, dpi=dpi, figsize=(4.0, 2.0), svg_hover_text=False)
    image = _saved_rgb(fig, dpi)
    plt.close(fig)
    return image


def _build_single_node_graph(style: NodeStyle) -> Tuple[DaguaGraph, torch.Tensor]:
    """Build a one-node graph for measuring node overrides.

    Parameters
    ----------
    style : NodeStyle
        Style applied to the node.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph and fixed position with shape ``[1, 2]``.
    """
    graph = DaguaGraph()
    graph.add_node("node", label="H", style=style)
    positions = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    return graph, positions


def _data_point_to_pixel(ax: Any, x: float, y: float) -> Tuple[int, int]:
    """Convert a data-coordinate point to image-array pixel coordinates.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    x : float
        Data x-coordinate.
    y : float
        Data y-coordinate.

    Returns
    -------
    tuple[int, int]
        Pixel ``(x, y)`` in image-array coordinates.
    """
    px, py = ax.transData.transform((x, y))
    canvas_height = int(ax.figure.canvas.get_width_height()[1])
    return int(round(px)), int(round(canvas_height - py))


def _dark_run_width(mask: np.ndarray, center: int) -> int:
    """Measure the contiguous dark-pixel run around one index.

    Parameters
    ----------
    mask : numpy.ndarray
        One-dimensional boolean mask.
    center : int
        Pixel index expected near the run.

    Returns
    -------
    int
        Contiguous run width in pixels.
    """
    dark_indices = np.flatnonzero(mask)
    if dark_indices.size == 0:
        return 0
    if center < 0 or center >= mask.shape[0] or not bool(mask[center]):
        center = int(dark_indices[np.argmin(np.abs(dark_indices - center))])
    left = center
    while left > 0 and bool(mask[left - 1]):
        left -= 1
    right = center
    while right + 1 < mask.shape[0] and bool(mask[right + 1]):
        right += 1
    return int(right - left + 1)


def _node_border_pixels(dpi: int) -> int:
    """Render a 2-point node stroke override and measure its pixel width.

    Parameters
    ----------
    dpi : int
        Rasterization DPI.

    Returns
    -------
    int
        Measured dark stroke width in pixels.
    """
    style = NodeStyle(
        fill="#FFFFFF",
        stroke="#000000",
        stroke_width=0.0,
        stroke_width_override_points=2.0,
        font_color="#FFFFFF",
        min_width=120.0,
        min_height=70.0,
    )
    graph, positions = _build_single_node_graph(style)
    graph.compute_node_sizes()
    sizes = graph.node_sizes.detach().cpu().numpy()
    fig, ax = render(graph, positions, dpi=dpi, figsize=(3.0, 2.0), svg_hover_text=False)
    image = _saved_rgb(fig, dpi)
    scale = float(dpi) / float(fig.dpi)
    left_x_raw, center_y_raw = _data_point_to_pixel(ax, -float(sizes[0, 0]) / 2.0, 0.0)
    left_x = int(round(left_x_raw * scale))
    center_y = int(round(center_y_raw * scale))
    dark = np.all(image < 80, axis=2)
    width = _dark_run_width(dark[center_y, :], left_x)
    plt.close(fig)
    return width


def _label_height_pixels(dpi: int) -> float:
    """Render a 14-point node label override and measure glyph height.

    Parameters
    ----------
    dpi : int
        Rasterization DPI.

    Returns
    -------
    float
        Label path height in display pixels.
    """
    style = NodeStyle(
        fill="#FFFFFF",
        stroke="#FFFFFF",
        stroke_width=0.0,
        font_color="#000000",
        font_size=2.0,
        font_size_override_points=14.0,
        min_width=120.0,
        min_height=70.0,
    )
    graph, positions = _build_single_node_graph(style)
    fig, _ = render(graph, positions, dpi=dpi, figsize=(3.0, 2.0), svg_hover_text=False)
    image = _saved_rgb(fig, dpi)
    dark = np.all(image < 80, axis=2)
    rows = np.flatnonzero(np.any(dark, axis=1))
    assert rows.size > 0
    height = float(rows[-1] - rows[0] + 1)
    plt.close(fig)
    return height


def _override_docs() -> Dict[str, str]:
    """Return override field documentation from dataclass metadata.

    Returns
    -------
    dict[str, str]
        Mapping from qualified field name to documentation text.
    """
    style_fields = {
        "NodeStyle": (NodeStyle, ("stroke_width_override_points", "font_size_override_points")),
        "EdgeStyle": (EdgeStyle, ("width_override_points", "font_size_override_points")),
        "ClusterStyle": (
            ClusterStyle,
            ("stroke_width_override_points", "font_size_override_points"),
        ),
    }
    docs: Dict[str, str] = {}
    for class_name, (style_type, field_names) in style_fields.items():
        metadata_by_name = {field.name: field.metadata for field in fields(style_type)}
        for field_name in field_names:
            docs[f"{class_name}.{field_name}"] = str(metadata_by_name[field_name].get("doc", ""))
    return docs


def test_override_none_default_unchanged() -> None:
    """Explicit ``None`` override values should match implicit defaults exactly."""
    base = _render_pair_image(
        NodeStyle(),
        EdgeStyle(),
        ClusterStyle(fill="#FFFFFF", opacity=0.1),
    )
    explicit_none = _render_pair_image(
        NodeStyle(stroke_width_override_points=None, font_size_override_points=None),
        EdgeStyle(width_override_points=None, font_size_override_points=None),
        ClusterStyle(
            fill="#FFFFFF",
            opacity=0.1,
            stroke_width_override_points=None,
            font_size_override_points=None,
        ),
    )
    np.testing.assert_array_equal(base, explicit_none)


def test_stroke_width_override_bypasses_data_coord() -> None:
    """Display-point stroke overrides should scale with raster DPI."""
    width_100 = _node_border_pixels(100)
    width_200 = _node_border_pixels(200)

    assert abs(width_100 - 2) <= 1
    assert abs(width_200 - 4) <= 2
    assert width_200 > width_100


def test_font_size_override_bypasses_data_coord() -> None:
    """Display-point font overrides should scale with raster DPI."""
    height_100 = _label_height_pixels(100)
    height_200 = _label_height_pixels(200)

    assert 10.0 <= height_100 <= 20.0
    assert 20.0 <= height_200 <= 40.0
    assert abs((height_200 / height_100) - 2.0) <= 0.15


def test_override_documented_not_differentiable() -> None:
    """All override fields must be documented as outside the optimizer manifold."""
    docs = _override_docs()
    assert len(docs) == 6
    for field_name, doc in docs.items():
        assert "NOT DIFFERENTIABLE" in doc, field_name
