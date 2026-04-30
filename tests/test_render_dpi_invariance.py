"""DPI-invariance checks for data-coordinate render geometry."""

from typing import Any, Dict, Tuple

import numpy as np
import torch
from matplotlib.patches import PathPatch

from dagua.edges import BezierCurve
from dagua.graph import DaguaGraph
from dagua.render.mpl import render


def _build_pair_graph() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build the canonical two-node render fixture.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph and fixed node positions with shape ``[2, 2]``.
    """
    graph = DaguaGraph()
    graph.add_node("source", label="A")
    graph.add_node("target", label="B")
    graph.add_edge("source", "target")
    positions = torch.tensor([[0.0, 0.0], [180.0, 0.0]], dtype=torch.float32)
    return graph, positions


def _canvas_rgb(fig: Any) -> np.ndarray:
    """Return the current figure canvas as an RGB array.

    Parameters
    ----------
    fig : Any
        Matplotlib figure with a drawn Agg canvas.

    Returns
    -------
    numpy.ndarray
        RGB image with shape ``[H, W, 3]``.
    """
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    return rgba[:, :, :3]


def _data_point_to_pixel(ax: Any, x: float, y: float) -> Tuple[int, int]:
    """Convert one data-coordinate point to integer canvas pixels.

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
    """Measure the contiguous true run around one pixel index.

    Parameters
    ----------
    mask : numpy.ndarray
        One-dimensional boolean mask.
    center : int
        Pixel index expected to be inside the run.

    Returns
    -------
    int
        Contiguous run width in pixels.
    """
    if center < 0 or center >= mask.shape[0] or not bool(mask[center]):
        true_indices = np.flatnonzero(mask)
        if true_indices.size == 0:
            return 0
        center = int(true_indices[np.argmin(np.abs(true_indices - center))])
    left = center
    while left > 0 and bool(mask[left - 1]):
        left -= 1
    right = center
    while right + 1 < mask.shape[0] and bool(mask[right + 1]):
        right += 1
    return int(right - left + 1)


def _label_pixel_bbox(ax: Any, gid: str) -> Tuple[float, float, float, float]:
    """Return a rendered label path bbox in display pixels.

    Parameters
    ----------
    ax : Any
        Matplotlib axes containing label path patches.
    gid : str
        Label patch gid to measure.

    Returns
    -------
    tuple[float, float, float, float]
        Display-pixel bbox as ``(x_min, x_max, y_min, y_max)``.
    """
    patches = [
        patch
        for patch in ax.patches
        if isinstance(patch, PathPatch) and str(patch.get_gid()) == gid
    ]
    assert patches
    vertices = np.concatenate([patch.get_path().vertices for patch in patches], axis=0)
    display_vertices = ax.transData.transform(vertices)
    return (
        float(display_vertices[:, 0].min()),
        float(display_vertices[:, 0].max()),
        float(display_vertices[:, 1].min()),
        float(display_vertices[:, 1].max()),
    )


def _render_ratios(dpi: int) -> Dict[str, float]:
    """Render the pair fixture and extract relative geometry ratios.

    Parameters
    ----------
    dpi : int
        Rasterization DPI for this render.

    Returns
    -------
    dict[str, float]
        Border, font, and edge ratios measured in pixels.
    """
    graph, positions = _build_pair_graph()
    graph.compute_node_sizes()
    sizes = graph.node_sizes.detach().cpu().numpy()
    source_x = float(positions[0, 0] + sizes[0, 0] / 2.0)
    target_x = float(positions[1, 0] - sizes[1, 0] / 2.0)
    curves = [
        BezierCurve(
            p0=(source_x, 0.0),
            cp1=(source_x + (target_x - source_x) / 3.0, 0.0),
            cp2=(source_x + 2.0 * (target_x - source_x) / 3.0, 0.0),
            p1=(target_x, 0.0),
        )
    ]
    fig, ax = render(
        graph,
        positions,
        dpi=dpi,
        figsize=(4.0, 2.0),
        curves=curves,
        svg_hover_text=False,
    )
    image = _canvas_rgb(fig)

    node_left, node_center_y = _data_point_to_pixel(
        ax,
        float(positions[0, 0] - sizes[0, 0] / 2.0),
        float(positions[0, 1]),
    )
    node_right, _ = _data_point_to_pixel(
        ax,
        float(positions[0, 0] + sizes[0, 0] / 2.0),
        float(positions[0, 1]),
    )
    node_width = abs(node_right - node_left)

    dark = np.all(image < 120, axis=2)
    row = dark[node_center_y, :]
    border_width = _dark_run_width(row, node_left)

    source_px, _ = _data_point_to_pixel(ax, float(positions[0, 0]), 0.0)
    target_px, _ = _data_point_to_pixel(ax, float(positions[1, 0]), 0.0)
    edge_patch = next(
        patch
        for patch in ax.patches
        if isinstance(patch, PathPatch) and patch.get_gid() is None and patch.get_zorder() < 2.0
    )
    edge_vertices = ax.transData.transform(edge_patch.get_path().vertices)
    edge_width = float(edge_vertices[:, 1].max() - edge_vertices[:, 1].min())
    node_separation = abs(target_px - source_px)

    _, _, label_y_min, label_y_max = _label_pixel_bbox(ax, "dagua-node-label-0")
    node_top_y = _data_point_to_pixel(
        ax,
        float(positions[0, 0]),
        float(positions[0, 1] + sizes[0, 1] / 2.0),
    )[1]
    node_bottom_y = _data_point_to_pixel(
        ax,
        float(positions[0, 0]),
        float(positions[0, 1] - sizes[0, 1] / 2.0),
    )[1]
    node_height = abs(node_bottom_y - node_top_y)

    return {
        "border_to_node": border_width / max(float(node_width), 1.0),
        "font_to_node": abs(label_y_max - label_y_min) / max(float(node_height), 1.0),
        "edge_to_separation": edge_width / max(float(node_separation), 1.0),
    }


def test_pair_fixture_geometry_ratios_are_dpi_invariant() -> None:
    """Border, text, and edge ratios should stay stable across raster DPI."""
    tolerance = 0.05
    ratios_by_dpi = {dpi: _render_ratios(dpi) for dpi in (100, 150, 200, 300)}
    baseline = ratios_by_dpi[100]

    for dpi, ratios in ratios_by_dpi.items():
        for key, ratio in ratios.items():
            assert ratio > 0.0, (dpi, key, ratio, ratios)
            allowed_delta = max(abs(baseline[key]) * tolerance, 1e-6)
            assert abs(ratio - baseline[key]) <= allowed_delta, (dpi, key, ratio, baseline[key])
