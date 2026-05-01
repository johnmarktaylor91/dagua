"""Regression tests for thin dashed and dotted edge visibility."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from dagua.graph import DaguaGraph
from dagua.render.mpl import render
from dagua.styles import GRAPHVIZ_STRICT_THEME, EdgeStyle


def _build_pair_graph(edge_style: str) -> Tuple[DaguaGraph, torch.Tensor]:
    """Build the canonical long pair fixture for one edge style.

    Parameters
    ----------
    edge_style : str
        Edge style name, such as ``"dashed"`` or ``"dotted"``.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph and fixed node positions with shape ``[2, 2]``.
    """
    graph = DaguaGraph.from_edge_list(
        [("Source", "Target")],
        _theme=copy.deepcopy(GRAPHVIZ_STRICT_THEME),
        direction="TB",
    )
    graph.edge_styles[0] = EdgeStyle(style=edge_style)
    graph.node_labels = ["", ""]
    positions = torch.tensor([[0.0, 130.0], [0.0, -130.0]], dtype=torch.float32)
    graph.compute_node_sizes()
    return graph, positions


def _data_to_pixel(
    ax: object,
    image_shape: Tuple[int, int, int],
    point: Tuple[float, float],
) -> Tuple[int, int]:
    """Convert an axes data point to image pixel coordinates.

    Parameters
    ----------
    ax : object
        Matplotlib axes used for the render.
    image_shape : tuple[int, int, int]
        RGB image shape as ``[height, width, channels]``.
    point : tuple[float, float]
        Data-coordinate point.

    Returns
    -------
    tuple[int, int]
        Pixel coordinate as ``(x, y)`` with image-origin coordinates.
    """
    display_x, display_y = ax.transData.transform(point)  # type: ignore[attr-defined]
    height = int(image_shape[0])
    return int(round(float(display_x))), int(round(float(height) - float(display_y)))


def _dark_mask(image: np.ndarray, threshold: float = 185.0) -> np.ndarray:
    """Return a mask for black and anti-aliased dark edge pixels.

    Parameters
    ----------
    image : numpy.ndarray
        RGB image data with shape ``[H, W, 3]``.
    threshold : float, default=185.0
        Luma cutoff for dark pixels.

    Returns
    -------
    numpy.ndarray
        Boolean mask with shape ``[H, W]``.
    """
    luma = (
        image[:, :, 0].astype(np.float32) * 0.2126
        + image[:, :, 1].astype(np.float32) * 0.7152
        + image[:, :, 2].astype(np.float32) * 0.0722
    )
    return luma < threshold


def _render_pair_image(
    tmp_path: Path,
    edge_style: str,
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """Render one long pair edge and return probe coordinates.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory for the rendered PNG.
    edge_style : str
        Edge style name.

    Returns
    -------
    tuple[numpy.ndarray, tuple[int, int], tuple[int, int]]
        RGB image, source-side edge-boundary pixel, and target-side boundary
        pixel. Boundary points sit at the analytic node intersections.
    """
    graph, positions = _build_pair_graph(edge_style)
    output_path = tmp_path / f"{edge_style}.png"
    fig, ax = render(
        graph,
        positions=positions,
        output=str(output_path),
        dpi=100,
        figsize=(8.0, 6.0),
    )
    fig.canvas.draw()

    assert graph.node_sizes is not None
    source_half_height = float(graph.node_sizes[0, 1]) / 2.0
    target_half_height = float(graph.node_sizes[1, 1]) / 2.0
    image = np.asarray(Image.open(output_path).convert("RGB"))
    source_boundary = _data_to_pixel(
        ax,
        image.shape,
        (0.0, float(positions[0, 1]) - source_half_height),
    )
    target_boundary = _data_to_pixel(
        ax,
        image.shape,
        (0.0, float(positions[1, 1]) + target_half_height),
    )
    plt.close(fig)
    return image, source_boundary, target_boundary


def _corridor_dark_pixels(
    image: np.ndarray,
    source_boundary: Tuple[int, int],
    target_boundary: Tuple[int, int],
) -> int:
    """Count dark pixels in the visible edge corridor between nodes.

    Parameters
    ----------
    image : numpy.ndarray
        RGB image data with shape ``[H, W, 3]``.
    source_boundary : tuple[int, int]
        Source-side edge-boundary pixel.
    target_boundary : tuple[int, int]
        Target-side edge-boundary pixel.

    Returns
    -------
    int
        Number of dark pixels in the edge corridor.
    """
    dark = _dark_mask(image)
    x_center = int(round((source_boundary[0] + target_boundary[0]) / 2.0))
    row_min = min(source_boundary[1], target_boundary[1]) + 8
    row_max = max(source_boundary[1], target_boundary[1]) - 24
    col_min = max(x_center - 7, 0)
    col_max = min(x_center + 8, dark.shape[1])
    return int(dark[row_min:row_max, col_min:col_max].sum())


def _arrowhead_dark_pixels(image: np.ndarray, target_boundary: Tuple[int, int]) -> int:
    """Count dark pixels in the target arrowhead region outside the node.

    Parameters
    ----------
    image : numpy.ndarray
        RGB image data with shape ``[H, W, 3]``.
    target_boundary : tuple[int, int]
        Target-side edge-boundary pixel.

    Returns
    -------
    int
        Number of dark pixels near the analytic target intersection.
    """
    dark = _dark_mask(image)
    x_center, y_tip = target_boundary
    row_min = max(y_tip - 24, 0)
    row_max = max(y_tip - 3, row_min)
    col_min = max(x_center - 18, 0)
    col_max = min(x_center + 19, dark.shape[1])
    return int(dark[row_min:row_max, col_min:col_max].sum())


def _dark_run_count(
    image: np.ndarray,
    source_boundary: Tuple[int, int],
    target_boundary: Tuple[int, int],
) -> int:
    """Count separated vertical dark runs in the edge corridor.

    Parameters
    ----------
    image : numpy.ndarray
        RGB image data with shape ``[H, W, 3]``.
    source_boundary : tuple[int, int]
        Source-side edge-boundary pixel.
    target_boundary : tuple[int, int]
        Target-side edge-boundary pixel.

    Returns
    -------
    int
        Number of separated dark runs along the corridor.
    """
    dark = _dark_mask(image, threshold=100.0)
    x_center = int(round((source_boundary[0] + target_boundary[0]) / 2.0))
    row_min = min(source_boundary[1], target_boundary[1]) + 8
    row_max = max(source_boundary[1], target_boundary[1]) - 24
    col_min = max(x_center - 5, 0)
    col_max = min(x_center + 6, dark.shape[1])
    per_row = dark[row_min:row_max, col_min:col_max].any(axis=1)
    starts = np.logical_and(per_row, np.concatenate([[True], ~per_row[:-1]]))
    return int(starts.sum())


def test_thin_dashed_edge_body_and_arrowhead_are_visible(tmp_path: Path) -> None:
    """Dashed thin GRAPHVIZ_STRICT edges should keep visible body and head ink.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    image, source_boundary, target_boundary = _render_pair_image(tmp_path, "dashed")

    assert _corridor_dark_pixels(image, source_boundary, target_boundary) >= 30
    assert _arrowhead_dark_pixels(image, target_boundary) >= 5


def test_thin_dotted_edge_body_and_arrowhead_are_visible(tmp_path: Path) -> None:
    """Dotted thin GRAPHVIZ_STRICT edges should show separated dots and a head.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    image, source_boundary, target_boundary = _render_pair_image(tmp_path, "dotted")

    assert _corridor_dark_pixels(image, source_boundary, target_boundary) >= 30
    assert _dark_run_count(image, source_boundary, target_boundary) >= 5
    assert _arrowhead_dark_pixels(image, target_boundary) >= 5
