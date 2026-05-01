"""DPI-invariance checks for data-coordinate render geometry."""

from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path

from dagua.edges import BezierCurve
from dagua.graph import DaguaGraph
from dagua.render.mpl import render
from dagua.styles import ClusterStyle, EdgeStyle, NodeStyle


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


def _build_cluster_graph() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build a three-node graph with one solid-bordered cluster.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph and fixed node positions with shape ``[3, 2]``.
    """
    graph = DaguaGraph()
    graph.add_node("left", label="L")
    graph.add_node("right", label="R")
    graph.add_node("outside", label="O")
    graph.add_cluster(
        "group",
        ["left", "right"],
        label="Group",
        style=ClusterStyle(
            fill="#FFFFFF",
            opacity=0.2,
            stroke="#000000",
            stroke_width=8.0,
            stroke_dash="solid",
            padding=20.0,
        ),
    )
    positions = torch.tensor([[-45.0, 0.0], [45.0, 0.0], [135.0, 0.0]], dtype=torch.float32)
    return graph, positions


def _build_shape_graph(shape: str) -> Tuple[DaguaGraph, torch.Tensor]:
    """Build a two-node graph whose nodes use one special shape.

    Parameters
    ----------
    shape : str
        Node shape name to apply to both nodes.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph and fixed node positions with shape ``[2, 2]``.
    """
    graph = DaguaGraph()
    style = NodeStyle(
        shape=shape,
        fill="#FFFFFF",
        stroke="#000000",
        stroke_width=8.0,
        min_width=72.0,
        min_height=54.0,
    )
    graph.add_node("source", label="A", style=style)
    graph.add_node("target", label="B", style=style)
    graph.add_edge("source", "target")
    positions = torch.tensor([[0.0, 0.0], [140.0, 0.0]], dtype=torch.float32)
    return graph, positions


def _build_text_outline_graph() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build a node fixture with text outline rendering enabled.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph and fixed node positions with shape ``[1, 2]``.
    """
    graph = DaguaGraph()
    graph.add_node(
        "outlined",
        label="Outline",
        style=NodeStyle(
            fill="#FFFFFF",
            stroke="#DDDDDD",
            stroke_width=2.0,
            min_width=140.0,
            min_height=70.0,
            font_color="#111111",
            font_size=24.0,
            text_outline=True,
            text_outline_color="#FFFFFF",
            text_outline_width=4.0,
        ),
    )
    positions = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    return graph, positions


def _build_bold_text_graph() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build a node fixture that triggers synthetic bold emphasis ribbons.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph and fixed node positions with shape ``[1, 2]``.
    """
    graph = DaguaGraph()
    graph.add_node(
        "bold",
        label="Bold",
        style=NodeStyle(
            fill="#FFFFFF",
            stroke="#DDDDDD",
            stroke_width=2.0,
            min_width=120.0,
            min_height=70.0,
            font_color="#111111",
            font_size=26.0,
            font_family="cmr10",
            font_weight="bold",
        ),
    )
    positions = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    return graph, positions


def _build_port_indicator_graph() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build a two-node fixture with edge endpoint port indicators.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph and fixed node positions with shape ``[2, 2]``.
    """
    graph = DaguaGraph()
    node_style = NodeStyle(
        fill="#FFFFFF",
        stroke="#444444",
        stroke_width=2.0,
        min_width=72.0,
        min_height=54.0,
    )
    graph.add_node("source", label="S", style=node_style)
    graph.add_node("target", label="T", style=node_style)
    graph.add_edge(
        "source",
        "target",
        style=EdgeStyle(
            color="#111111",
            width=8.0,
            opacity=1.0,
            port_indicator="circle",
            port_indicator_size=14.0,
            port_indicator_color="#111111",
        ),
    )
    positions = torch.tensor([[0.0, 0.0], [160.0, 0.0]], dtype=torch.float32)
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


def _data_span_to_pixels(ax: Any, width: float, height: float) -> Tuple[float, float]:
    """Convert a data-coordinate span to display-pixel lengths.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    width : float
        Data-coordinate horizontal span.
    height : float
        Data-coordinate vertical span.

    Returns
    -------
    tuple[float, float]
        Horizontal and vertical display-pixel lengths.
    """
    origin = ax.transData.transform((0.0, 0.0))
    corner = ax.transData.transform((width, height))
    return abs(float(corner[0] - origin[0])), abs(float(corner[1] - origin[1]))


def _split_path_subpaths(path: Path) -> List[np.ndarray]:
    """Split a compound matplotlib path into closed subpath vertex arrays.

    Parameters
    ----------
    path : matplotlib.path.Path
        Compound path to split.

    Returns
    -------
    list[numpy.ndarray]
        One vertex array per non-empty subpath.
    """
    subpaths: List[List[np.ndarray]] = []
    current: List[np.ndarray] = []
    for vertex, code in zip(path.vertices, path.codes):
        if code == Path.MOVETO:
            if current:
                subpaths.append(current)
            current = [vertex]
            continue
        if code == Path.CLOSEPOLY:
            if current:
                subpaths.append(current)
                current = []
            continue
        current.append(vertex)
    if current:
        subpaths.append(current)
    return [np.asarray(subpath, dtype=float) for subpath in subpaths if len(subpath) >= 3]


def _compound_ring_width_ratio(ax: Any, path: Path, owner_width_data: float) -> float:
    """Measure a compound annular ring's border-width ratio in pixels.

    Parameters
    ----------
    ax : Any
        Matplotlib axes containing the ring.
    path : matplotlib.path.Path
        Compound ring path with outer and inner subpaths in data coordinates.
    owner_width_data : float
        Width of the owning node or cluster in data coordinates.

    Returns
    -------
    float
        Ring thickness divided by owner width, both in display pixels.
    """
    subpaths = _split_path_subpaths(path)
    assert len(subpaths) >= 2
    display_bboxes = []
    for subpath in subpaths[:2]:
        display_vertices = ax.transData.transform(subpath)
        display_bboxes.append(
            (
                float(display_vertices[:, 0].min()),
                float(display_vertices[:, 0].max()),
                float(display_vertices[:, 1].min()),
                float(display_vertices[:, 1].max()),
            )
        )
    display_bboxes.sort(key=lambda bbox: (bbox[1] - bbox[0]) * (bbox[3] - bbox[2]), reverse=True)
    outer_bbox, inner_bbox = display_bboxes[:2]
    horizontal_width = ((outer_bbox[1] - outer_bbox[0]) - (inner_bbox[1] - inner_bbox[0])) / 2.0
    vertical_width = ((outer_bbox[3] - outer_bbox[2]) - (inner_bbox[3] - inner_bbox[2])) / 2.0
    owner_width_px, _ = _data_span_to_pixels(ax, owner_width_data, 0.0)
    return max(min(horizontal_width, vertical_width), 0.0) / max(owner_width_px, 1.0)


def _paths_at_zorder(ax: Any, zorder: float) -> List[Path]:
    """Return patch-collection paths drawn at one z-order.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    zorder : float
        Artist z-order to match.

    Returns
    -------
    list[matplotlib.path.Path]
        Paths from matching patch collections.
    """
    paths: List[Path] = []
    for collection in ax.collections:
        if (
            isinstance(collection, PatchCollection)
            and abs(float(collection.get_zorder()) - zorder) < 1e-6
        ):
            paths.extend(collection.get_paths())
    return paths


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


def _path_patch_pixel_bbox(ax: Any, gid: str) -> Tuple[float, float, float, float]:
    """Return one path patch's bbox in display pixels.

    Parameters
    ----------
    ax : Any
        Matplotlib axes containing the path patch.
    gid : str
        Patch gid to measure.

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
    vertices = ax.transData.transform(
        np.concatenate([patch.get_path().vertices for patch in patches], axis=0)
    )
    return (
        float(vertices[:, 0].min()),
        float(vertices[:, 0].max()),
        float(vertices[:, 1].min()),
        float(vertices[:, 1].max()),
    )


def _bbox_height(bbox: Tuple[float, float, float, float]) -> float:
    """Return a display-pixel bbox height.

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        Display-pixel bbox as ``(x_min, x_max, y_min, y_max)``.

    Returns
    -------
    float
        Absolute bbox height in pixels.
    """
    return abs(float(bbox[3] - bbox[2]))


def _bbox_width(bbox: Tuple[float, float, float, float]) -> float:
    """Return a display-pixel bbox width.

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        Display-pixel bbox as ``(x_min, x_max, y_min, y_max)``.

    Returns
    -------
    float
        Absolute bbox width in pixels.
    """
    return abs(float(bbox[1] - bbox[0]))


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
        fit_to_canvas=True,
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

    ratios = {
        "border_to_node": border_width / max(float(node_width), 1.0),
        "font_to_node": abs(label_y_max - label_y_min) / max(float(node_height), 1.0),
        "edge_to_separation": edge_width / max(float(node_separation), 1.0),
    }
    plt.close(fig)
    return ratios


def _render_cluster_border_ratio(dpi: int) -> float:
    """Render the cluster fixture and measure the solid border ratio.

    Parameters
    ----------
    dpi : int
        Rasterization DPI for this render.

    Returns
    -------
    float
        Cluster border pixel width divided by cluster pixel width.
    """
    graph, positions = _build_cluster_graph()
    graph.compute_node_sizes()
    fig, ax = render(graph, positions, dpi=dpi, figsize=(4.0, 2.2), svg_hover_text=False)
    paths = _paths_at_zorder(ax, 0.05)
    assert len(paths) == 1
    cluster_path = paths[0]
    owner_width = float(cluster_path.vertices[:, 0].max() - cluster_path.vertices[:, 0].min())
    ratio = _compound_ring_width_ratio(ax, cluster_path, owner_width)
    plt.close(fig)
    return ratio


def _render_shape_inner_ring_ratio(shape: str, dpi: int) -> float:
    """Render a special node shape and measure its inner-ring ratio.

    Parameters
    ----------
    shape : str
        Node shape name to render.
    dpi : int
        Rasterization DPI for this render.

    Returns
    -------
    float
        Inner ring or rim pixel width divided by node pixel width.
    """
    graph, positions = _build_shape_graph(shape)
    graph.compute_node_sizes()
    sizes = graph.node_sizes.detach().cpu().numpy()
    fig, ax = render(graph, positions, dpi=dpi, figsize=(4.0, 2.0), svg_hover_text=False)
    paths = _paths_at_zorder(ax, 2.08)
    assert len(paths) == graph.num_nodes
    ratio = _compound_ring_width_ratio(ax, paths[0], float(sizes[0, 0]))
    plt.close(fig)
    return ratio


def _render_text_outline_ratios(dpi: int) -> Dict[str, float]:
    """Render the outline fixture and measure text-outline ratios.

    Parameters
    ----------
    dpi : int
        Rasterization DPI for this render.

    Returns
    -------
    dict[str, float]
        Outline expansion ratio measured against label height.
    """
    graph, positions = _build_text_outline_graph()
    graph.compute_node_sizes()
    fig, ax = render(graph, positions, dpi=dpi, figsize=(4.0, 2.0), svg_hover_text=False)
    label_bbox = _label_pixel_bbox(ax, "dagua-node-label-0")
    outline_bbox = _path_patch_pixel_bbox(ax, "dagua-node-label-0-outline-0-0")
    outline_pixel_width = max(
        (_bbox_width(outline_bbox) - _bbox_width(label_bbox)) / 2.0,
        (_bbox_height(outline_bbox) - _bbox_height(label_bbox)) / 2.0,
    )
    ratios = {"outline_to_label_height": outline_pixel_width / max(_bbox_height(label_bbox), 1.0)}
    plt.close(fig)
    return ratios


def _render_bold_emphasis_ratios(dpi: int) -> Dict[str, float]:
    """Render the bold fixture and measure bold-emphasis ratios.

    Parameters
    ----------
    dpi : int
        Rasterization DPI for this render.

    Returns
    -------
    dict[str, float]
        Bold-emphasis expansion ratio measured against label height.
    """
    graph, positions = _build_bold_text_graph()
    graph.compute_node_sizes()
    fig, ax = render(graph, positions, dpi=dpi, figsize=(4.0, 2.0), svg_hover_text=False)
    label_bbox = _label_pixel_bbox(ax, "dagua-node-label-0")
    emphasis_bbox = _path_patch_pixel_bbox(ax, "dagua-node-label-0-embolden-0-0")
    emphasis_pixel_width = max(
        (_bbox_width(emphasis_bbox) - _bbox_width(label_bbox)) / 2.0,
        (_bbox_height(emphasis_bbox) - _bbox_height(label_bbox)) / 2.0,
    )
    ratios = {
        "bold_emphasis_to_label_height": emphasis_pixel_width / max(_bbox_height(label_bbox), 1.0)
    }
    plt.close(fig)
    return ratios


def _render_port_indicator_ratios(dpi: int) -> Dict[str, float]:
    """Render the port fixture and measure marker-to-edge ratios.

    Parameters
    ----------
    dpi : int
        Rasterization DPI for this render.

    Returns
    -------
    dict[str, float]
        Port marker dimensions measured relative to edge stroke width.
    """
    graph, positions = _build_port_indicator_graph()
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
    edge_path = next(
        path
        for collection in ax.collections
        if (
            isinstance(collection, PatchCollection)
            and abs(float(collection.get_zorder()) - 1.0) < 1e-6
        )
        for path in collection.get_paths()
    )
    edge_vertices = ax.transData.transform(edge_path.vertices)
    edge_width = float(edge_vertices[:, 1].max() - edge_vertices[:, 1].min())
    source_marker_bbox = _path_patch_pixel_bbox(ax, "dagua-port-indicator-0-source")
    target_marker_bbox = _path_patch_pixel_bbox(ax, "dagua-port-indicator-0-target")
    ratios = {
        "source_marker_to_edge": _bbox_width(source_marker_bbox) / max(edge_width, 1.0),
        "target_marker_to_edge": _bbox_width(target_marker_bbox) / max(edge_width, 1.0),
    }
    plt.close(fig)
    return ratios


def _assert_dpi_invariant(values_by_dpi: Dict[int, float]) -> None:
    """Assert scalar ratios remain stable across DPI values.

    Parameters
    ----------
    values_by_dpi : dict[int, float]
        Ratio values keyed by rasterization DPI.
    """
    tolerance = 0.05
    baseline = values_by_dpi[100]
    assert baseline > 0.0, values_by_dpi
    for dpi, ratio in values_by_dpi.items():
        assert ratio > 0.0, (dpi, ratio, values_by_dpi)
        allowed_delta = max(abs(baseline) * tolerance, 1e-6)
        assert abs(ratio - baseline) <= allowed_delta, (dpi, ratio, baseline, values_by_dpi)


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


def test_cluster_border_dpi_invariance() -> None:
    """Cluster solid borders should be filled data-coordinate ribbons."""
    ratios_by_dpi = {dpi: _render_cluster_border_ratio(dpi) for dpi in (100, 150, 200, 300)}
    _assert_dpi_invariant(ratios_by_dpi)


def test_text_outline_dpi_invariance() -> None:
    """Text outline ratio should stay stable across raster DPI."""
    tolerance = 0.05
    ratios_by_dpi = {dpi: _render_text_outline_ratios(dpi) for dpi in (100, 150, 200, 300)}
    baseline = ratios_by_dpi[100]

    for dpi, ratios in ratios_by_dpi.items():
        for key, ratio in ratios.items():
            assert ratio > 0.0, (dpi, key, ratio, ratios)
            allowed_delta = max(abs(baseline[key]) * tolerance, 1e-6)
            assert abs(ratio - baseline[key]) <= allowed_delta, (dpi, key, ratio, baseline[key])


def test_port_indicator_dpi_invariance() -> None:
    """Port indicator marker ratios should stay stable across raster DPI."""
    tolerance = 0.05
    ratios_by_dpi = {dpi: _render_port_indicator_ratios(dpi) for dpi in (100, 150, 200, 300)}
    baseline = ratios_by_dpi[100]

    for dpi, ratios in ratios_by_dpi.items():
        for key, ratio in ratios.items():
            assert ratio > 0.0, (dpi, key, ratio, ratios)
            allowed_delta = max(abs(baseline[key]) * tolerance, 1e-6)
            assert abs(ratio - baseline[key]) <= allowed_delta, (dpi, key, ratio, baseline[key])


def test_bold_emphasis_dpi_invariance() -> None:
    """Bold emphasis ratio should stay stable across raster DPI."""
    tolerance = 0.05
    ratios_by_dpi = {dpi: _render_bold_emphasis_ratios(dpi) for dpi in (100, 150, 200, 300)}
    baseline = ratios_by_dpi[100]

    for dpi, ratios in ratios_by_dpi.items():
        for key, ratio in ratios.items():
            assert ratio > 0.0, (dpi, key, ratio, ratios)
            allowed_delta = max(abs(baseline[key]) * tolerance, 1e-6)
            assert abs(ratio - baseline[key]) <= allowed_delta, (dpi, key, ratio, baseline[key])


def test_double_circle_inner_ring_dpi_invariance() -> None:
    """Double-circle inner rings should be filled data-coordinate ribbons."""
    ratios_by_dpi = {
        dpi: _render_shape_inner_ring_ratio("double_circle", dpi) for dpi in (100, 150, 200, 300)
    }
    _assert_dpi_invariant(ratios_by_dpi)


def test_cylinder_rim_dpi_invariance() -> None:
    """Cylinder rims should be filled data-coordinate ribbons."""
    ratios_by_dpi = {
        dpi: _render_shape_inner_ring_ratio("cylinder", dpi) for dpi in (100, 150, 200, 300)
    }
    _assert_dpi_invariant(ratios_by_dpi)
