"""Cluster-label rendering parity tests."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from matplotlib.collections import PatchCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import PathPatch
from matplotlib.path import Path

from dagua.graph import DaguaGraph
from dagua.render import mpl as mpl_renderer
from dagua.render.text import DaguaText
from dagua.styles import ClusterStyle, get_theme


def _path_bbox(path: Path) -> Tuple[float, float, float, float]:
    """Return a path bounding box as ``(xmin, xmax, ymin, ymax)``.

    Parameters
    ----------
    path : matplotlib.path.Path
        Path whose vertices are in data coordinates.

    Returns
    -------
    tuple[float, float, float, float]
        Data-coordinate path extents.
    """
    vertices = path.vertices
    return (
        float(vertices[:, 0].min()),
        float(vertices[:, 0].max()),
        float(vertices[:, 1].min()),
        float(vertices[:, 1].max()),
    )


def _background_patches(ax: Any) -> List[PathPatch]:
    """Return cluster-label background patches from an axes.

    Parameters
    ----------
    ax : Any
        Matplotlib axes containing rendered cluster labels.

    Returns
    -------
    list[PathPatch]
        Background mask patches for cluster labels.
    """
    return [
        patch
        for patch in ax.patches
        if isinstance(patch, PathPatch)
        and isinstance(patch.get_gid(), str)
        and patch.get_gid().startswith("dagua-cluster-label-")
        and patch.get_gid().endswith("-background")
    ]


def _render_single_cluster(
    style: ClusterStyle,
    monkeypatch: pytest.MonkeyPatch,
) -> Tuple[Any, Any, List[DaguaText]]:
    """Render a small single-cluster graph and capture text specs.

    Parameters
    ----------
    style : ClusterStyle
        Cluster style applied to the rendered cluster.
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture used to capture text render specs.

    Returns
    -------
    tuple[Any, Any, list[DaguaText]]
        Matplotlib figure, axes, and captured cluster-label specs.
    """
    graph = DaguaGraph()
    graph.graph_style.background_color = "#FDF6E3"
    graph.add_node("left")
    graph.add_node("right")
    graph.add_cluster("outer", ["left", "right"], label="Outer Group", style=style)

    captured_specs: List[DaguaText] = []
    original_render_text = mpl_renderer.render_text

    def capture_render_text(
        ax: Any,
        specs: Sequence[DaguaText],
        display_scale: float,
        svg_hover_map: Optional[dict[str, str]] = None,
    ) -> Any:
        """Capture text specs before delegating to the real renderer."""
        captured_specs.extend(specs)
        return original_render_text(ax, specs, display_scale, svg_hover_map)

    monkeypatch.setattr(mpl_renderer, "render_text", capture_render_text)
    monkeypatch.setattr(mpl_renderer, "measure_text_data", lambda *args, **kwargs: (56.0, 12.0))

    fig, ax = plt.subplots(figsize=(4.0, 3.0), dpi=100)
    ax.set_xlim(-80.0, 80.0)
    ax.set_ylim(-60.0, 80.0)
    fig.canvas.draw()

    mpl_renderer._draw_clusters(
        ax=ax,
        graph=graph,
        pos=np.array([[-24.0, 0.0], [24.0, 0.0]], dtype=float),
        sizes=np.array([[20.0, 20.0], [20.0, 20.0]], dtype=float),
    )
    return fig, ax, captured_specs


def test_cluster_label_top_center_anchor_matches_bbox_center(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Top-center cluster labels should anchor at the cluster bbox center."""
    style = ClusterStyle(
        label_position="top-center",
        label_offset=(0.0, 10.0),
        padding=12.0,
        stroke_width=1.0,
    )
    fig, ax, specs = _render_single_cluster(style, monkeypatch)

    label_spec = next(spec for spec in specs if spec.gid == "dagua-cluster-label-outer")
    fill_path = ax.collections[0].get_paths()[0]
    x_min, x_max, _, _ = _path_bbox(fill_path)
    bbox_center_x = (x_min + x_max) / 2.0
    label_px = ax.transData.transform((label_spec.x, label_spec.y))[0]
    center_px = ax.transData.transform((bbox_center_x, label_spec.y))[0]

    assert abs(label_px - center_px) <= 2.0
    plt.close(fig)


def test_cluster_label_background_sentinel_renders_opaque_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``@background`` sentinel should render an opaque graph-color mask."""
    style = ClusterStyle(
        label_background="@background",
        label_background_opacity=1.0,
        label_background_padding=(4.0, 2.0),
        label_position="top-center",
    )
    fig, ax, _ = _render_single_cluster(style, monkeypatch)

    patches = _background_patches(ax)
    assert len(patches) == 1
    assert patches[0].get_facecolor() == pytest.approx(to_rgba("#FDF6E3", 1.0))
    assert patches[0].get_alpha() == pytest.approx(1.0)
    plt.close(fig)


def test_graphviz_strict_empty_label_background_does_not_emit_masks() -> None:
    """Graphviz-strict labels should use their reserved band without a background patch."""
    graph = DaguaGraph()
    graph._theme = get_theme("graphviz_strict")
    for node_id in ("a", "b", "c"):
        graph.add_node(node_id)
    graph.add_cluster("outer", ["a", "b", "c"], label="Outer Group")
    graph.add_cluster("inner", ["b", "c"], label="Inner Group", parent="outer")

    fig, ax = plt.subplots(figsize=(5.0, 4.0), dpi=100)
    ax.set_xlim(-100.0, 100.0)
    ax.set_ylim(-80.0, 100.0)
    fig.canvas.draw()

    mpl_renderer._draw_clusters(
        ax=ax,
        graph=graph,
        pos=np.array([[-50.0, 0.0], [20.0, 0.0], [60.0, 0.0]], dtype=float),
        sizes=np.array([[28.0, 24.0], [28.0, 24.0], [28.0, 24.0]], dtype=float),
    )

    assert _background_patches(ax) == []
    plt.close(fig)


def test_cluster_label_background_renders_above_node_fills() -> None:
    """Explicit cluster label masks should stay above node fills when labels overlap nodes."""
    graph = DaguaGraph()
    graph._theme = get_theme("graphviz_strict")
    graph.add_node("a")
    graph.add_cluster(
        "outer",
        ["a"],
        label="Outer Group",
        style=ClusterStyle(
            label_background="@background",
            label_background_opacity=1.0,
        ),
    )

    fig, ax = mpl_renderer.render(
        graph,
        positions=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        show=False,
    )

    backgrounds = _background_patches(ax)
    node_fill_zorders = [
        float(collection.get_zorder())
        for collection in ax.collections
        if isinstance(collection, PatchCollection)
        and abs(float(collection.get_zorder()) - 2.0) < 1e-6
    ]
    glyph_zorders = [
        float(patch.get_zorder())
        for patch in ax.patches
        if isinstance(patch, PathPatch)
        and isinstance(patch.get_gid(), str)
        and patch.get_gid().startswith("dagua-cluster-label-outer")
        and not patch.get_gid().endswith("-background")
    ]

    assert backgrounds
    assert node_fill_zorders
    assert glyph_zorders
    assert min(float(patch.get_zorder()) for patch in backgrounds) > max(node_fill_zorders)
    assert {float(patch.get_zorder()) for patch in backgrounds} == set(glyph_zorders)
    plt.close(fig)


def test_top_cap_does_not_collapse_cluster_render_height(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Graphviz top caps should preserve a visible label band and content height."""
    graph = DaguaGraph()
    graph._theme = get_theme("graphviz_strict")
    graph.add_node("source")
    graph.add_node("inside")
    graph.add_edge("source", "inside")
    graph.add_cluster("child", ["inside"], label="Child")

    fig, ax = plt.subplots(figsize=(4.0, 3.0), dpi=100)
    ax.set_xlim(-50.0, 50.0)
    ax.set_ylim(-50.0, 50.0)
    fig.canvas.draw()

    ordered_clusters = mpl_renderer._cluster_render_order(graph)
    cluster_depths = mpl_renderer._cluster_depths(graph, ordered_clusters)
    pos = np.array([[0.0, 40.0], [0.0, 0.0]], dtype=float)
    sizes = np.array([[20.0, 20.0], [20.0, 20.0]], dtype=float)
    display_scale = mpl_renderer._compute_display_scale(ax)
    label_gap = 2.0 * display_scale
    cluster_y_mins = {"child": -10.0}
    cluster_y_maxes = {"child": 10.0}

    monkeypatch.setattr(
        mpl_renderer,
        "_graphviz_strict_cluster_top_cap",
        lambda *args, **kwargs: -25.0,
    )

    boxes = mpl_renderer._compute_cluster_render_bboxes(
        ax,
        graph,
        pos,
        sizes,
        ordered_clusters,
        cluster_depths,
        cluster_y_mins,
        cluster_y_maxes,
        display_scale,
        cluster_aware=False,
        label_gap=label_gap,
    )
    x_min, y_min, x_max, y_max = boxes["child"].bbox

    assert x_max > x_min
    assert y_max > y_min
    assert y_max - y_min >= 20.0
    plt.close(fig)
