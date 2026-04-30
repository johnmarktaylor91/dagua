"""Regression tests for density-aware node label scaling."""

from typing import Any, Tuple

import matplotlib
import numpy as np
import torch
from matplotlib.patches import PathPatch

import scripts.build_gallery_audit as gallery_audit
from dagua.graph import DaguaGraph
from dagua.render.mpl import render

matplotlib.use("Agg")


def _build_combo_graph() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build the gallery combo workflow fixture used by combo_pie_bold.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Configured graph and fixed positions with shape ``[5, 2]``.
    """
    item = next(
        item for item in gallery_audit.build_combo_items() if item.card_id == "combo_pie_bold"
    )
    fixture = gallery_audit._choose_combo_fixture(item.spec.settings)
    direction = str(item.spec.settings.get("direction", "TB"))
    graph, positions = gallery_audit._build_fixture(fixture, direction=direction)
    params = gallery_audit._combo_params(item.spec.settings, fixture)
    positions = gallery_audit._apply_reference_params(graph, positions, params, fixture)
    return graph, positions


def _apply_gallery_axes(ax: Any, graph: DaguaGraph, positions: torch.Tensor) -> None:
    """Apply the fixed-canvas axis normalization used by gallery audit renders.

    Parameters
    ----------
    ax : Any
        Matplotlib axes returned by the renderer.
    graph : DaguaGraph
        Rendered graph with computed node sizes.
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.

    Returns
    -------
    None
        The axes limits are updated in place.
    """
    graph.compute_node_sizes()
    sizes = graph.node_sizes.detach().cpu()
    pos = positions.detach().cpu()
    x_min = float((pos[:, 0] - sizes[:, 0] / 2.0).min())
    x_max = float((pos[:, 0] + sizes[:, 0] / 2.0).max())
    y_min = float((pos[:, 1] - sizes[:, 1] / 2.0).min())
    y_max = float((pos[:, 1] + sizes[:, 1] / 2.0).max())
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    ax.set_xlim(
        x_center - gallery_audit.CARD_SIZE[0] / 2.0,
        x_center + gallery_audit.CARD_SIZE[0] / 2.0,
    )
    ax.set_ylim(
        y_center - gallery_audit.CARD_SIZE[1] / 2.0,
        y_center + gallery_audit.CARD_SIZE[1] / 2.0,
    )


def _display_width(ax: Any, vertices: np.ndarray) -> float:
    """Return a data-space path width after axes transformation to pixels.

    Parameters
    ----------
    ax : Any
        Matplotlib axes containing the path.
    vertices : numpy.ndarray
        Path vertices with shape ``[N, 2]``.

    Returns
    -------
    float
        Width in display pixels.
    """
    display_vertices = ax.transData.transform(vertices)
    return float(display_vertices[:, 0].max() - display_vertices[:, 0].min())


def test_density_aware_labels_fit_inside_shrunk_combo_nodes() -> None:
    """The Ingest label should fit inside the density-shrunk node."""
    graph, positions = _build_combo_graph()
    fig, ax = render(
        graph,
        positions,
        dpi=gallery_audit.RENDER_DPI,
        figsize=(
            gallery_audit.CARD_SIZE[0] / gallery_audit.RENDER_DPI,
            gallery_audit.CARD_SIZE[1] / gallery_audit.RENDER_DPI,
        ),
    )
    _apply_gallery_axes(ax, graph, positions)
    fig.canvas.draw()

    label_patch = next(patch for patch in ax.patches if patch.get_gid() == "dagua-node-label-0")
    node_patches = [
        patch
        for patch in ax.patches
        if isinstance(patch, PathPatch)
        and patch.get_gid() is None
        and abs(
            (patch.get_path().vertices[:, 0].min() + patch.get_path().vertices[:, 0].max()) / 2.0
        )
        < 1.0
        and abs(
            (patch.get_path().vertices[:, 1].min() + patch.get_path().vertices[:, 1].max()) / 2.0
            - 210.0
        )
        < 1.0
    ]
    assert node_patches

    label_width = _display_width(ax, label_patch.get_path().vertices)
    node_width = max(_display_width(ax, patch.get_path().vertices) for patch in node_patches)

    assert label_width / node_width <= 0.95
