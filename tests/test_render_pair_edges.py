"""Regression tests for thin pair-fixture edge stems."""

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

import scripts.build_gallery_audit as gallery_audit
from dagua.graph import DaguaGraph


def _render_pair_fixture(tmp_path: Path, edge_width: float) -> Path:
    """Render the gallery pair fixture with a requested edge width.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    edge_width : float
        Edge stroke width in typographic points.

    Returns
    -------
    pathlib.Path
        Path to the rendered PNG.
    """
    item = next(
        item
        for item in gallery_audit.build_reference_items()
        if item.card_id == "nodes_shapes_box3d"
    )
    graph, positions = gallery_audit._prepare_reference_render(item)
    assert isinstance(graph, DaguaGraph)
    graph.edge_styles[0].width = edge_width
    output_path = tmp_path / f"box3d_pair_width_{edge_width:.1f}.png"
    gallery_audit._render_dagua_png(
        graph,
        positions,
        output_path,
        gallery_audit.PANEL_SIZE,
    )
    return output_path


def _dark_pixel_count(
    path: Path,
    x_range: Tuple[int, int],
    y_range: Tuple[int, int],
    threshold: int = 80,
) -> int:
    """Count dark pixels in an image corridor.

    Parameters
    ----------
    path : pathlib.Path
        PNG image to inspect.
    x_range : tuple[int, int]
        Half-open pixel-column range.
    y_range : tuple[int, int]
        Half-open pixel-row range.
    threshold : int, default=80
        Exclusive RGB upper bound used to classify a pixel as dark.

    Returns
    -------
    int
        Number of pixels whose RGB channels are all below the dark threshold.
    """
    image = np.asarray(Image.open(path).convert("RGB"))
    region = image[y_range[0] : y_range[1], x_range[0] : x_range[1]]
    mask = (
        (region[:, :, 0] < threshold)
        & (region[:, :, 1] < threshold)
        & (region[:, :, 2] < threshold)
    )
    return int(mask.sum())


def test_pair_fixture_edge_stem_visible_at_thin_widths(tmp_path: Path) -> None:
    """Default and thinner pair-fixture edges should leave visible stem pixels."""
    counts = []
    for edge_width in (0.5, 1.0, 1.5):
        path = _render_pair_fixture(tmp_path, edge_width)
        counts.append(
            _dark_pixel_count(
                path,
                x_range=(370, 430),
                y_range=(160, 440),
                threshold=160,
            )
        )

    assert counts[0] >= 150
    assert counts[1] >= 150
    assert counts[2] >= 150
    assert counts[0] <= counts[1] <= counts[2]
