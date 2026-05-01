"""Regression coverage for graphviz-compatible natural canvas rendering."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest
import torch
from PIL import Image
from skimage.metrics import structural_similarity

import dagua
from dagua.styles import GRAPHVIZ_STRICT_THEME, Theme


def _read_rgb_on_white(path: Path) -> np.ndarray:
    """Read an image as RGB composited over white.

    Parameters
    ----------
    path : Path
        PNG image path.

    Returns
    -------
    np.ndarray
        RGB image array with shape ``[H, W, 3]`` and dtype ``uint8``.
    """
    image = Image.open(path).convert("RGBA")
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    return np.asarray(Image.alpha_composite(background, image).convert("RGB"))


def _plain_node_positions(dot_text: str) -> Dict[str, Tuple[float, float]]:
    """Return graphviz ``dot -Tplain`` node positions in Dagua point units.

    Parameters
    ----------
    dot_text : str
        DOT source text.

    Returns
    -------
    dict[str, tuple[float, float]]
        Mapping from node id to ``(x, y)`` positions in points.
    """
    completed = subprocess.run(
        ["dot", "-Tplain", "-Gdpi=96"],
        input=dot_text,
        check=True,
        capture_output=True,
        text=True,
    )
    positions: Dict[str, Tuple[float, float]] = {}
    for line in completed.stdout.splitlines():
        parts = line.split()
        if parts and parts[0] == "node":
            positions[parts[1]] = (float(parts[2]) * 72.0, float(parts[3]) * 72.0)
    return positions


def _blank_canvas_theme() -> Theme:
    """Return a graphviz-strict theme that isolates canvas math.

    Parameters
    ----------
    None

    Returns
    -------
    Theme
        Theme with invisible nodes and edges but graphviz canvas defaults.
    """
    theme = GRAPHVIZ_STRICT_THEME.copy()
    theme.graph_style = replace(theme.graph_style, density_aware_node_shrink=False)
    theme.node_styles = {
        name: replace(
            style,
            fill="#FFFFFF",
            stroke="#FFFFFF",
            stroke_width=0.0,
            font_color="#FFFFFF",
        )
        for name, style in theme.node_styles.items()
    }
    theme.edge_styles = {
        name: replace(style, color="#FFFFFF", opacity=0.0, width=0.0, arrow="none")
        for name, style in theme.edge_styles.items()
    }
    return theme


def test_graphviz_natural_canvas_matches_dot_png(tmp_path: Path) -> None:
    """Dagua natural canvas should match dot PNG dimensions and pixels."""
    if shutil.which("dot") is None:
        pytest.skip("graphviz dot CLI is not available")

    dot_text = """
digraph G {
  graph [rankdir=TB margin=0.055 bgcolor="white"];
  node [style=invis label="" width=0.75 height=0.5];
  edge [style=invis];
  A -> B -> C;
  A -> D -> E;
}
"""
    expected_path = tmp_path / "expected.png"
    actual_path = tmp_path / "actual.png"
    subprocess.run(
        ["dot", "-Tpng", "-Gdpi=96", "-o", str(expected_path)],
        input=dot_text,
        check=True,
        text=True,
    )

    graph = dagua.from_dot(dot_text, _theme=_blank_canvas_theme())
    graph.node_labels = [""] * graph.num_nodes
    graphviz_positions = _plain_node_positions(dot_text)
    positions = torch.zeros((graph.num_nodes, 2), dtype=torch.float32)
    for index, node_id in enumerate(graph._index_to_id):
        positions[index] = torch.tensor(graphviz_positions[str(node_id)], dtype=torch.float32)

    dagua.render(graph, positions, output=str(actual_path), backend="cairo")

    expected = _read_rgb_on_white(expected_path)
    actual = _read_rgb_on_white(actual_path)
    assert actual.shape == expected.shape

    mean_l1 = float(np.mean(np.abs(expected.astype(np.float32) - actual.astype(np.float32))))
    similarity = float(structural_similarity(expected, actual, channel_axis=2, data_range=255))
    assert similarity >= 0.95
    assert mean_l1 <= 5.0
