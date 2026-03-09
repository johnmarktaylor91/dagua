"""Graphviz comparison utilities.

Provides functions to:
1. Layout a DaguaGraph using Graphviz's dot engine
2. Render using Graphviz's native renderer
3. Produce side-by-side comparison images (Dagua vs Graphviz)
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph


def to_dot(graph: DaguaGraph, positions: Optional[torch.Tensor] = None) -> str:
    """Convert DaguaGraph to DOT format.

    If positions are provided, pins them with pos="x,y!" attributes
    so Graphviz respects the layout (neato -n2 mode).
    """
    lines = ["digraph G {"]
    lines.append('  rankdir=TB;')
    lines.append('  node [shape=box, style=filled, fontname="Helvetica"];')
    lines.append('  edge [fontname="Helvetica", fontsize=9];')

    if positions is not None:
        pos = positions.detach().cpu()

    for i in range(graph.num_nodes):
        label = graph.node_labels[i].replace('"', '\\"')
        style = graph.get_style_for_node(i)
        attrs = [
            f'label="{label}"',
            f'fillcolor="{style.fill}"',
            f'fontcolor="{style.font_color}"',
            f'fontsize={style.font_size}',
        ]
        if style.shape == "ellipse":
            attrs.append('shape=ellipse')
        elif style.shape == "circle":
            attrs.append('shape=circle')
        elif style.shape == "diamond":
            attrs.append('shape=diamond')
        else:
            attrs.append('shape=box')
            if style.shape == "roundrect":
                attrs.append('style="filled,rounded"')

        if positions is not None:
            # Graphviz uses inches, 72 points/inch. Flip y for Graphviz convention.
            x_inch = pos[i, 0].item() / 72.0
            y_inch = -pos[i, 1].item() / 72.0  # flip y
            attrs.append(f'pos="{x_inch},{y_inch}!"')

        attrs_str = ", ".join(attrs)
        lines.append(f'  n{i} [{attrs_str}];')

    # Edges
    if graph.edge_index.numel() > 0:
        for e in range(graph.edge_index.shape[1]):
            s = graph.edge_index[0, e].item()
            t = graph.edge_index[1, e].item()
            edge_attrs = []
            if e < len(graph.edge_labels) and graph.edge_labels[e]:
                lbl = graph.edge_labels[e].replace('"', '\\"')
                edge_attrs.append(f'label="{lbl}"')
            style = graph.get_style_for_edge(e)
            edge_attrs.append(f'color="{style.color}"')
            if style.style == "dashed":
                edge_attrs.append('style=dashed')
            elif style.style == "dotted":
                edge_attrs.append('style=dotted')
            attrs_str = ", ".join(edge_attrs) if edge_attrs else ""
            lines.append(f'  n{s} -> n{t} [{attrs_str}];')

    # Clusters (subgraphs)
    if graph.clusters:
        for name, members in graph.clusters.items():
            if isinstance(members, dict):
                # Nested clusters — flatten for DOT
                from dagua.utils import collect_cluster_leaves
                indices = collect_cluster_leaves(members)
            else:
                indices = members
            if not indices:
                continue
            cluster_label = graph.cluster_labels.get(name, name)
            lines.append(f'  subgraph cluster_{name.replace(".", "_")} {{')
            lines.append(f'    label="{cluster_label}";')
            lines.append('    style=filled; color=lightgrey; fillcolor="#f0f0f0";')
            for idx in indices:
                lines.append(f'    n{idx};')
            lines.append('  }')

    lines.append("}")
    return "\n".join(lines)


def layout_with_graphviz(
    graph: DaguaGraph,
    engine: str = "dot",
) -> torch.Tensor:
    """Layout a DaguaGraph using Graphviz and return positions as a tensor.

    Runs `dot -Tjson` to get computed positions, then parses them back
    into a [N, 2] tensor matching node order.
    """
    dot_str = to_dot(graph)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".dot", delete=False) as f:
        f.write(dot_str)
        dot_path = f.name

    try:
        result = subprocess.run(
            [engine, "-Tjson", dot_path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Graphviz failed: {result.stderr}")
        data = json.loads(result.stdout)
    finally:
        Path(dot_path).unlink(missing_ok=True)

    # Parse positions from JSON output
    positions = torch.zeros(graph.num_nodes, 2)

    # Graphviz JSON has objects array with _gvid and pos
    if "objects" in data:
        for obj in data["objects"]:
            name = obj.get("name", "")
            if name.startswith("n") and name[1:].isdigit():
                idx = int(name[1:])
                if idx < graph.num_nodes and "pos" in obj:
                    coords = obj["pos"].split(",")
                    x = float(coords[0])
                    y = float(coords[1])
                    # Convert from Graphviz coordinates (y increases upward)
                    # to our coordinates (y increases downward for TB layout)
                    positions[idx, 0] = x
                    positions[idx, 1] = -y  # flip y

    return positions


def render_graphviz_native(
    graph: DaguaGraph,
    output: str,
    engine: str = "dot",
    fmt: Optional[str] = None,
) -> str:
    """Render a DaguaGraph using Graphviz's native renderer.

    Returns path to the output file.
    """
    dot_str = to_dot(graph)

    if fmt is None:
        ext = Path(output).suffix.lstrip(".")
        fmt = ext if ext else "png"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".dot", delete=False) as f:
        f.write(dot_str)
        dot_path = f.name

    try:
        result = subprocess.run(
            [engine, f"-T{fmt}", "-o", output, dot_path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Graphviz render failed: {result.stderr}")
    finally:
        Path(dot_path).unlink(missing_ok=True)

    return output


def render_comparison(
    graph: DaguaGraph,
    dagua_positions: torch.Tensor,
    graphviz_positions: torch.Tensor,
    output: str,
    config=None,
    dpi: int = 150,
) -> str:
    """Render side-by-side comparison: Dagua layout vs Graphviz layout.

    Both use the same matplotlib renderer for fair visual comparison.
    Returns path to the output file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from dagua.render.mpl import render as mpl_render
    from dagua.metrics import compute_all_metrics

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Render Dagua layout
    _render_on_axes(ax1, graph, dagua_positions, "Dagua")

    # Render Graphviz layout
    _render_on_axes(ax2, graph, graphviz_positions, "Graphviz (dot)")

    # Compute metrics for subtitle
    graph.compute_node_sizes()
    dagua_m = compute_all_metrics(dagua_positions, graph.edge_index, graph.node_sizes)
    gv_m = compute_all_metrics(graphviz_positions, graph.edge_index, graph.node_sizes)

    ax1.set_title(
        f"Dagua\ncrossings={dagua_m['edge_crossings']}, "
        f"overlaps={dagua_m['node_overlaps']}, "
        f"dag%={dagua_m['dag_fraction']:.2f}, "
        f"quality={dagua_m['overall_quality']:.1f}",
        fontsize=10,
    )
    ax2.set_title(
        f"Graphviz (dot)\ncrossings={gv_m['edge_crossings']}, "
        f"overlaps={gv_m['node_overlaps']}, "
        f"dag%={gv_m['dag_fraction']:.2f}, "
        f"quality={gv_m['overall_quality']:.1f}",
        fontsize=10,
    )

    plt.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return output


def _render_on_axes(ax, graph, positions, title: str):
    """Render a graph layout onto a specific matplotlib axes."""
    from dagua.edges import route_edges
    from dagua.render.mpl import _draw_clusters, _draw_edges, _draw_nodes, _draw_node_labels

    pos = positions.detach().cpu().numpy()
    graph.compute_node_sizes()
    sizes = graph.node_sizes.detach().cpu().numpy()

    n = graph.num_nodes
    if n == 0:
        return

    x_min = (pos[:, 0] - sizes[:, 0] / 2).min() - 30
    x_max = (pos[:, 0] + sizes[:, 0] / 2).max() + 30
    y_min = (pos[:, 1] - sizes[:, 1] / 2).min() - 30
    y_max = (pos[:, 1] + sizes[:, 1] / 2).max() + 30

    _draw_clusters(ax, graph, pos, sizes)

    curves = route_edges(positions, graph.edge_index, graph.node_sizes, graph.direction)
    _draw_edges(ax, graph, curves)
    _draw_nodes(ax, graph, pos, sizes)
    _draw_node_labels(ax, graph, pos, sizes)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.axis("off")


def render_comparison_native(
    graph: DaguaGraph,
    dagua_output: str,
    graphviz_output: str,
    combined_output: str,
    dpi: int = 150,
) -> str:
    """Combine pre-rendered Dagua and Graphviz images side-by-side.

    Uses PIL to concatenate two images horizontally.
    Returns path to the combined output.
    """
    from PIL import Image

    img1 = Image.open(dagua_output)
    img2 = Image.open(graphviz_output)

    # Resize to same height
    h = max(img1.height, img2.height)
    if img1.height != h:
        ratio = h / img1.height
        img1 = img1.resize((int(img1.width * ratio), h))
    if img2.height != h:
        ratio = h / img2.height
        img2 = img2.resize((int(img2.width * ratio), h))

    combined = Image.new("RGB", (img1.width + img2.width + 20, h), "white")
    combined.paste(img1, (0, 0))
    combined.paste(img2, (img1.width + 20, 0))
    combined.save(combined_output, dpi=(dpi, dpi))

    return combined_output
