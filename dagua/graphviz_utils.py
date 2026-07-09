"""Graphviz comparison utilities.

Provides functions to:
1. Layout a DaguaGraph using Graphviz's dot engine
2. Render using Graphviz's native renderer
3. Produce side-by-side comparison images (Dagua vs Graphviz)
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, cast

import torch

_CLUSTER_NAME_PATTERN = re.compile(r"[^0-9A-Za-z_]")

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from dagua.config import LayoutConfig
    from dagua.graph import DaguaGraph


def to_dot(
    graph: DaguaGraph,
    positions: Optional[torch.Tensor] = None,
    node_sizes: Optional[torch.Tensor] = None,
) -> str:
    """Convert a graph to DOT source.

    If positions are provided, they are emitted as ``pos="x,y!"`` attributes
    so Graphviz respects the layout in ``neato -n2`` mode. This exporter
    intentionally maps Dagua's richer style space down to a small DOT subset.
    Use it for placement and comparison workflows rather than full-fidelity
    round-tripping.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to export.
    positions : torch.Tensor | None, optional
        Optional node positions with shape ``[N, 2]`` in point units.
    node_sizes : torch.Tensor | None, optional
        Optional per-node ``[N, 2]`` width/height in point units (dagua's
        native coordinate scale). When provided, each node is emitted with
        ``width``/``height`` (converted to Graphviz's inch convention) and
        ``fixedsize=true`` so Graphviz lays out the node at its real size
        instead of auto-sizing to the label text. ``None`` (the default)
        preserves the prior size-blind DOT output exactly -- existing
        callers of this function are unaffected unless they opt in.

    Returns
    -------
    str
        DOT source for the graph.
    """
    lines = ["digraph G {"]
    lines.append("  rankdir=TB;")
    lines.append('  node [shape=box, style=filled, fontname="Helvetica"];')
    lines.append('  edge [fontname="Helvetica", fontsize=9];')

    if positions is not None:
        pos = positions.detach().cpu()
    if node_sizes is not None:
        sizes = node_sizes.detach().cpu()

    for i in range(graph.num_nodes):
        label = graph.node_labels[i].replace("\\", "\\\\").replace('"', '\\"')
        style = graph.get_style_for_node(i)
        attrs = [
            f'label="{label}"',
            f'fillcolor="{style.fill}"',
            f'fontcolor="{style.font_color}"',
            f"fontsize={style.font_size}",
        ]
        if style.shape == "ellipse":
            attrs.append("shape=ellipse")
        elif style.shape == "circle":
            attrs.append("shape=circle")
        elif style.shape == "diamond":
            attrs.append("shape=diamond")
        else:
            attrs.append("shape=box")
            if style.shape == "roundrect":
                attrs.append('style="filled,rounded"')

        if positions is not None:
            # Graphviz uses inches, 72 points/inch. Flip y for Graphviz convention.
            x_inch = pos[i, 0].item() / 72.0
            y_inch = -pos[i, 1].item() / 72.0  # flip y
            attrs.append(f'pos="{x_inch},{y_inch}!"')

        if node_sizes is not None:
            # Graphviz uses inches, 72 points/inch (same convention as pos above).
            width_in = max(float(sizes[i, 0].item()) / 72.0, 0.01)
            height_in = max(float(sizes[i, 1].item()) / 72.0, 0.01)
            attrs.append(f"width={width_in:.4f}")
            attrs.append(f"height={height_in:.4f}")
            attrs.append("fixedsize=true")

        attrs_str = ", ".join(attrs)
        lines.append(f"  n{i} [{attrs_str}];")

    # Edges
    if graph.edge_index.numel() > 0:
        for e in range(graph.edge_index.shape[1]):
            s = int(graph.edge_index[0, e].item())
            t = int(graph.edge_index[1, e].item())
            edge_attrs: list[str] = []
            if e < len(graph.edge_labels) and graph.edge_labels[e]:
                lbl = str(graph.edge_labels[e]).replace("\\", "\\\\").replace('"', '\\"')
                edge_attrs.append(f'label="{lbl}"')
            edge_style = graph.get_style_for_edge(e)
            edge_attrs.append(f'color="{edge_style.color}"')
            if edge_style.style == "dashed":
                edge_attrs.append("style=dashed")
            elif edge_style.style == "dotted":
                edge_attrs.append("style=dotted")
            attrs_str = ", ".join(edge_attrs) if edge_attrs else ""
            lines.append(f"  n{s} -> n{t} [{attrs_str}];")

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
            cluster_label = (
                graph.cluster_labels.get(name, name).replace("\\", "\\\\").replace('"', '\\"')
            )
            safe_id = _CLUSTER_NAME_PATTERN.sub("_", name)
            lines.append(f"  subgraph cluster_{safe_id} {{")
            lines.append(f'    label="{cluster_label}";')
            lines.append('    style=filled; color=lightgrey; fillcolor="#f0f0f0";')
            for idx in indices:
                lines.append(f"    n{idx};")
            lines.append("  }")

    lines.append("}")
    return "\n".join(lines)


def layout_with_graphviz(
    graph: DaguaGraph,
    engine: str = "dot",
    timeout: float = 300.0,
) -> torch.Tensor:
    """Layout a graph with Graphviz and parse the resulting node positions.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to lay out.
    engine : str, default="dot"
        Graphviz engine executable to invoke.
    timeout : float, default=300.0
        Maximum subprocess runtime in seconds.

    Returns
    -------
    torch.Tensor
        Node positions with shape ``[N, 2]`` in graph node order.
    """
    dot_str = to_dot(graph)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".dot", delete=False) as f:
        f.write(dot_str)
        dot_path = f.name

    try:
        result = subprocess.run(
            [engine, "-Tjson", dot_path],
            capture_output=True,
            text=True,
            timeout=timeout,
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
    timeout: float = 300.0,
) -> str:
    """Render a graph using Graphviz's native renderer.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to render.
    output : str
        Output file path.
    engine : str, default="dot"
        Graphviz engine executable to invoke.
    fmt : str | None, optional
        Explicit Graphviz output format. When omitted, it is inferred from
        ``output``.
    timeout : float, default=300.0
        Maximum subprocess runtime in seconds.

    Returns
    -------
    str
        Output path.
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
            capture_output=True,
            text=True,
            timeout=timeout,
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
    config: Optional[LayoutConfig] = None,
    dpi: int = 150,
) -> str:
    """Render side-by-side comparison: Dagua layout vs Graphviz layout.

    Both use the same matplotlib renderer for fair visual comparison. That means
    this helper isolates node-placement differences and intentionally discards
    Graphviz-native text/edge/cluster geometry choices.

    Returns path to the output file.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from dagua.metrics import compute_all_metrics

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Render Dagua layout
    _render_on_axes(ax1, graph, dagua_positions, "Dagua")

    # Render Graphviz layout
    _render_on_axes(ax2, graph, graphviz_positions, "Graphviz (dot)")

    # Compute metrics for subtitle
    graph.compute_node_sizes()
    assert graph.node_sizes is not None
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


def _render_on_axes(
    ax: Axes,
    graph: DaguaGraph,
    positions: torch.Tensor,
    title: str,
) -> None:
    """Render a graph layout onto a specific matplotlib axes.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to draw into.
    graph : DaguaGraph
        Graph being rendered.
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    title : str
        Title for the axes.

    Returns
    -------
    None
        The function draws directly onto ``ax``.
    """
    from dagua.edges import route_edges
    from dagua.render.mpl import _draw_clusters, _draw_edges, _draw_node_labels, _draw_nodes

    pos = positions.detach().cpu().numpy()
    graph.compute_node_sizes()
    assert graph.node_sizes is not None
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


def render_multi_comparison(
    graph: DaguaGraph,
    positions_dict: Dict[str, torch.Tensor],
    output: str,
    config: Optional[LayoutConfig] = None,
    dpi: int = 150,
) -> str:
    """Render N engine layouts side-by-side for the same graph.

    Args:
        graph: The DaguaGraph to render.
        positions_dict: Mapping of engine_name -> [N, 2] position tensor.
        output: Path for the output image.
        config: Optional LayoutConfig (unused currently, reserved for future).
        dpi: Output DPI.

    Returns:
        Path to the output file.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from dagua.metrics import compute_all_metrics

    n_engines = len(positions_dict)
    if n_engines == 0:
        raise ValueError("positions_dict must contain at least one engine")

    fig, axes = plt.subplots(1, n_engines, figsize=(10 * n_engines, 10))
    if n_engines == 1:
        axes = [axes]

    graph.compute_node_sizes()
    assert graph.node_sizes is not None

    for ax, (engine_name, pos) in zip(axes, positions_dict.items()):
        try:
            _render_on_axes(ax, graph, pos, engine_name)
            metrics = compute_all_metrics(pos, graph.edge_index, graph.node_sizes)
            ax.set_title(
                f"{engine_name}\ncrossings={metrics['edge_crossings']}, "
                f"overlaps={metrics['node_overlaps']}, "
                f"dag%={metrics['dag_fraction']:.2f}, "
                f"quality={metrics['overall_quality']:.1f}",
                fontsize=10,
            )
        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"{engine_name}\nError: {e}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
            )
            ax.axis("off")

    plt.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return output


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

    img1 = cast(Any, Image.open(dagua_output))
    img2 = cast(Any, Image.open(graphviz_output))

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
