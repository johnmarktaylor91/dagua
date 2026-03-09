"""Report generation — HTML dashboards and PNG grids.

Generates visual reports for parameter sweep results and
Graphviz comparisons.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch

from dagua.config import LayoutConfig
from dagua.eval.graphs import TestGraph, get_test_graphs


def generate_grid(
    graphs: Optional[List[TestGraph]] = None,
    config: Optional[LayoutConfig] = None,
    output_dir: str = "eval_output/grids",
    cols: int = 4,
    dpi: int = 100,
) -> str:
    """Generate a grid of all test graph layouts.

    Returns path to the grid image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from dagua.layout import layout
    from dagua.render.mpl import _draw_clusters, _draw_edges, _draw_nodes, _draw_node_labels
    from dagua.edges import route_edges

    if graphs is None:
        graphs = get_test_graphs(max_nodes=200)
    if config is None:
        config = LayoutConfig()

    n = len(graphs)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for i, tg in enumerate(graphs):
        ax = axes[i]
        try:
            tg.graph.compute_node_sizes()
            pos = layout(tg.graph, config)
            pos_np = pos.detach().cpu().numpy()
            sizes = tg.graph.node_sizes.detach().cpu().numpy()

            x_min = (pos_np[:, 0] - sizes[:, 0] / 2).min() - 20
            x_max = (pos_np[:, 0] + sizes[:, 0] / 2).max() + 20
            y_min = (pos_np[:, 1] - sizes[:, 1] / 2).min() - 20
            y_max = (pos_np[:, 1] + sizes[:, 1] / 2).max() + 20

            _draw_clusters(ax, tg.graph, pos_np, sizes)
            curves = route_edges(pos, tg.graph.edge_index, tg.graph.node_sizes, tg.graph.direction)
            _draw_edges(ax, tg.graph, curves)
            _draw_nodes(ax, tg.graph, pos_np, sizes)
            _draw_node_labels(ax, tg.graph, pos_np, sizes)

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect("equal")
        except Exception as e:
            ax.text(0.5, 0.5, f"Error:\n{e}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)

        ax.set_title(f"{tg.name}\n({', '.join(sorted(tg.tags)[:2])})", fontsize=8)
        ax.axis("off")

    # Hide empty axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = str(Path(output_dir) / "test_graph_grid.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Grid saved to {path}")
    return path


def generate_comparison_grid(
    graphs: Optional[List[TestGraph]] = None,
    config: Optional[LayoutConfig] = None,
    output_dir: str = "eval_output/grids",
    dpi: int = 100,
) -> str:
    """Generate a grid comparing Dagua vs Graphviz for each test graph.

    Each row: [Dagua layout | Graphviz layout]
    Returns path to the grid image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from dagua.layout import layout
    from dagua.graphviz_utils import layout_with_graphviz
    from dagua.render.mpl import _draw_clusters, _draw_edges, _draw_nodes, _draw_node_labels
    from dagua.edges import route_edges
    from dagua.metrics import compute_all_metrics

    if graphs is None:
        graphs = get_test_graphs(max_nodes=200)
    if config is None:
        config = LayoutConfig()

    n = len(graphs)
    fig, axes = plt.subplots(n, 2, figsize=(14, n * 3.5))
    if n == 1:
        axes = [axes]

    for i, tg in enumerate(graphs):
        ax_dagua = axes[i][0] if n > 1 else axes[0]
        ax_gv = axes[i][1] if n > 1 else axes[1]

        tg.graph.compute_node_sizes()

        # Dagua
        try:
            dagua_pos = layout(tg.graph, config)
            _render_on_ax(ax_dagua, tg.graph, dagua_pos)
            dagua_m = compute_all_metrics(dagua_pos, tg.graph.edge_index, tg.graph.node_sizes)
            ax_dagua.set_title(
                f"Dagua: {tg.name}\nQ={dagua_m['overall_quality']:.0f} "
                f"X={dagua_m['edge_crossings']} O={dagua_m['node_overlaps']}",
                fontsize=8,
            )
        except Exception as e:
            ax_dagua.text(0.5, 0.5, str(e), ha="center", va="center",
                         transform=ax_dagua.transAxes, fontsize=8)
            ax_dagua.set_title(f"Dagua: {tg.name} (error)", fontsize=8)

        # Graphviz
        try:
            gv_pos = layout_with_graphviz(tg.graph)
            _render_on_ax(ax_gv, tg.graph, gv_pos)
            gv_m = compute_all_metrics(gv_pos, tg.graph.edge_index, tg.graph.node_sizes)
            ax_gv.set_title(
                f"Graphviz: {tg.name}\nQ={gv_m['overall_quality']:.0f} "
                f"X={gv_m['edge_crossings']} O={gv_m['node_overlaps']}",
                fontsize=8,
            )
        except Exception as e:
            ax_gv.text(0.5, 0.5, str(e), ha="center", va="center",
                       transform=ax_gv.transAxes, fontsize=8)
            ax_gv.set_title(f"Graphviz: {tg.name} (error)", fontsize=8)

        ax_dagua.axis("off")
        ax_gv.axis("off")

    plt.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = str(Path(output_dir) / "comparison_grid.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Comparison grid saved to {path}")
    return path


def _render_on_ax(ax, graph, positions):
    """Helper to render a graph on a matplotlib axes."""
    from dagua.render.mpl import _draw_clusters, _draw_edges, _draw_nodes, _draw_node_labels
    from dagua.edges import route_edges

    pos_np = positions.detach().cpu().numpy()
    sizes = graph.node_sizes.detach().cpu().numpy()

    x_min = (pos_np[:, 0] - sizes[:, 0] / 2).min() - 20
    x_max = (pos_np[:, 0] + sizes[:, 0] / 2).max() + 20
    y_min = (pos_np[:, 1] - sizes[:, 1] / 2).min() - 20
    y_max = (pos_np[:, 1] + sizes[:, 1] / 2).max() + 20

    _draw_clusters(ax, graph, pos_np, sizes)
    curves = route_edges(positions, graph.edge_index, graph.node_sizes, graph.direction)
    _draw_edges(ax, graph, curves)
    _draw_nodes(ax, graph, pos_np, sizes)
    _draw_node_labels(ax, graph, pos_np, sizes)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")


def generate_html_dashboard(
    comparison_results: list,
    sweep_results: list,
    output_dir: str = "eval_output",
) -> str:
    """Generate an HTML dashboard summarizing all evaluation results.

    Returns path to the HTML file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = str(Path(output_dir) / "dashboard.html")

    html = ['<!DOCTYPE html><html><head>']
    html.append('<meta charset="utf-8">')
    html.append('<title>Dagua Evaluation Dashboard</title>')
    html.append('<style>')
    html.append('body { font-family: -apple-system, sans-serif; margin: 20px; }')
    html.append('table { border-collapse: collapse; margin: 10px 0; }')
    html.append('th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: right; }')
    html.append('th { background: #f5f5f5; }')
    html.append('.better { color: green; font-weight: bold; }')
    html.append('.worse { color: red; }')
    html.append('h1,h2,h3 { color: #333; }')
    html.append('</style></head><body>')

    html.append('<h1>Dagua Evaluation Dashboard</h1>')

    # Comparison table
    if comparison_results:
        html.append('<h2>Dagua vs Graphviz</h2>')
        html.append('<table><tr><th>Graph</th><th>Nodes</th>')
        html.append('<th>Dagua Quality</th><th>Graphviz Quality</th>')
        html.append('<th>Dagua Crossings</th><th>GV Crossings</th>')
        html.append('<th>Dagua Overlaps</th><th>GV Overlaps</th>')
        html.append('<th>Winner</th></tr>')

        for r in comparison_results:
            dq = r.dagua_metrics.get("overall_quality", 0)
            gq = r.graphviz_metrics.get("overall_quality", 0)
            dc = r.dagua_metrics.get("edge_crossings", 0)
            gc = r.graphviz_metrics.get("edge_crossings", 0)
            do_ = r.dagua_metrics.get("node_overlaps", 0)
            go = r.graphviz_metrics.get("node_overlaps", 0)
            nn = r.dagua_metrics.get("num_nodes", 0)
            winner = "Dagua" if r.dagua_better else "Graphviz"
            wclass = "better" if r.dagua_better else "worse"

            html.append(f'<tr><td style="text-align:left">{r.graph_name}</td>')
            html.append(f'<td>{nn}</td>')
            html.append(f'<td>{dq:.1f}</td><td>{gq:.1f}</td>')
            html.append(f'<td>{dc}</td><td>{gc}</td>')
            html.append(f'<td>{do_}</td><td>{go}</td>')
            html.append(f'<td class="{wclass}">{winner}</td></tr>')

        wins = sum(1 for r in comparison_results if r.dagua_better)
        html.append(f'</table><p><b>Dagua wins: {wins}/{len(comparison_results)}</b></p>')

    # Sweep summary
    if sweep_results:
        html.append('<h2>Parameter Sweep Summary</h2>')
        from collections import defaultdict
        by_param = defaultdict(list)
        for r in sweep_results:
            by_param[r.param_name].append(r)

        html.append('<table><tr><th>Parameter</th><th>Best Value</th>')
        html.append('<th>Best Quality</th><th>Worst Value</th><th>Worst Quality</th></tr>')

        for param, rs in sorted(by_param.items()):
            by_value = defaultdict(list)
            for r in rs:
                by_value[r.param_value].append(r.quality)
            avg_quality = {v: sum(qs) / len(qs) for v, qs in by_value.items()}
            best_val = max(avg_quality, key=avg_quality.get)
            worst_val = min(avg_quality, key=avg_quality.get)
            html.append(f'<tr><td style="text-align:left">{param}</td>')
            html.append(f'<td>{best_val}</td><td>{avg_quality[best_val]:.1f}</td>')
            html.append(f'<td>{worst_val}</td><td>{avg_quality[worst_val]:.1f}</td></tr>')

        html.append('</table>')

    # Images
    html.append('<h2>Visual Comparisons</h2>')
    grid_path = Path(output_dir) / "grids" / "comparison_grid.png"
    if grid_path.exists():
        html.append(f'<img src="grids/comparison_grid.png" style="max-width:100%">')
    test_grid = Path(output_dir) / "grids" / "test_graph_grid.png"
    if test_grid.exists():
        html.append(f'<img src="grids/test_graph_grid.png" style="max-width:100%">')

    html.append('</body></html>')

    with open(path, "w") as f:
        f.write("\n".join(html))

    print(f"Dashboard saved to {path}")
    return path
