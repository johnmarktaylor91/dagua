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


def generate_multi_comparison_grid(
    graphs: Optional[List[TestGraph]] = None,
    competitors: Optional[List[str]] = None,
    config: Optional[LayoutConfig] = None,
    output_dir: str = "eval_output/grids",
    dpi: int = 100,
) -> str:
    """Generate a grid comparing N engines for each test graph.

    Rows = test graphs, columns = competitors.
    Each cell: rendered graph with metrics in title.

    Args:
        graphs: Test graphs. If None, uses all test graphs.
        competitors: Engine names to include. If None, uses all available.
        config: LayoutConfig for Dagua competitor.
        output_dir: Output directory.
        dpi: Output DPI.

    Returns:
        Path to the grid image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from dagua.eval.competitors import get_available_competitors
    from dagua.metrics import compute_all_metrics

    if graphs is None:
        graphs = get_test_graphs(max_nodes=200)
    if config is None:
        config = LayoutConfig()

    available = get_available_competitors()
    if competitors is not None:
        comp_set = set(competitors)
        available = [c for c in available if c.name in comp_set]

    if not available:
        raise RuntimeError("No competitors available")

    n_rows = len(graphs)
    n_cols = len(available)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [list(axes)]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    for i, tg in enumerate(graphs):
        tg.graph.compute_node_sizes()

        for j, comp in enumerate(available):
            ax = axes[i][j]
            try:
                if tg.graph.num_nodes > comp.max_nodes:
                    ax.text(0.5, 0.5, "Too large", ha="center", va="center",
                            transform=ax.transAxes, fontsize=8)
                    ax.set_title(f"{comp.name}: {tg.name}", fontsize=7)
                else:
                    result = comp.layout(tg.graph)
                    if result.pos is not None:
                        _render_on_ax(ax, tg.graph, result.pos)
                        m = compute_all_metrics(
                            result.pos, tg.graph.edge_index, tg.graph.node_sizes
                        )
                        ax.set_title(
                            f"{comp.name}: {tg.name}\n"
                            f"Q={m['overall_quality']:.0f} "
                            f"X={m['edge_crossings']} O={m['node_overlaps']}",
                            fontsize=7,
                        )
                    else:
                        ax.text(0.5, 0.5, f"Error:\n{result.error}", ha="center",
                                va="center", transform=ax.transAxes, fontsize=7)
                        ax.set_title(f"{comp.name}: {tg.name}", fontsize=7)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error:\n{e}", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7)
                ax.set_title(f"{comp.name}: {tg.name}", fontsize=7)
            ax.axis("off")

    plt.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = str(Path(output_dir) / "multi_comparison_grid.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Multi-comparison grid saved to {path}")
    return path


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


def generate_benchmark_markdown(results: list, output_path: str) -> str:
    """Generate a GitHub-viewable markdown report from benchmark results.

    Args:
        results: List of BenchmarkResult dataclass instances.
        output_path: Path for the output .md file.

    Returns:
        Path to the generated file.
    """
    import platform
    from collections import defaultdict
    from datetime import datetime

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Group results by graph size tier
    def _tier_label(n_nodes: int) -> str:
        if n_nodes <= 2_000:
            return "small"
        elif n_nodes <= 50_000:
            return "medium"
        elif n_nodes <= 1_000_000:
            return "large"
        else:
            return "huge"

    _TIER_HEADINGS = {
        "small": "Small Graphs (up to 2K nodes)",
        "medium": "Medium Graphs (5K-50K nodes)",
        "large": "Large Graphs (100K-1M nodes)",
        "huge": "Huge Graphs (5M+ nodes)",
    }

    # Gather data
    by_graph: Dict[str, dict] = defaultdict(dict)
    all_competitors: set = set()
    for r in results:
        by_graph[r.graph_name][r.competitor] = r
        all_competitors.add(r.competitor)

    competitor_order = sorted(all_competitors)

    # Compute summary stats per competitor
    comp_stats: Dict[str, dict] = {}
    for comp in competitor_order:
        scores: List[float] = []
        runtimes: List[float] = []
        max_n = 0
        wins = 0
        for gname, comp_results in by_graph.items():
            if comp not in comp_results:
                continue
            r = comp_results[comp]
            if r.error:
                continue
            if r.composite_score is not None:
                scores.append(r.composite_score)
            runtimes.append(r.runtime_seconds)
            max_n = max(max_n, r.graph_nodes)

            # Check if this competitor won on this graph
            best_score = -1.0
            best_comp = ""
            for c2, r2 in comp_results.items():
                if r2.composite_score is not None and r2.composite_score > best_score:
                    best_score = r2.composite_score
                    best_comp = c2
            if best_comp == comp:
                wins += 1

        comp_stats[comp] = {
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "avg_runtime": sum(runtimes) / len(runtimes) if runtimes else 0.0,
            "max_nodes": max_n,
            "wins": wins,
            "n_graphs": len(runtimes),
        }

    # Build markdown
    lines: List[str] = []
    lines.append("# Dagua Competitive Benchmark Report\n")
    lines.append(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
        f"Platform: {platform.platform()} | "
        f"Graphs: {len(by_graph)}\n"
    )

    # Summary table
    lines.append("## Summary\n")
    lines.append(
        "| Competitor | Avg Score | Avg Runtime | Max Nodes | Wins | Graphs Run |"
    )
    lines.append("|---|---|---|---|---|---|")
    for comp in competitor_order:
        s = comp_stats[comp]
        lines.append(
            f"| {comp} | {s['avg_score']:.1f} | {s['avg_runtime']:.3f}s | "
            f"{s['max_nodes']:,} | {s['wins']}/{s['n_graphs']} | {s['n_graphs']} |"
        )
    lines.append("")

    # Per-tier tables
    by_tier: Dict[str, list] = defaultdict(list)
    graph_nodes: Dict[str, int] = {}
    for gname, comp_results in by_graph.items():
        sample = next(iter(comp_results.values()))
        tier = _tier_label(sample.graph_nodes)
        by_tier[tier].append(gname)
        graph_nodes[gname] = sample.graph_nodes

    for tier in ["small", "medium", "large", "huge"]:
        if tier not in by_tier:
            continue

        graph_names = sorted(by_tier[tier], key=lambda g: graph_nodes[g])
        heading = _TIER_HEADINGS.get(tier, tier.title())
        lines.append(f"## {heading}\n")

        # Determine which competitors actually ran in this tier
        tier_comps = []
        for comp in competitor_order:
            for gname in graph_names:
                if comp in by_graph[gname]:
                    tier_comps.append(comp)
                    break

        header = "| Graph | Nodes | Edges |"
        sep = "|---|---|---|"
        for comp in tier_comps:
            header += f" {comp} |"
            sep += "---|"
        lines.append(header)
        lines.append(sep)

        for gname in graph_names:
            comp_results = by_graph[gname]
            sample = next(iter(comp_results.values()))
            row = f"| {gname} | {sample.graph_nodes:,} | {sample.graph_edges:,} |"
            for comp in tier_comps:
                if comp not in comp_results:
                    row += " - |"
                else:
                    r = comp_results[comp]
                    if r.error:
                        row += f" {r.error[:15]} |"
                    elif r.composite_score is not None:
                        row += f" {r.composite_score:.1f} ({r.runtime_seconds:.2f}s) |"
                    else:
                        row += f" ({r.runtime_seconds:.2f}s) |"
            lines.append(row)
        lines.append("")

    # Key metrics breakdown (only if results have metrics data)
    has_metrics = any(
        r.metrics for r in results if r.metrics and "_metrics_error" not in r.metrics
    )
    if has_metrics:
        lines.append("## Key Metrics Breakdown\n")

        metric_keys = [
            ("dag_consistency", "DAG Consistency"),
            ("edge_length_cv", "Edge Length CV"),
            ("depth_correlation", "Depth-Position Correlation"),
            ("overlap_count", "Node Overlaps"),
            ("edge_straightness_mean_deg", "Edge Straightness (deg)"),
        ]

        for mk, label in metric_keys:
            lines.append(f"### {label}\n")
            lines.append(
                "| Graph |" + "".join(f" {c} |" for c in competitor_order)
            )
            lines.append("|---|" + "".join("---|" for _ in competitor_order))

            for gname in sorted(by_graph.keys()):
                comp_results = by_graph[gname]
                row = f"| {gname} |"
                for comp in competitor_order:
                    if comp not in comp_results:
                        row += " - |"
                    else:
                        r = comp_results[comp]
                        if r.metrics and mk in r.metrics:
                            v = r.metrics[mk]
                            if isinstance(v, float):
                                row += f" {v:.3f} |"
                            else:
                                row += f" {v} |"
                        else:
                            row += " - |"
                lines.append(row)
            lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Benchmark report saved to {output_path}")
    return output_path
