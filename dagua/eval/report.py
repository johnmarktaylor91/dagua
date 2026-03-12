"""Report generation — HTML dashboards and PNG grids.

Generates visual reports for parameter sweep results and
Graphviz comparisons.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from statistics import mean
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import torch

from dagua.config import LayoutConfig
from dagua.eval.graphs import TestGraph, get_test_graphs
from dagua.metrics import compare


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _require_node_sizes(graph: Any) -> torch.Tensor:
    """Return graph node sizes after forcing lazy computation."""
    graph.compute_node_sizes()
    if graph.node_sizes is None:
        raise ValueError("graph.node_sizes unavailable after compute_node_sizes()")
    return graph.node_sizes


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
            pos = layout(tg.graph, config)
            pos_np = pos.detach().cpu().numpy()
            sizes_t = _require_node_sizes(tg.graph)
            sizes = sizes_t.detach().cpu().numpy()

            x_min = (pos_np[:, 0] - sizes[:, 0] / 2).min() - 20
            x_max = (pos_np[:, 0] + sizes[:, 0] / 2).max() + 20
            y_min = (pos_np[:, 1] - sizes[:, 1] / 2).min() - 20
            y_max = (pos_np[:, 1] + sizes[:, 1] / 2).max() + 20

            _draw_clusters(ax, tg.graph, pos_np, sizes)
            curves = route_edges(pos, tg.graph.edge_index, sizes_t, tg.graph.direction)
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

        sizes_t = _require_node_sizes(tg.graph)

        # Dagua
        try:
            dagua_pos = layout(tg.graph, config)
            _render_on_ax(ax_dagua, tg.graph, dagua_pos)
            dagua_m = compute_all_metrics(dagua_pos, tg.graph.edge_index, sizes_t)
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
            gv_m = compute_all_metrics(gv_pos, tg.graph.edge_index, sizes_t)
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
    sizes_t = _require_node_sizes(graph)
    sizes = sizes_t.detach().cpu().numpy()

    x_min = (pos_np[:, 0] - sizes[:, 0] / 2).min() - 20
    x_max = (pos_np[:, 0] + sizes[:, 0] / 2).max() + 20
    y_min = (pos_np[:, 1] - sizes[:, 1] / 2).min() - 20
    y_max = (pos_np[:, 1] + sizes[:, 1] / 2).max() + 20

    _draw_clusters(ax, graph, pos_np, sizes)
    curves = route_edges(positions, graph.edge_index, sizes_t, graph.direction)
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
                        assert tg.graph.node_sizes is not None
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
            best_val = max(avg_quality, key=lambda value: avg_quality[value])
            worst_val = min(avg_quality, key=lambda value: avg_quality[value])
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


def _benchmark_graph_lookup() -> Dict[str, TestGraph]:
    from dagua.eval.benchmark import get_standard_suite_graphs

    lookup: Dict[str, TestGraph] = {}
    for bg in list(get_standard_suite_graphs()):
        lookup[bg.test_graph.name] = bg.test_graph
    return lookup


def _resolve_positions_path(results_root: Path, positions_path: str) -> Path:
    path = Path(positions_path)
    return path if path.is_absolute() else results_root / path


def _normalize_positions(
    positions: torch.Tensor,
    node_sizes: torch.Tensor,
    target_width: float = 600.0,
    target_height: float = 420.0,
    padding: float = 30.0,
) -> torch.Tensor:
    pos = positions.detach().cpu().clone()
    sizes = node_sizes.detach().cpu()
    x_min = (pos[:, 0] - sizes[:, 0] / 2).min()
    x_max = (pos[:, 0] + sizes[:, 0] / 2).max()
    y_min = (pos[:, 1] - sizes[:, 1] / 2).min()
    y_max = (pos[:, 1] + sizes[:, 1] / 2).max()
    width = max((x_max - x_min).item(), 1.0)
    height = max((y_max - y_min).item(), 1.0)
    scale = min((target_width - 2 * padding) / width, (target_height - 2 * padding) / height)
    pos[:, 0] = (pos[:, 0] - x_min) * scale + padding
    pos[:, 1] = (pos[:, 1] - y_min) * scale + padding
    return pos


def _status_panel(ax, title: str, subtitle: str) -> None:
    ax.set_facecolor("#F3F4F6")
    ax.text(0.5, 0.58, title, ha="center", va="center", transform=ax.transAxes, fontsize=11, color="#374151")
    ax.text(0.5, 0.42, subtitle, ha="center", va="center", transform=ax.transAxes, fontsize=8, color="#6B7280")
    ax.axis("off")


def _metric_from_result(result: Dict[str, Any], *keys: str) -> Optional[float]:
    metrics = result.get("metrics", {})
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def generate_comparison_visuals(
    output_dir: str = "eval_output",
    combined_results: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Render per-graph comparison grids from stored benchmark positions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from dagua.eval.benchmark import DEFAULT_COMPETITOR_ORDER

    if combined_results is None:
        from dagua.eval.benchmark import load_combined_results

        combined_results = load_combined_results(output_dir)

    graph_lookup = _benchmark_graph_lookup()
    root = Path(output_dir)
    comparison_dir = root / "visuals" / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: List[str] = []

    for graph_name, graph_payload in combined_results.get("graphs", {}).items():
        if graph_payload.get("n_nodes", 0) > 2_000:
            continue
        if graph_name not in graph_lookup:
            continue

        tg = graph_lookup[graph_name]
        tg.graph.compute_node_sizes()
        competitors = graph_payload.get("competitors", {})
        ordered = [name for name in DEFAULT_COMPETITOR_ORDER if name in competitors]
        if not ordered:
            continue

        cols = min(3, max(1, len(ordered)))
        rows = (len(ordered) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.8, rows * 3.8))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = list(axes)
        else:
            axes = [ax for row in axes for ax in row]

        for idx, comp_name in enumerate(ordered):
            ax = axes[idx]
            result = competitors[comp_name]
            if result.get("status") != "OK" or not result.get("positions_path"):
                _status_panel(ax, comp_name, result.get("reason", result.get("status", "N/A")))
                continue

            standard_id = combined_results.get("generated_from", {}).get("standard")
            rare_id = combined_results.get("generated_from", {}).get("rare")
            positions_root = Path(output_dir) / "benchmark_db" / "standard" / standard_id if standard_id else None
            if graph_name.startswith("scale_") and graph_payload.get("n_nodes", 0) > 100_000 and rare_id is not None:
                positions_root = Path(output_dir) / "benchmark_db" / "rare" / rare_id
            if positions_root is None:
                _status_panel(ax, comp_name, "missing positions root")
                continue

            pos = torch.load(_resolve_positions_path(positions_root, result["positions_path"]))
            pos = _normalize_positions(pos, _require_node_sizes(tg.graph))
            _render_on_ax(ax, tg.graph, pos)
            runtime = result.get("runtime_seconds")
            score = result.get("composite_score")
            ax.set_title(
                f"{comp_name}\n{runtime:.2f}s, score={score:.1f}" if runtime is not None and score is not None else comp_name,
                fontsize=8,
            )
            ax.axis("off")

        for ax in axes[len(ordered):]:
            ax.axis("off")

        fig.suptitle(f"{graph_name} ({graph_payload.get('n_nodes', 0):,} nodes)", fontsize=12)
        plt.tight_layout()
        out_path = comparison_dir / f"{graph_name}_comparison.png"
        fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        figure_paths.append(str(out_path))

    return figure_paths


def generate_scaling_curve(
    output_dir: str = "eval_output",
    combined_results: Optional[Dict[str, Any]] = None,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from dagua.eval.benchmark import DEFAULT_COMPETITOR_ORDER

    if combined_results is None:
        from dagua.eval.benchmark import load_combined_results

        combined_results = load_combined_results(output_dir)

    points: Dict[str, List[Tuple[int, float]]] = {name: [] for name in DEFAULT_COMPETITOR_ORDER}
    for graph_payload in combined_results.get("graphs", {}).values():
        n_nodes = graph_payload.get("n_nodes", 0)
        for comp_name, result in graph_payload.get("competitors", {}).items():
            if result.get("status") == "OK" and result.get("runtime_seconds") is not None:
                points.setdefault(comp_name, []).append((n_nodes, float(result["runtime_seconds"])))

    palette = {
        "dagua": "#0072B2",
        "graphviz_dot": "#D55E00",
        "graphviz_sfdp": "#009E73",
        "elk_layered": "#E69F00",
        "dagre": "#CC79A7",
        "nx_spring": "#6B7280",
    }

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for comp_name in DEFAULT_COMPETITOR_ORDER:
        if not points.get(comp_name):
            continue
        series = sorted(points[comp_name])
        xs = [x for x, _ in series]
        ys = [y for _, y in series]
        ax.plot(xs, ys, marker="o", linewidth=2.0, markersize=5, label=comp_name, color=palette.get(comp_name))
        if comp_name == "dagua":
            ax.annotate(f"{xs[-1]:,}", (xs[-1], ys[-1]), textcoords="offset points", xytext=(6, -4), fontsize=8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Scaling Curve")
    ax.grid(True, which="both", color="#E5E7EB", linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()

    out_path = Path(output_dir) / "scaling_curve.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(out_path)


def generate_layout_similarity_artifacts(
    output_dir: str = "eval_output",
    combined_results: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """Compute pairwise layout similarity across competitors from stored positions."""
    from dagua.eval.benchmark import DEFAULT_COMPETITOR_ORDER

    if combined_results is None:
        from dagua.eval.benchmark import load_combined_results

        combined_results = load_combined_results(output_dir)

    graph_lookup = _benchmark_graph_lookup()
    report_dir = Path(output_dir) / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    standard_id = combined_results.get("generated_from", {}).get("standard")
    rare_id = combined_results.get("generated_from", {}).get("rare")
    standard_root = Path(output_dir) / "benchmark_db" / "standard" / standard_id if standard_id else None
    rare_root = Path(output_dir) / "benchmark_db" / "rare" / rare_id if rare_id else None

    payload: Dict[str, Any] = {
        "generated_from": combined_results.get("generated_from", {}),
        "graphs": {},
        "aggregate": {
            "pairwise_mean_disparity": {},
            "graphs_compared": 0,
        },
    }
    pairwise_buckets: Dict[Tuple[str, str], List[float]] = {}

    for graph_name, graph_payload in combined_results.get("graphs", {}).items():
        if graph_name not in graph_lookup:
            continue
        tg = graph_lookup[graph_name]
        competitors = graph_payload.get("competitors", {})
        valid_positions: Dict[str, torch.Tensor] = {}
        ordered = [name for name in DEFAULT_COMPETITOR_ORDER if name in competitors]
        for comp_name in ordered:
            result = competitors[comp_name]
            if result.get("status") != "OK" or not result.get("positions_path"):
                continue
            root = standard_root
            if graph_payload.get("n_nodes", 0) > 100_000 and rare_root is not None and graph_name.startswith("scale_"):
                root = rare_root
            if root is None:
                continue
            pos_path = _resolve_positions_path(root, result["positions_path"])
            if not pos_path.exists():
                continue
            valid_positions[comp_name] = torch.load(pos_path)

        if len(valid_positions) < 2:
            continue

        matrix: Dict[str, Dict[str, Optional[float]]] = {name: {} for name in valid_positions}
        for a in valid_positions:
            for b in valid_positions:
                if a == b:
                    matrix[a][b] = 0.0
                    continue
                pair = cast(Tuple[str, str], tuple(sorted((a, b))))
                disparity = float(compare(valid_positions[a], valid_positions[b]).get("procrustes_disparity", 1.0))
                matrix[a][b] = disparity
                pairwise_buckets.setdefault(pair, []).append(disparity)

        payload["graphs"][graph_name] = {
            "n_nodes": graph_payload.get("n_nodes", 0),
            "n_edges": graph_payload.get("n_edges", 0),
            "competitors": list(valid_positions.keys()),
            "pairwise_procrustes_disparity": matrix,
        }

    aggregate_pairs = {
        f"{a}__vs__{b}": {
            "mean_procrustes_disparity": mean(values),
            "graphs_compared": len(values),
        }
        for (a, b), values in sorted(pairwise_buckets.items())
        if values
    }
    payload["aggregate"]["pairwise_mean_disparity"] = aggregate_pairs
    payload["aggregate"]["graphs_compared"] = len(payload["graphs"])

    json_path = report_dir / "layout_similarity.json"
    md_path = report_dir / "layout_similarity.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Layout Similarity",
        "",
        "Pairwise Procrustes disparity between competitor layouts for the same graph.",
        "Lower means more structurally similar after translation/rotation/scale alignment.",
        "",
        "## Aggregate",
        "",
    ]
    if aggregate_pairs:
        for pair_name, pair_payload in aggregate_pairs.items():
            lines.append(
                f"- `{pair_name}`: mean disparity `{pair_payload['mean_procrustes_disparity']:.4f}` "
                f"across `{pair_payload['graphs_compared']}` graphs"
            )
    else:
        lines.append("- No pairwise comparisons available.")
    lines.append("")
    lines.append("## Per Graph")
    lines.append("")
    for graph_name, graph_info in sorted(payload["graphs"].items()):
        lines.append(f"### {graph_name}")
        lines.append("")
        lines.append(f"- nodes: `{graph_info['n_nodes']}`")
        lines.append(f"- competitors: `{', '.join(graph_info['competitors'])}`")
        matrix = graph_info["pairwise_procrustes_disparity"]
        for row_name, row in matrix.items():
            rendered = ", ".join(f"{col}={val:.4f}" for col, val in row.items() if val is not None)
            lines.append(f"- `{row_name}`: {rendered}")
        lines.append("")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(json_path), str(md_path)


def generate_placement_summary_artifacts(
    output_dir: str = "eval_output",
    combined_results: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """Summarize placement-only quality across competitors from stored metrics."""
    from dagua.eval.benchmark import DEFAULT_COMPETITOR_ORDER

    if combined_results is None:
        from dagua.eval.benchmark import load_combined_results

        combined_results = load_combined_results(output_dir)

    report_dir = Path(output_dir) / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    placement_metrics = {
        "dag_consistency": ("dag_consistency",),
        "edge_crossings": ("edge_crossings",),
        "overlap_count": ("overlap_count", "node_overlaps"),
        "edge_length_cv": ("edge_length_cv",),
        "depth_correlation": ("depth_correlation",),
    }
    aggregate: Dict[str, Dict[str, Any]] = {}
    per_graph: Dict[str, Any] = {}

    for comp_name in DEFAULT_COMPETITOR_ORDER:
        aggregate[comp_name] = {
            "graphs_compared": 0,
            "mean_metrics": {},
        }

    buckets: Dict[str, Dict[str, List[float]]] = {
        comp_name: {metric_name: [] for metric_name in placement_metrics}
        for comp_name in DEFAULT_COMPETITOR_ORDER
    }

    for graph_name, graph_payload in combined_results.get("graphs", {}).items():
        graph_result = {
            "n_nodes": graph_payload.get("n_nodes", 0),
            "n_edges": graph_payload.get("n_edges", 0),
            "competitors": {},
        }
        for comp_name in DEFAULT_COMPETITOR_ORDER:
            result = graph_payload.get("competitors", {}).get(comp_name)
            if not result or result.get("status") != "OK":
                continue
            metrics_payload: Dict[str, Optional[float]] = {}
            for metric_name, keys in placement_metrics.items():
                value = _metric_from_result(result, *keys)
                metrics_payload[metric_name] = value
                if value is not None:
                    buckets[comp_name][metric_name].append(value)
            graph_result["competitors"][comp_name] = metrics_payload
        if graph_result["competitors"]:
            per_graph[graph_name] = graph_result

    for comp_name, metric_buckets in buckets.items():
        compared = 0
        for values in metric_buckets.values():
            compared = max(compared, len(values))
        aggregate[comp_name]["graphs_compared"] = compared
        aggregate[comp_name]["mean_metrics"] = {
            metric_name: (mean(values) if values else None)
            for metric_name, values in metric_buckets.items()
        }

    payload = {
        "generated_from": combined_results.get("generated_from", {}),
        "metric_definitions": {
            "dag_consistency": "Higher is better; ideal 1.0 for DAG-respecting layouts.",
            "edge_crossings": "Lower is better.",
            "overlap_count": "Lower is better.",
            "edge_length_cv": "Lower is better.",
            "depth_correlation": "Higher is better.",
        },
        "aggregate": aggregate,
        "graphs": per_graph,
    }

    json_path = report_dir / "placement_summary.json"
    md_path = report_dir / "placement_summary.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Placement Summary",
        "",
        "Placement-only benchmark view. This report intentionally ignores styling and focuses on node-placement metrics stored by the benchmark pipeline.",
        "",
        "## Aggregate",
        "",
    ]
    for comp_name in DEFAULT_COMPETITOR_ORDER:
        comp_payload = aggregate.get(comp_name, {})
        lines.append(f"### {comp_name}")
        lines.append("")
        lines.append(f"- graphs compared: `{comp_payload.get('graphs_compared', 0)}`")
        for metric_name, value in comp_payload.get("mean_metrics", {}).items():
            rendered = f"{value:.4f}" if value is not None else "N/A"
            lines.append(f"- mean {metric_name}: `{rendered}`")
        lines.append("")

    lines.append("## Per Graph")
    lines.append("")
    for graph_name, graph_info in sorted(per_graph.items()):
        lines.append(f"### {graph_name}")
        lines.append("")
        lines.append(f"- nodes: `{graph_info['n_nodes']}`")
        for comp_name, metric_payload in graph_info["competitors"].items():
            rendered = ", ".join(
                f"{metric}={value:.4f}" if value is not None else f"{metric}=N/A"
                for metric, value in metric_payload.items()
            )
            lines.append(f"- `{comp_name}`: {rendered}")
        lines.append("")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(json_path), str(md_path)


def generate_placement_dashboard_artifacts(
    output_dir: str = "eval_output",
    combined_results: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """Generate an iteration-oriented placement dashboard from stored metrics."""
    from dagua.eval.benchmark import DEFAULT_COMPETITOR_ORDER

    if combined_results is None:
        from dagua.eval.benchmark import load_combined_results

        combined_results = load_combined_results(output_dir)

    report_dir = Path(output_dir) / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "dag_consistency": ("dag_consistency", True),
        "edge_crossings": ("edge_crossings", False),
        "overlap_count": ("overlap_count", False),
        "edge_length_cv": ("edge_length_cv", False),
        "depth_correlation": ("depth_correlation", True),
    }

    payload: Dict[str, Any] = {
        "generated_from": combined_results.get("generated_from", {}),
        "aggregate": {
            "dagua_metric_wins": {metric: 0 for metric in metrics},
            "graphs_considered": 0,
            "mean_metric_delta_vs_best_non_dagua": {},
        },
        "graphs": {},
    }
    delta_buckets: Dict[str, List[float]] = {metric: [] for metric in metrics}

    for graph_name, graph_payload in combined_results.get("graphs", {}).items():
        competitors = graph_payload.get("competitors", {})
        if "dagua" not in competitors or competitors["dagua"].get("status") != "OK":
            continue
        dagua_result = competitors["dagua"]
        graph_entry = {
            "n_nodes": graph_payload.get("n_nodes", 0),
            "n_edges": graph_payload.get("n_edges", 0),
            "metrics": {},
        }
        any_metric = False
        for metric_name, (raw_key, higher_is_better) in metrics.items():
            dagua_value = _metric_from_result(
                dagua_result,
                raw_key,
                "node_overlaps" if raw_key == "overlap_count" else raw_key,
            )
            if dagua_value is None:
                continue
            best_other_name = None
            best_other_value = None
            for comp_name in DEFAULT_COMPETITOR_ORDER:
                if comp_name == "dagua":
                    continue
                result = competitors.get(comp_name)
                if not result or result.get("status") != "OK":
                    continue
                value = _metric_from_result(
                    result,
                    raw_key,
                    "node_overlaps" if raw_key == "overlap_count" else raw_key,
                )
                if value is None:
                    continue
                if best_other_value is None:
                    best_other_name, best_other_value = comp_name, value
                else:
                    if (higher_is_better and value > best_other_value) or (not higher_is_better and value < best_other_value):
                        best_other_name, best_other_value = comp_name, value
            delta = None
            dagua_wins = None
            if best_other_value is not None:
                delta = dagua_value - best_other_value if higher_is_better else best_other_value - dagua_value
                dagua_wins = delta >= 0
                delta_buckets[metric_name].append(delta)
                if dagua_wins:
                    payload["aggregate"]["dagua_metric_wins"][metric_name] += 1
            graph_entry["metrics"][metric_name] = {
                "dagua": dagua_value,
                "best_non_dagua_competitor": best_other_name,
                "best_non_dagua_value": best_other_value,
                "delta_vs_best_non_dagua": delta,
                "dagua_wins": dagua_wins,
            }
            any_metric = True
        if any_metric:
            payload["graphs"][graph_name] = graph_entry
            payload["aggregate"]["graphs_considered"] += 1

    payload["aggregate"]["mean_metric_delta_vs_best_non_dagua"] = {
        metric: (mean(values) if values else None)
        for metric, values in delta_buckets.items()
    }

    json_path = report_dir / "placement_dashboard.json"
    md_path = report_dir / "placement_dashboard.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Placement Dashboard",
        "",
        "Iteration-oriented placement summary. This is the fastest report to consult when the question is: how is Dagua's node placement doing, before worrying about rendering language?",
        "",
        "## Aggregate",
        "",
        f"- graphs considered: `{payload['aggregate']['graphs_considered']}`",
    ]
    for metric_name, wins in payload["aggregate"]["dagua_metric_wins"].items():
        total = payload["aggregate"]["graphs_considered"]
        mean_delta = payload["aggregate"]["mean_metric_delta_vs_best_non_dagua"].get(metric_name)
        mean_text = f"{mean_delta:.4f}" if mean_delta is not None else "N/A"
        lines.append(f"- `{metric_name}`: Dagua wins `{wins}/{total}` graphs, mean delta vs best non-Dagua `{mean_text}`")
    lines.append("")
    lines.append("## Per Graph")
    lines.append("")
    for graph_name, graph_info in sorted(payload["graphs"].items()):
        lines.append(f"### {graph_name}")
        lines.append("")
        lines.append(f"- nodes: `{graph_info['n_nodes']}`")
        for metric_name, metric_payload in graph_info["metrics"].items():
            lines.append(
                f"- `{metric_name}`: dagua=`{metric_payload['dagua']:.4f}`, "
                f"best_other=`{metric_payload['best_non_dagua_competitor']}` "
                f"({metric_payload['best_non_dagua_value']:.4f})"
                if metric_payload["best_non_dagua_value"] is not None
                else f"- `{metric_name}`: dagua=`{metric_payload['dagua']:.4f}`, best_other=`N/A`"
            )
        lines.append("")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(json_path), str(md_path)


def _latest_standard_run_id(combined_results: Dict[str, Any]) -> Optional[str]:
    return combined_results.get("generated_from", {}).get("standard")


def _previous_standard_results(output_dir: str, current_run_id: Optional[str]) -> Optional[Dict[str, Any]]:
    root = Path(output_dir) / "benchmark_db" / "standard"
    if not root.exists():
        return None
    run_dirs = [path for path in root.iterdir() if path.is_dir() and path.name != "latest"]
    run_dirs.sort(key=lambda path: path.name)
    candidates = [path for path in run_dirs if path.name != current_run_id and (path / "results.json").exists()]
    if not candidates:
        return None
    return _load_json(candidates[-1] / "results.json")


def generate_benchmark_deltas(
    output_dir: str = "eval_output",
    combined_results: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    if combined_results is None:
        from dagua.eval.benchmark import load_combined_results

        combined_results = load_combined_results(output_dir)

    current_run_id = _latest_standard_run_id(combined_results)
    current_payload = None
    if current_run_id is not None:
        current_path = Path(output_dir) / "benchmark_db" / "standard" / current_run_id / "results.json"
        if current_path.exists():
            current_payload = _load_json(current_path)
    previous_payload = _previous_standard_results(output_dir, current_run_id)

    report_dir = Path(output_dir) / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "benchmark_deltas.json"
    md_path = report_dir / "benchmark_deltas.md"

    if current_payload is None or previous_payload is None:
        payload: Dict[str, Any] = {
            "current_run_id": current_run_id,
            "baseline_run_id": previous_payload.get("run_id") if previous_payload else None,
            "status": "insufficient_history",
            "aggregate": {},
            "graphs": {},
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        md_path.write_text("# Benchmark Deltas\n\nNo previous standard run was available for delta reporting.\n", encoding="utf-8")
        return str(json_path), str(md_path)

    overlapping = sorted(set(current_payload.get("graphs", {})) & set(previous_payload.get("graphs", {})))
    graph_deltas: Dict[str, Any] = {}
    runtime_deltas: List[float] = []
    score_deltas: List[float] = []
    dag_deltas: List[float] = []
    win_count = 0
    loss_count = 0

    for graph_name in overlapping:
        current_result = current_payload["graphs"][graph_name]["competitors"].get("dagua", {})
        baseline_result = previous_payload["graphs"][graph_name]["competitors"].get("dagua", {})
        if current_result.get("status") != "OK" or baseline_result.get("status") != "OK":
            continue
        current_metrics = current_result.get("metrics", {})
        baseline_metrics = baseline_result.get("metrics", {})
        metric_delta: Dict[str, float] = {}
        for key in sorted(set(current_metrics) & set(baseline_metrics)):
            cur = current_metrics.get(key)
            old = baseline_metrics.get(key)
            if isinstance(cur, (int, float)) and isinstance(old, (int, float)):
                metric_delta[key] = float(cur) - float(old)
        runtime_delta = None
        if current_result.get("runtime_seconds") is not None and baseline_result.get("runtime_seconds") is not None:
            runtime_delta = float(current_result["runtime_seconds"]) - float(baseline_result["runtime_seconds"])
            runtime_deltas.append(runtime_delta)
        score_delta = None
        if current_result.get("composite_score") is not None and baseline_result.get("composite_score") is not None:
            score_delta = float(current_result["composite_score"]) - float(baseline_result["composite_score"])
            score_deltas.append(score_delta)
            if score_delta > 0:
                win_count += 1
            elif score_delta < 0:
                loss_count += 1
        dag_delta = metric_delta.get("dag_consistency")
        if dag_delta is not None:
            dag_deltas.append(dag_delta)
        graph_deltas[graph_name] = {
            "runtime_delta_seconds": runtime_delta,
            "composite_score_delta": score_delta,
            "metric_deltas": metric_delta,
        }

    aggregate = {
        "graphs_compared": len(graph_deltas),
        "mean_runtime_delta_seconds": mean(runtime_deltas) if runtime_deltas else None,
        "mean_composite_score_delta": mean(score_deltas) if score_deltas else None,
        "mean_dag_consistency_delta": mean(dag_deltas) if dag_deltas else None,
        "score_improved_graphs": win_count,
        "score_regressed_graphs": loss_count,
    }
    payload = {
        "current_run_id": current_payload.get("run_id"),
        "baseline_run_id": previous_payload.get("run_id"),
        "status": "ok",
        "aggregate": aggregate,
        "graphs": graph_deltas,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Benchmark Deltas",
        "",
        f"Current standard run: `{payload['current_run_id']}`",
        f"Baseline standard run: `{payload['baseline_run_id']}`",
        "",
        f"- Graphs compared: {aggregate['graphs_compared']}",
        f"- Mean composite score delta: {aggregate['mean_composite_score_delta']:.3f}" if aggregate["mean_composite_score_delta"] is not None else "- Mean composite score delta: N/A",
        f"- Mean runtime delta (s): {aggregate['mean_runtime_delta_seconds']:.3f}" if aggregate["mean_runtime_delta_seconds"] is not None else "- Mean runtime delta (s): N/A",
        "",
        "| Graph | Score delta | Runtime delta (s) |",
        "|---|---:|---:|",
    ]
    for graph_name in sorted(graph_deltas):
        gd = graph_deltas[graph_name]
        score = gd["composite_score_delta"]
        runtime = gd["runtime_delta_seconds"]
        lines.append(
            f"| {graph_name} | {score:.3f} | {runtime:.3f} |"
            if score is not None and runtime is not None
            else f"| {graph_name} | {score if score is not None else 'N/A'} | {runtime if runtime is not None else 'N/A'} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(json_path), str(md_path)


def _summary_statistics(combined_results: Dict[str, Any]) -> Dict[str, Any]:
    aggregate: Dict[str, Any] = {"competitors": {}}
    for graph_name, graph_payload in combined_results.get("graphs", {}).items():
        n_nodes = graph_payload.get("n_nodes", 0)
        for comp_name, result in graph_payload.get("competitors", {}).items():
            stats = aggregate["competitors"].setdefault(comp_name, {"scores": [], "runtimes": [], "small_scores": [], "dag_consistency": []})
            if result.get("status") != "OK":
                continue
            if result.get("composite_score") is not None:
                stats["scores"].append(float(result["composite_score"]))
                if n_nodes <= 500:
                    stats["small_scores"].append(float(result["composite_score"]))
            if result.get("runtime_seconds") is not None:
                stats["runtimes"].append(float(result["runtime_seconds"]))
            dag_val = result.get("metrics", {}).get("dag_consistency")
            if dag_val is not None:
                stats["dag_consistency"].append(float(dag_val))
    return aggregate


def _format_mean(values: Sequence[float]) -> str:
    return f"{sum(values) / len(values):.2f}" if values else "N/A"


def _aesthetic_critic_prompt() -> str:
    return (
        "Use the following rubric to judge a competition between graph layout algorithms.\n"
        "Below are renderings of the same graph laid out by different algorithms. Each image\n"
        "is labeled with the algorithm name.\n\n"
        "Judge which layout is the most aesthetically pleasing and readable. Consider all\n"
        "standard graph drawing aesthetic criteria, including but not limited to:\n"
        "- Edge crossing minimization\n"
        "- Edge length uniformity\n"
        "- Symmetry and balance\n"
        "- Effective use of space\n"
        "- Clear hierarchical/directional flow\n"
        "- Visual separation of clusters or groups\n"
        "- Angular resolution at nodes\n"
        "- Edge straightness and routing quality\n"
        "- Node overlap avoidance\n"
        "- Overall cleanliness and readability\n\n"
        "Return your answer as JSON:\n"
        "{\n"
        '  "winner": "<algorithm_name>",\n'
        '  "ranking": ["<best>", "<second>", "...", "<worst>"],\n'
        '  "reasoning": "<2-3 sentences explaining your choice, citing specific visual features>"\n'
        "}\n\n"
        "Be decisive. Pick a clear winner. Do not hedge.\n"
    )


def _review_prompt() -> str:
    return (
        "Use the following skeptical benchmarking review rubric when evaluating a benchmark\n"
        "report for Dagua. The goal is to identify unclear claims, missing context, unfair\n"
        "comparisons, and any weakness in the evidence presented.\n\n"
        "Do NOT tolerate:\n\n"
        "- Unclear or ambiguous claims\n"
        "- Cherry-picked results that hide weaknesses\n"
        "- Missing context that would change interpretation\n"
        "- Vague language where precise numbers should appear\n"
        "- Unfair comparisons (different settings, missing competitors)\n"
        "- Visual clutter or confusing figures\n"
        "- Claims not directly supported by the presented data\n"
        "- Burying unfavorable results\n"
        "- Statistical claims without proper methodology\n"
        "- Beautiful formatting masking thin content\n"
        "- Vague appeals to quality or aesthetics without grounding in specific criteria\n\n"
        "Rate the report on: Organization, Writing Clarity, Visual Quality, Convincingness,\n"
        "Statistical Rigor, Honesty/Balance. For each criterion below 8, provide specific\n"
        "revision instructions. For each criterion at 8+, note what works well and one improvement.\n"
    )


def _write_prompt_file(report_dir: Path) -> None:
    text = (
        "Write the benchmark report in a precise, skeptical, technically grounded voice.\n"
        "Every claim should cite the relevant figure or table. State weaknesses as plainly as\n"
        "strengths. Do not use marketing language. Ground quality claims in the stored metrics,\n"
        "the aesthetic score, and the style-guide anti-pattern flags.\n"
    )
    (report_dir / "prose_prompt.md").write_text(text, encoding="utf-8")


def _write_review_placeholders(report_dir: Path, rounds: int = 5) -> None:
    for idx in range(rounds):
        payload = {
            "round": idx + 1,
            "status": "skipped",
            "reason": "No separate review pass was executed in this environment",
            "prompt": _review_prompt(),
            "changes_made": "No automated review applied.",
        }
        with open(report_dir / f"review_round_{idx + 1}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def _latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def _latex_path(path: str) -> str:
    return path.replace("\\", "/")


def _render_latex_report(
    report_dir: Path,
    combined_results: Dict[str, Any],
    scaling_curve_path: str,
    comparison_paths: Sequence[str],
    delta_markdown_path: Optional[str] = None,
    similarity_markdown_path: Optional[str] = None,
    placement_markdown_path: Optional[str] = None,
    placement_dashboard_path: Optional[str] = None,
) -> str:
    summary = _summary_statistics(combined_results)
    generated_from = combined_results.get("generated_from", {})
    comp_lines = []
    for comp_name, stats in summary["competitors"].items():
        comp_lines.append(
            f"{_latex_escape(comp_name)} & {_format_mean(stats['small_scores'])} & {_format_mean(stats['dag_consistency'])} & {_format_mean(stats['runtimes'])} \\\\"
        )

    gallery_items = []
    for path in list(comparison_paths)[:4]:
        gallery_items.append(
            "\\includegraphics[width=0.48\\linewidth]{" + _latex_path(os.path.relpath(path, report_dir)) + "}"
        )
    gallery_block = "\\\\[6pt]".join(gallery_items) if gallery_items else "No comparison images available for this run."

    tex = f"""
\\documentclass[11pt]{{article}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage[table]{{xcolor}}
\\usepackage{{longtable}}
\\usepackage{{hyperref}}
\\definecolor{{daguagreen}}{{HTML}}{{D4EDDA}}
\\definecolor{{daguared}}{{HTML}}{{F8D7DA}}
\\definecolor{{daguayellow}}{{HTML}}{{FFF3CD}}
\\title{{Dagua Benchmark Report}}
\\date{{\\today}}
\\begin{{document}}
\\maketitle

\\section{{Executive Summary}}
This report summarizes the latest standard benchmark run {generated_from.get('standard', 'N/A')} and the latest rare run {generated_from.get('rare', 'N/A')}. Results are stored in \\texttt{{eval\\_output/benchmark\\_db}} and merged through \\texttt{{combined\\_latest.json}}.

\\begin{{itemize}}
\\item Dagua remains the only engine in the default roster expected to scale through the rare ladder.
\\item DAG-aware engines are compared directly on runtime, composite score, and DAG consistency.
\\item Force-directed baselines are included for contrast, not as DAG-faithful references.
\\item A separate visual-critic pass was not executed in this environment; the stored prompt can be reused for either local or external review.
\\end{{itemize}}

\\section{{Methodology}}
The suite mixes small structural motifs, real architecture traces, and a scale ladder. Metric computation uses the existing Dagua metric stack: full metrics for smaller graphs, quick metrics for larger graphs. Anti-pattern flags are derived from the stored quality metrics rather than subjective post-hoc inspection.

\\section{{Aggregate Results}}
\\begin{{figure}}[h]
\\centering
\\includegraphics[width=0.9\\linewidth]{{{_latex_path(os.path.relpath(scaling_curve_path, report_dir))}}}
\\caption{{Scaling curve across all stored benchmark runs.}}
\\end{{figure}}

\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{lrrr}}
\\toprule
Competitor & Avg. small-graph score & Avg. DAG consistency & Avg. runtime (s) \\\\
\\midrule
{' '.join(comp_lines)}
\\bottomrule
\\end{{tabular}}
\\caption{{Aggregate stored results. Pale green/red/yellow cell coloring is reserved for richer future comparisons; the current report keeps the table neutral when not all competitors are installed.}}
\\end{{table}}

\\paragraph{{Layout similarity.}} Pairwise Procrustes similarity summaries between competitor layouts are stored in \\texttt{{{_latex_escape(os.path.basename(similarity_markdown_path or 'layout_similarity.md'))}}}. This helps distinguish genuinely different geometric solutions from stylistic differences layered on top of similar placements.

\\paragraph{{Placement-only summary.}} A styling-agnostic placement summary is stored in \\texttt{{{_latex_escape(os.path.basename(placement_markdown_path or 'placement_summary.md'))}}}. This is the right artifact to consult when judging node placement independently of rendering choices.

\\paragraph{{Placement dashboard.}} The trench-view placement dashboard is stored in \\texttt{{{_latex_escape(os.path.basename(placement_dashboard_path or 'placement_dashboard.md'))}}}. Use it during optimization sprints when the only question is whether Dagua's placement is winning or losing on the core metrics.

\\section{{Visual Gallery}}
{gallery_block}

\\section{{Appendix}}
The aesthetic critic prompt used for optional visual evaluation is stored verbatim in \\texttt{{prose\\_prompt.md}} and review placeholders are stored as \\texttt{{review\\_round\\_*.json}}.

\\paragraph{{Benchmark deltas.}} Round-over-round Dagua deltas, when available, are stored in \\texttt{{{_latex_escape(os.path.basename(delta_markdown_path or 'benchmark_deltas.md'))}}}.

\\end{{document}}
"""
    tex_path = report_dir / "benchmark_report.tex"
    tex_path.write_text(tex, encoding="utf-8")
    return str(tex_path)


def _compile_pdf(tex_path: str) -> Optional[str]:
    if shutil.which("pdflatex") is None:
        return None
    tex_file = Path(tex_path)
    try:
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_file.name],
            cwd=tex_file.parent,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    pdf_path = tex_file.with_suffix(".pdf")
    return str(pdf_path) if pdf_path.exists() else None


def generate_report(
    output_dir: str = "eval_output",
    combined_results: Optional[Dict[str, Any]] = None,
    compile_pdf: bool = True,
) -> Dict[str, str]:
    """Generate scaling figure, comparison gallery, LaTeX report, and prompts."""
    from dagua.eval.benchmark import load_combined_results

    if combined_results is None:
        combined_results = load_combined_results(output_dir)

    root = Path(output_dir)
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "figures").mkdir(parents=True, exist_ok=True)

    scaling_curve_path = generate_scaling_curve(output_dir=output_dir, combined_results=combined_results)
    comparison_paths = generate_comparison_visuals(output_dir=output_dir, combined_results=combined_results)
    delta_json_path, delta_md_path = generate_benchmark_deltas(output_dir=output_dir, combined_results=combined_results)
    similarity_json_path, similarity_md_path = generate_layout_similarity_artifacts(output_dir=output_dir, combined_results=combined_results)
    placement_json_path, placement_md_path = generate_placement_summary_artifacts(output_dir=output_dir, combined_results=combined_results)
    placement_dashboard_json_path, placement_dashboard_md_path = generate_placement_dashboard_artifacts(output_dir=output_dir, combined_results=combined_results)
    _write_prompt_file(report_dir)
    _write_review_placeholders(report_dir)
    with open(report_dir / "aesthetic_critic_prompt.md", "w", encoding="utf-8") as f:
        f.write(_aesthetic_critic_prompt())

    tex_path = _render_latex_report(
        report_dir,
        combined_results,
        scaling_curve_path,
        comparison_paths,
        delta_md_path,
        similarity_md_path,
        placement_md_path,
        placement_dashboard_md_path,
    )
    pdf_path = _compile_pdf(tex_path) if compile_pdf else None
    return {
        "tex": tex_path,
        "pdf": pdf_path or "",
        "scaling_curve": scaling_curve_path,
        "comparisons_dir": str(root / "visuals" / "comparisons"),
        "benchmark_deltas_json": delta_json_path,
        "benchmark_deltas_md": delta_md_path,
        "layout_similarity_json": similarity_json_path,
        "layout_similarity_md": similarity_md_path,
        "placement_summary_json": placement_json_path,
        "placement_summary_md": placement_md_path,
        "placement_dashboard_json": placement_dashboard_json_path,
        "placement_dashboard_md": placement_dashboard_md_path,
    }
