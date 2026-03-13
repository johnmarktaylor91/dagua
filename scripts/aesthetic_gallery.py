"""Render all bundled test graphs for aesthetic review.

Produces:
  - aesthetic_review/gallery.png — grid overview of all graphs
  - aesthetic_review/individual/<name>.png — high-quality individual renders
  - aesthetic_review/clusters.png — cluster-specific graphs at larger size
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", "findfont")

from dagua.graphs import list_graphs, load
from dagua.config import LayoutConfig
from dagua.layout import layout
from dagua.edges import route_edges
from dagua.render.mpl import render, _draw_clusters, _draw_edges, _draw_nodes, _draw_node_labels

OUTPUT_DIR = Path("aesthetic_review")

# Graphs to skip (too large for rapid iteration)
SKIP = {"medium_mixed"}  # can add more if needed

# Graphs that showcase clusters (render larger)
CLUSTER_GRAPHS = {
    "nested_clusters", "deep_nesting_4", "deep_nesting_6",
    "flat_many_clusters", "cross_cluster_edges",
}


def render_individual(name, graph, config, output_dir):
    """Render a single graph to file using the full pipeline."""
    graph.compute_node_sizes()
    pos = layout(graph, config)
    curves = route_edges(pos, graph.edge_index, graph.node_sizes, graph.direction, graph)

    # Optimize edges (skip if result contains NaN control points)
    try:
        from dagua.layout.edge_optimization import optimize_edges
        import math
        curves_opt = optimize_edges(curves, pos, graph.edge_index, graph.node_sizes, config, graph)
        has_nan = any(
            math.isnan(c.cp1[0]) or math.isnan(c.cp1[1]) or
            math.isnan(c.cp2[0]) or math.isnan(c.cp2[1])
            for c in curves_opt
        )
        if not has_nan:
            curves = curves_opt
    except Exception:
        pass

    # Place edge labels
    from dagua.edges import place_edge_labels
    label_positions = place_edge_labels(curves, pos, graph.node_sizes, graph.edge_labels, graph)

    outpath = output_dir / f"{name}.png"
    fig, ax = render(graph, pos, config, output=str(outpath), dpi=200,
                     curves=curves, label_positions=label_positions)
    plt.close(fig)
    return outpath


def render_gallery(graphs_dict, config, output_path, cols=5, cell_size=(4, 3.5)):
    """Render all graphs in a tiled gallery."""
    names = sorted(graphs_dict.keys())
    n = len(names)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * cell_size[0], rows * cell_size[1]))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [list(axes)]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    bg = "#FAFAFA"
    fig.patch.set_facecolor(bg)

    for idx, name in enumerate(names):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        ax.set_facecolor(bg)

        graph = graphs_dict[name]
        try:
            graph.compute_node_sizes()
            pos = layout(graph, config)
            pos_np = pos.detach().cpu().numpy()
            sizes = graph.node_sizes.detach().cpu().numpy()

            curves = route_edges(pos, graph.edge_index, graph.node_sizes, graph.direction, graph)

            _draw_clusters(ax, graph, pos_np, sizes)
            _draw_edges(ax, graph, curves)
            _draw_nodes(ax, graph, pos_np, sizes)
            _draw_node_labels(ax, graph, pos_np, sizes)

            margin = 15
            x_min = (pos_np[:, 0] - sizes[:, 0] / 2).min() - margin
            x_max = (pos_np[:, 0] + sizes[:, 0] / 2).max() + margin
            y_min = (pos_np[:, 1] - sizes[:, 1] / 2).min() - margin
            y_max = (pos_np[:, 1] + sizes[:, 1] / 2).max() + margin

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect("equal")
        except Exception as e:
            ax.text(0.5, 0.5, f"Error:\n{str(e)[:60]}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7, color="red")

        # Clean title
        display_name = name.replace("_", " ").title()
        ax.set_title(display_name, fontsize=8, fontweight="medium", color="#4A4A4A", pad=4)
        ax.axis("off")

    # Hide empty axes
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")

    plt.tight_layout(pad=1.0)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight", facecolor=bg)
    plt.close(fig)
    print(f"Gallery saved to {output_path}")


def main():
    # Setup output dirs
    OUTPUT_DIR.mkdir(exist_ok=True)
    individual_dir = OUTPUT_DIR / "individual"
    individual_dir.mkdir(exist_ok=True)

    config = LayoutConfig(seed=42)

    # Load all graphs
    all_names = list_graphs()
    graphs = {}
    for name in all_names:
        if name in SKIP:
            continue
        try:
            graphs[name] = load(name)
        except Exception as e:
            print(f"  Skipping {name}: {e}")

    print(f"Loaded {len(graphs)} graphs")

    # Render individual high-quality images
    print("\n--- Rendering individual graphs ---")
    for name, graph in sorted(graphs.items()):
        try:
            path = render_individual(name, graph, config, individual_dir)
            print(f"  {name}: {path}")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")

    # Render gallery grid
    print("\n--- Rendering gallery ---")
    render_gallery(graphs, config, OUTPUT_DIR / "gallery.png", cols=6)

    # Render cluster-focused graphs at larger size
    cluster_graphs = {k: v for k, v in graphs.items() if k in CLUSTER_GRAPHS}
    if cluster_graphs:
        print("\n--- Rendering cluster showcase ---")
        render_gallery(cluster_graphs, config, OUTPUT_DIR / "clusters.png",
                       cols=3, cell_size=(6, 5))

    print(f"\nDone! Review images in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
