"""Quick evaluation — subset of graphs, defaults only, no sweep.

Usage: python -m dagua.eval.quick [--output-dir eval_output]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Dagua quick evaluation")
    parser.add_argument("--output-dir", default="eval_output", help="Output directory")
    parser.add_argument("--max-nodes", type=int, default=200, help="Max nodes per graph")
    args = parser.parse_args()

    from dagua.config import LayoutConfig
    from dagua.eval.graphs import get_test_graphs
    from dagua.eval.compare import compare_with_graphviz, print_comparison_table
    from dagua.eval.report import generate_grid, generate_comparison_grid

    output_dir = args.output_dir
    config = LayoutConfig()

    print("=== Dagua Quick Evaluation ===\n")

    # Get test graphs
    graphs = get_test_graphs(max_nodes=args.max_nodes)
    print(f"Test graphs: {len(graphs)}")
    for tg in graphs:
        print(f"  {tg.name} ({tg.graph.num_nodes} nodes, tags: {', '.join(sorted(tg.tags))})")

    # Compare with Graphviz
    print("\n--- Graphviz Comparison ---")
    results = compare_with_graphviz(
        graphs=graphs,
        config=config,
        output_dir=str(Path(output_dir) / "comparisons"),
        max_nodes=args.max_nodes,
    )
    print_comparison_table(results)

    # Generate grids
    print("\n--- Generating Grids ---")
    generate_grid(graphs, config, str(Path(output_dir) / "grids"))
    generate_comparison_grid(graphs, config, str(Path(output_dir) / "grids"))

    print(f"\nDone. Results in {output_dir}/")


if __name__ == "__main__":
    main()
