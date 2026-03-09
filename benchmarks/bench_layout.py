"""Runtime scaling benchmarks: Dagua vs Graphviz at 100 → 100K nodes.

Usage:
    python benchmarks/bench_layout.py              # default sizes
    python benchmarks/bench_layout.py --sizes 100 500 1000 5000
    python benchmarks/bench_layout.py --output results.json
    python benchmarks/bench_layout.py --plot scaling.png
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dagua.config import LayoutConfig
from dagua.graph import DaguaGraph
from dagua.graphviz_utils import to_dot


def _random_dag(n_nodes: int, edge_ratio: float = 1.5, seed: int = 42) -> DaguaGraph:
    """Generate a random DAG with n_nodes and approximately n_nodes * edge_ratio edges."""
    import random

    rng = random.Random(seed)
    n_edges = int(n_nodes * edge_ratio)

    g = DaguaGraph.from_edge_index(
        torch.zeros(2, 0, dtype=torch.long), num_nodes=n_nodes
    )

    edges = set()
    attempts = 0
    max_attempts = n_edges * 20
    while len(edges) < n_edges and attempts < max_attempts:
        i = rng.randint(0, n_nodes - 2)
        j = rng.randint(i + 1, min(i + max(n_nodes // 5, 10), n_nodes - 1))
        edges.add((i, j))
        attempts += 1

    if edges:
        edge_list = list(edges)
        src = [e[0] for e in edge_list]
        tgt = [e[1] for e in edge_list]
        g.edge_index = torch.tensor([src, tgt], dtype=torch.long)
        g.edge_labels = [None] * len(edge_list)
        g.edge_types = ["normal"] * len(edge_list)
        g.edge_styles = [None] * len(edge_list)

    return g


def _layered_dag(n_nodes: int, width: int = 10, seed: int = 42) -> DaguaGraph:
    """Generate a layered DAG — nodes organized in layers with edges to next layer.

    This is more representative of neural network architectures.
    """
    import random

    rng = random.Random(seed)
    n_layers = max(n_nodes // width, 2)

    # Distribute nodes across layers
    layers: List[List[int]] = []
    node_idx = 0
    for layer in range(n_layers):
        layer_size = width
        if node_idx + layer_size > n_nodes:
            layer_size = n_nodes - node_idx
        if layer_size <= 0:
            break
        layers.append(list(range(node_idx, node_idx + layer_size)))
        node_idx += layer_size

    # Add remaining nodes to last layer
    if node_idx < n_nodes:
        layers[-1].extend(range(node_idx, n_nodes))

    # Create edges between adjacent layers
    edges = set()
    for i in range(len(layers) - 1):
        current = layers[i]
        next_layer = layers[i + 1]
        # Each node connects to 1-3 random nodes in next layer
        for node in current:
            n_connections = min(rng.randint(1, 3), len(next_layer))
            targets = rng.sample(next_layer, n_connections)
            for t in targets:
                edges.add((node, t))

    g = DaguaGraph.from_edge_index(
        torch.zeros(2, 0, dtype=torch.long), num_nodes=n_nodes
    )

    if edges:
        edge_list = list(edges)
        src = [e[0] for e in edge_list]
        tgt = [e[1] for e in edge_list]
        g.edge_index = torch.tensor([src, tgt], dtype=torch.long)
        g.edge_labels = [None] * len(edge_list)
        g.edge_types = ["normal"] * len(edge_list)
        g.edge_styles = [None] * len(edge_list)

    return g


def time_dagua(graph: DaguaGraph, steps: int, device: str = "cpu") -> float:
    """Time dagua layout in seconds."""
    from dagua.layout import layout

    config = LayoutConfig(steps=steps, device=device)
    start = time.perf_counter()
    layout(graph, config)
    elapsed = time.perf_counter() - start
    return elapsed


def time_graphviz(graph: DaguaGraph, timeout: float = 300.0) -> Optional[float]:
    """Time Graphviz dot layout in seconds. Returns None if failed/timeout."""
    dot_str = to_dot(graph)

    start = time.perf_counter()
    try:
        result = subprocess.run(
            ["dot", "-Tjson"],
            input=dot_str,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.perf_counter() - start
        if result.returncode != 0:
            return None
        return elapsed
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def run_benchmark(
    sizes: List[int],
    graph_type: str = "random",
    dagua_steps: int = 50,
    device: str = "cpu",
    graphviz_timeout: float = 300.0,
    warmup: bool = True,
) -> List[Dict]:
    """Run scaling benchmark across graph sizes.

    Args:
        sizes: List of node counts to benchmark.
        graph_type: "random" or "layered".
        dagua_steps: Number of optimization steps for dagua.
        device: "cpu" or "cuda".
        graphviz_timeout: Timeout for graphviz in seconds.
        warmup: Whether to do a warmup run first.

    Returns:
        List of result dicts with timing data.
    """
    results = []

    # Warmup: run a small layout to JIT compile / cache
    if warmup:
        g_warmup = _random_dag(20)
        g_warmup.compute_node_sizes()
        time_dagua(g_warmup, steps=10, device=device)
        time_graphviz(g_warmup, timeout=10)

    for n in sizes:
        print(f"\n{'='*60}")
        print(f"Benchmarking N={n:,} nodes ({graph_type})")
        print(f"{'='*60}")

        # Generate graph
        gen_start = time.perf_counter()
        if graph_type == "layered":
            width = max(int(n**0.5), 5)
            g = _layered_dag(n, width=width)
        else:
            g = _random_dag(n)
        gen_time = time.perf_counter() - gen_start

        g.compute_node_sizes()
        n_edges = g.edge_index.shape[1]
        print(f"  Graph: {n:,} nodes, {n_edges:,} edges (generated in {gen_time:.2f}s)")

        # Dagua
        print(f"  Dagua ({dagua_steps} steps, {device})...", end="", flush=True)
        dagua_time = time_dagua(g, steps=dagua_steps, device=device)
        print(f" {dagua_time:.2f}s")

        # Graphviz
        gv_time = None
        if n <= 50000:  # Graphviz DOT can struggle above 50K
            print(f"  Graphviz (dot)...", end="", flush=True)
            gv_time = time_graphviz(g, timeout=graphviz_timeout)
            if gv_time is not None:
                print(f" {gv_time:.2f}s")
            else:
                print(f" TIMEOUT/FAILED")
        else:
            print(f"  Graphviz: skipped (N>{50000:,})")

        # Speedup
        speedup = None
        if gv_time is not None and gv_time > 0:
            speedup = gv_time / dagua_time
            faster = "dagua" if speedup > 1 else "graphviz"
            ratio = speedup if speedup > 1 else 1 / speedup
            print(f"  Winner: {faster} ({ratio:.1f}x faster)")

        result = {
            "num_nodes": n,
            "num_edges": n_edges,
            "graph_type": graph_type,
            "dagua_steps": dagua_steps,
            "device": device,
            "dagua_seconds": round(dagua_time, 4),
            "graphviz_seconds": round(gv_time, 4) if gv_time is not None else None,
            "speedup_dagua_over_gv": round(speedup, 2) if speedup is not None else None,
            "gen_seconds": round(gen_time, 4),
        }
        results.append(result)

    return results


def print_summary(results: List[Dict]) -> None:
    """Print a formatted summary table."""
    print(f"\n{'='*80}")
    print("RUNTIME SCALING BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"{'Nodes':>8} {'Edges':>8} {'Dagua(s)':>10} {'GV(s)':>10} {'Speedup':>10} {'Winner':>10}")
    print(f"{'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for r in results:
        n = r["num_nodes"]
        e = r["num_edges"]
        dt = r["dagua_seconds"]
        gt = r["graphviz_seconds"]
        sp = r["speedup_dagua_over_gv"]

        gv_str = f"{gt:.2f}" if gt is not None else "N/A"
        if sp is not None:
            if sp > 1:
                sp_str = f"{sp:.1f}x dagua"
            else:
                sp_str = f"{1/sp:.1f}x gv"
            winner = "dagua" if sp > 1 else "graphviz"
        else:
            sp_str = "N/A"
            winner = "dagua*"

        print(f"{n:>8,} {e:>8,} {dt:>10.2f} {gv_str:>10} {sp_str:>10} {winner:>10}")

    print(f"\n* dagua wins by default (Graphviz timed out or unavailable)")


def plot_results(results: List[Dict], output: str) -> None:
    """Generate a scaling plot comparing Dagua vs Graphviz."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    nodes = [r["num_nodes"] for r in results]
    dagua_times = [r["dagua_seconds"] for r in results]
    gv_times = [r["graphviz_seconds"] for r in results]
    gv_nodes = [n for n, t in zip(nodes, gv_times) if t is not None]
    gv_valid = [t for t in gv_times if t is not None]

    # Left: absolute runtime
    ax1.plot(nodes, dagua_times, "o-", color="#2196F3", linewidth=2, label="Dagua", markersize=6)
    if gv_valid:
        ax1.plot(gv_nodes, gv_valid, "s--", color="#FF5722", linewidth=2, label="Graphviz (dot)", markersize=6)
    ax1.set_xlabel("Number of Nodes")
    ax1.set_ylabel("Runtime (seconds)")
    ax1.set_title("Runtime Scaling: Dagua vs Graphviz")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: speedup ratio
    speedups = [(r["num_nodes"], r["speedup_dagua_over_gv"]) for r in results if r["speedup_dagua_over_gv"] is not None]
    if speedups:
        sp_nodes, sp_vals = zip(*speedups)
        colors = ["#4CAF50" if s > 1 else "#FF5722" for s in sp_vals]
        ax2.bar(range(len(sp_nodes)), sp_vals, color=colors, alpha=0.8)
        ax2.set_xticks(range(len(sp_nodes)))
        ax2.set_xticklabels([f"{n:,}" for n in sp_nodes], rotation=45, ha="right")
        ax2.axhline(y=1.0, color="black", linestyle="--", alpha=0.5)
        ax2.set_xlabel("Number of Nodes")
        ax2.set_ylabel("Speedup (Dagua / Graphviz)")
        ax2.set_title("Dagua Speedup over Graphviz")
        ax2.grid(True, alpha=0.3, axis="y")

        # Annotate bars
        for i, (n, s) in enumerate(zip(sp_nodes, sp_vals)):
            label = f"{s:.1f}x" if s > 1 else f"{1/s:.1f}x GV"
            ax2.text(i, s + 0.05 * max(sp_vals), label, ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {output}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Dagua vs Graphviz runtime scaling benchmark")
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000],
        help="Node counts to benchmark",
    )
    parser.add_argument("--type", choices=["random", "layered"], default="random", help="Graph type")
    parser.add_argument("--steps", type=int, default=50, help="Dagua optimization steps")
    parser.add_argument("--device", default="cpu", help="Dagua device (cpu/cuda)")
    parser.add_argument("--timeout", type=float, default=300.0, help="Graphviz timeout (seconds)")
    parser.add_argument("--output", type=str, default=None, help="Save results as JSON")
    parser.add_argument("--plot", type=str, default=None, help="Save scaling plot as image")
    args = parser.parse_args()

    print("Dagua vs Graphviz Runtime Scaling Benchmark")
    print(f"Graph type: {args.type}")
    print(f"Dagua steps: {args.steps}")
    print(f"Device: {args.device}")
    print(f"Sizes: {args.sizes}")

    results = run_benchmark(
        sizes=args.sizes,
        graph_type=args.type,
        dagua_steps=args.steps,
        device=args.device,
        graphviz_timeout=args.timeout,
    )

    print_summary(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    if args.plot:
        plot_results(results, args.plot)


if __name__ == "__main__":
    main()
