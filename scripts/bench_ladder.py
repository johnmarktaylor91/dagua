#!/usr/bin/env python3
"""Benchmark ladder: test layout performance across graph sizes and structures.

Usage:
    python scripts/bench_ladder.py                    # default wide-dag ladder
    python scripts/bench_ladder.py --variant chain    # chain DAG variant
    python scripts/bench_ladder.py --variant all      # run all variants
    python scripts/bench_ladder.py --sizes 1m,10m,100m  # custom sizes
    python scripts/bench_ladder.py --variant wide-dag --max-size 100m
    python scripts/bench_ladder.py --list-variants    # show available variants
    python scripts/bench_ladder.py --generate-only    # pre-generate graphs, skip layout

Graph variants test different structural properties:
- wide-dag: backbone + cross edges, uniform width (current default)
- chain: E=N, single chain per layer, minimal edges
- binary-tree: balanced binary tree, deep hierarchy
- clustered: groups of tightly-connected nodes with sparse inter-cluster edges
- scale-free: power-law degree distribution (few hubs, many leaves)
- grid: 2D grid DAG, regular degree-4 structure
- bipartite: alternating dense/sparse layers
- skip-heavy: many long-range skip connections across layers
- neural-net: realistic DNN structure (varying layer widths, residual connections)
- dense-random: E ~ N*log(N) random edges, stress test for crossing minimization
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import torch

# ─── Size parsing ────────────────────────────────────────────────────────────


def parse_size(s: str) -> int:
    """Parse size string: '1m' -> 1_000_000, '1b' -> 1_000_000_000."""
    s = s.strip().lower().replace("_", "").replace(",", "")
    if s.endswith("b"):
        return int(float(s[:-1]) * 1_000_000_000)
    elif s.endswith("m"):
        return int(float(s[:-1]) * 1_000_000)
    elif s.endswith("k"):
        return int(float(s[:-1]) * 1_000)
    return int(s)


DEFAULT_SIZES = [
    10,
    20,
    50,
    100,
    200,
    500,
    1_000,
    2_000,
    5_000,
    10_000,
    20_000,
    50_000,
    100_000,
    200_000,
    500_000,
    1_000_000,
    2_000_000,
    5_000_000,
    10_000_000,
    20_000_000,
    50_000_000,
    100_000_000,
    200_000_000,
    500_000_000,
    1_000_000_000,
]


# ─── Graph generators ───────────────────────────────────────────────────────


def _make_wide_dag(n: int, seed: int = 42) -> Tuple[torch.Tensor, int, int]:
    """Wide DAG: uniform layers with backbone + 50% cross edges.

    This is the standard benchmark graph. Each layer has N/L nodes.
    Backbone edges connect each node to the node directly below.
    Cross edges connect 50% of nodes to a random node in the next layer.

    Returns (edge_index, num_nodes, num_layers).
    """
    torch.manual_seed(seed)
    layers = max(int(n**0.5 / 10) * 10, 10)
    width = max(n // layers, 1)
    n = width * layers  # round to exact multiple

    idx_dtype = torch.int32 if n <= 2_147_483_647 else torch.long
    # Backbone: node i -> node i + width
    src = torch.arange(0, n - width, dtype=idx_dtype)
    tgt = src + width

    # Cross edges: 50% chance per backbone edge
    cross_mask = torch.rand(n - width) < 0.5
    cross_src = torch.arange(0, n - width, dtype=idx_dtype)[cross_mask]
    cross_offset = torch.randint(0, width, (cross_src.shape[0],), dtype=idx_dtype)
    cross_tgt_layer = cross_src // width + 1
    cross_tgt = cross_tgt_layer * width + cross_offset

    edge_index = torch.stack(
        [
            torch.cat([src, cross_src]),
            torch.cat([tgt, cross_tgt]),
        ]
    )
    return edge_index, n, layers


def _make_chain(n: int, seed: int = 42) -> Tuple[torch.Tensor, int, int]:
    """Chain DAG: E = N-1, each node connects to the next. Minimal edges.

    Tests layering speed and basic DAG ordering with zero redundancy.
    """
    src = torch.arange(0, n - 1, dtype=torch.long)
    tgt = src + 1
    return torch.stack([src, tgt]), n, n


def _make_binary_tree(n: int, seed: int = 42) -> Tuple[torch.Tensor, int, int]:
    """Balanced binary tree DAG. Deep hierarchy, no cross edges.

    Tests tree detection fast-path and hierarchical layout quality.
    """
    # Each node i has children 2i+1, 2i+2
    parents = []
    children = []
    for i in range(n):
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n:
            parents.append(i)
            children.append(left)
        if right < n:
            parents.append(i)
            children.append(right)
    edge_index = torch.tensor([parents, children], dtype=torch.long)
    depth = max(1, int(torch.tensor(n).float().log2().item()))
    return edge_index, n, depth


def _make_clustered(n: int, seed: int = 42) -> Tuple[torch.Tensor, int, int]:
    """Clustered DAG: groups of tightly-connected nodes with sparse inter-cluster edges.

    Tests cluster detection, cluster compactness loss, and visual separation.
    """
    torch.manual_seed(seed)
    n_clusters = max(int(n**0.3), 4)
    cluster_size = n // n_clusters
    n = cluster_size * n_clusters  # round
    layers_per_cluster = max(int(cluster_size**0.5), 3)
    width = max(cluster_size // layers_per_cluster, 1)

    src_list = []
    tgt_list = []

    for c in range(n_clusters):
        base = c * cluster_size
        # Intra-cluster: dense layered connections
        for i in range(cluster_size - width):
            src_list.append(base + i)
            tgt_list.append(base + i + width)
            # Extra intra-cluster edge with 30% probability
            if torch.rand(1).item() < 0.3 and i + width < cluster_size:
                offset = torch.randint(0, width, (1,)).item()
                tgt_node = base + (i // width + 1) * width + offset
                if tgt_node < base + cluster_size:
                    src_list.append(base + i)
                    tgt_list.append(tgt_node)

    # Inter-cluster edges: sparse connections between adjacent clusters
    for c in range(n_clusters - 1):
        base_src = c * cluster_size
        base_tgt = (c + 1) * cluster_size
        n_inter = max(int(cluster_size * 0.05), 1)
        for _ in range(n_inter):
            s = base_src + torch.randint(0, cluster_size, (1,)).item()
            t = base_tgt + torch.randint(0, cluster_size, (1,)).item()
            src_list.append(s)
            tgt_list.append(t)

    edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)
    return edge_index, n, layers_per_cluster * n_clusters


def _make_scale_free(n: int, seed: int = 42) -> Tuple[torch.Tensor, int, int]:
    """Scale-free DAG: power-law degree distribution via preferential attachment.

    Few hub nodes with many connections, most nodes with few. Tests repulsion
    sampling quality and hub isolation in coarsening.
    """
    torch.manual_seed(seed)
    # Barabási-Albert-style preferential attachment, made acyclic
    m = 3  # edges per new node
    src_list = []
    tgt_list = []
    degree = torch.zeros(n, dtype=torch.float)

    # Seed: small complete DAG
    for i in range(1, min(m + 1, n)):
        for j in range(i):
            src_list.append(j)
            tgt_list.append(i)
            degree[j] += 1
            degree[i] += 1

    # Preferential attachment
    for i in range(m + 1, n):
        probs = degree[:i] + 1  # +1 to avoid zero probability
        probs = probs / probs.sum()
        targets = torch.multinomial(probs, min(m, i), replacement=False)
        for t in targets:
            src_list.append(t.item())  # earlier node -> later node (acyclic)
            tgt_list.append(i)
            degree[t] += 1
            degree[i] += 1

    edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)
    layers = max(int(n**0.3), 5)
    return edge_index, n, layers


def _make_grid(n: int, seed: int = 42) -> Tuple[torch.Tensor, int, int]:
    """2D grid DAG: regular degree-4 structure. Tests uniform spacing.

    Nodes arranged in rows/cols, edges point right and down (acyclic).
    """
    cols = max(int(n**0.5), 2)
    rows = max(n // cols, 2)
    n = rows * cols

    src_list = []
    tgt_list = []
    for r in range(rows):
        for c in range(cols):
            node = r * cols + c
            # Right edge
            if c + 1 < cols:
                src_list.append(node)
                tgt_list.append(node + 1)
            # Down edge
            if r + 1 < rows:
                src_list.append(node)
                tgt_list.append(node + cols)

    edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)
    return edge_index, n, rows


def _make_bipartite(n: int, seed: int = 42) -> Tuple[torch.Tensor, int, int]:
    """Bipartite DAG: alternating dense layers. Tests crossing minimization.

    Even layers have N/L nodes, odd layers have N/(2L). All even-to-odd edges.
    """
    torch.manual_seed(seed)
    n_layers = max(int(n**0.3), 6)
    if n_layers % 2 == 1:
        n_layers += 1

    wide = n * 2 // (3 * n_layers // 2)  # wide layers
    narrow = wide // 2  # narrow layers

    nodes_per_layer = []
    for i in range(n_layers):
        nodes_per_layer.append(wide if i % 2 == 0 else narrow)

    actual_n = sum(nodes_per_layer)
    offsets = [0]
    for count in nodes_per_layer:
        offsets.append(offsets[-1] + count)

    src_list = []
    tgt_list = []
    for layer in range(n_layers - 1):
        n_src = nodes_per_layer[layer]
        n_tgt = nodes_per_layer[layer + 1]
        # Connect each source to 2-3 random targets
        n_edges = min(n_src * 3, n_src * n_tgt)
        for _ in range(n_edges):
            s = offsets[layer] + torch.randint(0, n_src, (1,)).item()
            t = offsets[layer + 1] + torch.randint(0, n_tgt, (1,)).item()
            src_list.append(s)
            tgt_list.append(t)

    edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)
    return edge_index, actual_n, n_layers


def _make_skip_heavy(n: int, seed: int = 42) -> Tuple[torch.Tensor, int, int]:
    """Skip-heavy DAG: many long-range connections across layers.

    Like wide-dag but 30% of edges skip 2-5 layers. Stress tests layering
    (longest path != BFS depth) and edge straightness.
    """
    torch.manual_seed(seed)
    layers = max(int(n**0.5 / 10) * 10, 10)
    width = max(n // layers, 1)
    n = width * layers

    idx_dtype = torch.int32 if n <= 2_147_483_647 else torch.long
    # Backbone
    src = torch.arange(0, n - width, dtype=idx_dtype)
    tgt = src + width

    # Skip edges: 30% of nodes get a skip edge spanning 2-5 layers
    skip_mask = torch.rand(n - width) < 0.3
    skip_src = torch.arange(0, n - width, dtype=idx_dtype)[skip_mask]
    skip_span = torch.randint(2, min(6, layers), (skip_src.shape[0],))
    skip_tgt_layer = skip_src // width + skip_span.to(idx_dtype)
    skip_tgt_layer = skip_tgt_layer.clamp(max=layers - 1)
    skip_offset = torch.randint(0, width, (skip_src.shape[0],), dtype=idx_dtype)
    skip_tgt = skip_tgt_layer * width + skip_offset

    edge_index = torch.stack(
        [
            torch.cat([src, skip_src]),
            torch.cat([tgt, skip_tgt]),
        ]
    )
    return edge_index, n, layers


def _make_neural_net(n: int, seed: int = 42) -> Tuple[torch.Tensor, int, int]:
    """Neural network DAG: realistic DNN structure with residual connections.

    Varying layer widths (like a real network: narrow input, wide hidden, narrow
    output). Residual skip connections every 2-3 layers. Tests real-world use case.
    """
    torch.manual_seed(seed)
    n_layers = max(int(n**0.3), 8)

    # Varying widths: narrow-wide-narrow pattern
    widths = []
    for i in range(n_layers):
        t = i / max(n_layers - 1, 1)
        # Bell curve: wide in the middle
        w = int(n / n_layers * (0.5 + 2.0 * (1 - abs(2 * t - 1) ** 2)))
        widths.append(max(w, 1))

    # Adjust to hit target n
    actual_n = sum(widths)
    scale = n / max(actual_n, 1)
    widths = [max(int(w * scale), 1) for w in widths]
    actual_n = sum(widths)

    offsets = [0]
    for w in widths:
        offsets.append(offsets[-1] + w)

    src_list = []
    tgt_list = []

    # Dense connections between adjacent layers
    for layer in range(n_layers - 1):
        n_src = widths[layer]
        n_tgt = widths[layer + 1]
        # Each source connects to ~3 targets
        for s_idx in range(n_src):
            s = offsets[layer] + s_idx
            n_conn = min(3, n_tgt)
            targets = torch.randint(0, n_tgt, (n_conn,))
            for t_idx in targets:
                src_list.append(s)
                tgt_list.append(offsets[layer + 1] + t_idx.item())

    # Residual connections every 2-3 layers
    for layer in range(n_layers - 2):
        skip = 2 if layer % 3 != 0 else 3
        tgt_layer = layer + skip
        if tgt_layer >= n_layers:
            continue
        n_residual = max(min(widths[layer], widths[tgt_layer]) // 4, 1)
        for _ in range(n_residual):
            s = offsets[layer] + torch.randint(0, widths[layer], (1,)).item()
            t = offsets[tgt_layer] + torch.randint(0, widths[tgt_layer], (1,)).item()
            src_list.append(s)
            tgt_list.append(t)

    edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)
    return edge_index, actual_n, n_layers


def _make_dense_random(n: int, seed: int = 42) -> Tuple[torch.Tensor, int, int]:
    """Dense random DAG: E ~ N*log(N). Stress test for crossing minimization.

    Random edges respecting topological order (src < tgt for acyclicity).
    """
    import math

    torch.manual_seed(seed)
    n_edges = int(n * max(math.log2(max(n, 2)), 1))
    n_edges = min(n_edges, n * (n - 1) // 2)  # cap at complete DAG

    src = torch.randint(0, n, (n_edges,), dtype=torch.long)
    tgt = torch.randint(0, n, (n_edges,), dtype=torch.long)

    # Enforce acyclicity: src < tgt
    low = torch.min(src, tgt)
    high = torch.max(src, tgt)
    keep = low != high
    edge_index = torch.stack([low[keep], high[keep]])

    # Deduplicate
    if edge_index.shape[1] > 0:
        hashes = edge_index[0].long() * n + edge_index[1].long()
        unique_hash = hashes.unique()
        edge_index = torch.stack([unique_hash // n, unique_hash % n])

    layers = max(int(n**0.3), 5)
    return edge_index, n, layers


# ─── Variant registry ───────────────────────────────────────────────────────

VARIANTS: Dict[str, dict] = {
    "wide-dag": {
        "fn": _make_wide_dag,
        "description": "Uniform-width layered DAG with backbone + cross edges (~1.5x E/N)",
        "max_recommended": 1_000_000_000,
    },
    "chain": {
        "fn": _make_chain,
        "description": "Simple chain (E=N-1). Minimal edges, tests pure layering speed",
        "max_recommended": 1_000_000_000,
    },
    "binary-tree": {
        "fn": _make_binary_tree,
        "description": "Balanced binary tree. Tests tree fast-path and deep hierarchy",
        "max_recommended": 100_000_000,  # Python loop in generator
    },
    "clustered": {
        "fn": _make_clustered,
        "description": "Clustered DAG with dense intra-cluster and sparse inter-cluster edges",
        "max_recommended": 10_000_000,  # Python loop
    },
    "scale-free": {
        "fn": _make_scale_free,
        "description": "Power-law degree distribution via preferential attachment",
        "max_recommended": 1_000_000,  # O(N*m) Python loop
    },
    "grid": {
        "fn": _make_grid,
        "description": "2D grid DAG. Regular degree-4, tests uniform spacing",
        "max_recommended": 10_000_000,  # Python loop
    },
    "bipartite": {
        "fn": _make_bipartite,
        "description": "Alternating wide/narrow layers. Tests crossing minimization",
        "max_recommended": 10_000_000,
    },
    "skip-heavy": {
        "fn": _make_skip_heavy,
        "description": "Wide DAG with 30% skip connections spanning 2-5 layers",
        "max_recommended": 1_000_000_000,
    },
    "neural-net": {
        "fn": _make_neural_net,
        "description": "Realistic DNN structure with varying widths and residual connections",
        "max_recommended": 10_000_000,
    },
    "dense-random": {
        "fn": _make_dense_random,
        "description": "Random DAG with E~N*log(N). Stress test for crossings",
        "max_recommended": 1_000_000,  # E grows fast
    },
}


# ─── Main ────────────────────────────────────────────────────────────────────


def run_layout(
    edge_index: torch.Tensor, num_nodes: int, device: str = "auto"
) -> Tuple[torch.Tensor, float]:
    """Run dagua layout and return (positions, elapsed_seconds)."""
    import dagua

    g = dagua.DaguaGraph()
    g.num_nodes = num_nodes
    g._edge_index_tensor = edge_index
    g.node_sizes = torch.full((num_nodes, 2), 20.0, dtype=torch.float16)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = dagua.LayoutConfig(device=device, verbose=True, seed=42)

    t0 = time.perf_counter()
    pos = dagua.layout(g, config)
    elapsed = time.perf_counter() - t0
    return pos, elapsed


def main():
    """Run the benchmark ladder."""
    parser = argparse.ArgumentParser(
        description="Benchmark ladder: test layout across graph sizes and structures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--variant",
        default="wide-dag",
        help="Graph variant name, or 'all' to run all variants. Default: wide-dag",
    )
    parser.add_argument(
        "--sizes",
        default=None,
        help="Comma-separated size list (e.g., '1k,10k,100k,1m'). Default: full ladder",
    )
    parser.add_argument(
        "--max-size",
        default=None,
        help="Maximum size to run (e.g., '100m'). Default: no limit",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for layout. Default: auto (CUDA if available)",
    )
    parser.add_argument(
        "--output",
        default="eval_output/ladder_results.csv",
        help="Output CSV path. Default: eval_output/ladder_results.csv",
    )
    parser.add_argument(
        "--list-variants",
        action="store_true",
        help="List available graph variants and exit",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate graphs and cache, but skip layout",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for graph generation. Default: 42",
    )
    args = parser.parse_args()

    if args.list_variants:
        print("Available graph variants:\n")
        for name, info in VARIANTS.items():
            max_rec = info["max_recommended"]
            max_str = f"{max_rec:,}" if max_rec < 1_000_000_000 else "1B"
            print(f"  {name:15s}  max={max_str:>10s}  {info['description']}")
        return

    # Parse sizes
    if args.sizes:
        sizes = [parse_size(s) for s in args.sizes.split(",")]
    else:
        sizes = DEFAULT_SIZES

    if args.max_size:
        max_size = parse_size(args.max_size)
        sizes = [s for s in sizes if s <= max_size]

    # Determine variants to run
    if args.variant == "all":
        variants_to_run = list(VARIANTS.keys())
    else:
        if args.variant not in VARIANTS:
            print(f"Unknown variant '{args.variant}'. Use --list-variants to see options.")
            sys.exit(1)
        variants_to_run = [args.variant]

    # Setup output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for variant_name in variants_to_run:
        variant = VARIANTS[variant_name]
        max_rec = variant["max_recommended"]
        variant_sizes = [s for s in sizes if s <= max_rec]

        print(f"\n{'=' * 60}")
        print(f"Variant: {variant_name} — {variant['description']}")
        print(f"Sizes: {len(variant_sizes)} ({variant_sizes[0]:,} to {variant_sizes[-1]:,})")
        print(f"{'=' * 60}\n")

        for size in variant_sizes:
            print(f"--- {variant_name} @ {size:,} nodes ---")

            try:
                t_gen = time.perf_counter()
                edge_index, actual_n, n_layers = variant["fn"](size, seed=args.seed)
                gen_time = time.perf_counter() - t_gen
                n_edges = edge_index.shape[1] if edge_index.numel() > 0 else 0

                print(
                    f"  Generated: {actual_n:,} nodes, {n_edges:,} edges, "
                    f"{n_layers} layers ({gen_time:.1f}s)"
                )

                if args.generate_only:
                    results.append(
                        {
                            "variant": variant_name,
                            "target_nodes": size,
                            "actual_nodes": actual_n,
                            "edges": n_edges,
                            "layers": n_layers,
                            "gen_time": gen_time,
                            "layout_time": 0,
                            "status": "generated",
                        }
                    )
                    continue

                pos, layout_time = run_layout(edge_index, actual_n, device=args.device)
                print(f"  Layout: {layout_time:.1f}s, pos={pos.shape}")

                results.append(
                    {
                        "variant": variant_name,
                        "target_nodes": size,
                        "actual_nodes": actual_n,
                        "edges": n_edges,
                        "layers": n_layers,
                        "gen_time": gen_time,
                        "layout_time": layout_time,
                        "status": "ok",
                    }
                )

            except Exception as e:
                print(f"  FAILED: {e}")
                results.append(
                    {
                        "variant": variant_name,
                        "target_nodes": size,
                        "actual_nodes": 0,
                        "edges": 0,
                        "layers": 0,
                        "gen_time": 0,
                        "layout_time": 0,
                        "status": f"FAILED: {e}",
                    }
                )
                if "OutOfMemoryError" in str(e):
                    print("  Stopping this variant due to OOM.")
                    break

    # Write results
    with open(output_path, "w") as f:
        f.write("variant,target_nodes,actual_nodes,edges,layers,gen_time,layout_time,status\n")
        for r in results:
            f.write(
                f"{r['variant']},{r['target_nodes']},{r['actual_nodes']},"
                f"{r['edges']},{r['layers']},{r['gen_time']:.2f},"
                f"{r['layout_time']:.2f},{r['status']}\n"
            )

    print(f"\nResults written to {output_path}")

    # Print summary table
    print(f"\n{'=' * 80}")
    print(f"{'Variant':15s} {'Nodes':>12s} {'Edges':>12s} {'Time':>10s} {'Status':>10s}")
    print(f"{'=' * 80}")
    for r in results:
        time_str = f"{r['layout_time']:.1f}s" if r["layout_time"] > 0 else "—"
        print(
            f"{r['variant']:15s} {r['actual_nodes']:>12,} {r['edges']:>12,} "
            f"{time_str:>10s} {r['status']:>10s}"
        )


if __name__ == "__main__":
    main()
