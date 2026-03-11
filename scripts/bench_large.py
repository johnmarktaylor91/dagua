"""Large-scale layout benchmark — wide DAG, CPU-only.

Usage:
    python scripts/bench_large.py 50m
    python scripts/bench_large.py 100m
    python scripts/bench_large.py 300m
    python scripts/bench_large.py 1b
    python scripts/bench_large.py 10_000_000          # arbitrary node count
    python scripts/bench_large.py 50m --layers 500 --workers 8
"""

import argparse
import gc
import os
import time

import torch

import dagua


# ─── Helpers ──────────────────────────────────────────────────────────────────


def rss_gb():
    """Current process RSS in GB (Linux /proc/self/statm)."""
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 1024**3
    except Exception:
        return 0.0


def mem(label):
    gc.collect()
    print(f"  [{label}] RSS={rss_gb():.1f} GB", flush=True)


def parse_node_count(s: str) -> int:
    """Parse node count from string: '50m' -> 50_000_000, '1b' -> 1_000_000_000."""
    s = s.strip().lower().replace("_", "").replace(",", "")
    if s.endswith("b"):
        return int(float(s[:-1]) * 1_000_000_000)
    elif s.endswith("m"):
        return int(float(s[:-1]) * 1_000_000)
    elif s.endswith("k"):
        return int(float(s[:-1]) * 1_000)
    else:
        return int(s)


# ─── Presets ──────────────────────────────────────────────────────────────────

PRESETS = {
    "50m": {"n": 50_000_000, "layers": 500},
    "100m": {"n": 100_000_000, "layers": 1000},
    "300m": {"n": 300_000_000, "layers": 1500},
    "1b": {"n": 1_000_000_000, "layers": 1500},
}

# Graphs above this threshold use chunked edge construction to limit peak memory.
CHUNK_THRESHOLD = 200_000_000


# ─── Edge construction ────────────────────────────────────────────────────────


def build_edges(n: int, layers: int) -> torch.Tensor:
    """Build wide-DAG edge_index: backbone + ~50% cross-connections per layer."""
    w = n // layers
    e_backbone = n - w

    if n >= CHUNK_THRESHOLD:
        return _build_edges_chunked(n, w, e_backbone)
    else:
        return _build_edges_simple(n, w, e_backbone)


def _build_edges_simple(n: int, w: int, e_backbone: int) -> torch.Tensor:
    src_backbone = torch.arange(0, n - w, dtype=torch.long)
    tgt_backbone = src_backbone + w

    cross_mask = torch.rand(n - w) < 0.5
    cross_src = torch.arange(0, n - w, dtype=torch.long)[cross_mask]
    cross_offset = torch.randint(0, w, (cross_src.shape[0],), dtype=torch.long)
    cross_tgt_layer = cross_src // w + 1
    cross_tgt = cross_tgt_layer * w + cross_offset

    edge_index = torch.stack([
        torch.cat([src_backbone, cross_src]),
        torch.cat([tgt_backbone, cross_tgt]),
    ])
    del src_backbone, tgt_backbone, cross_src, cross_offset, cross_tgt_layer, cross_tgt, cross_mask
    return edge_index


def _build_edges_chunked(n: int, w: int, e_backbone: int) -> torch.Tensor:
    """Build edges in chunks to limit peak memory for very large graphs."""
    cross_prob = 0.5
    e_cross_est = int(e_backbone * cross_prob * 1.02)  # 2% margin
    e_est = e_backbone + e_cross_est

    edge_src = torch.empty(e_est, dtype=torch.long)
    edge_tgt = torch.empty(e_est, dtype=torch.long)

    # Backbone: node i → node i + w
    edge_src[:e_backbone] = torch.arange(0, e_backbone, dtype=torch.long)
    edge_tgt[:e_backbone] = torch.arange(w, n, dtype=torch.long)
    mem("backbone")

    # Cross edges in chunks
    write_pos = e_backbone
    chunk_size = 50_000_000
    for start in range(0, e_backbone, chunk_size):
        end = min(start + chunk_size, e_backbone)
        chunk_n = end - start
        mask = torch.rand(chunk_n) < cross_prob
        n_cross = mask.sum().item()
        if n_cross == 0:
            continue
        cross_src = torch.arange(start, end, dtype=torch.long)[mask]
        cross_offset = torch.randint(0, w, (n_cross,), dtype=torch.long)
        cross_tgt_layer = cross_src // w + 1
        cross_tgt = cross_tgt_layer * w + cross_offset
        edge_src[write_pos:write_pos + n_cross] = cross_src
        edge_tgt[write_pos:write_pos + n_cross] = cross_tgt
        write_pos += n_cross

    edge_index = torch.stack([edge_src[:write_pos], edge_tgt[:write_pos]])
    del edge_src, edge_tgt
    mem("edges done")
    return edge_index


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Large-scale layout benchmark")
    parser.add_argument(
        "size",
        help="Node count: preset name (50m, 100m, 300m, 1b) or number (e.g. 10_000_000, 5m)",
    )
    parser.add_argument("--layers", type=int, default=0, help="Number of layers (0 = auto)")
    parser.add_argument("--workers", type=int, default=4, help="Num parallel workers")
    parser.add_argument("--steps", type=int, default=500, help="Layout optimization steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Resolve size
    key = args.size.lower().replace("_", "").replace(",", "")
    if key in PRESETS:
        n = PRESETS[key]["n"]
        layers = args.layers if args.layers > 0 else PRESETS[key]["layers"]
    else:
        n = parse_node_count(args.size)
        layers = args.layers if args.layers > 0 else max(int(n**0.5 / 10) * 10, 10)

    w = n // layers
    n = w * layers  # round to exact multiple

    mem("start")
    print(f"Building wide DAG: {n:,} nodes, {layers} layers, ~{w:,} nodes/layer...", flush=True)
    t0 = time.perf_counter()

    edge_index = build_edges(n, layers)
    print(f"Edge index ready: {edge_index.shape[1]:,} edges in {time.perf_counter() - t0:.1f}s", flush=True)

    g = dagua.DaguaGraph()
    g.num_nodes = n
    g._edge_index_tensor = edge_index
    g.node_sizes = torch.full((n, 2), 20.0)
    mem("graph built")

    config = dagua.LayoutConfig(
        device="cpu",
        verbose=True,
        num_workers=args.workers,
        multilevel_threshold=50000,
        multilevel_min_nodes=2000,
        multilevel_coarse_steps=50,
        multilevel_refine_steps=15,
        steps=args.steps,
        seed=args.seed,
    )

    print(f"\nStarting layout (num_workers={config.num_workers})...", flush=True)
    t1 = time.perf_counter()
    pos = dagua.layout(g, config)
    total = time.perf_counter() - t1

    print(f"\nResult: {pos.shape}", flush=True)
    print(f"Total layout time: {total:.1f}s", flush=True)
    print(f"  x range: [{pos[:, 0].min():.0f}, {pos[:, 0].max():.0f}]", flush=True)
    print(f"  y range: [{pos[:, 1].min():.0f}, {pos[:, 1].max():.0f}]", flush=True)
    mem("done")


if __name__ == "__main__":
    main()
