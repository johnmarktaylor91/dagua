"""1B node benchmark — wide DAG, CPU-only, streaming coarsening."""

import gc
import os
import time
import torch
import dagua


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


LAYERS = 1500
NODES_PER_LAYER = 1_000_000_000 // LAYERS
N = NODES_PER_LAYER * LAYERS

mem("start")
print(f"Building wide DAG: {N:,} nodes, {LAYERS} layers, ~{NODES_PER_LAYER:,} nodes/layer...", flush=True)
t0 = time.perf_counter()

W = NODES_PER_LAYER

# Build edge_index in chunks to avoid peak memory from temporaries
E_backbone = N - W
E_cross_est = int(E_backbone * 0.06)  # 5% cross + margin — realistic for neural net DAGs
E_est = E_backbone + E_cross_est

# Pre-allocate final tensor
edge_src = torch.empty(E_est, dtype=torch.long)
edge_tgt = torch.empty(E_est, dtype=torch.long)

# Backbone edges: node i → node i + W
edge_src[:E_backbone] = torch.arange(0, E_backbone, dtype=torch.long)
edge_tgt[:E_backbone] = torch.arange(W, N, dtype=torch.long)
mem("backbone")

# Cross edges: build in chunks to avoid 8 GB arange + 4 GB mask simultaneously
write_pos = E_backbone
CHUNK = 50_000_000
for start in range(0, E_backbone, CHUNK):
    end = min(start + CHUNK, E_backbone)
    chunk_n = end - start
    mask = torch.rand(chunk_n) < 0.05  # 5% — realistic for neural net DAGs
    n_cross = mask.sum().item()
    if n_cross == 0:
        continue
    cross_src = torch.arange(start, end, dtype=torch.long)[mask]
    cross_offset = torch.randint(0, W, (n_cross,), dtype=torch.long)
    cross_tgt_layer = cross_src // W + 1
    cross_tgt = cross_tgt_layer * W + cross_offset
    edge_src[write_pos:write_pos + n_cross] = cross_src
    edge_tgt[write_pos:write_pos + n_cross] = cross_tgt
    write_pos += n_cross

# Trim to actual size
edge_index = torch.stack([edge_src[:write_pos], edge_tgt[:write_pos]])
del edge_src, edge_tgt

mem("edges done")
print(f"Edge index ready: {edge_index.shape[1]:,} edges in {time.perf_counter() - t0:.1f}s", flush=True)

g = dagua.DaguaGraph()
g.num_nodes = N
g._edge_index_tensor = edge_index
g.node_sizes = torch.full((N, 2), 20.0)
mem("graph built")

config = dagua.LayoutConfig(
    device="cpu",
    verbose=True,
    num_workers=4,
    multilevel_threshold=50000,
    multilevel_min_nodes=2000,
    multilevel_coarse_steps=50,
    multilevel_refine_steps=15,
    steps=500,
    seed=42,
)

print(f"\nStarting layout (num_workers={config.num_workers})...", flush=True)
t1 = time.perf_counter()
pos = dagua.layout(g, config)
total = time.perf_counter() - t1

print(f"\nResult: {pos.shape}", flush=True)
print(f"Total layout time: {total:.1f}s", flush=True)
print(f"  x range: [{pos[:, 0].min():.0f}, {pos[:, 0].max():.0f}]", flush=True)
print(f"  y range: [{pos[:, 1].min():.0f}, {pos[:, 1].max():.0f}]", flush=True)
