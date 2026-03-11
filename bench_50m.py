"""50M node benchmark — wide DAG (like a neural network), CPU-only."""

import time
import torch
import dagua

N = 50_000_000
LAYERS = 500        # 500 layers, ~100K nodes per layer
NODES_PER_LAYER = N // LAYERS

print(f"Building wide DAG: {N:,} nodes, {LAYERS} layers, ~{NODES_PER_LAYER:,} nodes/layer...", flush=True)
t0 = time.perf_counter()

# Build edges: each node connects to a random node in the next layer
# Layer i has nodes [i*W, (i+1)*W) where W = NODES_PER_LAYER
W = NODES_PER_LAYER

# Edge type 1: each node → same-position node in next layer (backbone)
src_backbone = torch.arange(0, N - W, dtype=torch.long)
tgt_backbone = src_backbone + W

# Edge type 2: each node → random offset in next layer (cross-connections, ~50% of nodes)
cross_mask = torch.rand(N - W) < 0.5
cross_src = torch.arange(0, N - W, dtype=torch.long)[cross_mask]
cross_offset = torch.randint(0, W, (cross_src.shape[0],), dtype=torch.long)
# Target = next layer base + random offset within layer
cross_tgt_layer = cross_src // W + 1
cross_tgt = cross_tgt_layer * W + cross_offset

edge_index = torch.stack([
    torch.cat([src_backbone, cross_src]),
    torch.cat([tgt_backbone, cross_tgt]),
])
del src_backbone, tgt_backbone, cross_src, cross_offset, cross_tgt_layer, cross_tgt, cross_mask

print(f"Edge index ready: {edge_index.shape[1]:,} edges in {time.perf_counter() - t0:.1f}s", flush=True)

# Minimal graph — pre-set uniform node sizes, skip labels
g = dagua.DaguaGraph()
g.num_nodes = N
g._edge_index_tensor = edge_index
g.node_sizes = torch.full((N, 2), 20.0)

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
