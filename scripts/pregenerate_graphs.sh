#!/usr/bin/env bash
# Pre-generate graph structures for all ladder sizes.
# Creates edge_index.pt and node_sizes.pt on the locker for each size.
# Subsequent bench_large.py --resume runs skip graph construction entirely.
set -euo pipefail

SIZES=(10 20 50 100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000 1000000 2000000 5000000 10000000 20000000 50000000 100000000 200000000 500000000 1000000000)

echo "Pre-generating graph structures for ${#SIZES[@]} sizes..."

for SIZE in "${SIZES[@]}"; do
    CKPT="/mnt/locker/jt3295/dagua_bench_large/$SIZE"
    if [ -f "$CKPT/edge_index.pt" ] && [ -f "$CKPT/node_sizes.pt" ]; then
        echo "  $SIZE: already cached, skipping"
        continue
    fi
    echo "  $SIZE: generating..."
    PYTHONUNBUFFERED=1 python -c "
import torch
from scripts.bench_large import resolve_size_and_layers, build_edges, _default_checkpoint_dir, _checkpoint_paths, _save_checkpoint_meta, _atomic_torch_save

size = '$SIZE'
n, layers, w = resolve_size_and_layers(size)
print(f'  Building {n:,} nodes, {layers} layers, {w:,} width...')

checkpoint_dir = _default_checkpoint_dir(size)
paths = _checkpoint_paths(checkpoint_dir)

edge_index = build_edges(n, layers)
node_sizes = torch.full((n, 2), 20.0, dtype=torch.float16)

checkpoint_dir.mkdir(parents=True, exist_ok=True)
_atomic_torch_save(paths['edge_index'], edge_index)
_atomic_torch_save(paths['node_sizes'], node_sizes)
_save_checkpoint_meta(paths['meta'], {
    'size': size, 'n': n, 'layers': layers, 'width': w,
    'device': 'cpu', 'seed': 42,
})
print(f'  Saved to {checkpoint_dir}')
print(f'  Edges: {edge_index.shape[1]:,}, Size: {edge_index.numel()*4/1e6:.0f} MB')
" 2>&1
done

echo "Done. All graph structures cached on locker."
