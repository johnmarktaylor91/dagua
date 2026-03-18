"""Pre-compute graph + layering for large benchmark sizes.

Generates the synthetic graph and runs longest-path layering, saving
checkpoints so bench_large.py --resume skips both steps entirely.

Usage:
    python scripts/precompute_layering.py 1.5b 2b
    python scripts/precompute_layering.py 1500000000 2000000000
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import torch


def rss_gb() -> float:
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 1024**3
    except Exception:
        return 0.0


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(
        description="Pre-compute graph + layering for large benchmarks"
    )
    parser.add_argument(
        "sizes",
        nargs="+",
        help="Node counts: preset names (1.5b, 2b) or numbers (1500000000)",
    )
    args = parser.parse_args()

    # Import bench_large helpers
    bench_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(bench_dir))
    from bench_large import (
        _atomic_torch_save,
        _checkpoint_paths,
        _default_checkpoint_dir,
        _save_checkpoint_meta,
        build_edges,
        resolve_size_and_layers,
    )

    from dagua.utils import longest_path_layering

    for size_str in args.sizes:
        print(f"\n{'=' * 60}")
        print(f"Pre-computing: {size_str}")
        print(f"{'=' * 60}")
        t0 = time.perf_counter()

        n, layers, w = resolve_size_and_layers(size_str)
        checkpoint_dir = _default_checkpoint_dir(size_str)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        paths = _checkpoint_paths(checkpoint_dir)

        print(f"  Nodes: {n:,}, Layers: {layers:,}, Width: {w:,}")
        print(f"  Checkpoint: {checkpoint_dir}")

        # --- Graph generation ---
        if paths["edge_index"].exists() and paths["node_sizes"].exists():
            print("  Graph checkpoint exists, loading...")
            edge_index = torch.load(paths["edge_index"], map_location="cpu")
            node_sizes = torch.load(paths["node_sizes"], map_location="cpu")
            print(f"  Loaded: {edge_index.shape[1]:,} edges  [RSS={rss_gb():.1f}GB]")
        else:
            print("  Building edges...")
            edge_index = build_edges(n, layers)
            node_sizes = torch.full((n, 2), 20.0, dtype=torch.float16)
            print(
                f"  Built {edge_index.shape[1]:,} edges in "
                f"{time.perf_counter() - t0:.1f}s  [RSS={rss_gb():.1f}GB]"
            )
            _atomic_torch_save(paths["edge_index"], edge_index)
            _atomic_torch_save(paths["node_sizes"], node_sizes)
            _save_checkpoint_meta(
                paths["meta"],
                {"size": size_str, "n": n, "layers": layers, "width": w},
            )
            print("  Saved graph checkpoint")

        # --- Layering ---
        if paths["layer_assignments"].exists():
            print("  Layer assignments checkpoint exists, skipping")
        else:
            print(f"  Computing layering ({n:,} nodes, {edge_index.shape[1]:,} edges)...")
            t_layer = time.perf_counter()
            layer_assignments = longest_path_layering(edge_index, n, device="cpu", verbose=True)
            if isinstance(layer_assignments, list):
                layer_assignments = torch.tensor(
                    layer_assignments,
                    dtype=torch.int32 if n <= 2**31 else torch.long,
                )
            elapsed = time.perf_counter() - t_layer
            max_layer = int(layer_assignments.max().item())
            print(
                f"  Layering done: {max_layer + 1:,} layers in "
                f"{elapsed:.1f}s  [RSS={rss_gb():.1f}GB]"
            )
            _atomic_torch_save(paths["layer_assignments"], layer_assignments)
            print("  Saved layer assignments checkpoint")

        total = time.perf_counter() - t0
        print(f"  Total: {total:.1f}s")

        # Free memory before next size
        del edge_index, node_sizes
        gc.collect()

    print("\nAll done.")


if __name__ == "__main__":
    main()
