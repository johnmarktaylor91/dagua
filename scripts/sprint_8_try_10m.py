"""Sprint 8: characterize the 10M wall.

Sprint 8 exit target: 10M in <=45 min, <=120 GB RAM on GPU. This script
tries to run dagua.layout at N=10M on whatever GPU is available,
captures wall time + memory, and traces where it OOMs if it does.

On an RTX 2080 Ti with 11.5 GB VRAM, 1M already peaked at 8.87 GB of
allocated CUDA memory, so 10M is expected to OOM -- we want to know
WHERE (which phase / which op) so Sprint 8.5 can scope the offload
fix.
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import psutil
import torch

import dagua
from dagua.config import LayoutConfig

OUT_PATH = Path("eval_output/native_algo/sprint_8_try_10m/report.json")


def _make_wide_dag(n: int, seed: int = 42):
    torch.manual_seed(seed)
    layers = max(int(n**0.5 / 10) * 10, 10)
    width = max(n // layers, 1)
    n = width * layers
    src = torch.arange(0, n - width, dtype=torch.long)
    tgt = src + width
    cross_mask = torch.rand(n - width) < 0.5
    cross_src = torch.arange(0, n - width, dtype=torch.long)[cross_mask]
    cross_offset = torch.randint(0, width, (cross_src.shape[0],), dtype=torch.long)
    cross_tgt_layer = cross_src // width + 1
    cross_tgt = cross_tgt_layer * width + cross_offset
    edge_index = torch.stack([torch.cat([src, cross_src]), torch.cat([tgt, cross_tgt])])
    return edge_index, n


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("cuda not available; aborting")
        return

    props = torch.cuda.get_device_properties(0)
    print(f"gpu: {props.name}, {props.total_memory / 1e9:.1f} GB VRAM", flush=True)

    n = 10_000_000
    print(f"\nbuilding N={n:,} wide_dag...", flush=True)
    t = time.perf_counter()
    edge_index, actual_n = _make_wide_dag(n)
    print(
        f"  built in {time.perf_counter() - t:.1f}s: N={actual_n:,}, E={edge_index.shape[1]:,}",
        flush=True,
    )

    g = dagua.DaguaGraph()
    g.num_nodes = actual_n
    g._edge_index_tensor = edge_index
    g.node_sizes = torch.full((actual_n, 2), 20.0, dtype=torch.float16)
    cfg = LayoutConfig(device="cuda", verbose=False, seed=42)

    proc = psutil.Process()
    rss_start = proc.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    result = {
        "target_nodes": n,
        "actual_nodes": actual_n,
        "edges": int(edge_index.shape[1]),
        "gpu": props.name,
        "gpu_total_vram_gb": props.total_memory / 1e9,
    }

    try:
        pos = dagua.layout(g, cfg)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        rss_peak = proc.memory_info().rss
        result.update(
            {
                "status": "ok",
                "wall_s": wall,
                "wall_min": wall / 60,
                "vram_peak_alloc_gb": torch.cuda.max_memory_allocated() / 1e9,
                "vram_peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
                "rss_peak_gb": max(rss_start, rss_peak) / 1e9,
                "pos_device": str(pos.device),
            }
        )
        print(f"\nPASS: wall={wall:.1f}s ({wall / 60:.1f} min)", flush=True)
    except Exception as e:
        wall = time.perf_counter() - t0
        rss_peak = proc.memory_info().rss
        tb = traceback.format_exc()
        # Look in the traceback for where the last dagua op was
        tb_lines = tb.splitlines()
        last_dagua = [line for line in tb_lines if "dagua/" in line and "File " in line]
        result.update(
            {
                "status": "error",
                "error_type": type(e).__name__,
                "error_msg": str(e)[:500],
                "wall_s_before_crash": wall,
                "rss_at_crash_gb": max(rss_start, rss_peak) / 1e9,
                "vram_alloc_at_crash_gb": torch.cuda.memory_allocated() / 1e9
                if torch.cuda.is_available()
                else 0,
                "vram_peak_alloc_gb": torch.cuda.max_memory_allocated() / 1e9
                if torch.cuda.is_available()
                else 0,
                "last_dagua_frames": last_dagua[-8:],
            }
        )
        print(f"\nFAIL ({type(e).__name__}): {e}", flush=True)
        print(f"  wall before crash: {wall:.1f}s", flush=True)
        print(f"  vram alloc at crash: {result['vram_alloc_at_crash_gb']:.2f}GB", flush=True)
        print(f"  vram peak: {result['vram_peak_alloc_gb']:.2f}GB", flush=True)
        print("  last dagua frames:", flush=True)
        for line in last_dagua[-8:]:
            print(f"    {line.strip()}", flush=True)

    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
