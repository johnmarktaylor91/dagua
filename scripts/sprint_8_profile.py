"""Sprint 8 scaffolding: profile current layout perf + memory across a scale ladder.

Goal: identify the top bottlenecks on the path to the Sprint 8 exit
criteria (02_sprint_map.md L320-L327):
  * 1M nodes: <=8 minutes wall on GPU.
  * 10M nodes: <=45 minutes wall on GPU, peak RAM <=120 GB.

This script runs dagua.layout on a fixed graph variant (wide-dag) across
a size ladder {1K, 10K, 100K, 1M} capturing:
 - wall time (per-phase via cProfile cumulative time on top callees)
 - peak process RSS (psutil)
 - peak CUDA allocated + reserved (torch.cuda.memory_stats)
 - top 20 cumulative-time functions as a first look at what the big
   rocks are today.

Writes eval_output/native_algo/sprint_8_profile/report.json with the
full numeric data + per-size top-20 hot-spot table, and prints a
compact summary table to stdout.

This is scaffolding: it produces the baseline numbers Sprint 8 will
improve against. No code changes to the engine in this commit.
"""

from __future__ import annotations

import cProfile
import io
import json
import pstats
import time
from pathlib import Path

import psutil
import torch

import dagua

OUT_PATH = Path("eval_output/native_algo/sprint_8_profile/report.json")


def _make_wide_dag(n: int, seed: int = 42):
    """Wide DAG: uniform layers with backbone + 50% cross edges."""
    torch.manual_seed(seed)
    layers = max(int(n**0.5 / 10) * 10, 10)
    width = max(n // layers, 1)
    n = width * layers
    idx_dtype = torch.int32 if n <= 2_147_483_647 else torch.long
    src = torch.arange(0, n - width, dtype=idx_dtype)
    tgt = src + width
    cross_mask = torch.rand(n - width) < 0.5
    cross_src = torch.arange(0, n - width, dtype=idx_dtype)[cross_mask]
    cross_offset = torch.randint(0, width, (cross_src.shape[0],), dtype=idx_dtype)
    cross_tgt_layer = cross_src // width + 1
    cross_tgt = cross_tgt_layer * width + cross_offset
    edge_index = torch.stack(
        [
            torch.cat([src, cross_src]).to(torch.long),
            torch.cat([tgt, cross_tgt]).to(torch.long),
        ]
    )
    return edge_index, n, layers


def _build_graph(n: int, seed: int = 42):
    edge_index, actual_n, _ = _make_wide_dag(n, seed=seed)
    g = dagua.DaguaGraph()
    g.num_nodes = actual_n
    g._edge_index_tensor = edge_index
    g.node_sizes = torch.full((actual_n, 2), 20.0, dtype=torch.float16)
    return g, actual_n, edge_index.shape[1]


def _profile_one(n: int, device: str, seed: int = 42) -> dict:
    g, actual_n, n_edges = _build_graph(n, seed=seed)
    cfg = dagua.LayoutConfig(device=device, verbose=False, seed=seed)

    proc = psutil.Process()
    rss_start = proc.memory_info().rss
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    pr = cProfile.Profile()
    pr.enable()
    pos = dagua.layout(g, cfg)
    if device == "cuda":
        torch.cuda.synchronize()
    pr.disable()

    wall = time.perf_counter() - t0
    rss_peak = max(proc.memory_info().rss, rss_start)

    vram_peak_alloc = 0
    vram_peak_reserved = 0
    if device == "cuda":
        vram_peak_alloc = int(torch.cuda.max_memory_allocated())
        vram_peak_reserved = int(torch.cuda.max_memory_reserved())

    # Top 20 hot spots by cumulative time.
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(20)
    top_stats = buf.getvalue()

    # Parse pstats into structured data: (ncalls, tottime, cumtime, filename:lineno(function)).
    hot_spots = []
    for line in top_stats.splitlines():
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        try:
            ncalls = parts[0]
            tottime = float(parts[1])
            cumtime = float(parts[3])
            where = parts[5]
        except (ValueError, IndexError):
            continue
        hot_spots.append(
            {
                "ncalls": ncalls,
                "tottime_s": tottime,
                "cumtime_s": cumtime,
                "where": where,
            }
        )
        if len(hot_spots) >= 20:
            break

    return {
        "target_nodes": n,
        "actual_nodes": actual_n,
        "edges": n_edges,
        "wall_s": wall,
        "rss_peak_bytes": int(rss_peak),
        "rss_peak_gb": rss_peak / 1e9,
        "vram_peak_alloc_bytes": vram_peak_alloc,
        "vram_peak_alloc_gb": vram_peak_alloc / 1e9,
        "vram_peak_reserved_bytes": vram_peak_reserved,
        "vram_peak_reserved_gb": vram_peak_reserved / 1e9,
        "device": device,
        "hot_spots": hot_spots,
        "pos_device": str(pos.device),
    }


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"gpu: {props.name}, {props.total_memory / 1e9:.1f} GB VRAM")

    sizes = [1_000, 10_000, 100_000, 1_000_000]
    results = []
    for n in sizes:
        print(f"\n=== profiling N={n:,} ===", flush=True)
        try:
            r = _profile_one(n, device=device)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
            results.append({"target_nodes": n, "error": f"{type(e).__name__}: {e}"})
            if "OutOfMemory" in type(e).__name__:
                break
            continue
        results.append(r)
        print(
            f"  N={r['actual_nodes']:>9,} E={r['edges']:>9,} "
            f"wall={r['wall_s']:6.1f}s  rss={r['rss_peak_gb']:5.2f}GB  "
            f"vram(alloc)={r['vram_peak_alloc_gb']:5.2f}GB  "
            f"vram(resv)={r['vram_peak_reserved_gb']:5.2f}GB",
            flush=True,
        )
        print("  top 5 hot spots (cumulative):", flush=True)
        for hs in r["hot_spots"][:5]:
            print(
                f"    {hs['cumtime_s']:6.2f}s  {hs['where'][:80]}",
                flush=True,
            )

    payload = {
        "device": device,
        "sizes": sizes,
        "results": results,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {OUT_PATH}")

    # Summary table.
    print("\n=== Summary ===")
    print(f"{'N':>9}  {'E':>9}  {'wall':>7}  {'rss':>7}  {'vram_alloc':>10}  {'vram_resv':>10}")
    for r in results:
        if "error" in r:
            print(f"{r['target_nodes']:>9,}  (error: {r['error'][:40]})")
            continue
        print(
            f"{r['actual_nodes']:>9,}  {r['edges']:>9,}  "
            f"{r['wall_s']:>6.1f}s  {r['rss_peak_gb']:>5.2f}GB  "
            f"{r['vram_peak_alloc_gb']:>8.2f}GB  {r['vram_peak_reserved_gb']:>8.2f}GB"
        )

    # Sprint 8 target check.
    result_1m = next(
        (r for r in results if r.get("actual_nodes", 0) >= 900_000 and "error" not in r),
        None,
    )
    if result_1m:
        wall_budget_s = 8 * 60
        if result_1m["wall_s"] <= wall_budget_s:
            print(f"\nPASS: 1M wall {result_1m['wall_s']:.1f}s <= {wall_budget_s}s target")
        else:
            overshoot = result_1m["wall_s"] - wall_budget_s
            print(
                f"\nFAIL: 1M wall {result_1m['wall_s']:.1f}s > {wall_budget_s}s "
                f"target by {overshoot:.1f}s ({overshoot / wall_budget_s * 100:.0f}%)"
            )


if __name__ == "__main__":
    main()
