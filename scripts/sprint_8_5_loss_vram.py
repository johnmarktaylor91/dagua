"""Sprint 8.5: profile per-loss peak VRAM on the finest level at 1M.

10M OOM'd inside ``LossGroup.term.backward()`` at 9.07 GB on an 11.5 GB
card. We need to know WHICH loss's forward + backward has the biggest
footprint so we can chunk/checkpoint the right one first.

Approach: run dagua.layout at 1M (doable on this card), patch every
LossOp.evaluate + the immediately-following term.backward() to record
the allocated-VRAM delta per loss call. Peak delta per loss tells us
which ones dominate the VRAM pressure.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import torch

import dagua
from dagua.config import LayoutConfig
from dagua.layout.ops.base import LossOp

OUT_PATH = Path("eval_output/native_algo/sprint_8_5_loss_vram/report.json")


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
    if not torch.cuda.is_available():
        print("cuda required")
        return
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    n = 1_000_000
    edge_index, actual_n = _make_wide_dag(n)
    g = dagua.DaguaGraph()
    g.num_nodes = actual_n
    g._edge_index_tensor = edge_index
    g.node_sizes = torch.full((actual_n, 2), 20.0, dtype=torch.float16)
    cfg = LayoutConfig(device="cuda", verbose=False, seed=42)

    # Per-loss peak allocation deltas across all evaluate calls.
    peaks: dict[str, dict] = defaultdict(
        lambda: {
            "calls": 0,
            "peak_fwd_delta_gb": 0.0,
            "max_abs_alloc_at_call_gb": 0.0,
            "sum_fwd_delta_gb": 0.0,
        }
    )

    patches = []

    def _all_subclasses(cls):
        seen = set()
        stack = [cls]
        while stack:
            c = stack.pop()
            for s in c.__subclasses__():
                if s in seen:
                    continue
                seen.add(s)
                stack.append(s)
                yield s

    def wrap(cls):
        if "evaluate" not in cls.__dict__:
            return
        orig = cls.__dict__["evaluate"]

        def patched(self, problem, state, ctx, _orig=orig, _cls=cls):
            name = getattr(self, "name", _cls.__name__)
            torch.cuda.synchronize()
            alloc_before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            result = _orig(self, problem, state, ctx)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            fwd_delta = (peak - alloc_before) / 1e9
            entry = peaks[name]
            entry["calls"] += 1
            entry["sum_fwd_delta_gb"] += fwd_delta
            if fwd_delta > entry["peak_fwd_delta_gb"]:
                entry["peak_fwd_delta_gb"] = fwd_delta
            alloc_at_call = alloc_before / 1e9
            if alloc_at_call > entry["max_abs_alloc_at_call_gb"]:
                entry["max_abs_alloc_at_call_gb"] = alloc_at_call
            return result

        setattr(cls, "evaluate", patched)
        patches.append((cls, "evaluate", orig))

    import importlib

    importlib.import_module("dagua.layout.ops")
    for cls in _all_subclasses(LossOp):
        wrap(cls)

    print(f"instrumenting {len(patches)} loss ops at N={actual_n:,} ...", flush=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        dagua.layout(g, cfg)
        torch.cuda.synchronize()
    finally:
        for cls, name, orig in patches:
            setattr(cls, name, orig)
    wall = time.perf_counter() - t0
    vram_peak = torch.cuda.max_memory_allocated() / 1e9

    # Sort losses by peak_fwd_delta descending.
    rows = sorted(
        [
            {
                "loss": name,
                **stats,
                "mean_fwd_delta_gb": stats["sum_fwd_delta_gb"] / max(stats["calls"], 1),
            }
            for name, stats in peaks.items()
        ],
        key=lambda r: -r["peak_fwd_delta_gb"],
    )

    payload = {
        "n": actual_n,
        "wall_s": wall,
        "vram_peak_gb": vram_peak,
        "per_loss": rows,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nwall={wall:.1f}s  vram_peak={vram_peak:.2f}GB")
    print(f"wrote {OUT_PATH}\n")
    print("Per-loss peak forward-pass VRAM delta (top 15):")
    print(
        f"  {'loss':<30}  {'calls':>6}  {'peak_fwd_delta':>14}  "
        f"{'mean_fwd':>10}  {'max_abs_alloc':>14}"
    )
    for r in rows[:15]:
        print(
            f"  {r['loss'][:30]:<30}  {r['calls']:>6}  "
            f"{r['peak_fwd_delta_gb']:>12.3f}GB  "
            f"{r['mean_fwd_delta_gb']:>8.3f}GB  "
            f"{r['max_abs_alloc_at_call_gb']:>12.3f}GB"
        )


if __name__ == "__main__":
    main()
