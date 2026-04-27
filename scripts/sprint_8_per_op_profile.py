"""Sprint 8: per-op instrumented profile at N=1M.

cProfile cumulative time is useless for our layout pipeline because the
hot loop goes through several layers of Op.apply wrappers that each
dominate cumtime. We want time PER concrete op (RepulsionLoss,
PeriodicOverlapProjection, HeavyEdgeMatching, CreateOptimizer, ...) so
we can see where the 636s of a 1M run is actually going.

Approach: monkey-patch Op.apply and LossOp.evaluate to record
(start_time, end_time, op_name) with CUDA sync at both ends. Aggregate
by op name; report top 20 by total time.

Runs ONCE at N=1,000,000 on wide_dag, writes
eval_output/native_algo/sprint_8_per_op/report.json.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import torch

import dagua
from dagua.config import LayoutConfig
from dagua.layout.ops.base import LossOp, Op

OUT_PATH = Path("eval_output/native_algo/sprint_8_per_op/report.json")


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
        print("CUDA required")
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    edge_index, actual_n = _make_wide_dag(1_000_000)
    g = dagua.DaguaGraph()
    g.num_nodes = actual_n
    g._edge_index_tensor = edge_index
    g.node_sizes = torch.full((actual_n, 2), 20.0, dtype=torch.float16)
    cfg = LayoutConfig(device="cuda", verbose=False, seed=42)

    # Per-op timings: op_name -> list of elapsed seconds per call.
    timings: dict[str, list[float]] = defaultdict(list)

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

    # Monkey-patching Op.apply alone doesn't fire for subclasses that
    # override apply -- Python looks up apply in the subclass dict first.
    # We walk every concrete Op / LossOp subclass and patch its apply /
    # evaluate individually, then restore afterward.
    patches: list[tuple[type, str, object]] = []

    def wrap(cls, method_name, key_prefix):
        if method_name not in cls.__dict__:
            return  # method inherited; skip so we don't double-count
        orig = cls.__dict__[method_name]

        def patched(self, problem, state, ctx, _orig=orig, _cls=cls):
            op_name = getattr(self, "name", _cls.__name__)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                return _orig(self, problem, state, ctx)
            finally:
                torch.cuda.synchronize()
                timings[f"{key_prefix}::{op_name}"].append(time.perf_counter() - t0)

        setattr(cls, method_name, patched)
        patches.append((cls, method_name, orig))

    # Ensure every Op subclass is imported so __subclasses__ sees it.
    # (Using importlib avoids shadowing the module-level `dagua` name.)
    import importlib

    importlib.import_module("dagua.layout.ops")

    for cls in _all_subclasses(Op):
        wrap(cls, "apply", "apply")
    for cls in _all_subclasses(LossOp):
        wrap(cls, "evaluate", "loss")

    print(f"instrumented run at N={actual_n:,} ({len(patches)} ops patched) ...", flush=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        dagua.layout(g, cfg)
        torch.cuda.synchronize()
    finally:
        for cls, method_name, orig in patches:
            setattr(cls, method_name, orig)
    wall = time.perf_counter() - t0
    vram_peak = torch.cuda.max_memory_allocated() / 1e9

    # Aggregate.
    agg = []
    for op_key, times in timings.items():
        agg.append(
            {
                "op": op_key,
                "calls": len(times),
                "total_s": sum(times),
                "mean_ms": 1e3 * sum(times) / max(len(times), 1),
                "max_ms": 1e3 * max(times),
            }
        )
    agg.sort(key=lambda r: -r["total_s"])

    payload = {
        "n": actual_n,
        "wall_s": wall,
        "vram_peak_gb": vram_peak,
        "ops": agg,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))

    print(f"\nwall={wall:.1f}s  vram_peak={vram_peak:.2f}GB")
    print(f"wrote {OUT_PATH}")
    print("\nTop 25 ops by total wall time:")
    print(f"  {'op':<52}  {'calls':>8}  {'total_s':>9}  {'mean_ms':>9}  {'max_ms':>9}")
    for r in agg[:25]:
        print(
            f"  {r['op'][:52]:<52}  {r['calls']:>8}  "
            f"{r['total_s']:>8.2f}s  {r['mean_ms']:>8.2f}  {r['max_ms']:>8.2f}"
        )

    # The apply:: rows include nested ops' time, so they'll top the
    # list. Emit a second view: LEAF ops only (loss ops + primitive
    # op_names that don't wrap inner pipelines). This is the actionable
    # target list.
    wrapper_names = {
        "apply::Pipeline",
        "apply::Repeat",
        "apply::LossGroup",
        "apply::Parallel",
        "apply::FixedSteps",
        "apply::fixed_steps",
        "apply::gradient_core",
        "apply::dagua_native_pipeline_vcycle",
        "apply::vcycle_refine",
        "apply::VCycleRefine",
    }
    print("\nTop 25 LEAF ops (wrappers excluded):")
    leaves = [r for r in agg if r["op"] not in wrapper_names][:25]
    for r in leaves:
        print(
            f"  {r['op'][:52]:<52}  {r['calls']:>8}  "
            f"{r['total_s']:>8.2f}s  {r['mean_ms']:>8.2f}  {r['max_ms']:>8.2f}"
        )


if __name__ == "__main__":
    main()
