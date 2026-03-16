#!/usr/bin/env python3
"""Generate layouts from dagua's classic reimplementations on the same test graphs.

Mirrors generate_ground_truth.py but uses our reimplementations instead of
the original competitor packages. Output is saved alongside the ground truth
for direct comparison.

Usage:
    python scripts/generate_reimpl_layouts.py
    python scripts/generate_reimpl_layouts.py --output-dir eval_output/reimpl_layouts
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch

# ─── Reimplementation registry ───────────────────────────────────────────────

# Map: reimpl name -> (import path, function name, is_stochastic, accepts_clusters)
# The reimpl name should match the classic_* competitor name for easy pairing
REIMPLEMENTATIONS: Dict[str, dict] = {}


def _register(
    name: str,
    module: str,
    function: str,
    stochastic: bool = True,
    accepts_clusters: bool = False,
    original_competitor: str = "",
    max_nodes: int = 50_000,
):
    """Register a reimplementation for ground truth comparison."""
    REIMPLEMENTATIONS[name] = {
        "module": module,
        "function": function,
        "stochastic": stochastic,
        "accepts_clusters": accepts_clusters,
        "original_competitor": original_competitor,
        "max_nodes": max_nodes,
    }


# Force-directed family
_register(
    "classic_fr",
    "dagua.layout.classic.fr",
    "layout_fr",
    stochastic=True,
    original_competitor="nx_spring",
    max_nodes=50_000,
)
_register(
    "classic_kk",
    "dagua.layout.classic.kk",
    "layout_kk",
    stochastic=True,
    original_competitor="nx_kamada_kawai",
    max_nodes=50_000,
)
_register(
    "classic_fa2",
    "dagua.layout.classic.fa2",
    "layout_fa2",
    stochastic=True,
    original_competitor="",
    max_nodes=50_000,
)
_register(
    "classic_gem",
    "dagua.layout.classic.gem",
    "layout_gem",
    stochastic=True,
    original_competitor="igraph_fr",
    max_nodes=50_000,
)

# Stress-based family
_register(
    "classic_stress_sgd",
    "dagua.layout.classic.stress_sgd",
    "layout_stress_sgd",
    stochastic=True,
    original_competitor="",
    max_nodes=50_000,
)
_register(
    "classic_maxent_stress",
    "dagua.layout.classic.maxent_stress",
    "layout_maxent_stress",
    stochastic=True,
    original_competitor="",
    max_nodes=100_000,
)

# Spectral / MDS family
_register(
    "classic_spectral",
    "dagua.layout.classic.spectral",
    "layout_spectral",
    stochastic=False,
    original_competitor="",
    max_nodes=100_000,
)
_register(
    "classic_pivot_mds",
    "dagua.layout.classic.pivot_mds",
    "layout_pivot_mds",
    stochastic=False,
    original_competitor="",
    max_nodes=500_000,
)

# Hierarchical / layered
_register(
    "classic_sugiyama",
    "dagua.layout.classic.sugiyama",
    "layout_sugiyama",
    stochastic=False,
    accepts_clusters=False,
    original_competitor="graphviz_dot",
    max_nodes=50_000,
)

# Information-theoretic
_register(
    "classic_tsnet",
    "dagua.layout.classic.tsnet",
    "layout_tsnet",
    stochastic=True,
    original_competitor="",
    max_nodes=10_000,
)

# Simulated annealing
_register(
    "classic_davidson_harel",
    "dagua.layout.classic.davidson_harel",
    "layout_davidson_harel",
    stochastic=True,
    original_competitor="",
    max_nodes=500,
)

# LinLog (community-revealing)
_register(
    "classic_linlog",
    "dagua.layout.classic.linlog",
    "layout_linlog",
    stochastic=True,
    original_competitor="",
    max_nodes=50_000,
)

# FM^3 (multilevel + multipole)
_register(
    "classic_fmmm",
    "dagua.layout.classic.fmmm",
    "layout_fmmm",
    stochastic=True,
    original_competitor="",
    max_nodes=500_000,
)


# ─── Data structures ─────────────────────────────────────────────────────────

STOCHASTIC_SEEDS = list(range(42, 52))  # 10 seeds, matching ground truth


@dataclass
class ReimplRecord:
    """Record of a single reimplementation layout attempt."""

    graph_name: str
    reimpl_name: str
    seed: Optional[int]
    success: bool
    error: Optional[str]
    runtime_seconds: float
    positions_file: Optional[str]
    num_nodes: int
    num_edges: int
    has_clusters: bool
    clusters_passed: bool
    is_stochastic: bool
    original_competitor: str
    input_summary: Dict[str, Any] = field(default_factory=dict)


def _set_seed(seed: int) -> None:
    """Set all random seeds."""
    torch.manual_seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import random

        random.seed(seed)
    except ImportError:
        pass


def _get_layout_fn(info: dict) -> Callable:
    """Dynamically import and return the layout function."""
    import importlib

    mod = importlib.import_module(info["module"])
    return getattr(mod, info["function"])


def _run_single(
    layout_fn: Callable,
    graph,
    reimpl_name: str,
    info: dict,
    seed: Optional[int],
    timeout: float,
) -> Tuple[Optional[ReimplRecord], Optional[torch.Tensor]]:
    """Run a single reimplementation layout."""
    graph_name = getattr(graph, "_test_name", "unknown")
    has_clusters = bool(getattr(graph, "clusters", None))
    clusters_passed = has_clusters and info["accepts_clusters"]

    if seed is not None:
        _set_seed(seed)

    edge_index = graph.edge_index
    num_nodes = graph.num_nodes
    node_sizes = graph.node_sizes
    num_edges = int(edge_index.shape[1]) if edge_index.numel() > 0 else 0

    input_summary = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "has_clusters": has_clusters,
        "clusters_passed": clusters_passed,
        "seed": seed,
        "reimpl_module": info["module"],
        "reimpl_function": info["function"],
    }

    t0 = time.perf_counter()
    try:
        # Build kwargs based on what the function accepts
        kwargs = {"seed": seed} if seed is not None else {"seed": 42}

        pos = layout_fn(edge_index, num_nodes, node_sizes=node_sizes, **kwargs)
        elapsed = time.perf_counter() - t0

        if pos is not None and pos.shape == (num_nodes, 2):
            record = ReimplRecord(
                graph_name=graph_name,
                reimpl_name=reimpl_name,
                seed=seed,
                success=True,
                error=None,
                runtime_seconds=elapsed,
                positions_file=None,
                num_nodes=num_nodes,
                num_edges=num_edges,
                has_clusters=has_clusters,
                clusters_passed=clusters_passed,
                is_stochastic=info["stochastic"],
                original_competitor=info["original_competitor"],
                input_summary=input_summary,
            )
            return record, pos
        else:
            record = ReimplRecord(
                graph_name=graph_name,
                reimpl_name=reimpl_name,
                seed=seed,
                success=False,
                error=f"Bad shape: {pos.shape if pos is not None else 'None'}",
                runtime_seconds=time.perf_counter() - t0,
                positions_file=None,
                num_nodes=num_nodes,
                num_edges=num_edges,
                has_clusters=has_clusters,
                clusters_passed=clusters_passed,
                is_stochastic=info["stochastic"],
                original_competitor=info["original_competitor"],
                input_summary=input_summary,
            )
            return record, None
    except Exception as e:
        elapsed = time.perf_counter() - t0
        record = ReimplRecord(
            graph_name=graph_name,
            reimpl_name=reimpl_name,
            seed=seed,
            success=False,
            error=f"{type(e).__name__}: {e}",
            runtime_seconds=elapsed,
            positions_file=None,
            num_nodes=num_nodes,
            num_edges=num_edges,
            has_clusters=has_clusters,
            clusters_passed=clusters_passed,
            is_stochastic=info["stochastic"],
            original_competitor=info["original_competitor"],
            input_summary=input_summary,
        )
        return record, None


def main():
    """Generate reimplementation layouts."""
    parser = argparse.ArgumentParser(description="Generate reimplementation layouts")
    parser.add_argument("--output-dir", default="eval_output/reimpl_layouts")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--stochastic-seeds", type=int, default=10)
    parser.add_argument("--max-nodes", type=int, default=500)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(42, 42 + args.stochastic_seeds))

    # Load test graphs
    from dagua.eval.graphs import get_test_graphs

    all_graphs = get_test_graphs(max_nodes=args.max_nodes)
    print(f"Loaded {len(all_graphs)} test graphs (max_nodes={args.max_nodes})")
    print(f"Reimplementations: {len(REIMPLEMENTATIONS)}")
    for name, info in REIMPLEMENTATIONS.items():
        s = "stochastic" if info["stochastic"] else "deterministic"
        c = "+clusters" if info["accepts_clusters"] else ""
        orig = f" (vs {info['original_competitor']})" if info["original_competitor"] else ""
        print(f"  {name:30s} {s:15s} {c} {orig}")

    all_records = []
    total = 0
    ok = 0
    fail = 0

    for gi, test_graph in enumerate(all_graphs):
        graph = test_graph.graph
        graph._test_name = test_graph.name
        graph.compute_node_sizes()

        graph_dir = output_dir / test_graph.name.replace(" ", "_").replace("/", "_")
        graph_dir.mkdir(parents=True, exist_ok=True)

        ne = int(graph.edge_index.shape[1]) if graph.edge_index.numel() > 0 else 0
        print(f"\n[{gi + 1}/{len(all_graphs)}] {test_graph.name} (N={graph.num_nodes}, E={ne})")

        graph_records = []

        for reimpl_name, info in REIMPLEMENTATIONS.items():
            if info["max_nodes"] > 0 and graph.num_nodes > info["max_nodes"]:
                print(f"  {reimpl_name:30s} SKIPPED (max={info['max_nodes']})")
                continue

            try:
                layout_fn = _get_layout_fn(info)
            except (ImportError, AttributeError) as e:
                print(f"  {reimpl_name:30s} IMPORT ERROR: {e}")
                continue

            if info["stochastic"]:
                seed_ok = 0
                for seed in seeds:
                    record, pos = _run_single(
                        layout_fn, graph, reimpl_name, info, seed, args.timeout
                    )
                    total += 1
                    if pos is not None:
                        fname = f"{reimpl_name}_seed{seed}.pt"
                        torch.save(pos, graph_dir / fname)
                        record.positions_file = str(graph_dir / fname)
                        ok += 1
                        seed_ok += 1
                    else:
                        fail += 1
                    graph_records.append(record)
                avg_t = sum(
                    r.runtime_seconds
                    for r in graph_records
                    if r.reimpl_name == reimpl_name and r.success
                ) / max(seed_ok, 1)
                print(f"  {reimpl_name:30s} {seed_ok}/{len(seeds)} seeds OK ({avg_t:.2f}s avg)")
            else:
                record, pos = _run_single(layout_fn, graph, reimpl_name, info, None, args.timeout)
                total += 1
                if pos is not None:
                    fname = f"{reimpl_name}.pt"
                    torch.save(pos, graph_dir / fname)
                    record.positions_file = str(graph_dir / fname)
                    ok += 1
                    print(f"  {reimpl_name:30s} OK ({record.runtime_seconds:.2f}s)")
                else:
                    fail += 1
                    print(f"  {reimpl_name:30s} FAILED: {record.error}")
                graph_records.append(record)

        all_records.extend(graph_records)

    # Save manifest
    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "settings": {
            "timeout": args.timeout,
            "stochastic_seeds": seeds,
            "max_nodes": args.max_nodes,
            "num_graphs": len(all_graphs),
            "num_reimplementations": len(REIMPLEMENTATIONS),
        },
        "summary": {
            "total": total,
            "success": ok,
            "failed": fail,
            "success_rate": ok / max(total, 1),
        },
        "reimplementations": {k: v for k, v in REIMPLEMENTATIONS.items()},
        "records": [asdict(r) for r in all_records],
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print("REIMPLEMENTATION LAYOUTS COMPLETE")
    print(f"{'=' * 60}")
    print(f"Layouts: {ok}/{total} succeeded ({fail} failed)")
    print(f"Output:  {output_dir}")
    print(f"Manifest: {manifest_path}")

    failures = [r for r in all_records if not r.success]
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for r in failures[:20]:
            seed_str = f" seed={r.seed}" if r.seed is not None else ""
            print(f"  {r.graph_name} / {r.reimpl_name}{seed_str}: {r.error[:80]}")


if __name__ == "__main__":
    main()
