#!/usr/bin/env python3
"""Generate ground truth layouts from all competitor algorithms on all test graphs.

One-time data generation for validating dagua's reimplementations against
original competitor algorithms. Saves position tensors + detailed metadata.

Usage:
    python scripts/generate_ground_truth.py
    python scripts/generate_ground_truth.py --output-dir eval_output/ground_truth
    python scripts/generate_ground_truth.py --timeout 120 --stochastic-seeds 10
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import torch

# ─── Stochastic competitor detection ─────────────────────────────────────────

# Competitors with random initialization or stochastic optimization.
# These need multiple seeds to characterize their output distribution.
STOCHASTIC_COMPETITORS: Set[str] = {
    # Classic reimplementations (all use random init or stochastic optimization)
    "classic_fr",
    "classic_kk",
    "classic_fa2",
    "classic_stress_sgd",
    "classic_linlog",
    "classic_gem",
    "classic_tsnet",
    "classic_maxent_stress",
    "classic_davidson_harel",
    "classic_fmmm",
    # External library wrappers (force-directed = random init)
    "nx_spring",
    "igraph_fr",
    # Graphviz force-directed variants (may have internal randomness)
    "graphviz_fdp",
    "graphviz_sfdp",
}

# Competitors that are deterministic for a given input.
DETERMINISTIC_COMPETITORS: Set[str] = {
    "dagua",
    "graphviz_dot",
    "graphviz_neato",
    "elk_layered",
    "dagre",
    "nx_kamada_kawai",
    "igraph_sugiyama",
    "igraph_rt",
    "classic_sugiyama",
    "classic_spectral",
    "classic_pivot_mds",
}

# Competitors that support cluster/compound layout.
CLUSTER_CAPABLE: Set[str] = {
    "dagua",
    "graphviz_dot",
    "elk_layered",
    "dagre",
}


# ─── Data structures ─────────────────────────────────────────────────────────


@dataclass
class LayoutRecord:
    """Record of a single layout generation attempt."""

    graph_name: str
    competitor_name: str
    seed: Optional[int]  # None for deterministic competitors
    success: bool
    error: Optional[str]
    runtime_seconds: float
    positions_file: Optional[str]  # relative path to .pt file
    num_nodes: int
    num_edges: int
    has_clusters: bool
    clusters_passed: bool  # whether clusters were passed to this competitor
    is_stochastic: bool
    input_summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphRecord:
    """Record of all layout attempts for a single graph."""

    graph_name: str
    num_nodes: int
    num_edges: int
    has_clusters: bool
    cluster_names: List[str]
    tags: List[str]
    layouts: List[LayoutRecord] = field(default_factory=list)


# ─── Main logic ──────────────────────────────────────────────────────────────


def _set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
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


def _run_single_layout(
    competitor,
    graph,
    timeout: float,
    seed: Optional[int] = None,
) -> LayoutRecord:
    """Run a single competitor on a single graph, return a LayoutRecord."""

    graph_name = getattr(graph, "_test_name", "unknown")
    has_clusters = bool(getattr(graph, "clusters", None))
    clusters_passed = has_clusters and competitor.name in CLUSTER_CAPABLE
    is_stochastic = competitor.name in STOCHASTIC_COMPETITORS

    # For stochastic competitors, set the seed before running
    if seed is not None:
        _set_seed(seed)

    # Build input summary for sanity checking
    input_summary = {
        "num_nodes": graph.num_nodes,
        "num_edges": int(graph.edge_index.shape[1]) if graph.edge_index.numel() > 0 else 0,
        "has_labels": bool(graph.node_labels),
        "has_clusters": has_clusters,
        "clusters_passed": clusters_passed,
        "competitor_max_nodes": competitor.max_nodes,
        "seed": seed,
        "timeout": timeout,
    }

    t0 = time.perf_counter()
    try:
        # If competitor doesn't support clusters and graph has them,
        # we still pass the graph (clusters are just metadata that
        # non-cluster competitors ignore)
        result = competitor.layout(graph, timeout=timeout)
        elapsed = time.perf_counter() - t0

        if result.pos is not None and result.error is None:
            return LayoutRecord(
                graph_name=graph_name,
                competitor_name=competitor.name,
                seed=seed,
                success=True,
                error=None,
                runtime_seconds=elapsed,
                positions_file=None,  # filled in later when saving
                num_nodes=graph.num_nodes,
                num_edges=input_summary["num_edges"],
                has_clusters=has_clusters,
                clusters_passed=clusters_passed,
                is_stochastic=is_stochastic,
                input_summary=input_summary,
            ), result.pos
        else:
            return LayoutRecord(
                graph_name=graph_name,
                competitor_name=competitor.name,
                seed=seed,
                success=False,
                error=result.error or "No positions returned",
                runtime_seconds=elapsed,
                positions_file=None,
                num_nodes=graph.num_nodes,
                num_edges=input_summary["num_edges"],
                has_clusters=has_clusters,
                clusters_passed=clusters_passed,
                is_stochastic=is_stochastic,
                input_summary=input_summary,
            ), None
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return LayoutRecord(
            graph_name=graph_name,
            competitor_name=competitor.name,
            seed=seed,
            success=False,
            error=f"{type(e).__name__}: {e}",
            runtime_seconds=elapsed,
            positions_file=None,
            num_nodes=graph.num_nodes,
            num_edges=input_summary["num_edges"],
            has_clusters=has_clusters,
            clusters_passed=clusters_passed,
            is_stochastic=is_stochastic,
            input_summary=input_summary,
        ), None


def main():
    """Generate ground truth layouts."""
    parser = argparse.ArgumentParser(description="Generate competitor ground truth layouts")
    parser.add_argument("--output-dir", default="eval_output/ground_truth")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-layout timeout (seconds)")
    parser.add_argument(
        "--stochastic-seeds", type=int, default=10, help="Seeds for stochastic algos"
    )
    parser.add_argument("--max-nodes", type=int, default=500, help="Skip graphs larger than this")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all test graphs
    from dagua.eval.graphs import get_test_graphs

    all_graphs = get_test_graphs(max_nodes=args.max_nodes)
    print(f"Loaded {len(all_graphs)} test graphs (max_nodes={args.max_nodes})")

    # Get available competitors
    from dagua.eval.competitors import get_available_competitors

    competitors = get_available_competitors()
    competitor_names = [c.name for c in competitors]
    print(f"Available competitors: {len(competitors)}")
    for c in competitors:
        stoch = "stochastic" if c.name in STOCHASTIC_COMPETITORS else "deterministic"
        cluster = "+clusters" if c.name in CLUSTER_CAPABLE else ""
        print(f"  {c.name:25s} max={c.max_nodes:>8,} {stoch} {cluster}")

    # Generate seeds for stochastic competitors
    stochastic_seeds = list(range(42, 42 + args.stochastic_seeds))
    print(f"\nStochastic seeds: {stochastic_seeds}")

    # Run all combinations
    all_records: List[GraphRecord] = []
    total_layouts = 0
    total_success = 0
    total_failed = 0

    for gi, test_graph in enumerate(all_graphs):
        graph = test_graph.graph
        graph_name = test_graph.name
        graph._test_name = graph_name  # attach for record-keeping

        has_clusters = bool(getattr(graph, "clusters", None))
        cluster_names = list(graph.clusters.keys()) if has_clusters else []

        graph_record = GraphRecord(
            graph_name=graph_name,
            num_nodes=graph.num_nodes,
            num_edges=int(graph.edge_index.shape[1]) if graph.edge_index.numel() > 0 else 0,
            has_clusters=has_clusters,
            cluster_names=cluster_names,
            tags=list(test_graph.tags) if hasattr(test_graph, "tags") else [],
        )

        graph_dir = output_dir / graph_name.replace(" ", "_").replace("/", "_")
        graph_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"\n[{gi + 1}/{len(all_graphs)}] {graph_name} "
            f"(N={graph.num_nodes}, E={graph_record.num_edges}"
            f"{', clusters=' + str(len(cluster_names)) if has_clusters else ''})"
        )

        for competitor in competitors:
            # Skip if graph is too large for this competitor
            if competitor.max_nodes > 0 and graph.num_nodes > competitor.max_nodes:
                print(f"  {competitor.name:25s} SKIPPED (max_nodes={competitor.max_nodes})")
                continue

            is_stochastic = competitor.name in STOCHASTIC_COMPETITORS

            if is_stochastic:
                # Multiple seeds
                for seed in stochastic_seeds:
                    record, pos = _run_single_layout(competitor, graph, args.timeout, seed=seed)
                    total_layouts += 1

                    if pos is not None:
                        filename = f"{competitor.name}_seed{seed}.pt"
                        torch.save(pos, graph_dir / filename)
                        record.positions_file = str(graph_dir / filename)
                        total_success += 1
                    else:
                        total_failed += 1

                    graph_record.layouts.append(record)

                # Print summary for this competitor
                seeds_ok = sum(
                    1
                    for r in graph_record.layouts
                    if r.competitor_name == competitor.name and r.success
                )
                avg_time = sum(
                    r.runtime_seconds
                    for r in graph_record.layouts
                    if r.competitor_name == competitor.name and r.success
                ) / max(seeds_ok, 1)
                print(
                    f"  {competitor.name:25s} {seeds_ok}/{len(stochastic_seeds)} seeds OK "
                    f"({avg_time:.2f}s avg)"
                )
            else:
                # Single deterministic run
                record, pos = _run_single_layout(competitor, graph, args.timeout, seed=None)
                total_layouts += 1

                if pos is not None:
                    filename = f"{competitor.name}.pt"
                    torch.save(pos, graph_dir / filename)
                    record.positions_file = str(graph_dir / filename)
                    total_success += 1
                    print(f"  {competitor.name:25s} OK ({record.runtime_seconds:.2f}s)")
                else:
                    total_failed += 1
                    print(f"  {competitor.name:25s} FAILED: {record.error}")

                graph_record.layouts.append(record)

        all_records.append(graph_record)

    # Save manifest
    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "settings": {
            "timeout": args.timeout,
            "stochastic_seeds": stochastic_seeds,
            "max_nodes": args.max_nodes,
            "num_graphs": len(all_graphs),
            "num_competitors": len(competitors),
            "competitor_names": competitor_names,
        },
        "summary": {
            "total_layouts": total_layouts,
            "total_success": total_success,
            "total_failed": total_failed,
            "success_rate": total_success / max(total_layouts, 1),
        },
        "stochastic_competitors": sorted(STOCHASTIC_COMPETITORS),
        "deterministic_competitors": sorted(DETERMINISTIC_COMPETITORS),
        "cluster_capable_competitors": sorted(CLUSTER_CAPABLE),
        "graphs": [asdict(r) for r in all_records],
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nManifest written to {manifest_path}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("GROUND TRUTH GENERATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Graphs:     {len(all_records)}")
    print(f"Layouts:    {total_success}/{total_layouts} succeeded ({total_failed} failed)")
    print(f"Output:     {output_dir}")
    print(f"Manifest:   {manifest_path}")

    # Print failure summary
    failures = [
        (r.graph_name, r.competitor_name, r.seed, r.error)
        for gr in all_records
        for r in gr.layouts
        if not r.success
    ]
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for graph_name, comp_name, seed, error in failures[:20]:
            seed_str = f" seed={seed}" if seed is not None else ""
            print(f"  {graph_name} / {comp_name}{seed_str}: {error[:80]}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")


if __name__ == "__main__":
    main()
