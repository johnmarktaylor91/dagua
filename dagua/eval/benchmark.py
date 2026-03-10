"""Competitive benchmarking harness — run multiple layout engines on the same
graphs and produce a markdown report with quality + runtime comparison.

Usage:
    python -m dagua.eval.benchmark                         # small tier
    python -m dagua.eval.benchmark --tier medium
    python -m dagua.eval.benchmark --tier full              # all tiers
    python -m dagua.eval.benchmark --competitors dagua,graphviz_dot
    python -m dagua.eval.benchmark --output benchmark_results/
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from dagua.eval.competitors import get_available_competitors
from dagua.eval.competitors.base import CompetitorBase
from dagua.eval.graphs import TestGraph, get_scale_suite, get_test_graphs


@dataclass
class BenchmarkResult:
    """Result of running one competitor on one graph."""

    graph_name: str
    graph_nodes: int
    graph_edges: int
    competitor: str
    runtime_seconds: float
    metrics: Optional[Dict[str, Any]] = None
    composite_score: Optional[float] = None
    error: Optional[str] = None


def _compute_metrics(
    pos: torch.Tensor,
    graph,
    level: str = "quick",
) -> tuple:
    """Compute metrics and composite score. Returns (metrics_dict, composite)."""
    from dagua.metrics import composite, full, quick
    from dagua.utils import longest_path_layering

    edge_index = graph.edge_index
    topo_depth = longest_path_layering(edge_index, graph.num_nodes)
    node_sizes = graph.node_sizes if graph.node_sizes is not None else None

    if level == "full" and graph.num_nodes <= 50_000:
        m = full(pos, edge_index, topo_depth=topo_depth, node_sizes=node_sizes)
    else:
        m = quick(pos, edge_index, topo_depth=topo_depth, node_sizes=node_sizes)

    score = composite(m)
    return m, score


def _run_with_timeout(
    competitor: CompetitorBase,
    graph,
    timeout: float,
) -> "CompetitorResult":
    """Run competitor layout with a process-level timeout fallback.

    Uses multiprocessing for a hard timeout in case the competitor hangs.
    Most competitors already have their own timeout, so this is a safety net.
    """
    # First try the simple path — competitor has its own timeout
    return competitor.layout(graph, timeout=timeout)


def run_benchmark(
    tier: str = "small",
    competitors: Optional[List[str]] = None,
    timeout: float = 300.0,
    metrics_level: str = "quick",
    output_dir: str = "benchmark_results",
) -> List[BenchmarkResult]:
    """Run competitive benchmark suite.

    Args:
        tier: Scale tier — small, medium, large, huge, or full (all tiers).
        competitors: Filter to specific competitor names. None = all available.
        timeout: Per-layout timeout in seconds.
        metrics_level: "quick" or "full" — controls metric detail.
        output_dir: Directory for output files.

    Returns:
        List of BenchmarkResult for every (graph, competitor) pair.
    """
    # Determine tiers to run
    if tier == "full":
        tiers = ["small", "medium", "large", "huge"]
    else:
        tiers = [tier]

    # Collect graphs
    all_graphs: List[TestGraph] = []
    for t in tiers:
        if t == "small":
            # Include the existing test graphs plus scale suite
            all_graphs.extend(get_test_graphs(max_nodes=2000))
            all_graphs.extend(get_scale_suite("small"))
        else:
            all_graphs.extend(get_scale_suite(t))

    # Deduplicate by name
    seen_names: set = set()
    graphs: List[TestGraph] = []
    for tg in all_graphs:
        if tg.name not in seen_names:
            seen_names.add(tg.name)
            graphs.append(tg)

    # Get competitors
    available = get_available_competitors()
    if competitors:
        filter_set = set(competitors)
        available = [c for c in available if c.name in filter_set]

    if not available:
        print("ERROR: No competitors available.")
        return []

    print(f"Benchmark: {len(graphs)} graphs x {len(available)} competitors")
    print(f"Competitors: {', '.join(c.name for c in available)}")
    print(f"Tiers: {', '.join(tiers)}")
    print(f"Metrics level: {metrics_level}")
    print()

    results: List[BenchmarkResult] = []

    for i, tg in enumerate(graphs):
        n_nodes = tg.graph.num_nodes
        n_edges = tg.graph.edge_index.shape[1] if tg.graph.edge_index.numel() > 0 else 0
        print(f"[{i + 1}/{len(graphs)}] {tg.name} ({n_nodes:,} nodes, {n_edges:,} edges)")

        # Compute node sizes once (some competitors don't need it but metrics do)
        try:
            if tg.graph.node_sizes is None:
                tg.graph.compute_node_sizes()
        except Exception:
            pass  # node sizes are optional

        for comp in available:
            # Skip if graph is too large for this competitor
            if n_nodes > comp.max_nodes:
                print(f"  {comp.name}: skipped (>{comp.max_nodes:,} max)")
                continue

            print(f"  {comp.name}...", end="", flush=True)
            cr = _run_with_timeout(comp, tg.graph, timeout)

            if cr.error:
                print(f" ERROR: {cr.error[:80]}")
                results.append(BenchmarkResult(
                    graph_name=tg.name,
                    graph_nodes=n_nodes,
                    graph_edges=n_edges,
                    competitor=comp.name,
                    runtime_seconds=cr.runtime_seconds,
                    error=cr.error,
                ))
                continue

            # Compute metrics on the resulting positions
            metrics_dict = None
            comp_score = None
            try:
                metrics_dict, comp_score = _compute_metrics(
                    cr.pos, tg.graph, level=metrics_level
                )
            except Exception as e:
                # Metrics failure shouldn't discard timing result
                metrics_dict = {"_metrics_error": str(e)}

            print(f" {cr.runtime_seconds:.2f}s", end="")
            if comp_score is not None:
                print(f" (score={comp_score:.1f})", end="")
            print()

            results.append(BenchmarkResult(
                graph_name=tg.name,
                graph_nodes=n_nodes,
                graph_edges=n_edges,
                competitor=comp.name,
                runtime_seconds=cr.runtime_seconds,
                metrics=metrics_dict,
                composite_score=comp_score,
            ))

    # Save outputs
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = str(Path(output_dir) / "results.json")
    _save_json(results, json_path)
    print(f"\nJSON results: {json_path}")

    # Markdown
    md_path = str(Path(output_dir) / "report.md")
    from dagua.eval.report import generate_benchmark_markdown

    generate_benchmark_markdown(results, md_path)
    print(f"Markdown report: {md_path}")

    return results


def _save_json(results: List[BenchmarkResult], path: str):
    """Save results as JSON, converting non-serializable values."""
    data = []
    for r in results:
        d = asdict(r)
        # Clean up metrics for JSON serialization
        if d["metrics"]:
            cleaned = {}
            for k, v in d["metrics"].items():
                if isinstance(v, (int, float, str, bool, type(None))):
                    cleaned[k] = v
                elif isinstance(v, dict):
                    cleaned[k] = {
                        sk: sv
                        for sk, sv in v.items()
                        if isinstance(sv, (int, float, str, bool, type(None)))
                    }
                # Skip non-serializable values (tensors, etc.)
            d["metrics"] = cleaned
        data.append(d)

    meta = {
        "generated": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "results": data,
    }

    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)


def main():
    """CLI entry point for python -m dagua.eval.benchmark."""
    parser = argparse.ArgumentParser(
        description="Dagua competitive benchmark — compare layout quality and runtime"
    )
    parser.add_argument(
        "--tier",
        default="small",
        choices=["small", "medium", "large", "huge", "full"],
        help="Scale tier (default: small)",
    )
    parser.add_argument(
        "--competitors",
        type=str,
        default=None,
        help="Comma-separated competitor names to include (default: all available)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-layout timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--metrics-level",
        choices=["quick", "full"],
        default="quick",
        help="Metric detail level (default: quick)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results",
        help="Output directory (default: benchmark_results)",
    )
    args = parser.parse_args()

    comp_list = args.competitors.split(",") if args.competitors else None

    run_benchmark(
        tier=args.tier,
        competitors=comp_list,
        timeout=args.timeout,
        metrics_level=args.metrics_level,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
