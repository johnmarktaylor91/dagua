"""Graphviz comparison — run Dagua and Graphviz on same graphs, compare metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch

from dagua.config import LayoutConfig
from dagua.eval.graphs import TestGraph, get_test_graphs
from dagua.graphviz_utils import layout_with_graphviz, render_comparison
from dagua.layout import layout
from dagua.metrics import compute_all_metrics, graphviz_delta


@dataclass
class ComparisonResult:
    """Result of comparing Dagua vs Graphviz on a single graph."""
    graph_name: str
    dagua_metrics: Dict[str, float]
    graphviz_metrics: Dict[str, float]
    delta: Dict[str, float]
    dagua_better: bool = False  # True if dagua quality > graphviz quality

    def __post_init__(self):
        dq = self.dagua_metrics.get("overall_quality", 0)
        gq = self.graphviz_metrics.get("overall_quality", 0)
        self.dagua_better = dq >= gq


def compare_with_graphviz(
    graphs: Optional[List[TestGraph]] = None,
    config: Optional[LayoutConfig] = None,
    output_dir: Optional[str] = None,
    max_nodes: int = 500,
) -> List[ComparisonResult]:
    """Compare Dagua layout vs Graphviz for a collection of graphs.

    Args:
        graphs: Test graphs to compare. If None, uses all test graphs.
        config: LayoutConfig for Dagua. If None, uses defaults.
        output_dir: If set, saves comparison images here.
        max_nodes: Skip graphs larger than this (Graphviz can be slow).

    Returns:
        List of ComparisonResult objects.
    """
    if graphs is None:
        graphs = get_test_graphs(max_nodes=max_nodes)
    if config is None:
        config = LayoutConfig()

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = []
    for tg in graphs:
        if tg.graph.num_nodes > max_nodes:
            continue

        try:
            result = _compare_single(tg, config, output_dir)
            results.append(result)
        except Exception as e:
            print(f"  [SKIP] {tg.name}: {e}")

    return results


def _compare_single(
    tg: TestGraph,
    config: LayoutConfig,
    output_dir: Optional[str] = None,
) -> ComparisonResult:
    """Compare a single graph."""
    graph = tg.graph
    graph.compute_node_sizes()

    # Dagua layout
    dagua_pos = layout(graph, config)

    # Graphviz layout
    try:
        gv_pos = layout_with_graphviz(graph)
    except Exception as e:
        raise RuntimeError(f"Graphviz layout failed: {e}")

    # Compute metrics
    dagua_m = compute_all_metrics(dagua_pos, graph.edge_index, graph.node_sizes)
    gv_m = compute_all_metrics(gv_pos, graph.edge_index, graph.node_sizes)
    delta = graphviz_delta(dagua_m, gv_m)

    # Save comparison image
    if output_dir:
        img_path = str(Path(output_dir) / f"compare_{tg.name}.png")
        try:
            render_comparison(graph, dagua_pos, gv_pos, img_path, config)
        except Exception:
            pass  # Image generation is non-critical

    return ComparisonResult(
        graph_name=tg.name,
        dagua_metrics=dagua_m,
        graphviz_metrics=gv_m,
        delta=delta,
    )


def print_comparison_table(results: List[ComparisonResult]):
    """Print a summary table of comparison results."""
    if not results:
        print("No comparison results.")
        return

    key_metrics = ["edge_crossings", "node_overlaps", "dag_fraction", "overall_quality"]

    # Header
    print(f"\n{'Graph':<30} | {'Metric':<20} | {'Dagua':>10} | {'Graphviz':>10} | {'Delta':>10}")
    print("-" * 90)

    for r in results:
        for i, metric in enumerate(key_metrics):
            name = r.graph_name if i == 0 else ""
            dv = r.dagua_metrics.get(metric, 0)
            gv = r.graphviz_metrics.get(metric, 0)
            delta = r.delta.get(metric, 0)
            indicator = "+" if delta > 0 else ("-" if delta < 0 else "=")
            print(f"{name:<30} | {metric:<20} | {dv:>10.2f} | {gv:>10.2f} | {indicator}{abs(delta):>9.2f}")
        print("-" * 90)

    # Summary
    wins = sum(1 for r in results if r.dagua_better)
    print(f"\nDagua wins: {wins}/{len(results)}")


@dataclass
class MultiComparisonResult:
    """Result of comparing N engines on a single graph."""
    graph_name: str
    engine_metrics: Dict[str, Dict[str, float]]
    engine_positions: Dict[str, Optional[torch.Tensor]]
    winner: str = ""

    def __post_init__(self):
        if not self.winner and self.engine_metrics:
            best_q = -1.0
            for name, m in self.engine_metrics.items():
                q = m.get("overall_quality", 0)
                if q > best_q:
                    best_q = q
                    self.winner = name


def compare_engines(
    graphs: Optional[List[TestGraph]] = None,
    engines: Optional[List[str]] = None,
    config: Optional[LayoutConfig] = None,
    output_dir: Optional[str] = None,
    max_nodes: int = 500,
) -> List[MultiComparisonResult]:
    """Compare multiple layout engines on a collection of graphs.

    Uses the competitor registry to run all available engines (or a filtered subset).

    Args:
        graphs: Test graphs. If None, uses all test graphs.
        engines: Engine names to include. If None, uses all available.
        config: LayoutConfig for Dagua. If None, uses defaults.
        output_dir: If set, saves multi-comparison images here.
        max_nodes: Skip graphs larger than this.

    Returns:
        List of MultiComparisonResult objects.
    """
    from dagua.eval.competitors import get_available_competitors
    from dagua.graphviz_utils import render_multi_comparison

    if graphs is None:
        graphs = get_test_graphs(max_nodes=max_nodes)
    if config is None:
        config = LayoutConfig()

    competitors = get_available_competitors()
    if engines is not None:
        engine_set = set(engines)
        competitors = [c for c in competitors if c.name in engine_set]

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = []
    for tg in graphs:
        if tg.graph.num_nodes > max_nodes:
            continue

        tg.graph.compute_node_sizes()
        engine_metrics: Dict[str, Dict[str, float]] = {}
        engine_positions: Dict[str, Optional[torch.Tensor]] = {}

        for comp in competitors:
            if tg.graph.num_nodes > comp.max_nodes:
                continue
            try:
                result = comp.layout(tg.graph)
                if result.pos is not None:
                    m = compute_all_metrics(
                        result.pos, tg.graph.edge_index, tg.graph.node_sizes
                    )
                    engine_metrics[comp.name] = m
                    engine_positions[comp.name] = result.pos
            except Exception as e:
                print(f"  [{comp.name}] {tg.name}: {e}")

        if not engine_metrics:
            continue

        mr = MultiComparisonResult(
            graph_name=tg.name,
            engine_metrics=engine_metrics,
            engine_positions=engine_positions,
        )
        results.append(mr)

        # Save multi-comparison image
        if output_dir and engine_positions:
            valid_pos = {k: v for k, v in engine_positions.items() if v is not None}
            if valid_pos:
                img_path = str(Path(output_dir) / f"multi_{tg.name}.png")
                try:
                    render_multi_comparison(tg.graph, valid_pos, img_path, config)
                except Exception:
                    pass

    return results


def print_multi_comparison_table(results: List[MultiComparisonResult]):
    """Print a summary table of multi-engine comparison results."""
    if not results:
        print("No comparison results.")
        return

    # Collect all engine names
    all_engines = set()
    for r in results:
        all_engines.update(r.engine_metrics.keys())
    engine_list = sorted(all_engines)

    # Header
    header = f"{'Graph':<25}"
    for eng in engine_list:
        header += f" | {eng:>12}"
    header += " | Winner"
    print(f"\n{header}")
    print("-" * len(header))

    # Rows (overall quality)
    wins: Dict[str, int] = {e: 0 for e in engine_list}
    for r in results:
        row = f"{r.graph_name:<25}"
        for eng in engine_list:
            q = r.engine_metrics.get(eng, {}).get("overall_quality", 0)
            row += f" | {q:>12.1f}" if eng in r.engine_metrics else f" | {'—':>12}"
        row += f" | {r.winner}"
        print(row)
        if r.winner in wins:
            wins[r.winner] += 1

    print("-" * len(header))
    print("Wins:", ", ".join(f"{e}={w}" for e, w in sorted(wins.items(), key=lambda x: -x[1])))


def save_comparison_json(results: List[ComparisonResult], path: str):
    """Save comparison results as JSON."""
    data = []
    for r in results:
        data.append({
            "graph_name": r.graph_name,
            "dagua_metrics": r.dagua_metrics,
            "graphviz_metrics": r.graphviz_metrics,
            "delta": r.delta,
            "dagua_better": r.dagua_better,
        })
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
