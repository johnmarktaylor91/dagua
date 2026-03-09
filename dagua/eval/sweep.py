"""Parameter sweep engine for aesthetic tuning.

Modes:
- focused: One-parameter-at-a-time sensitivity analysis
- interaction: 2D grid for known interacting parameter pairs
- full: Full sweep across all graphs and parameters
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from dagua.config import LayoutConfig, PARAM_REGISTRY_DICT as PARAM_REGISTRY
from dagua.eval.graphs import TestGraph, get_test_graphs
from dagua.layout import layout
from dagua.metrics import compute_all_metrics


@dataclass
class SweepResult:
    """Result of a single sweep evaluation."""
    graph_name: str
    param_name: str
    param_value: Any
    metrics: Dict[str, float]
    quality: float = 0.0

    def __post_init__(self):
        self.quality = self.metrics.get("overall_quality", 0.0)


def focused_sweep(
    graphs: Optional[List[TestGraph]] = None,
    params: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
) -> List[SweepResult]:
    """One-parameter-at-a-time sweep.

    For each parameter, varies it across its registered sweep values
    while holding all others at defaults.

    Args:
        graphs: Test graphs. If None, uses a small subset.
        params: Parameter names to sweep. If None, sweeps all registered.
        output_dir: If set, saves results here.

    Returns:
        List of SweepResult objects.
    """
    if graphs is None:
        graphs = get_test_graphs(max_nodes=100)[:5]  # Small subset for speed
    if params is None:
        params = list(PARAM_REGISTRY.keys())

    results = []
    total = sum(len(PARAM_REGISTRY[p].sweep_values) for p in params if p in PARAM_REGISTRY)
    done = 0

    for param_name in params:
        if param_name not in PARAM_REGISTRY:
            continue
        param_info = PARAM_REGISTRY[param_name]

        for value in param_info.sweep_values:
            config = LayoutConfig()
            if hasattr(config, param_name):
                setattr(config, param_name, value)

            for tg in graphs:
                try:
                    tg.graph.compute_node_sizes()
                    pos = layout(tg.graph, config)
                    metrics = compute_all_metrics(
                        pos, tg.graph.edge_index, tg.graph.node_sizes
                    )
                    results.append(SweepResult(
                        graph_name=tg.name,
                        param_name=param_name,
                        param_value=value,
                        metrics=metrics,
                    ))
                except Exception as e:
                    print(f"  [ERROR] {tg.name} @ {param_name}={value}: {e}")

            done += 1

    if output_dir:
        _save_sweep_results(results, output_dir, "focused")

    return results


def interaction_sweep(
    pairs: List[Tuple[str, str]],
    graphs: Optional[List[TestGraph]] = None,
    output_dir: Optional[str] = None,
    grid_size: int = 5,
) -> List[SweepResult]:
    """2D grid sweep for interacting parameter pairs.

    Args:
        pairs: List of (param_a, param_b) tuples to sweep jointly.
        graphs: Test graphs. If None, uses small subset.
        output_dir: If set, saves results here.
        grid_size: Number of values per dimension.

    Returns:
        List of SweepResult objects.
    """
    if graphs is None:
        graphs = get_test_graphs(max_nodes=100)[:3]

    results = []

    for param_a, param_b in pairs:
        if param_a not in PARAM_REGISTRY or param_b not in PARAM_REGISTRY:
            continue

        values_a = PARAM_REGISTRY[param_a].sweep_values[:grid_size]
        values_b = PARAM_REGISTRY[param_b].sweep_values[:grid_size]

        for va in values_a:
            for vb in values_b:
                config = LayoutConfig()
                if hasattr(config, param_a):
                    setattr(config, param_a, va)
                if hasattr(config, param_b):
                    setattr(config, param_b, vb)

                for tg in graphs:
                    try:
                        tg.graph.compute_node_sizes()
                        pos = layout(tg.graph, config)
                        metrics = compute_all_metrics(
                            pos, tg.graph.edge_index, tg.graph.node_sizes
                        )
                        results.append(SweepResult(
                            graph_name=tg.name,
                            param_name=f"{param_a}×{param_b}",
                            param_value=f"{va},{vb}",
                            metrics=metrics,
                        ))
                    except Exception as e:
                        pass

    if output_dir:
        _save_sweep_results(results, output_dir, "interaction")

    return results


# ─── Known Interacting Pairs ─────────────────────────────────────────────────

KNOWN_INTERACTIONS = [
    ("w_dag", "w_attract"),
    ("w_repel", "w_overlap"),
    ("w_attract", "w_repel"),
    ("w_crossing", "w_straightness"),
    ("node_sep", "rank_sep"),
    ("w_dag", "w_straightness"),
]


def _save_sweep_results(results: List[SweepResult], output_dir: str, mode: str):
    """Save sweep results to JSON."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    data = []
    for r in results:
        data.append({
            "graph_name": r.graph_name,
            "param_name": r.param_name,
            "param_value": r.param_value,
            "quality": r.quality,
            "metrics": r.metrics,
        })
    path = Path(output_dir) / f"sweep_{mode}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(results)} results to {path}")


def print_sweep_summary(results: List[SweepResult]):
    """Print a summary of sweep results: best value per parameter."""
    from collections import defaultdict

    by_param = defaultdict(list)
    for r in results:
        by_param[r.param_name].append(r)

    print(f"\n{'Parameter':<25} | {'Best Value':>12} | {'Quality':>10} | {'Worst Value':>12} | {'Quality':>10}")
    print("-" * 80)

    for param, rs in sorted(by_param.items()):
        # Average quality per value
        by_value = defaultdict(list)
        for r in rs:
            by_value[r.param_value].append(r.quality)

        avg_quality = {v: sum(qs) / len(qs) for v, qs in by_value.items()}
        best_val = max(avg_quality, key=avg_quality.get)
        worst_val = min(avg_quality, key=avg_quality.get)

        print(f"{param:<25} | {str(best_val):>12} | {avg_quality[best_val]:>10.1f} | {str(worst_val):>12} | {avg_quality[worst_val]:>10.1f}")
