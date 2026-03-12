"""Parameter sweep engine for placement and aesthetic tuning.

Modes:
- focused: One-parameter-at-a-time sensitivity analysis
- interaction: 2D grid for known interacting parameter pairs
- full: Full sweep across all graphs and parameters
- placement tuning: staged, placement-only coordinate search with Pareto output
"""

from __future__ import annotations

import itertools
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from dagua.config import LayoutConfig, PARAM_REGISTRY_DICT as PARAM_REGISTRY
from dagua.eval.graphs import TestGraph, get_scale_suite, get_test_graphs
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


@dataclass
class PlacementCandidate:
    """One evaluated placement candidate from the staged search."""

    config: LayoutConfig
    score: float
    mean_runtime_seconds: float
    aggregate_metrics: Dict[str, float]
    per_graph_metrics: Dict[str, Dict[str, float]]
    stage: str
    tuned_param: str
    tuned_value: Any


@dataclass
class PlacementTuningResult:
    """Top-level result for the staged placement tuning workflow."""

    search_suite_graphs: List[str]
    validation_suite_graphs: List[str]
    baseline: PlacementCandidate
    best: PlacementCandidate
    pareto_frontier: List[PlacementCandidate]
    all_candidates: List[PlacementCandidate]
    stages: List[Dict[str, Any]]


PLACEMENT_STAGE_GROUPS: List[Tuple[str, List[str]]] = [
    ("spacing_core", ["node_sep", "rank_sep", "w_crossing", "w_dag", "w_overlap"]),
    ("structure_refine", ["w_straightness", "w_length_variance", "w_attract", "w_attract_x_bias", "w_repel"]),
    ("runtime_schedule", ["steps"]),
]

PLACEMENT_TUNING_GRAPH_NAMES = [
    "chain_100",
    "binary_tree_127",
    "dense_skip_200",
    "random_sparse_500",
    "random_dense_300",
    "long_range_residual_ladder",
    "interleaved_cluster_crosstalk",
    "width_skew_late_merge",
]


def get_placement_search_suite() -> List[TestGraph]:
    """Return the search suite used for the broad brute-force stages.

    The suite intentionally mixes:
    - easy sanity graphs
    - crossing-heavy DAGs
    - skip-heavy / asymmetry cases

    It intentionally excludes the 2k-scale graph because that graph dominates
    runtime and is better used as a validation gate for the top candidates.
    """
    test_graphs = {tg.name: tg for tg in get_test_graphs(max_nodes=1_000)}
    return [test_graphs[name] for name in PLACEMENT_TUNING_GRAPH_NAMES if name in test_graphs]


def get_placement_validation_suite() -> List[TestGraph]:
    """Return the scale-aware validation suite for top placement candidates."""
    scale_graphs = get_scale_suite("small")
    preferred = ("random_dag_2000", "wide_dag_2000")
    chosen = [tg for tg in scale_graphs if tg.name in preferred]
    if chosen:
        return chosen
    fallback = next((tg for tg in scale_graphs if tg.graph.num_nodes >= 2_000), None)
    return [fallback] if fallback is not None else []


def get_placement_tuning_suite() -> List[TestGraph]:
    """Return the full default suite used by the placement tuning pipeline."""
    return get_placement_search_suite() + get_placement_validation_suite()


def placement_score(metrics: Dict[str, float]) -> float:
    """Compute a placement-only scalar score from benchmark metrics.

    The score strongly prioritizes:
    - DAG consistency
    - crossing minimization
    - overlap avoidance

    Edge-length regularity matters, but it should not outweigh basic layered
    readability. Higher is better.
    """
    dag_consistency = float(metrics.get("dag_consistency", 0.0))
    crossings = max(float(metrics.get("edge_crossings", 0.0)), 0.0)
    overlaps = max(float(metrics.get("overlap_count", 0.0)), 0.0)
    edge_length_cv = max(float(metrics.get("edge_length_cv", 0.0)), 0.0)

    return (
        dag_consistency * 100.0
        - 7.5 * math.log1p(crossings)
        - 9.0 * math.log1p(overlaps)
        - 18.0 * edge_length_cv
    )


def combined_placement_score(search_score: float, validation_score: float) -> float:
    """Combine fast-suite and scale-validation scores into one selection scalar."""
    return search_score * 0.7 + validation_score * 0.3


def _clone_config(config: LayoutConfig) -> LayoutConfig:
    return LayoutConfig(**asdict(config))


def _evaluate_config(
    graphs: List[TestGraph],
    config: LayoutConfig,
    stage: str,
    tuned_param: str,
    tuned_value: Any,
) -> PlacementCandidate:
    aggregate_keys = ("dag_consistency", "edge_crossings", "overlap_count", "edge_length_cv")
    aggregate = {key: 0.0 for key in aggregate_keys}
    per_graph: Dict[str, Dict[str, float]] = {}
    runtimes: List[float] = []

    for tg in graphs:
        tg.graph.compute_node_sizes()
        assert tg.graph.node_sizes is not None
        start = torch.cuda.Event(enable_timing=True) if config.device == "cuda" and torch.cuda.is_available() else None
        end = torch.cuda.Event(enable_timing=True) if start is not None else None
        if start is not None and end is not None:
            start.record()
        else:
            cpu_t0 = time.perf_counter()

        pos = layout(tg.graph, config)

        if start is not None and end is not None:
            end.record()
            torch.cuda.synchronize()
            runtime = float(start.elapsed_time(end)) / 1000.0
        else:
            runtime = time.perf_counter() - cpu_t0
        runtimes.append(runtime)

        metrics = compute_all_metrics(pos, tg.graph.edge_index, tg.graph.node_sizes)
        per_graph[tg.name] = {key: float(metrics.get(key, 0.0)) for key in aggregate_keys}
        for key in aggregate_keys:
            aggregate[key] += per_graph[tg.name][key]

    graph_count = max(len(graphs), 1)
    aggregate = {key: value / graph_count for key, value in aggregate.items()}
    score = placement_score(aggregate)
    mean_runtime = sum(runtimes) / len(runtimes) if runtimes else 0.0
    return PlacementCandidate(
        config=_clone_config(config),
        score=score,
        mean_runtime_seconds=mean_runtime,
        aggregate_metrics=aggregate,
        per_graph_metrics=per_graph,
        stage=stage,
        tuned_param=tuned_param,
        tuned_value=tuned_value,
    )


def _dominates(a: PlacementCandidate, b: PlacementCandidate) -> bool:
    """Return whether candidate ``a`` Pareto-dominates candidate ``b``."""
    def _metric(candidate: PlacementCandidate, key: str) -> float:
        if key in candidate.aggregate_metrics:
            return float(candidate.aggregate_metrics[key])
        return float(candidate.aggregate_metrics.get(f"search_{key}", 0.0))

    better_or_equal = (
        _metric(a, "dag_consistency") >= _metric(b, "dag_consistency")
        and _metric(a, "edge_crossings") <= _metric(b, "edge_crossings")
        and _metric(a, "overlap_count") <= _metric(b, "overlap_count")
        and a.mean_runtime_seconds <= b.mean_runtime_seconds
    )
    strictly_better = (
        _metric(a, "dag_consistency") > _metric(b, "dag_consistency")
        or _metric(a, "edge_crossings") < _metric(b, "edge_crossings")
        or _metric(a, "overlap_count") < _metric(b, "overlap_count")
        or a.mean_runtime_seconds < b.mean_runtime_seconds
    )
    return better_or_equal and strictly_better


def pareto_frontier(candidates: List[PlacementCandidate]) -> List[PlacementCandidate]:
    """Return non-dominated candidates across the core placement/runtime axes."""
    frontier: List[PlacementCandidate] = []
    for candidate in candidates:
        if any(_dominates(other, candidate) for other in candidates if other is not candidate):
            continue
        frontier.append(candidate)
    return sorted(frontier, key=lambda c: (-c.score, c.mean_runtime_seconds))


def _candidate_for_stage(
    search_graphs: List[TestGraph],
    validation_graphs: List[TestGraph],
    config: LayoutConfig,
    stage: str,
    tuned_param: str,
    tuned_value: Any,
) -> PlacementCandidate:
    """Evaluate one candidate on the search suite and validation suite."""
    search_candidate = _evaluate_config(search_graphs, config, stage, tuned_param, tuned_value)
    if not validation_graphs:
        return search_candidate

    validation_candidate = _evaluate_config(validation_graphs, config, f"{stage}_validation", tuned_param, tuned_value)
    combined_metrics = {
        "search_score": search_candidate.score,
        "validation_score": validation_candidate.score,
        **{f"search_{k}": v for k, v in search_candidate.aggregate_metrics.items()},
        **{f"validation_{k}": v for k, v in validation_candidate.aggregate_metrics.items()},
    }
    return PlacementCandidate(
        config=_clone_config(config),
        score=combined_placement_score(search_candidate.score, validation_candidate.score),
        mean_runtime_seconds=search_candidate.mean_runtime_seconds + validation_candidate.mean_runtime_seconds,
        aggregate_metrics=combined_metrics,
        per_graph_metrics={
            **search_candidate.per_graph_metrics,
            **validation_candidate.per_graph_metrics,
        },
        stage=stage,
        tuned_param=tuned_param,
        tuned_value=tuned_value,
    )


def _stage_combinations(params: List[str]) -> List[Dict[str, Any]]:
    """Return the full Cartesian product for one stage."""
    values_by_param = []
    for param in params:
        meta = PARAM_REGISTRY.get(param)
        if meta is None:
            continue
        values_by_param.append((param, meta.sweep_values))

    combinations: List[Dict[str, Any]] = []
    for combo in itertools.product(*(values for _, values in values_by_param)):
        combinations.append({param: value for (param, _), value in zip(values_by_param, combo)})
    return combinations


def run_placement_tuning(
    graphs: Optional[List[TestGraph]] = None,
    output_dir: Optional[str] = None,
    base_config: Optional[LayoutConfig] = None,
) -> PlacementTuningResult:
    """Run staged automatic placement tuning over representative graph suites.

    The pipeline is intentionally split into:
    - a broad brute-force search on a fast challenge suite
    - validation of every candidate on a 2k-scale suite

    This keeps the 2k graph in the objective without making the broad search
    prohibitively slow.
    """
    if graphs is not None:
        search_graphs = graphs
        validation_graphs: List[TestGraph] = []
    else:
        search_graphs = get_placement_search_suite()
        validation_graphs = get_placement_validation_suite()
    current = _clone_config(base_config or LayoutConfig())
    all_candidates: List[PlacementCandidate] = []
    stages: List[Dict[str, Any]] = []

    baseline = _candidate_for_stage(
        search_graphs,
        validation_graphs,
        current,
        stage="baseline",
        tuned_param="baseline",
        tuned_value="baseline",
    )
    best = baseline
    all_candidates.append(baseline)

    for stage_name, params in PLACEMENT_STAGE_GROUPS:
        stage_record: Dict[str, Any] = {"stage": stage_name, "params": params, "evaluations": []}
        stage_best = best
        for combo in _stage_combinations(params):
            trial = _clone_config(best.config)
            for param, value in combo.items():
                setattr(trial, param, value)
            candidate = _candidate_for_stage(
                search_graphs,
                validation_graphs,
                trial,
                stage=stage_name,
                tuned_param="+".join(combo.keys()),
                tuned_value=combo,
            )
            all_candidates.append(candidate)
            stage_record["evaluations"].append(
                {
                    "values": combo,
                    "score": candidate.score,
                    "mean_runtime_seconds": candidate.mean_runtime_seconds,
                    "aggregate_metrics": candidate.aggregate_metrics,
                }
            )
            if candidate.score > stage_best.score:
                stage_best = candidate
        best = stage_best
        stage_record["selected"] = {
            "values": stage_best.tuned_value,
            "score": stage_best.score,
            "mean_runtime_seconds": stage_best.mean_runtime_seconds,
            "aggregate_metrics": stage_best.aggregate_metrics,
        }
        stages.append(stage_record)

    result = PlacementTuningResult(
        search_suite_graphs=[tg.name for tg in search_graphs],
        validation_suite_graphs=[tg.name for tg in validation_graphs],
        baseline=baseline,
        best=best,
        pareto_frontier=pareto_frontier(all_candidates),
        all_candidates=sorted(all_candidates, key=lambda c: (-c.score, c.mean_runtime_seconds)),
        stages=stages,
    )

    if output_dir:
        _save_placement_tuning(result, output_dir)
    return result


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
                    assert tg.graph.node_sizes is not None
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
                        assert tg.graph.node_sizes is not None
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


def _candidate_to_dict(candidate: PlacementCandidate) -> Dict[str, Any]:
    return {
        "config": asdict(candidate.config),
        "score": candidate.score,
        "mean_runtime_seconds": candidate.mean_runtime_seconds,
        "aggregate_metrics": candidate.aggregate_metrics,
        "per_graph_metrics": candidate.per_graph_metrics,
        "stage": candidate.stage,
        "tuned_param": candidate.tuned_param,
        "tuned_value": candidate.tuned_value,
    }


def _save_placement_tuning(result: PlacementTuningResult, output_dir: str) -> None:
    """Persist placement tuning artifacts as JSON + markdown."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "search_suite_graphs": result.search_suite_graphs,
        "validation_suite_graphs": result.validation_suite_graphs,
        "baseline": _candidate_to_dict(result.baseline),
        "best": _candidate_to_dict(result.best),
        "pareto_frontier": [_candidate_to_dict(candidate) for candidate in result.pareto_frontier],
        "all_candidates": [_candidate_to_dict(candidate) for candidate in result.all_candidates],
        "stages": result.stages,
    }
    (out / "placement_tuning.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Placement Tuning",
        "",
        "Automatic placement-only tuning run over a representative graph suite.",
        "",
        "## Search Suite",
        "",
    ]
    for name in result.search_suite_graphs:
        lines.append(f"- `{name}`")

    if result.validation_suite_graphs:
        lines.extend(["", "## Validation Suite", ""])
        for name in result.validation_suite_graphs:
            lines.append(f"- `{name}`")

    lines.extend(
        [
            "",
            "## Baseline vs Best",
            "",
            f"- baseline score: `{result.baseline.score:.2f}`",
            f"- best score: `{result.best.score:.2f}`",
            f"- baseline runtime: `{result.baseline.mean_runtime_seconds:.3f}s`",
            f"- best runtime: `{result.best.mean_runtime_seconds:.3f}s`",
            f"- best tuned params: `{result.best.tuned_param}` = `{result.best.tuned_value}`",
            "",
            "## Best Aggregate Metrics",
            "",
        ]
    )
    for key, value in result.best.aggregate_metrics.items():
        lines.append(f"- `{key}`: `{value:.4f}`")

    lines.extend(["", "## Pareto Frontier", ""])
    for candidate in result.pareto_frontier[:10]:
        lines.append(
            f"- `{candidate.stage}` / `{candidate.tuned_param}` = `{candidate.tuned_value}`"
            f" | score `{candidate.score:.2f}`"
            f" | runtime `{candidate.mean_runtime_seconds:.3f}s`"
        )

    (out / "placement_tuning.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        best_val = max(avg_quality, key=lambda value: avg_quality[value])
        worst_val = min(avg_quality, key=lambda value: avg_quality[value])

        print(f"{param:<25} | {str(best_val):>12} | {avg_quality[best_val]:>10.1f} | {str(worst_val):>12} | {avg_quality[worst_val]:>10.1f}")
