#!/usr/bin/env python3
"""Quality/runtime analysis pipeline for dagua benchmark output.

Reads a completed or partial ``variant_bench_full`` benchmark, reloads saved
positions, recomputes quality metrics with stable seeding, aggregates per-graph
rankings into family scorecards, extracts Pareto fronts, and writes sidecar CSVs
for downstream reporting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

# Pre-load numba/llvmlite FIRST. A transitive ``dagua.eval`` import below clobbers
# llvmlite's lazy ``libllvmlite.so`` load path (import-order bug: numba then fails
# with "Could not find/load shared object file"). Loading it here caches the lib
# before the breaker runs. Keep this above all heavy imports.
import llvmlite.binding  # noqa: F401
import numba  # noqa: F401
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dagua.eval.graphs import get_test_graphs  # noqa: E402
from dagua.eval.pipeline_io import (  # noqa: E402
    aspect_ratio_deviation,
    compute_quick_metrics_seeded,
    compute_sampled_metrics_seeded,
    load_position_tensor,
    open_h5_for_worker,
    stable_seed,
    validate_positions,
)
from dagua.eval.variants import algorithm_family  # noqa: E402

LOGGER = logging.getLogger("qr_analysis")

QR_QUICK_METRICS: FrozenSet[str] = frozenset(
    {
        "edge_length_cv",
        "dag_consistency",
        "depth_spearman_rho",
        "overlap_count",
        "edge_straightness_mean_deg",
    }
)

QR_SAMPLED_METRICS: FrozenSet[str] = frozenset(
    {
        "sampled_stress",
        "crossing_rate",
        "edge_crossings",
    }
)

ALL_ANALYSIS_METRICS: Tuple[str, ...] = tuple(
    sorted(QR_QUICK_METRICS | QR_SAMPLED_METRICS | {"aspect_ratio_deviation"})
)
INLINE_PARETO_METRICS: FrozenSet[str] = frozenset({"sampled_stress", "crossing_rate"})
DAG_ORDER_METRICS: FrozenSet[str] = frozenset({"dag_consistency", "depth_spearman_rho"})

# Metric direction: True = higher is better, False = lower is better.
METRIC_HIGHER_IS_BETTER: Dict[str, bool] = {
    "edge_length_cv": False,
    "dag_consistency": True,
    "depth_spearman_rho": True,
    "overlap_count": False,
    "edge_straightness_mean_deg": False,
    "sampled_stress": False,
    "crossing_rate": False,
    "edge_crossings": False,
    "aspect_ratio_deviation": False,
}

FAMILY_TAG_MAP: List[Tuple[str, str]] = [
    ("hub-spoke", "hub_spoke"),
    ("compound", "compound"),
    ("tree", "tree"),
    ("dependency", "dependency"),
    ("small-world", "small_world"),
    ("scale-free", "scale_free"),
    ("community", "community"),
    ("bipartite", "bipartite"),
    ("grid", "grid"),
    ("mesh", "grid"),
    ("lattice", "grid"),
    ("neural-net", "neural_net"),
    ("geometric", "geometric"),
    ("spatial", "geometric"),
    ("linear-shallow", "linear_shallow"),
    ("linear-deep", "linear_deep"),
    ("skip-light", "skip_light"),
    ("skip-heavy", "skip_heavy"),
    ("diamond", "diamond"),
    ("nested-shallow", "nested_shallow"),
    ("nested-deep", "nested_deep"),
    ("mixed-width", "mixed_width"),
    ("large-sparse", "large_sparse"),
    ("large-dense", "large_dense"),
    ("wide-layer", "wide_layer"),
    ("wide-parallel", "wide_layer"),
    ("erdos-renyi", "random"),
    ("random", "random"),
    ("clustered", "clustered"),
    ("cyclic", "cyclic"),
]

THRESHOLDS: Dict[str, Dict[str, float]] = {
    "dag_consistency": {"steal_abs": 0.05, "premium_abs": 0.10},
    "depth_spearman_rho": {"steal_abs": 0.05, "premium_abs": 0.10},
    "edge_straightness_mean_deg": {"steal_abs": 3.0, "premium_abs": 5.0},
    "overlap_count": {"steal_abs": 5.0, "premium_abs": 20.0},
    "sampled_stress": {"steal_pct": 0.15, "premium_pct": 0.30, "floor": 1e-3},
    "edge_length_cv": {"steal_pct": 0.15, "premium_pct": 0.30, "floor": 1e-3},
    "crossing_rate": {"steal_pct": 0.15, "premium_pct": 0.30, "floor": 1e-4},
    "edge_crossings": {"steal_pct": 0.15, "premium_pct": 0.30, "floor": 1.0},
    "aspect_ratio_deviation": {"steal_abs": 0.10, "premium_abs": 0.30},
}

SUMMARY_COLUMNS: Tuple[str, ...] = (
    "graph_family",
    "metric_name",
    "engine_name",
    "engine_family",
    "higher_is_better",
    "graphs_in_family_total",
    "graphs_in_family_available",
    "graphs_scheduled",
    "graphs_covered",
    "coverage_ratio",
    "graphs_ranked",
    "scorecard_eligible",
    "metric_median",
    "metric_p25",
    "metric_p75",
    "median_graph_rank",
    "win_rate",
    "top3_rate",
    "median_rel_best",
    "median_runtime_rel_fastest",
)

INSIGHT_COLUMNS: Tuple[str, ...] = (
    "graph_family",
    "metric_name",
    "insight_type",
    "dagua_engine_name",
    "competitor_engine_name",
    "competitor_engine_family",
    "dagua_metric_median",
    "competitor_metric_median",
    "quality_advantage",
    "quality_advantage_norm",
    "runtime_ratio",
    "dagua_runtime_rel_fastest",
    "competitor_runtime_rel_fastest",
    "family_metric_p25",
    "family_metric_p50",
    "family_metric_p75",
)

BEST_OF_BREED_COLUMNS: Tuple[str, ...] = (
    "engine_name",
    "engine_family",
    "pareto_appearances",
    "pareto_family_count",
    "pareto_metric_count",
    "best_quality_count",
    "fastest_count",
    "balanced_count",
    "dagua_anchor_count",
    "families",
    "metrics",
)

FamilyMetricTableMap = Dict[Tuple[str, str], pd.DataFrame]
FamilySummaryTables = Dict[str, pd.DataFrame]


@dataclass
class MetricTask:
    """One unit of metric recomputation work.

    Parameters
    ----------
    record_key : str
        Stable key from ``results.json``.
    graph_name : str
        Benchmark graph name.
    engine_name : str
        Engine identifier for the layout.
    layout_seed : Optional[int]
        Seed attached to the benchmark run.
    positions_file : Optional[str]
        Relative path to the saved ``.pt`` tensor.
    input_dir : Path
        Benchmark artifact directory.
    num_nodes : int
        Graph node count used for tensor validation.
    cache_path : Optional[Path]
        Cache location for this layout's recomputed metrics.
    stress_sources : int, optional
        ``sampled_stress`` source budget.
    stress_targets : int, optional
        ``sampled_stress`` target budget.
    crossing_samples : int, optional
        ``sampled_crossing_rate`` pair-sampling budget.
    skip_sampled : bool, optional
        Whether sampled metrics should be replaced with ``NaN``.
    """

    record_key: str
    graph_name: str
    engine_name: str
    layout_seed: Optional[int]
    positions_file: Optional[str]
    input_dir: Path
    num_nodes: int
    cache_path: Optional[Path]
    stress_sources: int = 32
    stress_targets: int = 128
    crossing_samples: int = 50_000
    skip_sampled: bool = False


_worker_h5: Any = None
_worker_graphs: Any = None


def optional_int(value: Any) -> Optional[int]:
    """Parse an optional integer.

    Parameters
    ----------
    value : Any
        Raw value from JSON.

    Returns
    -------
    Optional[int]
        Parsed integer, or ``None`` when conversion fails.
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def optional_float(value: Any) -> Optional[float]:
    """Parse an optional float.

    Parameters
    ----------
    value : Any
        Raw value from JSON.

    Returns
    -------
    Optional[float]
        Parsed float, or ``None`` when conversion fails.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def finite_numeric_values(values: Iterable[Any]) -> List[float]:
    """Collect finite numeric values in order.

    Parameters
    ----------
    values : Iterable[Any]
        Candidate values.

    Returns
    -------
    List[float]
        Finite numeric values converted to ``float``.
    """
    collected: List[float] = []
    for value in values:
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            collected.append(float(value))
    return collected


def is_finite_number(value: Any) -> bool:
    """Return whether a value is a finite numeric scalar.

    Parameters
    ----------
    value : Any
        Candidate value.

    Returns
    -------
    bool
        ``True`` when the value is an ``int`` or ``float`` and finite.
    """
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def safe_median(values: Sequence[float]) -> float:
    """Return the median of a numeric series.

    Parameters
    ----------
    values : Sequence[float]
        Numeric values.

    Returns
    -------
    float
        Median value, or ``nan`` when empty.
    """
    if not values:
        return float("nan")
    return float(np.median(np.asarray(values, dtype=np.float64)))


def safe_quantile(values: Sequence[float], quantile: float) -> float:
    """Return a quantile of a numeric series.

    Parameters
    ----------
    values : Sequence[float]
        Numeric values.
    quantile : float
        Requested quantile in ``[0, 1]``.

    Returns
    -------
    float
        Quantile value, or ``nan`` when empty.
    """
    if not values:
        return float("nan")
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def sanitize_slug(value: str) -> str:
    """Convert a label to a conservative filename slug.

    Parameters
    ----------
    value : str
        Input label.

    Returns
    -------
    str
        Lower-risk ASCII-ish slug suitable for output filenames.
    """
    allowed = []
    for char in value:
        if char.isalnum() or char in {"_", "-"}:
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed)


def derive_graph_family(
    graph_name: str,
    tags: List[str],
    num_nodes: int,
) -> Tuple[str, str, str]:
    """Derive structural family and size labels for one graph.

    Parameters
    ----------
    graph_name : str
        Stable graph name from the benchmark.
    tags : List[str]
        Benchmark graph tags.
    num_nodes : int
        Graph node count.

    Returns
    -------
    Tuple[str, str, str]
        ``(graph_family, graph_size_token, graph_size_bucket)``.
    """
    import re

    tag_set = set(tags or [])
    family = "misc"
    for tag_pattern, family_label in FAMILY_TAG_MAP:
        if tag_pattern in tag_set:
            family = family_label
            break

    match = re.search(r"_([0-9]+x[0-9]+(?:x[0-9]+)*|[0-9]+[km]?)$", graph_name or "")
    size_token = match.group(1) if match else str(num_nodes)

    if num_nodes < 20:
        bucket = "tiny"
    elif num_nodes < 100:
        bucket = "small"
    elif num_nodes < 1000:
        bucket = "medium"
    elif num_nodes < 10000:
        bucket = "large"
    else:
        bucket = "xlarge"
    return family, size_token, bucket


def load_benchmark_records(input_dir: Path) -> pd.DataFrame:
    """Load benchmark results and manifest data into a tidy DataFrame.

    Parameters
    ----------
    input_dir : Path
        Benchmark directory containing ``results.json`` and ``manifest.json``.

    Returns
    -------
    pandas.DataFrame
        Long-format records with all benchmark statuses preserved.
    """
    results_path = input_dir / "results.json"
    manifest_path = input_dir / "manifest.json"

    with results_path.open("r", encoding="utf-8") as handle:
        results = json.load(handle)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    graph_meta: Dict[str, Dict[str, Any]] = {}
    for graph_payload in manifest.get("graphs", []):
        if not isinstance(graph_payload, dict):
            continue
        name = str(graph_payload.get("name", ""))
        graph_meta[name] = {
            "tags": sorted(str(tag) for tag in graph_payload.get("tags", [])),
            "num_nodes": optional_int(graph_payload.get("num_nodes")),
            "num_edges": optional_int(graph_payload.get("num_edges")),
        }

    engine_meta: Dict[str, Dict[str, Any]] = {}
    for engine_payload in manifest.get("engines", []):
        if not isinstance(engine_payload, dict):
            continue
        name = str(engine_payload.get("name", ""))
        engine_meta[name] = {
            "available": bool(engine_payload.get("available", False)),
            "max_nodes": optional_int(engine_payload.get("max_nodes")),
            "is_heavy": bool(engine_payload.get("is_heavy", False)),
            "is_stochastic_manifest": bool(engine_payload.get("is_stochastic", False)),
            "original_for": sorted(str(item) for item in engine_payload.get("original_for", [])),
            "reimpl_of": sorted(str(item) for item in engine_payload.get("reimpl_of", [])),
        }

    rows: List[Dict[str, Any]] = []
    for record_key, payload in results.items():
        if not isinstance(payload, dict):
            continue
        graph_name = str(payload.get("graph_name", ""))
        engine_name = str(payload.get("engine_name", ""))
        graph_info = graph_meta.get(graph_name, {})
        engine_info = engine_meta.get(engine_name, {})
        num_nodes = optional_int(payload.get("num_nodes"))
        if num_nodes is None:
            num_nodes = optional_int(graph_info.get("num_nodes")) or 0
        num_edges = optional_int(payload.get("num_edges"))
        if num_edges is None:
            num_edges = optional_int(graph_info.get("num_edges")) or 0
        graph_tags = list(graph_info.get("tags", []))
        graph_family, size_token, size_bucket = derive_graph_family(
            graph_name, graph_tags, num_nodes
        )

        rows.append(
            {
                "record_key": str(record_key),
                "graph_name": graph_name,
                "engine_name": engine_name,
                "seed": optional_int(payload.get("seed")),
                "status": str(payload.get("status", "")),
                "runtime_seconds": optional_float(payload.get("runtime_seconds")),
                "positions_file": payload.get("positions_file"),
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "is_stochastic": bool(payload.get("is_stochastic", False)),
                "error": payload.get("error"),
                "skip_reason": payload.get("skip_reason"),
                "graph_tags": graph_tags,
                "num_nodes_manifest": optional_int(graph_info.get("num_nodes")),
                "num_edges_manifest": optional_int(graph_info.get("num_edges")),
                "engine_available_manifest": engine_info.get("available"),
                "engine_max_nodes_manifest": engine_info.get("max_nodes"),
                "engine_is_heavy_manifest": engine_info.get("is_heavy"),
                "engine_is_stochastic_manifest": engine_info.get("is_stochastic_manifest"),
                "engine_original_for": list(engine_info.get("original_for", [])),
                "engine_reimpl_of": list(engine_info.get("reimpl_of", [])),
                "engine_family": algorithm_family(engine_name) if engine_name else "unknown",
                "graph_family": graph_family,
                "graph_size_token": size_token,
                "graph_size_bucket": size_bucket,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _worker_init(h5_path: Path) -> None:
    """Initialize per-worker shared state.

    Parameters
    ----------
    h5_path : Path
        Benchmark ``positions.h5`` path.

    Returns
    -------
    None
        Worker globals are initialized in place.
    """
    global _worker_h5, _worker_graphs
    _worker_h5 = open_h5_for_worker(h5_path)
    _worker_graphs = {test_graph.name: test_graph.graph for test_graph in get_test_graphs()}


def _compute_metrics_for_task(task: MetricTask) -> Dict[str, Any]:
    """Load and recompute metrics for one benchmark record.

    Parameters
    ----------
    task : MetricTask
        Metric recomputation task.

    Returns
    -------
    Dict[str, Any]
        Result payload containing ``record_key`` plus metrics or rejection
        reason fields.
    """
    global _worker_h5, _worker_graphs

    if task.cache_path is not None and task.cache_path.exists():
        try:
            with task.cache_path.open("r", encoding="utf-8") as handle:
                cached = json.load(handle)
            return {
                "record_key": task.record_key,
                "metrics": cached,
                "reason": None,
                "from_cache": True,
            }
        except Exception:
            LOGGER.debug("Ignoring unreadable cache file %s", task.cache_path, exc_info=True)

    tensor, reason = load_position_tensor(
        record_key=task.record_key,
        positions_file=task.positions_file,
        input_dir=task.input_dir,
        h5_file=_worker_h5,
    )
    if tensor is None:
        return {
            "record_key": task.record_key,
            "metrics": None,
            "reason": reason,
            "from_cache": False,
        }

    graph = _worker_graphs.get(task.graph_name)
    if graph is None:
        return {
            "record_key": task.record_key,
            "metrics": None,
            "reason": "graph_not_in_registry",
            "from_cache": False,
        }

    graph.compute_node_sizes()
    edge_index = graph.edge_index
    node_sizes = graph.node_sizes
    if node_sizes is None:
        return {
            "record_key": task.record_key,
            "metrics": None,
            "reason": "missing_node_sizes",
            "from_cache": False,
        }

    shape_reason = validate_positions(tensor, int(node_sizes.shape[0]))
    if shape_reason is not None:
        return {
            "record_key": task.record_key,
            "metrics": None,
            "reason": shape_reason,
            "from_cache": False,
        }

    seed = stable_seed(task.graph_name, task.engine_name, str(task.layout_seed or 0))
    try:
        quick_metrics = compute_quick_metrics_seeded(
            tensor,
            edge_index,
            node_sizes,
            seed=seed,
            metric_filter=QR_QUICK_METRICS,
        )
        quick_metrics["aspect_ratio_deviation"] = aspect_ratio_deviation(tensor)
        if task.skip_sampled:
            sampled_metrics = {metric_name: float("nan") for metric_name in QR_SAMPLED_METRICS}
        else:
            sampled_metrics = compute_sampled_metrics_seeded(
                tensor,
                edge_index,
                int(node_sizes.shape[0]),
                seed=seed,
                stress_sources=task.stress_sources,
                stress_targets=task.stress_targets,
                crossing_samples=task.crossing_samples,
            )
    except Exception as exc:
        return {
            "record_key": task.record_key,
            "metrics": None,
            "reason": f"compute_failure: {exc}",
            "from_cache": False,
        }

    all_metrics = {**quick_metrics, **sampled_metrics}
    if task.cache_path is not None:
        try:
            task.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = task.cache_path.with_suffix(".tmp")
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(all_metrics, handle)
            tmp_path.replace(task.cache_path)
        except Exception:
            LOGGER.debug("Ignoring cache write failure for %s", task.cache_path, exc_info=True)

    return {
        "record_key": task.record_key,
        "metrics": all_metrics,
        "reason": None,
        "from_cache": False,
    }


def compute_cache_key(
    record_key: str,
    stress_sources: int,
    stress_targets: int,
    crossing_samples: int,
    metrics_module_hash: str,
    fix_s_version: str = "v1",
) -> str:
    """Build a stable cache key from record identity and metric config.

    Parameters
    ----------
    record_key : str
        Stable benchmark record key.
    stress_sources : int
        ``sampled_stress`` source budget.
    stress_targets : int
        ``sampled_stress`` target budget.
    crossing_samples : int
        ``sampled_crossing_rate`` pair-sampling budget.
    metrics_module_hash : str
        Content hash of ``dagua/metrics.py``.
    fix_s_version : str, optional
        Seed-policy version tag.

    Returns
    -------
    str
        Stable truncated SHA-256 key.
    """
    joined = "::".join(
        [
            record_key,
            f"ss{stress_sources}",
            f"st{stress_targets}",
            f"cs{crossing_samples}",
            f"mh{metrics_module_hash}",
            f"fs{fix_s_version}",
        ]
    )
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:32]


def compute_metrics_module_hash() -> str:
    """Hash the current ``dagua.metrics`` module file contents.

    Returns
    -------
    str
        Truncated SHA-256 digest of ``dagua/metrics.py``.
    """
    import dagua.metrics

    path = Path(dagua.metrics.__file__)
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def score_engines_on_graph(
    df: pd.DataFrame,
    metric: str,
    higher_is_better: bool,
) -> pd.DataFrame:
    """Assign per-graph ranks and relative-best normalization.

    Parameters
    ----------
    df : pandas.DataFrame
        Rows for one graph and one metric.
    metric : str
        Metric column name.
    higher_is_better : bool
        Whether larger metric values are preferred.

    Returns
    -------
    pandas.DataFrame
        Ranked rows with ``graph_rank``, ``rel_best``, and
        ``runtime_rel_fastest`` columns.
    """
    ranked = df.copy()
    ranked = ranked[ranked[metric].map(is_finite_number)]
    if ranked.empty:
        return ranked

    ascending = not higher_is_better
    ranked = ranked.sort_values(
        by=[metric, "runtime_seconds", "engine_name"],
        ascending=[ascending, True, True],
        na_position="last",
    ).reset_index(drop=True)
    ranked["graph_rank"] = np.arange(1, len(ranked) + 1, dtype=np.int64)

    best = float(ranked[metric].iloc[0])
    typical_scale = max(abs(float(ranked[metric].median())), 1e-3)

    def rel_best(value: Any) -> float:
        numeric_value = float(value)
        if higher_is_better:
            gap = best - numeric_value
            denom = max(abs(best), typical_scale)
        else:
            gap = numeric_value - best
            denom = max(best, typical_scale)
        raw = gap / denom if denom > 0 else 0.0
        return min(max(raw, 0.0), 10.0)

    ranked["rel_best"] = ranked[metric].map(rel_best)
    runtime_values = finite_numeric_values(ranked["runtime_seconds"].tolist())
    if runtime_values:
        fastest = min(runtime_values)
        ranked["runtime_rel_fastest"] = ranked["runtime_seconds"].map(
            lambda value: (
                float(value) / max(fastest, 1e-6)
                if isinstance(value, (int, float)) and math.isfinite(float(value))
                else float("nan")
            )
        )
    else:
        ranked["runtime_rel_fastest"] = float("nan")
    return ranked


def compute_engine_coverage(
    records_df: pd.DataFrame,
    graph_family_name: str,
) -> Dict[str, Dict[str, float]]:
    """Compute per-engine family coverage from all statuses.

    Parameters
    ----------
    records_df : pandas.DataFrame
        Benchmark records with all statuses included.
    graph_family_name : str
        Family label to summarize.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Coverage statistics keyed by engine name.
    """
    family_rows = records_df[records_df["graph_family"] == graph_family_name]
    scheduled: Dict[str, set[str]] = {}
    covered: Dict[str, set[str]] = {}

    for row in family_rows.itertuples():
        engine_name = str(row.engine_name)
        graph_name = str(row.graph_name)
        scheduled.setdefault(engine_name, set()).add(graph_name)
        if row.status == "ok":
            covered.setdefault(engine_name, set()).add(graph_name)

    return {
        engine_name: {
            "graphs_scheduled": float(len(scheduled_graphs)),
            "graphs_covered": float(len(covered.get(engine_name, set()))),
            "coverage_ratio": float(
                len(covered.get(engine_name, set())) / max(len(scheduled_graphs), 1)
            ),
        }
        for engine_name, scheduled_graphs in scheduled.items()
    }


def should_skip_family_metric(records_df: pd.DataFrame, family_name: str, metric_name: str) -> bool:
    """Return whether a family/metric pair should be excluded.

    Parameters
    ----------
    records_df : pandas.DataFrame
        Benchmark records including metric columns.
    family_name : str
        Graph family label.
    metric_name : str
        Metric under consideration.

    Returns
    -------
    bool
        ``True`` when the family/metric pair should be skipped.
    """
    if metric_name not in DAG_ORDER_METRICS:
        return False
    if "cyclic" in family_name:
        return True
    family_rows = records_df[
        (records_df["graph_family"] == family_name)
        & (records_df["status"] == "ok")
        & records_df["dag_consistency"].map(is_finite_number)
    ]
    if family_rows.empty:
        return False
    return float(family_rows["dag_consistency"].median()) < 0.5


def compute_family_summary(
    records_df: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    FamilyMetricTableMap,
    FamilyMetricTableMap,
    FamilySummaryTables,
]:
    """Aggregate per-graph rankings into family-level scorecards.

    Parameters
    ----------
    records_df : pandas.DataFrame
        Benchmark records with recomputed metric columns.

    Returns
    -------
    Tuple[pandas.DataFrame, FamilyMetricTableMap, FamilyMetricTableMap, FamilySummaryTables]
        Summary table, per-family/metric top-k tables, Pareto tables, and
        per-family leader tables.
    """
    summary_rows: List[Dict[str, Any]] = []
    topk_tables: FamilyMetricTableMap = {}
    pareto_tables: FamilyMetricTableMap = {}
    family_tables: FamilySummaryTables = {}

    if records_df.empty:
        return (
            pd.DataFrame(columns=SUMMARY_COLUMNS),
            topk_tables,
            pareto_tables,
            family_tables,
        )

    family_names = sorted(str(family) for family in records_df["graph_family"].dropna().unique())
    for family_name in family_names:
        coverage = compute_engine_coverage(records_df, family_name)
        family_rows = records_df[records_df["graph_family"] == family_name]
        graphs_in_family_total = int(family_rows["graph_name"].nunique())
        graphs_in_family_available = int(
            family_rows.loc[family_rows["status"] == "ok", "graph_name"].nunique()
        )
        leader_rows: List[Dict[str, Any]] = []

        for metric_name in ALL_ANALYSIS_METRICS:
            if should_skip_family_metric(records_df, family_name, metric_name):
                continue
            metric_rows = family_rows[
                (family_rows["status"] == "ok") & family_rows[metric_name].map(is_finite_number)
            ].copy()
            if metric_rows.empty:
                continue

            ranked_parts: List[pd.DataFrame] = []
            for graph_name, graph_group in metric_rows.groupby("graph_name", sort=True):
                ranked_group = score_engines_on_graph(
                    graph_group,
                    metric_name,
                    METRIC_HIGHER_IS_BETTER[metric_name],
                )
                if ranked_group.empty:
                    continue
                ranked_group["graph_name"] = graph_name
                ranked_group["graph_family"] = family_name
                ranked_group["metric_name"] = metric_name
                ranked_parts.append(ranked_group)

            if not ranked_parts:
                continue
            scored_df = pd.concat(ranked_parts, ignore_index=True)

            metric_summary_rows: List[Dict[str, Any]] = []
            for engine_name, engine_rows in scored_df.groupby("engine_name", sort=True):
                coverage_info = coverage.get(
                    str(engine_name),
                    {"graphs_scheduled": 0.0, "graphs_covered": 0.0, "coverage_ratio": 0.0},
                )
                metric_values = finite_numeric_values(engine_rows[metric_name].tolist())
                graph_ranks = finite_numeric_values(engine_rows["graph_rank"].tolist())
                rel_best_values = finite_numeric_values(engine_rows["rel_best"].tolist())
                runtime_rel_values = finite_numeric_values(
                    engine_rows["runtime_rel_fastest"].tolist()
                )

                summary_row = {
                    "graph_family": family_name,
                    "metric_name": metric_name,
                    "engine_name": str(engine_name),
                    "engine_family": algorithm_family(str(engine_name)),
                    "higher_is_better": METRIC_HIGHER_IS_BETTER[metric_name],
                    "graphs_in_family_total": graphs_in_family_total,
                    "graphs_in_family_available": graphs_in_family_available,
                    "graphs_scheduled": int(coverage_info["graphs_scheduled"]),
                    "graphs_covered": int(coverage_info["graphs_covered"]),
                    "coverage_ratio": float(coverage_info["coverage_ratio"]),
                    "graphs_ranked": int(engine_rows["graph_name"].nunique()),
                    "scorecard_eligible": bool(
                        graphs_in_family_available >= 3
                        and float(coverage_info["coverage_ratio"]) >= 0.5
                    ),
                    "metric_median": safe_median(metric_values),
                    "metric_p25": safe_quantile(metric_values, 0.25),
                    "metric_p75": safe_quantile(metric_values, 0.75),
                    "median_graph_rank": safe_median(graph_ranks),
                    "win_rate": float((engine_rows["graph_rank"] == 1).mean()),
                    "top3_rate": float((engine_rows["graph_rank"] <= 3).mean()),
                    "median_rel_best": safe_median(rel_best_values),
                    "median_runtime_rel_fastest": safe_median(runtime_rel_values),
                }
                summary_rows.append(summary_row)
                metric_summary_rows.append(summary_row)

            metric_summary_df = pd.DataFrame(metric_summary_rows, columns=SUMMARY_COLUMNS)
            metric_summary_df = metric_summary_df.sort_values(
                by=[
                    "median_graph_rank",
                    "median_rel_best",
                    "median_runtime_rel_fastest",
                    "engine_name",
                ],
                ascending=[True, True, True, True],
                na_position="last",
            ).reset_index(drop=True)
            topk_tables[(family_name, metric_name)] = metric_summary_df.head(10).copy()

            pareto_input = metric_summary_df[
                metric_summary_df["scorecard_eligible"]
                & metric_summary_df["median_rel_best"].map(math.isfinite)
                & metric_summary_df["median_runtime_rel_fastest"].map(math.isfinite)
            ].copy()
            pareto_tables[(family_name, metric_name)] = compute_pareto_front(pareto_input)

            if not metric_summary_df.empty:
                winner = metric_summary_df.iloc[0].to_dict()
                leader_rows.append(
                    {
                        "metric_name": metric_name,
                        "winner_engine_name": winner["engine_name"],
                        "winner_engine_family": winner["engine_family"],
                        "median_graph_rank": winner["median_graph_rank"],
                        "median_rel_best": winner["median_rel_best"],
                        "median_runtime_rel_fastest": winner["median_runtime_rel_fastest"],
                        "coverage_ratio": winner["coverage_ratio"],
                        "pareto_engines": ";".join(
                            pareto_tables[(family_name, metric_name)]["engine_name"].tolist()
                        ),
                    }
                )

        family_tables[family_name] = pd.DataFrame(leader_rows)

    summary_df = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    if summary_df.empty:
        summary_df = pd.DataFrame(columns=SUMMARY_COLUMNS)
    else:
        summary_df = summary_df.sort_values(
            by=[
                "graph_family",
                "metric_name",
                "median_graph_rank",
                "median_rel_best",
                "engine_name",
            ],
            ascending=[True, True, True, True, True],
            na_position="last",
        ).reset_index(drop=True)
    return summary_df, topk_tables, pareto_tables, family_tables


def compute_pareto_front(summary_df: pd.DataFrame, epsilon: float = 1e-9) -> pd.DataFrame:
    """Compute the Pareto front for runtime-vs-quality points.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        Family/metric summary rows with runtime and quality coordinates.
    epsilon : float, optional
        Dominance tolerance.

    Returns
    -------
    pandas.DataFrame
        Pareto-optimal rows annotated with role columns.
    """
    if summary_df.empty:
        return summary_df.copy()

    rows = summary_df.copy().reset_index(drop=True)
    x_values = rows["median_runtime_rel_fastest"]
    y_values = rows["median_rel_best"]
    rows = rows[x_values.map(is_finite_number) & y_values.map(is_finite_number)].reset_index(
        drop=True
    )
    if rows.empty:
        return rows

    dominated: List[bool] = []
    for idx, row in rows.iterrows():
        row_x = float(row["median_runtime_rel_fastest"])
        row_y = float(row["median_rel_best"])
        is_dominated = False
        for other_idx, other_row in rows.iterrows():
            if idx == other_idx:
                continue
            other_x = float(other_row["median_runtime_rel_fastest"])
            other_y = float(other_row["median_rel_best"])
            no_worse = other_x <= row_x + epsilon and other_y <= row_y + epsilon
            strictly_better = other_x < row_x - epsilon or other_y < row_y - epsilon
            if no_worse and strictly_better:
                is_dominated = True
                break
        dominated.append(is_dominated)

    pareto = rows.loc[[not flag for flag in dominated]].copy().reset_index(drop=True)
    if pareto.empty:
        return pareto

    min_x = float(pareto["median_runtime_rel_fastest"].min())
    min_y = float(pareto["median_rel_best"].min())
    distances = np.sqrt(
        (pareto["median_runtime_rel_fastest"].to_numpy(dtype=np.float64) - 1.0) ** 2
        + pareto["median_rel_best"].to_numpy(dtype=np.float64) ** 2
    )
    min_distance = float(distances.min())

    pareto["is_best_quality"] = pareto["median_rel_best"].map(
        lambda value: abs(float(value) - min_y) <= epsilon
    )
    pareto["is_fastest"] = pareto["median_runtime_rel_fastest"].map(
        lambda value: abs(float(value) - min_x) <= epsilon
    )
    pareto["is_balanced"] = [
        abs(float(distance) - min_distance) <= epsilon for distance in distances
    ]
    pareto["is_dagua_anchor"] = pareto["engine_name"] == "dagua"

    def build_roles(row: pd.Series) -> str:
        """Build the compact role label string for one Pareto row."""
        roles: List[str] = []
        if bool(row["is_best_quality"]):
            roles.append("best_quality")
        if bool(row["is_fastest"]):
            roles.append("fastest")
        if bool(row["is_balanced"]):
            roles.append("balanced")
        if bool(row["is_dagua_anchor"]):
            roles.append("dagua_anchor")
        return ",".join(roles)

    pareto["roles"] = pareto.apply(build_roles, axis=1)
    pareto = pareto.sort_values(
        by=["median_runtime_rel_fastest", "median_rel_best", "engine_name"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    return pareto


def metric_advantage(
    metric_name: str,
    baseline_value: float,
    candidate_value: float,
    higher_is_better: bool,
) -> Tuple[float, float]:
    """Compute absolute and normalized advantage of a candidate over baseline.

    Parameters
    ----------
    metric_name : str
        Metric identifier.
    baseline_value : float
        Baseline metric value.
    candidate_value : float
        Candidate metric value.
    higher_is_better : bool
        Whether larger metric values are preferred.

    Returns
    -------
    Tuple[float, float]
        Absolute and normalized advantage, both positive when the candidate is
        better than the baseline.
    """
    if higher_is_better:
        absolute_advantage = candidate_value - baseline_value
    else:
        absolute_advantage = baseline_value - candidate_value

    threshold_config = THRESHOLDS.get(metric_name, {})
    if "floor" in threshold_config:
        denom = max(abs(baseline_value), float(threshold_config["floor"]))
    else:
        denom = 1.0
    return absolute_advantage, absolute_advantage / denom


def extract_dagua_default_insights(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Extract dagua-vs-competitor insights from family scorecards.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        Family/metric summary rows.

    Returns
    -------
    pandas.DataFrame
        One row per extracted insight.
    """
    insight_rows: List[Dict[str, Any]] = []
    if summary_df.empty:
        return pd.DataFrame(columns=INSIGHT_COLUMNS)

    grouped = summary_df.groupby(["graph_family", "metric_name"], sort=True)
    for (graph_family, metric_name), group in grouped:
        dagua_rows = group[group["engine_name"] == "dagua"]
        if dagua_rows.empty:
            continue

        dagua_row = dagua_rows.iloc[0]
        if int(dagua_row["graphs_covered"]) < 3 or float(dagua_row["coverage_ratio"]) < 0.5:
            continue
        if int(dagua_row["graphs_ranked"]) < 3:
            continue

        family_metric_values = finite_numeric_values(group["metric_median"].tolist())
        higher_is_better = bool(dagua_row["higher_is_better"])
        threshold_config = THRESHOLDS.get(str(metric_name), {})
        dagua_metric = float(dagua_row["metric_median"])
        dagua_runtime = float(dagua_row["median_runtime_rel_fastest"])
        dagua_rel_best = float(dagua_row["median_rel_best"])

        for _, competitor_row in group.iterrows():
            if str(competitor_row["engine_name"]) == "dagua":
                continue
            if (
                int(competitor_row["graphs_covered"]) < 3
                or float(competitor_row["coverage_ratio"]) < 0.5
            ):
                continue
            if int(competitor_row["graphs_ranked"]) < 3:
                continue

            competitor_metric = float(competitor_row["metric_median"])
            competitor_runtime = float(competitor_row["median_runtime_rel_fastest"])
            competitor_rel_best = float(competitor_row["median_rel_best"])
            runtime_ratio = competitor_runtime / max(dagua_runtime, 1e-6)
            quality_advantage, quality_advantage_norm = metric_advantage(
                str(metric_name),
                dagua_metric,
                competitor_metric,
                higher_is_better,
            )
            reverse_advantage, reverse_advantage_norm = metric_advantage(
                str(metric_name),
                competitor_metric,
                dagua_metric,
                higher_is_better,
            )

            def add_row(insight_type: str, advantage: float, normalized: float) -> None:
                """Append one insight row with shared family context."""
                insight_rows.append(
                    {
                        "graph_family": graph_family,
                        "metric_name": metric_name,
                        "insight_type": insight_type,
                        "dagua_engine_name": "dagua",
                        "competitor_engine_name": competitor_row["engine_name"],
                        "competitor_engine_family": competitor_row["engine_family"],
                        "dagua_metric_median": dagua_metric,
                        "competitor_metric_median": competitor_metric,
                        "quality_advantage": advantage,
                        "quality_advantage_norm": normalized,
                        "runtime_ratio": runtime_ratio,
                        "dagua_runtime_rel_fastest": dagua_runtime,
                        "competitor_runtime_rel_fastest": competitor_runtime,
                        "family_metric_p25": safe_quantile(family_metric_values, 0.25),
                        "family_metric_p50": safe_median(family_metric_values),
                        "family_metric_p75": safe_quantile(family_metric_values, 0.75),
                    }
                )

            steal_abs = threshold_config.get("steal_abs")
            premium_abs = threshold_config.get("premium_abs")
            steal_pct = threshold_config.get("steal_pct")
            premium_pct = threshold_config.get("premium_pct")

            if (
                steal_abs is not None
                and quality_advantage >= float(steal_abs)
                and runtime_ratio <= 1.25
            ):
                add_row("steal_from", quality_advantage, quality_advantage_norm)
            if (
                premium_abs is not None
                and quality_advantage >= float(premium_abs)
                and runtime_ratio <= 2.0
            ):
                add_row("premium_quality", quality_advantage, quality_advantage_norm)
            if (
                steal_pct is not None
                and quality_advantage_norm >= float(steal_pct)
                and runtime_ratio <= 1.25
            ):
                add_row("steal_from", quality_advantage, quality_advantage_norm)
            if (
                premium_pct is not None
                and quality_advantage_norm >= float(premium_pct)
                and runtime_ratio <= 2.0
            ):
                add_row("premium_quality", quality_advantage, quality_advantage_norm)

            competitor_dominates = (
                competitor_runtime <= dagua_runtime + 1e-9
                and competitor_rel_best <= dagua_rel_best + 1e-9
                and (
                    competitor_runtime < dagua_runtime - 1e-9
                    or competitor_rel_best < dagua_rel_best - 1e-9
                )
            )
            if competitor_dominates:
                add_row("dagua_dominated", quality_advantage, quality_advantage_norm)

            dagua_runtime_ratio = dagua_runtime / max(competitor_runtime, 1e-6)
            dagua_wins = False
            if (
                steal_abs is not None
                and reverse_advantage >= float(steal_abs)
                and dagua_runtime_ratio <= 1.25
            ):
                dagua_wins = True
            if (
                steal_pct is not None
                and reverse_advantage_norm >= float(steal_pct)
                and dagua_runtime_ratio <= 1.25
            ):
                dagua_wins = True
            if dagua_wins:
                add_row("dagua_competitor_winner", reverse_advantage, reverse_advantage_norm)

    insights_df = pd.DataFrame(insight_rows, columns=INSIGHT_COLUMNS)
    if insights_df.empty:
        return pd.DataFrame(columns=INSIGHT_COLUMNS)
    return (
        insights_df.drop_duplicates()
        .sort_values(
            by=["graph_family", "metric_name", "insight_type", "competitor_engine_name"],
            ascending=[True, True, True, True],
        )
        .reset_index(drop=True)
    )


def extract_best_of_breed(pareto_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Pareto-optimal engines across families and metrics.

    Parameters
    ----------
    pareto_df : pandas.DataFrame
        Concatenated Pareto rows across families and metrics.

    Returns
    -------
    pandas.DataFrame
        Cross-family best-of-breed configuration summary.
    """
    if pareto_df.empty:
        return pd.DataFrame(columns=BEST_OF_BREED_COLUMNS)

    rows: List[Dict[str, Any]] = []
    for engine_name, engine_rows in pareto_df.groupby("engine_name", sort=True):
        families = sorted(str(value) for value in engine_rows["graph_family"].unique())
        metrics = sorted(str(value) for value in engine_rows["metric_name"].unique())
        rows.append(
            {
                "engine_name": str(engine_name),
                "engine_family": algorithm_family(str(engine_name)),
                "pareto_appearances": int(len(engine_rows)),
                "pareto_family_count": int(len(families)),
                "pareto_metric_count": int(len(metrics)),
                "best_quality_count": int(engine_rows["is_best_quality"].sum()),
                "fastest_count": int(engine_rows["is_fastest"].sum()),
                "balanced_count": int(engine_rows["is_balanced"].sum()),
                "dagua_anchor_count": int(engine_rows["is_dagua_anchor"].sum()),
                "families": ";".join(families),
                "metrics": ";".join(metrics),
            }
        )

    best_df = pd.DataFrame(rows, columns=BEST_OF_BREED_COLUMNS)
    return best_df.sort_values(
        by=[
            "pareto_family_count",
            "pareto_metric_count",
            "best_quality_count",
            "balanced_count",
            "fastest_count",
            "engine_name",
        ],
        ascending=[False, False, False, False, False, True],
    ).reset_index(drop=True)


def write_pareto_plot(
    all_rows: pd.DataFrame,
    pareto_rows: pd.DataFrame,
    output_path: Path,
    metric_name: str,
) -> bool:
    """Write a Pareto scatter plot when plotting dependencies are available.

    Parameters
    ----------
    all_rows : pandas.DataFrame
        Full metric summary rows for one family/metric.
    pareto_rows : pandas.DataFrame
        Pareto-optimal subset.
    output_path : Path
        Destination PNG path.
    metric_name : str
        Metric identifier for axis labeling.

    Returns
    -------
    bool
        ``True`` when the plot was written.
    """
    if all_rows.empty or pareto_rows.empty:
        return False
    try:
        import matplotlib.pyplot as plt
    except Exception:
        LOGGER.warning("Skipping Pareto plot for %s because matplotlib is unavailable", metric_name)
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.0, 4.0))
    axis.scatter(
        all_rows["median_runtime_rel_fastest"],
        all_rows["median_rel_best"],
        color="#b0b0b0",
        alpha=0.7,
        label="eligible engines",
    )
    axis.scatter(
        pareto_rows["median_runtime_rel_fastest"],
        pareto_rows["median_rel_best"],
        color="#0b7285",
        label="Pareto front",
    )
    for row in pareto_rows.itertuples():
        axis.annotate(
            str(row.engine_name),
            (float(row.median_runtime_rel_fastest), float(row.median_rel_best)),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )

    runtime_values = finite_numeric_values(all_rows["median_runtime_rel_fastest"].tolist())
    if runtime_values and max(runtime_values) / max(min(runtime_values), 1e-6) > 10.0:
        axis.set_xscale("log")
    axis.set_xlabel("Median runtime relative to fastest")
    axis.set_ylabel("Median rel_best")
    axis.set_title(f"Pareto: {metric_name}")
    axis.grid(True, alpha=0.2)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return True


def write_csvs(
    output_dir: Path,
    records_df: pd.DataFrame,
    family_summary_df: pd.DataFrame,
    topk_tables: Dict[Tuple[str, str], pd.DataFrame],
    pareto_tables: Dict[Tuple[str, str], pd.DataFrame],
    family_tables: Dict[str, pd.DataFrame],
    insights_df: pd.DataFrame,
    best_of_breed_df: pd.DataFrame,
    *,
    write_plots: bool,
) -> pd.DataFrame:
    """Write sidecar CSVs and return an artifact index table.

    Parameters
    ----------
    output_dir : Path
        Destination directory.
    records_df : pandas.DataFrame
        Benchmark records snapshot.
    family_summary_df : pandas.DataFrame
        Aggregated family summary rows.
    topk_tables : Dict[Tuple[str, str], pandas.DataFrame]
        Per-family/metric top-k rows.
    pareto_tables : Dict[Tuple[str, str], pandas.DataFrame]
        Per-family/metric Pareto rows.
    family_tables : Dict[str, pandas.DataFrame]
        Per-family leader tables.
    insights_df : pandas.DataFrame
        Extracted dagua insight rows.
    best_of_breed_df : pandas.DataFrame
        Cross-family Pareto aggregation table.
    write_plots : bool
        Whether Pareto plots should be emitted.

    Returns
    -------
    pandas.DataFrame
        Artifact index table.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_rows: List[Dict[str, Any]] = []

    def record_artifact(
        path: Path, artifact_type: str, family: str = "", metric: str = "", rows: int = 0
    ) -> None:
        """Append one artifact-index row."""
        artifact_rows.append(
            {
                "path": str(path.relative_to(output_dir)),
                "artifact_type": artifact_type,
                "graph_family": family,
                "metric_name": metric,
                "rows": rows,
            }
        )

    snapshot_path = output_dir / "analysis_records_snapshot.csv"
    records_df.to_csv(snapshot_path, index=False)
    record_artifact(snapshot_path, "records_snapshot", rows=int(len(records_df)))

    summary_path = output_dir / "family_metric_summary.csv"
    family_summary_df.to_csv(summary_path, index=False)
    record_artifact(summary_path, "family_metric_summary", rows=int(len(family_summary_df)))

    for (family_name, metric_name), table in sorted(topk_tables.items()):
        topk_path = output_dir / (
            f"family_{sanitize_slug(family_name)}__metric_{sanitize_slug(metric_name)}__topk.csv"
        )
        table.to_csv(topk_path, index=False)
        record_artifact(
            topk_path, "family_topk", family=family_name, metric=metric_name, rows=int(len(table))
        )

    for (family_name, metric_name), table in sorted(pareto_tables.items()):
        pareto_path = output_dir / (
            f"family_{sanitize_slug(family_name)}__metric_{sanitize_slug(metric_name)}__pareto.csv"
        )
        table.to_csv(pareto_path, index=False)
        record_artifact(
            pareto_path,
            "family_pareto",
            family=family_name,
            metric=metric_name,
            rows=int(len(table)),
        )

        if write_plots and metric_name in INLINE_PARETO_METRICS and not table.empty:
            plot_path = output_dir / (
                f"family_{sanitize_slug(family_name)}__metric_{sanitize_slug(metric_name)}__pareto.png"
            )
            summary_rows = family_summary_df[
                (family_summary_df["graph_family"] == family_name)
                & (family_summary_df["metric_name"] == metric_name)
                & family_summary_df["scorecard_eligible"]
            ]
            if write_pareto_plot(summary_rows, table, plot_path, metric_name):
                record_artifact(
                    plot_path,
                    "family_pareto_plot",
                    family=family_name,
                    metric=metric_name,
                    rows=int(len(table)),
                )

    for family_name, table in sorted(family_tables.items()):
        family_path = output_dir / f"family_{sanitize_slug(family_name)}__summary.csv"
        table.to_csv(family_path, index=False)
        record_artifact(family_path, "family_summary", family=family_name, rows=int(len(table)))

    insights_path = output_dir / "dagua_default_insights.csv"
    insights_df.to_csv(insights_path, index=False)
    record_artifact(insights_path, "dagua_default_insights", rows=int(len(insights_df)))

    best_path = output_dir / "best_of_breed_configs.csv"
    best_of_breed_df.to_csv(best_path, index=False)
    record_artifact(best_path, "best_of_breed", rows=int(len(best_of_breed_df)))

    telemetry_path = output_dir / "validate_sync_telemetry.json"
    if telemetry_path.exists():
        record_artifact(telemetry_path, "validate_sync_telemetry", rows=1)

    record_artifact(
        output_dir / "artifact_index.csv", "artifact_index", rows=len(artifact_rows) + 1
    )
    artifact_df = pd.DataFrame(artifact_rows)
    artifact_index_path = output_dir / "artifact_index.csv"
    artifact_df.to_csv(artifact_index_path, index=False)
    return artifact_df


def run_analysis(
    input_dir: Path,
    output_dir: Path,
    workers: int = 8,
    cache: bool = True,
    cache_dir: Optional[Path] = None,
    max_nodes_for_sampled: int = 5000,
    stress_sources: int = 32,
    stress_targets: int = 128,
    crossing_samples: int = 50_000,
    write_plots: bool = True,
) -> Dict[str, Any]:
    """Run the full quality/runtime analysis pipeline.

    Parameters
    ----------
    input_dir : Path
        Benchmark artifact directory.
    output_dir : Path
        Destination directory for sidecar outputs.
    workers : int, optional
        Worker-process count.
    cache : bool, optional
        Whether metric recomputation should use the JSON cache.
    cache_dir : Optional[Path], optional
        Cache directory override.
    max_nodes_for_sampled : int, optional
        Skip sampled metrics above this graph size.
    stress_sources : int, optional
        ``sampled_stress`` source budget.
    stress_targets : int, optional
        ``sampled_stress`` target budget.
    crossing_samples : int, optional
        ``sampled_crossing_rate`` pair-sampling budget.
    write_plots : bool, optional
        Whether Pareto PNG plots should be emitted.

    Returns
    -------
    Dict[str, Any]
        Compact run summary for CLI display.
    """
    LOGGER.info("Loading benchmark records from %s", input_dir)
    records_df = load_benchmark_records(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    telemetry: Dict[str, Any] = {}
    h5_path = input_dir / "positions.h5"
    if h5_path.exists():
        try:
            scripts_dir = Path(__file__).resolve().parent
            if str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))
            from validate_benchmark_integrity import validate_sync

            telemetry["validate_sync_errors"] = validate_sync(input_dir / "results.json", h5_path)
        except Exception as exc:
            telemetry["validate_sync_exception"] = str(exc)
    telemetry_path = output_dir / "validate_sync_telemetry.json"
    with telemetry_path.open("w", encoding="utf-8") as handle:
        json.dump(telemetry, handle, indent=2, default=str)

    if records_df.empty:
        empty_summary_df = pd.DataFrame(columns=SUMMARY_COLUMNS)
        artifact_df = write_csvs(
            output_dir,
            records_df,
            empty_summary_df,
            {},
            {},
            {},
            pd.DataFrame(columns=INSIGHT_COLUMNS),
            pd.DataFrame(columns=BEST_OF_BREED_COLUMNS),
            write_plots=write_plots,
        )
        artifact_df.to_csv(output_dir / "artifact_index.csv", index=False)
        return {
            "records": 0,
            "ok_records": 0,
            "metric_successes": 0,
            "metric_failures": 0,
            "telemetry": telemetry,
        }

    for metric_name in ALL_ANALYSIS_METRICS:
        records_df[metric_name] = float("nan")
    records_df["metric_rejection_reason"] = None
    records_df["metrics_from_cache"] = False

    ok_rows = records_df[records_df["status"] == "ok"].copy()
    metrics_module_hash = compute_metrics_module_hash() if cache else "nocache"
    cache_root = cache_dir or (output_dir / "cache")

    tasks: List[MetricTask] = []
    for row in ok_rows.itertuples():
        cache_key = compute_cache_key(
            str(row.record_key),
            stress_sources,
            stress_targets,
            crossing_samples,
            metrics_module_hash,
        )
        tasks.append(
            MetricTask(
                record_key=str(row.record_key),
                graph_name=str(row.graph_name),
                engine_name=str(row.engine_name),
                layout_seed=optional_int(row.seed),
                positions_file=row.positions_file if isinstance(row.positions_file, str) else None,
                input_dir=input_dir,
                num_nodes=int(row.num_nodes or 0),
                cache_path=(cache_root / cache_key[:2] / f"{cache_key}.json") if cache else None,
                stress_sources=stress_sources,
                stress_targets=stress_targets,
                crossing_samples=crossing_samples,
                skip_sampled=bool(int(row.num_nodes or 0) > max_nodes_for_sampled),
            )
        )

    LOGGER.info("Recomputing metrics for %d ok records", len(tasks))
    results_by_key: Dict[str, Dict[str, Any]] = {}
    started_at = time.time()
    if workers <= 1:
        _worker_init(h5_path)
        iterator = map(_compute_metrics_for_task, tasks)
        for result in iterator:
            results_by_key[str(result["record_key"])] = result
    else:
        with mp.Pool(workers, initializer=_worker_init, initargs=(h5_path,)) as pool:
            for index, result in enumerate(
                pool.imap_unordered(_compute_metrics_for_task, tasks, chunksize=32),
                start=1,
            ):
                results_by_key[str(result["record_key"])] = result
                if index % 1000 == 0:
                    LOGGER.info(
                        "Metrics %d/%d (%.1f%%) elapsed=%.1fs",
                        index,
                        len(tasks),
                        100.0 * index / max(len(tasks), 1),
                        time.time() - started_at,
                    )
    LOGGER.info("Metric recomputation finished in %.1fs", time.time() - started_at)

    result_rows: List[Dict[str, Any]] = []
    for record_key, result in results_by_key.items():
        row: Dict[str, Any] = {
            "record_key": record_key,
            "metric_rejection_reason": result.get("reason"),
            "metrics_from_cache": bool(result.get("from_cache", False)),
        }
        metrics = result.get("metrics") or {}
        for metric_name in ALL_ANALYSIS_METRICS:
            row[metric_name] = metrics.get(metric_name, float("nan"))
        result_rows.append(row)

    if result_rows:
        metrics_df = pd.DataFrame(result_rows)
        metric_columns = ["metric_rejection_reason", "metrics_from_cache"]
        metric_columns.extend(list(ALL_ANALYSIS_METRICS))
        records_df = records_df.drop(columns=metric_columns)
        records_df = records_df.merge(metrics_df, on="record_key", how="left")
    else:
        for metric_name in ALL_ANALYSIS_METRICS:
            records_df[metric_name] = float("nan")
        records_df["metric_rejection_reason"] = None
        records_df["metrics_from_cache"] = False

    family_summary_df, topk_tables, pareto_tables, family_tables = compute_family_summary(
        records_df
    )
    pareto_frames = [table for table in pareto_tables.values() if not table.empty]
    pareto_df = pd.concat(pareto_frames, ignore_index=True) if pareto_frames else pd.DataFrame()
    insights_df = extract_dagua_default_insights(family_summary_df)
    best_of_breed_df = extract_best_of_breed(pareto_df)
    artifact_df = write_csvs(
        output_dir,
        records_df,
        family_summary_df,
        topk_tables,
        pareto_tables,
        family_tables,
        insights_df,
        best_of_breed_df,
        write_plots=write_plots,
    )
    artifact_df.to_csv(output_dir / "artifact_index.csv", index=False)

    metric_successes = sum(1 for result in results_by_key.values() if result.get("reason") is None)
    metric_failures = sum(
        1 for result in results_by_key.values() if result.get("reason") is not None
    )
    return {
        "records": int(len(records_df)),
        "ok_records": int(len(ok_rows)),
        "metric_successes": int(metric_successes),
        "metric_failures": int(metric_failures),
        "artifacts": int(len(artifact_df)),
        "telemetry": telemetry,
    }


def main() -> int:
    """Run the CLI entry point.

    Returns
    -------
    int
        Exit status code.
    """
    parser = argparse.ArgumentParser(description="Quality/Runtime analysis pipeline.")
    parser.add_argument("--input", type=Path, default=Path("eval_output/variant_bench_full"))
    parser.add_argument("--output", type=Path, default=Path("eval_output/quality_runtime_report"))
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 2))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--cache-invalidate", action="store_true")
    parser.add_argument("--max-nodes-for-sampled-metrics", type=int, default=5000)
    parser.add_argument("--stress-sources", type=int, default=32)
    parser.add_argument("--stress-targets", type=int, default=128)
    parser.add_argument("--crossing-samples", type=int, default=50_000)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    if args.cache_invalidate:
        cache_root = args.cache_dir or (args.output / "cache")
        if cache_root.exists():
            shutil.rmtree(cache_root)

    result = run_analysis(
        input_dir=args.input,
        output_dir=args.output,
        workers=args.workers,
        cache=not args.no_cache,
        cache_dir=args.cache_dir,
        max_nodes_for_sampled=args.max_nodes_for_sampled_metrics,
        stress_sources=args.stress_sources,
        stress_targets=args.stress_targets,
        crossing_samples=args.crossing_samples,
        write_plots=not args.no_plots,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
