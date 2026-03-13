"""Persistent competitive benchmarking pipeline for Dagua.

Standard suite:
- Thorough but runtime-bounded benchmark run intended for local comparison and
  report generation.
- Stores timestamped results, metadata, and saved position tensors.

Rare suite:
- Extreme-scale Dagua-focused runs whose results are merged into the latest
  standard view without requiring re-execution.

The benchmark database layout is:

eval_output/
├── benchmark_db/
│   ├── standard/<run_id>/
│   ├── rare/<run_id>/
│   └── combined_latest.json
├── visuals/
├── report/
└── scaling_curve.png
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from dagua.edges import place_edge_labels, route_edges
from dagua.eval.competitors import get_competitors
from dagua.eval.competitors.base import CompetitorBase
from dagua.eval.graphs import (
    TestGraph,
    get_test_graphs,
    make_chain,
    make_random_dag,
    make_sparse_layered,
    make_tree,
    make_wide_dag,
)
from dagua.metrics import compute_all_metrics, composite, full, quick
from dagua.utils import longest_path_layering


DEFAULT_OUTPUT_DIR = "eval_output"
DEFAULT_TIMEOUT = 300.0
STANDARD_SUITE = "standard"
RARE_SUITE = "rare"
DEFAULT_COMPETITOR_ORDER = [
    "dagua",
    "graphviz_dot",
    "graphviz_sfdp",
    "elk_layered",
    "dagre",
    "nx_spring",
]
VISUAL_MAX_NODES = 2_000
CRITIC_MAX_NODES = 500


@dataclass
class BenchmarkResult:
    """Compatibility record for one graph/competitor outcome."""

    graph_name: str
    graph_nodes: int
    graph_edges: int
    competitor: str
    status: str
    runtime_seconds: Optional[float]
    metrics: Dict[str, Any]
    composite_score: Optional[float]
    reason: Optional[str] = None
    error: Optional[str] = None
    positions_path: Optional[str] = None


@dataclass
class BenchmarkGraph:
    """Graph plus benchmark metadata."""

    test_graph: TestGraph
    structural_category: str
    suite: str
    visualize: bool = False
    scale_tier: Optional[str] = None


def _clone_test_graph(tg: TestGraph) -> TestGraph:
    from dagua.graph import DaguaGraph

    return TestGraph(
        name=tg.name,
        graph=DaguaGraph.from_json(tg.graph.to_json()),
        tags=set(tg.tags),
        description=tg.description,
        source=tg.source,
        expected_challenges=tg.expected_challenges,
    )


def _named_graphs(max_nodes: Optional[int] = None) -> Dict[str, TestGraph]:
    return {tg.name: tg for tg in get_test_graphs(max_nodes=max_nodes)}


def get_standard_suite_graphs() -> List[BenchmarkGraph]:
    """Representative benchmark roster designed to stay within a local budget."""
    named = _named_graphs(max_nodes=2_500)
    selected: List[BenchmarkGraph] = []

    def add_named(name: str, category: str, visualize: bool = True) -> None:
        if name not in named:
            return
        selected.append(
            BenchmarkGraph(
                test_graph=_clone_test_graph(named[name]),
                structural_category=category,
                suite=STANDARD_SUITE,
                visualize=visualize,
            )
        )

    # Small structural variety
    chain_100 = make_chain(100, seed=42)
    chain_100.name = "chain_100"
    chain_100.description = "Linear chain benchmark at 100 nodes"
    selected.append(BenchmarkGraph(chain_100, "linear", STANDARD_SUITE, True, "small"))

    tree_127 = make_tree(127, branching=2, seed=42)
    tree_127.name = "binary_tree_127"
    tree_127.description = "Balanced binary tree benchmark at 127 nodes"
    selected.append(BenchmarkGraph(tree_127, "tree", STANDARD_SUITE, True, "small"))

    wide_200 = make_wide_dag(200, seed=42)
    wide_200.name = "wide_parallel_200"
    wide_200.description = "Wide layered DAG with strong fan-out/fan-in pressure"
    selected.append(BenchmarkGraph(wide_200, "wide_parallel", STANDARD_SUITE, True, "small"))

    dense_200 = make_random_dag(200, density=6.0, seed=42)
    dense_200.name = "dense_skip_200"
    dense_200.description = "Dense random DAG stress test with many skip-like chords"
    selected.append(BenchmarkGraph(dense_200, "dense_skip", STANDARD_SUITE, True, "small"))

    sparse_500 = make_random_dag(500, density=3.0, seed=42)
    sparse_500.name = "random_sparse_500"
    sparse_500.description = "Sparse random DAG benchmark near universal-competitor range"
    selected.append(BenchmarkGraph(sparse_500, "random_sparse", STANDARD_SUITE, True, "small"))

    dense_300 = make_random_dag(300, density=8.0, seed=42)
    dense_300.name = "random_dense_300"
    dense_300.description = "Dense random DAG benchmark for crossing-heavy comparisons"
    selected.append(BenchmarkGraph(dense_300, "random_dense", STANDARD_SUITE, True, "small"))

    # Real/synthetic architecture cases from the existing corpus
    add_named("residual_block", "residual")
    add_named("nested_shallow_enc_dec", "clustered")
    add_named("kitchen_sink_hybrid_net", "kitchen_sink")
    add_named("tl_cnn_small", "cnn")
    add_named("tl_resnet_2block", "resnet")
    add_named("tl_transformer_1layer", "transformer")

    # Scale ladder
    scale_specs = [
        (2_000, "scale_2k", "small"),
        (5_000, "scale_5k", "medium"),
        (20_000, "scale_20k", "medium"),
        (50_000, "scale_50k", "medium"),
        (100_000, "scale_100k", "large"),
    ]
    for n, name, tier in scale_specs:
        tg = make_sparse_layered(n, seed=42)
        tg.name = name
        tg.description = f"Sparse layered scale benchmark at {n:,} nodes"
        selected.append(BenchmarkGraph(tg, "scale", STANDARD_SUITE, False, tier))

    return selected


def get_rare_suite_graphs() -> List[BenchmarkGraph]:
    """Extreme-scale ladder. Dagua-first by design."""
    from dagua.graph import DaguaGraph

    selected: List[BenchmarkGraph] = []
    for n in [
        500_000,
        1_000_000,
        2_000_000,
        5_000_000,
        10_000_000,
        20_000_000,
        50_000_000,
        100_000_000,
        250_000_000,
        500_000_000,
        1_000_000_000,
    ]:
        name = f"scale_{n:,}".replace(",", "")
        tg = TestGraph(
            name=name,
            graph=DaguaGraph(),
            tags={"large-sparse"},
            description=f"Rare sparse layered scale benchmark at {n:,} nodes",
            source="synthetic",
            expected_challenges="Extreme scale",
        )
        selected.append(BenchmarkGraph(tg, "scale", RARE_SUITE, False, "rare"))
    return selected


def _tool_version(cmd: Sequence[str], stderr: bool = False) -> Optional[str]:
    try:
        result = subprocess.run(
            list(cmd),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    output = result.stderr if stderr else result.stdout
    text = (output or "").strip()
    return text.splitlines()[0] if text else None


def _node_package_version(package_name: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["node", "-e", f"console.log(require('{package_name}/package.json').version)"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    text = (result.stdout or "").strip()
    return text or None


def _system_metadata() -> Dict[str, Any]:
    try:
        import psutil

        ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        ram_gb = None

    try:
        import dagua

        dagua_version = dagua.__version__
    except Exception:
        dagua_version = None

    gpu_name = None
    cuda_version = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda

    git_hash = _tool_version(["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"])

    return {
        "cpu": platform.processor() or platform.machine(),
        "gpu": gpu_name,
        "ram_gb": ram_gb,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": cuda_version,
        "graphviz": _tool_version(["dot", "-V"], stderr=True),
        "elk": _node_package_version("elkjs"),
        "dagre": _node_package_version("dagre"),
        "networkx": _safe_import_version("networkx"),
        "dagua": dagua_version,
        "dagua_git_hash": git_hash,
        "platform": platform.platform(),
    }


def _safe_import_version(module_name: str) -> Optional[str]:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", None)


def _benchmark_db_root(output_dir: str) -> Path:
    return Path(output_dir) / "benchmark_db"


def _run_dir(output_dir: str, suite: str, run_id: str) -> Path:
    return _benchmark_db_root(output_dir) / suite / run_id


def _partial_results_path(run_dir: Path) -> Path:
    return run_dir / "results.partial.json"


def _progress_path(run_dir: Path) -> Path:
    return run_dir / "progress.json"


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False, default=str)


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _update_latest_symlink(parent: Path, run_id: str) -> None:
    latest = parent / "latest"
    target = parent / run_id
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(target.name)


def _graph_signature(graph) -> str:
    payload = json.dumps(graph.to_json(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _graph_signature_map(graphs: Sequence[BenchmarkGraph]) -> Dict[str, str]:
    return {bg.test_graph.name: _graph_signature(bg.test_graph.graph) for bg in graphs}


def _competitor_signature(name: str, system: Dict[str, Any]) -> str:
    version_keys = {
        "dagua": "dagua_git_hash",
        "graphviz_dot": "graphviz",
        "graphviz_sfdp": "graphviz",
        "elk_layered": "elk",
        "dagre": "dagre",
        "nx_spring": "networkx",
    }
    key = version_keys.get(name)
    value = system.get(key) if key is not None else None
    return f"{name}:{value}"


def _competitor_signature_map(
    competitors: Sequence[CompetitorBase],
    system: Dict[str, Any],
) -> Dict[str, str]:
    return {competitor.name: _competitor_signature(competitor.name, system) for competitor in competitors}


def _load_latest_payload_and_metadata(output_dir: str, suite: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Path]]:
    latest_dir = _benchmark_db_root(output_dir) / suite / "latest"
    results_path = latest_dir / "results.json"
    metadata_path = latest_dir / "metadata.json"
    if not results_path.exists():
        return None, None, None
    payload = _load_json(results_path)
    metadata = _load_json(metadata_path) if metadata_path.exists() else {}
    return payload, metadata, results_path.resolve().parent


def _load_resumable_payload_and_metadata(
    output_dir: str,
    suite: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Path], Optional[str]]:
    suite_root = _benchmark_db_root(output_dir) / suite
    if not suite_root.exists():
        return None, None, None, None

    run_dirs = [path for path in suite_root.iterdir() if path.is_dir() and path.name != "latest"]
    run_dirs.sort(key=lambda path: path.name, reverse=True)
    for run_dir in run_dirs:
        partial_path = _partial_results_path(run_dir)
        metadata_path = run_dir / "metadata.json"
        if partial_path.exists():
            payload = _load_json(partial_path)
            metadata = _load_json(metadata_path) if metadata_path.exists() else {}
            return payload, metadata, run_dir, run_dir.name
    return None, None, None, None


def _copy_cached_positions(
    latest_run_dir: Path,
    cached_result: Dict[str, Any],
    run_dir: Path,
) -> Optional[str]:
    rel_path = cached_result.get("positions_path")
    if not rel_path:
        return None
    src = latest_run_dir / rel_path
    if not src.exists():
        return None
    dst = run_dir / rel_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return rel_path


def _reuse_cached_result(
    graph_name: str,
    competitor_name: str,
    run_dir: Path,
    cached_payload: Optional[Dict[str, Any]],
    cached_metadata: Optional[Dict[str, Any]],
    latest_run_dir: Optional[Path],
    graph_signatures: Dict[str, str],
    competitor_signatures: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    if cached_payload is None or latest_run_dir is None:
        return None
    cached_graphs = cached_payload.get("graphs", {})
    cached_graph = cached_graphs.get(graph_name)
    if not cached_graph:
        return None
    cached_result = cached_graph.get("competitors", {}).get(competitor_name)
    if not cached_result:
        return None
    meta_graph_sigs = (cached_metadata or {}).get("graph_signatures", {})
    meta_comp_sigs = (cached_metadata or {}).get("competitor_signatures", {})
    if meta_graph_sigs.get(graph_name) != graph_signatures.get(graph_name):
        return None
    if meta_comp_sigs.get(competitor_name) != competitor_signatures.get(competitor_name):
        return None
    reused = copy.deepcopy(cached_result)
    reused_path = _copy_cached_positions(latest_run_dir, cached_result, run_dir)
    reused["positions_path"] = reused_path
    reused["reused_from"] = str(cached_payload.get("run_id"))
    return reused


def _metric_payload(
    graph,
    pos: torch.Tensor,
    compute_level: str,
) -> Tuple[Dict[str, Any], float, List[str], List[str]]:
    edge_index = graph.edge_index
    topo_depth = longest_path_layering(edge_index, graph.num_nodes) if edge_index.numel() > 0 else None
    node_sizes = graph.node_sizes if graph.node_sizes is not None else None

    metrics: Dict[str, Any]
    computed = ["tier1"]
    skipped: List[str] = []

    if compute_level == "full":
        curves = route_edges(pos, edge_index, graph.node_sizes, graph.direction, graph)
        label_positions = place_edge_labels(curves, pos, graph.node_sizes, graph.edge_labels, graph)
        metrics = full(
            pos,
            edge_index,
            topo_depth=topo_depth,
            node_sizes=node_sizes,
            direction=graph.direction,
            curves=curves,
            label_positions=label_positions,
            edge_labels=graph.edge_labels,
            stress_sources=100,
            stress_targets=250,
            crossing_samples=100_000,
            neighborhood_samples=1_000,
        )
        metrics.update(
            compute_all_metrics(
                pos,
                edge_index,
                graph.node_sizes,
                clusters=graph.clusters,
                direction=graph.direction,
            )
        )
        computed.extend(["tier2", "tier3"])
    else:
        metrics = quick(
            pos,
            edge_index,
            topo_depth=topo_depth,
            node_sizes=node_sizes,
            direction=graph.direction,
        )
        metrics.update(
            compute_all_metrics(
                pos,
                edge_index,
                graph.node_sizes,
                clusters=graph.clusters,
                direction=graph.direction,
            )
        )
        skipped.extend(["tier2", "tier3"])

    metrics["aesthetic_score"] = metrics.get("overall_quality", composite(metrics))
    metrics["anti_patterns"] = _style_anti_patterns(metrics)
    comp_score = float(metrics.get("composite_score", composite(metrics)))
    return metrics, comp_score, computed, skipped


def _style_anti_patterns(metrics: Dict[str, Any]) -> List[str]:
    flags: List[str] = []
    if metrics.get("node_overlaps", metrics.get("overlap_count", 0)) > 0:
        flags.append("node_overlaps")
    if metrics.get("edge_crossings", 0) > 10:
        flags.append("high_crossings")
    if metrics.get("edge_straightness", metrics.get("edge_straightness_mean_deg", 0.0)) > 25.0:
        flags.append("excessive_edge_deviation")
    if metrics.get("edge_length_cv", 0.0) > 1.0:
        flags.append("inconsistent_edge_lengths")
    if metrics.get("aspect_ratio", 1.5) > 4.0:
        flags.append("extreme_aspect_ratio")
    if metrics.get("label_overlaps", 0) + metrics.get("label_node_overlaps", 0) > 0:
        flags.append("label_collisions")
    return flags


def _compute_level_for_graph(n_nodes: int) -> str:
    return "full" if n_nodes <= 2_000 else "quick"


def _normalize_error(err: Exception) -> str:
    text = str(err).strip()
    return text[:1000] if text else err.__class__.__name__


def _positions_dir(run_dir: Path) -> Path:
    d = run_dir / "positions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _competitor_map(names: Optional[Sequence[str]] = None) -> List[CompetitorBase]:
    competitors = {c.name: c for c in get_competitors()}
    order = list(names) if names is not None else DEFAULT_COMPETITOR_ORDER
    return [competitors[name] for name in order if name in competitors]


def _suite_graphs(suite: str) -> List[BenchmarkGraph]:
    if suite == STANDARD_SUITE:
        return get_standard_suite_graphs()
    if suite == RARE_SUITE:
        selected: List[BenchmarkGraph] = []
        for bg in get_rare_suite_graphs():
            n = int(bg.test_graph.name.split("_", 1)[1])
            tg = make_sparse_layered(n, seed=42)
            tg.name = bg.test_graph.name
            tg.description = bg.test_graph.description
            selected.append(BenchmarkGraph(tg, bg.structural_category, bg.suite, bg.visualize, bg.scale_tier))
        return selected
    raise ValueError(f"Unknown suite {suite!r}")


def _graph_summary(bg: BenchmarkGraph) -> Dict[str, Any]:
    tg = bg.test_graph
    return {
        "n_nodes": tg.graph.num_nodes,
        "n_edges": int(tg.graph.edge_index.shape[1]) if tg.graph.edge_index.numel() > 0 else 0,
        "structural_category": bg.structural_category,
        "description": tg.description,
        "expected_challenges": tg.expected_challenges,
        "tags": sorted(tg.tags),
        "source": tg.source,
        "visualize": bg.visualize,
        "scale_tier": bg.scale_tier,
        "competitors": {},
    }


def _write_progress(
    run_dir: Path,
    suite: str,
    run_id: str,
    graphs: Sequence[BenchmarkGraph],
    competitors: Sequence[CompetitorBase],
    payload: Dict[str, Any],
    *,
    current_graph: Optional[str] = None,
    current_competitor: Optional[str] = None,
    step: str = "running",
    last_artifact: Optional[str] = None,
) -> None:
    total_graphs = len(graphs)
    total_competitors = len(competitors)
    total_pairs = total_graphs * total_competitors
    completed_graphs = 0
    completed_pairs = 0
    graph_status: Dict[str, Dict[str, Any]] = {}

    for bg in graphs:
        graph_payload = payload.get("graphs", {}).get(bg.test_graph.name, {})
        comp_payload = graph_payload.get("competitors", {})
        statuses = {}
        for competitor in competitors:
            status = comp_payload.get(competitor.name, {}).get("status", "PENDING")
            statuses[competitor.name] = status
            if status in {"OK", "FAILED", "SKIPPED"}:
                completed_pairs += 1
        if statuses and all(status in {"OK", "FAILED", "SKIPPED"} for status in statuses.values()):
            completed_graphs += 1
            graph_step = "complete"
        elif bg.test_graph.name == current_graph and current_competitor:
            graph_step = f"running:{current_competitor}"
        else:
            graph_step = "pending"
        graph_status[bg.test_graph.name] = {
            "status": graph_step,
            "competitors": statuses,
        }

    progress = {
        "suite": suite,
        "run_id": run_id,
        "step": step,
        "current_graph": current_graph,
        "current_competitor": current_competitor,
        "completed_graphs": completed_graphs,
        "total_graphs": total_graphs,
        "completed_pairs": completed_pairs,
        "total_pairs": total_pairs,
        "last_artifact": last_artifact,
        "graphs": graph_status,
    }
    _save_json(_progress_path(run_dir), progress)


def _run_one_competitor(
    bg: BenchmarkGraph,
    competitor: CompetitorBase,
    timeout: float,
    run_dir: Path,
) -> Dict[str, Any]:
    graph = bg.test_graph.graph
    graph.compute_node_sizes()
    n_nodes = graph.num_nodes

    if not competitor.available():
        return {
            "status": "SKIPPED",
            "reason": "not installed",
            "runtime_seconds": None,
            "metrics": {},
            "composite_score": None,
            "metrics_computed": [],
            "metrics_skipped": ["tier1", "tier2", "tier3"],
            "positions_path": None,
        }

    if n_nodes > competitor.max_nodes:
        return {
            "status": "SKIPPED",
            "reason": "exceeds known limit",
            "runtime_seconds": None,
            "metrics": {},
            "composite_score": None,
            "metrics_computed": [],
            "metrics_skipped": ["tier1", "tier2", "tier3"],
            "positions_path": None,
        }

    try:
        result = competitor.layout(graph, timeout=timeout)
    except Exception as exc:
        return {
            "status": "FAILED",
            "reason": "exception",
            "error": _normalize_error(exc),
            "runtime_seconds": None,
            "metrics": {},
            "composite_score": None,
            "metrics_computed": [],
            "metrics_skipped": ["tier1", "tier2", "tier3"],
            "positions_path": None,
        }

    if result.pos is None:
        return {
            "status": "FAILED",
            "reason": result.error or "layout failed",
            "error": result.error,
            "runtime_seconds": result.runtime_seconds,
            "metrics": {},
            "composite_score": None,
            "metrics_computed": [],
            "metrics_skipped": ["tier1", "tier2", "tier3"],
            "positions_path": None,
        }

    compute_level = _compute_level_for_graph(n_nodes)
    metrics, comp_score, computed, skipped = _metric_payload(graph, result.pos, compute_level)
    rel_positions = Path("positions") / f"{bg.test_graph.name}__{competitor.name}.pt"
    torch.save(result.pos.detach().cpu(), run_dir / rel_positions)

    return {
        "status": "OK",
        "runtime_seconds": result.runtime_seconds,
        "metrics": metrics,
        "composite_score": comp_score,
        "metrics_computed": computed,
        "metrics_skipped": skipped,
        "positions_path": str(rel_positions),
    }


def _build_results_payload(
    suite: str,
    run_id: str,
    graphs: Sequence[BenchmarkGraph],
    competitors: Sequence[CompetitorBase],
    timeout: float,
    output_dir: str,
    cached_payload: Optional[Dict[str, Any]] = None,
    cached_metadata: Optional[Dict[str, Any]] = None,
    latest_run_dir: Optional[Path] = None,
    graph_signatures: Optional[Dict[str, str]] = None,
    competitor_signatures: Optional[Dict[str, str]] = None,
    rerun_competitors: Optional[Sequence[str]] = None,
    existing_payload: Optional[Dict[str, Any]] = None,
    checkpoint_each_graph: bool = False,
) -> Dict[str, Any]:
    run_dir = _run_dir(output_dir, suite, run_id)
    _positions_dir(run_dir)
    graph_signatures = graph_signatures or {}
    competitor_signatures = competitor_signatures or {}
    rerun_set = set(rerun_competitors or [])

    payload = copy.deepcopy(existing_payload) if existing_payload is not None else {
        "run_id": run_id,
        "suite": suite,
        "system": _system_metadata(),
        "graphs": {},
    }
    payload["run_id"] = run_id
    payload["suite"] = suite
    payload["system"] = _system_metadata()
    _write_progress(
        run_dir,
        suite,
        run_id,
        graphs,
        competitors,
        payload,
        step="starting",
    )

    for bg in graphs:
        existing_graph = payload["graphs"].get(bg.test_graph.name, {})
        graph_payload = copy.deepcopy(existing_graph) if existing_graph else _graph_summary(bg)
        graph_payload.update({k: v for k, v in _graph_summary(bg).items() if k != "competitors"})
        graph_payload.setdefault("competitors", {})
        for competitor in competitors:
            existing_result = graph_payload["competitors"].get(competitor.name)
            if existing_result and existing_result.get("status") in {"OK", "FAILED", "SKIPPED"}:
                continue
            _write_progress(
                run_dir,
                suite,
                run_id,
                graphs,
                competitors,
                payload,
                current_graph=bg.test_graph.name,
                current_competitor=competitor.name,
                step="running",
                last_artifact=str(_partial_results_path(run_dir)) if checkpoint_each_graph else None,
            )
            reused = None
            if competitor.name not in rerun_set:
                reused = _reuse_cached_result(
                    graph_name=bg.test_graph.name,
                    competitor_name=competitor.name,
                    run_dir=run_dir,
                    cached_payload=cached_payload,
                    cached_metadata=cached_metadata,
                    latest_run_dir=latest_run_dir,
                    graph_signatures=graph_signatures,
                    competitor_signatures=competitor_signatures,
                )
            graph_payload["competitors"][competitor.name] = reused or _run_one_competitor(
                bg,
                competitor,
                timeout=timeout,
                run_dir=run_dir,
            )
            _write_progress(
                run_dir,
                suite,
                run_id,
                graphs,
                competitors,
                payload | {"graphs": payload.get("graphs", {}) | {bg.test_graph.name: graph_payload}},
                current_graph=bg.test_graph.name,
                current_competitor=competitor.name,
                step="running",
                last_artifact=graph_payload["competitors"][competitor.name].get("positions_path"),
            )
        payload["graphs"][bg.test_graph.name] = graph_payload
        if checkpoint_each_graph:
            _save_json(_partial_results_path(run_dir), payload)
            _write_progress(
                run_dir,
                suite,
                run_id,
                graphs,
                competitors,
                payload,
                current_graph=bg.test_graph.name,
                step="checkpointed",
                last_artifact=str(_partial_results_path(run_dir)),
            )

    _write_progress(
        run_dir,
        suite,
        run_id,
        graphs,
        competitors,
        payload,
        step="complete",
        last_artifact=str(run_dir / "results.json"),
    )

    return payload


def _flatten_results(payload: Dict[str, Any]) -> List[BenchmarkResult]:
    flat: List[BenchmarkResult] = []
    for graph_name, graph_payload in payload.get("graphs", {}).items():
        n_nodes = graph_payload.get("n_nodes", 0)
        n_edges = graph_payload.get("n_edges", 0)
        for competitor, result in graph_payload.get("competitors", {}).items():
            flat.append(
                BenchmarkResult(
                    graph_name=graph_name,
                    graph_nodes=n_nodes,
                    graph_edges=n_edges,
                    competitor=competitor,
                    status=result.get("status", "UNKNOWN"),
                    runtime_seconds=result.get("runtime_seconds"),
                    metrics=result.get("metrics", {}),
                    composite_score=result.get("composite_score"),
                    reason=result.get("reason"),
                    error=result.get("error"),
                    positions_path=result.get("positions_path"),
                )
            )
    return flat


def merge_latest_results(output_dir: str = DEFAULT_OUTPUT_DIR) -> Dict[str, Any]:
    """Merge latest standard + latest rare runs into a combined view."""
    root = _benchmark_db_root(output_dir)
    standard_latest = root / STANDARD_SUITE / "latest" / "results.json"
    rare_latest = root / RARE_SUITE / "latest" / "results.json"

    combined: Dict[str, Any] = {
        "generated_from": {},
        "graphs": {},
    }
    system: Dict[str, Any] = {}

    if standard_latest.exists():
        standard = _load_json(standard_latest)
        combined["generated_from"][STANDARD_SUITE] = standard.get("run_id")
        combined["graphs"].update(standard.get("graphs", {}))
        system.update(standard.get("system", {}))

    if rare_latest.exists():
        rare = _load_json(rare_latest)
        combined["generated_from"][RARE_SUITE] = rare.get("run_id")
        for name, graph_payload in rare.get("graphs", {}).items():
            if name not in combined["graphs"]:
                combined["graphs"][name] = graph_payload
        system.update({k: v for k, v in rare.get("system", {}).items() if v is not None})

    combined["system"] = system
    combined_path = root / "combined_latest.json"
    _save_json(combined_path, combined)
    return combined


def benchmark_run_status(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    suite: str = STANDARD_SUITE,
) -> Dict[str, Any]:
    """Return progress information for the latest complete or partial run."""
    partial_payload, partial_metadata, partial_run_dir, partial_run_id = _load_resumable_payload_and_metadata(
        output_dir, suite
    )
    latest_payload, latest_metadata, latest_run_dir = _load_latest_payload_and_metadata(output_dir, suite)

    payload = partial_payload or latest_payload or {"graphs": {}}
    metadata = partial_metadata or latest_metadata or {}
    run_dir = partial_run_dir or latest_run_dir
    run_id = partial_run_id or (payload.get("run_id") if payload else None)
    progress = _load_json(_progress_path(run_dir)) if run_dir is not None and _progress_path(run_dir).exists() else {}

    total_graphs = len(metadata.get("graphs", [])) or len(payload.get("graphs", {}))
    completed_graphs = 0
    graph_status: Dict[str, Any] = {}
    for graph_name, graph_payload in payload.get("graphs", {}).items():
        competitors = graph_payload.get("competitors", {})
        statuses = [result.get("status", "UNKNOWN") for result in competitors.values()]
        if statuses and all(status in {"OK", "FAILED", "SKIPPED"} for status in statuses):
            completed_graphs += 1
            graph_status[graph_name] = {"status": "complete", "competitors": {name: res.get("status", "UNKNOWN") for name, res in competitors.items()}}
        else:
            graph_status[graph_name] = {"status": "incomplete", "competitors": {name: res.get("status", "UNKNOWN") for name, res in competitors.items()}}

    return {
        "suite": suite,
        "run_id": run_id,
        "run_dir": str(run_dir) if run_dir is not None else None,
        "is_partial": partial_payload is not None,
        "completed_graphs": completed_graphs,
        "total_graphs": total_graphs,
        "remaining_graphs": max(total_graphs - completed_graphs, 0),
        "completed_pairs": progress.get("completed_pairs"),
        "total_pairs": progress.get("total_pairs"),
        "step": progress.get("step"),
        "current_graph": progress.get("current_graph"),
        "current_competitor": progress.get("current_competitor"),
        "last_artifact": progress.get("last_artifact"),
        "graphs": graph_status,
        "progress": progress or None,
    }


def run_suite(
    suite: str = STANDARD_SUITE,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    competitors: Optional[Sequence[str]] = None,
    timeout: float = DEFAULT_TIMEOUT,
    generate_report_artifacts: bool = True,
    reuse_cached: bool = True,
    rerun_competitors: Optional[Sequence[str]] = None,
    resume_incomplete: bool = True,
    checkpoint_each_graph: bool = False,
) -> Dict[str, Any]:
    graphs = _suite_graphs(suite)
    competitor_list = _competitor_map(competitors)
    system = _system_metadata()
    graph_signatures = _graph_signature_map(graphs)
    competitor_signatures = _competitor_signature_map(competitor_list, system)
    rerun_list = list(rerun_competitors) if rerun_competitors is not None else (["dagua"] if reuse_cached else [])
    cached_payload = cached_metadata = latest_run_dir = None
    if reuse_cached:
        cached_payload, cached_metadata, latest_run_dir = _load_latest_payload_and_metadata(output_dir, suite)

    existing_payload = existing_metadata = None
    existing_run_dir = None
    run_id = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    if resume_incomplete:
        existing_payload, existing_metadata, existing_run_dir, existing_run_id = _load_resumable_payload_and_metadata(output_dir, suite)
        if existing_run_dir is not None and existing_run_id is not None:
            run_id = existing_run_id
    run_dir = _run_dir(output_dir, suite, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    _update_latest_symlink(run_dir.parent, run_id)
    _save_json(
        run_dir / "metadata.json",
        {
            "run_id": run_id,
            "suite": suite,
            "system": system,
            "graphs": [bg.test_graph.name for bg in graphs],
            "competitors": [c.name for c in competitor_list],
            "graph_signatures": graph_signatures,
            "competitor_signatures": competitor_signatures,
            "reuse_cached": reuse_cached,
            "rerun_competitors": rerun_list,
            "resume_incomplete": resume_incomplete,
            "checkpoint_each_graph": checkpoint_each_graph,
            "resumed_from_partial": existing_run_dir is not None,
        },
    )
    _write_progress(
        run_dir,
        suite,
        run_id,
        graphs,
        competitor_list,
        existing_payload or {"graphs": {}},
        step="initialized",
        last_artifact=str(run_dir / "metadata.json"),
    )

    payload = _build_results_payload(
        suite=suite,
        run_id=run_id,
        graphs=graphs,
        competitors=competitor_list,
        timeout=timeout,
        output_dir=output_dir,
        cached_payload=cached_payload,
        cached_metadata=cached_metadata,
        latest_run_dir=latest_run_dir,
        graph_signatures=graph_signatures,
        competitor_signatures=competitor_signatures,
        rerun_competitors=rerun_list,
        existing_payload=existing_payload,
        checkpoint_each_graph=checkpoint_each_graph,
    )
    _save_json(run_dir / "results.json", payload)
    _write_progress(
        run_dir,
        suite,
        run_id,
        graphs,
        competitor_list,
        payload,
        step="results_saved",
        last_artifact=str(run_dir / "results.json"),
    )
    partial_path = _partial_results_path(run_dir)
    if partial_path.exists():
        partial_path.unlink()
    combined = merge_latest_results(output_dir=output_dir)

    if suite == STANDARD_SUITE and generate_report_artifacts:
        from dagua.eval.report import generate_report

        generate_report(output_dir=output_dir, combined_results=combined, compile_pdf=True)

    return payload


def run_standard_suite(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    competitors: Optional[Sequence[str]] = None,
    timeout: float = DEFAULT_TIMEOUT,
    reuse_cached: bool = True,
    rerun_competitors: Optional[Sequence[str]] = None,
    resume_incomplete: bool = True,
    checkpoint_each_graph: bool = True,
) -> Dict[str, Any]:
    return run_suite(
        suite=STANDARD_SUITE,
        output_dir=output_dir,
        competitors=competitors,
        timeout=timeout,
        generate_report_artifacts=True,
        reuse_cached=reuse_cached,
        rerun_competitors=rerun_competitors,
        resume_incomplete=resume_incomplete,
        checkpoint_each_graph=checkpoint_each_graph,
    )


def run_rare_suite(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    competitors: Optional[Sequence[str]] = None,
    timeout: float = DEFAULT_TIMEOUT,
    reuse_cached: bool = True,
    rerun_competitors: Optional[Sequence[str]] = None,
    resume_incomplete: bool = True,
) -> Dict[str, Any]:
    if competitors is None:
        competitors = ["dagua", "graphviz_sfdp"]
    return run_suite(
        suite=RARE_SUITE,
        output_dir=output_dir,
        competitors=competitors,
        timeout=timeout,
        generate_report_artifacts=False,
        reuse_cached=reuse_cached,
        rerun_competitors=rerun_competitors,
        resume_incomplete=resume_incomplete,
        checkpoint_each_graph=True,
    )


def load_combined_results(output_dir: str = DEFAULT_OUTPUT_DIR) -> Dict[str, Any]:
    path = _benchmark_db_root(output_dir) / "combined_latest.json"
    return _load_json(path)


def run_benchmark(
    tier: str = "standard",
    competitors: Optional[List[str]] = None,
    timeout: float = DEFAULT_TIMEOUT,
    metrics_level: str = "auto",
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> List[BenchmarkResult]:
    """Compatibility wrapper returning flattened results."""
    del metrics_level  # tiered metrics are chosen automatically by graph size
    if tier in {"standard", "small", "medium", "large", "full"}:
        payload = run_standard_suite(output_dir=output_dir, competitors=competitors, timeout=timeout)
    elif tier in {"rare", "huge"}:
        payload = run_rare_suite(output_dir=output_dir, competitors=competitors, timeout=timeout)
    else:
        raise ValueError(f"Unknown benchmark tier {tier!r}")
    return _flatten_results(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Dagua benchmark suites")
    parser.add_argument("--suite", choices=[STANDARD_SUITE, RARE_SUITE], default=STANDARD_SUITE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--competitors", default=None, help="Comma-separated competitor list")
    parser.add_argument("--rerun-competitors", default=None, help="Comma-separated competitors to force rerun")
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Only merge latest standard and rare results into combined_latest.json",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Print status for the latest complete/partial run of the selected suite",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume an incomplete run; always start a fresh timestamped run",
    )
    parser.add_argument(
        "--no-reuse-cached",
        action="store_true",
        help="Do not reuse cached competitor results from the latest run",
    )
    args = parser.parse_args()

    if args.merge_only:
        merge_latest_results(output_dir=args.output_dir)
        return
    if args.status_only:
        print(json.dumps(benchmark_run_status(output_dir=args.output_dir, suite=args.suite), indent=2))
        return

    competitors = args.competitors.split(",") if args.competitors else None
    rerun_competitors = args.rerun_competitors.split(",") if args.rerun_competitors else None
    if args.suite == STANDARD_SUITE:
        run_standard_suite(
            output_dir=args.output_dir,
            competitors=competitors,
            timeout=args.timeout,
            reuse_cached=not args.no_reuse_cached,
            rerun_competitors=rerun_competitors,
            resume_incomplete=not args.no_resume,
        )
    else:
        run_rare_suite(
            output_dir=args.output_dir,
            competitors=competitors,
            timeout=args.timeout,
            reuse_cached=not args.no_reuse_cached,
            rerun_competitors=rerun_competitors,
            resume_incomplete=not args.no_resume,
        )


if __name__ == "__main__":
    main()
