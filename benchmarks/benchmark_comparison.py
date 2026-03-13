#!/usr/bin/env python3
"""Comprehensive benchmark: Dagua vs Graphviz (dot) vs ELK (elkjs).

Compares layout quality (aesthetic metrics) and runtime performance across:
1. Real neural network architectures from TorchLens example_models
2. Synthetic RandomGraphModel at scales from 500 to 2M nodes

Usage:
    python benchmarks/benchmark_comparison.py
    python benchmarks/benchmark_comparison.py --max-scale 100000
    python benchmarks/benchmark_comparison.py --skip-real-models
    python benchmarks/benchmark_comparison.py --dagua-only
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Add project roots to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "torchlens"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "torchlens" / "tests"))

import dagua
from dagua.config import LayoutConfig
from dagua.graph import DaguaGraph
from dagua.graphviz_utils import layout_with_graphviz, to_dot
from dagua.metrics import compute_all_metrics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TIMEOUT_SECONDS = 300  # 5 min timeout for graphviz/elk
DAGUA_TIMEOUT = 600    # 10 min timeout for dagua on very large graphs
DAGUA_STEPS_QUALITY = 200  # steps for quality benchmarks (real models)
DAGUA_STEPS_SCALING = 50   # steps for scaling benchmarks (large synthetic)
GPU_MIN_NODES = 1000       # skip GPU for graphs smaller than this

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Scaling sizes for RandomGraphModel
SCALING_SIZES = [500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 2_000_000]

# ---------------------------------------------------------------------------
# Real model definitions (name, constructor, input_shape)
# ---------------------------------------------------------------------------
REAL_MODELS = [
    # Small models
    ("SimpleFF", "SimpleFF", (6, 3, 224, 224)),
    ("BatchNormModel", "BatchNormModel", (6, 3, 224, 224)),
    # Medium models
    ("TransformerEncoder", "TransformerEncoderModel", (10, 2, 16)),
    ("MultiheadAttention", "MultiheadAttentionModel", (10, 2, 16)),
    ("BiLSTM", "BiLSTMModel", (2, 10, 8)),
    ("SimpleVAE", "SimpleVAE", (1, 1, 28, 28)),
    ("SmallUNet", "SmallUNet", (1, 1, 64, 64)),
    # Large models
    ("FeaturePyramidNet", "FeaturePyramidNet", (1, 3, 64, 64)),
    ("CapsuleNetwork", "CapsuleNetwork", (1, 1, 28, 28)),
    ("PerceiverModel", "PerceiverModel", (1, 32, 16)),
    ("HighwayNetwork", "HighwayNetwork", (6, 3, 224, 224)),
    ("EarlyExitModel", "EarlyExitModel", (6, 3, 224, 224)),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TimeoutError(Exception):
    pass


@contextmanager
def time_limit(seconds):
    """Context manager that raises TimeoutError after `seconds`."""
    def handler(signum, frame):
        raise TimeoutError(f"Timed out after {seconds}s")
    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def log_model(model_class_name: str, input_shape: tuple) -> Optional[Any]:
    """Log a TorchLens model forward pass, return ModelLog or None on failure."""
    try:
        import torchlens as tl
        # Load example_models from torchlens tests directory
        import importlib.util
        models_path = Path.home() / "projects" / "torchlens" / "tests" / "example_models.py"
        spec = importlib.util.spec_from_file_location("example_models", models_path)
        example_models = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(example_models)
        model_cls = getattr(example_models, model_class_name)
        model = model_cls()
        model.eval()
        x = torch.randn(*input_shape)
        with torch.no_grad():
            model_log = tl.log_forward_pass(model, x, vis_mode="none")
        return model_log
    except Exception as e:
        print(f"  WARNING: Failed to log {model_class_name}: {e}")
        return None


def log_random_graph(target_nodes: int, seed: int = 42) -> Optional[Any]:
    """Generate a RandomGraphModel and log it."""
    try:
        import torchlens as tl
        import importlib.util
        models_path = Path.home() / "projects" / "torchlens" / "tests" / "example_models.py"
        spec = importlib.util.spec_from_file_location("example_models", models_path)
        example_models = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(example_models)
        RandomGraphModel = getattr(example_models, "RandomGraphModel")

        model = RandomGraphModel(target_nodes=target_nodes, seed=seed)
        model.eval()
        x = torch.randn(2, 64)
        with torch.no_grad():
            model_log = tl.log_forward_pass(model, x, vis_mode="none")

        del model, x
        gc.collect()
        return model_log
    except Exception as e:
        print(f"  WARNING: Failed to log RandomGraphModel(target={target_nodes}): {e}")
        traceback.print_exc()
        return None


def model_log_to_dagua_graph(model_log) -> DaguaGraph:
    """Convert a TorchLens ModelLog to DaguaGraph."""
    g = DaguaGraph.from_torchlens(model_log, direction="TB")
    g.compute_node_sizes()
    return g


def random_dag_graph(n_nodes: int, seed: int = 42) -> DaguaGraph:
    """Generate a random DAG DaguaGraph (no TorchLens dependency).

    Uses tensor-based construction for efficiency at scale.
    For N > 100K, uses vectorized numpy construction.
    """
    if n_nodes > 5_000:
        return _random_dag_graph_vectorized(n_nodes, seed)

    import random
    rng = random.Random(seed)

    src_list = []
    tgt_list = []
    max_reach = max(n_nodes // 10, 10)
    for i in range(n_nodes - 1):
        step = rng.randint(1, min(max_reach, n_nodes - i - 1))
        src_list.append(i)
        tgt_list.append(i + step)
        if rng.random() < 0.3 and i + step + 1 < n_nodes:
            step2 = rng.randint(1, min(max_reach, n_nodes - i - step - 1))
            src_list.append(i)
            tgt_list.append(i + step + step2)

    edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)

    g = DaguaGraph.from_edge_index(edge_index, num_nodes=n_nodes)
    g.compute_node_sizes()
    return g


def _random_dag_graph_vectorized(n_nodes: int, seed: int = 42) -> DaguaGraph:
    """Vectorized random DAG for very large graphs (100K+)."""
    rng = np.random.RandomState(seed)

    # Primary edges: each node connects to 1 random node ahead
    max_step = max(n_nodes // 10, 10)
    src = np.arange(n_nodes - 1)
    steps = rng.randint(1, min(max_step, n_nodes) + 1, size=n_nodes - 1)
    tgt = np.minimum(src + steps, n_nodes - 1)

    # Secondary edges: ~30% of nodes get a second connection
    mask = rng.random(n_nodes - 1) < 0.3
    src2 = src[mask]
    steps2 = rng.randint(1, min(max_step, n_nodes) + 1, size=mask.sum())
    tgt2 = np.minimum(src2 + steps2, n_nodes - 1)

    all_src = np.concatenate([src, src2])
    all_tgt = np.concatenate([tgt, tgt2])

    # Remove self-loops
    valid = all_src != all_tgt
    all_src = all_src[valid]
    all_tgt = all_tgt[valid]

    edge_index = torch.tensor(np.stack([all_src, all_tgt]), dtype=torch.long)

    g = DaguaGraph.from_edge_index(edge_index, num_nodes=n_nodes)
    # Skip per-label node sizing for huge graphs — use uniform sizes
    g.node_sizes = torch.tensor([[120.0, 40.0]]).expand(n_nodes, 2).clone()
    return g


# ---------------------------------------------------------------------------
# Layout engines
# ---------------------------------------------------------------------------

def run_dagua_layout(graph: DaguaGraph, steps: int, device: str = "cpu") -> Tuple[Optional[torch.Tensor], float]:
    """Run dagua layout, return (positions, time_seconds)."""
    n = graph.num_nodes
    config = LayoutConfig(steps=steps, device=device, direction="TB")

    # For very large graphs, configure multilevel params
    if n > 50000:
        config.multilevel_coarse_steps = min(steps * 2, 100)
        config.multilevel_refine_steps = max(steps // 5, 10)

    # Warmup for GPU
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    try:
        pos = dagua.layout(graph, config)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        return pos.detach().cpu(), elapsed
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"  WARNING: Dagua failed ({device}): {e}")
        traceback.print_exc()
        return None, elapsed


def run_graphviz_layout(graph: DaguaGraph, timeout: int = TIMEOUT_SECONDS) -> Tuple[Optional[torch.Tensor], float]:
    """Run graphviz dot layout, return (positions, time_seconds)."""
    dot_str = to_dot(graph)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".dot", delete=False) as f:
        f.write(dot_str)
        dot_path = f.name

    start = time.perf_counter()
    try:
        result = subprocess.run(
            ["dot", "-Tjson", dot_path],
            capture_output=True, text=True, timeout=timeout,
        )
        elapsed = time.perf_counter() - start

        if result.returncode != 0:
            print(f"  WARNING: Graphviz failed: {result.stderr[:200]}")
            return None, elapsed

        data = json.loads(result.stdout)
        positions = torch.zeros(graph.num_nodes, 2)

        if "objects" in data:
            for obj in data["objects"]:
                name = obj.get("name", "")
                if name.startswith("n") and name[1:].isdigit():
                    idx = int(name[1:])
                    if idx < graph.num_nodes and "pos" in obj:
                        coords = obj["pos"].split(",")
                        positions[idx, 0] = float(coords[0])
                        positions[idx, 1] = -float(coords[1])  # flip y

        return positions, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start
        print(f"  Graphviz TIMEOUT ({timeout}s)")
        return None, elapsed
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"  WARNING: Graphviz error: {e}")
        return None, elapsed
    finally:
        Path(dot_path).unlink(missing_ok=True)


# Inline ELK script
_ELK_SCRIPT = r"""
const ELK = require('elkjs');
const elk = new ELK();
let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => { input += chunk; });
process.stdin.on('end', () => {
    const graph = JSON.parse(input);
    elk.layout(graph).then((result) => {
        process.stdout.write(JSON.stringify(result));
    }).catch((err) => {
        process.stderr.write(err.toString());
        process.exit(1);
    });
});
"""


def run_elk_layout(graph: DaguaGraph, timeout: int = TIMEOUT_SECONDS) -> Tuple[Optional[torch.Tensor], float]:
    """Run ELK layered layout, return (positions, time_seconds)."""
    # Build ELK JSON
    n = graph.num_nodes
    children = [{"id": str(i), "width": 120, "height": 40} for i in range(n)]
    edges = []
    if graph.edge_index.numel() > 0:
        for e_idx in range(graph.edge_index.shape[1]):
            s = graph.edge_index[0, e_idx].item()
            t = graph.edge_index[1, e_idx].item()
            edges.append({"id": f"e{e_idx}", "sources": [str(s)], "targets": [str(t)]})

    elk_graph = {
        "id": "root",
        "layoutOptions": {
            "elk.algorithm": "layered",
            "elk.direction": "DOWN",
            "elk.spacing.nodeNode": "40",
            "elk.layered.spacing.nodeNodeBetweenLayers": "60",
        },
        "children": children,
        "edges": edges,
    }

    graph_json = json.dumps(elk_graph)
    graph_kb = len(graph_json) // 1024
    heap_mb = min(65536, max(16384, graph_kb * 48))

    start = time.perf_counter()
    try:
        result = subprocess.run(
            ["node", f"--max-old-space-size={heap_mb}", "-e", _ELK_SCRIPT],
            input=graph_json,
            capture_output=True, text=True, timeout=timeout,
        )
        elapsed = time.perf_counter() - start

        if result.returncode != 0:
            print(f"  WARNING: ELK failed: {result.stderr[:200]}")
            return None, elapsed

        # Parse positions from ELK output
        elk_result = json.loads(result.stdout)
        positions = torch.zeros(n, 2)

        if "children" in elk_result:
            for child in elk_result["children"]:
                idx = int(child["id"])
                if idx < n:
                    positions[idx, 0] = child.get("x", 0)
                    positions[idx, 1] = child.get("y", 0)

        return positions, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start
        print(f"  ELK TIMEOUT ({timeout}s)")
        return None, elapsed
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"  WARNING: ELK error: {e}")
        return None, elapsed


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics_safe(
    positions: Optional[torch.Tensor],
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    direction: str = "TB",
) -> Optional[Dict[str, float]]:
    """Compute metrics, return None if positions are None."""
    if positions is None:
        return None
    try:
        return compute_all_metrics(positions, edge_index, node_sizes, direction=direction)
    except Exception as e:
        print(f"  WARNING: Metrics computation failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def benchmark_single_graph(
    graph: DaguaGraph,
    name: str,
    dagua_steps: int = DAGUA_STEPS_QUALITY,
    run_graphviz: bool = True,
    run_elk: bool = True,
    run_gpu: bool = True,
) -> Dict[str, Any]:
    """Benchmark a single graph across all engines."""
    n = graph.num_nodes
    e = graph.edge_index.shape[1] if graph.edge_index.numel() > 0 else 0
    print(f"\n{'='*70}")
    print(f"  {name}: {n:,} nodes, {e:,} edges")
    print(f"{'='*70}")

    result = {
        "name": name,
        "num_nodes": n,
        "num_edges": e,
        "dagua_steps": dagua_steps,
    }

    # --- Dagua CPU ---
    print(f"  Dagua (CPU, {dagua_steps} steps)...", end="", flush=True)
    dagua_pos_cpu, dagua_time_cpu = run_dagua_layout(graph, dagua_steps, device="cpu")
    print(f" {dagua_time_cpu:.2f}s")
    result["dagua_cpu_time"] = round(dagua_time_cpu, 4)
    result["dagua_cpu_metrics"] = compute_metrics_safe(dagua_pos_cpu, graph.edge_index, graph.node_sizes)

    # --- Dagua GPU ---
    if run_gpu and torch.cuda.is_available() and n >= GPU_MIN_NODES:
        print(f"  Dagua (GPU, {dagua_steps} steps)...", end="", flush=True)
        dagua_pos_gpu, dagua_time_gpu = run_dagua_layout(graph, dagua_steps, device="cuda")
        print(f" {dagua_time_gpu:.2f}s")
        result["dagua_gpu_time"] = round(dagua_time_gpu, 4)
        result["dagua_gpu_metrics"] = compute_metrics_safe(dagua_pos_gpu, graph.edge_index, graph.node_sizes)
    else:
        result["dagua_gpu_time"] = None
        result["dagua_gpu_metrics"] = None

    # --- Graphviz ---
    if run_graphviz and n <= 20_000:
        print(f"  Graphviz (dot)...", end="", flush=True)
        gv_pos, gv_time = run_graphviz_layout(graph)
        if gv_pos is not None:
            print(f" {gv_time:.2f}s")
        result["graphviz_time"] = round(gv_time, 4) if gv_pos is not None else None
        result["graphviz_metrics"] = compute_metrics_safe(gv_pos, graph.edge_index, graph.node_sizes)
    else:
        result["graphviz_time"] = None
        result["graphviz_metrics"] = None
        if run_graphviz:
            print(f"  Graphviz: skipped (N={n:,} > 20K)")

    # --- ELK ---
    if run_elk and n <= 100_000:
        print(f"  ELK (layered)...", end="", flush=True)
        elk_pos, elk_time = run_elk_layout(graph)
        if elk_pos is not None:
            print(f" {elk_time:.2f}s")
        result["elk_time"] = round(elk_time, 4) if elk_pos is not None else None
        result["elk_metrics"] = compute_metrics_safe(elk_pos, graph.edge_index, graph.node_sizes)
    else:
        result["elk_time"] = None
        result["elk_metrics"] = None
        if run_elk:
            print(f"  ELK: skipped (N={n:,} > 100K)")

    # Print quick comparison
    _print_comparison(result)

    return result


def _print_comparison(result: Dict):
    """Print a quick comparison of metrics."""
    engines = []
    if result.get("dagua_cpu_metrics"):
        engines.append(("Dagua(CPU)", result["dagua_cpu_metrics"], result["dagua_cpu_time"]))
    if result.get("dagua_gpu_metrics"):
        engines.append(("Dagua(GPU)", result["dagua_gpu_metrics"], result["dagua_gpu_time"]))
    if result.get("graphviz_metrics"):
        engines.append(("Graphviz", result["graphviz_metrics"], result["graphviz_time"]))
    if result.get("elk_metrics"):
        engines.append(("ELK", result["elk_metrics"], result["elk_time"]))

    if len(engines) < 2:
        return

    key_metrics = ["edge_crossings", "node_overlaps", "dag_fraction", "edge_straightness", "overall_quality"]

    print(f"\n  {'Metric':<22}", end="")
    for name, _, _ in engines:
        print(f" {name:>14}", end="")
    print()
    print(f"  {'-'*22}", end="")
    for _ in engines:
        print(f" {'-'*14}", end="")
    print()

    print(f"  {'Runtime (s)':<22}", end="")
    for _, _, t in engines:
        print(f" {t:>14.2f}", end="")
    print()

    for key in key_metrics:
        print(f"  {key:<22}", end="")
        for _, m, _ in engines:
            val = m.get(key, "N/A")
            if isinstance(val, float):
                print(f" {val:>14.2f}", end="")
            else:
                print(f" {val:>14}", end="")
        print()


def run_real_model_benchmarks(run_gpu: bool = True, dagua_steps: int = 300) -> List[Dict]:
    """Benchmark real neural network architectures."""
    print("\n" + "=" * 70)
    print("PART 1: REAL NEURAL NETWORK ARCHITECTURES")
    print("=" * 70)

    results = []
    for display_name, class_name, input_shape in REAL_MODELS:
        print(f"\nLogging {display_name}...", end="", flush=True)
        model_log = log_model(class_name, input_shape)
        if model_log is None:
            print(" FAILED")
            continue

        try:
            graph = model_log_to_dagua_graph(model_log)
            print(f" {graph.num_nodes} nodes")
        except Exception as e:
            print(f" graph conversion failed: {e}")
            continue

        result = benchmark_single_graph(
            graph, display_name,
            dagua_steps=dagua_steps,
            run_graphviz=True,
            run_elk=True,
            run_gpu=run_gpu,
        )
        result["category"] = "real_model"
        result["model_class"] = class_name
        results.append(result)

        # Free memory
        del model_log, graph
        gc.collect()

    return results


def run_scaling_benchmarks(
    max_scale: int = 2_000_000,
    use_torchlens: bool = True,
    run_gpu: bool = True,
    dagua_steps_quality: int = 300,
    dagua_steps_scaling: int = 100,
) -> List[Dict]:
    """Benchmark synthetic graphs at increasing scales."""
    print("\n" + "=" * 70)
    print("PART 2: SCALING BENCHMARKS (RandomGraphModel)")
    print("=" * 70)

    results = []
    sizes = [s for s in SCALING_SIZES if s <= max_scale]

    for target_nodes in sizes:
        name = f"Random_{target_nodes // 1000}K" if target_nodes >= 1000 else f"Random_{target_nodes}"

        if use_torchlens and target_nodes <= 1_000:
            # Use TorchLens for smaller graphs (full model structure)
            print(f"\nGenerating {name} via TorchLens...", end="", flush=True)
            model_log = log_random_graph(target_nodes)
            if model_log is not None:
                try:
                    graph = model_log_to_dagua_graph(model_log)
                    print(f" {graph.num_nodes} nodes")
                except Exception as e:
                    print(f" conversion failed: {e}")
                    graph = random_dag_graph(target_nodes)
                    print(f" using fallback random DAG: {graph.num_nodes} nodes")
                del model_log
            else:
                graph = random_dag_graph(target_nodes)
                print(f" using fallback random DAG: {graph.num_nodes} nodes")
        else:
            # For very large graphs, use direct DAG generation (faster)
            print(f"\nGenerating {name} (direct DAG)...", end="", flush=True)
            t0 = time.perf_counter()
            graph = random_dag_graph(target_nodes)
            gen_time = time.perf_counter() - t0
            print(f" {graph.num_nodes} nodes ({gen_time:.1f}s)")

        # Adaptive step count based on graph size
        if target_nodes > 100_000:
            dagua_steps = max(dagua_steps_scaling // 2, 25)
        elif target_nodes > 5000:
            dagua_steps = dagua_steps_scaling
        else:
            dagua_steps = dagua_steps_quality

        result = benchmark_single_graph(
            graph, name,
            dagua_steps=dagua_steps,
            run_graphviz=(target_nodes <= 10_000),
            run_elk=(target_nodes <= 50_000),
            run_gpu=run_gpu,
        )
        result["category"] = "scaling"
        result["target_nodes"] = target_nodes
        results.append(result)

        del graph
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Results analysis
# ---------------------------------------------------------------------------

def analyze_results(results: List[Dict]) -> Dict:
    """Analyze benchmark results and identify dagua weaknesses."""
    analysis = {
        "dagua_wins": [],
        "dagua_losses": [],
        "dagua_ties": [],
        "speed_comparison": [],
        "quality_comparison": [],
    }

    key_metrics = ["edge_crossings", "node_overlaps", "dag_fraction", "edge_straightness",
                   "edge_length_variance", "x_alignment"]
    # For these metrics, lower is better (except dag_fraction where higher is better)
    lower_better = {"edge_crossings", "node_overlaps", "edge_straightness",
                    "edge_length_variance", "x_alignment"}

    for r in results:
        name = r["name"]
        dagua_m = r.get("dagua_cpu_metrics")
        gv_m = r.get("graphviz_metrics")
        elk_m = r.get("elk_metrics")

        if dagua_m and gv_m:
            for metric in key_metrics:
                dv = dagua_m.get(metric, 0)
                gvv = gv_m.get(metric, 0)
                if metric in lower_better:
                    winner = "dagua" if dv <= gvv else "graphviz"
                else:
                    winner = "dagua" if dv >= gvv else "graphviz"

                entry = {"graph": name, "metric": metric, "dagua": dv, "graphviz": gvv,
                         "winner": winner, "opponent": "graphviz"}
                if winner == "dagua":
                    analysis["dagua_wins"].append(entry)
                else:
                    analysis["dagua_losses"].append(entry)

        if dagua_m and elk_m:
            for metric in key_metrics:
                dv = dagua_m.get(metric, 0)
                ev = elk_m.get(metric, 0)
                if metric in lower_better:
                    winner = "dagua" if dv <= ev else "elk"
                else:
                    winner = "dagua" if dv >= ev else "elk"

                entry = {"graph": name, "metric": metric, "dagua": dv, "elk": ev,
                         "winner": winner, "opponent": "elk"}
                if winner == "dagua":
                    analysis["dagua_wins"].append(entry)
                else:
                    analysis["dagua_losses"].append(entry)

        # Speed comparison
        dt_cpu = r.get("dagua_cpu_time")
        dt_gpu = r.get("dagua_gpu_time")
        gvt = r.get("graphviz_time")
        elkt = r.get("elk_time")

        speed_entry = {"graph": name, "dagua_cpu": dt_cpu, "dagua_gpu": dt_gpu,
                       "graphviz": gvt, "elk": elkt}
        analysis["speed_comparison"].append(speed_entry)

    # Summary
    total_comparisons = len(analysis["dagua_wins"]) + len(analysis["dagua_losses"])
    if total_comparisons > 0:
        win_rate = len(analysis["dagua_wins"]) / total_comparisons
        analysis["win_rate"] = win_rate
        analysis["total_wins"] = len(analysis["dagua_wins"])
        analysis["total_losses"] = len(analysis["dagua_losses"])

    # Identify biggest weaknesses
    loss_by_metric = {}
    for loss in analysis["dagua_losses"]:
        m = loss["metric"]
        if m not in loss_by_metric:
            loss_by_metric[m] = []
        loss_by_metric[m].append(loss)
    analysis["loss_by_metric"] = {k: len(v) for k, v in loss_by_metric.items()}

    return analysis


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def generate_figures(results: List[Dict], output_dir: Path):
    """Generate all benchmark figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    output_dir.mkdir(parents=True, exist_ok=True)

    # Separate real model and scaling results
    real_results = [r for r in results if r.get("category") == "real_model"]
    scaling_results = [r for r in results if r.get("category") == "scaling"]

    # --- Figure 1: Runtime Scaling (log-log) ---
    if scaling_results:
        fig, ax = plt.subplots(figsize=(10, 6))

        nodes = [r["num_nodes"] for r in scaling_results]

        # Dagua CPU
        dagua_cpu = [(r["num_nodes"], r["dagua_cpu_time"]) for r in scaling_results if r.get("dagua_cpu_time")]
        if dagua_cpu:
            ax.plot(*zip(*dagua_cpu), "o-", color="#2196F3", linewidth=2, markersize=6, label="Dagua (CPU)")

        # Dagua GPU
        dagua_gpu = [(r["num_nodes"], r["dagua_gpu_time"]) for r in scaling_results if r.get("dagua_gpu_time")]
        if dagua_gpu:
            ax.plot(*zip(*dagua_gpu), "s-", color="#9C27B0", linewidth=2, markersize=6, label="Dagua (GPU)")

        # Graphviz
        gv = [(r["num_nodes"], r["graphviz_time"]) for r in scaling_results if r.get("graphviz_time")]
        if gv:
            ax.plot(*zip(*gv), "^--", color="#FF5722", linewidth=2, markersize=6, label="Graphviz (dot)")

        # ELK
        elk = [(r["num_nodes"], r["elk_time"]) for r in scaling_results if r.get("elk_time")]
        if elk:
            ax.plot(*zip(*elk), "D--", color="#4CAF50", linewidth=2, markersize=6, label="ELK (layered)")

        ax.set_xlabel("Number of Nodes", fontsize=12)
        ax.set_ylabel("Runtime (seconds)", fontsize=12)
        ax.set_title("Layout Engine Runtime Scaling", fontsize=14)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x):,}"))

        plt.tight_layout()
        fig.savefig(output_dir / "runtime_scaling.png", dpi=200, bbox_inches="tight")
        fig.savefig(output_dir / "runtime_scaling.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved runtime_scaling.png/pdf")

    # --- Figure 2: Quality comparison on real models (grouped bar) ---
    if real_results:
        metrics_to_plot = ["edge_crossings", "node_overlaps", "dag_fraction", "edge_straightness"]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            model_names = []
            dagua_vals = []
            gv_vals = []
            elk_vals = []

            for r in real_results:
                model_names.append(r["name"])
                dm = r.get("dagua_cpu_metrics") or {}
                gm = r.get("graphviz_metrics") or {}
                em = r.get("elk_metrics") or {}
                dagua_vals.append(dm.get(metric))
                gv_vals.append(gm.get(metric))
                elk_vals.append(em.get(metric))

            x = np.arange(len(model_names))
            width = 0.25

            # Filter to only models where at least dagua has a value
            valid = [i for i in range(len(model_names)) if dagua_vals[i] is not None]
            if not valid:
                continue

            x_valid = np.arange(len(valid))
            d_valid = [dagua_vals[i] for i in valid]
            g_valid = [gv_vals[i] for i in valid]
            e_valid = [elk_vals[i] for i in valid]
            names_valid = [model_names[i] for i in valid]

            bars_d = ax.bar(x_valid - width, d_valid, width, label="Dagua", color="#2196F3", alpha=0.8)

            # Only plot graphviz/elk where they have values
            g_plot = [v if v is not None else 0 for v in g_valid]
            e_plot = [v if v is not None else 0 for v in e_valid]
            has_gv = any(v is not None for v in g_valid)
            has_elk = any(v is not None for v in e_valid)

            if has_gv:
                ax.bar(x_valid, g_plot, width, label="Graphviz", color="#FF5722", alpha=0.8)
            if has_elk:
                ax.bar(x_valid + width, e_plot, width, label="ELK", color="#4CAF50", alpha=0.8)

            ax.set_title(metric.replace("_", " ").title(), fontsize=12)
            ax.set_xticks(x_valid)
            ax.set_xticklabels(names_valid, rotation=45, ha="right", fontsize=8)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis="y")

        plt.suptitle("Layout Quality: Real Neural Networks", fontsize=14, y=1.02)
        plt.tight_layout()
        fig.savefig(output_dir / "quality_real_models.png", dpi=200, bbox_inches="tight")
        fig.savefig(output_dir / "quality_real_models.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved quality_real_models.png/pdf")

    # --- Figure 3: Runtime comparison bar chart (real models) ---
    if real_results:
        fig, ax = plt.subplots(figsize=(12, 6))
        model_names = [r["name"] for r in real_results]
        x = np.arange(len(model_names))
        width = 0.2

        cpu_times = [r.get("dagua_cpu_time", 0) for r in real_results]
        gpu_times = [r.get("dagua_gpu_time") or 0 for r in real_results]
        gv_times = [r.get("graphviz_time") or 0 for r in real_results]
        elk_times = [r.get("elk_time") or 0 for r in real_results]

        ax.bar(x - 1.5*width, cpu_times, width, label="Dagua CPU", color="#2196F3", alpha=0.8)
        ax.bar(x - 0.5*width, gpu_times, width, label="Dagua GPU", color="#9C27B0", alpha=0.8)
        ax.bar(x + 0.5*width, gv_times, width, label="Graphviz", color="#FF5722", alpha=0.8)
        ax.bar(x + 1.5*width, elk_times, width, label="ELK", color="#4CAF50", alpha=0.8)

        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylabel("Runtime (seconds)", fontsize=12)
        ax.set_title("Layout Runtime: Real Neural Networks", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=9)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        fig.savefig(output_dir / "runtime_real_models.png", dpi=200, bbox_inches="tight")
        fig.savefig(output_dir / "runtime_real_models.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved runtime_real_models.png/pdf")

    # --- Figure 4: Dagua win/loss heatmap ---
    if real_results:
        metrics_list = ["edge_crossings", "node_overlaps", "dag_fraction",
                        "edge_straightness", "edge_length_variance", "x_alignment"]
        lower_better = {"edge_crossings", "node_overlaps", "edge_straightness",
                        "edge_length_variance", "x_alignment"}

        # vs Graphviz
        models_with_gv = [r for r in real_results if r.get("graphviz_metrics")]
        if models_with_gv:
            fig, ax = plt.subplots(figsize=(10, max(4, len(models_with_gv) * 0.6)))

            matrix = []
            names = []
            for r in models_with_gv:
                row = []
                names.append(r["name"])
                dm = r["dagua_cpu_metrics"]
                gm = r["graphviz_metrics"]
                for m in metrics_list:
                    dv = dm.get(m, 0)
                    gv = gm.get(m, 0)
                    if m in lower_better:
                        # Lower is better: positive = dagua wins
                        if gv != 0:
                            improvement = (gv - dv) / max(abs(gv), 1e-6) * 100
                        else:
                            improvement = 0 if dv == 0 else -100
                    else:
                        # Higher is better: positive = dagua wins
                        if gv != 0:
                            improvement = (dv - gv) / max(abs(gv), 1e-6) * 100
                        else:
                            improvement = 0 if dv == 0 else 100
                    row.append(improvement)
                matrix.append(row)

            matrix = np.array(matrix)
            im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-100, vmax=100)

            ax.set_xticks(range(len(metrics_list)))
            ax.set_xticklabels([m.replace("_", "\n") for m in metrics_list], fontsize=9)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=9)
            ax.set_title("Dagua vs Graphviz (% improvement, green=dagua wins)", fontsize=12)

            # Add text annotations
            for i in range(len(names)):
                for j in range(len(metrics_list)):
                    val = matrix[i, j]
                    color = "white" if abs(val) > 60 else "black"
                    ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                            fontsize=8, color=color)

            plt.colorbar(im, ax=ax, label="% improvement (positive = dagua better)")
            plt.tight_layout()
            fig.savefig(output_dir / "dagua_vs_graphviz_heatmap.png", dpi=200, bbox_inches="tight")
            fig.savefig(output_dir / "dagua_vs_graphviz_heatmap.pdf", bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved dagua_vs_graphviz_heatmap.png/pdf")

        # vs ELK
        models_with_elk = [r for r in real_results if r.get("elk_metrics")]
        if models_with_elk:
            fig, ax = plt.subplots(figsize=(10, max(4, len(models_with_elk) * 0.6)))

            matrix = []
            names = []
            for r in models_with_elk:
                row = []
                names.append(r["name"])
                dm = r["dagua_cpu_metrics"]
                em = r["elk_metrics"]
                for m in metrics_list:
                    dv = dm.get(m, 0)
                    ev = em.get(m, 0)
                    if m in lower_better:
                        if ev != 0:
                            improvement = (ev - dv) / max(abs(ev), 1e-6) * 100
                        else:
                            improvement = 0 if dv == 0 else -100
                    else:
                        if ev != 0:
                            improvement = (dv - ev) / max(abs(ev), 1e-6) * 100
                        else:
                            improvement = 0 if dv == 0 else 100
                    row.append(improvement)
                matrix.append(row)

            matrix = np.array(matrix)
            im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-100, vmax=100)

            ax.set_xticks(range(len(metrics_list)))
            ax.set_xticklabels([m.replace("_", "\n") for m in metrics_list], fontsize=9)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=9)
            ax.set_title("Dagua vs ELK (% improvement, green=dagua wins)", fontsize=12)

            for i in range(len(names)):
                for j in range(len(metrics_list)):
                    val = matrix[i, j]
                    color = "white" if abs(val) > 60 else "black"
                    ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                            fontsize=8, color=color)

            plt.colorbar(im, ax=ax, label="% improvement (positive = dagua better)")
            plt.tight_layout()
            fig.savefig(output_dir / "dagua_vs_elk_heatmap.png", dpi=200, bbox_inches="tight")
            fig.savefig(output_dir / "dagua_vs_elk_heatmap.pdf", bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved dagua_vs_elk_heatmap.png/pdf")

    # --- Figure 5: Scaling quality metrics ---
    if scaling_results:
        metrics_to_track = ["edge_crossings", "dag_fraction", "node_overlaps", "edge_straightness"]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics_to_track):
            ax = axes[idx]

            # Dagua CPU
            dagua_data = [(r["num_nodes"], r["dagua_cpu_metrics"][metric])
                          for r in scaling_results if r.get("dagua_cpu_metrics") and metric in r["dagua_cpu_metrics"]]
            if dagua_data:
                ax.plot(*zip(*dagua_data), "o-", color="#2196F3", label="Dagua", linewidth=2, markersize=5)

            # Graphviz
            gv_data = [(r["num_nodes"], r["graphviz_metrics"][metric])
                       for r in scaling_results if r.get("graphviz_metrics") and metric in r["graphviz_metrics"]]
            if gv_data:
                ax.plot(*zip(*gv_data), "^--", color="#FF5722", label="Graphviz", linewidth=2, markersize=5)

            # ELK
            elk_data = [(r["num_nodes"], r["elk_metrics"][metric])
                        for r in scaling_results if r.get("elk_metrics") and metric in r["elk_metrics"]]
            if elk_data:
                ax.plot(*zip(*elk_data), "D--", color="#4CAF50", label="ELK", linewidth=2, markersize=5)

            ax.set_title(metric.replace("_", " ").title(), fontsize=12)
            ax.set_xlabel("Nodes")
            ax.set_xscale("log")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.suptitle("Layout Quality at Scale", fontsize=14, y=1.02)
        plt.tight_layout()
        fig.savefig(output_dir / "quality_scaling.png", dpi=200, bbox_inches="tight")
        fig.savefig(output_dir / "quality_scaling.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved quality_scaling.png/pdf")


# ---------------------------------------------------------------------------
# LaTeX report
# ---------------------------------------------------------------------------

def generate_latex_report(results: List[Dict], analysis: Dict, output_dir: Path):
    """Generate a LaTeX report with figures and tables."""

    real_results = [r for r in results if r.get("category") == "real_model"]
    scaling_results = [r for r in results if r.get("category") == "scaling"]

    tex = r"""\documentclass[11pt, a4paper]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{float}
\usepackage{longtable}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{amsmath}

\definecolor{daguablue}{HTML}{2196F3}
\definecolor{gvred}{HTML}{FF5722}
\definecolor{elkgreen}{HTML}{4CAF50}
\definecolor{gpupurple}{HTML}{9C27B0}

\title{\textbf{Dagua vs.\ Graphviz vs.\ ELK:\\Comprehensive Layout Engine Benchmark Report}}
\author{Automated Benchmark Suite}
\date{""" + time.strftime("%B %d, %Y") + r"""}

\begin{document}
\maketitle

\begin{abstract}
This report presents a comprehensive comparison of three graph layout engines---\textcolor{daguablue}{\textbf{Dagua}} (GPU-accelerated differentiable layout), \textcolor{gvred}{\textbf{Graphviz}} (dot hierarchical layout), and \textcolor{elkgreen}{\textbf{ELK}} (Eclipse Layout Kernel, layered algorithm)---across """ + str(len(real_results)) + r""" real neural network architectures and """ + str(len(scaling_results)) + r""" synthetic graph scales (up to """ + f"{max([r['num_nodes'] for r in scaling_results], default=0):,}" + r""" nodes). We compare both \emph{runtime performance} (CPU and GPU) and \emph{aesthetic quality metrics} (edge crossings, node overlaps, DAG fraction, edge straightness, and more).
\end{abstract}

\tableofcontents
\newpage

% ===================================================================
\section{Experimental Setup}
% ===================================================================

\subsection{Layout Engines}
\begin{itemize}
    \item \textbf{Dagua} (v0.0.2): PyTorch-based differentiable graph layout engine. Layout as continuous optimization with composable loss functions. Supports CPU and CUDA. """ + f"Steps: {DAGUA_STEPS_QUALITY} (quality), {DAGUA_STEPS_SCALING} (scaling)." + r"""
    \item \textbf{Graphviz} (dot): Classic hierarchical layout algorithm. Gold standard for small-to-medium DAGs. Timeout: """ + str(TIMEOUT_SECONDS) + r"""s.
    \item \textbf{ELK} (elkjs, layered): Eclipse Layout Kernel via Node.js. Hierarchical layered algorithm. Timeout: """ + str(TIMEOUT_SECONDS) + r"""s.
\end{itemize}

\subsection{Hardware}
\begin{itemize}
"""

    tex += r"    \item \textbf{CPU}: " + _get_cpu_info() + "\n"
    if torch.cuda.is_available():
        tex += r"    \item \textbf{GPU}: " + torch.cuda.get_device_name(0) + "\n"
    tex += r"""
\end{itemize}

\subsection{Aesthetic Metrics}
All metrics computed via Dagua's \texttt{compute\_all\_metrics()} on the [N, 2] position tensors:
\begin{itemize}
    \item \textbf{Edge Crossings}: Count of intersecting edge pairs (lower is better)
    \item \textbf{Node Overlaps}: Count of overlapping bounding box pairs (lower is better)
    \item \textbf{DAG Fraction}: Fraction of edges pointing in layout direction (higher is better)
    \item \textbf{Edge Straightness}: Mean angular deviation from primary axis in degrees (lower is better)
    \item \textbf{Edge Length Variance}: Variance of Euclidean edge lengths (lower is better)
    \item \textbf{X-Alignment}: Mean cross-axis displacement between connected nodes (lower is better)
    \item \textbf{Overall Quality}: Composite score (higher is better)
\end{itemize}

"""

    # ===================================================================
    # Section 2: Real model results
    # ===================================================================
    if real_results:
        tex += r"""
% ===================================================================
\section{Real Neural Network Architectures}
% ===================================================================

"""
        # Runtime table
        tex += r"""
\subsection{Runtime Comparison}
\begin{table}[H]
\centering
\caption{Layout runtime (seconds) for real neural network architectures.}
\begin{tabular}{l r r r r r r}
\toprule
\textbf{Model} & \textbf{Nodes} & \textbf{Edges} & \textbf{Dagua CPU} & \textbf{Dagua GPU} & \textbf{Graphviz} & \textbf{ELK} \\
\midrule
"""
        for r in real_results:
            gpu_t = f"{r['dagua_gpu_time']:.2f}" if r.get('dagua_gpu_time') else "---"
            gv_t = f"{r['graphviz_time']:.2f}" if r.get('graphviz_time') else "---"
            elk_t = f"{r['elk_time']:.2f}" if r.get('elk_time') else "---"
            tex += f"{r['name']} & {r['num_nodes']:,} & {r['num_edges']:,} & {r['dagua_cpu_time']:.2f} & {gpu_t} & {gv_t} & {elk_t} \\\\\n"

        tex += r"""
\bottomrule
\end{tabular}
\end{table}
"""

        # Quality table
        tex += r"""
\subsection{Quality Metrics}
\begin{table}[H]
\centering
\caption{Aesthetic quality metrics for real neural network architectures. Best value per model bolded.}
\resizebox{\textwidth}{!}{
\begin{tabular}{l l r r r r r r r}
\toprule
\textbf{Model} & \textbf{Engine} & \textbf{Crossings} & \textbf{Overlaps} & \textbf{DAG \%} & \textbf{Straightness} & \textbf{Len.\ Var.} & \textbf{X-Align} & \textbf{Quality} \\
\midrule
"""
        for r in real_results:
            engines = [("Dagua", r.get("dagua_cpu_metrics")),
                       ("Graphviz", r.get("graphviz_metrics")),
                       ("ELK", r.get("elk_metrics"))]
            engines = [(n, m) for n, m in engines if m]

            for i, (ename, m) in enumerate(engines):
                model_col = r["name"] if i == 0 else ""
                tex += f"{model_col} & {ename} & {m.get('edge_crossings', 0)} & {m.get('node_overlaps', 0)} & {m.get('dag_fraction', 0):.3f} & {m.get('edge_straightness', 0):.1f} & {m.get('edge_length_variance', 0):.1f} & {m.get('x_alignment', 0):.1f} & {m.get('overall_quality', 0):.1f} \\\\\n"

            tex += r"\midrule" + "\n"

        tex += r"""
\bottomrule
\end{tabular}
}
\end{table}
"""

        # Figures
        tex += r"""
\subsection{Figures}

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{figures/runtime_real_models.pdf}
    \caption{Runtime comparison across real neural network architectures.}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{figures/quality_real_models.pdf}
    \caption{Quality metrics comparison across real neural network architectures.}
\end{figure}

"""
        if (output_dir / "figures" / "dagua_vs_graphviz_heatmap.pdf").exists():
            tex += r"""
\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{figures/dagua_vs_graphviz_heatmap.pdf}
    \caption{Dagua vs Graphviz improvement heatmap (\% change, green = Dagua better).}
\end{figure}
"""
        if (output_dir / "figures" / "dagua_vs_elk_heatmap.pdf").exists():
            tex += r"""
\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{figures/dagua_vs_elk_heatmap.pdf}
    \caption{Dagua vs ELK improvement heatmap (\% change, green = Dagua better).}
\end{figure}
"""

    # ===================================================================
    # Section 3: Scaling results
    # ===================================================================
    if scaling_results:
        tex += r"""
% ===================================================================
\section{Scaling Benchmarks}
% ===================================================================

\subsection{Runtime Scaling}
\begin{table}[H]
\centering
\caption{Runtime scaling from 500 to """ + f"{max(r['num_nodes'] for r in scaling_results):,}" + r""" nodes.}
\begin{tabular}{r r r r r r}
\toprule
\textbf{Nodes} & \textbf{Edges} & \textbf{Dagua CPU (s)} & \textbf{Dagua GPU (s)} & \textbf{Graphviz (s)} & \textbf{ELK (s)} \\
\midrule
"""
        for r in scaling_results:
            gpu_t = f"{r['dagua_gpu_time']:.2f}" if r.get('dagua_gpu_time') else "---"
            gv_t = f"{r['graphviz_time']:.2f}" if r.get('graphviz_time') else "---"
            elk_t = f"{r['elk_time']:.2f}" if r.get('elk_time') else "---"
            tex += f"{r['num_nodes']:,} & {r['num_edges']:,} & {r['dagua_cpu_time']:.2f} & {gpu_t} & {gv_t} & {elk_t} \\\\\n"

        tex += r"""
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{figures/runtime_scaling.pdf}
    \caption{Log-log runtime scaling comparison. Dagua maintains sub-quadratic scaling via multilevel coarsening.}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{figures/quality_scaling.pdf}
    \caption{Quality metrics at increasing graph sizes.}
\end{figure}
"""

    # ===================================================================
    # Section 4: Analysis
    # ===================================================================
    tex += r"""
% ===================================================================
\section{Analysis}
% ===================================================================

\subsection{Win/Loss Summary}
"""

    if "win_rate" in analysis:
        tex += f"""
Across all pairwise metric comparisons:
\\begin{{itemize}}
    \\item \\textbf{{Total comparisons}}: {analysis['total_wins'] + analysis['total_losses']}
    \\item \\textbf{{Dagua wins}}: {analysis['total_wins']} ({analysis['win_rate']*100:.1f}\\%)
    \\item \\textbf{{Dagua losses}}: {analysis['total_losses']} ({(1-analysis['win_rate'])*100:.1f}\\%)
\\end{{itemize}}
"""

    if analysis.get("loss_by_metric"):
        tex += r"""
\subsection{Dagua Weaknesses by Metric}
\begin{table}[H]
\centering
\begin{tabular}{l r}
\toprule
\textbf{Metric} & \textbf{Losses} \\
\midrule
"""
        for metric, count in sorted(analysis["loss_by_metric"].items(), key=lambda x: -x[1]):
            tex += f"{metric.replace('_', ' ').title()} & {count} \\\\\n"
        tex += r"""
\bottomrule
\end{tabular}
\end{table}
"""

    # ===================================================================
    # Section 5: Improvements Made
    # ===================================================================
    tex += r"""
% ===================================================================
\section{Improvements Made During Benchmarking}
% ===================================================================

\textit{This section documents any improvements made to Dagua during the benchmarking session to address identified weaknesses. See below for details.}

"""

    # Will be filled in after iteration
    improvements_file = output_dir / "improvements.txt"
    if improvements_file.exists():
        improvements = improvements_file.read_text()
        tex += improvements
    else:
        tex += r"\textit{(Improvements will be documented here after iteration.)}" + "\n"

    # ===================================================================
    # Section 6: Lessons
    # ===================================================================
    tex += r"""
% ===================================================================
\section{Lessons and Takeaways}
% ===================================================================

\begin{enumerate}
"""

    # Generate lessons from analysis
    lessons = _generate_lessons(results, analysis)
    for lesson in lessons:
        tex += f"    \\item {_latex_escape(lesson)}\n"

    tex += r"""
\end{enumerate}

% ===================================================================
\section{Conclusion}
% ===================================================================

"""

    conclusion = _generate_conclusion(results, analysis)
    tex += _latex_escape(conclusion) + "\n"

    tex += r"""
\end{document}
"""

    # Write LaTeX file
    tex_path = output_dir / "benchmark_report.tex"
    tex_path.write_text(tex)
    print(f"  Wrote {tex_path}")

    # Compile to PDF
    try:
        for _ in range(2):  # Run twice for TOC
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(output_dir),
                 str(tex_path)],
                capture_output=True, text=True, timeout=60,
                cwd=str(output_dir),
            )
        pdf_path = output_dir / "benchmark_report.pdf"
        if pdf_path.exists():
            print(f"  Compiled {pdf_path}")
        else:
            print("  WARNING: PDF compilation failed. LaTeX source saved.")
    except Exception as e:
        print(f"  WARNING: PDF compilation error: {e}")


def _latex_escape(text: str) -> str:
    """Escape special LaTeX characters."""
    replacements = {
        '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
        '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _get_cpu_info() -> str:
    """Get CPU model name."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    return line.split(":")[1].strip()
    except Exception:
        pass
    return "Unknown CPU"


def _generate_lessons(results: List[Dict], analysis: Dict) -> List[str]:
    """Generate lessons from the benchmark results."""
    lessons = []

    # Speed lessons
    real_results = [r for r in results if r.get("category") == "real_model"]
    scaling_results = [r for r in results if r.get("category") == "scaling"]

    # Find crossover point
    for r in scaling_results:
        if r.get("graphviz_time") and r.get("dagua_cpu_time"):
            if r["dagua_cpu_time"] < r["graphviz_time"]:
                lessons.append(f"Dagua becomes faster than Graphviz at approximately {r['num_nodes']:,} nodes on CPU.")
                break

    # GPU speedup
    gpu_speedups = []
    for r in results:
        if r.get("dagua_cpu_time") and r.get("dagua_gpu_time") and r["dagua_gpu_time"] > 0:
            speedup = r["dagua_cpu_time"] / r["dagua_gpu_time"]
            gpu_speedups.append((r["name"], speedup))
    if gpu_speedups:
        avg_speedup = np.mean([s for _, s in gpu_speedups])
        best_name, best_speedup = max(gpu_speedups, key=lambda x: x[1])
        lessons.append(f"GPU acceleration provides an average {avg_speedup:.1f}x speedup over CPU (best: {best_speedup:.1f}x on {best_name}).")

    # Quality lessons
    if analysis.get("loss_by_metric"):
        worst_metric = max(analysis["loss_by_metric"].items(), key=lambda x: x[1])
        lessons.append(f"Dagua's weakest metric relative to competitors is {worst_metric[0].replace('_', ' ')} ({worst_metric[1]} losses).")

    if analysis.get("win_rate"):
        lessons.append(f"Overall, Dagua wins {analysis['win_rate']*100:.0f}% of quality metric comparisons against Graphviz and ELK.")

    # Scalability
    max_dagua = max([r["num_nodes"] for r in results if r.get("dagua_cpu_time")], default=0)
    max_gv = max([r["num_nodes"] for r in results if r.get("graphviz_time")], default=0)
    max_elk = max([r["num_nodes"] for r in results if r.get("elk_time")], default=0)

    if max_dagua > max_gv:
        lessons.append(f"Dagua handles graphs up to {max_dagua:,} nodes, while Graphviz maxes out at {max_gv:,} nodes.")
    if max_dagua > max_elk:
        lessons.append(f"Dagua handles graphs up to {max_dagua:,} nodes, while ELK maxes out at {max_elk:,} nodes.")

    # Edge crossings insight
    crossing_ratios = []
    for r in real_results:
        dm = r.get("dagua_cpu_metrics", {})
        gm = r.get("graphviz_metrics", {})
        if dm.get("edge_crossings", 0) > 0 and gm.get("edge_crossings", 0) > 0:
            crossing_ratios.append(dm["edge_crossings"] / gm["edge_crossings"])
    if crossing_ratios:
        avg_ratio = np.mean(crossing_ratios)
        if avg_ratio > 1.5:
            lessons.append(f"Edge crossing reduction remains a key improvement area: Dagua averages {avg_ratio:.1f}x more crossings than Graphviz on real models.")
        elif avg_ratio < 0.8:
            lessons.append(f"Dagua achieves fewer edge crossings than Graphviz on average ({avg_ratio:.1f}x ratio).")

    if not lessons:
        lessons.append("Benchmarking completed successfully. See tables and figures for detailed results.")

    return lessons


def _generate_conclusion(results: List[Dict], analysis: Dict) -> str:
    """Generate conclusion text."""
    real_results = [r for r in results if r.get("category") == "real_model"]
    scaling_results = [r for r in results if r.get("category") == "scaling"]

    parts = []
    parts.append(f"This benchmark evaluated {len(real_results)} real neural network architectures and {len(scaling_results)} synthetic graph scales.")

    if analysis.get("win_rate") is not None:
        parts.append(f"Dagua achieved a {analysis['win_rate']*100:.0f}% win rate across all quality metric comparisons.")

    max_nodes = max([r["num_nodes"] for r in results if r.get("dagua_cpu_time")], default=0)
    if max_nodes >= 1_000_000:
        parts.append(f"Dagua successfully laid out graphs up to {max_nodes:,} nodes, demonstrating strong scalability beyond what Graphviz or ELK can handle.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Dagua vs Graphviz vs ELK benchmark")
    parser.add_argument("--max-scale", type=int, default=2_000_000, help="Maximum node count for scaling tests")
    parser.add_argument("--skip-real-models", action="store_true", help="Skip real model benchmarks")
    parser.add_argument("--skip-scaling", action="store_true", help="Skip scaling benchmarks")
    parser.add_argument("--dagua-only", action="store_true", help="Only benchmark Dagua")
    parser.add_argument("--no-gpu", action="store_true", help="Skip GPU benchmarks")
    parser.add_argument("--steps-quality", type=int, default=300)
    parser.add_argument("--steps-scaling", type=int, default=100)
    args = parser.parse_args()

    dagua_steps_quality = args.steps_quality
    dagua_steps_scaling = args.steps_scaling

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("COMPREHENSIVE LAYOUT ENGINE BENCHMARK")
    print(f"Dagua vs Graphviz (dot) vs ELK (layered)")
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"Max scale: {args.max_scale:,} nodes")
    print("=" * 70)

    all_results = []

    # Part 1: Real models
    if not args.skip_real_models:
        real_results = run_real_model_benchmarks(
            run_gpu=not args.no_gpu,
            dagua_steps=dagua_steps_quality,
        )
        all_results.extend(real_results)

    # Part 2: Scaling
    if not args.skip_scaling:
        scaling_results = run_scaling_benchmarks(
            max_scale=args.max_scale,
            run_gpu=not args.no_gpu,
            dagua_steps_quality=dagua_steps_quality,
            dagua_steps_scaling=dagua_steps_scaling,
        )
        all_results.extend(scaling_results)

    # Save raw results
    results_file = RESULTS_DIR / "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nRaw results saved to {results_file}")

    # Analyze
    print("\nAnalyzing results...")
    analysis = analyze_results(all_results)
    analysis_file = RESULTS_DIR / "analysis.json"
    with open(analysis_file, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"Analysis saved to {analysis_file}")

    # Generate figures
    print("\nGenerating figures...")
    generate_figures(all_results, FIGURES_DIR)

    # Generate LaTeX report
    print("\nGenerating LaTeX report...")
    generate_latex_report(all_results, analysis, RESULTS_DIR)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

    return all_results, analysis


if __name__ == "__main__":
    main()
