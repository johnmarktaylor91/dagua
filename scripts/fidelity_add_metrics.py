#!/usr/bin/env python3
"""Add quality metrics to existing fidelity analysis CSVs.

Second-pass script: reads per_seed_detail.csv, loads positions from HDF5,
computes quick and sampled quality metrics on each layout, and merges metric
columns into both per_seed_detail.csv and per_graph_detail.csv.

Does NOT recompute Procrustes -- only adds quality metrics.

Usage:
    python scripts/fidelity_add_metrics.py \
        --input eval_output/variant_bench_full \
        --data eval_output/fidelity_report/data
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fidelity_analysis import (  # noqa: E402
    ALL_QUALITY_METRICS,
    FIDELITY_CROSSING_N_SAMPLES,
    FIDELITY_STRESS_N_SOURCES,
    FIDELITY_STRESS_N_TARGETS,
    QUALITY_METRICS,
    SAMPLED_QUALITY_METRICS,
)

from dagua.eval.pipeline_io import stable_seed  # noqa: E402
from dagua.eval.variants import get_variant, original_variant_name  # noqa: E402
from dagua.metrics import quick, sampled_crossing_rate, sampled_stress  # noqa: E402


def _compute_metrics_for_row(
    result_key: str,
    positions_np: np.ndarray,
    graph_name: str,
    variant_id: str,
    side: str,
    seed_value: str,
) -> dict[str, float]:
    """Compute quality metrics for one layout.

    Parameters
    ----------
    result_key : str
        Benchmark record key for this layout.
    positions_np : numpy.ndarray
        Position array [N, 2].
    graph_name : str
        Graph name for looking up edge_index and node_sizes.
    variant_id : str
        Fidelity variant identifier.
    side : str
        Layout side, either ``"orig"`` or ``"reimpl"``.
    seed_value : str
        Seed text from the per-seed CSV row.

    Returns
    -------
    dict[str, float]
        Metric name -> value mapping.
    """
    try:
        from dagua.eval.graphs import get_test_graphs

        # Cache graph registry in function attribute
        if not hasattr(_compute_metrics_for_row, "_graph_cache"):
            _compute_metrics_for_row._graph_cache = {  # type: ignore[attr-defined]
                tg.name: tg for tg in get_test_graphs()
            }
        graph_cache = _compute_metrics_for_row._graph_cache  # type: ignore[attr-defined]

        tg = graph_cache.get(graph_name)
        if tg is None:
            return {m: math.nan for m in ALL_QUALITY_METRICS}

        pos = torch.from_numpy(positions_np).to(dtype=torch.float32)
        graph = tg.graph
        graph.compute_node_sizes()
        layout_seed = stable_seed(graph_name, variant_id, side, seed_value or "0")
        raw_metrics = quick(pos, graph.edge_index, node_sizes=graph.node_sizes, seed=layout_seed)
        metrics = {m: float(raw_metrics.get(m, math.nan)) for m in QUALITY_METRICS}
        stress_result = sampled_stress(
            pos,
            graph.edge_index,
            num_nodes=int(graph.node_sizes.shape[0]),
            n_sources=FIDELITY_STRESS_N_SOURCES,
            n_targets=FIDELITY_STRESS_N_TARGETS,
        )
        crossing_result = sampled_crossing_rate(
            pos,
            graph.edge_index,
            n_samples=FIDELITY_CROSSING_N_SAMPLES,
            seed=layout_seed,
        )
        for metric_name in SAMPLED_QUALITY_METRICS:
            if metric_name in stress_result:
                metrics[metric_name] = float(stress_result[metric_name])
            elif metric_name in crossing_result:
                metrics[metric_name] = float(crossing_result[metric_name])
            else:
                metrics[metric_name] = math.nan
        return metrics
    except Exception:
        return {m: math.nan for m in ALL_QUALITY_METRICS}


def _worker_batch(
    batch: list[tuple[str, str, str, str, str, str]],
    h5_path: str,
) -> list[tuple[str, dict[str, float]]]:
    """Process a batch of per-seed metric requests in one worker.

    Opens its own HDF5 handle to avoid cross-process sharing.

    Parameters
    ----------
    batch : list[tuple[str, str, str, str, str, str]]
        List of ``(result_key, graph_name, variant_id, side, seed_text, row_index_str)`` tuples.
    h5_path : str
        Path to positions.h5.

    Returns
    -------
    list[tuple[str, dict[str, float]]]
        List of (row_index_str, metrics_dict) tuples.
    """
    results = []
    with h5py.File(h5_path, "r") as h5:
        for result_key, graph_name, variant_id, side, seed_text, row_idx in batch:
            if result_key not in h5:
                results.append((row_idx, {m: math.nan for m in ALL_QUALITY_METRICS}))
                continue
            positions_np = h5[result_key][:]
            metrics = _compute_metrics_for_row(
                result_key,
                positions_np,
                graph_name,
                variant_id,
                side,
                seed_text,
            )
            results.append((row_idx, metrics))
    return results


def reconstruct_result_key(row: dict[str, str]) -> str:
    """Rebuild the ``results.json`` / HDF5 key for one per-seed CSV row.

    Parameters
    ----------
    row : dict[str, str]
        Per-seed CSV row.

    Returns
    -------
    str
        Stable benchmark record key.

    Raises
    ------
    ValueError
        Raised when an original-side row cannot be mapped to its synthetic
        original variant engine name.
    """
    graph_name = row.get("graph_name", "")
    variant_id = row.get("variant_id", "")
    side = row.get("side", "")
    seed = row.get("seed", "")

    engine_name = variant_id
    if side == "orig":
        variant = get_variant(variant_id)
        original_name = None if variant is None else original_variant_name(variant)
        if original_name is None:
            raise ValueError(f"Cannot resolve original engine for variant '{variant_id}'")
        engine_name = original_name

    if seed and seed != "None":
        return f"{graph_name}::{engine_name}::seed{seed}"
    return f"{graph_name}::{engine_name}::deterministic"


def main() -> None:
    """Merge quick metrics into existing fidelity-analysis CSV outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("eval_output/variant_bench_full"),
        help="Benchmark artifact directory",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("eval_output/fidelity_report/data"),
        help="Fidelity report data directory with existing CSVs",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(os.cpu_count() or 1, 8),
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Rows per worker batch",
    )
    args = parser.parse_args()

    h5_path = args.input / "positions.h5"
    if not h5_path.exists():
        print(
            f"ERROR: {h5_path} not found. Run consolidate_positions_hdf5.py first.", file=sys.stderr
        )
        sys.exit(1)

    seed_csv = args.data / "per_seed_detail.csv"
    graph_csv = args.data / "per_graph_detail.csv"

    if not seed_csv.exists():
        print(f"ERROR: {seed_csv} not found. Run fidelity_analysis.py first.", file=sys.stderr)
        sys.exit(1)

    # Load existing per_seed_detail.csv
    print(f"[metrics] Loading {seed_csv}...", file=sys.stderr)
    with open(seed_csv) as f:
        reader = csv.DictReader(f)
        seed_rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    print(f"[metrics] {len(seed_rows)} seed rows loaded", file=sys.stderr)

    # Build work items: (result_key, graph_name, row_index)
    # Reconstruct result_key from the CSV columns
    work_items: list[tuple[str, str, str, str, str, str]] = []
    for idx, row in enumerate(seed_rows):
        graph = row.get("graph_name", "")
        result_key = reconstruct_result_key(row)
        work_items.append(
            (
                result_key,
                graph,
                row.get("variant_id", ""),
                row.get("side", ""),
                row.get("seed", ""),
                str(idx),
            )
        )

    # Process in parallel batches
    batches = [
        work_items[i : i + args.batch_size] for i in range(0, len(work_items), args.batch_size)
    ]
    print(
        f"[metrics] Computing metrics: {len(work_items)} layouts, "
        f"{len(batches)} batches, {args.workers} workers",
        file=sys.stderr,
    )

    start = time.perf_counter()
    metrics_by_idx: dict[str, dict[str, float]] = {}
    completed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker_batch, batch, str(h5_path)): batch for batch in batches}
        from concurrent.futures import as_completed

        for future in as_completed(futures):
            results = future.result()
            for row_idx, metrics in results:
                metrics_by_idx[row_idx] = metrics
            completed += 1
            if completed % 10 == 0:
                pct = 100 * completed / len(batches)
                elapsed = time.perf_counter() - start
                print(
                    f"[metrics] {completed}/{len(batches)} batches ({pct:.1f}%) {elapsed:.0f}s",
                    file=sys.stderr,
                )

    elapsed = time.perf_counter() - start
    print(
        f"[metrics] Done: {len(metrics_by_idx)} metrics computed in {elapsed:.0f}s", file=sys.stderr
    )

    # Merge metrics into seed rows
    for metric_name in ALL_QUALITY_METRICS:
        if metric_name not in fieldnames:
            fieldnames.append(metric_name)

    for idx, row in enumerate(seed_rows):
        metrics = metrics_by_idx.get(str(idx), {})
        for metric_name in ALL_QUALITY_METRICS:
            row[metric_name] = metrics.get(metric_name, "")

    # Write updated per_seed_detail.csv
    print(f"[metrics] Writing {seed_csv}...", file=sys.stderr)
    with open(seed_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(seed_rows)

    # Aggregate metrics into per_graph_detail.csv
    print(f"[metrics] Updating {graph_csv}...", file=sys.stderr)
    with open(graph_csv) as f:
        reader = csv.DictReader(f)
        graph_rows = list(reader)
        graph_fieldnames = list(reader.fieldnames or [])

    # Build aggregation: group seed metrics by (variant_id, graph_name, side)
    from collections import defaultdict

    grouped: dict[tuple[str, str, str], list[dict[str, float]]] = defaultdict(list)
    for idx, row in enumerate(seed_rows):
        metrics = metrics_by_idx.get(str(idx), {})
        key = (row.get("variant_id", ""), row.get("graph_name", ""), row.get("side", ""))
        grouped[key].append(metrics)

    # Add metric columns to graph_rows
    for metric_name in ALL_QUALITY_METRICS:
        for suffix in ["_orig_mean", "_orig_std", "_reimpl_mean", "_reimpl_std", "_cohens_d"]:
            col = metric_name + suffix
            if col not in graph_fieldnames:
                graph_fieldnames.append(col)

    for grow in graph_rows:
        vid = grow.get("variant_id", "")
        gname = grow.get("graph_name", "")
        orig_metrics = grouped.get((vid, gname, "orig"), [])
        reimpl_metrics = grouped.get((vid, gname, "reimpl"), [])

        for metric_name in ALL_QUALITY_METRICS:
            orig_vals = [
                m.get(metric_name, math.nan)
                for m in orig_metrics
                if not math.isnan(m.get(metric_name, math.nan))
            ]
            reimpl_vals = [
                m.get(metric_name, math.nan)
                for m in reimpl_metrics
                if not math.isnan(m.get(metric_name, math.nan))
            ]

            grow[f"{metric_name}_orig_mean"] = f"{np.mean(orig_vals):.6f}" if orig_vals else ""
            grow[f"{metric_name}_orig_std"] = (
                f"{np.std(orig_vals):.6f}" if len(orig_vals) > 1 else ""
            )
            grow[f"{metric_name}_reimpl_mean"] = (
                f"{np.mean(reimpl_vals):.6f}" if reimpl_vals else ""
            )
            grow[f"{metric_name}_reimpl_std"] = (
                f"{np.std(reimpl_vals):.6f}" if len(reimpl_vals) > 1 else ""
            )

            # Cohen's d
            if orig_vals and reimpl_vals:
                pooled_std = np.sqrt(
                    (
                        (len(orig_vals) - 1) * np.var(orig_vals)
                        + (len(reimpl_vals) - 1) * np.var(reimpl_vals)
                    )
                    / max(len(orig_vals) + len(reimpl_vals) - 2, 1)
                )
                if pooled_std > 0:
                    grow[f"{metric_name}_cohens_d"] = (
                        f"{(np.mean(reimpl_vals) - np.mean(orig_vals)) / pooled_std:.6f}"
                    )
                else:
                    grow[f"{metric_name}_cohens_d"] = "0.0"
            else:
                grow[f"{metric_name}_cohens_d"] = ""

    with open(graph_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=graph_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(graph_rows)

    print("[metrics] All done.", file=sys.stderr)


if __name__ == "__main__":
    main()
