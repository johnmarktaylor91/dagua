#!/usr/bin/env python3
"""Run the r70 definitive distributional fidelity per-combo analysis.

This script is Task B from
``SPEC_definitive_fidelity_analysis.md`` version 6.  It is intentionally only
an incremental per-combo runner: global FDR, final rung assignment, and report
aggregation belong to Task C.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import psutil
import torch

from dagua.eval import distributional_fidelity as df
from dagua.eval.equivalence_metrics import compute_equivalence_metrics
from dagua.eval.graphs import get_test_graphs

SPEC_VERSION = "r70-v6"
DEFAULT_DATA_DIR = Path("eval_output/benchmark_100seed_escalation_final")
DEFAULT_REFRESH_DIR = Path("eval_output/benchmark_5seed_deterministic_refresh")
DEFAULT_OUTPUT = Path("eval_output/fidelity_definitive/per_combo.jsonl")
DEFAULT_CONTROL_DIR = Path("eval_output/fidelity_definitive/controls")
FAILING_MAP_PATH = Path(".project-context/research/sprint_rng_matching/failing_map_final.json")
PROGRESS_EVERY = 25
RSS_WARN_FRACTION = 0.70
RSS_ABORT_FRACTION = 0.85
MIN_MODE_SEEDS = 30
FREE_ASPECT_PREFIX = "classic_sugiyama"
DETERMINISTIC_DIFFERENT_ENGINES = {
    "classic_kk_steps100",
    "classic_kk_steps300",
    "classic_kk_steps1000",
    "classic_rt_horizontal",
    "classic_spectral_default",
    "classic_spectral_nx_fidelity",
    "classic_spectral_random_walk",
    "classic_spectral_unnormalized",
}
TOKEN_SET = (
    "classical_mds",
    "davidson_harel",
    "drl",
    "fa2",
    "fmmm",
    "fr",
    "gem",
    "graphopt",
    "kk",
    "lgl",
    "linlog",
    "maxent_stress",
    "neato",
    "pivot_mds",
    "reingold_tilford",
    "rt",
    "sfdp",
    "sgd2_multi",
    "spectral",
    "stress_maj",
    "stress_sgd",
    "sugiyama",
    "tsnet",
    "umap",
    "neulay",
    "fcose",
)


@dataclass(frozen=True)
class PositionRow:
    """Compact benchmark row payload for worker processes.

    Parameters
    ----------
    key : str
        Original ``results.json`` key.
    graph : str
        Graph name.
    engine : str
        Engine name.
    seed : Optional[int]
        Integer seed for stochastic rows, otherwise ``None``.
    status : str
        Benchmark status.  Only ``ok`` rows are load candidates.
    positions_file : Optional[str]
        Relative or absolute path from the row's ``positions_file`` field.
    runtime_seconds : Optional[float]
        Runtime reported by the benchmark row.
    num_nodes : Optional[int]
        Node count reported by the benchmark row.
    """

    key: str
    graph: str
    engine: str
    seed: Optional[int]
    status: str
    positions_file: Optional[str]
    runtime_seconds: Optional[float]
    num_nodes: Optional[int]


@dataclass(frozen=True)
class ComboPayload:
    """Self-contained work item for one analysis row.

    Parameters
    ----------
    combo_id : str
        Stable combo identifier.
    graph : str
        Graph name.
    engine : str
        Reimplementation engine name.
    reference : str
        Reference engine name.
    data_dir : str
        Benchmark root directory.
    reimpl_rows : tuple[PositionRow, ...]
        Candidate reimplementation rows.
    ref_rows : tuple[PositionRow, ...]
        Candidate reference rows.
    graph_edges : tuple[tuple[int, int], ...]
        Graph edge list for stress calculations.
    graph_n_nodes : int
        Number of graph nodes.
    git_sha : str
        Code git SHA recorded in output.
    chance_permute : bool
        Whether to permute reference seed labels once before Mode A analysis.
    force_mode_b_seed42 : bool
        Whether to truncate the reference cloud to the seed-42 row.
    control_kind : Optional[str]
        Control mode label, when applicable.
    source_combo_id : Optional[str]
        Original combo id for synthetic controls.
    """

    combo_id: str
    graph: str
    engine: str
    reference: str
    data_dir: str
    reimpl_rows: tuple[PositionRow, ...]
    ref_rows: tuple[PositionRow, ...]
    graph_edges: tuple[tuple[int, int], ...]
    graph_n_nodes: int
    git_sha: str
    chance_permute: bool = False
    force_mode_b_seed42: bool = False
    control_kind: Optional[str] = None
    source_combo_id: Optional[str] = None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed runner options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        action="append",
        default=None,
        help="Benchmark root; repeatable -- later dirs override earlier per record key.",
    )
    parser.add_argument("--refresh-dir", type=Path, default=DEFAULT_REFRESH_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--combos-file", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=(
            "full",
            "negative-control",
            "chance-control",
            "modeb-positive-control",
            "deterministic",
            "rung0-reverify",
        ),
        default="full",
    )
    return parser.parse_args()


def main() -> int:
    """Run the requested r70 analysis mode.

    Returns
    -------
    int
        Process exit status.
    """
    args = parse_args()
    configure_thread_environment()
    output_path = default_output_path(args.mode) if args.output is None else args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    git_sha = git_rev_parse()

    if args.mode == "deterministic":
        rows = run_deterministic_mode(args.refresh_dir, output_path, args.combos_file, git_sha)
        print_summary(rows)
        return 0
    if args.mode == "rung0-reverify":
        rows = run_rung0_reverify(args.refresh_dir, output_path, args.combos_file, git_sha)
        print_summary(rows)
        return 0

    failing_map = load_failing_map(FAILING_MAP_PATH)
    data_dirs = args.data_dir or [DEFAULT_DATA_DIR]
    args.data_dir = data_dirs[0]
    results = load_results_multi(data_dirs)
    index = index_results(results)
    graph_data = load_graph_data()
    combo_pairs = load_combo_pairs(failing_map, args.combos_file)
    if args.mode == "negative-control":
        combo_pairs = select_negative_controls(combo_pairs, failing_map, index, args.data_dir)
    elif args.mode == "chance-control":
        combo_pairs = select_chance_controls(combo_pairs, failing_map, index)
    elif args.mode == "modeb-positive-control":
        if args.combos_file is None:
            raise ValueError("--combos-file is required for modeb-positive-control.")

    completed = read_completed(output_path, git_sha) if args.resume else set()
    payloads = build_payloads(
        args.mode,
        combo_pairs,
        failing_map,
        index,
        graph_data,
        args.data_dir,
        git_sha,
    )
    payloads = [payload for payload in payloads if payload.combo_id not in completed]
    run_payloads(payloads, output_path, args.workers)
    rows = read_jsonl(output_path)
    print_summary(rows[-min(10, len(rows)) :])
    return 0


def configure_thread_environment() -> None:
    """Limit native thread fan-out before worker processes start.

    Returns
    -------
    None
        Environment variables are updated in place.
    """
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(name, "1")


def default_output_path(mode: str) -> Path:
    """Return the registered default output path for a mode.

    Parameters
    ----------
    mode : str
        Runner mode.

    Returns
    -------
    pathlib.Path
        Default JSONL output path.
    """
    if mode == "full":
        return DEFAULT_OUTPUT
    return DEFAULT_CONTROL_DIR / f"{mode}.jsonl"


def git_rev_parse() -> str:
    """Return the current git commit SHA.

    Returns
    -------
    str
        ``git rev-parse HEAD`` output, or ``unknown`` when unavailable.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def load_failing_map(path: Path) -> dict[str, dict[str, Any]]:
    """Load the approved failing-map scope.

    Parameters
    ----------
    path : pathlib.Path
        JSON path.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping from engine to reference and graph list.
    """
    with path.open() as file_obj:
        return json.load(file_obj)


def load_results(data_dir: Path) -> dict[str, Any]:
    """Load benchmark ``results.json`` from a data directory.

    Parameters
    ----------
    data_dir : pathlib.Path
        Benchmark root directory.

    Returns
    -------
    dict[str, Any]
        Raw benchmark result mapping.
    """
    with (data_dir / "results.json").open() as file_obj:
        return json.load(file_obj)


def load_results_multi(data_dirs: list[Path]) -> dict[str, Any]:
    """Load and overlay results from multiple benchmark roots.

    Later directories override earlier ones PER RECORD KEY (r71 union-store
    semantics: e.g. post-fix umap rows supersede pre-fix rows without mutating
    either store). positions_file paths are absolutized against their own root
    and rows are tagged with source_dir.

    Parameters
    ----------
    data_dirs : list[pathlib.Path]
        Benchmark roots in precedence order (last wins).

    Returns
    -------
    dict[str, Any]
        Merged result mapping.
    """
    merged: dict[str, Any] = {}
    for data_dir in data_dirs:
        rows = load_results(data_dir)
        for key, row in rows.items():
            if isinstance(row, dict):
                row = dict(row)
                pos = row.get("positions_file")
                if pos and not Path(pos).is_absolute():
                    row["positions_file"] = str((data_dir / pos).resolve())
                row.setdefault("source_dir", data_dir.name)
            merged[key] = row
    return merged


def index_results(results: dict[str, Any]) -> dict[tuple[str, str], list[PositionRow]]:
    """Index raw results by ``(graph, engine)``.

    Parameters
    ----------
    results : dict[str, Any]
        Raw ``results.json`` mapping.

    Returns
    -------
    dict[tuple[str, str], list[PositionRow]]
        Compact rows grouped by graph and engine.
    """
    index: dict[tuple[str, str], list[PositionRow]] = defaultdict(list)
    for key, value in results.items():
        graph = str(value.get("graph_name") or split_key(key)[0])
        engine = str(value.get("engine_name") or split_key(key)[1])
        seed = normalize_seed(value.get("seed"))
        status = str(value.get("status", ""))
        runtime_raw = value.get("runtime_seconds")
        nodes_raw = value.get("num_nodes")
        row = PositionRow(
            key=key,
            graph=graph,
            engine=engine,
            seed=seed,
            status=status,
            positions_file=value.get("positions_file"),
            runtime_seconds=None if runtime_raw is None else float(runtime_raw),
            num_nodes=None if nodes_raw is None else int(nodes_raw),
        )
        index[(graph, engine)].append(row)
    for rows in index.values():
        rows.sort(
            key=lambda item: (-int(item.status == "ok"), seed_sort_value(item.seed), item.key)
        )
    return dict(index)


def split_key(key: str) -> tuple[str, str, str]:
    """Split a benchmark key conservatively.

    Parameters
    ----------
    key : str
        Key of the form ``graph::engine::seedN`` or similar.

    Returns
    -------
    tuple[str, str, str]
        Graph, engine, and seed label components.
    """
    parts = key.split("::")
    if len(parts) < 3:
        return key, "", ""
    return parts[0], parts[1], "::".join(parts[2:])


def normalize_seed(value: Any) -> Optional[int]:
    """Normalize a benchmark seed value.

    Parameters
    ----------
    value : Any
        Raw seed from JSON.

    Returns
    -------
    Optional[int]
        Integer seed, or ``None`` for deterministic/seedless rows.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped in {"", "None", "deterministic"}:
            return None
        if stripped.startswith("seed"):
            stripped = stripped[4:]
        if stripped.lstrip("-").isdigit():
            return int(stripped)
    return None


def seed_sort_value(seed: Optional[int]) -> int:
    """Return a stable sort key for optional seeds.

    Parameters
    ----------
    seed : Optional[int]
        Seed value.

    Returns
    -------
    int
        Sort value with deterministic rows last.
    """
    return 10**12 if seed is None else int(seed)


def load_graph_data() -> dict[str, tuple[int, tuple[tuple[int, int], ...]]]:
    """Load graph node counts and edge lists from the evaluation registry.

    Returns
    -------
    dict[str, tuple[int, tuple[tuple[int, int], ...]]]
        Mapping from graph name to ``(n_nodes, edges)``.
    """
    graph_data: dict[str, tuple[int, tuple[tuple[int, int], ...]]] = {}
    for item in get_test_graphs():
        edge_index = item.graph.edge_index
        if hasattr(edge_index, "detach"):
            edge_array = edge_index.detach().cpu().numpy()
        else:
            edge_array = np.asarray(edge_index)
        if edge_array.ndim == 2 and edge_array.shape[0] == 2:
            edge_array = edge_array.T
        edges = tuple((int(src), int(dst)) for src, dst in np.asarray(edge_array).reshape(-1, 2))
        graph_data[item.name] = (int(item.graph.num_nodes), edges)
    return graph_data


def load_combo_pairs(
    failing_map: dict[str, dict[str, Any]],
    combos_file: Optional[Path],
) -> list[tuple[str, str]]:
    """Load requested ``(graph, engine)`` pairs.

    Parameters
    ----------
    failing_map : dict[str, dict[str, Any]]
        Approved failing-map scope.
    combos_file : Optional[pathlib.Path]
        Optional text file with ``graph::engine`` lines.

    Returns
    -------
    list[tuple[str, str]]
        Sorted combo pairs.
    """
    if combos_file is not None:
        pairs: list[tuple[str, str]] = []
        for line in combos_file.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            graph, engine = stripped.split("::", 1)
            pairs.append((graph, engine))
        return sorted(set(pairs))
    pairs = []
    for engine, entry in failing_map.items():
        pairs.extend((str(graph), engine) for graph in entry["graphs"])
    return sorted(set(pairs))


def build_payloads(
    mode: str,
    combo_pairs: list[tuple[str, str]],
    failing_map: dict[str, dict[str, Any]],
    index: dict[tuple[str, str], list[PositionRow]],
    graph_data: dict[str, tuple[int, tuple[tuple[int, int], ...]]],
    data_dir: Path,
    git_sha: str,
) -> list[ComboPayload]:
    """Build process-pool payloads.

    Parameters
    ----------
    mode : str
        Runner mode.
    combo_pairs : list[tuple[str, str]]
        Requested ``(graph, engine)`` pairs.
    failing_map : dict[str, dict[str, Any]]
        Approved failing-map scope.
    index : dict[tuple[str, str], list[PositionRow]]
        Results index.
    graph_data : dict[str, tuple[int, tuple[tuple[int, int], ...]]]
        Graph edge payloads.
    data_dir : pathlib.Path
        Benchmark root directory.
    git_sha : str
        Code git SHA.

    Returns
    -------
    list[ComboPayload]
        Work items for the parent executor.
    """
    payloads = []
    for graph, engine in combo_pairs:
        source_combo_id = None
        if "\t" in engine:
            engine, reference = engine.split("\t", 1)
            source_combo_id = f"{graph}::{engine}"
            combo_id = f"{graph}::{engine}::NEGREF::{reference}"
        else:
            reference = reference_for_engine(engine, failing_map)
            combo_id = f"{graph}::{engine}"
        graph_n_nodes, edges = graph_data.get(graph, fallback_graph_data(graph, index, engine))
        payloads.append(
            ComboPayload(
                combo_id=combo_id,
                graph=graph,
                engine=engine,
                reference=reference,
                data_dir=str(data_dir),
                reimpl_rows=tuple(index.get((graph, engine), [])),
                ref_rows=tuple(resolve_rows(index, graph, reference, None)),
                graph_edges=edges,
                graph_n_nodes=graph_n_nodes,
                git_sha=git_sha,
                chance_permute=mode == "chance-control",
                force_mode_b_seed42=mode == "modeb-positive-control",
                control_kind=None if mode == "full" else mode,
                source_combo_id=source_combo_id,
            )
        )
    return payloads


def reference_for_engine(engine: str, failing_map: dict[str, dict[str, Any]]) -> str:
    """Return the registered reference engine for an implementation.

    Parameters
    ----------
    engine : str
        Reimplementation engine.
    failing_map : dict[str, dict[str, Any]]
        Approved failing-map scope.

    Returns
    -------
    str
        Reference engine.
    """
    if engine not in failing_map:
        raise KeyError(f"No failing-map reference for {engine!r}.")
    return str(failing_map[engine]["ref"])


def fallback_graph_data(
    graph: str,
    index: dict[tuple[str, str], list[PositionRow]],
    engine: str,
) -> tuple[int, tuple[tuple[int, int], ...]]:
    """Build a minimal graph payload if the registry lacks a graph.

    Parameters
    ----------
    graph : str
        Graph name.
    index : dict[tuple[str, str], list[PositionRow]]
        Results index.
    engine : str
        Engine name.

    Returns
    -------
    tuple[int, tuple[tuple[int, int], ...]]
        Node count and an empty edge list.
    """
    for row in index.get((graph, engine), []):
        if row.num_nodes is not None:
            return row.num_nodes, ()
    return 0, ()


def resolve_rows(
    index: dict[tuple[str, str], list[PositionRow]],
    graph: str,
    engine: str,
    seed: Optional[int],
) -> list[PositionRow]:
    """Resolve rows using the registered fallback key order.

    Parameters
    ----------
    index : dict[tuple[str, str], list[PositionRow]]
        Results index.
    graph : str
        Graph name.
    engine : str
        Engine name.
    seed : Optional[int]
        Preferred seed.

    Returns
    -------
    list[PositionRow]
        Matching rows.  For ``seed is None`` all rows for the pair are returned.
    """
    rows = index.get((graph, engine), [])
    if seed is None:
        return list(rows)
    candidates = [seed, None]
    selected: list[PositionRow] = []
    for candidate in candidates:
        selected = [row for row in rows if row.seed == candidate]
        if selected:
            return selected
    return []


def run_payloads(payloads: list[ComboPayload], output_path: Path, workers: int) -> None:
    """Run payloads in a process pool and append rows as futures finish.

    Parameters
    ----------
    payloads : list[ComboPayload]
        Work items.
    output_path : pathlib.Path
        JSONL output path.
    workers : int
        Maximum worker count.

    Returns
    -------
    None
        Rows and progress heartbeat are written to disk.
    """
    total = len(payloads)
    done = 0
    progress_path = output_path.with_name("progress.json")
    if total == 0:
        write_progress(progress_path, done, total)
        return
    with output_path.open("a") as out_file:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=configure_thread_environment,
        ) as executor:
            futures = [executor.submit(analyze_payload, payload) for payload in payloads]
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                out_file.write(json.dumps(jsonify(row), sort_keys=True) + "\n")
                out_file.flush()
                done += 1
                if done % PROGRESS_EVERY == 0 or done == total:
                    write_progress(progress_path, done, total)
                    warn_parent_rss()
                    print(f"progress {done}/{total}", flush=True)


def analyze_payload(payload: ComboPayload) -> dict[str, Any]:
    """Analyze one process-pool payload.

    Parameters
    ----------
    payload : ComboPayload
        Combo work item.

    Returns
    -------
    dict[str, Any]
        JSON-ready per-combo analysis row.
    """
    abort_if_worker_rss_high()
    rng = combo_rng(payload.graph, payload.engine)
    base = base_row(payload)
    reimpl = collect_layouts(payload.data_dir, payload.reimpl_rows)
    reference = collect_layouts(payload.data_dir, payload.ref_rows)
    if payload.force_mode_b_seed42:
        reference = truncate_reference_to_seed42(reference)
    if payload.chance_permute:
        reference = permute_reference_labels(reference, payload.combo_id)
    mode_info = classify_mode(reimpl, reference)
    base.update(mode_info)
    if mode_info["insufficient_data"]:
        return base
    free_aspect = is_free_aspect(payload.engine)
    if mode_info["mode"] == "A":
        seeds = mode_info["matched_seeds"]
        d_layouts = [reimpl["seeded"][seed] for seed in seeds]
        r_layouts = [reference["seeded"][seed] for seed in seeds]
        analysis = df.analyze_mode_a(d_layouts, r_layouts, rng, free_aspect=free_aspect)
        stress = compute_mode_a_stress(payload, d_layouts, r_layouts)
    else:
        seeds = mode_info["reimpl_seeds"]
        d_layouts = [reimpl["seeded"][seed] for seed in seeds]
        r_layout = reference["deterministic"]
        analysis = df.analyze_mode_b(d_layouts, r_layout, rng, free_aspect=free_aspect)
        stress = compute_mode_b_stress(payload, d_layouts, r_layout)
    base.update(analysis)
    base.update(stress)
    base.update(runtime_ratio(payload.reimpl_rows, payload.ref_rows))
    base["flags"] = collect_flags(base)
    return base


def base_row(payload: ComboPayload) -> dict[str, Any]:
    """Create fields common to every output row.

    Parameters
    ----------
    payload : ComboPayload
        Combo work item.

    Returns
    -------
    dict[str, Any]
        Base JSON row.
    """
    row: dict[str, Any] = {
        "spec_version": SPEC_VERSION,
        "git_sha": payload.git_sha,
        "combo_id": payload.combo_id,
        "graph": payload.graph,
        "engine": payload.engine,
        "reference": payload.reference,
        "free_aspect": is_free_aspect(payload.engine),
        "control_kind": payload.control_kind,
        "source_combo_id": payload.source_combo_id,
    }
    return row


def collect_layouts(data_dir: str, rows: Iterable[PositionRow]) -> dict[Any, Any]:
    """Load valid ok layouts from compact rows.

    Parameters
    ----------
    data_dir : str
        Benchmark root directory.
    rows : Iterable[PositionRow]
        Candidate rows.

    Returns
    -------
    dict[Any, Any]
        ``seed -> ndarray`` for seeded layouts plus optional deterministic layout.
    """
    seeded: dict[int, np.ndarray] = {}
    deterministic: Optional[np.ndarray] = None
    dropped: dict[str, int] = defaultdict(int)
    for row in rows:
        if row.status != "ok":
            dropped[f"status_{row.status}"] += 1
            continue
        if row.positions_file is None:
            dropped["missing_positions_field"] += 1
            continue
        try:
            layout = load_position(Path(data_dir), row.positions_file)
        except (OSError, ValueError, RuntimeError) as exc:
            dropped[f"load_error:{type(exc).__name__}"] += 1
            continue
        if row.seed is None:
            deterministic = layout
        else:
            seeded[row.seed] = layout
    return {"seeded": seeded, "deterministic": deterministic, "dropped": dict(dropped)}


def load_position(data_dir: Path, positions_file: str) -> np.ndarray:
    """Load one ``positions_file`` path from a benchmark row.

    Parameters
    ----------
    data_dir : pathlib.Path
        Benchmark root directory.
    positions_file : str
        Stored row path.  Relative paths are resolved against ``data_dir``.

    Returns
    -------
    numpy.ndarray
        Finite ``float64`` position array shaped ``[N, 2]``.
    """
    path = Path(positions_file)
    if not path.is_absolute():
        path = data_dir / path
    tensor = torch.load(path, map_location="cpu", weights_only=False)
    array = np.asarray(
        tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else tensor,
        dtype=np.float64,
    )
    if array.ndim != 2 or array.shape[1] != 2 or array.size == 0:
        raise ValueError(f"Invalid position shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Position array contains non-finite values.")
    return array


def classify_mode(reimpl: dict[Any, Any], reference: dict[Any, Any]) -> dict[str, Any]:
    """Classify a combo as Mode A, Mode B, or insufficient data.

    Parameters
    ----------
    reimpl : dict[Any, Any]
        Loaded reimplementation layouts.
    reference : dict[Any, Any]
        Loaded reference layouts.

    Returns
    -------
    dict[str, Any]
        Mode, seed sets, and insufficiency reason fields.
    """
    reimpl_seeds = sorted(reimpl["seeded"])
    ref_seeds = sorted(reference["seeded"])
    ref_det = reference["deterministic"] is not None
    matched = sorted(set(reimpl_seeds) & set(ref_seeds))
    result: dict[str, Any] = {
        "n_reimpl_ok": len(reimpl_seeds),
        "n_ref_seeded_ok": len(ref_seeds),
        "has_ref_deterministic": ref_det,
        "n_dropped_reimpl": reimpl["dropped"],
        "n_dropped_ref": reference["dropped"],
        "matched_seeds": matched,
        "reimpl_seeds": reimpl_seeds,
        "insufficient_data": False,
        "insufficient_reason": None,
    }
    if len(ref_seeds) >= MIN_MODE_SEEDS:
        if len(matched) < MIN_MODE_SEEDS:
            result.update(insufficient("matched_seeds_lt_30"))
        else:
            result["mode"] = "A"
            result["n"] = len(matched)
        return result
    if 1 <= len(ref_seeds) < MIN_MODE_SEEDS and not ref_det:
        result.update(insufficient("ref_seeds_lt_30"))
        return result
    if not ref_seeds and not ref_det:
        result.update(insufficient("no_reference_rows"))
        return result
    if len(reimpl_seeds) < MIN_MODE_SEEDS:
        result.update(insufficient("reimpl_seeds_lt_30"))
        return result
    result["mode"] = "B"
    result["n"] = len(reimpl_seeds)
    if 1 <= len(ref_seeds) < MIN_MODE_SEEDS and ref_det:
        result["ref_seeds_lt_30"] = True
        result.setdefault("flags", []).append("ref_seeds_lt_30")
    return result


def insufficient(reason: str) -> dict[str, Any]:
    """Build an insufficient-data marker.

    Parameters
    ----------
    reason : str
        Registered insufficiency reason.

    Returns
    -------
    dict[str, Any]
        Marker fields.
    """
    return {"insufficient_data": True, "insufficient_reason": reason, "mode": "INSUFFICIENT_DATA"}


def truncate_reference_to_seed42(reference: dict[Any, Any]) -> dict[Any, Any]:
    """Treat reference seed 42 as a deterministic single-draw reference.

    Parameters
    ----------
    reference : dict[Any, Any]
        Loaded reference layouts.

    Returns
    -------
    dict[Any, Any]
        Reference layout mapping with ``deterministic`` set.
    """
    layout = reference["seeded"].get(42)
    return {"seeded": {}, "deterministic": layout, "dropped": reference["dropped"]}


def permute_reference_labels(reference: dict[Any, Any], combo_id: str) -> dict[Any, Any]:
    """Permute reference seed labels once for chance-control analysis.

    Parameters
    ----------
    reference : dict[Any, Any]
        Loaded reference layouts.
    combo_id : str
        Stable combo id used for deterministic seeding.

    Returns
    -------
    dict[Any, Any]
        Reference layout mapping with permuted seeded labels.
    """
    seeds = sorted(reference["seeded"])
    if not seeds:
        return reference
    rng = purpose_rng(f"r70::chance::{combo_id}")
    layouts = [reference["seeded"][seed] for seed in seeds]
    perm = rng.permutation(len(seeds))
    return {
        "seeded": {seed: layouts[int(perm[index])] for index, seed in enumerate(seeds)},
        "deterministic": reference["deterministic"],
        "dropped": reference["dropped"],
    }


def compute_mode_a_stress(
    payload: ComboPayload,
    d_layouts: list[np.ndarray],
    r_layouts: list[np.ndarray],
) -> dict[str, Any]:
    """Compute paired stress diagnostics and raw TOST for Mode A.

    Parameters
    ----------
    payload : ComboPayload
        Combo work item.
    d_layouts : list[numpy.ndarray]
        Reimplementation layouts.
    r_layouts : list[numpy.ndarray]
        Reference layouts.

    Returns
    -------
    dict[str, Any]
        Stress values, raw p-values, and direct-branch verdict fields.
    """
    pairs, dists, disconnected = stress_pairs(payload)
    stress_d = np.asarray([df.stress_per_layout(layout, pairs, dists) for layout in d_layouts])
    stress_r = np.asarray([df.stress_per_layout(layout, pairs, dists) for layout in r_layouts])
    margin = max(0.05 * float(np.mean(stress_r)), 1.0e-6)
    tost = df.paired_tost(stress_d - stress_r, margin)
    return stress_record(stress_d, stress_r, margin, tost, disconnected)


def compute_mode_b_stress(
    payload: ComboPayload,
    d_layouts: list[np.ndarray],
    r_layout: np.ndarray,
) -> dict[str, Any]:
    """Compute one-sample stress diagnostics and raw TOST for Mode B.

    Parameters
    ----------
    payload : ComboPayload
        Combo work item.
    d_layouts : list[numpy.ndarray]
        Reimplementation layouts.
    r_layout : numpy.ndarray
        Deterministic reference layout.

    Returns
    -------
    dict[str, Any]
        Stress values, raw p-values, and direct-branch verdict fields.
    """
    pairs, dists, disconnected = stress_pairs(payload)
    stress_d = np.asarray([df.stress_per_layout(layout, pairs, dists) for layout in d_layouts])
    stress_r = float(df.stress_per_layout(r_layout, pairs, dists))
    margin = max(0.05 * stress_r, 1.0e-6)
    tost = df.one_sample_tost(stress_d, stress_r, margin)
    return stress_record(stress_d, np.asarray([stress_r]), margin, tost, disconnected)


def stress_pairs(payload: ComboPayload) -> tuple[np.ndarray, np.ndarray, bool]:
    """Prepare graph distances and registered pair sample for stress.

    Parameters
    ----------
    payload : ComboPayload
        Combo work item.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, bool]
        Pair indices, distance matrix, and disconnected flag.
    """
    edges = np.asarray(payload.graph_edges, dtype=np.int64)
    if edges.size == 0:
        edges = np.empty((0, 2), dtype=np.int64)
    dists = df.prepare_graph_distances(edges, payload.graph_n_nodes)
    pairs = df.sample_pairs(dists, payload.graph)
    disconnected = bool(np.isinf(dists).any())
    return pairs, dists, disconnected


def stress_record(
    stress_d: np.ndarray,
    stress_r: np.ndarray,
    margin: float,
    tost: dict[str, Any],
    disconnected: bool,
) -> dict[str, Any]:
    """Build common stress output fields.

    Parameters
    ----------
    stress_d : numpy.ndarray
        Reimplementation stress values.
    stress_r : numpy.ndarray
        Reference stress values.
    margin : float
        Equivalence margin.
    tost : dict[str, Any]
        TOST result from Task A.
    disconnected : bool
        Whether graph distances include disconnected components.

    Returns
    -------
    dict[str, Any]
        JSON-ready stress fields.
    """
    record = {
        "stress_D_mean": float(np.mean(stress_d)) if stress_d.size else float("nan"),
        "stress_D_sd": float(np.std(stress_d, ddof=1)) if stress_d.size > 1 else 0.0,
        "stress_R_mean": float(np.mean(stress_r)) if stress_r.size else float("nan"),
        "stress_margin": margin,
        "stress_p_tost": tost.get("p_tost"),
        "stress_wilcoxon_p_tost": tost.get("wilcoxon_p_tost"),
        "stress_degenerate_sd": tost.get("degenerate_sd", False),
        "stress_direct_equivalent": tost.get("direct_equivalent"),
        "disconnected": disconnected,
    }
    # SPEC-INTERPRETATION: Task C applies BH to q_tost; raw direct decisions are
    # still emitted so the report can keep degenerate-sd combos outside the BH family.
    record["quality_equivalent_raw"] = bool(
        record["stress_direct_equivalent"]
        if record["stress_degenerate_sd"]
        else (
            math.isfinite(float(record["stress_p_tost"])) and float(record["stress_p_tost"]) < 0.05
        )
    )
    return record


def runtime_ratio(
    reimpl_rows: tuple[PositionRow, ...],
    ref_rows: tuple[PositionRow, ...],
) -> dict[str, Any]:
    """Compute benchmark runtime medians and ratio.

    Parameters
    ----------
    reimpl_rows : tuple[PositionRow, ...]
        Reimplementation rows.
    ref_rows : tuple[PositionRow, ...]
        Reference rows.

    Returns
    -------
    dict[str, Any]
        Runtime median fields.
    """
    d_values = [
        row.runtime_seconds
        for row in reimpl_rows
        if row.status == "ok" and row.runtime_seconds is not None
    ]
    r_values = [
        row.runtime_seconds
        for row in ref_rows
        if row.status == "ok" and row.runtime_seconds is not None
    ]
    d_med = float(np.median(d_values)) if d_values else float("nan")
    r_med = float(np.median(r_values)) if r_values else float("nan")
    return {
        "runtime_D_median": d_med,
        "runtime_R_median": r_med,
        "runtime_ratio": d_med / r_med if r_med > 0.0 else float("nan"),
    }


def collect_flags(row: dict[str, Any]) -> list[str]:
    """Collect prominent boolean output flags.

    Parameters
    ----------
    row : dict[str, Any]
        Output row.

    Returns
    -------
    list[str]
        Sorted flag names.
    """
    flags = set(row.get("flags") or [])
    for key in (
        "near_deterministic",
        "one_sided_degenerate",
        "degenerate_heavy",
        "typicality_uninformative",
        "disconnected",
        "ref_seeds_lt_30",
    ):
        if bool(row.get(key, False)):
            flags.add(key)
    return sorted(flags)


def is_free_aspect(engine: str) -> bool:
    """Return whether an engine uses free-aspect registered distances.

    Parameters
    ----------
    engine : str
        Engine name.

    Returns
    -------
    bool
        ``True`` for classic Sugiyama variants.
    """
    return engine == FREE_ASPECT_PREFIX or engine.startswith(f"{FREE_ASPECT_PREFIX}_")


def combo_rng(graph: str, engine: str) -> np.random.Generator:
    """Return the registered per-combo RNG.

    Parameters
    ----------
    graph : str
        Graph name.
    engine : str
        Engine name.

    Returns
    -------
    numpy.random.Generator
        Deterministically seeded generator.
    """
    return purpose_rng(f"{graph}::{engine}::r70")


def purpose_rng(purpose: str) -> np.random.Generator:
    """Return a SHA-256 seeded RNG for a purpose string.

    Parameters
    ----------
    purpose : str
        Seed purpose string.

    Returns
    -------
    numpy.random.Generator
        Deterministically seeded generator.
    """
    digest = hashlib.sha256(purpose.encode()).digest()
    seed = int.from_bytes(digest[:8], "little")
    return np.random.default_rng(seed)


def read_completed(output_path: Path, git_sha: str) -> set[str]:
    """Read resume-compatible completed combo ids.

    Parameters
    ----------
    output_path : pathlib.Path
        Existing JSONL output.
    git_sha : str
        Current git SHA.

    Returns
    -------
    set[str]
        Combo ids with matching version and SHA rows.
    """
    completed = set()
    for row in read_jsonl(output_path):
        if row.get("spec_version") == SPEC_VERSION and row.get("git_sha") == git_sha:
            completed.add(str(row.get("combo_id")))
    return completed


def read_jsonl(output_path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file if it exists.

    Parameters
    ----------
    output_path : pathlib.Path
        JSONL path.

    Returns
    -------
    list[dict[str, Any]]
        Parsed rows.
    """
    if not output_path.exists():
        return []
    rows = []
    for line in output_path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_progress(progress_path: Path, done: int, total: int) -> None:
    """Write a progress heartbeat JSON file.

    Parameters
    ----------
    progress_path : pathlib.Path
        Heartbeat path.
    done : int
        Completed count.
    total : int
        Total count.

    Returns
    -------
    None
        File is overwritten.
    """
    progress_path.write_text(
        json.dumps({"done": done, "total": total, "ts": time.time()}, sort_keys=True)
    )


def warn_parent_rss() -> None:
    """Print a warning when parent RSS exceeds the registered guard.

    Returns
    -------
    None
        Warning is printed to stderr when needed.
    """
    memory = psutil.virtual_memory()
    process = psutil.Process()
    if process.memory_info().rss > RSS_WARN_FRACTION * memory.total:
        print("WARNING: parent RSS exceeds 70% system RAM", file=sys.stderr, flush=True)


def abort_if_worker_rss_high() -> None:
    """Abort a worker before analysis if RSS exceeds the registered guard.

    Returns
    -------
    None
        Raises ``MemoryError`` when the guard trips.
    """
    memory = psutil.virtual_memory()
    process = psutil.Process()
    if process.memory_info().rss > RSS_ABORT_FRACTION * memory.total:
        raise MemoryError("Worker RSS exceeds 85% system RAM.")


def select_chance_controls(
    combo_pairs: list[tuple[str, str]],
    failing_map: dict[str, dict[str, Any]],
    index: dict[tuple[str, str], list[PositionRow]],
) -> list[tuple[str, str]]:
    """Draw 20 real Mode-A-eligible combos for chance controls.

    Parameters
    ----------
    combo_pairs : list[tuple[str, str]]
        Candidate combo pairs.
    failing_map : dict[str, dict[str, Any]]
        Approved failing-map scope.
    index : dict[tuple[str, str], list[PositionRow]]
        Results index.

    Returns
    -------
    list[tuple[str, str]]
        Deterministically drawn controls.
    """
    eligible = []
    for graph, engine in combo_pairs:
        reference = reference_for_engine(engine, failing_map)
        d_seeds = ok_seed_set(index.get((graph, engine), []))
        r_seeds = ok_seed_set(index.get((graph, reference), []))
        if len(d_seeds & r_seeds) >= MIN_MODE_SEEDS:
            eligible.append((graph, engine))
    return draw_sorted(eligible, 20, "r70::chance")


def select_negative_controls(
    combo_pairs: list[tuple[str, str]],
    failing_map: dict[str, dict[str, Any]],
    index: dict[tuple[str, str], list[PositionRow]],
    data_dir: Path,
) -> list[tuple[str, str]]:
    """Draw registered negative-control mispairs.

    Parameters
    ----------
    combo_pairs : list[tuple[str, str]]
        Candidate real combos.
    failing_map : dict[str, dict[str, Any]]
        Approved failing-map scope.
    index : dict[tuple[str, str], list[PositionRow]]
        Results index.
    data_dir : pathlib.Path
        Benchmark root directory.

    Returns
    -------
    list[tuple[str, str]]
        Synthetic pairs encoded as ``(graph, "engine\tref")``.
    """
    token_map = {engine: algorithm_token(engine) for engine in sorted(failing_map)}
    controls_dir = DEFAULT_CONTROL_DIR
    controls_dir.mkdir(parents=True, exist_ok=True)
    (controls_dir / "token_map.json").write_text(json.dumps(token_map, indent=2, sort_keys=True))
    print(json.dumps(token_map, indent=2, sort_keys=True))
    candidates: list[tuple[str, str]] = []
    by_graph: dict[str, list[str]] = defaultdict(list)
    for graph, engine in combo_pairs:
        by_graph[graph].append(engine)
    for graph, engines in by_graph.items():
        for engine in sorted(engines):
            for other in sorted(engines):
                if engine == other:
                    continue
                ref = reference_for_engine(engine, failing_map)
                other_ref = reference_for_engine(other, failing_map)
                if token_map[engine] == token_map[other]:
                    continue
                if ref_base(ref) == ref_base(other_ref):
                    continue
                if not reimpl_cloud_informative(index, data_dir, graph, engine):
                    continue
                if reference_distance_ok(index, data_dir, graph, ref, other_ref):
                    candidates.append((graph, f"{engine}\t{other_ref}"))
    return draw_sorted(candidates, 20, "r70::negctl")


def ok_seed_set(rows: Iterable[PositionRow]) -> set[int]:
    """Return ok integer seeds for rows.

    Parameters
    ----------
    rows : Iterable[PositionRow]
        Candidate rows.

    Returns
    -------
    set[int]
        Seeds with ok rows.
    """
    return {int(row.seed) for row in rows if row.status == "ok" and row.seed is not None}


def draw_sorted(items: list[tuple[str, str]], count: int, purpose: str) -> list[tuple[str, str]]:
    """Draw a deterministic sample from sorted candidates.

    Parameters
    ----------
    items : list[tuple[str, str]]
        Candidate items.
    count : int
        Requested sample size.
    purpose : str
        Purpose string for seeding.

    Returns
    -------
    list[tuple[str, str]]
        Drawn items in sorted order.
    """
    unique = sorted(set(items))
    if len(unique) < count:
        raise ValueError(f"Need {count} candidates for {purpose}, found {len(unique)}.")
    rng = purpose_rng(purpose)
    indices = sorted(rng.choice(len(unique), size=count, replace=False).tolist())
    return [unique[int(index)] for index in indices]


def algorithm_token(engine: str) -> str:
    """Map an engine to its pre-registered algorithm token.

    Parameters
    ----------
    engine : str
        Engine name.

    Returns
    -------
    str
        Longest matching token after stripping ``classic_``.
    """
    stripped = engine.removeprefix("classic_")
    matches = [
        token for token in TOKEN_SET if stripped == token or stripped.startswith(f"{token}_")
    ]
    if not matches:
        raise ValueError(f"No registered algorithm token for {engine}.")
    return max(matches, key=len)


def ref_base(reference: str) -> str:
    """Return the base reference engine before ``__for__``.

    Parameters
    ----------
    reference : str
        Reference engine name.

    Returns
    -------
    str
        Base reference name.
    """
    return reference.split("__for__", 1)[0]


def reimpl_cloud_informative(
    index: dict[tuple[str, str], list[PositionRow]],
    data_dir: Path,
    graph: str,
    engine: str,
) -> bool:
    """Pre-screen negative controls by mean plain within distance.

    Parameters
    ----------
    index : dict[tuple[str, str], list[PositionRow]]
        Results index.
    data_dir : pathlib.Path
        Benchmark root directory.
    graph : str
        Graph name.
    engine : str
        Engine name.

    Returns
    -------
    bool
        ``True`` when mean plain ``W_D <= 1.0``.
    """
    layouts = []
    sorted_rows = sorted(
        index.get((graph, engine), []),
        key=lambda item: seed_sort_value(item.seed),
    )
    for row in sorted_rows[:30]:
        if row.status == "ok" and row.positions_file is not None:
            try:
                layouts.append(load_position(data_dir, row.positions_file))
            except (OSError, ValueError, RuntimeError):
                pass
    if len(layouts) < 2:
        return False
    matrix = df.pairwise_procrustes_matrix(layouts, free_aspect=False)
    return offdiag_mean(matrix) <= 1.0


def reference_distance_ok(
    index: dict[tuple[str, str], list[PositionRow]],
    data_dir: Path,
    graph: str,
    first_ref: str,
    second_ref: str,
) -> bool:
    """Check negative-control reference layouts differ by Procrustes distance.

    Parameters
    ----------
    index : dict[tuple[str, str], list[PositionRow]]
        Results index.
    data_dir : pathlib.Path
        Benchmark root directory.
    graph : str
        Graph name.
    first_ref : str
        First reference engine.
    second_ref : str
        Second reference engine.

    Returns
    -------
    bool
        ``True`` when the selected reference layouts differ by more than ``0.1``.
    """
    first = load_reference_draw(index, data_dir, graph, first_ref)
    second = load_reference_draw(index, data_dir, graph, second_ref)
    if first is None or second is None:
        return False
    matrix = df.pairwise_procrustes_matrix([first, second], free_aspect=False)
    return float(matrix[0, 1]) > 0.1


def load_reference_draw(
    index: dict[tuple[str, str], list[PositionRow]],
    data_dir: Path,
    graph: str,
    reference: str,
) -> Optional[np.ndarray]:
    """Load the deterministic or seed-42 reference layout for controls.

    Parameters
    ----------
    index : dict[tuple[str, str], list[PositionRow]]
        Results index.
    data_dir : pathlib.Path
        Benchmark root directory.
    graph : str
        Graph name.
    reference : str
        Reference engine.

    Returns
    -------
    Optional[numpy.ndarray]
        Loaded layout when available.
    """
    rows = index.get((graph, reference), [])
    preferred = [row for row in rows if row.seed is None] or [row for row in rows if row.seed == 42]
    for row in preferred:
        if row.status == "ok" and row.positions_file is not None:
            try:
                return load_position(data_dir, row.positions_file)
            except (OSError, ValueError, RuntimeError):
                return None
    return None


def offdiag_mean(matrix: np.ndarray) -> float:
    """Return a square matrix off-diagonal mean.

    Parameters
    ----------
    matrix : numpy.ndarray
        Square numeric matrix.

    Returns
    -------
    float
        Off-diagonal mean.
    """
    if matrix.shape[0] < 2:
        return 0.0
    return float((np.sum(matrix) - np.trace(matrix)) / (matrix.shape[0] * (matrix.shape[0] - 1)))


def run_deterministic_mode(
    refresh_dir: Path,
    output_path: Path,
    combos_file: Optional[Path],
    git_sha: str,
) -> list[dict[str, Any]]:
    """Run deterministic-different refresh analysis.

    Parameters
    ----------
    refresh_dir : pathlib.Path
        Refresh benchmark root.
    output_path : pathlib.Path
        JSONL output.
    combos_file : Optional[pathlib.Path]
        Optional combo restriction.
    git_sha : str
        Code git SHA.

    Returns
    -------
    list[dict[str, Any]]
        Rows written.
    """
    index = index_results(load_results(refresh_dir))
    graph_data = load_graph_data()
    # The 8 DETERMINISTIC_DIFFERENT engines are NOT in the failing map (that is what made
    # them deterministic-different) -- enumerate their combos from the refresh data itself,
    # and resolve each reference as the co-benchmarked "<ref>__for__<engine>" entry.
    ref_by_engine: dict[str, str] = {}
    for _graph, idx_engine in index:
        if "__for__" in idx_engine:
            target = idx_engine.split("__for__", 1)[1]
            if target in DETERMINISTIC_DIFFERENT_ENGINES:
                ref_by_engine.setdefault(target, idx_engine)
    pairs = sorted(
        (graph, engine) for graph, engine in index if engine in DETERMINISTIC_DIFFERENT_ENGINES
    )
    if combos_file is not None:
        requested = set(load_combo_pairs({}, combos_file))
        pairs = [pair for pair in pairs if pair in requested]
    done: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    if output_path.exists():
        for line in output_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("spec_version") == SPEC_VERSION and not row.get("toolkit_timeout"):
                done.add((row["graph"], row["engine"]))
                rows.append(row)
    work = [
        (graph, engine, ref_by_engine[engine])
        for graph, engine in pairs
        if engine in ref_by_engine and (graph, engine) not in done
    ]
    print(f"[deterministic] resume: {len(done)} done, {len(work)} to run", flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(
                deterministic_row, refresh_dir, index, graph_data, graph, engine, ref, git_sha
            ): (graph, engine)
            for graph, engine, ref in work
        }
        # Rewrite kept rows first so recomputed timeout rows do not duplicate.
        with output_path.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            for future in concurrent.futures.as_completed(futures):
                graph, engine = futures[future]
                try:
                    row = future.result()
                except Exception as exc:  # pragma: no cover - defensive
                    row = {
                        "spec_version": SPEC_VERSION,
                        "git_sha": git_sha,
                        "combo_id": f"{graph}::{engine}",
                        "graph": graph,
                        "engine": engine,
                        "mode": "deterministic",
                        "error": str(exc)[:300],
                    }
                rows.append(row)
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                if len(rows) % PROGRESS_EVERY == 0:
                    print(f"[deterministic] {len(rows)}/{len(work)}", flush=True)
    return rows


def deterministic_row(
    refresh_dir: Path,
    index: dict[tuple[str, str], list[PositionRow]],
    graph_data: dict[str, tuple[int, tuple[tuple[int, int], ...]]],
    graph: str,
    engine: str,
    reference: str,
    git_sha: str,
) -> dict[str, Any]:
    """Analyze one deterministic refresh combo.

    Parameters
    ----------
    refresh_dir : pathlib.Path
        Refresh benchmark root.
    index : dict[tuple[str, str], list[PositionRow]]
        Results index.
    graph_data : dict[str, tuple[int, tuple[tuple[int, int], ...]]]
        Graph data.
    graph : str
        Graph name.
    engine : str
        Engine name.
    reference : str
        Reference name.
    git_sha : str
        Code git SHA.

    Returns
    -------
    dict[str, Any]
        Deterministic row.
    """
    d_layout = load_reference_draw(index, refresh_dir, graph, engine)
    r_layout = load_reference_draw(index, refresh_dir, graph, reference)
    base = {
        "spec_version": SPEC_VERSION,
        "git_sha": git_sha,
        "combo_id": f"{graph}::{engine}",
        "graph": graph,
        "engine": engine,
        "reference": reference,
        "mode": "deterministic",
    }
    if d_layout is None or r_layout is None:
        base.update(insufficient("no_reference_rows" if r_layout is None else "reimpl_seeds_lt_30"))
        return base
    n_nodes, edges = graph_data.get(graph, (d_layout.shape[0], ()))
    edge_array = np.asarray(edges, dtype=np.int64)
    # The automorphism SEARCH (single igraph/BLISS C call) is intractable on twin-heavy
    # graphs (chung_lu_150 measured >490s) and cannot be interrupted by signals -- run the
    # toolkit in a hard-killed subprocess. On timeout, plain Procrustes is a sound
    # CONSERVATIVE substitute: toolkit_distance = min(aligned variants) <= plain, so
    # plain < 1e-3 still proves INVARIANCE_EQUIVALENT; plain >= 1e-3 is flagged
    # toolkit_timeout and falls through to the quality axis.
    budget = float(os.environ.get("DAGUA_R70_TOOLKIT_BUDGET_S", "90"))
    metrics = toolkit_metrics_with_timeout(d_layout, r_layout, edge_array, engine, budget)
    plain_distance = float(df.pairwise_procrustes_matrix([d_layout, r_layout])[0, 1])
    toolkit_timeout = metrics is None
    if metrics is None:
        aut_rmsd: Optional[float] = None
        component_rmsd: Optional[float] = None
        anisotropic_rmsd: Optional[float] = None
        toolkit_distance = plain_distance
    else:
        aut_rmsd = metrics.aut_procrustes_rmsd
        component_rmsd = metrics.component_aligned_rmsd
        anisotropic_rmsd = metrics.anisotropic_rmsd
        toolkit_values = [aut_rmsd, component_rmsd]
        if is_free_aspect(engine) and anisotropic_rmsd is not None:
            toolkit_values.append(anisotropic_rmsd)
        toolkit_distance = float(min(v for v in toolkit_values if v is not None))
    dists = df.prepare_graph_distances(edge_array.reshape(-1, 2), n_nodes)
    pairs = df.sample_pairs(dists, graph)
    stress_d = df.stress_per_layout(d_layout, pairs, dists)
    stress_r = df.stress_per_layout(r_layout, pairs, dists)
    stress_margin = 0.05 * stress_r
    base.update(
        {
            "toolkit_distance": toolkit_distance,
            "plain_distance": plain_distance,
            "toolkit_timeout": toolkit_timeout,
            "aut_procrustes_rmsd": aut_rmsd,
            "component_aligned_rmsd": component_rmsd,
            "anisotropic_rmsd": anisotropic_rmsd,
            "stress_D": stress_d,
            "stress_R": stress_r,
            "stress_delta_abs": abs(stress_d - stress_r),
            "stress_margin": stress_margin,
            "deterministic_verdict": deterministic_verdict(toolkit_distance, stress_d, stress_r),
        }
    )
    return base


def _toolkit_child(conn, d_layout, r_layout, edge_array, engine: str) -> None:
    """Compute toolkit metrics in a killable child process."""
    try:
        metrics = compute_equivalence_metrics(d_layout, r_layout, edge_array, engine_name=engine)
        conn.send(
            (
                metrics.aut_procrustes_rmsd,
                metrics.component_aligned_rmsd,
                metrics.anisotropic_rmsd,
            )
        )
    except Exception:  # pragma: no cover - defensive
        conn.send(None)
    finally:
        conn.close()


def toolkit_metrics_with_timeout(d_layout, r_layout, edge_array, engine: str, budget_s: float):
    """Run compute_equivalence_metrics with a hard subprocess kill at ``budget_s``.

    Returns a small namespace-like object or None on timeout/failure.
    """
    import multiprocessing as mp
    from types import SimpleNamespace

    ctx = mp.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_toolkit_child, args=(child_conn, d_layout, r_layout, edge_array, engine)
    )
    proc.start()
    child_conn.close()
    result = None
    if parent_conn.poll(budget_s):
        try:
            result = parent_conn.recv()
        except EOFError:
            result = None
    proc.join(timeout=1.0)
    if proc.is_alive():
        proc.kill()
        proc.join()
    parent_conn.close()
    if result is None:
        return None
    return SimpleNamespace(
        aut_procrustes_rmsd=result[0],
        component_aligned_rmsd=result[1],
        anisotropic_rmsd=result[2],
    )


def deterministic_verdict(toolkit_distance: float, stress_d: float, stress_r: float) -> str:
    """Assign the deterministic-mode local verdict.

    Parameters
    ----------
    toolkit_distance : float
        Minimum allowed toolkit invariance distance.
    stress_d : float
        Reimplementation stress.
    stress_r : float
        Reference stress.

    Returns
    -------
    str
        ``INVARIANCE_EQUIVALENT``, ``QUALITY_EQUIVALENT``, or ``DIFFERENT``.
    """
    if toolkit_distance < 1.0e-3:
        return "INVARIANCE_EQUIVALENT"
    if abs(stress_d - stress_r) <= 0.05 * stress_r:
        return "QUALITY_EQUIVALENT"
    return "DIFFERENT"


def run_rung0_reverify(
    refresh_dir: Path,
    output_path: Path,
    combos_file: Optional[Path],
    git_sha: str,
) -> list[dict[str, Any]]:
    """Reverify non-failing-map Sugiyama rung-0 combos in refresh data.

    Parameters
    ----------
    refresh_dir : pathlib.Path
        Refresh benchmark root.
    output_path : pathlib.Path
        JSONL output.
    combos_file : Optional[pathlib.Path]
        Optional combo restriction.
    git_sha : str
        Code git SHA.

    Returns
    -------
    list[dict[str, Any]]
        Rows written.
    """
    failing_map = load_failing_map(FAILING_MAP_PATH)
    index = index_results(load_results(refresh_dir))
    all_pairs = sorted((graph, engine) for graph, engine in index if is_free_aspect(engine))
    failing_pairs = set(load_combo_pairs(failing_map, None))
    requested = set(load_combo_pairs(failing_map, combos_file)) if combos_file else None
    rows = []
    for graph, engine in all_pairs:
        if (graph, engine) in failing_pairs:
            continue
        if requested is not None and (graph, engine) not in requested:
            continue
        reference = failing_map.get(engine, {}).get("ref", "igraph_sugiyama__for__" + engine)
        rows.append(rung0_row(refresh_dir, index, graph, engine, str(reference), git_sha))
    write_rows(output_path, rows)
    return rows


def rung0_row(
    refresh_dir: Path,
    index: dict[tuple[str, str], list[PositionRow]],
    graph: str,
    engine: str,
    reference: str,
    git_sha: str,
) -> dict[str, Any]:
    """Compute max per-seed Procrustes RMSD for one rung-0 reverify combo.

    Parameters
    ----------
    refresh_dir : pathlib.Path
        Refresh benchmark root.
    index : dict[tuple[str, str], list[PositionRow]]
        Results index.
    graph : str
        Graph name.
    engine : str
        Engine name.
    reference : str
        Reference name.
    git_sha : str
        Code git SHA.

    Returns
    -------
    dict[str, Any]
        Reverification row.
    """
    d_rows = {
        row.seed: row
        for row in index.get((graph, engine), [])
        if row.status == "ok" and row.seed is not None
    }
    r_rows = {
        row.seed: row
        for row in index.get((graph, reference), [])
        if row.status == "ok" and row.seed is not None
    }
    distances = []
    if r_rows:
        for seed in sorted(set(d_rows) & set(r_rows)):
            if d_rows[seed].positions_file is None or r_rows[seed].positions_file is None:
                continue
            d_layout = load_position(refresh_dir, d_rows[seed].positions_file or "")
            r_layout = load_position(refresh_dir, r_rows[seed].positions_file or "")
            distances.append(float(df.pairwise_procrustes_matrix([d_layout, r_layout])[0, 1]))
    else:
        # Deterministic reference (e.g. igraph_sugiyama): a single seed-None row. Compare
        # every seeded reimpl layout against that one layout -- the same pairing the 5-seed
        # triage used for these combos.
        det_rows = [
            row
            for row in index.get((graph, reference), [])
            if row.status == "ok" and row.seed is None and row.positions_file
        ]
        if det_rows:
            r_layout = load_position(refresh_dir, det_rows[0].positions_file or "")
            for seed in sorted(d_rows):
                if d_rows[seed].positions_file is None:
                    continue
                d_layout = load_position(refresh_dir, d_rows[seed].positions_file or "")
                distances.append(float(df.pairwise_procrustes_matrix([d_layout, r_layout])[0, 1]))
    max_rmsd = max(distances) if distances else float("nan")
    return {
        "spec_version": SPEC_VERSION,
        "git_sha": git_sha,
        "combo_id": f"{graph}::{engine}",
        "graph": graph,
        "engine": engine,
        "reference": reference,
        "mode": "rung0-reverify",
        "n": len(distances),
        "max_rmsd": max_rmsd,
        "still_bit_exact": bool(math.isfinite(max_rmsd) and max_rmsd < 1.0e-3),
    }


def write_rows(output_path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL rows.

    Parameters
    ----------
    output_path : pathlib.Path
        Output path.
    rows : list[dict[str, Any]]
        Rows to write.

    Returns
    -------
    None
        File is overwritten.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(jsonify(row), sort_keys=True) + "\n")


def jsonify(value: Any) -> Any:
    """Convert NumPy/scalar values to JSON-compatible objects.

    Parameters
    ----------
    value : Any
        Arbitrary value.

    Returns
    -------
    Any
        JSON-compatible value.
    """
    if isinstance(value, dict):
        return {str(key): jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonify(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def print_summary(rows: list[dict[str, Any]]) -> None:
    """Print a concise row summary for smoke verification.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Rows to summarize.

    Returns
    -------
    None
        Summary is printed to stdout.
    """
    print("row summary:")
    for row in rows:
        fields = {
            "combo_id": row.get("combo_id"),
            "mode": row.get("mode"),
            "n": row.get("n"),
            "insufficient_reason": row.get("insufficient_reason"),
            "mean_W_D": row.get("mean_W_D"),
            "mean_W_R": row.get("mean_W_R"),
            "d_R": row.get("d_R"),
            "p_typ": row.get("p_typ"),
            "p_track": row.get("p_track"),
            "stress_p_tost": row.get("stress_p_tost"),
            "flags": row.get("flags"),
        }
        print(json.dumps(jsonify(fields), sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
