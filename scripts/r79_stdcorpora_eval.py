"""Evaluate Dagua on held-out standard graph-drawing corpora."""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from dagua.eval.competitors import get_competitor  # noqa: E402
from dagua.eval.competitors.base import CompetitorBase, CompetitorResult  # noqa: E402
from dagua.graph import DaguaGraph  # noqa: E402
from dagua.metrics import composite_auto, evaluate  # noqa: E402

# Loader extraction (GLADOS_RUNNER_SPEC.md section 3): bodies moved verbatim to
# scripts/stdcorpora_loaders.py; every name is re-imported here so this module's
# public surface (pinned by tests/test_stdcorpora_eval.py) is unchanged. The
# F401 suppressions are load-bearing: these names ARE the re-export contract.
from scripts.stdcorpora_loaders import (  # noqa: E402
    MAX_NODES,
    LoadedGraph,
    _load_mtx_coordinate_fallback,  # noqa: F401  (re-export)
    _load_mtx_with_scipy,  # noqa: F401  (re-export)
    _numeric_lines,  # noqa: F401  (re-export)
    build_graph,  # noqa: F401  (re-export)
    infer_corpus,  # noqa: F401  (re-export)
    infer_directed,  # noqa: F401  (re-export)
    load_corpus,
    load_gml_file,  # noqa: F401  (re-export)
    load_graph_file,  # noqa: F401  (re-export)
    load_graphml_file,  # noqa: F401  (re-export)
    load_mtx_file,  # noqa: F401  (re-export)
)

OUTPUT_DIR = Path("eval_output/stdcorpora")
CORPUS_DIR = Path("eval_output/stdcorpora")
SEED = 42
TIMEOUT_SECONDS = 120.0
TIE_BAND = 0.5
REPORT_METRICS = (
    "sampled_stress",
    "neighborhood_preservation",
    "crossing_rate",
    "edge_length_cv",
)
CORPUS_NAMES = ("rome", "north", "suitesparse", "misc")
# Memory guard: the harness runs hundreds of in-process "dagua" layouts plus
# thousands of forked external-engine subprocesses. r80 holdout run r80_holdout
# was OOM-killed at anon-rss ~101GB (dmesg: pid=344661, total-vm ~1.86TB,
# pgtables ~3.4GB -- evidence of leaked/fragmented native allocations, not a
# Python-level container growing per row). GC_TRIM_INTERVAL_ROWS forces a full
# gc pass + libc malloc_trim(0) periodically to return freed heap pages to the
# OS. RSS_WARN_BYTES/RSS_ABORT_BYTES are a hard backstop so a still-unknown or
# host-specific leak can never again silently exhaust the box.
GC_TRIM_INTERVAL_ROWS = 10
RSS_WARN_BYTES = 16 * 1024**3
RSS_ABORT_BYTES = 32 * 1024**3
ENGINE_NAMES = [
    "dagua",
    "graphviz_dot",
    "graphviz_sfdp",
    "graphviz_neato",
    "elk_layered",
    "dagre",
    "nx_spring",
    "igraph_kamada_kawai",
    "igraph_sugiyama",
]
EXTERNAL_ENGINE_NAMES = [name for name in ENGINE_NAMES if name != "dagua"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-nodes", type=int, default=MAX_NODES)
    parser.add_argument("--dagua-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--engines", nargs="+", default=None)
    parser.add_argument("--graphs", nargs="+", default=None)
    parser.add_argument(
        "--corpus",
        choices=CORPUS_NAMES,
        default=None,
        help="Restrict the run to one reporting corpus (rome, north, suitesparse, misc).",
    )
    return parser.parse_args()


def git_sha() -> str:
    """Return the current git commit SHA.

    Returns
    -------
    str
        Current commit SHA, or ``unknown`` if git cannot provide one.
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


def safe_component(value: str) -> str:
    """Return a filesystem-safe artifact name component.

    Parameters
    ----------
    value : str
        Raw graph or engine name.

    Returns
    -------
    str
        Sanitized path component.
    """
    return "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in value)


def position_relpath(graph_name: str, engine_name: str) -> str:
    """Return a relative position tensor path.

    Parameters
    ----------
    graph_name : str
        Corpus graph name.
    engine_name : str
        Layout engine name.

    Returns
    -------
    str
        POSIX-style path below the output directory.
    """
    filename = f"{safe_component(graph_name)}__{safe_component(engine_name)}.pt"
    return str(Path("positions") / filename)


def json_clean(value: Any) -> Any:
    """Convert nested metric values to JSON-safe primitives.

    Parameters
    ----------
    value : Any
        Raw metric, tensor, collection, or scalar.

    Returns
    -------
    Any
        JSON-serializable value.
    """
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return json_clean(value.item())
        return [json_clean(item) for item in value.detach().cpu().flatten().tolist()]
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return numeric if math.isfinite(numeric) else None


def current_rss_bytes() -> Optional[int]:
    """Return the current process resident set size in bytes.

    Returns
    -------
    int | None
        RSS in bytes, or ``None`` when ``psutil`` is unavailable.
    """
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.Process(os.getpid()).memory_info().rss)


def release_native_heap() -> None:
    """Force a full GC pass and return freed heap pages to the OS.

    Called periodically from the row loop to counter native-allocator
    fragmentation from thousands of short-lived per-row tensors and forked
    subprocess bookkeeping. Cheap relative to one layout call; safe to call
    every row if ever needed.

    Returns
    -------
    None
    """
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (AttributeError, OSError):
        pass


def load_existing_results(output_dir: Path) -> Dict[str, Any]:
    """Load an existing result payload.

    Parameters
    ----------
    output_dir : Path
        Harness output directory.

    Returns
    -------
    Dict[str, Any]
        Parsed ``results.json`` payload.
    """
    with (output_dir / "results.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_results(output_dir: Path, payload: Dict[str, Any]) -> None:
    """Write the result payload.

    Parameters
    ----------
    output_dir : Path
        Harness output directory.
    payload : Dict[str, Any]
        JSON-serializable payload.

    Returns
    -------
    None
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def rows_path(output_dir: Path) -> Path:
    """Return the resumable row store path.

    Parameters
    ----------
    output_dir : Path
        Harness output directory.

    Returns
    -------
    Path
        JSONL row path.
    """
    return output_dir / "results.rows.jsonl"


def append_row(output_dir: Path, row: Dict[str, Any]) -> None:
    """Append one completed result row.

    Parameters
    ----------
    output_dir : Path
        Harness output directory.
    row : Dict[str, Any]
        Completed result row.

    Returns
    -------
    None
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with rows_path(output_dir).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True))
        handle.write("\n")


def load_jsonl_rows(output_dir: Path) -> List[Dict[str, Any]]:
    """Load resumable rows from disk.

    Parameters
    ----------
    output_dir : Path
        Harness output directory.

    Returns
    -------
    List[Dict[str, Any]]
        Completed rows.
    """
    path = rows_path(output_dir)
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_jsonl_rows(output_dir: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Rewrite the resumable row store.

    Parameters
    ----------
    output_dir : Path
        Harness output directory.
    rows : Sequence[Dict[str, Any]]
        Rows to write.

    Returns
    -------
    None
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with rows_path(output_dir).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def staging_dir(output_dir: Path) -> Path:
    """Return the staging directory used before atomic publish.

    Parameters
    ----------
    output_dir : Path
        Final output directory.

    Returns
    -------
    Path
        Sibling staging directory.
    """
    return output_dir.with_name(f"{output_dir.name}.tmp")


def previous_dir(output_dir: Path) -> Path:
    """Return the temporary backup directory used during publish.

    Parameters
    ----------
    output_dir : Path
        Final output directory.

    Returns
    -------
    Path
        Sibling previous-output directory.
    """
    return output_dir.with_name(f"{output_dir.name}.prev")


def _write_worker_message(result_path: str, message: Dict[str, Any]) -> None:
    """Write a child-process result message.

    Parameters
    ----------
    result_path : str
        JSON path used by the child process.
    message : Dict[str, Any]
        Serializable result metadata.

    Returns
    -------
    None
    """
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(message, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _external_layout_worker(
    graph: DaguaGraph,
    engine_name: str,
    temp_position_path: str,
    result_path: str,
) -> None:
    """Run one external layout in an isolated child process.

    Parameters
    ----------
    graph : DaguaGraph
        Corpus graph to lay out.
    engine_name : str
        Registered competitor name.
    temp_position_path : str
        Temporary position tensor path.
    result_path : str
        Child status JSON path.

    Returns
    -------
    None
    """
    try:
        competitor = get_competitor(engine_name)
        if competitor is None:
            _write_worker_message(
                result_path,
                {"status": "ERROR", "runtime_s": 0.0, "error": "adapter not registered"},
            )
            os._exit(0)
        result = competitor.layout(graph, timeout=TIMEOUT_SECONDS, seed=SEED)
        if result.pos is None:
            _write_worker_message(
                result_path,
                {
                    "status": "ERROR",
                    "runtime_s": float(result.runtime_seconds),
                    "error": result.error or "adapter returned no positions",
                },
            )
            os._exit(0)
        torch.save(result.pos.detach().cpu().to(dtype=torch.float32), temp_position_path)
        _write_worker_message(
            result_path,
            {
                "status": "OK",
                "runtime_s": float(result.runtime_seconds),
                "error": result.error,
                "temp_positions_path": temp_position_path,
            },
        )
        os._exit(0)
    except BaseException:
        os._exit(1)


def make_skip_row(graph: LoadedGraph, engine_name: str, reason: str) -> Dict[str, Any]:
    """Create a skipped result row.

    Parameters
    ----------
    graph : LoadedGraph
        Corpus graph metadata.
    engine_name : str
        Engine name.
    reason : str
        Skip reason.

    Returns
    -------
    Dict[str, Any]
        Result row.
    """
    return {
        "graph": graph.name,
        "corpus": graph.corpus,
        "engine": engine_name,
        "status": "SKIP",
        "runtime_s": 0.0,
        "metrics": {},
        "reported_metrics": {},
        "composite": None,
        "positions_path": None,
        "nodes": graph.graph.num_nodes,
        "edges": int(graph.graph.edge_index.shape[1]),
        "directed": graph.directed,
        "source_path": str(graph.source_path),
        "reason": reason,
    }


def make_error_row(
    graph: LoadedGraph,
    engine_name: str,
    runtime_s: float,
    error: str,
    status_detail: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a failed result row.

    Parameters
    ----------
    graph : LoadedGraph
        Corpus graph metadata.
    engine_name : str
        Engine name.
    runtime_s : float
        Runtime before failure was observed.
    error : str
        Error detail.
    status_detail : str | None, default=None
        Optional machine-readable failure detail.

    Returns
    -------
    Dict[str, Any]
        Result row.
    """
    row = {
        "graph": graph.name,
        "corpus": graph.corpus,
        "engine": engine_name,
        "status": "ERROR",
        "runtime_s": float(runtime_s),
        "metrics": {},
        "reported_metrics": {},
        "composite": None,
        "positions_path": None,
        "nodes": graph.graph.num_nodes,
        "edges": int(graph.graph.edge_index.shape[1]),
        "directed": graph.directed,
        "source_path": str(graph.source_path),
        "error": error,
    }
    if status_detail is not None:
        row["status_detail"] = status_detail
    return row


def external_layout_result(
    graph: LoadedGraph,
    engine_name: str,
    output_dir: Path,
) -> Tuple[Optional[CompetitorResult], Optional[Dict[str, Any]]]:
    """Run one external layout with timeout isolation.

    Parameters
    ----------
    graph : LoadedGraph
        Corpus graph metadata.
    engine_name : str
        Registered external competitor name.
    output_dir : Path
        Output directory for temporary artifacts.

    Returns
    -------
    Tuple[CompetitorResult | None, Dict[str, Any] | None]
        Successful result, or an error row.
    """
    # dagua's child runs the torch optimizer: fork-after-torch-import
    # deadlocks (known gotcha, see ba1dc95 spawn-context pool fix in the
    # main repo). r80 holdout symptom: first dagua row OK (forked before
    # the parent imported torch for scoring), every later fork hung until
    # timeout. External engines exec subprocess binaries and stay on fork
    # (spawn would cost ~10s torch re-import per child for nothing).
    context = mp.get_context("spawn" if engine_name == "dagua" else "fork")
    temp_path = output_dir / "positions" / f".{safe_component(graph.name)}__{engine_name}.child.pt"
    result_path = output_dir / "positions" / f".{safe_component(graph.name)}__{engine_name}.json"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    for path in (temp_path, result_path):
        if path.exists():
            path.unlink()
    process = context.Process(
        target=_external_layout_worker,
        args=(graph.graph, engine_name, str(temp_path), str(result_path)),
    )
    start = time.perf_counter()
    try:
        process.start()
        process.join(TIMEOUT_SECONDS)
        elapsed = time.perf_counter() - start
        if process.is_alive():
            process.terminate()
            process.join(5.0)
            if process.is_alive():
                process.kill()
                process.join()
            for path in (temp_path, result_path):
                if path.exists():
                    path.unlink()
            return None, make_error_row(graph, engine_name, elapsed, "timeout", "timeout")
        exitcode = process.exitcode
        if not result_path.is_file():
            if temp_path.exists():
                temp_path.unlink()
            detail = f"child exited {exitcode}" if exitcode is not None else "child exited"
            return None, make_error_row(graph, engine_name, elapsed, detail)
        with result_path.open("r", encoding="utf-8") as handle:
            message = json.load(handle)
        result_path.unlink()
        if message.get("status") != "OK":
            if temp_path.exists():
                temp_path.unlink()
            return None, make_error_row(
                graph,
                engine_name,
                float(message.get("runtime_s") or elapsed),
                str(message.get("error") or "external child error"),
            )
        position_path = Path(str(message.get("temp_positions_path")))
        if not position_path.is_file():
            return None, make_error_row(
                graph,
                engine_name,
                elapsed,
                "external child returned no positions",
            )
        positions = torch.load(position_path, map_location="cpu")
        position_path.unlink()
        return (
            CompetitorResult(
                name=engine_name,
                pos=positions,
                runtime_seconds=float(message.get("runtime_s") or elapsed),
                error=message.get("error"),
            ),
            None,
        )
    finally:
        # Process objects retain OS-level handles (sentinel fd, pipe fds) until
        # explicitly closed; over a ~19k-fork run (274 graphs x 8 external
        # engines) never calling close() leaks those handles for the life of
        # the parent process. See the module-level comment on RSS_ABORT_BYTES.
        if process.is_alive():
            process.kill()
            process.join()
        try:
            process.close()
        except ValueError:
            pass


def run_engine(graph: LoadedGraph, competitor: CompetitorBase, output_dir: Path) -> Dict[str, Any]:
    """Run one engine on one corpus graph.

    Parameters
    ----------
    graph : LoadedGraph
        Corpus graph metadata.
    competitor : CompetitorBase
        Registered layout engine adapter.
    output_dir : Path
        Output directory for positions.

    Returns
    -------
    Dict[str, Any]
        Completed result row.
    """
    # Every engine -- including "dagua" itself -- runs in a forked, isolated
    # child process. This was NOT always true: "dagua" used to run directly
    # in-process (competitor.layout() called inline, no fork, and its
    # `timeout` argument was silently ignored -- DaguaCompetitor.layout()
    # does `del timeout`). Live reproduction during the r80 OOM investigation
    # confirmed the root cause: on certain dense small graphs (e.g.
    # suitesparse/Journals, 124 nodes / 5972 edges, avg degree ~96) dagua's
    # own layout optimizer runs far longer than on sparse graphs of similar
    # node count -- multiple minutes with no enforced ceiling -- and its
    # resident memory climbs the entire time it runs (observed tens of GB
    # mid-run in a live repro before the process was killed as a
    # precaution). The r80_holdout crash (OOM at anon-rss ~101GB, dmesg
    # pid=344661) landed exactly on the first of these dense outlier graphs
    # after 261 clean graphs. The underlying optimizer behavior lives in
    # dagua/layout/ and is out of scope to fix here. Isolating "dagua" the
    # same way as the other 8 engines means that unbounded runtime/memory
    # growth is contained to a short-lived child process whose memory is
    # released to the OS the instant it exits (killed at TIMEOUT_SECONDS if
    # still running), so it can never accumulate in -- or take down -- the
    # long-lived parent harness process. This also gives "dagua" the same
    # TIMEOUT_SECONDS wall-clock enforcement the
    # other engines already had (previously unenforced for "dagua").
    result, error_row = external_layout_result(graph, competitor.name, output_dir)
    if error_row is not None:
        return error_row
    if result is None:
        return make_error_row(graph, competitor.name, 0.0, "external child failed")

    base_row = {
        "graph": graph.name,
        "corpus": graph.corpus,
        "engine": competitor.name,
        "runtime_s": float(result.runtime_seconds),
        "nodes": graph.graph.num_nodes,
        "edges": int(graph.graph.edge_index.shape[1]),
        "directed": graph.directed,
        "source_path": str(graph.source_path),
    }
    if result.pos is None:
        return {
            **base_row,
            "status": "ERROR",
            "metrics": {},
            "reported_metrics": {},
            "composite": None,
            "positions_path": None,
            "error": result.error or "adapter returned no positions",
        }

    positions = result.pos.detach().cpu().to(dtype=torch.float32)
    try:
        metrics = evaluate(graph.graph, positions, tier="full")
        composite = composite_auto(metrics, graph.directed)
    except Exception as exc:  # noqa: BLE001
        return {
            **base_row,
            "status": "ERROR",
            "metrics": {},
            "reported_metrics": {},
            "composite": None,
            "positions_path": None,
            "error": f"metrics {type(exc).__name__}: {exc}",
        }

    relpath = position_relpath(graph.name, competitor.name)
    position_path = output_dir / relpath
    position_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(positions, position_path)
    clean_metrics = json_clean(metrics)
    return {
        **base_row,
        "status": "OK",
        "metrics": clean_metrics,
        "reported_metrics": {name: clean_metrics.get(name) for name in REPORT_METRICS},
        "composite": float(composite),
        "positions_path": relpath,
        "error": None,
    }


def engine_availability(engine_names: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Inspect availability for requested engines.

    Parameters
    ----------
    engine_names : Sequence[str]
        Engine names to check.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Availability metadata keyed by engine.
    """
    availability: Dict[str, Dict[str, Any]] = {}
    for engine_name in engine_names:
        competitor = get_competitor(engine_name)
        if competitor is None:
            availability[engine_name] = {"available": False, "reason": "adapter not registered"}
            continue
        try:
            available = competitor.available()
        except Exception as exc:  # noqa: BLE001
            availability[engine_name] = {"available": False, "reason": str(exc)}
            continue
        availability[engine_name] = {
            "available": bool(available),
            "reason": None if available else "adapter unavailable",
        }
    return availability


def validate_store(output_dir: Path) -> None:
    """Validate that OK rows and position tensors match one-to-one.

    Parameters
    ----------
    output_dir : Path
        Published output directory.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If an OK row is missing its tensor or a tensor has no OK row.
    """
    payload = load_existing_results(output_dir)
    rows = payload.get("rows", [])
    ok_paths = {
        str(row.get("positions_path"))
        for row in rows
        if row.get("status") == "OK" and row.get("positions_path")
    }
    missing = sorted(path for path in ok_paths if not (output_dir / path).is_file())
    position_dir = output_dir / "positions"
    position_paths = (
        {str(path.relative_to(output_dir)) for path in position_dir.glob("*.pt") if path.is_file()}
        if position_dir.is_dir()
        else set()
    )
    orphaned = sorted(position_paths - ok_paths)
    if missing or orphaned:
        details = []
        if missing:
            details.append(f"missing position files for OK rows: {missing}")
        if orphaned:
            details.append(f"orphaned position files without OK rows: {orphaned}")
        raise RuntimeError("; ".join(details))


def graph_best_external(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Return the best external OK row for each graph.

    Parameters
    ----------
    rows : Iterable[Dict[str, Any]]
        Result rows.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Best external row keyed by graph name.
    """
    best: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if row.get("engine") not in EXTERNAL_ENGINE_NAMES or row.get("status") != "OK":
            continue
        composite = row.get("composite")
        if composite is None:
            continue
        graph_name = str(row["graph"])
        if graph_name not in best or float(composite) > float(best[graph_name]["composite"]):
            best[graph_name] = row
    return best


def summarize_wtl(rows: List[Dict[str, Any]], corpus: Optional[str] = None) -> Tuple[int, int, int]:
    """Summarize Dagua wins, ties, and losses.

    Parameters
    ----------
    rows : List[Dict[str, Any]]
        Result rows.
    corpus : str | None, default=None
        Optional corpus filter.

    Returns
    -------
    Tuple[int, int, int]
        Win, tie, and loss counts.
    """
    filtered = [row for row in rows if corpus is None or row.get("corpus") == corpus]
    best_external = graph_best_external(filtered)
    dagua_rows = {
        str(row["graph"]): row
        for row in filtered
        if row.get("engine") == "dagua" and row.get("status") == "OK"
    }
    wins = ties = losses = 0
    for graph_name, dagua_row in dagua_rows.items():
        external_row = best_external.get(graph_name)
        if external_row is None:
            continue
        delta = float(dagua_row["composite"]) - float(external_row["composite"])
        if delta > TIE_BAND:
            wins += 1
        elif delta >= -TIE_BAND:
            ties += 1
        else:
            losses += 1
    return wins, ties, losses


def write_markdown(output_dir: Path, payload: Dict[str, Any]) -> None:
    """Write the standard-corpora Markdown report.

    Parameters
    ----------
    output_dir : Path
        Harness output directory.
    payload : Dict[str, Any]
        Published result payload.

    Returns
    -------
    None
    """
    rows = list(payload.get("rows", []))
    corpora = sorted({str(row.get("corpus")) for row in rows if row.get("corpus")})
    lines = [
        "# Standard Corpora Heldout Evaluation",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Git SHA: {payload['git_sha']}",
        f"- Tie band: +/-{TIE_BAND:.1f} composite points",
        f"- Graphs loaded: {payload['graph_count']}",
        "",
        "## Dagua Native vs Best External",
        "",
        "| Corpus | W | T | L |",
        "|---|---:|---:|---:|",
    ]
    overall = summarize_wtl(rows)
    lines.append(f"| all | {overall[0]} | {overall[1]} | {overall[2]} |")
    for corpus in corpora:
        wins, ties, losses = summarize_wtl(rows, corpus)
        lines.append(f"| {corpus} | {wins} | {ties} | {losses} |")
    lines.extend(["", "## Rows", "", "| Graph | Corpus | Dagua | Best external | Delta | Result |"])
    lines.append("|---|---|---:|---:|---:|---|")
    best_external = graph_best_external(rows)
    dagua_rows = {
        str(row["graph"]): row
        for row in rows
        if row.get("engine") == "dagua" and row.get("status") == "OK"
    }
    for graph_name in sorted(dagua_rows):
        dagua_row = dagua_rows[graph_name]
        external_row = best_external.get(graph_name)
        if external_row is None:
            lines.append(
                f"| {graph_name} | {dagua_row.get('corpus')} | "
                f"{float(dagua_row['composite']):.2f} | n/a | n/a | no external OK |"
            )
            continue
        delta = float(dagua_row["composite"]) - float(external_row["composite"])
        label = "W" if delta > TIE_BAND else "T" if delta >= -TIE_BAND else "L"
        lines.append(
            f"| {graph_name} | {dagua_row.get('corpus')} | "
            f"{float(dagua_row['composite']):.2f} | "
            f"{float(external_row['composite']):.2f} ({external_row['engine']}) | "
            f"{delta:.2f} | {label} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Corpus graphs are capped at 2000 nodes by default.",
            "- Rome and SuiteSparse inputs are scored as undirected.",
            "- North DAG inputs are directed only when file metadata or path names say so.",
            "- Missing external engines are recorded as SKIP rows, not as losses.",
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "STDCORPORA.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def publish_results(output_dir: Path, staging: Path, payload: Dict[str, Any]) -> None:
    """Publish a staging directory atomically enough for local benchmark use.

    Parameters
    ----------
    output_dir : Path
        Final output directory.
    staging : Path
        Staging directory containing rows and positions.
    payload : Dict[str, Any]
        Final result payload.

    Returns
    -------
    None
    """
    write_results(staging, payload)
    write_markdown(staging, payload)
    previous = previous_dir(output_dir)
    if previous.exists():
        shutil.rmtree(previous)
    if output_dir.exists():
        output_dir.rename(previous)
    staging.rename(output_dir)
    if previous.exists():
        shutil.rmtree(previous)
    validate_store(output_dir)


def selected_engines(args: argparse.Namespace) -> List[str]:
    """Resolve the engine list for this run.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line options.

    Returns
    -------
    List[str]
        Engine names to run or preserve.
    """
    engines = list(args.engines) if args.engines else list(ENGINE_NAMES)
    if args.dagua_only and "dagua" not in engines:
        engines.insert(0, "dagua")
    return engines


class _MemoryGuardAbort(Exception):
    """Raised internally to unwind the row loop once the RSS ceiling is hit."""


def main() -> int:
    """Run the standard-corpora heldout harness.

    Returns
    -------
    int
        Process exit status. ``3`` when the run stopped early because of the
        RSS abort guard (results up to that point are still published).
    """
    args = parse_args()
    # Create the final output directory immediately so a user (or a watcher
    # script) checking mid-run finds something, even before the first row
    # lands in the staging directory below. The published results.json/
    # STDCORPORA.md still only appear at publish_results() time, but
    # results.rows.jsonl in the staging dir streams every row as it completes.
    args.output_dir.mkdir(parents=True, exist_ok=True)

    graphs = load_corpus(args.corpus_dir, args.max_nodes)
    if args.corpus:
        graphs = [graph for graph in graphs if graph.corpus == args.corpus]
    if args.graphs:
        requested = set(args.graphs)
        graphs = [
            graph
            for graph in graphs
            if graph.name in requested or Path(graph.name).name in requested
        ]
    if not graphs:
        readme = args.output_dir / "README.md"
        if not readme.exists():
            readme.write_text(
                "# Standard Corpora Inputs\n\n"
                "Drop Rome/North `.graph` or `.gml` files and SuiteSparse `.mtx` files here, "
                "then rerun `scripts/r79_stdcorpora_eval.py`.\n",
                encoding="utf-8",
            )
        print(f"No supported graphs found under {args.corpus_dir}", file=sys.stderr)
        return 2

    engines = selected_engines(args)
    availability = engine_availability(engines)
    existing_rows: List[Dict[str, Any]] = []
    if args.dagua_only and (args.output_dir / "results.json").is_file():
        existing_rows = [
            row
            for row in load_existing_results(args.output_dir).get("rows", [])
            if row.get("engine") != "dagua"
        ]

    staging = staging_dir(args.output_dir)
    if staging.exists() and not args.resume:
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    completed = load_jsonl_rows(staging) if args.resume else []
    if not args.resume:
        write_jsonl_rows(staging, existing_rows)
        completed = list(existing_rows)
    completed_keys = {(str(row.get("graph")), str(row.get("engine"))) for row in completed}

    rows = list(completed)
    rss_warned = False
    aborted_reason: Optional[str] = None
    row_index = 0
    try:
        for graph in graphs:
            for engine_name in engines:
                if args.dagua_only and engine_name != "dagua":
                    continue
                key = (graph.name, engine_name)
                if key in completed_keys:
                    continue
                available = availability.get(
                    engine_name, {"available": False, "reason": "not checked"}
                )
                competitor = get_competitor(engine_name)
                if competitor is None or not available["available"]:
                    reason = str(available.get("reason") or "unavailable")
                    row = make_skip_row(graph, engine_name, reason)
                else:
                    row = run_engine(graph, competitor, staging)
                append_row(staging, row)
                rows.append(row)
                completed_keys.add(key)
                print(f"{row['status']} {graph.name} {engine_name}")
                del row
                row_index += 1

                if row_index % GC_TRIM_INTERVAL_ROWS == 0:
                    release_native_heap()
                    rss = current_rss_bytes()
                    if rss is not None:
                        print(
                            f"progress: {row_index} rows, RSS={rss / 1024**3:.2f}GB",
                            file=sys.stderr,
                        )
                        if rss >= RSS_ABORT_BYTES:
                            aborted_reason = (
                                f"RSS {rss / 1024**3:.1f}GB >= abort ceiling "
                                f"{RSS_ABORT_BYTES / 1024**3:.0f}GB after {row_index} rows"
                            )
                            print(f"ABORT: {aborted_reason}", file=sys.stderr)
                            raise _MemoryGuardAbort(aborted_reason)
                        if rss >= RSS_WARN_BYTES and not rss_warned:
                            rss_warned = True
                            print(
                                f"WARNING: RSS {rss / 1024**3:.1f}GB >= warn threshold "
                                f"{RSS_WARN_BYTES / 1024**3:.0f}GB after {row_index} rows",
                                file=sys.stderr,
                            )
    except _MemoryGuardAbort:
        pass

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": git_sha(),
        "seed": SEED,
        "timeout_seconds": TIMEOUT_SECONDS,
        "max_nodes": args.max_nodes,
        "tie_band": TIE_BAND,
        "graph_count": len(graphs),
        "engines": engines,
        "availability": availability,
        "rows": rows,
        "aborted": aborted_reason is not None,
        "aborted_reason": aborted_reason,
    }
    publish_results(args.output_dir, staging, payload)
    return 3 if aborted_reason is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
