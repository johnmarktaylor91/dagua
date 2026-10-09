"""Lay out the fresh-pairs corpus with Dagua and its competitors and score it.

Writes ``graphs.json`` (topology and node boxes) plus one ``layouts-<tag>.jsonl``
row per (graph, engine). Used by the 2026-10-08 pairwise rejudge; never touches
the ruler v4 sealed, holdout, calibration or adversarial material (only
``get_test_graphs`` and the top-level ``dagua/graphs/*.yaml`` files are read).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import signal
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ENGINES = (
    "dagua",
    "graphviz_dot",
    "graphviz_neato",
    "graphviz_sfdp",
    "elk_layered",
    "dagre",
    "igraph_sugiyama",
    "igraph_kamada_kawai",
    "nx_spring",
    "ogdf_sugiyama",
    "ogdf_fmmm",
)
# Max nodes for the July-style suite (r83 rescore used max_nodes=500).
MAX_NODES = 500
PER_CALL_TIMEOUT_S = 100
SEED = 42
SCORE_TERMS = (
    "ksm_score",
    "edge_crossing_score",
    "node_occlusion_score",
    "neighborhood_preservation_score",
    "edge_length_deviation_score",
    "gabriel_score",
    "crossing_angle_score",
    "angular_resolution_score",
    "path_continuity_score",
    "cluster_silhouette_score",
    "directed_flow_score",
    "depth_order_score",
)

_STATE: Dict[str, Any] = {}


class _Timeout(Exception):
    """Raised by the SIGALRM handler when one engine call overruns."""


def _on_alarm(signum: int, frame: Any) -> None:
    """Signal handler that aborts the current engine call."""
    raise _Timeout("per-call timeout")


def build_corpus() -> List[Tuple[str, str, Any, bool, bool, List[str]]]:
    """Build the unprotected corpus.

    Returns
    -------
    list of tuple
        ``(name, source, DaguaGraph, semantically_directed, hierarchical, tags)``.
        ``source`` is ``july`` (get_test_graphs, <= 500 nodes) or ``yaml``
        (top-level dagua/graphs/*.yaml, the standard reference graphs).
    """
    from dagua.eval.benchmark import _declares_hierarchy
    from dagua.eval.graphs import get_test_graphs, is_semantically_directed
    from dagua.graph import DaguaGraph

    out = []
    seen = set()
    for tg in get_test_graphs(max_nodes=MAX_NODES):
        tg.graph.compute_node_sizes()
        out.append(
            (
                tg.name,
                "july",
                tg.graph,
                bool(is_semantically_directed(tg)),
                bool(_declares_hierarchy(tg)),
                sorted(tg.tags),
            )
        )
        seen.add(tg.name)
    gdir = Path(__file__).resolve().parent.parent / "dagua" / "graphs"
    for path in sorted(gdir.glob("*.yaml")):  # top level only: holdout/ is excluded
        if path.stem in seen:
            continue
        try:
            graph = DaguaGraph.from_yaml(str(path))
            graph.compute_node_sizes()
        except Exception as exc:  # noqa: BLE001
            print(f"yaml skip {path.name}: {exc}", flush=True)
            continue
        out.append((path.stem, "yaml", graph, True, False, []))
    return out


def _init_worker() -> None:
    """Prepare one worker: single thread, size-aware externals (corpus is inherited by fork)."""
    import torch

    from dagua.eval.size_policy import set_size_aware_externals

    torch.set_num_threads(1)
    set_size_aware_externals(True)
    signal.signal(signal.SIGALRM, _on_alarm)


def _run_one(task: Tuple[str, str]) -> Dict[str, Any]:
    """Lay out and score one (graph, engine)."""
    import torch

    import dagua.eval.competitors  # noqa: F401  (registers adapters)
    from dagua.eval.competitors.base import get_competitor
    from dagua.metrics import composite_auto, evaluate

    name, engine = task
    _, source, graph, directed, hier, _tags = _STATE["corpus"][name]
    row: Dict[str, Any] = {"graph": name, "engine": engine, "source": source}
    comp = get_competitor(engine)
    if comp is None or not comp.available():
        row["error"] = "unavailable"
        return row
    t0 = time.perf_counter()
    try:
        signal.alarm(PER_CALL_TIMEOUT_S + 10)
        res = comp.layout(graph, timeout=PER_CALL_TIMEOUT_S, seed=SEED)
        signal.alarm(0)
    except BaseException as exc:  # noqa: BLE001
        signal.alarm(0)
        row["error"] = f"{type(exc).__name__}: {exc}"[:200]
        return row
    row["runtime_s"] = round(time.perf_counter() - t0, 3)
    if res.pos is None:
        row["error"] = (res.error or "no positions")[:200]
        return row
    pos = res.pos.detach().cpu().to(torch.float32)
    if tuple(pos.shape) != (graph.num_nodes, 2) or not bool(torch.isfinite(pos).all()):
        row["error"] = f"bad positions shape {tuple(pos.shape)} or non-finite"
        return row
    try:
        signal.alarm(120)
        metrics = evaluate(graph, pos, tier="full")
        metrics["declared_hierarchical"] = hier
        row["score"] = float(composite_auto(metrics, directed))
        signal.alarm(0)
        row["terms"] = {
            k: (None if metrics.get(k) is None else float(metrics[k])) for k in SCORE_TERMS
        }
    except BaseException as exc:  # noqa: BLE001
        signal.alarm(0)
        row["score_error"] = f"{type(exc).__name__}: {exc}"[:200]
    row["pos"] = [[round(float(x), 3), round(float(y), 3)] for x, y in pos.tolist()]
    return row


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the layout sweep."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--engines", default=",".join(ENGINES))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="first N graphs only (probe)")
    args = ap.parse_args(argv)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only: forked workers, repeatable
    args.out.mkdir(parents=True, exist_ok=True)

    corpus = build_corpus()
    if args.limit:
        corpus = corpus[: args.limit]
    _STATE["corpus"] = {row[0]: row for row in corpus}
    graphs_meta = {}
    for name, source, graph, directed, hier, tags in corpus:
        graphs_meta[name] = {
            "source": source,
            "n": graph.num_nodes,
            "e": int(graph.edge_index.shape[1]),
            "edges": graph.edge_index.t().tolist(),
            "node_sizes": [[round(float(w), 2), round(float(h), 2)] for w, h in graph.node_sizes],
            "directed": directed,
            "hierarchical": hier,
            "tags": tags,
            "n_clusters": len(graph.clusters),
        }
    (args.out / "graphs.json").write_text(json.dumps(graphs_meta))
    engines = [e for e in args.engines.split(",") if e]
    tasks = [(row[0], eng) for row in corpus for eng in engines]
    print(f"{len(corpus)} graphs x {len(engines)} engines = {len(tasks)} tasks", flush=True)
    ctx = mp.get_context("fork")
    done = 0
    with (args.out / f"layouts-{args.tag}.jsonl").open("w") as fh:
        with ctx.Pool(args.workers, initializer=_init_worker, maxtasksperchild=40) as pool:
            for row in pool.imap_unordered(_run_one, tasks):
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                done += 1
                if done % 25 == 0:
                    print(f"{done}/{len(tasks)}", flush=True)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
