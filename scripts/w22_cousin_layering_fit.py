"""W2-2 cousin fit for the directed low-layering gate threshold (review F4).

The stress-family arm's directed admission uses
``LOW_LAYERING_MIN_AVG_LAYER_WIDTH``. Per the sprint's binding fitting rule
(measure/COUSINS.md), that threshold must be FITTED on training cousins and
only CONFIRMED on dev63. This script produces the fitting evidence:

Phase ``native`` (run at the PRE-ARM base, e.g. corrected W2-1) regenerates
the native incumbent for every directed training cousin under the exact
holdout deterministic envelope and scores it with the frozen V3 ruler. The
base run avoids circularity: the gate decides where the arm may challenge the
incumbent, so the reference must not itself contain the arm.

Phase ``fit`` (run at the packet head) builds the frozen-seed stress-family
candidates for the same cousins, scores them with the same ruler, joins the
two score sets, and emits the admitted/rejected feature table: per cousin,
``avg_layer_width``, acyclicity, the native and best-stress V3 composites,
and the margin. The threshold is then established from this table alone;
dev63 rows are only ever confirmation.

Usage::

    # at the pre-arm base checkout (script copied in, untracked there)
    PYTHONPATH=$PWD python scripts/w22_cousin_layering_fit.py native \
        --train-dir ./projects/dagua/eval_output/sprint2_train \
        --out-dir eval_output/w22_layering_fit

    # at the packet head
    PYTHONPATH=$PWD python scripts/w22_cousin_layering_fit.py fit \
        --train-dir ./projects/dagua/eval_output/sprint2_train \
        --native-json <base>/eval_output/w22_layering_fit/cousins_native.json \
        --out-dir eval_output/w22_layering_fit
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.glados_holdout_run import (  # noqa: E402
    DEFAULT_CHILD_RSS_ABORT_GB,
    DEFAULT_MAX_EDGES,
    DEFAULT_MAX_NODES,
    DEFAULT_MIN_AVAIL_GB,
    DEFAULT_NATIVE_TIMEOUT,
    GraphEntry,
    RowExecutor,
    build_test_graph_map,
    load_phase,
)

# The stress candidates could flip a contest whenever they land within the
# frozen tie band of the incumbent; that is the competitiveness bar the gate
# exists to predict.
COMPETITIVE_MARGIN = -0.5


def _load_directed_cousins(train_dir: Path) -> List[GraphEntry]:
    """Load every directed training cousin through the holdout load phase.

    Parameters
    ----------
    train_dir : Path
        Training corpus root (corpus-shaped subtree, e.g. sprint2_train).

    Returns
    -------
    List[GraphEntry]
        Directed cousins sorted by name.
    """
    relpaths = {
        str(path.relative_to(train_dir))
        for path in train_dir.rglob("*")
        if path.is_file() and path.suffix in {".graphml", ".mtx"}
    }
    entries, load_rows, _stats = load_phase(
        train_dir, relpaths, DEFAULT_MAX_NODES, DEFAULT_MAX_EDGES
    )
    problems = [
        f"{row['graph']}: {row['status']}"
        for row in load_rows
        if row.get("status") not in {"OK", None}
    ]
    if problems:
        raise RuntimeError("cousin load problems: " + "; ".join(problems))
    return sorted((entry for entry in entries if entry.directed), key=lambda e: e.name)


def _classifier_features(entry: GraphEntry) -> Dict[str, Any]:
    """Return the gate-relevant classifier features for one cousin.

    Uses the same ``classify_graph`` output the engine attaches to
    ``LayoutProblem.structure``, so the fitted feature is exactly the value
    the gate reads at runtime.
    """
    from dagua.layout.graph_classify import classify_graph

    graph = entry.loaded.graph
    structure = classify_graph(graph.edge_index, graph.num_nodes)
    return {
        "n": int(graph.num_nodes),
        "e": int(graph.edge_index.shape[1]) if graph.edge_index.numel() else 0,
        "is_directed_acyclic": bool(getattr(structure, "is_directed_acyclic", True)),
        "avg_layer_width": float(getattr(structure, "avg_layer_width", 0.0)),
    }


def _score_positions(
    entries: Sequence[GraphEntry],
    jobs_hint: int,
    tasks: List[Dict[str, Any]],
) -> None:
    """Score saved position tensors in-place under the frozen V3 ruler.

    Parameters
    ----------
    entries : Sequence[GraphEntry]
        Loaded cousins backing the tasks.
    jobs_hint : int
        Unused (sequential is fine at cousin scale); kept for symmetry.
    tasks : List[Dict[str, Any]]
        Mutated: ``v3_tiered`` merged into each task with a ``path``.
    """
    import scripts.native_sprint_score as nss

    graph_map = build_test_graph_map(entries)
    signature = nss.scoring_signature()
    nss.init_worker(graph_map, signature)
    for task in tasks:
        score = nss.score_position(
            nss._WORKER_GRAPHS[task["graph"]],
            task["path"],
            task["engine"],
            signature,
            ruler="v3",
        )
        task["v3_tiered"] = float(score["v3_tiered"])


def run_native(args: argparse.Namespace) -> int:
    """Regenerate + score the pre-arm native incumbent for every cousin."""
    entries = _load_directed_cousins(args.train_dir)
    if not entries:
        print("ERROR: no directed cousins found", file=sys.stderr)
        return 3
    run_dir = args.out_dir / "native_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    executor = RowExecutor(
        argparse.Namespace(
            native_deterministic=True,
            child_rss_abort_gb=DEFAULT_CHILD_RSS_ABORT_GB,
            min_avail_gb=DEFAULT_MIN_AVAIL_GB,
            rss_guard_selftest=False,
            seed=args.seed,
        ),
        run_dir,
    )

    def _one(entry: GraphEntry) -> Dict[str, Any]:
        row = executor.run_row(
            entry,
            "dagua",
            seed=None,
            child_seed=args.seed,
            timeout_s=DEFAULT_NATIVE_TIMEOUT,
            is_native=True,
        )
        print(f"{row['status']} {entry.name}", flush=True)
        return row

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        rows = list(pool.map(_one, entries))
    tasks: List[Dict[str, Any]] = []
    for entry, row in zip(entries, rows):
        if row["status"] != "OK":
            print(f"ERROR: native row failed for {entry.name}: {row}", file=sys.stderr)
            return 1
        tasks.append(
            {
                "graph": entry.name,
                "engine": "dagua",
                "path": str(run_dir / row["positions_path"]),
            }
        )
    _score_positions(entries, args.jobs, tasks)
    payload = {
        task["graph"]: {"native_v3_tiered": task["v3_tiered"], "positions_path": task["path"]}
        for task in tasks
    }
    out_path = args.out_dir / "cousins_native.json"
    out_path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {len(payload)} native cousin scores -> {out_path}")
    return 0


def run_fit(args: argparse.Namespace) -> int:
    """Build + score stress candidates, join, and emit the fitting table."""
    from dagua.layout.graph_classify import classify_graph
    from dagua.layout.ops.pipelines.native_stress_family_arm import (
        build_stress_family_candidates,
    )
    from dagua.layout.ops.state import LayoutProblem

    native_scores: Dict[str, Any] = json.loads(args.native_json.read_text())
    entries = _load_directed_cousins(args.train_dir)
    entries = [entry for entry in entries if entry.name in native_scores]
    missing = sorted(set(native_scores) - {entry.name for entry in entries})
    if missing:
        print(f"ERROR: cousins missing from train dir: {missing}", file=sys.stderr)
        return 3
    candidate_dir = args.out_dir / "stress_candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    graph_map = build_test_graph_map(entries)
    tasks: List[Dict[str, Any]] = []
    features: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        features[entry.name] = _classifier_features(entry)
        test_graph = graph_map[entry.name]
        graph = test_graph.graph
        structure = classify_graph(graph.edge_index, graph.num_nodes)
        problem = LayoutProblem(
            edge_index=graph.edge_index,
            num_nodes=graph.num_nodes,
            node_sizes=graph.node_sizes,
            structure=structure,
            seed=args.seed,
        )
        candidates = build_stress_family_candidates(problem, node_sep=args.node_sep)
        for name, pos in candidates.items():
            path = candidate_dir / f"{entry.name.replace('/', '__')}__{name}.pt"
            torch.save(pos.detach().to(device="cpu", dtype=torch.float32), path)
            tasks.append({"graph": entry.name, "engine": name, "path": str(path)})
    _score_positions(entries, args.jobs, tasks)

    rows: List[Dict[str, Any]] = []
    for entry in entries:
        entry_tasks = [task for task in tasks if task["graph"] == entry.name]
        best = max(entry_tasks, key=lambda task: task["v3_tiered"])
        native_v3 = float(native_scores[entry.name]["native_v3_tiered"])
        margin = best["v3_tiered"] - native_v3
        rows.append(
            {
                "graph": entry.name,
                **features[entry.name],
                "native_v3": round(native_v3, 3),
                "stress_best_v3": round(best["v3_tiered"], 3),
                "stress_best_engine": best["engine"],
                "margin": round(margin, 3),
                "competitive": margin >= COMPETITIVE_MARGIN,
            }
        )

    # Threshold establishment (acyclic rows only: cyclic digraphs have no
    # faithful layering and bypass the width check by construction). Report
    # the exact separation the cousins support.
    acyclic = [row for row in rows if row["is_directed_acyclic"]]
    competitive_widths = sorted(row["avg_layer_width"] for row in acyclic if row["competitive"])
    hopeless_widths = sorted(row["avg_layer_width"] for row in acyclic if not row["competitive"])
    fit = {
        "competitive_min_width": competitive_widths[0] if competitive_widths else None,
        "competitive_widths": competitive_widths,
        "hopeless_widths": hopeless_widths,
        "competitive_margin_bar": COMPETITIVE_MARGIN,
    }

    out_json = args.out_dir / "layering_fit.json"
    out_json.write_text(json.dumps({"rows": rows, "fit": fit}, indent=1, sort_keys=True) + "\n")
    lines = [
        "| graph | n | acyclic | avg_layer_width | native | stress_best (engine) | margin |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in sorted(rows, key=lambda r: r["avg_layer_width"]):
        lines.append(
            f"| {row['graph']} | {row['n']} | {row['is_directed_acyclic']} | "
            f"{row['avg_layer_width']:.3f} | {row['native_v3']:.2f} | "
            f"{row['stress_best_v3']:.2f} ({row['stress_best_engine']}) | "
            f"{row['margin']:+.2f} |"
        )
    table = "\n".join(lines)
    (args.out_dir / "layering_fit.md").write_text(table + "\n")
    print(table)
    print(f"\nfit: {json.dumps(fit)}")
    print(f"wrote {out_json}")
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("native", "fit"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--train-dir", type=Path, required=True)
        cmd.add_argument("--out-dir", type=Path, default=Path("eval_output/w22_layering_fit"))
        cmd.add_argument("--seed", type=int, default=42)
        cmd.add_argument("--jobs", type=int, default=8)
        if name == "fit":
            cmd.add_argument("--native-json", type=Path, required=True)
            cmd.add_argument("--node-sep", type=float, default=70.0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Dispatch subcommands."""
    args = parse_args(argv)
    if args.command == "native":
        return run_native(args)
    return run_fit(args)


if __name__ == "__main__":
    raise SystemExit(main())
