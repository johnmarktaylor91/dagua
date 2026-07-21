"""R0.4 row forensics: per-row native-vs-field deficit decomposition.

For every extended-corpus graph this tool reports:

* native's extended composite (fresh-scored from the deterministic baseline
  positions via the CORRECTED harness: ``build_graph_map`` + ``score_position``),
* the best V2 competitor and which engine it is,
* the gap and best-or-tied status (frozen tie band),
* a dominant-failure-mode tag: the ruler facet whose leave-one-swap (native
  facet value replaced by the field-best value) buys native the most composite
  under the real ruler,
* a degenerate-tie flag for tie/win rows whose native layout is essentially a
  field engine's layout (position sha identity or unit-cloud Procrustes RMSD
  below the bit-exact tier), which mechanizes the borrow-best palette
  blindspot check.

Outputs a regenerable round-packet table (markdown + json). Scores are cached
by position sha + ruler signature, so round-over-round reruns only pay for
changed positions.

Usage
-----
python scripts/row_forensics.py \\
    [--baseline-dir .../s1_out] [--v2 .../R8_EVENTA_RAW_SCORES_V2_BACKFILL.json] \\
    [--output-dir .../roundloop] [--workers 8] [--skip-tie-audit]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import roundloop_common as rl  # noqa: E402
import torch  # noqa: E402
from native_sprint_score import scoring_signature, sha256_file  # noqa: E402

#: Fresh-rescore drift beyond this vs the V2 cached composite is flagged
#: (an integrity tripwire on the frozen scale, not a hard failure).
FIELD_RESCORE_DRIFT_WARN = 0.05


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line options.

    Parameters
    ----------
    argv : Optional[Sequence[str]], optional
        Explicit argument sequence, or ``None`` for ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=rl.DEFAULT_BASELINE_DIR)
    parser.add_argument("--v2", type=Path, default=rl.V2_FIELD_PATH)
    parser.add_argument("--output-dir", type=Path, default=rl.ROUNDLOOP_DIR)
    parser.add_argument("--cache", type=Path, default=rl.DEFAULT_CACHE_PATH)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--skip-tie-audit",
        action="store_true",
        help="Skip the Procrustes degenerate-tie scan (fast mode).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="row_forensics",
        help="Output basename tag (round packets can use e.g. r1_row_forensics).",
    )
    return parser.parse_args(argv)


def native_tasks(
    baseline_dir: Path, names: Sequence[str]
) -> Tuple[List[Tuple[str, str, str]], List[str]]:
    """Collect native scoring tasks from the baseline positions dir.

    Parameters
    ----------
    baseline_dir : Path
        Deterministic native baseline benchmark output dir.
    names : Sequence[str]
        Extended corpus names.

    Returns
    -------
    Tuple[List[Tuple[str, str, str]], List[str]]
        Score tasks and names with missing position files.
    """
    tasks: List[Tuple[str, str, str]] = []
    missing: List[str] = []
    for name in names:
        path = rl.native_position_path(baseline_dir, name)
        if path.exists():
            tasks.append((name, "dagua", str(path)))
        else:
            missing.append(name)
    return tasks, missing


def field_metric_rows(
    field_best: Dict[str, Dict[str, Any]],
    graphs: Dict[str, Any],
    cache: rl.ScoreCache,
    signature: str,
    workers: int,
) -> Dict[str, Dict[str, Any]]:
    """Ensure every field-best row has a facet-level metrics payload.

    V2 legacy-cache rows carry composites but empty metrics; those field-best
    positions are re-scored fresh (cached) so facet decomposition has both
    sides. Fresh V2 rows keep their stored metrics.

    Parameters
    ----------
    field_best : Dict[str, Dict[str, Any]]
        V2 field-best row per graph.
    graphs : Dict[str, Any]
        Graph map.
    cache : rl.ScoreCache
        Score cache.
    signature : str
        Scoring signature.
    workers : int
        Fresh-scoring worker count.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Per graph: ``{"metrics": ..., "fresh_extended": float | None,
        "rescored": bool}``.
    """
    need_rescore: List[Tuple[str, str, str]] = []
    out: Dict[str, Dict[str, Any]] = {}
    for graph, row in field_best.items():
        if row.get("metrics"):
            out[graph] = {
                "metrics": dict(row["metrics"]),
                "fresh_extended": float(row["extended_composite"]),
                "rescored": False,
            }
            continue
        path = row.get("position_path")
        if path and Path(str(path)).exists():
            need_rescore.append((graph, str(row["engine"]), str(path)))
        else:
            out[graph] = {"metrics": {}, "fresh_extended": None, "rescored": False}
    if need_rescore:
        scored = rl.score_positions_cached(need_rescore, graphs, cache, signature, workers)
        for task in need_rescore:
            row = scored[task]
            out[task[0]] = {
                "metrics": dict(row.get("metrics", {})),
                "fresh_extended": row.get("extended_composite"),
                "rescored": True,
            }
    return out


def build_rows(args: argparse.Namespace) -> Dict[str, Any]:
    """Build the full forensics payload.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI options.

    Returns
    -------
    Dict[str, Any]
        Round-packet payload (header + rows).
    """
    v2 = rl.load_v2_field(args.v2)
    names = list(v2["header"]["extended_names"])
    signature = scoring_signature()
    graphs = rl.graphs_for_names(names)
    directed_flags = rl.semantic_direction_flags(graphs)
    field_rows = rl.field_rows_by_graph(v2["rows"])
    field_best = rl.field_best_by_graph(v2["rows"])
    v2_native = rl.v2_native_scores(v2["rows"])

    cache = rl.ScoreCache(args.cache)
    tasks, missing = native_tasks(args.baseline_dir, names)
    native_scored = rl.score_positions_cached(tasks, graphs, cache, signature, args.workers)
    native_by_graph = {task[0]: row for task, row in native_scored.items()}
    field_metrics = field_metric_rows(field_best, graphs, cache, signature, args.workers)

    rows: List[Dict[str, Any]] = []
    tallies = {"strictly_best": 0, "tied": 0, "behind": 0, "missing": 0}
    for name in names:
        native_row = native_by_graph.get(name)
        best_row = field_best.get(name)
        test_graph = graphs[name]
        base: Dict[str, Any] = {
            "graph": name,
            "family": ",".join(sorted(test_graph.tags)) if test_graph.tags else "",
            "ruler": "directed" if directed_flags[name] else "undirected",
            "num_nodes": test_graph.graph.num_nodes,
        }
        if native_row is None or native_row.get("extended_composite") is None or best_row is None:
            tallies["missing"] += 1
            base.update(
                {
                    "status": "missing",
                    "native": None,
                    "field_best": None,
                    "errors": (native_row or {}).get("errors"),
                }
            )
            rows.append(base)
            continue
        native_score = float(native_row["extended_composite"])
        best_score = float(best_row["extended_composite"])
        delta = native_score - best_score
        status = rl.classify(delta)
        tallies[status] += 1
        base.update(
            {
                "status": status,
                "native": native_score,
                "field_best": best_score,
                "field_best_engine": best_row["engine"],
                "delta": delta,
                "v2_native": v2_native.get(name),
                "native_position_sha256": native_row["position_sha256"],
                "native_position_path": native_row["position_path"],
                "field_best_path": best_row.get("position_path"),
            }
        )
        # Facet decomposition (both sides need metrics).
        field_info = field_metrics.get(name, {"metrics": {}, "fresh_extended": None})
        if native_row.get("metrics") and field_info["metrics"]:
            gains = rl.facet_swap_gains(
                native_row["metrics"], field_info["metrics"], directed_flags[name]
            )
            positive = [(facet, gain) for facet, gain in gains if gain > 1e-9]
            base["dominant_facet"] = positive[0][0] if positive else None
            base["dominant_facet_gain"] = positive[0][1] if positive else 0.0
            base["top_facets"] = [{"facet": facet, "gain": gain} for facet, gain in gains[:3]]
            base["facet_table"] = rl.facet_table(native_row["metrics"], field_info["metrics"])
        else:
            base["dominant_facet"] = None
            base["dominant_facet_gain"] = None
            base["top_facets"] = []
            base["facet_table"] = {}
        if field_info.get("rescored") and field_info.get("fresh_extended") is not None:
            drift = float(field_info["fresh_extended"]) - best_score
            base["field_rescore_drift"] = drift
            base["field_rescore_drift_flag"] = abs(drift) > FIELD_RESCORE_DRIFT_WARN
        # Degenerate-tie audit for banked (tie/win) rows.
        if status in ("strictly_best", "tied") and not args.skip_tie_audit:
            match = rl.closest_field_layout(
                str(native_row["position_path"]),
                str(native_row["position_sha256"]),
                field_rows.get(name, []),
            )
            if match is not None:
                base["closest_field_engine"] = match.engine
                base["closest_field_rmsd"] = match.rmsd
                base["degenerate_tie"] = match.degenerate
                base["near_field_engine"] = match.near
        rows.append(base)

    header = {
        "tool": "row_forensics",
        "generated_at": rl.utc_now_iso(),
        "git_sha": rl.git_sha(SCRIPTS_DIR.parent),
        "scoring_signature": signature,
        "ruler_schema": str(v2["header"]["ruler_schema"]),
        "tie_band": rl.TIE_BAND,
        "baseline_dir": str(args.baseline_dir),
        "v2_path": str(args.v2),
        "v2_sha256": sha256_file(args.v2),
        "corpus_size": len(names),
        "tallies": tallies,
        "best_or_tied": tallies["strictly_best"] + tallies["tied"],
        "missing_positions": missing,
        "degenerate_rmsd_threshold": rl.DEGENERATE_RMSD,
        "near_rmsd_threshold": rl.NEAR_RMSD,
    }
    return {"header": header, "rows": rows}


def render_markdown(payload: Dict[str, Any]) -> str:
    """Render the round-packet markdown table.

    Parameters
    ----------
    payload : Dict[str, Any]
        Forensics payload.

    Returns
    -------
    str
        Markdown document.
    """
    header = payload["header"]
    rows = payload["rows"]
    lines: List[str] = []
    lines.append("# Row forensics: native vs V2 field")
    lines.append("")
    lines.append(f"- Generated: {header['generated_at']}  (git {header['git_sha'][:12]})")
    lines.append(f"- Baseline: `{header['baseline_dir']}`")
    lines.append(f"- Ruler: {header['ruler_schema']}  sig `{header['scoring_signature'][:16]}...`")
    tallies = header["tallies"]
    lines.append(
        f"- **Best-or-tied {header['best_or_tied']}/{header['corpus_size']}** "
        f"(best {tallies['strictly_best']}, tied {tallies['tied']}, "
        f"behind {tallies['behind']}, missing {tallies['missing']}; "
        f"tie band {header['tie_band']})"
    )
    lines.append("")

    behind = [row for row in rows if row.get("status") == "behind"]
    behind.sort(key=lambda row: row.get("delta", 0.0))
    lines.append(f"## Behind rows ({len(behind)})")
    lines.append("")
    if behind:
        lines.append(
            "| graph | native | field best | engine | gap | dominant facet (gain) | family |"
        )
        lines.append("| --- | ---: | ---: | --- | ---: | --- | --- |")
        for row in behind:
            facet = row.get("dominant_facet") or "-"
            gain = row.get("dominant_facet_gain")
            facet_cell = f"{facet} ({gain:+.2f})" if facet != "-" and gain is not None else "-"
            lines.append(
                f"| {row['graph']} | {row['native']:.3f} | {row['field_best']:.3f} "
                f"| {row['field_best_engine']} | {row['delta']:+.3f} | {facet_cell} "
                f"| {row['family']} |"
            )
        lines.append("")
        for row in behind:
            lines.append(f"### {row['graph']}")
            lines.append("")
            top = row.get("top_facets") or []
            if top:
                lines.append("Top swap gains (facet -> composite gain if native matched field):")
                lines.append("")
                table = row.get("facet_table", {})
                lines.append("| facet | native | field | swap gain |")
                lines.append("| --- | ---: | ---: | ---: |")
                for entry in top:
                    facet = entry["facet"]
                    values = table.get(facet, {})
                    native_v = values.get("native")
                    field_v = values.get("field")
                    native_cell = "-" if native_v is None else f"{native_v:.4f}"
                    field_cell = "-" if field_v is None else f"{field_v:.4f}"
                    lines.append(
                        f"| {facet} | {native_cell} | {field_cell} | {entry['gain']:+.3f} |"
                    )
            else:
                lines.append("(no facet decomposition available)")
            lines.append("")

    degenerate = [row for row in rows if row.get("degenerate_tie")]
    near = [row for row in rows if row.get("near_field_engine") and not row.get("degenerate_tie")]
    lines.append(f"## Degenerate ties ({len(degenerate)}) / near-field ties ({len(near)})")
    lines.append("")
    if degenerate or near:
        lines.append("| graph | status | closest engine | rmsd | flag |")
        lines.append("| --- | --- | --- | ---: | --- |")
        for row in degenerate + near:
            flag = "DEGENERATE" if row.get("degenerate_tie") else "near"
            lines.append(
                f"| {row['graph']} | {row['status']} | {row.get('closest_field_engine')} "
                f"| {row.get('closest_field_rmsd', float('nan')):.2e} | {flag} |"
            )
    else:
        lines.append("None: no native tie/win is a copy of a field layout.")
    lines.append("")

    drifted = [row for row in rows if row.get("field_rescore_drift_flag")]
    if drifted:
        lines.append(f"## Field rescore drift flags ({len(drifted)})")
        lines.append("")
        lines.append("| graph | v2 field best | fresh drift |")
        lines.append("| --- | ---: | ---: |")
        for row in drifted:
            lines.append(
                f"| {row['graph']} | {row['field_best']:.3f} | {row['field_rescore_drift']:+.4f} |"
            )
        lines.append("")

    lines.append("## Full table")
    lines.append("")
    lines.append("| graph | status | native | field best | engine | gap | dominant facet |")
    lines.append("| --- | --- | ---: | ---: | --- | ---: | --- |")
    order = {"behind": 0, "missing": 1, "tied": 2, "strictly_best": 3}
    for row in sorted(rows, key=lambda r: (order.get(r.get("status", ""), 9), r["graph"])):
        if row.get("status") == "missing":
            lines.append(f"| {row['graph']} | missing | - | - | - | - | - |")
            continue
        lines.append(
            f"| {row['graph']} | {row['status']} | {row['native']:.3f} "
            f"| {row['field_best']:.3f} | {row['field_best_engine']} "
            f"| {row['delta']:+.3f} | {row.get('dominant_facet') or '-'} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run row forensics and write the round packet.

    Parameters
    ----------
    argv : Optional[Sequence[str]], optional
        Explicit argument sequence, or ``None`` for ``sys.argv``.

    Returns
    -------
    int
        Process exit code (0 even when rows are behind; forensics reports,
        the regression-lock harness gates).
    """
    torch.set_num_threads(1)
    args = parse_args(argv)
    payload = build_rows(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.tag}.json"
    md_path = args.output_dir / f"{args.tag}.md"
    json_path.write_text(json.dumps(payload, indent=1))
    md_path.write_text(render_markdown(payload))
    header = payload["header"]
    behind = [row["graph"] for row in payload["rows"] if row.get("status") == "behind"]
    print(
        f"[row_forensics] best-or-tied {header['best_or_tied']}/{header['corpus_size']}; "
        f"behind: {behind}",
        flush=True,
    )
    print(f"[row_forensics] wrote {json_path} and {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
