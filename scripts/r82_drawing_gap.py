"""r82: break down the composite_drawing gap dagua-vs-dot into weighted sub-terms.

For each probe graph, runs dagua (its own placement + router) and graphviz_dot
(native splines), then attributes the composite_drawing gap
``dot_native - dagua`` to the individual weighted terms:

    gap = sum_t weight_t * (term_t(dot native) - term_t(dagua))

Emits a per-graph table, a mean-contribution ranking (the diagnosis), and a
full JSON dump of every metric field for downstream use.

Usage:
  .venv/bin/python scripts/r82_drawing_gap.py [--json /tmp/r82_gap.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dagua.edges import place_edge_labels, route_edges
from dagua.eval.benchmark import _drawing_metrics
from dagua.eval.competitors import get_competitor
from dagua.eval.graphs import get_test_graphs

SEED = 42
TIMEOUT_S = 120.0

PROBE_GRAPHS = [
    "citation_dag_300",
    "random_dag_200",
    "long_skip_only_24",
    "r79_undirected_sbm_low_mix_4x25",
    "chung_lu_150",
    "protein_ppi_200",
    "clustered_medium_5x20",
    "r79_nested_clusters_3x2x10",
    "heavy_tail_weights_50",
    "r79_weighted_community_4x18",
]

# Must mirror dagua.metrics.composite_drawing weights.
TERM_WEIGHTS = {
    "crossing": 30.0,
    "edge_node": 20.0,
    "label_node": 7.5,
    "label_label": 7.5,
    "port": 12.0,
    "overlap": 10.0,
    "curvature": 8.0,
    "bend": 5.0,
}


def run_engine(graph, engine_name: str) -> Optional[Dict[str, Any]]:
    """Layout one graph with one engine and score both drawing variants."""
    competitor = get_competitor(engine_name)
    if competitor is None or not competitor.available():
        return None
    result = competitor.layout(graph, timeout=TIMEOUT_S, seed=SEED)
    if result.pos is None:
        return None
    pos = result.pos
    curves = route_edges(pos, graph.edge_index, graph.node_sizes, graph.direction, graph)
    label_positions = place_edge_labels(curves, pos, graph.node_sizes, graph.edge_labels, graph)
    return _drawing_metrics(
        graph,
        pos,
        curves,
        label_positions,
        native_routes=result.routes,
        native_edge_label_positions=result.edge_label_positions,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=Path("/tmp/r82_gap.json"))
    parser.add_argument(
        "--engines",
        nargs="+",
        default=["dagua", "graphviz_dot"],
        help="First engine is 'ours' (dagua-routed), second is the reference (native).",
    )
    args = parser.parse_args()

    ours_name, ref_name = args.engines[0], args.engines[1]

    by_name = {tg.name: tg for tg in get_test_graphs(max_nodes=500)}
    missing = [n for n in PROBE_GRAPHS if n not in by_name]
    if missing:
        print(f"FATAL: probe graphs missing from corpus: {missing}", file=sys.stderr)
        return 1

    all_rows: List[Dict[str, Any]] = []
    contribs_per_graph: Dict[str, Dict[str, float]] = {}
    gaps: Dict[str, float] = {}

    for name in PROBE_GRAPHS:
        tg = by_name[name]
        tg.graph.compute_node_sizes()
        t0 = time.perf_counter()
        print(f"[gap] {name}: {ours_name} ...", flush=True)
        ours = run_engine(tg.graph, ours_name)
        print(f"[gap] {name}: {ref_name} ...", flush=True)
        ref = run_engine(tg.graph, ref_name)
        dt = time.perf_counter() - t0
        if ours is None or ref is None:
            print(f"[gap] {name}: SKIP (engine failure)", flush=True)
            continue

        ours_score = ours.get("composite_drawing_dagua_routed")
        ref_score = ref.get("composite_drawing_native")
        if ours_score is None or ref_score is None:
            print(f"[gap] {name}: SKIP (missing variant)", flush=True)
            continue

        contribs: Dict[str, float] = {}
        for term, weight in TERM_WEIGHTS.items():
            ours_t = float(ours.get(f"drawing_term_{term}_dagua_routed", 0.0))
            ref_t = float(ref.get(f"drawing_term_{term}_native", 0.0))
            contribs[term] = weight * (ref_t - ours_t)
        gap = float(ref_score) - float(ours_score)
        contribs_per_graph[name] = contribs
        gaps[name] = gap
        all_rows.append({"graph": name, "ours": ours, "ref": ref, "runtime_s": dt})
        recon = sum(contribs.values())
        print(
            f"[gap] {name}: ours={ours_score:.1f} ref={ref_score:.1f} "
            f"gap={gap:+.1f} (term-sum {recon:+.1f}) [{dt:.0f}s]",
            flush=True,
        )

    if not gaps:
        print("FATAL: no graphs scored", file=sys.stderr)
        return 1

    terms = list(TERM_WEIGHTS)
    print()
    header = f"| {'graph':32s} | {'gap':>6s} | " + " | ".join(f"{t:>10s}" for t in terms) + " |"
    print(header)
    print("|" + "-" * (len(header) - 2) + "|")
    for name in gaps:
        c = contribs_per_graph[name]
        row = f"| {name:32s} | {gaps[name]:+6.1f} | "
        row += " | ".join(f"{c[t]:+10.2f}" for t in terms) + " |"
        print(row)
    n = len(gaps)
    mean_contrib = {t: sum(contribs_per_graph[g][t] for g in gaps) / n for t in terms}
    mean_gap = sum(gaps.values()) / n
    row = f"| {'MEAN':32s} | {mean_gap:+6.1f} | "
    row += " | ".join(f"{mean_contrib[t]:+10.2f}" for t in terms) + " |"
    print(row)

    print("\nMean weighted gap contribution ranking (positive = dot wins that term):")
    for t, v in sorted(mean_contrib.items(), key=lambda kv: -kv[1]):
        share = 100.0 * v / mean_gap if mean_gap else 0.0
        print(f"  {t:12s} {v:+7.2f} pts  ({share:5.1f}% of mean gap)")

    args.json.write_text(
        json.dumps(
            {
                "seed": SEED,
                "engines": [ours_name, ref_name],
                "gaps": gaps,
                "mean_gap": mean_gap,
                "contribs_per_graph": contribs_per_graph,
                "mean_contrib": mean_contrib,
                "rows": all_rows,
            },
            indent=2,
            default=str,
        )
    )
    print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
