"""r80-S6 proof run: composite_drawing across 10 graphs x 4 engines.

Runs a representative 10-graph mix (3 layered DAGs, 3 undirected community,
2 clustered, 2 weighted) through dagua, graphviz_dot, graphviz_sfdp, and
elk_layered, then reports composite_drawing plus its component terms for
BOTH variants:

- native:       the engine's own captured routing (splines/bendPoints) and,
                when available, its own edge-label anchors
- dagua_routed: the engine's positions with DAGUA's router + label placement
                applied (the deployment-relevant combined-system score; the
                only fair variant for force engines with no native routing)

Usage:
  .venv/bin/python scripts/r80_drawing_probe.py \
      [--output .project-context/research/r79_native/P9_DRAWING_BASELINE.md]
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dagua.edges import place_edge_labels, route_edges
from dagua.eval.benchmark import _drawing_metrics
from dagua.eval.competitors import get_competitor
from dagua.eval.drawing import native_route_coverage
from dagua.eval.graphs import get_test_graphs

SEED = 42
TIMEOUT_S = 120.0

PROBE_GRAPHS: List[tuple] = [
    # (name, category)
    ("citation_dag_300", "layered DAG"),
    ("random_dag_200", "layered DAG"),
    ("long_skip_only_24", "layered DAG"),
    ("r79_undirected_sbm_low_mix_4x25", "undirected community"),
    ("chung_lu_150", "undirected community"),
    ("protein_ppi_200", "undirected community"),
    ("clustered_medium_5x20", "clustered"),
    ("r79_nested_clusters_3x2x10", "clustered"),
    ("heavy_tail_weights_50", "weighted"),
    ("r79_weighted_community_4x18", "weighted"),
]

ENGINES = ["dagua", "graphviz_dot", "graphviz_sfdp", "elk_layered"]


def probe_one(graph, engine_name: str) -> Optional[Dict[str, Any]]:
    """Run one engine on one graph and score both drawing variants.

    Parameters
    ----------
    graph : DaguaGraph
        Graph under test (node sizes computed).
    engine_name : str
        Registered competitor name.

    Returns
    -------
    Dict[str, Any] | None
        Drawing metric payload plus bookkeeping, or ``None`` on failure.
    """
    competitor = get_competitor(engine_name)
    if competitor is None or not competitor.available():
        return {"error": "not installed"}
    start = time.perf_counter()
    result = competitor.layout(graph, timeout=TIMEOUT_S, seed=SEED)
    if result.pos is None:
        return {"error": result.error or "layout failed"}

    pos = result.pos
    curves = route_edges(pos, graph.edge_index, graph.node_sizes, graph.direction, graph)
    label_positions = place_edge_labels(curves, pos, graph.node_sizes, graph.edge_labels, graph)
    payload = _drawing_metrics(
        graph,
        pos,
        curves,
        label_positions,
        native_routes=result.routes,
        native_edge_label_positions=result.edge_label_positions,
    )
    num_edges = int(graph.edge_index.shape[1]) if graph.edge_index.numel() > 0 else 0
    payload["probe_runtime_s"] = time.perf_counter() - start
    payload["native_coverage"] = native_route_coverage(result.routes, num_edges)
    return payload


def fmt(value: Any, width: int = 6, decimals: int = 1) -> str:
    """Format one probe cell."""
    if value is None:
        return " " * (width - 2) + "--"
    if isinstance(value, bool):
        return ("yes" if value else "no").rjust(width)
    if isinstance(value, float):
        return f"{value:.{decimals}f}".rjust(width)
    return str(value).rjust(width)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".project-context/research/r79_native/P9_DRAWING_BASELINE.md"),
    )
    args = parser.parse_args()

    by_name = {tg.name: tg for tg in get_test_graphs(max_nodes=500)}
    missing = [name for name, _ in PROBE_GRAPHS if name not in by_name]
    if missing:
        print(f"FATAL: probe graphs not in corpus: {missing}", file=sys.stderr)
        return 1

    rows: List[Dict[str, Any]] = []
    for graph_name, category in PROBE_GRAPHS:
        tg = by_name[graph_name]
        tg.graph.compute_node_sizes()
        n = tg.graph.num_nodes
        e = int(tg.graph.edge_index.shape[1]) if tg.graph.edge_index.numel() > 0 else 0
        for engine in ENGINES:
            print(f"[probe] {graph_name} x {engine} ...", flush=True)
            payload = probe_one(tg.graph, engine) or {}
            rows.append(
                {
                    "graph": graph_name,
                    "category": category,
                    "n": n,
                    "e": e,
                    "engine": engine,
                    **payload,
                }
            )

    header = (
        f"| {'graph':32s} | {'engine':13s} | {'nat':>6s} | {'dgr':>6s} "
        f"| {'natX':>6s} | {'dgrX':>6s} | {'enX':>5s} | {'port':>6s} "
        f"| {'curv':>6s} | {'bend':>5s} | {'lblN':>4s} | {'cov':>4s} |"
    )
    sep = "|" + "-" * (len(header) - 2) + "|"
    lines = [header, sep]
    for row in rows:
        if "error" in row:
            lines.append(f"| {row['graph']:32s} | {row['engine']:13s} | ERROR: {row['error']}")
            continue
        native_score = row.get("composite_drawing_native")

        def pick(stem: str) -> Any:
            """Prefer the native-variant component when native was captured."""
            if native_score is not None:
                return row.get(f"{stem}_native")
            return row.get(f"{stem}_dagua_routed")

        lines.append(
            f"| {row['graph']:32s} | {row['engine']:13s} "
            f"| {fmt(native_score)} "
            f"| {fmt(row.get('composite_drawing_dagua_routed'))} "
            f"| {fmt(row.get('drawing_crossing_rate_native'), decimals=3)} "
            f"| {fmt(row.get('drawing_crossing_rate_dagua_routed'), decimals=3)} "
            f"| {fmt(pick('drawing_edge_node_crossings'), width=5)} "
            f"| {fmt(pick('drawing_port_angular_deg'))} "
            f"| {fmt(pick('drawing_curvature_cv'), decimals=2)} "
            f"| {fmt(pick('drawing_bend_mean_per_edge'), width=5, decimals=2)} "
            f"| {fmt(pick('drawing_label_node_overlaps'), width=4)} "
            f"| {fmt(row.get('native_coverage'), width=4, decimals=2)} |"
        )

    table = "\n".join(lines)
    print()
    print(table)

    legend = (
        "Legend: nat = composite_drawing on the engine's NATIVE routing; "
        "dgr = composite_drawing on the engine's positions with DAGUA's router "
        "('external positions + dagua routing'); natX/dgrX = routed crossing "
        "rate per variant; enX = edge-node crossings; port = port angular "
        "resolution (deg); curv = curvature CV; bend = mean bends/edge; "
        "lblN = edge-label-vs-node overlaps; cov = fraction of edges with "
        "captured native routes. dagua has no 'native vs dagua-routed' split "
        "(its native routing IS the dagua router), so nat is blank and dgr is "
        "its full-drawing score. Component columns describe the native "
        "variant when captured, else the dagua-routed variant."
    )

    doc = [
        "# P9: Full-drawing quality baseline (r80-S6 proof run)",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Seed: {SEED}. Engines: {', '.join(ENGINES)}. "
        f"Graphs: 10 (3 layered DAG, 3 undirected community, 2 clustered, 2 weighted).",
        "Scorer: dagua.metrics.composite_drawing (0-100; weights: crossings 30,",
        "edge-node 20, labels 15, ports 12, overlap 10, curvature 8, bends 5).",
        "",
        "```",
        table,
        "```",
        "",
        legend,
        "",
        "## Observations",
        "",
        "(filled in by the sprint report)",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(doc))
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
