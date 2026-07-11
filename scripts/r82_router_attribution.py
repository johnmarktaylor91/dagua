"""r82: attribute drawing defects to the ROUTER vs the PLACEMENT.

For each probe graph:
  - dagua positions: score composite_drawing under (a) dagua's actual router,
    (b) straight-line segments on the same positions. Delta (a)-(b) is what
    the router itself adds/removes per weighted term.
  - graphviz_dot positions: score (a) dot's native splines, (b) straight
    lines on dot's positions. Delta = what dot's spline router buys.

This separates "dot wins because its placement has fewer straight-line
defects" (not addressable by routing) from "dot wins because its router
improves on straight lines while ours degrades them" (addressable).

Usage:
  .venv/bin/python scripts/r82_router_attribution.py [--json /tmp/r82_attr.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from dagua.edges import BezierCurve, place_edge_labels, route_edges
from dagua.eval.competitors import get_competitor
from dagua.eval.drawing import routes_to_curves
from dagua.eval.graphs import get_test_graphs
from dagua.metrics import composite_drawing

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


def straight_curves(pos, edge_index) -> List[BezierCurve]:
    """Straight segments, node-center to node-center.

    Control points sit at the 1/3 and 2/3 chord points so tangents are
    well-defined everywhere (a p0,p0,p1,p1 degenerate bezier has a zero
    tangent at both endpoints, which would corrupt the port term).
    """
    out = []
    xs = pos[:, 0].tolist()
    ys = pos[:, 1].tolist()
    for e in range(edge_index.shape[1]):
        s = int(edge_index[0, e])
        t = int(edge_index[1, e])
        p0 = (xs[s], ys[s])
        p1 = (xs[t], ys[t])
        cp1 = (p0[0] + (p1[0] - p0[0]) / 3.0, p0[1] + (p1[1] - p0[1]) / 3.0)
        cp2 = (p0[0] + 2.0 * (p1[0] - p0[0]) / 3.0, p0[1] + 2.0 * (p1[1] - p0[1]) / 3.0)
        out.append(BezierCurve(p0, cp1, cp2, p1, routing="bezier"))
    return out


def score(graph, pos, curves, label_positions=None, edge_labels=None) -> Dict[str, Any]:
    return composite_drawing(
        pos,
        graph.edge_index,
        graph.node_sizes,
        curves,
        label_positions=label_positions,
        edge_labels=edge_labels,
        seed=0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=Path("/tmp/r82_attr.json"))
    args = parser.parse_args()

    by_name = {tg.name: tg for tg in get_test_graphs(max_nodes=500)}
    results: Dict[str, Any] = {}

    for name in PROBE_GRAPHS:
        tg = by_name.get(name)
        if tg is None:
            print(f"[attr] {name}: missing", flush=True)
            continue
        g = tg.graph
        g.compute_node_sizes()
        entry: Dict[str, Any] = {}

        for engine in ("dagua", "graphviz_dot"):
            comp = get_competitor(engine)
            if comp is None or not comp.available():
                continue
            res = comp.layout(g, timeout=TIMEOUT_S, seed=SEED)
            if res.pos is None:
                continue
            pos = res.pos

            straight = score(g, pos, straight_curves(pos, g.edge_index))
            if engine == "dagua":
                curves = route_edges(pos, g.edge_index, g.node_sizes, g.direction, g)
                lp = place_edge_labels(curves, pos, g.node_sizes, g.edge_labels, g)
                routed = score(g, pos, curves, lp, g.edge_labels)
                entry["dagua_routed"] = routed
                entry["dagua_straight"] = straight
            else:
                native_curves = routes_to_curves(res.routes, pos, g.edge_index)
                if native_curves is None:
                    continue
                native = score(
                    g,
                    pos,
                    native_curves,
                    label_positions=res.edge_label_positions,
                    edge_labels=g.edge_labels if res.edge_label_positions is not None else None,
                )
                entry["dot_native"] = native
                entry["dot_straight"] = straight
        results[name] = entry
        parts = []
        for key in ("dagua_routed", "dagua_straight", "dot_native", "dot_straight"):
            if key in entry:
                parts.append(f"{key}={entry[key]['composite_drawing']:.1f}")
        print(f"[attr] {name}: " + " ".join(parts), flush=True)

    # Router-added deltas per term (routed - straight), weighted.
    print("\nRouter effect per weighted term (positive = router IMPROVES on straight lines):")
    header = (
        f"| {'graph':32s} | {'who':6s} | {'total':>6s} | "
        + " | ".join(f"{t:>9s}" for t in TERM_WEIGHTS)
        + " |"
    )
    print(header)
    print("|" + "-" * (len(header) - 2) + "|")
    agg = {"dagua": {t: 0.0 for t in TERM_WEIGHTS}, "dot": {t: 0.0 for t in TERM_WEIGHTS}}
    agg_total = {"dagua": 0.0, "dot": 0.0}
    counts = {"dagua": 0, "dot": 0}
    for name, entry in results.items():
        for who, routed_key, straight_key in (
            ("dagua", "dagua_routed", "dagua_straight"),
            ("dot", "dot_native", "dot_straight"),
        ):
            if routed_key not in entry or straight_key not in entry:
                continue
            r = entry[routed_key]
            st = entry[straight_key]
            deltas = {
                t: TERM_WEIGHTS[t]
                * (float(r[f"drawing_term_{t}"]) - float(st[f"drawing_term_{t}"]))
                for t in TERM_WEIGHTS
            }
            total = float(r["composite_drawing"]) - float(st["composite_drawing"])
            row = f"| {name:32s} | {who:6s} | {total:+6.1f} | "
            row += " | ".join(f"{deltas[t]:+9.2f}" for t in TERM_WEIGHTS) + " |"
            print(row)
            for t in TERM_WEIGHTS:
                agg[who][t] += deltas[t]
            agg_total[who] += total
            counts[who] += 1
    for who in ("dagua", "dot"):
        if counts[who]:
            n = counts[who]
            row = f"| {'MEAN':32s} | {who:6s} | {agg_total[who] / n:+6.1f} | "
            row += " | ".join(f"{agg[who][t] / n:+9.2f}" for t in TERM_WEIGHTS) + " |"
            print(row)

    args.json.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
