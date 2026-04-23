"""Sprint 6 exit audit: edge-node crossings, heuristic vs differentiable.

Runs the 39-graph held-out suite TWICE -- once with
``LayoutConfig(edge_routing="heuristic")``, once with
``LayoutConfig(edge_routing="differentiable")`` -- and compares the
``edge_node_crossings`` count + ``edge_node_crossing_rate`` per graph.

Exit criterion (02_sprint_map.md L253):
  Held-out visual audit: edge-node crossings drop >=30% vs Sprint 5.

"Sprint 5" here means the heuristic path (no CP refinement), which is
what ``edge_routing="heuristic"`` produces. We report both absolute and
relative drop per graph and aggregate across the suite.

Writes eval_output/native_algo/sprint_6_edge_routing/report.json.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import dagua
from dagua.config import LayoutConfig
from dagua.eval.graph_generator import make_holdout_suite
from dagua.metrics import edge_node_crossing_count

OUT_PATH = Path("eval_output/native_algo/sprint_6_edge_routing/report.json")


def _route_with_mode(g, mode: str):
    """Run the full draw-equivalent pipeline and return (pos, curves, ns, ei)."""
    cfg = LayoutConfig(seed=42, edge_routing=mode)
    # Run layout + route + (maybe optimize) without actually rendering.
    pos = dagua.layout(g, cfg)
    g.compute_node_sizes()
    from dagua.edges import route_edges

    curves = route_edges(pos, g.edge_index, g.node_sizes, g.direction, g)
    if mode == "differentiable":
        from dagua.layout.edge_optimization import optimize_edges

        curves = optimize_edges(curves, pos, g.edge_index, g.node_sizes, cfg, g)
    return pos, curves, g.node_sizes, g.edge_index


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    graphs, manifest = make_holdout_suite()

    results = []
    print(f"Running {len(graphs)} graphs, heuristic + differentiable edge routing ...", flush=True)
    for idx, tg in enumerate(graphs):
        family = manifest.entries[idx]["family"]
        g = tg.graph
        n = g.num_nodes

        t = time.perf_counter()
        pos_h, curves_h, ns_h, ei_h = _route_with_mode(g, "heuristic")
        wall_h = time.perf_counter() - t
        m_h = edge_node_crossing_count(curves_h, pos_h, ns_h, ei_h)

        # Fresh layout (don't reuse cached pos from the heuristic run).
        t = time.perf_counter()
        pos_d, curves_d, ns_d, ei_d = _route_with_mode(g, "differentiable")
        wall_d = time.perf_counter() - t
        m_d = edge_node_crossing_count(curves_d, pos_d, ns_d, ei_d)

        h_count = int(m_h["edge_node_crossings"])
        d_count = int(m_d["edge_node_crossings"])
        drop_pct = 100.0 * (h_count - d_count) / max(h_count, 1) if h_count > 0 else 0.0

        results.append(
            {
                "index": idx,
                "family": family,
                "n": n,
                "heuristic_crossings": h_count,
                "differentiable_crossings": d_count,
                "drop_pct": drop_pct,
                "heuristic_runtime_s": wall_h,
                "differentiable_runtime_s": wall_d,
            }
        )
        print(
            f"  [{idx + 1:>2d}/{len(graphs)}] {family:<22} n={n:>5d} "
            f"heuristic={h_count:>4d} differentiable={d_count:>4d} drop={drop_pct:+6.1f}%",
            flush=True,
        )

    # Aggregate.
    fam_agg = defaultdict(lambda: {"h": [], "d": []})
    for r in results:
        fam_agg[r["family"]]["h"].append(r["heuristic_crossings"])
        fam_agg[r["family"]]["d"].append(r["differentiable_crossings"])

    family_summary = {}
    for fam, a in fam_agg.items():
        h = sum(a["h"])
        d = sum(a["d"])
        family_summary[fam] = {
            "sum_heuristic": h,
            "sum_differentiable": d,
            "drop_pct": 100.0 * (h - d) / max(h, 1) if h > 0 else 0.0,
            "n_graphs": len(a["h"]),
        }

    total_h = sum(r["heuristic_crossings"] for r in results)
    total_d = sum(r["differentiable_crossings"] for r in results)
    total_drop_pct = 100.0 * (total_h - total_d) / max(total_h, 1) if total_h > 0 else 0.0

    payload = {
        "n_graphs": len(results),
        "total_heuristic_crossings": total_h,
        "total_differentiable_crossings": total_d,
        "total_drop_pct": total_drop_pct,
        "results": results,
        "family_summary": family_summary,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {OUT_PATH}")

    print("\nPer-family drop:")
    for fam, s in sorted(family_summary.items(), key=lambda kv: -kv[1]["drop_pct"]):
        print(
            f"  {fam:<25} heuristic={s['sum_heuristic']:>4d} "
            f"differentiable={s['sum_differentiable']:>4d} drop={s['drop_pct']:+6.1f}% "
            f"(n={s['n_graphs']})"
        )

    print(f"\nTOTAL: heuristic={total_h}, differentiable={total_d}, drop={total_drop_pct:+.1f}%")
    if total_drop_pct >= 30.0:
        print("PASS: >=30% edge-node crossing drop")
    else:
        print(f"FAIL: {total_drop_pct:.1f}% < 30% target")


if __name__ == "__main__":
    main()
