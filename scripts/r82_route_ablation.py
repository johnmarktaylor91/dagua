"""r82: fast routing-variant ablation on CACHED positions.

Layouts are expensive (minutes); routing + scoring is cheap (seconds). This
harness caches per-(graph, engine) positions + native routes once, then
scores arbitrary routing variants against the cache so routing changes can
be iterated quickly and deterministically on IDENTICAL positions.

Usage:
  # once (slow): build the position cache
  .venv/bin/python scripts/r82_route_ablation.py --build-cache

  # fast: score the current router against the cache
  .venv/bin/python scripts/r82_route_ablation.py --label after_fix

Cache: /tmp/r82_pos_cache.pt  {graph: {engine: {"pos": tensor, "routes": ...}}}
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict

from dagua.edges import place_edge_labels, route_edges
from dagua.eval.competitors import get_competitor
from dagua.eval.drawing import routes_to_curves
from dagua.eval.graphs import get_test_graphs
from dagua.metrics import composite_drawing

SEED = 42
TIMEOUT_S = 120.0
CACHE = Path("/tmp/r82_pos_cache.pkl")

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

ENGINES = ["dagua", "graphviz_dot"]


def graphs_by_name():
    return {tg.name: tg for tg in get_test_graphs(max_nodes=500)}


def build_cache() -> int:
    by_name = graphs_by_name()
    cache: Dict[str, Dict[str, Any]] = {}
    for name in PROBE_GRAPHS:
        g = by_name[name].graph
        g.compute_node_sizes()
        cache[name] = {}
        for engine in ENGINES:
            comp = get_competitor(engine)
            if comp is None or not comp.available():
                continue
            res = comp.layout(g, timeout=TIMEOUT_S, seed=SEED)
            if res.pos is None:
                continue
            cache[name][engine] = {
                "pos": res.pos.detach().cpu(),
                "routes": res.routes,
                "edge_label_positions": res.edge_label_positions,
            }
            print(f"[cache] {name} x {engine}: ok", flush=True)
    with CACHE.open("wb") as fh:
        pickle.dump(cache, fh)
    print(f"[cache] wrote {CACHE}")
    return 0


def score_variants(label: str, json_out: Path) -> int:
    with CACHE.open("rb") as fh:
        cache = pickle.load(fh)
    by_name = graphs_by_name()

    rows: Dict[str, Dict[str, Any]] = {}
    print(f"\n=== routing variant: {label} ===")
    for name in PROBE_GRAPHS:
        if name not in cache:
            continue
        g = by_name[name].graph
        g.compute_node_sizes()
        entry: Dict[str, Any] = {}
        for engine in ENGINES:
            if engine not in cache[name]:
                continue
            pos = cache[name][engine]["pos"]
            curves = route_edges(pos, g.edge_index, g.node_sizes, g.direction, g)
            lp = place_edge_labels(curves, pos, g.node_sizes, g.edge_labels, g)
            scored = composite_drawing(
                pos,
                g.edge_index,
                g.node_sizes,
                curves,
                label_positions=lp,
                edge_labels=g.edge_labels,
                seed=0,
            )
            entry[f"{engine}_dgr"] = scored
            if engine == "graphviz_dot":
                native_curves = routes_to_curves(cache[name][engine]["routes"], pos, g.edge_index)
                if native_curves is not None:
                    nl = cache[name][engine]["edge_label_positions"]
                    native = composite_drawing(
                        pos,
                        g.edge_index,
                        g.node_sizes,
                        native_curves,
                        label_positions=nl,
                        edge_labels=g.edge_labels if nl is not None else None,
                        seed=0,
                    )
                    entry["dot_native"] = native
        rows[name] = entry
        parts = []
        for key in ("dagua_dgr", "graphviz_dot_dgr", "dot_native"):
            if key in entry:
                parts.append(f"{key}={entry[key]['composite_drawing']:.1f}")
        print(f"[{label}] {name}: " + " ".join(parts), flush=True)

    # Summary vs dot native
    total_gap = 0.0
    n = 0
    for name, entry in rows.items():
        if "dagua_dgr" in entry and "dot_native" in entry:
            gap = entry["dot_native"]["composite_drawing"] - entry["dagua_dgr"]["composite_drawing"]
            total_gap += gap
            n += 1
    if n:
        print(f"[{label}] mean gap vs dot-native over {n} graphs: {total_gap / n:+.2f}")

    payload = {name: {k: v for k, v in entry.items()} for name, entry in rows.items()}
    json_out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[{label}] wrote {json_out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-cache", action="store_true")
    parser.add_argument("--label", type=str, default="current")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    if args.build_cache:
        return build_cache()
    json_out = args.json or Path(f"/tmp/r82_ablation_{args.label}.json")
    return score_variants(args.label, json_out)


if __name__ == "__main__":
    sys.exit(main())
