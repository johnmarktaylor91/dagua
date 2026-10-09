"""Select ~200 stratified pairs from scored layouts (stdlib only, hub-safe).

Input: graphs.json and layouts-*.jsonl from fp_layout.py. Output: pairs.json
(render list) and pairs_manifest_base.json (stratum, metric scores, names).
The metric is the r83 honest-ruler composite (``composite_auto``, higher is
better, tie band 0.5) already stored per layout.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

TIE_BAND = 0.5
NEAR_TIE = 2.0
CONTROL_MARGIN = 25.0
HEADLINE_MAX = 15.0
PER_GRAPH_CAP = 4
STRUCT_TERMS = ("ksm_score", "neighborhood_preservation_score", "edge_crossing_score")
TARGETS = {
    "near_tie": 50,
    "term_split": 40,
    "headline_dagua_ahead": 25,
    "headline_dagua_behind": 25,
    "clear_control": 30,
    "dagua_branch_variant": 30,
}
# The July headline engines (scripts/r83_rescore.py CLASSICAL_ENGINES).
JULY_ENGINES = {
    "dagre", "elk_layered", "graphviz_dot", "graphviz_neato", "graphviz_sfdp",
    "igraph_kamada_kawai", "igraph_sugiyama", "nx_spring",
}


def load(paths: Sequence[Path]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Index rows by (graph, 'engine@tag')."""
    out = {}
    for path in paths:
        tag = path.stem.split("layouts-", 1)[1]
        for line in path.open():
            row = json.loads(line)
            if row.get("pos") is not None and row.get("score") is not None:
                out[(row["graph"], f"{row['engine']}@{tag}")] = row
    return out


def term_votes(a: Dict[str, Any], b: Dict[str, Any]) -> int:
    """Return +1 per structural term favouring a, -1 per term favouring b."""
    v = 0
    for t in STRUCT_TERMS:
        ta, tb = (a.get("terms") or {}).get(t), (b.get("terms") or {}).get(t)
        if ta is None or tb is None or abs(ta - tb) < 1e-9:
            continue
        v += 1 if ta > tb else -1
    return v


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Build the pair list."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", type=Path, required=True)
    ap.add_argument("--layouts", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--alt-graphs", type=Path, nargs="*", default=[],
                    help="TAG=path graphs.json of other branches; variants need identical topology")
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    graphs = json.loads(args.graphs.read_text())
    rows = load(args.layouts)
    alt = {}
    for spec in args.alt_graphs:
        tag, path = str(spec).split("=", 1)
        alt[tag] = json.loads(Path(path).read_text())
    by_graph = defaultdict(dict)
    for (g, lay), row in rows.items():
        by_graph[g][lay] = row

    cands: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for g, lays in by_graph.items():
        d = lays.get("dagua@main")
        if d is None:
            continue
        for lay, row in lays.items():
            eng, tag = lay.split("@")
            if lay == "dagua@main":
                continue
            delta = d["score"] - row["score"]
            rec = {"graph": g, "x": "dagua@main", "y": lay, "delta": delta,
                   "xs": d["score"], "ys": row["score"], "votes": term_votes(d, row)}
            if eng == "dagua" and tag != "main":
                other = alt.get(tag, {}).get(g)
                same_topology = other is not None and other["edges"] == graphs[g]["edges"] \
                    and other["node_sizes"] == graphs[g]["node_sizes"]
                if same_topology and d["pos"] != row["pos"]:
                    cands["dagua_branch_variant"].append(rec)
                continue
            if eng == "dagua":
                continue
            comp_dir = 0 if abs(delta) <= TIE_BAND else (1 if delta > 0 else -1)
            if abs(delta) <= NEAR_TIE:
                cands["near_tie"].append(rec)
            if abs(delta) > NEAR_TIE and comp_dir * rec["votes"] <= -2:
                cands["term_split"].append(rec)
            if eng in JULY_ENGINES:
                if NEAR_TIE < delta <= HEADLINE_MAX:
                    cands["headline_dagua_ahead"].append(rec)
                if -HEADLINE_MAX <= delta < -NEAR_TIE:
                    cands["headline_dagua_behind"].append(rec)
            if abs(delta) > CONTROL_MARGIN:
                cands["clear_control"].append(rec)

    chosen: List[Dict[str, Any]] = []
    used = set()
    per_graph: Dict[str, int] = defaultdict(int)
    # rarest strata first so they are not starved
    for stratum in sorted(TARGETS, key=lambda s: len(cands[s])):
        pool = [c for c in cands[stratum] if (c["graph"], c["y"]) not in used]
        rng.shuffle(pool)
        by_engine = defaultdict(list)
        for c in pool:
            by_engine[c["y"]].append(c)
        order = sorted(by_engine)
        picked = 0
        while picked < TARGETS[stratum] and any(by_engine.values()):
            for lay in order:
                if picked >= TARGETS[stratum]:
                    break
                while by_engine[lay]:
                    c = by_engine[lay].pop()
                    if per_graph[c["graph"]] >= PER_GRAPH_CAP or (c["graph"], c["y"]) in used:
                        continue
                    used.add((c["graph"], c["y"]))
                    per_graph[c["graph"]] += 1
                    chosen.append({**c, "stratum": stratum})
                    picked += 1
                    break
        print(f"{stratum}: candidates {len(cands[stratum])} picked {picked}/{TARGETS[stratum]}")

    rng.shuffle(chosen)
    pairs, base = [], []
    for i, c in enumerate(chosen):
        pid = f"fp{i + 1:03d}"
        pairs.append({"pair_id": pid, "graph": c["graph"],
                      "x": {"layout": c["x"]}, "y": {"layout": c["y"]}})
        base.append({"pair_id": pid, "graph": c["graph"], "source": graphs[c["graph"]]["source"],
                     "n": graphs[c["graph"]]["n"], "e": graphs[c["graph"]]["e"],
                     "directed": graphs[c["graph"]]["directed"],
                     "x": c["x"], "y": c["y"], "stratum": c["stratum"],
                     "score_x": round(c["xs"], 3), "score_y": round(c["ys"], 3),
                     "delta_x_minus_y": round(c["delta"], 3), "struct_votes_x": c["votes"]})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pairs.json").write_text(json.dumps(pairs))
    (args.out / "pairs_manifest_base.json").write_text(json.dumps(base, indent=1))
    print(f"{len(pairs)} pairs over {len(per_graph)} graphs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
