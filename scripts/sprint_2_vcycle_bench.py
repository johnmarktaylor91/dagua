"""Sprint 2 head-to-head: V-cycle (ops) vs legacy multilevel_layout.

Runs the same large-graph set through both paths and reports composite
score + runtime. Writes eval_output/native_algo/sprint_2_vcycle/report.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from dagua.config import LayoutConfig
from dagua.eval.graphs import make_chain, make_random_dag, make_tree
from dagua.layout.engine import layout as engine_layout
from dagua.metrics import composite, composite_large, full, quick

OUT_PATH = Path("eval_output/native_algo/sprint_2_vcycle/report.json")

GRAPHS = [
    ("chain_10000", lambda: make_chain(10000, seed=42).graph),
    ("chain_25000", lambda: make_chain(25000, seed=42).graph),
    ("random_dag_10000", lambda: make_random_dag(10000, density=1.5, seed=42).graph),
    ("random_dag_25000", lambda: make_random_dag(25000, density=1.5, seed=42).graph),
    ("tree_10000", lambda: make_tree(10000, branching=3, seed=42).graph),
    ("tree_25000", lambda: make_tree(25000, branching=3, seed=42).graph),
]


def score(g, pos):
    g.compute_node_sizes()
    if g.num_nodes <= 2000:
        m = full(pos, g.edge_index, node_sizes=g.node_sizes)
        return composite(m), "profile_small"
    m = quick(pos, g.edge_index, node_sizes=g.node_sizes)
    return composite_large(m), "profile_large"


def run_one(name, factory, algorithm):
    g = factory()
    cfg = LayoutConfig(seed=42, steps=60, algorithm=algorithm)
    t = time.perf_counter()
    try:
        pos = engine_layout(g, cfg)
        wall = time.perf_counter() - t
        s, profile = score(g, pos)
        return {
            "graph": name,
            "algorithm": algorithm or "ops_native",
            "n": g.num_nodes,
            "score": s,
            "runtime_s": wall,
            "profile": profile,
            "error": None,
        }
    except Exception as e:
        wall = time.perf_counter() - t
        return {
            "graph": name,
            "algorithm": algorithm or "ops_native",
            "n": g.num_nodes,
            "score": None,
            "runtime_s": wall,
            "error": f"{type(e).__name__}: {e}",
        }


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for name, factory in GRAPHS:
        for algorithm in (None, "_legacy"):
            tag = "ops_native" if algorithm is None else "legacy_mlvl"
            print(f"  running {name} via {tag} ...", flush=True)
            r = run_one(name, factory, algorithm)
            results.append(r)
            if r["error"]:
                print(f"    ERROR: {r['error'][:200]}")
            else:
                print(f"    n={r['n']:>6d} score={r['score']:6.2f} runtime={r['runtime_s']:6.1f}s")

    # Pair by graph name to report deltas.
    by_graph = {}
    for r in results:
        by_graph.setdefault(r["graph"], {})[r["algorithm"]] = r
    pairs = []
    for name, variants in by_graph.items():
        ops = variants.get("ops_native", {})
        leg = variants.get("_legacy", {})
        if not ops or not leg:
            continue
        if ops.get("error") or leg.get("error"):
            pairs.append(
                {
                    "graph": name,
                    "ops_error": ops.get("error"),
                    "legacy_error": leg.get("error"),
                }
            )
            continue
        pairs.append(
            {
                "graph": name,
                "n": ops["n"],
                "ops_score": ops["score"],
                "legacy_score": leg["score"],
                "score_delta": ops["score"] - leg["score"],
                "ops_runtime_s": ops["runtime_s"],
                "legacy_runtime_s": leg["runtime_s"],
                "runtime_ratio_ops_over_legacy": ops["runtime_s"] / max(leg["runtime_s"], 0.001),
            }
        )

    payload = {"results": results, "pairs": pairs}
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {OUT_PATH}")
    print("\nPair summary:")
    for p in pairs:
        if p.get("ops_error") or p.get("legacy_error"):
            print(f"  {p['graph']}: ERROR")
            continue
        print(
            f"  {p['graph']:<22} n={p['n']:>6d} "
            f"score ops={p['ops_score']:5.1f} legacy={p['legacy_score']:5.1f} "
            f"delta={p['score_delta']:+5.2f}  "
            f"runtime ops={p['ops_runtime_s']:6.1f}s legacy={p['legacy_runtime_s']:6.1f}s "
            f"ratio={p['runtime_ratio_ops_over_legacy']:.2f}x"
        )


if __name__ == "__main__":
    main()
