"""Sprint 0 Task 0.6: record baseline metrics on the MVP iteration suite.

Sprint 0 uses only the MVP harness graphs (chain_N, random_dag_N, diamond_N).
Sprint 0.5 expands to the full 25-graph iteration suite and the opaque
held-out set. Baseline is committed under seed=42 at the Sprint 0 commit SHA.

Writes eval_output/native_algo/baseline_sprint_0/metrics.json with per-graph
composite score, runtime, and key metric breakdowns.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from dagua.config import LayoutConfig
from dagua.eval.graphs import make_chain, make_diamond, make_random_dag
from dagua.layout.engine import layout as engine_layout
from dagua.metrics import composite, composite_large, full, quick

OUT_PATH = Path("eval_output/native_algo/baseline_sprint_0/metrics.json")

ITERATION_SUITE_MVP = [
    ("chain_10", lambda: make_chain(10, seed=42).graph),
    ("chain_100", lambda: make_chain(100, seed=42).graph),
    ("chain_500", lambda: make_chain(500, seed=42).graph),
    ("random_dag_100", lambda: make_random_dag(100, density=2.0, seed=42).graph),
    ("random_dag_200", lambda: make_random_dag(200, density=2.5, seed=42).graph),
    ("diamond_40", lambda: make_diamond(40, seed=42).graph),
    ("diamond_100", lambda: make_diamond(100, seed=42).graph),
]


def score_graph(name: str, factory) -> dict:
    g = factory()
    n = g.num_nodes

    t0 = time.perf_counter()
    pos = engine_layout(g, LayoutConfig(seed=42))
    wall = time.perf_counter() - t0

    g.compute_node_sizes()
    # Sprint 0 rule: full() + composite at N<=2000; quick() + composite_large at N>2000.
    if n <= 2000:
        m = full(pos, g.edge_index, node_sizes=g.node_sizes)
        score = composite(m)
        profile = "profile_small"
    else:
        m = quick(pos, g.edge_index, node_sizes=g.node_sizes)
        score = composite_large(m)
        profile = "profile_large"

    return {
        "name": name,
        "n": n,
        "runtime_s": wall,
        "score": score,
        "profile": profile,
        "dag_consistency": m.get("dag_consistency", 0.0),
        "edge_length_cv": m.get("edge_length_cv", 0.0),
        "depth_spearman_rho": m.get("depth_spearman_rho", 0.0),
        "overlap_count": m.get("overlap_count", 0),
        "edge_straightness_mean_deg": m.get("edge_straightness_mean_deg", 0.0),
        "crossing_rate": m.get("crossing_rate"),
    }


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = []
    print(f"Running {len(ITERATION_SUITE_MVP)} graphs ...", flush=True)
    for name, factory in ITERATION_SUITE_MVP:
        r = score_graph(name, factory)
        print(
            f"  {name:<18} n={r['n']:>5d} score={r['score']:6.2f} "
            f"runtime={r['runtime_s'] * 1000:6.1f}ms profile={r['profile']}"
        )
        results.append(r)

    # Note: git SHA is captured by the commit that lands this file, not
    # embedded here (detect-secrets flags hex strings).
    payload = {
        "seed": 42,
        "mvp_suite_note": (
            "Sprint 0 uses MVP harness graphs only (chain_N, random_dag_N, "
            "diamond_N). Sprint 0.5 expands to the full 25-graph iteration "
            "suite + the opaque held-out set."
        ),
        "results": results,
        "summary": {
            "n_graphs": len(results),
            "mean_score_small": (
                sum(r["score"] for r in results if r["profile"] == "profile_small")
                / max(1, sum(1 for r in results if r["profile"] == "profile_small"))
            ),
            "mean_score_large": (
                sum(r["score"] for r in results if r["profile"] == "profile_large")
                / max(1, sum(1 for r in results if r["profile"] == "profile_large"))
            ),
            "mean_runtime_ms": 1000 * sum(r["runtime_s"] for r in results) / len(results),
        },
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
