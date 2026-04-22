"""Sprint 3 hybrid-classical: sgd2 warm-start hypothesis.

Runs the 30-graph held-out suite TWICE:
1. Baseline: default dagua_native (Sprint 2 state).
2. Warm-start: sgd2_multi runs for ~300 iter to get init positions,
   then dagua_native refines.

Writes eval_output/native_algo/sprint_3_warmstart/report.json with
per-graph deltas. Decision rule: keep warm-start if at least 3 families
improve >=5% composite AND no family regresses >3%.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from dagua.config import LayoutConfig
from dagua.eval.graph_generator import make_holdout_suite
from dagua.layout.engine import layout as engine_layout
from dagua.layout.ops.pipelines.dagua_native import layout_dagua_native_pipeline
from dagua.layout.ops.pipelines.sgd2_multi import layout_sgd2_multi_pipeline
from dagua.metrics import composite, composite_large, full, quick

OUT_PATH = Path("eval_output/native_algo/sprint_3_warmstart/report.json")

SGD2_WARM_STEPS = 300


def score_graph(g, pos, family):
    g.compute_node_sizes()
    if g.num_nodes <= 2000:
        m = full(pos, g.edge_index, node_sizes=g.node_sizes)
        return composite(m), "profile_small"
    m = quick(pos, g.edge_index, node_sizes=g.node_sizes)
    return composite_large(m), "profile_large"


def run_baseline(g, family):
    t = time.perf_counter()
    pos = engine_layout(g, LayoutConfig(seed=42))
    wall = time.perf_counter() - t
    s, prof = score_graph(g, pos, family)
    return s, wall, prof


def run_warmstart(g, family):
    # 1. sgd2_multi warm-start positions.
    t = time.perf_counter()
    try:
        init_pos = layout_sgd2_multi_pipeline(
            edge_index=g.edge_index,
            num_nodes=g.num_nodes,
            seed=42,
            steps=SGD2_WARM_STEPS,
        )
    except Exception as e:
        return None, time.perf_counter() - t, f"sgd2_warmstart_error: {e}", None

    # 2. Hand to dagua_native via init_pos.
    try:
        g.compute_node_sizes()
        pos = layout_dagua_native_pipeline(
            edge_index=g.edge_index,
            num_nodes=g.num_nodes,
            node_sizes=g.node_sizes,
            config=LayoutConfig(seed=42),
            init_pos=init_pos,
        )
    except Exception as e:
        return None, time.perf_counter() - t, f"native_refine_error: {e}", None

    wall = time.perf_counter() - t
    s, prof = score_graph(g, pos, family)
    return s, wall, None, prof


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    graphs, manifest = make_holdout_suite()

    results = []
    print(f"Running {len(graphs)} graphs, baseline + warm-start ...", flush=True)
    for idx, tg in enumerate(graphs):
        family = manifest.entries[idx]["family"]
        g = tg.graph
        n = g.num_nodes

        baseline_score, baseline_wall, baseline_prof = run_baseline(g, family)

        warm_score, warm_wall, warm_err, warm_prof = run_warmstart(g, family)

        delta = None
        if warm_score is not None:
            delta = warm_score - baseline_score
        results.append(
            {
                "index": idx,
                "family": family,
                "n": n,
                "baseline_score": baseline_score,
                "baseline_runtime_s": baseline_wall,
                "warmstart_score": warm_score,
                "warmstart_runtime_s": warm_wall,
                "warmstart_error": warm_err,
                "delta": delta,
            }
        )
        tag = (
            f"delta={delta:+.2f}"
            if delta is not None
            else f"ERR: {warm_err[:50] if warm_err else ''}"
        )
        print(
            f"  [{idx + 1:>2d}/{len(graphs)}] {family:<22} n={n:>5d} "
            f"baseline={baseline_score:5.1f} warm="
            f"{warm_score if warm_score is not None else '---':>5} {tag}"
        )

    # Family-level rollup.
    from collections import defaultdict

    family_agg = defaultdict(lambda: {"base": [], "warm": []})
    for r in results:
        if r["delta"] is None:
            continue
        family_agg[r["family"]]["base"].append(r["baseline_score"])
        family_agg[r["family"]]["warm"].append(r["warmstart_score"])

    family_summary = {}
    for fam, agg in family_agg.items():
        mean_base = sum(agg["base"]) / len(agg["base"])
        mean_warm = sum(agg["warm"]) / len(agg["warm"])
        family_summary[fam] = {
            "mean_baseline": mean_base,
            "mean_warmstart": mean_warm,
            "delta_pct": 100 * (mean_warm - mean_base) / max(mean_base, 1e-3),
            "n_graphs": len(agg["base"]),
        }

    payload = {
        "sgd2_warm_steps": SGD2_WARM_STEPS,
        "n_graphs": len(results),
        "results": results,
        "family_summary": family_summary,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {OUT_PATH}")

    print("\nPer-family delta:")
    for fam, s in sorted(family_summary.items(), key=lambda kv: -kv[1]["delta_pct"]):
        print(
            f"  {fam:<25} base={s['mean_baseline']:5.1f} warm={s['mean_warmstart']:5.1f} "
            f"delta={s['delta_pct']:+6.2f}% (n={s['n_graphs']})"
        )


if __name__ == "__main__":
    main()
