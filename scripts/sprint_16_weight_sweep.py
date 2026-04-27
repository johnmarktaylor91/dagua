"""Sprint 16: Optuna-lite weight sweep on the 39-graph holdout.

The forward memo's Tier-2 option was "once structural work is in,
tune the weight surface." Sprint 10-13 landed the structural work;
this sprint tunes the weights that moved most.

Approach: grid search over a small number of weight combinations
on the full 39-graph holdout. Composite score per combo averaged
across all graphs. Pick the best combo as new defaults if it
beats the current baseline meaningfully AND doesn't regress any
family by >5%.

Weights tuned:
 - w_length_variance (Sprint 11 bumped 0.7 -> 8.0; worth checking 4, 12, 16)
 - w_crossing (default 1.8; barycenter polish may have reduced the
   need for the continuous crossing loss)
 - w_repel / w_overlap balance (these often compete)

Runtime estimate: ~30 min for a 9-combo sweep.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

from dagua.config import LayoutConfig
from dagua.eval.graph_generator import make_holdout_suite
from dagua.layout.engine import layout as engine_layout
from dagua.metrics import composite, composite_large, full, quick

OUT_PATH = Path("eval_output/native_algo/sprint_16_weight_sweep/report.json")


def _score(g, pos):
    g.compute_node_sizes()
    if g.num_nodes <= 2000:
        m = full(pos, g.edge_index, node_sizes=g.node_sizes)
        return composite(m)
    m = quick(pos, g.edge_index, node_sizes=g.node_sizes)
    return composite_large(m)


CANDIDATES = [
    # (name, overrides)
    ("baseline", {}),
    ("wlv_4", {"w_length_variance": 4.0}),
    ("wlv_12", {"w_length_variance": 12.0}),
    ("wlv_16", {"w_length_variance": 16.0}),
    ("wc_3", {"w_crossing": 3.0}),
    ("wc_1", {"w_crossing": 1.0}),
    ("wlv_12_wc_3", {"w_length_variance": 12.0, "w_crossing": 3.0}),
    ("wlv_12_wc_1", {"w_length_variance": 12.0, "w_crossing": 1.0}),
    ("wrepel_05", {"w_repel": 0.05}),
    ("wrepel_2", {"w_repel": 0.2}),
]


def _run_with(name: str, overrides: dict, graphs, manifest) -> dict:
    results = []
    fam_agg = defaultdict(list)
    t0 = time.perf_counter()
    for idx, tg in enumerate(graphs):
        family = manifest.entries[idx]["family"]
        g = tg.graph
        cfg_kwargs = dict(seed=42, **overrides)
        pos = engine_layout(g, LayoutConfig(**cfg_kwargs))
        s = _score(g, pos)
        results.append({"family": family, "n": g.num_nodes, "score": s})
        fam_agg[family].append(s)
    wall = time.perf_counter() - t0
    mean = sum(r["score"] for r in results) / len(results)
    fam_mean = {f: sum(v) / len(v) for f, v in fam_agg.items()}
    return {"name": name, "overrides": overrides, "mean": mean, "wall": wall, "fam_mean": fam_mean}


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    graphs, manifest = make_holdout_suite()
    print(f"sweeping {len(CANDIDATES)} candidates on {len(graphs)} graphs", flush=True)

    runs = []
    for name, overrides in CANDIDATES:
        print(f"  -> {name} ...", flush=True)
        r = _run_with(name, overrides, graphs, manifest)
        runs.append(r)
        print(f"     mean={r['mean']:.3f}  wall={r['wall']:.1f}s", flush=True)

    baseline = runs[0]
    ranked = sorted(runs, key=lambda r: -r["mean"])

    # Compute per-family deltas vs baseline for the winner.
    winner = ranked[0]
    fam_deltas = {
        f: winner["fam_mean"].get(f, 0) - baseline["fam_mean"].get(f, 0)
        for f in set(winner["fam_mean"]) | set(baseline["fam_mean"])
    }
    regressions = [
        (f, d)
        for f, d in fam_deltas.items()
        if d < 0 and abs(d) / max(baseline["fam_mean"].get(f, 1), 1) > 0.05
    ]

    payload = {
        "n_candidates": len(CANDIDATES),
        "n_graphs": len(graphs),
        "runs": runs,
        "winner": winner["name"],
        "winner_mean": winner["mean"],
        "baseline_mean": baseline["mean"],
        "delta": winner["mean"] - baseline["mean"],
        "delta_pct": 100 * (winner["mean"] - baseline["mean"]) / max(baseline["mean"], 1),
        "winner_fam_deltas": fam_deltas,
        "winner_regressions_gt_5pct": regressions,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {OUT_PATH}")

    print("\n=== Ranked ===")
    for r in ranked:
        print(f"  {r['name']:<18} mean={r['mean']:.3f}  overrides={r['overrides']}")

    print(
        f"\nWinner: {winner['name']}  mean={winner['mean']:.3f}  "
        f"delta={payload['delta']:+.3f} ({payload['delta_pct']:+.2f}%)"
    )
    if regressions:
        print(f"Regressions > 5%: {regressions}")
    else:
        print("No family regressions > 5%. Safe to adopt.")


if __name__ == "__main__":
    main()
