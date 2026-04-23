"""Sprint 9 preflight: held-out quality snapshot vs baseline.

Sprint 9's binding ship checklist (02_sprint_map.md L425) demands a final
benchmark table and suite-wide regression check before release. This
script is the cheap first half: run the 39-graph held-out suite on the
current branch HEAD, compare per-family composite scores against the
Sprint 4 snapshot (post-vectorization), flag any regressions > 5%, and
write a preflight report.

The expensive half -- Optuna hyperparameter search and full competitor
refresh -- is still outstanding and NOT attempted here.

Writes eval_output/native_algo/sprint_9_preflight/report.json.
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

OUT_PATH = Path("eval_output/native_algo/sprint_9_preflight/report.json")
BASELINE_PATH = Path("eval_output/native_algo/holdout_v1/metrics.json")


def score(g, pos):
    g.compute_node_sizes()
    if g.num_nodes <= 2000:
        m = full(pos, g.edge_index, node_sizes=g.node_sizes)
        return composite(m)
    m = quick(pos, g.edge_index, node_sizes=g.node_sizes)
    return composite_large(m)


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    graphs, manifest = make_holdout_suite()

    results = []
    print(f"running {len(graphs)} held-out graphs on HEAD ...", flush=True)
    for idx, tg in enumerate(graphs):
        family = manifest.entries[idx]["family"]
        g = tg.graph
        n = g.num_nodes
        t = time.perf_counter()
        pos = engine_layout(g, LayoutConfig(seed=42))
        wall = time.perf_counter() - t
        s = score(g, pos)
        results.append({"family": family, "n": n, "score": s, "wall": wall})
        print(
            f"  [{idx + 1:>2d}/{len(graphs)}] {family:<22} n={n:>5d} "
            f"score={s:6.2f} wall={wall:6.2f}s",
            flush=True,
        )

    # Load baseline for comparison.
    baseline = {}
    if BASELINE_PATH.exists():
        baseline_raw = json.loads(BASELINE_PATH.read_text())
        baseline = {f"{r['family']}_{r['n']}": r["score"] for r in baseline_raw.get("results", [])}

    # Per-family rollup (current).
    fam_agg = defaultdict(list)
    for r in results:
        fam_agg[r["family"]].append(r["score"])

    # Per-family rollup (baseline) when available.
    baseline_fam = defaultdict(list)
    if baseline:
        for r in json.loads(BASELINE_PATH.read_text()).get("results", []):
            baseline_fam[r["family"]].append(r["score"])

    family_summary = {}
    regressions = []
    for fam, scores in fam_agg.items():
        mean_now = sum(scores) / len(scores)
        mean_base = (
            sum(baseline_fam[fam]) / len(baseline_fam[fam]) if baseline_fam.get(fam) else None
        )
        delta_pct = None
        if mean_base is not None:
            delta_pct = 100 * (mean_now - mean_base) / max(abs(mean_base), 1e-3)
            if delta_pct < -5.0:
                regressions.append(
                    {
                        "family": fam,
                        "baseline": mean_base,
                        "current": mean_now,
                        "delta_pct": delta_pct,
                    }
                )
        family_summary[fam] = {
            "mean_now": mean_now,
            "mean_baseline": mean_base,
            "delta_pct": delta_pct,
            "n_graphs": len(scores),
        }

    mean_now_suite = sum(r["score"] for r in results) / len(results)
    mean_base_suite = None
    if baseline:
        vals = list(baseline.values())
        mean_base_suite = sum(vals) / len(vals)

    payload = {
        "n_graphs": len(results),
        "mean_score_now": mean_now_suite,
        "mean_score_baseline": mean_base_suite,
        "delta_pct": (
            100 * (mean_now_suite - mean_base_suite) / max(abs(mean_base_suite), 1e-3)
            if mean_base_suite is not None
            else None
        ),
        "regressions_gt_5pct": regressions,
        "family_summary": family_summary,
        "results": results,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {OUT_PATH}")

    # Summary.
    print("\n=== Sprint 9 preflight ===")
    print(
        f"suite mean: now={mean_now_suite:.2f}  "
        f"baseline={mean_base_suite if mean_base_suite else 'n/a'}"
    )
    if payload["delta_pct"] is not None:
        print(f"suite delta: {payload['delta_pct']:+.2f}%")
    print("\nPer-family deltas vs baseline:")
    for fam, s in sorted(family_summary.items(), key=lambda kv: (kv[1]["delta_pct"] or 0)):
        delta_str = f"{s['delta_pct']:+6.2f}%" if s["delta_pct"] is not None else "  n/a"
        base_str = f"{s['mean_baseline']:6.2f}" if s["mean_baseline"] is not None else "  ?  "
        print(
            f"  {fam:<25} n={s['n_graphs']:>2d}  "
            f"now={s['mean_now']:6.2f}  base={base_str}  delta={delta_str}"
        )

    if regressions:
        print(f"\nREGRESSIONS > 5% on {len(regressions)} families:")
        for r in regressions:
            print(
                f"  {r['family']:<25}  {r['baseline']:6.2f} -> {r['current']:6.2f}  "
                f"({r['delta_pct']:+.1f}%)"
            )
        print("\nSprint 9 ship-checklist item 'Suite-wide Pareto gate' BLOCKED.")
    else:
        print("\nNo >5% family regressions. Suite-wide Pareto floor clear for a tuning pass.")


if __name__ == "__main__":
    main()
