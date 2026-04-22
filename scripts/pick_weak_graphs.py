"""Sprint 0.5 Task 6: pick the k weakest graphs for iteration.

Reads a metrics.json (from ``scripts/sprint_0_baseline.py``, the benchmark
CLI, or ``iterate_native.sh`` output) and selects the k graphs where the
current layout is most likely to benefit from work. The per-sprint
iteration loop (10_iteration_loop.md) uses this to pick 5 weak graphs plus
3 random ones each cycle.

Priority for "weak" (per 10_iteration_loop.md):
1. Graphs where Dagua is DOMINATED by a competitor on both (quality, runtime).
   Requires a head-to-head metrics file; falls back to (2) when absent.
2. Graphs with the lowest composite score (raw quality gap).
3. Graphs with the largest runtime deviation above the family median.
4. Graphs in a priority family (P0) with any regression from Sprint 0.

Spread: at most 2 graphs from any one family.

Usage:
    python scripts/pick_weak_graphs.py <metrics.json> [--k=5]
    # Returns a space-separated list of graph specs for iterate_native.sh
    # consumption, e.g. "chain_100 random_dag_200 diamond_40 ..."
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def _family_of(result: Dict[str, Any]) -> str:
    # New salt-derived metrics always carry 'family'.
    if "family" in result:
        return str(result["family"])
    # Sprint 0 baseline: derive family from name prefix.
    name = str(result.get("name", ""))
    for prefix in ("chain_", "random_dag_", "diamond_", "tree_", "grid_"):
        if name.startswith(prefix):
            return prefix.rstrip("_")
    return "unknown"


def _graph_spec(result: Dict[str, Any]) -> str:
    """Return the spec string iterate_native.sh consumes (chain_N, diamond_N, ...).

    For salt-derived graphs we can't reconstruct a deterministic spec without
    the salt, so fall back to the holdout/rolling tag + index.
    """
    if "name" in result:
        return str(result["name"])
    family = _family_of(result)
    idx = result.get("index", "?")
    n = result.get("n", "?")
    return f"{family}[idx={idx},n={n}]"


def pick_weak(metrics_path: Path, k: int = 5) -> List[Dict[str, Any]]:
    """Pick k weakest graphs, normalizing by family to avoid rubric artifacts.

    Raw composite is DAG-biased (dag_consistency weight 25, crossing weight
    10, etc.), so undirected families like small_world floor below ~30 even
    at perfect layout. Without normalization, `pick_weak` would always
    return the same family's graphs (per the round-2 adversarial finding).

    Strategy (revised): rank by `(score - family_max) / family_span` so a
    graph is "weak" relative to what its family actually achieves in the
    metrics file. Families with only one entry use raw score. Cross-family
    spread (at-most-2-per-family) is still enforced.

    Future (Sprint 1+): once head-to-head competitor metrics are available,
    rank by Pareto dominance instead (dagua_score - best_competitor_score).
    """
    payload = json.loads(metrics_path.read_text())
    results = payload.get("results", [])
    if not results:
        raise SystemExit(f"No 'results' in {metrics_path}")

    scored = [r for r in results if r.get("error") is None and r.get("score") is not None]
    if not scored:
        raise SystemExit(f"No non-error results in {metrics_path}")

    # Compute per-family max score for normalization.
    by_family: Dict[str, List[float]] = defaultdict(list)
    for r in scored:
        by_family[_family_of(r)].append(r["score"])
    family_max = {f: max(scores) for f, scores in by_family.items()}

    def family_normalized(r: Dict[str, Any]) -> float:
        """Return a rank key where lower = weaker within family.

        raw_gap_to_family_max = family_max[f] - r['score']. Larger gap = weaker.
        Negate so that sort ascending = weakest first.
        """
        fmax = family_max[_family_of(r)]
        gap = fmax - r["score"]
        return -gap  # ascending sort = most-negative (largest gap) first

    scored.sort(key=family_normalized)

    # Spread: at most 2 per family.
    per_family: Dict[str, int] = defaultdict(int)
    picks: List[Dict[str, Any]] = []
    for r in scored:
        family = _family_of(r)
        if per_family[family] >= 2:
            continue
        picks.append(r)
        per_family[family] += 1
        if len(picks) >= k:
            break

    return picks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_path", type=Path)
    parser.add_argument("--k", type=int, default=5, help="Number of weak graphs to pick")
    parser.add_argument(
        "--format",
        choices=["specs", "json", "table"],
        default="table",
        help="specs = newline-separated graph specs for iterate_native.sh; "
        "json = full records; table = human-readable (default).",
    )
    args = parser.parse_args()

    picks = pick_weak(args.metrics_path, k=args.k)

    if args.format == "specs":
        for p in picks:
            print(_graph_spec(p))
    elif args.format == "json":
        print(json.dumps(picks, indent=2))
    else:
        print(f"Top {len(picks)} weakest graphs from {args.metrics_path}:")
        for p in picks:
            spec = _graph_spec(p)
            family = _family_of(p)
            score = p.get("score", float("nan"))
            runtime_ms = p.get("runtime_s", 0.0) * 1000
            print(f"  {spec:<35} family={family:<25} score={score:6.2f} runtime={runtime_ms:.0f}ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
