#!/usr/bin/env python3
"""r71 P1b seedability probe (plan sec. 2b).

Per reference family: 2 mid-size graphs from its OWN failing map x seeds {42, 43, 44}
plus a same-seed repeat. Emits the probe table consumed by report v2.
PASS criteria per plan: varies across seeds AND stable within seed.
Includes a positive control (igraph_drl, known seedable). "PROVABLY unseedable" further
requires upstream-source evidence -- recorded as a separate column, filled by CC review.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fast_fidelity_report import procrustes_rmsd  # noqa: E402

from dagua.eval import competitors as comp_mod  # noqa: E402
from dagua.eval.graphs import get_test_graphs  # noqa: E402

FAILING_MAP = json.load(
    open(".project-context/research/sprint_rng_matching/failing_map_final.json")
)

# reference base -> a variant whose failing map supplies probe graphs
FAMILIES = {
    "graphviz_neato": "classic_neato",
    "graphviz_sfdp": "classic_sfdp_default",
    "graphviz_fdp": "classic_fmmm_graphviz_fdp_fidelity",
    "graphviz_dot": "classic_sugiyama_graphviz_fidelity",
    "ogdf_fmmm": "classic_fmmm_steps100",
    "ogdf_gem": "classic_gem_iters100",
    "ogdf_stress": "classic_maxent_stress_default",
    "ogdf_pivot_mds": "classic_pivot_mds_50",
    "igraph_mds": "classic_classical_mds_default",
    "igraph_sugiyama": "classic_sugiyama_default",
    # positive control (known seedable, Mode A in r70):
    "igraph_drl": "classic_drl_default",
}
SEEDS = (42, 43, 44)


def pick_graphs(variant: str, graphs: dict) -> list[str]:
    entry = FAILING_MAP.get(variant)
    pool = entry["graphs"] if entry else list(graphs)
    sized = []
    for name in pool:
        g = graphs.get(name)
        if g is None:
            continue
        n = getattr(g, "num_nodes", None) or len(
            getattr(g, "nodes", lambda: [])()
            if callable(getattr(g, "nodes", None))
            else getattr(g, "nodes", []) or []
        )
        if 40 <= n <= 600:
            sized.append((n, name))
    sized.sort()
    mid = sized[len(sized) // 2 :][:2] if sized else []
    return [name for _n, name in mid] or pool[:2]


def _pos(result):
    arr = getattr(result, "pos", result)
    if arr is None:
        raise RuntimeError(getattr(result, "error", "no positions"))
    import numpy as _np

    a = _np.asarray(arr, dtype=float)
    if a.ndim != 2:  # dict of node -> (x, y)
        a = _np.asarray([arr[k] for k in sorted(arr)], dtype=float)
    return a


def main() -> int:
    graphs = {g.name: g.graph for g in get_test_graphs()}
    table = {}
    for ref, variant in FAMILIES.items():
        try:
            competitor = comp_mod.get_competitor(ref)
        except Exception:
            competitor = None
        if competitor is None:
            table[ref] = {"status": "ADAPTER_NOT_FOUND", "probe_verdict": "PROBE_ERROR"}
            continue
        rows = []
        for gname in pick_graphs(variant, graphs):
            g = graphs[gname]
            try:
                lays = {s: _pos(competitor.layout(g, timeout=120, seed=s)) for s in SEEDS}
                repeat = _pos(competitor.layout(g, timeout=120, seed=SEEDS[0]))
            except Exception as exc:
                rows.append({"graph": gname, "error": str(exc)[:160]})
                continue
            cross = [
                procrustes_rmsd(lays[a], lays[b])
                for i, a in enumerate(SEEDS)
                for b in SEEDS[i + 1 :]
            ]
            within = procrustes_rmsd(lays[SEEDS[0]], repeat)
            rows.append(
                {
                    "graph": gname,
                    "n_nodes": int(lays[SEEDS[0]].shape[0]),
                    "cross_seed_min_rmsd": float(min(cross)),
                    "cross_seed_max_rmsd": float(max(cross)),
                    "within_seed_rmsd": float(within),
                }
            )
        ok_rows = [r for r in rows if "cross_seed_min_rmsd" in r]
        varies = bool(ok_rows) and all(r["cross_seed_min_rmsd"] > 1e-9 for r in ok_rows)
        stable = bool(ok_rows) and all(r["within_seed_rmsd"] < 1e-9 for r in ok_rows)
        table[ref] = {
            "variant_for_graphs": variant,
            "rows": rows,
            "varies_across_seeds": varies,
            "stable_within_seed": stable,
            "probe_verdict": "SEEDABLE"
            if varies and stable
            else ("NO_VARIATION" if ok_rows else "PROBE_ERROR"),
            "upstream_source_evidence": "PENDING_CC_REVIEW",
        }
        print(f"{ref:18} -> {table[ref]['probe_verdict']}", flush=True)

    ctl = table.get("igraph_drl", {})
    table["_positive_control_ok"] = ctl.get("probe_verdict") == "SEEDABLE"
    out = Path("eval_output/fidelity_definitive/r71_seedability_probe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=1))
    print(f"positive control ok: {table['_positive_control_ok']}")
    return 0 if table["_positive_control_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
