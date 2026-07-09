"""Per-graph before/after comparison for the r80/projector sweep gate.

Compares dagua rows between the pre-change baseline snapshot
(``results.pre_r80_at_base.json``, produced at the r79/native branch base)
and the post-change ``results.json`` (produced by ``r79_baseline.py
--dagua-only`` on r80/projector), using the same frozen external rows and
the sprint's TIE_BAND=0.5 win/tie/loss definition.

Usage: python scripts/r80_projector_delta.py [output_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

TIE_BAND = 0.5
EXTERNAL_EXCLUDE = {"dagua"}


def load_rows(path: Path) -> list[Dict[str, Any]]:
    """Load result rows from a results.json payload.

    Parameters
    ----------
    path : Path
        Path to a results.json file.

    Returns
    -------
    list[Dict[str, Any]]
        Result rows.
    """
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)["rows"]


def dagua_by_graph(rows: list[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Index OK dagua rows by graph name.

    Parameters
    ----------
    rows : list[Dict[str, Any]]
        Result rows.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        OK dagua rows keyed by graph.
    """
    return {
        str(r["graph"]): r for r in rows if r.get("engine") == "dagua" and r.get("status") == "OK"
    }


def best_external_by_graph(rows: list[Dict[str, Any]]) -> Dict[str, float]:
    """Best external OK composite per graph (mirrors r79_baseline.py).

    Parameters
    ----------
    rows : list[Dict[str, Any]]
        Result rows.

    Returns
    -------
    Dict[str, float]
        Best external composite keyed by graph.
    """
    best: Dict[str, float] = {}
    for r in rows:
        if r.get("engine") in EXTERNAL_EXCLUDE or r.get("status") != "OK":
            continue
        comp = r.get("composite")
        if comp is None:
            continue
        g = str(r["graph"])
        if g not in best or float(comp) > best[g]:
            best[g] = float(comp)
    return best


def bucket(delta: float) -> str:
    """Classify a dagua-minus-best-external delta into W/T/L.

    Parameters
    ----------
    delta : float
        Composite delta.

    Returns
    -------
    str
        ``"W"``, ``"T"``, or ``"L"``.
    """
    if delta > TIE_BAND:
        return "W"
    if delta >= -TIE_BAND:
        return "T"
    return "L"


def main() -> int:
    """Emit the per-graph delta table and W/T/L summary.

    Returns
    -------
    int
        Process exit status (0 = acceptance met, 1 = not met).
    """
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("eval_output/r79_baseline")
    pre_rows = load_rows(out_dir / "results.pre_r80_at_base.json")
    post_rows = load_rows(out_dir / "results.json")

    pre_dagua = dagua_by_graph(pre_rows)
    post_dagua = dagua_by_graph(post_rows)
    # One frozen external anchor for BOTH sides (post store carries the same
    # externals the staging copy started from).
    best_ext = best_external_by_graph(post_rows)

    graphs = sorted(set(pre_dagua) | set(post_dagua))
    lines = []
    net = 0.0
    flips_w_to_l = []
    pre_wtl = {"W": 0, "T": 0, "L": 0}
    post_wtl = {"W": 0, "T": 0, "L": 0}
    regressions = []
    status_changes = []

    lines.append("| graph | pre comp | post comp | delta | best ext | pre W/T/L | post W/T/L |")
    lines.append("|---|---:|---:|---:|---:|:-:|:-:|")
    for g in graphs:
        pre = pre_dagua.get(g)
        post = post_dagua.get(g)
        if pre is None:
            # Corpus drift (graph absent from the pre-change snapshot), not a
            # regression caused by this diff -- report but exclude from the
            # net/flip acceptance math.
            status_changes.append((g, "pre", "graph not in pre-change snapshot"))
            continue
        if post is None:
            status_changes.append((g, "post", "dagua row not OK post-change"))
            continue
        ext = best_ext.get(g)
        pc = float(pre["composite"])
        qc = float(post["composite"])
        d = qc - pc
        net += d
        if ext is None:
            pb = qb = "?"
        else:
            pb = bucket(pc - ext)
            qb = bucket(qc - ext)
            pre_wtl[pb] += 1
            post_wtl[qb] += 1
            if pb == "W" and qb == "L":
                flips_w_to_l.append(g)
        if d < -0.5:
            regressions.append((g, d))
        lines.append(
            f"| {g} | {pc:.3f} | {qc:.3f} | {d:+.3f} | {ext:.3f} | {pb} | {qb} |"
            if ext is not None
            else f"| {g} | {pc:.3f} | {qc:.3f} | {d:+.3f} | n/a | ? | ? |"
        )

    print("\n".join(lines))
    print()
    print(f"graphs compared: {sum(post_wtl.values())}")
    print(f"net composite delta: {net:+.3f}")
    print(f"pre  W/T/L: {pre_wtl['W']}/{pre_wtl['T']}/{pre_wtl['L']}")
    print(f"post W/T/L: {post_wtl['W']}/{post_wtl['T']}/{post_wtl['L']}")
    print(f"WIN->LOSS flips: {len(flips_w_to_l)} {flips_w_to_l}")
    if status_changes:
        print(f"excluded graphs: {status_changes}")
    if regressions:
        print("per-graph composite drops < -0.5:")
        for g, d in sorted(regressions, key=lambda x: x[1]):
            print(f"  {g}: {d:+.3f}")
    post_failures = [t for t in status_changes if t[1] == "post"]
    accepted = net >= 0.0 and not flips_w_to_l and not post_failures
    print(
        f"ACCEPTANCE: {'PASS' if accepted else 'FAIL'} "
        f"(net >= 0: {net >= 0.0}; zero W->L flips: {not flips_w_to_l}; "
        f"no post-change row failures: {not post_failures})"
    )
    return 0 if accepted else 1


if __name__ == "__main__":
    sys.exit(main())
