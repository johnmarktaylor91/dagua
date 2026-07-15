"""r80-S4 Stage 4 gate 3: acceptance analysis for the full dagua-only sweep.

Compares the NEW store (this worktree's eval_output/r79_baseline after the
--dagua-only sweep) against the FROZEN reference store in the main worktree.

Acceptance criteria (from the brief):
- ZERO WIN->LOSS flips anywhere in the corpus.
- The undirected class gains >= +6 graphs in best-or-tied (WIN or TIE)
  vs the frozen baseline.

Also reports: full W/T/L before/after per population, per-graph deltas,
biggest flips, and wall-time impact on the undirected class.

Usage
-----
    .venv/bin/python scripts/r80_gate3_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

P2_ROOT = Path(__file__).resolve().parents[1]
NEW_RESULTS = P2_ROOT / "eval_output" / "r79_baseline" / "results.json"
FROZEN_RESULTS = Path(
    "/home/jtaylor/.claude/worktrees/dagua-native/eval_output/r79_baseline/results.json"
)
REPORT_PATH = P2_ROOT / ".project-context" / "research" / "r79_native" / "P8_SWEEP_DELTAS.md"

TIE_BAND = 0.5


def _load(path: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    with path.open() as handle:
        payload = json.load(handle)
    by_graph: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in payload["rows"]:
        if row.get("status") != "OK":
            continue
        by_graph.setdefault(row["graph"], {})[row["engine"]] = row
    return by_graph


def _best_external(engines: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    for engine, row in engines.items():
        if engine == "dagua" or row.get("composite") is None:
            continue
        if best is None or float(row["composite"]) > float(best["composite"]):
            best = row
    return best


def _verdict(dagua_score: float, external_score: float) -> str:
    delta = dagua_score - external_score
    if delta > TIE_BAND:
        return "WIN"
    if delta >= -TIE_BAND:
        return "TIE"
    return "LOSS"


def main() -> None:
    sys.path.insert(0, str(P2_ROOT))
    from dagua.eval.graphs import get_test_graphs, is_semantically_directed

    corpus = get_test_graphs(max_nodes=500)
    undirected_names = {tg.name for tg in corpus if not is_semantically_directed(tg)}
    populations = {}
    sys.path.insert(0, str(P2_ROOT / "scripts"))
    from r79_baseline import graph_population  # noqa: E402

    for tg in corpus:
        populations[tg.name] = graph_population(tg.name)

    frozen = _load(FROZEN_RESULTS)
    new = _load(NEW_RESULTS)

    rows_out = []
    win_to_loss = []
    undirected_before_bot = 0  # best-or-tied
    undirected_after_bot = 0
    wtl_before: Dict[str, Dict[str, int]] = {}
    wtl_after: Dict[str, Dict[str, int]] = {}
    undirected_runtime_before = 0.0
    undirected_runtime_after = 0.0

    for tg in corpus:
        name = tg.name
        frozen_engines = frozen.get(name, {})
        new_engines = new.get(name, {})
        frozen_dagua = frozen_engines.get("dagua")
        new_dagua = new_engines.get("dagua")
        best_ext = _best_external(frozen_engines)
        if best_ext is None or frozen_dagua is None or new_dagua is None:
            continue
        ext_score = float(best_ext["composite"])
        before_score = float(frozen_dagua["composite"])
        after_score = float(new_dagua["composite"])
        before_verdict = _verdict(before_score, ext_score)
        after_verdict = _verdict(after_score, ext_score)
        population = populations.get(name, "legacy")
        wtl_before.setdefault(population, {"WIN": 0, "TIE": 0, "LOSS": 0})[before_verdict] += 1
        wtl_after.setdefault(population, {"WIN": 0, "TIE": 0, "LOSS": 0})[after_verdict] += 1
        is_undirected = name in undirected_names
        if is_undirected:
            undirected_before_bot += before_verdict in ("WIN", "TIE")
            undirected_after_bot += after_verdict in ("WIN", "TIE")
            undirected_runtime_before += float(frozen_dagua.get("runtime_s", 0.0))
            undirected_runtime_after += float(new_dagua.get("runtime_s", 0.0))
        if before_verdict == "WIN" and after_verdict == "LOSS":
            win_to_loss.append(name)
        rows_out.append(
            {
                "graph": name,
                "population": population,
                "undirected": is_undirected,
                "before": before_score,
                "after": after_score,
                "delta": after_score - before_score,
                "best_ext": best_ext["engine"],
                "ext_score": ext_score,
                "before_verdict": before_verdict,
                "after_verdict": after_verdict,
                "runtime_before": float(frozen_dagua.get("runtime_s", 0.0)),
                "runtime_after": float(new_dagua.get("runtime_s", 0.0)),
            }
        )

    undirected_gain = undirected_after_bot - undirected_before_bot
    gate_pass = (len(win_to_loss) == 0) and (undirected_gain >= 6)

    lines = []
    lines.append("# r80-S4 Gate 3: Full Sweep Before/After Deltas")
    lines.append("")
    lines.append("## W/T/L by population")
    lines.append("")
    lines.append("| population | before W/T/L | after W/T/L |")
    lines.append("|---|---|---|")
    for population in sorted(wtl_before):
        b = wtl_before[population]
        a = wtl_after.get(population, {"WIN": 0, "TIE": 0, "LOSS": 0})
        lines.append(
            f"| {population} | {b['WIN']}/{b['TIE']}/{b['LOSS']} "
            f"| {a['WIN']}/{a['TIE']}/{a['LOSS']} |"
        )
    lines.append("")
    lines.append(
        f"**Undirected class best-or-tied: {undirected_before_bot} -> "
        f"{undirected_after_bot} (gain {undirected_gain:+d}; acceptance >= +6)**"
    )
    lines.append("")
    lines.append(f"**WIN->LOSS flips: {len(win_to_loss)}** {win_to_loss}")
    lines.append("")
    lines.append(f"**GATE 3 {'PASSES' if gate_pass else 'FAILS'}**")
    lines.append("")
    lines.append(
        f"Undirected-class dagua wall time: {undirected_runtime_before:.1f}s -> "
        f"{undirected_runtime_after:.1f}s (frozen store recorded vs this sweep; "
        f"note this sweep ran under heavy shared-machine load, so absolute "
        f"multipliers overstate the route cost -- see directed-graph runtimes "
        f"in the same sweep for the load factor)."
    )
    lines.append("")
    lines.append("## Per-graph table (all scored graphs)")
    lines.append("")
    lines.append(
        "| graph | pop | undirected | before | after | delta | best-ext | verdict "
        "before->after | runtime before->after (s) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for row in sorted(rows_out, key=lambda item: -abs(item["delta"])):
        lines.append(
            f"| {row['graph']} | {row['population']} | "
            f"{'Y' if row['undirected'] else ''} | {row['before']:.2f} | "
            f"{row['after']:.2f} | {row['delta']:+.2f} | {row['best_ext']} "
            f"{row['ext_score']:.2f} | {row['before_verdict']}->{row['after_verdict']} | "
            f"{row['runtime_before']:.1f}->{row['runtime_after']:.1f} |"
        )
    lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n")

    print(f"Undirected best-or-tied: {undirected_before_bot} -> {undirected_after_bot}")
    print(f"WIN->LOSS flips: {len(win_to_loss)} {win_to_loss}")
    for population in sorted(wtl_before):
        b = wtl_before[population]
        a = wtl_after[population]
        print(
            f"{population}: {b['WIN']}/{b['TIE']}/{b['LOSS']} -> {a['WIN']}/{a['TIE']}/{a['LOSS']}"
        )
    print(f"GATE 3 {'PASSES' if gate_pass else 'FAILS'}")
    print(f"Report: {REPORT_PATH}")
    sys.exit(0 if gate_pass else 1)


if __name__ == "__main__":
    main()
