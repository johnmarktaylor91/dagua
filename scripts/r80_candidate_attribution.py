"""r80-S4: attribute each undirected-class dagua row to its winning candidate.

For every undirected corpus graph in the published store, decide which
contest candidate produced the final positions by comparing the published
composite against the Stage-1 probe candidate scores and the frozen
incumbent score:

- score == frozen incumbent score  -> incumbent won (or contest skipped)
- score == probe sfdp+proj score   -> sfdp challenger won
- score == probe neato+proj score  -> neato challenger won

Matches use a small tolerance; the probe and the route produce candidates
through identical code paths, so genuine matches agree to ~1e-6.

Usage
-----
    .venv/bin/python scripts/r80_candidate_attribution.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

P2_ROOT = Path(__file__).resolve().parents[1]
NEW_RESULTS = P2_ROOT / "eval_output" / "r79_baseline" / "results.json"
FROZEN_RESULTS = Path(
    "/home/jtaylor/.claude/worktrees/dagua-native/eval_output/r79_baseline/results.json"
)
PROBE_REPORT = P2_ROOT / ".project-context" / "research" / "r79_native" / "P8_PORTFOLIO_PROBE.md"

TOLERANCE = 0.05


def _dagua_scores(path: Path) -> dict[str, float]:
    with path.open() as handle:
        payload = json.load(handle)
    return {
        row["graph"]: float(row["composite"])
        for row in payload["rows"]
        if row.get("engine") == "dagua" and row.get("status") == "OK"
    }


def _probe_rows() -> dict[str, dict[str, float]]:
    """Parse per-graph sfdp/neato/kk candidate scores from the probe table."""
    rows: dict[str, dict[str, float]] = {}
    pattern = re.compile(r"^\| (\S+) \| \d+ \| ")
    for line in PROBE_REPORT.read_text().splitlines():
        if not pattern.match(line):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 8:
            continue
        name = cells[0]
        if name in rows:
            continue  # keep the first (full) table only

        def _num(cell: str) -> float | None:
            try:
                return float(cell)
            except ValueError:
                return None

        rows[name] = {
            key: value
            for key, value in {
                "sfdp": _num(cells[4]),
                "neato": _num(cells[5]),
                "kk": _num(cells[6]),
            }.items()
            if value is not None
        }
    return rows


def main() -> None:
    sys.path.insert(0, str(P2_ROOT))
    from dagua.eval.graphs import get_test_graphs, is_semantically_directed

    corpus = get_test_graphs(max_nodes=500)
    undirected = [tg.name for tg in corpus if not is_semantically_directed(tg)]

    new = _dagua_scores(NEW_RESULTS)
    frozen = _dagua_scores(FROZEN_RESULTS)
    probe = _probe_rows()

    counts = {"incumbent": 0, "sfdp": 0, "neato": 0, "unmatched": 0}
    print(f"{'graph':35s} {'winner':10s} {'score':>8s}")
    for name in sorted(undirected):
        if name not in new:
            continue
        score = new[name]
        winner = "unmatched"
        if name in frozen and abs(score - frozen[name]) < TOLERANCE:
            winner = "incumbent"
        else:
            for candidate in ("sfdp", "neato"):
                candidate_score = probe.get(name, {}).get(candidate)
                if candidate_score is not None and abs(score - candidate_score) < TOLERANCE:
                    winner = candidate
                    break
        counts[winner] += 1
        print(f"{name:35s} {winner:10s} {score:8.2f}")
    print()
    print("Candidate win counts (undirected class):", counts)


if __name__ == "__main__":
    main()
