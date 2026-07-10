"""Summarize r81 P2 resistance-distance probe results into a Markdown table.

Reads the probe JSONL (scripts/r81_resdist_probe.py output) and prints a
per-graph table: baseline core | dijkstra K-scaled | resistance variants |
best external, with beat/tie/lose verdicts at the r79 tie band (+/-0.5).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

TIE_BAND = 0.5


def verdict(candidate: float, reference: float) -> str:
    """Return BEAT/TIE/LOSE for a candidate against a reference score."""
    delta = candidate - reference
    if delta > TIE_BAND:
        return "BEAT"
    if delta >= -TIE_BAND:
        return "TIE"
    return "LOSE"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--stage", default="best", choices=["raw", "prism", "conv", "best"])
    args = parser.parse_args()

    rows = defaultdict(dict)
    meta = {}
    for line in args.jsonl.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        rows[record["graph"]][record["variant"]] = record
        meta[record["graph"]] = record

    variants = sorted({variant for by_variant in rows.values() for variant in by_variant})
    key = f"{args.stage}_composite"

    header = (
        ["graph", "n", "frozen_dagua"]
        + variants
        + ["best_ext", "res_verdict_vs_ext", "res_verdict_vs_dij"]
    )
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for graph in rows:
        record = meta[graph]
        best_ext = record.get("best_external")
        cells = [
            graph,
            str(record.get("nodes")),
            f"{record.get('frozen_dagua'):.2f}" if record.get("frozen_dagua") else "-",
        ]
        scores = {}
        for variant in variants:
            entry = rows[graph].get(variant)
            if entry is None:
                cells.append("-")
                continue
            scores[variant] = float(entry[key])
            cells.append(f"{scores[variant]:.2f}")
        res_scores = [value for name, value in scores.items() if name.startswith("res_")]
        dij_scores = [
            value for name, value in scores.items() if name.startswith("dij_") or name == "base"
        ]
        best_res = max(res_scores) if res_scores else None
        best_dij = max(dij_scores) if dij_scores else None
        cells.append(f"{best_ext:.2f}" if best_ext is not None else "-")
        cells.append(
            verdict(best_res, best_ext) if best_res is not None and best_ext is not None else "-"
        )
        cells.append(
            verdict(best_res, best_dij) if best_res is not None and best_dij is not None else "-"
        )
        print("| " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
