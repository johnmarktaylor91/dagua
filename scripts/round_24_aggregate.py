#!/usr/bin/env python3
"""Aggregate per-family TOST verdicts from Round 24 multi_seed_summary.json files.

Reads each `eval_output/algo_fidelity/round_24/<family>/multi_seed_summary.json`
and produces a per-family roll-up: at each TOST margin (0.25x/0.5x/1x/1.5x/2x),
how many graphs reach equivalence?

A family is judged CONVERGED at margin X if all (or all-but-one) graphs are
equivalent_at_X.

Usage:
    python3 scripts/round_24_aggregate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROUND = "round_24"
ROUND_DIR = (
    REPO_ROOT
    / "eval_output"
    / "algo_fidelity"
    / (sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROUND)
)

MARGINS = ["0.25x", "0.5x", "1x", "1.5x", "2x"]
FAMILIES = [
    "classical_mds",
    "fa2",
    "fmmm",
    "fr",
    "gem",
    "kk",
    "lgl",
    "maxent_stress",
    "pivot_mds",
    "rt",
    "sgd2_multi",
    "spectral",
    "stress_maj",
    "stress_sgd",
    "sugiyama",
    "umap",
]


DET_PERFECT_THRESHOLD = 1e-3
"""RMSD below this counts as numerical noise / perfect match for deterministic algos."""


def family_verdict(summary: dict) -> dict:
    """Roll up per-graph TOST verdicts to a family-level summary."""
    graphs = summary.get("graphs", {})
    n_graphs = len(graphs)
    medians: List[float] = []
    eq_counts: Dict[str, int] = {m: 0 for m in MARGINS}
    not_tested_count = 0
    not_tested_perfect_count = 0
    not_tested_divergent_count = 0
    for graph_name, graph_summary in graphs.items():
        rmsd_block = graph_summary.get("dagua_vs_graphviz") or graph_summary.get(
            "dagua_vs_target", {}
        )
        med = rmsd_block.get("median")
        if med is not None:
            medians.append(med)
        tost = graph_summary.get("tost", {})
        # When TOST can't run (zero target variance), treat near-zero RMSD as
        # deterministic match and significant RMSD as divergence from a
        # deterministic reference.
        any_tested = any(
            tost.get(m, {}).get("verdict", "not_tested") != "not_tested" for m in MARGINS
        )
        if not any_tested:
            not_tested_count += 1
            if med is not None and med < DET_PERFECT_THRESHOLD:
                not_tested_perfect_count += 1
            else:
                not_tested_divergent_count += 1
            continue
        for margin in MARGINS:
            if tost.get(margin, {}).get("equivalent"):
                eq_counts[margin] += 1
    overall_median = sum(medians) / len(medians) if medians else None
    return {
        "n_graphs": n_graphs,
        "n_not_tested": not_tested_count,
        "n_not_tested_perfect": not_tested_perfect_count,
        "n_not_tested_divergent": not_tested_divergent_count,
        "median_of_medians": overall_median,
        "max_median": max(medians) if medians else None,
        "eq_counts": eq_counts,
    }


def classify_family(verdict: dict) -> str:
    """Classify family by strictest margin where ALL graphs are equivalent."""
    n = verdict["n_graphs"]
    n_nt_perfect = verdict["n_not_tested_perfect"]
    n_nt_divergent = verdict["n_not_tested_divergent"]
    n_tested = n - n_nt_perfect - n_nt_divergent

    # All graphs deterministic-perfect (zero target variance, near-zero RMSD)
    if n_nt_perfect == n:
        return "DETERMINISTIC_PERFECT"
    if n_nt_divergent == n:
        return "DIVERGENT_FROM_DETERMINISTIC_REF"

    eq = verdict["eq_counts"]
    # Need ALL tested graphs to be equivalent at margin (and no divergent-from-deterministic).
    if n_nt_divergent == 0:
        for m in MARGINS:
            if eq[m] >= n_tested and n_tested > 0:
                if n_nt_perfect > 0:
                    return f"CONVERGED_at_{m}_with_{n_nt_perfect}det_perfect"
                return f"CONVERGED_at_{m}"
        # All tested were equivalent at no margin, but at least some perfect
        if n_nt_perfect > 0 and n_tested == 0:
            return f"DETERMINISTIC_PERFECT_partial_{n_nt_perfect}of{n}"

    # Partial: count how many at strictest equivalent for any graph
    for m in MARGINS:
        if eq[m] >= 1:
            return f"PARTIAL_some_eq_at_{m}_div{n_nt_divergent}"
    if n_nt_divergent > 0:
        return f"NOT_EQUIVALENT_div{n_nt_divergent}"
    return "NOT_EQUIVALENT"


def main() -> int:
    """Iterate families, print summary table, write report file."""
    if not ROUND_DIR.is_dir():
        print(f"FATAL: round dir missing: {ROUND_DIR}", file=sys.stderr)
        return 1

    print(f"Round 24 aggregate (source dir: {ROUND_DIR})\n")
    rows: List[List[str]] = []
    rows.append(
        [
            "family",
            "n_graphs",
            "n_perf",
            "n_div",
            "med_of_med",
            "max_med",
            *[f"eq_{m}" for m in MARGINS],
            "verdict",
        ]
    )
    family_records: Dict[str, dict] = {}
    for family in FAMILIES:
        summary_path = ROUND_DIR / family / "multi_seed_summary.json"
        if not summary_path.is_file():
            rows.append(
                [
                    family,
                    "?",
                    "?",
                    "?",
                    "?",
                    "?",
                    *["?"] * len(MARGINS),
                    "MISSING",
                ]
            )
            family_records[family] = {"verdict": "MISSING"}
            continue
        with summary_path.open() as f:
            summary = json.load(f)
        verdict = family_verdict(summary)
        classification = classify_family(verdict)
        family_records[family] = {
            **verdict,
            "verdict": classification,
        }
        rows.append(
            [
                family,
                str(verdict["n_graphs"]),
                str(verdict["n_not_tested_perfect"]),
                str(verdict["n_not_tested_divergent"]),
                f"{verdict['median_of_medians']:.4f}"
                if verdict["median_of_medians"] is not None
                else "n/a",
                f"{verdict['max_median']:.4f}" if verdict["max_median"] is not None else "n/a",
                *[str(verdict["eq_counts"][m]) for m in MARGINS],
                classification,
            ]
        )

    # Print as a tab-separated table.
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for r_idx, row in enumerate(rows):
        line = "  ".join(
            cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i]) for i, cell in enumerate(row)
        )
        print(line)
        if r_idx == 0:
            print("  ".join("-" * w for w in widths))

    # Write JSON report.
    report_path = ROUND_DIR / "_aggregate_report.json"
    with report_path.open("w") as f:
        json.dump(family_records, f, indent=2)
    print(f"\nWrote report to {report_path}")

    # Identify stragglers (anything not CONVERGED at 1x or stricter / DETERMINISTIC_PERFECT).
    def is_clean(v: str) -> bool:
        if v == "DETERMINISTIC_PERFECT":
            return True
        if v.startswith("CONVERGED_at_0.25x"):
            return True
        if v.startswith("CONVERGED_at_0.5x"):
            return True
        if v.startswith("CONVERGED_at_1x"):
            return True
        return False

    stragglers = [
        fam for fam, rec in family_records.items() if not is_clean(rec.get("verdict", "MISSING"))
    ]
    print(f"\nStragglers ({len(stragglers)}): {', '.join(stragglers) if stragglers else 'NONE'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
