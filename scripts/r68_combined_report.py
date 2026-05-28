#!/usr/bin/env python3
"""Combine the per-seed Procrustes RMSD report (bit-exact verdicts) with the
TOST equivalence report (statistical-distribution verdicts) into one final
fidelity report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--per-seed",
        required=True,
        help="Directory containing per_variant.json from fast_fidelity_report.py",
    )
    parser.add_argument(
        "--tost",
        required=True,
        help="Directory containing tost_results.json from r68_tost_followup.py",
    )
    parser.add_argument("--output", required=True, help="Output path for combined report.md")
    args = parser.parse_args()

    per_seed = json.load((Path(args.per_seed) / "per_variant.json").open())
    summary_rows = per_seed.get("summary", [])  # list of (variant, verdict, mean, median, max, n)

    tost_path = Path(args.tost) / "tost_results.json"
    tost_results = {}
    if tost_path.is_file():
        tost_results = json.load(tost_path.open())

    # Combined verdict per variant
    lines = [
        "# Dagua Fidelity Report -- Combined (per-seed + TOST)",
        "",
        "Hybrid framework: bit-exact verdicts where dagua reproduces the reference per-seed at <1e-3 RMSD;",  # noqa: E501
        "TOST statistical equivalence for variants where chaotic dynamics produce different basins per seed",  # noqa: E501
        "but the algorithmic distribution matches the reference over 100 seeds.",
        "",
        "## Tiers",
        "",
        "- **MACHINE_EPSILON** -- per-seed max RMSD < 1e-6 (float epsilon)",
        "- **BIT_EXACT** -- per-seed max RMSD < 1e-3",
        "- **STRONG_EQUIVALENT** -- TOST: dagua and reference distributions statistically equivalent within ±0.05",  # noqa: E501
        "- **WEAK_EQUIVALENT** -- TOST: equivalent on 60-89% of (variant, graph) groups",
        "- **PARTIAL_MATCH** -- TOST: equivalent on 30-59% of groups",
        "- **NO_EQUIVALENCE** -- TOST: equivalent on <30% of groups, OR no TOST data",
        "",
        "## Per-variant",
        "",
        "| Variant | n_seeds | Final tier | Median RMSD | Max RMSD | TOST %equiv |",
        "|---|---:|:--|---:|---:|---:|",
    ]

    tier_counts: dict[str, int] = {}
    for row in summary_rows:
        variant, _verdict, mean_r, median_r, max_r, n = row
        if max_r < 1e-6:
            tier = "MACHINE_EPSILON"
            tost_str = "--"
        elif max_r < 1e-3:
            tier = "BIT_EXACT"
            tost_str = "--"
        elif variant in tost_results:
            tr = tost_results[variant]
            tier = tr["tier"].upper()
            tost_str = f"{tr['pct_equiv']:.1f}%"
        else:
            tier = "NO_EQUIVALENCE"
            tost_str = "no_data"

        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        lines.append(f"| {variant} | {n} | {tier} | {median_r:.3e} | {max_r:.3e} | {tost_str} |")

    lines.append("")
    lines.append("## Tier totals")
    lines.append("")
    for tier in [
        "MACHINE_EPSILON",
        "BIT_EXACT",
        "STRONG_EQUIVALENT",
        "WEAK_EQUIVALENT",
        "PARTIAL_MATCH",
        "NO_EQUIVALENCE",
    ]:
        lines.append(f"- {tier}: {tier_counts.get(tier, 0)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
