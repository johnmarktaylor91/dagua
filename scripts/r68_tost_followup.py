#!/usr/bin/env python3
"""For variants flagged as non-bit-exact, run TOST equivalence test on the
100-seed distribution to determine if dagua and reference samples come from
statistically equivalent distributions.

Strategy: for each (variant, graph) group with seeds >= 30, compute the
distribution of per-pair Procrustes RMSDs and the distribution of pairwise
distances. Test that both sides have the same distribution within an
equivalence band.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from scipy import stats


def procrustes_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """Procrustes RMSD on unit-Frobenius-normalized clouds."""
    if a.shape != b.shape or a.size == 0:
        return float("nan")
    a_c = a - a.mean(0)
    b_c = b - b.mean(0)
    a_n = float(np.linalg.norm(a_c))
    b_n = float(np.linalg.norm(b_c))
    if a_n < 1e-12 or b_n < 1e-12:
        return 0.0 if (a_n < 1e-12 and b_n < 1e-12) else float(a_n + b_n)
    a_u = a_c / a_n
    b_u = b_c / b_n
    u, _, vt = np.linalg.svd(b_u.T @ a_u)
    rotation = u @ vt
    return float(np.linalg.norm((a_u @ rotation.T) - b_u))


def stress_score(positions: np.ndarray, edges: list[tuple[int, int]] | None = None) -> float:
    """Simple shape descriptor: sum of pairwise distances (scale-invariant)."""
    if positions.shape[0] < 2:
        return 0.0
    pos_c = positions - positions.mean(0)
    norm = np.linalg.norm(pos_c)
    if norm < 1e-12:
        return 0.0
    pos_u = pos_c / norm
    # Pairwise squared distance sum (scale-invariant after normalization)
    diffs = pos_u[:, None, :] - pos_u[None, :, :]
    return float(np.sqrt((diffs**2).sum()))


def tost_test(
    a_samples: list[float], b_samples: list[float], delta: float = 0.05
) -> tuple[bool, float, float]:
    """Two One-Sided T-tests for equivalence within ±delta.

    Returns (is_equivalent, p_lower, p_upper).
    Equivalent if both p-values < 0.05 (95% confidence both bounds satisfied).
    """
    if len(a_samples) < 5 or len(b_samples) < 5:
        return False, float("nan"), float("nan")
    a_mean = float(np.mean(a_samples))
    b_mean = float(np.mean(b_samples))
    diff = a_mean - b_mean
    # Welch's t-test
    se = math.sqrt(
        np.var(a_samples, ddof=1) / len(a_samples) + np.var(b_samples, ddof=1) / len(b_samples)
    )
    if se < 1e-12:
        # Effectively zero variance -- equivalent if means are within delta
        return abs(diff) < delta, 0.0, 0.0
    t_lower = (diff + delta) / se  # H0: diff <= -delta
    t_upper = (diff - delta) / se  # H0: diff >= delta
    df = len(a_samples) + len(b_samples) - 2
    p_lower = 1 - stats.t.cdf(t_lower, df)  # reject H0_lower if p_lower < alpha
    p_upper = stats.t.cdf(t_upper, df)  # reject H0_upper if p_upper < alpha
    is_equiv = p_lower < 0.05 and p_upper < 0.05
    return is_equiv, p_lower, p_upper


def load_pos(h5, key):
    try:
        if key not in h5:
            return None
        arr = np.asarray(h5[key][...], dtype=np.float64)
        return arr if arr.ndim == 2 and arr.size > 0 else None
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-variant-json", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--positions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bit-exact-threshold", type=float, default=1e-3)
    parser.add_argument("--tost-delta", type=float, default=0.05)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Identify chaotic-faithful and diverged variants from the fast report
    data = json.load(Path(args.per_variant_json).open())
    summary_rows = data.get("summary", [])
    candidates = [r for r in summary_rows if r[4] >= args.bit_exact_threshold]
    print(f"Variants requiring TOST: {len(candidates)}")
    for row in candidates:
        print(f"  {row[0]} mean={row[2]:.3e} med={row[3]:.3e} max={row[4]:.3e}")

    if not candidates:
        report_path = output_dir / "tost_report.md"
        report_path.write_text("# TOST Report\n\nNo variants require TOST -- all are bit-exact.\n")
        print(f"Wrote {report_path}")
        return 0

    # Load results + positions
    sys.path.insert(0, ".")
    from dagua.eval.variants import VARIANT_REGISTRY, original_variant_name

    pair_map = {
        v.variant_id: original_variant_name(v)
        for v in VARIANT_REGISTRY
        if original_variant_name(v) is not None
    }

    with Path(args.results).open() as f:
        results = json.load(f)

    valid = set()
    for v in results.values():
        if v.get("status") == "ok":
            valid.add((v.get("engine_name", ""), v.get("graph_name", ""), v.get("seed", -1)))

    h5 = h5py.File(args.positions, "r")

    tost_results: dict[str, dict] = {}
    for row in candidates:
        variant = row[0]
        ref = pair_map.get(variant)
        if ref is None:
            continue

        # collect per-graph stress distributions for both sides
        graph_groups = defaultdict(lambda: {"reimp_stress": [], "ref_stress": [], "rmsds": []})
        for eng, graph, seed in valid:
            if eng != variant:
                continue
            if (ref, graph, seed) not in valid:
                continue
            a = load_pos(h5, f"{graph}::{variant}::seed{seed}")
            b = load_pos(h5, f"{graph}::{ref}::seed{seed}")
            if a is None or b is None:
                continue
            graph_groups[graph]["reimp_stress"].append(stress_score(a))
            graph_groups[graph]["ref_stress"].append(stress_score(b))
            graph_groups[graph]["rmsds"].append(procrustes_rmsd(a, b))

        # Aggregate
        n_equiv_graphs = 0
        n_total_graphs = 0
        graph_verdicts = []
        for graph, gdata in graph_groups.items():
            if len(gdata["reimp_stress"]) < 5:
                continue
            n_total_graphs += 1
            is_equiv, p_lo, p_hi = tost_test(
                gdata["reimp_stress"], gdata["ref_stress"], args.tost_delta
            )
            if is_equiv:
                n_equiv_graphs += 1
            graph_verdicts.append(
                {
                    "graph": graph,
                    "n_seeds": len(gdata["reimp_stress"]),
                    "rmsd_mean": float(np.mean(gdata["rmsds"])),
                    "rmsd_max": float(np.max(gdata["rmsds"])),
                    "is_equiv": is_equiv,
                    "p_lower": float(p_lo),
                    "p_upper": float(p_hi),
                }
            )

        pct_equiv = (n_equiv_graphs / n_total_graphs * 100) if n_total_graphs > 0 else 0.0
        if pct_equiv >= 90:
            tier = "strong_equivalent"
        elif pct_equiv >= 60:
            tier = "weak_equivalent"
        elif pct_equiv >= 30:
            tier = "partial_match"
        else:
            tier = "no_equivalence"
        tost_results[variant] = {
            "tier": tier,
            "n_equiv_graphs": n_equiv_graphs,
            "n_total_graphs": n_total_graphs,
            "pct_equiv": pct_equiv,
            "per_seed_max_rmsd": row[4],
            "per_seed_median_rmsd": row[3],
            "graph_verdicts": graph_verdicts,
        }
        print(f"  {variant}: {tier} ({n_equiv_graphs}/{n_total_graphs} = {pct_equiv:.1f}%)")

    h5.close()

    # Write report
    lines = [
        "# R68 TOST Equivalence Report",
        "",
        f"For variants with max per-seed Procrustes RMSD >= {args.bit_exact_threshold:.0e},",
        "tests whether dagua and reference produce STATISTICALLY EQUIVALENT distributions",
        "over 100 seeds (chaotic ports + faithful sampling from same distribution).",
        "",
        "TOST equivalence band: ±{:.2f} on stress/shape score.".format(args.tost_delta),
        "",
        "| Variant | Tier | %Equiv | n_graphs | per-seed median | per-seed max |",
        "|---|:--|---:|---:|---:|---:|",
    ]
    tier_counts = defaultdict(int)
    for variant in sorted(tost_results):
        r = tost_results[variant]
        tier_counts[r["tier"]] += 1
        lines.append(
            f"| {variant} | {r['tier']} | {r['pct_equiv']:.1f}% | {r['n_total_graphs']} "
            f"| {r['per_seed_median_rmsd']:.3e} | {r['per_seed_max_rmsd']:.3e} |"
        )

    lines.append("")
    lines.append("## Tier totals")
    lines.append("")
    for tier in ["strong_equivalent", "weak_equivalent", "partial_match", "no_equivalence"]:
        lines.append(f"- {tier}: {tier_counts[tier]}")

    report_path = output_dir / "tost_report.md"
    report_path.write_text("\n".join(lines))
    print(f"\nWrote {report_path}")

    json_path = output_dir / "tost_results.json"
    with json_path.open("w") as f:
        json.dump(tost_results, f, indent=2)
    print(f"Wrote {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
