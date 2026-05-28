#!/usr/bin/env python3
"""Fast fidelity report: per-(variant, graph, seed) Procrustes RMSD vs reference.

Aggregates per-variant: mean/median/max RMSD + bit-exact verdict.
For variants that exceed the bit-exact threshold, dumps the failing (graph, seed)
list for TOST-style follow-up.

Runs in ~10-20 minutes on the full results.json + positions.h5 instead of the
20-hour pipeline of fidelity_analysis.py.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np


def procrustes_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """Compute scale/rotation/translation-invariant Procrustes RMSD on unit-normalized clouds."""
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


def load_position_h5(h5, key: str) -> np.ndarray | None:
    """Load positions from HDF5 store. Returns None if missing or invalid."""
    try:
        if key not in h5:
            return None
        arr = np.asarray(h5[key][...], dtype=np.float64)
        if arr.ndim != 2 or arr.size == 0:
            return None
        return arr
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="eval_output/benchmark_100seed_final/results.json")
    parser.add_argument("--positions", default="eval_output/benchmark_100seed_final/positions.h5")
    parser.add_argument("--output", default="eval_output/fidelity_report_fast")
    parser.add_argument("--bit-exact-threshold", type=float, default=1e-3)
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=5,
        help="Cap seeds per variant; default 5 for the smart benchmark contract",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    positions_path = Path(args.positions)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {results_path}...")
    with results_path.open() as f:
        results = json.load(f)
    print(f"  {len(results)} entries")

    # Build the variant pairing
    sys.path.insert(0, ".")
    from dagua.eval.variants import VARIANT_REGISTRY, original_variant_name

    pair_map: dict[str, str] = {}
    for v in VARIANT_REGISTRY:
        ref = original_variant_name(v)
        if ref is not None:
            pair_map[v.variant_id] = ref

    # Index ok results by (engine, graph, seed)
    valid = set()
    seeds_per_pair: dict[tuple[str, str], set[int]] = defaultdict(set)
    for v in results.values():
        if v.get("status") != "ok":
            continue
        eng = v.get("engine_name", "")
        graph = v.get("graph_name", "")
        seed = v.get("seed", -1)
        valid.add((eng, graph, seed))
        seeds_per_pair[(eng, graph)].add(seed)
    print(f"  {len(valid)} ok entries")

    # Compute per (reimp, graph, seed) Procrustes RMSD vs reference
    print(f"Opening {positions_path}...")
    h5 = h5py.File(positions_path, "r")

    per_variant_rmsds: dict[str, list[tuple[str, int, float]]] = defaultdict(list)
    per_variant_failures: dict[str, list[tuple[str, int, float]]] = defaultdict(list)
    total_pairs = 0
    skipped_no_pos = 0
    skipped_no_pair = 0
    start = time.time()

    for reimp, ref in pair_map.items():
        # find graphs where both reimp and ref have ok results at matching seeds
        for graph in set(g for (e, g) in seeds_per_pair if e == reimp):
            reimp_seeds = seeds_per_pair.get((reimp, graph), set())
            ref_seeds = seeds_per_pair.get((ref, graph), set())
            common = sorted(reimp_seeds & ref_seeds)[: args.max_seeds]
            if not common:
                skipped_no_pair += 1
                continue
            for seed in common:
                reimp_key = f"{graph}::{reimp}::seed{seed}"
                ref_key = f"{graph}::{ref}::seed{seed}"
                a = load_position_h5(h5, reimp_key)
                b = load_position_h5(h5, ref_key)
                if a is None or b is None:
                    # try without seed suffix
                    a = a if a is not None else load_position_h5(h5, f"{graph}::{reimp}")
                    b = b if b is not None else load_position_h5(h5, f"{graph}::{ref}")
                if a is None or b is None:
                    skipped_no_pos += 1
                    continue
                rmsd = procrustes_rmsd(a, b)
                if math.isfinite(rmsd):
                    per_variant_rmsds[reimp].append((graph, seed, rmsd))
                    if rmsd >= args.bit_exact_threshold:
                        per_variant_failures[reimp].append((graph, seed, rmsd))
                total_pairs += 1
                if total_pairs % 500 == 0:
                    elapsed = time.time() - start
                    print(f"  {total_pairs} pairs processed ({elapsed:.1f}s)")

    h5.close()
    elapsed = time.time() - start
    print(f"\nTotal pairs: {total_pairs} processed in {elapsed:.1f}s")
    print(f"  no-position skips: {skipped_no_pos}, no-pair skips: {skipped_no_pair}")

    # Build report
    lines = [
        "# Dagua Fidelity Report (fast)",
        "",
        "Per-variant Procrustes RMSD against reference, using the most recent benchmark data.",  # noqa: E501
        f"Up to {args.max_seeds} seeds per variant.",  # noqa: E501
        f"Bit-exact threshold: {args.bit_exact_threshold:.0e}",
        "",
        "## Verdict legend",
        "",
        "- **MACHINE_EPSILON**: max RMSD < 1e-6 (truly bit-identical modulo float epsilon)",
        "- **BIT_EXACT**: max RMSD < 1e-3 (Procrustes-normalized 0.1% geometric difference)",
        "- **STRONG_EQUIV**: max RMSD < 1e-2 (statistically indistinguishable; TOST recommended)",
        "- **PARTIAL**: max RMSD >= 1e-2 (visible geometric differences)",
        "",
        "## Per-variant summary",
        "",
        "| Variant | N | Mean | Median | Max | Verdict |",
        "|---|---:|---:|---:|---:|:--|",
    ]
    machine_eps = bit_exact = strong = partial = no_data = 0
    summary_rows: list[tuple[str, str, float, float, float, int]] = []
    for variant in sorted(per_variant_rmsds):
        entries = per_variant_rmsds[variant]
        if not entries:
            lines.append(f"| {variant} | 0 | -- | -- | -- | NO_DATA |")
            no_data += 1
            continue
        rmsds = [r for (_g, _s, r) in entries]
        mean_r = float(np.mean(rmsds))
        median_r = float(np.median(rmsds))
        max_r = float(np.max(rmsds))
        if max_r < 1e-6:
            verdict = "MACHINE_EPSILON"
            machine_eps += 1
        elif max_r < 1e-3:
            verdict = "BIT_EXACT"
            bit_exact += 1
        elif max_r < 1e-2:
            verdict = "STRONG_EQUIV"
            strong += 1
        else:
            verdict = "PARTIAL"
            partial += 1
        lines.append(
            f"| {variant} | {len(rmsds)} | {mean_r:.3e} | {median_r:.3e} | {max_r:.3e} | {verdict} |"  # noqa: E501
        )
        summary_rows.append((variant, verdict, mean_r, median_r, max_r, len(rmsds)))

    lines.append("")
    lines.append("## Totals")
    lines.append("")
    lines.append(f"- MACHINE_EPSILON: **{machine_eps}** variants")
    lines.append(f"- BIT_EXACT: **{bit_exact}** variants")
    lines.append(f"- STRONG_EQUIV: **{strong}** variants")
    lines.append(f"- PARTIAL: **{partial}** variants")
    if no_data:
        lines.append(f"- NO_DATA: {no_data} variants")

    # Failure breakdown
    if per_variant_failures:
        lines.append("")
        lines.append("## Non-bit-exact (graph, seed) pairs")
        lines.append("")
        lines.append(
            "Variants exceeding the bit-exact threshold (1e-3). Use 100-seed TOST for these."
        )
        lines.append("")
        for variant, fails in sorted(per_variant_failures.items()):
            lines.append(f"### {variant} ({len(fails)} failing pairs)")
            lines.append("")
            lines.append("| Graph | Seed | RMSD |")
            lines.append("|---|---:|---:|")
            for graph, seed, rmsd in sorted(fails, key=lambda x: -x[2])[:20]:
                lines.append(f"| {graph} | {seed} | {rmsd:.3e} |")
            if len(fails) > 20:
                lines.append(f"| ... | ... | ({len(fails) - 20} more) |")
            lines.append("")

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines))
    print(f"\nWrote {report_path}")
    print(
        f"Verdicts: MACHINE_EPSILON={machine_eps} BIT_EXACT={bit_exact} STRONG_EQUIV={strong} PARTIAL={partial}"  # noqa: E501
    )

    # Also dump JSON for the per-variant table
    json_path = output_dir / "per_variant.json"
    with json_path.open("w") as f:
        json.dump(
            {
                "summary": summary_rows,
                "failures": {k: v for k, v in per_variant_failures.items()},
                "total_pairs": total_pairs,
            },
            f,
            indent=2,
            default=lambda o: float(o) if isinstance(o, np.floating) else str(o),
        )
    print(f"Wrote {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
