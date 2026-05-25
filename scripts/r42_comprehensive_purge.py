#!/usr/bin/env python3
"""Comprehensive purge of all R36-R41 affected entries from results.json.

R36-R41 sprint pushed every engine's fidelity_mode path closer to bit-exact
against the reference. The 'igraph'/'ogdf'/'graphviz' aliases used by existing
classic_* variants in variants.py all had their algorithmic behavior changed.

This purges all classic_* entries (forcing refill under current code) plus
references that got re-paired during R41 pairing_audit.

References whose seed handling was already correct (per R41 ref_audit) are
left intact -- no need to refill them since the underlying reference adapter
didn't change.

Backup at results.json.r42_pre_purge_backup.
"""

import json
from pathlib import Path

BENCH_OUT = Path("eval_output/benchmark_100seed_final")
RESULTS = BENCH_OUT / "results.json"
BACKUP = BENCH_OUT / "results.json.r42_pre_purge_backup"

# All engines touched by R36-R41 (effectively all dagua engines).
# Purging at the classic_* level catches every variant per engine.
AFFECTED_CLASSIC_PREFIXES = [
    # Graphviz-paired (R36-R40)
    "classic_sugiyama",
    "classic_neato",
    "classic_sfdp",
    "classic_fmmm",
    # R41 wave 1
    "classic_fr",
    "classic_kk",
    "classic_tsnet",
    "classic_umap",
    "classic_drl",
    "classic_davidson_harel",
    # R41 wave 1b
    "classic_fa2",
    "classic_lgl",
    "classic_stress_majorization",
    "classic_classical_mds",
    "classic_spectral",
    "classic_reingold_tilford",
    "classic_dagua_native",
    # R41 wave 2
    "classic_graphopt",
    "classic_sgd2_multi",
    "classic_neulay",
    "classic_stress_sgd",
    "classic_gem",
    "classic_maxent_stress",
    # R41 wave 3
    "classic_pivot_mds",
    "classic_linlog",
    # Newer engines from R33 (not touched in R41 but listed for completeness)
    "classic_fcose",
    "classic_yifanhu",
]

# References that got re-paired in R41 pairing_audit:
# - classic_fmmm_steps10/100/200 now paired with graphviz_fdp (was ogdf_fmmm)
# - classic_spectral_unnormalized now paired with nx_spectral
# - classic_rt_horizontal now paired with igraph_rt
# Need to purge the OLD synthetic original-variant entries so they get refilled
# under the NEW pairing.
REPAIRED_ORIGINAL_PREFIXES = [
    "ogdf_fmmm__for__classic_fmmm_steps",  # 10/100/200 re-paired to graphviz_fdp
    "nx_spectral__for__classic_spectral_unnormalized",
    "igraph_rt__for__classic_rt_horizontal",
    "graphviz_fdp__for__classic_fmmm_steps",  # ensure fresh
]

ALL_AFFECTED = AFFECTED_CLASSIC_PREFIXES + REPAIRED_ORIGINAL_PREFIXES


def matches_affected(engine_name: str) -> bool:
    """Return True if engine matches any affected prefix."""
    return any(engine_name.startswith(p) for p in ALL_AFFECTED)


def main() -> int:
    """Purge R36-R41 affected entries and write back."""
    if not RESULTS.is_file():
        print(f"FATAL: {RESULTS} missing")
        return 1

    print(f"Reading {RESULTS}...")
    with RESULTS.open() as f:
        results = json.load(f)
    total = len(results)
    print(f"  total entries: {total}")

    keep: dict[str, dict] = {}
    purged_counts: dict[str, int] = {}
    for key, value in results.items():
        engine = value.get("engine_name", "")
        if matches_affected(engine):
            purged_counts[engine] = purged_counts.get(engine, 0) + 1
        else:
            keep[key] = value

    purged = total - len(keep)
    print(f"  purged: {purged} entries ({purged * 100 / total:.1f}%)")
    print(f"  kept:   {len(keep)} entries")
    print()
    print("Purge breakdown (top 30):")
    for eng, n in sorted(purged_counts.items(), key=lambda kv: -kv[1])[:30]:
        print(f"  {n:6}  {eng}")

    if not BACKUP.is_file():
        print(f"\nBacking up to {BACKUP}")
        BACKUP.write_text(json.dumps(results, indent=2, sort_keys=True))
    else:
        print(f"\n(backup already exists at {BACKUP}; not overwriting)")

    print(f"Writing trimmed results.json to {RESULTS}")
    with RESULTS.open("w") as f:
        json.dump(keep, f, indent=2, sort_keys=True)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
