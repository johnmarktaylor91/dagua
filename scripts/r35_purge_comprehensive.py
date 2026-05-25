#!/usr/bin/env python3
"""Comprehensive purge of R31-R35 affected entries from results.json.

After all R31/R32/R33/R34/R35 commits land, this purges:
- ALL dagua reimplementation entries (classic_*) for engines that changed
- References that had R35 seed-threading bugs (so their existing single-seed
  data is wrong)
- References that became newly available in R34 (NeuLay, sgd2_multi_ref, linlog)

A backup is saved to results.json.r35_pre_purge_backup if not already present.

Then `scripts/run_benchmark.py --resume ...` will refill them with current code.
"""

import json
from pathlib import Path

BENCH_OUT = Path("eval_output/benchmark_100seed_final")
RESULTS = BENCH_OUT / "results.json"
BACKUP = BENCH_OUT / "results.json.r35_pre_purge_backup"

# Dagua reimplementation engines changed in R31-R35
AFFECTED_CLASSIC_PREFIXES = [
    "classic_umap",  # R31 + R33 D5+D6+D9
    "classic_lgl",  # R31 + R33 bounded grid
    "classic_graphopt",  # R31 + R33 zero-distance predicate
    "classic_neulay",  # R31 + R34 reference recovered
    "classic_sgd2_multi",  # R31 + R34 reference recovered
    "classic_davidson_harel",  # R31 only
    "classic_drl",  # R32 + R33 DE1+DE2+DE3
    "classic_tsnet",  # R32
    "classic_gem",  # R32 deep OGDF port
    "classic_fa2",  # R32 dissuade_hubs alias
    "classic_stress_sgd",  # R32 term order
    "classic_sfdp",  # R33 graphviz_bitexact attraction
    "classic_fcose",  # R33 NEW engine
    "classic_yifanhu",  # R33 NEW engine
    "classic_linlog",  # R34 reference + variants
    # Also affected via R35 reference seed fixes (paired refs got different data)
    "classic_fr",  # R35 igraph_fr seed bug
    "classic_kk",  # R35 igraph_kk seed bug
]

# References fixed/added in R34/R35 (their existing rows are stale or absent)
AFFECTED_REFERENCE_PREFIXES = [
    "igraph_fr",  # R35 seed bug -- positions were identical across seeds
    "igraph_davidson_harel",  # R35 seed bug
    "igraph_kamada_kawai",  # R35 seed bug
    "cytoscape_fcose",  # R35 seed bug
    "neulay",  # R34 newly available
    "sgd2_multi_ref",  # R34 newly available
    "linlog",  # R34 newly available
]

ALL_AFFECTED = AFFECTED_CLASSIC_PREFIXES + AFFECTED_REFERENCE_PREFIXES


def matches_affected(engine_name: str) -> bool:
    """Return True if engine matches any affected prefix."""
    for prefix in ALL_AFFECTED:
        if engine_name.startswith(prefix):
            return True
    return False


def main() -> int:
    """Purge affected entries and write back."""
    if not RESULTS.is_file():
        print(f"FATAL: {RESULTS} missing")
        return 1

    print(f"Reading {RESULTS}...")
    with RESULTS.open() as f:
        results = json.load(f)
    total = len(results)
    print(f"  total entries: {total}")

    keep = {}
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
