#!/usr/bin/env python3
"""Purge R31-affected engine entries from results.json before focal rerun.

Reads `eval_output/benchmark_100seed_final/results.json`, removes entries for
engines whose dagua-side code changed in R31 commits, writes back. The
focal rerun then re-fills them via `--resume`.

R31-affected engines:
- classic_umap_*  (per a6fd45c)
- classic_lgl_*   (per 6175275)
- classic_graphopt_*  (per e69de63)
- Plus infra-recovery: classic_neulay_*, classic_sgd2_multi_*, classic_davidson_harel_*
  (these may have error/skipped status; force redo)
"""

import json
from pathlib import Path

BENCH_OUT = Path("eval_output/benchmark_100seed_final")
RESULTS = BENCH_OUT / "results.json"
BACKUP = BENCH_OUT / "results.json.r31_pre_purge_backup"

AFFECTED_ENGINE_PREFIXES = [
    "classic_umap",
    "classic_lgl",
    "classic_graphopt",
    "classic_neulay",
    "classic_sgd2_multi",
    "classic_davidson_harel",
]


def matches_affected(engine_name: str) -> bool:
    """Return True if engine name starts with an R31-affected prefix."""
    for prefix in AFFECTED_ENGINE_PREFIXES:
        if engine_name.startswith(prefix):
            return True
    return False


def main() -> int:
    """Purge affected entries and write back."""
    if not RESULTS.is_file():
        print(f"FATAL: {RESULTS} not found")
        return 1

    print(f"Reading {RESULTS}...")
    with RESULTS.open() as f:
        results = json.load(f)
    total = len(results)
    print(f"  total entries: {total}")

    # Filter
    keep = {}
    purged_engine_counts: dict[str, int] = {}
    for key, value in results.items():
        engine = value.get("engine_name", "")
        if matches_affected(engine):
            purged_engine_counts[engine] = purged_engine_counts.get(engine, 0) + 1
        else:
            keep[key] = value

    print(f"  purged {total - len(keep)} entries; kept {len(keep)}")
    print()
    print("Purge breakdown:")
    for eng, n in sorted(purged_engine_counts.items()):
        print(f"  {n:6} {eng}")

    # Backup + write
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
