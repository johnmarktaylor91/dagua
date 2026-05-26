#!/usr/bin/env python3
"""Purge gem + drl entries from results.json for 100-seed rerun.

Only purges classic_gem* and classic_drl* (+ their paired reference entries).
Leaves the 22 bit-exact engines' 5-seed data intact.
"""

import json
from pathlib import Path

BENCH_OUT = Path("eval_output/benchmark_100seed_final")
RESULTS = BENCH_OUT / "results.json"
BACKUP = BENCH_OUT / "results.json.r67_pre_purge_backup"

AFFECTED_PREFIXES = [
    "classic_gem",
    "classic_drl",
    "ogdf_gem__for__classic_gem",
    "igraph_drl__for__classic_drl",
]


def matches_affected(engine_name: str) -> bool:
    """Return True if engine matches any affected prefix."""
    return any(engine_name.startswith(p) for p in AFFECTED_PREFIXES)


def main() -> int:
    """Purge gem + drl entries."""
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
    print("Purge breakdown:")
    for eng, n in sorted(purged_counts.items(), key=lambda kv: -kv[1])[:20]:
        print(f"  {n:6}  {eng}")

    if not BACKUP.is_file():
        print(f"\nBacking up to {BACKUP}")
        BACKUP.write_text(json.dumps(results, indent=2, sort_keys=True))

    print(f"Writing trimmed results.json to {RESULTS}")
    with RESULTS.open("w") as f:
        json.dump(keep, f, indent=2, sort_keys=True)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
