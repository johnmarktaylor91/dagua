#!/usr/bin/env python3
"""Run the RNG-match harness for one engine and print fixture RMSDs."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rng_match.bitexact_harness import (  # noqa: E402
    DEFAULT_SEEDS,
    expand_engine_names,
    parse_seeds,
    run_harness,
    write_status,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", help="Variant ID or base engine to check.")
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-delimited matched seeds. Default: 1,2,3.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-competitor run timeout in seconds.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run one engine check and print a compact table.

    Parameters
    ----------
    argv : Sequence[str] | None, default=None
        Optional argument vector for tests.

    Returns
    -------
    int
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    seeds = parse_seeds(args.seeds)
    engine_names = expand_engine_names(args.engine)
    rows = run_harness(engine_names, seeds, args.timeout)
    write_status(engine_names, rows)

    print("")
    print("| engine | graph | seed | rmsd | exact_match | status |")
    print("|---|---|---:|---:|---|---|")
    ok_rmsds: list[float] = []
    for row in rows:
        rmsd_text = "--" if row.rmsd is None else f"{row.rmsd:.9e}"
        seed_text = "--" if row.seed is None else str(row.seed)
        row_text = (
            f"| {row.engine} | {row.graph} | {seed_text} | {rmsd_text} | "
            f"{row.exact_match} | {row.status} |"
        )
        print(row_text)
        if row.rmsd is not None:
            ok_rmsds.append(float(row.rmsd))
    max_text = "--" if not ok_rmsds else f"{max(ok_rmsds):.9e}"
    print(f"\nmax RMSD: {max_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
