"""W2-2 target-row acceptance verification (review F1).

Runs the sprint2 dev tally on the packet's 12 declared target rows and
ASSERTS the packet's measured acceptance state, so the claims in
W2_2_DONE.md are machine-checked instead of narrated:

- no regression tripwire (exit 0 from the tally),
- every row's status vs the frozen field bests matches the expectation
  table below (which encodes the HONEST post-W2-1/W2-3 result: zero
  incremental tie/behind -> strict conversions from this packet),
- ``north/g.10.75`` -- the row the acceptance bar singled out -- is
  explicitly asserted NOT converted (it stays behind; the in-house stress
  family tops out ~10 points below the incumbent there, see W2_2_DONE.md).

Usage::

    # regenerate + assert (the normal, ~10 min path)
    PYTHONPATH=$PWD python scripts/w22_verify_target_rows.py --jobs 8

    # assert on an existing eval_output/sprint2_dev/run/last_tally.json
    # produced by exactly the TARGET_GRAPHS subset
    PYTHONPATH=$PWD python scripts/w22_verify_target_rows.py --reuse-last

Exit codes: 0 = every assertion holds; 1 = an assertion failed;
3 = tally could not run / wrong subset.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
LAST_TALLY = REPO_ROOT / "eval_output" / "sprint2_dev" / "run" / "last_tally.json"

TARGET_GRAPHS = (
    "north/g.10.0",
    "north/g.10.4",
    "north/g.10.42",
    "north/g.10.45",
    "north/g.10.6",
    "north/g.10.75",
    "north/g.10.78",
    "rome/grafo1001.12",
    "rome/grafo1003.11",
    "rome/grafo1005.11",
    "rome/grafo10044.38",
    "suitesparse/bcspwr01",
)

# The measured post-fixup2 acceptance state on the frozen dev baseline
# (native regenerated at the packet head, deterministic envelope, jobs=8).
# Every value is a REQUIRED status vs the frozen field bests; a mismatch in
# either direction (regression OR unexpected conversion) fails the script so
# the DONE report can never drift from measured reality.
EXPECTED_STATUS = {
    "north/g.10.0": "tied",
    "north/g.10.4": "tied",
    "north/g.10.42": "tied",
    "north/g.10.45": "tied",
    "north/g.10.6": "tied",
    "north/g.10.75": "behind",
    "north/g.10.78": "tied",
    "rome/grafo1001.12": "tied",
    "rome/grafo1003.11": "tied",
    "rome/grafo1005.11": "tied",
    "rome/grafo10044.38": "tied",
    "suitesparse/bcspwr01": "strictly_best",
}
# Rows that must additionally reproduce the frozen baseline positions
# byte-for-byte (the packet's no-movement / gate-inert evidence).
EXPECTED_BYTE_IDENTICAL = ()


def _run_tally(jobs: int) -> int:
    """Regenerate the target-row tally via scripts/sprint2_dev_tally.py."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "sprint2_dev_tally.py"),
        "tally",
        "--graphs",
        ",".join(TARGET_GRAPHS),
        "--jobs",
        str(jobs),
    ]
    return subprocess.run(cmd, cwd=REPO_ROOT, check=False).returncode


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run (or reuse) the target-row tally and assert the acceptance state."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--reuse-last",
        action="store_true",
        help="assert on the existing last_tally.json instead of regenerating",
    )
    args = parser.parse_args(argv)

    if not args.reuse_last:
        tally_rc = _run_tally(args.jobs)
        if tally_rc != 0:
            print(f"FAIL: dev tally exited {tally_rc} (regression tripwire or failed row)")
            return 1
    if not LAST_TALLY.is_file():
        print(f"ERROR: {LAST_TALLY} missing", file=sys.stderr)
        return 3

    payload = json.loads(LAST_TALLY.read_text())
    details = {row["graph"]: row for row in payload["details"]}
    if set(details) != set(TARGET_GRAPHS):
        print(
            "ERROR: last_tally.json covers a different graph subset; rerun without --reuse-last",
            file=sys.stderr,
        )
        return 3

    failures = []
    for graph in TARGET_GRAPHS:
        row = details[graph]
        status = row.get("status")
        if status != EXPECTED_STATUS[graph]:
            failures.append(
                f"{graph}: status {status!r} != expected {EXPECTED_STATUS[graph]!r} "
                f"(delta {row.get('delta', float('nan')):+.3f})"
            )
        if graph in EXPECTED_BYTE_IDENTICAL and row.get("position_match") is not True:
            failures.append(f"{graph}: expected byte-identical positions, got movement")
        if status != row["baseline_status"]:
            failures.append(
                f"{graph}: status changed vs frozen baseline "
                f"({row['baseline_status']} -> {status}); W2-2 claims ZERO "
                "status changes -- update W2_2_DONE.md and this table"
            )

    conversions = [
        graph
        for graph in TARGET_GRAPHS
        if details[graph].get("status") == "strictly_best"
        and details[graph]["baseline_status"] in ("tied", "behind")
    ]
    print(
        f"verified {len(TARGET_GRAPHS)} rows: "
        f"{sum(1 for g in TARGET_GRAPHS if details[g]['status'] == 'strictly_best')} strict / "
        f"{sum(1 for g in TARGET_GRAPHS if details[g]['status'] == 'tied')} tied / "
        f"{sum(1 for g in TARGET_GRAPHS if details[g]['status'] == 'behind')} behind; "
        f"incremental conversions: {len(conversions)}"
    )
    if failures:
        print("FAIL:")
        for line in failures:
            print(f"  {line}")
        return 1
    print("OK: measured acceptance state matches W2_2_DONE.md (0 conversions, no regressions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
