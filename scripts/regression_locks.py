"""R0.5 regression locks: cheap guards on every banked best-or-tied row.

``arm`` scores the deterministic native baseline fresh (corrected harness),
classifies each extended-corpus row against the pinned V2 field, and writes
one lock per best-or-tied row: the banked position sha, the banked native
extended composite, and the lock floor (``field_best - tie_band``).

``check`` verifies a candidate run dir against the armed locks. Position sha
identity passes without scoring (the deterministic fast path); a changed
position is re-scored and FAILS the check when it drops below its lock floor.
A firing lock means a banked row regressed: STOP and BISECT before banking
anything new.

Usage
-----
python scripts/regression_locks.py arm \\
    [--baseline-dir .../s1_out] [--v2 ...] [--locks .../regression_locks.json]

python scripts/regression_locks.py check --candidate-dir <run_dir> \\
    [--locks ...] [--graphs a,b,c] [--allow-missing] [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import roundloop_common as rl  # noqa: E402
import torch  # noqa: E402
from native_sprint_score import scoring_signature, sha256_file  # noqa: E402


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line options.

    Parameters
    ----------
    argv : Optional[Sequence[str]], optional
        Explicit argument sequence, or ``None`` for ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    arm = sub.add_parser("arm", help="Arm locks from a banked baseline run.")
    arm.add_argument("--baseline-dir", type=Path, default=rl.DEFAULT_BASELINE_DIR)
    arm.add_argument("--v2", type=Path, default=rl.V2_FIELD_PATH)
    arm.add_argument("--locks", type=Path, default=rl.DEFAULT_LOCKS_PATH)
    arm.add_argument("--cache", type=Path, default=rl.DEFAULT_CACHE_PATH)
    arm.add_argument("--workers", type=int, default=8)
    arm.add_argument(
        "--expect-locks",
        type=int,
        default=None,
        help="Fail arming unless exactly this many locks are produced (e.g. 117).",
    )

    check = sub.add_parser("check", help="Check a candidate run against armed locks.")
    check.add_argument("--candidate-dir", type=Path, required=True)
    check.add_argument("--locks", type=Path, default=rl.DEFAULT_LOCKS_PATH)
    check.add_argument("--cache", type=Path, default=rl.DEFAULT_CACHE_PATH)
    check.add_argument("--workers", type=int, default=8)
    check.add_argument(
        "--graphs",
        type=str,
        default=None,
        help="Comma-separated lock subset to check (default: every armed lock).",
    )
    check.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "Treat locks whose candidate position is absent as skipped instead of "
            "failing. Required for subset runs like the fast native gate."
        ),
    )
    return parser.parse_args(argv)


def arm_locks(args: argparse.Namespace) -> int:
    """Arm regression locks from the banked baseline.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``arm`` options.

    Returns
    -------
    int
        Process exit code.
    """
    v2 = rl.load_v2_field(args.v2)
    names = list(v2["header"]["extended_names"])
    signature = scoring_signature()
    graphs = rl.graphs_for_names(names)
    field_best = rl.field_best_by_graph(v2["rows"])

    cache = rl.ScoreCache(args.cache)
    tasks: List[Tuple[str, str, str]] = []
    for name in names:
        path = rl.native_position_path(args.baseline_dir, name)
        if path.exists():
            tasks.append((name, "dagua", str(path)))
    scored = rl.score_positions_cached(tasks, graphs, cache, signature, args.workers)
    native_rows = {task[0]: row for task, row in scored.items()}

    statuses: Dict[str, str] = {}
    for name in names:
        row = native_rows.get(name)
        best = field_best.get(name)
        if row is None or row.get("extended_composite") is None or best is None:
            statuses[name] = "missing"
            continue
        delta = float(row["extended_composite"]) - float(best["extended_composite"])
        statuses[name] = rl.classify(delta)

    locks = rl.build_locks(statuses, native_rows, field_best)
    behind = sorted(name for name, status in statuses.items() if status == "behind")
    if args.expect_locks is not None and len(locks) != args.expect_locks:
        print(
            f"[locks] REFUSING to arm: expected {args.expect_locks} locks, "
            f"got {len(locks)} (behind rows: {behind})",
            flush=True,
        )
        return 1

    payload = {
        "header": {
            "schema": rl.LOCK_SCHEMA,
            "created_at": rl.utc_now_iso(),
            "git_sha": rl.git_sha(SCRIPTS_DIR.parent),
            "scoring_signature": signature,
            "tie_band": rl.TIE_BAND,
            "baseline_dir": str(args.baseline_dir),
            "v2_path": str(args.v2),
            "v2_sha256": sha256_file(args.v2),
            "corpus_size": len(names),
            "lock_count": len(locks),
            "behind_rows": behind,
        },
        "locks": [lock.to_json() for lock in locks],
    }
    args.locks.parent.mkdir(parents=True, exist_ok=True)
    args.locks.write_text(json.dumps(payload, indent=1))
    print(
        f"[locks] armed {len(locks)} locks (behind: {behind}) -> {args.locks}",
        flush=True,
    )
    return 0


def load_locks(path: Path) -> Tuple[Dict[str, Any], List[rl.Lock]]:
    """Load an armed locks file.

    Parameters
    ----------
    path : Path
        Locks JSON path.

    Returns
    -------
    Tuple[Dict[str, Any], List[rl.Lock]]
        Header and locks.

    Raises
    ------
    RuntimeError
        On schema mismatch.
    """
    payload = json.loads(path.read_text())
    header = payload["header"]
    if header.get("schema") != rl.LOCK_SCHEMA:
        raise RuntimeError(f"unexpected locks schema: {header.get('schema')}")
    return header, [rl.Lock.from_json(entry) for entry in payload["locks"]]


def check_locks(args: argparse.Namespace) -> int:
    """Check a candidate run dir against armed locks.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``check`` options.

    Returns
    -------
    int
        0 when every checked lock holds; 1 when any lock fires (or a lock's
        candidate position is missing without ``--allow-missing``).
    """
    header, locks = load_locks(args.locks)
    signature = scoring_signature()
    if header["scoring_signature"] != signature:
        print(
            "[locks] FAIL: scoring signature changed since arming "
            f"({header['scoring_signature'][:16]}... -> {signature[:16]}...). "
            "Locked floors are not comparable; re-arm deliberately after a ruler review.",
            flush=True,
        )
        return 1

    subset = None
    if args.graphs:
        subset = {name.strip() for name in args.graphs.split(",") if name.strip()}
        unknown = subset - {lock.graph for lock in locks}
        if unknown:
            print(f"[locks] FAIL: unknown lock graphs requested: {sorted(unknown)}", flush=True)
            return 1
    selected = [lock for lock in locks if subset is None or lock.graph in subset]

    # Pass 1: cheap sha comparison; collect what needs a fresh score.
    candidate_sha: Dict[str, Optional[str]] = {}
    to_score: List[Tuple[str, str, str]] = []
    for lock in selected:
        path = rl.native_position_path(args.candidate_dir, lock.graph)
        if not path.exists():
            candidate_sha[lock.graph] = None
            continue
        sha = sha256_file(path)
        candidate_sha[lock.graph] = sha
        if sha != lock.position_sha256:
            to_score.append((lock.graph, "dagua", str(path)))

    scored: Dict[str, Optional[float]] = {}
    if to_score:
        graphs = rl.graphs_for_names(sorted({task[0] for task in to_score}))
        cache = rl.ScoreCache(args.cache)
        rows = rl.score_positions_cached(to_score, graphs, cache, signature, args.workers)
        for task, row in rows.items():
            value = row.get("extended_composite")
            scored[task[0]] = None if value is None else float(value)

    def _rescore_lookup(graph: str) -> Callable[[], Optional[float]]:
        def _lookup() -> Optional[float]:
            return scored.get(graph)

        return _lookup

    results: List[rl.LockResult] = []
    skipped = 0
    for lock in selected:
        lock_sha: Optional[str] = candidate_sha[lock.graph]
        if lock_sha is None and args.allow_missing:
            skipped += 1
            continue
        results.append(rl.evaluate_lock(lock, lock_sha, _rescore_lookup(lock.graph)))

    summary = rl.summarize_lock_results(results)
    counts = summary["counts"]
    print(
        f"[locks] checked {len(results)}/{len(selected)} locks "
        f"(skipped missing: {skipped}): "
        f"sha-pass {counts['pass_sha']}, rescored-pass {counts['pass_rescored']}, "
        f"missing {counts['missing']}, FIRED {counts['fired']}",
        flush=True,
    )
    for result in results:
        if result.status == "pass_sha":
            continue
        print(f"  [{result.status.upper():13s}] {result.graph}: {result.detail}", flush=True)
    if not summary["ok"]:
        print(
            "[locks] LOCK FIRED: a banked best-or-tied row regressed. "
            "STOP and BISECT before banking any new change.",
            flush=True,
        )
        return 1
    print("[locks] all checked locks hold", flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Parameters
    ----------
    argv : Optional[Sequence[str]], optional
        Explicit argument sequence, or ``None`` for ``sys.argv``.

    Returns
    -------
    int
        Process exit code.
    """
    torch.set_num_threads(1)
    args = parse_args(argv)
    if args.command == "arm":
        return arm_locks(args)
    return check_locks(args)


if __name__ == "__main__":
    sys.exit(main())
