"""R0.5 fast native gate: minutes-scale sanity tier between banks.

Runs the native engine on a small, family-diverse, fast subset of the frozen
old-108 corpus, checks the armed regression locks on those rows, and runs a
determinism smoke (two graphs laid out twice must produce byte-identical
position tensors). Total wall clock is minutes, not hours -- run it BETWEEN
banks; the full 121-row sweep + lock check remains the banking gate.

The gate subset is curated from the deterministic baseline's runtime profile
(fastest rows) with family coverage: cycles/disconnected, grids, regular,
classic undirected, directed feedback, scale-free, label-heavy, kitchen-sink
clustered, neural-net DAG, bipartite, clustered labels, deep tree. All twelve
are banked (locked) old-108 rows.

Usage
-----
python scripts/native_gate.py [--scratch-dir DIR] [--locks ...] [--keep-scratch]
python scripts/native_gate.py --score-only --scratch-dir DIR   # rescore existing
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import roundloop_common as rl  # noqa: E402
from native_sprint_score import sha256_file  # noqa: E402

#: Curated fast + family-diverse gate subset (old-108, all banked/locked).
GATE_GRAPHS: tuple = (
    "parallel_cycles_4x5",  # cyclic + disconnected packing
    "grid_5x5",  # grid / lattice
    "regular_3_30",  # regular undirected
    "petersen_10",  # classic undirected
    "braided_feedback_tails",  # directed feedback
    "scale_free_ba_120",  # scale-free hubs
    "mixed_width_labels",  # label-driven geometry
    "kitchen_sink_hybrid_net",  # kitchen sink, clustered
    "unet_small",  # neural-net DAG with skips
    "complete_bipartite_8x12",  # dense bipartite crossing pressure
    "clustered_longlabel_handoffs",  # clustered + long labels
    "org_chart_deep",  # deep tree
)
#: Two gate graphs re-run for the determinism smoke.
DETERMINISM_GRAPHS: tuple = ("grid_5x5", "braided_feedback_tails")
DEFAULT_SCRATCH = rl.ROUNDLOOP_DIR / "native_gate_scratch"


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
    parser.add_argument("--scratch-dir", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--locks", type=Path, default=rl.DEFAULT_LOCKS_PATH)
    parser.add_argument("--cache", type=Path, default=rl.DEFAULT_CACHE_PATH)
    parser.add_argument("--workers", type=int, default=4, help="Scoring workers.")
    parser.add_argument("--timeout", type=int, default=300, help="Per-layout timeout (s).")
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Skip layout; check locks against positions already in the scratch dir.",
    )
    parser.add_argument(
        "--skip-determinism",
        action="store_true",
        help="Skip the second determinism pass.",
    )
    parser.add_argument(
        "--keep-scratch",
        action="store_true",
        help="Keep the scratch benchmark dirs after the gate.",
    )
    return parser.parse_args(argv)


def run_native(out_dir: Path, graph_names: Sequence[str], timeout: int) -> None:
    """Run the native engine on a graph subset via the standard harness.

    Parameters
    ----------
    out_dir : Path
        Benchmark output dir (created fresh).
    graph_names : Sequence[str]
        Graphs to lay out.
    timeout : int
        Per-layout timeout seconds.

    Raises
    ------
    RuntimeError
        If the benchmark run exits nonzero.
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "run_benchmark.py"),
        "--engines",
        "dagua",
        "--graphs",
        ",".join(graph_names),
        "--seeds",
        "1",
        "--seed-start",
        "42",
        "--workers",
        "1",
        "--timeout",
        str(timeout),
        "--output-dir",
        str(out_dir),
    ]
    log_path = out_dir / "gate_run.log"
    with log_path.open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"native gate benchmark run failed (rc={result.returncode}); see {log_path}"
        )


def determinism_smoke(primary_dir: Path, repeat_dir: Path, graph_names: Sequence[str]) -> List[str]:
    """Compare position shas between two runs of the same graphs.

    Parameters
    ----------
    primary_dir : Path
        First run output dir.
    repeat_dir : Path
        Second run output dir.
    graph_names : Sequence[str]
        Graphs run in both dirs.

    Returns
    -------
    List[str]
        Human-readable failures (empty when deterministic).
    """
    failures: List[str] = []
    for name in graph_names:
        first = rl.native_position_path(primary_dir, name)
        second = rl.native_position_path(repeat_dir, name)
        if not first.exists() or not second.exists():
            failures.append(f"{name}: missing position file in one of the runs")
            continue
        if sha256_file(first) != sha256_file(second):
            failures.append(f"{name}: position tensors differ between identical runs")
    return failures


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the fast native gate.

    Parameters
    ----------
    argv : Optional[Sequence[str]], optional
        Explicit argument sequence, or ``None`` for ``sys.argv``.

    Returns
    -------
    int
        0 on PASS; 1 on any lock firing, determinism failure, or run error.
    """
    args = parse_args(argv)
    started = time.time()
    scratch = args.scratch_dir
    repeat_dir = scratch.parent / (scratch.name + "_repeat")
    failures: List[str] = []

    if not args.locks.exists():
        print(f"[gate] FAIL: no armed locks at {args.locks}; arm locks first", flush=True)
        return 1

    if not args.score_only:
        print(f"[gate] laying out {len(GATE_GRAPHS)} gate graphs -> {scratch}", flush=True)
        try:
            run_native(scratch, GATE_GRAPHS, args.timeout)
        except RuntimeError as exc:
            print(f"[gate] FAIL: {exc}", flush=True)
            return 1
        if not args.skip_determinism:
            print(
                f"[gate] determinism smoke: re-running {list(DETERMINISM_GRAPHS)}",
                flush=True,
            )
            try:
                run_native(repeat_dir, DETERMINISM_GRAPHS, args.timeout)
            except RuntimeError as exc:
                print(f"[gate] FAIL: {exc}", flush=True)
                return 1
            failures.extend(determinism_smoke(scratch, repeat_dir, DETERMINISM_GRAPHS))

    # Lock check on the gate subset (delegates to the regression-lock harness).
    import regression_locks

    check_args = argparse.Namespace(
        command="check",
        candidate_dir=scratch,
        locks=args.locks,
        cache=args.cache,
        workers=args.workers,
        graphs=",".join(GATE_GRAPHS),
        allow_missing=False,
    )
    lock_rc = regression_locks.check_locks(check_args)

    for failure in failures:
        print(f"[gate] DETERMINISM FAIL: {failure}", flush=True)

    if not args.keep_scratch and not args.score_only:
        shutil.rmtree(scratch, ignore_errors=True)
        shutil.rmtree(repeat_dir, ignore_errors=True)

    elapsed = time.time() - started
    ok = lock_rc == 0 and not failures
    print(
        f"[gate] {'PASS' if ok else 'FAIL'} in {elapsed / 60.0:.1f} min "
        f"({len(GATE_GRAPHS)} rows, determinism "
        f"{'skipped' if args.skip_determinism or args.score_only else 'checked'})",
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
