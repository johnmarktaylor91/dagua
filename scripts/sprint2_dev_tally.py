"""Sprint-2 fast inner-loop dev tally: regen native ONLY, score vs frozen field bests.

The GLaDOS holdout run (eval_output/glados_holdout) already scored the full
engine field on the 63 dev graphs. This script freezes those field champions
once (``extract``) and then makes the inner loop cheap: regenerate ONLY the
native rows under the exact holdout deterministic envelope, score them with
the same imported V3 machinery, and diff the tally against the frozen
36/14/13 baseline.

Usage::

    # One-time: freeze field bests + copy the 63 dev graphs out of stdcorpora
    python scripts/sprint2_dev_tally.py extract

    # Inner loop: full 63 graphs on 10 workers (<15 min)
    python scripts/sprint2_dev_tally.py tally --jobs 10

    # Subset iteration
    python scripts/sprint2_dev_tally.py tally --graphs north/g.10.0,rome/grafo147.17

Exit codes: 0 = clean; 1 = regression tripwire (a previously best-or-tied row
is now behind) or a native row failed; 3 = preflight/verification failure.

Design pins (all inherited from scripts/glados_holdout_run.py -- reuse, never
reimplement):

- Native rows run in spawn children through ``RowExecutor`` with
  ``deterministic_native_runtime`` + ``deterministic_native=True``
  (the WP03-F02 seam), child seed = run seed, seedless row keys.
- Scoring is ``native_sprint_score.score_position(..., ruler="v3")`` via the
  imported score-pool helpers (NEVER ``score_group``: it silently scores v2).
- Champion/native selection and the 0.5 tie band come from
  ``best_rows_by_graph`` / ``classify`` (symmetric-G6 semantics included).
- Field bests are extracted with the SAME ``best_rows_by_graph`` call the
  holdout tally used, then cross-checked 1:1 against the published
  ``tally.details`` before anything is written -- a mismatch aborts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.glados_holdout_run import (  # noqa: E402
    DEFAULT_CHILD_RSS_ABORT_GB,
    DEFAULT_MAX_EDGES,
    DEFAULT_MAX_NODES,
    DEFAULT_MIN_AVAIL_GB,
    DEFAULT_NATIVE_TIMEOUT,
    DEFAULT_SEED,
    GraphEntry,
    PreflightError,
    RowExecutor,
    _preflight_native_env,
    _score_pool_initializer,
    _score_pool_task,
    build_test_graph_map,
    load_phase,
)

DEFAULT_HOLDOUT_DIR = Path("eval_output/glados_holdout")
DEFAULT_DEV_DIR = Path("eval_output/sprint2_dev")
DEFAULT_CORPUS_DIR = Path("eval_output/stdcorpora")
DEFAULT_JOBS = 10
FIELD_BESTS_NAME = "field_bests.json"
GRAPHS_SUBDIR = "graphs"
RUN_SUBDIR = "run"
# The frozen holdout baseline this harness must reproduce unchanged.
BASELINE_OVERALL = {"strictly_best": 36, "tied": 14, "behind": 13}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Parameters
    ----------
    argv : Sequence[str] | None
        Arguments (``None`` = ``sys.argv[1:]``).

    Returns
    -------
    argparse.Namespace
        Parsed arguments with ``command`` set to ``extract`` or ``tally``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract", help="freeze field bests + copy dev graphs (one-time)")
    extract.add_argument("--holdout-dir", type=Path, default=DEFAULT_HOLDOUT_DIR)
    extract.add_argument("--dev-dir", type=Path, default=DEFAULT_DEV_DIR)
    extract.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    extract.add_argument(
        "--force", action="store_true", help="overwrite an existing field_bests.json"
    )

    tally = sub.add_parser("tally", help="regen native only and diff vs frozen field bests")
    tally.add_argument("--dev-dir", type=Path, default=DEFAULT_DEV_DIR)
    tally.add_argument(
        "--graphs",
        type=str,
        default=None,
        help="comma-separated graph names (e.g. north/g.10.0) to run; default all",
    )
    tally.add_argument("--jobs", type=int, default=DEFAULT_JOBS, help="parallel native regens")
    tally.add_argument("--seed", type=int, default=DEFAULT_SEED)
    tally.add_argument("--native-timeout", type=float, default=DEFAULT_NATIVE_TIMEOUT)
    tally.add_argument("--child-rss-abort-gb", type=float, default=DEFAULT_CHILD_RSS_ABORT_GB)
    tally.add_argument("--min-avail-gb", type=float, default=DEFAULT_MIN_AVAIL_GB)
    tally.add_argument(
        "--allow-signature-drift",
        action="store_true",
        help=(
            "proceed when scoring_signature() differs from the frozen manifest "
            "(scores are then NOT comparable to the frozen field bests unless the "
            "drift is a no-op seam edit; the drift is always printed)"
        ),
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    """Return the sha256 hex digest of a file.

    Parameters
    ----------
    path : Path
        File to hash.

    Returns
    -------
    str
        Hex digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------


def _load_allocation(corpus_dir: Path) -> Dict[str, Set[str]]:
    """Load the dev/sealed relpath allocation from the frozen subset files.

    Parameters
    ----------
    corpus_dir : Path
        stdcorpora root holding SUBSET.json + SEALED_REMAINDER.json.

    Returns
    -------
    Dict[str, Set[str]]
        ``{"dev": relpaths, "sealed": relpaths}``.
    """
    subset = json.loads((corpus_dir / "SUBSET.json").read_text())
    sealed = json.loads((corpus_dir / "SEALED_REMAINDER.json").read_text())
    return {
        "dev": {entry["relpath"] for entry in subset["selected"]},
        "sealed": {entry["relpath"] for entry in sealed["sealed"]},
    }


def run_extract(args: argparse.Namespace) -> int:
    """Freeze per-graph field bests and copy the 63 dev graphs.

    Reads the published holdout ``results.json``, re-derives the field
    champion per graph with the imported selection machinery, cross-checks
    every derived value against the published ``tally.details``, then writes
    ``field_bests.json`` and copies the dev graph files (sha-verified, never
    sealed ones) into the dev graphs directory.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``extract`` arguments.

    Returns
    -------
    int
        Process exit code.
    """
    import scripts.native_sprint_score as nss

    results_path = args.holdout_dir / "results.json"
    if not results_path.is_file():
        print(f"ERROR: {results_path} not found", file=sys.stderr)
        return 3
    manifest_path = args.dev_dir / FIELD_BESTS_NAME
    if manifest_path.is_file() and not args.force:
        print(f"ERROR: {manifest_path} exists; pass --force to rebuild", file=sys.stderr)
        return 3

    payload = json.loads(results_path.read_text())
    rows: List[Dict[str, Any]] = payload["rows"]
    details = {d["graph"]: d for d in payload["tally"]["details"]}

    # Mirror compute_tally: OK rows with a v3_tiered, record_key-sorted so the
    # first-exact-tie winner is deterministic and matches the published run.
    scored = sorted(
        (row for row in rows if row.get("status") == "OK" and row.get("v3_tiered") is not None),
        key=lambda row: str(row.get("record_key")),
    )
    field_best = nss.best_rows_by_graph(scored, "v3_tiered", engine=None)
    native_best = nss.best_rows_by_graph(scored, "v3_tiered", engine="dagua")

    allocation = _load_allocation(args.corpus_dir)
    graphs_dir = args.dev_dir / GRAPHS_SUBDIR
    mismatches: List[str] = []
    recomputed = {"strictly_best": 0, "tied": 0, "behind": 0}
    bests: Dict[str, Dict[str, Any]] = {}
    for graph, detail in sorted(details.items()):
        field_row = field_best.get(graph)
        native_row = native_best.get(graph)
        if field_row is None or native_row is None:
            mismatches.append(f"{graph}: missing {'field' if field_row is None else 'native'} row")
            continue
        checks = [
            ("field_engine", str(field_row["engine"]), detail["field_engine"]),
            ("field_seed", field_row.get("seed"), detail["field_seed"]),
            ("field_v3_tiered", float(field_row["v3_tiered"]), detail["field_v3_tiered"]),
            ("native_v3_tiered", float(native_row["v3_tiered"]), detail["native_v3_tiered"]),
        ]
        for label, derived, published in checks:
            if derived != published:
                mismatches.append(f"{graph}: {label} derived={derived!r} published={published!r}")
        status = nss.classify(float(native_row["v3_tiered"]) - float(field_row["v3_tiered"]))
        if status != detail["status"]:
            mismatches.append(f"{graph}: status derived={status!r} published={detail['status']!r}")
        else:
            recomputed[status] += 1

        source_path = Path(str(native_row["source_path"]))
        relpath = source_path.relative_to(args.corpus_dir).as_posix()
        if relpath in allocation["sealed"]:
            mismatches.append(f"{graph}: {relpath} is SEALED -- refusing to touch it")
            continue
        if relpath not in allocation["dev"]:
            mismatches.append(f"{graph}: {relpath} is not in the dev-63 allocation")
            continue
        bests[graph] = {
            "corpus": native_row["corpus"],
            "relpath": relpath,
            "directed": native_row["directed"],
            "directed_source": native_row["directed_source"],
            "source_path": str(source_path),
            "graph_file_sha256": native_row["graph_file_sha256"],
            "field_engine": str(field_row["engine"]),
            "field_seed": field_row.get("seed"),
            "field_v3_tiered": float(field_row["v3_tiered"]),
            "field_record_key": field_row["record_key"],
            "baseline_native_v3_tiered": float(native_row["v3_tiered"]),
            "baseline_native_position_sha256": native_row.get("position_sha256"),
            "baseline_status": detail["status"],
            "baseline_delta": float(detail["delta"]),
        }

    overall = payload["tally"]["overall"]
    for key, expected in BASELINE_OVERALL.items():
        if overall.get(key) != expected or recomputed.get(key) != expected:
            mismatches.append(
                f"overall[{key}]: published={overall.get(key)} recomputed={recomputed.get(key)} "
                f"expected={expected}"
            )
    if float(overall["tie_band"]) != float(nss.TIE_BAND):
        mismatches.append(f"tie_band: published={overall['tie_band']} != nss.TIE_BAND")
    if mismatches:
        print("EXTRACT ABORTED -- derived field bests do not match the published tally:")
        for line in mismatches:
            print(f"  {line}")
        return 3

    graphs_dir.mkdir(parents=True, exist_ok=True)
    for graph, record in bests.items():
        source = Path(record["source_path"])
        target = graphs_dir / record["relpath"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied_sha = sha256_file(target)
        if copied_sha != record["graph_file_sha256"]:
            print(
                f"EXTRACT ABORTED: {target} sha {copied_sha} != holdout-row sha "
                f"{record['graph_file_sha256']} (source drifted since the holdout run?)",
                file=sys.stderr,
            )
            return 3

    manifest = {
        "holdout_dir": str(args.holdout_dir),
        "holdout_git_sha": payload["git_sha"],
        "holdout_generated_at": payload["generated_at"],
        "scoring_signature": payload["scoring_signature"],
        "subset_sha256": payload["subset_sha256"],
        "tie_band": float(nss.TIE_BAND),
        "baseline_overall": dict(overall),
        "graphs": bests,
    }
    args.dev_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(
        f"extracted {len(bests)} field bests -> {manifest_path}; graphs -> {graphs_dir} "
        f"(baseline {overall['strictly_best']}/{overall['tied']}/{overall['behind']})"
    )
    return 0


# ---------------------------------------------------------------------------
# tally
# ---------------------------------------------------------------------------


def _executor_args(args: argparse.Namespace) -> argparse.Namespace:
    """Build the namespace ``RowExecutor`` expects from tally arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``tally`` arguments.

    Returns
    -------
    argparse.Namespace
        Executor knobs (deterministic envelope always on).
    """
    return argparse.Namespace(
        native_deterministic=True,
        child_rss_abort_gb=args.child_rss_abort_gb,
        min_avail_gb=args.min_avail_gb,
        rss_guard_selftest=False,
        seed=args.seed,
    )


def _load_dev_entries(
    graphs_dir: Path, manifest_graphs: Dict[str, Dict[str, Any]]
) -> List[GraphEntry]:
    """Load and sha-verify the dev graphs through the holdout load phase.

    Parameters
    ----------
    graphs_dir : Path
        Dev graphs directory (corpus-shaped subtree).
    manifest_graphs : Dict[str, Dict[str, Any]]
        Frozen per-graph records from ``field_bests.json``.

    Returns
    -------
    List[GraphEntry]
        Loaded graphs, one per manifest entry.

    Raises
    ------
    PreflightError
        On missing/extra/sha-drifted graphs or load errors.
    """
    relpaths = {record["relpath"] for record in manifest_graphs.values()}
    entries, load_rows, _stats = load_phase(
        graphs_dir, relpaths, DEFAULT_MAX_NODES, DEFAULT_MAX_EDGES
    )
    problems = [
        f"{row['graph']}: {row['status']} {row.get('error') or row.get('reason')}"
        for row in load_rows
    ]
    if problems:
        raise PreflightError("dev graph load problems: " + "; ".join(problems))
    by_name = {entry.name: entry for entry in entries}
    missing = sorted(set(manifest_graphs) - set(by_name))
    if missing:
        raise PreflightError(f"dev graphs missing from {graphs_dir}: {missing}")
    for graph, record in manifest_graphs.items():
        entry = by_name[graph]
        if entry.source_sha256 != record["graph_file_sha256"]:
            raise PreflightError(
                f"{graph}: dev copy sha {entry.source_sha256} != frozen sha "
                f"{record['graph_file_sha256']} -- graphs/ drifted, re-run extract"
            )
        if entry.directed != record["directed"]:
            raise PreflightError(
                f"{graph}: loaded directedness {entry.directed} != frozen {record['directed']}"
            )
    return [by_name[graph] for graph in sorted(manifest_graphs)]


def _regen_native_rows(
    entries: Sequence[GraphEntry], run_dir: Path, args: argparse.Namespace
) -> List[Dict[str, Any]]:
    """Regenerate the native row for each entry in parallel spawn children.

    Parameters
    ----------
    entries : Sequence[GraphEntry]
        Dev graphs to lay out.
    run_dir : Path
        Working directory for positions.
    args : argparse.Namespace
        Parsed ``tally`` arguments.

    Returns
    -------
    List[Dict[str, Any]]
        One layout row per entry (status OK or ERROR).
    """
    executor = RowExecutor(_executor_args(args), run_dir)
    started = time.perf_counter()
    done = {"count": 0}

    def one(entry: GraphEntry) -> Dict[str, Any]:
        row = executor.run_row(
            entry,
            "dagua",
            seed=None,
            child_seed=args.seed,
            timeout_s=args.native_timeout,
            is_native=True,
        )
        done["count"] += 1
        print(
            f"[{done['count']}/{len(entries)}] {row['status']} {entry.name} "
            f"({row['runtime_s']:.1f}s)",
            flush=True,
        )
        return row

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        rows = list(pool.map(one, entries))
    print(f"native regen: {len(rows)} rows in {time.perf_counter() - started:.1f}s")
    return rows


def _score_native_rows(
    rows: List[Dict[str, Any]],
    entries: Sequence[GraphEntry],
    run_dir: Path,
    signature: str,
    jobs: int,
) -> None:
    """Score OK rows in-place under the V3 ruler via the holdout score pool.

    Parameters
    ----------
    rows : List[Dict[str, Any]]
        Layout rows (mutated: score fields merged into OK rows).
    entries : Sequence[GraphEntry]
        Loaded graphs backing the rows.
    run_dir : Path
        Working directory holding the position tensors.
    signature : str
        Current scoring signature.
    jobs : int
        Score-pool worker count.

    Returns
    -------
    None
    """
    graph_map = build_test_graph_map(entries)
    tasks = []
    by_key: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if row["status"] != "OK" or not row.get("positions_path"):
            continue
        key = str(row["record_key"])
        by_key[key] = row
        tasks.append((row["graph"], "dagua", str(run_dir / row["positions_path"]), key))
    if not tasks:
        return
    context = mp.get_context("spawn")
    workers = max(1, min(jobs, len(tasks)))
    with context.Pool(
        processes=workers,
        initializer=_score_pool_initializer,
        initargs=(graph_map, signature),
    ) as pool:
        for key, score, error in pool.imap_unordered(_score_pool_task, tasks):
            row = by_key[key]
            if error is not None:
                row["status"] = "ERROR"
                row["error"] = f"scoring failed: {error}"
                continue
            assert score is not None
            row.update(score)


def run_tally(args: argparse.Namespace) -> int:
    """Regenerate native rows, score them, and diff against the frozen bests.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``tally`` arguments.

    Returns
    -------
    int
        0 clean; 1 regression tripwire or failed native rows; 3 preflight.
    """
    import scripts.native_sprint_score as nss

    manifest_path = args.dev_dir / FIELD_BESTS_NAME
    if not manifest_path.is_file():
        print(f"ERROR: {manifest_path} missing -- run the extract step first", file=sys.stderr)
        return 3
    manifest = json.loads(manifest_path.read_text())
    manifest_graphs: Dict[str, Dict[str, Any]] = manifest["graphs"]

    if args.graphs:
        wanted = {name.strip() for name in args.graphs.split(",") if name.strip()}
        unknown = sorted(wanted - set(manifest_graphs))
        if unknown:
            print(f"ERROR: unknown graphs {unknown}", file=sys.stderr)
            return 3
        manifest_graphs = {name: manifest_graphs[name] for name in wanted}

    try:
        _preflight_native_env()
    except PreflightError as exc:
        print(f"PREFLIGHT FAILED: {exc}", file=sys.stderr)
        return 3

    signature = nss.scoring_signature()
    if signature != manifest["scoring_signature"]:
        drift = (
            "SCORING SIGNATURE DRIFT: current scoring_signature() != the frozen manifest's. "
            "A score-affecting source file changed; fresh native scores are NOT comparable "
            "to the frozen field bests unless the edit is score-inert."
        )
        if not args.allow_signature_drift:
            print(f"{drift}\nPass --allow-signature-drift to proceed anyway.", file=sys.stderr)
            return 3
        print(f"WARNING: {drift} (proceeding under --allow-signature-drift)", flush=True)

    try:
        entries = _load_dev_entries(args.dev_dir / GRAPHS_SUBDIR, manifest_graphs)
    except PreflightError as exc:
        print(f"PREFLIGHT FAILED: {exc}", file=sys.stderr)
        return 3

    run_dir = args.dev_dir / RUN_SUBDIR
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = _regen_native_rows(entries, run_dir, args)
    _score_native_rows(rows, entries, run_dir, signature, args.jobs)

    native_best = nss.best_rows_by_graph(
        [row for row in rows if row["status"] == "OK" and row.get("v3_tiered") is not None],
        "v3_tiered",
        engine="dagua",
    )
    rows_by_graph = {str(row["graph"]): row for row in rows}

    counts = {"strictly_best": 0, "tied": 0, "behind": 0, "missing": 0}
    regressions: List[str] = []
    changes: List[str] = []
    failures: List[str] = []
    details: List[Dict[str, Any]] = []
    header = (
        f"{'graph':<28} {'native':>9} {'field':>9} {'delta':>8}  "
        f"{'status':<13} {'baseline':<13} {'change':<10} pos"
    )
    print("\n" + header)
    print("-" * len(header))
    for graph in sorted(manifest_graphs):
        record = manifest_graphs[graph]
        native_row = native_best.get(graph)
        baseline_status = record["baseline_status"]
        detail: Dict[str, Any] = {
            "graph": graph,
            "field_engine": record["field_engine"],
            "field_v3_tiered": record["field_v3_tiered"],
            "baseline_status": baseline_status,
            "baseline_native_v3_tiered": record["baseline_native_v3_tiered"],
        }
        if native_row is None:
            counts["missing"] += 1
            error = rows_by_graph.get(graph, {}).get("error", "no row")
            failures.append(f"{graph}: native row failed ({error})")
            detail.update({"status": "missing", "error": error})
            details.append(detail)
            print(
                f"{graph:<28} {'-':>9} {record['field_v3_tiered']:>9.3f} {'-':>8}  MISSING: {error}"
            )
            continue
        native_score = float(native_row["v3_tiered"])
        delta = native_score - float(record["field_v3_tiered"])
        status = nss.classify(delta)
        counts[status] += 1
        position_match: Optional[bool] = None
        baseline_sha = record.get("baseline_native_position_sha256")
        if baseline_sha:
            fresh_sha = sha256_file(run_dir / str(native_row["positions_path"]))
            position_match = fresh_sha == baseline_sha
        change = ""
        if status != baseline_status:
            change = f"{baseline_status}->{status}"
            changes.append(
                f"{graph}: {change} (delta {record['baseline_delta']:+.3f} -> {delta:+.3f})"
            )
            if baseline_status in ("strictly_best", "tied") and status == "behind":
                regressions.append(f"{graph}: {change}")
        pos_note = "" if position_match is None else ("match" if position_match else "DIFF")
        detail.update(
            {
                "native_v3_tiered": native_score,
                "delta": delta,
                "status": status,
                "position_match": position_match,
            }
        )
        details.append(detail)
        print(
            f"{graph:<28} {native_score:>9.3f} {record['field_v3_tiered']:>9.3f} "
            f"{delta:>+8.3f}  {status:<13} {baseline_status:<13} {change:<10} {pos_note}"
        )

    total = len(manifest_graphs)
    best_or_tied = counts["strictly_best"] + counts["tied"]
    print(
        f"\nTALLY: {counts['strictly_best']} strict / {counts['tied']} tied / "
        f"{counts['behind']} behind"
        + (f" / {counts['missing']} missing" if counts["missing"] else "")
        + f" of {total} (best-or-tied {best_or_tied}, tie_band {manifest['tie_band']})"
    )
    if changes:
        print("STATUS CHANGES vs frozen baseline:")
        for line in changes:
            print(f"  {line}")
    else:
        print("STATUS CHANGES vs frozen baseline: none")
    for line in failures:
        print(f"FAILED ROW: {line}")

    (run_dir / "last_tally.json").write_text(
        json.dumps(
            {
                "counts": counts,
                "total": total,
                "tie_band": manifest["tie_band"],
                "scoring_signature": signature,
                "signature_matches_manifest": signature == manifest["scoring_signature"],
                "details": details,
            },
            indent=1,
            sort_keys=True,
        )
        + "\n"
    )

    if regressions:
        print("\nREGRESSION TRIPWIRE: previously best-or-tied row(s) now behind:")
        for line in regressions:
            print(f"  {line}")
        return 1
    if failures:
        print("\nFAILED native row(s) -- tally incomplete", file=sys.stderr)
        return 1
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Parameters
    ----------
    argv : Sequence[str] | None
        Arguments (``None`` = ``sys.argv[1:]``).

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    if args.command == "extract":
        return run_extract(args)
    return run_tally(args)


if __name__ == "__main__":
    raise SystemExit(main())
