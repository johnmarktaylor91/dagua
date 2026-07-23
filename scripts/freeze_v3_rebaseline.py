"""Re-score the 121-row native corpus under the frozen V3 ruler."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.native_sprint_score import (  # noqa: E402
    TIE_BAND,
    build_graph_map,
    graph_meta_for_v3,
    score_position,
    scoring_signature,
    selected_names,
)

DEFAULT_BACKFILL = Path(
    "/home/jtaylor/.claude/research/dagua/native_sprint/R8_EVENTA_RAW_SCORES_V2_BACKFILL.json"
)
DEFAULT_OLD_BASELINE = Path(
    "/home/jtaylor/.claude/research/dagua/native_sprint/R8_EVENTA_OLD108_V2.json"
)
DEFAULT_EXTENDED_BASELINE = Path(
    "/home/jtaylor/.claude/research/dagua/native_sprint/R8_EVENTA_EXTENDED121_V2.json"
)
DEFAULT_CORPUS_POSITIONS = ROOT / "eval_output" / "r81_regate2" / "positions"
DEFAULT_OUTPUT_DIR = Path("/home/jtaylor/.claude/research/dagua/megasprint/freeze_v3_rebaseline")
SEVERE_G6_FLOOR = 0.55
DEGENERACY_CHAMPION_INELIGIBLE_FLAGS = frozenset(
    {"DEGENERATE_SCALE", "SPRAWL_COLLAPSE", "COINCIDENT_COLLAPSE"}
)
_WORKER_GRAPHS: Dict[str, Any] = {}
_WORKER_SIGNATURE = ""


@dataclass(frozen=True)
class ScoreTask:
    """One backfill row to re-score under V3."""

    index: int
    graph: str
    engine: str
    position_path: str
    v2_old: Optional[float]
    v2_extended: Optional[float]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : Optional[Sequence[str]], optional
        Explicit command-line arguments, or ``None`` for ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backfill", type=Path, default=DEFAULT_BACKFILL)
    parser.add_argument("--old-baseline", type=Path, default=DEFAULT_OLD_BASELINE)
    parser.add_argument("--extended-baseline", type=Path, default=DEFAULT_EXTENDED_BASELINE)
    parser.add_argument("--corpus-positions", type=Path, default=DEFAULT_CORPUS_POSITIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def load_tasks(backfill_path: Path, limit: int) -> List[ScoreTask]:
    """Load V2 backfill rows as the fixed V3 re-score population.

    Parameters
    ----------
    backfill_path : Path
        V2 backfill cache with a ``rows`` array.
    limit : int
        Optional positive row limit for debugging.

    Returns
    -------
    List[ScoreTask]
        Ordered scoring tasks.
    """
    rows = json.loads(backfill_path.read_text())["rows"]
    tasks: List[ScoreTask] = []
    for index, row in enumerate(rows):
        path = str(row["position_path"])
        if not Path(path).exists():
            raise FileNotFoundError(path)
        tasks.append(
            ScoreTask(
                index=index,
                graph=str(row["graph"]),
                engine=str(row["engine"]),
                position_path=path,
                v2_old=_optional_float(row.get("old_composite")),
                v2_extended=_optional_float(row.get("extended_composite")),
            )
        )
    if limit > 0:
        return tasks[:limit]
    return tasks


def _optional_float(value: Any) -> Optional[float]:
    """Convert a nullable score to ``float``.

    Parameters
    ----------
    value : Any
        Raw score value.

    Returns
    -------
    Optional[float]
        Float value, or ``None`` when absent.
    """
    return None if value is None else float(value)


def init_worker(graphs: Dict[str, Any], signature: str) -> None:
    """Initialize worker-global graph state.

    Parameters
    ----------
    graphs : Dict[str, Any]
        Reconstructed graph map.
    signature : str
        Current scoring signature.

    Returns
    -------
    None
        Mutates process-local globals.
    """
    global _WORKER_GRAPHS, _WORKER_SIGNATURE
    _WORKER_GRAPHS = graphs
    _WORKER_SIGNATURE = signature


def score_task(task: ScoreTask) -> Dict[str, Any]:
    """Score one saved position tensor under V3.

    Parameters
    ----------
    task : ScoreTask
        Backfill row descriptor.

    Returns
    -------
    Dict[str, Any]
        V3 row plus V2 audit columns.
    """
    row = score_position(
        _WORKER_GRAPHS[task.graph],
        task.position_path,
        task.engine,
        _WORKER_SIGNATURE,
        ruler="v3",
    )
    row["source_index"] = task.index
    row["v2_old_composite"] = task.v2_old
    row["v2_extended_composite"] = task.v2_extended
    return row


def score_tasks(
    tasks: Sequence[ScoreTask], graphs: Dict[str, Any], workers: int, signature: str
) -> List[Dict[str, Any]]:
    """Score all tasks, optionally with a worker pool.

    Parameters
    ----------
    tasks : Sequence[ScoreTask]
        Ordered V3 scoring tasks.
    graphs : Dict[str, Any]
        Reconstructed graph map.
    workers : int
        Number of worker processes.
    signature : str
        Current scoring signature.

    Returns
    -------
    List[Dict[str, Any]]
        Ordered V3 rows.
    """
    if workers <= 1:
        init_worker(graphs, signature)
        return [score_task(task) for task in tasks]
    with mp.Pool(workers, initializer=init_worker, initargs=(graphs, signature)) as pool:
        return list(pool.imap(score_task, tasks, chunksize=1))


def score_tasks_incremental(
    tasks: Sequence[ScoreTask],
    graphs: Dict[str, Any],
    signature: str,
    jsonl_path: Path,
    *,
    force: bool,
) -> List[Dict[str, Any]]:
    """Score tasks with a resumable JSONL cache.

    Parameters
    ----------
    tasks : Sequence[ScoreTask]
        Ordered V3 scoring tasks.
    graphs : Dict[str, Any]
        Reconstructed graph map.
    signature : str
        Current scoring signature.
    jsonl_path : Path
        Incremental row cache path.
    force : bool
        Recompute rows even when a cache exists.

    Returns
    -------
    List[Dict[str, Any]]
        Ordered V3 rows.
    """
    init_worker(graphs, signature)
    existing = {} if force else load_jsonl_rows(jsonl_path, signature)
    total = len(tasks)
    started = time.monotonic()
    with jsonl_path.open("w" if force else "a") as handle:
        for offset, task in enumerate(tasks, start=1):
            if task.index in existing:
                continue
            row = score_task(task)
            existing[task.index] = row
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            if offset == total or offset % 100 == 0:
                elapsed = max(time.monotonic() - started, 1.0e-9)
                rate = len(existing) / elapsed
                print(
                    f"[rebaseline] scored {len(existing)}/{total} rows ({rate:.2f} rows/s)",
                    flush=True,
                )
    missing = [task.index for task in tasks if task.index not in existing]
    if missing:
        raise RuntimeError(f"incremental score cache missing {len(missing)} rows")
    return [existing[task.index] for task in tasks]


def load_jsonl_rows(jsonl_path: Path, signature: str) -> Dict[int, Dict[str, Any]]:
    """Load previously scored V3 rows from an incremental cache.

    Parameters
    ----------
    jsonl_path : Path
        JSONL cache path.
    signature : str
        Required scoring signature.

    Returns
    -------
    Dict[int, Dict[str, Any]]
        Cached rows keyed by source index.
    """
    rows: Dict[int, Dict[str, Any]] = {}
    if not jsonl_path.exists():
        return rows
    with jsonl_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("scoring_signature") != signature:
                continue
            rows[int(row["source_index"])] = row
    return rows


def best_by_graph(
    rows: Iterable[Mapping[str, Any]], names: Sequence[str], *, native: bool
) -> Dict[str, Mapping[str, Any]]:
    """Select best V3 row per graph.

    Parameters
    ----------
    rows : Iterable[Mapping[str, Any]]
        Candidate rows.
    names : Sequence[str]
        Graph names to include.
    native : bool
        Whether to select native ``dagua`` rows or field rows.

    Returns
    -------
    Dict[str, Mapping[str, Any]]
        Best row keyed by graph.
    """
    wanted = set(names)
    best: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        graph = str(row["graph"])
        if graph not in wanted:
            continue
        is_native = str(row["engine"]) == "dagua"
        if is_native != native:
            continue
        current_key = _selection_key(row, native=native)
        previous = best.get(graph)
        if previous is None or current_key > _selection_key(previous, native=native):
            best[graph] = row
    return best


def _selection_key(row: Mapping[str, Any], *, native: bool) -> Tuple[int, Tuple[int, float], float]:
    """Build the V3 row-selection key.

    Parameters
    ----------
    row : Mapping[str, Any]
        V3 score row.
    native : bool
        Whether native referee eligibility is binding.

    Returns
    -------
    Tuple[int, Tuple[int, float], float]
        Degeneracy eligibility, severe-G6 eligibility, and V3 tiered score.
    """
    degeneracy_eligible = _row_degeneracy_eligibility_key(row)
    if native:
        raw_key = row.get("v3_referee_eligibility_key", [1, -0.0])
        prefix = (int(raw_key[0]), float(raw_key[1]))
    else:
        prefix = (1, -0.0)
    return (degeneracy_eligible, prefix, float(row["v3_tiered"]))


def _row_degeneracy_eligibility_key(row: Mapping[str, Any]) -> int:
    """Return the row-local degeneracy champion eligibility prefix.

    Parameters
    ----------
    row : Mapping[str, Any]
        V3 score row.

    Returns
    -------
    int
        ``1`` when the row can be selected as champion, otherwise ``0``.
    """
    flags = {str(flag) for flag in row.get("v3_row_flags", [])}
    return int(flags.isdisjoint(DEGENERACY_CHAMPION_INELIGIBLE_FLAGS))


def build_table(
    rows: Sequence[Mapping[str, Any]], names: Sequence[str], graphs: Mapping[str, Any]
) -> Dict[str, Any]:
    """Build a native-vs-field V3 standing table.

    Parameters
    ----------
    rows : Sequence[Mapping[str, Any]]
        V3 score rows.
    names : Sequence[str]
        Graph names in the table.
    graphs : Mapping[str, Any]
        Reconstructed graph map.

    Returns
    -------
    Dict[str, Any]
        Standing table.
    """
    native_best = best_by_graph(rows, names, native=True)
    field_best = best_by_graph(rows, names, native=False)
    tallies = {"strictly_best": 0, "tied": 0, "behind": 0, "missing": 0}
    per_graph: List[Dict[str, Any]] = []
    for graph in names:
        native_row = native_best.get(graph)
        field_row = field_best.get(graph)
        if native_row is None or field_row is None:
            tallies["missing"] += 1
            per_graph.append({"graph": graph, "status": "missing"})
            continue
        delta = float(native_row["v3_tiered"]) - float(field_row["v3_tiered"])
        status = classify(delta)
        tallies[status] += 1
        per_graph.append(
            {
                "graph": graph,
                "family": ",".join(sorted(graphs[graph].tags)) if graphs[graph].tags else "",
                "status": status,
                "native": float(native_row["v3_tiered"]),
                "native_linear": float(native_row["v3_tiered_linear"]),
                "native_tier1_only": float(native_row["v3_tier1_only"]),
                "native_v2_extended": native_row.get("v2_extended_composite"),
                "native_delta_v2_to_v3": _score_delta(
                    native_row.get("v2_extended_composite"), native_row.get("v3_tiered")
                ),
                "native_severe_g6_breach": bool(native_row.get("v3_severe_g6_breach")),
                "native_referee_eligibility_key": native_row.get("v3_referee_eligibility_key"),
                "field_best": float(field_row["v3_tiered"]),
                "field_best_linear": float(field_row["v3_tiered_linear"]),
                "field_best_engine": field_row["engine"],
                "field_best_v2_extended": field_row.get("v2_extended_composite"),
                "field_best_delta_v2_to_v3": _score_delta(
                    field_row.get("v2_extended_composite"), field_row.get("v3_tiered")
                ),
                "delta": delta,
                "native_path": native_row.get("position_path"),
                "field_best_path": field_row.get("position_path"),
            }
        )
    return {
        "corpus_size": len(names),
        "tie_band": TIE_BAND,
        "best_or_tied": tallies["strictly_best"] + tallies["tied"],
        "tallies": tallies,
        "per_graph": per_graph,
    }


def _score_delta(before: Any, after: Any) -> Optional[float]:
    """Return a nullable score delta.

    Parameters
    ----------
    before : Any
        Baseline score.
    after : Any
        New score.

    Returns
    -------
    Optional[float]
        ``after - before`` when both are present.
    """
    if before is None or after is None:
        return None
    return float(after) - float(before)


def classify(delta: float) -> str:
    """Classify native-vs-field standing using the frozen tie band.

    Parameters
    ----------
    delta : float
        Native score minus field-best score.

    Returns
    -------
    str
        ``strictly_best``, ``tied``, or ``behind``.
    """
    if delta > TIE_BAND:
        return "strictly_best"
    if delta >= -TIE_BAND:
        return "tied"
    return "behind"


def rank_flips(rows: Sequence[Mapping[str, Any]], output_path: Path) -> int:
    """Write every within-graph V2-to-V3 engine-pair rank flip.

    Parameters
    ----------
    rows : Sequence[Mapping[str, Any]]
        V3 rows carrying V2 extended scores.
    output_path : Path
        JSONL output path.

    Returns
    -------
    int
        Number of flips written.
    """
    by_graph: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("v2_extended_composite") is None:
            continue
        by_graph.setdefault(str(row["graph"]), []).append(row)
    count = 0
    with output_path.open("w") as handle:
        for graph, graph_rows in sorted(by_graph.items()):
            ordered = sorted(
                graph_rows,
                key=lambda row: (str(row["engine"]), str(row["position_path"])),
            )
            for left_index, left in enumerate(ordered):
                for right in ordered[left_index + 1 :]:
                    v2_delta = float(left["v2_extended_composite"]) - float(
                        right["v2_extended_composite"]
                    )
                    v3_delta = float(left["v3_tiered"]) - float(right["v3_tiered"])
                    if _sign(v2_delta) == 0 or _sign(v3_delta) == 0:
                        continue
                    if _sign(v2_delta) == _sign(v3_delta):
                        continue
                    handle.write(
                        json.dumps(
                            {
                                "graph": graph,
                                "left_engine": left["engine"],
                                "right_engine": right["engine"],
                                "v2_delta": v2_delta,
                                "v3_delta": v3_delta,
                                "left_v2": left["v2_extended_composite"],
                                "right_v2": right["v2_extended_composite"],
                                "left_v3": left["v3_tiered"],
                                "right_v3": right["v3_tiered"],
                                "left_path": left["position_path"],
                                "right_path": right["position_path"],
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    count += 1
    return count


def _sign(value: float) -> int:
    """Return a strict numeric sign with exact-zero neutrality.

    Parameters
    ----------
    value : float
        Numeric value.

    Returns
    -------
    int
        ``-1``, ``0``, or ``1``.
    """
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def severe_g6_native_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """List native rows breaching the severe-G6 floor.

    Parameters
    ----------
    rows : Sequence[Mapping[str, Any]]
        V3 rows.

    Returns
    -------
    List[Dict[str, Any]]
        Breaching native rows with facet detail.
    """
    breaches: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("engine") != "dagua":
            continue
        facet_scores = severe_g6_scores(row)
        breached = {code: score for code, score in facet_scores.items() if score < SEVERE_G6_FLOOR}
        if breached:
            breaches.append(
                {
                    "graph": row["graph"],
                    "v3_tiered": row["v3_tiered"],
                    "eligibility_key": row.get("v3_referee_eligibility_key"),
                    "breached_facets": breached,
                    "position_path": row.get("position_path"),
                }
            )
    return breaches


def severe_g6_scores(row: Mapping[str, Any]) -> Dict[str, float]:
    """Extract applicable severe-G6 facet scores.

    Parameters
    ----------
    row : Mapping[str, Any]
        V3 score row.

    Returns
    -------
    Dict[str, float]
        Applicable severe-G6 facet scores.
    """
    scores: Dict[str, float] = {}
    for code in ("G6_weighted_ksm", "G6_local_weight_monotonicity"):
        facet = row.get("v3_facets", {}).get(code)
        if (
            isinstance(facet, Mapping)
            and facet.get("applicable")
            and facet.get("score") is not None
        ):
            scores[code] = float(facet["score"])
    return scores


def applicability_counts(rows: Sequence[Mapping[str, Any]], *, native_only: bool) -> Dict[str, int]:
    """Count applicable conditional groups.

    Parameters
    ----------
    rows : Sequence[Mapping[str, Any]]
        V3 rows.
    native_only : bool
        Restrict counts to native rows when true.

    Returns
    -------
    Dict[str, int]
        Applicable row counts keyed by group id.
    """
    counts = {group: 0 for group in ("G1", "G2", "G3", "G4", "G5", "G6", "G7")}
    for row in rows:
        if native_only and row.get("engine") != "dagua":
            continue
        applicability = row.get("v3_applicability", {})
        for group in counts:
            if any(
                bool(value) and (str(code) == group or str(code).startswith(f"{group}_"))
                for code, value in applicability.items()
            ):
                counts[group] += 1
    return counts


def softmin_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarize family softmin haircuts.

    Parameters
    ----------
    rows : Sequence[Mapping[str, Any]]
        V3 score rows.

    Returns
    -------
    Dict[str, Any]
        Corpus-wide and clustered-family haircut summary.
    """
    all_haircuts: List[float] = []
    clustered_haircuts: List[float] = []
    clustered_tail = 0
    clustered_rows = 0
    for row in rows:
        haircut = max(0.0, float(row["v3_tiered_linear"]) - float(row["v3_tiered"]))
        all_haircuts.append(haircut)
        family = score_family(row)
        if family == "clustered":
            clustered_rows += 1
            clustered_haircuts.append(haircut)
            if haircut > 1.0:
                clustered_tail += 1
    return {
        "all_rows": len(all_haircuts),
        "clustered_rows": clustered_rows,
        "clustered_tail_gt_1": clustered_tail,
        "clustered_p50": percentile(clustered_haircuts, 0.50),
        "clustered_p90": percentile(clustered_haircuts, 0.90),
        "clustered_max": max(clustered_haircuts) if clustered_haircuts else None,
        "all_bound_rows": sum(1 for value in all_haircuts if value > 1.0e-9),
    }


def score_family(row: Mapping[str, Any]) -> str:
    """Infer the softmin family for a serialized V3 row.

    Parameters
    ----------
    row : Mapping[str, Any]
        V3 score row.

    Returns
    -------
    str
        Frozen family-envelope key.
    """
    meta = row.get("graph_meta", {})
    for key in ("ruler_family", "family", "v3_family"):
        value = meta.get(key) if isinstance(meta, Mapping) else None
        if isinstance(value, str) and value:
            return value
    if isinstance(meta, Mapping) and "edge_weights" in meta:
        return "weighted"
    if any(str(code).startswith("G2_") for code in row.get("v3_facets", {})):
        return "clustered"
    if any(str(code).startswith("G1_") for code in row.get("v3_facets", {})):
        return "dag"
    if any(str(code).startswith("G4_") for code in row.get("v3_facets", {})):
        return "tree"
    return "generic_force"


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    """Compute a nearest-rank percentile for a small audit list.

    Parameters
    ----------
    values : Sequence[float]
        Numeric values.
    q : float
        Quantile in ``[0, 1]``.

    Returns
    -------
    Optional[float]
        Percentile value, or ``None`` for an empty list.
    """
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(float(q) * len(ordered)) - 1))
    return ordered[index]


def load_baseline(path: Path) -> Dict[str, Any]:
    """Load one prior V2 baseline table.

    Parameters
    ----------
    path : Path
        JSON table path.

    Returns
    -------
    Dict[str, Any]
        Baseline payload.
    """
    return json.loads(path.read_text())


def write_report(
    output_path: Path,
    old_table: Mapping[str, Any],
    extended_table: Mapping[str, Any],
    old_baseline: Mapping[str, Any],
    extended_baseline: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    flip_count: int,
    flip_path: Path,
    breach_rows: Sequence[Mapping[str, Any]],
    softmin: Mapping[str, Any],
) -> None:
    """Write the re-baseline markdown report.

    Parameters
    ----------
    output_path : Path
        Markdown report path.
    old_table : Mapping[str, Any]
        OLD-108 V3 table.
    extended_table : Mapping[str, Any]
        EXTENDED-121 V3 table.
    old_baseline : Mapping[str, Any]
        Prior OLD-108 baseline table.
    extended_baseline : Mapping[str, Any]
        Prior EXTENDED-121 baseline table.
    rows : Sequence[Mapping[str, Any]]
        V3 raw rows.
    flip_count : int
        Number of engine-pair flips.
    flip_path : Path
        Rank-flip JSONL path.
    breach_rows : Sequence[Mapping[str, Any]]
        Severe-G6 native breach rows.
    softmin : Mapping[str, Any]
        Softmin audit summary.

    Returns
    -------
    None
        Writes a markdown file.
    """
    native_counts = applicability_counts(rows, native_only=True)
    all_counts = applicability_counts(rows, native_only=False)
    lines = [
        "# V3 Native Corpus Re-baseline",
        "",
        f"- OLD-108 native best-or-tied: {old_table['best_or_tied']}/108 "
        f"(prior V2 baseline {old_baseline['best_or_tied']}/108, "
        f"delta {int(old_table['best_or_tied']) - int(old_baseline['best_or_tied']):+d})",
        f"- EXTENDED-121 native best-or-tied: {extended_table['best_or_tied']}/121 "
        f"(prior V2 baseline {extended_baseline['best_or_tied']}/121, "
        f"delta {int(extended_table['best_or_tied']) - int(extended_baseline['best_or_tied']):+d})",
        f"- OLD tallies: {old_table['tallies']}",
        f"- EXTENDED tallies: {extended_table['tallies']}",
        f"- Conditional applicability, native rows: {native_counts}",
        f"- Conditional applicability, all rows: {all_counts}",
        f"- Rank flips: {flip_count} (`{flip_path}`)",
        f"- Native severe-G6 tripwire: {'FIRED' if breach_rows else 'CLEAR'}",
        f"- Native severe-G6 breach rows: {len(breach_rows)}",
        f"- Softmin audit: {dict(softmin)}",
        "",
        "## Native Severe-G6 Rows",
        "",
    ]
    if breach_rows:
        lines.extend(["| graph | tiered | facets | path |", "|---|---:|---|---|"])
        for row in breach_rows:
            lines.append(
                f"| {row['graph']} | {float(row['v3_tiered']):.6f} | "
                f"{row['breached_facets']} | `{row['position_path']}` |"
            )
    else:
        lines.append("None.")
    lines.extend(
        [
            "",
            "## Behind Rows",
            "",
            "| graph | delta | native | field best | by |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in extended_table["per_graph"]:
        if row.get("status") == "behind":
            lines.append(
                f"| {row['graph']} | {float(row['delta']):.6f} | {float(row['native']):.6f} | "
                f"{float(row['field_best']):.6f} | {row['field_best_engine']} |"
            )
    output_path.write_text("\n".join(lines) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the V3 re-baseline.

    Parameters
    ----------
    argv : Optional[Sequence[str]], optional
        Explicit command-line arguments.

    Returns
    -------
    int
        Process exit status.
    """
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    old_names, extended_names = selected_names(args.corpus_positions, include_r8=True)
    graphs = build_graph_map(extended_names)
    for name in extended_names:
        meta = graph_meta_for_v3(graphs[name])
        graphs[name].graph_meta_for_v3 = meta  # type: ignore[attr-defined]
    tasks = load_tasks(args.backfill, args.limit)
    signature = scoring_signature()
    if args.workers > 1:
        print(
            "[rebaseline] multiprocessing is disabled for this measurement helper; "
            "using resumable single-process scoring",
            flush=True,
        )
    rows = score_tasks_incremental(
        tasks,
        graphs,
        signature,
        args.output_dir / "v3_rebaseline_rows.jsonl",
        force=bool(args.force),
    )
    rows_path = args.output_dir / "v3_rebaseline_rows.json"
    rows_path.write_text(json.dumps({"scoring_signature": signature, "rows": rows}, indent=1))
    old_table = build_table(rows, old_names, graphs)
    extended_table = build_table(rows, extended_names, graphs)
    old_path = args.output_dir / "V3_OLD108.json"
    extended_path = args.output_dir / "V3_EXTENDED121.json"
    old_path.write_text(json.dumps(old_table, indent=1))
    extended_path.write_text(json.dumps(extended_table, indent=1))
    flip_path = args.output_dir / "rank_flips_v2_to_v3.jsonl"
    flip_count = rank_flips(rows, flip_path)
    breach_rows = severe_g6_native_rows(rows)
    breach_path = args.output_dir / "native_severe_g6_breaches.json"
    breach_path.write_text(json.dumps(breach_rows, indent=1))
    softmin = softmin_summary(rows)
    softmin_path = args.output_dir / "softmin_summary.json"
    softmin_path.write_text(json.dumps(softmin, indent=1))
    write_report(
        args.output_dir / "V3_REBASELINE_REPORT.md",
        old_table,
        extended_table,
        load_baseline(args.old_baseline),
        load_baseline(args.extended_baseline),
        rows,
        flip_count,
        flip_path,
        breach_rows,
        softmin,
    )
    print(
        f"[rebaseline] old {old_table['best_or_tied']}/108; "
        f"extended {extended_table['best_or_tied']}/121; flips {flip_count}; "
        f"native severe-G6 breaches {len(breach_rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
