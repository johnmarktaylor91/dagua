"""Shared round-loop tooling: V2 field access, cached scoring, facet forensics, locks.

This module backs the megasprint round-loop tools (``row_forensics.py``,
``regression_locks.py``, ``native_gate.py``). It deliberately reuses the
CORRECTED scoring harness in ``native_sprint_score.py`` (``build_graph_map`` +
``score_position``; the node_sizes tripwire is in) so every tool scores on the
exact V2 scale. Nothing here mutates the frozen ruler.

Data-plane conventions
----------------------
* The V2 competitor field is the pinned
  ``R8_EVENTA_RAW_SCORES_V2_BACKFILL.json`` (121 extended graphs, ~13.6k rows).
* The deterministic native baseline is a benchmark output dir with
  ``results.json`` + ``positions/<graph>__dagua.pt`` (default: megasprint
  ``s1_out``, the banked 117/121 baseline).
* Fresh scores are cached in a JSON score cache keyed by
  ``(graph, position_sha256, scoring_signature)`` so round-over-round reruns
  only pay for changed positions.
"""

from __future__ import annotations

import datetime as _dt
import json
import multiprocessing as mp
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from native_sprint_score import (
    TIE_BAND,
    build_graph_map,
    classify,
    init_worker,
    score_position,
    scoring_signature,
    sha256_file,
)

from dagua.eval.equivalence_metrics import procrustes_rmsd
from dagua.eval.graphs import TestGraph, is_semantically_directed
from dagua.metrics import (
    _CLUSTER_WEIGHTS,
    _COMMON_WEIGHTS,
    _DIRECTED_WEIGHTS,
    composite_auto,
)

V2_FIELD_PATH = Path(
    "/home/jtaylor/.claude/research/dagua/native_sprint/R8_EVENTA_RAW_SCORES_V2_BACKFILL.json"
)
DEFAULT_BASELINE_DIR = Path("/home/jtaylor/.claude/research/dagua/megasprint/s1_out")
ROUNDLOOP_DIR = Path("/home/jtaylor/.claude/research/dagua/megasprint/roundloop")
DEFAULT_CACHE_PATH = ROUNDLOOP_DIR / "scores_cache.json"
DEFAULT_LOCKS_PATH = ROUNDLOOP_DIR / "regression_locks.json"

#: Procrustes RMSD below this = the two layouts are the same drawing
#: (bit-exact tier; unit-cloud normalized so this is scale-free).
DEGENERATE_RMSD = 1e-3
#: Procrustes RMSD below this = suspiciously close to a field engine's layout.
NEAR_RMSD = 0.05

FACET_KEYS: Tuple[str, ...] = tuple(
    {**_COMMON_WEIGHTS, **_DIRECTED_WEIGHTS, **_CLUSTER_WEIGHTS}.keys()
)


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Returns
    -------
    str
        Timezone-aware ISO timestamp.
    """
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def git_sha(repo: Path) -> str:
    """Return the repo HEAD sha (best effort).

    Parameters
    ----------
    repo : Path
        Repository root.

    Returns
    -------
    str
        HEAD sha, or ``"unknown"`` when git is unavailable.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 - provenance only, never fatal
        return "unknown"


# ---------------------------------------------------------------------------
# V2 field access
# ---------------------------------------------------------------------------


def load_v2_field(
    path: Path = V2_FIELD_PATH, require_signature_match: bool = True
) -> Dict[str, Any]:
    """Load the pinned V2 competitor field and verify the ruler signature.

    Parameters
    ----------
    path : Path, optional
        V2 backfill JSON path.
    require_signature_match : bool, optional
        Hard-fail when the current checkout's scoring signature differs from
        the V2 header signature (fresh scores would not be on the V2 scale).

    Returns
    -------
    Dict[str, Any]
        Parsed payload with ``header`` and ``rows``.

    Raises
    ------
    RuntimeError
        If the frozen-ruler scoring signature does not match this checkout.
    """
    payload = json.loads(path.read_text())
    header = payload["header"]
    current = scoring_signature()
    if header["scoring_signature"] != current:
        message = (
            "scoring signature mismatch: V2 field was scored under "
            f"{header['scoring_signature']} but this checkout computes {current}. "
            "Fresh scores would NOT be comparable to the V2 field."
        )
        if require_signature_match:
            raise RuntimeError(message)
        print(f"[roundloop] WARNING: {message}", flush=True)
    return payload


def field_rows_by_graph(
    v2_rows: Iterable[Mapping[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Group scoreable non-native field rows by graph.

    Parameters
    ----------
    v2_rows : Iterable[Mapping[str, Any]]
        Raw V2 rows.

    Returns
    -------
    Dict[str, List[Dict[str, Any]]]
        Non-dagua rows with a usable extended composite, keyed by graph.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in v2_rows:
        if row.get("engine") == "dagua":
            continue
        if row.get("extended_composite") is None:
            continue
        grouped.setdefault(str(row["graph"]), []).append(dict(row))
    return grouped


def field_best_by_graph(
    v2_rows: Iterable[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Select the best non-native field row per graph (extended composite).

    Parameters
    ----------
    v2_rows : Iterable[Mapping[str, Any]]
        Raw V2 rows.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Best field row keyed by graph.
    """
    best: Dict[str, Dict[str, Any]] = {}
    for graph, rows in field_rows_by_graph(v2_rows).items():
        best[graph] = max(rows, key=lambda row: float(row["extended_composite"]))
    return best


def v2_native_scores(v2_rows: Iterable[Mapping[str, Any]]) -> Dict[str, float]:
    """Extract the V2 native (dagua) extended composite per graph.

    Parameters
    ----------
    v2_rows : Iterable[Mapping[str, Any]]
        Raw V2 rows.

    Returns
    -------
    Dict[str, float]
        V2 dagua extended composite keyed by graph.
    """
    return {
        str(row["graph"]): float(row["extended_composite"])
        for row in v2_rows
        if row.get("engine") == "dagua" and row.get("extended_composite") is not None
    }


# ---------------------------------------------------------------------------
# Cached fresh scoring
# ---------------------------------------------------------------------------


class ScoreCache:
    """JSON-backed score cache keyed by (graph, position sha, ruler signature)."""

    def __init__(self, path: Path) -> None:
        """Open (or lazily create) a score cache.

        Parameters
        ----------
        path : Path
            Cache file location (durable dir, not /tmp).
        """
        self.path = path
        self._rows: Dict[str, Dict[str, Any]] = {}
        if path.exists():
            payload = json.loads(path.read_text())
            self._rows = dict(payload.get("rows", {}))

    @staticmethod
    def key(graph: str, position_sha: str, signature: str) -> str:
        """Build the cache key for one scored position.

        Parameters
        ----------
        graph : str
            Graph name.
        position_sha : str
            SHA-256 of the position tensor file.
        signature : str
            Scoring signature the row was produced under.

        Returns
        -------
        str
            Composite cache key.
        """
        return f"{graph}::{position_sha}::{signature}"

    def get(self, graph: str, position_sha: str, signature: str) -> Optional[Dict[str, Any]]:
        """Look up a cached score row.

        Parameters
        ----------
        graph : str
            Graph name.
        position_sha : str
            Position file SHA-256.
        signature : str
            Scoring signature.

        Returns
        -------
        Optional[Dict[str, Any]]
            Cached raw score row, or ``None``.
        """
        return self._rows.get(self.key(graph, position_sha, signature))

    def put(self, row: Mapping[str, Any]) -> None:
        """Insert a fresh score row.

        Parameters
        ----------
        row : Mapping[str, Any]
            Raw row from ``score_position`` (must carry graph, sha, signature).

        Returns
        -------
        None
        """
        key = self.key(
            str(row["graph"]), str(row["position_sha256"]), str(row["scoring_signature"])
        )
        self._rows[key] = dict(row)

    def save(self) -> None:
        """Persist the cache to disk.

        Returns
        -------
        None
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"rows": self._rows}, indent=None))


_WORKER_SIG = ""


def _score_one(task: Tuple[str, str, str]) -> Dict[str, Any]:
    """Worker: score one (graph, engine, path) task.

    Parameters
    ----------
    task : Tuple[str, str, str]
        Graph name, engine name, absolute position path.

    Returns
    -------
    Dict[str, Any]
        Raw score row, or an error row with ``score_origin='fresh_positions_error'``.
    """
    import native_sprint_score as nss

    graph_name, engine, path_string = task
    test_graph = nss._WORKER_GRAPHS[graph_name]
    try:
        return score_position(test_graph, path_string, engine, _WORKER_SIG)
    except Exception as exc:  # noqa: BLE001 - error rows keep the sweep going
        return {
            "graph": graph_name,
            "engine": engine,
            "position_path": path_string,
            "position_sha256": None,
            "metrics": {},
            "old_composite": None,
            "extended_composite": None,
            "scoring_signature": _WORKER_SIG,
            "score_origin": "fresh_positions_error",
            "errors": [f"{type(exc).__name__}: {exc}"],
        }


def _pool_init(graphs: Dict[str, TestGraph], signature: str) -> None:
    """Pool initializer that also pins the module-level signature.

    Parameters
    ----------
    graphs : Dict[str, TestGraph]
        Reconstructed corpus graphs.
    signature : str
        Current scoring signature.

    Returns
    -------
    None
    """
    global _WORKER_SIG
    _WORKER_SIG = signature
    init_worker(graphs, signature)


def score_positions_cached(
    tasks: Sequence[Tuple[str, str, str]],
    graphs: Dict[str, TestGraph],
    cache: ScoreCache,
    signature: str,
    workers: int = 1,
) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """Score position files, reusing the cache for unchanged tensors.

    Parameters
    ----------
    tasks : Sequence[Tuple[str, str, str]]
        ``(graph, engine, absolute position path)`` tuples.
    graphs : Dict[str, TestGraph]
        Graph map from ``build_graph_map`` (node sizes computed).
    cache : ScoreCache
        Persistent score cache.
    signature : str
        Current scoring signature.
    workers : int, optional
        Process count for fresh scoring.

    Returns
    -------
    Dict[Tuple[str, str, str], Dict[str, Any]]
        Raw score row per input task.
    """
    results: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    fresh: List[Tuple[str, str, str]] = []
    for graph_name, engine, path_string in tasks:
        sha = sha256_file(Path(path_string))
        cached = cache.get(graph_name, sha, signature)
        if cached is not None:
            results[(graph_name, engine, path_string)] = dict(cached)
        else:
            fresh.append((graph_name, engine, path_string))
    if fresh:
        print(
            f"[roundloop] scoring {len(fresh)} fresh positions "
            f"({len(results)} cache hits, workers={workers})",
            flush=True,
        )
        if workers <= 1:
            _pool_init(graphs, signature)
            rows = [_score_one(task) for task in fresh]
        else:
            context = mp.get_context("fork")
            with context.Pool(
                workers, initializer=_pool_init, initargs=(graphs, signature)
            ) as pool:
                rows = list(pool.imap(_score_one, fresh, chunksize=1))
        for task, row in zip(fresh, rows):
            results[task] = row
            if row.get("extended_composite") is not None:
                cache.put(row)
        cache.save()
    else:
        print(f"[roundloop] all {len(results)} positions served from cache", flush=True)
    return results


# ---------------------------------------------------------------------------
# Facet forensics
# ---------------------------------------------------------------------------


def facet_swap_gains(
    native_metrics: Mapping[str, Any],
    field_metrics: Mapping[str, Any],
    semantically_directed: bool,
) -> List[Tuple[str, float]]:
    """Rank ruler facets by leave-one-swap composite gain.

    For each weighted facet, replace native's value with the field-best value
    and measure the composite delta under the REAL ruler (``composite_auto``,
    so renormalization / directed gating / cluster conditioning all apply).
    The top facet is the dominant failure mode: the single facet whose parity
    would buy native the most composite.

    Parameters
    ----------
    native_metrics : Mapping[str, Any]
        Native raw metrics payload (must include ``declared_hierarchical``).
    field_metrics : Mapping[str, Any]
        Field-best raw metrics payload.
    semantically_directed : bool
        The graph's frozen semantic-direction routing flag.

    Returns
    -------
    List[Tuple[str, float]]
        ``(facet, gain)`` sorted by descending gain. Positive gain means the
        field is better on that facet.
    """
    base_metrics = dict(native_metrics)
    base = composite_auto(base_metrics, semantically_directed)
    gains: List[Tuple[str, float]] = []
    for facet in FACET_KEYS:
        if facet not in base_metrics and facet not in field_metrics:
            continue
        candidate = dict(base_metrics)
        candidate[facet] = field_metrics.get(facet)
        gains.append((facet, composite_auto(candidate, semantically_directed) - base))
    gains.sort(key=lambda pair: -pair[1])
    return gains


def facet_table(
    native_metrics: Mapping[str, Any], field_metrics: Mapping[str, Any]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Build a per-facet native-vs-field value table.

    Parameters
    ----------
    native_metrics : Mapping[str, Any]
        Native raw metrics payload.
    field_metrics : Mapping[str, Any]
        Field-best raw metrics payload.

    Returns
    -------
    Dict[str, Dict[str, Optional[float]]]
        ``{facet: {"native": ..., "field": ...}}`` for facets present on
        either side.
    """
    table: Dict[str, Dict[str, Optional[float]]] = {}
    for facet in FACET_KEYS:
        native_value = native_metrics.get(facet)
        field_value = field_metrics.get(facet)
        if native_value is None and field_value is None:
            continue
        table[facet] = {
            "native": None if native_value is None else float(native_value),
            "field": None if field_value is None else float(field_value),
        }
    return table


# ---------------------------------------------------------------------------
# Degenerate-tie detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TieMatch:
    """Closest field layout to a native layout for one graph."""

    engine: str
    rmsd: float
    sha_match: bool

    @property
    def degenerate(self) -> bool:
        """Whether the native layout is effectively the field layout.

        Returns
        -------
        bool
            ``True`` on sha identity or bit-exact-tier Procrustes RMSD.
        """
        return self.sha_match or self.rmsd < DEGENERATE_RMSD

    @property
    def near(self) -> bool:
        """Whether the native layout is suspiciously close to the field layout.

        Returns
        -------
        bool
            ``True`` when RMSD is under the near threshold.
        """
        return self.degenerate or self.rmsd < NEAR_RMSD


def closest_field_layout(
    native_path: str,
    native_sha: str,
    field_rows: Sequence[Mapping[str, Any]],
    load_positions: Optional[Callable[[str], torch.Tensor]] = None,
) -> Optional[TieMatch]:
    """Find the field layout closest to a native layout (sha, then Procrustes).

    Parameters
    ----------
    native_path : str
        Native position tensor path.
    native_sha : str
        Native position file SHA-256.
    field_rows : Sequence[Mapping[str, Any]]
        V2 field rows for the graph (need ``engine``, ``position_path``,
        ``position_sha256``).
    load_positions : Optional[Callable[[str], torch.Tensor]], optional
        Position loader override (tests inject synthetic tensors here).

    Returns
    -------
    Optional[TieMatch]
        Closest match, or ``None`` when no field layout was comparable.
    """

    def _default_loader(path: str) -> torch.Tensor:
        return torch.load(path, map_location="cpu", weights_only=True)

    check_exists = load_positions is None
    loader = load_positions if load_positions is not None else _default_loader
    for row in field_rows:
        if row.get("position_sha256") == native_sha:
            return TieMatch(engine=str(row["engine"]), rmsd=0.0, sha_match=True)
    try:
        native = loader(native_path).to(dtype=torch.float64).numpy()
    except Exception:  # noqa: BLE001 - unreadable native tensor: no verdict
        return None
    best: Optional[TieMatch] = None
    for row in field_rows:
        path = row.get("position_path")
        if not path or (check_exists and not Path(str(path)).exists()):
            continue
        try:
            other = loader(str(path)).to(dtype=torch.float64).numpy()
        except Exception:  # noqa: BLE001 - skip unreadable field tensors
            continue
        if other.shape != native.shape:
            continue
        rmsd = float(procrustes_rmsd(native, other))
        if rmsd != rmsd:  # NaN guard
            continue
        if best is None or rmsd < best.rmsd:
            best = TieMatch(engine=str(row["engine"]), rmsd=rmsd, sha_match=False)
    return best


# ---------------------------------------------------------------------------
# Regression locks (pure logic; CLI in regression_locks.py)
# ---------------------------------------------------------------------------

LOCK_SCHEMA = "roundloop-regression-lock-v1"
#: Absolute float-noise epsilon under the floor before a lock fires.
LOCK_EPSILON = 1e-9


@dataclass(frozen=True)
class Lock:
    """One banked best-or-tied row's regression lock."""

    graph: str
    position_sha256: str
    native_extended: float
    field_best: float
    field_best_engine: str
    floor: float

    def to_json(self) -> Dict[str, Any]:
        """Serialize the lock.

        Returns
        -------
        Dict[str, Any]
            JSON-compatible payload.
        """
        return {
            "graph": self.graph,
            "position_sha256": self.position_sha256,
            "native_extended": self.native_extended,
            "field_best": self.field_best,
            "field_best_engine": self.field_best_engine,
            "floor": self.floor,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "Lock":
        """Deserialize a lock.

        Parameters
        ----------
        payload : Mapping[str, Any]
            JSON payload.

        Returns
        -------
        Lock
            Reconstructed lock.
        """
        return cls(
            graph=str(payload["graph"]),
            position_sha256=str(payload["position_sha256"]),
            native_extended=float(payload["native_extended"]),
            field_best=float(payload["field_best"]),
            field_best_engine=str(payload["field_best_engine"]),
            floor=float(payload["floor"]),
        )


@dataclass(frozen=True)
class LockResult:
    """Outcome of checking one lock against a candidate run."""

    graph: str
    status: str  # pass_sha | pass_rescored | fired | missing
    new_score: Optional[float] = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        """Whether the lock held.

        Returns
        -------
        bool
            ``True`` for both pass statuses.
        """
        return self.status in ("pass_sha", "pass_rescored")


def build_locks(
    statuses: Mapping[str, str],
    native_rows: Mapping[str, Mapping[str, Any]],
    field_best: Mapping[str, Mapping[str, Any]],
    tie_band: float = TIE_BAND,
) -> List[Lock]:
    """Build locks for every best-or-tied graph.

    Parameters
    ----------
    statuses : Mapping[str, str]
        Graph name to ``classify`` status.
    native_rows : Mapping[str, Mapping[str, Any]]
        Fresh native raw rows keyed by graph.
    field_best : Mapping[str, Mapping[str, Any]]
        V2 field-best rows keyed by graph.
    tie_band : float, optional
        Frozen tie band; the floor is ``field_best - tie_band``.

    Returns
    -------
    List[Lock]
        Locks for graphs whose status is ``strictly_best`` or ``tied``.
    """
    locks: List[Lock] = []
    for graph, status in sorted(statuses.items()):
        if status not in ("strictly_best", "tied"):
            continue
        native = native_rows[graph]
        best = field_best[graph]
        best_score = float(best["extended_composite"])
        locks.append(
            Lock(
                graph=graph,
                position_sha256=str(native["position_sha256"]),
                native_extended=float(native["extended_composite"]),
                field_best=best_score,
                field_best_engine=str(best["engine"]),
                floor=best_score - tie_band,
            )
        )
    return locks


def evaluate_lock(
    lock: Lock,
    candidate_sha: Optional[str],
    rescore: Callable[[], Optional[float]],
) -> LockResult:
    """Check one lock against a candidate position.

    Parameters
    ----------
    lock : Lock
        Armed lock.
    candidate_sha : Optional[str]
        Candidate position file SHA-256, or ``None`` when the position is
        missing from the candidate run.
    rescore : Callable[[], Optional[float]]
        Lazy scorer for the candidate position (only invoked when the sha
        differs). Returns the extended composite or ``None`` on failure.

    Returns
    -------
    LockResult
        Lock outcome. ``fired`` means a banked row dropped below its floor:
        stop and bisect.
    """
    if candidate_sha is None:
        return LockResult(lock.graph, "missing", detail="candidate position not found")
    if candidate_sha == lock.position_sha256:
        return LockResult(lock.graph, "pass_sha", new_score=lock.native_extended)
    new_score = rescore()
    if new_score is None:
        return LockResult(lock.graph, "fired", detail="candidate position failed to score")
    if new_score < lock.floor - LOCK_EPSILON:
        return LockResult(
            lock.graph,
            "fired",
            new_score=new_score,
            detail=(
                f"score {new_score:.4f} fell below lock floor {lock.floor:.4f} "
                f"(banked native {lock.native_extended:.4f}, "
                f"field best {lock.field_best:.4f} by {lock.field_best_engine})"
            ),
        )
    drift = new_score - lock.native_extended
    return LockResult(
        lock.graph,
        "pass_rescored",
        new_score=new_score,
        detail=f"drift {drift:+.4f} vs banked native",
    )


def summarize_lock_results(results: Sequence[LockResult]) -> Dict[str, Any]:
    """Tally lock results.

    Parameters
    ----------
    results : Sequence[LockResult]
        Per-lock outcomes.

    Returns
    -------
    Dict[str, Any]
        Counts per status, fired graph list, and overall pass flag (missing
        rows do not pass).
    """
    counts: Dict[str, int] = {"pass_sha": 0, "pass_rescored": 0, "fired": 0, "missing": 0}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    fired = [result.graph for result in results if result.status == "fired"]
    return {
        "counts": counts,
        "fired_graphs": fired,
        "ok": counts["fired"] == 0 and counts["missing"] == 0,
    }


# ---------------------------------------------------------------------------
# Baseline access
# ---------------------------------------------------------------------------


def native_position_path(run_dir: Path, graph: str) -> Path:
    """Return the conventional native position path in a benchmark output dir.

    Parameters
    ----------
    run_dir : Path
        Benchmark output dir.
    graph : str
        Graph name.

    Returns
    -------
    Path
        ``<run_dir>/positions/<graph>__dagua.pt``.
    """
    return run_dir / "positions" / f"{graph}__dagua.pt"


def graphs_for_names(names: Sequence[str]) -> Dict[str, TestGraph]:
    """Reconstruct corpus graphs through the corrected harness.

    Parameters
    ----------
    names : Sequence[str]
        Graph names (V2 header ``extended_names`` for full runs).

    Returns
    -------
    Dict[str, TestGraph]
        Graph map with measured node sizes (``build_graph_map`` contract).
    """
    return build_graph_map(list(names))


def semantic_direction_flags(graphs: Mapping[str, TestGraph]) -> Dict[str, bool]:
    """Compute the frozen semantic-direction routing flag per graph.

    Parameters
    ----------
    graphs : Mapping[str, TestGraph]
        Graph map.

    Returns
    -------
    Dict[str, bool]
        ``is_semantically_directed`` per graph name.
    """
    return {name: is_semantically_directed(graph) for name, graph in graphs.items()}


__all__ = [
    "DEFAULT_BASELINE_DIR",
    "DEFAULT_CACHE_PATH",
    "DEFAULT_LOCKS_PATH",
    "DEGENERATE_RMSD",
    "FACET_KEYS",
    "LOCK_EPSILON",
    "LOCK_SCHEMA",
    "Lock",
    "LockResult",
    "NEAR_RMSD",
    "ROUNDLOOP_DIR",
    "ScoreCache",
    "TIE_BAND",
    "TieMatch",
    "V2_FIELD_PATH",
    "build_locks",
    "classify",
    "closest_field_layout",
    "evaluate_lock",
    "facet_swap_gains",
    "facet_table",
    "field_best_by_graph",
    "field_rows_by_graph",
    "git_sha",
    "graphs_for_names",
    "load_v2_field",
    "native_position_path",
    "score_positions_cached",
    "semantic_direction_flags",
    "summarize_lock_results",
    "utc_now_iso",
    "v2_native_scores",
]
