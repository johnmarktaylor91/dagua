"""Run bounded corpus-level freeze gates for the V3 ruler."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.native_sprint_score import (  # noqa: E402
    build_graph_map,
    graph_meta_for_v3,
    normalize_position_units_for_scoring,
    score_position,
    scoring_signature,
)

BACKFILL_PATH = Path(
    "/home/jtaylor/.claude/research/dagua/native_sprint/R8_EVENTA_RAW_SCORES_V2_BACKFILL.json"
)
DEFAULT_REPORT_PATH = Path(
    "/home/jtaylor/.claude/research/dagua/megasprint/FREEZE_PART_B_BOUND_REPORT.md"
)
HIGH_CORRELATION_THRESHOLD = 0.85
RANDOM_LAYOUTS_PER_GRAPH = 3
DEFAULT_MAX_REAL_ROWS = 1800
GG6_TIE_EPSILON = 0.5

SPECIALIST_RULES: Tuple[Tuple[str, str, Tuple[str, ...], Tuple[str, ...]], ...] = (
    ("tree", "G4", ("tidy", "rt", "tree", "mrtree"), ("tree",)),
    ("declared_hierarchy", "G1_directed_flow", ("dot", "elk", "dagre", "sugiyama"), ("dag",)),
    ("planar", "C2", ("planar", "fpp", "schnyder", "bertault"), ("planar",)),
    (
        "mesh_lattice",
        "C1",
        ("stress", "mds", "neato", "kk", "kamada"),
        ("mesh", "lattice", "grid"),
    ),
    (
        "multi_scale_np",
        "C3",
        # Defensibility correction: backfill rows show tsne_graph/umap_graph
        # exist but lose C3; neighborhood/stress optimizers are the measured
        # specialist class for multi-radius graph neighborhoods.
        ("tfdp", "sgd2", "elk_stress", "ogdf_stress", "stress"),
        ("scale", "large-sparse", "small-world", "scale-free"),
    ),
    ("declared_cluster", "G2", ("cise", "fcose"), ("clustered",)),
)

PREREGISTERED_SUSPECTS: Tuple[Tuple[str, str], ...] = (
    ("C1", "C3"),
    ("C2", "C1"),
    ("C2", "C6"),
    ("C7", "C1"),
    ("C7", "C4"),
    ("G2_cluster_exclusion", "C4"),
    ("G2_cluster_sibling_overlap", "C4"),
    ("G2_cluster_nesting_fidelity", "C4"),
    ("G2_cluster_edge_intrusion", "C4"),
    ("G2_cluster_label_occlusion", "C4"),
)

SPECIALIST_TOKENS = tuple(
    sorted({token for _family, _facet, tokens, _tags in SPECIALIST_RULES for token in tokens})
)


@dataclass(frozen=True)
class PositionTask:
    """One scoreable graph-engine position tensor."""

    graph: str
    engine: str
    path: Path


@dataclass(frozen=True)
class WinnerResult:
    """GG-7 winner table row."""

    family: str
    facet: str
    expected: str
    actual: str
    rows: int
    status: str


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments for the bounded Part B analysis.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--store",
        type=Path,
        default=Path("eval_output/benchmark_closeout_fullseed"),
    )
    parser.add_argument("--backfill", type=Path, default=BACKFILL_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--max-real-rows", type=int, default=DEFAULT_MAX_REAL_ROWS)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def load_backfill_rows(path: Path) -> List[Mapping[str, Any]]:
    """Load V2 backfill rows used as the engine and graph coverage authority.

    Parameters
    ----------
    path : Path
        JSON file containing a ``rows`` array.

    Returns
    -------
    List[Mapping[str, Any]]
        Backfill score rows.
    """
    payload = json.loads(path.read_text())
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not contain a rows list")
    return [row for row in rows if isinstance(row, Mapping)]


def load_store_tasks(
    store: Path, graph_names: Iterable[str], engines: Iterable[str]
) -> List[PositionTask]:
    """Load one deterministic position path per graph-engine from a result store.

    Parameters
    ----------
    store : Path
        Benchmark directory containing ``results.json`` and relative position files.
    graph_names : Iterable[str]
        Corpus graph names to retain.
    engines : Iterable[str]
        Engine names to retain.

    Returns
    -------
    List[PositionTask]
        Existing one-seed graph-engine position tasks.
    """
    wanted_graphs = set(graph_names)
    wanted_engines = set(engines)
    payload = json.loads((store / "results.json").read_text())
    grouped: Dict[Tuple[str, str], List[Tuple[int, str, Path]]] = defaultdict(list)
    for key, record in payload.items():
        if not isinstance(record, Mapping):
            continue
        if record.get("status") != "ok" or not record.get("positions_file"):
            continue
        graph = str(record.get("graph_name", record.get("graph", "")))
        engine = str(record.get("engine_name", record.get("engine", "")))
        if graph not in wanted_graphs or engine not in wanted_engines:
            continue
        path = (store / str(record["positions_file"])).resolve()
        if not path.exists():
            continue
        seed_value = record.get("seed")
        seed = -1 if seed_value is None else int(seed_value)
        grouped[(graph, engine)].append((seed, str(key), path))
    tasks = [
        PositionTask(graph=graph, engine=engine, path=sorted(candidates)[0][2])
        for (graph, engine), candidates in sorted(grouped.items())
    ]
    return tasks


def bounded_tasks(tasks: Sequence[PositionTask], max_real_rows: int) -> List[PositionTask]:
    """Select a deterministic bounded subset while preserving family coverage.

    Parameters
    ----------
    tasks : Sequence[PositionTask]
        Full one-seed candidate task list.
    max_real_rows : int
        Maximum real position rows to score. Non-positive values keep all rows.

    Returns
    -------
    List[PositionTask]
        Bounded task list.
    """
    if max_real_rows <= 0 or len(tasks) <= max_real_rows:
        return list(tasks)
    graph_seed_limit = min(len({task.graph for task in tasks}), max_real_rows // 2)
    selected = round_robin_by_graph(tasks, graph_seed_limit)
    already = {(task.graph, task.engine, task.path) for task in selected}
    priority: List[PositionTask] = []
    remainder: List[PositionTask] = []
    for task in tasks:
        if (task.graph, task.engine, task.path) in already:
            continue
        is_priority = any(token in task.engine.lower() for token in SPECIALIST_TOKENS)
        target = priority if is_priority else remainder
        target.append(task)
    selected.extend(round_robin_by_engine(priority, max_real_rows - len(selected)))
    if len(selected) >= max_real_rows:
        return selected
    already = {(task.graph, task.engine, task.path) for task in selected}
    remaining = [task for task in remainder if (task.graph, task.engine, task.path) not in already]
    selected.extend(round_robin_by_graph(remaining, max_real_rows - len(selected)))
    return selected


def round_robin_by_engine(tasks: Sequence[PositionTask], limit: int) -> List[PositionTask]:
    """Select tasks by cycling through engines.

    Parameters
    ----------
    tasks : Sequence[PositionTask]
        Candidate tasks.
    limit : int
        Maximum tasks to return.

    Returns
    -------
    List[PositionTask]
        Selected tasks with broad engine coverage.
    """
    by_engine: Dict[str, List[PositionTask]] = defaultdict(list)
    for task in tasks:
        by_engine[task.engine].append(task)
    return drain_round_robin(by_engine, limit)


def round_robin_by_graph(tasks: Sequence[PositionTask], limit: int) -> List[PositionTask]:
    """Select tasks by cycling through graphs.

    Parameters
    ----------
    tasks : Sequence[PositionTask]
        Candidate tasks.
    limit : int
        Maximum tasks to return.

    Returns
    -------
    List[PositionTask]
        Selected tasks with broad graph coverage.
    """
    by_graph: Dict[str, List[PositionTask]] = defaultdict(list)
    for task in tasks:
        by_graph[task.graph].append(task)
    return drain_round_robin(by_graph, limit)


def drain_round_robin(buckets: Mapping[str, List[PositionTask]], limit: int) -> List[PositionTask]:
    """Drain keyed task buckets in deterministic round-robin order.

    Parameters
    ----------
    buckets : Mapping[str, List[PositionTask]]
        Task buckets keyed by graph or engine.
    limit : int
        Maximum tasks to return.

    Returns
    -------
    List[PositionTask]
        Round-robin selected tasks.
    """
    mutable = {key: list(value) for key, value in buckets.items()}
    selected: List[PositionTask] = []
    keys = sorted(mutable)
    cursor = 0
    while len(selected) < limit and keys:
        key = keys[cursor % len(keys)]
        bucket = mutable[key]
        if bucket:
            selected.append(bucket.pop(0))
        if not bucket:
            keys.remove(key)
            if not keys:
                break
            cursor %= len(keys)
        else:
            cursor += 1
    return selected


def score_real_tasks(
    tasks: Sequence[PositionTask], graphs: Mapping[str, Any], signature: str
) -> List[Dict[str, Any]]:
    """Score real positions through the wired ``score_position(..., ruler='v3')`` path.

    Parameters
    ----------
    tasks : Sequence[PositionTask]
        Position tensor tasks to score.
    graphs : Mapping[str, Any]
        Reconstructed graph map from ``build_graph_map``.
    signature : str
        Scoring signature to stamp on rows.

    Returns
    -------
    List[Dict[str, Any]]
        V3 score rows for successfully scored positions.
    """
    rows: List[Dict[str, Any]] = []
    for index, task in enumerate(tasks, start=1):
        row = score_position(graphs[task.graph], str(task.path), task.engine, signature, ruler="v3")
        row["analysis_order"] = index
        rows.append(row)
    return rows


def position_bbox(
    path: Path,
    graph: Any,
    engine: str,
) -> Tuple[float, float, float, float]:
    """Read a position tensor and return its rendered-unit bounding box.

    Parameters
    ----------
    path : Path
        Position tensor path.
    graph : Any
        Graph object whose measured node sizes define rendered point units.
    engine : str
        Engine identifier used to apply the store-unit convention.

    Returns
    -------
    Tuple[float, float, float, float]
        Minimum x, minimum y, maximum x, and maximum y.
    """
    positions = torch.load(path, map_location="cpu", weights_only=True).to(dtype=torch.float32)
    normalized = normalize_position_units_for_scoring(graph, positions, engine)
    mins = normalized.positions.amin(dim=0)
    maxes = normalized.positions.amax(dim=0)
    return (float(mins[0]), float(mins[1]), float(maxes[0]), float(maxes[1]))


def graph_bboxes(
    tasks: Sequence[PositionTask],
    graphs: Mapping[str, Any],
) -> Dict[str, Tuple[float, float, float, float]]:
    """Compute one rendered-layout bounding box per graph for random floors.

    Parameters
    ----------
    tasks : Sequence[PositionTask]
        Real score tasks.
    graphs : Mapping[str, Any]
        Reconstructed graph map from ``build_graph_map``.

    Returns
    -------
    Dict[str, Tuple[float, float, float, float]]
        Graph name to bounding box.
    """
    boxes: Dict[str, Tuple[float, float, float, float]] = {}
    for task in tasks:
        if task.graph not in boxes:
            boxes[task.graph] = position_bbox(task.path, graphs[task.graph].graph, task.engine)
    return boxes


def random_positions(
    num_nodes: int, bbox: Tuple[float, float, float, float], seed: int
) -> torch.Tensor:
    """Generate deterministic uniform random positions in a graph bounding box.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    bbox : Tuple[float, float, float, float]
        Minimum x, minimum y, maximum x, and maximum y.
    seed : int
        Torch generator seed.

    Returns
    -------
    torch.Tensor
        Random positions with shape ``[N, 2]``.
    """
    min_x, min_y, max_x, max_y = bbox
    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    unit = torch.rand((num_nodes, 2), generator=generator, dtype=torch.float32)
    return unit * torch.tensor([width, height], dtype=torch.float32) + torch.tensor(
        [min_x, min_y], dtype=torch.float32
    )


def score_random_rows(
    graph_names: Sequence[str],
    graphs: Mapping[str, Any],
    bboxes: Mapping[str, Tuple[float, float, float, float]],
    signature: str,
) -> List[Dict[str, Any]]:
    """Score deterministic random layouts through the wired V3 path.

    Parameters
    ----------
    graph_names : Sequence[str]
        Graphs to score.
    graphs : Mapping[str, Any]
        Reconstructed graph map from ``build_graph_map``.
    bboxes : Mapping[str, Tuple[float, float, float, float]]
        Real-layout bounding boxes keyed by graph.
    signature : str
        Scoring signature to stamp on rows.

    Returns
    -------
    List[Dict[str, Any]]
        Random-layout V3 score rows.
    """
    rows: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="ruler_v3_random_") as tmpdir:
        tmp_path = Path(tmpdir)
        for graph_index, graph_name in enumerate(sorted(graph_names)):
            test_graph = graphs[graph_name]
            for random_index in range(RANDOM_LAYOUTS_PER_GRAPH):
                seed = 7300 + graph_index * 101 + random_index
                positions = random_positions(test_graph.graph.num_nodes, bboxes[graph_name], seed)
                path = tmp_path / f"{graph_name}__random_{random_index}.pt"
                torch.save(positions, path)
                row = score_position(
                    test_graph,
                    str(path),
                    f"random_uniform_{random_index}",
                    signature,
                    ruler="v3",
                )
                row["random_seed"] = seed
                rows.append(row)
    return rows


def row_facet_score(row: Mapping[str, Any], facet: str) -> Optional[float]:
    """Extract a facet score from a V3 row.

    Parameters
    ----------
    row : Mapping[str, Any]
        V3 score row.
    facet : str
        Exact facet code or prefix such as ``"G4"``.

    Returns
    -------
    Optional[float]
        Facet score, or the mean across matching applicable facet codes.
    """
    facets = row.get("v3_facets", {})
    if not isinstance(facets, Mapping):
        return None
    matches: List[float] = []
    for code, record in facets.items():
        if not isinstance(record, Mapping):
            continue
        if code != facet and not str(code).startswith(f"{facet}_"):
            continue
        if not record.get("applicable", False):
            continue
        value = record.get("score")
        if value is None:
            continue
        matches.append(float(value))
    if not matches:
        return None
    return sum(matches) / len(matches)


def family_for_graph(graph_name: str, graphs: Mapping[str, Any]) -> str:
    """Classify a corpus graph into the primary family used for GG-6 reporting.

    Parameters
    ----------
    graph_name : str
        Corpus graph name.
    graphs : Mapping[str, Any]
        Reconstructed graph map.

    Returns
    -------
    str
        Primary family label.
    """
    test_graph = graphs[graph_name]
    tags = {str(tag).lower() for tag in test_graph.tags}
    meta = graph_meta_for_v3(test_graph)
    if "ports" in meta or "declared_ports" in meta:
        return "port"
    if meta.get("declared_tree"):
        return "tree"
    if meta.get("declared_hierarchical"):
        return "declared_hierarchy"
    if meta.get("clusters"):
        return "declared_cluster"
    if tags & {"mesh", "lattice", "grid"}:
        return "mesh_lattice"
    if "planar" in tags:
        return "planar"
    if tags & {"scale", "large-sparse", "small-world", "scale-free"}:
        return "multi_scale_np"
    return "generic"


def rows_by_family(
    rows: Sequence[Mapping[str, Any]], graphs: Mapping[str, Any]
) -> Dict[str, List[Mapping[str, Any]]]:
    """Group score rows by corpus family.

    Parameters
    ----------
    rows : Sequence[Mapping[str, Any]]
        V3 score rows.
    graphs : Mapping[str, Any]
        Reconstructed graph map.

    Returns
    -------
    Dict[str, List[Mapping[str, Any]]]
        Rows grouped by family.
    """
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        graph_name = str(row["graph"])
        grouped[family_for_graph(graph_name, graphs)].append(row)
    return dict(grouped)


def row_flags(row: Mapping[str, Any]) -> Tuple[str, ...]:
    """Return V3 row flags from a scored row.

    Parameters
    ----------
    row : Mapping[str, Any]
        V3 score row from ``score_position``.

    Returns
    -------
    Tuple[str, ...]
        Row flags such as ``DEGENERATE_SCALE`` and ``OCCLUSION_FLOOR``.
    """
    flags = row.get("v3_row_flags")
    if isinstance(flags, (list, tuple)):
        return tuple(str(flag) for flag in flags)
    metrics = row.get("metrics", {})
    if isinstance(metrics, Mapping):
        metric_flags = metrics.get("v3_flags")
        if isinstance(metric_flags, (list, tuple)):
            return tuple(str(flag) for flag in metric_flags)
    return tuple()


def gg6_block_signal(
    random_row: Mapping[str, Any],
    real_row: Mapping[str, Any],
) -> Optional[str]:
    """Return a GG-6 block line when random beats real beyond acceptance rules.

    Parameters
    ----------
    random_row : Mapping[str, Any]
        Random-layout V3 score row.
    real_row : Mapping[str, Any]
        Real-engine V3 score row.

    Returns
    -------
    Optional[str]
        Human-readable block signal, or ``None`` for ties, flagged rows, and
        non-breaches.
    """
    random_score = float(random_row["v3_tiered"])
    real_score = float(real_row["v3_tiered"])
    if random_score <= real_score + GG6_TIE_EPSILON:
        return None
    random_flags = row_flags(random_row)
    real_flags = row_flags(real_row)
    if random_flags or real_flags:
        return None
    graph = str(random_row["graph"])
    return (
        f"{graph}: {random_row['engine']} {random_score:.3f} > "
        f"{real_row['engine']} {real_score:.3f} (delta={random_score - real_score:.3f})"
    )


def summarize_random_floor(
    real_rows: Sequence[Mapping[str, Any]],
    random_rows: Sequence[Mapping[str, Any]],
    graphs: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Summarize GG-6 random floors by family.

    Parameters
    ----------
    real_rows : Sequence[Mapping[str, Any]]
        V3 rows for real engines.
    random_rows : Sequence[Mapping[str, Any]]
        V3 rows for random layouts.
    graphs : Mapping[str, Any]
        Reconstructed graph map.

    Returns
    -------
    Tuple[List[Dict[str, Any]], List[str]]
        Summary rows and explicit random-beats-real block signals.
    """
    real_by_family = rows_by_family(real_rows, graphs)
    random_by_family = rows_by_family(random_rows, graphs)
    summaries: List[Dict[str, Any]] = []
    block_signals: List[str] = []
    real_by_graph: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in real_rows:
        real_by_graph[str(row["graph"])].append(row)
    for random_row in random_rows:
        graph = str(random_row["graph"])
        for real_row in real_by_graph.get(graph, []):
            signal = gg6_block_signal(random_row, real_row)
            if signal is not None:
                block_signals.append(signal)
    for family in sorted(set(real_by_family) | set(random_by_family)):
        real_scores = [float(row["v3_tiered"]) for row in real_by_family.get(family, [])]
        random_scores = [float(row["v3_tiered"]) for row in random_by_family.get(family, [])]
        if not real_scores or not random_scores:
            continue
        summaries.append(
            {
                "family": family,
                "real_best": max(real_scores),
                "random_best": max(random_scores),
                "gap": max(real_scores) - max(random_scores),
                "real_rows": len(real_scores),
                "random_rows": len(random_scores),
            }
        )
    return summaries, block_signals


def graph_matches_rule(
    graph_name: str, graphs: Mapping[str, Any], tags: Sequence[str], family: str
) -> bool:
    """Return whether a graph belongs to a GG-7 rule family.

    Parameters
    ----------
    graph_name : str
        Corpus graph name.
    graphs : Mapping[str, Any]
        Reconstructed graph map.
    tags : Sequence[str]
        Accepted tag names.
    family : str
        Family label for declaration-aware checks.

    Returns
    -------
    bool
        ``True`` when the graph should be included in the rule.
    """
    test_graph = graphs[graph_name]
    tag_set = {str(tag).lower() for tag in test_graph.tags}
    if family == "declared_hierarchy":
        return bool(graph_meta_for_v3(test_graph).get("declared_hierarchical"))
    if family == "declared_cluster":
        return bool(graph_meta_for_v3(test_graph).get("clusters"))
    if family == "tree":
        return bool(graph_meta_for_v3(test_graph).get("declared_tree"))
    return bool(tag_set & set(tags))


def summarize_facet_winners(
    real_rows: Sequence[Mapping[str, Any]], graphs: Mapping[str, Any]
) -> List[WinnerResult]:
    """Build the GG-7 specialist winner table.

    Parameters
    ----------
    real_rows : Sequence[Mapping[str, Any]]
        V3 rows for real engines.
    graphs : Mapping[str, Any]
        Reconstructed graph map.

    Returns
    -------
    List[WinnerResult]
        Per-family expected-vs-actual winner rows.
    """
    results: List[WinnerResult] = []
    for family, facet, expected_tokens, graph_tags in SPECIALIST_RULES:
        candidates: List[Tuple[float, str, Mapping[str, Any]]] = []
        for row in real_rows:
            graph_name = str(row["graph"])
            if not graph_matches_rule(graph_name, graphs, graph_tags, family):
                continue
            value = row_facet_score(row, facet)
            if value is None or not math.isfinite(value):
                continue
            candidates.append((value, str(row["engine"]), row))
        if not candidates:
            results.append(WinnerResult(family, facet, "/".join(expected_tokens), "N/A", 0, "FLAG"))
            continue
        best_value = max(value for value, _engine, _row in candidates)
        winners = sorted({engine for value, engine, _row in candidates if value == best_value})
        actual = ",".join(winners[:4])
        matched = any(
            any(token in engine.lower() for token in expected_tokens) for engine in winners
        )
        status = "PASS" if matched else "FLAG"
        results.append(
            WinnerResult(family, facet, "/".join(expected_tokens), actual, len(candidates), status)
        )
    results.append(
        WinnerResult(
            "port",
            "G7_port_hard_compliance",
            "elk/dagre/webcola",
            "N/A",
            0,
            "FLAG",
        )
    )
    return results


def rank_values(values: Sequence[float]) -> List[float]:
    """Return average ranks for Spearman correlation.

    Parameters
    ----------
    values : Sequence[float]
        Numeric values.

    Returns
    -------
    List[float]
        Average ranks with ties sharing the midpoint rank.
    """
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0 for _ in values]
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + end - 1) / 2.0 + 1.0
        for index in order[cursor:end]:
            ranks[index] = rank
        cursor = end
    return ranks


def pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Compute a Pearson correlation.

    Parameters
    ----------
    xs : Sequence[float]
        First numeric vector.
    ys : Sequence[float]
        Second numeric vector.

    Returns
    -------
    Optional[float]
        Pearson correlation, or ``None`` for degenerate vectors.
    """
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    denom_x = math.sqrt(sum(value * value for value in centered_x))
    denom_y = math.sqrt(sum(value * value for value in centered_y))
    if denom_x == 0.0 or denom_y == 0.0:
        return None
    return sum(x * y for x, y in zip(centered_x, centered_y)) / (denom_x * denom_y)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Compute Spearman rank correlation.

    Parameters
    ----------
    xs : Sequence[float]
        First numeric vector.
    ys : Sequence[float]
        Second numeric vector.

    Returns
    -------
    Optional[float]
        Spearman rho, or ``None`` for insufficient data.
    """
    return pearson(rank_values(xs), rank_values(ys))


def facet_value_map(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[int, float]]:
    """Collect applicable facet scores by row index.

    Parameters
    ----------
    rows : Sequence[Mapping[str, Any]]
        V3 score rows.

    Returns
    -------
    Dict[str, Dict[int, float]]
        Facet code to row-indexed scores.
    """
    values: Dict[str, Dict[int, float]] = defaultdict(dict)
    for index, row in enumerate(rows):
        facets = row.get("v3_facets", {})
        if not isinstance(facets, Mapping):
            continue
        for code, record in facets.items():
            if not isinstance(record, Mapping) or not record.get("applicable", False):
                continue
            score = record.get("score")
            if score is None:
                continue
            value = float(score)
            if math.isfinite(value):
                values[str(code)][index] = value
    return dict(values)


def summarize_correlations(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Compute high Spearman facet-correlation pairs.

    Parameters
    ----------
    rows : Sequence[Mapping[str, Any]]
        V3 rows for real engines.

    Returns
    -------
    List[Dict[str, Any]]
        Correlation pairs with ``abs(rho)`` above the freeze threshold.
    """
    values = facet_value_map(rows)
    flagged: List[Dict[str, Any]] = []
    codes = sorted(values)
    for left_index, left in enumerate(codes):
        for right in codes[left_index + 1 :]:
            overlap = sorted(set(values[left]) & set(values[right]))
            if len(overlap) < 10:
                continue
            xs = [values[left][index] for index in overlap]
            ys = [values[right][index] for index in overlap]
            rho = spearman(xs, ys)
            if rho is None or abs(rho) <= HIGH_CORRELATION_THRESHOLD:
                continue
            flagged.append({"left": left, "right": right, "rho": rho, "n": len(overlap)})
    return sorted(flagged, key=lambda item: (-abs(float(item["rho"])), item["left"], item["right"]))


def suspect_notes(flagged: Sequence[Mapping[str, Any]]) -> List[str]:
    """Identify pre-registered suspect pairs present in the flagged list.

    Parameters
    ----------
    flagged : Sequence[Mapping[str, Any]]
        High-correlation pair records.

    Returns
    -------
    List[str]
        Human-readable suspect notes.
    """
    notes: List[str] = []
    for left, right in PREREGISTERED_SUSPECTS:
        for item in flagged:
            pair = {str(item["left"]), str(item["right"])}
            if left in pair and right in pair:
                notes.append(f"{left} vs {right}: rho={float(item['rho']):.3f}, n={item['n']}")
    return notes


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Render a simple Markdown table.

    Parameters
    ----------
    headers : Sequence[str]
        Column headers.
    rows : Sequence[Sequence[Any]]
        Table rows.

    Returns
    -------
    str
        Markdown table text.
    """
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    store: Path,
    tasks: Sequence[PositionTask],
    selected_tasks: Sequence[PositionTask],
    random_summary: Sequence[Mapping[str, Any]],
    random_blocks: Sequence[str],
    winners: Sequence[WinnerResult],
    correlations: Sequence[Mapping[str, Any]],
) -> None:
    """Write the Part B bounded-pass report.

    Parameters
    ----------
    path : Path
        Report output path.
    store : Path
        Position store used for real rows.
    tasks : Sequence[PositionTask]
        Full discovered one-seed task field.
    selected_tasks : Sequence[PositionTask]
        Bounded task field actually scored.
    random_summary : Sequence[Mapping[str, Any]]
        GG-6 family gap rows.
    random_blocks : Sequence[str]
        Random-beats-real block signals.
    winners : Sequence[WinnerResult]
        GG-7 winner table.
    correlations : Sequence[Mapping[str, Any]]
        High-correlation pairs.

    Returns
    -------
    None
        Writes the report file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    engines = sorted({task.engine for task in selected_tasks})
    graphs = sorted({task.graph for task in selected_tasks})
    gg6_status = "BLOCK" if random_blocks else "PASS"
    gg7_status = "FLAG" if any(row.status != "PASS" for row in winners) else "PASS"
    redundancy_status = "FLAG" if correlations else "PASS"
    gap_rows = [
        (
            row["family"],
            row["real_rows"],
            row["random_rows"],
            f"{float(row['real_best']):.3f}",
            f"{float(row['random_best']):.3f}",
            f"{float(row['gap']):.3f}",
        )
        for row in random_summary
    ]
    winner_rows = [
        (row.family, row.facet, row.expected, row.actual, row.rows, row.status) for row in winners
    ]
    correlation_rows = [
        (row["left"], row["right"], f"{float(row['rho']):.3f}", row["n"])
        for row in correlations[:80]
    ]
    notes = suspect_notes(correlations)
    text = "\n\n".join(
        [
            "# Freeze Ceremony Part B Bounded Pass",
            (
                f"Data source: `{store}`. Discovered one-seed field: {len(tasks)} rows; "
                f"bounded scored field: {len(selected_tasks)} rows, {len(graphs)} graphs, "
                f"{len(engines)} engines. Random layouts: "
                f"{RANDOM_LAYOUTS_PER_GRAPH} per scored graph."
            ),
            f"Gate status: GG-6 {gg6_status}; GG-7 {gg7_status}; sec 2.3 {redundancy_status}.",
            "## GG-6 Random-vs-Best Gaps\n\n"
            + markdown_table(
                ["family", "real_rows", "random_rows", "real_best", "random_best", "gap"],
                gap_rows,
            ),
            "Random-beats-real aggregate signals:\n"
            + (
                "\n".join(f"- {item}" for item in random_blocks[:50]) if random_blocks else "- none"
            ),
            "## GG-7 Facet Winners\n\n"
            + markdown_table(
                ["family", "facet", "expected winner", "actual winner", "rows", "status"],
                winner_rows,
            ),
            "Port note: base corpus G7 is inapplicable; synthetic-port GG-7 is deferred "
            "to Part C/SA ported family and existing G7 unit tests.",
            "## sec 2.3 High-Correlation Pairs\n\n"
            + (
                markdown_table(["left", "right", "rho", "n"], correlation_rows)
                if correlation_rows
                else "No pairs above |rho| > 0.85."
            ),
            "Pre-registered suspect hits:\n"
            + ("\n".join(f"- {note}" for note in notes) if notes else "- none"),
        ]
    )
    path.write_text(text + "\n")


def main() -> None:
    """Run the bounded Part B analysis.

    Returns
    -------
    None
        Writes a Markdown report and prints its path.
    """
    args = parse_args()
    backfill_rows = load_backfill_rows(args.backfill)
    graph_names = sorted({str(row["graph"]) for row in backfill_rows})
    engines = sorted({str(row["engine"]) for row in backfill_rows})
    tasks = load_store_tasks(args.store, graph_names, engines)
    selected_tasks = bounded_tasks(tasks, int(args.max_real_rows))
    selected_graphs = sorted({task.graph for task in selected_tasks})
    graphs = build_graph_map(selected_graphs)
    signature = scoring_signature()
    real_rows = score_real_tasks(selected_tasks, graphs, signature)
    bboxes = graph_bboxes(selected_tasks, graphs)
    random_rows = score_random_rows(selected_graphs, graphs, bboxes, signature)
    random_summary, random_blocks = summarize_random_floor(real_rows, random_rows, graphs)
    winners = summarize_facet_winners(real_rows, graphs)
    correlations = summarize_correlations(real_rows)
    write_report(
        args.report,
        args.store,
        tasks,
        selected_tasks,
        random_summary,
        random_blocks,
        winners,
        correlations,
    )
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
