#!/usr/bin/env python3
"""Build fidelity analysis datasets for reimplementation-vs-original benchmarks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
from scipy.stats import ks_2samp, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.weightstats import ttost_ind

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dagua.eval.variants import (  # noqa: E402
    VARIANT_REGISTRY,
    AlgorithmVariant,
    algorithm_family,
    original_variant_name,
)
from dagua.metrics import quick  # noqa: E402

QUALITY_METRICS: tuple[str, ...] = (
    "aspect_ratio",
    "dag_consistency",
    "edge_length_cv",
    "edge_length_mean",
    "overlap_count",
)
TOST_MARGIN_FACTORS: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
TOST_MARGIN_LABELS: dict[float, str] = {
    0.5: "0_5x",
    1.0: "1x",
    1.5: "1_5x",
    2.0: "2x",
}
METRIC_MARGIN_FLOORS: dict[str, float] = {
    "aspect_ratio": 0.05,
    "dag_consistency": 0.02,
    "edge_length_cv": 0.05,
    "edge_length_mean": 1.0,
    "overlap_count": 1.0,
}
MIN_VALID_NODE_COUNT = 3
MIN_PROCRUSTES_NODE_COUNT = 5
MIN_STOCHASTIC_SEEDS = 10
PAIRWISE_SAMPLE_SIZE = 10
PAIRWISE_PROGRESS_INTERVAL = 1_000
POWER_TARGET = 0.8
POWER_ALPHA = 0.05
POWER_EFFECT_GRID = tuple(round(step * 0.05, 2) for step in range(2, 51))
POWER_SIMULATIONS = 2_000
PROCRUSTES_ANOMALY_THRESHOLD = 1.0
IDENTICAL_DISPLACEMENT_THRESHOLD = 1e-4
SCALE_RATIO_LOWER = 0.8
SCALE_RATIO_UPPER = 1.25
RUNTIME_RATIO_WARNING_LOWER = 0.5
RUNTIME_RATIO_WARNING_UPPER = 2.0
RUNTIME_RATIO_ANOMALY_LOWER = 0.33
RUNTIME_RATIO_ANOMALY_UPPER = 3.0
README_HASH_PREFIX = "Results SHA-256:"


@dataclass(frozen=True)
class LayoutRecord:
    """One validated layout sample.

    Parameters
    ----------
    graph_name : str
        Stable benchmark graph name.
    variant_id : str
        Reimplementation variant identifier.
    side : str
        Either ``"orig"`` or ``"reimpl"``.
    seed : int | None
        Benchmark seed when present.
    runtime_seconds : float | None
        Recorded benchmark runtime.
    positions : torch.Tensor
        Layout coordinates with shape ``[N, 2]``.
    metrics : dict[str, float]
        Selected quick-metric values for this layout.
    """

    graph_name: str
    variant_id: str
    side: str
    seed: Optional[int]
    runtime_seconds: Optional[float]
    positions: torch.Tensor
    metrics: dict[str, float]


@dataclass(frozen=True)
class ResultRecord:
    """Minimal benchmark record used by the fidelity analysis.

    Parameters
    ----------
    graph_name : str
        Stable benchmark graph name.
    engine_name : str
        Variant or original-side engine identifier.
    seed : int | None
        Benchmark seed when present.
    status : str
        Benchmark run status.
    runtime_seconds : float | None
        Recorded runtime in seconds.
    positions_file : str | None
        Relative path to the saved position tensor.
    """

    graph_name: str
    engine_name: str
    seed: Optional[int]
    status: str
    runtime_seconds: Optional[float]
    positions_file: Optional[str]
    result_key: str = ""


@dataclass(frozen=True)
class GraphDescriptor:
    """Structural metadata used for anomaly interpretation and reporting.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    num_edges : int
        Number of graph edges.
    density_bucket : str
        Coarse density label.
    size_bucket : str
        Coarse size label.
    structure_bucket : str
        Structural family label.
    is_disconnected : bool
        Whether the undirected graph has multiple connected components.
    structural_note : str
        Human-readable note carried into the report.
    tags : tuple[str, ...]
        Stable, sorted graph tags from the registry.
    """

    num_nodes: int
    num_edges: int
    density_bucket: str
    size_bucket: str
    structure_bucket: str
    is_disconnected: bool
    structural_note: str
    tags: tuple[str, ...]


@dataclass(frozen=True)
class PairwiseComparison:
    """One pairwise Procrustes comparison.

    Parameters
    ----------
    comparison_type : str
        Pair label: ``orig-orig``, ``orig-reimpl``, or ``reimpl-reimpl``.
    seed_a : int | None
        Seed on the first side.
    seed_b : int | None
        Seed on the second side.
    procrustes_rmsd : float
        RMSD after alignment with proper rotation only.
    scale_ratio : float
        Frobenius scale ratio between centered layouts.
    reflected : bool
        Whether a reflected alignment would improve the fit materially.
    max_node_displacement : float
        Maximum per-node displacement after proper-rotation alignment.
    """

    comparison_type: str
    seed_a: Optional[int]
    seed_b: Optional[int]
    procrustes_rmsd: float
    scale_ratio: float
    reflected: bool
    max_node_displacement: float


@dataclass
class GroupResult:
    """Collected outputs for one variant/graph group.

    Parameters
    ----------
    row : dict[str, Any]
        Pending per-graph CSV row.
    seed_rows : list[dict[str, Any]]
        Per-seed CSV rows for this group.
    pairwise_rows : list[dict[str, Any]]
        Pairwise CSV rows for this group.
    rejection_count : int
        Number of layouts rejected by tensor integrity checks.
    """

    row: dict[str, Any]
    seed_rows: list[dict[str, Any]]
    pairwise_rows: list[dict[str, Any]]
    rejection_count: int


@dataclass
class PValueBucket:
    """Deferred p-value correction state for one family of tests.

    Parameters
    ----------
    entries : list[tuple[int, str, float]]
        Tuples of row index, output column, and raw p-value.
    """

    entries: list[tuple[int, str, float]] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    """Parse CLI options.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("eval_output/variant_bench_full"),
        help="Benchmark artifact directory containing results.json and positions/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_output/fidelity_report/data"),
        help="Destination directory for CSV outputs and README",
    )
    parser.add_argument(
        "--max-graphs",
        type=int,
        default=None,
        help="Optional cap on the number of graph names to analyze",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10_000,
        help="Bootstrap samples per (algorithm, graph, metric) comparison",
    )
    parser.add_argument(
        "--power-simulations",
        type=int,
        default=POWER_SIMULATIONS,
        help="Simulation count for the Mann-Whitney minimum detectable effect estimate",
    )
    return parser.parse_args()


def compute_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file.

    Parameters
    ----------
    path : Path
        File to hash.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optional_int(value: object) -> Optional[int]:
    """Parse an optional integer value.

    Parameters
    ----------
    value : object
        Raw value.

    Returns
    -------
    int | None
        Parsed integer or ``None``.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def optional_float(value: object) -> Optional[float]:
    """Parse an optional float value.

    Parameters
    ----------
    value : object
        Raw value.

    Returns
    -------
    float | None
        Parsed float or ``None``.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def optional_str(value: object) -> Optional[str]:
    """Parse an optional string value.

    Parameters
    ----------
    value : object
        Raw value.

    Returns
    -------
    str | None
        Parsed string or ``None``.
    """
    if value is None:
        return None
    parsed = str(value)
    return parsed if parsed else None


def result_record_from_dict(payload: Mapping[str, object]) -> ResultRecord:
    """Build a result record from one JSON object.

    Parameters
    ----------
    payload : Mapping[str, object]
        Raw JSON payload.

    Returns
    -------
    ResultRecord
        Parsed record.
    """
    return ResultRecord(
        graph_name=str(payload.get("graph_name", "")),
        engine_name=str(payload.get("engine_name", "")),
        seed=optional_int(payload.get("seed")),
        status=str(payload.get("status", "")),
        runtime_seconds=optional_float(payload.get("runtime_seconds")),
        positions_file=optional_str(payload.get("positions_file")),
    )


def load_results(path: Path) -> dict[str, ResultRecord]:
    """Load benchmark records from ``results.json``.

    Parameters
    ----------
    path : Path
        Path to the benchmark results JSON file.

    Returns
    -------
    dict[str, ResultRecord]
        Parsed benchmark records keyed by record key.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")

    def _with_key(key: str, value: Mapping[str, object]) -> ResultRecord:
        rec = result_record_from_dict(value)
        rec.result_key = key
        return rec

    return {
        str(key): _with_key(str(key), value)
        for key, value in payload.items()
        if isinstance(value, Mapping)
    }


def selected_graph_names(
    records: Mapping[str, ResultRecord],
    max_graphs: Optional[int],
) -> Optional[set[str]]:
    """Resolve the optional graph-name filter.

    Parameters
    ----------
    records : Mapping[str, ResultRecord]
        Parsed benchmark records.
    max_graphs : int | None
        Maximum number of graph names to keep.

    Returns
    -------
    set[str] | None
        Selected graph names, or ``None`` when all graphs should be analyzed.
    """
    if max_graphs is None:
        return None
    graph_names = sorted({record.graph_name for record in records.values()})
    return set(graph_names[: max(0, max_graphs)])


def stable_seed(*parts: str) -> int:
    """Build a reproducible integer seed from stable string parts.

    Parameters
    ----------
    *parts : str
        Components to hash.

    Returns
    -------
    int
        Unsigned 32-bit integer derived from SHA-256.
    """
    joined = "::".join(parts)
    return int(hashlib.sha256(joined.encode("utf-8")).hexdigest()[:8], 16)


def sample_without_replacement(
    layouts: Sequence[LayoutRecord],
    sample_size: int,
    seed: int,
) -> list[LayoutRecord]:
    """Select a stable random subset of layout samples.

    Parameters
    ----------
    layouts : Sequence[LayoutRecord]
        Available layouts.
    sample_size : int
        Requested sample count.
    seed : int
        Deterministic random seed.

    Returns
    -------
    list[LayoutRecord]
        Selected layouts sorted by seed.
    """
    ordered = sorted(layouts, key=lambda layout: (layout.seed is None, layout.seed))
    if len(ordered) <= sample_size:
        return ordered
    rng = random.Random(seed)
    picked = rng.sample(ordered, sample_size)
    return sorted(picked, key=lambda layout: (layout.seed is None, layout.seed))


def safe_mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean, or ``nan`` for empty input.

    Parameters
    ----------
    values : Sequence[float]
        Numeric inputs.

    Returns
    -------
    float
        Mean value or ``nan`` when empty.
    """
    if not values:
        return math.nan
    return float(statistics.fmean(values))


def safe_median(values: Sequence[float]) -> float:
    """Return the median, or ``nan`` for empty input.

    Parameters
    ----------
    values : Sequence[float]
        Numeric inputs.

    Returns
    -------
    float
        Median value or ``nan`` when empty.
    """
    if not values:
        return math.nan
    return float(statistics.median(values))


def safe_std(values: Sequence[float]) -> float:
    """Return the sample standard deviation, or ``0`` for short input.

    Parameters
    ----------
    values : Sequence[float]
        Numeric inputs.

    Returns
    -------
    float
        Sample standard deviation or ``0.0`` when ``len(values) < 2``.
    """
    if len(values) < 2:
        return 0.0
    return float(statistics.stdev(values))


def finite_values(values: Iterable[float]) -> list[float]:
    """Collect finite numeric values.

    Parameters
    ----------
    values : Iterable[float]
        Candidate values.

    Returns
    -------
    list[float]
        Finite values in input order.
    """
    return [value for value in values if math.isfinite(value)]


def load_graph_registry() -> dict[str, Any]:
    """Load benchmark graph objects keyed by graph name.

    Returns
    -------
    dict[str, Any]
        Graph registry keyed by stable graph names.
    """
    from dagua.eval.graphs import get_test_graphs

    return {graph.name: graph for graph in get_test_graphs()}


def is_disconnected_graph(edge_index: torch.Tensor, num_nodes: int) -> bool:
    """Return whether the underlying undirected graph is disconnected.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor shaped ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    bool
        ``True`` when multiple connected components are present.
    """
    if num_nodes <= 1:
        return False
    if edge_index.numel() == 0:
        return num_nodes > 1

    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    for source, target in edge_index.t().tolist():
        src_idx = int(source)
        tgt_idx = int(target)
        adjacency[src_idx].append(tgt_idx)
        adjacency[tgt_idx].append(src_idx)

    seen = {0}
    queue: deque[int] = deque([0])
    while queue:
        node = queue.popleft()
        for child in adjacency[node]:
            if child in seen:
                continue
            seen.add(child)
            queue.append(child)
    return len(seen) != num_nodes


def structure_bucket(
    edge_index: torch.Tensor,
    num_nodes: int,
    tags: Sequence[str],
    disconnected: bool,
) -> str:
    """Classify a graph into a coarse structure bucket.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor shaped ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    tags : Sequence[str]
        Graph tags from the registry.
    disconnected : bool
        Precomputed disconnected flag.

    Returns
    -------
    str
        Coarse structure label.
    """
    tag_set = set(tags)
    if disconnected:
        return "disconnected"
    if "tree" in tag_set or (num_nodes > 0 and int(edge_index.shape[1]) == num_nodes - 1):
        return "tree"
    if "cycle" in tag_set or "self_loop" in tag_set:
        return "cyclic"
    if "random" in tag_set or "sbm" in tag_set or "erdos_renyi" in tag_set:
        return "random"
    if "dense" in tag_set:
        return "dense"
    return "cyclic" if int(edge_index.shape[1]) >= num_nodes else "random"


def describe_graph(test_graph: Any) -> GraphDescriptor:
    """Build report-friendly metadata for one benchmark graph.

    Parameters
    ----------
    test_graph : Any
        Loaded benchmark graph object.

    Returns
    -------
    GraphDescriptor
        Derived graph metadata and interpretive notes.
    """
    graph = test_graph.graph
    num_nodes = int(graph.num_nodes)
    num_edges = int(graph.edge_index.shape[1]) if graph.edge_index.numel() > 0 else 0
    density = 0.0 if num_nodes <= 1 else num_edges / max(num_nodes * (num_nodes - 1), 1)
    edges_per_node = num_edges / max(num_nodes, 1)
    disconnected = is_disconnected_graph(graph.edge_index, num_nodes)
    tags = tuple(sorted(test_graph.tags))

    if num_nodes < 50:
        size_label = "small"
    elif num_nodes < 200:
        size_label = "medium"
    else:
        size_label = "large"

    if density >= 0.1 or edges_per_node >= 3.0:
        density_label = "dense"
    elif density <= 0.02 and edges_per_node <= 1.2:
        density_label = "sparse"
    else:
        density_label = "medium"

    note_parts: list[str] = []
    if disconnected:
        note_parts.append("disconnected components may legitimately drift between implementations")
    if num_nodes < MIN_PROCRUSTES_NODE_COUNT:
        note_parts.append("excluded from Procrustes because N < 5")
    note = "; ".join(note_parts) if note_parts else "none"
    return GraphDescriptor(
        num_nodes=num_nodes,
        num_edges=num_edges,
        density_bucket=density_label,
        size_bucket=size_label,
        structure_bucket=structure_bucket(graph.edge_index, num_nodes, tags, disconnected),
        is_disconnected=disconnected,
        structural_note=note,
        tags=tags,
    )


def validate_positions(
    positions: torch.Tensor,
    expected_nodes: int,
) -> Optional[str]:
    """Validate a loaded layout tensor against integrity requirements.

    Parameters
    ----------
    positions : torch.Tensor
        Candidate position tensor.
    expected_nodes : int
        Expected node count for the graph.

    Returns
    -------
    str | None
        Rejection reason, or ``None`` when valid.
    """
    if positions.ndim != 2:
        return "tensor_not_2d"
    if positions.shape[1] != 2:
        return "tensor_not_xy"
    if positions.shape[0] < MIN_VALID_NODE_COUNT:
        return "too_few_nodes"
    if positions.shape[0] != expected_nodes:
        return "node_count_mismatch"
    if torch.isnan(positions).any().item():
        return "contains_nan"
    if torch.isinf(positions).any().item():
        return "contains_inf"
    return None


def load_layout(
    record: ResultRecord,
    variant_id: str,
    side: str,
    input_dir: Path,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
) -> tuple[Optional[LayoutRecord], Optional[str]]:
    """Load and validate one benchmark layout.

    Parameters
    ----------
    record : ResultRecord
        Benchmark metadata record.
    variant_id : str
        Reimplementation variant identifier.
    side : str
        Either ``"orig"`` or ``"reimpl"``.
    input_dir : Path
        Benchmark artifact root.
    edge_index : torch.Tensor
        Graph edge tensor shaped ``[2, E]``.
    node_sizes : torch.Tensor
        Graph node sizes shaped ``[N, 2]``.

    Returns
    -------
    tuple[LayoutRecord | None, str | None]
        Loaded layout and optional rejection reason.
    """
    if record.positions_file is None:
        return None, "missing_positions_file"
    # Try HDF5 first (fast), fall back to individual .pt files
    h5_file = getattr(load_layout, "_h5_file", None)
    record_key = record.result_key
    if h5_file is not None and record_key and record_key in h5_file:
        try:
            positions = torch.from_numpy(h5_file[record_key][:]).to(dtype=torch.float32)
        except Exception:
            return None, "h5_load_failure"
    else:
        path = input_dir / record.positions_file
        try:
            tensor = torch.load(path, map_location="cpu")
        except Exception:
            return None, "load_failure"
        if not isinstance(tensor, torch.Tensor):
            return None, "not_tensor"
        positions = tensor.detach().to(dtype=torch.float32, device="cpu")
    rejection = validate_positions(positions, int(node_sizes.shape[0]))
    if rejection is not None:
        return None, rejection
    metrics = {
        metric_name: float(metric_value)
        for metric_name, metric_value in quick(
            positions,
            edge_index,
            node_sizes=node_sizes,
        ).items()
        if metric_name in QUALITY_METRICS
    }
    return (
        LayoutRecord(
            graph_name=record.graph_name,
            variant_id=variant_id,
            side=side,
            seed=record.seed,
            runtime_seconds=record.runtime_seconds,
            positions=positions,
            metrics=metrics,
        ),
        None,
    )


def fidelity_procrustes(
    pos_a: torch.Tensor,
    pos_b: torch.Tensor,
) -> tuple[float, float, bool, torch.Tensor]:
    """Align two layouts WITH scale normalization, without reflection.

    Both layouts are centered and normalized to unit Frobenius norm before
    alignment. This makes the RMSD measure pure shape difference, invariant
    to the arbitrary coordinate scale each implementation uses.

    Scale ratio is reported separately as a diagnostic.

    Parameters
    ----------
    pos_a : torch.Tensor
        First position tensor shaped ``[N, 2]``.
    pos_b : torch.Tensor
        Second position tensor shaped ``[N, 2]``.

    Returns
    -------
    tuple[float, float, bool, torch.Tensor]
        RMSD after scale-normalized proper-rotation alignment, scale ratio
        (informational), reflection-help flag, and per-node displacements.
    """
    a_centered = pos_a - pos_a.mean(dim=0, keepdim=True)
    b_centered = pos_b - pos_b.mean(dim=0, keepdim=True)
    norm_a = float(a_centered.norm().item())
    norm_b = float(b_centered.norm().item())
    scale_ratio = (norm_b / norm_a) if norm_a > 0.0 else math.nan

    # Normalize to unit Frobenius norm -- makes RMSD scale-invariant
    if norm_a > 0.0:
        a_centered = a_centered / norm_a
    if norm_b > 0.0:
        b_centered = b_centered / norm_b

    covariance = a_centered.t() @ b_centered
    left_singular, _, right_singular_t = torch.linalg.svd(covariance)
    det_value = torch.det(left_singular @ right_singular_t)
    correction = torch.diag(
        torch.tensor([1.0, float(torch.sign(det_value).item())], dtype=a_centered.dtype)
    )
    rotation = left_singular @ correction @ right_singular_t
    aligned = a_centered @ rotation
    per_node = torch.norm(aligned - b_centered, dim=1)
    rmsd = float(torch.sqrt(torch.mean(per_node.square())).item())

    reflected_rotation = left_singular @ right_singular_t
    reflected_aligned = a_centered @ reflected_rotation
    reflected_rmsd = float(
        torch.sqrt(torch.mean(torch.norm(reflected_aligned - b_centered, dim=1).square())).item()
    )
    reflected = reflected_rmsd < (0.9 * rmsd)
    return rmsd, scale_ratio, reflected, per_node


def pairwise_statistics(values: Sequence[float]) -> dict[str, float]:
    """Return compact summary statistics for one numeric series.

    Parameters
    ----------
    values : Sequence[float]
        Numeric inputs.

    Returns
    -------
    dict[str, float]
        Mean, median, standard deviation, and maximum.
    """
    return {
        "mean": safe_mean(values),
        "median": safe_median(values),
        "std": safe_std(values),
        "max": max(values) if values else math.nan,
    }


def collect_metric_values(layouts: Sequence[LayoutRecord], metric_name: str) -> np.ndarray:
    """Collect one metric across layout samples.

    Parameters
    ----------
    layouts : Sequence[LayoutRecord]
        Validated layout samples.
    metric_name : str
        Metric identifier.

    Returns
    -------
    numpy.ndarray
        Metric values as a ``float64`` array.
    """
    return np.asarray([layout.metrics[metric_name] for layout in layouts], dtype=np.float64)


def cohens_d(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    """Compute Cohen's d for two independent samples.

    Parameters
    ----------
    sample_a : numpy.ndarray
        First sample.
    sample_b : numpy.ndarray
        Second sample.

    Returns
    -------
    float
        Standardized mean difference.
    """
    n_a = int(sample_a.size)
    n_b = int(sample_b.size)
    if n_a == 0 or n_b == 0:
        return math.nan
    mean_diff = float(sample_b.mean() - sample_a.mean())
    if n_a < 2 or n_b < 2:
        return 0.0 if abs(mean_diff) < 1e-12 else math.copysign(math.inf, mean_diff)
    var_a = float(sample_a.var(ddof=1))
    var_b = float(sample_b.var(ddof=1))
    pooled_num = (n_a - 1) * var_a + (n_b - 1) * var_b
    pooled_den = n_a + n_b - 2
    if pooled_den <= 0:
        return math.nan
    pooled_sd = math.sqrt(max(pooled_num / pooled_den, 0.0))
    if pooled_sd <= 1e-12:
        return 0.0 if abs(mean_diff) < 1e-12 else math.copysign(math.inf, mean_diff)
    return mean_diff / pooled_sd


def cliffs_delta(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    """Compute Cliff's delta for two independent samples.

    Parameters
    ----------
    sample_a : numpy.ndarray
        First sample.
    sample_b : numpy.ndarray
        Second sample.

    Returns
    -------
    float
        Signed effect size in ``[-1, 1]``.
    """
    if sample_a.size == 0 or sample_b.size == 0:
        return math.nan
    wins = 0
    losses = 0
    for value_a in sample_a.tolist():
        for value_b in sample_b.tolist():
            if value_b > value_a:
                wins += 1
            elif value_b < value_a:
                losses += 1
    total = sample_a.size * sample_b.size
    return (wins - losses) / total


def rank_biserial_from_delta(delta: float) -> float:
    """Return the rank-biserial correlation corresponding to Cliff's delta.

    Parameters
    ----------
    delta : float
        Cliff's delta.

    Returns
    -------
    float
        Rank-biserial correlation.
    """
    return delta


def bootstrap_mean_difference_ci(
    sample_a: np.ndarray,
    sample_b: np.ndarray,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    """Estimate a percentile bootstrap CI for the mean difference.

    Parameters
    ----------
    sample_a : numpy.ndarray
        Original-side sample.
    sample_b : numpy.ndarray
        Reimplementation-side sample.
    samples : int
        Bootstrap draw count.
    seed : int
        Deterministic bootstrap seed.

    Returns
    -------
    tuple[float, float]
        Lower and upper percentile bounds for ``mean(sample_b) - mean(sample_a)``.
    """
    if sample_a.size == 0 or sample_b.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    bootstrap_values = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        draw_a = rng.choice(sample_a, size=sample_a.size, replace=True)
        draw_b = rng.choice(sample_b, size=sample_b.size, replace=True)
        bootstrap_values[index] = float(draw_b.mean() - draw_a.mean())
    lower = float(np.percentile(bootstrap_values, 2.5))
    upper = float(np.percentile(bootstrap_values, 97.5))
    return lower, upper


def margin_for_metric(
    metric_name: str,
    original_values: np.ndarray,
    factor: float,
) -> float:
    """Return the TOST equivalence margin for one metric.

    Parameters
    ----------
    metric_name : str
        Metric identifier.
    original_values : numpy.ndarray
        Original-side metric values.
    factor : float
        Within-original standard deviation multiplier.

    Returns
    -------
    float
        Metric-specific equivalence margin.
    """
    std_orig = float(original_values.std(ddof=1)) if original_values.size >= 2 else 0.0
    if metric_name == "edge_length_mean":
        mean_orig = float(original_values.mean()) if original_values.size > 0 else 0.0
        floor = max(1.0, 0.01 * abs(mean_orig))
    else:
        floor = METRIC_MARGIN_FLOORS[metric_name]
    return max(factor * std_orig, floor)


def tost_pvalue(
    original_values: np.ndarray,
    reimpl_values: np.ndarray,
    margin: float,
) -> float:
    """Run a Welch-style TOST equivalence test on mean differences.

    Parameters
    ----------
    original_values : numpy.ndarray
        Original-side sample.
    reimpl_values : numpy.ndarray
        Reimplementation-side sample.
    margin : float
        Symmetric equivalence bound.

    Returns
    -------
    float
        Raw TOST p-value.
    """
    pvalue, _, _ = ttost_ind(
        reimpl_values,
        original_values,
        low=-margin,
        upp=margin,
        usevar="unequal",
    )
    return float(pvalue)


def metric_test_columns(metric_name: str) -> list[str]:
    """Return dynamic per-metric output columns.

    Parameters
    ----------
    metric_name : str
        Metric identifier.

    Returns
    -------
    list[str]
        CSV columns emitted for the metric.
    """
    columns = [
        f"{metric_name}_orig_mean",
        f"{metric_name}_orig_std",
        f"{metric_name}_reimpl_mean",
        f"{metric_name}_reimpl_std",
        f"{metric_name}_cohens_d",
        f"{metric_name}_cliffs_delta",
        f"{metric_name}_rank_biserial",
        f"{metric_name}_ks_pvalue_raw",
        f"{metric_name}_ks_pvalue_bh",
        f"{metric_name}_mannwhitney_pvalue_raw",
        f"{metric_name}_mannwhitney_pvalue_bh",
        f"{metric_name}_bootstrap_diff_ci_low",
        f"{metric_name}_bootstrap_diff_ci_high",
    ]
    for factor in TOST_MARGIN_FACTORS:
        label = TOST_MARGIN_LABELS[factor]
        columns.extend(
            (
                f"{metric_name}_tost_margin_{label}",
                f"{metric_name}_tost_pvalue_{label}_raw",
                f"{metric_name}_tost_pvalue_{label}_bh",
            )
        )
    return columns


def per_graph_fieldnames() -> list[str]:
    """Return the per-graph CSV header.

    Returns
    -------
    list[str]
        Output column names.
    """
    columns = [
        "algorithm_family",
        "variant_id",
        "graph_name",
        "num_nodes",
        "num_edges",
        "density_bucket",
        "size_bucket",
        "structure_bucket",
        "structural_note",
        "num_reimpl_seeds",
        "num_orig_seeds",
        "procrustes_rmsd_mean",
        "procrustes_rmsd_std",
        "procrustes_rmsd_max",
        "scale_ratio_mean",
        "scale_ratio_std",
        "reflected",
        "max_node_displacement",
        "ks_pvalue_raw",
        "mannwhitney_pvalue_raw",
        "tost_pvalue_at_1x",
        "ks_pvalue_bh",
        "mannwhitney_pvalue_bh",
        "tost_pvalue_at_1x_bh",
        "cliffs_delta",
        "runtime_orig_mean",
        "runtime_reimpl_mean",
        "runtime_ratio",
        "verdict",
        "anomaly_reason",
    ]
    for metric_name in QUALITY_METRICS:
        columns.extend(metric_test_columns(metric_name))
    return columns


def per_seed_fieldnames() -> list[str]:
    """Return the per-seed CSV header.

    Returns
    -------
    list[str]
        Output column names.
    """
    columns = [
        "algorithm_family",
        "variant_id",
        "graph_name",
        "seed",
        "side",
        "runtime_seconds",
        "nearest_procrustes",
    ]
    columns.extend(QUALITY_METRICS)
    return columns


def pairwise_fieldnames() -> list[str]:
    """Return the pairwise-similarity CSV header.

    Returns
    -------
    list[str]
        Output column names.
    """
    return [
        "algorithm_family",
        "graph_name",
        "seed_a",
        "seed_b",
        "comparison_type",
        "procrustes_rmsd",
        "scale_ratio",
    ]


def algorithm_summary_fieldnames() -> list[str]:
    """Return the algorithm-summary CSV header.

    Returns
    -------
    list[str]
        Output column names.
    """
    return [
        "algorithm_family",
        "is_stochastic",
        "num_graphs_tested",
        "num_graphs_paired_ok",
        "num_graphs_insufficient_data",
        "num_nan_rejected",
        "procrustes_rmsd_mean",
        "procrustes_rmsd_median",
        "procrustes_rmsd_max",
        "scale_ratio_mean",
        "scale_ratio_std",
        "num_mirror_matches",
        "mean_runtime_ratio",
        "std_runtime_ratio",
        "verdict",
        "anomaly_count",
        "anomaly_graphs",
        "tost_pass_rate_at_1x",
        "tost_pass_rate_at_1_5x",
    ]


def compute_pairwise_comparisons(
    first: Sequence[LayoutRecord],
    second: Sequence[LayoutRecord],
    comparison_type: str,
) -> list[PairwiseComparison]:
    """Compute Procrustes comparisons between two layout sets.

    Parameters
    ----------
    first : Sequence[LayoutRecord]
        First layout collection.
    second : Sequence[LayoutRecord]
        Second layout collection.
    comparison_type : str
        Output label for the comparison family.

    Returns
    -------
    list[PairwiseComparison]
        Pairwise Procrustes measurements.
    """
    comparisons: list[PairwiseComparison] = []
    for first_index, layout_a in enumerate(first):
        start_index = first_index + 1 if first is second else 0
        for second_index in range(start_index, len(second)):
            layout_b = second[second_index]
            rmsd, scale_ratio, reflected, per_node = fidelity_procrustes(
                layout_a.positions,
                layout_b.positions,
            )
            comparisons.append(
                PairwiseComparison(
                    comparison_type=comparison_type,
                    seed_a=layout_a.seed,
                    seed_b=layout_b.seed,
                    procrustes_rmsd=rmsd,
                    scale_ratio=scale_ratio,
                    reflected=reflected,
                    max_node_displacement=float(per_node.max().item()),
                )
            )
    return comparisons


def nearest_cross_procrustes(
    source_layouts: Sequence[LayoutRecord],
    target_layouts: Sequence[LayoutRecord],
) -> dict[tuple[str, Optional[int]], float]:
    """Return nearest cross-side Procrustes RMSD for each source layout.

    Parameters
    ----------
    source_layouts : Sequence[LayoutRecord]
        Layouts whose nearest opposite-side match is required.
    target_layouts : Sequence[LayoutRecord]
        Opposite-side layout set.

    Returns
    -------
    dict[tuple[str, int | None], float]
        Mapping from ``(side, seed)`` to nearest cross-side RMSD.
    """
    nearest: dict[tuple[str, Optional[int]], float] = {}
    for layout_a in source_layouts:
        best = math.nan
        for layout_b in target_layouts:
            rmsd, _, _, _ = fidelity_procrustes(layout_a.positions, layout_b.positions)
            if not math.isfinite(best) or rmsd < best:
                best = rmsd
        nearest[(layout_a.side, layout_a.seed)] = best
    return nearest


def graph_metrics_summary(
    layouts: Sequence[LayoutRecord],
    metric_name: str,
) -> tuple[float, float]:
    """Return mean and standard deviation for one metric across layouts.

    Parameters
    ----------
    layouts : Sequence[LayoutRecord]
        Layout samples.
    metric_name : str
        Metric identifier.

    Returns
    -------
    tuple[float, float]
        Mean and sample standard deviation.
    """
    values = [layout.metrics[metric_name] for layout in layouts]
    return safe_mean(values), safe_std(values)


def estimate_mw_min_detectable_effect(simulations: int) -> float:
    """Estimate the Mann-Whitney minimum detectable effect size at 80% power.

    Parameters
    ----------
    simulations : int
        Monte Carlo simulation count per effect-size candidate.

    Returns
    -------
    float
        Smallest Cohen's d whose estimated power reaches the target.
    """
    rng = np.random.default_rng(7)
    for effect_size in POWER_EFFECT_GRID:
        hits = 0
        for _ in range(simulations):
            sample_a = rng.normal(loc=0.0, scale=1.0, size=MIN_STOCHASTIC_SEEDS)
            sample_b = rng.normal(loc=effect_size, scale=1.0, size=MIN_STOCHASTIC_SEEDS)
            pvalue = float(mannwhitneyu(sample_a, sample_b, alternative="two-sided").pvalue)
            if pvalue < POWER_ALPHA:
                hits += 1
        power = hits / simulations
        if power >= POWER_TARGET:
            return effect_size
    return POWER_EFFECT_GRID[-1]


def runtime_ratio(
    original_layouts: Sequence[LayoutRecord],
    reimpl_layouts: Sequence[LayoutRecord],
) -> float:
    """Compute the mean runtime ratio between reimplementation and original.

    Parameters
    ----------
    original_layouts : Sequence[LayoutRecord]
        Original-side layouts.
    reimpl_layouts : Sequence[LayoutRecord]
        Reimplementation-side layouts.

    Returns
    -------
    float
        ``mean(reimpl_runtime) / mean(orig_runtime)``.
    """
    orig_values = finite_values(
        layout.runtime_seconds if layout.runtime_seconds is not None else math.nan
        for layout in original_layouts
    )
    reimpl_values = finite_values(
        layout.runtime_seconds if layout.runtime_seconds is not None else math.nan
        for layout in reimpl_layouts
    )
    if not orig_values or not reimpl_values:
        return math.nan
    mean_orig = safe_mean(orig_values)
    mean_reimpl = safe_mean(reimpl_values)
    if not math.isfinite(mean_orig) or abs(mean_orig) <= 1e-12:
        return math.nan
    return mean_reimpl / mean_orig


def build_variant_groups(
    records: Mapping[str, ResultRecord],
    allowed_graphs: Optional[set[str]],
) -> dict[tuple[str, str], dict[str, list[ResultRecord]]]:
    """Group benchmark records by variant and graph.

    Parameters
    ----------
    records : Mapping[str, ResultRecord]
        Parsed benchmark results.
    allowed_graphs : set[str] | None
        Optional graph filter.

    Returns
    -------
    dict[tuple[str, str], dict[str, list[ResultRecord]]]
        Grouped metadata, with ``orig`` and ``reimpl`` record lists.
    """
    original_name_to_variant: dict[str, AlgorithmVariant] = {}
    variant_by_id = {candidate.variant_id: candidate for candidate in VARIANT_REGISTRY}
    for candidate in VARIANT_REGISTRY:
        original_name = original_variant_name(candidate)
        if original_name is not None:
            original_name_to_variant[original_name] = candidate

    groups: dict[tuple[str, str], dict[str, list[ResultRecord]]] = defaultdict(
        lambda: {"orig": [], "reimpl": []}
    )
    for record in records.values():
        if allowed_graphs is not None and record.graph_name not in allowed_graphs:
            continue
        if record.status != "ok":
            continue
        variant: Optional[AlgorithmVariant]
        side: Optional[str]
        if record.engine_name in variant_by_id:
            variant = variant_by_id[record.engine_name]
            if original_variant_name(variant) is None:
                continue
            side = "reimpl"
        elif record.engine_name in original_name_to_variant:
            variant = original_name_to_variant[record.engine_name]
            side = "orig"
        else:
            continue
        groups[(variant.variant_id, record.graph_name)][side].append(record)
    return groups


def add_metric_tests_to_row(
    row: dict[str, Any],
    row_index: int,
    pvalue_buckets: dict[str, PValueBucket],
    original_layouts: Sequence[LayoutRecord],
    reimpl_layouts: Sequence[LayoutRecord],
    variant_id: str,
    graph_name: str,
    bootstrap_samples: int,
) -> None:
    """Populate per-metric statistics and deferred p-value corrections.

    Parameters
    ----------
    row : dict[str, Any]
        Pending per-graph output row.
    row_index : int
        Index of the per-graph row in the global list.
    pvalue_buckets : dict[str, PValueBucket]
        Deferred correction registry.
    original_layouts : Sequence[LayoutRecord]
        Original-side layouts.
    reimpl_layouts : Sequence[LayoutRecord]
        Reimplementation-side layouts.
    variant_id : str
        Variant identifier.
    graph_name : str
        Graph identifier.
    bootstrap_samples : int
        Bootstrap sample count.
    """
    for metric_name in QUALITY_METRICS:
        original_values = collect_metric_values(original_layouts, metric_name)
        reimpl_values = collect_metric_values(reimpl_layouts, metric_name)
        delta = cliffs_delta(original_values, reimpl_values)
        ci_low, ci_high = bootstrap_mean_difference_ci(
            original_values,
            reimpl_values,
            samples=bootstrap_samples,
            seed=stable_seed(variant_id, graph_name, metric_name),
        )
        row[f"{metric_name}_orig_mean"] = float(original_values.mean())
        row[f"{metric_name}_orig_std"] = (
            float(original_values.std(ddof=1)) if original_values.size > 1 else 0.0
        )
        row[f"{metric_name}_reimpl_mean"] = float(reimpl_values.mean())
        row[f"{metric_name}_reimpl_std"] = (
            float(reimpl_values.std(ddof=1)) if reimpl_values.size > 1 else 0.0
        )
        row[f"{metric_name}_cohens_d"] = cohens_d(original_values, reimpl_values)
        row[f"{metric_name}_cliffs_delta"] = delta
        row[f"{metric_name}_rank_biserial"] = rank_biserial_from_delta(delta)
        row[f"{metric_name}_bootstrap_diff_ci_low"] = ci_low
        row[f"{metric_name}_bootstrap_diff_ci_high"] = ci_high

        ks_pvalue = float(ks_2samp(original_values, reimpl_values).pvalue)
        mw_pvalue = float(
            mannwhitneyu(original_values, reimpl_values, alternative="two-sided").pvalue
        )
        row[f"{metric_name}_ks_pvalue_raw"] = ks_pvalue
        row[f"{metric_name}_mannwhitney_pvalue_raw"] = mw_pvalue
        pvalue_buckets["ks"].entries.append((row_index, f"{metric_name}_ks_pvalue_bh", ks_pvalue))
        pvalue_buckets["mannwhitney"].entries.append(
            (row_index, f"{metric_name}_mannwhitney_pvalue_bh", mw_pvalue)
        )

        for factor in TOST_MARGIN_FACTORS:
            label = TOST_MARGIN_LABELS[factor]
            margin = margin_for_metric(metric_name, original_values, factor)
            pvalue = tost_pvalue(original_values, reimpl_values, margin)
            row[f"{metric_name}_tost_margin_{label}"] = margin
            row[f"{metric_name}_tost_pvalue_{label}_raw"] = pvalue
            bucket_name = f"tost_{label}"
            pvalue_buckets[bucket_name].entries.append(
                (row_index, f"{metric_name}_tost_pvalue_{label}_bh", pvalue)
            )


def initialize_metric_columns(row: dict[str, Any]) -> None:
    """Fill dynamic metric columns with ``nan`` defaults.

    Parameters
    ----------
    row : dict[str, Any]
        Pending per-graph row to mutate.
    """
    for metric_name in QUALITY_METRICS:
        for column in metric_test_columns(metric_name):
            row[column] = math.nan


def ensure_node_sizes(test_graph: Any) -> torch.Tensor:
    """Return graph node sizes, computing them eagerly when needed.

    Parameters
    ----------
    test_graph : Any
        Benchmark graph wrapper with a ``graph`` attribute.

    Returns
    -------
    torch.Tensor
        Node-size tensor shaped ``[N, 2]`` on CPU.
    """
    node_sizes = getattr(test_graph.graph, "node_sizes", None)
    if node_sizes is None:
        test_graph.graph.compute_node_sizes()
        node_sizes = getattr(test_graph.graph, "node_sizes", None)
    if node_sizes is None:
        raise ValueError(
            f"Graph {test_graph.name} is missing node sizes after compute_node_sizes()"
        )
    return node_sizes.detach().cpu()


def process_group(
    variant: AlgorithmVariant,
    graph_name: str,
    records: dict[str, list[ResultRecord]],
    input_dir: Path,
    test_graph: Any,
    row_index: int,
    pvalue_buckets: dict[str, PValueBucket],
    bootstrap_samples: int,
    load_counter: Counter[str],
) -> GroupResult:
    """Process one variant/graph pair into CSV-ready outputs.

    Parameters
    ----------
    variant : AlgorithmVariant
        Variant definition.
    graph_name : str
        Graph name for the group.
    records : dict[str, list[ResultRecord]]
        Side-partitioned benchmark records.
    input_dir : Path
        Benchmark artifact root.
    test_graph : Any
        Graph object used for metrics and metadata.
    row_index : int
        Per-graph row index in the global output list.
    pvalue_buckets : dict[str, PValueBucket]
        Deferred correction registry.
    bootstrap_samples : int
        Bootstrap draw count.
    load_counter : Counter[str]
        Global file-load progress counter.

    Returns
    -------
    GroupResult
        Per-graph row plus streamed child rows.
    """
    descriptor = describe_graph(test_graph)
    edge_index = test_graph.graph.edge_index.detach().cpu()
    node_sizes = ensure_node_sizes(test_graph)
    family_name = algorithm_family(variant.variant_id)
    row: dict[str, Any] = {
        "algorithm_family": family_name,
        "variant_id": variant.variant_id,
        "graph_name": graph_name,
        "num_nodes": descriptor.num_nodes,
        "num_edges": descriptor.num_edges,
        "density_bucket": descriptor.density_bucket,
        "size_bucket": descriptor.size_bucket,
        "structure_bucket": descriptor.structure_bucket,
        "structural_note": descriptor.structural_note,
        "num_reimpl_seeds": 0,
        "num_orig_seeds": 0,
        "procrustes_rmsd_mean": math.nan,
        "procrustes_rmsd_std": math.nan,
        "procrustes_rmsd_max": math.nan,
        "scale_ratio_mean": math.nan,
        "scale_ratio_std": math.nan,
        "reflected": False,
        "max_node_displacement": math.nan,
        "ks_pvalue_raw": math.nan,
        "mannwhitney_pvalue_raw": math.nan,
        "tost_pvalue_at_1x": math.nan,
        "ks_pvalue_bh": math.nan,
        "mannwhitney_pvalue_bh": math.nan,
        "tost_pvalue_at_1x_bh": math.nan,
        "cliffs_delta": math.nan,
        "runtime_orig_mean": math.nan,
        "runtime_reimpl_mean": math.nan,
        "runtime_ratio": math.nan,
        "verdict": "insufficient_data",
        "anomaly_reason": "",
        "_variant_is_stochastic": variant.is_stochastic,
        "_structural_note_flag": descriptor.structural_note != "none",
    }
    initialize_metric_columns(row)

    loaded_layouts: dict[str, list[LayoutRecord]] = {"orig": [], "reimpl": []}
    rejection_count = 0

    # Parallel loading -- I/O bound so threads give ~4-8x speedup
    load_tasks: list[tuple[str, Any]] = []
    for side in ("orig", "reimpl"):
        for record in sorted(
            records[side], key=lambda candidate: (candidate.seed is None, candidate.seed)
        ):
            load_tasks.append((side, record))

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(
                load_layout,
                record,
                variant_id=variant.variant_id,
                side=side,
                input_dir=input_dir,
                edge_index=edge_index,
                node_sizes=node_sizes,
            ): side
            for side, record in load_tasks
        }
        for future in as_completed(futures):
            side = futures[future]
            layout, rejection = future.result()
            load_counter["files"] += 1
            if load_counter["files"] % PAIRWISE_PROGRESS_INTERVAL == 0:
                print(
                    f"[fidelity] loaded {load_counter['files']} position tensors", file=sys.stderr
                )
            if layout is None:
                rejection_count += 1
                continue
            loaded_layouts[side].append(layout)

    original_layouts = loaded_layouts["orig"]
    reimpl_layouts = loaded_layouts["reimpl"]
    row["num_orig_seeds"] = len(original_layouts)
    row["num_reimpl_seeds"] = len(reimpl_layouts)
    row["runtime_orig_mean"] = safe_mean(
        finite_values(
            layout.runtime_seconds if layout.runtime_seconds is not None else math.nan
            for layout in original_layouts
        )
    )
    row["runtime_reimpl_mean"] = safe_mean(
        finite_values(
            layout.runtime_seconds if layout.runtime_seconds is not None else math.nan
            for layout in reimpl_layouts
        )
    )
    row["runtime_ratio"] = runtime_ratio(original_layouts, reimpl_layouts)

    seed_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    if not original_layouts or not reimpl_layouts:
        return GroupResult(
            row=row,
            seed_rows=seed_rows,
            pairwise_rows=pairwise_rows,
            rejection_count=rejection_count,
        )

    pairwise_orig_reimpl: list[PairwiseComparison] = []
    if variant.is_stochastic:
        if (
            len(original_layouts) < MIN_STOCHASTIC_SEEDS
            or len(reimpl_layouts) < MIN_STOCHASTIC_SEEDS
        ):
            for layouts in (original_layouts, reimpl_layouts):
                for layout in layouts:
                    seed_row = {
                        "algorithm_family": family_name,
                        "variant_id": variant.variant_id,
                        "graph_name": graph_name,
                        "seed": layout.seed,
                        "side": layout.side,
                        "runtime_seconds": layout.runtime_seconds,
                        "nearest_procrustes": math.nan,
                    }
                    for metric_name in QUALITY_METRICS:
                        seed_row[metric_name] = layout.metrics[metric_name]
                    seed_rows.append(seed_row)
            return GroupResult(
                row=row,
                seed_rows=seed_rows,
                pairwise_rows=pairwise_rows,
                rejection_count=rejection_count,
            )

        sampled_orig = sample_without_replacement(
            original_layouts,
            PAIRWISE_SAMPLE_SIZE,
            stable_seed(variant.variant_id, graph_name, "orig"),
        )
        sampled_reimpl = sample_without_replacement(
            reimpl_layouts,
            PAIRWISE_SAMPLE_SIZE,
            stable_seed(variant.variant_id, graph_name, "reimpl"),
        )
        pairwise_orig_reimpl = compute_pairwise_comparisons(
            sampled_orig,
            sampled_reimpl,
            "orig-reimpl",
        )
        pairwise_orig = compute_pairwise_comparisons(sampled_orig, sampled_orig, "orig-orig")
        pairwise_reimpl = compute_pairwise_comparisons(
            sampled_reimpl, sampled_reimpl, "reimpl-reimpl"
        )
        for comparison in (*pairwise_orig, *pairwise_orig_reimpl, *pairwise_reimpl):
            pairwise_rows.append(
                {
                    "algorithm_family": family_name,
                    "graph_name": graph_name,
                    "seed_a": comparison.seed_a,
                    "seed_b": comparison.seed_b,
                    "comparison_type": comparison.comparison_type,
                    "procrustes_rmsd": comparison.procrustes_rmsd,
                    "scale_ratio": comparison.scale_ratio,
                }
            )
        if descriptor.num_nodes >= MIN_PROCRUSTES_NODE_COUNT:
            rmsd_values = [comparison.procrustes_rmsd for comparison in pairwise_orig_reimpl]
            scale_values = [comparison.scale_ratio for comparison in pairwise_orig_reimpl]
            max_displacements = [
                comparison.max_node_displacement for comparison in pairwise_orig_reimpl
            ]
            reflected = any(comparison.reflected for comparison in pairwise_orig_reimpl)
            pairwise_summary = pairwise_statistics(rmsd_values)
            row["procrustes_rmsd_mean"] = pairwise_summary["mean"]
            row["procrustes_rmsd_std"] = pairwise_summary["std"]
            row["procrustes_rmsd_max"] = pairwise_summary["max"]
            row["scale_ratio_mean"] = safe_mean(scale_values)
            row["scale_ratio_std"] = safe_std(scale_values)
            row["reflected"] = reflected
            row["max_node_displacement"] = max(max_displacements) if max_displacements else math.nan
        nearest_orig = nearest_cross_procrustes(original_layouts, reimpl_layouts)
        nearest_reimpl = nearest_cross_procrustes(reimpl_layouts, original_layouts)
        for layouts in (original_layouts, reimpl_layouts):
            for layout in layouts:
                nearest_value = (
                    nearest_orig.get((layout.side, layout.seed))
                    if layout.side == "orig"
                    else nearest_reimpl.get((layout.side, layout.seed))
                )
                seed_row = {
                    "algorithm_family": family_name,
                    "variant_id": variant.variant_id,
                    "graph_name": graph_name,
                    "seed": layout.seed,
                    "side": layout.side,
                    "runtime_seconds": layout.runtime_seconds,
                    "nearest_procrustes": nearest_value,
                }
                for metric_name in QUALITY_METRICS:
                    seed_row[metric_name] = layout.metrics[metric_name]
                seed_rows.append(seed_row)
        add_metric_tests_to_row(
            row=row,
            row_index=row_index,
            pvalue_buckets=pvalue_buckets,
            original_layouts=original_layouts,
            reimpl_layouts=reimpl_layouts,
            variant_id=variant.variant_id,
            graph_name=graph_name,
            bootstrap_samples=bootstrap_samples,
        )
        row["ks_pvalue_raw"] = min(
            row[f"{metric_name}_ks_pvalue_raw"] for metric_name in QUALITY_METRICS
        )
        row["mannwhitney_pvalue_raw"] = min(
            row[f"{metric_name}_mannwhitney_pvalue_raw"] for metric_name in QUALITY_METRICS
        )
        row["tost_pvalue_at_1x"] = max(
            row[f"{metric_name}_tost_pvalue_1x_raw"] for metric_name in QUALITY_METRICS
        )
        row["cliffs_delta"] = max(
            abs(float(row[f"{metric_name}_cliffs_delta"])) for metric_name in QUALITY_METRICS
        )
        return GroupResult(
            row=row,
            seed_rows=seed_rows,
            pairwise_rows=pairwise_rows,
            rejection_count=rejection_count,
        )

    original_layout = original_layouts[0]
    reimpl_layout = reimpl_layouts[0]
    if descriptor.num_nodes >= MIN_PROCRUSTES_NODE_COUNT:
        rmsd, scale_ratio, reflected, per_node = fidelity_procrustes(
            original_layout.positions,
            reimpl_layout.positions,
        )
        row["procrustes_rmsd_mean"] = rmsd
        row["procrustes_rmsd_std"] = 0.0
        row["procrustes_rmsd_max"] = rmsd
        row["scale_ratio_mean"] = scale_ratio
        row["scale_ratio_std"] = 0.0
        row["reflected"] = reflected
        row["max_node_displacement"] = float(per_node.max().item())
        pairwise_orig_reimpl = [
            PairwiseComparison(
                comparison_type="orig-reimpl",
                seed_a=original_layout.seed,
                seed_b=reimpl_layout.seed,
                procrustes_rmsd=rmsd,
                scale_ratio=scale_ratio,
                reflected=reflected,
                max_node_displacement=float(per_node.max().item()),
            )
        ]
        pairwise_rows.append(
            {
                "algorithm_family": family_name,
                "graph_name": graph_name,
                "seed_a": original_layout.seed,
                "seed_b": reimpl_layout.seed,
                "comparison_type": "orig-reimpl",
                "procrustes_rmsd": rmsd,
                "scale_ratio": scale_ratio,
            }
        )
    for layout, nearest_value in (
        (original_layout, row["procrustes_rmsd_mean"]),
        (reimpl_layout, row["procrustes_rmsd_mean"]),
    ):
        seed_row = {
            "algorithm_family": family_name,
            "variant_id": variant.variant_id,
            "graph_name": graph_name,
            "seed": layout.seed,
            "side": layout.side,
            "runtime_seconds": layout.runtime_seconds,
            "nearest_procrustes": nearest_value,
        }
        for metric_name in QUALITY_METRICS:
            seed_row[metric_name] = layout.metrics[metric_name]
            if layout.side == "orig":
                row[f"{metric_name}_orig_mean"] = layout.metrics[metric_name]
                row[f"{metric_name}_orig_std"] = 0.0
            else:
                row[f"{metric_name}_reimpl_mean"] = layout.metrics[metric_name]
                row[f"{metric_name}_reimpl_std"] = 0.0
        seed_rows.append(seed_row)
    for metric_name in QUALITY_METRICS:
        original_value = original_layout.metrics[metric_name]
        reimpl_value = reimpl_layout.metrics[metric_name]
        delta_value = reimpl_value - original_value
        row[f"{metric_name}_cohens_d"] = (
            0.0 if abs(delta_value) < 1e-12 else math.copysign(math.inf, delta_value)
        )
        row[f"{metric_name}_cliffs_delta"] = (
            0.0 if abs(delta_value) < 1e-12 else math.copysign(1.0, delta_value)
        )
        row[f"{metric_name}_rank_biserial"] = row[f"{metric_name}_cliffs_delta"]
        row[f"{metric_name}_bootstrap_diff_ci_low"] = delta_value
        row[f"{metric_name}_bootstrap_diff_ci_high"] = delta_value
    return GroupResult(
        row=row, seed_rows=seed_rows, pairwise_rows=pairwise_rows, rejection_count=rejection_count
    )


def apply_bh_correction(
    rows: list[dict[str, Any]],
    pvalue_buckets: Mapping[str, PValueBucket],
) -> None:
    """Apply Benjamini-Hochberg correction to deferred p-value columns.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Pending per-graph rows.
    pvalue_buckets : Mapping[str, PValueBucket]
        Deferred p-value registry.
    """
    for bucket in pvalue_buckets.values():
        if not bucket.entries:
            continue
        pvalues = [entry[2] for entry in bucket.entries]
        _, corrected, _, _ = multipletests(pvalues, alpha=0.05, method="fdr_bh")
        for (row_index, column_name, _), corrected_value in zip(bucket.entries, corrected):
            rows[row_index][column_name] = float(corrected_value)


def explainable_only(row: Mapping[str, Any]) -> bool:
    """Return whether a row has only structural-note anomalies.

    Parameters
    ----------
    row : Mapping[str, Any]
        Finalized per-graph row.

    Returns
    -------
    bool
        ``True`` when anomaly text is empty or purely structural.
    """
    anomaly_reason = str(row.get("anomaly_reason", ""))
    if not anomaly_reason:
        return True
    reasons = {part.strip() for part in anomaly_reason.split(";") if part.strip()}
    return reasons <= {"structural_note"}


def finalize_group_row(row: dict[str, Any]) -> None:
    """Resolve row-level BH summaries, anomalies, and verdicts.

    Parameters
    ----------
    row : dict[str, Any]
        Pending per-graph row.
    """
    anomaly_reasons: list[str] = []
    stochastic = bool(row["_variant_is_stochastic"])
    if row["structural_note"] != "none":
        anomaly_reasons.append("structural_note")

    if stochastic:
        row["ks_pvalue_bh"] = min(
            float(row[f"{metric_name}_ks_pvalue_bh"]) for metric_name in QUALITY_METRICS
        )
        row["mannwhitney_pvalue_bh"] = min(
            float(row[f"{metric_name}_mannwhitney_pvalue_bh"]) for metric_name in QUALITY_METRICS
        )
        row["tost_pvalue_at_1x_bh"] = max(
            float(row[f"{metric_name}_tost_pvalue_1x_bh"]) for metric_name in QUALITY_METRICS
        )
        row["cliffs_delta"] = max(
            abs(float(row[f"{metric_name}_cliffs_delta"])) for metric_name in QUALITY_METRICS
        )
    counts_ok = int(row["num_orig_seeds"]) >= 1 and int(row["num_reimpl_seeds"]) >= 1
    if not counts_ok:
        row["verdict"] = "insufficient_data"
    elif stochastic and (
        int(row["num_orig_seeds"]) < MIN_STOCHASTIC_SEEDS
        or int(row["num_reimpl_seeds"]) < MIN_STOCHASTIC_SEEDS
    ):
        row["verdict"] = "insufficient_data"
    else:
        reflected = bool(row["reflected"])
        runtime_ratio_value = (
            float(row["runtime_ratio"]) if math.isfinite(float(row["runtime_ratio"])) else math.nan
        )
        scale_ratio_value = (
            float(row["scale_ratio_mean"])
            if math.isfinite(float(row["scale_ratio_mean"]))
            else math.nan
        )
        max_displacement = (
            float(row["max_node_displacement"])
            if math.isfinite(float(row["max_node_displacement"]))
            else math.nan
        )

        if reflected:
            anomaly_reasons.append("mirror_match")
        if math.isfinite(scale_ratio_value) and (
            scale_ratio_value < SCALE_RATIO_LOWER or scale_ratio_value > SCALE_RATIO_UPPER
        ):
            anomaly_reasons.append("scale_ratio_out_of_range")
        if math.isfinite(runtime_ratio_value) and (
            runtime_ratio_value < RUNTIME_RATIO_ANOMALY_LOWER
            or runtime_ratio_value > RUNTIME_RATIO_ANOMALY_UPPER
        ):
            anomaly_reasons.append("runtime_ratio_outlier")
        if math.isfinite(runtime_ratio_value) and (
            runtime_ratio_value < RUNTIME_RATIO_WARNING_LOWER
            or runtime_ratio_value > RUNTIME_RATIO_WARNING_UPPER
        ):
            anomaly_reasons.append("runtime_ratio_warning")

        if stochastic:
            mw_anomaly = any(
                float(row[f"{metric_name}_mannwhitney_pvalue_bh"]) < 0.05
                for metric_name in QUALITY_METRICS
            )
            if mw_anomaly:
                anomaly_reasons.append("mannwhitney_difference")
            tost_1x = all(
                float(row[f"{metric_name}_tost_pvalue_1x_bh"]) < 0.05
                for metric_name in QUALITY_METRICS
            )
            tost_1_5x = all(
                float(row[f"{metric_name}_tost_pvalue_1_5x_bh"]) < 0.05
                for metric_name in QUALITY_METRICS
            )
            tost_2x = all(
                float(row[f"{metric_name}_tost_pvalue_2x_bh"]) < 0.05
                for metric_name in QUALITY_METRICS
            )
            if tost_1x and not mw_anomaly and "mirror_match" not in anomaly_reasons:
                row["verdict"] = "strong_equivalent"
            elif tost_1_5x and explainable_only({"anomaly_reason": "; ".join(anomaly_reasons)}):
                row["verdict"] = "weak_equivalent"
            elif not tost_2x:
                row["verdict"] = "divergent"
            else:
                row["verdict"] = "partial_match"
        else:
            if math.isfinite(max_displacement) and max_displacement > PROCRUSTES_ANOMALY_THRESHOLD:
                anomaly_reasons.append("max_node_displacement")
            if (
                math.isfinite(max_displacement)
                and max_displacement < IDENTICAL_DISPLACEMENT_THRESHOLD
            ):
                row["verdict"] = "identical"
            elif not anomaly_reasons or explainable_only(
                {"anomaly_reason": "; ".join(anomaly_reasons)}
            ):
                row["verdict"] = "strong_equivalent"
            elif (
                not math.isfinite(max_displacement)
                or max_displacement <= PROCRUSTES_ANOMALY_THRESHOLD
            ):
                row["verdict"] = "partial_match"
            else:
                row["verdict"] = "divergent"

    reasons = sorted(set(reason for reason in anomaly_reasons if reason))
    row["anomaly_reason"] = "; ".join(reasons)
    row["_tost_pass_1x"] = bool(
        row["verdict"] in {"strong_equivalent", "identical"}
        or all(
            math.isfinite(float(row.get(f"{metric_name}_tost_pvalue_1x_bh", math.nan)))
            and float(row[f"{metric_name}_tost_pvalue_1x_bh"]) < 0.05
            for metric_name in QUALITY_METRICS
        )
    )
    row["_tost_pass_1_5x"] = bool(
        row["verdict"] in {"strong_equivalent", "weak_equivalent", "identical"}
        or all(
            math.isfinite(float(row.get(f"{metric_name}_tost_pvalue_1_5x_bh", math.nan)))
            and float(row[f"{metric_name}_tost_pvalue_1_5x_bh"]) < 0.05
            for metric_name in QUALITY_METRICS
        )
    )


def family_summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-graph rows into algorithm-family summaries.

    Parameters
    ----------
    rows : Sequence[Mapping[str, Any]]
        Finalized per-graph rows.

    Returns
    -------
    list[dict[str, Any]]
        Algorithm-summary output rows.
    """
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["algorithm_family"])].append(row)

    summaries: list[dict[str, Any]] = []
    for family_name, family_rows in sorted(grouped.items()):
        paired_ok = [row for row in family_rows if row["verdict"] != "insufficient_data"]
        insufficient = [row for row in family_rows if row["verdict"] == "insufficient_data"]
        procrustes_values = finite_values(float(row["procrustes_rmsd_mean"]) for row in paired_ok)
        scale_values = finite_values(float(row["scale_ratio_mean"]) for row in paired_ok)
        runtime_values = finite_values(float(row["runtime_ratio"]) for row in paired_ok)
        anomalies = [row for row in paired_ok if str(row["anomaly_reason"])]
        verdicts = {str(row["verdict"]) for row in paired_ok}
        is_stochastic = any(bool(row["_variant_is_stochastic"]) for row in family_rows)
        if paired_ok and all(str(row["verdict"]) == "identical" for row in paired_ok):
            verdict = "identical"
        elif paired_ok and all(
            str(row["verdict"]) in {"strong_equivalent", "identical"} for row in paired_ok
        ):
            verdict = "strong_equivalent"
        elif paired_ok and all(
            str(row["verdict"]) in {"strong_equivalent", "weak_equivalent", "identical"}
            for row in paired_ok
        ):
            verdict = "weak_equivalent"
        elif "divergent" in verdicts:
            verdict = "divergent"
        else:
            verdict = "partial_match"
        summaries.append(
            {
                "algorithm_family": family_name,
                "is_stochastic": is_stochastic,
                "num_graphs_tested": len(family_rows),
                "num_graphs_paired_ok": len(paired_ok),
                "num_graphs_insufficient_data": len(insufficient),
                "num_nan_rejected": sum(int(row["_rejection_count"]) for row in family_rows),
                "procrustes_rmsd_mean": safe_mean(procrustes_values),
                "procrustes_rmsd_median": safe_median(procrustes_values),
                "procrustes_rmsd_max": max(procrustes_values) if procrustes_values else math.nan,
                "scale_ratio_mean": safe_mean(scale_values),
                "scale_ratio_std": safe_std(scale_values),
                "num_mirror_matches": sum(bool(row["reflected"]) for row in paired_ok),
                "mean_runtime_ratio": safe_mean(runtime_values),
                "std_runtime_ratio": safe_std(runtime_values),
                "verdict": verdict,
                "anomaly_count": len(anomalies),
                "anomaly_graphs": "; ".join(
                    sorted({f"{row['variant_id']}:{row['graph_name']}" for row in anomalies})
                ),
                "tost_pass_rate_at_1x": (
                    sum(bool(row["_tost_pass_1x"]) for row in paired_ok) / len(paired_ok)
                    if paired_ok
                    else math.nan
                ),
                "tost_pass_rate_at_1_5x": (
                    sum(bool(row["_tost_pass_1_5x"]) for row in paired_ok) / len(paired_ok)
                    if paired_ok
                    else math.nan
                ),
            }
        )
    return summaries


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    """Write rows to CSV with a fixed header.

    Parameters
    ----------
    path : Path
        Destination CSV file.
    rows : Sequence[Mapping[str, Any]]
        Output rows.
    fieldnames : Sequence[str]
        CSV header order.
    """
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_readme(
    path: Path,
    results_hash: str,
    previous_hash: Optional[str],
    power_effect: float,
) -> None:
    """Write the analysis data README.

    Parameters
    ----------
    path : Path
        Destination README path.
    results_hash : str
        Current ``results.json`` SHA-256 hash.
    previous_hash : str | None
        Prior hash from an earlier run when available.
    power_effect : float
        Simulated minimum detectable effect for the Mann-Whitney test.
    """
    hash_changed = previous_hash is not None and previous_hash != results_hash
    lines = [
        "# Fidelity Analysis Data",
        "",
        f"{README_HASH_PREFIX} `{results_hash}`",
        "",
        "## Provenance",
        "",
        "- Source benchmark directory: `eval_output/variant_bench_full`",
        "- Position tensors loaded from `eval_output/variant_bench_full/positions/`",
        "- Metrics computed with `dagua.metrics.quick`",
        (
            "- Estimated Mann-Whitney minimum detectable effect at n=10/side "
            f"and 80% power: `d >= {power_effect:.2f}`"
        ),
        "",
        "## Methodology Notes",
        "",
        (
            "- Procrustes alignment uses centering plus optimal proper rotation "
            "only; scale is reported, not normalized away."
        ),
        "- Reflections are detected and flagged, but never used to improve the reported RMSD.",
        (
            "- Stochastic comparisons require at least 10 valid seeds on each "
            "side; otherwise the group is marked `insufficient_data`."
        ),
        (
            "- TOST equivalence p-values are reported for margins of 0.5x, "
            "1.0x, 1.5x, and 2.0x the within-original standard deviation, "
            "subject to metric-specific floors."
        ),
        (
            "- KS and Mann-Whitney are secondary difference-detection tests; "
            "failure to reject the KS null is not evidence of equivalence."
        ),
        (
            "- Benjamini-Hochberg FDR correction is applied separately to "
            "each test family across all graph-level metric tests."
        ),
        "",
        "## Columns",
        "",
        (
            "- `algorithm_summary.csv`: one row per algorithm family, "
            "aggregating all analyzed variant/graph comparisons in that family."
        ),
        (
            "- `per_graph_detail.csv`: one row per `(variant_id, graph_name)` "
            "pair, including metric summaries and corrected p-values."
        ),
        (
            "- `per_seed_detail.csv`: one row per valid seed/layout sample, "
            "with quick metrics and nearest cross-side Procrustes distance."
        ),
        (
            "- `pairwise_similarity.csv`: downsampled pairwise Procrustes "
            "comparisons. Deterministic groups emit the single "
            "`orig-reimpl` pair."
        ),
    ]
    if hash_changed:
        lines.extend(
            [
                "",
                "## Warning",
                "",
                (
                    f"- Previous run hash differed: `{previous_hash}`. "
                    "The underlying benchmark data changed between runs."
                ),
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def previous_results_hash(readme_path: Path) -> Optional[str]:
    """Read the previous benchmark hash from an existing README.

    Parameters
    ----------
    readme_path : Path
        README path.

    Returns
    -------
    str | None
        Previously recorded hash when present.
    """
    if not readme_path.exists():
        return None
    for line in readme_path.read_text(encoding="utf-8").splitlines():
        if line.startswith(README_HASH_PREFIX):
            parts = line.split("`")
            if len(parts) >= 2:
                return parts[1]
    return None


def run_analysis(
    input_dir: Path,
    output_dir: Path,
    max_graphs: Optional[int],
    bootstrap_samples: int,
    power_simulations: int,
) -> None:
    """Run the end-to-end fidelity analysis.

    Parameters
    ----------
    input_dir : Path
        Benchmark artifact root.
    output_dir : Path
        Destination directory for structured outputs.
    max_graphs : int | None
        Optional graph-count cap.
    bootstrap_samples : int
        Bootstrap sample count.
    power_simulations : int
        Simulation count for the power estimate.
    """
    results_path = input_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing benchmark results: {results_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    previous_hash = previous_results_hash(output_dir / "README.md")
    results_hash = compute_sha256(results_path)
    records = load_results(results_path)
    graph_filter = selected_graph_names(records, max_graphs)
    graph_registry = load_graph_registry()
    groups = build_variant_groups(records, graph_filter)

    # Use HDF5 for fast loading if available
    h5_path = input_dir / "positions.h5"
    h5_file = None
    if h5_path.exists():
        import h5py

        h5_file = h5py.File(str(h5_path), "r")
        load_layout._h5_file = h5_file  # type: ignore[attr-defined]
        print(f"[fidelity] Using HDF5 cache: {h5_path}", file=sys.stderr)
    else:
        print(
            "[fidelity] No HDF5 cache found. Loading individual .pt files. "
            "Run scripts/consolidate_positions_hdf5.py to speed this up.",
            file=sys.stderr,
        )
    pvalue_buckets = {
        "ks": PValueBucket(),
        "mannwhitney": PValueBucket(),
        "tost_0_5x": PValueBucket(),
        "tost_1x": PValueBucket(),
        "tost_1_5x": PValueBucket(),
        "tost_2x": PValueBucket(),
    }
    per_graph_rows: list[dict[str, Any]] = []
    per_seed_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    load_counter: Counter[str] = Counter()

    sorted_groups = sorted(
        groups.items(),
        key=lambda item: (
            algorithm_family(item[0][0]),
            item[0][0],
            item[0][1],
        ),
    )
    variant_by_id = {variant.variant_id: variant for variant in VARIANT_REGISTRY}
    for (variant_id, graph_name), grouped_records in sorted_groups:
        variant = variant_by_id[variant_id]
        if graph_name not in graph_registry:
            continue
        group_result = process_group(
            variant=variant,
            graph_name=graph_name,
            records=grouped_records,
            input_dir=input_dir,
            test_graph=graph_registry[graph_name],
            row_index=len(per_graph_rows),
            pvalue_buckets=pvalue_buckets,
            bootstrap_samples=bootstrap_samples,
            load_counter=load_counter,
        )
        group_result.row["_rejection_count"] = group_result.rejection_count
        per_graph_rows.append(group_result.row)
        per_seed_rows.extend(group_result.seed_rows)
        pairwise_rows.extend(group_result.pairwise_rows)

    apply_bh_correction(per_graph_rows, pvalue_buckets)
    for row in per_graph_rows:
        finalize_group_row(row)

    summary_rows = family_summary_rows(per_graph_rows)
    power_effect = estimate_mw_min_detectable_effect(power_simulations)

    write_csv(output_dir / "algorithm_summary.csv", summary_rows, algorithm_summary_fieldnames())
    write_csv(output_dir / "per_graph_detail.csv", per_graph_rows, per_graph_fieldnames())
    write_csv(output_dir / "per_seed_detail.csv", per_seed_rows, per_seed_fieldnames())
    write_csv(output_dir / "pairwise_similarity.csv", pairwise_rows, pairwise_fieldnames())
    write_readme(output_dir / "README.md", results_hash, previous_hash, power_effect)


def main() -> None:
    """Parse arguments and run the fidelity analysis."""
    args = parse_args()
    run_analysis(
        input_dir=args.input,
        output_dir=args.output,
        max_graphs=args.max_graphs,
        bootstrap_samples=args.bootstrap_samples,
        power_simulations=args.power_simulations,
    )


if __name__ == "__main__":
    main()
