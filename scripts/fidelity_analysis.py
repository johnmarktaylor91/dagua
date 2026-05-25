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
from scipy.optimize import linear_sum_assignment
from scipy.stats import ks_2samp, mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.weightstats import ttost_ind

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dagua.eval.pipeline_io import (  # noqa: E402
    load_position_tensor,
    stable_seed,
    validate_positions,
)
from dagua.eval.variants import (  # noqa: E402
    VARIANT_REGISTRY,
    AlgorithmVariant,
    algorithm_family,
    original_variant_name,
)
from dagua.metrics import (  # noqa: E402
    quick,  # noqa: E402
    sampled_crossing_rate,
    sampled_stress,
)

QUALITY_METRICS: tuple[str, ...] = (
    "aspect_ratio",
    "dag_consistency",
    "edge_length_cv",
    "edge_straightness_mean_deg",
    "depth_spearman_rho",
    "overlap_count",
)
SAMPLED_QUALITY_METRICS: tuple[str, ...] = (
    "sampled_stress",
    "crossing_rate",
)
ALL_QUALITY_METRICS: tuple[str, ...] = QUALITY_METRICS + SAMPLED_QUALITY_METRICS
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
    "edge_straightness_mean_deg": 3.0,
    "depth_spearman_rho": 0.05,
    "overlap_count": 5.0,
    "sampled_stress": 1e-3,
    "crossing_rate": 1e-4,
}
QUALITY_GATE_STRONG_MAX_REGRESSION_PCT = 10.0
QUALITY_GATE_WEAK_MAX_REGRESSION_PCT = 25.0
HIGHER_IS_BETTER_QUALITY_METRICS: frozenset[str] = frozenset(
    {
        "dag_consistency",
        "depth_spearman_rho",
    }
)
LOWER_IS_BETTER_QUALITY_METRICS: frozenset[str] = frozenset(
    {
        "edge_length_cv",
        "edge_straightness_mean_deg",
        "overlap_count",
        "sampled_stress",
        "crossing_rate",
    }
)
FIDELITY_STRESS_N_SOURCES = 32
FIDELITY_STRESS_N_TARGETS = 128
FIDELITY_CROSSING_N_SAMPLES = 50_000
COMPUTE_SAMPLED_METRICS = True
MIN_PROCRUSTES_NODE_COUNT = 5
MAX_HUNGARIAN_NODE_COUNT = 2_000
MIN_STOCHASTIC_SEEDS = 10
PAIRWISE_SAMPLE_SIZE = 30
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
        Selected per-layout quality metrics for this layout.
    """

    graph_name: str
    variant_id: str
    side: str
    seed: Optional[int]
    runtime_seconds: Optional[float]
    positions: torch.Tensor
    metrics: dict[str, float]


@dataclass
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
    result_key : str
        Stable key from ``results.json`` used for HDF5 lookups.
    error_message : str | None
        Benchmark error message when ``status == "error"``.
    skip_reason : str | None
        Benchmark skip reason when ``status == "skipped"``.
    """

    graph_name: str
    engine_name: str
    seed: Optional[int]
    status: str
    runtime_seconds: Optional[float]
    positions_file: Optional[str]
    result_key: str = ""
    error_message: Optional[str] = None
    skip_reason: Optional[str] = None


@dataclass
class SideRecordGroup:
    """Partitioned benchmark records for one side of a variant/graph group.

    Parameters
    ----------
    ok_records : list[ResultRecord]
        Records with ``status == "ok"`` that are eligible for layout loading.
    non_ok_records : list[ResultRecord]
        Records retained for rejection accounting.
    """

    ok_records: list[ResultRecord] = field(default_factory=list)
    non_ok_records: list[ResultRecord] = field(default_factory=list)


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
    hungarian_rmsd : float
        RMSD after Procrustes alignment plus optimal point assignment.
    scale_ratio : float
        Frobenius scale ratio between centered layouts.
    reflected : bool
        Whether a reflected alignment would improve the fit materially.
    max_node_displacement : float
        Maximum per-node displacement after proper-rotation alignment.
    variant_id : str
        Variant identifier shared by the compared layouts.
    """

    comparison_type: str = ""
    seed_a: Optional[int] = None
    seed_b: Optional[int] = None
    procrustes_rmsd: float = 0.0
    hungarian_rmsd: float = 0.0
    scale_ratio: float = 0.0
    reflected: bool = False
    max_node_displacement: float = 0.0
    variant_id: str = ""


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

    def add(self, row_idx: int, raw_pvalue: float, column: str = "") -> None:
        """Append a raw p-value entry for later BH correction."""
        self.entries.append((row_idx, column, raw_pvalue))


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
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Skip quality metrics computation (Procrustes + stats only, much faster)",
    )
    parser.add_argument(
        "--without-sampled-metrics",
        action="store_true",
        help=(
            "Skip sampled_stress and crossing_rate for faster iteration; avoids the "
            "extra 10-50 ms per layout sampled-metric cost in full fidelity runs."
        ),
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
        error_message=optional_str(payload.get("error")),
        skip_reason=optional_str(payload.get("skip_reason")),
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
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    return float(np.std(arr, ddof=1))


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
    positions_cache = getattr(load_layout, "_positions_cache", None)
    h5_file = getattr(load_layout, "_h5_file", None)
    record_key = record.result_key

    if positions_cache is not None and record_key and record_key in positions_cache:
        positions = positions_cache[record_key]
    else:
        positions, reason = load_position_tensor(
            record_key=record_key,
            positions_file=record.positions_file,
            input_dir=input_dir,
            h5_file=h5_file,
        )
        if positions is None:
            return None, reason

    rejection = validate_positions(positions, int(node_sizes.shape[0]))
    if rejection is not None:
        return None, rejection
    skip_metrics = getattr(load_layout, "_skip_metrics", False)
    compute_sampled = getattr(load_layout, "_compute_sampled_metrics", COMPUTE_SAMPLED_METRICS)
    layout_seed = stable_seed(record.graph_name, variant_id, side, str(record.seed or 0))
    if skip_metrics:
        metrics = {}
    else:
        metrics = {
            metric_name: float(metric_value)
            for metric_name, metric_value in quick(
                positions,
                edge_index,
                node_sizes=node_sizes,
                seed=layout_seed,
            ).items()
            if metric_name in QUALITY_METRICS
        }
        if compute_sampled:
            # Sampled metrics live outside quick(); keep budgets explicit so
            # the fidelity runtime cost is deliberate and reproducible.
            stress_result = sampled_stress(
                positions,
                edge_index,
                num_nodes=int(node_sizes.shape[0]),
                n_sources=FIDELITY_STRESS_N_SOURCES,
                n_targets=FIDELITY_STRESS_N_TARGETS,
            )
            crossing_result = sampled_crossing_rate(
                positions,
                edge_index,
                n_samples=FIDELITY_CROSSING_N_SAMPLES,
                seed=layout_seed,
            )
            for key, value in stress_result.items():
                if key in SAMPLED_QUALITY_METRICS:
                    metrics[key] = float(value)
            for key, value in crossing_result.items():
                if key in SAMPLED_QUALITY_METRICS:
                    metrics[key] = float(value)
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
    """Align two layouts WITH scale normalization, using best-of-two rotations.

    Both layouts are centered and normalized to unit Frobenius norm before
    alignment. Both proper rotation and reflected rotation are tested; the
    one with lower RMSD is returned. This handles mirror matches (SVD sign
    ambiguity) that are common in graph layout algorithms.

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
    reflected_per_node = torch.norm(reflected_aligned - b_centered, dim=1)
    reflected_rmsd = float(torch.sqrt(torch.mean(reflected_per_node.square())).item())
    if reflected_rmsd < rmsd:
        return reflected_rmsd, scale_ratio, True, reflected_per_node
    return rmsd, scale_ratio, False, per_node


def _procrustes_aligned_normalized_points(
    pos_a: torch.Tensor,
    pos_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return scale-normalized Procrustes-aligned point clouds.

    Parameters
    ----------
    pos_a : torch.Tensor
        First position tensor shaped ``[N, 2]``.
    pos_b : torch.Tensor
        Second position tensor shaped ``[N, 2]``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``pos_a`` aligned to ``pos_b`` using the same scale-normalized
        Procrustes behavior as :func:`fidelity_procrustes`, followed by the
        normalized ``pos_b`` reference points.
    """
    a_centered = pos_a - pos_a.mean(dim=0, keepdim=True)
    b_centered = pos_b - pos_b.mean(dim=0, keepdim=True)
    norm_a = float(a_centered.norm().item())
    norm_b = float(b_centered.norm().item())

    if norm_a > 0.0:
        a_centered = a_centered / norm_a
    if norm_b > 0.0:
        b_centered = b_centered / norm_b

    covariance = a_centered.t() @ b_centered
    left_singular, _, right_singular_t = torch.linalg.svd(covariance)
    det_value = torch.det(left_singular @ right_singular_t)
    correction = torch.diag(
        torch.tensor(
            [1.0, float(torch.sign(det_value).item())],
            dtype=a_centered.dtype,
            device=a_centered.device,
        )
    )
    rotation = left_singular @ correction @ right_singular_t
    aligned = a_centered @ rotation

    reflected_rotation = left_singular @ right_singular_t
    reflected_aligned = a_centered @ reflected_rotation
    rotation_rmsd = torch.sqrt(((aligned - b_centered).square()).sum(dim=1).mean())
    reflection_rmsd = torch.sqrt(((reflected_aligned - b_centered).square()).sum(dim=1).mean())
    if reflection_rmsd < rotation_rmsd:
        return reflected_aligned, b_centered
    return aligned, b_centered


def hungarian_matched_rmsd(
    positions_a: torch.Tensor,
    positions_b: torch.Tensor,
) -> float:
    """Return RMSD after Procrustes alignment and optimal point assignment.

    Parameters
    ----------
    positions_a : torch.Tensor
        First position tensor shaped ``[N, 2]``.
    positions_b : torch.Tensor
        Second position tensor shaped ``[N, 2]``.

    Returns
    -------
    float
        Scale-normalized RMSD after assigning aligned points with the Hungarian
        algorithm. Returns ``math.nan`` when shapes differ or the exact
        assignment would exceed the configured node-count guardrail.
    """
    if positions_a.shape != positions_b.shape or positions_a.ndim != 2:
        return math.nan
    if positions_a.shape[0] > MAX_HUNGARIAN_NODE_COUNT:
        return math.nan

    aligned_a, normalized_b = _procrustes_aligned_normalized_points(positions_a, positions_b)
    cost_matrix = torch.cdist(aligned_a, normalized_b).detach().cpu().numpy()
    row_indices, column_indices = linear_sum_assignment(cost_matrix)
    matched_costs = cost_matrix[row_indices, column_indices]
    return float(np.sqrt(np.mean(np.square(matched_costs))))


def procrustes_align_rigid(
    pos_a: torch.Tensor,
    pos_b: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    """Return ``pos_b`` aligned to ``pos_a`` via centering plus rotation only.

    Unlike :func:`fidelity_procrustes`, this helper does not normalize scale.
    Use it for deterministic equivalence checks where scale drift should fail
    the geometric match tier.

    Parameters
    ----------
    pos_a : torch.Tensor
        Reference position tensor shaped ``[N, 2]``.
    pos_b : torch.Tensor
        Candidate position tensor shaped ``[N, 2]``.

    Returns
    -------
    tuple[torch.Tensor, bool]
        ``pos_b`` aligned into ``pos_a`` coordinates and a reflection flag
        indicating whether the best rigid alignment used a reflected basis.
    """
    center_a = pos_a.mean(dim=0, keepdim=True)
    center_b = pos_b.mean(dim=0, keepdim=True)
    centered_a = pos_a - center_a
    centered_b = pos_b - center_b

    covariance = centered_b.t() @ centered_a
    left_singular, _, right_singular_t = torch.linalg.svd(covariance)
    rotation = left_singular @ right_singular_t
    reflected = False
    if torch.det(rotation) < 0:
        right_singular_t_reflected = right_singular_t.clone()
        right_singular_t_reflected[-1] = -right_singular_t_reflected[-1]
        reflected_rotation = left_singular @ right_singular_t_reflected
        aligned_rotation = centered_b @ rotation + center_a
        aligned_reflection = centered_b @ reflected_rotation + center_a
        rotation_rmsd = torch.sqrt(((pos_a - aligned_rotation).square()).sum(dim=1).mean())
        reflection_rmsd = torch.sqrt(((pos_a - aligned_reflection).square()).sum(dim=1).mean())
        if reflection_rmsd < rotation_rmsd:
            return aligned_reflection, True
        return aligned_rotation, reflected
    aligned = centered_b @ rotation + center_a
    return aligned, reflected


def deterministic_verdict_from_layouts(
    original_layout: LayoutRecord,
    reimpl_layout: LayoutRecord,
) -> tuple[Optional[str], int, str]:
    """Return the deterministic verdict tier for one orig/reimpl pair.

    Parameters
    ----------
    original_layout : LayoutRecord
        Original implementation layout for one graph.
    reimpl_layout : LayoutRecord
        Reimplementation layout for the same graph.

    Returns
    -------
    tuple[str | None, int, str]
        Verdict label when a deterministic tier matches, the winning tier
        number, and semicolon-delimited rejection reasons when all direct
        comparators fail and the caller must fall back to heuristics.
    """
    original_positions = original_layout.positions
    reimpl_positions = reimpl_layout.positions
    rejection_reasons: list[str] = []

    if original_positions.shape == reimpl_positions.shape and torch.equal(
        original_positions, reimpl_positions
    ):
        return "identical", 1, ""
    rejection_reasons.append("tier1_raw_tensor_mismatch")

    if original_positions.shape == reimpl_positions.shape:
        try:
            aligned_reimpl, _ = procrustes_align_rigid(original_positions, reimpl_positions)
        except RuntimeError:
            rejection_reasons.append("tier2_rigid_alignment_failed")
        else:
            if torch.allclose(original_positions, aligned_reimpl, atol=1e-6, rtol=1e-4):
                return "geometric_equivalent", 2, ""
            rejection_reasons.append("tier2_rigid_geometric_mismatch")
    else:
        rejection_reasons.append("tier2_shape_mismatch")

    original_metrics = original_layout.metrics or {}
    reimpl_metrics = reimpl_layout.metrics or {}
    all_metrics_close = True
    compared_any = False
    for metric_name in QUALITY_METRICS:
        if metric_name in original_metrics and metric_name in reimpl_metrics:
            compared_any = True
            original_value = float(original_metrics[metric_name])
            reimpl_value = float(reimpl_metrics[metric_name])
            if not math.isclose(original_value, reimpl_value, abs_tol=1e-6, rel_tol=1e-4):
                all_metrics_close = False
                break

    if compared_any and all_metrics_close:
        return "metric_equivalent", 3, ""
    if compared_any:
        rejection_reasons.append("tier3_metric_mismatch")
    else:
        rejection_reasons.append("tier3_metrics_unavailable")
    return None, 0, "; ".join(rejection_reasons)


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
    return np.asarray(
        [layout.metrics.get(metric_name, math.nan) for layout in layouts], dtype=np.float64
    )


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
    floor = METRIC_MARGIN_FLOORS[metric_name]
    return max(factor * std_orig, floor)


def relative_delta_pct(reference_value: float, candidate_value: float, floor: float) -> float:
    """Return percent difference from a reference value.

    Parameters
    ----------
    reference_value : float
        Reference-side mean value.
    candidate_value : float
        Reimplementation-side mean value.
    floor : float
        Minimum denominator used for near-zero reference metrics.

    Returns
    -------
    float
        Percent delta ``(candidate - reference) / denominator * 100``.
    """
    if not math.isfinite(reference_value) or not math.isfinite(candidate_value):
        return math.nan
    denominator = max(abs(reference_value), floor)
    return ((candidate_value - reference_value) / denominator) * 100.0


def metric_regression_pct(
    metric_name: str,
    reference_value: float,
    candidate_value: float,
) -> float:
    """Return directional quality regression as a non-negative percentage.

    Parameters
    ----------
    metric_name : str
        Quality metric identifier.
    reference_value : float
        Reference-side mean metric value.
    candidate_value : float
        Reimplementation-side mean metric value.

    Returns
    -------
    float
        Regression percentage where ``0`` means equal or improved quality.
    """
    floor = METRIC_MARGIN_FLOORS[metric_name]
    if metric_name == "aspect_ratio":
        reference_deviation = abs(math.log(max(reference_value, 1e-12)))
        candidate_deviation = abs(math.log(max(candidate_value, 1e-12)))
        return max(relative_delta_pct(reference_deviation, candidate_deviation, floor), 0.0)
    if metric_name in HIGHER_IS_BETTER_QUALITY_METRICS:
        return max(relative_delta_pct(reference_value, candidate_value, floor) * -1.0, 0.0)
    if metric_name in LOWER_IS_BETTER_QUALITY_METRICS:
        return max(relative_delta_pct(reference_value, candidate_value, floor), 0.0)
    raise KeyError(f"unknown quality metric direction: {metric_name}")


def quality_gate_status(
    row: Mapping[str, Any],
    threshold_pct: float,
) -> tuple[bool, float, list[str]]:
    """Evaluate a directional quality-regression gate for one row.

    Parameters
    ----------
    row : Mapping[str, Any]
        Per-graph row containing metric mean columns.
    threshold_pct : float
        Maximum allowed regression percentage for each metric.

    Returns
    -------
    tuple[bool, float, list[str]]
        Whether all available metrics pass, the maximum observed regression
        percentage, and metric names that exceed the threshold.
    """
    regressions: list[tuple[str, float]] = []
    for metric_name in ALL_QUALITY_METRICS:
        regression = _safe_float(row.get(f"{metric_name}_regression_pct"))
        if math.isfinite(regression):
            regressions.append((metric_name, regression))
    if not regressions:
        return False, math.nan, ["quality_metrics_unavailable"]
    failing = [metric_name for metric_name, value in regressions if value > threshold_pct]
    max_regression = max(value for _, value in regressions)
    return not failing, max_regression, failing


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
        f"{metric_name}_delta",
        f"{metric_name}_delta_pct",
        f"{metric_name}_regression_pct",
        f"{metric_name}_cohens_d",
        f"{metric_name}_cliffs_delta",
        f"{metric_name}_rank_biserial",
        f"{metric_name}_ks_pvalue_raw",
        f"{metric_name}_ks_pvalue_bh",
        f"{metric_name}_mannwhitney_pvalue_raw",
        f"{metric_name}_mannwhitney_pvalue_bh",
        f"{metric_name}_welch_pvalue_raw",
        f"{metric_name}_welch_pvalue_bh",
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


def procrustes_tost_columns() -> list[str]:
    """Return per-graph columns for Procrustes within-vs-between TOST tests.

    Returns
    -------
    list[str]
        Output column names for the Procrustes-specific TOST families.
    """
    columns: list[str] = []
    for factor in TOST_MARGIN_FACTORS:
        label = TOST_MARGIN_LABELS[factor]
        columns.extend(
            (
                f"procrustes_tost_margin_{label}",
                f"procrustes_tost_pvalue_{label}_raw",
                f"procrustes_tost_pvalue_{label}_bh",
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
        "hungarian_rmsd_mean",
        "hungarian_rmsd_std",
        "hungarian_rmsd_max",
        "scale_ratio_mean",
        "scale_ratio_std",
        "reflected",
        "max_node_displacement",
        "within_vs_between_pvalue",
        "within_vs_between_pvalue_bh",
        "procrustes_mannwhitney_pvalue_raw",
        "procrustes_mannwhitney_pvalue_bh",
        "within_rmsd_mean",
        "within_rmsd_std",
        "reimpl_rmsd_mean",
        "reimpl_rmsd_std",
        "between_rmsd_mean",
        "rmsd_ratio",
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
        "rejection_breakdown_json",
        "total_rejected",
        "_deterministic_tier",
        "verdict",
        "anomaly_reason",
        "quality_gate_strong_pass",
        "quality_gate_weak_pass",
        "quality_regression_max_pct",
        "quality_gate_failures",
    ]
    columns.extend(procrustes_tost_columns())
    for metric_name in ALL_QUALITY_METRICS:
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
    columns.extend(ALL_QUALITY_METRICS)
    return columns


def pairwise_fieldnames() -> list[str]:
    """Return the pairwise-similarity CSV header.

    Returns
    -------
    list[str]
        Output column names.
    """
    columns = [
        "algorithm_family",
        "graph_name",
        "seed_a",
        "seed_b",
        "comparison_type",
        "procrustes_rmsd",
        "hungarian_rmsd",
        "scale_ratio",
        "variant_id",
        "reflected",
        "max_node_displacement",
    ]
    return columns


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
        "hungarian_rmsd_mean",
        "hungarian_rmsd_median",
        "hungarian_rmsd_max",
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
            matched_rmsd = hungarian_matched_rmsd(layout_a.positions, layout_b.positions)
            comparisons.append(
                PairwiseComparison(
                    comparison_type=comparison_type,
                    seed_a=layout_a.seed,
                    seed_b=layout_b.seed,
                    procrustes_rmsd=rmsd,
                    hungarian_rmsd=matched_rmsd,
                    scale_ratio=scale_ratio,
                    reflected=reflected,
                    max_node_displacement=float(per_node.max().item()),
                    variant_id=layout_a.variant_id,
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
    values = [layout.metrics.get(metric_name, math.nan) for layout in layouts]
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


REJECTION_BREAKDOWN_KEYS: tuple[str, ...] = (
    "orig_error",
    "orig_timeout",
    "orig_skipped",
    "orig_running",
    "reimpl_error",
    "reimpl_timeout",
    "reimpl_skipped",
    "reimpl_running",
    "missing_positions_file",
    "h5_load_failure",
    "load_failure",
    "not_tensor",
    "tensor_not_2d",
    "tensor_not_xy",
    "too_few_nodes",
    "node_count_mismatch",
    "contains_nan",
    "contains_inf",
    "too_few_seeds",
)


def default_rejection_breakdown() -> dict[str, int]:
    """Return the canonical rejection breakdown mapping.

    Returns
    -------
    dict[str, int]
        Zero-initialized counters keyed by canonical rejection reason.
    """
    return {reason: 0 for reason in REJECTION_BREAKDOWN_KEYS}


def scheduling_rejection_reason(side: str, status: str) -> Optional[str]:
    """Map a non-``ok`` benchmark status to a rejection bucket.

    Parameters
    ----------
    side : str
        Benchmark side, either ``"orig"`` or ``"reimpl"``.
    status : str
        Benchmark status string from ``results.json``.

    Returns
    -------
    str | None
        Canonical rejection bucket, or ``None`` for statuses that should not
        be counted at scheduling time.
    """
    if status in {"error", "timeout", "skipped", "running"}:
        return f"{side}_{status}"
    return None


def increment_rejection_breakdown(
    rejection_breakdown: dict[str, int],
    reason: Optional[str],
) -> None:
    """Increment one rejection bucket in place.

    Parameters
    ----------
    rejection_breakdown : dict[str, int]
        Mutable rejection counter mapping.
    reason : str | None
        Rejection reason to increment.
    """
    if reason is None:
        return
    rejection_breakdown[reason] = rejection_breakdown.get(reason, 0) + 1


def finalize_rejection_columns(
    row: dict[str, Any],
    rejection_breakdown: Mapping[str, int],
) -> None:
    """Write structured rejection counters into the per-graph row.

    Parameters
    ----------
    row : dict[str, Any]
        Pending per-graph output row.
    rejection_breakdown : Mapping[str, int]
        Structured rejection counts collected for the group.
    """
    row["rejection_breakdown_json"] = json.dumps(dict(rejection_breakdown))
    row["total_rejected"] = int(sum(rejection_breakdown.values()))


def build_variant_groups(
    records: Mapping[str, ResultRecord],
    allowed_graphs: Optional[set[str]],
) -> dict[tuple[str, str], dict[str, SideRecordGroup]]:
    """Group benchmark records by variant and graph.

    Parameters
    ----------
    records : Mapping[str, ResultRecord]
        Parsed benchmark results.
    allowed_graphs : set[str] | None
        Optional graph filter.

    Returns
    -------
    dict[tuple[str, str], dict[str, SideRecordGroup]]
        Grouped metadata, with ``orig`` and ``reimpl`` side partitions that
        retain both successful and non-successful benchmark records.
    """
    original_name_to_variant: dict[str, AlgorithmVariant] = {}
    variant_by_id = {candidate.variant_id: candidate for candidate in VARIANT_REGISTRY}
    for candidate in VARIANT_REGISTRY:
        original_name = original_variant_name(candidate)
        if original_name is not None:
            original_name_to_variant[original_name] = candidate

    groups: dict[tuple[str, str], dict[str, SideRecordGroup]] = defaultdict(
        lambda: {"orig": SideRecordGroup(), "reimpl": SideRecordGroup()}
    )
    for record in records.values():
        if allowed_graphs is not None and record.graph_name not in allowed_graphs:
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
        side_group = groups[(variant.variant_id, record.graph_name)][side]
        if record.status == "ok":
            side_group.ok_records.append(record)
        else:
            side_group.non_ok_records.append(record)
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
    # Skip metric analysis entirely when metrics were not computed (--skip-metrics)
    skip = getattr(load_layout, "_skip_metrics", False)
    if skip:
        return
    for metric_name in ALL_QUALITY_METRICS:
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
        row[f"{metric_name}_delta"] = float(row[f"{metric_name}_reimpl_mean"]) - float(
            row[f"{metric_name}_orig_mean"]
        )
        row[f"{metric_name}_delta_pct"] = relative_delta_pct(
            float(row[f"{metric_name}_orig_mean"]),
            float(row[f"{metric_name}_reimpl_mean"]),
            METRIC_MARGIN_FLOORS[metric_name],
        )
        row[f"{metric_name}_regression_pct"] = metric_regression_pct(
            metric_name,
            float(row[f"{metric_name}_orig_mean"]),
            float(row[f"{metric_name}_reimpl_mean"]),
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
        _, welch_pvalue = ttest_ind(
            original_values,
            reimpl_values,
            equal_var=False,
            alternative="two-sided",
        )
        row[f"{metric_name}_ks_pvalue_raw"] = ks_pvalue
        row[f"{metric_name}_mannwhitney_pvalue_raw"] = mw_pvalue
        row[f"{metric_name}_welch_pvalue_raw"] = (
            float(welch_pvalue) if np.isfinite(welch_pvalue) else math.nan
        )
        pvalue_buckets["ks"].entries.append((row_index, f"{metric_name}_ks_pvalue_bh", ks_pvalue))
        pvalue_buckets["mannwhitney"].entries.append(
            (row_index, f"{metric_name}_mannwhitney_pvalue_bh", mw_pvalue)
        )
        pvalue_buckets["welch"].entries.append(
            (row_index, f"{metric_name}_welch_pvalue_bh", float(welch_pvalue))
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
    for metric_name in ALL_QUALITY_METRICS:
        for column in metric_test_columns(metric_name):
            row[column] = math.nan

    for column in procrustes_tost_columns():
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
    records: dict[str, SideRecordGroup],
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
    records : dict[str, SideRecordGroup]
        Side-partitioned benchmark records, preserving both successful and
        non-successful runs.
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
        "hungarian_rmsd_mean": math.nan,
        "hungarian_rmsd_std": math.nan,
        "hungarian_rmsd_max": math.nan,
        "scale_ratio_mean": math.nan,
        "scale_ratio_std": math.nan,
        "reflected": False,
        "max_node_displacement": math.nan,
        "within_vs_between_pvalue": math.nan,
        "within_vs_between_pvalue_bh": math.nan,
        "procrustes_mannwhitney_pvalue_raw": math.nan,
        "procrustes_mannwhitney_pvalue_bh": math.nan,
        "within_rmsd_mean": math.nan,
        "within_rmsd_std": math.nan,
        "reimpl_rmsd_mean": math.nan,
        "reimpl_rmsd_std": math.nan,
        "between_rmsd_mean": math.nan,
        "rmsd_ratio": math.nan,
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
        "rejection_breakdown_json": "{}",
        "total_rejected": 0,
        "_deterministic_tier": 0,
        "verdict": "insufficient_data",
        "anomaly_reason": "",
        "_deterministic_verdict": "",
        "_deterministic_rejection_reasons": "",
        "_variant_is_stochastic": variant.is_stochastic,
        "_structural_note_flag": descriptor.structural_note != "none",
    }
    initialize_metric_columns(row)

    loaded_layouts: dict[str, list[LayoutRecord]] = {"orig": [], "reimpl": []}
    rejection_count = 0
    rejection_breakdown = default_rejection_breakdown()
    for side in ("orig", "reimpl"):
        for record in records[side].non_ok_records:
            increment_rejection_breakdown(
                rejection_breakdown,
                scheduling_rejection_reason(side, record.status),
            )

    # Parallel loading -- I/O bound so threads give ~4-8x speedup
    load_tasks: list[tuple[str, Any]] = []
    for side in ("orig", "reimpl"):
        for record in sorted(
            records[side].ok_records,
            key=lambda candidate: (candidate.seed is None, candidate.seed),
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
                increment_rejection_breakdown(rejection_breakdown, rejection)
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
        finalize_rejection_columns(row, rejection_breakdown)
        return GroupResult(
            row=row,
            seed_rows=seed_rows,
            pairwise_rows=pairwise_rows,
            rejection_count=rejection_count,
        )

    pairwise_orig_reimpl: list[PairwiseComparison] = []
    if variant.is_stochastic:
        if len(reimpl_layouts) < MIN_STOCHASTIC_SEEDS:
            increment_rejection_breakdown(rejection_breakdown, "too_few_seeds")
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
                    for metric_name in ALL_QUALITY_METRICS:
                        seed_row[metric_name] = layout.metrics.get(metric_name, math.nan)
                    seed_rows.append(seed_row)
            finalize_rejection_columns(row, rejection_breakdown)
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
                    "hungarian_rmsd": comparison.hungarian_rmsd,
                    "scale_ratio": comparison.scale_ratio,
                    "variant_id": comparison.variant_id,
                    "reflected": str(comparison.reflected),
                    "max_node_displacement": comparison.max_node_displacement,
                }
            )
        if descriptor.num_nodes >= MIN_PROCRUSTES_NODE_COUNT:
            rmsd_values = [comparison.procrustes_rmsd for comparison in pairwise_orig_reimpl]
            matched_rmsd_values = finite_values(
                comparison.hungarian_rmsd for comparison in pairwise_orig_reimpl
            )
            scale_values = [comparison.scale_ratio for comparison in pairwise_orig_reimpl]
            max_displacements = [
                comparison.max_node_displacement for comparison in pairwise_orig_reimpl
            ]
            reflected = any(comparison.reflected for comparison in pairwise_orig_reimpl)
            pairwise_summary = pairwise_statistics(rmsd_values)
            row["procrustes_rmsd_mean"] = pairwise_summary["mean"]
            row["procrustes_rmsd_std"] = pairwise_summary["std"]
            row["procrustes_rmsd_max"] = pairwise_summary["max"]
            matched_pairwise_summary = pairwise_statistics(matched_rmsd_values)
            row["hungarian_rmsd_mean"] = matched_pairwise_summary["mean"]
            row["hungarian_rmsd_std"] = matched_pairwise_summary["std"]
            row["hungarian_rmsd_max"] = matched_pairwise_summary["max"]
            row["scale_ratio_mean"] = safe_mean(scale_values)
            row["scale_ratio_std"] = safe_std(scale_values)
            row["reflected"] = reflected
            row["max_node_displacement"] = max(max_displacements) if max_displacements else math.nan

            # Within-vs-between Procrustes test: is between-engine RMSD
            # significantly greater than within-engine RMSD?
            within_orig_rmsd = [c.procrustes_rmsd for c in pairwise_orig]
            within_reimpl_rmsd = [c.procrustes_rmsd for c in pairwise_reimpl]
            between_rmsd = rmsd_values
            if len(within_orig_rmsd) >= 2 and len(between_rmsd) >= 2:
                # One-sided: is between > within?
                _, wb_pval = mannwhitneyu(
                    between_rmsd,
                    within_orig_rmsd,
                    alternative="greater",
                )
                row["within_vs_between_pvalue"] = float(wb_pval)
                pvalue_buckets["procrustes_one_sided"].entries.append(
                    (row_index, "within_vs_between_pvalue_bh", float(wb_pval))
                )

                _, wb_two_sided = mannwhitneyu(
                    between_rmsd,
                    within_orig_rmsd,
                    alternative="two-sided",
                )
                row["procrustes_mannwhitney_pvalue_raw"] = float(wb_two_sided)
                pvalue_buckets["procrustes_mannwhitney"].entries.append(
                    (
                        row_index,
                        "procrustes_mannwhitney_pvalue_bh",
                        float(wb_two_sided),
                    )
                )

                row["within_rmsd_mean"] = safe_mean(within_orig_rmsd)
                row["within_rmsd_std"] = safe_std(within_orig_rmsd)
                row["reimpl_rmsd_mean"] = safe_mean(within_reimpl_rmsd)
                row["reimpl_rmsd_std"] = safe_std(within_reimpl_rmsd)
                row["between_rmsd_mean"] = safe_mean(between_rmsd)
                row["rmsd_ratio"] = safe_mean(between_rmsd) / max(
                    safe_mean(within_orig_rmsd),
                    1e-12,
                )
            else:
                row["within_vs_between_pvalue"] = math.nan
                row["within_rmsd_mean"] = (
                    safe_mean(within_orig_rmsd) if within_orig_rmsd else math.nan
                )
                row["within_rmsd_std"] = (
                    safe_std(within_orig_rmsd) if within_orig_rmsd else math.nan
                )
                row["reimpl_rmsd_mean"] = (
                    safe_mean(within_reimpl_rmsd) if within_reimpl_rmsd else math.nan
                )
                row["reimpl_rmsd_std"] = (
                    safe_std(within_reimpl_rmsd) if within_reimpl_rmsd else math.nan
                )
                row["between_rmsd_mean"] = safe_mean(between_rmsd) if between_rmsd else math.nan
                row["rmsd_ratio"] = math.nan

            # Procrustes TOST: is the between-engine distribution
            # statistically equivalent to within-original variation?
            if len(within_orig_rmsd) >= 2 and len(between_rmsd) >= 2:
                std_within_orig = float(np.std(within_orig_rmsd, ddof=1))
                # Zero-variance within-original baselines need a floor so the
                # equivalence band does not collapse to an exact-equality test.
                std_floor = max(std_within_orig, 1e-6)
                for factor in TOST_MARGIN_FACTORS:
                    label = TOST_MARGIN_LABELS[factor]
                    margin = factor * std_floor
                    try:
                        pvalue = tost_pvalue(
                            np.asarray(within_orig_rmsd, dtype=np.float64),
                            np.asarray(between_rmsd, dtype=np.float64),
                            margin,
                        )
                    except Exception:
                        pvalue = math.nan
                    row[f"procrustes_tost_margin_{label}"] = float(margin)
                    row[f"procrustes_tost_pvalue_{label}_raw"] = float(pvalue)
                    if math.isfinite(pvalue):
                        pvalue_buckets[f"procrustes_tost_{label}"].entries.append(
                            (
                                row_index,
                                f"procrustes_tost_pvalue_{label}_bh",
                                float(pvalue),
                            )
                        )
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
                for metric_name in ALL_QUALITY_METRICS:
                    seed_row[metric_name] = layout.metrics.get(metric_name, math.nan)
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
            row[f"{metric_name}_ks_pvalue_raw"] for metric_name in ALL_QUALITY_METRICS
        )
        row["mannwhitney_pvalue_raw"] = min(
            row[f"{metric_name}_mannwhitney_pvalue_raw"] for metric_name in ALL_QUALITY_METRICS
        )
        row["tost_pvalue_at_1x"] = max(
            row[f"{metric_name}_tost_pvalue_1x_raw"] for metric_name in ALL_QUALITY_METRICS
        )
        row["cliffs_delta"] = max(
            abs(float(row[f"{metric_name}_cliffs_delta"])) for metric_name in ALL_QUALITY_METRICS
        )
        finalize_rejection_columns(row, rejection_breakdown)
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
        matched_rmsd = hungarian_matched_rmsd(original_layout.positions, reimpl_layout.positions)
        row["hungarian_rmsd_mean"] = matched_rmsd
        row["hungarian_rmsd_std"] = 0.0 if math.isfinite(matched_rmsd) else math.nan
        row["hungarian_rmsd_max"] = matched_rmsd
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
                hungarian_rmsd=matched_rmsd,
                scale_ratio=scale_ratio,
                reflected=reflected,
                max_node_displacement=float(per_node.max().item()),
                variant_id=variant.variant_id,
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
                "hungarian_rmsd": matched_rmsd,
                "scale_ratio": scale_ratio,
                "variant_id": variant.variant_id,
                "reflected": str(reflected),
                "max_node_displacement": float(per_node.max().item()),
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
        for metric_name in ALL_QUALITY_METRICS:
            seed_row[metric_name] = layout.metrics.get(metric_name, math.nan)
            if layout.side == "orig":
                row[f"{metric_name}_orig_mean"] = layout.metrics.get(metric_name, math.nan)
                row[f"{metric_name}_orig_std"] = 0.0
            else:
                row[f"{metric_name}_reimpl_mean"] = layout.metrics.get(metric_name, math.nan)
                row[f"{metric_name}_reimpl_std"] = 0.0
        seed_rows.append(seed_row)
    for metric_name in ALL_QUALITY_METRICS:
        original_value = original_layout.metrics.get(metric_name, math.nan)
        reimpl_value = reimpl_layout.metrics.get(metric_name, math.nan)
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
    deterministic_verdict, deterministic_tier, deterministic_rejections = (
        deterministic_verdict_from_layouts(original_layout, reimpl_layout)
    )
    row["_deterministic_tier"] = deterministic_tier
    row["_deterministic_verdict"] = deterministic_verdict or ""
    row["_deterministic_rejection_reasons"] = deterministic_rejections
    finalize_rejection_columns(row, rejection_breakdown)
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
        # Separate finite and NaN entries -- NaN contaminates multipletests(),
        # causing ALL corrected values to become NaN (statsmodels bug/design).
        finite_entries = [
            (idx, col, pval) for idx, col, pval in bucket.entries if math.isfinite(pval)
        ]
        nan_entries = [
            (idx, col, pval) for idx, col, pval in bucket.entries if not math.isfinite(pval)
        ]
        if finite_entries:
            pvalues = [entry[2] for entry in finite_entries]
            _, corrected, _, _ = multipletests(pvalues, alpha=0.05, method="fdr_bh")
            for (row_index, column_name, _), corrected_value in zip(finite_entries, corrected):
                rows[row_index][column_name] = float(corrected_value)
        for row_index, column_name, _ in nan_entries:
            rows[row_index][column_name] = math.nan


def explainable_only(row: Mapping[str, Any]) -> bool:
    """Return whether a row has only benign anomalies.

    Mirror matches are explainable because most layout algorithms have
    arbitrary axis orientation (SVD sign ambiguity, etc.).  A mirrored
    layout is a valid equivalent output, not a fidelity failure.
    ``fidelity_procrustes`` now tests both rotations and returns the
    better fit, so mirror_match mostly appears when the reflected
    alignment was used.

    Parameters
    ----------
    row : Mapping[str, Any]
        Finalized per-graph row.

    Returns
    -------
    bool
        ``True`` when anomaly text is empty or purely benign.
    """
    anomaly_reason = str(row.get("anomaly_reason", ""))
    if not anomaly_reason:
        return True
    reasons = {part.strip() for part in anomaly_reason.split(";") if part.strip()}
    return reasons <= {
        "structural_note",
        "mirror_match",
        "scale_ratio_out_of_range",
        "runtime_ratio_outlier",
        "runtime_ratio_warning",
    }


def _safe_float(val: Any, default: float = math.nan) -> float:
    """Convert value to float, returning *default* on empty/invalid strings."""
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def finalize_group_row(row: dict[str, Any]) -> None:
    """Resolve row-level BH summaries, anomalies, and verdicts.

    Parameters
    ----------
    row : dict[str, Any]
        Pending per-graph row.
    """
    anomaly_reasons: list[str] = []
    stochastic = str(row["_variant_is_stochastic"]).lower() in ("true", "1")
    if row["structural_note"] != "none":
        anomaly_reasons.append("structural_note")

    if stochastic:
        _ks_vals = [
            v
            for v in (_safe_float(row.get(f"{m}_ks_pvalue_bh")) for m in ALL_QUALITY_METRICS)
            if math.isfinite(v)
        ]
        row["ks_pvalue_bh"] = min(_ks_vals) if _ks_vals else math.nan
        _mw_vals = [
            v
            for v in (
                _safe_float(row.get(f"{m}_mannwhitney_pvalue_bh")) for m in ALL_QUALITY_METRICS
            )
            if math.isfinite(v)
        ]
        row["mannwhitney_pvalue_bh"] = min(_mw_vals) if _mw_vals else math.nan
        _tost_vals = [
            v
            for v in (_safe_float(row.get(f"{m}_tost_pvalue_1x_bh")) for m in ALL_QUALITY_METRICS)
            if math.isfinite(v)
        ]
        row["tost_pvalue_at_1x_bh"] = max(_tost_vals) if _tost_vals else math.nan
        _cd_vals = [
            v
            for v in (abs(_safe_float(row.get(f"{m}_cliffs_delta"))) for m in ALL_QUALITY_METRICS)
            if math.isfinite(v)
        ]
        row["cliffs_delta"] = max(_cd_vals) if _cd_vals else math.nan
    counts_ok = int(row["num_orig_seeds"]) >= 1 and int(row["num_reimpl_seeds"]) >= 1
    if not counts_ok:
        row["verdict"] = "insufficient_data"
    elif stochastic and int(row["num_reimpl_seeds"]) < MIN_STOCHASTIC_SEEDS:
        # Only require MIN_STOCHASTIC_SEEDS on reimpl side. Many originals are
        # truly deterministic (OGDF, Graphviz, igraph) producing exactly 1 seed.
        # Comparing 1 deterministic orig against 10+ stochastic reimpl is valid.
        row["verdict"] = "insufficient_data"
    else:
        reflected = str(row.get("reflected", "")).lower() in ("true", "1")
        runtime_ratio_value = _safe_float(row.get("runtime_ratio"))
        scale_ratio_value = _safe_float(row.get("scale_ratio_mean"))
        max_displacement = _safe_float(row.get("max_node_displacement"))

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

        # Asymmetric case: stochastic reimpl but deterministic orig (1 seed).
        # Route through Procrustes-based verdict since TOST needs n>=2 both sides.
        _orig_has_enough = int(row["num_orig_seeds"]) >= MIN_STOCHASTIC_SEEDS
        if stochastic and _orig_has_enough:

            def _tost_passes(metric: str, label: str) -> bool:
                """Return True when TOST passes OR distributions are identical.

                NaN BH-corrected p-values arise from two sources:
                1. Identical distributions (pooled SD = 0 -> NaN raw p-value)
                2. BH correction contamination (NaN in bucket poisons all)
                For case 2, fall back to the raw p-value.
                """
                val = _safe_float(row.get(f"{metric}_tost_pvalue_{label}_bh"))
                if math.isnan(val):
                    orig_std = _safe_float(row.get(f"{metric}_orig_std"))
                    reimpl_std = _safe_float(row.get(f"{metric}_reimpl_std"))
                    orig_mean = _safe_float(row.get(f"{metric}_orig_mean"))
                    reimpl_mean = _safe_float(row.get(f"{metric}_reimpl_mean"))
                    # Case 1: truly identical distributions -> trivial equivalence
                    if (
                        math.isfinite(orig_mean)
                        and math.isfinite(reimpl_mean)
                        and abs(orig_mean - reimpl_mean) < 1e-9
                        and math.isfinite(orig_std)
                        and math.isfinite(reimpl_std)
                        and orig_std < 1e-9
                        and reimpl_std < 1e-9
                    ):
                        return True
                    # Case 2: BH correction failed -- fall back to raw p-value
                    raw_key = f"{metric}_tost_pvalue_{label}_raw"
                    raw_val = _safe_float(row.get(raw_key))
                    if math.isfinite(raw_val):
                        return raw_val < 0.05
                    return False  # genuinely missing data
                return val < 0.05

            # Within-vs-between Procrustes verdict: is the between-engine
            # RMSD significantly greater than the within-engine RMSD?
            wb_pval = _safe_float(row.get("within_vs_between_pvalue"))
            if not math.isfinite(wb_pval):
                # Not enough data for the test -- fall back to TOST
                tost_2x = all(
                    _tost_passes(metric_name, "2x") for metric_name in ALL_QUALITY_METRICS
                )
                if tost_2x:
                    row["verdict"] = "partial_match"
                else:
                    row["verdict"] = "insufficient_data"
            else:
                # TOST-based verdict replaces the old "failed to reject
                # difference" heuristic. The between-engine distribution now
                # has to be statistically equivalent to within-original
                # variation, not merely non-significantly different.
                def _procrustes_tost_pass(label: str) -> bool:
                    """Return whether one Procrustes TOST p-value passes."""

                    corrected = _safe_float(row.get(f"procrustes_tost_pvalue_{label}_bh"))
                    if math.isfinite(corrected):
                        return corrected < 0.05
                    raw = _safe_float(row.get(f"procrustes_tost_pvalue_{label}_raw"))
                    return math.isfinite(raw) and raw < 0.05

                procrustes_0_5x_pass = _procrustes_tost_pass("0_5x")
                procrustes_1x_pass = _procrustes_tost_pass("1x")
                strong_quality_pass, _, _ = quality_gate_status(
                    row,
                    QUALITY_GATE_STRONG_MAX_REGRESSION_PCT,
                )
                weak_quality_pass, _, _ = quality_gate_status(
                    row,
                    QUALITY_GATE_WEAK_MAX_REGRESSION_PCT,
                )

                if procrustes_0_5x_pass and strong_quality_pass:
                    row["verdict"] = "strong_equivalent"
                elif procrustes_1x_pass and weak_quality_pass:
                    row["verdict"] = "weak_equivalent"
                else:
                    row["verdict"] = "partial_match"
        else:
            deterministic_tier = int(row.get("_deterministic_tier", 0) or 0)
            deterministic_verdict = str(row.get("_deterministic_verdict", ""))
            deterministic_rejections = [
                reason.strip()
                for reason in str(row.get("_deterministic_rejection_reasons", "")).split(";")
                if reason.strip()
            ]
            if math.isfinite(max_displacement) and max_displacement > PROCRUSTES_ANOMALY_THRESHOLD:
                anomaly_reasons.append("max_node_displacement")
            if deterministic_tier > 0 and deterministic_verdict:
                row["verdict"] = deterministic_verdict
            else:
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
                anomaly_reasons.extend(deterministic_rejections)

    reasons = sorted(set(reason for reason in anomaly_reasons if reason))
    row["anomaly_reason"] = "; ".join(reasons)
    strong_quality_pass, max_regression, strong_failures = quality_gate_status(
        row,
        QUALITY_GATE_STRONG_MAX_REGRESSION_PCT,
    )
    weak_quality_pass, _, weak_failures = quality_gate_status(
        row,
        QUALITY_GATE_WEAK_MAX_REGRESSION_PCT,
    )
    row["quality_gate_strong_pass"] = strong_quality_pass
    row["quality_gate_weak_pass"] = weak_quality_pass
    row["quality_regression_max_pct"] = max_regression
    row["quality_gate_failures"] = "; ".join(
        sorted(set(strong_failures if not strong_quality_pass else weak_failures))
    )
    row["_tost_pass_1x"] = bool(
        row["verdict"]
        in {"strong_equivalent", "identical", "geometric_equivalent", "metric_equivalent"}
        or all(
            math.isfinite(_safe_float(row.get(f"{metric_name}_tost_pvalue_1x_bh")))
            and _safe_float(row.get(f"{metric_name}_tost_pvalue_1x_bh")) < 0.05
            for metric_name in ALL_QUALITY_METRICS
        )
    )
    row["_tost_pass_1_5x"] = bool(
        row["verdict"]
        in {
            "strong_equivalent",
            "weak_equivalent",
            "identical",
            "geometric_equivalent",
            "metric_equivalent",
        }
        or all(
            math.isfinite(_safe_float(row.get(f"{metric_name}_tost_pvalue_1_5x_bh")))
            and _safe_float(row.get(f"{metric_name}_tost_pvalue_1_5x_bh")) < 0.05
            for metric_name in ALL_QUALITY_METRICS
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
        matched_values = finite_values(float(row["hungarian_rmsd_mean"]) for row in paired_ok)
        scale_values = finite_values(float(row["scale_ratio_mean"]) for row in paired_ok)
        runtime_values = finite_values(float(row["runtime_ratio"]) for row in paired_ok)
        anomalies = [row for row in paired_ok if str(row["anomaly_reason"])]
        is_stochastic = any(
            str(row["_variant_is_stochastic"]).lower() in ("true", "1") for row in family_rows
        )
        # Proportion-based aggregation: use the fraction of graphs that
        # pass at each level instead of all-or-nothing.
        n = len(paired_ok) if paired_ok else 0
        if n == 0:
            verdict = "insufficient_data"
        else:
            n_identical = sum(1 for r in paired_ok if str(r["verdict"]) == "identical")
            n_strong = sum(
                1
                for r in paired_ok
                if str(r["verdict"])
                in {
                    "strong_equivalent",
                    "identical",
                    "geometric_equivalent",
                    "metric_equivalent",
                }
            )
            n_weak = sum(
                1
                for r in paired_ok
                if str(r["verdict"])
                in {
                    "strong_equivalent",
                    "weak_equivalent",
                    "identical",
                    "geometric_equivalent",
                    "metric_equivalent",
                }
            )
            n_divergent = sum(1 for r in paired_ok if str(r["verdict"]) == "divergent")
            # Family-level verdict aggregation rule (conservative majority):
            #   100% identical            -> "identical"
            #   >= 90% strong_equivalent  -> "strong_equivalent"
            #   >= 90% weak or stronger   -> "weak_equivalent"
            #   > 50% divergent           -> "divergent"
            #   else                      -> "partial_match"
            #
            # Rationale: a family verdict of "strong_equivalent" requires the
            # vast majority of graphs in that family to have strongly
            # equivalent per-graph verdicts. A handful of divergent graphs
            # within an otherwise-matching family downgrades the family to
            # "partial_match" rather than hiding the divergence under a
            # majority pass. The 0.90 thresholds were chosen empirically
            # during early iteration to tolerate occasional numerical flukes
            # without whitewashing real fidelity gaps.
            if n_identical == n:
                verdict = "identical"
            elif n_strong / n >= 0.90:
                verdict = "strong_equivalent"
            elif n_weak / n >= 0.90:
                verdict = "weak_equivalent"
            elif n_divergent / n > 0.50:
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
                "hungarian_rmsd_mean": safe_mean(matched_values),
                "hungarian_rmsd_median": safe_median(matched_values),
                "hungarian_rmsd_max": max(matched_values) if matched_values else math.nan,
                "scale_ratio_mean": safe_mean(scale_values),
                "scale_ratio_std": safe_std(scale_values),
                "num_mirror_matches": sum(
                    str(row["reflected"]).lower() in ("true", "1") for row in paired_ok
                ),
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
            "- Procrustes alignment uses centering plus unit-Frobenius "
            "normalization before the optimal fit, so RMSD is scale-invariant "
            "while the scale ratio is still reported separately."
        ),
        (
            "- Procrustes fitting evaluates both proper and reflected alignments, "
            "reports the lower RMSD, and flags when the reflected solution wins."
        ),
        (
            "- Hungarian RMSD is reported beside raw Procrustes RMSD as an "
            "alternative geometric-only metric: it uses the same alignment, "
            "then optimally assigns point labels with "
            "`scipy.optimize.linear_sum_assignment`."
        ),
        (
            f"- Exact Hungarian assignment is skipped above `{MAX_HUNGARIAN_NODE_COUNT}` "
            "nodes and reported as `NaN` to avoid quadratic cost matrices in "
            "large benchmark runs."
        ),
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
            "- `pairwise_similarity.csv`: downsampled pairwise Procrustes and "
            "Hungarian-matched comparisons. Deterministic groups emit the single "
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
    skip_metrics: bool = False,
    compute_sampled_metrics: bool = COMPUTE_SAMPLED_METRICS,
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
    skip_metrics : bool, optional
        Whether to skip all quality-metric computation.
    compute_sampled_metrics : bool, optional
        Whether to compute the sampled quality metrics in addition to quick().
    """
    results_path = input_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing benchmark results: {results_path}")

    # Telemetry-only sync check: retain the retro guardrail signal, but do not
    # abort the full analysis when a subset of records is still desynced.
    h5_path = input_dir / "positions.h5"
    if h5_path.exists():
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from validate_benchmark_integrity import validate_sync

        sync_errors = validate_sync(results_path, h5_path)
        desync_count = sum(1 for e in sync_errors if "DESYNC" in e)
        if sync_errors:
            print(
                f"[fidelity] {len(sync_errors)} results/H5 sync issues "
                f"({desync_count} missing positions):",
                file=sys.stderr,
            )
            for err in sync_errors[:20]:
                print(f"  {err}", file=sys.stderr)
            if desync_count > 10:
                telemetry_path = output_dir / "validate_sync_telemetry.json"
                try:
                    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(telemetry_path, "w", encoding="utf-8") as handle:
                        json.dump(
                            {
                                "total_errors": len(sync_errors),
                                "desync_count": desync_count,
                                "sample_errors": sync_errors[:20],
                            },
                            handle,
                            indent=2,
                        )
                except Exception:
                    pass
                print(
                    f"[fidelity] WARNING: {desync_count} engines have results "
                    f"but missing positions. Continuing with best-effort "
                    f"loading; see {telemetry_path} for details.",
                    file=sys.stderr,
                )
            else:
                print(
                    "[fidelity] Minor desync -- proceeding with available data.",
                    file=sys.stderr,
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    previous_hash = previous_results_hash(output_dir / "README.md")
    results_hash = compute_sha256(results_path)
    records = load_results(results_path)
    graph_filter = selected_graph_names(records, max_graphs)
    graph_registry = load_graph_registry()
    groups = build_variant_groups(records, graph_filter)
    h5_file = None
    if h5_path.exists():
        import h5py

        h5_file = h5py.File(str(h5_path), "r")
        load_layout._h5_file = h5_file  # type: ignore[attr-defined]
        print(f"[fidelity] Using HDF5 cache: {h5_path}", file=sys.stderr)
    if skip_metrics:
        load_layout._skip_metrics = True  # type: ignore[attr-defined]
        print("[fidelity] Skipping quality metrics (Procrustes only)", file=sys.stderr)
    else:
        load_layout._skip_metrics = False  # type: ignore[attr-defined]
    load_layout._compute_sampled_metrics = compute_sampled_metrics  # type: ignore[attr-defined]
    if not compute_sampled_metrics:
        print("[fidelity] Sampled metrics disabled (--without-sampled-metrics)", file=sys.stderr)
    if h5_file is None:
        print(
            "[fidelity] No HDF5 cache found. Loading individual .pt files. "
            "Run scripts/consolidate_positions_hdf5.py to speed this up.",
            file=sys.stderr,
        )
    pvalue_buckets = {
        "ks": PValueBucket(),
        "mannwhitney": PValueBucket(),
        "welch": PValueBucket(),
        "procrustes_one_sided": PValueBucket(),
        "procrustes_mannwhitney": PValueBucket(),
        "procrustes_tost_0_5x": PValueBucket(),
        "procrustes_tost_1x": PValueBucket(),
        "procrustes_tost_1_5x": PValueBucket(),
        "procrustes_tost_2x": PValueBucket(),
        "tost_0_5x": PValueBucket(),
        "tost_1x": PValueBucket(),
        "tost_1_5x": PValueBucket(),
        "tost_2x": PValueBucket(),
    }
    per_graph_rows: list[dict[str, Any]] = []
    per_seed_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []

    sorted_groups = sorted(
        groups.items(),
        key=lambda item: (
            algorithm_family(item[0][0]),
            item[0][0],
            item[0][1],
        ),
    )
    variant_by_id = {variant.variant_id: variant for variant in VARIANT_REGISTRY}

    # Build task list for parallel execution
    tasks = []
    for (variant_id, graph_name), grouped_records in sorted_groups:
        variant = variant_by_id.get(variant_id)
        if variant is None or graph_name not in graph_registry:
            continue
        tasks.append((variant, graph_name, grouped_records, graph_registry[graph_name]))

    print(f"[fidelity] Processing {len(tasks)} groups (serial)", file=sys.stderr)

    # No pre-load needed: processing is serial so load_layout reads HDF5 on
    # demand via the _h5_file handle set above (no thread contention).

    def _process_one(args: tuple[Any, Any, Any, Any], row_index: int) -> GroupResult:
        """Process one queued group with the live global row index.

        Parameters
        ----------
        args : tuple[Any, Any, Any, Any]
            Queued ``(variant, graph_name, records, test_graph)`` payload.
        row_index : int
            Index the group will occupy in ``per_graph_rows``.

        Returns
        -------
        GroupResult
            Group output with p-values already registered in the global buckets.
        """
        variant, gname, records, test_graph = args
        local_counter: Counter[str] = Counter()
        return process_group(
            variant=variant,
            graph_name=gname,
            records=records,
            input_dir=input_dir,
            test_graph=test_graph,
            row_index=row_index,
            pvalue_buckets=pvalue_buckets,
            bootstrap_samples=bootstrap_samples,
            load_counter=local_counter,
        )

    # Serial processing -- ThreadPoolExecutor hangs on CPU-bound Procrustes SVDs
    # due to Python GIL starving the main thread (see retro_2026-03-27)
    import time as _time

    _t_start = _time.monotonic()
    for idx, task in enumerate(tasks):
        row_index = len(per_graph_rows)
        group_result = _process_one(task, row_index)
        group_result.row["_rejection_count"] = group_result.rejection_count
        group_result.row["_row_index"] = row_index
        per_graph_rows.append(group_result.row)
        per_seed_rows.extend(group_result.seed_rows)
        pairwise_rows.extend(group_result.pairwise_rows)
        if (idx + 1) % 50 == 0 or idx + 1 == len(tasks):
            elapsed = _time.monotonic() - _t_start
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(tasks) - idx - 1) / rate if rate > 0 else 0
            print(
                f"[fidelity] {idx + 1}/{len(tasks)} groups done "
                f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining, "
                f"{rate:.1f} groups/s)",
                file=sys.stderr,
            )

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
    load_layout._compute_sampled_metrics = not args.without_sampled_metrics  # type: ignore[attr-defined]
    run_analysis(
        input_dir=args.input,
        output_dir=args.output,
        max_graphs=args.max_graphs,
        bootstrap_samples=args.bootstrap_samples,
        power_simulations=args.power_simulations,
        skip_metrics=args.skip_metrics,
        compute_sampled_metrics=not args.without_sampled_metrics,
    )


if __name__ == "__main__":
    main()
