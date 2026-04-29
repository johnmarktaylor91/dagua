#!/usr/bin/env python3
"""Cross-compare dagua reimplementations against Graphviz benchmark layouts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dagua.eval.pipeline_io import load_position_tensor, validate_positions  # noqa: E402

QUALITY_METRICS: tuple[str, ...] = (
    "aspect_ratio",
    "dag_consistency",
    "edge_length_cv",
    "edge_straightness_mean_deg",
    "depth_spearman_rho",
    "overlap_count",
)
RMSD_ALERT_THRESHOLD = 0.15


@dataclass(frozen=True)
class Pairing:
    """One dagua-vs-target engine pairing.

    Parameters
    ----------
    dagua_engine : str
        Dagua reimplementation engine name recorded in ``results.json``.
    target_engine : str
        Target engine name recorded in ``results.json``.
    family_label : str
        Stable family label used in summary output.
    priority : str
        Triage priority label, currently ``"P0"`` or ``"P1"``.
    """

    dagua_engine: str
    target_engine: str
    family_label: str
    priority: str


@dataclass(frozen=True)
class ResultRecord:
    """Minimal benchmark result record used by the cross-comparator.

    Parameters
    ----------
    key : str
        Stable key from ``results.json``.
    graph : str
        Benchmark graph name.
    engine : str
        Engine name.
    seed : int | None
        Layout seed, when recorded.
    status : str
        Benchmark status.
    positions_file : str | None
        Benchmark-relative position tensor path.
    n_nodes : int
        Number of graph nodes.
    metrics : dict[str, float | None]
        Quality metrics copied from ``results.json`` when present.
    """

    key: str
    graph: str
    engine: str
    seed: Optional[int]
    status: str
    positions_file: Optional[str]
    n_nodes: int
    metrics: dict[str, Optional[float]]


PAIRINGS: tuple[Pairing, ...] = (
    Pairing("classic_sugiyama", "graphviz_dot", "dot", "P0"),
    Pairing("classic_stress_maj", "graphviz_neato", "neato_stress", "P0"),
    Pairing("classic_classical_mds", "graphviz_neato", "neato_mds", "P0"),
    Pairing("classic_fmmm", "graphviz_fdp", "fdp", "P0"),
    Pairing("classic_sfdp", "graphviz_sfdp", "sfdp", "P0"),
    Pairing("classic_fr", "graphviz_neato", "neato_fr_proxy", "P1"),
    Pairing("classic_kk", "graphviz_neato", "neato_kk_proxy", "P1"),
)


def finite_float(value: Any) -> Optional[float]:
    """Convert a JSON scalar to a finite float when possible.

    Parameters
    ----------
    value : Any
        Candidate scalar value from ``results.json``.

    Returns
    -------
    float | None
        Finite float value, or ``None`` when missing/non-finite.
    """
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(converted):
        return None
    return converted


def load_results(input_dir: Path) -> dict[str, list[ResultRecord]]:
    """Load and index benchmark results by ``graph::engine``.

    Parameters
    ----------
    input_dir : Path
        Benchmark root containing ``results.json``.

    Returns
    -------
    dict[str, list[ResultRecord]]
        Records keyed by ``"{graph}::{engine}"``.
    """
    results_path = input_dir / "results.json"
    raw = json.loads(results_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{results_path} must contain a JSON object")

    indexed: dict[str, list[ResultRecord]] = {}
    for key, payload in raw.items():
        if not isinstance(payload, Mapping):
            continue
        graph = str(payload.get("graph_name", ""))
        engine = str(payload.get("engine_name", ""))
        if not graph or not engine:
            continue
        metrics = {metric: finite_float(payload.get(metric)) for metric in QUALITY_METRICS}
        record = ResultRecord(
            key=str(key),
            graph=graph,
            engine=engine,
            seed=payload.get("seed") if isinstance(payload.get("seed"), int) else None,
            status=str(payload.get("status", "")),
            positions_file=str(payload["positions_file"])
            if payload.get("positions_file") is not None
            else None,
            n_nodes=int(payload.get("num_nodes") or 0),
            metrics=metrics,
        )
        indexed.setdefault(f"{graph}::{engine}", []).append(record)
    for records in indexed.values():
        records.sort(key=record_sort_key)
    return indexed


def record_sort_key(record: ResultRecord) -> tuple[int, int, str]:
    """Return a deterministic preference key for duplicate engine records.

    Parameters
    ----------
    record : ResultRecord
        Candidate benchmark record.

    Returns
    -------
    tuple[int, int, str]
        Sort key preferring successful runs, then lower seed, then key.
    """
    status_rank = 0 if record.status == "ok" else 1
    seed_rank = -1 if record.seed is None else record.seed
    return status_rank, seed_rank, record.key


def select_ok_record(
    indexed: Mapping[str, list[ResultRecord]],
    graph: str,
    engine: str,
) -> Optional[ResultRecord]:
    """Select the canonical successful result for a graph/engine.

    Parameters
    ----------
    indexed : Mapping[str, list[ResultRecord]]
        Records returned by :func:`load_results`.
    graph : str
        Benchmark graph name.
    engine : str
        Engine name.

    Returns
    -------
    ResultRecord | None
        First successful record after deterministic sorting, if available.
    """
    for record in indexed.get(f"{graph}::{engine}", []):
        if record.status == "ok":
            return record
    return None


def shared_graphs(indexed: Mapping[str, list[ResultRecord]], pairing: Pairing) -> list[str]:
    """Return graph names with successful records on both sides.

    Parameters
    ----------
    indexed : Mapping[str, list[ResultRecord]]
        Records returned by :func:`load_results`.
    pairing : Pairing
        Engine pairing to evaluate.

    Returns
    -------
    list[str]
        Sorted graph names with successful dagua and target layouts.
    """
    graphs = {
        key.rsplit("::", 1)[0] for key in indexed if key.endswith(f"::{pairing.dagua_engine}")
    }
    target_graphs = {
        key.rsplit("::", 1)[0] for key in indexed if key.endswith(f"::{pairing.target_engine}")
    }
    return sorted(
        graph
        for graph in graphs & target_graphs
        if select_ok_record(indexed, graph, pairing.dagua_engine)
        and select_ok_record(indexed, graph, pairing.target_engine)
    )


def fidelity_procrustes(
    pos_a: torch.Tensor,
    pos_b: torch.Tensor,
) -> tuple[float, torch.Tensor]:
    """Align two layouts with scale normalization and best-of-two rotations.

    Parameters
    ----------
    pos_a : torch.Tensor
        First position tensor with shape ``[N, 2]``.
    pos_b : torch.Tensor
        Target position tensor with shape ``[N, 2]``.

    Returns
    -------
    tuple[float, torch.Tensor]
        RMSD after scale-normalized alignment and per-node displacements.
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
        return reflected_rmsd, reflected_per_node
    return rmsd, per_node


def load_valid_positions(
    record: ResultRecord,
    input_dir: Path,
) -> tuple[Optional[torch.Tensor], str]:
    """Load and validate one position tensor.

    Parameters
    ----------
    record : ResultRecord
        Benchmark record with a position path.
    input_dir : Path
        Benchmark root directory.

    Returns
    -------
    tuple[torch.Tensor | None, str]
        Loaded tensor and empty error on success, otherwise ``None`` and a
        canonical rejection reason.
    """
    positions, error = load_position_tensor(
        record_key=record.key,
        positions_file=record.positions_file,
        input_dir=input_dir,
    )
    if error is not None:
        return None, error
    if positions is None:
        return None, "missing_positions"
    validation_error = validate_positions(positions, record.n_nodes)
    if validation_error is not None:
        return None, validation_error
    return positions, ""


def percentile(values: Sequence[float], quantile: float) -> Optional[float]:
    """Compute a linear-interpolated percentile.

    Parameters
    ----------
    values : Sequence[float]
        Numeric values.
    quantile : float
        Percentile in ``[0, 1]``.

    Returns
    -------
    float | None
        Percentile value, or ``None`` for an empty input.
    """
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def build_pairwise_rows(
    indexed: Mapping[str, list[ResultRecord]],
    input_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build RMSD and metric-delta rows for all configured pairings.

    Parameters
    ----------
    indexed : Mapping[str, list[ResultRecord]]
        Records returned by :func:`load_results`.
    input_dir : Path
        Benchmark root directory.

    Returns
    -------
    tuple[list[dict[str, Any]], list[dict[str, Any]]]
        Pairwise RMSD CSV rows and quality-delta CSV rows.
    """
    rmsd_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    for pairing in PAIRINGS:
        for graph in shared_graphs(indexed, pairing):
            dagua_record = select_ok_record(indexed, graph, pairing.dagua_engine)
            target_record = select_ok_record(indexed, graph, pairing.target_engine)
            if dagua_record is None or target_record is None:
                continue
            dagua_positions, dagua_error = load_valid_positions(dagua_record, input_dir)
            target_positions, target_error = load_valid_positions(target_record, input_dir)
            error = dagua_error or target_error
            rmsd: Optional[float] = None
            n_aligned = 0
            if not error and dagua_positions is not None and target_positions is not None:
                if dagua_positions.shape[0] != target_positions.shape[0]:
                    error = "node_count_mismatch_between_engines"
                else:
                    rmsd, _ = fidelity_procrustes(dagua_positions, target_positions)
                    n_aligned = int(dagua_positions.shape[0])

            rmsd_rows.append(
                {
                    "graph": graph,
                    "dagua_engine": pairing.dagua_engine,
                    "target_engine": pairing.target_engine,
                    "family_label": pairing.family_label,
                    "priority": pairing.priority,
                    "n_nodes": dagua_record.n_nodes,
                    "rmsd": rmsd,
                    "n_aligned": n_aligned,
                    "error": error,
                }
            )
            delta_rows.extend(build_delta_rows(graph, pairing, dagua_record, target_record))
    return rmsd_rows, delta_rows


def build_delta_rows(
    graph: str,
    pairing: Pairing,
    dagua_record: ResultRecord,
    target_record: ResultRecord,
) -> list[dict[str, Any]]:
    """Build metric-delta rows for one graph pairing.

    Parameters
    ----------
    graph : str
        Benchmark graph name.
    pairing : Pairing
        Engine pairing being compared.
    dagua_record : ResultRecord
        Dagua-side benchmark record.
    target_record : ResultRecord
        Target-side benchmark record.

    Returns
    -------
    list[dict[str, Any]]
        One row per configured quality metric.
    """
    rows: list[dict[str, Any]] = []
    for metric in QUALITY_METRICS:
        dagua_value = dagua_record.metrics.get(metric)
        target_value = target_record.metrics.get(metric)
        abs_delta: Optional[float] = None
        rel_delta: Optional[float] = None
        if dagua_value is not None and target_value is not None:
            abs_delta = abs(dagua_value - target_value)
            denominator = abs(target_value)
            rel_delta = abs_delta / denominator if denominator > 1e-12 else None
        rows.append(
            {
                "graph": graph,
                "dagua_engine": pairing.dagua_engine,
                "target_engine": pairing.target_engine,
                "family_label": pairing.family_label,
                "priority": pairing.priority,
                "metric": metric,
                "dagua_value": dagua_value,
                "target_value": target_value,
                "abs_delta": abs_delta,
                "rel_delta": rel_delta,
            }
        )
    return rows


def summarize_pairwise_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Summarize RMSD rows by family pairing.

    Parameters
    ----------
    rows : Sequence[Mapping[str, Any]]
        Pairwise RMSD rows.

    Returns
    -------
    dict[str, dict[str, Any]]
        Summary keyed by family label.
    """
    summary: dict[str, dict[str, Any]] = {}
    for pairing in PAIRINGS:
        family_rows = [
            row
            for row in rows
            if row["dagua_engine"] == pairing.dagua_engine
            and row["target_engine"] == pairing.target_engine
        ]
        rmsd_values = [
            float(row["rmsd"])
            for row in family_rows
            if row.get("rmsd") is not None and math.isfinite(float(row["rmsd"]))
        ]
        worst_row = max(
            (row for row in family_rows if row.get("rmsd") is not None),
            key=lambda row: float(row["rmsd"]),
            default=None,
        )
        summary[pairing.family_label] = {
            "family_label": pairing.family_label,
            "priority": pairing.priority,
            "dagua_engine": pairing.dagua_engine,
            "target_engine": pairing.target_engine,
            "graph_count": len(rmsd_values),
            "median_rmsd": statistics.median(rmsd_values) if rmsd_values else None,
            "p25_rmsd": percentile(rmsd_values, 0.25),
            "p75_rmsd": percentile(rmsd_values, 0.75),
            "p95_rmsd": percentile(rmsd_values, 0.95),
            "worst_rmsd": float(worst_row["rmsd"]) if worst_row is not None else None,
            "worst_graph": str(worst_row["graph"]) if worst_row is not None else None,
            "rmsd_gt_0_05_count": sum(value > 0.05 for value in rmsd_values),
            "rmsd_gt_0_15_count": sum(value > RMSD_ALERT_THRESHOLD for value in rmsd_values),
        }
    return summary


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    """Write rows to a CSV file with stable columns.

    Parameters
    ----------
    path : Path
        Destination path.
    rows : Sequence[Mapping[str, Any]]
        Row mappings.
    columns : Sequence[str]
        CSV column order.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    output_dir: Path,
    rmsd_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
) -> None:
    """Write all comparator output files.

    Parameters
    ----------
    output_dir : Path
        Round output root.
    rmsd_rows : list[dict[str, Any]]
        Pairwise RMSD rows.
    delta_rows : list[dict[str, Any]]
        Quality metric delta rows.
    """
    data_dir = output_dir / "data"
    write_csv(
        data_dir / "pairwise_rmsd.csv",
        rmsd_rows,
        ("graph", "dagua_engine", "target_engine", "n_nodes", "rmsd", "n_aligned", "error"),
    )
    write_csv(
        data_dir / "quality_deltas.csv",
        delta_rows,
        (
            "graph",
            "dagua_engine",
            "target_engine",
            "metric",
            "dagua_value",
            "target_value",
            "abs_delta",
            "rel_delta",
        ),
    )
    summary = summarize_pairwise_rows(rmsd_rows)
    (data_dir / "per_family_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Parameters
    ----------
    argv : Sequence[str] | None
        Optional argument vector for tests.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("eval_output/benchmark_full"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_output/algo_fidelity/round_1"),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the cross-comparator CLI.

    Parameters
    ----------
    argv : Sequence[str] | None
        Optional argument vector for tests.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    indexed = load_results(args.input_dir)
    rmsd_rows, delta_rows = build_pairwise_rows(indexed, args.input_dir)
    write_outputs(args.output_dir, rmsd_rows, delta_rows)
    print(f"Wrote {len(rmsd_rows)} RMSD rows and {len(delta_rows)} delta rows to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
