"""Regression tests for the fidelity analysis pipeline helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence

import torch

from scripts.fidelity_analysis import (
    algorithm_summary_fieldnames,
    fidelity_procrustes,
    margin_for_metric,
    per_graph_fieldnames,
)
from scripts.generate_fidelity_report import build_report_tex


def write_csv_fixture(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Write a CSV fixture with a fixed header order.

    Parameters
    ----------
    path : Path
        Destination CSV path.
    fieldnames : Sequence[str]
        Header order.
    rows : Sequence[Mapping[str, object]]
        Rows to serialize.
    """
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def test_fidelity_procrustes_matches_rotation_without_reflection() -> None:
    """A pure rotation+translation should align with near-zero RMSD."""
    positions = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float32)
    shifted = (positions @ rotation) + torch.tensor([4.0, -3.0], dtype=torch.float32)

    rmsd, scale_ratio, reflected, per_node = fidelity_procrustes(positions, shifted)

    assert rmsd < 1e-5
    assert abs(scale_ratio - 1.0) < 1e-5
    assert not reflected
    assert torch.max(per_node).item() < 1e-5


def test_fidelity_procrustes_flags_mirror_match() -> None:
    """A mirrored layout should be flagged rather than silently accepted."""
    positions = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 2.0],
        ],
        dtype=torch.float32,
    )
    mirrored = positions.clone()
    mirrored[:, 0] = -mirrored[:, 0]
    mirrored += torch.tensor([3.0, 1.5], dtype=torch.float32)

    rmsd, _, reflected, _ = fidelity_procrustes(positions, mirrored)

    assert reflected
    assert rmsd > 0.1


def test_margin_for_metric_respects_edge_length_mean_floor() -> None:
    """The edge-length-mean margin should never fall below the documented floor."""
    original_values = torch.tensor([250.0] * 10, dtype=torch.float32).numpy()

    margin = margin_for_metric("edge_length_mean", original_values, factor=0.5)

    assert margin == 2.5


def test_build_report_tex_renders_title_and_sections(tmp_path: Path) -> None:
    """The report generator should produce LaTeX from the CSV contract."""
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "report"
    data_dir.mkdir()

    summary_row = {
        "algorithm_family": "fr",
        "is_stochastic": "True",
        "num_graphs_tested": "1",
        "num_graphs_paired_ok": "1",
        "num_graphs_insufficient_data": "0",
        "num_nan_rejected": "0",
        "procrustes_rmsd_mean": "0.012",
        "procrustes_rmsd_median": "0.012",
        "procrustes_rmsd_max": "0.012",
        "scale_ratio_mean": "1.001",
        "scale_ratio_std": "0.0",
        "num_mirror_matches": "0",
        "mean_runtime_ratio": "0.95",
        "std_runtime_ratio": "0.02",
        "verdict": "strong_equivalent",
        "anomaly_count": "0",
        "anomaly_graphs": "",
        "tost_pass_rate_at_1x": "1.0",
        "tost_pass_rate_at_1_5x": "1.0",
    }
    graph_row = {field: "" for field in per_graph_fieldnames()}
    graph_row.update(
        {
            "algorithm_family": "fr",
            "variant_id": "classic_fr_steps50",
            "graph_name": "karate_club",
            "num_nodes": "34",
            "num_edges": "78",
            "density_bucket": "medium",
            "size_bucket": "small",
            "structure_bucket": "random",
            "structural_note": "none",
            "num_reimpl_seeds": "10",
            "num_orig_seeds": "10",
            "procrustes_rmsd_mean": "0.012",
            "scale_ratio_mean": "1.001",
            "runtime_ratio": "0.95",
            "verdict": "strong_equivalent",
        }
    )
    for metric_name in (
        "aspect_ratio",
        "dag_consistency",
        "edge_length_cv",
        "edge_length_mean",
        "overlap_count",
    ):
        graph_row[f"{metric_name}_orig_mean"] = "1.0"
        graph_row[f"{metric_name}_orig_std"] = "0.1"
        graph_row[f"{metric_name}_reimpl_mean"] = "1.0"
        graph_row[f"{metric_name}_reimpl_std"] = "0.1"
        graph_row[f"{metric_name}_cohens_d"] = "0.0"
        graph_row[f"{metric_name}_tost_pvalue_0_5x_bh"] = "0.01"
        graph_row[f"{metric_name}_tost_pvalue_1x_bh"] = "0.01"
        graph_row[f"{metric_name}_tost_pvalue_1_5x_bh"] = "0.01"
        graph_row[f"{metric_name}_tost_pvalue_2x_bh"] = "0.01"

    write_csv_fixture(
        data_dir / "algorithm_summary.csv", algorithm_summary_fieldnames(), [summary_row]
    )
    write_csv_fixture(data_dir / "per_graph_detail.csv", per_graph_fieldnames(), [graph_row])
    (data_dir / "README.md").write_text(
        "\n".join(
            [
                "# Fidelity Analysis Data",
                "",
                "Results SHA-256: `deadbeef`",
                "",
                (
                    "- Estimated Mann-Whitney minimum detectable effect at "
                    "n=10/side and 80% power: `d >= 1.10`"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    tex_path = build_report_tex(data_dir, output_dir)
    tex_source = tex_path.read_text(encoding="utf-8")

    assert tex_path.exists()
    assert "Dagua Algorithm Reimplementation Fidelity Report" in tex_source
    assert "\\section{Executive summary}" in tex_source
    assert "\\section{Methodology}" in tex_source
