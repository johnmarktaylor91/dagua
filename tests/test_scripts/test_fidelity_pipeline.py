"""Regression tests for the fidelity analysis pipeline helpers."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import pytest
import torch
from pytest import MonkeyPatch

from dagua.eval.variants import VARIANT_REGISTRY, original_variant_name
from scripts import fidelity_analysis
from scripts.consolidate_positions_hdf5 import write_hdf5_atomic
from scripts.fidelity_add_metrics import reconstruct_result_key
from scripts.fidelity_analysis import (
    GroupResult,
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
    """A mirrored layout should use the reflected fit and still be flagged."""
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
    assert rmsd < 1e-5


def test_margin_for_metric_respects_edge_length_cv_floor() -> None:
    """The edge-length-CV margin should never fall below the documented floor."""
    original_values = torch.tensor([0.2] * 10, dtype=torch.float32).numpy()

    margin = margin_for_metric("edge_length_cv", original_values, factor=0.5)

    assert margin == 0.05


def test_per_graph_fieldnames_include_within_vs_between_columns() -> None:
    """The per-graph CSV contract should expose within-vs-between RMSD fields."""
    fieldnames = per_graph_fieldnames()

    assert "within_vs_between_pvalue" in fieldnames
    assert "within_rmsd_mean" in fieldnames
    assert "between_rmsd_mean" in fieldnames
    assert "rmsd_ratio" in fieldnames


def test_reconstruct_result_key_uses_original_variant_name_for_orig_side() -> None:
    """Original-side per-seed rows should target the synthetic original engine key."""
    variant = next(
        candidate for candidate in VARIANT_REGISTRY if original_variant_name(candidate) is not None
    )

    result_key = reconstruct_result_key(
        {
            "graph_name": "chain_5",
            "variant_id": variant.variant_id,
            "side": "orig",
            "seed": "7",
        }
    )

    assert result_key == f"chain_5::{original_variant_name(variant)}::seed7"


def test_write_hdf5_atomic_preserves_existing_file_on_rename_failure(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Rename failure should leave the previous HDF5 cache untouched."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_path = tmp_path / "positions.h5"
    tensor_path = input_dir / "positions" / "graph.pt"
    tensor_path.parent.mkdir()
    torch.save(torch.zeros((2, 2), dtype=torch.float32), tensor_path)

    with h5py.File(output_path, "w") as handle:
        handle.create_dataset("existing", data=[[1.0, 2.0]])

    ok_records = [
        ("graph::engine::seed42", {"positions_file": "positions/graph.pt"}),
    ]

    def _failing_rename(src: object, dst: object) -> None:
        """Raise during the final atomic rename step."""
        del src, dst
        raise RuntimeError("rename failed")

    monkeypatch.setattr("scripts.consolidate_positions_hdf5.os.rename", _failing_rename)

    with pytest.raises(RuntimeError, match="rename failed"):
        write_hdf5_atomic(output_path, ok_records, input_dir, time.perf_counter())

    with h5py.File(output_path, "r") as handle:
        assert list(handle.keys()) == ["existing"]
    assert output_path.with_suffix(".h5.tmp").exists()


def test_run_analysis_passes_global_buckets_and_row_indexes(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Serial analysis should reuse the global BH buckets with monotonic row indexes."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "results.json").write_text("{}", encoding="utf-8")

    variant = next(
        candidate for candidate in VARIANT_REGISTRY if original_variant_name(candidate) is not None
    )
    fake_groups = {
        (variant.variant_id, "graph_a"): {"orig": [], "reimpl": []},
        (variant.variant_id, "graph_b"): {"orig": [], "reimpl": []},
    }
    row_indexes: list[int] = []
    bucket_ids: list[int] = []

    def _fake_process_group(**kwargs: object) -> GroupResult:
        """Capture the shared p-value bucket object and assigned row index."""
        row_indexes.append(int(kwargs["row_index"]))
        bucket_ids.append(id(kwargs["pvalue_buckets"]))
        return GroupResult(
            row={
                "algorithm_family": "family",
                "variant_id": variant.variant_id,
                "graph_name": str(kwargs["graph_name"]),
                "verdict": "insufficient_data",
                "anomaly_reason": "",
                "_variant_is_stochastic": True,
                "_structural_note_flag": False,
            },
            seed_rows=[],
            pairwise_rows=[],
            rejection_count=0,
        )

    captured_csv_rows: dict[str, list[dict[str, object]]] = {}

    def _capture_write_csv(
        path: Path,
        rows: Sequence[Mapping[str, object]],
        fieldnames: Sequence[str],
    ) -> None:
        """Record CSV writes without touching disk."""
        del fieldnames
        captured_csv_rows[path.name] = [dict(row) for row in rows]

    monkeypatch.setattr(fidelity_analysis, "compute_sha256", lambda path: "hash")
    monkeypatch.setattr(fidelity_analysis, "previous_results_hash", lambda path: None)
    monkeypatch.setattr(fidelity_analysis, "load_results", lambda path: {})
    monkeypatch.setattr(
        fidelity_analysis,
        "selected_graph_names",
        lambda records, max_graphs: None,
    )
    monkeypatch.setattr(
        fidelity_analysis,
        "load_graph_registry",
        lambda: {"graph_a": object(), "graph_b": object()},
    )
    monkeypatch.setattr(
        fidelity_analysis,
        "build_variant_groups",
        lambda records, graph_filter: fake_groups,
    )
    monkeypatch.setattr(fidelity_analysis, "process_group", _fake_process_group)
    monkeypatch.setattr(fidelity_analysis, "apply_bh_correction", lambda rows, buckets: None)
    monkeypatch.setattr(fidelity_analysis, "finalize_group_row", lambda row: None)
    monkeypatch.setattr(fidelity_analysis, "family_summary_rows", lambda rows: [])
    monkeypatch.setattr(
        fidelity_analysis,
        "estimate_mw_min_detectable_effect",
        lambda power_simulations: 1.0,
    )
    monkeypatch.setattr(fidelity_analysis, "write_csv", _capture_write_csv)
    monkeypatch.setattr(
        fidelity_analysis,
        "write_readme",
        lambda path, results_hash, previous_hash, power_effect: None,
    )

    fidelity_analysis.run_analysis(
        input_dir=input_dir,
        output_dir=output_dir,
        max_graphs=None,
        bootstrap_samples=10,
        power_simulations=10,
    )

    assert row_indexes == [0, 1]
    assert len(set(bucket_ids)) == 1
    assert len(captured_csv_rows["per_graph_detail.csv"]) == 2


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
    for metric_name in ("aspect_ratio", "dag_consistency", "edge_length_cv"):
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
