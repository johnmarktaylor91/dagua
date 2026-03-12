from pathlib import Path

import pytest
import torch

from dagua.eval.benchmark import (
    get_rare_suite_graphs,
    get_standard_suite_graphs,
    merge_latest_results,
)
from dagua.eval.report import generate_report


@pytest.mark.smoke
def test_standard_suite_contains_expected_cases():
    suite = get_standard_suite_graphs()
    names = {bg.test_graph.name for bg in suite}

    assert "chain_100" in names
    assert "binary_tree_127" in names
    assert "tl_cnn_small" in names
    assert "tl_resnet_2block" in names
    assert "tl_transformer_1layer" in names
    assert "scale_100k" in names
    assert len(suite) >= 15


@pytest.mark.smoke
def test_rare_suite_sizes_present():
    suite = get_rare_suite_graphs()
    names = {bg.test_graph.name for bg in suite}
    assert {"scale_500000", "scale_1000000", "scale_5000000", "scale_10000000", "scale_50000000"} <= names


@pytest.mark.smoke
def test_merge_latest_results_and_generate_report(tmp_path):
    output_dir = tmp_path / "eval_output"
    standard_run = output_dir / "benchmark_db" / "standard" / "2026-03-12T00:00:00+00:00"
    rare_run = output_dir / "benchmark_db" / "rare" / "2026-03-12T01:00:00+00:00"
    standard_positions = standard_run / "positions"
    rare_positions = rare_run / "positions"
    standard_positions.mkdir(parents=True, exist_ok=True)
    rare_positions.mkdir(parents=True, exist_ok=True)

    standard_suite = {bg.test_graph.name: bg.test_graph for bg in get_standard_suite_graphs()}
    residual = standard_suite["residual_block"]
    residual.graph.compute_node_sizes()
    pos = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 60.0],
            [0.0, 120.0],
            [0.0, 180.0],
            [0.0, 240.0],
            [-80.0, 240.0],
            [-80.0, 300.0],
            [0.0, 300.0],
            [0.0, 360.0],
            [0.0, 420.0],
        ],
        dtype=torch.float32,
    )
    torch.save(pos, standard_positions / "residual_block__dagua.pt")

    standard_payload = {
        "run_id": "2026-03-12T00:00:00+00:00",
        "suite": "standard",
        "system": {"python": "3.11"},
        "graphs": {
            "residual_block": {
                "n_nodes": residual.graph.num_nodes,
                "n_edges": int(residual.graph.edge_index.shape[1]),
                "structural_category": "residual",
                "description": residual.description,
                "expected_challenges": residual.expected_challenges,
                "tags": sorted(residual.tags),
                "source": residual.source,
                "visualize": True,
                "scale_tier": None,
                "competitors": {
                    "dagua": {
                        "status": "OK",
                        "runtime_seconds": 0.12,
                        "metrics": {
                            "dag_consistency": 1.0,
                            "overall_quality": 90.0,
                            "edge_crossings": 0,
                            "node_overlaps": 0,
                            "edge_length_cv": 0.1,
                        },
                        "composite_score": 91.5,
                        "metrics_computed": ["tier1", "tier2", "tier3"],
                        "metrics_skipped": [],
                        "positions_path": "positions/residual_block__dagua.pt",
                    },
                    "graphviz_dot": {
                        "status": "SKIPPED",
                        "reason": "not installed",
                        "runtime_seconds": None,
                        "metrics": {},
                        "composite_score": None,
                        "metrics_computed": [],
                        "metrics_skipped": ["tier1", "tier2", "tier3"],
                        "positions_path": None,
                    },
                },
            }
        },
    }

    rare_payload = {
        "run_id": "2026-03-12T01:00:00+00:00",
        "suite": "rare",
        "system": {"python": "3.11"},
        "graphs": {
            "scale_500000": {
                "n_nodes": 500_000,
                "n_edges": 750_000,
                "structural_category": "scale",
                "description": "Rare scale graph",
                "expected_challenges": "Scale",
                "tags": ["large-sparse"],
                "source": "synthetic",
                "visualize": False,
                "scale_tier": "rare",
                "competitors": {
                    "dagua": {
                        "status": "OK",
                        "runtime_seconds": 12.0,
                        "metrics": {"dag_consistency": 1.0, "overall_quality": 70.0},
                        "composite_score": 74.0,
                        "metrics_computed": ["tier1"],
                        "metrics_skipped": ["tier2", "tier3"],
                        "positions_path": None,
                    }
                },
            }
        },
    }

    (standard_run / "results.json").write_text(__import__("json").dumps(standard_payload), encoding="utf-8")
    (rare_run / "results.json").write_text(__import__("json").dumps(rare_payload), encoding="utf-8")
    (standard_run.parent / "latest").symlink_to(standard_run.name)
    (rare_run.parent / "latest").symlink_to(rare_run.name)

    combined = merge_latest_results(str(output_dir))
    assert "residual_block" in combined["graphs"]
    assert "scale_500000" in combined["graphs"]

    artifacts = generate_report(output_dir=str(output_dir), combined_results=combined, compile_pdf=False)
    assert Path(artifacts["tex"]).exists()
    assert Path(artifacts["scaling_curve"]).exists()
    assert (output_dir / "visuals" / "comparisons" / "residual_block_comparison.png").exists()
    assert (output_dir / "report" / "prose_prompt.md").exists()
    assert (output_dir / "report" / "review_round_1.json").exists()
