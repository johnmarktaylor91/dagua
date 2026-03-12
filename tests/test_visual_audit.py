"""Smoke coverage for the visual audit suite."""

from __future__ import annotations

from pathlib import Path

from dagua.eval.visual_audit import build_visual_audit_suite, build_visual_review_session


def test_visual_audit_suite_rebuilds(tmp_path: Path):
    result = build_visual_audit_suite(
        output_dir=str(tmp_path / "visual_audit"),
        steps=18,
        edge_opt_steps=-1,
        graph_names=["linear_3layer_mlp", "residual_block", "transformer_layer"],
    )

    out = Path(result.output_dir)
    assert out.exists()
    assert Path(result.manifest_path).exists()
    assert Path(result.readme_path).exists()
    assert (out / "complexity_ladder" / "linear_3layer_mlp_ladder.png").exists()
    assert (out / "decomposition" / "linear_3layer_mlp_decomposition.png").exists()
    assert (out / "kill_switches" / "linear_3layer_mlp_kill_switches.png").exists()
    assert (out / "diff_dashboard" / "linear_3layer_mlp_diff.png").exists()
    assert (out / "competitor_stepwise" / "linear_3layer_mlp_competitors.png").exists()
    assert (out / "sheets" / "typography_stress.png").exists()
    assert (out / "sheets" / "edge_language_sheet.png").exists()
    assert (out / "metric_cards" / "README.md").exists()
    assert (out / "frozen_baselines" / "current" / "linear_3layer_mlp_ladder.png").exists()


def test_visual_audit_supports_fast_partial_rebuilds(tmp_path: Path):
    result = build_visual_audit_suite(
        output_dir=str(tmp_path / "visual_audit_partial"),
        steps=10,
        edge_opt_steps=-1,
        graph_names=["linear_3layer_mlp"],
        panels=["ladder", "manifest"],
        compare_to_baseline=None,
    )

    out = Path(result.output_dir)
    assert (out / "complexity_ladder" / "linear_3layer_mlp_ladder.png").exists()
    assert Path(result.manifest_path).exists()
    assert not (out / "decomposition" / "linear_3layer_mlp_decomposition.png").exists()
    assert not (out / "sheets" / "typography_stress.png").exists()


def test_visual_review_session_builds_numbered_sequence(tmp_path: Path):
    result = build_visual_review_session(
        output_dir=str(tmp_path / "visual_session"),
        steps=10,
        edge_opt_steps=-1,
        graph_names=["linear_3layer_mlp", "residual_block"],
    )

    out = Path(result.output_dir)
    assert out.exists()
    assert Path(result.manifest_path).exists()
    assert Path(result.readme_path).exists()
    assert Path(result.notes_path).exists()
    assert (out / "01_linear_3layer_mlp.png").exists()
    assert (out / "02_residual_block.png").exists()
