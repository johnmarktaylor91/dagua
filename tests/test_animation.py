"""Tests for optimization animation export."""

from pathlib import Path

import pytest

import dagua
from dagua import AnimationConfig, DaguaGraph, LayoutConfig


def _animated_graph():
    g = DaguaGraph.from_edge_list(
        [
            ("input", "stem"),
            ("stem", "branch_a"),
            ("stem", "branch_b"),
            ("branch_a", "merge"),
            ("branch_b", "merge"),
            ("merge", "head"),
        ]
    )
    g.edge_labels = ["", "left", "right", "", "", "out"]
    g.add_cluster("block", ["stem", "branch_a", "branch_b", "merge"], label="Block")
    return g


class TestAnimationExport:
    @pytest.mark.slow
    def test_animate_gif_exports_file(self, tmp_path):
        g = _animated_graph()
        out = tmp_path / "opt.gif"
        result = dagua.animate(
            g,
            LayoutConfig(steps=12, edge_opt_steps=6, seed=42),
            output=str(out),
            animation_config=AnimationConfig(
                fps=10,
                max_layout_frames=6,
                max_edge_frames=4,
                hold_start_frames=1,
                hold_end_frames=1,
                transition_frames=1,
            ),
        )

        assert out.exists()
        assert out.stat().st_size > 0
        assert result.output == str(out)
        assert result.frame_count > 0
        assert result.layout_snapshots >= 2
        assert result.edge_snapshots >= 1

    @pytest.mark.slow
    def test_animate_respects_focus_camera(self, tmp_path):
        g = _animated_graph()
        out = tmp_path / "focus.webp"
        result = dagua.animate(
            g,
            LayoutConfig(steps=10, edge_opt_steps=-1, seed=42),
            output=str(out),
            animation_config=AnimationConfig(
                format="webp",
                camera="focus",
                center_on="merge",
                max_layout_frames=5,
                hold_start_frames=1,
                hold_end_frames=1,
                transition_frames=0,
            ),
        )

        assert Path(result.output).exists()
        assert Path(result.output).suffix == ".webp"

    @pytest.mark.slow
    def test_animate_does_not_dump_frames_by_default(self, tmp_path):
        g = _animated_graph()
        out = tmp_path / "clean.gif"
        frames_dir = tmp_path / "frames"
        dagua.animate(
            g,
            LayoutConfig(steps=8, edge_opt_steps=-1, seed=42),
            output=str(out),
            animation_config=AnimationConfig(
                max_layout_frames=4,
                hold_start_frames=1,
                hold_end_frames=1,
                transition_frames=0,
            ),
        )

        assert out.exists()
        assert not frames_dir.exists()
