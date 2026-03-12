"""Tests for optimization animation export."""

from pathlib import Path

import pytest

import dagua
from dagua import AnimationConfig, CameraKeyframe, DaguaGraph, LayoutConfig, TourConfig


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


class TestTourExport:
    @pytest.mark.slow
    def test_tour_auto_scene_exports_mp4(self, tmp_path):
        g = _animated_graph()
        pos = dagua.layout(g, LayoutConfig(steps=12, edge_opt_steps=-1, seed=42))
        out = tmp_path / "tour.mp4"
        result = dagua.tour(
            g,
            positions=pos,
            output=str(out),
            tour_config=TourConfig(
                fps=12,
                scene="auto",
                hold_start_frames=1,
                hold_end_frames=1,
            ),
        )

        assert out.exists()
        assert out.stat().st_size > 0
        assert result.output == str(out)
        assert result.frame_count > 10

    @pytest.mark.slow
    def test_tour_keyframes_support_custom_sweep(self, tmp_path):
        g = _animated_graph()
        pos = dagua.layout(g, LayoutConfig(steps=10, edge_opt_steps=-1, seed=42))
        out = tmp_path / "tour.gif"
        result = dagua.tour(
            g,
            positions=pos,
            output=str(out),
            tour_config=TourConfig(
                format="gif",
                scene="keyframes",
                hold_start_frames=1,
                hold_end_frames=1,
                keyframes=[
                    CameraKeyframe(duration_frames=8, bounds=(-120, 120, -40, 120), title="Start"),
                    CameraKeyframe(duration_frames=10, center_on="merge", scale=0.7, easing="ease_in", title="Merge"),
                ],
            ),
        )

        assert out.exists()
        assert result.format == "gif"

    @pytest.mark.slow
    def test_tour_zoom_pan_scene_exports(self, tmp_path):
        g = _animated_graph()
        pos = dagua.layout(g, LayoutConfig(steps=10, edge_opt_steps=-1, seed=42))
        out = tmp_path / "zoom-pan.webp"
        result = dagua.tour(
            g,
            positions=pos,
            output=str(out),
            tour_config=TourConfig(
                format="webp",
                scene="zoom_pan",
                hold_start_frames=1,
                hold_end_frames=1,
            ),
        )

        assert out.exists()
        assert result.frame_count > 12

    @pytest.mark.slow
    def test_tour_cathedral_scene_exports(self, tmp_path):
        g = _animated_graph()
        pos = dagua.layout(g, LayoutConfig(steps=10, edge_opt_steps=-1, seed=42))
        out = tmp_path / "cathedral.gif"
        result = dagua.tour(
            g,
            positions=pos,
            output=str(out),
            tour_config=TourConfig(
                format="gif",
                scene="cathedral",
                hold_start_frames=1,
                hold_end_frames=1,
            ),
        )

        assert out.exists()
        assert result.frame_count > 12

    @pytest.mark.slow
    def test_tour_large_lod_mode_exports(self, tmp_path):
        edges = [(f"n{i}", f"n{i+1}") for i in range(799)]
        g = DaguaGraph.from_edge_list(edges)
        pos = dagua.layout(g, LayoutConfig(steps=12, edge_opt_steps=-1, seed=42))
        out = tmp_path / "lod-tour.gif"
        result = dagua.tour(
            g,
            positions=pos,
            output=str(out),
            tour_config=TourConfig(
                format="gif",
                scene="zoom_pan",
                lod_threshold=100,
                detail_node_limit=120,
                edge_sample_limit=500,
                hold_start_frames=1,
                hold_end_frames=1,
            ),
        )

        assert out.exists()
        assert result.frame_count > 12
