"""Tests for matplotlib renderer."""

from pathlib import Path

import pytest

import dagua
from dagua.config import LayoutConfig
from dagua.graph import DaguaGraph
from dagua.layout import layout
from dagua.render import render
from dagua.styles import EdgeStyle, NodeStyle


class TestRenderBasic:
    @pytest.mark.slow
    def test_returns_fig_ax(self, simple_chain, fast_config):
        pos = layout(simple_chain, fast_config)
        fig, ax = render(simple_chain, pos)
        assert fig is not None
        assert ax is not None

    @pytest.mark.slow
    def test_empty_graph(self, empty_graph, fast_config):
        pos = layout(empty_graph, fast_config)
        fig, ax = render(empty_graph, pos)
        assert fig is not None

    @pytest.mark.slow
    def test_save_png(self, simple_chain, fast_config, tmp_path):
        pos = layout(simple_chain, fast_config)
        out = str(tmp_path / "test.png")
        render(simple_chain, pos, output=out)
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0

    @pytest.mark.slow
    def test_save_svg(self, simple_chain, fast_config, tmp_path):
        pos = layout(simple_chain, fast_config)
        out = str(tmp_path / "test.svg")
        render(simple_chain, pos, output=out)
        assert Path(out).exists()
        content = Path(out).read_text(encoding="utf-8")
        assert "<title>a</title>" in content or "<title>b</title>" in content

    @pytest.mark.slow
    def test_save_pdf(self, simple_chain, fast_config, tmp_path):
        pos = layout(simple_chain, fast_config)
        out = str(tmp_path / "test.pdf")
        render(simple_chain, pos, output=out)
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0

    @pytest.mark.slow
    def test_save_eps(self, simple_chain, fast_config, tmp_path):
        pos = layout(simple_chain, fast_config)
        out = str(tmp_path / "test.eps")
        render(simple_chain, pos, output=out)
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0

    @pytest.mark.slow
    def test_save_jpeg(self, simple_chain, fast_config, tmp_path):
        pytest.importorskip("PIL")
        pos = layout(simple_chain, fast_config)
        out = str(tmp_path / "test.jpg")
        render(simple_chain, pos, output=out)
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0

    @pytest.mark.slow
    def test_format_override_uses_requested_format(self, simple_chain, fast_config, tmp_path):
        pos = layout(simple_chain, fast_config)
        out = str(tmp_path / "forced.bin")
        render(simple_chain, pos, output=out, format="png")
        assert Path(out).exists()
        with open(out, "rb") as f:
            assert f.read(8) == b"\x89PNG\r\n\x1a\n"

    @pytest.mark.slow
    def test_vector_format_override_uses_requested_format(
        self, simple_chain, fast_config, tmp_path
    ):
        pos = layout(simple_chain, fast_config)
        out = str(tmp_path / "forced-vector.bin")
        render(simple_chain, pos, output=out, format="svg")
        assert Path(out).exists()
        with open(out, "rt", encoding="utf-8") as f:
            content = f.read(256)
        assert "<svg" in content or ":svg" in content

    @pytest.mark.slow
    def test_svg_hover_text_can_be_disabled(self, simple_chain, fast_config, tmp_path):
        pos = layout(simple_chain, fast_config)
        out = str(tmp_path / "no-hover.svg")
        render(simple_chain, pos, output=out, svg_hover_text=False)
        content = Path(out).read_text(encoding="utf-8")
        assert "<title>a</title>" not in content
        assert "<title>b</title>" not in content

    @pytest.mark.slow
    def test_custom_figsize(self, simple_chain, fast_config):
        pos = layout(simple_chain, fast_config)
        fig, ax = render(simple_chain, pos, figsize=(10, 8))
        w, h = fig.get_size_inches()
        assert abs(w - 10) < 0.1
        assert abs(h - 8) < 0.1

    @pytest.mark.slow
    def test_render_can_use_cached_positions(self, simple_chain, fast_config):
        layout(simple_chain, fast_config)
        fig, ax = render(simple_chain)
        assert fig is not None
        assert ax is not None

    @pytest.mark.slow
    def test_draw_relayout_false_requires_fresh_layout(self, simple_chain, fast_config):
        with pytest.raises(ValueError, match="Graph layout is missing"):
            dagua.draw(simple_chain, fast_config, relayout=False)

    @pytest.mark.slow
    def test_draw_uses_graph_direction_when_config_is_implicit(self):
        g = DaguaGraph.from_edge_list([("a", "b"), ("b", "c"), ("c", "d")], direction="LR")
        fig, ax = dagua.draw(g)
        pos = g.last_positions
        assert fig is not None
        assert ax is not None
        assert pos is not None
        x_span = float(pos[:, 0].max().item() - pos[:, 0].min().item())
        y_span = float(pos[:, 1].max().item() - pos[:, 1].min().item())
        assert x_span > y_span

    @pytest.mark.slow
    def test_draw_direction_override_wins(self):
        g = DaguaGraph.from_edge_list([("a", "b"), ("b", "c"), ("c", "d")], direction="TB")
        config = LayoutConfig(steps=60, edge_opt_steps=-1, direction="TB", seed=42)
        fig, ax = dagua.draw(g, config=config, direction="LR")
        pos = g.last_positions
        assert fig is not None
        assert ax is not None
        assert pos is not None
        x_span = float(pos[:, 0].max().item() - pos[:, 0].min().item())
        y_span = float(pos[:, 1].max().item() - pos[:, 1].min().item())
        assert x_span > y_span


class TestRenderWithClusters:
    @pytest.mark.slow
    def test_clustered_graph(self, clustered_graph, fast_config, tmp_path):
        pos = layout(clustered_graph, fast_config)
        out = str(tmp_path / "clustered.png")
        render(clustered_graph, pos, output=out)
        assert Path(out).exists()


class TestRenderEdgeLabels:
    @pytest.mark.slow
    def test_edge_labels(self, fast_config, tmp_path):
        g = DaguaGraph.from_edge_list([("a", "b"), ("b", "c")])
        g.edge_labels = ["first", "second"]
        pos = layout(g, fast_config)
        out = str(tmp_path / "labeled.png")
        render(g, pos, output=out)
        assert Path(out).exists()


class TestRenderStyleFlexibility:
    @pytest.mark.slow
    def test_dotted_border_renders(self, fast_config, tmp_path):
        g = DaguaGraph.from_edge_list([("a", "b")])
        g.node_styles[0] = NodeStyle(stroke_dash="dotted")
        pos = layout(g, fast_config)
        out = str(tmp_path / "dotted.png")
        render(g, pos, output=out)
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "shape",
        [
            "triangle",
            "hexagon",
            "pentagon",
            "octagon",
            "star",
            "parallelogram",
            "trapezoid",
            "cylinder",
        ],
    )
    def test_new_shapes_render(self, fast_config, tmp_path, shape):
        g = DaguaGraph.from_edge_list([("a", "b")])
        g.node_styles[0] = NodeStyle(shape=shape, shadow=True, gradient="linear")
        pos = layout(g, fast_config)
        out = str(tmp_path / f"{shape}.png")
        render(g, pos, output=out)
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0

    @pytest.mark.slow
    def test_rich_label_renders(self, fast_config, tmp_path):
        g = DaguaGraph.from_edge_list([("a", "b")])
        g.node_labels[0] = "**Bold** and *italic*\n{color:#FF0000}red{/color} `mono`"
        g.node_styles[0] = NodeStyle(
            label_format="rich",
            text_align="left",
            text_valign="top",
            text_outline=True,
        )
        pos = layout(g, fast_config)
        out = str(tmp_path / "rich.png")
        render(g, pos, output=out)
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "arrow",
        ["normal", "vee", "dot", "diamond", "tee", "crow", "circle", "open", "none"],
    )
    def test_arrow_types_render(self, fast_config, tmp_path, arrow):
        g = DaguaGraph.from_edge_list([("a", "b")])
        g.edge_styles[0] = EdgeStyle(arrow=arrow, tail_arrow="diamond", arrow_fill="hollow")
        pos = layout(g, fast_config)
        out = str(tmp_path / f"{arrow}.png")
        render(g, pos, output=out)
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0

    @pytest.mark.slow
    def test_edge_label_font_fields_render(self, fast_config, tmp_path):
        g = DaguaGraph.from_edge_list([("a", "b")])
        g.edge_labels = ["edge label"]
        g.edge_styles[0] = EdgeStyle(
            label_font_family="DejaVu Sans",
            label_font_weight="bold",
        )
        pos = layout(g, fast_config)
        out = str(tmp_path / "edge-label-fonts.png")
        render(g, pos, output=out)
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0
