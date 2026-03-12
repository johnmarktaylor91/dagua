"""Tests for matplotlib renderer."""

import pytest
import tempfile
from pathlib import Path

import torch

from dagua.graph import DaguaGraph
from dagua.config import LayoutConfig
from dagua.layout import layout
from dagua.render import render


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
    def test_vector_format_override_uses_requested_format(self, simple_chain, fast_config, tmp_path):
        pos = layout(simple_chain, fast_config)
        out = str(tmp_path / "forced-vector.bin")
        render(simple_chain, pos, output=out, format="svg")
        assert Path(out).exists()
        with open(out, "rt", encoding="utf-8") as f:
            content = f.read(256)
        assert "<svg" in content

    @pytest.mark.slow
    def test_custom_figsize(self, simple_chain, fast_config):
        pos = layout(simple_chain, fast_config)
        fig, ax = render(simple_chain, pos, figsize=(10, 8))
        w, h = fig.get_size_inches()
        assert abs(w - 10) < 0.1
        assert abs(h - 8) < 0.1


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
