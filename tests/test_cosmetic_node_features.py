"""Tests for cosmetic node rendering features."""

from __future__ import annotations

import matplotlib
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import PathPatch

from dagua.graph import DaguaGraph
from dagua.render import render
from dagua.styles import EdgeStyle, NodeStyle
from dagua.utils import compute_node_size, prepare_label_text

matplotlib.use("Agg")


class TestItalicFont:
    """Cover the italic font-style field."""

    def test_field_exists(self) -> None:
        """NodeStyle should expose the italic font-style option."""
        style = NodeStyle(font_style="italic")
        assert style.font_style == "italic"

    def test_default_is_normal(self) -> None:
        """NodeStyle should default to normal font style."""
        assert NodeStyle().font_style == "normal"


class TestTextWrapping:
    """Cover text wrapping and truncation helpers."""

    def test_text_wrap_field(self) -> None:
        """NodeStyle should store wrap configuration."""
        style = NodeStyle(text_wrap="wrap", text_max_width=50.0)
        assert style.text_wrap == "wrap"
        assert style.text_max_width == 50.0

    def test_wrap_prepares_multiline_text(self) -> None:
        """Wrapped labels should gain explicit line breaks."""
        wrapped = prepare_label_text(
            "wrap this label please",
            font_size=10.0,
            text_wrap="wrap",
            text_max_width=36.0,
        )
        assert "\n" in wrapped

    def test_ellipsis_prepares_truncated_text(self) -> None:
        """Ellipsis mode should trim long labels."""
        ellipsized = prepare_label_text(
            "truncate me now",
            font_size=10.0,
            text_wrap="ellipsis",
            text_max_width=30.0,
        )
        assert ellipsized.endswith("...")

    def test_wrapped_text_changes_node_size_measurement(self) -> None:
        """Wrapping should trade width for height during node sizing."""
        plain_width, plain_height, _ = compute_node_size(
            "wrap this label please",
            font_size=10.0,
        )
        wrapped_width, wrapped_height, _ = compute_node_size(
            "wrap this label please",
            font_size=10.0,
            text_wrap="wrap",
            text_max_width=36.0,
        )
        assert wrapped_width < plain_width
        assert wrapped_height > plain_height


class TestTextTransform:
    """Cover uppercase and lowercase transforms."""

    def test_uppercase(self) -> None:
        """NodeStyle should store the uppercase transform."""
        style = NodeStyle(text_transform="uppercase")
        assert style.text_transform == "uppercase"

    def test_lowercase(self) -> None:
        """NodeStyle should store the lowercase transform."""
        style = NodeStyle(text_transform="lowercase")
        assert style.text_transform == "lowercase"

    def test_default_none(self) -> None:
        """NodeStyle should default to no transform."""
        assert NodeStyle().text_transform == "none"

    def test_transform_helper_applies_case(self) -> None:
        """Prepared labels should reflect the configured transform."""
        assert prepare_label_text("MiXeD", font_size=10.0, text_transform="uppercase") == "MIXED"
        assert prepare_label_text("MiXeD", font_size=10.0, text_transform="lowercase") == "mixed"


class TestDoubleBorder:
    """Cover the double-border style field."""

    def test_border_count(self) -> None:
        """NodeStyle should store the requested border count."""
        style = NodeStyle(border_count=2)
        assert style.border_count == 2

    def test_default_single(self) -> None:
        """NodeStyle should default to a single border."""
        assert NodeStyle().border_count == 1


class TestLineCap:
    """Cover node and edge cap/join style fields."""

    def test_node_cap(self) -> None:
        """NodeStyle should store the requested cap style."""
        style = NodeStyle(stroke_cap="round")
        assert style.stroke_cap == "round"

    def test_edge_cap(self) -> None:
        """EdgeStyle should store the requested cap style."""
        style = EdgeStyle(line_cap="round")
        assert style.line_cap == "round"


class TestStripedFill:
    """Cover fill-pattern fields."""

    def test_striped(self) -> None:
        """NodeStyle should store striped fill settings."""
        style = NodeStyle(fill_pattern="striped", fill_pattern_colors=["#FF0000", "#0000FF"])
        assert style.fill_pattern == "striped"
        assert style.fill_pattern_colors == ["#FF0000", "#0000FF"]

    def test_hatched(self) -> None:
        """NodeStyle should store the hatched fill pattern."""
        style = NodeStyle(fill_pattern="hatched")
        assert style.fill_pattern == "hatched"


class TestRenderSmoke:
    """Smoke-test the new cosmetics through the matplotlib renderer."""

    def test_render_with_italic(self) -> None:
        """Italic node labels should render without error."""
        graph = DaguaGraph()
        graph.add_node("A", label="Italic", style=NodeStyle(font_style="italic"))
        graph.add_node("B", label="Normal")
        graph.add_edge("A", "B")
        graph.compute_node_sizes()
        graph.cache_layout(torch.tensor([[0, 20], [0, -20]], dtype=torch.float32))
        fig, _ax = render(graph, show=False)
        assert fig is not None
        plt.close(fig)

    def test_render_with_double_border(self) -> None:
        """Double-border nodes should render border strokes separately."""
        graph = DaguaGraph()
        graph.add_node(
            "A",
            label="",
            style=NodeStyle(border_count=2, stroke_width=1.5),
        )
        graph.compute_node_sizes()
        graph.cache_layout(torch.tensor([[0, 0]], dtype=torch.float32))
        fig, ax = render(graph, show=False)
        border_patches = [
            patch
            for patch in ax.patches
            if isinstance(patch, PathPatch) and float(patch.get_linewidth()) > 0.0
        ]
        assert len(border_patches) >= 2
        plt.close(fig)

    def test_render_with_striped_fill(self) -> None:
        """Striped fills should create a clipped image artist."""
        graph = DaguaGraph()
        graph.add_node(
            "A",
            label="",
            style=NodeStyle(
                fill_pattern="striped",
                fill_pattern_colors=["#FF0000", "#0000FF"],
                fill_pattern_angle=25.0,
            ),
        )
        graph.compute_node_sizes()
        graph.cache_layout(torch.tensor([[0, 0]], dtype=torch.float32))
        fig, ax = render(graph, show=False)
        assert len(ax.images) >= 1
        plt.close(fig)

    def test_render_with_round_node_border_styles(self) -> None:
        """Node border patches should receive cap and join settings."""
        graph = DaguaGraph()
        graph.add_node(
            "A",
            label="",
            style=NodeStyle(stroke_cap="round", stroke_join="round"),
        )
        graph.compute_node_sizes()
        graph.cache_layout(torch.tensor([[0, 0]], dtype=torch.float32))
        fig, ax = render(graph, show=False)
        matching = [
            patch
            for patch in ax.patches
            if isinstance(patch, PathPatch)
            and float(patch.get_linewidth()) > 0.0
            and patch.get_capstyle() == "round"
            and patch.get_joinstyle() == "round"
        ]
        assert matching
        plt.close(fig)

    def test_render_with_round_edge_styles(self) -> None:
        """Direct edge rendering should forward line cap and join."""
        graph = DaguaGraph()
        graph.add_node("A", label="")
        graph.add_node("B", label="")
        graph.add_edge("A", "B", style=EdgeStyle(line_cap="round", line_join="bevel"))
        graph.compute_node_sizes()
        graph.cache_layout(torch.tensor([[0, 20], [0, -20]], dtype=torch.float32))
        fig, ax = render(graph, show=False)
        matching = [
            patch
            for patch in ax.patches
            if isinstance(patch, PathPatch)
            and patch.get_capstyle() == "round"
            and patch.get_joinstyle() == "bevel"
        ]
        assert matching
        plt.close(fig)
