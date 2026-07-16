"""Regression tests for the Graphviz theme comparison script."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from dagua import DaguaGraph
from scripts import graphviz_theme_comparison


def test_render_dagua_theme_uses_graphviz_positions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dagua panels should render on Graphviz-derived positions.

    Parameters
    ----------
    tmp_path : Path
        Temporary output directory.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to isolate external dependencies.

    Returns
    -------
    None
        The assertions verify that Graphviz positions are forwarded directly to
        ``dagua.render()`` and that ``dagua.layout()`` is not used.
    """

    graph = DaguaGraph.from_edge_list([("a", "b"), ("b", "c")])
    expected_positions = torch.tensor([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
    observed: dict[str, Any] = {}

    def _fake_layout_with_graphviz(graph: DaguaGraph, engine: str = "dot") -> torch.Tensor:
        """Return fixed Graphviz positions for the themed graph.

        Parameters
        ----------
        graph : DaguaGraph
            Themed graph passed to Graphviz.
        engine : str, default="dot"
            Requested Graphviz engine.

        Returns
        -------
        torch.Tensor
            Stable node positions used for the regression assertion.
        """

        observed["graph"] = graph
        observed["engine"] = engine
        return expected_positions.clone()

    def _fail_layout(*args: object, **kwargs: object) -> torch.Tensor:
        """Fail fast if the old Dagua layout path is used.

        Parameters
        ----------
        *args : object
            Positional arguments forwarded to the unexpected call.
        **kwargs : object
            Keyword arguments forwarded to the unexpected call.

        Returns
        -------
        torch.Tensor
            This function never returns.

        Raises
        ------
        AssertionError
            Always raised when the deprecated code path is hit.
        """

        del args, kwargs
        raise AssertionError("render_dagua_theme() should not call dagua.layout()")

    def _fake_render(
        graph: DaguaGraph,
        positions: torch.Tensor,
        *,
        output: str,
        dpi: int,
        **kwargs: object,
    ) -> tuple[None, None]:
        """Capture the render call for assertions.

        Parameters
        ----------
        graph : DaguaGraph
            Themed graph being rendered.
        positions : torch.Tensor
            Node positions forwarded to the renderer.
        output : str
            Output image path.
        dpi : int
            Raster resolution.
        **kwargs : object
            Additional render keyword arguments.

        Returns
        -------
        tuple[None, None]
            Minimal stub matching ``dagua.render()``'s tuple return shape.
        """

        observed["render_graph"] = graph
        observed["render_positions"] = positions.clone()
        observed["output"] = output
        observed["dpi"] = dpi
        observed["kwargs"] = kwargs
        return None, None

    monkeypatch.setattr(
        graphviz_theme_comparison,
        "layout_with_graphviz",
        _fake_layout_with_graphviz,
    )
    monkeypatch.setattr(graphviz_theme_comparison.dagua, "layout", _fail_layout)
    monkeypatch.setattr(graphviz_theme_comparison.dagua, "render", _fake_render)
    monkeypatch.setattr(graphviz_theme_comparison.dagua, "set_theme", lambda theme_name: None)

    output_path = tmp_path / "strict.png"
    graphviz_theme_comparison.render_dagua_theme(
        graph,
        graphviz_theme_comparison.STRICT_THEME_NAME,
        output_path,
    )

    assert observed["engine"] == "dot"
    assert torch.equal(observed["render_positions"], expected_positions)
    assert observed["output"] == str(output_path)
    assert observed["dpi"] == 210
    assert observed["graph"] is observed["render_graph"]


def test_arrowhead_atlas_categories_match_expected_counts() -> None:
    """The four labeled arrowhead atlas categories should match F2's counts."""

    categories = graphviz_theme_comparison._arrowhead_atlas_categories()

    assert len(categories["primitive"]) == 23
    assert len(categories["alias"]) == 4
    assert len(categories["gv_modifier"]) == 42
    assert len(categories["compound"]) >= 12
    assert set(categories.keys()) == {"primitive", "alias", "gv_modifier", "compound"}


def test_arrowhead_atlas_graph_has_four_labeled_clusters() -> None:
    """The atlas graph should carry one cluster per category, each labeled."""

    graph, title = graphviz_theme_comparison._make_arrowhead_atlas()

    assert title == "Arrowhead Atlas"
    assert len(graph.cluster_labels) == 4
    for label in graph.cluster_labels.values():
        assert any(
            label.startswith(f"{category} (")
            for category in ("primitive", "alias", "gv_modifier", "compound")
        )


def test_registry_arrow_types_replaces_hard_coded_tuple() -> None:
    """Arrow type enumeration should come from the live arrowhead registry."""

    from dagua.render.edges.arrowheads import available_arrowheads

    assert graphviz_theme_comparison._registry_arrow_types() == tuple(available_arrowheads())
    assert not hasattr(graphviz_theme_comparison, "ARROW_TYPES")


def test_shape_atlas_covers_all_supported_shapes_without_placeholders() -> None:
    """The shape atlas should render every implemented shape directly."""

    graph, title = graphviz_theme_comparison._make_shape_atlas()

    assert title == "Shape Atlas"
    labels = graph.node_labels
    assert any(label == "ellipse" for label in labels)
    assert "promoter" in labels
    assert "invtrapezium" in labels
    assert not any(label.startswith("GAP: ") for label in labels)
    gap_labels = {label.removeprefix("GAP: ") for label in labels if label.startswith("GAP: ")}
    assert gap_labels == set(graphviz_theme_comparison.GV_SHAPE_GAP_COMMON) | set(
        graphviz_theme_comparison.GV_SHAPE_WAIVED_SAMPLE
    )


def test_compose_stress_stacks_cosmetics_and_nested_structures() -> None:
    """The composition panel should combine cosmetics rather than isolate them."""

    graph, title = graphviz_theme_comparison._make_compose_stress()

    assert title == "Compose Stress"
    assert graph.num_nodes == 6
    assert len(graph.cluster_labels) == 3
    assert len(graph.cluster_parents) == 2
    assert all(style is not None and style.border_count == 2 for style in graph.node_styles)
    assert {style.fill_pattern for style in graph.node_styles if style is not None} == {
        "gradient",
        "striped",
        "pie",
    }
    assert any(source == target for source, target in graph.edge_index.T.tolist())
    assert any(
        source > target for source, target in graph.edge_index.T.tolist() if source != target
    )


def test_compose_stress_dot_preserves_explicit_pattern_styles() -> None:
    """Explicit Graphviz striped and wedged styles must survive style synthesis."""

    graph, _ = graphviz_theme_comparison._make_compose_stress()
    dot_source = graphviz_theme_comparison.graph_to_dot(graph)

    assert "style=striped" in dot_source
    assert "style=wedged" in dot_source
    assert "peripheries=2" in dot_source


def test_spline_stress_scene_has_back_edges_flat_edge_and_self_loops() -> None:
    """The spline-stress case should exercise back-edges, a flat edge, and self-loops."""

    graph, title = graphviz_theme_comparison._make_spline_stress()

    assert title == "Spline Stress"
    edge_index = graph.edge_index.detach().cpu().numpy()
    sources = [graph.node_labels[i] for i in edge_index[0]]
    targets = [graph.node_labels[i] for i in edge_index[1]]
    self_loops = sum(1 for s, t in zip(sources, targets) if s == t)
    assert self_loops >= 2
    assert "flat a" in graph.node_labels and "flat b" in graph.node_labels


def test_cluster_nest_deep_scene_has_five_nesting_levels() -> None:
    """The deep-nesting cluster case should chain five parent-linked clusters."""

    graph, title = graphviz_theme_comparison._make_cluster_nest_deep()

    assert title == "Cluster Nest Deep"
    assert len(graph.cluster_labels) == 5
    assert len(graph.cluster_parents) == 4
