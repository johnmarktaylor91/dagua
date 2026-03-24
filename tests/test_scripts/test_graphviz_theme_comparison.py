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
