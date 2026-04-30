"""Tests for Graphviz competitor adapters."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from dagua.eval.competitors.graphviz_competitor import GraphvizDot, GraphvizFdp
from dagua.graph import DaguaGraph


def _tiny_graph() -> DaguaGraph:
    """Build a tiny graph for Graphviz adapter tests.

    Returns
    -------
    DaguaGraph
        Two-node directed graph.
    """
    graph = DaguaGraph()
    graph.add_node("a")
    graph.add_node("b")
    graph.add_edge("a", "b")
    graph.compute_node_sizes()
    return graph


def _graphviz_json_stdout() -> str:
    """Return minimal Graphviz JSON output with node positions.

    Returns
    -------
    str
        JSON payload accepted by the adapter parser.
    """
    return '{"objects": [{"name": "n0", "pos": "0,0"}, {"name": "n1", "pos": "1,2"}]}'


def test_fdp_seed_reaches_graphviz_subprocess(monkeypatch: Any) -> None:
    """The fdp adapter should pass seed and start attributes to Graphviz.

    Parameters
    ----------
    monkeypatch : Any
        Pytest monkeypatch fixture.

    Returns
    -------
    None
        Assertions validate subprocess arguments.
    """
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        """Capture subprocess arguments and return a successful JSON payload.

        Parameters
        ----------
        command : list[str]
            Subprocess command passed by the adapter.
        **kwargs : Any
            Additional subprocess keyword arguments.

        Returns
        -------
        SimpleNamespace
            Minimal object matching ``subprocess.CompletedProcess`` fields used
            by the adapter.
        """
        del kwargs
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=_graphviz_json_stdout(), stderr="")

    monkeypatch.setattr("dagua.eval.competitors.graphviz_competitor.subprocess.run", fake_run)

    result = GraphvizFdp().layout(_tiny_graph(), seed=17)

    assert result.error is None
    assert calls
    assert calls[0][:3] == ["dot", "-Tjson", "-Kfdp"]
    assert "-Gseed=17" in calls[0]
    assert "-Gstart=17" in calls[0]


def test_dot_keeps_seed_out_of_deterministic_subprocess(monkeypatch: Any) -> None:
    """The dot adapter should ignore seed because dot is deterministic.

    Parameters
    ----------
    monkeypatch : Any
        Pytest monkeypatch fixture.

    Returns
    -------
    None
        Assertions validate subprocess arguments.
    """
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        """Capture subprocess arguments and return a successful JSON payload.

        Parameters
        ----------
        command : list[str]
            Subprocess command passed by the adapter.
        **kwargs : Any
            Additional subprocess keyword arguments.

        Returns
        -------
        SimpleNamespace
            Minimal object matching ``subprocess.CompletedProcess`` fields used
            by the adapter.
        """
        del kwargs
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=_graphviz_json_stdout(), stderr="")

    monkeypatch.setattr("dagua.eval.competitors.graphviz_competitor.subprocess.run", fake_run)

    result = GraphvizDot().layout(_tiny_graph(), seed=17)

    assert result.error is None
    assert calls
    assert calls[0][:2] == ["dot", "-Tjson"]
    assert "-Gseed=17" not in calls[0]
    assert "-Gstart=17" not in calls[0]
