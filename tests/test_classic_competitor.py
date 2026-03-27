"""Tests for classic competitor adapters."""

from __future__ import annotations

import subprocess
from typing import Any

import pytest
import torch

from dagua.eval.competitors import classic_competitor, get_available_competitors
from dagua.eval.competitors.classic_competitor import ClassicFR, ClassicNeuLay, ClassicSGD2Multi
from dagua.eval.graphs import get_test_graphs
from dagua.graph import DaguaGraph

EXPECTED_CLASSIC_NAMES = {
    "classic_fr",
    "classic_kk",
    "classic_fa2",
    "classic_stress_sgd",
    "classic_sugiyama",
    "classic_spectral",
    "classic_classical_mds",
    "classic_stress_maj",
    "classic_pivot_mds",
    "classic_rt",
    "classic_linlog",
    "classic_gem",
    "classic_tsnet",
    "classic_maxent_stress",
    "classic_davidson_harel",
    "classic_fmmm",
}


def _make_small_graph() -> DaguaGraph:
    """Create a connected 10-node graph suitable for all classic layouts.

    Returns
    -------
    DaguaGraph
        Graph with computed node sizes.
    """
    edges = [(f"node_{index}", f"node_{index + 1}") for index in range(9)]
    graph = DaguaGraph.from_edge_list(edges)
    graph.compute_node_sizes()
    if graph.node_sizes is None:
        raise AssertionError("node sizes should be available after compute_node_sizes()")
    return graph


def _make_clustered_graph() -> DaguaGraph:
    """Create a small clustered graph for compound-layout competitors.

    Returns
    -------
    DaguaGraph
        Four-node path graph with two sibling clusters.
    """
    graph = DaguaGraph()
    graph.add_node("a")
    graph.add_node("b")
    graph.add_node("c")
    graph.add_node("d")
    graph.add_edge("a", "b")
    graph.add_edge("b", "c")
    graph.add_edge("c", "d")
    graph.add_cluster("group1", ["a", "b"])
    graph.add_cluster("group2", ["c", "d"])
    graph.compute_node_sizes()
    if graph.node_sizes is None:
        raise AssertionError("node sizes should be available after compute_node_sizes()")
    return graph


def test_classic_competitors_are_discoverable() -> None:
    """All classic competitors should appear in the available registry."""
    competitor_names = {competitor.name for competitor in get_available_competitors()}
    assert EXPECTED_CLASSIC_NAMES.issubset(competitor_names)


def test_classic_competitor_names_match_expected_values() -> None:
    """The registered classic competitor names should match the expected set."""
    classic_names = {
        competitor.name
        for competitor in get_available_competitors()
        if competitor.name.startswith("classic_")
    }
    assert classic_names == EXPECTED_CLASSIC_NAMES


def test_competitor_max_node_limits_match_regressions() -> None:
    """Regression limits should match the known safe adapter ceilings."""
    from dagua.eval.competitors import get_competitors

    competitors = {competitor.name: competitor for competitor in get_competitors()}

    assert competitors["classic_davidson_harel"].max_nodes == 50
    assert competitors["classic_stress_maj"].max_nodes == 500
    assert competitors["elk_layered"].max_nodes == 15_000


def test_each_classic_competitor_produces_a_valid_result() -> None:
    """Each classic competitor should return a successful layout result."""
    graph = _make_small_graph()
    competitors = {
        competitor.name: competitor
        for competitor in get_available_competitors()
        if competitor.name in EXPECTED_CLASSIC_NAMES
    }

    for competitor_name in EXPECTED_CLASSIC_NAMES:
        result = competitors[competitor_name].layout(graph)
        assert result.name == competitor_name
        assert result.error is None
        assert result.pos is not None
        assert result.runtime_seconds >= 0.0


def test_classic_competitor_positions_have_expected_shape() -> None:
    """Classic competitor layouts should return position tensors with shape ``[N, 2]``."""
    graph = _make_small_graph()
    competitors = {
        competitor.name: competitor
        for competitor in get_available_competitors()
        if competitor.name in EXPECTED_CLASSIC_NAMES
    }

    for competitor_name in EXPECTED_CLASSIC_NAMES:
        result = competitors[competitor_name].layout(graph)
        assert result.pos is not None
        assert tuple(result.pos.shape) == (graph.num_nodes, 2)


def test_classic_fr_seed_override_changes_layout() -> None:
    """Classic FR should preserve the default seed and honor explicit overrides."""
    graph = next(tg.graph for tg in get_test_graphs() if tg.name == "residual_block")
    graph.compute_node_sizes()

    competitor = ClassicFR()
    default_result = competitor.layout(graph)
    seeded_result = competitor.layout(graph, seed=42)
    other_seed_result = competitor.layout(graph, seed=99)

    assert default_result.pos is not None
    assert seeded_result.pos is not None
    assert other_seed_result.pos is not None
    assert torch.allclose(default_result.pos, seeded_result.pos)
    assert not torch.allclose(seeded_result.pos, other_seed_result.pos)


def test_classic_neulay_uses_full_two_phase_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Classic NeuLay should benchmark the full GCN-enabled configuration."""
    graph = _make_small_graph()
    observed: dict[str, object] = {}

    def _fake_quick_classic(
        name: str,
        import_path: str,
        fn_name: str,
        graph: DaguaGraph,
        seed: int,
        **extra_kwargs: Any,
    ) -> object:
        """Capture classic NeuLay forwarding parameters."""
        del graph
        observed["name"] = name
        observed["import_path"] = import_path
        observed["fn_name"] = fn_name
        observed["seed"] = seed
        observed["extra_kwargs"] = extra_kwargs
        return object()

    monkeypatch.setattr(classic_competitor, "_quick_classic", _fake_quick_classic)

    result = ClassicNeuLay().layout(graph, seed=23)

    assert result is not None
    assert observed["name"] == "classic_neulay"
    assert observed["import_path"] == "dagua.layout.classic.neulay"
    assert observed["fn_name"] == "layout_neulay"
    assert observed["seed"] == 23
    assert observed["extra_kwargs"] == {
        "steps": 20_000,
        "gcn_steps": 2_000,
        "use_gcn": True,
    }


def test_classic_sgd2_multi_enables_multiple_criteria(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Classic ``(SGD)^2`` should not fall back to pure stress."""
    graph = _make_small_graph()
    observed: dict[str, object] = {}

    def _fake_quick_classic(
        name: str,
        import_path: str,
        fn_name: str,
        graph: DaguaGraph,
        seed: int,
        **extra_kwargs: Any,
    ) -> object:
        """Capture classic ``(SGD)^2`` forwarding parameters."""
        del graph
        observed["name"] = name
        observed["import_path"] = import_path
        observed["fn_name"] = fn_name
        observed["seed"] = seed
        observed["extra_kwargs"] = extra_kwargs
        return object()

    monkeypatch.setattr(classic_competitor, "_quick_classic", _fake_quick_classic)

    result = ClassicSGD2Multi().layout(graph, seed=17)

    assert result is not None
    assert observed["name"] == "classic_sgd2_multi"
    assert observed["import_path"] == "dagua.layout.classic.sgd2_multi"
    assert observed["fn_name"] == "layout_sgd2_multi"
    assert observed["seed"] == 17
    assert observed["extra_kwargs"] == {"criteria": {"stress": 1.0, "ideal_edge_length": 1.0}}


def test_graphviz_dot_with_clusters() -> None:
    """Graphviz dot should lay out clustered graphs without dropping nodes.

    Returns
    -------
    None
        The assertion validates clustered layout output.
    """
    graph = _make_clustered_graph()

    from dagua.eval.competitors.graphviz_competitor import GraphvizDot

    competitor = GraphvizDot()
    if not competitor.available():
        pytest.skip("Graphviz dot is not installed")

    result = competitor.layout(graph)
    assert result.error is None
    assert result.pos is not None
    assert tuple(result.pos.shape) == (4, 2)


def test_graphviz_base_forwards_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Graphviz base competitors should forward the requested timeout.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace the shared Graphviz utility.

    Returns
    -------
    None
        The assertion validates timeout propagation.
    """
    import dagua.graphviz_utils as graphviz_utils
    from dagua.eval.competitors.graphviz_competitor import GraphvizSfdp

    graph = _make_small_graph()
    observed: dict[str, float | str] = {}

    def _fake_layout_with_graphviz(
        input_graph: DaguaGraph,
        engine: str = "dot",
        timeout: float = 300.0,
    ) -> torch.Tensor:
        """Capture Graphviz utility arguments for the regression test.

        Parameters
        ----------
        input_graph : DaguaGraph
            Graph passed through the competitor.
        engine : str, default="dot"
            Requested Graphviz engine.
        timeout : float, default=300.0
            Requested timeout in seconds.

        Returns
        -------
        torch.Tensor
            Dummy position tensor with shape ``[N, 2]``.
        """
        observed["engine"] = engine
        observed["timeout"] = timeout
        return torch.zeros((input_graph.num_nodes, 2))

    monkeypatch.setattr(graphviz_utils, "layout_with_graphviz", _fake_layout_with_graphviz)

    result = GraphvizSfdp().layout(graph, timeout=123.0)

    assert result.error is None
    assert observed == {"engine": "sfdp", "timeout": 123.0}


def test_graphviz_base_classifies_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Graphviz base competitors should normalize subprocess timeouts."""
    import dagua.graphviz_utils as graphviz_utils
    from dagua.eval.competitors.graphviz_competitor import GraphvizSfdp

    graph = _make_small_graph()

    def _fake_layout_with_graphviz(
        input_graph: DaguaGraph,
        engine: str = "dot",
        timeout: float = 300.0,
    ) -> torch.Tensor:
        """Raise a timeout to validate adapter error normalization.

        Parameters
        ----------
        input_graph : DaguaGraph
            Graph passed through the competitor.
        engine : str, default="dot"
            Requested Graphviz engine.
        timeout : float, default=300.0
            Requested timeout in seconds.

        Returns
        -------
        torch.Tensor
            This helper never returns because it always raises.
        """
        del input_graph, engine
        raise subprocess.TimeoutExpired(cmd="sfdp", timeout=timeout)

    monkeypatch.setattr(graphviz_utils, "layout_with_graphviz", _fake_layout_with_graphviz)

    result = GraphvizSfdp().layout(graph, timeout=7.0)

    assert result.pos is None
    assert result.error == "timeout"
    assert result.runtime_seconds >= 0.0


def test_graphviz_dot_classifies_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """The dot-specific adapter should normalize subprocess timeouts."""
    from dagua.eval.competitors import graphviz_competitor
    from dagua.eval.competitors.graphviz_competitor import GraphvizDot

    graph = _make_small_graph()

    def _fake_layout_with_dot(input_graph: DaguaGraph, timeout: float) -> torch.Tensor:
        """Raise a timeout from the dot execution path.

        Parameters
        ----------
        input_graph : DaguaGraph
            Graph passed through the competitor.
        timeout : float
            Requested timeout in seconds.

        Returns
        -------
        torch.Tensor
            This helper never returns because it always raises.
        """
        del input_graph
        raise subprocess.TimeoutExpired(cmd="dot", timeout=timeout)

    monkeypatch.setattr(graphviz_competitor, "_layout_with_dot", _fake_layout_with_dot)

    result = GraphvizDot().layout(graph, timeout=9.0)

    assert result.pos is None
    assert result.error == "timeout"
    assert result.runtime_seconds >= 0.0


def test_dagua_competitor_uses_detected_device_and_returns_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dagua competitor should detect the runtime device and return CPU tensors."""
    import importlib

    import dagua.config as dagua_config
    from dagua.eval.competitors.dagua_competitor import DaguaCompetitor

    dagua_layout_module = importlib.import_module("dagua.layout")
    graph = _make_small_graph()
    competitor = DaguaCompetitor()
    observed: dict[str, str | bool] = {}

    def _fake_is_available() -> bool:
        """Pretend CUDA is available for device-selection coverage.

        Returns
        -------
        bool
            Always ``True``.
        """
        return True

    def _fake_layout_config(*args: object, **kwargs: object) -> object:
        """Capture the requested device without constructing a real config.

        Parameters
        ----------
        *args : object
            Positional arguments forwarded to ``LayoutConfig``.
        **kwargs : object
            Keyword arguments forwarded to ``LayoutConfig``.

        Returns
        -------
        object
            Minimal config-like object with ``device`` and ``verbose`` fields.
        """
        del args
        observed["config_device"] = str(kwargs["device"])
        observed["config_verbose"] = bool(kwargs["verbose"])
        return type("FakeConfig", (), {"device": kwargs["device"], "verbose": kwargs["verbose"]})()

    def _fake_layout(input_graph: DaguaGraph, config: object) -> torch.Tensor:
        """Return a differentiable tensor so the adapter must detach it.

        Parameters
        ----------
        input_graph : DaguaGraph
            Graph passed through the competitor.
        config : object
            Config-like object constructed by the patched ``LayoutConfig``.

        Returns
        -------
        torch.Tensor
            CPU tensor with gradients enabled.
        """
        observed["layout_device"] = str(getattr(config, "device"))
        return torch.ones((input_graph.num_nodes, 2), requires_grad=True)

    monkeypatch.setattr(torch.cuda, "is_available", _fake_is_available)
    monkeypatch.setattr(dagua_config, "LayoutConfig", _fake_layout_config)
    monkeypatch.setattr(dagua_layout_module, "layout", _fake_layout)

    result = competitor.layout(graph)

    assert competitor.device == "cuda"
    assert observed == {
        "config_device": "cuda",
        "config_verbose": False,
        "layout_device": "cuda",
    }
    assert result.error is None
    assert result.pos is not None
    assert result.pos.device.type == "cpu"
    assert not result.pos.requires_grad


def test_supports_clusters_flag() -> None:
    """Cluster-capable competitors should be explicitly marked.

    Returns
    -------
    None
        The assertion validates the registry metadata.
    """
    from dagua.eval.competitors import get_competitors

    cluster_names = {
        competitor.name for competitor in get_competitors() if competitor.supports_clusters
    }
    assert {"graphviz_dot", "elk_layered", "dagre"}.issubset(cluster_names)
