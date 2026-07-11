"""Tests for classic competitor adapters."""

from __future__ import annotations

import subprocess
import sys
import types
from typing import Any, Optional

import pytest
import torch

from dagua.eval.competitors import classic_competitor, get_available_competitors
from dagua.eval.competitors.classic_competitor import (
    ClassicClassicalMDS,
    ClassicFR,
    ClassicMaxentStress,
    ClassicNeato,
    ClassicNeuLay,
    ClassicPivotMDS,
    ClassicSGD2Multi,
    ClassicStressMajorization,
    ClassicStressSGD,
    ClassicTsNET,
    ClassicUMAP,
)
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
    "classic_neato",
    "classic_pivot_mds",
    "classic_rt",
    "classic_linlog",
    "classic_gem",
    "classic_tsnet",
    "classic_maxent_stress",
    "classic_davidson_harel",
    "classic_fmmm",
    "classic_graphopt",
    "classic_drl",
    "classic_lgl",
    "classic_sfdp",
    "classic_umap",
    "classic_neulay",
    "classic_sgd2_multi",
    "classic_fcose",
    "classic_fr_kk",
    "classic_kk_fr",
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


def _make_weighted_path_graph() -> DaguaGraph:
    """Create a small weighted path graph for adapter forwarding tests.

    Returns
    -------
    DaguaGraph
        Three-node path with edge weights ``[2.0, 3.0]``.
    """
    graph = DaguaGraph()
    graph.add_edge(0, 1, weight=2.0)
    graph.add_edge(1, 2, weight=3.0)
    _ = graph.edge_index
    graph.compute_node_sizes()
    return graph


def _install_classic_layout_spy(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    fn_name: str,
    seen: dict[str, dict[str, Any]],
) -> None:
    """Install a fake classic layout module that captures forwarded kwargs.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace the imported layout module.
    module_name : str
        Fully qualified module path imported by the adapter.
    fn_name : str
        Layout function name to expose on the fake module.
    seen : dict[str, dict[str, Any]]
        Mutable capture dictionary populated with the received kwargs.

    Returns
    -------
    None
        The fake module is registered in ``sys.modules``.
    """
    fake_module = types.ModuleType(module_name)

    def _layout_spy(
        edge_index: torch.Tensor,
        num_nodes: int,
        node_sizes: Optional[torch.Tensor] = None,
        seed: int = 42,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Capture classic layout kwargs and return zero coordinates.

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge tensor shaped ``[2, E]``.
        num_nodes : int
            Number of nodes in the graph.
        node_sizes : torch.Tensor | None, default=None
            Node-size tensor forwarded by the adapter.
        seed : int, default=42
            Seed forwarded by the adapter.
        **kwargs : Any
            Additional layout parameters forwarded by the adapter.

        Returns
        -------
        torch.Tensor
            Zero coordinates shaped ``[N, 2]``.
        """
        del edge_index, seed
        seen["node_sizes"] = None if node_sizes is None else node_sizes.detach().clone()
        seen["kwargs"] = dict(kwargs)
        return torch.zeros((num_nodes, 2), dtype=torch.float32)

    setattr(fake_module, fn_name, _layout_spy)
    monkeypatch.setitem(sys.modules, module_name, fake_module)


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
    assert observed["import_path"] == "dagua.layout.ops.pipelines.neulay"
    assert observed["fn_name"] == "layout_neulay_pipeline"
    assert observed["seed"] == 23
    assert observed["extra_kwargs"] == {
        "steps": 20_000,
        "gcn_steps": 2_000,
        "use_gcn": True,
        "lr": 0.1,
        "radius": 0.4,
    }


def test_classic_embedding_variant_param_names_match_registry_contract() -> None:
    """Classic embedding adapters should declare their supported override names."""
    assert ClassicTsNET.variant_param_names == frozenset({"perplexity", "steps", "fidelity_mode"})
    assert ClassicUMAP.variant_param_names == frozenset(
        {"n_neighbors", "min_dist", "spread", "fidelity_mode"}
    )
    assert ClassicNeuLay.variant_param_names == frozenset(
        {"steps", "gcn_steps", "use_gcn", "lr", "radius", "fidelity_mode"}
    )


def test_classic_layout_with_variant_warns_on_unrecognized_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Classic variant dispatch should warn on unsupported override names."""
    graph = _make_small_graph()

    def _fake_quick_classic(
        name: str,
        import_path: str,
        fn_name: str,
        graph: DaguaGraph,
        seed: int,
        **extra_kwargs: Any,
    ) -> object:
        """Return a dummy result payload for warning validation."""
        del name, import_path, fn_name, graph, seed, extra_kwargs
        return object()

    monkeypatch.setattr(classic_competitor, "_quick_classic", _fake_quick_classic)

    with pytest.warns(
        UserWarning, match="classic_umap received unrecognized variant params: bogus"
    ):
        result = ClassicUMAP().layout_with_variant(graph, seed=19, variant_params={"bogus": 1.0})

    assert result is not None


def test_classic_sfdp_graphviz_fidelity_forwards_dot_node_boxes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Graphviz-fidelity SFDP should pack with DOT label-sized node boxes."""
    graph = DaguaGraph.from_edge_list(
        [
            ("short", "a substantially wider label"),
            ("isolated wide component", "tail"),
        ]
    )
    graph.compute_node_sizes()
    original_sizes = None if graph.node_sizes is None else graph.node_sizes.detach().clone()
    seen: dict[str, Any] = {}
    _install_classic_layout_spy(
        monkeypatch=monkeypatch,
        module_name="dagua.layout.ops.pipelines.sfdp",
        fn_name="layout_sfdp_pipeline",
        seen=seen,
    )

    result = classic_competitor.ClassicSFDP().layout_with_variant(
        graph,
        seed=100,
        variant_params={"fidelity_mode": "graphviz", "steps": 500},
    )

    assert result.pos is not None
    assert seen["kwargs"]["fidelity_mode"] == "graphviz"
    forwarded_sizes = seen["node_sizes"]
    assert isinstance(forwarded_sizes, torch.Tensor)
    assert forwarded_sizes.shape == (graph.num_nodes, 2)
    if original_sizes is not None:
        assert not torch.equal(forwarded_sizes, original_sizes)
    assert float(forwarded_sizes[:, 0].max().item()) > 54.0


def test_classic_sfdp_graphviz_fidelity_preserves_small_modest_label_pack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Small modest-label SFDP graphs should keep the C4d default pack boxes."""
    graph = DaguaGraph.from_edge_list(
        [
            ("enc_in", "enc_conv"),
            ("enc_conv", "enc_relu"),
            ("enc_relu", "enc_out"),
            ("res_in", "res_conv1"),
            ("res_conv1", "res_conv2"),
            ("res_in", "res_add"),
            ("res_conv2", "res_add"),
            ("res_add", "res_out"),
        ]
    )
    graph.compute_node_sizes()
    original_sizes = None if graph.node_sizes is None else graph.node_sizes.detach().clone()
    seen: dict[str, Any] = {}
    _install_classic_layout_spy(
        monkeypatch=monkeypatch,
        module_name="dagua.layout.ops.pipelines.sfdp",
        fn_name="layout_sfdp_pipeline",
        seen=seen,
    )

    result = classic_competitor.ClassicSFDP().layout_with_variant(
        graph,
        seed=100,
        variant_params={"fidelity_mode": "graphviz", "steps": 500},
    )

    assert result.pos is not None
    forwarded_sizes = seen["node_sizes"]
    assert isinstance(forwarded_sizes, torch.Tensor)
    assert original_sizes is not None
    assert torch.equal(forwarded_sizes, original_sizes)


def test_classic_sfdp_graphviz_fidelity_preserves_connected_pack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Connected Graphviz-fidelity SFDP should not use disconnected-pack boxes."""
    graph = DaguaGraph.from_edge_list(
        [
            ("source", "VeryLongConnectedLabelThatWouldOtherwiseTriggerThePackGate"),
            ("VeryLongConnectedLabelThatWouldOtherwiseTriggerThePackGate", "sink"),
        ]
    )
    graph.compute_node_sizes()
    original_sizes = None if graph.node_sizes is None else graph.node_sizes.detach().clone()
    seen: dict[str, Any] = {}
    _install_classic_layout_spy(
        monkeypatch=monkeypatch,
        module_name="dagua.layout.ops.pipelines.sfdp",
        fn_name="layout_sfdp_pipeline",
        seen=seen,
    )

    result = classic_competitor.ClassicSFDP().layout_with_variant(
        graph,
        seed=100,
        variant_params={"fidelity_mode": "graphviz", "steps": 500},
    )

    assert result.pos is not None
    forwarded_sizes = seen["node_sizes"]
    assert isinstance(forwarded_sizes, torch.Tensor)
    assert original_sizes is not None
    assert torch.equal(forwarded_sizes, original_sizes)


def test_classic_sugiyama_graphviz_fidelity_forwards_label_only_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Graphviz-fidelity Sugiyama should pass edge-label boxes for label-only DOT."""
    graph = DaguaGraph()
    graph.add_edge("source", "target", label="handoff label")
    graph.compute_node_sizes()
    seen: dict[str, Any] = {}
    _install_classic_layout_spy(
        monkeypatch=monkeypatch,
        module_name="dagua.layout.ops.pipelines.sugiyama",
        fn_name="layout_sugiyama_pipeline",
        seen=seen,
    )

    result = classic_competitor.ClassicSugiyama().layout_with_variant(
        graph,
        seed=100,
        variant_params={"fidelity_mode": "graphviz"},
    )

    assert result.pos is not None
    kwargs = seen["kwargs"]
    assert isinstance(kwargs["graphviz_node_sizes"], torch.Tensor)
    assert graph.node_sizes is not None
    assert torch.equal(kwargs["graphviz_node_sizes"], graph.node_sizes)
    assert isinstance(kwargs["graphviz_edge_label_sizes"], torch.Tensor)
    assert kwargs["graphviz_edge_label_sizes"].shape == (1, 2)
    assert "clusters" not in kwargs
    assert "cluster_parents" not in kwargs
    assert "graphviz_apply_cluster_constraints" not in kwargs


def test_classic_sugiyama_graphviz_fidelity_forwards_cluster_only_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Graphviz-fidelity Sugiyama should opt into clusters for cluster-only DOT."""
    graph = _make_clustered_graph()
    seen: dict[str, Any] = {}
    _install_classic_layout_spy(
        monkeypatch=monkeypatch,
        module_name="dagua.layout.ops.pipelines.sugiyama",
        fn_name="layout_sugiyama_pipeline",
        seen=seen,
    )

    result = classic_competitor.ClassicSugiyama().layout_with_variant(
        graph,
        seed=100,
        variant_params={"fidelity_mode": "graphviz"},
    )

    assert result.pos is not None
    kwargs = seen["kwargs"]
    assert isinstance(kwargs["graphviz_node_sizes"], torch.Tensor)
    assert isinstance(kwargs["graphviz_typed_node_sizes"], torch.Tensor)
    assert kwargs["clusters"] is graph.clusters
    assert kwargs["cluster_parents"] is graph.cluster_parents
    assert kwargs["graphviz_apply_cluster_constraints"] is True
    assert "graphviz_edge_label_sizes" not in kwargs


def test_classic_sugiyama_typed_boxes_use_dot_fallback_linespacing() -> None:
    """Match Graphviz fallback label height when fitting typed ellipse boxes."""
    width, height = classic_competitor._graphviz_dot_node_box(
        label="encoder.stage_1_attention_projection",
        font_size=12.0,
        shape="ellipse",
        text_height_factor=1.2,
    )

    assert width / 2.0 == pytest.approx(138.48179404552744)
    assert height == 36.0


def test_classic_sugiyama_uses_times_metrics_for_dot_cluster_labels() -> None:
    """Match DOT's default Times-Roman widths for cluster labels."""
    graph = DaguaGraph.from_edge_list([("source", "target")])
    graph.add_cluster("encoder", ["source", "target"], label="Encoder")
    graph.add_cluster("cross", ["source"], label="Cross Attention", parent="encoder")

    widths = classic_competitor._graphviz_dot_cluster_label_widths(graph)

    assert widths["encoder"] == 62
    assert round(widths["cross"]) == 104


def test_classic_sugiyama_enables_only_certified_cluster_inventory() -> None:
    """Enable the typed cluster path only for an instrumented exact oracle."""
    edges = [
        ("input", "embed"),
        ("embed", "router"),
        ("router", "expert_0"),
        ("router", "expert_3"),
        ("embed", "expert_1"),
        ("embed", "expert_2"),
        ("expert_0", "combine"),
        ("expert_1", "combine"),
        ("expert_2", "combine"),
        ("expert_3", "combine"),
        ("combine", "output"),
    ]
    graph = DaguaGraph.from_edge_list(edges)
    node_by_label = {label: index for index, label in enumerate(graph.node_labels)}
    graph.add_cluster(
        "experts",
        [node_by_label[f"expert_{index}"] for index in range(4)],
        label="Experts",
    )
    kwargs: dict[str, Any] = {}

    classic_competitor._apply_sugiyama_graphviz_metadata(graph=graph, extra_kwargs=kwargs)

    assert kwargs["graphviz_enable_cluster_skeleton"] is True
    assert kwargs["graphviz_expected_x_inventory"][0] == 28
    assert sum(record[2] for record in kwargs["graphviz_expected_x_inventory"][1]) == 36


def test_classic_sugiyama_graphviz_fidelity_guards_mixed_label_cluster_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Graphviz-fidelity Sugiyama should leave mixed label+cluster DOT unaugmented."""
    graph = DaguaGraph()
    graph.add_edge("source", "target", label="handoff label")
    graph.add_cluster("group", ["source", "target"])
    graph.compute_node_sizes()
    seen: dict[str, Any] = {}
    _install_classic_layout_spy(
        monkeypatch=monkeypatch,
        module_name="dagua.layout.ops.pipelines.sugiyama",
        fn_name="layout_sugiyama_pipeline",
        seen=seen,
    )

    result = classic_competitor.ClassicSugiyama().layout_with_variant(
        graph,
        seed=100,
        variant_params={"fidelity_mode": "graphviz"},
    )

    assert result.pos is not None
    kwargs = seen["kwargs"]
    assert isinstance(kwargs["graphviz_node_sizes"], torch.Tensor)
    assert "graphviz_edge_label_sizes" not in kwargs
    assert "clusters" not in kwargs
    assert "cluster_parents" not in kwargs
    assert "graphviz_apply_cluster_constraints" not in kwargs


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
    assert observed["import_path"] == "dagua.layout.ops.pipelines.sgd2_multi"
    assert observed["fn_name"] == "layout_sgd2_multi_pipeline"
    assert observed["seed"] == 17
    assert observed["extra_kwargs"] == {
        "criteria": {"stress": 1.0, "ideal_edge_length": 1.0},
        "lr": 0.01,
    }


def test_classic_pivot_mds_variant_does_not_forward_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Classic Pivot-MDS fidelity dispatch should match OGDF's unweighted BFS."""
    graph = _make_weighted_path_graph()
    seen: dict[str, dict[str, Any]] = {}
    _install_classic_layout_spy(
        monkeypatch=monkeypatch,
        module_name="dagua.layout.ops.pipelines.pivot_mds",
        fn_name="layout_pivot_mds_pipeline",
        seen=seen,
    )

    result = ClassicPivotMDS().layout_with_variant(
        graph,
        seed=17,
        variant_params={
            "n_pivots": 2,
            "first_pivot": "first_node",
            "compute_dtype": "float64",
            "distance_scale": 100.0,
            "ogdf_path_special_case": True,
        },
    )

    assert result.error is None
    assert seen["kwargs"] == {
        "n_pivots": 2,
        "first_pivot": "first_node",
        "compute_dtype": "float64",
        "distance_scale": 100.0,
        "ogdf_path_special_case": True,
    }


@pytest.mark.parametrize(
    ("competitor_factory", "module_name", "fn_name"),
    [
        (
            ClassicClassicalMDS,
            "dagua.layout.ops.pipelines.classical_mds",
            "layout_classical_mds_pipeline",
        ),
        (
            ClassicSGD2Multi,
            "dagua.layout.ops.pipelines.sgd2_multi",
            "layout_sgd2_multi_pipeline",
        ),
        (
            ClassicNeato,
            "dagua.layout.ops.pipelines.neato",
            "layout_neato_pipeline",
        ),
    ],
)
def test_reference_unweighted_classic_adapters_skip_edge_weights(
    monkeypatch: pytest.MonkeyPatch,
    competitor_factory: type[ClassicClassicalMDS] | type[ClassicSGD2Multi] | type[ClassicNeato],
    module_name: str,
    fn_name: str,
) -> None:
    """Reference-unweighted classic adapters should not forward edge weights."""
    graph = _make_weighted_path_graph()
    seen: dict[str, dict[str, Any]] = {}
    _install_classic_layout_spy(monkeypatch, module_name, fn_name, seen)

    result = competitor_factory().layout(graph, seed=7)

    assert result.error is None
    assert result.pos is not None
    assert "edge_weights" not in seen["kwargs"]


@pytest.mark.parametrize(
    ("competitor_factory", "module_name", "fn_name"),
    [
        (
            ClassicStressSGD,
            "dagua.layout.ops.pipelines.stress_sgd",
            "layout_stress_sgd_pipeline",
        ),
        (
            ClassicStressMajorization,
            "dagua.layout.ops.pipelines.stress_majorization",
            "layout_stress_majorization_pipeline",
        ),
        (
            ClassicMaxentStress,
            "dagua.layout.ops.pipelines.maxent_stress",
            "layout_maxent_stress_pipeline",
        ),
    ],
)
def test_stress_classic_adapters_still_forward_edge_weights(
    monkeypatch: pytest.MonkeyPatch,
    competitor_factory: type[ClassicStressSGD]
    | type[ClassicStressMajorization]
    | type[ClassicMaxentStress],
    module_name: str,
    fn_name: str,
) -> None:
    """Stress-family classic adapters should keep forwarding edge weights."""
    graph = _make_weighted_path_graph()
    seen: dict[str, dict[str, Any]] = {}
    _install_classic_layout_spy(monkeypatch, module_name, fn_name, seen)

    result = competitor_factory().layout(graph, seed=7)

    assert result.error is None
    assert result.pos is not None
    assert seen["kwargs"]["edge_weights"] is graph.edge_weights


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
    from dagua.eval.competitors import graphviz_competitor
    from dagua.eval.competitors.graphviz_competitor import GraphvizSfdp

    graph = _make_small_graph()
    observed: dict[str, float | str] = {}

    def _fake_layout_with_graphviz(
        graph: DaguaGraph,
        engine: str = "dot",
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Capture Graphviz utility arguments for the regression test.

        Parameters
        ----------
        graph : DaguaGraph
            Graph passed through the competitor.
        engine : str, default="dot"
            Requested Graphviz engine.
        timeout : float, default=300.0
            Requested timeout in seconds.
        seed : int | None, default=None
            Optional Graphviz seed.

        Returns
        -------
        torch.Tensor
            Dummy position tensor with shape ``[N, 2]``.
        """
        del seed
        observed["engine"] = engine
        observed["timeout"] = timeout
        return torch.zeros((graph.num_nodes, 2))

    monkeypatch.setattr(
        graphviz_competitor, "_layout_with_graphviz_engine", _fake_layout_with_graphviz
    )

    result = GraphvizSfdp().layout(graph, timeout=123.0)

    assert result.error is None
    assert observed == {"engine": "sfdp", "timeout": 123.0}


def test_graphviz_base_classifies_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Graphviz base competitors should normalize subprocess timeouts."""
    from dagua.eval.competitors import graphviz_competitor
    from dagua.eval.competitors.graphviz_competitor import GraphvizSfdp

    graph = _make_small_graph()

    def _fake_layout_with_graphviz(
        graph: DaguaGraph,
        engine: str = "dot",
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Raise a timeout to validate adapter error normalization.

        Parameters
        ----------
        graph : DaguaGraph
            Graph passed through the competitor.
        engine : str, default="dot"
            Requested Graphviz engine.
        timeout : float, default=300.0
            Requested timeout in seconds.
        seed : int | None, default=None
            Optional Graphviz seed.

        Returns
        -------
        torch.Tensor
            This helper never returns because it always raises.
        """
        del graph, engine, seed
        raise subprocess.TimeoutExpired(cmd="sfdp", timeout=timeout)

    monkeypatch.setattr(
        graphviz_competitor, "_layout_with_graphviz_engine", _fake_layout_with_graphviz
    )

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
