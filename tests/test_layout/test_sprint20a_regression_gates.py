"""Sprint-20a regression coverage for native DAG heuristic gates."""

from __future__ import annotations

import copy

import pytest
import torch

from dagua.config import LayoutConfig
from dagua.eval.graphs import _make_sierpinski_graph, _random_dag, get_test_graphs
from dagua.graph import DaguaGraph
from dagua.layout import layout
from dagua.layout.graph_classify import GraphFamily, GraphStructure, classify_graph
from dagua.layout.ops.pipelines.dagua_native import (
    _should_apply_brandes_koepf_refine,
    _should_use_native_dummy_nodes,
    _should_use_native_median_transpose,
    build_dagua_pipeline,
)
from dagua.layout.resolve import prepare_pipeline_config
from dagua.metrics import evaluate


def _edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    """Build a directed edge tensor from Python edge tuples.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Directed edges.

    Returns
    -------
    torch.Tensor
        Long edge tensor with shape ``[2, E]``.
    """
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)
    sources, targets = zip(*edges)
    return torch.tensor([list(sources), list(targets)], dtype=torch.long)


def _general_dag_structure(
    num_layers: int,
    max_layer_width: int,
    avg_layer_width: float = 1.0,
) -> GraphStructure:
    """Build a minimal connected DAG structure for native gate tests.

    Parameters
    ----------
    num_layers : int
        Number of classified layers.
    max_layer_width : int
        Maximum number of nodes in any classified layer.
    avg_layer_width : float, default=1.0
        Average classified layer width.

    Returns
    -------
    GraphStructure
        Structure metadata shaped like the native classifier output.
    """
    return GraphStructure(
        family=GraphFamily.GENERAL,
        num_components=1,
        max_degree=4,
        num_layers=num_layers,
        avg_layer_width=avg_layer_width,
        is_planar_hint=True,
        is_acyclic=True,
        max_layer_width=max_layer_width,
        is_directed_acyclic=True,
    )


def _benchmark_graph(name: str) -> DaguaGraph:
    """Return a benchmark graph copy with node sizes populated.

    Parameters
    ----------
    name : str
        Benchmark graph name.

    Returns
    -------
    DaguaGraph
        Graph ready for native layout and metric evaluation.
    """
    graphs = {entry.name: entry.graph for entry in get_test_graphs(max_nodes=250)}
    graph = copy.deepcopy(graphs[name])
    graph.compute_node_sizes()
    return graph


def _resolved_config(graph: DaguaGraph) -> LayoutConfig:
    """Return native pipeline config resolved for one graph.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to classify and resolve.

    Returns
    -------
    LayoutConfig
        Shallow config copy annotated with native private attrs.
    """
    return prepare_pipeline_config(
        config=LayoutConfig(algorithm="dagua_native", seed=42, device="cpu"),
        num_nodes=graph.num_nodes,
        edge_index=graph.edge_index,
        device="cpu",
        layer_assignments=None,
        prebuilt_layer_index=None,
        graph_structure=None,
        skip_classification=False,
    )


def _composite_score(graph: DaguaGraph) -> float:
    """Return the native composite score for a benchmark graph.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to lay out and score.

    Returns
    -------
    float
        Full-tier composite score.
    """
    pos = layout(
        graph,
        LayoutConfig(algorithm="dagua_native", seed=42, device="cpu"),
    )
    return float(evaluate(graph, pos, tier="full")["composite_score"])


def test_single_width_chain_without_long_edges_skips_dummy_nodes() -> None:
    """Single-node layers without skip edges should not trigger dummy expansion."""
    layers = torch.arange(20, dtype=torch.long)
    edge_index = _edge_index([(index, index + 1) for index in range(19)])
    structure = _general_dag_structure(num_layers=20, max_layer_width=1)
    config = LayoutConfig(insert_dummy_nodes=True, brandes_koepf_refine=True)

    assert not _should_use_native_dummy_nodes(
        config=config,
        structure=structure,
        edge_index=edge_index,
        layer_assignments=layers,
    )
    assert _should_apply_brandes_koepf_refine(
        config=config,
        structure=structure,
        layer_assignments=layers,
    )


def test_single_width_chain_with_skip_edges_uses_dummy_nodes_and_bk_refine() -> None:
    """Single-node layers with skip edges should admit dummy expansion and BK."""
    layers = torch.arange(20, dtype=torch.long)
    edge_index = _edge_index([(0, 2)])
    structure = _general_dag_structure(num_layers=20, max_layer_width=1)
    config = LayoutConfig(insert_dummy_nodes=True, brandes_koepf_refine=True)

    assert _should_use_native_dummy_nodes(
        config=config,
        structure=structure,
        edge_index=edge_index,
        layer_assignments=layers,
    )
    assert _should_apply_brandes_koepf_refine(
        config=config,
        structure=structure,
        layer_assignments=layers,
    )


def test_tiny_dags_skip_median_transpose_until_node_31() -> None:
    """Median/transpose should stay off through N=30 and re-enter at N=31."""
    structure = _general_dag_structure(num_layers=6, max_layer_width=6, avg_layer_width=5.0)
    config_30 = LayoutConfig(use_native_median_transpose=True)
    config_31 = LayoutConfig(use_native_median_transpose=True)
    setattr(config_30, "_dagua_native_num_nodes", 30)
    setattr(config_30, "_dagua_native_structure", structure)
    setattr(config_31, "_dagua_native_num_nodes", 31)
    setattr(config_31, "_dagua_native_structure", structure)

    assert not _should_use_native_median_transpose(config=config_30, is_acyclic=True)
    assert _should_use_native_median_transpose(config=config_31, is_acyclic=True)

    op_names_30 = [op.name for op in build_dagua_pipeline(config_30).ops]
    op_names_31 = [op.name for op in build_dagua_pipeline(config_31).ops]

    assert "median_sweep" not in op_names_30
    assert "transpose_heuristic" not in op_names_30
    assert "median_sweep" in op_names_31
    assert "transpose_heuristic" in op_names_31


@pytest.mark.parametrize(
    ("graph_name", "pre_patch_plus_two_floor"),
    [
        ("planar_60", 67.82),
        ("ragged_feature_pyramid", 71.52),
        ("regular_3_30", 70.37),
    ],
)
def test_sprint20a_target_graphs_clear_pre_patch_plus_two_floor(
    graph_name: str,
    pre_patch_plus_two_floor: float,
) -> None:
    """Each target graph should recover at least two composite points."""
    graph = _benchmark_graph(graph_name)

    assert _composite_score(graph) >= pre_patch_plus_two_floor


def test_planar_dag_aspect_uses_045_without_retuning_random_dag() -> None:
    """Planar DAGs with real size get 0.45 while random DAGs keep 0.25."""
    planar_graph = _make_sierpinski_graph(depth=3)
    random_graph = _random_dag(200, 300, seed=42)
    planar_structure = classify_graph(planar_graph.edge_index, planar_graph.num_nodes)
    random_structure = classify_graph(random_graph.edge_index, random_graph.num_nodes)

    assert planar_graph.num_nodes >= 20
    assert "planar_dag" in planar_structure.topology_tags
    assert random_structure.topology_tags == ()
    assert getattr(_resolved_config(planar_graph), "_dagua_native_target_aspect") == pytest.approx(
        0.45,
    )
    assert getattr(_resolved_config(random_graph), "_dagua_native_target_aspect") == pytest.approx(
        0.25,
    )
