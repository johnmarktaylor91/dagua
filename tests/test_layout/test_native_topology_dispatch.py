"""Regression tests for sprint-20e native topology dispatch."""

from __future__ import annotations

import torch

from dagua.config import LayoutConfig
from dagua.layout.graph_classify import GraphFamily, GraphStructure, classify_graph
from dagua.layout.ops.pipelines.dagua_native import (
    _choose_native_pipeline,
    layout_dagua_native_pipeline,
)
from dagua.layout.ops.pipelines.dagua_native_legacy import (
    layout_dagua_native_pipeline as layout_legacy_native_pipeline,
)
from dagua.metrics import composite, full


def _ring_edges(num_nodes: int) -> torch.Tensor:
    """Return a directed ring edge tensor.

    Parameters
    ----------
    num_nodes : int
        Number of ring nodes.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, N]``.
    """
    sources = torch.arange(num_nodes, dtype=torch.long)
    targets = (sources + 1) % num_nodes
    return torch.stack([sources, targets])


def test_classifier_exposes_dispatch_fields_for_cyclic_graph() -> None:
    """Cyclic graphs should expose feedback and effective-layer metadata."""
    result = classify_graph(_ring_edges(12), 12)

    assert result.family in {GraphFamily.HYBRID, GraphFamily.FORCE_DIRECTED}
    assert result.cyclicity_ratio > 0.0
    assert result.num_layers_effective >= 1
    assert result.has_dominant_component


def test_dispatch_routes_force_directed_family_to_force_pipeline() -> None:
    """FORCE_DIRECTED + high cyclicity should select the force sub-pipeline.

    Sprint-20g tightened the auto-route gate from ``cyclicity_ratio > 0.3``
    to ``cyclicity_ratio > 0.5`` after benchmarking found force_directed
    losing to layered_dag/hybrid on every cyclic candidate at the lower
    threshold (recurrent_feedback_cell, kitchen_sink_*, parallel_cycles).
    The strict gate keeps the force_directed branch reachable by its
    intended target (densely-cyclic non-hierarchical graphs) while
    excluding mostly-DAG graphs with sparse back-edges.
    """
    structure = GraphStructure(
        family=GraphFamily.FORCE_DIRECTED,
        num_components=1,
        max_degree=6,
        num_layers=1,
        avg_layer_width=100.0,
        is_planar_hint=False,
        is_directed_acyclic=False,
        cyclicity_ratio=0.6,
    )

    selected = _choose_native_pipeline(structure, LayoutConfig())

    assert selected == "force_directed"


def test_dispatch_routes_karate_like_feedback_to_hybrid() -> None:
    """Low-feedback cyclic directed graphs should select the hybrid path."""
    structure = GraphStructure(
        family=GraphFamily.HYBRID,
        num_components=1,
        max_degree=8,
        num_layers=5,
        avg_layer_width=7.0,
        is_planar_hint=True,
        is_directed_acyclic=False,
        cyclicity_ratio=0.1,
    )

    selected = _choose_native_pipeline(structure, LayoutConfig())

    assert selected == "hybrid"


def test_force_pipeline_legacy_monolith_matches_legacy_module() -> None:
    """The legacy_monolith escape hatch should preserve sprint-20d output."""
    edge_index = torch.tensor([[0, 1, 2, 0], [1, 2, 3, 3]], dtype=torch.long)
    node_sizes = torch.full((4, 2), 20.0, dtype=torch.float32)
    config = LayoutConfig(seed=42, steps=5, force_pipeline="legacy_monolith")

    actual = layout_dagua_native_pipeline(
        edge_index=edge_index,
        num_nodes=4,
        node_sizes=node_sizes,
        config=config,
    )
    expected = layout_legacy_native_pipeline(
        edge_index=edge_index,
        num_nodes=4,
        node_sizes=node_sizes,
        config=config,
    )

    assert torch.allclose(actual, expected)


def test_force_pipeline_planar_runs_planar_pipeline_not_layered_fallback() -> None:
    """force_pipeline='planar' must invoke the planar pipeline, not layered_dag.

    Sprint-20g originally registered the planar branch in
    ``_choose_native_pipeline`` but forgot to add the matching arm in
    ``build_dagua_pipeline``, so every planar dispatch silently fell through
    to ``build_native_layered_dag_pipeline``. This test pins the wiring to
    catch a regression.
    """
    from dagua.eval.graphs import get_test_graphs

    graph = next(t.graph for t in get_test_graphs() if t.name == "hexagonal_lattice_42")
    graph.compute_node_sizes()

    planar_pos = layout_dagua_native_pipeline(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        node_sizes=graph.node_sizes,
        config=LayoutConfig(seed=42, force_pipeline="planar"),
        seed=42,
    )
    layered_pos = layout_dagua_native_pipeline(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        node_sizes=graph.node_sizes,
        config=LayoutConfig(seed=42, force_pipeline="layered_dag"),
        seed=42,
    )

    assert not torch.equal(planar_pos, layered_pos)


def test_native_default_hexagonal_lattice_polish_score_stays_high() -> None:
    """Sprint-21a polish candidates should keep hex lattice out of the loss band.

    Returns
    -------
    None
        This test asserts that the default native pipeline still produces a
        high composite score under the stress-inclusive metric.
    """
    from dagua.eval.graphs import get_test_graphs
    from dagua.layout.engine import layout as engine_layout

    graph = next(t.graph for t in get_test_graphs() if t.name == "hexagonal_lattice_42")
    graph.compute_node_sizes()

    pos = engine_layout(graph, LayoutConfig(seed=42))
    torch.manual_seed(0)
    score = float(composite(full(pos, graph.edge_index, node_sizes=graph.node_sizes)))

    assert 80.0 < score < 100.0
