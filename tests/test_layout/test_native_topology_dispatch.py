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


def test_dispatch_routes_force_directed_family_to_common_contest() -> None:
    """A densely cyclic digraph follows the ruler into the common contest."""
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

    assert selected == "undirected_portfolio"


def test_dispatch_routes_karate_like_feedback_to_common_contest() -> None:
    """A sparse-feedback digraph follows the ruler into the common contest."""
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

    assert selected == "undirected_portfolio"


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


def test_declared_undirected_mesh_lattice_enters_common_contest() -> None:
    """Mesh-strong declared-undirected lattice DAGs enter the common contest.

    Router-v2 (native-sprint r2 wave 2) supersedes the r80-era exclusion that
    kept every lattice-tagged declared-undirected DAG on the layered baseline
    route: when the structural mesh features fire (near-constant degree, no
    hub tail, sqrt-N diameter), the graph re-enters the undirected portfolio
    contest. This is monotone-safe because the contest's candidate A IS the
    layered baseline route (including its polish battery) and ties go to the
    incumbent -- the property the old exclusion protected is preserved by
    construction, while stress-family candidates get a chance to win where
    the frozen common-table ruler says they are better.

    Returns
    -------
    None
        This test asserts the mesh-strong lattice DAG routes to the
        undirected portfolio while the incumbent guarantee holds.
    """
    from dagua.eval.graphs import get_test_graphs
    from dagua.layout.ops.pipelines.dagua_native import _mesh_features_strong

    graph = next(t.graph for t in get_test_graphs() if t.name == "hexagonal_lattice_42")
    structure = classify_graph(graph.edge_index, graph.num_nodes, graph=graph)
    config = LayoutConfig(seed=42)
    setattr(config, "_dagua_native_num_nodes", graph.num_nodes)

    assert structure.is_semantically_directed is False
    assert structure.direction_is_declared is True
    assert "lattice_like" in structure.topology_tags
    assert _mesh_features_strong(structure, graph.num_nodes)
    assert _choose_native_pipeline(structure, config) == "undirected_portfolio"

    # The old exclusion still protects lattice-tagged DAGs whose mesh
    # features do NOT fire (unknown node count keeps the gate closed).
    weak_config = LayoutConfig(seed=42)
    assert _choose_native_pipeline(structure, weak_config) == "layered_dag"


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
