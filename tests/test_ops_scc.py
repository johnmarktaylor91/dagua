"""Tests for SCC condensation and hybrid-v2 routing."""

from __future__ import annotations

import torch

from dagua.config import LayoutConfig
from dagua.layout.graph_classify import GraphFamily, GraphStructure
from dagua.layout.ops.pipelines import get_pipeline_function
from dagua.layout.ops.pipelines.dagua_native import _choose_native_pipeline
from dagua.layout.ops.scc import (
    SCCCondensation,
    SCCExpand,
    SCCPredicateStats,
    build_scc_condensation,
    hybrid_v2_predicate_matches,
)
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


def _component_sets(condensation: SCCCondensation) -> set[frozenset[int]]:
    """Return SCC components as an order-independent set.

    Parameters
    ----------
    condensation : SCCCondensation
        Condensation metadata under test.

    Returns
    -------
    set[frozenset[int]]
        Components as hashable sets.
    """
    return {frozenset(component) for component in condensation.components}


def _cyclic_structure(*, max_degree: int = 8) -> GraphStructure:
    """Return a semantically directed cyclic structure fixture.

    Parameters
    ----------
    max_degree : int, default=8
        Maximum degree to store on the fixture.

    Returns
    -------
    GraphStructure
        Directed cyclic graph classification fixture.
    """
    return GraphStructure(
        family=GraphFamily.HYBRID,
        num_components=1,
        max_degree=max_degree,
        num_layers=4,
        avg_layer_width=10.0,
        is_planar_hint=True,
        is_directed_acyclic=False,
        cyclicity_ratio=0.2,
        is_semantically_directed=True,
    )


def test_scc_condense_nested_cycles_builds_condensation_dag() -> None:
    """Nested cycle fixtures should condense to two SCC cores and a tail."""
    edge_index = torch.tensor(
        [
            [0, 1, 2, 2, 3, 4, 4],
            [1, 2, 0, 3, 4, 3, 5],
        ],
        dtype=torch.long,
    )

    condensation = build_scc_condensation(edge_index=edge_index, num_nodes=6)

    assert _component_sets(condensation) == {
        frozenset({0, 1, 2}),
        frozenset({3, 4}),
        frozenset({5}),
    }
    assert condensation.stats.covered_nodes == 5
    assert condensation.stats.max_scc_size == 3
    assert condensation.meta_edge_index.shape[0] == 2
    assert condensation.meta_edge_index.shape[1] == 2


def test_scc_condense_self_loop_counts_singleton_as_nontrivial() -> None:
    """Singleton self-loops should count as cyclic SCC coverage."""
    edge_index = torch.tensor([[0, 1], [0, 2]], dtype=torch.long)

    condensation = build_scc_condensation(edge_index=edge_index, num_nodes=3)

    assert _component_sets(condensation) == {
        frozenset({0}),
        frozenset({1}),
        frozenset({2}),
    }
    assert condensation.stats.covered_nodes == 1
    assert condensation.stats.nontrivial_count == 1
    assert condensation.stats.max_scc_size == 1


def test_scc_condense_all_singleton_dag_preserves_edges() -> None:
    """All-singleton DAGs should pass through as ordinary meta nodes."""
    edge_index = torch.tensor([[0, 0, 1], [1, 2, 3]], dtype=torch.long)

    condensation = build_scc_condensation(edge_index=edge_index, num_nodes=4)

    assert _component_sets(condensation) == {
        frozenset({0}),
        frozenset({1}),
        frozenset({2}),
        frozenset({3}),
    }
    assert condensation.stats.covered_nodes == 0
    assert condensation.stats.max_scc_size == 0
    assert condensation.meta_edge_multiplicity.tolist() == [1.0, 1.0, 1.0]


def test_scc_expand_applies_offsets_and_meta_layers() -> None:
    """Expansion should place members at meta position plus internal offset."""
    edge_index = torch.tensor([[0, 1, 1], [1, 0, 2]], dtype=torch.long)
    condensation = build_scc_condensation(edge_index=edge_index, num_nodes=3)
    offsets = []
    for members in condensation.components:
        if set(members) == {0, 1}:
            offsets.append(torch.tensor([[-10.0, 0.0], [10.0, 0.0]], dtype=torch.float32))
        else:
            offsets.append(torch.zeros((len(members), 2), dtype=torch.float32))
    meta_pos = torch.zeros((len(condensation.components), 2), dtype=torch.float32)
    for component_id, members in enumerate(condensation.components):
        if set(members) == {0, 1}:
            meta_pos[component_id] = torch.tensor([100.0, 50.0])
        else:
            meta_pos[component_id] = torch.tensor([300.0, 150.0])
    state = SolveState(
        extras={
            "scc_condensation": condensation,
            "scc_internal_offsets": tuple(offsets),
            "scc_meta_pos": meta_pos,
            "scc_bbox_sizes": torch.tensor([[40.0, 20.0], [20.0, 20.0]]),
        }
    )
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=3,
        node_sizes=torch.full((3, 2), 20.0),
    )
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))

    expanded = SCCExpand().apply(problem, state, ctx)

    assert expanded.pos is not None
    assert torch.allclose(expanded.pos.mean(dim=0), torch.zeros(2), atol=1.0e-5)
    assert expanded.layers is not None
    assert expanded.layers.shape == (3,)
    assert "scc_expanded_bbox" in expanded.extras


def test_hybrid_v2_predicate_is_evidence_gated_off_by_default() -> None:
    """The default route should stay on existing paths until evidence passes."""
    structure = _cyclic_structure()
    config = LayoutConfig()
    stats = SCCPredicateStats(
        total_nodes=20,
        covered_nodes=12,
        max_scc_size=12,
        coverage_ratio=0.6,
        nontrivial_count=1,
    )
    setattr(
        config,
        "_dagua_native_scc_stats",
        stats,
    )

    assert hybrid_v2_predicate_matches(stats)
    assert _choose_native_pipeline(structure, config) == "hybrid"
    setattr(config, "_dagua_native_enable_hybrid_v2_auto", True)
    assert _choose_native_pipeline(structure, config) == "hybrid_v2"


def test_hybrid_v2_predicate_leaves_dag_and_small_scc_graphs_untouched() -> None:
    """Pure DAGs and small SCC graphs should keep existing routes."""
    dag_structure = GraphStructure(
        family=GraphFamily.GENERAL,
        num_components=1,
        max_degree=3,
        num_layers=4,
        avg_layer_width=2.0,
        is_planar_hint=True,
        is_directed_acyclic=True,
        cyclicity_ratio=0.0,
        is_semantically_directed=True,
    )
    small_scc_config = LayoutConfig()
    setattr(
        small_scc_config,
        "_dagua_native_scc_stats",
        SCCPredicateStats(
            total_nodes=20,
            covered_nodes=12,
            max_scc_size=6,
            coverage_ratio=0.6,
            nontrivial_count=2,
        ),
    )

    assert _choose_native_pipeline(dag_structure, LayoutConfig()) == "layered_dag"
    assert _choose_native_pipeline(_cyclic_structure(), small_scc_config) == "hybrid"


def test_native_hybrid_v2_registry_dispatches() -> None:
    """The public pipeline registry should expose native_hybrid_v2."""
    pipeline_fn = get_pipeline_function("native_hybrid_v2")

    assert pipeline_fn.__name__ == "layout_native_hybrid_v2_pipeline"
