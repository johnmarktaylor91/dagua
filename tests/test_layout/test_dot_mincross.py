"""Regression tests for Graphviz dot mincross fidelity helpers."""

from __future__ import annotations

import pytest
import torch

from dagua.eval.competitors.classic_competitor import (
    _graphviz_dot_node_box,
    _graphviz_dot_node_sizes,
)
from dagua.eval.graphs import get_test_graphs
from dagua.layout.ops._dot_mincross import graphviz_mincross
from dagua.layout.ops.pipelines.sugiyama import build_sugiyama_pipeline, layout_sugiyama_pipeline
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.sugiyama import (
    _expand_long_edges_with_dummy_nodes,
    _graphviz_contain_cluster_ordering,
    _graphviz_decompose_node_order,
)


def _cluster_skeleton_visible_order(graph_name: str) -> list[list[int]]:
    """Return visible rank order from the inactive A12 cluster skeleton path.

    Parameters
    ----------
    graph_name : str
        Name from the evaluation graph registry.

    Returns
    -------
    list[list[int]]
        Original-node ids by non-empty expanded rank.
    """
    graphs = {test_graph.name: test_graph.graph for test_graph in get_test_graphs(max_nodes=300)}
    graph = graphs[graph_name]
    state = SolveState()
    state.extras["sugiyama_graphviz_node_sizes"] = _graphviz_dot_node_sizes(graph=graph)
    problem = LayoutProblem(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        node_sizes=graph.node_sizes,
        clusters=graph.clusters,
        cluster_parents=graph.cluster_parents,
    )
    pipeline = build_sugiyama_pipeline(
        rank_sep=1.0,
        node_sep=1.0,
        fidelity_mode="graphviz",
        center_coordinates=False,
        graphviz_enable_cluster_skeleton=True,
    )
    final_state = pipeline.apply(problem, state, RuntimeContext(plan=ExecutionPlan()))
    ordered_layers = final_state.extras["sugiyama_ordered_layers"]
    return [
        [int(node) for node in layer if int(node) < graph.num_nodes]
        for layer in ordered_layers
        if any(int(node) < graph.num_nodes for node in layer)
    ]


def test_graphviz_mincross_two_rank_golden_order() -> None:
    """Match a two-rank golden ordering captured from Graphviz 7.0.5 dot."""
    ranks = [[0, 1], [2, 3]]
    edges = torch.tensor([[0, 1], [3, 2]], dtype=torch.long)

    ordering = graphviz_mincross(ranks=ranks, edges=edges, iterations=24)

    assert ordering == [[0, 1], [3, 2]]


def test_graphviz_mincross_three_rank_golden_order() -> None:
    """Match a three-rank golden ordering captured from Graphviz 7.0.5 dot."""
    ranks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    edges = [
        (0, 4),
        (1, 3),
        (2, 5),
        (3, 7),
        (4, 6),
        (5, 8),
    ]

    ordering = graphviz_mincross(ranks=ranks, edges=edges, iterations=24)

    assert ordering == [[0, 1, 2], [4, 3, 5], [6, 7, 8]]


def test_graphviz_mincross_ignores_unexpanded_long_edges() -> None:
    """Require callers to pass dot-style adjacent-rank virtual-node chains."""
    ranks = [[0], [1], [2]]
    edges = [(0, 2)]

    ordering = graphviz_mincross(ranks=ranks, edges=edges, iterations=24)

    assert ordering == ranks


def test_graphviz_mincross_counts_sparse_wide_rank_edges() -> None:
    """Count crossings when lower-rank node order exceeds edge count."""
    ranks = [[0, 1], [2, 3, 4, 5, 6]]
    edges = [(0, 6), (1, 2)]

    ordering = graphviz_mincross(ranks=ranks, edges=edges, iterations=1)

    assert ordering[0] == [0, 1]
    assert sorted(ordering[1]) == [2, 3, 4, 5, 6]


def test_graphviz_mincross_orders_weak_components_independently() -> None:
    """Concatenate independently minimized components in decomposition order."""
    ranks = [[0, 2], [1, 3]]
    edges = [(0, 1), (2, 3)]

    ordering = graphviz_mincross(
        ranks=ranks,
        edges=edges,
        node_order=[2, 3, 0, 1],
    )

    assert ordering == [[2, 0], [3, 1]]


def test_graphviz_cluster_containment_keeps_rank_blocks_contiguous() -> None:
    """Collect same-rank cluster members into one mincross block."""
    ranks = [[0, 1, 2, 3], [4, 5]]

    ordering = _graphviz_contain_cluster_ordering(
        ranks=ranks,
        graphviz_cluster_members={"cluster_c": (0, 2, 3)},
    )

    assert ordering == [[0, 2, 3, 1], [4, 5]]


def test_graphviz_cluster_skeleton_flag_preserves_interleaved_order() -> None:
    """Keep the A12c interleaved-cluster ordering verifier behind a flag."""
    assert _cluster_skeleton_visible_order("interleaved_cluster_crosstalk") == [
        [0],
        [2, 1],
        [5, 3, 8, 9],
        [6, 4, 10],
        [11, 7],
    ]


def test_graphviz_cluster_skeleton_flag_preserves_platform_order() -> None:
    """Pin the inactive platform skeleton state without the removed x tie."""
    assert _cluster_skeleton_visible_order("kitchen_sink_platform_graph") == [
        [12, 16],
        [13, 17],
        [0, 14],
        [1, 15],
        [2],
        [3, 5, 4],
        [6],
        [7, 10],
        [8, 11],
        [9],
    ]


def test_graphviz_decompose_order_discovers_virtual_nodes_from_real_roots() -> None:
    """Match Graphviz 7.0.5 ``decompose(g, 1)`` DFS root and edge order."""
    edge_index = torch.tensor(
        [
            [0, 0, 2, 1],
            [2, 3, 4, 4],
        ],
        dtype=torch.long,
    )

    node_order = _graphviz_decompose_node_order(
        edge_index=edge_index,
        num_nodes=5,
        num_original_nodes=2,
    )

    assert node_order == [0, 2, 4, 1, 3]


def test_sugiyama_graphviz_fidelity_ignores_benchmark_edge_weights() -> None:
    """Mirror the benchmark DOT adapter, which omits ``weight=`` attributes."""
    edge_index = torch.tensor(
        [
            [0, 0, 1, 2],
            [1, 2, 3, 3],
        ],
        dtype=torch.long,
    )
    edge_weights = torch.tensor([100.0, 1.0, 5.0, 1.0], dtype=torch.float32)

    weighted_positions = layout_sugiyama_pipeline(
        edge_index=edge_index,
        num_nodes=4,
        edge_weights=edge_weights,
        fidelity_mode="graphviz",
        center_coordinates=False,
    )
    unit_positions = layout_sugiyama_pipeline(
        edge_index=edge_index,
        num_nodes=4,
        edge_weights=None,
        fidelity_mode="graphviz",
        center_coordinates=False,
    )

    assert torch.equal(weighted_positions, unit_positions)


def test_graphviz_dot_node_box_matches_default_ellipse_widths() -> None:
    """Use Graphviz point-unit boxes instead of Dagua's narrower theme boxes."""
    numeric_width, numeric_height = _graphviz_dot_node_box("10", 12.0, "ellipse")
    pair_width, pair_height = _graphviz_dot_node_box("pair_10", 12.0, "ellipse")

    assert numeric_width == pytest.approx(54.0)
    assert numeric_height == pytest.approx(36.0)
    assert pair_width == pytest.approx(69.9227938258)
    assert pair_height == pytest.approx(36.0)


def test_graphviz_duplicate_long_edges_inflate_virtual_width() -> None:
    """Mirror ``merge_chain()``, which widens virtual nodes on merged chains."""
    edge_index = torch.tensor([[0, 0], [2, 2]], dtype=torch.long)
    layers = torch.tensor([0, 1, 2], dtype=torch.long)
    node_sizes = torch.full((3, 2), 54.0, dtype=torch.float32)

    expanded_graph, _ = _expand_long_edges_with_dummy_nodes(
        edge_index=edge_index,
        layer_assignments=layers,
        node_sizes=node_sizes,
        num_original_nodes=3,
        use_graphviz_edge_order=True,
        graphviz_virtual_node_sep=72.0,
    )

    assert expanded_graph.edge_paths == [[0, 3, 2], [0, 3, 2]]
    assert float(expanded_graph.node_sizes[3, 0].item()) == pytest.approx(146.0)


def test_sugiyama_dot_fidelity_uses_graphviz_mincross() -> None:
    """Run the public Sugiyama pipeline with Graphviz dot mincross enabled."""
    edge_index = torch.tensor([[0, 1], [3, 2]], dtype=torch.long)

    positions = layout_sugiyama_pipeline(
        edge_index=edge_index,
        num_nodes=4,
        fidelity_mode="dot",
        center_coordinates=False,
    )

    assert isinstance(positions, torch.Tensor)
    assert float(positions[3, 0].item()) < float(positions[2, 0].item())
