"""Tests for layering ops."""

from __future__ import annotations

import torch

from dagua.layout.ops import Pipeline
from dagua.layout.ops.coordinate import BrandesKopf4Pass
from dagua.layout.ops.layering import (
    BuildLayerIndex,
    BuildLayerIndexConfig,
    InsertDummyNodes,
    LayerPromotion,
    LongestPathLayering,
)
from dagua.layout.ops.ordering import BarycenterSweep
from dagua.layout.ops.postprocess import StripDummyNodes
from dagua.layout.ops.preprocess import BuildAdjacency, MakeAcyclic
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState


def _make_dag_problem() -> LayoutProblem:
    """Build a small DAG used across layering tests.

    Returns
    -------
    LayoutProblem
        Ten-node DAG with several merge and skip edges.
    """
    edge_index = torch.tensor(
        [
            [0, 0, 1, 2, 3, 4, 6, 5, 7, 8],
            [1, 3, 2, 4, 4, 5, 7, 8, 8, 9],
        ],
        dtype=torch.long,
    )
    return LayoutProblem(
        edge_index=edge_index,
        num_nodes=10,
        node_sizes=torch.ones((10, 2), dtype=torch.float32),
    )


def _edge_list(edge_index: torch.Tensor) -> list[tuple[int, int]]:
    """Convert an edge tensor to a Python edge list.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.

    Returns
    -------
    list[tuple[int, int]]
        Edge list in column order.
    """
    return [(int(src), int(dst)) for src, dst in edge_index.t().tolist()]


def test_longest_path_layering_produces_valid_dag_layers() -> None:
    """Longest-path layering should respect every DAG edge."""
    problem = _make_dag_problem()
    state = LongestPathLayering().apply(problem, SolveState(), RuntimeContext())

    assert state.layers is not None
    assert state.layers.shape == (10,)
    assert int(state.layers[0].item()) == 0

    for src, dst in _edge_list(problem.edge_index):
        assert int(state.layers[dst].item()) > int(state.layers[src].item())


def test_layer_promotion_pushes_nodes_to_deepest_legal_layer() -> None:
    """Promotion should deepen nodes when successors leave slack."""
    problem = LayoutProblem(
        edge_index=torch.tensor(
            [
                [0, 1, 2],
                [3, 2, 3],
            ],
            dtype=torch.long,
        ),
        num_nodes=4,
    )
    ctx = RuntimeContext()
    state = LongestPathLayering().apply(problem, SolveState(), ctx)

    assert state.layers is not None
    assert state.layers.tolist() == [0, 0, 1, 2]

    promoted = LayerPromotion().apply(problem, state, ctx)

    assert promoted.layers is not None
    assert promoted.layers.tolist() == [1, 0, 1, 2]
    for src, dst in _edge_list(problem.edge_index):
        assert int(promoted.layers[dst].item()) > int(promoted.layers[src].item())


def test_build_layer_index_preserves_stable_node_order_within_layers() -> None:
    """Layer-index sorting should remain stable within each layer."""
    state = SolveState(layers=torch.tensor([2, 0, 1, 2, 1, 0], dtype=torch.long))
    op = BuildLayerIndex(config=BuildLayerIndexConfig(enable_cuda_sort=False))
    updated = op.apply(
        LayoutProblem(edge_index=torch.zeros((2, 0), dtype=torch.long), num_nodes=6),
        state,
        RuntimeContext(),
    )

    assert updated.layer_index is not None
    assert updated.layer_index.sorted_nodes.tolist() == [1, 5, 2, 4, 0, 3]
    assert updated.layer_index.layer_offsets.tolist() == [0, 2, 4, 6]
    assert updated.layer_index.nodes_in_layer(1).tolist() == [2, 4]


def test_insert_dummy_nodes_expands_only_long_edges() -> None:
    """Dummy-node insertion should split multi-layer edges into unit steps."""
    problem = LayoutProblem(
        edge_index=torch.tensor(
            [
                [0, 1, 2, 0],
                [1, 2, 3, 3],
            ],
            dtype=torch.long,
        ),
        num_nodes=4,
        node_sizes=torch.ones((4, 2), dtype=torch.float32),
    )
    ctx = RuntimeContext()
    state = LongestPathLayering().apply(problem, SolveState(), ctx)
    state = InsertDummyNodes().apply(problem, state, ctx)

    expanded = state.extras["expanded_graph"]
    assert expanded.num_nodes == 6
    assert len(expanded.edge_paths) == problem.edge_index.shape[1]
    assert max(len(path) for path in expanded.edge_paths) == 4

    layer_by_node: dict[int, int] = {}
    for layer_index, nodes in enumerate(expanded.layers):
        for node in nodes:
            layer_by_node[node] = layer_index

    for src, dst in _edge_list(expanded.edge_index):
        assert layer_by_node[dst] - layer_by_node[src] == 1

    assert torch.all(expanded.node_sizes[problem.num_nodes :] == 0)


def test_insert_dummy_nodes_expands_skip_edges_into_full_paths() -> None:
    """Dummy insertion should add one intermediate node per skipped layer."""

    problem = LayoutProblem(
        edge_index=torch.tensor([[0, 0], [1, 3]], dtype=torch.long),
        num_nodes=4,
        node_sizes=torch.ones((4, 2), dtype=torch.float32),
    )
    state = SolveState(layers=torch.tensor([0, 1, 2, 3], dtype=torch.long))

    result = InsertDummyNodes().apply(problem, state, RuntimeContext())

    expanded = result.extras["expanded_graph"]
    assert expanded.num_nodes == 6
    assert expanded.edge_paths[0] == [0, 1]
    assert expanded.edge_paths[1] == [0, 4, 5, 3]


def test_full_sugiyama_pipeline_produces_finite_original_node_positions() -> None:
    """The composed Sugiyama pipeline should produce finite positions for original nodes."""

    problem = LayoutProblem(
        edge_index=torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]], dtype=torch.long),
        num_nodes=4,
        node_sizes=torch.ones((4, 2), dtype=torch.float32),
    )
    pipeline = Pipeline(
        [
            MakeAcyclic(),
            BuildAdjacency(),
            LongestPathLayering(),
            LayerPromotion(),
            InsertDummyNodes(),
            BarycenterSweep(),
            BrandesKopf4Pass(),
            StripDummyNodes(),
        ]
    )

    result = pipeline.apply(
        problem,
        SolveState(back_edge_mask=torch.zeros(problem.edge_index.shape[1], dtype=torch.bool)),
        RuntimeContext(),
    )

    assert result.pos is not None
    assert result.pos.shape == (problem.num_nodes, 2)
    assert torch.isfinite(result.pos).all()
    assert result.layers is not None
    assert result.ordering is not None
