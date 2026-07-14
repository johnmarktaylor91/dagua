"""Unit tests for reusable d3-dag Sugiyama operations."""

from __future__ import annotations

import torch

from dagua.layout.ops.d3dag import (
    D3DagCoffmanGrahamLayering,
    D3DagDecross,
    D3DagLayering,
    D3DagOptimalCrossingOrder,
    D3DagPrepare,
    D3DagSugify,
)
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


def _context() -> RuntimeContext:
    """Return a CPU runtime context for op tests.

    Returns
    -------
    RuntimeContext
        Minimal execution context.
    """
    return RuntimeContext(plan=ExecutionPlan(device="cpu"))


def test_coffman_graham_layering_respects_width_bound() -> None:
    """Layer a fanout DAG without exceeding the configured width.

    Returns
    -------
    None
        Every layer contains at most two nodes.
    """
    edge_index = torch.tensor([[0, 0, 0, 0, 1, 2], [1, 2, 3, 4, 5, 5]], dtype=torch.long)
    problem = LayoutProblem(edge_index=edge_index, num_nodes=6)
    state = D3DagCoffmanGrahamLayering(width=2).apply(problem, SolveState(), _context())
    assert state.layers is not None
    layers = state.layers.tolist()
    for layer in set(layers):
        assert layers.count(layer) <= 2
    for source, target in edge_index.t().tolist():
        assert layers[target] > layers[source]


def test_optimal_crossing_order_reduces_crossings_on_two_layer_case() -> None:
    """Exact decrossing removes the classic two-edge crossing.

    Returns
    -------
    None
        The optimized lower layer order swaps to remove the crossing.
    """
    edge_index = torch.tensor([[0, 1], [3, 2]], dtype=torch.long)
    node_sizes = torch.ones((4, 2), dtype=torch.float64)
    problem = LayoutProblem(edge_index=edge_index, num_nodes=4, node_sizes=node_sizes)
    state = SolveState()
    for op in (D3DagPrepare(), D3DagLayering(method="longestPath"), D3DagSugify()):
        state = op.apply(problem, state, _context())

    before = [layer[:] for layer in state.extras["d3dag_graph"].layers]
    state = D3DagOptimalCrossingOrder().apply(problem, state, _context())
    after = state.extras["d3dag_stage_ordering"]

    assert before[1] == [2, 3]
    assert after[1] == [3, 2]


def test_decross_opt_path_is_available_through_stage_op() -> None:
    """Run exact crossing minimization through ``D3DagDecross``.

    Returns
    -------
    None
        The public decross op accepts ``method='opt'``.
    """
    edge_index = torch.tensor([[0, 1], [3, 2]], dtype=torch.long)
    problem = LayoutProblem(edge_index=edge_index, num_nodes=4, node_sizes=torch.ones((4, 2)))
    state = SolveState()
    for op in (
        D3DagPrepare(),
        D3DagLayering(method="longestPath"),
        D3DagSugify(),
        D3DagDecross(method="opt"),
    ):
        state = op.apply(problem, state, _context())
    assert state.extras["d3dag_stage_ordering"][1] == [3, 2]
