"""Op-level regression tests for ELK Layered stages."""

from __future__ import annotations

import pytest
import torch

from dagua.layout.ops.elk import ElkAssignLayers, ElkBreakCycles, ElkPrepareGraph
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


def _context() -> RuntimeContext:
    """Return a CPU runtime context for deterministic op tests.

    Returns
    -------
    RuntimeContext
        Runtime context with a CPU execution plan.
    """
    return RuntimeContext(plan=ExecutionPlan(device="cpu"))


def test_elk_cycle_breaking_makes_cycle_layerable() -> None:
    """Break a directed 3-cycle before layer assignment.

    Returns
    -------
    None
        The active edge set must become compatible with increasing layers.
    """
    problem = LayoutProblem(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        num_nodes=3,
        node_sizes=torch.tensor([[40.0, 20.0]] * 3),
    )
    state = ElkPrepareGraph().apply(problem, SolveState(), _context())
    state = ElkBreakCycles().apply(problem, state, _context())
    state = ElkAssignLayers().apply(problem, state, _context())

    layers = {
        node: layer for layer, nodes in enumerate(state.extras["elk_layers"]) for node in nodes
    }
    for source, target in state.extras["elk_graph"].active_edges:
        assert layers[source] < layers[target]


def test_elk_prepare_rejects_invalid_strategy() -> None:
    """Reject unknown ELK strategy names at construction time.

    Returns
    -------
    None
        Invalid public options must raise ``ValueError``.
    """
    with pytest.raises(ValueError, match="layering_strategy"):
        ElkPrepareGraph(layering_strategy="definitely-not-elk")


def test_elk_prepare_accepts_direction_alias() -> None:
    """Accept dagua/dagre-style direction aliases.

    Returns
    -------
    None
        ``TB`` must normalize to ELK ``DOWN``.
    """
    problem = LayoutProblem(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=1,
        node_sizes=torch.tensor([[40.0, 20.0]]),
    )
    state = ElkPrepareGraph(direction="TB").apply(problem, SolveState(), _context())

    assert state.extras["elk_graph"].direction == "DOWN"
