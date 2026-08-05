"""Degenerate-input regression pins for Family B reference ops.

These tests pin the empty-graph crash fixes from the GLaDOS-prep hardening
sweep: reference ports must tolerate zero-node problems (including the
0-element node-size tensors the engine produces for them) instead of
crashing with shape or reduction errors.
"""

from __future__ import annotations

import math

import torch

from dagua.layout.ops.d3dag import (
    D3DagCoordinate,
    D3DagDecross,
    D3DagLayering,
    D3DagPrepare,
    D3DagSugify,
)
from dagua.layout.ops.dagre import DagrePrepareGraph
from dagua.layout.ops.fcose import FCoSEPrepareState
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


def _context() -> RuntimeContext:
    """Return a CPU runtime context for deterministic op tests.

    Returns
    -------
    RuntimeContext
        Runtime context with a CPU execution plan.
    """
    return RuntimeContext(plan=ExecutionPlan(device="cpu"))


def _empty_problem() -> LayoutProblem:
    """Build a zero-node problem with a degenerate flat size tensor.

    Returns
    -------
    LayoutProblem
        Empty problem whose ``node_sizes`` has shape ``[0]`` (the shape the
        engine produces when computing sizes for zero nodes).
    """
    return LayoutProblem(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=0,
        node_sizes=torch.empty((0,), dtype=torch.float32),
    )


def test_dagre_prepare_accepts_zero_element_sizes_on_empty_graph() -> None:
    """Accept degenerate 0-element node sizes for the empty graph.

    Returns
    -------
    None
        Dagre preparation must succeed and stash a working graph.
    """
    state = DagrePrepareGraph().apply(_empty_problem(), SolveState(), _context())

    assert "dagre_graph" in state.extras


def test_d3dag_greedy_coordinate_tolerates_empty_layers() -> None:
    """Run the full d3-dag op chain with greedy coords on a zero-node graph.

    The d3dag pipeline guards ``num_nodes == 0`` before its ops run, so this
    frame is only reachable through direct op composition; the greedy layer
    spacing must skip empty layers instead of indexing into them.

    Returns
    -------
    None
        The chain must produce an empty ``[0, 2]`` position tensor.
    """
    problem = _empty_problem()
    state = SolveState(pos=torch.zeros((0, 2), dtype=torch.float64))

    state = D3DagPrepare().apply(problem, state, _context())
    state = D3DagLayering().apply(problem, state, _context())
    state = D3DagSugify().apply(problem, state, _context())
    state = D3DagDecross().apply(problem, state, _context())
    state = D3DagCoordinate(method="greedy").apply(problem, state, _context())

    assert state.pos is not None
    assert tuple(state.pos.shape) == (0, 2)


def test_fcose_prepare_state_tolerates_empty_positions() -> None:
    """Initialize fCoSE spring state on a zero-node problem.

    Returns
    -------
    None
        The span reduction over zero positions must not raise, and the
        resulting temperature must be finite and positive.
    """
    state = SolveState(pos=torch.zeros((0, 2), dtype=torch.float64))

    result = FCoSEPrepareState().apply(_empty_problem(), state, _context())

    assert result.temperature is not None
    assert math.isfinite(float(result.temperature))
    assert float(result.temperature) > 0.0
