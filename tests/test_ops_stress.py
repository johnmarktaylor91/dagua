"""Tests for stress-majorization primitive operations."""

from __future__ import annotations

import pytest
import torch

from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.stress import (
    FinalizeStressMajorizationPositions,
    SmacofStep,
)


def _triangle_problem() -> LayoutProblem:
    """Build a 3-node triangle problem.

    Returns
    -------
    LayoutProblem
        Triangle graph ``0-1-2-0``.
    """
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    return LayoutProblem(edge_index=edge_index, num_nodes=3, seed=42)


def test_smacof_step_raises_descriptive_error_without_sm_state() -> None:
    """A missing SM initialization must raise ``ValueError``, not ``KeyError``.

    Regression: the raw ``extras[...]`` lookups used to raise ``KeyError``
    before the op's own descriptive guards could fire.
    """

    state = SolveState(pos=torch.zeros((3, 2), dtype=torch.float32))
    state.distance_matrix = torch.ones((3, 3), dtype=torch.float64)

    with pytest.raises(ValueError, match="sm_current_positions"):
        SmacofStep().apply(_triangle_problem(), state, RuntimeContext())


def test_finalize_positions_raises_descriptive_error_without_sm_state() -> None:
    """Finalize must raise its descriptive ``ValueError`` when SM state is absent."""

    state = SolveState(pos=torch.zeros((3, 2), dtype=torch.float32))

    with pytest.raises(ValueError, match="sm_current_positions"):
        FinalizeStressMajorizationPositions().apply(_triangle_problem(), state, RuntimeContext())
