"""Reference-fidelity tests for the Stress-SGD pipeline."""

from __future__ import annotations

import pytest
import torch

from dagua.layout.ops.pipelines.stress_sgd import build_stress_sgd_pipeline
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState


def _edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    """Build an edge-index tensor from edge tuples.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Directed edge pairs.

    Returns
    -------
    torch.Tensor
        Edge index with shape ``[2, E]``.
    """
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    sources, targets = zip(*edges)
    return torch.tensor([sources, targets], dtype=torch.long)


def _apply_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    edge_weights: torch.Tensor | None = None,
    fidelity_mode: bool,
) -> SolveState:
    """Run the Stress-SGD pipeline and return the final solve state.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge index with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    edge_weights : torch.Tensor | None, default=None
        Optional edge weights with shape ``[E]``.
    fidelity_mode : bool
        Whether to enable reference-fidelity behavior.

    Returns
    -------
    SolveState
        Final pipeline state.
    """
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weights=edge_weights,
        seed=7,
    )
    return build_stress_sgd_pipeline(steps=0, fidelity_mode=fidelity_mode).apply(
        problem,
        SolveState(),
        RuntimeContext(plan=ExecutionPlan(device="cpu")),
    )


def test_stress_sgd_fidelity_mode_sums_reverse_weighted_multiedges() -> None:
    """Fidelity mode mirrors ``s_gd2`` summed symmetrized weighted edges."""
    edges = _edge_index([(0, 1), (1, 0)])
    weights = torch.tensor([2.0, 3.0], dtype=torch.float64)

    state = _apply_pipeline(edges, 2, edge_weights=weights, fidelity_mode=True)

    assert state.adjacency == [[(1, 5.0)], [(0, 5.0)]]


def test_stress_sgd_default_mode_keeps_legacy_min_weight_dedup() -> None:
    """Default mode preserves the existing classic minimum-weight policy."""
    edges = _edge_index([(0, 1), (1, 0)])
    weights = torch.tensor([2.0, 3.0], dtype=torch.float64)

    state = _apply_pipeline(edges, 2, edge_weights=weights, fidelity_mode=False)

    assert state.adjacency == [[(1, 2.0)], [(0, 2.0)]]


def test_stress_sgd_fidelity_mode_uses_float64_exact_terms() -> None:
    """Fidelity mode keeps exact stress distances and weights in double precision."""
    edges = _edge_index([(0, 1), (1, 2)])
    weights = torch.tensor([0.25, 0.5], dtype=torch.float64)

    state = _apply_pipeline(edges, 3, edge_weights=weights, fidelity_mode=True)

    assert state.extras["stress_sgd_distances"].dtype == "float64"
    assert state.extras["stress_sgd_weights"].dtype == "float64"


def test_stress_sgd_fidelity_mode_returns_zeros_for_edgeless_graph() -> None:
    """Fidelity mode matches the reference adapter for edgeless ``N > 1`` graphs."""
    state = _apply_pipeline(_edge_index([]), 3, fidelity_mode=True)

    assert state.pos is not None
    assert torch.equal(state.pos, torch.zeros((3, 2), dtype=torch.float32))


def test_stress_sgd_fidelity_mode_rejects_disconnected_graphs() -> None:
    """Fidelity mode raises on disconnected non-empty graphs like native ``s_gd2``."""
    edges = _edge_index([(0, 1), (2, 3)])

    with pytest.raises(ValueError, match="connected graph"):
        _apply_pipeline(edges, 4, fidelity_mode=True)
