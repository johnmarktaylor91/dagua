"""Reference-fidelity tests for the Stress-SGD pipeline."""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest
import torch

from dagua.eval.competitors.classic_competitor import _CLASSIC_LAYOUT_SPECS
from dagua.eval.competitors.sgd2_competitor import SGD2
from dagua.graph import DaguaGraph
from dagua.layout.ops.pipelines.stress_sgd import (
    build_stress_sgd_pipeline,
    layout_stress_sgd_pipeline,
)
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.stress_sgd import _build_exact_terms


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


def test_stress_sgd_reference_term_order_uses_bfs_discovery_order() -> None:
    """Fidelity term order follows native ``s_gd2`` BFS discovery order."""
    adjacency = [[(2, 1.0)], [(2, 1.0)], [(0, 1.0), (1, 1.0)]]

    sources, targets, distances, _ = _build_exact_terms(
        adjacency=adjacency,
        weighted=False,
        exact_float64_terms=True,
        reference_term_order=True,
    )

    assert list(zip(sources.tolist(), targets.tolist(), distances.tolist())) == [
        (0, 2, 1.0),
        (0, 1, 2.0),
        (1, 2, 1.0),
    ]


def test_stress_sgd_default_term_order_stays_target_sorted() -> None:
    """Default exact terms preserve the historical target-index order."""
    adjacency = [[(2, 1.0)], [(2, 1.0)], [(0, 1.0), (1, 1.0)]]

    sources, targets, distances, _ = _build_exact_terms(
        adjacency=adjacency,
        weighted=False,
        exact_float64_terms=True,
    )

    assert list(zip(sources.tolist(), targets.tolist(), distances.tolist())) == [
        (0, 1, 2.0),
        (0, 2, 1.0),
        (1, 2, 1.0),
    ]


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


def test_stress_sgd_pipeline_accepts_initial_positions() -> None:
    """Explicit ``init_pos`` bypasses the default NumPy initialization."""
    edges = _edge_index([(0, 1), (1, 2)])
    init_pos = torch.tensor([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=torch.float64)

    positions = layout_stress_sgd_pipeline(edges, 3, init_pos=init_pos, steps=0)

    assert torch.equal(positions, init_pos.to(dtype=torch.float32))


def test_stress_sgd_pipeline_rejects_bad_initial_position_shape() -> None:
    """Initial coordinates must cover every node exactly once."""
    edges = _edge_index([(0, 1)])
    init_pos = torch.zeros((1, 2), dtype=torch.float32)

    with pytest.raises(ValueError, match="init_pos"):
        layout_stress_sgd_pipeline(edges, 2, init_pos=init_pos, steps=0)


def test_classic_stress_sgd_defaults_to_reference_exact_node_cap() -> None:
    """The fidelity competitor keeps exact mode up to the ``s_gd2`` node cap."""
    spec = _CLASSIC_LAYOUT_SPECS["classic_stress_sgd"]

    assert spec.default_params["max_exact_nodes"] == SGD2.max_nodes


def test_stress_sgd_ogdf_mode_matches_seed_42_path_fixture() -> None:
    """OGDF fidelity mode matches the local runner fixture for a path graph."""
    edges = _edge_index([(0, 1), (1, 2), (2, 3), (3, 4)])
    expected = torch.tensor(
        [
            [157.48316461972911, -96.39363027933655],
            [87.42398954824397, -24.617892170241596],
            [27.61844541205235, 56.00507809849135],
            [-18.675736594414914, 145.076183262894],
            [-52.64898253054323, 239.4451229507347],
        ],
        dtype=torch.float32,
    )

    actual = layout_stress_sgd_pipeline(
        edges,
        5,
        steps=200,
        seed=42,
        fidelity_mode="ogdf",
    )

    assert torch.allclose(actual, expected, atol=1.0e-4, rtol=1.0e-6)


def test_stress_sgd_ogdf_mode_accepts_disconnected_graph() -> None:
    """OGDF fidelity uses finite fallback distances for disconnected pairs."""
    edges = _edge_index([(0, 1), (2, 3)])

    positions = layout_stress_sgd_pipeline(
        edges,
        4,
        steps=3,
        seed=7,
        fidelity_mode="ogdf",
    )

    assert positions.shape == (4, 2)
    assert torch.isfinite(positions).all()


def test_sgd2_adapter_rejects_trailing_isolated_node_outputs(monkeypatch: Any) -> None:
    """The reference adapter reports mismatched output from trailing isolated nodes."""
    fake_module = types.ModuleType("s_gd2")

    def _layout_fake(
        sources: list[int],
        targets: list[int],
        **kwargs: Any,
    ) -> np.ndarray:
        """Return the native shape produced from only edge endpoint IDs.

        Parameters
        ----------
        sources : list[int]
            Source endpoint IDs.
        targets : list[int]
            Target endpoint IDs.
        **kwargs : Any
            Ignored ``s_gd2`` compatibility arguments.

        Returns
        -------
        np.ndarray
            Two-node coordinate array, omitting the trailing isolated node.
        """
        del sources, targets, kwargs
        return np.zeros((2, 2), dtype=np.float64)

    fake_module.layout = _layout_fake
    monkeypatch.setitem(sys.modules, "s_gd2", fake_module)
    graph = DaguaGraph.from_edge_index(torch.tensor([[0], [1]], dtype=torch.long), num_nodes=3)

    result = SGD2().layout(graph, seed=7)

    assert result.pos is None
    assert result.error is not None
    assert "trailing isolated nodes" in result.error
