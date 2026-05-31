"""Parity tests for the in-pipeline LinLog fidelity solver."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import pytest
import torch

from dagua.eval.competitors.linlog_competitor import _layout_linlog_reference, _resolve_config
from dagua.layout.ops.pipelines.linlog import layout_linlog_pipeline


def _edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    """Build an edge tensor from edge tuples.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Edge tuples.

    Returns
    -------
    torch.Tensor
        Edge index with shape ``[2, E]``.
    """
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    src, dst = zip(*edges)
    return torch.tensor([list(src), list(dst)], dtype=torch.long)


def _path_edge_index(num_nodes: int) -> torch.Tensor:
    """Build a path edge tensor.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.

    Returns
    -------
    torch.Tensor
        Edge index with shape ``[2, E]``.
    """
    return _edge_index([(index, index + 1) for index in range(max(num_nodes - 1, 0))])


def _cycle_chords_edge_index(num_nodes: int) -> torch.Tensor:
    """Build a cycle edge tensor with deterministic chords.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.

    Returns
    -------
    torch.Tensor
        Edge index with shape ``[2, E]``.
    """
    edges = [(index, (index + 1) % num_nodes) for index in range(num_nodes)]
    edges.extend((index, (index + 3) % num_nodes) for index in range(0, num_nodes, 2))
    return _edge_index(edges)


def _procrustes_rmsd(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Compute Procrustes RMSD after optimal rotation or reflection.

    Parameters
    ----------
    actual : torch.Tensor
        Actual positions with shape ``[N, 2]``.
    expected : torch.Tensor
        Expected positions with shape ``[N, 2]``.

    Returns
    -------
    float
        Root-mean-square deviation after alignment.
    """
    if actual.numel() == 0:
        return 0.0
    actual_centered = actual.to(torch.float64) - actual.to(torch.float64).mean(
        dim=0,
        keepdim=True,
    )
    expected_centered = expected.to(torch.float64) - expected.to(torch.float64).mean(
        dim=0,
        keepdim=True,
    )
    u, _, vh = torch.linalg.svd(actual_centered.T @ expected_centered)
    aligned = actual_centered @ (u @ vh)
    return float(torch.sqrt(torch.mean((aligned - expected_centered) ** 2)).item())


_PARITY_CASES = [
    pytest.param(
        "empty",
        torch.empty((2, 0), dtype=torch.long),
        0,
        3,
        None,
        id="empty",
    ),
    pytest.param("path_exact", _path_edge_index(8), 8, 3, None, id="path-exact"),
    pytest.param(
        "weighted_exact",
        _cycle_chords_edge_index(12),
        12,
        3,
        torch.linspace(0.25, 2.0, 18),
        id="weighted-exact",
    ),
    pytest.param(
        "disconnected_exact",
        _edge_index([(0, 1), (1, 2), (5, 6), (8, 9)]),
        11,
        3,
        None,
        id="disconnected-exact",
    ),
    pytest.param(
        "barnes_hut_weighted",
        _path_edge_index(2001),
        2001,
        1,
        torch.linspace(0.5, 1.5, 2000),
        id="barnes-hut-weighted",
    ),
]


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize(
    ("case_name", "edge_index", "num_nodes", "steps", "edge_weights"),
    _PARITY_CASES,
)
def test_linlog_fidelity_matches_reference_without_runtime_delegation(
    case_name: str,
    edge_index: torch.Tensor,
    num_nodes: int,
    steps: int,
    edge_weights: Optional[torch.Tensor],
    seed: int,
) -> None:
    """Fidelity mode should match the independent reference numerically.

    Parameters
    ----------
    case_name : str
        Human-readable case name used by pytest ids.
    edge_index : torch.Tensor
        Edge index with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    steps : int
        Number of fidelity iterations.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.
    seed : int
        Random seed for initial coordinates.

    Returns
    -------
    None
        This test asserts parity with the LinLog reference adapter.
    """
    config = _resolve_config({"steps": steps, "a": 1.0, "r": 0.0})
    graph = SimpleNamespace(
        num_nodes=num_nodes,
        edge_index=edge_index,
        edge_weights=edge_weights,
        node_sizes=None,
    )

    expected = _layout_linlog_reference(graph=graph, config=config, seed=seed)
    actual = layout_linlog_pipeline(
        edge_index=edge_index,
        num_nodes=num_nodes,
        steps=steps,
        seed=seed,
        edge_weights=edge_weights,
        fidelity_mode=True,
    )

    max_abs_diff = float((actual - expected).abs().max().item()) if actual.numel() else 0.0
    rmsd = _procrustes_rmsd(actual, expected)
    assert torch.allclose(actual, expected, atol=1.0e-6, rtol=0.0), case_name
    assert max_abs_diff <= 1.0e-6, case_name
    assert rmsd < 1.0e-6, case_name
