"""Regression tests for GRIP, omega, and tidy reference adapters."""

from __future__ import annotations

import pytest
import torch

from dagua.eval.competitors import get_competitor
from dagua.eval.variants import base_pairings, variant_pairings
from dagua.graph import DaguaGraph


def _edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    """Build an edge-index tensor.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Directed edge pairs.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def test_grip_omega_tidy_references_are_registered_and_paired() -> None:
    """Reference and reimplementation competitors should be paired both ways."""
    expected = {
        "grip_reimpl": "grip_reference",
        "omega_reimpl": "omega_reference",
        "tidy_reimpl": "tidy_reference",
    }
    base = base_pairings()
    variants = variant_pairings()
    for reimplementation, reference in expected.items():
        assert get_competitor(reimplementation) is not None
        assert get_competitor(reference) is not None
        assert base[reimplementation] == [reference]
        variant_name = f"{reimplementation}_default"
        original_name = f"{reference}__for__{variant_name}"
        assert variants[variant_name] == [original_name]
        assert variants[original_name] == [variant_name]


@pytest.mark.parametrize(
    ("name", "edge_index", "num_nodes", "variant_params"),
    [
        (
            "grip_reference",
            _edge_index([(0, 1), (1, 2), (2, 3)]),
            4,
            {"rounds": 2, "final_rounds": 2, "init_vertices": 3, "dim": 2},
        ),
        (
            "omega_reference",
            _edge_index([(0, 1), (1, 2), (2, 3), (0, 2)]),
            4,
            {"k": 2, "sgd_iterations": 2, "unit_edge_length": 1.0},
        ),
        (
            "tidy_reference",
            _edge_index([(0, 1), (0, 2), (1, 3)]),
            4,
            {"parent_child_margin": 7.0, "peer_margin": 5.0},
        ),
    ],
)
def test_grip_omega_tidy_reference_smoke(
    name: str,
    edge_index: torch.Tensor,
    num_nodes: int,
    variant_params: dict[str, float | int],
) -> None:
    """Built native references should return finite coordinates."""
    competitor = get_competitor(name)
    assert competitor is not None
    if not competitor.available():
        pytest.skip(f"{name} binary is not available")
    node_sizes = (
        torch.full((num_nodes, 2), 10.0, dtype=torch.float64) if name == "tidy_reference" else None
    )
    graph = DaguaGraph.from_edge_index(edge_index, num_nodes, node_sizes=node_sizes)
    result = competitor.layout_with_variant(graph, seed=11, variant_params=variant_params)
    assert result.error is None
    assert result.pos is not None
    assert result.pos.shape == (num_nodes, 2)
    assert torch.isfinite(result.pos).all()
