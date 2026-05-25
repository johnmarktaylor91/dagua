"""Tests for opt-in Graphviz neato overlap-removal fidelity."""

from __future__ import annotations

import pytest
import torch

from dagua.layout.ops.pipelines import neato
from dagua.layout.ops.pipelines.neato import remove_neato_overlap_fidelity


def test_neato_vpsc_two_node_golden_spacing() -> None:
    """VPSC two-node spacing matches Graphviz 7.0.5's gap and x-scale constants."""
    positions = torch.tensor([[0.0, 0.0], [0.5, 0.0]], dtype=torch.float64)
    node_sizes = torch.ones((2, 2), dtype=torch.float64)

    adjusted = remove_neato_overlap_fidelity(positions=positions, node_sizes=node_sizes)

    expected_separation = 1.0001 + (1.0 / 9.0)
    assert torch.allclose(adjusted[:, 1], positions[:, 1])
    assert adjusted[1, 0] - adjusted[0, 0] == pytest.approx(expected_separation)
    assert adjusted[:, 0].mean() == pytest.approx(positions[:, 0].mean())


def test_neato_overlap_fidelity_noops_without_node_sizes() -> None:
    """Overlap fidelity leaves coordinates unchanged when rectangles are unavailable."""
    positions = torch.tensor([[0.0, 0.0], [0.5, 0.0]], dtype=torch.float32)

    adjusted = remove_neato_overlap_fidelity(positions=positions, node_sizes=None)

    assert adjusted is positions


def test_neato_pipeline_overlap_removal_is_fidelity_gated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The new overlap pass is inactive unless neato fidelity mode is enabled."""

    def fake_stress_pipeline(**kwargs: object) -> torch.Tensor:
        """Return a deliberately overlapping stress layout for gate testing."""
        return torch.tensor([[0.0, 0.0], [0.5, 0.0]], dtype=torch.float64)

    monkeypatch.setattr(neato, "layout_stress_majorization_pipeline", fake_stress_pipeline)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    node_sizes = torch.ones((2, 2), dtype=torch.float64)

    default_result = neato.layout_neato_pipeline(
        edge_index=edge_index,
        num_nodes=2,
        node_sizes=node_sizes,
        pack=False,
        fidelity_mode=False,
        overlap_removal=True,
    )
    fidelity_result = neato.layout_neato_pipeline(
        edge_index=edge_index,
        num_nodes=2,
        node_sizes=node_sizes,
        pack=False,
        fidelity_mode=True,
        overlap_removal=True,
    )

    assert torch.equal(default_result, torch.tensor([[0.0, 0.0], [0.5, 0.0]], dtype=torch.float64))
    assert fidelity_result[1, 0] - fidelity_result[0, 0] > 1.0
