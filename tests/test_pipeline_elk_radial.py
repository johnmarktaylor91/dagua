"""Tests for the ELK Radial pipeline."""

from __future__ import annotations

from pathlib import Path

import torch

from dagua.layout.ops.pipelines import get_pipeline_function
from dagua.layout.ops.pipelines.elk_radial import layout_elk_radial_pipeline


def test_elk_radial_pipeline_is_registered() -> None:
    """ELK Radial should resolve through the shared pipeline registry."""
    assert get_pipeline_function("elk_radial") is layout_elk_radial_pipeline


def test_elk_radial_produces_concentric_tree() -> None:
    """Radial should place deeper tree nodes farther from the root."""
    edge_index = torch.tensor([[0, 0, 1, 1], [1, 2, 3, 4]], dtype=torch.long)
    node_sizes = torch.full((5, 2), 20.0, dtype=torch.float64)
    pos = layout_elk_radial_pipeline(edge_index, 5, node_sizes, roots=[0])
    root = pos[0]
    child_radius = torch.linalg.norm(pos[1] - root)
    grandchild_radius = torch.linalg.norm(pos[3] - root)
    assert pos.shape == (5, 2)
    assert torch.isfinite(pos).all()
    assert grandchild_radius > child_radius


def test_elk_radial_pipeline_does_not_delegate_to_elkjs() -> None:
    """Runtime pipeline source must not import or execute elkjs/Node."""
    source = Path("dagua/layout/ops/pipelines/elk_radial.py").read_text()
    assert "elkjs" not in source
    assert "subprocess" not in source
