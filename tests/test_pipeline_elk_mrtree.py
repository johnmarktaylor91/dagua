"""Tests for the ELK MrTree pipeline."""

from __future__ import annotations

from pathlib import Path

import torch

from dagua.layout.ops.pipelines import get_pipeline_function
from dagua.layout.ops.pipelines.elk_mrtree import layout_elk_mrtree_pipeline


def test_elk_mrtree_pipeline_is_registered() -> None:
    """ELK MrTree should resolve through the shared pipeline registry."""
    assert get_pipeline_function("elk_mrtree") is layout_elk_mrtree_pipeline


def test_elk_mrtree_places_tree_by_depth() -> None:
    """MrTree should increase y by tree depth on a directed tree."""
    edge_index = torch.tensor([[0, 0, 1, 1], [1, 2, 3, 4]], dtype=torch.long)
    node_sizes = torch.full((5, 2), 20.0, dtype=torch.float64)
    pos = layout_elk_mrtree_pipeline(edge_index, 5, node_sizes, roots=[0])
    assert pos.shape == (5, 2)
    assert torch.isfinite(pos).all()
    assert pos[0, 1] < pos[1, 1] < pos[3, 1]


def test_elk_mrtree_pipeline_does_not_delegate_to_elkjs() -> None:
    """Runtime pipeline source must not import or execute elkjs/Node."""
    source = Path("dagua/layout/ops/pipelines/elk_mrtree.py").read_text()
    assert "elkjs" not in source
    assert "subprocess" not in source
