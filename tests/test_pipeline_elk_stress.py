"""Tests for the ELK Stress pipeline."""

from __future__ import annotations

from pathlib import Path

import torch

from dagua.layout.ops.pipelines import get_pipeline_function
from dagua.layout.ops.pipelines.elk_stress import layout_elk_stress_pipeline


def test_elk_stress_pipeline_is_registered() -> None:
    """ELK Stress should resolve through the shared pipeline registry."""
    assert get_pipeline_function("elk_stress") is layout_elk_stress_pipeline


def test_elk_stress_is_deterministic_and_finite() -> None:
    """ELK Stress should produce repeatable finite coordinates."""
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    node_sizes = torch.full((4, 2), 20.0, dtype=torch.float64)
    first = layout_elk_stress_pipeline(edge_index, 4, node_sizes, iteration_limit=4)
    second = layout_elk_stress_pipeline(edge_index, 4, node_sizes, iteration_limit=4)
    assert first.shape == (4, 2)
    assert torch.isfinite(first).all()
    assert torch.allclose(first, second)


def test_elk_stress_pipeline_does_not_delegate_to_elkjs() -> None:
    """Runtime pipeline source must not import or execute elkjs/Node."""
    source = Path("dagua/layout/ops/pipelines/elk_stress.py").read_text()
    assert "elkjs" not in source
    assert "subprocess" not in source
