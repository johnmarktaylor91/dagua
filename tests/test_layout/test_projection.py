"""Tests for hard overlap projection."""

import pytest
import torch

from dagua.layout.projection import project_overlaps
from dagua.metrics import count_overlaps


class TestProjectOverlaps:
    def test_resolves_overlaps(self):
        # Create overlapping nodes
        pos = torch.tensor([[0.0, 0.0], [10.0, 5.0], [5.0, 10.0]])
        ns = torch.tensor([[40.0, 20.0]] * 3)
        projected = project_overlaps(pos, ns, iterations=20)
        overlaps = count_overlaps(projected, ns)
        assert overlaps == 0

    def test_preserves_non_overlapping(self):
        # Nodes already well-separated
        pos = torch.tensor([[0.0, 0.0], [200.0, 0.0], [0.0, 200.0]])
        ns = torch.tensor([[40.0, 20.0]] * 3)
        projected = project_overlaps(pos, ns, iterations=10)
        # Should not move much
        assert torch.allclose(pos, projected, atol=1.0)

    def test_single_node(self):
        pos = torch.tensor([[50.0, 50.0]])
        ns = torch.tensor([[40.0, 20.0]])
        projected = project_overlaps(pos, ns)
        assert torch.allclose(pos, projected)
