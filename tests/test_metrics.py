"""Tests for aesthetic quality metrics."""

import pytest
import torch

from dagua.metrics import (
    compute_all_metrics,
    count_crossings,
    compute_dag_fraction,
    count_overlaps,
    compute_mean_edge_length,
    overall_quality,
)


class TestCountCrossings:
    def test_no_crossings(self):
        pos = torch.tensor([[0.0, 0.0], [0.0, 100.0], [50.0, 0.0], [50.0, 100.0]])
        ei = torch.tensor([[0, 2], [1, 3]])  # parallel edges
        assert count_crossings(pos, ei) == 0

    def test_one_crossing(self):
        pos = torch.tensor([[0.0, 0.0], [100.0, 100.0], [100.0, 0.0], [0.0, 100.0]])
        ei = torch.tensor([[0, 2], [1, 3]])  # X pattern
        assert count_crossings(pos, ei) == 1

    def test_empty_edges(self):
        pos = torch.randn(5, 2)
        ei = torch.zeros(2, 0, dtype=torch.long)
        assert count_crossings(pos, ei) == 0


class TestDagFraction:
    def test_perfect_dag(self):
        pos = torch.tensor([[0.0, 0.0], [0.0, 50.0], [0.0, 100.0]])
        ei = torch.tensor([[0, 1], [1, 2]])
        assert compute_dag_fraction(pos, ei) == 1.0

    def test_reversed_dag(self):
        pos = torch.tensor([[0.0, 100.0], [0.0, 50.0], [0.0, 0.0]])
        ei = torch.tensor([[0, 1], [1, 2]])
        assert compute_dag_fraction(pos, ei) == 0.0

    def test_empty(self):
        pos = torch.randn(3, 2)
        ei = torch.zeros(2, 0, dtype=torch.long)
        assert compute_dag_fraction(pos, ei) == 1.0


class TestCountOverlaps:
    def test_overlapping(self):
        pos = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
        ns = torch.tensor([[40.0, 20.0], [40.0, 20.0]])
        assert count_overlaps(pos, ns) == 1

    def test_non_overlapping(self):
        pos = torch.tensor([[0.0, 0.0], [200.0, 0.0]])
        ns = torch.tensor([[40.0, 20.0], [40.0, 20.0]])
        assert count_overlaps(pos, ns) == 0


class TestComputeAllMetrics:
    def test_returns_all_keys(self):
        pos = torch.tensor([[0.0, 0.0], [0.0, 50.0]])
        ei = torch.tensor([[0], [1]])
        ns = torch.tensor([[40.0, 20.0], [40.0, 20.0]])
        m = compute_all_metrics(pos, ei, ns)
        assert "num_nodes" in m
        assert "edge_crossings" in m
        assert "dag_fraction" in m
        assert "node_overlaps" in m
        assert "overall_quality" in m

    def test_overall_quality_higher_is_better(self):
        # Good layout
        pos_good = torch.tensor([[0.0, 0.0], [0.0, 50.0], [0.0, 100.0]])
        # Bad layout (reversed, overlapping)
        pos_bad = torch.tensor([[0.0, 100.0], [5.0, 50.0], [0.0, 0.0]])
        ei = torch.tensor([[0, 1], [1, 2]])
        ns = torch.tensor([[40.0, 20.0]] * 3)
        q_good = compute_all_metrics(pos_good, ei, ns)["overall_quality"]
        q_bad = compute_all_metrics(pos_bad, ei, ns)["overall_quality"]
        assert q_good > q_bad
