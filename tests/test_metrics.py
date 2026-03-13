"""Tests for aesthetic quality metrics."""

import pytest
import torch

from dagua.metrics import (
    compute_all_metrics,
    count_crossings,
    compute_dag_fraction,
    compute_edge_straightness,
    compute_x_alignment,
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


class TestDirectionAwareMetrics:
    """Test that metrics respect layout direction."""

    def test_dag_fraction_bt(self):
        """BT: edges should go upward (target.y < source.y)."""
        pos = torch.tensor([[0.0, 100.0], [0.0, 50.0], [0.0, 0.0]])
        ei = torch.tensor([[0, 1], [1, 2]])
        assert compute_dag_fraction(pos, ei, direction="BT") == 1.0
        assert compute_dag_fraction(pos, ei, direction="TB") == 0.0

    def test_dag_fraction_lr(self):
        """LR: edges should go rightward (target.x > source.x)."""
        pos = torch.tensor([[0.0, 0.0], [50.0, 0.0], [100.0, 0.0]])
        ei = torch.tensor([[0, 1], [1, 2]])
        assert compute_dag_fraction(pos, ei, direction="LR") == 1.0

    def test_dag_fraction_rl(self):
        """RL: edges should go leftward (target.x < source.x)."""
        pos = torch.tensor([[100.0, 0.0], [50.0, 0.0], [0.0, 0.0]])
        ei = torch.tensor([[0, 1], [1, 2]])
        assert compute_dag_fraction(pos, ei, direction="RL") == 1.0

    def test_edge_straightness_lr(self):
        """LR: deviation from horizontal should be small for horizontal edges."""
        pos = torch.tensor([[0.0, 0.0], [100.0, 0.0]])  # perfectly horizontal
        ei = torch.tensor([[0], [1]])
        angle = compute_edge_straightness(pos, ei, direction="LR")
        assert angle < 1.0  # nearly 0 degrees deviation from horizontal

    def test_edge_straightness_tb(self):
        """TB: deviation from vertical should be small for vertical edges."""
        pos = torch.tensor([[0.0, 0.0], [0.0, 100.0]])  # perfectly vertical
        ei = torch.tensor([[0], [1]])
        angle = compute_edge_straightness(pos, ei, direction="TB")
        assert angle < 1.0

    def test_x_alignment_lr(self):
        """LR: cross-axis displacement is along y."""
        pos = torch.tensor([[0.0, 0.0], [100.0, 10.0]])
        ei = torch.tensor([[0], [1]])
        # LR: cross-axis is y, so alignment = abs(0 - 10) = 10
        assert compute_x_alignment(pos, ei, direction="LR") == pytest.approx(10.0)
        # TB: cross-axis is x, so alignment = abs(0 - 100) = 100
        assert compute_x_alignment(pos, ei, direction="TB") == pytest.approx(100.0)

    def test_compute_all_metrics_with_direction(self):
        """compute_all_metrics passes direction through."""
        pos = torch.tensor([[0.0, 100.0], [0.0, 50.0], [0.0, 0.0]])
        ei = torch.tensor([[0, 1], [1, 2]])
        ns = torch.tensor([[40.0, 20.0]] * 3)
        m_bt = compute_all_metrics(pos, ei, ns, direction="BT")
        m_tb = compute_all_metrics(pos, ei, ns, direction="TB")
        assert m_bt["dag_fraction"] == 1.0
        assert m_tb["dag_fraction"] == 0.0
