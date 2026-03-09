"""Tests for individual constraint loss functions."""

import pytest
import torch

from dagua.layout.constraints import (
    dag_ordering_loss,
    edge_attraction_loss,
    repulsion_loss,
    overlap_avoidance_loss,
    crossing_loss,
    edge_straightness_loss,
    edge_length_variance_loss,
)


@pytest.fixture
def chain_data():
    """Simple chain: 0→1→2→3, perfectly ordered vertically."""
    pos = torch.tensor([[0.0, 0.0], [0.0, 50.0], [0.0, 100.0], [0.0, 150.0]])
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    node_sizes = torch.tensor([[40.0, 20.0]] * 4)
    return pos, edge_index, node_sizes


@pytest.fixture
def bad_order_data():
    """Chain where targets are ABOVE sources (wrong DAG ordering)."""
    pos = torch.tensor([[0.0, 150.0], [0.0, 100.0], [0.0, 50.0], [0.0, 0.0]])
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    node_sizes = torch.tensor([[40.0, 20.0]] * 4)
    return pos, edge_index, node_sizes


class TestDagOrderingLoss:
    def test_perfect_order_zero_loss(self, chain_data):
        pos, ei, ns = chain_data
        loss = dag_ordering_loss(pos, ei, ns)
        assert loss.item() < 1.0  # near zero for good ordering

    def test_bad_order_high_loss(self, bad_order_data):
        pos, ei, ns = bad_order_data
        loss = dag_ordering_loss(pos, ei, ns)
        assert loss.item() > 10.0  # significant penalty

    def test_empty_edges(self):
        pos = torch.randn(5, 2)
        ei = torch.zeros(2, 0, dtype=torch.long)
        ns = torch.ones(5, 2) * 20
        loss = dag_ordering_loss(pos, ei, ns)
        assert loss.item() == 0.0


class TestEdgeAttractionLoss:
    def test_close_nodes_low_loss(self):
        pos = torch.tensor([[0.0, 0.0], [0.0, 30.0]])
        ei = torch.tensor([[0], [1]])
        loss = edge_attraction_loss(pos, ei, x_bias=1.0)
        assert loss.item() < 1000

    def test_far_nodes_higher_loss(self):
        pos_close = torch.tensor([[0.0, 0.0], [0.0, 30.0]])
        pos_far = torch.tensor([[0.0, 0.0], [0.0, 300.0]])
        ei = torch.tensor([[0], [1]])
        loss_close = edge_attraction_loss(pos_close, ei, x_bias=1.0)
        loss_far = edge_attraction_loss(pos_far, ei, x_bias=1.0)
        assert loss_far > loss_close


class TestRepulsionLoss:
    def test_overlapping_high_loss(self):
        pos = torch.tensor([[0.0, 0.0], [5.0, 0.0]])  # very close
        loss = repulsion_loss(pos, num_nodes=2)
        assert loss.item() > 0

    def test_far_apart_low_loss(self):
        pos = torch.tensor([[0.0, 0.0], [500.0, 0.0]])  # far apart
        loss = repulsion_loss(pos, num_nodes=2)
        # Should be lower for distant nodes
        pos_close = torch.tensor([[0.0, 0.0], [5.0, 0.0]])
        loss_close = repulsion_loss(pos_close, num_nodes=2)
        assert loss.item() < loss_close.item()

    def test_single_node(self):
        pos = torch.tensor([[0.0, 0.0]])
        loss = repulsion_loss(pos, num_nodes=1)
        assert loss.item() == 0.0


class TestOverlapAvoidanceLoss:
    def test_overlapping_nodes(self):
        pos = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
        ns = torch.tensor([[40.0, 20.0], [40.0, 20.0]])
        loss = overlap_avoidance_loss(pos, ns)
        assert loss.item() > 0

    def test_non_overlapping(self):
        pos = torch.tensor([[0.0, 0.0], [200.0, 0.0]])
        ns = torch.tensor([[40.0, 20.0], [40.0, 20.0]])
        loss = overlap_avoidance_loss(pos, ns)
        assert loss.item() == 0.0


class TestCrossingLoss:
    def test_crossing_edges(self):
        # Two edges that cross: (0,0)→(100,100) and (100,0)→(0,100)
        pos = torch.tensor([
            [0.0, 0.0], [100.0, 100.0],
            [100.0, 0.0], [0.0, 100.0],
        ])
        ei = torch.tensor([[0, 2], [1, 3]])
        loss = crossing_loss(pos, ei, alpha=5.0)
        assert loss.item() > 0

    def test_parallel_edges_no_crossing(self):
        pos = torch.tensor([
            [0.0, 0.0], [0.0, 100.0],
            [50.0, 0.0], [50.0, 100.0],
        ])
        ei = torch.tensor([[0, 2], [1, 3]])
        loss = crossing_loss(pos, ei, alpha=5.0)
        # Parallel non-crossing edges should have low loss
        assert loss.item() < 0.5


class TestEdgeStraightnessLoss:
    def test_straight_vertical_zero(self):
        pos = torch.tensor([[0.0, 0.0], [0.0, 100.0]])
        ei = torch.tensor([[0], [1]])
        loss = edge_straightness_loss(pos, ei)
        assert loss.item() < 0.01

    def test_diagonal_higher(self):
        pos_straight = torch.tensor([[0.0, 0.0], [0.0, 100.0]])
        pos_diag = torch.tensor([[0.0, 0.0], [100.0, 100.0]])
        ei = torch.tensor([[0], [1]])
        assert edge_straightness_loss(pos_diag, ei) > edge_straightness_loss(pos_straight, ei)


class TestEdgeLengthVarianceLoss:
    def test_uniform_lengths_low(self):
        pos = torch.tensor([[0.0, 0.0], [0.0, 50.0], [0.0, 100.0]])
        ei = torch.tensor([[0, 1], [1, 2]])
        loss = edge_length_variance_loss(pos, ei)
        assert loss.item() < 1.0

    def test_varied_lengths_higher(self):
        pos_uniform = torch.tensor([[0.0, 0.0], [0.0, 50.0], [0.0, 100.0]])
        pos_varied = torch.tensor([[0.0, 0.0], [0.0, 10.0], [0.0, 200.0]])
        ei = torch.tensor([[0, 1], [1, 2]])
        assert edge_length_variance_loss(pos_varied, ei) > edge_length_variance_loss(pos_uniform, ei)
