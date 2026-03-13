"""Tests for individual constraint loss functions."""

import pytest
import torch

from dagua.config import LayoutConfig
from dagua.eval.graphs import get_test_graphs
from dagua.layout import layout
from dagua.layout.constraints import (
    back_edge_compactness_loss,
    crossing_loss,
    dag_ordering_loss,
    edge_attraction_loss,
    edge_length_variance_loss,
    edge_straightness_loss,
    fanout_distribution_loss,
    overlap_avoidance_loss,
    repulsion_loss,
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


class TestFanoutDistributionLoss:
    def test_star_graph_clustered_children(self):
        """Star graph with 10 children clustered to one side should have loss > 0."""
        # Hub at origin, 10 children all clustered to the right
        pos = torch.zeros(11, 2)
        pos[0] = torch.tensor([0.0, 0.0])  # hub
        for i in range(10):
            pos[i + 1] = torch.tensor([50.0 + i * 2.0, 50.0])  # clustered
        ei = torch.tensor([
            [0] * 10,
            list(range(1, 11)),
        ])
        loss = fanout_distribution_loss(pos, ei, degree_threshold=5)
        assert loss.item() > 0

    def test_star_graph_spread_vs_clustered(self):
        """Evenly spread children should have lower loss than clustered ones."""
        import math
        ei = torch.tensor([[0] * 10, list(range(1, 11))])

        # Clustered: all children to the right
        pos_clustered = torch.zeros(11, 2)
        for i in range(10):
            pos_clustered[i + 1] = torch.tensor([50.0 + i * 2.0, 50.0])

        # Spread: children evenly around hub
        pos_spread = torch.zeros(11, 2)
        for i in range(10):
            angle = 2 * math.pi * i / 10
            pos_spread[i + 1] = torch.tensor([50.0 * math.cos(angle), 50.0 * math.sin(angle)])

        loss_clustered = fanout_distribution_loss(pos_clustered, ei, degree_threshold=5)
        loss_spread = fanout_distribution_loss(pos_spread, ei, degree_threshold=5)
        assert loss_spread < loss_clustered

    def test_no_hubs_returns_zero(self):
        """Graph with max degree < threshold returns 0."""
        pos = torch.tensor([[0.0, 0.0], [50.0, 50.0], [100.0, 100.0]])
        ei = torch.tensor([[0, 1], [1, 2]])  # chain, max out-degree = 1
        loss = fanout_distribution_loss(pos, ei, degree_threshold=5)
        assert loss.item() == 0.0

    def test_empty_edges(self):
        pos = torch.randn(5, 2)
        ei = torch.zeros(2, 0, dtype=torch.long)
        loss = fanout_distribution_loss(pos, ei)
        assert loss.item() == 0.0


class TestBackEdgeCompactnessLoss:
    def test_back_edge_has_loss(self):
        """Graph with a back edge (target above source) should have loss > 0."""
        pos = torch.tensor([
            [0.0, 0.0],    # node 0 at top
            [0.0, 100.0],  # node 1 below
            [80.0, 0.0],   # node 2 at top, far right
        ])
        # Edge 1→2 is a back edge (node 2 y=0 < node 1 y=100)
        ei = torch.tensor([[1], [2]])
        loss = back_edge_compactness_loss(pos, ei)
        assert loss.item() > 0

    def test_no_back_edges_returns_zero(self):
        """All forward edges (target below source) → loss == 0."""
        pos = torch.tensor([
            [0.0, 0.0],
            [0.0, 50.0],
            [0.0, 100.0],
        ])
        # All edges point downward (increasing y)
        ei = torch.tensor([[0, 1], [1, 2]])
        loss = back_edge_compactness_loss(pos, ei)
        assert loss.item() == 0.0

    def test_empty_edges(self):
        pos = torch.randn(5, 2)
        ei = torch.zeros(2, 0, dtype=torch.long)
        loss = back_edge_compactness_loss(pos, ei)
        assert loss.item() == 0.0


@pytest.mark.smoke
@pytest.mark.parametrize("graph_name", ["kitchen_sink_hybrid_net", "disconnected_label_cycle_collage"])
def test_self_loop_stress_graphs_layout_without_nan(graph_name):
    graphs = {graph.name: graph.graph for graph in get_test_graphs(max_nodes=120)}
    graph = graphs[graph_name]
    graph.compute_node_sizes()

    pos = layout(
        graph,
        config=LayoutConfig(
            steps=20,
            edge_opt_steps=-1,
            seed=42,
        ),
    )

    assert torch.isfinite(pos).all(), f"{graph_name} produced NaN/Inf positions"
