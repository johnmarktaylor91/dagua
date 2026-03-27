"""Tests for the classic Sugiyama layered layout."""

import pytest
import torch

from dagua.layout.classic import layout_sugiyama


def test_layout_sugiyama_returns_position_tensor() -> None:
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]], dtype=torch.long)

    positions = layout_sugiyama(edge_index=edge_index, num_nodes=4)

    assert isinstance(positions, torch.Tensor)
    assert positions.shape == (4, 2)
    assert torch.isfinite(positions).all()


def test_layout_sugiyama_is_deterministic_for_same_seed() -> None:
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]], dtype=torch.long)

    positions_a = layout_sugiyama(edge_index=edge_index, num_nodes=4, seed=7)
    positions_b = layout_sugiyama(edge_index=edge_index, num_nodes=4, seed=7)

    assert isinstance(positions_a, torch.Tensor)
    assert isinstance(positions_b, torch.Tensor)
    torch.testing.assert_close(positions_a, positions_b)


def test_layout_sugiyama_preserves_dag_y_ordering() -> None:
    edge_index = torch.tensor([[0, 0, 1, 2, 2], [1, 2, 3, 3, 4]], dtype=torch.long)

    positions = layout_sugiyama(edge_index=edge_index, num_nodes=5)

    assert isinstance(positions, torch.Tensor)
    for src, dst in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        assert positions[src, 1].item() < positions[dst, 1].item()


def test_layout_sugiyama_respects_node_separation_within_layers() -> None:
    edge_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
    node_sizes = torch.tensor(
        [[20.0, 20.0], [30.0, 20.0], [10.0, 20.0]],
        dtype=torch.float32,
    )

    positions = layout_sugiyama(
        edge_index=edge_index,
        num_nodes=3,
        node_sizes=node_sizes,
        node_sep=28.0,
    )

    assert isinstance(positions, torch.Tensor)
    layer_nodes = sorted([1, 2], key=lambda node: positions[node, 0].item())
    left_node, right_node = layer_nodes
    center_gap = positions[right_node, 0].item() - positions[left_node, 0].item()
    min_gap = (node_sizes[left_node, 0].item() + node_sizes[right_node, 0].item()) / 2.0 + 28.0
    assert center_gap >= min_gap


def test_layout_sugiyama_diamond_graph_is_layered_sensibly() -> None:
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]], dtype=torch.long)

    positions = layout_sugiyama(edge_index=edge_index, num_nodes=4, seed=1)

    assert isinstance(positions, torch.Tensor)
    assert positions[0, 1].item() < positions[1, 1].item()
    assert positions[0, 1].item() < positions[2, 1].item()
    assert positions[1, 1].item() == positions[2, 1].item()
    assert positions[3, 1].item() > positions[1, 1].item()
    assert positions[1, 0].item() != positions[2, 0].item()


def test_layout_sugiyama_chain_graph_produces_vertical_line() -> None:
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

    positions = layout_sugiyama(edge_index=edge_index, num_nodes=4)

    assert isinstance(positions, torch.Tensor)
    torch.testing.assert_close(positions[:, 0], torch.zeros(4))
    assert torch.equal(positions[:, 1], torch.tensor([0.0, 1.0, 2.0, 3.0]))


def test_layout_sugiyama_trace_mode_returns_snapshots() -> None:
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]], dtype=torch.long)

    positions, traces = layout_sugiyama(
        edge_index=edge_index,
        num_nodes=4,
        barycenter_passes=4,
        trace_every=2,
    )

    assert positions.shape == (4, 2)
    assert len(traces) == 2
    assert all(trace.shape == (4, 2) for trace in traces)
    torch.testing.assert_close(positions, traces[-1])


def test_layout_sugiyama_routes_long_edges_through_intermediate_layers() -> None:
    edge_index = torch.tensor([[0, 1, 2, 0], [1, 2, 3, 3]], dtype=torch.long)

    positions, edge_routes = layout_sugiyama(
        edge_index=edge_index,
        num_nodes=4,
        return_edge_routes=True,
    )

    long_edge_route = edge_routes[-1]

    assert positions.shape == (4, 2)
    assert len(edge_routes) == edge_index.shape[1]
    assert long_edge_route.shape == (4, 2)
    torch.testing.assert_close(long_edge_route[0], positions[0])
    torch.testing.assert_close(long_edge_route[-1], positions[3])
    torch.testing.assert_close(
        long_edge_route[:, 1],
        torch.tensor([0.0, 1.0, 2.0, 3.0]),
    )


def test_layout_sugiyama_handles_isolated_nodes() -> None:
    edge_index = torch.zeros((2, 0), dtype=torch.long)

    positions = layout_sugiyama(edge_index=edge_index, num_nodes=3)

    assert isinstance(positions, torch.Tensor)
    assert torch.equal(positions[:, 1], torch.zeros(3))
    assert positions[0, 0].item() < positions[1, 0].item() < positions[2, 0].item()


def test_layout_sugiyama_promotes_layers_to_reduce_dummy_nodes() -> None:
    edge_index = torch.tensor([[4, 1, 2, 0], [1, 2, 3, 3]], dtype=torch.long)

    positions, edge_routes = layout_sugiyama(
        edge_index=edge_index,
        num_nodes=5,
        return_edge_routes=True,
    )

    assert positions[0, 1].item() == pytest.approx(2.0)
    assert edge_routes[-1].shape == (2, 2)
    torch.testing.assert_close(edge_routes[-1][0], positions[0])
    torch.testing.assert_close(edge_routes[-1][-1], positions[3])


def test_layout_sugiyama_coordinates_follow_adjacent_layer_barycenters() -> None:
    edge_index = torch.tensor([[0, 1, 2], [3, 4, 4]], dtype=torch.long)

    positions = layout_sugiyama(edge_index=edge_index, num_nodes=5, seed=0)

    assert abs(positions[3, 0].item() - positions[0, 0].item()) <= 0.5
    assert abs(positions[4, 0].item() - ((positions[1, 0] + positions[2, 0]) / 2.0).item()) <= 0.5


def test_layout_sugiyama_layer_sep_alias_overrides_rank_sep() -> None:
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

    positions = layout_sugiyama(edge_index=edge_index, num_nodes=4, rank_sep=10.0, layer_sep=2.0)

    torch.testing.assert_close(positions[:, 1], torch.tensor([0.0, 2.0, 4.0, 6.0]))


def test_layout_sugiyama_edge_weights_bias_weighted_barycenters() -> None:
    edge_index = torch.tensor([[0, 2, 1], [3, 3, 4]], dtype=torch.long)
    edge_weights = torch.tensor([8.0, 1.0, 1.0], dtype=torch.float32)

    positions = layout_sugiyama(
        edge_index=edge_index,
        num_nodes=5,
        seed=0,
        edge_weights=edge_weights,
    )

    assert positions[3, 0].item() < positions[4, 0].item()


def test_layout_sugiyama_rejects_mismatched_edge_weights() -> None:
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

    with pytest.raises(ValueError, match="edge_weights length"):
        layout_sugiyama(
            edge_index=edge_index,
            num_nodes=3,
            edge_weights=torch.ones(1, dtype=torch.float32),
        )


def test_layout_sugiyama_handles_cyclic_input_robustly() -> None:
    """Cycle breaking should still produce a finite layered layout."""
    edge_index = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]], dtype=torch.long)

    positions = layout_sugiyama(edge_index=edge_index, num_nodes=3)

    assert positions.shape == (3, 2)
    assert torch.isfinite(positions).all()
