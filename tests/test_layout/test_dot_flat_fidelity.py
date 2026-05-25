"""Graphviz-dot flat/self/multi-edge fidelity tests."""

from __future__ import annotations

import torch

from dagua.config import LayoutConfig
from dagua.layout.ops.pipelines.dagua_native import (
    _dot_flat_adjacency_mask,
    _dot_flat_preprocess_edges,
    _is_graphviz_dot_fidelity_mode,
    _is_graphviz_dot_flat_fidelity_mode,
    layout_dagua_native_pipeline,
)


def test_dot_flat_adjacency_matches_flat_c_blockers() -> None:
    """Flat adjacency follows ``flat.c`` normal/labeled-virtual blocker rules."""
    edge_index = torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=torch.long)
    ranks = torch.tensor([0, 0, 0, 0], dtype=torch.long)
    orders = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    default_adjacent = _dot_flat_adjacency_mask(edge_index, ranks, orders)
    assert default_adjacent.tolist() == [True, False, False]

    node_is_normal = torch.tensor([True, False, True, True], dtype=torch.bool)
    unlabeled_virtual_adjacent = _dot_flat_adjacency_mask(
        edge_index,
        ranks,
        orders,
        node_is_normal=node_is_normal,
    )
    assert unlabeled_virtual_adjacent.tolist() == [True, True, False]

    labeled_virtual = torch.tensor([False, True, False, False], dtype=torch.bool)
    labeled_virtual_adjacent = _dot_flat_adjacency_mask(
        edge_index,
        ranks,
        orders,
        node_is_normal=node_is_normal,
        virtual_label_mask=labeled_virtual,
    )
    assert labeled_virtual_adjacent.tolist() == [True, False, False]


def test_dot_flat_preprocess_filters_self_loops_and_duplicate_representatives() -> None:
    """Fidelity preprocessing keeps first non-self edge per ordered node pair."""
    edge_index = torch.tensor(
        [
            [0, 0, 0, 1, 2, 2, 3],
            [0, 1, 1, 0, 2, 3, 3],
        ],
        dtype=torch.long,
    )
    weights = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
    ranks = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    result = _dot_flat_preprocess_edges(
        edge_index=edge_index,
        num_nodes=4,
        edge_weights=weights,
        layer_assignments=ranks,
    )

    assert result.edge_index.tolist() == [[0, 1, 2], [1, 0, 3]]
    assert result.edge_weights is not None
    assert result.edge_weights.tolist() == [20.0, 40.0, 60.0]
    assert result.metadata.original_edge_count == 7
    assert result.metadata.representative_edge_ids.tolist() == [1, 3, 5]
    assert result.metadata.self_loop_edge_ids.tolist() == [0, 4, 6]
    assert result.metadata.duplicate_edge_ids.tolist() == [2]
    assert result.metadata.flat_edge_ids.tolist() == [1, 2, 3, 5]
    assert result.metadata.flat_representative_edge_ids.tolist() == [1, 1, 3, 5]
    assert result.metadata.flat_adjacent_mask.tolist() == [True, True, True, True]


def test_graphviz_dot_fidelity_selector_is_explicit() -> None:
    """Default mode is off and only dot fidelity aliases enable preprocessing."""
    assert _is_graphviz_dot_fidelity_mode(None) is False
    assert _is_graphviz_dot_fidelity_mode(False) is False
    assert _is_graphviz_dot_fidelity_mode("igraph") is False
    assert _is_graphviz_dot_fidelity_mode(True) is True
    assert _is_graphviz_dot_fidelity_mode("dot") is True
    assert _is_graphviz_dot_fidelity_mode("graphviz_dot") is True
    assert _is_graphviz_dot_flat_fidelity_mode("dot_position") is False
    assert _is_graphviz_dot_flat_fidelity_mode("dot_flat") is True


def test_dagua_native_dot_fidelity_handles_self_and_multi_edges() -> None:
    """The native pipeline remains invokable with dot fidelity preprocessing."""
    edge_index = torch.tensor(
        [
            [0, 0, 0, 1, 2, 2, 3],
            [0, 1, 1, 0, 2, 3, 3],
        ],
        dtype=torch.long,
    )
    node_sizes = torch.full((4, 2), 24.0, dtype=torch.float32)
    ranks = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    pos = layout_dagua_native_pipeline(
        edge_index=edge_index,
        num_nodes=4,
        node_sizes=node_sizes,
        config=LayoutConfig(steps=1, edge_equalize_polish=False),
        layer_assignments=ranks,
        fidelity_mode="graphviz_dot",
    )

    assert pos.shape == (4, 2)
    assert torch.isfinite(pos).all()
