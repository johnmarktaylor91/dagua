"""Regression tests for Sugiyama igraph-fidelity edge cases."""

import pytest
import torch

from dagua.layout.ops.pipelines.sugiyama import layout_sugiyama_pipeline
from dagua.layout.ops.sugiyama import (
    _igraph_eades_layer_assignments,
    _igraph_glpk_layer_assignments,
    _igraph_glpk_objective_coefficients,
    _igraph_undirected_layer_assignments,
)


def test_sugiyama_ignores_self_loops_before_layering() -> None:
    """Self-loops should not keep otherwise acyclic graphs from layering."""
    edge_index = torch.tensor(
        [
            [0, 0, 1, 2],
            [0, 1, 2, 2],
        ],
        dtype=torch.long,
    )

    positions = layout_sugiyama_pipeline(edge_index=edge_index, num_nodes=3)

    assert positions.shape == (3, 2)
    assert torch.isfinite(positions).all()
    assert positions[0, 1] < positions[1, 1] < positions[2, 1]


def test_sugiyama_igraph_fidelity_stops_after_stable_ordering() -> None:
    """Igraph fidelity mode should stop sweeps once a full pass is stable."""
    edge_index = torch.tensor(
        [
            [0, 1, 2],
            [1, 2, 3],
        ],
        dtype=torch.long,
    )

    _, default_traces = layout_sugiyama_pipeline(
        edge_index=edge_index,
        num_nodes=4,
        barycenter_passes=10,
        trace_every=1,
    )
    _, fidelity_traces = layout_sugiyama_pipeline(
        edge_index=edge_index,
        num_nodes=4,
        barycenter_passes=10,
        trace_every=1,
        fidelity_mode="igraph",
    )

    assert len(default_traces) == 10
    assert len(fidelity_traces) == 1


def test_sugiyama_igraph_fidelity_uses_multiedge_incidence_barycenters() -> None:
    """Igraph fidelity mode should count duplicate edges as incidences."""
    edge_index = torch.tensor(
        [
            [0, 0, 4, 1, 0, 1, 3],
            [5, 5, 5, 6, 1, 2, 1],
        ],
        dtype=torch.long,
    )

    default_positions = layout_sugiyama_pipeline(
        edge_index=edge_index,
        num_nodes=7,
        rank_sep=1.0,
        node_sep=1.0,
    )
    fidelity_positions = layout_sugiyama_pipeline(
        edge_index=edge_index,
        num_nodes=7,
        rank_sep=1.0,
        node_sep=1.0,
        fidelity_mode="igraph",
    )

    assert default_positions[5, 0] < default_positions[6, 0]
    assert fidelity_positions[6, 0] < fidelity_positions[5, 0]


def test_sugiyama_igraph_glpk_objective_matches_in_in_strength_quirk() -> None:
    """Igraph LP coefficients should preserve the 1.0.0 IN/IN source quirk."""
    edge_index = torch.tensor(
        [
            [0, 0, 1, 2],
            [1, 2, 3, 3],
        ],
        dtype=torch.long,
    )

    objective = _igraph_glpk_objective_coefficients(
        edge_index=edge_index,
        num_nodes=4,
        feedback_edges=set(),
        edge_weights=None,
    )

    assert objective == [0.0, 0.0, 0.0, 0.0]


def test_sugiyama_igraph_glpk_two_hubs_bridge_matches_installed_igraph() -> None:
    """The LP layering should match installed igraph on an IN/IN-only distinguisher."""
    igraph = pytest.importorskip("igraph")
    edge_index = torch.tensor(
        [
            [0, 1, 2, 2, 3, 4, 5, 6, 6],
            [2, 2, 3, 4, 6, 6, 6, 7, 8],
        ],
        dtype=torch.long,
    )
    edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    graph = igraph.Graph(n=9, edges=edges, directed=True)
    layout = graph.layout("sugiyama")
    if isinstance(layout, tuple):
        layout = layout[0]
    y_values = [float(coord[1]) for coord in layout.coords]
    ordered_y_values = sorted(set(y_values))
    expected_layers = torch.tensor(
        [ordered_y_values.index(y_value) for y_value in y_values],
        dtype=torch.long,
    )

    layers = _igraph_glpk_layer_assignments(edge_index=edge_index, num_nodes=9)

    assert torch.equal(expected_layers, torch.tensor([0, 0, 1, 2, 2, 0, 3, 4, 4]))
    assert torch.equal(layers, expected_layers)


def test_sugiyama_igraph_conflict_quirk_matches_installed_igraph() -> None:
    """The BK conflict pass should match installed igraph on a tie-sensitive DAG."""
    igraph = pytest.importorskip("igraph")
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 4, 3, 2, 5, 6, 7, 8, 2, 3, 4, 9, 10, 11, 12, 13],
            [1, 2, 3, 4, 5, 6, 7, 8, 6, 7, 8, 9, 10, 11, 12, 9, 10, 11, 13, 13, 13, 13, 14],
        ],
        dtype=torch.long,
    )
    graph = igraph.Graph(
        n=15,
        edges=list(zip(edge_index[0].tolist(), edge_index[1].tolist())),
        directed=True,
    )
    reference = torch.tensor(
        graph.layout("sugiyama", maxiter=24, vgap=1.0, hgap=1.0).coords,
        dtype=torch.float32,
    )

    positions = layout_sugiyama_pipeline(
        edge_index=edge_index,
        num_nodes=15,
        rank_sep=1.0,
        node_sep=1.0,
        barycenter_passes=24,
        fidelity_mode="igraph",
    )

    assert torch.equal(positions, reference)


def test_sugiyama_igraph_glpk_falls_back_above_1000_nodes() -> None:
    """Igraph fidelity mode should use Eades layering above GLPK's node gate."""
    edge_index = torch.stack(
        [
            torch.arange(1000, dtype=torch.long),
            torch.arange(1, 1001, dtype=torch.long),
        ]
    )

    layers = _igraph_glpk_layer_assignments(edge_index=edge_index, num_nodes=1001)
    expected = _igraph_eades_layer_assignments(edge_index=edge_index, num_nodes=1001)

    assert torch.equal(layers, expected)


def test_sugiyama_igraph_undirected_gate_uses_bfs_fallback() -> None:
    """Undirected igraph Sugiyama should use the non-LP BFS fallback."""
    edge_index = torch.tensor(
        [
            [0, 1, 2],
            [1, 2, 3],
        ],
        dtype=torch.long,
    )

    layers = _igraph_glpk_layer_assignments(
        edge_index=edge_index,
        num_nodes=4,
        is_directed=False,
    )
    expected = _igraph_undirected_layer_assignments(
        edge_index=edge_index,
        edge_weights=None,
        num_nodes=4,
    )

    assert torch.equal(layers, expected)
    assert torch.equal(layers, torch.tensor([1, 0, 1, 2]))


def test_sugiyama_graphviz_fidelity_uses_dot_x_assignment() -> None:
    """Graphviz fidelity should keep ranks while using dot x assignment."""
    edge_index = torch.tensor(
        [
            [0, 0, 1, 2, 2, 3],
            [1, 2, 3, 3, 4, 4],
        ],
        dtype=torch.long,
    )

    positions = layout_sugiyama_pipeline(
        edge_index=edge_index,
        num_nodes=5,
        rank_sep=1.0,
        node_sep=1.0,
        fidelity_mode="graphviz",
    )

    expected = torch.tensor(
        [
            [-0.2, 0.0],
            [-0.6, 1.0],
            [0.2, 1.0],
            [-0.6, 2.0],
            [0.0, 3.0],
        ]
    )
    assert torch.equal(positions, expected)


def test_sugiyama_igraph_fidelity_packs_weak_components_independently() -> None:
    """Igraph fidelity mode should not globally order disconnected layers."""
    edge_index = torch.tensor(
        [
            [0, 2],
            [1, 3],
        ],
        dtype=torch.long,
    )

    positions = layout_sugiyama_pipeline(
        edge_index=edge_index,
        num_nodes=4,
        rank_sep=1.0,
        node_sep=1.0,
        fidelity_mode="igraph",
    )

    assert torch.allclose(positions[:, 0], torch.tensor([0.0, 0.0, 1.0, 1.0]))
    assert torch.allclose(positions[:, 1], torch.tensor([0.0, 1.0, 0.0, 1.0]))


def test_sugiyama_igraph_fidelity_ignores_node_width_spacing_by_default() -> None:
    """Igraph fidelity mode should use hgap-only compaction by default."""
    edge_index = torch.empty((2, 0), dtype=torch.long)
    node_sizes = torch.tensor([[10.0, 1.0], [20.0, 1.0]], dtype=torch.float32)

    positions = layout_sugiyama_pipeline(
        edge_index=edge_index,
        num_nodes=2,
        node_sizes=node_sizes,
        rank_sep=1.0,
        node_sep=1.0,
        fidelity_mode="igraph",
    )

    assert torch.allclose(positions[:, 0], torch.tensor([0.0, 1.0]))


def test_sugiyama_default_keeps_centered_node_width_spacing() -> None:
    """Default mode should preserve graphviz-style node-width spacing."""
    edge_index = torch.empty((2, 0), dtype=torch.long)
    node_sizes = torch.tensor([[10.0, 1.0], [20.0, 1.0]], dtype=torch.float32)

    positions = layout_sugiyama_pipeline(
        edge_index=edge_index,
        num_nodes=2,
        node_sizes=node_sizes,
        rank_sep=1.0,
        node_sep=1.0,
    )

    assert torch.allclose(positions[:, 0], torch.tensor([-8.0, 8.0]))
