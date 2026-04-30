"""Regression tests for Sugiyama igraph-fidelity edge cases."""

import torch

from dagua.layout.ops.pipelines.sugiyama import layout_sugiyama_pipeline


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
            [0, 0, 4, 1],
            [5, 5, 5, 6],
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
