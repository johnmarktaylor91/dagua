"""Tests for Graphviz neato-compatible dispatch."""

from __future__ import annotations

import pytest
import torch

import dagua
from dagua.layout.ops.pipelines import neato


def test_public_neato_algorithm_dispatch() -> None:
    """Public ``dagua.layout`` accepts ``algorithm="neato"``."""
    graph = dagua.DaguaGraph.from_edge_list([("a", "b"), ("b", "c"), ("c", "a")])

    pos = dagua.layout(graph, dagua.LayoutConfig(algorithm="neato", seed=42))

    assert pos.shape == (3, 2)
    assert torch.isfinite(pos).all()


def test_stress_majorization_graphviz_neato_fidelity_switch() -> None:
    """The stress pipeline accepts the neato fidelity compatibility switch."""
    graph = dagua.DaguaGraph.from_edge_list([("a", "b"), ("b", "c"), ("d", "e")])

    pos = dagua.layout(
        graph,
        dagua.LayoutConfig(
            algorithm="stress_majorization",
            seed=42,
            algorithm_params={"graphviz_neato_fidelity": True},
        ),
    )

    assert pos.shape == (5, 2)
    assert torch.isfinite(pos).all()


def test_neato_polyomino_packing_matches_graphviz_scan_golden() -> None:
    """Disconnected neato packing follows Graphviz pack.c's polyomino scan."""
    components = [[0], [1], [2]]
    component_positions = [
        torch.tensor([[0.0, 0.0]], dtype=torch.float64),
        torch.tensor([[0.0, 0.0]], dtype=torch.float64),
        torch.tensor([[0.0, 0.0]], dtype=torch.float64),
    ]
    component_edges = [torch.empty((2, 0), dtype=torch.long) for _ in components]
    node_sizes = torch.tensor(
        [[0.75, 0.5], [2.0, 0.5], [0.75, 1.5]],
        dtype=torch.float64,
    )

    packed = neato._pack_component_positions(
        components=components,
        component_positions=component_positions,
        component_edges=component_edges,
        num_nodes=3,
        node_sizes=node_sizes,
    )

    expected = torch.tensor(
        [[-0.375, 0.25], [-0.25, -0.75], [0.625, 0.5]],
        dtype=torch.float64,
    )
    assert torch.allclose(packed, expected)


def test_neato_connected_graph_skips_component_packer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Connected neato layouts do not call the disconnected-component packer."""

    def fake_stress_pipeline(**kwargs: object) -> torch.Tensor:
        """Return a finite connected layout for the packer bypass check."""
        return torch.zeros((3, 2), dtype=torch.float64)

    def fail_pack(**kwargs: object) -> torch.Tensor:
        """Raise if the disconnected packer is called for one component."""
        raise AssertionError("component packer should not run")

    monkeypatch.setattr(neato, "layout_stress_majorization_pipeline", fake_stress_pipeline)
    monkeypatch.setattr(neato, "_pack_component_positions", fail_pack)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

    result = neato.layout_neato_pipeline(
        edge_index=edge_index,
        num_nodes=3,
        pack=True,
        fidelity_mode=False,
    )

    assert torch.equal(result, torch.zeros((3, 2), dtype=torch.float64))
