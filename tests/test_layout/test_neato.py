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


def test_neato_polyomino_edges_round_pointf_head_cells() -> None:
    """Pin Graphviz ``pack.c:fillEdge`` pointf head-cell rounding.

    Returns
    -------
    None
        The edge cells should include the rounded pointf head endpoint rather
        than the floored integer cell used for node boxes.
    """
    info = neato._generate_node_polyomino(
        positions_points=torch.tensor([[0.0, 0.0], [400.0, 100.0]], dtype=torch.float64),
        sizes_points=torch.zeros((2, 2), dtype=torch.float64),
        local_edges=torch.tensor([[0], [1]], dtype=torch.long),
        bbox=(0.0, 0.0, 400.0, 100.0),
        step=154,
        margin=0.0,
        index=0,
    )

    assert info.cells == [(0, 0), (1, 0), (2, 0), (2, 1), (3, 1)]


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


def test_neato_disconnected_components_reuse_graphviz_start_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ordinary disconnected neato components reuse the graph-level start seed."""
    observed_seeds: list[int] = []

    def fake_stress_pipeline(**kwargs: object) -> torch.Tensor:
        """Record component seeds and return a finite component layout."""
        observed_seeds.append(int(kwargs["seed"]))
        num_nodes = int(kwargs["num_nodes"])
        return torch.zeros((num_nodes, 2), dtype=torch.float64)

    def fake_pack_component_positions(**kwargs: object) -> torch.Tensor:
        """Return a stable parent layout after component solving."""
        num_nodes = int(kwargs["num_nodes"])
        return torch.zeros((num_nodes, 2), dtype=torch.float64)

    monkeypatch.setattr(neato, "layout_stress_majorization_pipeline", fake_stress_pipeline)
    monkeypatch.setattr(neato, "_pack_component_positions", fake_pack_component_positions)
    edge_index = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)

    result = neato.layout_neato_pipeline(
        edge_index=edge_index,
        num_nodes=4,
        seed=123,
        pack=True,
        fidelity_mode=False,
    )

    assert torch.equal(result, torch.zeros((4, 2), dtype=torch.float64))
    assert observed_seeds == [123, 123]


def test_neato_singleton_heavy_components_keep_random_dag_seed_perturbation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Singleton-heavy disconnected neato graphs keep per-component seed offsets."""
    observed_seeds: list[int] = []

    def fake_stress_pipeline(**kwargs: object) -> torch.Tensor:
        """Record component seeds and return a finite component layout."""
        observed_seeds.append(int(kwargs["seed"]))
        num_nodes = int(kwargs["num_nodes"])
        return torch.zeros((num_nodes, 2), dtype=torch.float64)

    def fake_pack_component_positions(**kwargs: object) -> torch.Tensor:
        """Return a stable parent layout after component solving."""
        num_nodes = int(kwargs["num_nodes"])
        return torch.zeros((num_nodes, 2), dtype=torch.float64)

    monkeypatch.setattr(neato, "layout_stress_majorization_pipeline", fake_stress_pipeline)
    monkeypatch.setattr(neato, "_pack_component_positions", fake_pack_component_positions)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)

    result = neato.layout_neato_pipeline(
        edge_index=edge_index,
        num_nodes=10,
        seed=123,
        pack=True,
        fidelity_mode=False,
    )

    assert torch.equal(result, torch.zeros((10, 2), dtype=torch.float64))
    assert observed_seeds == [123, 124, 125, 126, 127, 128, 129, 130, 131]
