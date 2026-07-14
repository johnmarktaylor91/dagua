"""Fidelity tests for the Chrobak-Payne planar pipeline."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
import torch

from dagua.config import LayoutConfig
from dagua.graph import DaguaGraph
from dagua.layout import layout
from dagua.layout.ops.pipelines import get_pipeline_function
from dagua.layout.ops.pipelines.planar import layout_planar_pipeline


def _edge_index_from_graph(graph: nx.Graph) -> torch.Tensor:
    """Return a Dagua edge tensor from a NetworkX graph.

    Parameters
    ----------
    graph : nx.Graph
        Graph with integer node ids.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    edges = list(graph.edges())
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _reference_planar_tensor(graph: nx.Graph) -> torch.Tensor:
    """Return NetworkX planar-layout positions as a tensor.

    Parameters
    ----------
    graph : nx.Graph
        Reference graph with integer node ids.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]`` and dtype ``torch.float64``.
    """
    pos = nx.planar_layout(graph)
    array = np.vstack([pos[node] for node in range(graph.number_of_nodes())])
    return torch.from_numpy(array).to(dtype=torch.float64)


@pytest.mark.parametrize(
    "graph",
    [
        nx.path_graph(4),
        nx.cycle_graph(6),
        nx.complete_graph(4),
        nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 3)),
        nx.triangular_lattice_graph(2, 3, with_positions=False),
    ],
)
def test_layout_planar_pipeline_matches_networkx_bit_exact(graph: nx.Graph) -> None:
    """The direct pipeline output should be bit-exact against NetworkX."""
    graph = nx.convert_node_labels_to_integers(graph)
    edge_index = _edge_index_from_graph(graph)

    actual = layout_planar_pipeline(
        edge_index,
        graph.number_of_nodes(),
        fidelity_dtype=torch.float64,
    )
    expected = _reference_planar_tensor(graph)

    assert torch.equal(actual.cpu(), expected)


def test_planar_algorithm_is_registered_without_colliding_with_native_planar() -> None:
    """The new planar key should resolve distinctly from native_planar."""
    planar_fn = get_pipeline_function("planar")
    native_planar_fn = get_pipeline_function("native_planar")

    assert planar_fn is layout_planar_pipeline
    assert planar_fn is not native_planar_fn


def test_layout_config_algorithm_planar_dispatches() -> None:
    """LayoutConfig should dispatch algorithm='planar' through the registry."""
    graph = DaguaGraph.from_edge_list([(0, 1), (1, 2), (2, 3)])

    pos = layout(graph, LayoutConfig(algorithm="planar", fidelity_dtype=torch.float64))
    expected = _reference_planar_tensor(nx.path_graph(4)).to(dtype=torch.float32)

    assert torch.equal(pos.cpu(), expected)
