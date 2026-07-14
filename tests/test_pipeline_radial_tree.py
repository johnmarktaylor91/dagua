"""Pipeline pins for igraph circular Reingold-Tilford radial trees."""

from __future__ import annotations

import math
from pathlib import Path

import igraph
import torch

import dagua
from dagua.config import LayoutConfig
from dagua.layout.ops.pipelines import PIPELINE_REGISTRY, get_pipeline_function
from dagua.layout.ops.pipelines.radial_tree import (
    layout_radial_tree_pipeline,
    radial_tree_from_rt_units,
)


def _edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    """Build a ``[2, E]`` edge tensor.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Source-target edge pairs.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _igraph_circular_reference(edges: list[tuple[int, int]], num_nodes: int) -> torch.Tensor:
    """Run igraph's circular RT reference and apply Dagua adapter scaling.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Source-target edge pairs.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    torch.Tensor
        Reference coordinates with shape ``[N, 2]``.
    """
    graph = igraph.Graph(n=num_nodes, edges=edges, directed=True)
    layout = graph.layout_reingold_tilford_circular(mode="out")
    positions = torch.zeros((num_nodes, 2), dtype=torch.float64)
    for node in range(num_nodes):
        positions[node, 0] = float(layout[node][0]) * 50.0
        positions[node, 1] = float(layout[node][1]) * 50.0
    return positions


def test_radial_tree_pipeline_is_registered() -> None:
    """Register the public radial-tree algorithm.

    Returns
    -------
    None
        Registry lookups must resolve the radial-tree entrypoint.
    """
    assert PIPELINE_REGISTRY["radial_tree"] == (
        "dagua.layout.ops.pipelines.radial_tree",
        "layout_radial_tree_pipeline",
    )
    assert get_pipeline_function("RADIAL_TREE") is layout_radial_tree_pipeline


def test_radial_tree_polar_transform_matches_igraph_formula() -> None:
    """Pin igraph's circular RT polar transform.

    Returns
    -------
    None
        The transform must use ``2*pi*(N-1)/N`` scaled by RT x-span.
    """
    rt = torch.tensor(
        [[0.0, 0.0], [-1.5, 2.0], [-0.5, 2.0], [0.5, 2.0], [1.5, 2.0]],
        dtype=torch.float64,
    )
    observed = radial_tree_from_rt_units(rt)
    ratio = 2.0 * math.pi * 4.0 / 5.0 / 3.0
    expected = torch.tensor(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0 * math.cos(ratio), 2.0 * math.sin(ratio)],
            [2.0 * math.cos(2.0 * ratio), 2.0 * math.sin(2.0 * ratio)],
            [2.0 * math.cos(3.0 * ratio), 2.0 * math.sin(3.0 * ratio)],
        ],
        dtype=torch.float64,
    )

    torch.testing.assert_close(observed, expected, rtol=0.0, atol=1.0e-12)


def test_radial_tree_matches_igraph_binary_tree() -> None:
    """Match igraph circular RT on a directed binary tree.

    Returns
    -------
    None
        Raw coordinates must match the igraph reference after adapter scaling.
    """
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
    observed = layout_radial_tree_pipeline(_edge_index(edges), 7)
    reference = _igraph_circular_reference(edges, 7)

    torch.testing.assert_close(observed, reference, rtol=0.0, atol=1.0e-5)


def test_radial_tree_matches_igraph_star_tree() -> None:
    """Match igraph circular RT on a star tree.

    Returns
    -------
    None
        The angular spacing must match igraph's ``N``-dependent ratio.
    """
    edges = [(0, 1), (0, 2), (0, 3), (0, 4)]
    observed = layout_radial_tree_pipeline(_edge_index(edges), 5)
    reference = _igraph_circular_reference(edges, 5)

    torch.testing.assert_close(observed, reference, rtol=0.0, atol=1.0e-5)


def test_layout_config_algorithm_radial_tree_dispatches() -> None:
    """Exercise public engine dispatch for ``algorithm='radial_tree'``.

    Returns
    -------
    None
        Dispatch must return finite coordinates.
    """
    graph = dagua.DaguaGraph.from_edge_list([("a", "b"), ("a", "c"), ("b", "d")])
    positions = dagua.layout(graph, LayoutConfig(algorithm="radial_tree"))

    assert positions.shape == (4, 2)
    assert torch.isfinite(positions).all()


def test_radial_tree_pipeline_has_no_runtime_igraph_delegation() -> None:
    """Guard the production pipeline against python-igraph delegation.

    Returns
    -------
    None
        The pipeline source must not import or call igraph.
    """
    source = (
        Path(__file__).parents[1] / "dagua" / "layout" / "ops" / "pipelines" / "radial_tree.py"
    ).read_text()
    assert "import igraph" not in source
    assert "layout_reingold_tilford_circular" not in source
