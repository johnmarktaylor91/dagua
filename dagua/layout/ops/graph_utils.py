"""Shared graph utility functions for layout operation pipelines.

Extracted from ``dagua.layout.classic`` to provide a single import point for
pipeline files under ``dagua.layout.ops.pipelines``.  Every function here is a
public, unmodified copy of its private classic-module original -- implementations
are bit-for-bit identical to their sources.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from dagua.layout._archive.classic._graph_distances import (
    all_pairs_shortest_paths as _shared_all_pairs_shortest_paths,
)
from dagua.layout._archive.classic._graph_distances import (
    bfs_distances,  # noqa: F401
    dijkstra_distances,  # noqa: F401
)
from dagua.layout._archive.classic._graph_distances import (
    bfs_distances as _shared_bfs_distances,  # noqa: F401, F811
)
from dagua.layout._archive.classic._graph_distances import (
    build_undirected_adjacency as _shared_build_undirected_adjacency,
)
from dagua.layout._archive.classic._graph_distances import (
    dijkstra_distances as _shared_dijkstra_distances,  # noqa: F401, F811
)

# ---------------------------------------------------------------------------
# 1. layout_device  (source: fr.py::_layout_device)
# ---------------------------------------------------------------------------


def layout_device(
    edge_index: torch.Tensor,
    node_sizes: Optional[torch.Tensor] = None,
) -> torch.device:
    """Choose the output device from input tensors.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor.

    Returns
    -------
    torch.device
        Output device for the final position tensor.
    """
    if edge_index.numel() > 0:
        return edge_index.device
    if node_sizes is not None:
        return node_sizes.device
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# 2. normalize_positions  (source: gem.py::_normalize_positions)
# ---------------------------------------------------------------------------


def normalize_positions(positions: torch.Tensor, extent: float) -> torch.Tensor:
    """Center and scale coordinates into a bounded drawing box.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    extent : float
        Target half-width.

    Returns
    -------
    torch.Tensor
        Normalized coordinates.
    """
    if positions.shape[0] <= 1:
        return torch.zeros_like(positions)

    centered = positions - positions.mean(dim=0, keepdim=True)
    span = centered.abs().max().clamp(min=1.0)
    return centered * (extent / span)


# ---------------------------------------------------------------------------
# 3. layout_extent  (source: gem.py::_layout_extent)
# ---------------------------------------------------------------------------


def layout_extent(num_nodes: int, node_sizes: Optional[torch.Tensor] = None) -> float:
    """Estimate a stable drawing extent from graph size and node sizes.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor.

    Returns
    -------
    float
        Target half-width.
    """
    if node_sizes is None or node_sizes.numel() == 0:
        return max(float(max(num_nodes, 1)) ** 0.5 * 5.0, 1.0)

    max_size = float(node_sizes.to(dtype=torch.float32, device="cpu").max().item())
    return max(max_size * max(float(max(num_nodes, 1)) ** 0.5, 1.0) * 2.0, 1.0)


# ---------------------------------------------------------------------------
# 4. build_undirected_adjacency  (source: gem.py::_build_undirected_adjacency)
#
# Note: gem.py accumulates duplicate-edge weights additively while
# _graph_distances.py keeps the minimum weight.  The gem.py variant is used
# here because it is the most common pattern across layout files (gem, fmmm,
# drl, maxent_stress, etc.).
# ---------------------------------------------------------------------------


def build_undirected_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor] = None,
) -> list[list[tuple[int, float]]]:
    """Build a symmetric weighted adjacency list from edge_index.

    Duplicate edges are accumulated additively (sum of weights).

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    edge_weights : torch.Tensor, optional
        Optional per-edge weights with shape ``[E]``.

    Returns
    -------
    list[list[tuple[int, float]]]
        One sorted undirected neighbor list per node storing
        ``(neighbor, edge_weight)`` pairs.
    """
    adjacency_maps: list[dict[int, float]] = [dict() for _ in range(num_nodes)]
    if edge_index.numel() == 0:
        return [[] for _ in range(num_nodes)]

    edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
    weights_cpu = (
        torch.ones((edge_index.shape[1],), dtype=torch.float32)
        if edge_weights is None
        else edge_weights.detach().to(device="cpu", dtype=torch.float32)
    )
    for edge_id, (source, target) in enumerate(
        zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist())
    ):
        if source == target:
            continue
        weight = float(weights_cpu[edge_id].item())
        adjacency_maps[source][target] = adjacency_maps[source].get(target, 0.0) + weight
        adjacency_maps[target][source] = adjacency_maps[target].get(source, 0.0) + weight
    return [sorted(neighbors.items()) for neighbors in adjacency_maps]


# ---------------------------------------------------------------------------
# 5. all_pairs_shortest_paths  (source: tsnet.py::_all_pairs_shortest_paths)
#
# Wraps the shared _graph_distances.all_pairs_shortest_paths with
# per-row unreachable fill and conversion to a torch float32 tensor.
# ---------------------------------------------------------------------------


def all_pairs_shortest_paths(
    adjacency: list[list[tuple[int, float]]],
    weighted: bool = False,
) -> torch.Tensor:
    """Compute the full shortest-path matrix with finite fills.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Undirected adjacency list.
    weighted : bool
        Whether to use Dijkstra instead of BFS.

    Returns
    -------
    torch.Tensor
        Distance matrix with shape ``[N, N]``.
    """
    raw_distances = _shared_all_pairs_shortest_paths(adjacency, weighted=weighted)
    if raw_distances.size == 0:
        return torch.empty((0, 0), dtype=torch.float32)

    cleaned = raw_distances.astype(np.float64, copy=True)
    for node in range(cleaned.shape[0]):
        row = cleaned[node]
        finite_mask = np.isfinite(row) if weighted else row >= 0
        max_distance = float(row[finite_mask].max()) if bool(finite_mask.any()) else 0.0
        fill_value = max_distance + 1.0 if cleaned.shape[0] > 1 else 0.0
        row[~finite_mask] = fill_value
    return torch.tensor(cleaned, dtype=torch.float32)


# ---------------------------------------------------------------------------
# 6. shortest_path_distances  (source: classical_mds.py::_shortest_path_distances)
#
# Uses the _graph_distances shared helpers for adjacency + APSP, then applies
# a global (not per-row) unreachable fill.  Returns numpy, not torch.
# ---------------------------------------------------------------------------


def shortest_path_distances(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """Compute a dense shortest-path distance matrix with finite fills.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.

    Returns
    -------
    numpy.ndarray
        Finite shortest-path distances with shape ``[N, N]``.
    """
    adjacency = _shared_build_undirected_adjacency(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weights=edge_weights,
    )
    weighted = edge_weights is not None
    raw_distances = _shared_all_pairs_shortest_paths(adjacency, weighted=weighted)
    finite_mask = np.isfinite(raw_distances) if weighted else raw_distances >= 0
    max_distance = float(raw_distances[finite_mask].max()) if bool(finite_mask.any()) else 0.0
    fill_value = max_distance + 1.0 if num_nodes > 1 else 0.0
    cleaned = np.where(finite_mask, raw_distances, fill_value).astype(np.float64, copy=False)
    np.fill_diagonal(cleaned, 0.0)
    return cleaned


# ---------------------------------------------------------------------------
# 7. build_directed_adjacency  (source: _graph_distances.py::build_directed_adjacency)
# ---------------------------------------------------------------------------


def build_directed_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor] = None,
) -> list[list[tuple[int, float]]]:
    """Build a directed weighted adjacency list from edge_index.

    Duplicate edges keep the minimum weight.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    edge_weights : torch.Tensor, optional
        Per-edge weights with shape ``[E]``. Defaults to 1.0 for all edges.

    Returns
    -------
    list[list[tuple[int, float]]]
        Directed adjacency list where each entry is ``(neighbor, weight)``.
    """
    adjacency: list[dict[int, float]] = [{} for _ in range(num_nodes)]
    if edge_index.numel() == 0:
        return [[] for _ in range(num_nodes)]

    ei_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    sources = ei_cpu[0].tolist()
    targets = ei_cpu[1].tolist()

    if edge_weights is not None:
        weights = edge_weights.detach().cpu().float().tolist()
    else:
        weights = [1.0] * len(sources)

    for src, tgt, weight in zip(sources, targets, weights):
        if tgt not in adjacency[src] or weight < adjacency[src][tgt]:
            adjacency[src][tgt] = weight

    return [sorted(neighbors.items()) for neighbors in adjacency]


# ---------------------------------------------------------------------------
# 8. rescale_layout  (source: fr.py::_rescale_layout)
# ---------------------------------------------------------------------------


def rescale_layout(positions: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Center and scale coordinates like ``networkx.rescale_layout``.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    scale : float
        Target half-width after rescaling.

    Returns
    -------
    torch.Tensor
        Rescaled position tensor with shape ``[N, 2]``.
    """
    centered = positions - positions.mean(dim=0, keepdim=True)
    limit = float(centered.abs().max().item())
    if limit > 0.0:
        centered = centered * (scale / limit)
    return centered


# ---------------------------------------------------------------------------
# 9. initialize_numpy_positions  (source: fr.py::_initialize_positions)
# ---------------------------------------------------------------------------


def initialize_numpy_positions(num_nodes: int, seed: int) -> torch.Tensor:
    """Create the exact NetworkX random initialization in the unit square.

    Uses ``numpy.random.RandomState`` for deterministic placement identical
    to the NetworkX Fruchterman-Reingold implementation.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    seed : int
        Random seed for the initial placement.

    Returns
    -------
    torch.Tensor
        Initial positions with shape ``[N, 2]`` and dtype ``float64``.
    """
    random_state = np.random.RandomState(seed)
    positions = np.asarray(random_state.rand(num_nodes, 2), dtype=np.float64)
    return torch.from_numpy(positions)


# ---------------------------------------------------------------------------
# 10. initialize_torch_positions  (source: linlog.py::_initialize_positions)
# ---------------------------------------------------------------------------


def initialize_torch_positions(
    num_nodes: int,
    seed: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create deterministic random initial coordinates via torch.randn.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    seed : int
        Random seed.
    device : torch.device, optional
        Device for the position tensor. Default is CPU.

    Returns
    -------
    torch.Tensor
        Initial positions with shape ``[N, 2]``.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    positions = torch.randn((num_nodes, 2), generator=generator, dtype=torch.float32)
    return positions.to(device)
