"""Classical multidimensional scaling from all-pairs graph distances."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from dagua.layout._archive.classic._graph_distances import (
    all_pairs_shortest_paths as _shared_all_pairs_shortest_paths,
)
from dagua.layout._archive.classic._graph_distances import (
    build_undirected_adjacency as _shared_build_undirected_adjacency,
)

_MIN_SPAN = 1.0e-6


def _layout_device(
    edge_index: torch.Tensor,
    node_sizes: Optional[torch.Tensor],
) -> torch.device:
    """Choose the device used for the returned layout tensor.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]``.

    Returns
    -------
    torch.device
        Device for the final output tensor.
    """
    if edge_index.numel() > 0:
        return edge_index.device
    if node_sizes is not None:
        return node_sizes.device
    return torch.device("cpu")


def _layout_extent(num_nodes: int, node_sizes: Optional[torch.Tensor]) -> float:
    """Estimate a stable output scale for the embedding.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]``.

    Returns
    -------
    float
        Target half-width after normalization.
    """
    if node_sizes is None or node_sizes.numel() == 0:
        return max(float(max(num_nodes, 1)) ** 0.5 * 5.0, 1.0)

    max_size = float(node_sizes.to(dtype=torch.float32, device="cpu").max().item())
    return max(max_size * max(float(max(num_nodes, 1)) ** 0.5, 1.0) * 2.0, 1.0)


def _normalize_positions(positions: torch.Tensor, extent: float) -> torch.Tensor:
    """Center and scale coordinates into a stable bounding box.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    extent : float
        Target half-width after normalization.

    Returns
    -------
    torch.Tensor
        Centered and scaled coordinates.
    """
    if positions.shape[0] <= 1:
        return torch.zeros_like(positions)

    centered = positions - positions.mean(dim=0, keepdim=True)
    span = float(centered.abs().max().item())
    if span < _MIN_SPAN:
        centered = centered.clone()
        centered[:, 0] = torch.linspace(
            -1.0,
            1.0,
            steps=positions.shape[0],
            device=positions.device,
            dtype=positions.dtype,
        )
        span = float(centered.abs().max().item())
    return centered * (extent / max(span, _MIN_SPAN))


def _shortest_path_distances(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor],
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


def _classical_mds_embedding(distances: np.ndarray) -> torch.Tensor:
    """Recover a rank-2 embedding from pairwise graph distances.

    Parameters
    ----------
    distances : numpy.ndarray
        Dense shortest-path distance matrix with shape ``[N, N]``.

    Returns
    -------
    torch.Tensor
        Raw coordinates with shape ``[N, 2]`` on CPU.
    """
    num_nodes = int(distances.shape[0])
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float32)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=torch.float32)

    squared = distances * distances
    centering = np.eye(num_nodes, dtype=np.float64) - (
        np.ones((num_nodes, num_nodes), dtype=np.float64) / float(num_nodes)
    )
    gram = -0.5 * centering @ squared @ centering

    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    positive_indices = [index for index in sorted_indices if eigenvalues[index] > 0.0][:2]

    coordinates = np.zeros((num_nodes, 2), dtype=np.float64)
    if positive_indices:
        selected_values = np.clip(eigenvalues[positive_indices], a_min=0.0, a_max=None)
        selected_vectors = eigenvectors[:, positive_indices]
        coordinates[:, : len(positive_indices)] = selected_vectors * np.sqrt(selected_values)
    else:
        coordinates[:, 0] = np.linspace(-1.0, 1.0, num_nodes, dtype=np.float64)

    return torch.from_numpy(coordinates).to(dtype=torch.float32)


def layout_classical_mds(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Lay out a graph with classical multidimensional scaling.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]`` used only to pick a
        stable drawing extent.
    seed : int, default=42
        Accepted for interface compatibility. Classical MDS is deterministic
        once the graph distances are fixed.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]`` used when computing
        shortest-path distances.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If ``num_nodes`` is negative or ``edge_weights`` has the wrong shape.
    """
    _ = seed

    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must have shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge count {edge_index.shape[1]}"
            )

    device = _layout_device(edge_index=edge_index, node_sizes=node_sizes)
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=device)

    distances = _shortest_path_distances(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weights=edge_weights,
    )
    raw_positions = _classical_mds_embedding(distances)
    extent = _layout_extent(num_nodes=num_nodes, node_sizes=node_sizes)
    normalized = _normalize_positions(raw_positions.to(device=device), extent=extent)
    return normalized.to(dtype=torch.float32, device=device)
