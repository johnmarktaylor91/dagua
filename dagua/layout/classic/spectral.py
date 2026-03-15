"""Spectral graph layout based on Laplacian eigenvectors.

This implementation follows the standard Hall/Koren-style spectral drawing
approach: build the unweighted graph Laplacian, extract the first non-trivial
eigenvectors, and use them as planar coordinates.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

_MIN_SPAN = 1.0e-6
_SPARSE_EIGH_THRESHOLD = 512


def _layout_device(
    edge_index: torch.Tensor,
    node_sizes: Optional[torch.Tensor],
) -> torch.device:
    """Choose the compute device for the returned layout tensor.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor.

    Returns
    -------
    torch.device
        Device used by the output tensor.
    """
    if edge_index.numel() > 0:
        return edge_index.device
    if node_sizes is not None:
        return node_sizes.device
    return torch.device("cpu")


def _layout_extent(num_nodes: int, node_sizes: Optional[torch.Tensor]) -> float:
    """Estimate a reasonable half-width for the final drawing.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor.

    Returns
    -------
    float
        Target extent for each axis after normalization.
    """
    if node_sizes is None or node_sizes.numel() == 0:
        return max(float(max(num_nodes, 1)) ** 0.5 * 5.0, 1.0)

    max_size = float(node_sizes.to(dtype=torch.float32, device="cpu").max().item())
    return max(max_size * max(float(max(num_nodes, 1)) ** 0.5, 1.0) * 2.0, 1.0)


def _fallback_positions(
    num_nodes: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """Create deterministic fallback positions for degenerate eigensystems.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    device : torch.device
        Device for the returned tensor.
    seed : int
        Random seed for deterministic jitter.

    Returns
    -------
    torch.Tensor
        Fallback coordinates with shape ``[N, 2]``.
    """
    if num_nodes <= 1:
        return torch.zeros((num_nodes, 2), dtype=torch.float32, device=device)

    angles = torch.linspace(0.0, 2.0 * torch.pi, steps=num_nodes + 1, dtype=torch.float32)[:-1]
    circle = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    jitter = 0.05 * torch.randn((num_nodes, 2), generator=generator, dtype=torch.float32)
    return (circle + jitter).to(device)


def _normalize_positions(
    positions: torch.Tensor,
    extent: float,
    seed: int,
) -> torch.Tensor:
    """Center and scale coordinates to a stable drawing box.

    Parameters
    ----------
    positions : torch.Tensor
        Raw coordinates with shape ``[N, 2]``.
    extent : float
        Target half-width of the drawing.
    seed : int
        Random seed used if the layout is degenerate.

    Returns
    -------
    torch.Tensor
        Normalized coordinates with shape ``[N, 2]``.
    """
    if positions.shape[0] <= 1:
        return torch.zeros_like(positions)

    centered = positions - positions.mean(dim=0, keepdim=True)
    span = float(centered.abs().max().item())
    if span < _MIN_SPAN:
        return _fallback_positions(int(positions.shape[0]), positions.device, seed) * extent
    return centered * (extent / span)


def _laplacian_matrix(edge_index: torch.Tensor, num_nodes: int) -> sparse.csr_matrix:
    """Build the unweighted symmetric graph Laplacian.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse Laplacian matrix with shape ``[N, N]``.

    Raises
    ------
    ValueError
        If ``edge_index`` does not have shape ``[2, E]`` or contains invalid
        node indices.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")

    if edge_index.numel() == 0:
        return sparse.csr_matrix((num_nodes, num_nodes), dtype=np.float64)

    edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
    sources = edge_index_cpu[0].tolist()
    targets = edge_index_cpu[1].tolist()

    rows: list[int] = []
    cols: list[int] = []
    for source, target in zip(sources, targets):
        if source < 0 or source >= num_nodes or target < 0 or target >= num_nodes:
            raise ValueError("edge_index contains a node index outside [0, num_nodes).")
        if source == target:
            continue
        rows.extend((source, target))
        cols.extend((target, source))

    if not rows:
        return sparse.csr_matrix((num_nodes, num_nodes), dtype=np.float64)

    adjacency = sparse.coo_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, cols)),
        shape=(num_nodes, num_nodes),
        dtype=np.float64,
    ).tocsr()
    adjacency.data[:] = 1.0
    adjacency.eliminate_zeros()
    degrees = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    degree_matrix = sparse.diags(degrees, offsets=0, format="csr")
    return degree_matrix - adjacency


def _spectral_coordinates(laplacian: sparse.csr_matrix, seed: int) -> torch.Tensor:
    """Extract the first two non-trivial Laplacian eigenvectors.

    Parameters
    ----------
    laplacian : scipy.sparse.csr_matrix
        Graph Laplacian.
    seed : int
        Random seed used by the sparse solver fallback.

    Returns
    -------
    torch.Tensor
        Raw coordinate matrix with shape ``[N, 2]``.
    """
    num_nodes = int(laplacian.shape[0])
    if num_nodes <= 1:
        return torch.zeros((num_nodes, 2), dtype=torch.float32)

    if num_nodes >= _SPARSE_EIGH_THRESHOLD:
        eigen_count = min(4, num_nodes - 1)
        if eigen_count > 0:
            values, vectors = sparse_linalg.eigsh(
                laplacian,
                k=eigen_count,
                which="SM",
                tol=1.0e-4,
                v0=np.random.default_rng(seed).normal(size=num_nodes),
            )
            order = np.argsort(values)
            values = values[order]
            vectors = vectors[:, order]
            valid = vectors[:, values > 1.0e-8]
            if valid.shape[1] >= 2:
                return torch.from_numpy(valid[:, :2]).to(dtype=torch.float32)

    dense_laplacian = torch.from_numpy(laplacian.toarray()).to(dtype=torch.float32)
    eigenvalues, eigenvectors = torch.linalg.eigh(dense_laplacian)
    valid_mask = eigenvalues > 1.0e-8
    valid_vectors = eigenvectors[:, valid_mask]
    if valid_vectors.shape[1] >= 2:
        return valid_vectors[:, :2]
    if valid_vectors.shape[1] == 1:
        zeros = torch.zeros((num_nodes, 1), dtype=torch.float32)
        return torch.cat((valid_vectors, zeros), dim=1)
    return torch.zeros((num_nodes, 2), dtype=torch.float32)


def layout_spectral(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: int = 42,
) -> torch.Tensor:
    """Lay out a graph with the Laplacian spectral embedding.

    Reference
    ---------
    Hall, "An r-Dimensional Quadratic Placement Algorithm" (1970); Koren,
    "On Spectral Graph Drawing" (2003).

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor, optional
        Optional node sizes used only to scale the final drawing.
    seed : int, default=42
        Random seed used by solver fallbacks.

    Returns
    -------
    torch.Tensor
        Node positions with shape ``[N, 2]``.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")

    device = _layout_device(edge_index, node_sizes)
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=device)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=torch.float32, device=device)

    laplacian = _laplacian_matrix(edge_index, num_nodes)
    raw_positions = _spectral_coordinates(laplacian, seed)
    extent = _layout_extent(num_nodes, node_sizes)
    normalized = _normalize_positions(raw_positions.to(device), extent, seed)
    return normalized.to(dtype=torch.float32, device=device)
