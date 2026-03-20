"""Spectral graph layout based on Laplacian eigenvectors.

This implementation mirrors NetworkX's spectral layout: build an adjacency
matrix, symmetrize directed inputs with ``A + A.T``, compute the unnormalized
Laplacian, extract the first non-trivial eigenvectors, and rescale the result
with the same centering rule.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

_EIGENVALUE_EPSILON = 1.0e-12
_SPARSE_EIG_THRESHOLD = 500


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


def _rescale_layout(positions: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Center and scale coordinates like ``networkx.rescale_layout``.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    scale : float, default=1.0
        Target half-width after rescaling.

    Returns
    -------
    torch.Tensor
        Rescaled positions with shape ``[N, 2]``.
    """
    centered = positions - positions.mean(dim=0, keepdim=True)
    limit = float(centered.abs().max().item())
    if limit > 0.0:
        centered = centered * (scale / limit)
    return centered


def _build_adjacency_matrix(edge_index: torch.Tensor, num_nodes: int) -> sparse.csr_matrix:
    """Build the symmetrized adjacency matrix used by NetworkX spectral layout.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    scipy.sparse.csr_matrix
        Symmetrized adjacency matrix with shape ``[N, N]``.

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
        rows.append(source)
        cols.append(target)

    adjacency = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, cols)),
        shape=(num_nodes, num_nodes),
        dtype=np.float64,
    )
    return (adjacency + adjacency.transpose()).tocsr()


def _select_nontrivial_vectors(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    num_nodes: int,
) -> torch.Tensor:
    """Select the first two non-trivial eigenvectors from an eigensystem.

    Parameters
    ----------
    eigenvalues : numpy.ndarray
        Eigenvalues with shape ``[K]``.
    eigenvectors : numpy.ndarray
        Eigenvectors with shape ``[N, K]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    torch.Tensor
        Coordinate matrix with shape ``[N, 2]``.
    """
    order = np.argsort(np.abs(eigenvalues))
    selected_indices = [
        index for index in order if abs(float(np.real(eigenvalues[index]))) > _EIGENVALUE_EPSILON
    ][:2]
    coordinates = np.real(eigenvectors[:, selected_indices])
    if coordinates.shape[1] == 2:
        return torch.from_numpy(coordinates).to(dtype=torch.float32)
    if coordinates.shape[1] == 1:
        zeros = torch.zeros((num_nodes, 1), dtype=torch.float32)
        return torch.cat((torch.from_numpy(coordinates).to(dtype=torch.float32), zeros), dim=1)
    return torch.zeros((num_nodes, 2), dtype=torch.float32)


def _dense_spectral(adjacency: np.ndarray, num_nodes: int) -> torch.Tensor:
    """Compute spectral coordinates with NumPy's dense eigensolver.

    Parameters
    ----------
    adjacency : numpy.ndarray
        Dense adjacency matrix with shape ``[N, N]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    torch.Tensor
        Raw spectral coordinates with shape ``[N, 2]``.
    """
    degree = np.identity(num_nodes, dtype=adjacency.dtype) * np.sum(adjacency, axis=1)
    laplacian = degree - adjacency
    eigenvalues, eigenvectors = np.linalg.eig(laplacian)
    return _select_nontrivial_vectors(eigenvalues, eigenvectors, num_nodes)


def _sparse_spectral(adjacency: sparse.csr_matrix, num_nodes: int) -> torch.Tensor:
    """Compute spectral coordinates with ARPACK like NetworkX.

    Parameters
    ----------
    adjacency : scipy.sparse.csr_matrix
        Sparse adjacency matrix with shape ``[N, N]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    torch.Tensor
        Raw spectral coordinates with shape ``[N, 2]``.
    """
    degree = sparse.dia_array((adjacency.sum(axis=1), 0), shape=(num_nodes, num_nodes)).tocsr()
    laplacian = degree - adjacency
    eigen_count = min(3, num_nodes - 1)
    if eigen_count <= 0:
        return torch.zeros((num_nodes, 2), dtype=torch.float32)

    ncv = max((2 * eigen_count) + 1, int(np.sqrt(num_nodes)))
    eigenvalues, eigenvectors = sparse_linalg.eigsh(
        laplacian,
        eigen_count,
        which="SM",
        ncv=ncv,
    )
    return _select_nontrivial_vectors(eigenvalues, eigenvectors, num_nodes)


def layout_spectral(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: int = 42,
) -> torch.Tensor:
    """Lay out a graph with the Laplacian spectral embedding.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Unused, accepted for interface compatibility.
    seed : int, default=42
        Accepted for interface compatibility. NetworkX's spectral layout does
        not use a random seed for the eigensolver path mirrored here.

    Returns
    -------
    torch.Tensor
        Node positions with shape ``[N, 2]``.
    """
    _ = node_sizes
    _ = seed

    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")

    device = _layout_device(edge_index, node_sizes)
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=device)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=torch.float32, device=device)
    if num_nodes == 2:
        return torch.zeros((2, 2), dtype=torch.float32, device=device)

    adjacency = _build_adjacency_matrix(edge_index=edge_index, num_nodes=num_nodes)
    if num_nodes < _SPARSE_EIG_THRESHOLD:
        raw_positions = _dense_spectral(adjacency.toarray(), num_nodes=num_nodes)
    else:
        raw_positions = _sparse_spectral(adjacency, num_nodes=num_nodes)

    return _rescale_layout(raw_positions.to(device=device), scale=1.0).to(dtype=torch.float32)
