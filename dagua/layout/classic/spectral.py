"""Spectral layout translated from NetworkX's reference implementation."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

SPARSE_EIGEN_THRESHOLD = 500


def _layout_device(
    edge_index: torch.Tensor,
    node_sizes: Optional[torch.Tensor],
) -> torch.device:
    """Choose the device used for the returned tensor.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor.

    Returns
    -------
    torch.device
        Output device for the final layout tensor.
    """
    if edge_index.numel() > 0:
        return edge_index.device
    if node_sizes is not None:
        return node_sizes.device
    return torch.device("cpu")


def _rescale_layout(positions: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Center and scale coordinates like ``networkx.rescale_layout``.

    Parameters
    ----------
    positions : numpy.ndarray
        Position array with shape ``[N, 2]``.
    scale : float, default=1.0
        Target half-width after rescaling.

    Returns
    -------
    numpy.ndarray
        Rescaled coordinate array.
    """
    positions = positions.copy()
    positions -= positions.mean(axis=0)
    limit = np.abs(positions).max()
    if limit > 0:
        positions *= scale / limit
    return positions


def _build_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor] = None,
) -> sparse.csr_matrix:
    """Build the directed sparse adjacency matrix from ``edge_index``.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse adjacency matrix with shape ``[N, N]``.

    Raises
    ------
    ValueError
        If ``edge_index`` has an invalid shape or contains out-of-range nodes.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")

    if edge_index.numel() == 0:
        return sparse.csr_matrix((num_nodes, num_nodes), dtype=np.float64)

    edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
    rows = edge_index_cpu[0].numpy()
    cols = edge_index_cpu[1].numpy()
    if (
        np.any(rows < 0)
        or np.any(rows >= num_nodes)
        or np.any(cols < 0)
        or np.any(cols >= num_nodes)
    ):
        raise ValueError("edge_index contains a node index outside [0, num_nodes).")

    if edge_weights is not None:
        data = edge_weights.detach().to(device="cpu").numpy().astype(np.float64)
    else:
        data = np.ones(rows.shape[0], dtype=np.float64)
    return sparse.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.float64)


def _spectral(adjacency: np.ndarray, dim: int = 2) -> np.ndarray:
    """Compute the dense spectral embedding exactly like NetworkX.

    Parameters
    ----------
    adjacency : numpy.ndarray
        Dense adjacency matrix with shape ``[N, N]``.
    dim : int, default=2
        Output dimension.

    Returns
    -------
    numpy.ndarray
        Raw spectral coordinates with shape ``[N, dim]``.
    """
    num_nodes, _ = adjacency.shape
    degree = np.identity(num_nodes, dtype=adjacency.dtype) * np.sum(adjacency, axis=1)
    laplacian = degree - adjacency
    eigenvalues, eigenvectors = np.linalg.eig(laplacian)
    index = np.argsort(eigenvalues)[1 : dim + 1]
    return np.real(eigenvectors[:, index])


def _sparse_spectral(adjacency: sparse.csr_matrix, dim: int = 2) -> np.ndarray:
    """Compute the sparse spectral embedding exactly like NetworkX.

    Parameters
    ----------
    adjacency : scipy.sparse.csr_matrix
        Sparse adjacency matrix with shape ``[N, N]``.
    dim : int, default=2
        Output dimension.

    Returns
    -------
    numpy.ndarray
        Raw spectral coordinates with shape ``[N, dim]``.
    """
    num_nodes, _ = adjacency.shape
    degree = sparse.dia_array((adjacency.sum(axis=1), 0), shape=(num_nodes, num_nodes)).tocsr()
    laplacian = degree - adjacency

    eigen_count = dim + 1
    lanczos_vectors = max((2 * eigen_count) + 1, int(np.sqrt(num_nodes)))
    eigenvalues, eigenvectors = sparse_linalg.eigsh(
        laplacian,
        eigen_count,
        which="SM",
        ncv=lanczos_vectors,
    )
    index = np.argsort(eigenvalues)[1:eigen_count]
    return np.real(eigenvectors[:, index])


def layout_spectral(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Lay out a graph with the NetworkX spectral layout translation.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Unused, accepted for interface compatibility.
    seed : int, default=42
        Accepted for interface compatibility. NetworkX's spectral layout does
        not use a seed on the translated path.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]`` that scales adjacency
        values before the Laplacian eigendecomposition.

    Returns
    -------
    torch.Tensor
        Final positions with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If ``num_nodes`` is negative.
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
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=torch.float32, device=device)
    if num_nodes == 2:
        return torch.zeros((2, 2), dtype=torch.float32, device=device)

    adjacency = _build_adjacency(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weights=edge_weights,
    )
    # Check if adjacency is already symmetric (edge_index has both directions).
    # NX only does A+A.T for directed graphs. If our edge_index already has
    # both (i,j) and (j,i), the adjacency is already symmetric — don't double.
    dense_adjacency = adjacency.toarray()
    is_symmetric = np.allclose(dense_adjacency, dense_adjacency.T)
    if not is_symmetric:
        dense_adjacency = dense_adjacency + dense_adjacency.T
    if num_nodes < SPARSE_EIGEN_THRESHOLD:
        positions = _spectral(dense_adjacency, dim=2)
    else:
        sparse_sym = sparse.csr_matrix(dense_adjacency)
        positions = _sparse_spectral(sparse_sym, dim=2)

    scaled = _rescale_layout(positions, scale=1.0)
    return torch.from_numpy(scaled).to(dtype=torch.float32, device=device)
