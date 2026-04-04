"""Spectral layout expressed as a composable ops pipeline."""

from __future__ import annotations

from typing import ClassVar, Optional, Tuple

import numpy as np
import torch
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from dagua.layout.ops.graph_utils import layout_device as _layout_device

SPARSE_EIGEN_THRESHOLD = 500
_EIGENVALUE_TOLERANCE = 1.0e-9


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


def _symmetrized_adjacency(adjacency: sparse.csr_matrix) -> sparse.csr_matrix:
    """Build the undirected adjacency used by spectral layouts.

    Parameters
    ----------
    adjacency : scipy.sparse.csr_matrix
        Possibly directed adjacency matrix with shape ``[N, N]``.

    Returns
    -------
    scipy.sparse.csr_matrix
        Symmetric adjacency matrix.
    """
    difference = adjacency - adjacency.T
    if difference.nnz == 0:
        return adjacency
    return (adjacency + adjacency.T).tocsr()


def _laplacian_matrix(
    adjacency: sparse.csr_matrix,
    normalization: str,
) -> tuple[sparse.csr_matrix, bool]:
    """Build the requested graph Laplacian.

    Parameters
    ----------
    adjacency : scipy.sparse.csr_matrix
        Symmetric adjacency matrix with shape ``[N, N]``.
    normalization : str
        One of ``"symmetric"``, ``"random_walk"``, or ``"unnormalized"``.

    Returns
    -------
    tuple[scipy.sparse.csr_matrix, bool]
        Laplacian matrix and whether it is symmetric.

    Raises
    ------
    ValueError
        If ``normalization`` is unsupported.
    """
    degrees = np.asarray(adjacency.sum(axis=1)).reshape(-1).astype(np.float64, copy=False)
    degree_matrix = sparse.diags(degrees, offsets=0, format="csr")

    if normalization == "unnormalized":
        return (degree_matrix - adjacency).tocsr(), True
    if normalization == "symmetric":
        inv_sqrt = np.zeros_like(degrees)
        nonzero_mask = degrees > 0.0
        inv_sqrt[nonzero_mask] = 1.0 / np.sqrt(degrees[nonzero_mask])
        normalized = sparse.diags(inv_sqrt, offsets=0, format="csr")
        identity = sparse.identity(adjacency.shape[0], format="csr", dtype=np.float64)
        return (identity - (normalized @ adjacency @ normalized)).tocsr(), True
    if normalization == "random_walk":
        inv_degree = np.zeros_like(degrees)
        nonzero_mask = degrees > 0.0
        inv_degree[nonzero_mask] = 1.0 / degrees[nonzero_mask]
        normalized = sparse.diags(inv_degree, offsets=0, format="csr")
        identity = sparse.identity(adjacency.shape[0], format="csr", dtype=np.float64)
        return (identity - (normalized @ adjacency)).tocsr(), False
    raise ValueError("normalization must be one of 'symmetric', 'random_walk', or 'unnormalized'.")


def _select_embedding_columns(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    dim: int,
) -> np.ndarray:
    """Select the first non-trivial spectral coordinates.

    Parameters
    ----------
    eigenvalues : numpy.ndarray
        Eigenvalues with shape ``[K]``.
    eigenvectors : numpy.ndarray
        Eigenvectors with shape ``[N, K]``.
    dim : int
        Requested output dimension.

    Returns
    -------
    numpy.ndarray
        Coordinate matrix with shape ``[N, dim]``.
    """
    sorted_indices = np.argsort(np.real(eigenvalues))
    nontrivial_indices = [
        index
        for index in sorted_indices
        if abs(float(np.real(eigenvalues[index]))) > _EIGENVALUE_TOLERANCE
    ][:dim]

    num_nodes = eigenvectors.shape[0]
    coordinates = np.zeros((num_nodes, dim), dtype=np.float64)
    if nontrivial_indices:
        coordinates[:, : len(nontrivial_indices)] = np.real(eigenvectors[:, nontrivial_indices])
    elif num_nodes > 0:
        coordinates[:, 0] = np.linspace(-1.0, 1.0, num_nodes, dtype=np.float64)
    return coordinates


def _dense_spectral_embedding(
    laplacian: sparse.csr_matrix,
    dim: int,
    symmetric: bool,
) -> np.ndarray:
    """Compute a dense spectral embedding.

    Parameters
    ----------
    laplacian : scipy.sparse.csr_matrix
        Laplacian matrix with shape ``[N, N]``.
    dim : int
        Output dimension.
    symmetric : bool
        Whether ``laplacian`` is symmetric.

    Returns
    -------
    numpy.ndarray
        Raw coordinates with shape ``[N, dim]``.
    """
    dense_laplacian = laplacian.toarray()
    if symmetric:
        eigenvalues, eigenvectors = np.linalg.eigh(dense_laplacian)
    else:
        eigenvalues, eigenvectors = np.linalg.eig(dense_laplacian)
    return _select_embedding_columns(eigenvalues=eigenvalues, eigenvectors=eigenvectors, dim=dim)


def _sparse_spectral_embedding(
    laplacian: sparse.csr_matrix,
    dim: int,
    symmetric: bool,
) -> np.ndarray:
    """Compute a sparse spectral embedding.

    Parameters
    ----------
    laplacian : scipy.sparse.csr_matrix
        Laplacian matrix with shape ``[N, N]``.
    dim : int
        Output dimension.
    symmetric : bool
        Whether ``laplacian`` is symmetric.

    Returns
    -------
    numpy.ndarray
        Raw coordinates with shape ``[N, dim]``.
    """
    num_nodes = laplacian.shape[0]
    eigen_count = min(num_nodes - 1, max(dim + 4, dim + 1))
    if eigen_count <= dim:
        return _dense_spectral_embedding(laplacian=laplacian, dim=dim, symmetric=symmetric)

    lanczos_vectors = max((2 * eigen_count) + 1, int(np.sqrt(num_nodes)))
    if symmetric:
        eigenvalues, eigenvectors = sparse_linalg.eigsh(
            laplacian,
            k=eigen_count,
            which="SM",
            ncv=min(max(lanczos_vectors, eigen_count + 2), num_nodes),
        )
    else:
        eigenvalues, eigenvectors = sparse_linalg.eigs(
            laplacian,
            k=eigen_count,
            which="SR",
            ncv=min(max(lanczos_vectors, eigen_count + 2), num_nodes),
        )
    return _select_embedding_columns(eigenvalues=eigenvalues, eigenvectors=eigenvectors, dim=dim)


from dagua.layout.ops.base import Op, Pipeline  # noqa: E402
from dagua.layout.ops.state import (  # noqa: E402
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)
from dagua.layout.ops.taxonomy import OpCategory  # noqa: E402

_SYMMETRIC_FLAG_KEY = "spectral_is_symmetric"


class _PrepareSpectralState(Op):
    """Build the exact Laplacian state required by the classic solver."""

    name: ClassVar[str] = "spectral_prepare_state"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    writes: ClassVar[Tuple[str, ...]] = ("pos", "laplacian", "extras")

    def __init__(self, normalization: str) -> None:
        """Store the requested Laplacian normalization mode.

        Parameters
        ----------
        normalization : str
            Laplacian normalization mode.

        Returns
        -------
        None
            This constructor stores configuration only.
        """
        self.normalization = normalization

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Prepare trivial outputs or cache the spectral Laplacian.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this deterministic op.

        Returns
        -------
        SolveState
            State with trivial positions for degenerate graphs or the cached
            Laplacian for the eigensolver step.
        """
        del ctx

        if problem.num_nodes == 0:
            state.pos = torch.empty((0, 2), dtype=torch.float32)
            return state
        if problem.num_nodes == 1:
            state.pos = torch.zeros((1, 2), dtype=torch.float32)
            return state

        adjacency = _build_adjacency(
            edge_index=problem.edge_index,
            num_nodes=problem.num_nodes,
            edge_weights=problem.edge_weights,
        )
        laplacian, is_symmetric = _laplacian_matrix(
            adjacency=_symmetrized_adjacency(adjacency),
            normalization=self.normalization,
        )
        state.laplacian = laplacian
        state.extras[_SYMMETRIC_FLAG_KEY] = is_symmetric
        return state


class _EmbedSpectral(Op):
    """Recover the raw 2D spectral embedding."""

    name: ClassVar[str] = "spectral_embed"
    category: ClassVar[OpCategory] = OpCategory.EMBED
    reads: ClassVar[Tuple[str, ...]] = ("pos", "laplacian", "extras")
    writes: ClassVar[Tuple[str, ...]] = ("pos",)

    def __init__(self, sparse_threshold: int) -> None:
        """Store the dense-versus-sparse eigensolver threshold.

        Parameters
        ----------
        sparse_threshold : int
            Node-count threshold for the sparse eigensolver branch.

        Returns
        -------
        None
            This constructor stores configuration only.
        """
        self.sparse_threshold = int(sparse_threshold)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run the same dense or sparse eigensolver as the classic layout.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state containing the cached Laplacian.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this deterministic op.

        Returns
        -------
        SolveState
            State with raw spectral coordinates stored in ``state.pos``.

        Raises
        ------
        RuntimeError
            If the Laplacian or symmetry flag is missing.
        """
        del ctx

        if state.pos is not None:
            return state
        if state.laplacian is None:
            raise RuntimeError("_EmbedSpectral requires state.laplacian to be set.")
        is_symmetric = state.extras.get(_SYMMETRIC_FLAG_KEY)
        if not isinstance(is_symmetric, bool):
            raise RuntimeError("_EmbedSpectral requires a cached symmetry flag.")

        if problem.num_nodes < self.sparse_threshold:
            coordinates = _dense_spectral_embedding(
                laplacian=state.laplacian,
                dim=2,
                symmetric=is_symmetric,
            )
        else:
            coordinates = _sparse_spectral_embedding(
                laplacian=state.laplacian,
                dim=2,
                symmetric=is_symmetric,
            )

        state.pos = torch.from_numpy(coordinates)
        return state


class _FinalizeSpectralPositions(Op):
    """Apply the classic spectral center-and-scale normalization."""

    name: ClassVar[str] = "spectral_finalize_positions"
    category: ClassVar[OpCategory] = OpCategory.POSTPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Rescale raw coordinates like classic ``layout_spectral``.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state containing raw coordinates.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with final output positions.

        Raises
        ------
        ValueError
            If ``state.pos`` is missing.
        """
        del ctx

        if state.pos is None:
            raise ValueError("_FinalizeSpectralPositions requires state.pos to be set.")

        output_device = _layout_device(
            edge_index=problem.edge_index,
            node_sizes=problem.node_sizes,
        )
        if problem.num_nodes == 0:
            state.pos = torch.empty((0, 2), dtype=torch.float32, device=output_device)
            return state
        if problem.num_nodes == 1:
            state.pos = torch.zeros((1, 2), dtype=torch.float32, device=output_device)
            return state

        positions = state.pos.detach().to(device="cpu", dtype=torch.float64).numpy()
        scaled = _rescale_layout(positions, scale=1.0)
        state.pos = torch.from_numpy(scaled).to(dtype=torch.float32, device=output_device)
        return state


def build_spectral_pipeline(
    normalization: str = "symmetric",
    sparse_threshold: int = SPARSE_EIGEN_THRESHOLD,
) -> Pipeline:
    """Build a spectral pipeline that matches classic ``layout_spectral``.

    Parameters
    ----------
    normalization : str, default="symmetric"
        Laplacian normalization mode.
    sparse_threshold : int, default=500
        Node-count threshold for the sparse eigensolver branch.

    Returns
    -------
    Pipeline
        Pipeline that reproduces classic spectral layout exactly.

    Raises
    ------
    ValueError
        If ``sparse_threshold`` is not positive.
    """
    if sparse_threshold <= 0:
        raise ValueError("sparse_threshold must be positive.")

    return Pipeline(
        [
            _PrepareSpectralState(normalization=normalization),
            _EmbedSpectral(sparse_threshold=sparse_threshold),
            _FinalizeSpectralPositions(),
        ],
        name="spectral_pipeline",
    )


def layout_spectral_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
    normalization: str = "symmetric",
) -> torch.Tensor:
    """Run the spectral pipeline as a drop-in replacement.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Unused, accepted for interface compatibility.
    seed : int, default=42
        Accepted for interface compatibility. Spectral layout is deterministic
        once the graph is fixed.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.
    normalization : str, default="symmetric"
        Laplacian normalization mode.

    Returns
    -------
    torch.Tensor
        Final position tensor with the same dtype, device, and values as
        classic ``layout_spectral``.

    Raises
    ------
    ValueError
        If ``num_nodes`` or ``edge_weights`` are invalid.
    RuntimeError
        If the pipeline fails to populate final positions.
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

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_spectral_pipeline(normalization=normalization).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("Spectral pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_spectral_pipeline", "layout_spectral_pipeline"]
