"""Embedding and dimensionality-reduction operations for layout pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log2
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy import optimize, sparse
from scipy.sparse import linalg as sparse_linalg
from torch import nn

from dagua.layout.ops.base import Op
from dagua.layout.ops.preprocess import _SYMMETRIC_FLAG_KEY
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_LAPLACIAN_KEY = "laplacian"
_NORMALIZED_ADJACENCY_KEY = "normalized_adjacency"
_EIGENPAIRS_KEY = "eigenpairs"
_SVD_RESULT_KEY = "svd_result"
_LAPLACIAN_PINV_KEY = "laplacian_pinv"
_GCN_MODEL_KEY = "gcn_model"
_GCN_ADJACENCY_REF_KEY = "gcn_model_adjacency"
_PROBABILITIES_KEY = "probabilities"
_SIGMAS_KEY = "sigmas"
_RHOS_KEY = "rhos"
_FUZZY_GRAPH_KEY = "fuzzy_graph"
_UMAP_A_KEY = "umap_a"
_UMAP_B_KEY = "umap_b"
_UMAP_N_NEIGHBORS_KEY = "umap_n_neighbors"

_EPSILON = 1.0e-12
_MIN_SPAN = 1.0e-6
_MIN_SIGMA_SCALE = 1.0e-3
_DEFAULT_CURVE_A = 1.93
_DEFAULT_CURVE_B = 0.79
_GCN_GAIN_FALLBACK_DIM = 2
_CLASSICAL_MDS_MIN_SPAN = 1.0e-6
_SPECTRAL_EIGENVALUE_TOLERANCE = 1.0e-9
_EMBEDDING_OUTPUT_DIM = 2
_SPECTRAL_EXTRA_EIGENPAIRS = 4
_SPECTRAL_LANCZOS_MULTIPLIER = 2
_SPECTRAL_LANCZOS_PADDING = 2
_GCN_REQUIRED_HIDDEN_LAYER_COUNT = 2
_TSNE_JOINT_PROBABILITY_DIVISOR = 2.0
_CURVE_FIT_INITIAL_GUESS = (_DEFAULT_CURVE_A, _DEFAULT_CURVE_B)

AdjacencyList = List[List[Union[int, Tuple[int, float]]]]
CSRAdjacency = Dict[str, torch.Tensor]
AdjacencyValue = Union[AdjacencyList, CSRAdjacency, torch.Tensor, np.ndarray]


def _adjacency_shape(adjacency: AdjacencyValue) -> Tuple[int, int]:
    """Infer the matrix shape for an adjacency payload.

    Parameters
    ----------
    adjacency : AdjacencyValue
        Supported adjacency representation.

    Returns
    -------
    tuple[int, int]
        Adjacency matrix shape.

    Raises
    ------
    TypeError
        If ``adjacency`` uses an unsupported representation.
    """
    if isinstance(adjacency, list):
        return len(adjacency), len(adjacency)
    if isinstance(adjacency, dict):
        indptr = adjacency["indptr"]
        num_nodes = max(int(indptr.numel()) - 1, 0)
        return num_nodes, num_nodes
    if isinstance(adjacency, torch.Tensor):
        return int(adjacency.shape[0]), int(adjacency.shape[1])
    if isinstance(adjacency, np.ndarray):
        return int(adjacency.shape[0]), int(adjacency.shape[1])
    raise TypeError(f"Unsupported adjacency type: {type(adjacency)!r}.")


def _adjacency_to_sparse_matrix(adjacency: AdjacencyValue) -> sparse.csr_matrix:
    """Convert an adjacency payload into a SciPy CSR matrix.

    Parameters
    ----------
    adjacency : AdjacencyValue
        Adjacency list, CSR payload, dense torch tensor, sparse torch tensor,
        or dense NumPy array.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse adjacency matrix in ``float64`` precision.

    Raises
    ------
    TypeError
        If ``adjacency`` uses an unsupported representation.
    """
    shape = _adjacency_shape(adjacency)

    if isinstance(adjacency, list):
        row_index: List[int] = []
        col_index: List[int] = []
        values: List[float] = []
        for source, neighbors in enumerate(adjacency):
            for entry in neighbors:
                if isinstance(entry, tuple):
                    target, weight = entry
                    value = float(weight)
                else:
                    target = int(entry)
                    value = 1.0
                row_index.append(source)
                col_index.append(int(target))
                values.append(value)
        return sparse.csr_matrix((values, (row_index, col_index)), shape=shape, dtype=np.float64)

    if isinstance(adjacency, dict):
        indptr = adjacency["indptr"].detach().to(device="cpu", dtype=torch.long).numpy()
        indices = adjacency["indices"].detach().to(device="cpu", dtype=torch.long).numpy()
        weights = adjacency["weights"].detach().to(device="cpu", dtype=torch.float64).numpy()
        return sparse.csr_matrix((weights, indices, indptr), shape=shape, dtype=np.float64)

    if isinstance(adjacency, torch.Tensor):
        if adjacency.is_sparse:
            coalesced = adjacency.coalesce()
            indices = coalesced.indices().detach().to(device="cpu", dtype=torch.long).numpy()
            values = coalesced.values().detach().to(device="cpu", dtype=torch.float64).numpy()
            return sparse.csr_matrix(
                (values, (indices[0], indices[1])),
                shape=shape,
                dtype=np.float64,
            )
        dense = adjacency.detach().to(device="cpu", dtype=torch.float64)
        mask = torch.isfinite(dense) & (dense != 0)
        row_index, col_index = torch.nonzero(mask, as_tuple=True)
        values = dense[row_index, col_index].numpy()
        return sparse.csr_matrix(
            (values, (row_index.numpy(), col_index.numpy())),
            shape=shape,
            dtype=np.float64,
        )

    if isinstance(adjacency, np.ndarray):
        dense = np.asarray(adjacency, dtype=np.float64)
        mask = np.isfinite(dense) & (dense != 0)
        row_index, col_index = np.nonzero(mask)
        values = dense[row_index, col_index]
        return sparse.csr_matrix((values, (row_index, col_index)), shape=shape, dtype=np.float64)

    raise TypeError(f"Unsupported adjacency type: {type(adjacency)!r}.")


def _sparse_matrix_to_list(matrix: sparse.csr_matrix) -> AdjacencyList:
    """Convert a sparse matrix into list-adjacency form.

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix
        Sparse matrix with shape ``[N, N]``.

    Returns
    -------
    AdjacencyList
        Weighted adjacency list with deterministic row order.
    """
    csr = matrix.tocsr()
    adjacency: AdjacencyList = []
    for node_idx in range(csr.shape[0]):
        start = int(csr.indptr[node_idx])
        end = int(csr.indptr[node_idx + 1])
        row = [
            (int(csr.indices[offset]), float(csr.data[offset]))
            for offset in range(start, end)
            if csr.data[offset] != 0
        ]
        adjacency.append(row)
    return adjacency


def _sparse_matrix_to_csr_payload(matrix: sparse.csr_matrix) -> CSRAdjacency:
    """Convert a sparse matrix into the ops-package CSR payload format.

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix
        Sparse matrix with shape ``[N, N]``.

    Returns
    -------
    CSRAdjacency
        Payload containing ``indptr``, ``indices``, and ``weights`` tensors.
    """
    csr = matrix.tocsr()
    return {
        "indptr": torch.from_numpy(csr.indptr.astype(np.int64, copy=False)),
        "indices": torch.from_numpy(csr.indices.astype(np.int64, copy=False)),
        "weights": torch.from_numpy(csr.data.astype(np.float64, copy=False)),
    }


def _sparse_matrix_to_torch(
    matrix: sparse.csr_matrix,
    template: Optional[torch.Tensor] = None,
    sparse_output: bool = False,
) -> torch.Tensor:
    """Convert a SciPy sparse matrix into a torch tensor.

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix
        Sparse matrix with shape ``[N, N]``.
    template : torch.Tensor | None, optional
        Optional tensor whose dtype and device should be preserved when
        reasonable.
    sparse_output : bool, default=False
        Whether to emit a sparse COO tensor.

    Returns
    -------
    torch.Tensor
        Dense or sparse torch tensor containing the matrix values.
    """
    device = template.device if template is not None else torch.device("cpu")
    dtype = (
        template.dtype if template is not None and template.is_floating_point() else torch.float32
    )
    coo = matrix.tocoo()
    indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64, copy=False))
    values = torch.from_numpy(coo.data.astype(np.float32, copy=False)).to(dtype=dtype)
    if sparse_output:
        return torch.sparse_coo_tensor(indices, values, matrix.shape, dtype=dtype, device=device)
    dense = torch.from_numpy(matrix.toarray()).to(dtype=dtype, device=device)
    return dense


def _restore_adjacency_type(
    matrix: sparse.csr_matrix,
    original: AdjacencyValue,
) -> AdjacencyValue:
    """Restore a sparse matrix to the original adjacency representation.

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix
        Sparse matrix to restore.
    original : AdjacencyValue
        Original adjacency payload used as the type template.

    Returns
    -------
    AdjacencyValue
        Matrix converted back into the original container type.
    """
    if isinstance(original, list):
        return _sparse_matrix_to_list(matrix)
    if isinstance(original, dict):
        return _sparse_matrix_to_csr_payload(matrix)
    if isinstance(original, torch.Tensor):
        return _sparse_matrix_to_torch(
            matrix,
            template=original,
            sparse_output=bool(original.is_sparse),
        )
    if isinstance(original, np.ndarray):
        return matrix.toarray()
    raise TypeError(f"Unsupported adjacency type: {type(original)!r}.")


def _pivot_mds_coordinates(distance_matrix: torch.Tensor) -> torch.Tensor:
    """Recover a 2D pivot-MDS embedding from pivot distance rows.

    Parameters
    ----------
    distance_matrix : torch.Tensor
        Pivot-to-node distances with shape ``[P, N]``.

    Returns
    -------
    torch.Tensor
        Coordinates with shape ``[N, 2]``.
    """
    if distance_matrix.shape[0] == 0:
        return torch.zeros((distance_matrix.shape[1], 2), dtype=torch.float32)

    squared = distance_matrix.square()
    row_means = squared.mean(dim=1, keepdim=True)
    col_means = squared.mean(dim=0, keepdim=True)
    grand_mean = squared.mean()
    centered = -0.5 * (squared - row_means - col_means + grand_mean)

    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    coord_dims = min(2, int(singular_values.shape[0]))
    if coord_dims == 0:
        return torch.zeros((distance_matrix.shape[1], 2), dtype=torch.float32)

    scales = singular_values[:coord_dims].clamp_min(0.0)
    coordinates = vh[:coord_dims].transpose(0, 1) * scales.unsqueeze(0)
    if coord_dims == 1:
        zeros = torch.zeros((coordinates.shape[0], 1), dtype=coordinates.dtype)
        coordinates = torch.cat((coordinates, zeros), dim=1)
    return coordinates.to(dtype=torch.float32)


def _classical_mds_embedding(
    distances: torch.Tensor,
    *,
    igraph_fidelity: bool = False,
) -> torch.Tensor:
    """Compute rank-2 classical MDS coordinates from a distance matrix.

    Parameters
    ----------
    distances : torch.Tensor
        Dense pairwise distances with shape ``[N, N]``.
    igraph_fidelity : bool, default=False
        If ``True``, match igraph's raw connected-component MDS semantics:
        the two-node special case, largest algebraic eigenpairs,
        ``sqrt(abs(lambda))`` scaling, and reversed output dimensions.

    Returns
    -------
    torch.Tensor
        Raw coordinates with shape ``[N, 2]``.
    """
    num_nodes = int(distances.shape[0])
    if num_nodes == 0:
        return torch.zeros((0, 2), dtype=torch.float32, device=distances.device)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=torch.float32, device=distances.device)

    if distances.shape[1] != num_nodes:
        raise ValueError("Classical-MDS embedding requires a square distance matrix.")

    if igraph_fidelity and num_nodes == 2:
        return torch.tensor(
            [[0.0, 0.0], [1.0, 1.0]],
            dtype=torch.float32,
            device=distances.device,
        )

    distances_np = distances.detach().to(dtype=torch.float64).numpy()
    squared = distances_np * distances_np
    centering = np.eye(num_nodes, dtype=np.float64) - (
        np.ones((num_nodes, num_nodes), dtype=np.float64) / float(num_nodes)
    )
    gram = -0.5 * centering @ squared @ centering

    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    sorted_indices = np.argsort(eigenvalues)[::-1]

    coordinates = np.zeros((num_nodes, 2), dtype=np.float64)
    if igraph_fidelity:
        selected_indices = sorted_indices[:2]
        selected_values = eigenvalues[selected_indices]
        selected_vectors = eigenvectors[:, selected_indices]
        scaled = selected_vectors * np.sqrt(np.abs(selected_values))
        coordinates[:, : len(selected_indices)] = scaled
        coordinates = coordinates[:, ::-1].copy()
    elif positive_indices := [index for index in sorted_indices if eigenvalues[index] > 0.0][:2]:
        selected_values = np.clip(eigenvalues[positive_indices], a_min=0.0, a_max=None)
        selected_vectors = eigenvectors[:, positive_indices]
        coordinates[:, : len(positive_indices)] = selected_vectors * np.sqrt(selected_values)
    else:
        coordinates[:, 0] = np.linspace(-1.0, 1.0, num_nodes, dtype=np.float64)

    return torch.from_numpy(coordinates).to(dtype=torch.float32, device=distances.device)


def _payload_to_sparse_matrix(payload: Any) -> sparse.csr_matrix:
    """Convert a matrix-like payload into SciPy CSR form.

    Parameters
    ----------
    payload : Any
        Sparse or dense matrix payload.

    Returns
    -------
    scipy.sparse.csr_matrix
        CSR matrix in ``float64`` precision.

    Raises
    ------
    TypeError
        If ``payload`` cannot be interpreted as a matrix.
    """
    if sparse.isspmatrix(payload):
        return payload.tocsr().astype(np.float64)
    if isinstance(payload, (list, dict, torch.Tensor, np.ndarray)):
        return _adjacency_to_sparse_matrix(payload)
    raise TypeError(f"Unsupported matrix payload type: {type(payload)!r}.")


def _is_symmetric(matrix: sparse.csr_matrix, tolerance: float = 1.0e-9) -> bool:
    """Check whether a sparse matrix is numerically symmetric.

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix
        Matrix to test.
    tolerance : float, default=1.0e-9
        Absolute tolerance for the matrix difference.

    Returns
    -------
    bool
        ``True`` when ``matrix`` and ``matrix.T`` match within ``tolerance``.
    """
    difference = (matrix - matrix.transpose()).tocsr()
    if difference.nnz == 0:
        return True
    return bool(np.max(np.abs(difference.data)) <= tolerance)


def _numpy_array_to_torch(values: np.ndarray) -> torch.Tensor:
    """Convert a NumPy array into a contiguous torch tensor.

    Parameters
    ----------
    values : numpy.ndarray
        Input array.

    Returns
    -------
    torch.Tensor
        Torch tensor preserving real or complex dtype.
    """
    converted = np.real_if_close(np.asarray(values))
    contiguous = np.ascontiguousarray(converted)
    return torch.from_numpy(contiguous)


def _knn_from_distances(
    distances: torch.Tensor,
    n_neighbors: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract the nearest neighbors from a dense distance matrix.

    Parameters
    ----------
    distances : torch.Tensor
        Dense distance matrix with shape ``[N, N]``.
    n_neighbors : int
        Requested number of neighbors per node.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Neighbor indices and distances with shapes ``[N, K]``.
    """
    num_nodes = int(distances.shape[0])
    if num_nodes == 0:
        empty_index = torch.empty((0, 0), dtype=torch.long)
        empty_distance = torch.empty((0, 0), dtype=torch.float32)
        return empty_index, empty_distance

    k = min(n_neighbors, max(num_nodes - 1, 1))
    adjusted = distances.detach().to(dtype=torch.float32).clone()
    adjusted.fill_diagonal_(float("inf"))
    knn_distances, knn_indices = torch.topk(adjusted, k=k, largest=False, dim=1)
    return knn_indices.to(dtype=torch.long), knn_distances.to(dtype=torch.float32)


def _perplexity_row(
    distances: torch.Tensor,
    perplexity: float,
    tol: float,
    max_iter: int,
) -> torch.Tensor:
    """Match one conditional Gaussian row to a target perplexity.

    Parameters
    ----------
    distances : torch.Tensor
        Distance vector with shape ``[N]``.
    perplexity : float
        Target perplexity.
    tol : float
        Absolute entropy tolerance.
    max_iter : int
        Maximum binary-search iterations.

    Returns
    -------
    torch.Tensor
        Conditional probability row with shape ``[N]``.
    """
    num_nodes = int(distances.shape[0])
    if num_nodes <= 1:
        return torch.zeros_like(distances, dtype=torch.float32)

    row = distances.detach().to(dtype=torch.float32)
    mask = torch.ones(num_nodes, dtype=torch.bool, device=row.device)
    self_index = int(torch.argmin(row).item())
    mask[self_index] = False

    squared = row.square()
    beta = 1.0
    beta_min: Optional[float] = None
    beta_max: Optional[float] = None
    target_entropy = float(np.log(perplexity))
    probabilities = torch.zeros_like(row, dtype=torch.float32)

    for _ in range(max_iter):
        weights = torch.exp(-squared * beta) * mask.to(dtype=torch.float32)
        weights_sum = weights.sum().clamp(min=_EPSILON)
        probabilities = weights / weights_sum
        active = probabilities[mask]
        entropy = -(active * active.clamp(min=_EPSILON).log()).sum()
        error = float(entropy.item()) - target_entropy
        if abs(error) <= tol:
            break
        if error > 0.0:
            beta_min = beta
            beta = beta * 2.0 if beta_max is None else 0.5 * (beta + beta_max)
        else:
            beta_max = beta
            beta = beta * 0.5 if beta_min is None else 0.5 * (beta + beta_min)

    probabilities[self_index] = 0.0
    return probabilities


def _smooth_knn_bandwidth(
    knn_distances: torch.Tensor,
    n_neighbors: int,
    tol: float,
    max_iter: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Solve UMAP's smooth-kNN bandwidth equation.

    Parameters
    ----------
    knn_distances : torch.Tensor
        K-nearest-neighbor distances with shape ``[N, K]``.
    n_neighbors : int
        Neighborhood size used to define the smooth-kNN target.
    tol : float
        Absolute binary-search tolerance for the membership sum.
    max_iter : int
        Maximum binary-search iterations.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``sigma`` and ``rho`` vectors with shape ``[N]``.
    """
    num_nodes = int(knn_distances.shape[0])
    if num_nodes == 0:
        empty = torch.empty((0,), dtype=torch.float32)
        return empty, empty

    sigmas = torch.empty((num_nodes,), dtype=torch.float32)
    rhos = torch.empty((num_nodes,), dtype=torch.float32)
    target = log2(float(max(n_neighbors, 2)))

    for index in range(num_nodes):
        distances = knn_distances[index]
        finite = distances[torch.isfinite(distances)]
        if finite.numel() == 0:
            sigmas[index] = 1.0
            rhos[index] = 0.0
            continue

        positive = finite[finite > 0]
        rho = float(positive.min().item()) if positive.numel() > 0 else 0.0
        rhos[index] = rho
        mean_distance = max(float(finite.mean().item()), _MIN_SPAN)
        sigma_min = mean_distance * _MIN_SIGMA_SCALE
        lower = 0.0
        upper = 1.0

        def _membership_sum(sigma: float) -> float:
            """Evaluate the smooth-kNN membership sum for one sigma.

            Parameters
            ----------
            sigma : float
                Candidate local bandwidth.

            Returns
            -------
            float
                Sum of soft memberships for the row.
            """
            if sigma <= 0.0:
                return float(finite[1:].numel())
            shifted = torch.clamp(finite[1:] - rho, min=0.0)
            values = torch.exp(-shifted / sigma)
            return float(values.sum().item())

        while _membership_sum(upper) < target:
            upper *= 2.0
            if upper > 1.0e6:
                break

        sigma = upper
        for _ in range(max_iter):
            sigma = 0.5 * (lower + upper)
            estimate = _membership_sum(max(sigma, sigma_min))
            if abs(estimate - target) <= tol:
                break
            if estimate > target:
                upper = sigma
            else:
                lower = sigma

        sigmas[index] = max(sigma, sigma_min)

    return sigmas, rhos


def _fuzzy_simplicial_graph(
    knn_indices: torch.Tensor,
    knn_distances: torch.Tensor,
    sigmas: torch.Tensor,
    rhos: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Build the symmetrized fuzzy simplicial set from local neighborhoods.

    Parameters
    ----------
    knn_indices : torch.Tensor
        Neighbor indices with shape ``[N, K]``.
    knn_distances : torch.Tensor
        Neighbor distances with shape ``[N, K]``.
    sigmas : torch.Tensor
        Smooth-kNN bandwidths with shape ``[N]``.
    rhos : torch.Tensor
        Local connectivity radii with shape ``[N]``.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary containing ``head``, ``tail``, and ``weight`` tensors.
    """
    directed_weights: Dict[Tuple[int, int], float] = {}
    num_nodes, num_neighbors = knn_indices.shape

    for row in range(num_nodes):
        sigma = float(sigmas[row].item())
        rho = float(rhos[row].item())
        for column in range(num_neighbors):
            neighbor = int(knn_indices[row, column].item())
            distance = float(knn_distances[row, column].item())
            if not np.isfinite(distance):
                continue
            if distance <= rho or sigma <= 0.0:
                weight = 1.0
            else:
                weight = float(np.exp(-(distance - rho) / sigma))
            directed_weights[(row, neighbor)] = weight

    undirected: Dict[Tuple[int, int], float] = {}
    handled: set[Tuple[int, int]] = set()
    for source, target in directed_weights:
        key = (min(source, target), max(source, target))
        if source == target or key in handled:
            continue
        handled.add(key)
        forward = directed_weights.get((source, target), 0.0)
        backward = directed_weights.get((target, source), 0.0)
        weight = forward + backward - (forward * backward)
        if weight > 0.0:
            undirected[key] = weight

    if not undirected:
        empty_index = torch.empty((0,), dtype=torch.long)
        empty_weight = torch.empty((0,), dtype=torch.float32)
        return {"head": empty_index, "tail": empty_index.clone(), "weight": empty_weight}

    pairs = list(undirected.keys())
    weights = list(undirected.values())
    return {
        "head": torch.tensor([pair[0] for pair in pairs], dtype=torch.long),
        "tail": torch.tensor([pair[1] for pair in pairs], dtype=torch.long),
        "weight": torch.tensor(weights, dtype=torch.float32),
    }


def _curve_function(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Evaluate the UMAP attraction curve used for fitting ``a`` and ``b``.

    Parameters
    ----------
    x : numpy.ndarray
        Input distances.
    a : float
        UMAP curve coefficient ``a``.
    b : float
        UMAP curve coefficient ``b``.

    Returns
    -------
    numpy.ndarray
        Attraction probabilities for ``x``.
    """
    return 1.0 / (1.0 + (a * np.power(x, 2.0 * b)))


def _resolve_generator(ctx: RuntimeContext, problem: LayoutProblem) -> torch.Generator:
    """Resolve the torch RNG used for deterministic GCN initialization.

    Parameters
    ----------
    ctx : RuntimeContext
        Execution context that may already supply a generator.
    problem : LayoutProblem
        Layout problem providing the fallback seed.

    Returns
    -------
    torch.Generator
        CPU generator for deterministic parameter initialization.
    """
    if ctx.generator is not None:
        return ctx.generator
    generator = torch.Generator(device="cpu")
    generator.manual_seed(problem.seed)
    return generator


def _xavier_uniform_(
    tensor: torch.Tensor,
    gain: float,
    generator: torch.Generator,
) -> None:
    """Initialize a tensor with Xavier-uniform weights using a private generator.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to initialize in place.
    gain : float
        Xavier gain factor.
    generator : torch.Generator
        CPU generator controlling the random draw sequence.

    Returns
    -------
    None
        The tensor is initialized in place.
    """
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    std = gain * np.sqrt(2.0 / float(fan_in + fan_out))
    bound = np.sqrt(3.0) * std
    cpu_values = torch.empty(tuple(tensor.shape), dtype=tensor.dtype, device="cpu")
    cpu_values.uniform_(-bound, bound, generator=generator)
    tensor.data.copy_(cpu_values.to(device=tensor.device, dtype=tensor.dtype))


class _SparseGCNLayer(nn.Module):
    """Sparse GCN layer implementing ``A_norm @ (X @ W)`` without bias."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        adj_norm: torch.Tensor,
        gain: float,
        generator: torch.Generator,
    ) -> None:
        """Initialize the sparse GCN layer.

        Parameters
        ----------
        in_dim : int
            Input feature dimension.
        out_dim : int
            Output feature dimension.
        adj_norm : torch.Tensor
            Sparse normalized adjacency with shape ``[N, N]``.
        gain : float
            Xavier gain factor.
        generator : torch.Generator
            Generator used for deterministic initialization.
        """
        super().__init__()
        self.adj_norm = adj_norm
        self.weight = nn.Parameter(
            torch.empty((in_dim, out_dim), device=adj_norm.device, dtype=torch.float32)
        )
        _xavier_uniform_(self.weight, gain=gain, generator=generator)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one sparse GCN message-passing step.

        Parameters
        ----------
        x : torch.Tensor
            Node features with shape ``[N, in_dim]``.

        Returns
        -------
        torch.Tensor
            Updated node features with shape ``[N, out_dim]``.
        """
        support = torch.mm(x, self.weight)
        return torch.sparse.mm(self.adj_norm, support)


class _ResGCNModel(nn.Module):
    """Residual GCN matching NeuLay's skip-concatenation architecture."""

    def __init__(
        self,
        num_nodes: int,
        hidden_sizes: Tuple[int, int],
        output_dim: int,
        adj_norm: torch.Tensor,
        generator: torch.Generator,
    ) -> None:
        """Initialize the residual GCN model.

        Parameters
        ----------
        num_nodes : int
            Number of graph nodes.
        hidden_sizes : tuple[int, int]
            Hidden widths ``(hidden_1, hidden_2)``.
        output_dim : int
            Output coordinate dimension.
        adj_norm : torch.Tensor
            Sparse normalized adjacency with shape ``[N, N]``.
        generator : torch.Generator
            Generator used for deterministic weight initialization.
        """
        super().__init__()
        hidden_1, hidden_2 = hidden_sizes
        effective_dim = max(output_dim, _GCN_GAIN_FALLBACK_DIM)
        gain = float(max(num_nodes, 1)) ** (1.0 / float(effective_dim))

        self.hidden_sizes = hidden_sizes
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.weight1 = nn.Parameter(
            torch.empty((num_nodes, hidden_1), device=adj_norm.device, dtype=torch.float32)
        )
        _xavier_uniform_(self.weight1, gain=gain, generator=generator)

        self.gcn1 = _SparseGCNLayer(
            in_dim=hidden_1,
            out_dim=hidden_1,
            adj_norm=adj_norm,
            gain=gain,
            generator=generator,
        )
        self.gcn2 = _SparseGCNLayer(
            in_dim=hidden_1,
            out_dim=hidden_2,
            adj_norm=adj_norm,
            gain=gain,
            generator=generator,
        )

        concat_dim = hidden_1 + hidden_1 + hidden_2
        self.weight2 = nn.Parameter(
            torch.empty((concat_dim, output_dim), device=adj_norm.device, dtype=torch.float32)
        )
        _xavier_uniform_(self.weight2, gain=gain, generator=generator)

    def forward(self) -> torch.Tensor:
        """Generate output coordinates from the learned graph embedding.

        Returns
        -------
        torch.Tensor
            Coordinate tensor with shape ``[N, output_dim]``.
        """
        h0 = self.weight1
        h1 = torch.tanh(self.gcn1(h0))
        h2 = self.gcn2(h1)
        return torch.mm(torch.cat([h0, h1, h2], dim=1), self.weight2)


def _normalized_adjacency_to_sparse_torch(payload: Any) -> torch.Tensor:
    """Convert a normalized adjacency payload into sparse torch COO format.

    Parameters
    ----------
    payload : Any
        Sparse or dense matrix-like normalized adjacency.

    Returns
    -------
    torch.Tensor
        Sparse COO tensor with shape ``[N, N]`` and dtype ``float32``.
    """
    if isinstance(payload, torch.Tensor):
        if payload.is_sparse:
            return payload.coalesce().to(dtype=torch.float32)
        dense = payload.to(dtype=torch.float32)
        indices = torch.nonzero(dense != 0, as_tuple=False).transpose(0, 1)
        values = dense[dense != 0]
        return torch.sparse_coo_tensor(indices, values, dense.shape, dtype=torch.float32).coalesce()

    matrix = _payload_to_sparse_matrix(payload).tocoo()
    indices = torch.from_numpy(np.vstack((matrix.row, matrix.col)).astype(np.int64, copy=False))
    values = torch.from_numpy(matrix.data.astype(np.float32, copy=False))
    return torch.sparse_coo_tensor(indices, values, matrix.shape, dtype=torch.float32).coalesce()


@register_op
class SymmetrizeAdjacency(Op):
    """Mirror the active adjacency so downstream embedding ops see an undirected graph."""

    name: ClassVar[str] = "symmetrize_adjacency"
    category: ClassVar[OpCategory] = OpCategory.EMBED
    reads: ClassVar[Tuple[str, ...]] = ("adjacency",)
    writes: ClassVar[Tuple[str, ...]] = ("adjacency",)
    requires: ClassVar[Tuple[str, ...]] = ("adjacency",)
    access_pattern: ClassVar[str] = "global"

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Symmetrize the active adjacency payload.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused for this deterministic op.
        state : SolveState
            Mutable solve state with ``adjacency`` populated.
        ctx : RuntimeContext
            Execution context. Unused for this deterministic op.

        Returns
        -------
        SolveState
            State with symmetric adjacency restored in the original container type.
        """
        _ = problem
        _ = ctx
        if state.adjacency is None:
            raise ValueError("SymmetrizeAdjacency requires state.adjacency to be set.")
        matrix = _adjacency_to_sparse_matrix(state.adjacency)
        symmetrized = (matrix + matrix.transpose()).tocsr()
        # Restore the caller's original container type so later ops can keep
        # using the same adjacency representation they started with.
        state.adjacency = _restore_adjacency_type(symmetrized, state.adjacency)
        return state


@dataclass(frozen=True)
class BuildLaplacianConfig:
    """Configuration for :class:`BuildLaplacian`.

    Parameters
    ----------
    normalization : str, default="symmetric"
        Laplacian normalization mode. Supported values are ``"symmetric"``,
        ``"random_walk"``, and ``"unnormalized"``.
    """

    normalization: str = "symmetric"


@register_op
class BuildLaplacian(Op):
    """Build the configured Laplacian variant and cache it for embedding ops."""

    name: ClassVar[str] = "build_laplacian"
    category: ClassVar[OpCategory] = OpCategory.EMBED
    reads: ClassVar[Tuple[str, ...]] = ("adjacency",)
    writes: ClassVar[Tuple[str, ...]] = (f"extras.{_LAPLACIAN_KEY}",)
    requires: ClassVar[Tuple[str, ...]] = ("adjacency",)
    access_pattern: ClassVar[str] = "global"

    def __init__(self, config: Optional[BuildLaplacianConfig] = None) -> None:
        """Store the Laplacian builder configuration.

        Parameters
        ----------
        config : BuildLaplacianConfig | None, optional
            Optional op configuration.
        """
        self.config = config or BuildLaplacianConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute and store the configured graph Laplacian.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused for this deterministic op.
        state : SolveState
            Mutable solve state with ``adjacency`` populated.
        ctx : RuntimeContext
            Execution context. Unused for this deterministic op.

        Returns
        -------
        SolveState
            State with ``extras["laplacian"]`` populated.
        """
        _ = problem
        _ = ctx
        if state.adjacency is None:
            raise ValueError("BuildLaplacian requires state.adjacency to be set.")

        adjacency = _adjacency_to_sparse_matrix(state.adjacency)
        degrees = np.asarray(adjacency.sum(axis=1)).reshape(-1).astype(np.float64, copy=False)
        degree_matrix = sparse.diags(degrees, offsets=0, format="csr")

        if self.config.normalization == "unnormalized":
            laplacian = degree_matrix - adjacency
        elif self.config.normalization == "symmetric":
            inv_sqrt = np.zeros_like(degrees)
            nonzero_mask = degrees > 0.0
            # Leave isolated nodes at zero so the normalized Laplacian stays
            # finite without inventing degree mass.
            inv_sqrt[nonzero_mask] = 1.0 / np.sqrt(degrees[nonzero_mask])
            normalized = sparse.diags(inv_sqrt, offsets=0, format="csr")
            identity = sparse.identity(adjacency.shape[0], format="csr", dtype=np.float64)
            laplacian = identity - (normalized @ adjacency @ normalized)
        elif self.config.normalization == "random_walk":
            inv_degree = np.zeros_like(degrees)
            nonzero_mask = degrees > 0.0
            # Random-walk normalization uses the same isolated-node convention
            # as the symmetric branch to preserve the legacy sparse output.
            inv_degree[nonzero_mask] = 1.0 / degrees[nonzero_mask]
            normalized = sparse.diags(inv_degree, offsets=0, format="csr")
            identity = sparse.identity(adjacency.shape[0], format="csr", dtype=np.float64)
            laplacian = identity - (normalized @ adjacency)
        else:
            raise ValueError(
                "normalization must be one of 'symmetric', 'random_walk', or 'unnormalized'."
            )

        state.extras[_LAPLACIAN_KEY] = laplacian.tocsr()
        return state


@dataclass(frozen=True)
class BuildNormalizedAdjacencyConfig:
    """Configuration for :class:`BuildNormalizedAdjacency`.

    Parameters
    ----------
    add_self_loops : bool, default=True
        Whether to add the identity matrix before degree normalization.
    """

    add_self_loops: bool = True


@register_op
class BuildNormalizedAdjacency(Op):
    """Build NeuLay's symmetric degree-normalized adjacency with optional self-loops."""

    name: ClassVar[str] = "build_normalized_adjacency"
    category: ClassVar[OpCategory] = OpCategory.EMBED
    reads: ClassVar[Tuple[str, ...]] = ("adjacency",)
    writes: ClassVar[Tuple[str, ...]] = (f"extras.{_NORMALIZED_ADJACENCY_KEY}",)
    requires: ClassVar[Tuple[str, ...]] = ("adjacency",)
    access_pattern: ClassVar[str] = "global"

    def __init__(self, config: Optional[BuildNormalizedAdjacencyConfig] = None) -> None:
        """Store the normalized-adjacency configuration.

        Parameters
        ----------
        config : BuildNormalizedAdjacencyConfig | None, optional
            Optional op configuration.
        """
        self.config = config or BuildNormalizedAdjacencyConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute ``D^(-1/2) (A + I) D^(-1/2)`` as sparse COO.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused for this deterministic op.
        state : SolveState
            Mutable solve state with ``adjacency`` populated.
        ctx : RuntimeContext
            Execution context. Unused for this deterministic op.

        Returns
        -------
        SolveState
            State with ``extras["normalized_adjacency"]`` populated.
        """
        _ = problem
        _ = ctx
        if state.adjacency is None:
            raise ValueError("BuildNormalizedAdjacency requires state.adjacency to be set.")

        adjacency = _adjacency_to_sparse_matrix(state.adjacency).tocsr()
        if adjacency.nnz > 0:
            # NeuLay treats the graph as unweighted at this stage, so collapse
            # any input edge weights before adding self-loops.
            adjacency.data[:] = 1.0
        if self.config.add_self_loops:
            adjacency = adjacency + sparse.identity(
                adjacency.shape[0],
                format="csr",
                dtype=np.float64,
            )
        adjacency.sum_duplicates()
        if adjacency.nnz > 0:
            # Adding the identity can introduce duplicate diagonal entries.
            # Collapse them back to binary connectivity before normalization.
            adjacency.data[:] = 1.0

        degrees = np.asarray(adjacency.sum(axis=1)).reshape(-1).astype(np.float64, copy=False)
        inv_sqrt = np.zeros_like(degrees)
        nonzero_mask = degrees > 0.0
        inv_sqrt[nonzero_mask] = 1.0 / np.sqrt(degrees[nonzero_mask])
        d_inv_sqrt = sparse.diags(inv_sqrt, offsets=0, format="csr")
        normalized = (d_inv_sqrt @ adjacency @ d_inv_sqrt).tocsr()

        template = state.adjacency if isinstance(state.adjacency, torch.Tensor) else None
        state.extras[_NORMALIZED_ADJACENCY_KEY] = _sparse_matrix_to_torch(
            normalized,
            template=template,
            sparse_output=True,
        ).coalesce()
        return state


def _select_embedding_columns(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    dim: int,
    skip_first: bool = False,
) -> np.ndarray:
    """Select the first non-trivial eigenvectors.

    Parameters
    ----------
    eigenvalues : numpy.ndarray
        Eigenvalues with shape ``[K]``.
    eigenvectors : numpy.ndarray
        Eigenvectors with shape ``[N, K]``.
    dim : int
        Requested output dimension.
    skip_first : bool, default=False
        Whether to mirror NetworkX by sorting eigenvalues, dropping only the
        first column, and taking the next ``dim`` vectors. When ``False``, all
        near-zero eigenvalues are skipped for Dagua's existing robust behavior.

    Returns
    -------
    numpy.ndarray
        Selected coordinates with shape ``[N, dim]``.
    """
    sorted_indices = np.argsort(np.real(eigenvalues))
    if skip_first:
        nontrivial_indices = list(sorted_indices[1 : dim + 1])
    else:
        nontrivial_indices = [
            index
            for index in sorted_indices
            if abs(float(np.real(eigenvalues[index]))) > _SPECTRAL_EIGENVALUE_TOLERANCE
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
    networkx_fidelity: bool = False,
) -> np.ndarray:
    """Compute dense spectral coordinates from a Laplacian matrix.

    Parameters
    ----------
    laplacian : scipy.sparse.csr_matrix
        Laplacian matrix with shape ``[N, N]``.
    dim : int
        Requested output dimension.
    symmetric : bool
        Whether the matrix can use a symmetric eigensolver.
    networkx_fidelity : bool, default=False
        Whether to mirror NetworkX eigenvector selection.

    Returns
    -------
    numpy.ndarray
        Dense spectral coordinates with shape ``[N, dim]``.
    """
    dense_laplacian = laplacian.toarray()
    if symmetric:
        eigenvalues, eigenvectors = np.linalg.eigh(dense_laplacian)
    else:
        eigenvalues, eigenvectors = np.linalg.eig(dense_laplacian)
    return _select_embedding_columns(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        dim=dim,
        skip_first=networkx_fidelity,
    )


def _sparse_spectral_embedding(
    laplacian: sparse.csr_matrix,
    dim: int,
    symmetric: bool,
    networkx_fidelity: bool = False,
) -> np.ndarray:
    """Compute sparse spectral coordinates from a Laplacian matrix.

    Parameters
    ----------
    laplacian : scipy.sparse.csr_matrix
        Laplacian matrix with shape ``[N, N]``.
    dim : int
        Requested output dimension.
    symmetric : bool
        Whether the matrix can use the symmetric sparse eigensolver.
    networkx_fidelity : bool, default=False
        Whether to mirror NetworkX eigenvector selection.

    Returns
    -------
    numpy.ndarray
        Sparse spectral coordinates with shape ``[N, dim]``.
    """
    num_nodes = laplacian.shape[0]
    eigen_count = min(num_nodes - 1, max(dim + _SPECTRAL_EXTRA_EIGENPAIRS, dim + 1))
    if eigen_count <= dim:
        return _dense_spectral_embedding(
            laplacian=laplacian,
            dim=dim,
            symmetric=symmetric,
            networkx_fidelity=networkx_fidelity,
        )

    lanczos_vectors = max(
        (_SPECTRAL_LANCZOS_MULTIPLIER * eigen_count) + 1,
        int(np.sqrt(num_nodes)),
    )
    if symmetric:
        eigenvalues, eigenvectors = sparse_linalg.eigsh(
            laplacian,
            k=eigen_count,
            which="SM",
            ncv=min(max(lanczos_vectors, eigen_count + _SPECTRAL_LANCZOS_PADDING), num_nodes),
        )
    else:
        eigenvalues, eigenvectors = sparse_linalg.eigs(
            laplacian,
            k=eigen_count,
            which="SR",
            ncv=min(max(lanczos_vectors, eigen_count + _SPECTRAL_LANCZOS_PADDING), num_nodes),
        )
    return _select_embedding_columns(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        dim=dim,
        skip_first=networkx_fidelity,
    )


@register_op
class SpectralEmbed(Op):
    """Compute 2D spectral coordinates from the cached Laplacian when no positions exist."""

    name: ClassVar[str] = "spectral_embed"
    category: ClassVar[OpCategory] = OpCategory.EMBED
    reads: ClassVar[Tuple[str, ...]] = ("pos", "laplacian", f"extras.{_SYMMETRIC_FLAG_KEY}")
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("laplacian", f"extras.{_SYMMETRIC_FLAG_KEY}")
    access_pattern: ClassVar[str] = "global"

    def __init__(self, sparse_threshold: int, networkx_fidelity: bool = False) -> None:
        """Store the dense-vs-sparse eigensolve threshold.

        Parameters
        ----------
        sparse_threshold : int
            Dense matrices smaller than this threshold use NumPy eigensolvers.
        networkx_fidelity : bool, default=False
            Whether to mirror NetworkX eigenvector-selection behavior.
        """
        self.sparse_threshold = int(sparse_threshold)
        self.networkx_fidelity = bool(networkx_fidelity)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Recover raw 2D spectral coordinates.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs used to choose dense or sparse eigensolve.
        state : SolveState
            Mutable solve state containing the cached Laplacian and symmetry flag.
        ctx : RuntimeContext
            Execution infrastructure. Unused for this deterministic op.

        Returns
        -------
        SolveState
            State with ``state.pos`` populated unless an earlier op already set it.

        Raises
        ------
        RuntimeError
            If the Laplacian or symmetry metadata is missing.
        """
        _ = ctx
        if state.pos is not None:
            # Preserve any upstream coordinates so spectral embedding can act as
            # a fallback initializer inside larger pipelines.
            return state
        if state.laplacian is None:
            raise RuntimeError("SpectralEmbed requires state.laplacian to be set.")

        is_symmetric = state.extras.get(_SYMMETRIC_FLAG_KEY)
        if not isinstance(is_symmetric, bool):
            raise RuntimeError("SpectralEmbed requires a cached symmetry flag.")

        if problem.num_nodes < self.sparse_threshold:
            coordinates = _dense_spectral_embedding(
                laplacian=state.laplacian,
                dim=_EMBEDDING_OUTPUT_DIM,
                symmetric=is_symmetric,
                networkx_fidelity=self.networkx_fidelity,
            )
        else:
            coordinates = _sparse_spectral_embedding(
                laplacian=state.laplacian,
                dim=_EMBEDDING_OUTPUT_DIM,
                symmetric=is_symmetric,
                networkx_fidelity=self.networkx_fidelity,
            )

        state.pos = torch.from_numpy(coordinates)
        return state


@dataclass(frozen=True)
class EigendecompositionConfig:
    """Configuration for :class:`Eigendecomposition`.

    Parameters
    ----------
    sparse_threshold : int, default=500
        Dense matrices smaller than this threshold use NumPy eigensolvers.
    k : int, default=2
        Number of smallest eigenpairs to return.
    """

    sparse_threshold: int = 500
    k: int = 2


@register_op
class Eigendecomposition(Op):
    """Compute the smallest eigenpairs of a stored Laplacian."""

    name: ClassVar[str] = "eigendecomposition"
    category: ClassVar[OpCategory] = OpCategory.EMBED
    reads: ClassVar[Tuple[str, ...]] = (f"extras.{_LAPLACIAN_KEY}",)
    writes: ClassVar[Tuple[str, ...]] = (f"extras.{_EIGENPAIRS_KEY}",)
    requires: ClassVar[Tuple[str, ...]] = (f"extras.{_LAPLACIAN_KEY}",)
    access_pattern: ClassVar[str] = "global"

    def __init__(self, config: Optional[EigendecompositionConfig] = None) -> None:
        """Store the eigendecomposition configuration.

        Parameters
        ----------
        config : EigendecompositionConfig | None, optional
            Optional op configuration.
        """
        self.config = config or EigendecompositionConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute and store the requested eigenpairs.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused for this deterministic op.
        state : SolveState
            Mutable solve state containing ``extras["laplacian"]``.
        ctx : RuntimeContext
            Execution context. Unused for this deterministic op.

        Returns
        -------
        SolveState
            State with ``extras["eigenpairs"]`` populated.
        """
        _ = problem
        _ = ctx
        raw_laplacian = state.extras.get(_LAPLACIAN_KEY)
        if raw_laplacian is None:
            raise ValueError("Eigendecomposition requires extras['laplacian'].")

        laplacian = _payload_to_sparse_matrix(raw_laplacian)
        num_nodes = int(laplacian.shape[0])
        if self.config.k <= 0:
            raise ValueError("k must be positive.")
        if self.config.sparse_threshold < 0:
            raise ValueError("sparse_threshold must be non-negative.")

        if num_nodes == 0:
            eigenvalues = torch.empty((0,), dtype=torch.float64)
            eigenvectors = torch.empty((0, 0), dtype=torch.float64)
        else:
            k = min(self.config.k, num_nodes)
            symmetric = _is_symmetric(laplacian)
            use_dense = num_nodes < self.config.sparse_threshold or k >= num_nodes
            if use_dense:
                # Small problems and full-rank requests are cheaper and more
                # stable through dense LAPACK routines.
                dense = laplacian.toarray()
                if symmetric:
                    raw_values, raw_vectors = np.linalg.eigh(dense)
                else:
                    raw_values, raw_vectors = np.linalg.eig(dense)
                order = np.argsort(np.real(raw_values))[:k]
                eigenvalues = _numpy_array_to_torch(raw_values[order])
                eigenvectors = _numpy_array_to_torch(raw_vectors[:, order])
            else:
                if symmetric:
                    raw_values, raw_vectors = sparse_linalg.eigsh(laplacian, k=k, which="SM")
                else:
                    raw_values, raw_vectors = sparse_linalg.eigs(laplacian, k=k, which="SR")
                order = np.argsort(np.real(raw_values))
                eigenvalues = _numpy_array_to_torch(raw_values[order])
                eigenvectors = _numpy_array_to_torch(raw_vectors[:, order])

        state.extras[_EIGENPAIRS_KEY] = {
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
        }
        return state


@register_op
class SVD(Op):
    """Compute and cache the compact SVD of the pivot-distance matrix."""

    name: ClassVar[str] = "svd"
    category: ClassVar[OpCategory] = OpCategory.EMBED
    reads: ClassVar[Tuple[str, ...]] = ("pivot_distances",)
    writes: ClassVar[Tuple[str, ...]] = (f"extras.{_SVD_RESULT_KEY}",)
    requires: ClassVar[Tuple[str, ...]] = ("pivot_distances",)
    access_pattern: ClassVar[str] = "global"

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute and store the compact SVD.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused for this deterministic op.
        state : SolveState
            Mutable solve state with ``pivot_distances`` populated.
        ctx : RuntimeContext
            Execution context. Unused for this deterministic op.

        Returns
        -------
        SolveState
            State with ``extras["svd_result"]`` populated.
        """
        _ = problem
        _ = ctx
        if state.pivot_distances is None:
            raise ValueError("SVD requires state.pivot_distances to be set.")
        u, singular_values, vh = torch.linalg.svd(state.pivot_distances, full_matrices=False)
        state.extras[_SVD_RESULT_KEY] = {
            "u": u,
            "s": singular_values,
            "vh": vh,
        }
        return state


@register_op
class PivotMDSComputeCoordinates(Op):
    """Recover 2D Pivot-MDS coordinates from the stored pivot-distance rows."""

    name: ClassVar[str] = "pivot_mds_compute_coordinates"
    category: ClassVar[OpCategory] = OpCategory.EMBED
    reads: ClassVar[Tuple[str, ...]] = ("pivot_distances",)
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pivot_distances",)
    access_pattern: ClassVar[str] = "global"

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute coordinates from ``state.pivot_distances``.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused for this deterministic op.
        state : SolveState
            Mutable solve state containing ``pivot_distances``.
        ctx : RuntimeContext
            Execution context.

        Returns
        -------
        SolveState
            State with ``state.pos`` populated.
        """
        _ = problem
        _ = ctx
        if state.pivot_distances is None:
            raise ValueError("PivotMDSComputeCoordinates requires state.pivot_distances to be set.")

        state.pos = _pivot_mds_coordinates(state.pivot_distances)
        return state


@register_op
class Pseudoinverse(Op):
    """Compute the dense Moore-Penrose pseudoinverse of the cached Laplacian."""

    name: ClassVar[str] = "pseudoinverse"
    category: ClassVar[OpCategory] = OpCategory.EMBED
    reads: ClassVar[Tuple[str, ...]] = (f"extras.{_LAPLACIAN_KEY}",)
    writes: ClassVar[Tuple[str, ...]] = (f"extras.{_LAPLACIAN_PINV_KEY}",)
    requires: ClassVar[Tuple[str, ...]] = (f"extras.{_LAPLACIAN_KEY}",)
    access_pattern: ClassVar[str] = "global"

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute and store the Laplacian pseudoinverse.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused for this deterministic op.
        state : SolveState
            Mutable solve state containing ``extras["laplacian"]``.
        ctx : RuntimeContext
            Execution context. Unused for this deterministic op.

        Returns
        -------
        SolveState
            State with ``extras["laplacian_pinv"]`` populated.
        """
        _ = problem
        _ = ctx
        raw_laplacian = state.extras.get(_LAPLACIAN_KEY)
        if raw_laplacian is None:
            raise ValueError("Pseudoinverse requires extras['laplacian'].")
        laplacian = _payload_to_sparse_matrix(raw_laplacian).toarray()
        state.extras[_LAPLACIAN_PINV_KEY] = _numpy_array_to_torch(np.linalg.pinv(laplacian))
        return state


@dataclass(frozen=True)
class GCNForwardConfig:
    """Configuration for :class:`GCNForward`.

    Parameters
    ----------
    hidden_sizes : tuple[int, int], default=(100, 3)
        NeuLay-style hidden widths for the two GCN blocks.
    output_dim : int, default=2
        Output coordinate dimension.
    """

    hidden_sizes: Tuple[int, int] = (100, 3)
    output_dim: int = 2


@register_op
class GCNForward(Op):
    """Run one forward pass of the cached NeuLay-style residual GCN model.

    Notes
    -----
    When the model is created, parameters use a deterministic torch-uniform
    Xavier initializer driven by ``ctx.generator`` when provided, otherwise a
    private CPU generator seeded from ``problem.seed``.
    """

    name: ClassVar[str] = "gcn_forward"
    category: ClassVar[OpCategory] = OpCategory.EMBED
    reads: ClassVar[Tuple[str, ...]] = (
        f"extras.{_NORMALIZED_ADJACENCY_KEY}",
        f"extras.{_GCN_MODEL_KEY}",
        f"extras.{_GCN_ADJACENCY_REF_KEY}",
    )
    writes: ClassVar[Tuple[str, ...]] = (
        "pos",
        f"extras.{_GCN_MODEL_KEY}",
        f"extras.{_GCN_ADJACENCY_REF_KEY}",
    )
    requires: ClassVar[Tuple[str, ...]] = (f"extras.{_NORMALIZED_ADJACENCY_KEY}",)
    access_pattern: ClassVar[str] = "global"

    def __init__(self, config: Optional[GCNForwardConfig] = None) -> None:
        """Store the GCN forward configuration.

        Parameters
        ----------
        config : GCNForwardConfig | None, optional
            Optional op configuration.
        """
        self.config = config or GCNForwardConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Materialize ``state.pos`` from the residual GCN.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state containing ``extras["normalized_adjacency"]``.
        ctx : RuntimeContext
            Execution context supplying the optional RNG generator.

        Returns
        -------
        SolveState
            State with ``pos`` and the cached model populated.
        """
        if len(self.config.hidden_sizes) != _GCN_REQUIRED_HIDDEN_LAYER_COUNT:
            raise ValueError("hidden_sizes must contain exactly two layer widths.")
        if self.config.output_dim <= 0:
            raise ValueError("output_dim must be positive.")

        raw_adjacency = state.extras.get(_NORMALIZED_ADJACENCY_KEY)
        if raw_adjacency is None:
            raise ValueError("GCNForward requires extras['normalized_adjacency'].")

        normalized_adjacency = _normalized_adjacency_to_sparse_torch(raw_adjacency)
        device = normalized_adjacency.device
        cached_model = state.extras.get(_GCN_MODEL_KEY)
        cached_adjacency = state.extras.get(_GCN_ADJACENCY_REF_KEY)

        rebuild_model = (
            not isinstance(cached_model, _ResGCNModel)
            or cached_model.num_nodes != problem.num_nodes
            or cached_model.hidden_sizes != self.config.hidden_sizes
            or cached_model.output_dim != self.config.output_dim
            or cached_model.weight1.device != device
            or cached_adjacency is not raw_adjacency
        )
        if rebuild_model:
            # Cache invalidation intentionally uses object identity for the
            # adjacency payload so we never reuse weights across a rebuilt graph.
            generator = _resolve_generator(ctx, problem)
            model = _ResGCNModel(
                num_nodes=problem.num_nodes,
                hidden_sizes=self.config.hidden_sizes,
                output_dim=self.config.output_dim,
                adj_norm=normalized_adjacency.to(device=device),
                generator=generator,
            )
            state.extras[_GCN_MODEL_KEY] = model
            state.extras[_GCN_ADJACENCY_REF_KEY] = raw_adjacency
        else:
            model = cached_model

        state.pos = model()
        return state


@dataclass(frozen=True)
class PerplexityMatchConfig:
    """Configuration for :class:`PerplexityMatch`.

    Parameters
    ----------
    perplexity : float, default=30.0
        Target t-SNE perplexity.
    tol : float, default=1e-5
        Binary-search entropy tolerance.
    max_iter : int, default=100
        Maximum binary-search iterations per row.
    """

    perplexity: float = 30.0
    tol: float = 1.0e-5
    max_iter: int = 100


@register_op
class PerplexityMatch(Op):
    """Convert pairwise distances into symmetric t-SNE joint probabilities."""

    name: ClassVar[str] = "perplexity_match"
    category: ClassVar[OpCategory] = OpCategory.EMBED
    reads: ClassVar[Tuple[str, ...]] = ("distance_matrix",)
    writes: ClassVar[Tuple[str, ...]] = (f"extras.{_PROBABILITIES_KEY}",)
    requires: ClassVar[Tuple[str, ...]] = ("distance_matrix",)
    access_pattern: ClassVar[str] = "global"

    def __init__(self, config: Optional[PerplexityMatchConfig] = None) -> None:
        """Store the perplexity-matching configuration.

        Parameters
        ----------
        config : PerplexityMatchConfig | None, optional
            Optional op configuration.
        """
        self.config = config or PerplexityMatchConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute and store symmetric t-SNE input probabilities.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused for this deterministic op.
        state : SolveState
            Mutable solve state with ``distance_matrix`` populated.
        ctx : RuntimeContext
            Execution context. Unused for this deterministic op.

        Returns
        -------
        SolveState
            State with ``extras["probabilities"]`` populated.
        """
        _ = problem
        _ = ctx
        if state.distance_matrix is None:
            raise ValueError("PerplexityMatch requires state.distance_matrix to be set.")
        if self.config.perplexity <= 0.0:
            raise ValueError("perplexity must be positive.")
        if self.config.tol <= 0.0:
            raise ValueError("tol must be positive.")
        if self.config.max_iter <= 0:
            raise ValueError("max_iter must be positive.")

        distance_matrix = state.distance_matrix.detach().to(dtype=torch.float32)
        rows = [
            _perplexity_row(
                distance_matrix[node],
                perplexity=self.config.perplexity,
                tol=self.config.tol,
                max_iter=self.config.max_iter,
            )
            for node in range(distance_matrix.shape[0])
        ]
        conditional = torch.stack(rows, dim=0) if rows else torch.empty((0, 0), dtype=torch.float32)
        # Mirror the conditional rows into the joint distribution that t-SNE
        # consumes, keeping the legacy normalization factor intact.
        probabilities = (conditional + conditional.transpose(0, 1)) / (
            _TSNE_JOINT_PROBABILITY_DIVISOR * max(distance_matrix.shape[0], 1)
        )
        state.extras[_PROBABILITIES_KEY] = probabilities.clamp(min=_EPSILON)
        return state


@dataclass(frozen=True)
class SmoothKNNBandwidthConfig:
    """Configuration for :class:`SmoothKNNBandwidth`.

    Parameters
    ----------
    n_neighbors : int, default=15
        Number of nearest neighbors used for each local bandwidth estimate.
    tol : float, default=1e-5
        Binary-search tolerance.
    max_iter : int, default=64
        Maximum binary-search iterations.
    """

    n_neighbors: int = 15
    tol: float = 1.0e-5
    max_iter: int = 64


@register_op
class SmoothKNNBandwidth(Op):
    """Compute UMAP smooth-kNN local bandwidths from the dense distance matrix."""

    name: ClassVar[str] = "smooth_knn_bandwidth"
    category: ClassVar[OpCategory] = OpCategory.EMBED
    reads: ClassVar[Tuple[str, ...]] = ("distance_matrix",)
    writes: ClassVar[Tuple[str, ...]] = (
        f"extras.{_SIGMAS_KEY}",
        f"extras.{_RHOS_KEY}",
        f"extras.{_UMAP_N_NEIGHBORS_KEY}",
    )
    requires: ClassVar[Tuple[str, ...]] = ("distance_matrix",)
    access_pattern: ClassVar[str] = "global"

    def __init__(self, config: Optional[SmoothKNNBandwidthConfig] = None) -> None:
        """Store the smooth-kNN bandwidth configuration.

        Parameters
        ----------
        config : SmoothKNNBandwidthConfig | None, optional
            Optional op configuration.
        """
        self.config = config or SmoothKNNBandwidthConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute and store the local UMAP bandwidth terms.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused for this deterministic op.
        state : SolveState
            Mutable solve state with ``distance_matrix`` populated.
        ctx : RuntimeContext
            Execution context. Unused for this deterministic op.

        Returns
        -------
        SolveState
            State with ``extras["sigmas"]`` and ``extras["rhos"]`` populated.
        """
        _ = problem
        _ = ctx
        if state.distance_matrix is None:
            raise ValueError("SmoothKNNBandwidth requires state.distance_matrix to be set.")
        if self.config.n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive.")
        if self.config.tol <= 0.0:
            raise ValueError("tol must be positive.")
        if self.config.max_iter <= 0:
            raise ValueError("max_iter must be positive.")

        # Reuse the exact neighbor extraction that the fuzzy-graph builder will
        # later consume so the cached bandwidths stay aligned with it.
        _, knn_distances = _knn_from_distances(
            state.distance_matrix,
            n_neighbors=self.config.n_neighbors,
        )
        sigmas, rhos = _smooth_knn_bandwidth(
            knn_distances=knn_distances,
            n_neighbors=self.config.n_neighbors,
            tol=self.config.tol,
            max_iter=self.config.max_iter,
        )
        state.extras[_SIGMAS_KEY] = sigmas
        state.extras[_RHOS_KEY] = rhos
        state.extras[_UMAP_N_NEIGHBORS_KEY] = int(self.config.n_neighbors)
        return state


@dataclass(frozen=True)
class FuzzySimplicialSetConfig:
    """Configuration for :class:`FuzzySimplicialSet`.

    Parameters
    ----------
    default_n_neighbors : int, default=15
        Fallback neighborhood size used when the smooth-kNN op has not stored
        an explicit value in ``extras["umap_n_neighbors"]``.
    """

    default_n_neighbors: int = 15


@register_op
class FuzzySimplicialSet(Op):
    """Build and cache UMAP's symmetric fuzzy simplicial graph edge list."""

    name: ClassVar[str] = "fuzzy_simplicial_set"
    category: ClassVar[OpCategory] = OpCategory.EMBED
    reads: ClassVar[Tuple[str, ...]] = (
        "distance_matrix",
        f"extras.{_SIGMAS_KEY}",
        f"extras.{_RHOS_KEY}",
        f"extras.{_UMAP_N_NEIGHBORS_KEY}",
    )
    writes: ClassVar[Tuple[str, ...]] = (f"extras.{_FUZZY_GRAPH_KEY}",)
    requires: ClassVar[Tuple[str, ...]] = (
        "distance_matrix",
        f"extras.{_SIGMAS_KEY}",
        f"extras.{_RHOS_KEY}",
    )
    access_pattern: ClassVar[str] = "global"

    def __init__(
        self,
        config: Optional[FuzzySimplicialSetConfig] = None,
        *,
        default_n_neighbors: Optional[int] = None,
    ) -> None:
        """Store the fuzzy-graph configuration.

        Parameters
        ----------
        config : FuzzySimplicialSetConfig | None, optional
            Optional op configuration.
        default_n_neighbors : int | None, optional
            Backward-compatible override for ``config.default_n_neighbors``.
        """
        resolved_config = config or FuzzySimplicialSetConfig()
        if default_n_neighbors is not None:
            resolved_config = FuzzySimplicialSetConfig(default_n_neighbors=default_n_neighbors)
        self.config = resolved_config

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute and store the fuzzy simplicial graph.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with ``distance_matrix``, ``sigmas``, and
            ``rhos`` populated.
        ctx : RuntimeContext
            Execution context. Unused for this deterministic op.

        Returns
        -------
        SolveState
            State with ``extras["fuzzy_graph"]`` populated.
        """
        _ = ctx
        if state.distance_matrix is None:
            raise ValueError("FuzzySimplicialSet requires state.distance_matrix to be set.")

        sigmas = state.extras.get(_SIGMAS_KEY)
        rhos = state.extras.get(_RHOS_KEY)
        if not isinstance(sigmas, torch.Tensor) or not isinstance(rhos, torch.Tensor):
            raise ValueError("FuzzySimplicialSet requires extras['sigmas'] and extras['rhos'].")

        n_neighbors = int(
            state.extras.get(
                _UMAP_N_NEIGHBORS_KEY,
                # Match the same clamped neighbor count that SmoothKNNBandwidth
                # would have used if it had already run.
                min(max(problem.num_nodes - 1, 1), self.config.default_n_neighbors),
            )
        )
        knn_indices, knn_distances = _knn_from_distances(
            state.distance_matrix,
            n_neighbors=n_neighbors,
        )
        state.extras[_FUZZY_GRAPH_KEY] = _fuzzy_simplicial_graph(
            knn_indices=knn_indices,
            knn_distances=knn_distances,
            sigmas=sigmas,
            rhos=rhos,
        )
        return state


@dataclass(frozen=True)
class ClassicalMDSComputeEmbeddingConfig:
    """Configuration for :class:`ClassicalMDSComputeEmbedding`.

    Parameters
    ----------
    igraph_fidelity : bool, default=False
        If ``True``, use igraph-compatible raw MDS eigensolver semantics.

    Notes
    -----
    The classic-MDS embedding step is deterministic once graph distances are
    fixed. The fidelity flag is opt-in so default Dagua layouts keep legacy
    positive-eigenvalue filtering.
    """

    igraph_fidelity: bool = False


@register_op
@dataclass(frozen=True)
class ClassicalMDSComputeEmbedding(Op):
    """Compute raw 2D classical-MDS coordinates from ``distance_matrix``."""

    config: ClassicalMDSComputeEmbeddingConfig = field(
        default_factory=ClassicalMDSComputeEmbeddingConfig
    )

    name: ClassVar[str] = "classical_mds_compute_embedding"
    category: ClassVar[OpCategory] = OpCategory.EMBED
    reads: ClassVar[Tuple[str, ...]] = ("distance_matrix",)
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("distance_matrix",)
    access_pattern: ClassVar[str] = "global"

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute and store the raw coordinates for classical MDS.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state containing a dense distance matrix.
        ctx : RuntimeContext
            Execution context (unused).

        Returns
        -------
        SolveState
            State with ``state.pos`` populated.
        """
        del problem, ctx

        if state.distance_matrix is None:
            raise ValueError(
                "ClassicalMDSComputeEmbedding requires state.distance_matrix to be set."
            )

        if state.distance_matrix.ndim != 2:
            raise ValueError("ClassicalMDSComputeEmbedding requires a square distance matrix.")
        state.pos = _classical_mds_embedding(
            state.distance_matrix,
            igraph_fidelity=self.config.igraph_fidelity,
        )
        return state


@dataclass(frozen=True)
class CurveFitABConfig:
    """Configuration for :class:`CurveFit_ab`.

    Parameters
    ----------
    min_dist : float, default=0.1
        Minimum embedded distance with full attraction.
    spread : float, default=1.0
        Characteristic embedding spread.
    sample_multiple : float, default=3.0
        Maximum ``x`` value expressed as a multiple of ``spread`` when fitting.
    sample_count : int, default=300
        Number of evenly spaced samples used to fit the attraction curve.
    maxfev : int, default=10000
        Maximum function evaluations for ``scipy.optimize.curve_fit``.
    """

    min_dist: float = 0.1
    spread: float = 1.0
    sample_multiple: float = 3.0
    sample_count: int = 300
    maxfev: int = 10_000


@register_op
class CurveFit_ab(Op):
    """Fit UMAP's scalar attraction-curve parameters ``a`` and ``b`` with fallback defaults."""

    name: ClassVar[str] = "curve_fit_ab"
    category: ClassVar[OpCategory] = OpCategory.EMBED
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = (f"extras.{_UMAP_A_KEY}", f"extras.{_UMAP_B_KEY}")
    requires: ClassVar[Tuple[str, ...]] = ()
    access_pattern: ClassVar[str] = "global"

    def __init__(self, config: Optional[CurveFitABConfig] = None) -> None:
        """Store the UMAP curve-fit configuration.

        Parameters
        ----------
        config : CurveFitABConfig | None, optional
            Optional op configuration.
        """
        self.config = config or CurveFitABConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Fit and store the UMAP attraction-curve parameters.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused for this deterministic op.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution context. Unused for this deterministic op.

        Returns
        -------
        SolveState
            State with ``extras["umap_a"]`` and ``extras["umap_b"]`` populated.
        """
        _ = problem
        _ = state
        _ = ctx
        if self.config.min_dist < 0.0:
            raise ValueError("min_dist must be non-negative.")
        if self.config.spread <= 0.0:
            raise ValueError("spread must be positive.")
        if self.config.sample_multiple <= 0.0:
            raise ValueError("sample_multiple must be positive.")
        if self.config.sample_count <= 1:
            raise ValueError("sample_count must be greater than one.")

        xv = np.linspace(
            0.0,
            self.config.sample_multiple * self.config.spread,
            self.config.sample_count,
        )
        # Fit against the same piecewise target curve UMAP uses: a flat
        # attraction zone inside ``min_dist`` followed by exponential decay.
        yv = np.where(
            xv < self.config.min_dist,
            1.0,
            np.exp(-(xv - self.config.min_dist) / self.config.spread),
        )
        try:
            params, _ = optimize.curve_fit(
                _curve_function,
                xv,
                yv,
                p0=_CURVE_FIT_INITIAL_GUESS,
                maxfev=self.config.maxfev,
            )
            umap_a = float(params[0])
            umap_b = float(params[1])
        except (RuntimeError, ValueError):
            # Fall back to the canonical UMAP defaults if SciPy cannot fit a
            # stable curve for the requested sampling window.
            umap_a = _DEFAULT_CURVE_A
            umap_b = _DEFAULT_CURVE_B

        state.extras[_UMAP_A_KEY] = umap_a
        state.extras[_UMAP_B_KEY] = umap_b
        return state
