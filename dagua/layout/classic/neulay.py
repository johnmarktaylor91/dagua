"""NeuLay two-phase graph layout.

This module implements the published NeuLay objective with an optional
PyTorch-Geometric GCN reparameterization phase followed by direct force-driven
refinement.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

_PATIENCE = 10
_GCN_REL_TOL = 1.0e-4
_LINEAR_REL_TOL = 1.0e-8
_LATENT_DIM = 10
_GNN_LR = 0.01
_EPS = 1.0e-9
_PAIR_QUERY_RADIUS_FACTOR = 4.0
_PAIR_REFRESH_INTERVAL = 5


def _layout_device(
    edge_index: torch.Tensor,
    node_sizes: Optional[torch.Tensor],
) -> torch.device:
    """Choose the output device for the layout result.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor | None
        Optional node-size tensor.

    Returns
    -------
    torch.device
        Device used for optimization and the returned coordinates.
    """
    if edge_index.numel() > 0:
        return edge_index.device
    if node_sizes is not None:
        return node_sizes.device
    return torch.device("cpu")


def _validate_inputs(
    edge_index: torch.Tensor,
    num_nodes: int,
    steps: int,
    gcn_steps: int,
    dim: int,
    lr: float,
    radius: float,
    magnitude: Optional[float],
    edge_weights: Optional[torch.Tensor],
) -> None:
    """Validate the public NeuLay inputs.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes in the graph.
    steps : int
        Total optimization budget across the GCN and direct-refinement phases.
    gcn_steps : int
        Number of GCN reparameterization steps.
    dim : int
        Embedding dimensionality.
    lr : float
        RMSprop learning rate for the direct phase.
    radius : float
        Gaussian repulsion radius.
    magnitude : float | None
        Gaussian repulsion magnitude.  When ``None`` the adaptive formula
        ``100 * N^(1/3) * radius`` is used (validated after resolution).
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.

    Returns
    -------
    None
        Raises ``ValueError`` when the configuration is invalid.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if gcn_steps < 0:
        raise ValueError("gcn_steps must be non-negative.")
    if dim <= 0:
        raise ValueError("dim must be positive.")
    if lr <= 0.0:
        raise ValueError("lr must be positive.")
    if radius <= 0.0:
        raise ValueError("radius must be positive.")
    if magnitude is not None and magnitude < 0.0:
        raise ValueError("magnitude must be non-negative.")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")
    if edge_weights is not None and edge_weights.shape[0] != edge_index.shape[1]:
        raise ValueError(
            f"edge_weights length {edge_weights.shape[0]} does not match "
            f"edge count {edge_index.shape[1]}"
        )
    if edge_index.numel() == 0:
        return

    if edge_index.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError("edge_index must use an integer dtype.")

    min_index = int(edge_index.min().item())
    max_index = int(edge_index.max().item())
    if min_index < 0 or max_index >= num_nodes:
        raise ValueError("edge_index contains node indices outside [0, num_nodes).")


def _set_seed(seed: int) -> None:
    """Seed the PyTorch RNGs used by NeuLay.

    Parameters
    ----------
    seed : int
        Requested random seed.

    Returns
    -------
    None
        The global RNG state is updated in-place.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clean_edge_index(edge_index: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Remove self-loops and move the edge list to the optimization device.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    device : torch.device
        Device used by the optimization loop.

    Returns
    -------
    torch.Tensor
        Cleaned edge tensor with shape ``[2, E_clean]``.
    """
    cleaned = edge_index.to(device=device, dtype=torch.long)
    if cleaned.numel() == 0:
        return cleaned.reshape(2, 0)
    non_self = cleaned[0] != cleaned[1]
    return cleaned[:, non_self].contiguous()


def _initial_positions(num_nodes: int, dim: int, device: torch.device) -> torch.Tensor:
    """Create the NeuLay random initialization.

    Matches the reference ``xavier_uniform_`` with gain ``N^(1/dim)``.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    dim : int
        Embedding dimensionality.
    device : torch.device
        Device used for the returned tensor.

    Returns
    -------
    torch.Tensor
        Initial coordinates with shape ``[N, dim]``.
    """
    gain = float(max(num_nodes, 1)) ** (1.0 / float(max(dim, 1)))
    initial = torch.empty((num_nodes, dim), device=device, dtype=torch.float32)
    nn.init.xavier_uniform_(initial, gain=gain)
    return initial


def _center_positions(pos: torch.Tensor) -> torch.Tensor:
    """Remove translational drift from a coordinate tensor.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, dim]``.

    Returns
    -------
    torch.Tensor
        Centered coordinates with the same shape as ``pos``.
    """
    if pos.numel() == 0:
        return pos
    return pos - pos.mean(dim=0, keepdim=True)


def _elastic_loss(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Evaluate the NeuLay elastic energy.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, dim]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.

    Returns
    -------
    torch.Tensor
        Scalar elastic loss.
    """
    if edge_index.numel() == 0:
        return pos.sum() * 0.0
    # Match the reference by collapsing directed duplicates into one
    # undirected spring before measuring the elastic energy.
    src, dst = edge_index[0], edge_index[1]
    low = torch.minimum(src, dst)
    high = torch.maximum(src, dst)
    pairs = torch.stack([low, high], dim=0)
    unique_pairs = torch.unique(pairs, dim=1)
    diff = pos[unique_pairs[0]] - pos[unique_pairs[1]]
    return diff.square().sum() * 0.5


def _kdtree_repulsion_pairs(pos: torch.Tensor, query_radius: float) -> np.ndarray:
    """Find nearby node pairs using SciPy's cKDTree.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, dim]``.
    query_radius : float
        cKDTree search radius.

    Returns
    -------
    numpy.ndarray
        Pair array with shape ``[M, 2]`` and dtype ``int64``.
    """
    from scipy.spatial import cKDTree

    if pos.shape[0] < 2:
        return np.empty((0, 2), dtype=np.int64)
    tree = cKDTree(pos.detach().cpu().numpy())
    pairs = tree.query_pairs(query_radius, output_type="ndarray")
    if pairs.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    return pairs.astype(np.int64)


def _kdtree_repulsion_loss(
    pos: torch.Tensor,
    pairs: np.ndarray,
    radius: float,
    magnitude: float,
) -> torch.Tensor:
    """Evaluate Gaussian repulsion over cached cKDTree pairs.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, dim]``.
    pairs : numpy.ndarray
        Pair array with shape ``[M, 2]``.
    radius : float
        Gaussian radius.
    magnitude : float
        Repulsion magnitude.

    Returns
    -------
    torch.Tensor
        Scalar repulsion loss.
    """
    if pairs.shape[0] == 0 or magnitude == 0.0:
        return pos.sum() * 0.0
    idx = torch.from_numpy(pairs).to(device=pos.device)
    sq_dist = ((pos[idx[:, 0]] - pos[idx[:, 1]]) ** 2).sum(dim=-1)
    return magnitude * torch.exp(-sq_dist / (4.0 * radius * radius)).sum()


def _relative_window_difference(loss_window: list[float]) -> float:
    """Measure the reference NeuLay relative-loss window difference.

    Parameters
    ----------
    loss_window : list[float]
        Sliding loss window of size ``_PATIENCE``.

    Returns
    -------
    float
        Relative range ``(max - min) / max``.
    """
    max_loss = max(loss_window)
    if max_loss <= 0.0:
        return 0.0
    return (max_loss - min(loss_window)) / max_loss


def _optimize_positions(
    initial_pos: torch.Tensor,
    edge_index: torch.Tensor,
    steps: int,
    lr: float,
    radius: float,
    magnitude: float,
) -> torch.Tensor:
    """Run the direct NeuLay refinement phase.

    Parameters
    ----------
    initial_pos : torch.Tensor
        Initial coordinates with shape ``[N, dim]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    steps : int
        Number of RMSprop steps for the direct refinement phase.
    lr : float
        RMSprop learning rate.
    radius : float
        Gaussian repulsion radius.
    magnitude : float
        Gaussian repulsion magnitude.

    Returns
    -------
    torch.Tensor
        Refined coordinates with shape ``[N, dim]``.
    """
    pos = nn.Parameter(initial_pos.clone())
    optimizer = torch.optim.RMSprop([pos], lr=lr)
    loss_window = [0.0] * _PATIENCE
    num_nodes = initial_pos.shape[0]
    query_radius = _PAIR_QUERY_RADIUS_FACTOR * radius
    pairs = np.empty((0, 2), dtype=np.int64)

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        if step % _PAIR_REFRESH_INTERVAL == 0:
            pairs = _kdtree_repulsion_pairs(pos, query_radius)
        loss = _elastic_loss(pos, edge_index) + _kdtree_repulsion_loss(
            pos,
            pairs=pairs,
            radius=radius,
            magnitude=magnitude,
        )
        loss.backward()
        optimizer.step()
        loss_window.append(float(loss.detach().item()))
        loss_window.pop(0)
        if _relative_window_difference(loss_window) < _LINEAR_REL_TOL * math.sqrt(float(num_nodes)):
            break

    return pos.detach()


def _build_normalized_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Build ``D^(-1/2) (A+I) D^(-1/2)`` as a sparse torch tensor.

    Matches the reference NeuLay normalization: make the graph undirected,
    add self-loops, then apply symmetric degree normalization.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]`` (self-loops already removed).
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    torch.Tensor
        Sparse normalized adjacency with shape ``[N, N]``.
    """
    if edge_index.numel() == 0:
        idx = torch.arange(num_nodes, dtype=torch.long)
        return torch.sparse_coo_tensor(
            torch.stack([idx, idx]),
            torch.ones(num_nodes, dtype=torch.float32),
            (num_nodes, num_nodes),
        ).coalesce()

    src, dst = edge_index[0].cpu(), edge_index[1].cpu()
    pairs = np.stack(
        [
            np.concatenate([src.numpy(), dst.numpy()]),
            np.concatenate([dst.numpy(), src.numpy()]),
        ],
        axis=0,
    ).astype(np.int64)
    vals = np.ones(pairs.shape[1], dtype=np.float32)
    adj = sp.coo_matrix((vals, (pairs[0], pairs[1])), shape=(num_nodes, num_nodes))
    adj = adj + sp.eye(num_nodes, dtype=np.float32, format="coo")
    adj.sum_duplicates()
    if adj.nnz > 0:
        adj.data[:] = 1.0

    degree = np.asarray(adj.sum(axis=0), dtype=np.float32).ravel()
    inv_sqrt = np.zeros_like(degree)
    mask = degree > 0
    inv_sqrt[mask] = 1.0 / np.sqrt(degree[mask])
    d_mat = sp.diags(inv_sqrt).tocsr()
    normalized = d_mat.dot(adj.dot(d_mat)).tocoo()

    indices = torch.from_numpy(np.vstack((normalized.row, normalized.col)).astype(np.int64))
    values = torch.from_numpy(normalized.data.astype(np.float32))
    return torch.sparse_coo_tensor(
        indices, values, (num_nodes, num_nodes), dtype=torch.float32
    ).coalesce()


class _SparseGCN(nn.Module):
    """Reference-matching GCN layer: ``A_norm @ (X @ W)``, no bias."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        adj_norm: torch.Tensor,
        gain: float,
    ) -> None:
        super().__init__()
        self.adj_norm = adj_norm
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``A_norm @ (X @ W)``."""
        support = torch.mm(x, self.weight)
        return torch.sparse.mm(self.adj_norm, support)


class _ResGCN(nn.Module):
    """Reference-matching NeuLay 3-layer GCN with skip concatenation.

    Architecture exactly matches the upstream NeuLay-2.py script:
    - Direct learnable weight1 (N x 100), Xavier-initialized
    - GCN layer 1: A_norm @ (h0 @ W_gcn1), 100 -> 100, Tanh, no bias
    - GCN layer 2: A_norm @ (h1 @ W_gcn2), 100 -> 3, no bias
    - Skip concatenation: [h0, h1, h2] -> N x 203
    - Direct weight2 multiply: 203 -> output dim, no bias
    """

    _HIDDEN = 100
    _GCN2_OUT = 3

    def __init__(
        self,
        num_nodes: int,
        dim: int,
        device: torch.device,
        edge_index: torch.Tensor,
    ) -> None:
        super().__init__()
        gain = float(max(num_nodes, 1)) ** (1.0 / float(max(dim, 1)))
        adj_norm = _build_normalized_adjacency(edge_index, num_nodes).to(device)

        self.weight1 = nn.Parameter(
            torch.empty((num_nodes, self._HIDDEN), device=device, dtype=torch.float32)
        )
        nn.init.xavier_uniform_(self.weight1, gain=gain)

        self.gcn1 = _SparseGCN(self._HIDDEN, self._HIDDEN, adj_norm, gain)
        self.gcn2 = _SparseGCN(self._HIDDEN, self._GCN2_OUT, adj_norm, gain)

        concat_dim = self._HIDDEN + self._HIDDEN + self._GCN2_OUT
        self.weight2 = nn.Parameter(
            torch.empty((concat_dim, dim), device=device, dtype=torch.float32)
        )
        nn.init.xavier_uniform_(self.weight2, gain=gain)

    def forward(self) -> torch.Tensor:
        """Generate layout coordinates."""
        h0 = self.weight1
        h1 = torch.tanh(self.gcn1(h0))
        h2 = self.gcn2(h1)
        return torch.mm(torch.cat([h0, h1, h2], dim=1), self.weight2)


def _optimize_gcn_phase(
    edge_index: torch.Tensor,
    num_nodes: int,
    dim: int,
    device: torch.device,
    steps: int,
    radius: float,
    magnitude: float,
) -> torch.Tensor:
    """Run the optional NeuLay GCN reparameterization phase.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    dim : int
        Embedding dimensionality.
    device : torch.device
        Optimization device.
    steps : int
        Number of GCN RMSprop steps.
    radius : float
        Gaussian repulsion radius.
    magnitude : float
        Gaussian repulsion magnitude.

    Returns
    -------
    torch.Tensor
        Coarse coordinates with shape ``[N, dim]``.
    """
    model = _ResGCN(
        num_nodes=num_nodes,
        dim=dim,
        device=device,
        edge_index=edge_index,
    )
    optimizer = torch.optim.RMSprop(model.parameters(), lr=_GNN_LR)
    loss_window = [0.0] * _PATIENCE
    query_radius = _PAIR_QUERY_RADIUS_FACTOR * radius
    pairs = np.empty((0, 2), dtype=np.int64)

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output = model()
        if step % _PAIR_REFRESH_INTERVAL == 0:
            pairs = _kdtree_repulsion_pairs(output, query_radius)
        loss = _elastic_loss(output, edge_index) + _kdtree_repulsion_loss(
            output,
            pairs=pairs,
            radius=radius,
            magnitude=magnitude,
        )
        loss.backward()
        optimizer.step()
        loss_window.append(float(loss.detach().item()))
        loss_window.pop(0)
        if _relative_window_difference(loss_window) < _GCN_REL_TOL * math.sqrt(float(num_nodes)):
            break

    with torch.no_grad():
        return model().detach()


def layout_neulay(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: int = 42,
    steps: int = 20_000,
    gcn_steps: int = 2_000,
    use_gcn: bool = True,
    dim: int = 2,
    lr: float = 0.01,
    radius: float = 0.4,
    magnitude: Optional[float] = None,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Lay out a graph with the published NeuLay objective.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor | None, default=None
        Unused placeholder kept for API compatibility with other layouts.
    seed : int, default=42
        Random seed used for the initialization and optimizer trajectory.
    steps : int, default=20000
        Total optimization budget across the GCN and direct phases.
    gcn_steps : int, default=2000
        Number of GCN reparameterization steps when PyG is available.
    use_gcn : bool, default=True
        Whether to run the optional GCN phase when PyG is installed.
    dim : int, default=2
        Output dimensionality.
    lr : float, default=0.01
        RMSprop learning rate for the direct phase.
    radius : float, default=0.4
        Gaussian repulsion radius.
    magnitude : float | None, default=None
        Gaussian repulsion magnitude.  When ``None`` (the default), an
        adaptive value of ``100 * N^(1/3) * radius`` is used, matching
        the reference implementation.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``. Accepted for interface
        consistency; the current NeuLay port does not yet thread them into the
        GCN message-passing or direct force phases.

    Returns
    -------
    torch.Tensor
        Coordinate tensor with shape ``[N, dim]``.
    """
    _validate_inputs(
        edge_index=edge_index,
        num_nodes=num_nodes,
        steps=steps,
        gcn_steps=gcn_steps,
        dim=dim,
        lr=lr,
        radius=radius,
        magnitude=magnitude,
        edge_weights=edge_weights,
    )
    device = _layout_device(edge_index=edge_index, node_sizes=node_sizes)
    if num_nodes == 0:
        return torch.empty((0, dim), dtype=torch.float32, device=device)
    if num_nodes == 1:
        return torch.zeros((1, dim), dtype=torch.float32, device=device)

    # Adaptive repulsion: 100 * N^(1/3) * radius (reference formula).
    if magnitude is None:
        magnitude = 100.0 * (num_nodes ** (1.0 / 3.0)) * radius

    _set_seed(seed)
    cleaned_edge_index = _clean_edge_index(edge_index=edge_index, device=device)

    if use_gcn and gcn_steps > 0:
        coarse = _optimize_gcn_phase(
            edge_index=cleaned_edge_index,
            num_nodes=num_nodes,
            dim=dim,
            device=device,
            steps=gcn_steps,
            radius=radius,
            magnitude=magnitude,
        )
    else:
        coarse = _initial_positions(num_nodes=num_nodes, dim=dim, device=device)

    linear_steps = max(steps - gcn_steps, 0) if use_gcn else steps
    if linear_steps <= 0:
        return coarse

    return _optimize_positions(
        initial_pos=coarse,
        edge_index=cleaned_edge_index,
        steps=linear_steps,
        lr=lr,
        radius=radius,
        magnitude=magnitude,
    )
