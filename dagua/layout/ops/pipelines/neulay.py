"""NeuLay two-phase graph layout expressed as a composable ops pipeline."""

from __future__ import annotations

import math
from typing import ClassVar, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from dagua.layout.ops.base import Op, Pipeline, Repeat  # noqa: E402
from dagua.layout.ops.converge import FixedSteps, FixedStepsConfig  # noqa: E402
from dagua.layout.ops.state import (  # noqa: E402
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)
from dagua.layout.ops.taxonomy import OpCategory  # noqa: E402

# ---------------------------------------------------------------------------
# Constants and functions copied from dagua/layout/classic/neulay.py
# (bit-identical to the classic originals)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Pipeline-local ops
# ---------------------------------------------------------------------------


class _SeedRNG(Op):
    """Seed PyTorch and NumPy RNGs exactly like classic NeuLay."""

    name: ClassVar[str] = "neulay_seed_rng"
    category: ClassVar[OpCategory] = OpCategory.INIT
    writes: ClassVar[Tuple[str, ...]] = ()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Set the global RNG state to match classic NeuLay seeding.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs containing ``seed``.
        state : SolveState
            Mutable solve state. Unchanged by this op.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            Unchanged state (RNG is global side-effect).
        """
        del ctx
        _set_seed(problem.seed)
        return state


class _PrepareNeuLayState(Op):
    """Clean edges and resolve NeuLay hyperparameters into extras."""

    name: ClassVar[str] = "neulay_prepare_state"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def __init__(
        self,
        *,
        dim: int = 2,
        lr: float = 0.01,
        radius: float = 0.4,
        magnitude: Optional[float] = None,
        use_gcn: bool = True,
        gcn_steps: int = 2_000,
        total_steps: int = 20_000,
    ) -> None:
        """Store NeuLay configuration.

        Parameters
        ----------
        dim : int
            Embedding dimensionality.
        lr : float
            RMSprop learning rate for the direct phase.
        radius : float
            Gaussian repulsion radius.
        magnitude : float or None
            Gaussian repulsion magnitude. ``None`` triggers the adaptive
            formula ``100 * N^(1/3) * radius``.
        use_gcn : bool
            Whether to run the GCN reparameterization phase.
        gcn_steps : int
            Number of GCN optimization steps.
        total_steps : int
            Total optimization budget across both phases.
        """
        self._dim = dim
        self._lr = lr
        self._radius = radius
        self._magnitude = magnitude
        self._use_gcn = use_gcn
        self._gcn_steps = gcn_steps
        self._total_steps = total_steps

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Clean edges and store resolved hyperparameters in extras.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state receiving NeuLay extras.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with NeuLay configuration in ``extras``.
        """
        del ctx

        device = _layout_device(problem.edge_index, problem.node_sizes)
        cleaned = _clean_edge_index(edge_index=problem.edge_index, device=device)

        magnitude = self._magnitude
        if magnitude is None:
            magnitude = 100.0 * (problem.num_nodes ** (1.0 / 3.0)) * self._radius

        linear_steps = (
            max(self._total_steps - self._gcn_steps, 0) if self._use_gcn else self._total_steps
        )

        state.extras["neulay_cleaned_edge_index"] = cleaned
        state.extras["neulay_device"] = device
        state.extras["neulay_dim"] = self._dim
        state.extras["neulay_lr"] = self._lr
        state.extras["neulay_radius"] = self._radius
        state.extras["neulay_magnitude"] = magnitude
        state.extras["neulay_use_gcn"] = self._use_gcn
        state.extras["neulay_gcn_steps"] = self._gcn_steps
        state.extras["neulay_linear_steps"] = linear_steps
        state.extras["neulay_query_radius"] = _PAIR_QUERY_RADIUS_FACTOR * self._radius
        return state


class _GCNPhase(Op):
    """Run the optional NeuLay GCN reparameterization phase.

    This op encapsulates the entire GCN training loop to match the classic
    implementation exactly. It creates the ``_ResGCN`` model, trains it with
    RMSprop, and writes the resulting coordinates to ``state.pos``.
    """

    name: ClassVar[str] = "neulay_gcn_phase"
    category: ClassVar[OpCategory] = OpCategory.INIT
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Train the GCN and produce coarse coordinates.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with ``pos`` set to the GCN output coordinates.
        """
        del ctx

        cleaned = state.extras["neulay_cleaned_edge_index"]
        device = state.extras["neulay_device"]
        dim = state.extras["neulay_dim"]
        radius = state.extras["neulay_radius"]
        magnitude = state.extras["neulay_magnitude"]
        gcn_steps = state.extras["neulay_gcn_steps"]
        use_gcn = state.extras["neulay_use_gcn"]
        query_radius = state.extras["neulay_query_radius"]

        if use_gcn and gcn_steps > 0:
            model = _ResGCN(
                num_nodes=problem.num_nodes,
                dim=dim,
                device=device,
                edge_index=cleaned,
            )
            optimizer = torch.optim.RMSprop(model.parameters(), lr=_GNN_LR)
            loss_window = [0.0] * _PATIENCE
            pairs = np.empty((0, 2), dtype=np.int64)

            for step in range(gcn_steps):
                optimizer.zero_grad(set_to_none=True)
                output = model()
                if step % _PAIR_REFRESH_INTERVAL == 0:
                    pairs = _kdtree_repulsion_pairs(output, query_radius)
                loss = _elastic_loss(output, cleaned) + _kdtree_repulsion_loss(
                    output,
                    pairs=pairs,
                    radius=radius,
                    magnitude=magnitude,
                )
                loss.backward()
                optimizer.step()
                loss_window.append(float(loss.detach().item()))
                loss_window.pop(0)
                if _relative_window_difference(loss_window) < (
                    _GCN_REL_TOL * math.sqrt(float(problem.num_nodes))
                ):
                    break

            with torch.no_grad():
                state.pos = model().detach()
        else:
            state.pos = _initial_positions(
                num_nodes=problem.num_nodes,
                dim=dim,
                device=device,
            )

        return state


class _InitDirectPhaseOptimizer(Op):
    """Initialize the RMSprop optimizer and loss window for the direct phase."""

    name: ClassVar[str] = "neulay_init_direct_optimizer"
    category: ClassVar[OpCategory] = OpCategory.INIT
    reads: ClassVar[Tuple[str, ...]] = ("pos", "extras")
    writes: ClassVar[Tuple[str, ...]] = ("optimizer", "extras")
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Create the RMSprop optimizer over a fresh Parameter clone.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state containing ``pos`` from the GCN phase.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with ``optimizer`` and direct-phase bookkeeping in extras.

        Raises
        ------
        ValueError
            If ``state.pos`` is missing.
        """
        del problem, ctx

        if state.pos is None:
            raise ValueError("_InitDirectPhaseOptimizer requires state.pos.")

        lr = state.extras["neulay_lr"]
        pos_param = nn.Parameter(state.pos.clone())
        state.pos = pos_param
        state.optimizer = torch.optim.RMSprop([pos_param], lr=lr)
        state.extras["neulay_loss_window"] = [0.0] * _PATIENCE
        state.extras["neulay_pairs"] = np.empty((0, 2), dtype=np.int64)
        return state


class _DirectPhaseStep(Op):
    """Execute one step of the NeuLay direct refinement phase."""

    name: ClassVar[str] = "neulay_direct_step"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[Tuple[str, ...]] = ("pos", "optimizer", "extras")
    writes: ClassVar[Tuple[str, ...]] = ("pos", "extras")
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute loss, backprop, and apply one RMSprop step.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state with positions and optimizer.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with updated positions and loss window.

        Raises
        ------
        ValueError
            If ``state.pos`` or ``state.optimizer`` is missing.
        """
        del problem, ctx

        if state.pos is None:
            raise ValueError("_DirectPhaseStep requires state.pos.")
        if state.optimizer is None:
            raise ValueError("_DirectPhaseStep requires state.optimizer.")

        cleaned = state.extras["neulay_cleaned_edge_index"]
        radius = state.extras["neulay_radius"]
        magnitude = state.extras["neulay_magnitude"]
        query_radius = state.extras["neulay_query_radius"]
        loss_window = state.extras["neulay_loss_window"]
        pairs = state.extras["neulay_pairs"]

        state.optimizer.zero_grad(set_to_none=True)
        if state.step % _PAIR_REFRESH_INTERVAL == 0:
            pairs = _kdtree_repulsion_pairs(state.pos, query_radius)
            state.extras["neulay_pairs"] = pairs

        loss = _elastic_loss(state.pos, cleaned) + _kdtree_repulsion_loss(
            state.pos,
            pairs=pairs,
            radius=radius,
            magnitude=magnitude,
        )
        loss.backward()
        state.optimizer.step()

        loss_window.append(float(loss.detach().item()))
        loss_window.pop(0)
        state.extras["neulay_loss_window"] = loss_window
        return state


class _DirectPhaseConvergenceCheck(Op):
    """Check the NeuLay sliding-window convergence criterion."""

    name: ClassVar[str] = "neulay_direct_convergence"
    category: ClassVar[OpCategory] = OpCategory.CONVERGE
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Set ``state.converged`` when the loss window stabilizes.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with loss window in extras.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with ``converged`` potentially set to ``True``.
        """
        del ctx

        loss_window = state.extras["neulay_loss_window"]
        threshold = _LINEAR_REL_TOL * math.sqrt(float(problem.num_nodes))
        if _relative_window_difference(loss_window) < threshold:
            state.converged = True
        return state


class _FinalizeNeuLayPositions(Op):
    """Detach the final positions from the autograd graph."""

    name: ClassVar[str] = "neulay_finalize_positions"
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
        """Detach positions to match classic ``_optimize_positions`` return.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state containing final positions.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with detached positions.

        Raises
        ------
        ValueError
            If ``state.pos`` is missing.
        """
        del problem, ctx

        if state.pos is None:
            raise ValueError("_FinalizeNeuLayPositions requires state.pos.")
        state.pos = state.pos.detach()
        return state


# ---------------------------------------------------------------------------
# Public pipeline constructors
# ---------------------------------------------------------------------------


def build_neulay_pipeline(
    steps: int = 20_000,
    gcn_steps: int = 2_000,
    use_gcn: bool = True,
    dim: int = 2,
    lr: float = 0.01,
    radius: float = 0.4,
    magnitude: Optional[float] = None,
) -> Pipeline:
    """Build a NeuLay pipeline that is bit-identical to classic ``layout_neulay``.

    Parameters
    ----------
    steps : int, default=20000
        Total optimization budget across both phases.
    gcn_steps : int, default=2000
        Number of GCN reparameterization steps.
    use_gcn : bool, default=True
        Whether to run the optional GCN phase.
    dim : int, default=2
        Output embedding dimensionality.
    lr : float, default=0.01
        RMSprop learning rate for the direct phase.
    radius : float, default=0.4
        Gaussian repulsion radius.
    magnitude : float or None, default=None
        Gaussian repulsion magnitude. ``None`` triggers the adaptive formula.

    Returns
    -------
    Pipeline
        Pipeline that reproduces classic NeuLay's GCN phase, direct
        RMSprop refinement, KD-tree repulsion, and convergence check.

    Raises
    ------
    ValueError
        If ``steps`` or ``gcn_steps`` is negative.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if gcn_steps < 0:
        raise ValueError("gcn_steps must be non-negative.")

    linear_steps = max(steps - gcn_steps, 0) if use_gcn else steps

    return Pipeline(
        [
            FixedSteps(FixedStepsConfig(n=steps)),
            _SeedRNG(),
            _PrepareNeuLayState(
                dim=dim,
                lr=lr,
                radius=radius,
                magnitude=magnitude,
                use_gcn=use_gcn,
                gcn_steps=gcn_steps,
                total_steps=steps,
            ),
            _GCNPhase(),
            _InitDirectPhaseOptimizer(),
            Repeat(
                n=linear_steps,
                ops=[
                    _DirectPhaseStep(),
                    _DirectPhaseConvergenceCheck(),
                ],
            ),
            _FinalizeNeuLayPositions(),
        ],
        name="neulay_pipeline",
    )


def layout_neulay_pipeline(
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
    """Run the NeuLay pipeline as a drop-in replacement for classic ``layout_neulay``.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor | None, default=None
        Optional node-size tensor (unused, kept for API compatibility).
    seed : int, default=42
        Random seed for initialization and optimizer trajectory.
    steps : int, default=20000
        Total optimization budget across both phases.
    gcn_steps : int, default=2000
        Number of GCN reparameterization steps.
    use_gcn : bool, default=True
        Whether to run the optional GCN phase.
    dim : int, default=2
        Output embedding dimensionality.
    lr : float, default=0.01
        RMSprop learning rate for the direct phase.
    radius : float, default=0.4
        Gaussian repulsion radius.
    magnitude : float or None, default=None
        Gaussian repulsion magnitude. ``None`` triggers the adaptive formula.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``. Accepted for interface
        consistency.

    Returns
    -------
    torch.Tensor
        Coordinate tensor with shape ``[N, dim]``, bit-identical to classic
        ``layout_neulay``.

    Raises
    ------
    ValueError
        If inputs are invalid.
    RuntimeError
        If the pipeline fails to produce positions.
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

    # Handle trivial cases exactly like classic.
    if num_nodes == 0:
        return torch.empty((0, dim), dtype=torch.float32, device=device)
    if num_nodes == 1:
        return torch.zeros((1, dim), dtype=torch.float32, device=device)

    # When linear_steps is 0, classic returns the GCN output directly.
    linear_steps = max(steps - gcn_steps, 0) if use_gcn else steps
    if use_gcn and linear_steps <= 0:
        # GCN-only path: seed, clean, run GCN, return.
        _set_seed(seed)
        cleaned = _clean_edge_index(edge_index=edge_index, device=device)
        if magnitude is None:
            magnitude = 100.0 * (num_nodes ** (1.0 / 3.0)) * radius
        return _optimize_gcn_phase(
            edge_index=cleaned,
            num_nodes=num_nodes,
            dim=dim,
            device=device,
            steps=gcn_steps,
            radius=radius,
            magnitude=magnitude,
        )

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_neulay_pipeline(
        steps=steps,
        gcn_steps=gcn_steps,
        use_gcn=use_gcn,
        dim=dim,
        lr=lr,
        radius=radius,
        magnitude=magnitude,
    ).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("NeuLay pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_neulay_pipeline", "layout_neulay_pipeline"]
