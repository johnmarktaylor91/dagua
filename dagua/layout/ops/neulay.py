"""Registered NeuLay ops for a composable two-phase layout pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from scipy.spatial import cKDTree
from torch import nn

from dagua.layout.ops.base import Op
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_NEULAY_CLEANED_EDGE_INDEX_KEY = "neulay_cleaned_edge_index"
_NEULAY_DEVICE_KEY = "neulay_device"
_NEULAY_DIM_KEY = "neulay_dim"
_NEULAY_LR_KEY = "neulay_lr"
_NEULAY_RADIUS_KEY = "neulay_radius"
_NEULAY_MAGNITUDE_KEY = "neulay_magnitude"
_NEULAY_USE_GCN_KEY = "neulay_use_gcn"
_NEULAY_GCN_STEPS_KEY = "neulay_gcn_steps"
_NEULAY_LINEAR_STEPS_KEY = "neulay_linear_steps"
_NEULAY_QUERY_RADIUS_KEY = "neulay_query_radius"
_NEULAY_LOSS_WINDOW_KEY = "neulay_loss_window"
_NEULAY_PAIRS_KEY = "neulay_pairs"
_NEULAY_DEFAULT_DIM = 2
_NEULAY_OLD_CODE_DIM = 3
_NEULAY_DEFAULT_GCN_STEPS = 2_000
_NEULAY_OLD_CODE_GCN_STEPS = 40_000
_NEULAY_OLD_CODE_FDL_STEPS = 1_000_000
_NEULAY_OLD_CODE_QUERY_RADIUS = 4.0


@dataclass(frozen=True)
class NeuLayPrepareStateConfig:
    """Configuration for :class:`NeuLayPrepareState`.

    Parameters
    ----------
    dim : int | None, default=None
        Output dimensionality. ``None`` resolves from ``fidelity_mode``.
    lr : float, default=0.01
        Direct-phase optimizer learning rate.
    radius : float, default=0.4
        Gaussian repulsion radius.
    magnitude : float | None, default=None
        NeuLay repulsion magnitude. ``None`` triggers adaptive resolution.
    use_gcn : bool, default=True
        Whether to run the optional GCN pre-phase.
    gcn_steps : int | None, default=None
        Number of gradient steps in the GCN phase. ``None`` resolves from
        ``fidelity_mode``.
    fdl_steps : int | None, default=None
        Direct refinement step budget. ``None`` preserves the historical
        total-budget interpretation except in old-code fidelity mode.
    total_steps : int, default=20_000
        Total optimization budget across both phases.
    magnitude_scale_base : float, default=100.0
        Base coefficient for the adaptive repulsion magnitude heuristic.
    query_radius_factor : float, default=4.0
        Multiplier that expands the KD-tree query radius beyond ``radius``.
    query_radius : float | None, default=None
        Absolute KD-tree query radius. When set, this overrides
        ``query_radius_factor``.
    fidelity_mode : {"old_code"} | None, default=None
        Optional reference-default bundle. ``"old_code"`` targets
        ``old_code/NeuLay-2.py`` defaults.
    """

    dim: Optional[int] = None
    lr: float = 0.01
    radius: float = 0.4
    magnitude: Optional[float] = None
    use_gcn: bool = True
    gcn_steps: Optional[int] = None
    fdl_steps: Optional[int] = None
    total_steps: int = 20_000
    magnitude_scale_base: float = 100.0
    query_radius_factor: float = 4.0
    query_radius: Optional[float] = None
    fidelity_mode: Optional[str] = None


@dataclass(frozen=True)
class NeuLayRunGCNPhaseConfig:
    """Configuration for :class:`NeuLayRunGCNPhase`.

    Parameters
    ----------
    patience : int, default=10
        Rolling loss-window length used for the GCN early-stop heuristic.
    relative_tolerance : float, default=1e-4
        Relative loss-window threshold, scaled by ``sqrt(num_nodes)``.
    optimizer_lr : float, default=0.01
        RMSprop learning rate for the GCN phase.
    pair_refresh_interval : int, default=5
        Steps between KD-tree neighborhood refreshes.
    """

    patience: int = 10
    relative_tolerance: float = 1.0e-4
    optimizer_lr: float = 0.01
    pair_refresh_interval: int = 5


@dataclass(frozen=True)
class NeuLayPrepareDirectOptimizerConfig:
    """Configuration for :class:`NeuLayPrepareDirectOptimizer`.

    Parameters
    ----------
    patience : int, default=10
        Rolling loss-window length used by the direct-phase convergence check.
    """

    patience: int = 10


@dataclass(frozen=True)
class NeuLayDirectStepConfig:
    """Configuration for :class:`NeuLayDirectStep`.

    Parameters
    ----------
    pair_refresh_interval : int, default=5
        Steps between KD-tree neighborhood refreshes.
    """

    pair_refresh_interval: int = 5


@dataclass(frozen=True)
class NeuLayDirectConvergenceCheckConfig:
    """Configuration for :class:`NeuLayDirectConvergenceCheck`.

    Parameters
    ----------
    relative_tolerance : float, default=1e-8
        Relative loss-window threshold, scaled by ``sqrt(num_nodes)``.
    """

    relative_tolerance: float = 1.0e-8


class _SparseGCN(nn.Module):
    """Sparse GCN layer implementing ``A_norm @ (X @ W)``."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        adj_norm: torch.Tensor,
        gain: float,
    ) -> None:
        super().__init__()
        self.adj_norm = adj_norm
        self.weight = nn.Parameter(
            torch.empty(in_dim, out_dim, device=adj_norm.device, dtype=torch.float32)
        )
        nn.init.xavier_uniform_(self.weight, gain=gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one sparse GCN message-passing step.

        Parameters
        ----------
        x : torch.Tensor
            Node features with shape ``[N, in_dim]``.

        Returns
        -------
        torch.Tensor
            Updated features with shape ``[N, out_dim]``.
        """
        support = torch.mm(x, self.weight)
        return torch.sparse.mm(self.adj_norm, support)


class _ResGCN(nn.Module):
    """NeuLay-style residual two-layer sparse GCN model."""

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
        adj_norm = _build_normalized_adjacency(edge_index=edge_index, num_nodes=num_nodes).to(
            device
        )

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
        """Generate coordinates from residual GCN features."""
        h0 = self.weight1
        h1 = torch.tanh(self.gcn1(h0))
        h2 = self.gcn2(h1)
        return torch.mm(torch.cat([h0, h1, h2], dim=1), self.weight2)


def _build_normalized_adjacency(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Build the NeuLay symmetric degree-normalized adjacency matrix."""
    if edge_index.numel() == 0:
        idx = torch.arange(num_nodes, dtype=torch.long)
        return torch.sparse_coo_tensor(
            torch.stack([idx, idx]),
            torch.ones(num_nodes, dtype=torch.float32),
            (num_nodes, num_nodes),
        ).coalesce()

    src = edge_index[0].cpu()
    dst = edge_index[1].cpu()
    pairs = np.stack(
        [
            np.concatenate([src.numpy(), dst.numpy()]),
            np.concatenate([dst.numpy(), src.numpy()]),
        ],
        axis=0,
    ).astype(np.int64)
    values = np.ones(pairs.shape[1], dtype=np.float32)

    adjacency = sp.coo_matrix((values, (pairs[0], pairs[1])), shape=(num_nodes, num_nodes))
    adjacency = adjacency + sp.eye(num_nodes, dtype=np.float32, format="coo")
    adjacency.sum_duplicates()
    if adjacency.nnz > 0:
        adjacency.data[:] = 1.0

    degree = np.asarray(adjacency.sum(axis=0), dtype=np.float32).ravel()
    inv_sqrt = np.zeros_like(degree)
    mask = degree > 0
    inv_sqrt[mask] = 1.0 / np.sqrt(degree[mask])
    d_mat = sp.diags(inv_sqrt).tocsr()
    normalized = d_mat.dot(adjacency.dot(d_mat)).tocoo()

    indices = torch.from_numpy(np.vstack((normalized.row, normalized.col)).astype(np.int64))
    values = torch.from_numpy(normalized.data.astype(np.float32))
    return torch.sparse_coo_tensor(
        indices,
        values,
        (num_nodes, num_nodes),
        dtype=torch.float32,
    ).coalesce()


def _initial_positions(num_nodes: int, dim: int, device: torch.device) -> torch.Tensor:
    """Create random Xavier-initialized start positions."""
    gain = float(max(num_nodes, 1)) ** (1.0 / float(max(dim, 1)))
    initial = torch.empty((num_nodes, dim), device=device, dtype=torch.float32)
    nn.init.xavier_uniform_(initial, gain=gain)
    return initial


def _relative_window_difference(loss_window: list[float]) -> float:
    """Compute relative range over a rolling loss window."""
    max_loss = max(loss_window)
    if max_loss <= 0.0:
        return 0.0
    return (max_loss - min(loss_window)) / max_loss


def _query_pairs(pos: torch.Tensor, query_radius: float) -> np.ndarray:
    """Return KD-tree pairs within ``query_radius``."""
    if pos.shape[0] < 2:
        return np.empty((0, 2), dtype=np.int64)
    tree = cKDTree(pos.detach().cpu().numpy())
    pairs = tree.query_pairs(query_radius, output_type="ndarray")
    if pairs.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    return pairs.astype(np.int64)


def _kdtree_repulsion_loss(
    pos: torch.Tensor, pairs: np.ndarray, radius: float, magnitude: float
) -> torch.Tensor:
    """Evaluate NeuLay Gaussian repulsion for cached KD-tree pairs."""
    if pairs.shape[0] == 0 or magnitude == 0.0:
        return pos.sum() * 0.0
    idx = torch.from_numpy(pairs).to(device=pos.device)
    sq_dist = ((pos[idx[:, 0]] - pos[idx[:, 1]]) ** 2).sum(dim=-1)
    return magnitude * torch.exp(-sq_dist / (4.0 * radius * radius)).sum()


def _is_old_code_fidelity(fidelity_mode: Optional[str]) -> bool:
    """Return whether NeuLay should resolve old-code reference defaults.

    Parameters
    ----------
    fidelity_mode : str | None
        Optional fidelity mode name.

    Returns
    -------
    bool
        ``True`` when ``fidelity_mode`` selects the old script defaults.

    Raises
    ------
    ValueError
        If the mode is not recognized.
    """
    if fidelity_mode is None:
        return False
    if fidelity_mode == "old_code":
        return True
    raise ValueError("NeuLay fidelity_mode must be None or 'old_code'.")


def _resolve_dim(dim: Optional[int], fidelity_mode: Optional[str]) -> int:
    """Resolve the output dimension from explicit config and fidelity mode.

    Parameters
    ----------
    dim : int | None
        Explicit output dimension, if supplied.
    fidelity_mode : str | None
        Optional fidelity mode name.

    Returns
    -------
    int
        Positive output dimension.
    """
    if dim is not None:
        return dim
    if _is_old_code_fidelity(fidelity_mode):
        return _NEULAY_OLD_CODE_DIM
    return _NEULAY_DEFAULT_DIM


def _resolve_gcn_steps(gcn_steps: Optional[int], fidelity_mode: Optional[str]) -> int:
    """Resolve the GCN step budget from explicit config and fidelity mode.

    Parameters
    ----------
    gcn_steps : int | None
        Explicit GCN step budget, if supplied.
    fidelity_mode : str | None
        Optional fidelity mode name.

    Returns
    -------
    int
        Non-negative GCN step budget.
    """
    if gcn_steps is not None:
        return gcn_steps
    if _is_old_code_fidelity(fidelity_mode):
        return _NEULAY_OLD_CODE_GCN_STEPS
    return _NEULAY_DEFAULT_GCN_STEPS


def _resolve_linear_steps(
    total_steps: int,
    gcn_steps: int,
    fdl_steps: Optional[int],
    use_gcn: bool,
    fidelity_mode: Optional[str],
) -> int:
    """Resolve the direct refinement budget.

    Parameters
    ----------
    total_steps : int
        Historical total optimization budget.
    gcn_steps : int
        Resolved GCN step budget.
    fdl_steps : int | None
        Explicit direct refinement budget, if supplied.
    use_gcn : bool
        Whether the GCN phase is enabled.
    fidelity_mode : str | None
        Optional fidelity mode name.

    Returns
    -------
    int
        Non-negative direct refinement step budget.
    """
    if fdl_steps is not None:
        return fdl_steps
    if _is_old_code_fidelity(fidelity_mode):
        return _NEULAY_OLD_CODE_FDL_STEPS
    return max(total_steps - gcn_steps, 0) if use_gcn else total_steps


@register_op
class NeuLaySeedRNG(Op):
    """Seed classic NeuLay RNG state (PyTorch and NumPy)."""

    name: ClassVar[str] = "neulay_seed_rng"
    category: ClassVar[OpCategory] = OpCategory.INIT
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = ()
    access_pattern: ClassVar[str] = "global"

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Set global PRNG state for deterministic NeuLay trajectories."""
        del ctx

        torch.manual_seed(problem.seed)
        np.random.seed(problem.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(problem.seed)
        return state


@register_op
class NeuLayPrepareState(Op):
    """Clean edge index and cache NeuLay hyper-parameters."""

    config: NeuLayPrepareStateConfig

    name: ClassVar[str] = "neulay_prepare_state"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = (
        f"extras.{_NEULAY_CLEANED_EDGE_INDEX_KEY}",
        f"extras.{_NEULAY_DEVICE_KEY}",
        f"extras.{_NEULAY_DIM_KEY}",
        f"extras.{_NEULAY_LR_KEY}",
        f"extras.{_NEULAY_RADIUS_KEY}",
        f"extras.{_NEULAY_MAGNITUDE_KEY}",
        f"extras.{_NEULAY_USE_GCN_KEY}",
        f"extras.{_NEULAY_GCN_STEPS_KEY}",
        f"extras.{_NEULAY_LINEAR_STEPS_KEY}",
        f"extras.{_NEULAY_QUERY_RADIUS_KEY}",
    )
    access_pattern: ClassVar[str] = "global"

    def __init__(self, config: Optional[NeuLayPrepareStateConfig] = None) -> None:
        """Store reusable NeuLay configuration.

        Parameters
        ----------
        config : NeuLayPrepareStateConfig | None, optional
            Optional preparation configuration.
        """
        self.config = config or NeuLayPrepareStateConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Resolve reusable NeuLay values into ``state.extras``."""
        del ctx

        old_code_fidelity = _is_old_code_fidelity(self.config.fidelity_mode)
        dim = _resolve_dim(self.config.dim, self.config.fidelity_mode)
        gcn_steps = _resolve_gcn_steps(self.config.gcn_steps, self.config.fidelity_mode)
        linear_steps = _resolve_linear_steps(
            total_steps=self.config.total_steps,
            gcn_steps=gcn_steps,
            fdl_steps=self.config.fdl_steps,
            use_gcn=self.config.use_gcn,
            fidelity_mode=self.config.fidelity_mode,
        )
        if dim <= 0:
            raise ValueError("NeuLay dim must be positive.")
        if self.config.lr <= 0.0:
            raise ValueError("NeuLay lr must be positive.")
        if self.config.radius <= 0.0:
            raise ValueError("NeuLay radius must be positive.")
        if self.config.magnitude is not None and self.config.magnitude < 0.0:
            raise ValueError("NeuLay magnitude must be non-negative.")
        if gcn_steps < 0:
            raise ValueError("NeuLay gcn_steps must be non-negative.")
        if linear_steps < 0:
            raise ValueError("NeuLay fdl_steps must be non-negative.")
        if self.config.total_steps < 0:
            raise ValueError("NeuLay total_steps must be non-negative.")
        if self.config.query_radius_factor <= 0.0:
            raise ValueError("NeuLay query_radius_factor must be positive.")
        if self.config.query_radius is not None and self.config.query_radius <= 0.0:
            raise ValueError("NeuLay query_radius must be positive.")

        device = (
            problem.edge_index.device
            if problem.edge_index.numel() > 0
            else problem.node_sizes.device
            if problem.node_sizes is not None
            else torch.device("cpu")
        )
        cleaned = problem.edge_index.to(device=device, dtype=torch.long)
        if cleaned.numel() == 0:
            cleaned = cleaned.reshape(2, 0)
        else:
            non_self = cleaned[0] != cleaned[1]
            cleaned = cleaned[:, non_self].contiguous()

        if self.config.magnitude is None:
            magnitude = (
                self.config.magnitude_scale_base
                * (problem.num_nodes ** (1.0 / 3.0))
                * self.config.radius
            )
        else:
            magnitude = float(self.config.magnitude)

        if self.config.query_radius is not None:
            query_radius = self.config.query_radius
        elif old_code_fidelity:
            query_radius = _NEULAY_OLD_CODE_QUERY_RADIUS
        else:
            query_radius = self.config.query_radius_factor * self.config.radius

        state.extras[_NEULAY_CLEANED_EDGE_INDEX_KEY] = cleaned
        state.extras[_NEULAY_DEVICE_KEY] = device
        state.extras[_NEULAY_DIM_KEY] = dim
        state.extras[_NEULAY_LR_KEY] = self.config.lr
        state.extras[_NEULAY_RADIUS_KEY] = self.config.radius
        state.extras[_NEULAY_MAGNITUDE_KEY] = magnitude
        state.extras[_NEULAY_USE_GCN_KEY] = self.config.use_gcn
        state.extras[_NEULAY_GCN_STEPS_KEY] = gcn_steps
        state.extras[_NEULAY_LINEAR_STEPS_KEY] = linear_steps
        state.extras[_NEULAY_QUERY_RADIUS_KEY] = query_radius
        return state


@register_op
class NeuLayRunGCNPhase(Op):
    """Run the optional NeuLay GCN pre-optimization phase."""

    config: NeuLayRunGCNPhaseConfig

    name: ClassVar[str] = "neulay_gcn_phase"
    category: ClassVar[OpCategory] = OpCategory.INIT
    reads: ClassVar[Tuple[str, ...]] = (
        f"extras.{_NEULAY_CLEANED_EDGE_INDEX_KEY}",
        f"extras.{_NEULAY_DEVICE_KEY}",
        f"extras.{_NEULAY_DIM_KEY}",
        f"extras.{_NEULAY_RADIUS_KEY}",
        f"extras.{_NEULAY_MAGNITUDE_KEY}",
        f"extras.{_NEULAY_GCN_STEPS_KEY}",
        f"extras.{_NEULAY_USE_GCN_KEY}",
        f"extras.{_NEULAY_QUERY_RADIUS_KEY}",
    )
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = (
        f"extras.{_NEULAY_CLEANED_EDGE_INDEX_KEY}",
        f"extras.{_NEULAY_DEVICE_KEY}",
        f"extras.{_NEULAY_DIM_KEY}",
    )
    access_pattern: ClassVar[str] = "global"

    def __init__(self, config: Optional[NeuLayRunGCNPhaseConfig] = None) -> None:
        """Store GCN-phase optimization controls.

        Parameters
        ----------
        config : NeuLayRunGCNPhaseConfig | None, optional
            Optional GCN-phase configuration.
        """
        self.config = config or NeuLayRunGCNPhaseConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run GCN training unless disabled by config and return coarse positions."""
        del ctx

        cleaned = state.extras[_NEULAY_CLEANED_EDGE_INDEX_KEY]
        device = state.extras[_NEULAY_DEVICE_KEY]
        dim = state.extras[_NEULAY_DIM_KEY]
        radius = state.extras[_NEULAY_RADIUS_KEY]
        magnitude = state.extras[_NEULAY_MAGNITUDE_KEY]
        gcn_steps = state.extras[_NEULAY_GCN_STEPS_KEY]
        use_gcn = state.extras[_NEULAY_USE_GCN_KEY]
        query_radius = state.extras[_NEULAY_QUERY_RADIUS_KEY]

        if use_gcn and gcn_steps > 0:
            model = _ResGCN(
                num_nodes=problem.num_nodes,
                dim=dim,
                device=device,
                edge_index=cleaned,
            )
            optimizer = torch.optim.RMSprop(model.parameters(), lr=self.config.optimizer_lr)
            loss_window = [0.0] * self.config.patience
            pairs = np.empty((0, 2), dtype=np.int64)

            for step in range(gcn_steps):
                optimizer.zero_grad(set_to_none=True)
                output = model()
                if step % self.config.pair_refresh_interval == 0:
                    pairs = _query_pairs(output, query_radius)

                if cleaned.numel() == 0:
                    elastic = output.sum() * 0.0
                else:
                    src = cleaned[0]
                    dst = cleaned[1]
                    low = torch.minimum(src, dst)
                    high = torch.maximum(src, dst)
                    pairs_matrix = torch.stack([low, high], dim=0)
                    # NeuLay treats edges as undirected in the elastic term,
                    # so parallel opposite directions must collapse here.
                    unique_pairs = torch.unique(pairs_matrix, dim=1)
                    diff = output[unique_pairs[0]] - output[unique_pairs[1]]
                    elastic = diff.square().sum() * 0.5

                repulsion = _kdtree_repulsion_loss(
                    output,
                    pairs=pairs,
                    radius=radius,
                    magnitude=magnitude,
                )
                loss = elastic + repulsion
                loss.backward()
                optimizer.step()
                loss_window.append(float(loss.detach().item()))
                loss_window.pop(0)
                if _relative_window_difference(loss_window) < (
                    self.config.relative_tolerance * math.sqrt(float(problem.num_nodes))
                ):
                    break

            with torch.no_grad():
                state.pos = model().detach()
            return state

        state.pos = _initial_positions(
            num_nodes=problem.num_nodes,
            dim=dim,
            device=device,
        )
        return state


@register_op
class NeuLayPrepareDirectOptimizer(Op):
    """Create RMSprop state for NeuLay's direct refinement phase."""

    config: NeuLayPrepareDirectOptimizerConfig

    name: ClassVar[str] = "neulay_init_direct_optimizer"
    category: ClassVar[OpCategory] = OpCategory.OPTIMIZE
    reads: ClassVar[Tuple[str, ...]] = ("pos", f"extras.{_NEULAY_LR_KEY}")
    writes: ClassVar[Tuple[str, ...]] = (
        "pos",
        "optimizer",
        f"extras.{_NEULAY_LOSS_WINDOW_KEY}",
        f"extras.{_NEULAY_PAIRS_KEY}",
    )
    requires: ClassVar[Tuple[str, ...]] = ("pos",)
    access_pattern: ClassVar[str] = "global"

    def __init__(self, config: Optional[NeuLayPrepareDirectOptimizerConfig] = None) -> None:
        """Store direct-phase optimizer initialization settings.

        Parameters
        ----------
        config : NeuLayPrepareDirectOptimizerConfig | None, optional
            Optional direct-optimizer configuration.
        """
        self.config = config or NeuLayPrepareDirectOptimizerConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Clone initial coordinates and attach an RMSprop optimizer."""
        del problem, ctx

        if state.pos is None:
            raise ValueError("NeuLayPrepareDirectOptimizer requires state.pos.")

        lr = state.extras[_NEULAY_LR_KEY]
        state.pos = nn.Parameter(state.pos.clone())
        state.optimizer = torch.optim.RMSprop([state.pos], lr=lr)
        state.extras[_NEULAY_LOSS_WINDOW_KEY] = [0.0] * self.config.patience
        state.extras[_NEULAY_PAIRS_KEY] = np.empty((0, 2), dtype=np.int64)
        return state


@register_op
class NeuLayDirectStep(Op):
    """Run one NeuLay direct refinement step."""

    config: NeuLayDirectStepConfig

    name: ClassVar[str] = "neulay_direct_step"
    category: ClassVar[OpCategory] = OpCategory.OPTIMIZE
    reads: ClassVar[Tuple[str, ...]] = (
        "pos",
        "optimizer",
        f"extras.{_NEULAY_CLEANED_EDGE_INDEX_KEY}",
        f"extras.{_NEULAY_RADIUS_KEY}",
        f"extras.{_NEULAY_MAGNITUDE_KEY}",
        f"extras.{_NEULAY_QUERY_RADIUS_KEY}",
        f"extras.{_NEULAY_LOSS_WINDOW_KEY}",
        f"extras.{_NEULAY_PAIRS_KEY}",
    )
    writes: ClassVar[Tuple[str, ...]] = (
        "pos",
        f"extras.{_NEULAY_LOSS_WINDOW_KEY}",
        f"extras.{_NEULAY_PAIRS_KEY}",
    )
    requires: ClassVar[Tuple[str, ...]] = ("pos",)
    access_pattern: ClassVar[str] = "global"

    def __init__(self, config: Optional[NeuLayDirectStepConfig] = None) -> None:
        """Store direct-phase step controls.

        Parameters
        ----------
        config : NeuLayDirectStepConfig | None, optional
            Optional direct-step configuration.
        """
        self.config = config or NeuLayDirectStepConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute NeuLay loss, backpropagate, and apply one RMSprop update."""
        del problem, ctx

        if state.pos is None:
            raise ValueError("NeuLayDirectStep requires state.pos.")
        if state.optimizer is None:
            raise ValueError("NeuLayDirectStep requires state.optimizer.")

        cleaned = state.extras[_NEULAY_CLEANED_EDGE_INDEX_KEY]
        radius = state.extras[_NEULAY_RADIUS_KEY]
        magnitude = state.extras[_NEULAY_MAGNITUDE_KEY]
        query_radius = state.extras[_NEULAY_QUERY_RADIUS_KEY]
        loss_window = state.extras[_NEULAY_LOSS_WINDOW_KEY]
        pairs = state.extras[_NEULAY_PAIRS_KEY]

        state.optimizer.zero_grad(set_to_none=True)
        if state.step % self.config.pair_refresh_interval == 0:
            pairs = _query_pairs(state.pos, query_radius)
            state.extras[_NEULAY_PAIRS_KEY] = pairs

        if cleaned.numel() == 0:
            elastic = state.pos.sum() * 0.0
        else:
            src = cleaned[0]
            dst = cleaned[1]
            low = torch.minimum(src, dst)
            high = torch.maximum(src, dst)
            pairs_matrix = torch.stack([low, high], dim=0)
            # The direct phase uses the same undirected elastic energy as the
            # GCN warm start, so mirror edges collapse to one spring.
            unique_pairs = torch.unique(pairs_matrix, dim=1)
            diff = state.pos[unique_pairs[0]] - state.pos[unique_pairs[1]]
            elastic = diff.square().sum() * 0.5

        repulsion = _kdtree_repulsion_loss(
            state.pos,
            pairs=pairs,
            radius=radius,
            magnitude=magnitude,
        )
        loss = elastic + repulsion
        loss.backward()
        state.optimizer.step()

        loss_window.append(float(loss.detach().item()))
        loss_window.pop(0)
        state.extras[_NEULAY_LOSS_WINDOW_KEY] = loss_window
        return state


@register_op
class NeuLayDirectConvergenceCheck(Op):
    """Check NeuLay convergence in direct refinement."""

    config: NeuLayDirectConvergenceCheckConfig

    name: ClassVar[str] = "neulay_direct_convergence"
    category: ClassVar[OpCategory] = OpCategory.CONVERGE
    reads: ClassVar[Tuple[str, ...]] = (f"extras.{_NEULAY_LOSS_WINDOW_KEY}",)
    writes: ClassVar[Tuple[str, ...]] = ()
    requires: ClassVar[Tuple[str, ...]] = (f"extras.{_NEULAY_LOSS_WINDOW_KEY}",)
    access_pattern: ClassVar[str] = "global"

    def __init__(
        self,
        config: Optional[NeuLayDirectConvergenceCheckConfig] = None,
    ) -> None:
        """Store direct-phase convergence settings.

        Parameters
        ----------
        config : NeuLayDirectConvergenceCheckConfig | None, optional
            Optional convergence configuration.
        """
        self.config = config or NeuLayDirectConvergenceCheckConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Set ``state.converged`` when the sliding loss window stabilizes."""
        del ctx
        loss_window = state.extras[_NEULAY_LOSS_WINDOW_KEY]
        threshold = self.config.relative_tolerance * math.sqrt(float(problem.num_nodes))
        if _relative_window_difference(loss_window) < threshold:
            state.converged = True
        return state


@register_op
class NeuLayFinalizePositions(Op):
    """Detach final coordinates for deterministic output."""

    name: ClassVar[str] = "neulay_finalize_positions"
    category: ClassVar[OpCategory] = OpCategory.POSTPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos",)
    access_pattern: ClassVar[str] = "global"

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Detach the final positions from autograd and return them."""
        del problem, ctx
        if state.pos is None:
            raise ValueError("NeuLayFinalizePositions requires state.pos.")
        state.pos = state.pos.detach()
        return state


__all__ = [
    "NeuLaySeedRNG",
    "NeuLayPrepareState",
    "NeuLayPrepareStateConfig",
    "NeuLayRunGCNPhaseConfig",
    "NeuLayRunGCNPhase",
    "NeuLayPrepareDirectOptimizerConfig",
    "NeuLayPrepareDirectOptimizer",
    "NeuLayDirectStepConfig",
    "NeuLayDirectStep",
    "NeuLayDirectConvergenceCheckConfig",
    "NeuLayDirectConvergenceCheck",
    "NeuLayFinalizePositions",
]
