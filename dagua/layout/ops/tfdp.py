"""t-FDP-compatible composable layout operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch

from dagua.layout.ops.base import Op
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_DEFAULT_ALPHA_DECAY = 0.012955423246736264
_DEFAULT_MAX_STEP = 1.0
_DEFAULT_ATTRACTION_CAP = 2000.0
_DEFAULT_JITTER_SCALE = 0.01
_DEFAULT_MOMENTUM_DECAY = 0.6
_DISPLACEMENT_EPS = 1.0e-32
_ATTRACTION_EPS = 1.0e-12
_PMDS_POWER_ITERATIONS = 100

TFDPInitMode = Literal["pmds", "random"]
TFDPForceMode = Literal["exact", "fft"]


@dataclass(frozen=True)
class TFDPConfig:
    """Parameter bundle for t-distributed force-directed placement.

    Parameters
    ----------
    init : {"pmds", "random"}, default="pmds"
        Initial coordinate strategy. ``"pmds"`` follows the reference
        PivotMDS initializer with small Gaussian jitter.
    force_mode : {"exact", "fft"}, default="exact"
        Force evaluation mode. ``"exact"`` uses the reference O(N^2)
        repulsion. ``"fft"`` currently falls back to exact force evaluation
        while preserving the public variant hook.
    max_iter : int, default=300
        Number of optimization iterations.
    alpha : float, default=0.1
        Reference long-range force parameter.
    beta : float, default=8.0
        Attraction force parameter.
    gamma : float, default=2.0
        t-force repulsion exponent.
    combine : bool, default=True
        Retained for reference API compatibility with ibFFT modes.
    seed : int or None, default=None
        Deterministic seed used for initialization and iteration jitter.
    pmds_pivots : int, default=100
        Number of PivotMDS pivots before clipping to ``N``.
    dtype : torch.dtype, default=torch.float32
        Internal coordinate dtype. The reference exact loop uses float32.
    """

    init: TFDPInitMode = "pmds"
    force_mode: TFDPForceMode = "exact"
    max_iter: int = 300
    alpha: float = 0.1
    beta: float = 8.0
    gamma: float = 2.0
    combine: bool = True
    seed: Optional[int] = None
    pmds_pivots: int = 100
    dtype: torch.dtype = torch.float32


def _make_generator(seed: Optional[int], device: torch.device) -> torch.Generator:
    """Build a deterministic torch generator.

    Parameters
    ----------
    seed : int or None
        Requested seed. ``None`` maps to ``0`` for deterministic Dagua runs.
    device : torch.device
        Device that will consume generated values.

    Returns
    -------
    torch.Generator
        Seeded generator for reproducible initialization and jitter.
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(0 if seed is None else int(seed))
    return generator


def _symmetrized_csr(
    edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the reference-style symmetric CSR adjacency.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    device : torch.device
        Device for returned tensors.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(indptr, indices)`` tensors matching SciPy CSR layout.
    """
    neighbors: list[set[int]] = [set() for _ in range(num_nodes)]
    if edge_index.numel() > 0:
        edges = edge_index.detach().to(device="cpu", dtype=torch.long)
        for source, target in edges.t().tolist():
            src = int(source)
            tgt = int(target)
            if src == tgt or src < 0 or tgt < 0 or src >= num_nodes or tgt >= num_nodes:
                continue
            neighbors[src].add(tgt)
            neighbors[tgt].add(src)
    indptr = [0]
    indices: list[int] = []
    for row_neighbors in neighbors:
        ordered = sorted(row_neighbors)
        indices.extend(ordered)
        indptr.append(len(indices))
    return (
        torch.tensor(indptr, dtype=torch.long, device=device),
        torch.tensor(indices, dtype=torch.long, device=device),
    )


def _pivot_distances(
    indptr: torch.Tensor,
    indices: torch.Tensor,
    pivots: torch.Tensor,
    num_nodes: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compute unweighted shortest path distances from selected pivots.

    Parameters
    ----------
    indptr : torch.Tensor
        CSR row pointer with shape ``[N + 1]``.
    indices : torch.Tensor
        CSR column indices with shape ``[2E]`` for the symmetrized graph.
    pivots : torch.Tensor
        Pivot node indices with shape ``[P]``.
    num_nodes : int
        Number of graph nodes.
    dtype : torch.dtype
        Floating dtype for returned distances.

    Returns
    -------
    torch.Tensor
        Distance matrix with shape ``[P, N]``.
    """
    indptr_cpu = indptr.detach().cpu().tolist()
    indices_cpu = indices.detach().cpu().tolist()
    pivots_cpu = pivots.detach().cpu().tolist()
    rows: list[list[float]] = []
    for pivot in pivots_cpu:
        distances = [-1.0] * num_nodes
        distances[int(pivot)] = 0.0
        queue = [int(pivot)]
        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            next_distance = distances[node] + 1.0
            for offset in range(indptr_cpu[node], indptr_cpu[node + 1]):
                neighbor = int(indices_cpu[offset])
                if distances[neighbor] >= 0.0:
                    continue
                distances[neighbor] = next_distance
                queue.append(neighbor)
        rows.append([0.0 if value < 0.0 else value for value in distances])
    return torch.tensor(rows, dtype=dtype, device=indptr.device)


def tfdp_scale_by_edge(
    pos: torch.Tensor,
    indptr: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Return the reference edge-length scale factor.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    indptr : torch.Tensor
        CSR row pointer with shape ``[N + 1]``.
    indices : torch.Tensor
        CSR column indices.

    Returns
    -------
    torch.Tensor
        Scalar ``sum(length) / sum(length ** 2)``.
    """
    if indices.numel() == 0:
        return torch.ones((), dtype=pos.dtype, device=pos.device)
    source_counts = indptr[1:] - indptr[:-1]
    sources = torch.repeat_interleave(torch.arange(pos.shape[0], device=pos.device), source_counts)
    deltas = pos[sources] - pos[indices]
    lengths = torch.linalg.vector_norm(deltas, dim=1)
    denominator = (lengths * lengths).sum().clamp_min(torch.finfo(pos.dtype).eps)
    return lengths.sum() / denominator


def tfdp_pivot_mds(
    indptr: torch.Tensor,
    indices: torch.Tensor,
    num_nodes: int,
    pivots_count: int,
    generator: torch.Generator,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compute the reference-style PivotMDS initialization.

    Parameters
    ----------
    indptr : torch.Tensor
        CSR row pointer with shape ``[N + 1]``.
    indices : torch.Tensor
        CSR column indices.
    num_nodes : int
        Number of graph nodes.
    pivots_count : int
        Requested pivot count before clipping to ``N``.
    generator : torch.Generator
        Deterministic random generator.
    dtype : torch.dtype
        Coordinate dtype.

    Returns
    -------
    torch.Tensor
        Initial positions with shape ``[N, 2]``.
    """
    if num_nodes == 0:
        return torch.zeros((0, 2), dtype=dtype, device=indptr.device)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=dtype, device=indptr.device)
    pivot_total = min(max(int(pivots_count), 1), num_nodes)
    if pivot_total >= num_nodes:
        pivots = torch.arange(num_nodes, dtype=torch.long, device=indptr.device)
    else:
        pivots = torch.randperm(num_nodes, generator=generator, device=indptr.device)[:pivot_total]
    distances = _pivot_distances(indptr, indices, pivots, num_nodes, dtype)
    squared = distances * distances
    delta_is = squared.sum(dim=0) / float(pivot_total)
    delta_rj = squared.sum(dim=1) / float(num_nodes)
    sum_all = squared.sum() / float(num_nodes * pivot_total)
    centered = -0.5 * (squared.T - delta_rj.unsqueeze(0) - delta_is.unsqueeze(1) + sum_all)
    gram = centered.T @ centered
    pos = torch.zeros((num_nodes, 2), dtype=dtype, device=indptr.device)
    working = gram
    for axis in range(2):
        vector = torch.rand((pivot_total,), generator=generator, dtype=dtype, device=indptr.device)
        for _ in range(_PMDS_POWER_ITERATIONS):
            next_vector = working @ vector
            norm = torch.linalg.vector_norm(next_vector).clamp_min(torch.finfo(dtype).eps)
            vector = next_vector / norm
        eigenvalue = vector @ (working @ vector)
        pos[:, axis] = centered @ vector
        working = working - eigenvalue * torch.outer(vector, vector)
    return pos


def _compute_bias(indptr: torch.Tensor, indices: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Compute degree bias for each CSR edge.

    Parameters
    ----------
    indptr : torch.Tensor
        CSR row pointer with shape ``[N + 1]``.
    indices : torch.Tensor
        CSR column indices.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    torch.Tensor
        Bias vector with shape ``[indices.numel()]``.
    """
    del num_nodes
    degrees = (indptr[1:] - indptr[:-1]).to(dtype=torch.float32)
    if indices.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32, device=indices.device)
    source_counts = indptr[1:] - indptr[:-1]
    sources = torch.repeat_interleave(
        torch.arange(degrees.shape[0], device=indices.device),
        source_counts,
    )
    source_degree = degrees[sources]
    target_degree = degrees[indices]
    return target_degree / (source_degree + target_degree).clamp_min(torch.finfo(torch.float32).eps)


def _center_by_range(pos: torch.Tensor) -> torch.Tensor:
    """Center coordinates by min/max range midpoint.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Range-centered positions.
    """
    if pos.numel() == 0:
        return pos
    midpoint = (pos.max(dim=0).values + pos.min(dim=0).values) / 2.0
    return pos - midpoint.unsqueeze(0)


@register_op
class TFDPInitialize(Op):
    """Initialize t-FDP state."""

    name = "tfdp_initialize"
    category = OpCategory.INIT
    writes = ("pos", "extras")
    access_pattern = "global"

    def __init__(self, config: TFDPConfig) -> None:
        """Store t-FDP initialization configuration.

        Parameters
        ----------
        config : TFDPConfig
            t-FDP parameter bundle.
        """
        self.config = config

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Initialize positions, adjacency, bias, and displacement memory.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout problem.
        state : SolveState
            Mutable solver state.
        ctx : RuntimeContext
            Runtime context; accepted for pipeline compatibility.

        Returns
        -------
        SolveState
            State populated with t-FDP working tensors.
        """
        del ctx
        device = problem.edge_index.device if problem.edge_index.numel() else torch.device("cpu")
        dtype = self.config.dtype
        generator = _make_generator(self.config.seed, device)
        indptr, indices = _symmetrized_csr(problem.edge_index, problem.num_nodes, device)
        if self.config.init == "random":
            pos = torch.randn(
                (problem.num_nodes, 2),
                generator=generator,
                dtype=dtype,
                device=device,
            )
        else:
            noise = _DEFAULT_JITTER_SCALE * torch.randn(
                (problem.num_nodes, 2),
                generator=generator,
                dtype=dtype,
                device=device,
            )
            pos = tfdp_pivot_mds(
                indptr=indptr,
                indices=indices,
                num_nodes=problem.num_nodes,
                pivots_count=self.config.pmds_pivots,
                generator=generator,
                dtype=dtype,
            )
            pos = 2.0 * tfdp_scale_by_edge(pos, indptr, indices) * pos + noise
        state.pos = _center_by_range(pos.to(dtype=dtype))
        state.extras["tfdp_indptr"] = indptr
        state.extras["tfdp_indices"] = indices
        state.extras["tfdp_bias"] = _compute_bias(indptr, indices, problem.num_nodes).to(
            dtype=dtype
        )
        state.extras["tfdp_displacement"] = torch.zeros_like(state.pos)
        state.extras["tfdp_generator"] = _make_generator(self.config.seed, device)
        edge_count = max(int(indices.numel()), 1)
        initial_alpha = 1.0
        alpha_min = 0.01
        average_degree = edge_count / max(float(problem.num_nodes), 1.0)
        if average_degree >= 15.0:
            initial_alpha /= 10.0
            alpha_min /= 10.0
        if average_degree >= 50.0:
            initial_alpha /= 10.0
            alpha_min /= 10.0
        state.extras["tfdp_d3alpha"] = torch.tensor(initial_alpha, dtype=dtype, device=device)
        state.extras["tfdp_d3alpha_min"] = torch.tensor(alpha_min, dtype=dtype, device=device)
        return state


@register_op
class TFDPIteration(Op):
    """Run t-FDP exact-force optimization iterations."""

    name = "tfdp_iteration"
    category = OpCategory.FORCE
    reads = ("pos", "extras")
    writes = ("pos", "extras")
    access_pattern = "global"

    def __init__(self, config: TFDPConfig) -> None:
        """Store t-FDP iteration configuration.

        Parameters
        ----------
        config : TFDPConfig
            t-FDP parameter bundle.
        """
        self.config = config

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Apply the reference exact t-force loop.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout problem.
        state : SolveState
            Mutable solver state with initialized positions.
        ctx : RuntimeContext
            Runtime context; accepted for pipeline compatibility.

        Returns
        -------
        SolveState
            State with optimized ``pos``.

        Raises
        ------
        ValueError
            If positions were not initialized.
        """
        del problem, ctx
        if state.pos is None:
            raise ValueError("TFDPIteration requires initialized positions.")
        pos = state.pos
        indptr = state.extras["tfdp_indptr"]
        indices = state.extras["tfdp_indices"]
        bias = state.extras["tfdp_bias"].to(dtype=pos.dtype, device=pos.device)
        displacement = state.extras["tfdp_displacement"].to(dtype=pos.dtype, device=pos.device)
        generator = state.extras["tfdp_generator"]
        alpha = state.extras["tfdp_d3alpha"].to(dtype=pos.dtype, device=pos.device)
        alpha_min = state.extras["tfdp_d3alpha_min"].to(dtype=pos.dtype, device=pos.device)
        para_factor = 1.0 / float(self.config.alpha) if float(self.config.alpha) != 0.0 else 1.0
        for _ in range(max(int(self.config.max_iter), 0)):
            attr_force = self._attraction_force(pos, displacement, indptr, indices, bias, alpha)
            displacement = displacement + attr_force
            jitter = torch.randn(
                pos.shape,
                generator=generator,
                dtype=pos.dtype,
                device=pos.device,
            )
            displacement = displacement - _DEFAULT_JITTER_SCALE * alpha * jitter
            displacement = displacement + self._repulsion_force(pos, para_factor, alpha)
            pos = self._apply_force(pos, displacement)
            alpha = alpha + (alpha_min - alpha) * _DEFAULT_ALPHA_DECAY
            pos = _center_by_range(pos)
            displacement = displacement * _DEFAULT_MOMENTUM_DECAY
        state.pos = pos
        state.extras["tfdp_displacement"] = displacement
        state.extras["tfdp_d3alpha"] = alpha
        return state

    def _attraction_force(
        self,
        pos: torch.Tensor,
        displacement: torch.Tensor,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        bias: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reference attraction forces.

        Parameters
        ----------
        pos : torch.Tensor
            Current positions with shape ``[N, 2]``.
        displacement : torch.Tensor
            Previous displacement accumulator with shape ``[N, 2]``.
        indptr : torch.Tensor
            CSR row pointer with shape ``[N + 1]``.
        indices : torch.Tensor
            CSR column indices.
        bias : torch.Tensor
            Per-CSR-edge bias values.
        alpha : torch.Tensor
            Current d3-style alpha scalar.

        Returns
        -------
        torch.Tensor
            Attraction force with shape ``[N, 2]``.
        """
        force = torch.zeros_like(pos)
        if indices.numel() == 0:
            return force
        moved = pos + displacement
        source_counts = indptr[1:] - indptr[:-1]
        sources = torch.repeat_interleave(
            torch.arange(pos.shape[0], device=pos.device),
            source_counts,
        )
        deltas = moved[sources] - moved[indices]
        radius = float(self.config.beta) / (1.0 + (deltas * deltas).sum(dim=1)) + bias
        edge_force = -(radius.unsqueeze(1) * deltas)
        force.index_add_(0, sources, edge_force)
        force = alpha * force
        lengths = torch.linalg.vector_norm(force, dim=1, keepdim=True).clamp_min(_ATTRACTION_EPS)
        clipped = torch.clamp(lengths, max=_DEFAULT_ATTRACTION_CAP)
        return force / lengths * clipped

    def _repulsion_force(
        self,
        pos: torch.Tensor,
        para_factor: float,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """Compute exact O(N^2) t-distributed repulsion.

        Parameters
        ----------
        pos : torch.Tensor
            Current positions with shape ``[N, 2]``.
        para_factor : float
            ``1 / alpha`` reference long-range scale factor.
        alpha : torch.Tensor
            Current d3-style alpha scalar.

        Returns
        -------
        torch.Tensor
            Repulsive force with shape ``[N, 2]``.
        """
        if pos.shape[0] == 0:
            return pos
        deltas = pos.unsqueeze(1) - pos.unsqueeze(0)
        squared = (deltas * deltas).sum(dim=2)
        weights = alpha * float(para_factor) * torch.pow(1.0 + squared, -float(self.config.gamma))
        return (weights.unsqueeze(2) * deltas).sum(dim=1)

    def _apply_force(self, pos: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
        """Apply capped displacement to positions.

        Parameters
        ----------
        pos : torch.Tensor
            Current positions with shape ``[N, 2]``.
        displacement : torch.Tensor
            Displacement accumulator with shape ``[N, 2]``.

        Returns
        -------
        torch.Tensor
            Updated positions with shape ``[N, 2]``.
        """
        lengths = torch.linalg.vector_norm(displacement, dim=1, keepdim=True).clamp_min(
            _DISPLACEMENT_EPS
        )
        capped = torch.clamp(lengths, max=_DEFAULT_MAX_STEP)
        return pos + displacement / lengths * capped


__all__ = [
    "TFDPConfig",
    "TFDPForceMode",
    "TFDPInitMode",
    "TFDPInitialize",
    "TFDPIteration",
    "tfdp_pivot_mds",
    "tfdp_scale_by_edge",
]
