"""Pivot MDS expressed as a composable ops pipeline."""

from __future__ import annotations

from typing import ClassVar, Optional, Tuple

import numpy as np
import torch

from dagua.layout.ops.graph_utils import (
    _shared_bfs_distances as bfs_distances,
)
from dagua.layout.ops.graph_utils import (
    _shared_dijkstra_distances as dijkstra_distances,
)
from dagua.layout.ops.graph_utils import (
    layout_device as _layout_device,
)
from dagua.layout.ops.graph_utils import (
    layout_extent as _layout_extent,
)

_MIN_SPAN = 1.0e-6


def _normalize_positions(positions: torch.Tensor, extent: float) -> torch.Tensor:
    """Center and scale coordinates into a stable bounding box.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    extent : float
        Target half-width.

    Returns
    -------
    torch.Tensor
        Normalized coordinates.
    """
    if positions.shape[0] <= 1:
        return torch.zeros_like(positions)

    centered = positions - positions.mean(dim=0, keepdim=True)
    span = float(centered.abs().max().item())
    if span < _MIN_SPAN:
        centered = centered.clone()
        centered[:, 0] = torch.linspace(
            -1.0, 1.0, steps=positions.shape[0], device=positions.device
        )
        span = float(centered.abs().max().item())
    return centered * (extent / max(span, _MIN_SPAN))


def _build_undirected_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor] = None,
) -> list[list[tuple[int, float]]]:
    """Build an undirected adjacency list.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    edge_weights : torch.Tensor, optional
        Optional per-edge weights with shape ``[E]``.

    Returns
    -------
    list[list[tuple[int, float]]]
        One neighbor list per node.
    """
    adjacency: list[dict[int, float]] = [{} for _ in range(num_nodes)]
    if edge_index.numel() == 0:
        return [[] for _ in range(num_nodes)]

    ei_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    sources = ei_cpu[0].tolist()
    targets = ei_cpu[1].tolist()

    if edge_weights is not None:
        weights = edge_weights.detach().cpu().float().tolist()
    else:
        weights = [1.0] * len(sources)

    for src, tgt, weight in zip(sources, targets, weights):
        if src == tgt:
            continue
        if tgt not in adjacency[src] or weight < adjacency[src][tgt]:
            adjacency[src][tgt] = weight
        if src not in adjacency[tgt] or weight < adjacency[tgt][src]:
            adjacency[tgt][src] = weight

    return [sorted(neighbors.items()) for neighbors in adjacency]


def _graph_distances(
    adjacency: list[list[tuple[int, float]]],
    start: int,
    weighted: bool,
) -> torch.Tensor:
    """Compute graph distances from one source.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Undirected adjacency list.
    start : int
        Source node index.
    weighted : bool
        Whether to use Dijkstra instead of BFS.

    Returns
    -------
    torch.Tensor
        Float distances with unreachable nodes replaced by ``max_distance + 1``.
    """
    if weighted:
        raw_distances = dijkstra_distances(adjacency, start)
        finite_mask = np.isfinite(raw_distances)
    else:
        raw_distances = bfs_distances(adjacency, start).astype(np.float64)
        finite_mask = raw_distances >= 0

    max_distance = float(raw_distances[finite_mask].max()) if bool(finite_mask.any()) else 0.0
    fill_value = max_distance + 1.0 if len(adjacency) > 1 else 0.0
    cleaned = np.where(finite_mask, raw_distances, fill_value)
    return torch.tensor(cleaned, dtype=torch.float32)


def _select_pivots(
    adjacency: list[list[tuple[int, float]]],
    n_pivots: int,
    seed: int,
    weighted: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select pivots with a deterministic max-min heuristic.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Undirected adjacency list.
    n_pivots : int
        Number of pivots to choose.
    seed : int
        Random seed for the first pivot.
    weighted : bool
        Whether to use Dijkstra instead of BFS.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Pivot indices ``[P]`` and pivot-to-node distance matrix ``[P, N]``.
    """
    num_nodes = len(adjacency)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    selected = torch.zeros(num_nodes, dtype=torch.bool)
    pivot_indices: list[int] = []
    pivot_distances: list[torch.Tensor] = []

    first_pivot = int(torch.randint(0, num_nodes, (1,), generator=generator).item())
    pivot_indices.append(first_pivot)
    selected[first_pivot] = True

    min_distances = _graph_distances(adjacency, first_pivot, weighted=weighted)
    pivot_distances.append(min_distances)

    while len(pivot_indices) < n_pivots:
        candidate_scores = min_distances.masked_fill(selected, -1.0)
        next_pivot = int(torch.argmax(candidate_scores).item())
        if selected[next_pivot]:
            break
        selected[next_pivot] = True
        pivot_indices.append(next_pivot)
        distances = _graph_distances(adjacency, next_pivot, weighted=weighted)
        pivot_distances.append(distances)
        min_distances = torch.minimum(min_distances, distances)

    return torch.tensor(pivot_indices, dtype=torch.long), torch.stack(pivot_distances, dim=0)


def _pivot_mds_coordinates(distance_matrix: torch.Tensor) -> torch.Tensor:
    """Recover a 2D embedding from pivot distances with SVD.

    Parameters
    ----------
    distance_matrix : torch.Tensor
        Pivot-to-node distance matrix with shape ``[P, N]``.

    Returns
    -------
    torch.Tensor
        Coordinate matrix with shape ``[N, 2]``.
    """
    squared = distance_matrix.square()
    row_means = squared.mean(dim=1, keepdim=True)
    col_means = squared.mean(dim=0, keepdim=True)
    grand_mean = squared.mean()
    centered = -0.5 * (squared - row_means - col_means + grand_mean)

    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    coord_dims = min(2, int(singular_values.shape[0]))
    if coord_dims == 0:
        return torch.zeros((distance_matrix.shape[1], 2), dtype=torch.float32)

    # Brandes-Pich 2007: for rectangular PxN pivot distance matrix C,
    # SVD gives C = U*S*V^T and coordinates are X = V[:,:k] * S[:k].
    # NOT sqrt(S) -- that's only for classical MDS on square NxN matrices.
    scales = singular_values[:coord_dims].clamp_min(0.0)
    coords = vh[:coord_dims].transpose(0, 1) * scales.unsqueeze(0)
    if coord_dims == 1:
        zeros = torch.zeros((coords.shape[0], 1), dtype=coords.dtype)
        coords = torch.cat((coords, zeros), dim=1)
    return coords.to(dtype=torch.float32)


from dagua.layout.ops.base import Op, Pipeline  # noqa: E402
from dagua.layout.ops.state import (  # noqa: E402
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)
from dagua.layout.ops.taxonomy import OpCategory  # noqa: E402


class _PreparePivotMDSState(Op):
    """Populate pivot-MDS adjacency and pivot-distance caches."""

    name: ClassVar[str] = "pivot_mds_prepare_state"
    category: ClassVar[OpCategory] = OpCategory.DISTANCE
    writes: ClassVar[Tuple[str, ...]] = ("pos", "pivot_indices", "pivot_distances")

    def __init__(self, n_pivots: int) -> None:
        """Store the pivot-selection budget.

        Parameters
        ----------
        n_pivots : int
            Maximum number of pivots to select.

        Returns
        -------
        None
            This constructor stores configuration only.
        """
        self.n_pivots = int(n_pivots)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Select pivots and cache their graph distances.

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
            State with pivot-MDS caches or trivial positions for graphs with
            fewer than two nodes.

        Raises
        ------
        ValueError
            If ``n_pivots`` is not positive.
        """
        del ctx

        if self.n_pivots <= 0:
            raise ValueError("n_pivots must be positive.")

        if problem.num_nodes == 0:
            state.pos = torch.empty((0, 2), dtype=torch.float32)
            state.pivot_indices = torch.empty((0,), dtype=torch.long)
            state.pivot_distances = torch.empty((0, 0), dtype=torch.float32)
            return state
        if problem.num_nodes == 1:
            state.pos = torch.zeros((1, 2), dtype=torch.float32)
            state.pivot_indices = torch.tensor([0], dtype=torch.long)
            state.pivot_distances = torch.zeros((1, 1), dtype=torch.float32)
            return state

        adjacency = _build_undirected_adjacency(
            edge_index=problem.edge_index,
            num_nodes=problem.num_nodes,
            edge_weights=problem.edge_weights,
        )
        pivot_indices, pivot_distances = _select_pivots(
            adjacency=adjacency,
            n_pivots=min(self.n_pivots, problem.num_nodes),
            seed=problem.seed,
            weighted=problem.edge_weights is not None,
        )
        state.pivot_indices = pivot_indices
        state.pivot_distances = pivot_distances
        return state


class _EmbedPivotMDS(Op):
    """Recover the raw 2D embedding from pivot distances."""

    name: ClassVar[str] = "pivot_mds_embed"
    category: ClassVar[OpCategory] = OpCategory.EMBED
    reads: ClassVar[Tuple[str, ...]] = ("pos", "pivot_distances")
    writes: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run the classic SVD embedding step.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state containing pivot distances.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this deterministic op.

        Returns
        -------
        SolveState
            State with raw pivot-MDS coordinates stored in ``state.pos``.

        Raises
        ------
        RuntimeError
            If the pivot-distance matrix is missing.
        """
        del problem, ctx

        if state.pos is not None:
            return state
        if state.pivot_distances is None:
            raise RuntimeError("_EmbedPivotMDS requires state.pivot_distances to be set.")

        state.pos = _pivot_mds_coordinates(state.pivot_distances)
        return state


class _FinalizePivotMDSPositions(Op):
    """Apply the classic Pivot-MDS centering and extent normalization."""

    name: ClassVar[str] = "pivot_mds_finalize_positions"
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
        """Normalize the embedding like classic ``layout_pivot_mds``.

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
            raise ValueError("_FinalizePivotMDSPositions requires state.pos to be set.")

        output_device = _layout_device(
            edge_index=problem.edge_index,
            node_sizes=problem.node_sizes,
        )
        extent = _layout_extent(num_nodes=problem.num_nodes, node_sizes=problem.node_sizes)
        normalized = _normalize_positions(state.pos.to(device=output_device), extent=extent)
        state.pos = normalized.to(dtype=torch.float32, device=output_device)
        return state


def build_pivot_mds_pipeline(n_pivots: int = 50) -> Pipeline:
    """Build a Pivot-MDS pipeline that matches classic ``layout_pivot_mds``.

    Parameters
    ----------
    n_pivots : int, default=50
        Maximum number of pivots to select.

    Returns
    -------
    Pipeline
        Pipeline that reproduces classic Pivot MDS exactly.

    Raises
    ------
    ValueError
        If ``n_pivots`` is not positive.
    """
    if n_pivots <= 0:
        raise ValueError("n_pivots must be positive.")

    return Pipeline(
        [
            _PreparePivotMDSState(n_pivots=n_pivots),
            _EmbedPivotMDS(),
            _FinalizePivotMDSPositions(),
        ],
        name="pivot_mds_pipeline",
    )


def layout_pivot_mds_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    n_pivots: int = 50,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the Pivot-MDS pipeline as a drop-in replacement.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor used to scale the final drawing extent.
    n_pivots : int, default=50
        Maximum number of pivots to select.
    seed : int, default=42
        Random seed for the first pivot.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Final position tensor with the same dtype, device, and values as
        classic ``layout_pivot_mds``.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``n_pivots``, or ``edge_weights`` are invalid.
    RuntimeError
        If the pipeline fails to populate final positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if n_pivots <= 0:
        raise ValueError("n_pivots must be positive.")
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
        seed=seed,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_pivot_mds_pipeline(n_pivots=n_pivots).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("Pivot-MDS pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_pivot_mds_pipeline", "layout_pivot_mds_pipeline"]
