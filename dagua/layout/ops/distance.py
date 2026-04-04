"""Distance-computation operations for composable layout pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from dagua.layout._archive.classic._graph_distances import (
    all_pairs_shortest_paths as _reference_all_pairs_shortest_paths,
)
from dagua.layout._archive.classic._graph_distances import bfs_distances as _reference_bfs_distances
from dagua.layout._archive.classic._graph_distances import (
    dijkstra_distances as _reference_dijkstra_distances,
)
from dagua.layout._archive.classic._graph_distances import is_connected as _reference_is_connected
from dagua.layout.ops.base import Op
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_ADJ_FORMAT_KEY = "adjacency_format"
_ADJ_WEIGHTED_KEY = "adjacency_weighted"
_DISTANCE_ROWS_KEY = "distance_rows"
_DISTANCE_SOURCES_KEY = "distance_sources"
_IS_CONNECTED_KEY = "is_connected"

AdjacencyList = List[List[Tuple[int, float]]]
CSRAdjacency = Dict[str, torch.Tensor]
AdjacencyValue = Union[AdjacencyList, CSRAdjacency, torch.Tensor]


def _listify_neighbors(neighbors: Sequence[Any]) -> List[Tuple[int, float]]:
    """Normalize one neighbor row into weighted tuple form.

    Parameters
    ----------
    neighbors : sequence[Any]
        Neighbor entries that may be integers or ``(neighbor, weight)`` tuples.

    Returns
    -------
    list[tuple[int, float]]
        Normalized weighted neighbors.
    """
    normalized: List[Tuple[int, float]] = []
    for entry in neighbors:
        if isinstance(entry, tuple):
            neighbor, weight = entry
            normalized.append((int(neighbor), float(weight)))
        else:
            normalized.append((int(entry), 1.0))
    return normalized


def _csr_to_list(adjacency: CSRAdjacency) -> AdjacencyList:
    """Convert a CSR-like payload into list adjacency form.

    Parameters
    ----------
    adjacency : dict[str, torch.Tensor]
        CSR payload with ``indptr``, ``indices``, and ``weights`` tensors.

    Returns
    -------
    list[list[tuple[int, float]]]
        Weighted adjacency list.
    """
    indptr = adjacency["indptr"].detach().to(device="cpu", dtype=torch.long)
    indices = adjacency["indices"].detach().to(device="cpu", dtype=torch.long)
    weights = adjacency["weights"].detach().to(device="cpu", dtype=torch.float64)

    rows: AdjacencyList = []
    for row_idx in range(max(int(indptr.numel()) - 1, 0)):
        start = int(indptr[row_idx].item())
        end = int(indptr[row_idx + 1].item())
        row = [
            (int(indices[offset].item()), float(weights[offset].item()))
            for offset in range(start, end)
        ]
        rows.append(row)
    return rows


def _dense_to_list(adjacency: torch.Tensor) -> AdjacencyList:
    """Convert a dense adjacency matrix into list form.

    Parameters
    ----------
    adjacency : torch.Tensor
        Dense cost matrix with shape ``[N, N]``.

    Returns
    -------
    list[list[tuple[int, float]]]
        Weighted adjacency list containing every finite off-diagonal edge.
    """
    dense = adjacency.detach().to(device="cpu", dtype=torch.float64)
    rows: AdjacencyList = []
    for source in range(int(dense.shape[0])):
        neighbors: List[Tuple[int, float]] = []
        for target in range(int(dense.shape[1])):
            if source == target:
                continue
            weight = float(dense[source, target].item())
            if np.isfinite(weight):
                neighbors.append((target, weight))
        rows.append(neighbors)
    return rows


def _adjacency_to_list(state: SolveState) -> AdjacencyList:
    """Resolve ``state.adjacency`` into the classic weighted-list format.

    Parameters
    ----------
    state : SolveState
        Mutable solve state.

    Returns
    -------
    list[list[tuple[int, float]]]
        Weighted adjacency list.

    Raises
    ------
    ValueError
        If adjacency is missing or has an unsupported representation.
    """
    adjacency = state.adjacency
    if adjacency is None:
        raise ValueError("Adjacency is required. Run BuildAdjacency before distance ops.")

    if isinstance(adjacency, list):
        return [_listify_neighbors(neighbors) for neighbors in adjacency]
    if isinstance(adjacency, dict):
        return _csr_to_list(adjacency)
    if isinstance(adjacency, torch.Tensor):
        return _dense_to_list(adjacency)
    raise ValueError(f"Unsupported adjacency type: {type(adjacency)!r}.")


def _is_weighted(problem: LayoutProblem, state: SolveState, adjacency: AdjacencyList) -> bool:
    """Infer whether shortest-path computations should use edge weights.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout inputs.
    state : SolveState
        Mutable solve state.
    adjacency : list[list[tuple[int, float]]]
        Weighted adjacency list.

    Returns
    -------
    bool
        ``True`` when Dijkstra should be used.
    """
    weighted_flag = state.extras.get(_ADJ_WEIGHTED_KEY)
    if isinstance(weighted_flag, bool):
        return weighted_flag
    if problem.edge_weights is not None:
        return True
    return any(abs(weight - 1.0) > 1.0e-12 for row in adjacency for _, weight in row)


def _resolve_sources(state: SolveState, num_nodes: int) -> Optional[List[int]]:
    """Resolve optional source-node selection from ``state.extras``.

    Parameters
    ----------
    state : SolveState
        Mutable solve state.
    num_nodes : int
        Number of nodes in the graph.

    Returns
    -------
    list[int] | None
        Selected sources, or ``None`` for all nodes.

    Raises
    ------
    ValueError
        If a requested source is out of range.
    """
    raw_sources = state.extras.get(_DISTANCE_SOURCES_KEY)
    if raw_sources is None:
        return None

    if isinstance(raw_sources, int):
        sources = [raw_sources]
    elif isinstance(raw_sources, torch.Tensor):
        sources = raw_sources.detach().to(device="cpu", dtype=torch.long).tolist()
    else:
        sources = [int(source) for source in raw_sources]

    for source in sources:
        if source < 0 or source >= num_nodes:
            raise ValueError(f"Distance source {source} is out of range for {num_nodes} nodes.")
    return sources


def _convert_unreachable_values(raw: np.ndarray, fill_value: Union[int, float]) -> np.ndarray:
    """Replace raw unreachable sentinels with a configured value.

    Parameters
    ----------
    raw : numpy.ndarray
        Distance array from the reference helpers.
    fill_value : int | float
        Replacement value for unreachable entries.

    Returns
    -------
    numpy.ndarray
        Distance array with configured unreachable values.
    """
    result = raw.copy()
    if np.issubdtype(result.dtype, np.integer):
        result[result < 0] = int(fill_value)
        return result
    if np.isinf(fill_value):
        return result
    result[np.isinf(result)] = float(fill_value)
    return result


def _to_torch(raw: np.ndarray) -> torch.Tensor:
    """Convert a NumPy distance array into a torch tensor with matching precision.

    Parameters
    ----------
    raw : numpy.ndarray
        Distance array.

    Returns
    -------
    torch.Tensor
        Torch tensor using ``int32`` for BFS data and ``float64`` for weighted
        distances.
    """
    if np.issubdtype(raw.dtype, np.integer):
        return torch.from_numpy(raw.astype(np.int32, copy=False))
    return torch.from_numpy(raw.astype(np.float64, copy=False))


def _fill_all_pairs_unreachable(raw: np.ndarray, mode: str) -> np.ndarray:
    """Apply unreachable-pair filling to an all-pairs matrix.

    Parameters
    ----------
    raw : numpy.ndarray
        All-pairs matrix from the reference helper.
    mode : str
        Fill strategy: ``"reference"`` or ``"max_plus_1"``.

    Returns
    -------
    numpy.ndarray
        Filled distance matrix.

    Raises
    ------
    ValueError
        If the fill mode is unknown.
    """
    if mode == "reference":
        return raw
    if mode != "max_plus_1":
        raise ValueError(f"Unsupported unreachable_fill mode: {mode!r}.")

    filled = raw.astype(np.float64, copy=True)
    if filled.size == 0:
        return filled

    finite_mask = np.isfinite(raw) if np.issubdtype(raw.dtype, np.floating) else raw >= 0
    max_distance = float(filled[finite_mask].max()) if bool(finite_mask.any()) else 0.0
    replacement = max_distance + 1.0 if filled.shape[0] > 1 else 0.0
    filled[~finite_mask] = replacement
    np.fill_diagonal(filled, 0.0)
    return filled


def _graph_distances_for_pivot(
    adjacency: AdjacencyList,
    source: int,
    weighted: bool,
) -> torch.Tensor:
    """Compute one pivot's distance row with Pivot-MDS disconnected fills.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Weighted adjacency list.
    source : int
        Pivot node index.
    weighted : bool
        Whether to use Dijkstra instead of BFS.

    Returns
    -------
    torch.Tensor
        Distance row with shape ``[N]`` and ``max_distance + 1`` fill for
        disconnected nodes.
    """
    if weighted:
        raw = _reference_dijkstra_distances(adjacency, source)
        finite_mask = np.isfinite(raw)
    else:
        raw = _reference_bfs_distances(adjacency, source).astype(np.float64)
        finite_mask = raw >= 0

    max_distance = float(raw[finite_mask].max()) if bool(finite_mask.any()) else 0.0
    fill_value = max_distance + 1.0 if len(adjacency) > 1 else 0.0
    cleaned = np.where(finite_mask, raw, fill_value)
    return torch.tensor(cleaned, dtype=torch.float32)


def _compute_distance_rows(
    adjacency: AdjacencyList,
    sources: Optional[Sequence[int]],
    weighted: bool,
    unreachable: Union[int, float],
) -> Tuple[Optional[torch.Tensor], Dict[int, torch.Tensor]]:
    """Compute either a full matrix or sparse source rows.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Weighted adjacency list.
    sources : sequence[int] | None
        Optional subset of source nodes. ``None`` means all sources.
    weighted : bool
        Whether to use Dijkstra instead of BFS.
    unreachable : int | float
        Configured unreachable sentinel.

    Returns
    -------
    tuple[torch.Tensor | None, dict[int, torch.Tensor]]
        Full matrix when every source is requested, otherwise a per-source row
        mapping stored in ``state.extras``.
    """
    num_nodes = len(adjacency)
    if sources is None:
        if weighted:
            raw = np.full((num_nodes, num_nodes), np.inf, dtype=np.float64)
            for source in range(num_nodes):
                raw[source] = _reference_dijkstra_distances(adjacency, source)
        else:
            raw = np.full((num_nodes, num_nodes), -1, dtype=np.int32)
            for source in range(num_nodes):
                raw[source] = _reference_bfs_distances(adjacency, source)
        converted = _convert_unreachable_values(raw, fill_value=unreachable)
        return _to_torch(converted), {}

    rows: Dict[int, torch.Tensor] = {}
    for source in sources:
        if weighted:
            raw_row = _reference_dijkstra_distances(adjacency, source)
        else:
            raw_row = _reference_bfs_distances(adjacency, source)
        rows[int(source)] = _to_torch(_convert_unreachable_values(raw_row, fill_value=unreachable))
    return None, rows


@dataclass(frozen=True)
class BFSDistancesConfig:
    """Configuration for ``BFSDistances``.

    Parameters
    ----------
    unreachable : int, default=-1
        Sentinel value used for unreachable nodes.
    """

    unreachable: int = -1


@register_op
class BFSDistances(Op):
    """Compute unweighted graph distances via BFS."""

    name = "bfs_distances"
    category = OpCategory.DISTANCE
    reads = ("adjacency", "extras.distance_sources")
    writes = ("distance_matrix", "extras.distance_rows")
    requires = ("adjacency",)

    def __init__(self, config: Optional[BFSDistancesConfig] = None) -> None:
        """Initialize the BFS-distance op.

        Parameters
        ----------
        config : BFSDistancesConfig | None, optional
            Optional op configuration.
        """
        self.config = config or BFSDistancesConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run repeated BFS traversals.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution context. Unused for this deterministic op.

        Returns
        -------
        SolveState
            Updated state with either ``distance_matrix`` or per-source rows.
        """
        _ = problem
        _ = ctx
        adjacency = _adjacency_to_list(state)
        sources = _resolve_sources(state, num_nodes=len(adjacency))
        matrix, rows = _compute_distance_rows(
            adjacency=adjacency,
            sources=sources,
            weighted=False,
            unreachable=self.config.unreachable,
        )
        state.distance_matrix = matrix
        if rows:
            state.extras[_DISTANCE_ROWS_KEY] = rows
        else:
            state.extras.pop(_DISTANCE_ROWS_KEY, None)
        return state


@dataclass(frozen=True)
class DijkstraDistancesConfig:
    """Configuration for ``DijkstraDistances``.

    Parameters
    ----------
    unreachable : float, default=inf
        Sentinel value used for unreachable nodes.
    """

    unreachable: float = float("inf")


@register_op
class DijkstraDistances(Op):
    """Compute weighted graph distances via Dijkstra."""

    name = "dijkstra_distances"
    category = OpCategory.DISTANCE
    reads = ("adjacency", "extras.distance_sources")
    writes = ("distance_matrix", "extras.distance_rows")
    requires = ("adjacency",)

    def __init__(self, config: Optional[DijkstraDistancesConfig] = None) -> None:
        """Initialize the Dijkstra-distance op.

        Parameters
        ----------
        config : DijkstraDistancesConfig | None, optional
            Optional op configuration.
        """
        self.config = config or DijkstraDistancesConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run repeated Dijkstra traversals.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution context. Unused for this deterministic op.

        Returns
        -------
        SolveState
            Updated state with either ``distance_matrix`` or per-source rows.
        """
        _ = problem
        _ = ctx
        adjacency = _adjacency_to_list(state)
        sources = _resolve_sources(state, num_nodes=len(adjacency))
        matrix, rows = _compute_distance_rows(
            adjacency=adjacency,
            sources=sources,
            weighted=True,
            unreachable=self.config.unreachable,
        )
        state.distance_matrix = matrix
        if rows:
            state.extras[_DISTANCE_ROWS_KEY] = rows
        else:
            state.extras.pop(_DISTANCE_ROWS_KEY, None)
        return state


@dataclass(frozen=True)
class AllPairsShortestPathsConfig:
    """Configuration for ``AllPairsShortestPaths``.

    Parameters
    ----------
    method : str, default="auto"
        Distance method: ``"auto"``, ``"bfs"``, or ``"dijkstra"``.
    unreachable_fill : str, default="max_plus_1"
        Post-processing policy for unreachable node pairs. ``"reference"``
        keeps the raw classic sentinel values; ``"max_plus_1"`` replaces
        unreachable entries with ``max_finite_distance + 1``.
    """

    method: str = "auto"
    unreachable_fill: str = "max_plus_1"


@register_op
class AllPairsShortestPaths(Op):
    """Compute all-pairs shortest-path distances."""

    name = "all_pairs_shortest_paths"
    category = OpCategory.DISTANCE
    reads = ("adjacency", "extras.adjacency_weighted")
    writes = ("distance_matrix",)
    requires = ("adjacency",)

    def __init__(self, config: Optional[AllPairsShortestPathsConfig] = None) -> None:
        """Initialize the APSP op.

        Parameters
        ----------
        config : AllPairsShortestPathsConfig | None, optional
            Optional op configuration.
        """
        self.config = config or AllPairsShortestPathsConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute the full shortest-path matrix.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution context. Unused for this deterministic op.

        Returns
        -------
        SolveState
            State with ``distance_matrix`` populated.

        Raises
        ------
        ValueError
            If the configured method is unsupported.
        """
        _ = ctx
        adjacency = _adjacency_to_list(state)
        if self.config.method == "bfs":
            weighted = False
        elif self.config.method == "dijkstra":
            weighted = True
        elif self.config.method == "auto":
            weighted = _is_weighted(problem, state, adjacency)
        else:
            raise ValueError(f"Unsupported APSP method: {self.config.method!r}.")

        raw = _reference_all_pairs_shortest_paths(adjacency, weighted=weighted)
        filled = _fill_all_pairs_unreachable(raw, mode=self.config.unreachable_fill)
        state.distance_matrix = _to_torch(filled)
        return state


@dataclass(frozen=True)
class PivotSelectionConfig:
    """Configuration for ``PivotSelection``.

    Parameters
    ----------
    n_pivots : int, default=50
        Maximum number of pivots to choose.
    method : str, default="maxmin"
        Pivot-selection strategy. Only ``"maxmin"`` is currently supported,
        matching the reference Pivot-MDS heuristic.
    """

    n_pivots: int = 50
    method: str = "maxmin"


def _resolve_generator(ctx: RuntimeContext, problem: LayoutProblem) -> torch.Generator:
    """Resolve the RNG generator for pivot selection.

    Parameters
    ----------
    ctx : RuntimeContext
        Execution context.
    problem : LayoutProblem
        Immutable layout inputs.

    Returns
    -------
    torch.Generator
        CPU generator matching the reference torch backend.
    """
    if ctx.generator is not None:
        return ctx.generator
    generator = torch.Generator(device="cpu")
    generator.manual_seed(problem.seed)
    return generator


@register_op
class PivotSelection(Op):
    """Select max-min pivots for approximate distance queries.

    Notes
    -----
    Randomness uses ``torch.Generator`` exactly once for the first pivot:
    ``torch.randint(0, num_nodes, (1,), generator=generator)``. The generator
    comes from ``ctx.generator`` when provided, otherwise a private CPU
    generator seeded from ``problem.seed`` is used.
    """

    name = "pivot_selection"
    category = OpCategory.DISTANCE
    reads = ("adjacency", "extras.adjacency_weighted")
    writes = ("pivot_indices",)
    requires = ("adjacency",)

    def __init__(self, config: Optional[PivotSelectionConfig] = None) -> None:
        """Initialize the pivot-selection op.

        Parameters
        ----------
        config : PivotSelectionConfig | None, optional
            Optional op configuration.
        """
        self.config = config or PivotSelectionConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Select pivots using the reference max-min heuristic.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution context that supplies the RNG generator.

        Returns
        -------
        SolveState
            State with ``pivot_indices`` populated.

        Raises
        ------
        ValueError
            If the configuration is invalid.
        """
        adjacency = _adjacency_to_list(state)
        num_nodes = len(adjacency)
        if self.config.n_pivots <= 0:
            raise ValueError("n_pivots must be positive.")
        if self.config.method != "maxmin":
            raise ValueError(f"Unsupported pivot selection method: {self.config.method!r}.")
        if num_nodes == 0:
            state.pivot_indices = torch.empty((0,), dtype=torch.long)
            return state

        n_pivots = min(self.config.n_pivots, num_nodes)
        generator = _resolve_generator(ctx, problem)
        weighted = _is_weighted(problem, state, adjacency)

        selected = torch.zeros(num_nodes, dtype=torch.bool)
        pivot_indices: List[int] = []
        first_pivot = int(torch.randint(0, num_nodes, (1,), generator=generator).item())
        pivot_indices.append(first_pivot)
        selected[first_pivot] = True

        min_distances = _graph_distances_for_pivot(adjacency, first_pivot, weighted=weighted)
        while len(pivot_indices) < n_pivots:
            candidate_scores = min_distances.masked_fill(selected, -1.0)
            next_pivot = int(torch.argmax(candidate_scores).item())
            if bool(selected[next_pivot].item()):
                break
            selected[next_pivot] = True
            pivot_indices.append(next_pivot)
            distances = _graph_distances_for_pivot(adjacency, next_pivot, weighted=weighted)
            min_distances = torch.minimum(min_distances, distances)

        state.pivot_indices = torch.tensor(pivot_indices, dtype=torch.long)
        return state


@register_op
class PivotDistanceQueries(Op):
    """Run distance queries from the selected pivots."""

    name = "pivot_distance_queries"
    category = OpCategory.DISTANCE
    reads = ("pivot_indices", "adjacency", "extras.adjacency_weighted")
    writes = ("pivot_distances",)
    requires = ("pivot_indices", "adjacency")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute pivot-to-node distance rows.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution context. Unused for this deterministic op.

        Returns
        -------
        SolveState
            State with ``pivot_distances`` populated.
        """
        _ = ctx
        adjacency = _adjacency_to_list(state)
        num_nodes = len(adjacency)
        pivots = state.pivot_indices
        if pivots is None or int(pivots.numel()) == 0:
            state.pivot_distances = torch.empty((0, num_nodes), dtype=torch.float32)
            return state

        weighted = _is_weighted(problem, state, adjacency)
        rows = [
            _graph_distances_for_pivot(adjacency, int(pivot.item()), weighted=weighted)
            for pivot in pivots.detach().to(device="cpu", dtype=torch.long)
        ]
        state.pivot_distances = torch.stack(rows, dim=0)
        return state


@register_op
class ConnectivityCheck(Op):
    """Check whether the current adjacency is connected."""

    name = "connectivity_check"
    category = OpCategory.DISTANCE
    reads = ("adjacency",)
    writes = ("extras.is_connected",)
    requires = ("adjacency",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Store a connectivity flag into ``state.extras``.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution context. Unused for this deterministic op.

        Returns
        -------
        SolveState
            State with ``extras["is_connected"]`` updated.
        """
        _ = problem
        _ = ctx
        adjacency = _adjacency_to_list(state)
        state.extras[_IS_CONNECTED_KEY] = bool(_reference_is_connected(adjacency))
        return state
