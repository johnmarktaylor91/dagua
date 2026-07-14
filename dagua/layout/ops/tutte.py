"""Tutte barycentric embedding operations."""

from __future__ import annotations

import math
from typing import ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import torch

from dagua.layout.ops.base import Op
from dagua.layout.ops.graph_utils import build_undirected_adjacency
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_TUTTE_METADATA_KEY = "tutte"
_TWO_PI = 2.0 * math.pi
_MAX_EXACT_CYCLE_NODES = 32


def _canonical_cycle(cycle: Sequence[int]) -> Tuple[int, ...]:
    """Return a rotation- and reflection-stable cycle tuple.

    Parameters
    ----------
    cycle : sequence[int]
        Simple cycle node IDs without a repeated closing node.

    Returns
    -------
    tuple[int, ...]
        Canonical ordering starting at the smallest node.
    """
    values = list(cycle)
    if not values:
        return ()
    variants: List[Tuple[int, ...]] = []
    for ordered in (values, list(reversed(values))):
        start = ordered.index(min(ordered))
        rotated = ordered[start:] + ordered[:start]
        variants.append(tuple(rotated))
    return min(variants)


def _undirected_neighbors(edge_index: torch.Tensor, num_nodes: int) -> List[List[int]]:
    """Build a sorted unweighted undirected adjacency list.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    list[list[int]]
        Neighbor node IDs sorted ascending per row.
    """
    weighted = build_undirected_adjacency(edge_index=edge_index, num_nodes=num_nodes)
    rows: List[List[int]] = []
    for neighbors in weighted:
        rows.append(sorted({int(neighbor) for neighbor, _weight in neighbors}))
    return rows


def _find_longest_simple_cycle(adjacency: Sequence[Sequence[int]]) -> List[int]:
    """Find the deterministic longest simple cycle for small graphs.

    Parameters
    ----------
    adjacency : sequence[sequence[int]]
        Undirected adjacency list.

    Returns
    -------
    list[int]
        Boundary cycle node order, or an empty list when no cycle exists.
    """
    num_nodes = len(adjacency)
    if num_nodes > _MAX_EXACT_CYCLE_NODES:
        return _two_core_boundary(adjacency)

    neighbor_sets = [set(neighbors) for neighbors in adjacency]
    best: Tuple[int, ...] = ()
    for start in range(num_nodes):
        stack: List[Tuple[int, List[int], Set[int]]] = [(start, [start], {start})]
        while stack:
            node, path, seen = stack.pop()
            for neighbor in sorted(adjacency[node], reverse=True):
                if neighbor == start and len(path) >= 3:
                    candidate = _canonical_cycle(path)
                    if _is_chordless_cycle(candidate, neighbor_sets) and (
                        len(candidate) > len(best)
                        or (len(candidate) == len(best) and candidate < best)
                    ):
                        best = candidate
                    continue
                if neighbor <= start or neighbor in seen:
                    continue
                stack.append((neighbor, [*path, neighbor], {*seen, neighbor}))
    return list(best)


def _is_chordless_cycle(cycle: Sequence[int], adjacency: Sequence[Set[int]]) -> bool:
    """Return whether a cycle has no non-boundary chords.

    Parameters
    ----------
    cycle : sequence[int]
        Simple cycle node IDs in cyclic order.
    adjacency : sequence[set[int]]
        Undirected neighbor sets.

    Returns
    -------
    bool
        ``True`` when no two non-consecutive cycle nodes share an edge.
    """
    cycle_length = len(cycle)
    if cycle_length < 3:
        return False
    for left_index, left_node in enumerate(cycle):
        for right_index in range(left_index + 1, cycle_length):
            if (right_index - left_index) in (1, cycle_length - 1):
                continue
            if int(cycle[right_index]) in adjacency[int(left_node)]:
                return False
    return True


def _two_core_boundary(adjacency: Sequence[Sequence[int]]) -> List[int]:
    """Return a deterministic 2-core fallback boundary.

    Parameters
    ----------
    adjacency : sequence[sequence[int]]
        Undirected adjacency list.

    Returns
    -------
    list[int]
        Sorted core nodes, or an empty list when no cycle-bearing core exists.
    """
    num_nodes = len(adjacency)
    degree = [len(set(neighbors)) for neighbors in adjacency]
    removed = [False] * num_nodes
    queue = [node for node, value in enumerate(degree) if value < 2]
    cursor = 0
    while cursor < len(queue):
        node = queue[cursor]
        cursor += 1
        if removed[node]:
            continue
        removed[node] = True
        for neighbor in adjacency[node]:
            degree[neighbor] -= 1
            if not removed[neighbor] and degree[neighbor] < 2:
                queue.append(neighbor)
    return [node for node, is_removed in enumerate(removed) if not is_removed]


def _regular_polygon(node_ids: Sequence[int], radius: float) -> Dict[int, Tuple[float, float]]:
    """Place boundary nodes on a regular convex polygon.

    Parameters
    ----------
    node_ids : sequence[int]
        Boundary node IDs in cyclic order.
    radius : float
        Polygon radius.

    Returns
    -------
    dict[int, tuple[float, float]]
        Boundary coordinates keyed by node ID.
    """
    count = len(node_ids)
    if count == 0:
        return {}
    if count == 1:
        return {int(node_ids[0]): (0.0, 0.0)}
    return {
        int(node): (
            radius * math.cos(_TWO_PI * index / count),
            radius * math.sin(_TWO_PI * index / count),
        )
        for index, node in enumerate(node_ids)
    }


def _edge_weights(problem: LayoutProblem) -> List[Tuple[int, int, float]]:
    """Return undirected weighted edges in input order.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable graph inputs.

    Returns
    -------
    list[tuple[int, int, float]]
        Source, target, and positive edge weight triples.
    """
    if problem.edge_index.numel() == 0:
        return []
    edges = problem.edge_index.detach().to(device="cpu", dtype=torch.long).t().tolist()
    if problem.edge_weights is None:
        weights = [1.0] * len(edges)
    else:
        weights = problem.edge_weights.detach().to(device="cpu", dtype=torch.float64).tolist()
    return [
        (int(source), int(target), float(weight))
        for (source, target), weight in zip(edges, weights)
        if int(source) != int(target) and float(weight) > 0.0
    ]


def tutte_embedding(
    problem: LayoutProblem,
    radius: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Compute a Tutte barycentric embedding.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable graph inputs with ``edge_index`` shape ``[2, E]``.
    radius : float, default=1.0
        Radius of the fixed convex boundary polygon.

    Returns
    -------
    tuple[torch.Tensor, dict[str, object]]
        Position tensor with shape ``[N, 2]`` and diagnostic metadata.

    Raises
    ------
    ValueError
        If ``radius`` is not positive.
    """
    if radius <= 0.0:
        raise ValueError("radius must be positive.")
    num_nodes = int(problem.num_nodes)
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float64), {"boundary": [], "fallback": "empty"}

    adjacency = _undirected_neighbors(problem.edge_index, num_nodes)
    boundary = _find_longest_simple_cycle(adjacency)
    fallback: Optional[str] = None
    if len(boundary) < 3:
        boundary = list(range(num_nodes))
        fallback = "no peripheral cycle; all nodes fixed on convex polygon"

    boundary_set = set(boundary)
    boundary_coordinates = _regular_polygon(boundary, radius=radius)
    positions = torch.zeros((num_nodes, 2), dtype=torch.float64)
    for node, coordinate in boundary_coordinates.items():
        positions[node] = torch.tensor(coordinate, dtype=torch.float64)

    interior = [node for node in range(num_nodes) if node not in boundary_set]
    if interior:
        interior_index = {node: index for index, node in enumerate(interior)}
        lhs = torch.zeros((len(interior), len(interior)), dtype=torch.float64)
        rhs = torch.zeros((len(interior), 2), dtype=torch.float64)
        for source, target, weight in _edge_weights(problem):
            for row_node, col_node in ((source, target), (target, source)):
                if row_node not in interior_index:
                    continue
                row = interior_index[row_node]
                lhs[row, row] += weight
                if col_node in interior_index:
                    lhs[row, interior_index[col_node]] -= weight
                elif col_node in boundary_coordinates:
                    rhs[row] += weight * positions[col_node]

        try:
            positions[interior] = torch.linalg.solve(lhs, rhs)
        except RuntimeError:
            # Singular systems occur when an interior component is disconnected
            # from the fixed boundary. The least-squares solution is the
            # minimum-norm barycentric fallback and keeps the layout finite.
            positions[interior] = torch.linalg.lstsq(lhs, rhs).solution
            fallback = "singular interior system; least-squares barycenter solution"

    metadata: Dict[str, object] = {
        "boundary": boundary,
        "interior": interior,
        "fallback": fallback,
        "reference": "Tutte barycentric embedding with fixed convex boundary",
    }
    return positions, metadata


@register_op
class TutteBarycentricEmbedding(Op):
    """Solve the fixed-boundary Tutte linear system."""

    name: ClassVar[str] = "tutte_barycentric_embedding"
    category: ClassVar[OpCategory] = OpCategory.EMBED
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = ("pos", "extras.tutte")

    def __init__(self, radius: float = 1.0) -> None:
        """Initialize the Tutte embedding op.

        Parameters
        ----------
        radius : float, default=1.0
            Radius of the fixed convex boundary polygon.
        """
        self.radius = float(radius)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run the Tutte solve and store the resulting coordinates.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context. Unused by this deterministic CPU solve.

        Returns
        -------
        SolveState
            State with ``pos`` and ``extras["tutte"]`` populated.
        """
        del ctx
        positions, metadata = tutte_embedding(problem=problem, radius=self.radius)
        state.pos = positions
        state.extras[_TUTTE_METADATA_KEY] = metadata
        return state


__all__ = ["TutteBarycentricEmbedding", "tutte_embedding"]
