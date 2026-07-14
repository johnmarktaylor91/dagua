"""Shared LargeVis/DRGraph dimensionality-reduction layout operations."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import ClassVar, Optional

import numpy as np
import torch

from dagua.layout.ops.base import Op
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_FLT_MIN = float(np.finfo(np.float32).tiny)
_PERPLEXITY_TOLERANCE = 1.0e-5
_PERPLEXITY_STEPS = 200
_GRADIENT_CLIP_LARGEVIS = 5.0
_GRADIENT_CLIP_DRGRAPH = 1.0
_NEGATIVE_TABLE_SIZE_CAP = 100_000
_REFERENCE_SEED = 314159265
_DEFAULT_PERPLEXITY = 50.0
_DEFAULT_LARGEVIS_GAMMA = 7.0
_DEFAULT_DRGRAPH_GAMMA = 0.01
_DEFAULT_ALPHA = 1.0


@dataclass(frozen=True)
class LargeVisGraph:
    """Directed weighted graph consumed by LargeVis-style SGD.

    Parameters
    ----------
    source : numpy.ndarray
        Directed edge sources with shape ``[E]``.
    target : numpy.ndarray
        Directed edge targets with shape ``[E]``.
    weight : numpy.ndarray
        Directed edge weights with shape ``[E]``.
    num_nodes : int
        Number of graph nodes.
    """

    source: np.ndarray
    target: np.ndarray
    weight: np.ndarray
    num_nodes: int


@dataclass(frozen=True)
class LargeVisConfig:
    """Parameter bundle for LargeVis-style embedding.

    Parameters
    ----------
    n_neighbors : int, default=150
        Number of geodesic neighbors used to build the sparse similarity graph.
    samples : int or None, default=None
        Number of positive-edge SGD samples. ``None`` follows the source
        default after converting the million-sample CLI unit to a bounded
        in-process count.
    alpha : float, default=1.0
        Initial learning rate.
    negative_samples : int, default=5
        Number of negative samples per positive edge.
    gamma : float, default=7.0
        Negative-sample repulsion weight.
    perplexity : float, default=50.0
        Target perplexity for row-wise Gaussian similarity calibration.
    seed : int, default=314159265
        Random seed. The C++ source fixes GSL ``rand48`` to this value.
    dtype : torch.dtype, default=torch.float32
        Output dtype used for returned positions.
    """

    n_neighbors: int = 150
    samples: Optional[int] = None
    alpha: float = _DEFAULT_ALPHA
    negative_samples: int = 5
    gamma: float = _DEFAULT_LARGEVIS_GAMMA
    perplexity: float = _DEFAULT_PERPLEXITY
    seed: int = _REFERENCE_SEED
    dtype: torch.dtype = torch.float32


@dataclass(frozen=True)
class DRGraphConfig(LargeVisConfig):
    """Parameter bundle for DRGraph graph-layout mode.

    Parameters
    ----------
    a : float, default=-1.0
        DRGraph curve parameter A. Values ``<= 0`` select the source fallback
        LargeVis force law.
    b : float, default=-1.0
        DRGraph curve parameter B. Used only when ``a > 0``.
    multilevel : bool, default=True
        Retained for API fidelity. The native port runs a single-level pass for
        deterministic graph-sized layouts.
    """

    gamma: float = _DEFAULT_DRGRAPH_GAMMA
    a: float = -1.0
    b: float = -1.0
    multilevel: bool = True


def _as_int_edges(edge_index: torch.Tensor, num_nodes: int) -> list[tuple[int, int]]:
    """Convert an edge tensor into validated integer pairs.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of valid nodes.

    Returns
    -------
    list[tuple[int, int]]
        Valid non-self edge pairs.
    """
    if edge_index.numel() == 0:
        return []
    edges = edge_index.detach().to(device="cpu", dtype=torch.long)
    pairs: list[tuple[int, int]] = []
    for source, target in edges.t().tolist():
        src = int(source)
        tgt = int(target)
        if src == tgt or src < 0 or tgt < 0 or src >= num_nodes or tgt >= num_nodes:
            continue
        pairs.append((src, tgt))
    return pairs


def _build_adjacency(edge_index: torch.Tensor, num_nodes: int) -> list[list[int]]:
    """Build sorted undirected adjacency from graph edges.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    list[list[int]]
        Sorted adjacency list.
    """
    neighbors: list[set[int]] = [set() for _ in range(num_nodes)]
    for source, target in _as_int_edges(edge_index, num_nodes):
        neighbors[source].add(target)
        neighbors[target].add(source)
    return [sorted(row) for row in neighbors]


def _bfs_distances(adjacency: list[list[int]], source: int) -> np.ndarray:
    """Compute unweighted geodesic distances from one node.

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected adjacency list.
    source : int
        Source node index.

    Returns
    -------
    numpy.ndarray
        Distance vector with shape ``[N]``. Unreachable nodes are ``inf``.
    """
    distances = np.full(len(adjacency), np.inf, dtype=np.float32)
    distances[source] = 0.0
    queue: deque[int] = deque([source])
    while queue:
        node = queue.popleft()
        next_distance = distances[node] + 1.0
        for neighbor in adjacency[node]:
            if np.isfinite(distances[neighbor]):
                continue
            distances[neighbor] = next_distance
            queue.append(neighbor)
    return distances


def build_geodesic_knn_graph(
    edge_index: torch.Tensor,
    num_nodes: int,
    n_neighbors: int,
) -> LargeVisGraph:
    """Build a directed KNN graph from graph geodesics.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    n_neighbors : int
        Maximum neighbors retained per source.

    Returns
    -------
    LargeVisGraph
        Directed neighbor graph with geodesic distances as raw weights.
    """
    if num_nodes <= 0:
        empty = np.empty((0,), dtype=np.int64)
        return LargeVisGraph(empty, empty.copy(), empty.astype(np.float32), num_nodes)

    adjacency = _build_adjacency(edge_index, num_nodes)
    k = max(1, min(int(n_neighbors), max(num_nodes - 1, 1)))
    sources: list[int] = []
    targets: list[int] = []
    weights: list[float] = []
    for node in range(num_nodes):
        distances = _bfs_distances(adjacency, node)
        candidates = [
            (float(distance), target)
            for target, distance in enumerate(distances.tolist())
            if target != node and np.isfinite(distance)
        ]
        candidates.sort(key=lambda item: (item[0], item[1]))
        for distance, target in candidates[:k]:
            sources.append(node)
            targets.append(target)
            weights.append(distance)
    return LargeVisGraph(
        source=np.asarray(sources, dtype=np.int64),
        target=np.asarray(targets, dtype=np.int64),
        weight=np.asarray(weights, dtype=np.float32),
        num_nodes=num_nodes,
    )


def _perplexity_weights(distances: list[float], perplexity: float) -> list[float]:
    """Calibrate one row of distances to a target perplexity.

    Parameters
    ----------
    distances : list[float]
        Raw neighbor distances.
    perplexity : float
        Target perplexity.

    Returns
    -------
    list[float]
        Row-normalized weights.
    """
    if not distances:
        return []
    beta = 1.0
    lo_beta = -1.0
    hi_beta = -1.0
    target_entropy = float(np.log(perplexity))
    arr = np.asarray(distances, dtype=np.float32)
    for _ in range(_PERPLEXITY_STEPS):
        exp_values = np.exp(-beta * arr, dtype=np.float32)
        sum_weight = float(exp_values.sum()) + _FLT_MIN
        entropy = float((beta * arr * exp_values).sum() / sum_weight) + float(np.log(sum_weight))
        if abs(entropy - target_entropy) < _PERPLEXITY_TOLERANCE:
            break
        if entropy > target_entropy:
            lo_beta = beta
            beta = beta * 2.0 if hi_beta < 0.0 else (beta + hi_beta) / 2.0
        else:
            hi_beta = beta
            beta = beta / 2.0 if lo_beta < 0.0 else (lo_beta + beta) / 2.0
        beta = min(beta, float(np.finfo(np.float32).max))
    weights = np.exp(-beta * arr, dtype=np.float32)
    weights_sum = float(weights.sum()) + _FLT_MIN
    return (weights / weights_sum).astype(np.float32).tolist()


def symmetrize_largevis_similarity(graph: LargeVisGraph, perplexity: float) -> LargeVisGraph:
    """Compute source-compatible symmetric LargeVis similarities.

    Parameters
    ----------
    graph : LargeVisGraph
        Directed KNN graph with raw distances.
    perplexity : float
        Target perplexity for row-wise Gaussian weights.

    Returns
    -------
    LargeVisGraph
        Directed graph where every edge has a reverse edge and reciprocal
        weights are averaged as in the C++ reference.
    """
    rows: list[list[tuple[int, float]]] = [[] for _ in range(graph.num_nodes)]
    for source, target, distance in zip(
        graph.source.tolist(),
        graph.target.tolist(),
        graph.weight.tolist(),
    ):
        rows[int(source)].append((int(target), float(distance)))

    directed: dict[tuple[int, int], float] = {}
    for source, row in enumerate(rows):
        row.sort(key=lambda item: (item[1], item[0]))
        weights = _perplexity_weights([distance for _, distance in row], perplexity)
        for (target, _), weight in zip(row, weights):
            directed[(source, target)] = float(weight)

    undirected_pairs = {tuple(sorted(pair)) for pair in directed if pair[0] != pair[1]}
    sources: list[int] = []
    targets: list[int] = []
    weights: list[float] = []
    for left, right in sorted(undirected_pairs):
        value = 0.5 * (directed.get((left, right), 0.0) + directed.get((right, left), 0.0))
        if value <= 0.0:
            continue
        sources.extend([left, right])
        targets.extend([right, left])
        weights.extend([value, value])
    return LargeVisGraph(
        source=np.asarray(sources, dtype=np.int64),
        target=np.asarray(targets, dtype=np.int64),
        weight=np.asarray(weights, dtype=np.float32),
        num_nodes=graph.num_nodes,
    )


def drgraph_similarity_graph(edge_index: torch.Tensor, num_nodes: int) -> LargeVisGraph:
    """Build DRGraph graph-layout similarity weights from input topology.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    LargeVisGraph
        Directed symmetric graph with DRGraph's ``exp(-distance)`` weights.
    """
    adjacency = _build_adjacency(edge_index, num_nodes)
    pairs: set[tuple[int, int]] = set()
    for source, row in enumerate(adjacency):
        for target in row:
            if source != target:
                pairs.add((min(source, target), max(source, target)))

    raw_weight = float(np.exp(-1.0))
    sum_weight = 2.0 * raw_weight * max(len(pairs), 1)
    scaled = raw_weight * float(num_nodes) / sum_weight if pairs else 1.0
    sources: list[int] = []
    targets: list[int] = []
    weights: list[float] = []
    for left, right in sorted(pairs):
        sources.extend([left, right])
        targets.extend([right, left])
        weights.extend([scaled, scaled])
    return LargeVisGraph(
        source=np.asarray(sources, dtype=np.int64),
        target=np.asarray(targets, dtype=np.int64),
        weight=np.asarray(weights, dtype=np.float32),
        num_nodes=num_nodes,
    )


def _alias_table(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build the source alias table for weighted edge sampling.

    Parameters
    ----------
    weights : numpy.ndarray
        Non-negative sample weights with shape ``[E]``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Alias indices and probability thresholds, each with shape ``[E]``.
    """
    n_items = int(weights.size)
    if n_items == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)
    total = float(weights.sum())
    if total <= 0.0:
        norm_prob = np.ones(n_items, dtype=np.float64)
    else:
        norm_prob = weights.astype(np.float64) * n_items / total
    alias = np.zeros(n_items, dtype=np.int64)
    prob = np.ones(n_items, dtype=np.float32)
    small = [index for index in range(n_items - 1, -1, -1) if norm_prob[index] < 1.0]
    large = [index for index in range(n_items - 1, -1, -1) if norm_prob[index] >= 1.0]
    while small and large:
        cur_small = small.pop()
        cur_large = large.pop()
        prob[cur_small] = float(norm_prob[cur_small])
        alias[cur_small] = cur_large
        norm_prob[cur_large] = norm_prob[cur_large] + norm_prob[cur_small] - 1.0
        if norm_prob[cur_large] < 1.0:
            small.append(cur_large)
        else:
            large.append(cur_large)
    return alias, prob


def _sample_alias(alias: np.ndarray, prob: np.ndarray, rng: np.random.RandomState) -> int:
    """Sample one alias-table item using the reference two-uniform scheme.

    Parameters
    ----------
    alias : numpy.ndarray
        Alias index table with shape ``[E]``.
    prob : numpy.ndarray
        Probability threshold table with shape ``[E]``.
    rng : numpy.random.RandomState
        Deterministic random number generator.

    Returns
    -------
    int
        Sampled item index.
    """
    n_items = int(prob.size)
    if n_items == 0:
        return 0
    index = int((n_items - 0.1) * float(rng.random_sample()))
    return index if float(rng.random_sample()) <= float(prob[index]) else int(alias[index])


def _negative_table(graph: LargeVisGraph) -> np.ndarray:
    """Build the LargeVis/DRGraph degree^0.75 negative-sampling table.

    Parameters
    ----------
    graph : LargeVisGraph
        Weighted directed graph.

    Returns
    -------
    numpy.ndarray
        Node IDs sampled according to weighted out-degree to the ``0.75`` power.
    """
    if graph.num_nodes <= 0:
        return np.empty((0,), dtype=np.int64)
    weights = np.zeros(graph.num_nodes, dtype=np.float64)
    np.add.at(weights, graph.source, graph.weight.astype(np.float64))
    weights = np.power(weights, 0.75)
    total = float(weights.sum())
    if total <= 0.0:
        return np.arange(graph.num_nodes, dtype=np.int64)
    table_size = max(graph.num_nodes, min(_NEGATIVE_TABLE_SIZE_CAP, graph.num_nodes * 1024))
    table = np.empty(table_size, dtype=np.int64)
    cumulative = weights[0]
    node = 0
    for index in range(table_size):
        table[index] = node
        if index / float(table_size) > cumulative / total and node < graph.num_nodes - 1:
            node += 1
            cumulative += weights[node]
    return table


def _default_sample_count(num_nodes: int) -> int:
    """Return a bounded in-process equivalent of the source sample heuristic.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    int
        Positive-edge SGD samples.
    """
    return max(100, int(num_nodes) * 200)


def optimize_largevis_embedding(
    graph: LargeVisGraph,
    config: LargeVisConfig,
    *,
    drgraph_ab: Optional[tuple[float, float]] = None,
) -> np.ndarray:
    """Run the LargeVis/DRGraph negative-sampling SGD loop.

    Parameters
    ----------
    graph : LargeVisGraph
        Directed weighted similarity graph.
    config : LargeVisConfig
        Optimization parameters.
    drgraph_ab : tuple[float, float] or None, optional
        DRGraph A/B curve parameters. ``None`` uses the LargeVis force law.

    Returns
    -------
    numpy.ndarray
        Position array with shape ``[N, 2]``.
    """
    rng = np.random.RandomState(int(config.seed))
    positions = ((rng.random_sample((graph.num_nodes, 2)) - 0.5) / 2.0 * 0.0001).astype(np.float32)
    if graph.num_nodes == 0 or graph.weight.size == 0:
        return positions

    alias, prob = _alias_table(graph.weight)
    neg_table = _negative_table(graph)
    samples = (
        int(config.samples)
        if config.samples is not None
        else _default_sample_count(graph.num_nodes)
    )
    samples = max(samples, 0)
    alpha0 = float(config.alpha)
    gamma = float(config.gamma)
    negative_samples = max(int(config.negative_samples), 0)
    clip = _GRADIENT_CLIP_DRGRAPH if drgraph_ab is not None else _GRADIENT_CLIP_LARGEVIS
    a = drgraph_ab[0] if drgraph_ab is not None else -1.0
    b = drgraph_ab[1] if drgraph_ab is not None else -1.0

    for step in range(samples):
        cur_alpha = alpha0 * (1.0 - step / (samples + 1.0))
        cur_alpha = max(cur_alpha, alpha0 * 0.0001)
        edge_id = _sample_alias(alias, prob, rng)
        source = int(graph.source[edge_id])
        positive_target = int(graph.target[edge_id])
        cur = positions[source].copy()
        error = np.zeros(2, dtype=np.float32)
        for sample_id in range(negative_samples + 1):
            if sample_id == 0:
                target = positive_target
            else:
                table_index = int((len(neg_table) - 0.1) * float(rng.random_sample()))
                target = int(neg_table[table_index])
                if target == positive_target or target == source:
                    continue
            diff = cur - positions[target]
            squared_distance = float(np.dot(diff, diff))
            if a > 0.0 and b > 0.0:
                powered = float(np.power(max(squared_distance, 1.0e-12), b))
                if sample_id == 0:
                    gradient = (
                        -2.0 * a * b * float(np.power(max(squared_distance, 1.0e-12), b - 1.0))
                    )
                    gradient /= 1.0 + a * powered
                else:
                    gradient = 2.0 * gamma * b / (1.0 + a * powered) / (0.001 + squared_distance)
            elif sample_id == 0:
                gradient = -2.0 / (1.0 + squared_distance)
            else:
                gradient = 2.0 * gamma / (1.0 + squared_distance) / (0.1 + squared_distance)
            update = np.clip(gradient * diff, -clip, clip).astype(np.float32) * cur_alpha
            error += update
            positions[target] -= update
        positions[source] += error
    return positions


@register_op
@dataclass(frozen=True)
class LargeVisBuildSimilarity(Op):
    """Build the LargeVis geodesic KNN similarity graph."""

    n_neighbors: int = 150
    perplexity: float = _DEFAULT_PERPLEXITY

    name: ClassVar[str] = "largevis_build_similarity"
    category: ClassVar[OpCategory] = OpCategory.EMBED
    writes: ClassVar[tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Apply the similarity construction operation.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph problem.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime execution context.

        Returns
        -------
        SolveState
            State containing ``largevis_graph`` in ``extras``.
        """
        del ctx
        knn_graph = build_geodesic_knn_graph(
            problem.edge_index,
            problem.num_nodes,
            self.n_neighbors,
        )
        state.extras["largevis_graph"] = symmetrize_largevis_similarity(knn_graph, self.perplexity)
        return state


@register_op
@dataclass(frozen=True)
class DRGraphBuildSimilarity(Op):
    """Build the DRGraph graph-layout similarity graph."""

    name: ClassVar[str] = "drgraph_build_similarity"
    category: ClassVar[OpCategory] = OpCategory.EMBED
    writes: ClassVar[tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Apply the DRGraph similarity construction operation.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph problem.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime execution context.

        Returns
        -------
        SolveState
            State containing ``largevis_graph`` in ``extras``.
        """
        del ctx
        state.extras["largevis_graph"] = drgraph_similarity_graph(
            problem.edge_index,
            problem.num_nodes,
        )
        return state


@register_op
@dataclass(frozen=True)
class LargeVisOptimizeEmbedding(Op):
    """Optimize a LargeVis-style embedding from the prepared graph."""

    config: LargeVisConfig
    drgraph_ab: Optional[tuple[float, float]] = None

    name: ClassVar[str] = "largevis_optimize_embedding"
    category: ClassVar[OpCategory] = OpCategory.OPTIMIZE
    reads: ClassVar[tuple[str, ...]] = ("extras",)
    writes: ClassVar[tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Apply the sampled SGD embedding operation.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph problem.
        state : SolveState
            Mutable solve state containing ``largevis_graph``.
        ctx : RuntimeContext
            Runtime execution context.

        Returns
        -------
        SolveState
            State with optimized ``pos`` tensor.
        """
        graph = state.extras.get("largevis_graph")
        if not isinstance(graph, LargeVisGraph):
            graph = build_geodesic_knn_graph(
                problem.edge_index,
                problem.num_nodes,
                self.config.n_neighbors,
            )
            graph = symmetrize_largevis_similarity(graph, self.config.perplexity)
        positions = optimize_largevis_embedding(graph, self.config, drgraph_ab=self.drgraph_ab)
        state.pos = torch.tensor(
            positions,
            dtype=self.config.dtype,
            device=torch.device(ctx.plan.device),
        )
        return state


__all__ = [
    "DRGraphBuildSimilarity",
    "DRGraphConfig",
    "LargeVisBuildSimilarity",
    "LargeVisConfig",
    "LargeVisGraph",
    "LargeVisOptimizeEmbedding",
    "build_geodesic_knn_graph",
    "drgraph_similarity_graph",
    "optimize_largevis_embedding",
    "symmetrize_largevis_similarity",
]
