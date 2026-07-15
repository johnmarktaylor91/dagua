"""Sparse stress layout pipeline after Ortmann's Java implementation."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import ClassVar, List, Optional, Tuple, Union

import numpy as np
import torch

from dagua.layout.ops.base import Op, Pipeline
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_DEFAULT_PIVOTS = 50
_DEFAULT_MDS_PIVOTS = 200
_DEFAULT_STEPS = 200
_DEFAULT_FACTOR = 1.0
_KMEANS_MAX_ITER = 50
_OVERLAP_SEED = 100
_BREAK_CONDITION_INTERVAL = 10
_BREAK_CONDITION_RELATIVE_DELTA = 0.0001
_PIVOT_MDS_DIMENSIONS = 2
_PIVOT_MDS_FACTOR = -0.5
_JAVA_RANDOM_MULTIPLIER = 0x5DEECE66D
_JAVA_RANDOM_ADDEND = 0xB
_JAVA_RANDOM_MASK = (1 << 48) - 1
_JAVA_RANDOM_SEED_XOR = 0x5DEECE66D


@dataclass(frozen=True)
class SparseStressConfig:
    """Configuration for Ortmann sparse stress.

    Parameters
    ----------
    pivots : int, default=50
        Number of sparse stress pivots, capped to the node count.
    sampler : str, default="kmeans"
        Pivot sampler: ``"random"``, ``"maxmin"``, or ``"kmeans"``.
    steps : int, default=200
        Maximum number of majorization iterations.
    seed : int, default=0
        Java ``Random`` seed used by the sampler.
    factor : float, default=1.0
        Final coordinate multiplier.
    weighted : bool, default=False
        Whether edge weights represent desired graph distances.
    break_condition : bool, default=False
        Whether to stop after sparse stress converges.
    mds_pivots : int, default=200
        Pivot count used for the PivotMDS initialization.
    kmeans_features : int | None, default=None
        Number of max-min distance features used by k-means. ``None`` uses
        ``min(pivots, mds_pivots)`` as a deterministic production default.
    dtype : torch.dtype, default=torch.float32
        Output tensor dtype.
    """

    pivots: int = _DEFAULT_PIVOTS
    sampler: str = "kmeans"
    steps: int = _DEFAULT_STEPS
    seed: int = 0
    factor: float = _DEFAULT_FACTOR
    weighted: bool = False
    break_condition: bool = False
    mds_pivots: int = _DEFAULT_MDS_PIVOTS
    kmeans_features: Optional[int] = None
    dtype: torch.dtype = torch.float32


@dataclass
class _SparseStressGraph:
    """Undirected adjacency used by the reference-compatible port.

    Parameters
    ----------
    neighbors : list[list[int]]
        Neighbor ids for each node.
    weights : list[list[float]]
        Desired distances aligned to ``neighbors``.
    edge_count : int
        Number of undirected edges.
    """

    neighbors: List[List[int]]
    weights: List[List[float]]
    edge_count: int


@dataclass
class _SparseStressData:
    """Sparse stress term arrays grouped by updated node.

    Parameters
    ----------
    distances : list[list[float]]
        Desired graph distances for each per-node term.
    weights : list[list[float]]
        Normalized stress weights aligned to ``distances``.
    positions : list[list[int]]
        Vote-node indices aligned to ``distances``.
    pivots : list[int]
        Sorted pivot ids used to construct the sparse terms.
    """

    distances: List[List[float]]
    weights: List[List[float]]
    positions: List[List[int]]
    pivots: List[int]


class _JavaRandom:
    """Minimal port of ``java.util.Random`` for sampler fidelity."""

    def __init__(self, seed: int) -> None:
        """Initialize the Java-compatible random state.

        Parameters
        ----------
        seed : int
            Public Java ``Random`` seed.

        Returns
        -------
        None
            The instance stores the 48-bit internal state.
        """
        self._seed = (int(seed) ^ _JAVA_RANDOM_SEED_XOR) & _JAVA_RANDOM_MASK

    def _next(self, bits: int) -> int:
        """Return the next Java ``Random.next(bits)`` value.

        Parameters
        ----------
        bits : int
            Number of high bits to return.

        Returns
        -------
        int
            Non-negative integer containing ``bits`` random bits.
        """
        self._seed = (
            self._seed * _JAVA_RANDOM_MULTIPLIER + _JAVA_RANDOM_ADDEND
        ) & _JAVA_RANDOM_MASK
        return self._seed >> (48 - bits)

    def next_int(self, bound: int) -> int:
        """Return ``Random.nextInt(bound)``.

        Parameters
        ----------
        bound : int
            Exclusive positive upper bound.

        Returns
        -------
        int
            Uniform integer in ``[0, bound)``.
        """
        if bound <= 0:
            raise ValueError("bound must be positive.")
        if (bound & -bound) == bound:
            return int((bound * self._next(31)) >> 31)
        while True:
            bits = self._next(31)
            value = bits % bound
            if bits - value + (bound - 1) >= 0:
                return value

    def next_double(self) -> float:
        """Return ``Random.nextDouble()``.

        Returns
        -------
        float
            Double in ``[0, 1)`` with Java's 26/27-bit composition.
        """
        return float(((self._next(26) << 27) + self._next(27)) / float(1 << 53))


class _StableHeap:
    """Decrease-key heap matching the reference heap's pop order."""

    def __init__(self, size: int) -> None:
        """Create a heap with infinite initial values.

        Parameters
        ----------
        size : int
            Number of addressable elements.

        Returns
        -------
        None
            The heap stores values and lazy queue entries.
        """
        self.values = [math.inf for _ in range(size)]
        self._queue: List[Tuple[float, int]] = []

    def upsert(self, index: int, value: float) -> None:
        """Insert or decrease one element.

        Parameters
        ----------
        index : int
            Element id.
        value : float
            Candidate priority.

        Returns
        -------
        None
            The heap is updated when ``value`` improves the current priority.
        """
        if value >= self.values[index]:
            return
        self.values[index] = value
        heapq.heappush(self._queue, (value, index))

    def pop(self) -> int:
        """Pop the minimum-priority element.

        Returns
        -------
        int
            Element id with minimum current value.
        """
        while self._queue:
            value, index = heapq.heappop(self._queue)
            if value == self.values[index]:
                return index
        raise IndexError("pop from empty heap")

    def is_empty(self) -> bool:
        """Return whether the heap contains no live entries.

        Returns
        -------
        bool
            ``True`` when the heap is empty.
        """
        while self._queue and self._queue[0][0] != self.values[self._queue[0][1]]:
            heapq.heappop(self._queue)
        return not self._queue

    def value(self, index: int) -> float:
        """Return the current value for an element.

        Parameters
        ----------
        index : int
            Element id.

        Returns
        -------
        float
            Current priority value.
        """
        return self.values[index]


@register_op
class PrepareSparseStressGraph(Op):
    """Build reference-style undirected adjacency."""

    name: ClassVar[str] = "sparse_stress_prepare_graph"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def __init__(self, weighted: bool) -> None:
        """Store graph preparation options.

        Parameters
        ----------
        weighted : bool
            Whether to use ``problem.edge_weights`` as desired distances.

        Returns
        -------
        None
            The op stores the weighted flag.
        """
        self.weighted = weighted

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build adjacency lists and store them in ``state.extras``.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs with ``edge_index`` shape ``[2, E]``.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context, unused by this CPU reference port.

        Returns
        -------
        SolveState
            State with ``extras["sparse_stress_graph"]`` populated.
        """
        del ctx
        graph = _build_sparse_stress_graph(
            problem.edge_index,
            problem.num_nodes,
            problem.edge_weights,
            self.weighted,
        )
        state.extras["sparse_stress_graph"] = graph
        return state


@register_op
class InitializeSparseStressPositions(Op):
    """Compute Ortmann PivotMDS initialization and edge-length scaling."""

    name: ClassVar[str] = "sparse_stress_initialize"
    category: ClassVar[OpCategory] = OpCategory.INIT
    requires: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("pos", "extras")

    def __init__(self, mds_pivots: int) -> None:
        """Store initialization settings.

        Parameters
        ----------
        mds_pivots : int
            Maximum number of PivotMDS pivots.

        Returns
        -------
        None
            The op stores the pivot count.
        """
        self.mds_pivots = mds_pivots

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Initialize coordinates with the Java PivotMDS path.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state containing ``sparse_stress_graph``.
        ctx : RuntimeContext
            Runtime context, unused by this CPU reference port.

        Returns
        -------
        SolveState
            State with double-precision numpy positions stored in extras and
            torch positions in ``state.pos``.
        """
        del ctx
        graph = _require_sparse_graph(state)
        layout = _pivot_mds_layout(graph, max(1, self.mds_pivots))
        _scale_average_edge_length(graph, layout)
        state.extras["sparse_stress_layout_np"] = layout
        state.pos = torch.as_tensor(layout, dtype=torch.float64).reshape(problem.num_nodes, 2)
        return state


@register_op
class BuildSparseStressTerms(Op):
    """Sample pivots and build sparse stress aggregation terms."""

    name: ClassVar[str] = "sparse_stress_terms"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    requires: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def __init__(self, config: SparseStressConfig) -> None:
        """Store sparse term configuration.

        Parameters
        ----------
        config : SparseStressConfig
            Pipeline configuration.

        Returns
        -------
        None
            The op stores the config.
        """
        self.config = config

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build sparse stress data and store it in ``state.extras``.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state containing ``sparse_stress_graph``.
        ctx : RuntimeContext
            Runtime context, unused by this CPU reference port.

        Returns
        -------
        SolveState
            State with sparse terms in ``extras["sparse_stress_data"]``.
        """
        del problem, ctx
        graph = _require_sparse_graph(state)
        data = _build_sparse_stress_data(graph, self.config)
        state.extras["sparse_stress_data"] = data
        state.extras["sparse_stress_pivots"] = list(data.pivots)
        return state


@register_op
class RunSparseStressMajorization(Op):
    """Run sequential in-place sparse stress majorization."""

    name: ClassVar[str] = "sparse_stress_majorization"
    category: ClassVar[OpCategory] = OpCategory.OPTIMIZE
    requires: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("pos", "extras")

    def __init__(self, config: SparseStressConfig) -> None:
        """Store majorization settings.

        Parameters
        ----------
        config : SparseStressConfig
            Pipeline configuration.

        Returns
        -------
        None
            The op stores the config.
        """
        self.config = config

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run majorization and finalize torch positions.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state containing initialization and sparse terms.
        ctx : RuntimeContext
            Runtime context, unused by this CPU reference port.

        Returns
        -------
        SolveState
            State with final positions in ``state.pos``.
        """
        del problem, ctx
        graph = _require_sparse_graph(state)
        stress_data = _require_sparse_data(state)
        layout = np.asarray(state.extras["sparse_stress_layout_np"], dtype=np.float64)
        iterations = _run_majorization(
            graph,
            layout,
            stress_data,
            self.config.steps,
            self.config.break_condition,
        )
        layout *= float(self.config.factor)
        state.extras["sparse_stress_iterations"] = iterations
        state.extras["sparse_stress_layout_np"] = layout
        state.pos = torch.as_tensor(layout, dtype=self.config.dtype).reshape(graph_size(graph), 2)
        return state


def _require_sparse_graph(state: SolveState) -> _SparseStressGraph:
    """Return the prepared sparse stress graph from state.

    Parameters
    ----------
    state : SolveState
        Mutable solve state.

    Returns
    -------
    _SparseStressGraph
        Prepared graph adjacency.
    """
    graph = state.extras.get("sparse_stress_graph")
    if not isinstance(graph, _SparseStressGraph):
        raise RuntimeError("sparse stress graph has not been prepared.")
    return graph


def _require_sparse_data(state: SolveState) -> _SparseStressData:
    """Return sparse stress terms from state.

    Parameters
    ----------
    state : SolveState
        Mutable solve state.

    Returns
    -------
    _SparseStressData
        Sparse term data.
    """
    data = state.extras.get("sparse_stress_data")
    if not isinstance(data, _SparseStressData):
        raise RuntimeError("sparse stress terms have not been prepared.")
    return data


def graph_size(graph: _SparseStressGraph) -> int:
    """Return the number of nodes in a sparse stress graph.

    Parameters
    ----------
    graph : _SparseStressGraph
        Prepared adjacency.

    Returns
    -------
    int
        Node count.
    """
    return len(graph.neighbors)


def _build_sparse_stress_graph(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor],
    weighted: bool,
) -> _SparseStressGraph:
    """Build simple undirected adjacency lists.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    edge_weights : torch.Tensor or None
        Optional desired edge distances with shape ``[E]``.
    weighted : bool
        Whether to read ``edge_weights``.

    Returns
    -------
    _SparseStressGraph
        Undirected graph with one stored weight per adjacency entry.
    """
    edge_map: dict[Tuple[int, int], float] = {}
    edge_array = edge_index.detach().cpu().long()
    weights = None if edge_weights is None else edge_weights.detach().cpu().to(dtype=torch.float64)
    for edge_pos in range(int(edge_array.shape[1])):
        source = int(edge_array[0, edge_pos].item())
        target = int(edge_array[1, edge_pos].item())
        if (
            source == target
            or source < 0
            or target < 0
            or source >= num_nodes
            or target >= num_nodes
        ):
            continue
        weight = float(weights[edge_pos].item()) if weighted and weights is not None else 1.0
        if weight <= 0.0 or not math.isfinite(weight):
            weight = 1.0
        key = (source, target) if source < target else (target, source)
        edge_map[key] = min(weight, edge_map.get(key, math.inf))
    neighbors = [[] for _ in range(num_nodes)]
    graph_weights = [[] for _ in range(num_nodes)]
    for (source, target), weight in sorted(edge_map.items()):
        neighbors[source].append(target)
        graph_weights[source].append(weight)
        neighbors[target].append(source)
        graph_weights[target].append(weight)
    return _SparseStressGraph(neighbors=neighbors, weights=graph_weights, edge_count=len(edge_map))


def _single_source_distances(graph: _SparseStressGraph, source: int) -> List[float]:
    """Compute Dijkstra distances from one source.

    Parameters
    ----------
    graph : _SparseStressGraph
        Prepared graph.
    source : int
        Source node id.

    Returns
    -------
    list[float]
        Distances of length ``N``.
    """
    heap = _StableHeap(graph_size(graph))
    marked = [False for _ in range(graph_size(graph))]
    distances = [math.inf for _ in range(graph_size(graph))]
    heap.upsert(source, 0.0)
    while not heap.is_empty():
        current = heap.pop()
        dist = heap.value(current)
        if marked[current]:
            continue
        distances[current] = dist
        marked[current] = True
        for neighbor, weight in zip(graph.neighbors[current], graph.weights[current]):
            if not marked[neighbor]:
                heap.upsert(neighbor, dist + weight)
    return distances


def _sample_random(num_pivots: int, node_count: int, seed: int) -> List[int]:
    """Sample pivots with the reference reservoir sampler.

    Parameters
    ----------
    num_pivots : int
        Number of pivots.
    node_count : int
        Number of nodes in the cluster.
    seed : int
        Java random seed.

    Returns
    -------
    list[int]
        Sampled pivot ids.
    """
    rng = _JavaRandom(seed)
    pivots = list(range(num_pivots))
    index = num_pivots
    while index < node_count:
        index += 1
        rand_val = rng.next_int(index)
        if rand_val < num_pivots:
            pivots[rand_val] = index - 1
    return pivots


def _maxmin_samples(
    graph: _SparseStressGraph,
    num_pivots: int,
    seed: int,
) -> Tuple[List[int], List[List[float]]]:
    """Sample pivots via max-min SSSP.

    Parameters
    ----------
    graph : _SparseStressGraph
        Prepared graph.
    num_pivots : int
        Number of pivots.
    seed : int
        Java random seed.

    Returns
    -------
    tuple[list[int], list[list[float]]]
        Pivot ids and node-major distance feature matrix ``[N][P]``.
    """
    node_count = graph_size(graph)
    rng = _JavaRandom(seed)
    pivot_index = rng.next_int(node_count)
    pivots: List[int] = []
    distance_matrix = [[0.0 for _ in range(num_pivots)] for _ in range(node_count)]
    min_distances = [math.inf for _ in range(node_count)]
    for pivot_position in range(num_pivots):
        pivots.append(pivot_index)
        distances = _single_source_distances(graph, pivot_index)
        min_distances[pivot_index] = 0.0
        for node_index in range(node_count):
            distance_matrix[node_index][pivot_position] = distances[node_index]
            min_distances[node_index] = min(min_distances[node_index], distances[node_index])
            if min_distances[node_index] > min_distances[pivot_index]:
                pivot_index = node_index
    return pivots, distance_matrix


def _feature_key(features: List[float]) -> Tuple[float, ...]:
    """Return an exact feature key for k-means duplicate checks.

    Parameters
    ----------
    features : list[float]
        Feature vector.

    Returns
    -------
    tuple[float, ...]
        Hashable exact feature tuple.
    """
    return tuple(features)


def _kmeans_initial_samples(
    num_pivots: int,
    features: List[List[float]],
    rng: _JavaRandom,
) -> List[int]:
    """Select k-means initial samples with reference reservoir semantics.

    Parameters
    ----------
    num_pivots : int
        Number of samples.
    features : list[list[float]]
        Node-major feature matrix.
    rng : _JavaRandom
        Java-compatible random stream.

    Returns
    -------
    list[int]
        Initial medoid ids.
    """
    size = len(features)
    pivots = [0 for _ in range(num_pivots)]
    seen: set[Tuple[float, ...]] = set()
    pos = 0
    processed = 0
    while processed < num_pivots and pos < size:
        key = _feature_key(features[pos])
        if key not in seen:
            pivots[processed] = pos
            processed += 1
            seen.add(key)
        pos += 1
    if processed < num_pivots:
        for node_features in features:
            for feature_index in range(len(node_features)):
                node_features[feature_index] += (rng.next_double() - 0.5) / 1000.0
        return _kmeans_initial_samples(num_pivots, features, rng)
    while pos < size:
        pos += 1
        rand_val = rng.next_int(pos)
        key = _feature_key(features[pos - 1])
        if rand_val < num_pivots and key not in seen:
            seen.remove(_feature_key(features[pivots[rand_val]]))
            seen.add(key)
            pivots[rand_val] = pos - 1
    return pivots


def _feature_distance(left: List[float], right: List[float]) -> float:
    """Return squared Euclidean distance between feature vectors.

    Parameters
    ----------
    left : list[float]
        First feature vector.
    right : list[float]
        Second feature vector.

    Returns
    -------
    float
        Squared distance.
    """
    return sum((left[index] - right[index]) ** 2 for index in range(len(left)))


def _sample_kmeans(
    graph: _SparseStressGraph,
    num_pivots: int,
    seed: int,
    kmeans_features: int,
) -> List[int]:
    """Sample pivots with Ortmann's k-means shortest-path sampler.

    Parameters
    ----------
    graph : _SparseStressGraph
        Prepared graph.
    num_pivots : int
        Number of requested pivots.
    seed : int
        Java random seed.
    kmeans_features : int
        Number of max-min SSSP features.

    Returns
    -------
    list[int]
        Sampled pivot ids.
    """
    _, features = _maxmin_samples(graph, min(kmeans_features, num_pivots), seed)
    rng = _JavaRandom(seed)
    rng.next_int(graph_size(graph))
    pivots = _kmeans_initial_samples(num_pivots, features, rng)
    assignments = [0 for _ in range(graph_size(graph))]
    feature_count = len(features[0]) if features else 0
    means = [[0.0 for _ in range(feature_count)] for _ in range(num_pivots)]
    changed = True
    repetitions = _KMEANS_MAX_ITER
    while repetitions > 0 and changed:
        repetitions -= 1
        for node_index, node_features in enumerate(features):
            min_dist = math.inf
            for pivot_pos, pivot in enumerate(pivots):
                dist = _feature_distance(node_features, features[pivot])
                if min_dist > dist:
                    min_dist = dist
                    assignments[node_index] = pivot_pos
        cluster_sizes = [0 for _ in range(num_pivots)]
        for node_index, node_features in enumerate(features):
            cluster = assignments[node_index]
            cluster_sizes[cluster] += 1
            for feature_index in range(feature_count):
                means[cluster][feature_index] += node_features[feature_index]
        for pivot_pos in range(num_pivots):
            size = cluster_sizes[pivot_pos]
            if size == 0:
                continue
            for feature_index in range(feature_count):
                means[pivot_pos][feature_index] /= size
        new_pivots = [0 for _ in range(num_pivots)]
        min_distances = [math.inf for _ in range(num_pivots)]
        for node_index, node_features in enumerate(features):
            cluster = assignments[node_index]
            dist = _feature_distance(node_features, means[cluster])
            if min_distances[cluster] > dist:
                min_distances[cluster] = dist
                new_pivots[cluster] = node_index
        changed = False
        for pivot_pos in range(num_pivots):
            changed = changed or pivots[pivot_pos] != new_pivots[pivot_pos]
            pivots[pivot_pos] = new_pivots[pivot_pos]
            for feature_index in range(feature_count):
                means[pivot_pos][feature_index] = 0.0
    return pivots


def _sample_pivots(graph: _SparseStressGraph, config: SparseStressConfig) -> List[int]:
    """Sample sparse stress pivots.

    Parameters
    ----------
    graph : _SparseStressGraph
        Prepared graph.
    config : SparseStressConfig
        Pipeline configuration.

    Returns
    -------
    list[int]
        Sampled pivot ids, not yet sorted.
    """
    node_count = graph_size(graph)
    num_pivots = min(max(1, config.pivots), node_count)
    sampler = config.sampler.lower()
    if sampler == "random":
        return _sample_random(num_pivots, node_count, config.seed)
    if sampler == "maxmin":
        pivots, _ = _maxmin_samples(graph, num_pivots, config.seed)
        return pivots
    if sampler == "kmeans":
        features = config.kmeans_features
        if features is None:
            features = min(num_pivots, max(1, config.mds_pivots))
        return _sample_kmeans(graph, num_pivots, config.seed, max(1, features))
    raise ValueError("sampler must be one of: random, maxmin, kmeans.")


def _stress_partitioning(
    graph: _SparseStressGraph,
    pivots: List[int],
    data: _SparseStressData,
    state: dict[str, object],
    current_distance_block: float,
    next_distance: float,
) -> None:
    """Process one multi-source shortest-path distance block.

    Parameters
    ----------
    graph : _SparseStressGraph
        Prepared graph.
    pivots : list[int]
        Sorted pivot ids.
    data : _SparseStressData
        Sparse terms being built.
    state : dict[str, object]
        Mutable MSSP arrays.
    current_distance_block : float
        Distance value for the completed block.
    next_distance : float
        Distance value of the next heap block.

    Returns
    -------
    None
        ``data`` and ``state`` are mutated.
    """
    node_count = graph_size(graph)
    cl_assignment = state["cl_assignment"]
    cl_size = state["cl_size"]
    sorted_distances = state["sorted_distances"]
    hanging_pointer = state["hanging_pointer"]
    block = state["block"]
    processed_block = state["processed_block"]
    pivot_neighbors = state["pivot_neighbors"]
    assert isinstance(cl_assignment, list)
    assert isinstance(cl_size, list)
    assert isinstance(sorted_distances, list)
    assert isinstance(hanging_pointer, list)
    assert isinstance(block, list)
    assert isinstance(processed_block, list)
    assert isinstance(pivot_neighbors, list)

    for index in block:
        pivot_index = index // node_count
        node_index = index - pivot_index * node_count
        if cl_assignment[node_index] < 0:
            cl_assignment[node_index] = pivot_index
            cl_size[pivot_index] += 1
            sorted_distances[pivot_index].append(current_distance_block)
        if cl_size[cl_assignment[node_index]] > cl_size[pivot_index] + 1:
            sorted_distances[cl_assignment[node_index]].pop()
            cl_size[cl_assignment[node_index]] -= 1
            sorted_distances[pivot_index].append(current_distance_block)
            cl_assignment[node_index] = pivot_index
            cl_size[pivot_index] += 1

    for index in processed_block:
        pivot_index = index // node_count
        node_index = index - pivot_index * node_count
        if (
            current_distance_block > 0.0
            and node_index not in pivot_neighbors[pivot_index]
            and math.isfinite(current_distance_block)
        ):
            data.distances[node_index].append(current_distance_block)
            data.weights[node_index].append(
                hanging_pointer[pivot_index] / (current_distance_block * current_distance_block)
            )
            data.positions[node_index].append(pivots[pivot_index])

    cutoff = next_distance / 2.0
    for pivot_index, distances in enumerate(sorted_distances):
        while (
            hanging_pointer[pivot_index] != len(distances)
            and distances[hanging_pointer[pivot_index]] <= cutoff
        ):
            hanging_pointer[pivot_index] += 1


def _multi_source_sparse_terms(graph: _SparseStressGraph, pivots: List[int]) -> _SparseStressData:
    """Build sparse pivot-to-node aggregation terms via Ortmann MSSP.

    Parameters
    ----------
    graph : _SparseStressGraph
        Prepared graph.
    pivots : list[int]
        Sorted pivot ids.

    Returns
    -------
    _SparseStressData
        Sparse pivot aggregation terms before neighbor terms are added.
    """
    node_count = graph_size(graph)
    pivot_count = len(pivots)
    data = _SparseStressData(
        distances=[[] for _ in range(node_count)],
        weights=[[] for _ in range(node_count)],
        positions=[[] for _ in range(node_count)],
        pivots=list(pivots),
    )
    heap = _StableHeap(pivot_count * node_count)
    marked = [False for _ in range(pivot_count * node_count)]
    for pivot_index, pivot in enumerate(pivots):
        heap.upsert(pivot_index * node_count + pivot, 0.0)
    state: dict[str, object] = {
        "hanging_pointer": [0 for _ in pivots],
        "cl_assignment": [-1 for _ in range(node_count)],
        "cl_size": [0 for _ in pivots],
        "sorted_distances": [[] for _ in pivots],
        "block": [],
        "processed_block": [],
        "pivot_neighbors": [set([pivot]).union(graph.neighbors[pivot]) for pivot in pivots],
    }
    current_distance_block = 0.0
    while not heap.is_empty():
        current_index = heap.pop()
        dist = heap.value(current_index)
        if current_distance_block != dist:
            _stress_partitioning(graph, pivots, data, state, current_distance_block, dist)
            state["block"] = []
            state["processed_block"] = []
            current_distance_block = dist
        processed_block = state["processed_block"]
        block = state["block"]
        assert isinstance(processed_block, list)
        assert isinstance(block, list)
        processed_block.append(current_index)
        pivot_index = current_index // node_count
        node_index = current_index - pivot_index * node_count
        cl_assignment = state["cl_assignment"]
        assert isinstance(cl_assignment, list)
        if cl_assignment[node_index] < 0:
            block.append(current_index)
        marked[current_index] = True
        for neighbor, weight in zip(graph.neighbors[node_index], graph.weights[node_index]):
            neighbor_index = current_index - node_index + neighbor
            if not marked[neighbor_index]:
                heap.upsert(neighbor_index, dist + weight)
    _stress_partitioning(graph, pivots, data, state, current_distance_block, current_distance_block)
    return data


def _add_neighbor_terms(graph: _SparseStressGraph, data: _SparseStressData) -> None:
    """Add exact neighbor terms to sparse stress data.

    Parameters
    ----------
    graph : _SparseStressGraph
        Prepared graph.
    data : _SparseStressData
        Sparse terms to mutate.

    Returns
    -------
    None
        Neighbor terms are appended in adjacency order.
    """
    for node_index in range(graph_size(graph)):
        for neighbor, weight in zip(graph.neighbors[node_index], graph.weights[node_index]):
            data.distances[node_index].append(weight)
            data.weights[node_index].append(1.0 / (weight * weight))
            data.positions[node_index].append(neighbor)


def _normalize_sparse_weights(data: _SparseStressData) -> None:
    """Normalize sparse stress weights per updated node.

    Parameters
    ----------
    data : _SparseStressData
        Sparse stress terms.

    Returns
    -------
    None
        Weights are divided by their per-node totals.
    """
    for weights in data.weights:
        total = sum(weights)
        if total == 0.0:
            continue
        for index in range(len(weights)):
            weights[index] /= total


def _build_sparse_stress_data(
    graph: _SparseStressGraph,
    config: SparseStressConfig,
) -> _SparseStressData:
    """Sample pivots and construct normalized sparse stress terms.

    Parameters
    ----------
    graph : _SparseStressGraph
        Prepared graph.
    config : SparseStressConfig
        Pipeline configuration.

    Returns
    -------
    _SparseStressData
        Sparse stress terms with normalized weights.
    """
    pivots = sorted(_sample_pivots(graph, config))
    data = _multi_source_sparse_terms(graph, pivots)
    _add_neighbor_terms(graph, data)
    _normalize_sparse_weights(data)
    return data


def _center_distance_matrix(distance_matrix: List[List[float]]) -> np.ndarray:
    """Center PivotMDS distances with Ortmann's formula.

    Parameters
    ----------
    distance_matrix : list[list[float]]
        Pivot-major shortest path distances with shape ``[P, N]``.

    Returns
    -------
    numpy.ndarray
        Centered matrix with shape ``[P, N]``.
    """
    matrix = np.asarray(distance_matrix, dtype=np.float64)
    number_of_pivots, component_size = matrix.shape
    normalization_factor = 0.0
    column_normalization = np.zeros(number_of_pivots, dtype=np.float64)
    for pivot_index in range(number_of_pivots):
        row_col_normalizer = 0.0
        for node_index in range(component_size):
            row_col_normalizer += float(matrix[pivot_index, node_index]) ** 2
        normalization_factor += row_col_normalizer
        column_normalization[pivot_index] = row_col_normalizer / component_size
    normalization_factor /= component_size * number_of_pivots
    for node_index in range(component_size):
        row_col_normalizer = 0.0
        for pivot_index in range(number_of_pivots):
            square = float(matrix[pivot_index, node_index]) ** 2
            matrix[pivot_index, node_index] = (
                square + normalization_factor - column_normalization[pivot_index]
            )
            row_col_normalizer += square
        row_col_normalizer /= number_of_pivots
        for pivot_index in range(number_of_pivots):
            matrix[pivot_index, node_index] = _PIVOT_MDS_FACTOR * (
                float(matrix[pivot_index, node_index]) - row_col_normalizer
            )
    return matrix


def _normalize_vector(vector: np.ndarray) -> float:
    """Normalize a vector in place.

    Parameters
    ----------
    vector : numpy.ndarray
        Mutable vector with shape ``[N]``.

    Returns
    -------
    float
        Vector norm before normalization.
    """
    norm = math.sqrt(float(np.dot(vector, vector)))
    if norm != 0.0:
        vector /= norm
    return norm


def _pivot_mds_layout(graph: _SparseStressGraph, number_of_pivots: int) -> np.ndarray:
    """Compute the Java reference PivotMDS initialization.

    Parameters
    ----------
    graph : _SparseStressGraph
        Prepared graph.
    number_of_pivots : int
        Maximum number of pivots.

    Returns
    -------
    numpy.ndarray
        Flat coordinate array with shape ``[2 * N]``.
    """
    node_count = graph_size(graph)
    if node_count == 0:
        return np.zeros(0, dtype=np.float64)
    pivot_count = min(number_of_pivots, node_count)
    pivot_index = node_count - 1
    min_distances = [math.inf for _ in range(node_count)]
    distance_matrix: List[List[float]] = []
    for _ in range(pivot_count):
        distances = _single_source_distances(graph, pivot_index)
        distance_matrix.append(distances)
        min_distances[pivot_index] = 0.0
        for node_index, distance in enumerate(distances):
            min_distances[node_index] = min(min_distances[node_index], distance)
            if min_distances[node_index] > min_distances[pivot_index]:
                pivot_index = node_index
    centered = _center_distance_matrix(distance_matrix)
    kernel = centered @ centered.T
    # The Java reference uses a custom power iteration here. NumPy's symmetric
    # eigensolver targets the same centered kernel and avoids non-termination
    # on tiny rank-deficient graphs in the Python port.
    evals_all, evecs_all = np.linalg.eigh(kernel)
    order = np.argsort(evals_all)[::-1]
    tmp = np.zeros((_PIVOT_MDS_DIMENSIONS, pivot_count), dtype=np.float64)
    evals = np.zeros(_PIVOT_MDS_DIMENSIONS, dtype=np.float64)
    for axis in range(min(_PIVOT_MDS_DIMENSIONS, pivot_count)):
        source_axis = int(order[axis])
        tmp[axis, :] = evecs_all[:, source_axis]
        evals[axis] = max(float(evals_all[source_axis]), 0.0)
    singular_values = np.sqrt(evals)
    coords = tmp @ centered
    for axis in range(_PIVOT_MDS_DIMENSIONS):
        _normalize_vector(coords[axis])
        coords[axis] *= math.sqrt(max(singular_values[axis], 0.0))
    layout = np.zeros(node_count * 2, dtype=np.float64)
    for node_index in range(node_count):
        layout[node_index * 2] = coords[0, node_index]
        layout[node_index * 2 + 1] = coords[1, node_index]
    return layout


def _euclidean_flat(layout: np.ndarray, source: int, target: int) -> float:
    """Compute Euclidean distance between two flat-layout nodes.

    Parameters
    ----------
    layout : numpy.ndarray
        Flat coordinate array with shape ``[2 * N]``.
    source : int
        Source node id.
    target : int
        Target node id.

    Returns
    -------
    float
        Euclidean distance, with NaN mapped to zero like the Java reference.
    """
    dx = float(layout[source * 2] - layout[target * 2])
    dy = float(layout[source * 2 + 1] - layout[target * 2 + 1])
    dist = math.sqrt(dx * dx + dy * dy)
    return 0.0 if math.isnan(dist) else dist


def _scale_average_edge_length(graph: _SparseStressGraph, layout: np.ndarray) -> None:
    """Scale initialization to average desired edge length and jitter overlaps.

    Parameters
    ----------
    graph : _SparseStressGraph
        Prepared graph.
    layout : numpy.ndarray
        Flat coordinate array with shape ``[2 * N]``.

    Returns
    -------
    None
        Coordinates are scaled and jittered in place.
    """
    if graph.edge_count == 0:
        return
    avg_dist = 0.0
    avg_cost = 0.0
    for node_index in range(graph_size(graph)):
        for neighbor, weight in zip(graph.neighbors[node_index], graph.weights[node_index]):
            if neighbor > node_index:
                avg_dist += _euclidean_flat(layout, node_index, neighbor) / graph.edge_count
                avg_cost += weight / graph.edge_count
    if avg_dist != 0.0:
        layout *= avg_cost / avg_dist
    rng = _JavaRandom(_OVERLAP_SEED)
    jitter_scale = avg_cost / 1000.0
    for index in range(layout.shape[0]):
        layout[index] += jitter_scale * (rng.next_double() - 0.5)


def _minimize_sparse_stress(layout: np.ndarray, data: _SparseStressData) -> None:
    """Run one sequential sparse stress majorization sweep.

    Parameters
    ----------
    layout : numpy.ndarray
        Flat coordinate array with shape ``[2 * N]``.
    data : _SparseStressData
        Sparse stress terms.

    Returns
    -------
    None
        Coordinates are updated in place.
    """
    for node_index in range(len(data.weights)):
        new_x = 0.0
        new_y = 0.0
        ref_x = float(layout[node_index * 2])
        ref_y = float(layout[node_index * 2 + 1])
        for weight, distance, vote_node in zip(
            data.weights[node_index],
            data.distances[node_index],
            data.positions[node_index],
        ):
            vote_x = float(layout[vote_node * 2])
            vote_y = float(layout[vote_node * 2 + 1])
            dx = ref_x - vote_x
            dy = ref_y - vote_y
            euclidean_distance = math.sqrt(dx * dx + dy * dy)
            if euclidean_distance != 0.0:
                ratio = distance / euclidean_distance
                new_x += weight * (vote_x + ratio * dx)
                new_y += weight * (vote_y + ratio * dy)
        layout[node_index * 2] = new_x
        layout[node_index * 2 + 1] = new_y


def _intermediate_sparse_stress(
    graph: _SparseStressGraph,
    layout: np.ndarray,
    data: _SparseStressData,
) -> float:
    """Compute the reference break-condition sparse stress.

    Parameters
    ----------
    graph : _SparseStressGraph
        Prepared graph.
    layout : numpy.ndarray
        Flat coordinate array with shape ``[2 * N]``.
    data : _SparseStressData
        Sparse stress terms.

    Returns
    -------
    float
        Sparse stress over non-neighbor pivot terms.
    """
    stress = 0.0
    for node_index, distances in enumerate(data.distances):
        end = len(distances) - len(graph.neighbors[node_index])
        for term_index in range(max(0, end)):
            distance = distances[term_index]
            addend = 0.0
            if distance > 0.0:
                addend = (
                    _euclidean_flat(layout, node_index, data.positions[node_index][term_index])
                    / distance
                    - 1.0
                )
            stress += addend * addend
    return stress


def _run_majorization(
    graph: _SparseStressGraph,
    layout: np.ndarray,
    data: _SparseStressData,
    steps: int,
    use_break_condition: bool,
) -> int:
    """Run reference sparse stress majorization.

    Parameters
    ----------
    graph : _SparseStressGraph
        Prepared graph.
    layout : numpy.ndarray
        Flat coordinate array with shape ``[2 * N]``.
    data : _SparseStressData
        Sparse stress terms.
    steps : int
        Maximum number of iterations.
    use_break_condition : bool
        Whether to enable the reference convergence check.

    Returns
    -------
    int
        Number of completed iterations.
    """
    time_to_break = _BREAK_CONDITION_INTERVAL
    prev_stress = 0.0
    completed = 0
    for iteration in range(1, steps + 1):
        _minimize_sparse_stress(layout, data)
        completed = iteration
        if use_break_condition:
            time_to_break -= 1
            if time_to_break == 1:
                prev_stress = _intermediate_sparse_stress(graph, layout, data)
            if time_to_break == 0:
                time_to_break = _BREAK_CONDITION_INTERVAL
                current = _intermediate_sparse_stress(graph, layout, data)
                if (
                    prev_stress != 0.0
                    and (prev_stress - current) / prev_stress < _BREAK_CONDITION_RELATIVE_DELTA
                ):
                    break
    return completed


def build_sparse_stress_pipeline(config: Optional[SparseStressConfig] = None) -> Pipeline:
    """Build the sparse stress pipeline.

    Parameters
    ----------
    config : SparseStressConfig, optional
        Pipeline configuration. ``None`` uses reference-like defaults.

    Returns
    -------
    Pipeline
        Pipeline containing graph preparation, PivotMDS initialization,
        sparse-term construction, and majorization.
    """
    resolved = SparseStressConfig() if config is None else config
    if resolved.pivots <= 0:
        raise ValueError("pivots must be positive.")
    if resolved.steps <= 0:
        raise ValueError("steps must be positive.")
    if resolved.mds_pivots <= 0:
        raise ValueError("mds_pivots must be positive.")
    if resolved.factor <= 0.0:
        raise ValueError("factor must be positive.")
    return Pipeline(
        [
            PrepareSparseStressGraph(weighted=resolved.weighted),
            InitializeSparseStressPositions(mds_pivots=resolved.mds_pivots),
            BuildSparseStressTerms(config=resolved),
            RunSparseStressMajorization(config=resolved),
        ],
        name="sparse_stress_pipeline",
    )


def layout_sparse_stress_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    edge_weights: Optional[torch.Tensor] = None,
    pivots: int = _DEFAULT_PIVOTS,
    sampler: str = "kmeans",
    steps: int = _DEFAULT_STEPS,
    seed: Optional[int] = 0,
    factor: float = _DEFAULT_FACTOR,
    weighted: bool = False,
    break_condition: bool = False,
    mds_pivots: int = _DEFAULT_MDS_PIVOTS,
    kmeans_features: Optional[int] = None,
    dtype: Union[torch.dtype, str] = torch.float32,
) -> torch.Tensor:
    """Lay out a graph with sparse stress.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    edge_weights : torch.Tensor, optional
        Optional desired distances with shape ``[E]``.
    pivots : int, default=50
        Number of sparse stress pivots.
    sampler : str, default="kmeans"
        Pivot sampler: ``"random"``, ``"maxmin"``, or ``"kmeans"``.
    steps : int, default=200
        Maximum number of majorization iterations.
    seed : int or None, default=0
        Sampler seed. ``None`` uses the reference default ``0``.
    factor : float, default=1.0
        Final coordinate multiplier.
    weighted : bool, default=False
        Whether ``edge_weights`` are desired graph distances.
    break_condition : bool, default=False
        Whether to enable convergence stopping.
    mds_pivots : int, default=200
        PivotMDS initialization pivot count.
    kmeans_features : int, optional
        K-means feature count.
    dtype : torch.dtype or str, default=torch.float32
        Output dtype.

    Returns
    -------
    torch.Tensor
        Final positions with shape ``[N, 2]``.
    """
    resolved_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    config = SparseStressConfig(
        pivots=pivots,
        sampler=sampler,
        steps=steps,
        seed=0 if seed is None else int(seed),
        factor=factor,
        weighted=weighted,
        break_condition=break_condition,
        mds_pivots=mds_pivots,
        kmeans_features=kmeans_features,
        dtype=resolved_dtype,
    )
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weights=edge_weights,
        seed=config.seed,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device=str(edge_index.device)))
    final_state = build_sparse_stress_pipeline(config).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("sparse stress pipeline did not produce positions.")
    return final_state.pos.to(device=edge_index.device, dtype=resolved_dtype)


__all__ = [
    "BuildSparseStressTerms",
    "InitializeSparseStressPositions",
    "PrepareSparseStressGraph",
    "RunSparseStressMajorization",
    "SparseStressConfig",
    "build_sparse_stress_pipeline",
    "layout_sparse_stress_pipeline",
]
