"""MulMent multilevel MaxEnt-Stress layout pipeline.

This module reimplements KaDraw's MulMent drawing algorithm
(``kadraw --preconfiguration fast``, the binary default) without invoking the
reference at runtime. Every stage was validated offline against an
instrumented KaDraw build:

* coarsening hierarchy (size-constrained label propagation, libstdc++
  ``std::unordered_map`` quotient-edge iteration order, cluster-factor decay),
* the two seeded RNG streams (``std::mt19937`` for label-propagation
  tie-breaks, glibc ``rand()`` for coordinate initialization and projection
  jitter),
* the coarsest-level exact MaxEnt optimizer and the ``faster_drawing``
  cluster-approximated refinement used during uncoarsening.

The reference computes coordinates in ``float`` (32-bit). Dagua computes in
``float64``, so per-stage agreement is bounded by the reference's float32
rounding (~1e-6 relative), not by any algorithmic difference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from dagua.layout.ops.base import Op, Pipeline
from dagua.layout.ops.graph_utils import layout_device
from dagua.layout.ops.state import (
    ExecutionPlan,
    HierarchyLevel,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)
from dagua.layout.ops.taxonomy import OpCategory, register_op

_DEFAULT_ALPHA = 1.0
_DEFAULT_MIN_ALPHA = 0.008
_DEFAULT_Q = 0.0
_DEFAULT_TOL = 1.0e-4
_DEFAULT_OUTER_ITERATIONS = 13
_DEFAULT_INNER_ITERATIONS = 2
_DEFAULT_MAX_LEVELS = 64
_KADRAW_LABEL_ITERATIONS = 5
_KADRAW_CLUSTER_COARSENING_FACTOR = 20.0
_KADRAW_SIZE_BASE = 2.0
_KADRAW_STOP_NODES = 2
_KADRAW_TWO_PI = 2.0 * 3.1415
_MT19937_N = 624
_MT19937_M = 397
_MT19937_MATRIX_A = 0x9908B0DF
_MT19937_UPPER_MASK = 0x80000000
_MT19937_LOWER_MASK = 0x7FFFFFFF
_UINT32_MAX = 0xFFFFFFFF
_GLIBC_RAND_MAX = 2147483647
_LIBSTDCXX_REHASH_PRIMES = [
    13,
    29,
    59,
    127,
    257,
    541,
    1109,
    2357,
    5087,
    10273,
    20753,
    42043,
    85229,
    172933,
    351061,
    712697,
    1447153,
    2938679,
    5967347,
    12117689,
    24607243,
    49969847,
]
_FASTER_DRAWING_LEVEL_LADDER = (
    (100, 3),
    (1_000, 5),
    (10_000, 6),
    (100_000, 7),
    (300_000, 8),
    (600_000, 9),
    (2_000_000, 10),
    (4_000_000, 11),
    (10_000_000, 12),
    (20_000_000, 15),
    (40_000_000, 17),
)
_FASTER_DRAWING_LEVEL_MAX = 20


@dataclass(frozen=True)
class MulMentConfig:
    """Configuration for KaDraw-style MulMent layout.

    Defaults mirror KaDraw's ``--preconfiguration fast``, which is the
    reference binary's default configuration.

    Parameters
    ----------
    steps : int
        MaxEnt outer iterations (KaDraw ``maxent_outer_iterations``). Each
        outer iteration runs ``inner_iterations + 1`` Jacobi sweeps and then
        decays alpha by 0.3. ``0`` skips optimization entirely.
    inner_iterations : int
        KaDraw ``maxent_inner_iterations``. The reference's inner loop is a
        ``do {} while (iterations-- > 0)``, so each outer iteration executes
        ``inner_iterations + 1`` sweeps unless the tolerance exit fires.
    alpha : float
        Initial MaxEnt repulsion weight.
    min_alpha : float
        Lower bound for the outer-loop alpha schedule.
    q : float
        KaDraw MaxEnt exponent. ``0`` gives inverse-square entropy forces;
        KaDraw's ``sgn`` macro maps ``q == 0`` to ``+1``, so repulsion stays
        active.
    tol : float
        Relative coordinate-change tolerance for each refinement stage.
    max_levels : int
        Safety cap on coarsening transitions. The reference is uncapped; the
        default is far above any depth reached in practice.
    fidelity_dtype : torch.dtype
        Output dtype. Internal computation always runs in ``float64``.
    """

    steps: int = _DEFAULT_OUTER_ITERATIONS
    inner_iterations: int = _DEFAULT_INNER_ITERATIONS
    alpha: float = _DEFAULT_ALPHA
    min_alpha: float = _DEFAULT_MIN_ALPHA
    q: float = _DEFAULT_Q
    tol: float = _DEFAULT_TOL
    max_levels: int = _DEFAULT_MAX_LEVELS
    fidelity_dtype: torch.dtype = torch.float32


class _KaDrawRng:
    """KaDraw-compatible ``std::mt19937`` stream for coarsening tie-breaks.

    Parameters
    ----------
    seed : int
        Seed passed to KaDraw's ``random_functions::setSeed``.
    """

    def __init__(self, seed: int) -> None:
        """Initialize the MT19937 state.

        Parameters
        ----------
        seed : int
            Seed value.

        Returns
        -------
        None
            The object stores mutable generator state.
        """
        self._state = [0] * _MT19937_N
        self._index = _MT19937_N
        self._state[0] = int(seed) & _UINT32_MAX
        for index in range(1, _MT19937_N):
            previous = self._state[index - 1]
            self._state[index] = (1812433253 * (previous ^ (previous >> 30)) + index) & _UINT32_MAX

    def _twist(self) -> None:
        """Regenerate the MT19937 state array.

        Returns
        -------
        None
            The internal state is advanced by one twist block.
        """
        for index in range(_MT19937_N):
            mixed = (self._state[index] & _MT19937_UPPER_MASK) | (
                self._state[(index + 1) % _MT19937_N] & _MT19937_LOWER_MASK
            )
            value = self._state[(index + _MT19937_M) % _MT19937_N] ^ (mixed >> 1)
            if mixed % 2:
                value ^= _MT19937_MATRIX_A
            self._state[index] = value & _UINT32_MAX
        self._index = 0

    def raw_uint32(self) -> int:
        """Return the next tempered 32-bit MT19937 value.

        Returns
        -------
        int
            Unsigned 32-bit random value.
        """
        if self._index >= _MT19937_N:
            self._twist()
        value = self._state[self._index]
        self._index += 1
        value ^= value >> 11
        value ^= (value << 7) & 0x9D2C5680
        value ^= (value << 15) & 0xEFC60000
        value ^= value >> 18
        return value & _UINT32_MAX

    def next_int(self, lower: int, upper: int) -> int:
        """Return libstdc++ ``uniform_int_distribution`` output.

        Parameters
        ----------
        lower : int
            Inclusive lower bound.
        upper : int
            Inclusive upper bound.

        Returns
        -------
        int
            Random integer in ``[lower, upper]``.
        """
        if upper < lower:
            raise ValueError("upper must be greater than or equal to lower.")
        urange = int(upper - lower)
        if urange == 0:
            return lower
        urngrange = _UINT32_MAX
        if urngrange > urange:
            scaling = urngrange // (urange + 1)
            past = (urange + 1) * scaling
            while True:
                value = self.raw_uint32()
                if value < past:
                    return lower + value // scaling
        while True:
            value = self.raw_uint32()
            if value <= urange:
                return lower + value

    def next_bool(self) -> bool:
        """Return KaDraw's ``random_functions::nextBool`` result.

        Returns
        -------
        bool
            Random boolean.
        """
        return bool(self.next_int(0, 1))


class _GlibcRand:
    """glibc ``rand()`` (TYPE_3 additive feedback) replica.

    KaDraw's ``random_functions::nextDouble`` draws from glibc's ``rand()``
    stream (seeded by ``srand``), not from the MT19937 stream. It feeds the
    coarsest-level coordinate initialization and the per-level projection
    jitter. Verified bit-exact against glibc for the fidelity seeds.

    Parameters
    ----------
    seed : int
        Seed passed to ``srand``.
    """

    def __init__(self, seed: int) -> None:
        """Initialize the additive feedback state like glibc ``srandom_r``.

        Parameters
        ----------
        seed : int
            Seed value. glibc maps ``0`` to ``1``.

        Returns
        -------
        None
            The object stores mutable generator state.
        """
        seed = int(seed) if int(seed) != 0 else 1
        state = [0] * 34
        state[0] = seed & _UINT32_MAX
        for index in range(1, 31):
            high, low = divmod(state[index - 1], 127773)
            word = 16807 * low - 2836 * high
            if word < 0:
                word += _GLIBC_RAND_MAX
            state[index] = word
        for index in range(31, 34):
            state[index] = state[index - 31]
        self._state = state
        self._index = 34
        for _ in range(310):
            self.rand()

    def rand(self) -> int:
        """Return the next ``rand()`` output in ``[0, 2^31 - 1]``.

        Returns
        -------
        int
            Next pseudo-random value.
        """
        state = self._state
        index = self._index
        value = (state[index - 31] + state[index - 3]) & _UINT32_MAX
        state.append(value)
        self._index += 1
        return value >> 1

    def next_double(self, lower: float, upper: float) -> float:
        """Return KaDraw's ``random_functions::nextDouble`` result.

        Parameters
        ----------
        lower : float
            Inclusive lower bound.
        upper : float
            Inclusive upper bound.

        Returns
        -------
        float
            Uniform double in ``[lower, upper]``.
        """
        return (self.rand() / float(_GLIBC_RAND_MAX)) * (upper - lower) + lower


def _float32(value: float) -> float:
    """Round a Python float through IEEE-754 binary32.

    Parameters
    ----------
    value : float
        Input value.

    Returns
    -------
    float
        The value after a float32 round trip, mirroring KaDraw's
        ``CoordType``/coordinate storage truncation.
    """
    return float(torch.tensor(value, dtype=torch.float32).item())


def _validate_config(config: MulMentConfig) -> None:
    """Validate MulMent configuration values.

    Parameters
    ----------
    config : MulMentConfig
        Configuration to validate.

    Returns
    -------
    None
        The function raises on invalid input.
    """
    if config.steps < 0:
        raise ValueError("steps must be non-negative.")
    if config.inner_iterations < 0:
        raise ValueError("inner_iterations must be non-negative.")
    if config.alpha < 0.0:
        raise ValueError("alpha must be non-negative.")
    if config.min_alpha < 0.0:
        raise ValueError("min_alpha must be non-negative.")
    if config.tol <= 0.0:
        raise ValueError("tol must be positive.")
    if config.max_levels < 0:
        raise ValueError("max_levels must be non-negative.")
    if config.fidelity_dtype not in (torch.float32, torch.float64):
        raise ValueError("fidelity_dtype must be torch.float32 or torch.float64.")


def _kadraw_node_order(adjacency: list[list[tuple[int, float]]]) -> list[int]:
    """Return KaDraw's default degree-ascending node order.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Symmetric weighted adjacency for the current level.

    Returns
    -------
    list[int]
        Node order reused across label-propagation passes.

    Notes
    -----
    KaDraw's default configuration selects ``DEGREE_NODEORDERING``.
    ``std::sort`` is not stable, but libstdc++ falls back to insertion sort
    below 16 elements and leaves equal degrees in ascending ID order for the
    deterministic ranges covered by the fidelity tests.
    """
    return sorted(range(len(adjacency)), key=lambda node: len(adjacency[node]))


def _weighted_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor] = None,
) -> list[list[tuple[int, float]]]:
    """Build KaDraw-style symmetric weighted adjacency.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    edge_weights : torch.Tensor, optional
        Optional per-edge weights with shape ``[E]``.

    Returns
    -------
    list[list[tuple[int, float]]]
        Undirected adjacency lists. Unweighted (finest-level) rows are sorted
        by target ID to mirror KaDraw's METIS reader on sorted input; weighted
        (quotient) rows preserve insertion order, which mirrors the quotient
        graph construction order.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")
    if edge_weights is not None and edge_weights.shape[0] != edge_index.shape[1]:
        raise ValueError("edge_weights length must match edge count.")

    rows: list[dict[int, float]] = [dict() for _ in range(num_nodes)]
    edge_index_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    weights_cpu = (
        torch.ones((edge_index_cpu.shape[1],), dtype=torch.float64)
        if edge_weights is None
        else edge_weights.detach().to(device="cpu", dtype=torch.float64)
    )
    for edge_id, (source, target) in enumerate(
        zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist())
    ):
        if source < 0 or source >= num_nodes or target < 0 or target >= num_nodes:
            raise ValueError("edge_index contains a node outside [0, num_nodes).")
        if source == target:
            continue
        weight = float(weights_cpu[edge_id].item())
        rows[source][target] = rows[source].get(target, 0.0) + weight
        rows[target][source] = rows[target].get(source, 0.0) + weight
    if edge_weights is None:
        return [sorted(row.items()) for row in rows]
    return [list(row.items()) for row in rows]


def _boundary_pair_hash(source: int, target: int, partition_count: int) -> int:
    """Return KaDraw's undirected quotient-edge hash.

    Parameters
    ----------
    source : int
        First quotient block.
    target : int
        Second quotient block.
    partition_count : int
        Number of quotient blocks.

    Returns
    -------
    int
        Hash value used by ``hash_boundary_pair`` in KaDraw.
    """
    lower = min(source, target)
    upper = max(source, target)
    return lower * partition_count + upper


def _next_libstdcxx_bucket_count(required_size: int) -> int:
    """Return libstdc++'s next rehash bucket count for small quotient maps.

    Parameters
    ----------
    required_size : int
        Number of elements after insertion.

    Returns
    -------
    int
        Prime bucket count used by libstdc++'s default rehash policy.
    """
    for prime in _LIBSTDCXX_REHASH_PRIMES:
        if prime >= required_size:
            return prime
    return max(required_size * 2 + 1, _LIBSTDCXX_REHASH_PRIMES[-1])


def _libstdcxx_unordered_boundary_order(
    inserted_pairs: list[tuple[int, int]],
    partition_count: int,
) -> list[tuple[int, int]]:
    """Replay KaDraw's ``std::unordered_map`` quotient edge iteration order.

    Parameters
    ----------
    inserted_pairs : list[tuple[int, int]]
        Directed cut-edge block pairs in the order seen by ``complete_boundary``.
    partition_count : int
        Number of quotient blocks.

    Returns
    -------
    list[tuple[int, int]]
        Unique undirected pairs in libstdc++ iteration order.
    """
    order: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    bucket_count = 0

    for source, target in inserted_pairs:
        pair = (source, target) if source < target else (target, source)
        if pair in seen:
            continue
        if bucket_count == 0 or len(seen) + 1 > bucket_count:
            bucket_count = _next_libstdcxx_bucket_count(len(seen) + 1)
            order.reverse()

        bucket = _boundary_pair_hash(pair[0], pair[1], partition_count) % bucket_count
        insert_at = 0
        for index, existing in enumerate(order):
            existing_bucket = (
                _boundary_pair_hash(existing[0], existing[1], partition_count) % bucket_count
            )
            if existing_bucket == bucket:
                insert_at = index
                break
        order.insert(insert_at, pair)
        seen.add(pair)
    return order


def _kadraw_label_propagation_mapping(
    adjacency: list[list[tuple[int, float]]],
    node_weights: list[int],
    block_upperbound: int,
    rng: _KaDrawRng,
) -> tuple[list[int], int, list[int]]:
    """Run KaDraw's size-constrained label propagation.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Symmetric weighted adjacency for the current level.
    node_weights : list[int]
        Node weights for the current level.
    block_upperbound : int
        Maximum cluster weight accepted by label propagation.
    rng : _KaDrawRng
        KaDraw-compatible random stream.

    Returns
    -------
    tuple[list[int], int, list[int]]
        Fine-to-coarse mapping, coarse node count, and node order.
    """
    num_nodes = len(adjacency)
    cluster_id = list(range(num_nodes))
    cluster_sizes = list(node_weights)
    permutation = _kadraw_node_order(adjacency)
    hash_map = [0.0] * num_nodes

    for _ in range(_KADRAW_LABEL_ITERATIONS):
        for node in permutation:
            for target, weight in adjacency[node]:
                hash_map[cluster_id[target]] += weight

            max_block = cluster_id[node]
            my_block = cluster_id[node]
            max_value = 0.0
            for target, _weight in adjacency[node]:
                cur_block = cluster_id[target]
                cur_value = hash_map[cur_block]
                can_fit = cluster_sizes[cur_block] + node_weights[node] <= block_upperbound
                if (cur_value > max_value or (cur_value == max_value and rng.next_bool())) and (
                    can_fit or cur_block == my_block
                ):
                    max_value = cur_value
                    max_block = cur_block
                hash_map[cur_block] = 0.0

            cluster_sizes[cluster_id[node]] -= node_weights[node]
            cluster_sizes[max_block] += node_weights[node]
            cluster_id[node] = max_block

    remap: dict[int, int] = {}
    fine_to_coarse: list[int] = []
    for cluster in cluster_id:
        if cluster not in remap:
            remap[cluster] = len(remap)
        fine_to_coarse.append(remap[cluster])
    return fine_to_coarse, len(remap), permutation


def _kadraw_contract(
    adjacency: list[list[tuple[int, float]]],
    fine_to_coarse: list[int],
    coarse_num_nodes: int,
    node_weights: list[int],
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Build KaDraw's clustering quotient graph.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Symmetric weighted fine-level adjacency.
    fine_to_coarse : list[int]
        Fine-to-coarse mapping.
    coarse_num_nodes : int
        Number of coarse nodes.
    node_weights : list[int]
        Fine-level node weights.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, list[int]]
        Coarse edge tensor ``[2, E]``, edge weights ``[E]``, and coarse node
        weights.
    """
    coarse_weights = [0] * coarse_num_nodes
    cut_weights: dict[tuple[int, int], float] = {}
    inserted_pairs: list[tuple[int, int]] = []
    for source, neighbors in enumerate(adjacency):
        source_block = fine_to_coarse[source]
        coarse_weights[source_block] += node_weights[source]
        for target, weight in neighbors:
            target_block = fine_to_coarse[target]
            if source_block == target_block:
                continue
            inserted_pairs.append((source_block, target_block))
            key = (
                (source_block, target_block)
                if source_block < target_block
                else (target_block, source_block)
            )
            cut_weights[key] = cut_weights.get(key, 0.0) + weight

    pairs = _libstdcxx_unordered_boundary_order(inserted_pairs, coarse_num_nodes)
    if not pairs:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
            coarse_weights,
        )

    weights = [cut_weights[pair] / 2.0 for pair in pairs]
    return (
        torch.tensor(pairs, dtype=torch.long).transpose(0, 1).contiguous(),
        torch.tensor(weights, dtype=torch.float32),
        coarse_weights,
    )


def _build_kadraw_hierarchy(
    problem: LayoutProblem,
    config: MulMentConfig,
) -> list[HierarchyLevel]:
    """Build KaDraw's label-propagation coarsening hierarchy.

    Parameters
    ----------
    problem : LayoutProblem
        Input graph problem.
    config : MulMentConfig
        MulMent configuration.

    Returns
    -------
    list[HierarchyLevel]
        Levels from finest transition to coarsest transition.

    Notes
    -----
    KaDraw computes the per-level block bound from
    ``upper_bound_partition = N - 1`` and decays the cluster coarsening factor
    when ``finer / coarser < 1.1`` evaluated with C++ *integer* division,
    both of which are mirrored here.
    """
    rng = _KaDrawRng(problem.seed)
    levels: list[HierarchyLevel] = []
    current_edge_index = problem.edge_index.detach().to(device="cpu", dtype=torch.long)
    current_edge_weights = (
        None
        if problem.edge_weights is None
        else problem.edge_weights.detach().to(device="cpu", dtype=torch.float32)
    )
    current_num_nodes = problem.num_nodes
    current_node_weights = [1] * problem.num_nodes
    upper_bound_partition = max(0, problem.num_nodes - 1)
    cluster_factor = _KADRAW_CLUSTER_COARSENING_FACTOR
    level_index = 0

    while level_index < config.max_levels:
        upper_bound = min(
            _KADRAW_SIZE_BASE ** float(level_index + 1),
            math.ceil(float(upper_bound_partition) / cluster_factor),
        )
        upper_bound = min(upper_bound, float(upper_bound_partition))
        block_upperbound = int(math.ceil(upper_bound))
        adjacency = _weighted_adjacency(
            current_edge_index,
            current_num_nodes,
            current_edge_weights,
        )
        fine_to_coarse_list, coarse_num_nodes, _permutation = _kadraw_label_propagation_mapping(
            adjacency,
            current_node_weights,
            block_upperbound,
            rng,
        )
        coarse_edges, coarse_weights, coarse_node_weights = _kadraw_contract(
            adjacency,
            fine_to_coarse_list,
            coarse_num_nodes,
            current_node_weights,
        )
        fine_to_coarse = torch.tensor(fine_to_coarse_list, dtype=torch.long)
        level = HierarchyLevel(
            num_nodes=coarse_num_nodes,
            num_fine=current_num_nodes,
            edge_index=coarse_edges,
            edge_weights=coarse_weights,
            node_masses=torch.tensor(coarse_node_weights, dtype=torch.float32),
            fine_to_coarse=fine_to_coarse,
            cluster_ids=fine_to_coarse.clone(),
        )
        levels.append(level)

        contraction_stop = coarse_num_nodes > _KADRAW_STOP_NODES and coarse_edges.numel() != 0
        if current_num_nodes // coarse_num_nodes < 1.1:
            cluster_factor *= 0.7
        if not contraction_stop:
            break

        current_edge_index = coarse_edges
        current_edge_weights = coarse_weights
        current_num_nodes = coarse_num_nodes
        current_node_weights = coarse_node_weights
        level_index += 1

    return levels


def _faster_drawing_num_levels(num_nodes: int) -> int:
    """Return KaDraw's ``faster_drawing_num_levels`` graph-size ladder.

    Parameters
    ----------
    num_nodes : int
        Node count of the input graph.

    Returns
    -------
    int
        Number of hierarchy levels skipped by the approximate repulsion.
    """
    for threshold, levels in _FASTER_DRAWING_LEVEL_LADDER:
        if num_nodes < threshold:
            return levels
    return _FASTER_DRAWING_LEVEL_MAX


def _directed_edges_with_distances(
    adjacency: list[list[tuple[int, float]]],
    node_weights: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build directed edge arrays and KaDraw's per-edge desired distances.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Symmetric weighted adjacency for one level.
    node_weights : list[int]
        Node weights for the level.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Source indices ``[E]``, target indices ``[E]``, and desired distances
        ``[E]`` mirroring ``local_optimizer::configure_distances`` with unit
        inter/intra-cluster factors:
        ``(sqrt(w_u) + sqrt(w_v)) / 2`` rounded through float32.
    """
    sources: list[int] = []
    targets: list[int] = []
    distances: list[float] = []
    for node, neighbors in enumerate(adjacency):
        for target, _weight in neighbors:
            sources.append(node)
            targets.append(target)
            distances.append(
                _float32((math.sqrt(node_weights[node]) + math.sqrt(node_weights[target])) / 2.0)
            )
    return (
        torch.tensor(sources, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
        torch.tensor(distances, dtype=torch.float64),
    )


def _stress_and_edge_terms(
    pos: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    lengths: torch.Tensor,
    rho: torch.Tensor,
    q: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the per-node stress centers and edge repulsion corrections.

    Parameters
    ----------
    pos : torch.Tensor
        Current positions ``[N, 2]``.
    src : torch.Tensor
        Directed edge sources ``[E]``.
    dst : torch.Tensor
        Directed edge targets ``[E]``.
    lengths : torch.Tensor
        Desired edge lengths ``[E]``.
    rho : torch.Tensor
        Per-node stress normalization ``[N]``.
    q : float
        MaxEnt exponent.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Stress centers ``[N, 2]`` and summed neighbor repulsion terms
        ``[N, 2]`` (subtracted from the entropy force like the reference).
    """
    diff = pos[src] - pos[dst]
    dist = torch.linalg.norm(diff, dim=1)
    inv_sq = torch.reciprocal(lengths * lengths)
    stress_terms = (pos[dst] + (lengths / dist).unsqueeze(1) * diff) * inv_sq.unsqueeze(1)
    stress = torch.zeros_like(pos).index_add_(0, src, stress_terms) * rho.unsqueeze(1)
    edge_rep_terms = diff / dist.pow(q + 2.0).unsqueeze(1)
    edge_rep = torch.zeros_like(pos).index_add_(0, src, edge_rep_terms)
    return stress, edge_rep


def _run_maxent_exact(
    pos: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    lengths: torch.Tensor,
    config: MulMentConfig,
) -> torch.Tensor:
    """Run KaDraw's exact MaxEnt optimizer (all-pairs repulsion).

    Parameters
    ----------
    pos : torch.Tensor
        Initial positions ``[N, 2]`` in float64.
    src : torch.Tensor
        Directed edge sources ``[E]``.
    dst : torch.Tensor
        Directed edge targets ``[E]``.
    lengths : torch.Tensor
        Desired edge lengths ``[E]``.
    config : MulMentConfig
        Optimizer parameters.

    Returns
    -------
    torch.Tensor
        Refined positions ``[N, 2]``.

    Notes
    -----
    Mirrors ``local_optimizer::run_maxent_optimization_internal``: Jacobi
    sweeps, a ``do {} while (iterations-- > 0)`` inner loop (``inner + 1``
    sweeps), a relative-change tolerance exit, and an alpha decay of 0.3 per
    outer iteration. Degree-zero nodes snap to the origin exactly like the
    reference's value-initialized ``new_coord`` buffer.
    """
    num_nodes = int(pos.shape[0])
    if num_nodes <= 1 or src.numel() == 0 or config.steps <= 0:
        return pos

    ones = torch.ones((src.shape[0],), dtype=torch.float64)
    degree = torch.zeros((num_nodes,), dtype=torch.float64).index_add_(0, src, ones)
    active = degree > 0
    inv_sq = torch.reciprocal(lengths * lengths)
    rho_den = torch.zeros((num_nodes,), dtype=torch.float64).index_add_(0, src, inv_sq)
    rho = torch.where(active, torch.reciprocal(rho_den), torch.zeros_like(rho_den))
    q = float(config.q)
    sign_q = -1.0 if q < 0.0 else 1.0
    alpha = float(config.alpha)

    for _ in range(config.steps):
        iterations = config.inner_iterations
        while True:
            stress, edge_rep = _stress_and_edge_terms(pos, src, dst, lengths, rho, q)
            pair_diff = pos.unsqueeze(1) - pos.unsqueeze(0)
            pair_dist = torch.linalg.norm(pair_diff, dim=2)
            pair_dist.fill_diagonal_(1.0)
            repulsion = (pair_diff / pair_dist.pow(q + 2.0).unsqueeze(2)).sum(dim=1)
            entropy = (repulsion - edge_rep) * (alpha * rho).unsqueeze(1)
            new_pos = stress + sign_q * entropy
            new_pos[~active] = 0.0
            norm_coords = float(torch.sum(pos.square()).item())
            norm_diff = float(torch.sum((pos - new_pos).square()).item())
            pos = new_pos
            if norm_coords > 0.0 and norm_diff / norm_coords < config.tol:
                return pos
            iterations -= 1
            if iterations < 0:
                break
        alpha = max(0.3 * alpha, float(config.min_alpha))
    return pos


def _run_maxent_fast_approx(
    pos: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    lengths: torch.Tensor,
    node_weights: torch.Tensor,
    cluster_of: torch.Tensor,
    num_clusters: int,
    config: MulMentConfig,
) -> torch.Tensor:
    """Run KaDraw's ``faster_drawing`` cluster-approximated MaxEnt optimizer.

    Parameters
    ----------
    pos : torch.Tensor
        Initial positions ``[N, 2]`` in float64.
    src : torch.Tensor
        Directed edge sources ``[E]``.
    dst : torch.Tensor
        Directed edge targets ``[E]``.
    lengths : torch.Tensor
        Desired edge lengths ``[E]``.
    node_weights : torch.Tensor
        Current-level node weights ``[N]``.
    cluster_of : torch.Tensor
        Plus-x coarse cluster per node ``[N]``.
    num_clusters : int
        Number of plus-x clusters.
    config : MulMentConfig
        Optimizer parameters.

    Returns
    -------
    torch.Tensor
        Refined positions ``[N, 2]``.

    Notes
    -----
    Mirrors ``run_maxent_optimization_internal_fast_approx``: cluster
    centroids are recomputed from the current fine coordinates at the start
    of every sweep, repulsion against other clusters uses the centroid scaled
    by the cluster's *node count*, and repulsion inside the node's own
    cluster is computed exactly.
    """
    num_nodes = int(pos.shape[0])
    if num_nodes <= 1 or src.numel() == 0 or config.steps <= 0:
        return pos

    ones = torch.ones((src.shape[0],), dtype=torch.float64)
    degree = torch.zeros((num_nodes,), dtype=torch.float64).index_add_(0, src, ones)
    active = degree > 0
    inv_sq = torch.reciprocal(lengths * lengths)
    rho_den = torch.zeros((num_nodes,), dtype=torch.float64).index_add_(0, src, inv_sq)
    rho = torch.where(active, torch.reciprocal(rho_den), torch.zeros_like(rho_den))
    q = float(config.q)
    sign_q = -1.0 if q < 0.0 else 1.0
    alpha = float(config.alpha)

    counts = torch.bincount(cluster_of, minlength=num_clusters).to(torch.float64)
    cluster_weight = torch.zeros((num_clusters,), dtype=torch.float64).index_add_(
        0, cluster_of, node_weights
    )
    own_cluster = torch.nn.functional.one_hot(cluster_of, num_clusters).to(torch.bool)
    members: list[torch.Tensor] = [
        torch.nonzero(cluster_of == cluster, as_tuple=False).flatten()
        for cluster in range(num_clusters)
    ]

    for _ in range(config.steps):
        iterations = config.inner_iterations
        while True:
            weighted = node_weights.unsqueeze(1) * pos
            centroids = torch.zeros((num_clusters, 2), dtype=torch.float64).index_add_(
                0, cluster_of, weighted
            ) / cluster_weight.unsqueeze(1)

            stress, edge_rep = _stress_and_edge_terms(pos, src, dst, lengths, rho, q)

            cluster_diff = pos.unsqueeze(1) - centroids.unsqueeze(0)
            cluster_dist = torch.linalg.norm(cluster_diff, dim=2)
            cluster_dist = torch.where(own_cluster, torch.ones_like(cluster_dist), cluster_dist)
            cluster_terms = (
                counts.unsqueeze(0).unsqueeze(2)
                * cluster_diff
                / cluster_dist.pow(q + 2.0).unsqueeze(2)
            )
            cluster_terms = torch.where(
                own_cluster.unsqueeze(2), torch.zeros_like(cluster_terms), cluster_terms
            )
            repulsion = cluster_terms.sum(dim=1)

            for member_ids in members:
                if member_ids.numel() < 2:
                    continue
                local = pos[member_ids]
                local_diff = local.unsqueeze(1) - local.unsqueeze(0)
                local_dist = torch.linalg.norm(local_diff, dim=2)
                local_dist.fill_diagonal_(1.0)
                repulsion[member_ids] += (local_diff / local_dist.pow(q + 2.0).unsqueeze(2)).sum(
                    dim=1
                )

            entropy = (repulsion - edge_rep) * (alpha * rho).unsqueeze(1)
            new_pos = stress + sign_q * entropy
            new_pos[~active] = 0.0
            norm_coords = float(torch.sum(pos.square()).item())
            norm_diff = float(torch.sum((pos - new_pos).square()).item())
            pos = new_pos
            if norm_coords > 0.0 and norm_diff / norm_coords < config.tol:
                return pos
            iterations -= 1
            if iterations < 0:
                break
        alpha = max(0.3 * alpha, float(config.min_alpha))
    return pos


def _coarsest_initial_positions(
    num_nodes: int,
    node_weights: list[int],
    rng: _GlibcRand,
) -> torch.Tensor:
    """Create KaDraw's coarsest-graph starting coordinates.

    Parameters
    ----------
    num_nodes : int
        Coarsest node count.
    node_weights : list[int]
        Coarsest node weights.
    rng : _GlibcRand
        glibc ``rand()`` replica stream.

    Returns
    -------
    torch.Tensor
        Initial coordinates ``[N, 2]`` in float64.

    Notes
    -----
    Mirrors ``graph_drawer::perform_drawing``: exactly two coarsest nodes are
    placed deterministically at their desired distance; any other count draws
    ``nextDouble(0, 1)`` per axis from the glibc stream.
    """
    if num_nodes == 2:
        distance = _float32((math.sqrt(node_weights[0]) + math.sqrt(node_weights[1])) / 2.0)
        return torch.tensor([[0.0, 0.0], [0.0, distance]], dtype=torch.float64)
    coords = []
    for _ in range(num_nodes):
        x = _float32(rng.next_double(0.0, 1.0))
        y = _float32(rng.next_double(0.0, 1.0))
        coords.append([x, y])
    return torch.tensor(coords, dtype=torch.float64)


def _project_with_polar_jitter(
    coarse_pos: torch.Tensor,
    mapping: torch.Tensor,
    coarse_weights: torch.Tensor,
    rng: _GlibcRand,
) -> torch.Tensor:
    """Project coarse coordinates to the finer level with KaDraw's jitter.

    Parameters
    ----------
    coarse_pos : torch.Tensor
        Coarse positions ``[C, 2]``.
    mapping : torch.Tensor
        Fine-to-coarse mapping ``[N_fine]``.
    coarse_weights : torch.Tensor
        Coarse node weights ``[C]``.
    rng : _GlibcRand
        glibc ``rand()`` replica stream.

    Returns
    -------
    torch.Tensor
        Projected fine positions ``[N_fine, 2]``.

    Notes
    -----
    Mirrors ``graph_hierarchy::pop_finer_and_project`` with
    ``use_polar_coordinates``: per fine node, an angle from
    ``nextDouble(0, 2 * 3.1415)`` and a radius from
    ``nextDouble(0, sqrt(w_coarse) / 2)`` are drawn in node order from the
    glibc stream.
    """
    mapping_list = mapping.tolist()
    weights_list = coarse_weights.tolist()
    coords = []
    for coarse_node in mapping_list:
        max_dist = _float32(math.sqrt(weights_list[coarse_node]) / 2.0)
        angle = rng.next_double(0.0, _KADRAW_TWO_PI)
        distance = rng.next_double(0.0, max_dist)
        offset_x = _float32(distance * math.cos(angle))
        offset_y = _float32(distance * math.sin(angle))
        coords.append(
            [
                float(coarse_pos[coarse_node, 0].item()) + offset_x,
                float(coarse_pos[coarse_node, 1].item()) + offset_y,
            ]
        )
    return torch.tensor(coords, dtype=torch.float64)


def _compose_plus_x_mapping(
    levels: list[HierarchyLevel],
    level_index: int,
    plus_levels: int,
) -> tuple[torch.Tensor, int]:
    """Compose fine-to-coarse mappings across ``plus_levels`` transitions.

    Parameters
    ----------
    levels : list[HierarchyLevel]
        Full hierarchy transitions.
    level_index : int
        Index of the current fine level.
    plus_levels : int
        KaDraw ``faster_drawing_num_levels``.

    Returns
    -------
    tuple[torch.Tensor, int]
        Cluster assignment per fine node and the cluster count, mirroring
        ``graph_hierarchy::get_mapping_plus_x_faster`` and
        ``get_coarser_plus_x``.
    """
    total = len(levels)
    target_level = min(level_index + plus_levels, total)
    assert levels[level_index].fine_to_coarse is not None
    mapping = levels[level_index].fine_to_coarse.clone()
    for index in range(level_index + 1, target_level):
        step = levels[index].fine_to_coarse
        assert step is not None
        mapping = step[mapping]
    return mapping, levels[target_level - 1].num_nodes


@register_op
class MulMentCoarsenAndRefine(Op):
    """Build a hierarchy, optimize the coarsest graph, and unroll levels.

    Parameters
    ----------
    config : MulMentConfig
        MulMent optimizer configuration.
    """

    name = "mulment_coarsen_refine"
    category = OpCategory.COARSEN
    writes = ("pos", "hierarchy")

    def __init__(self, config: MulMentConfig) -> None:
        """Store the MulMent configuration.

        Parameters
        ----------
        config : MulMentConfig
            Validated MulMent configuration.

        Returns
        -------
        None
            The operation stores the configuration.
        """
        self.config = config

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Apply multilevel MaxEnt-Stress layout.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context carrying execution options.

        Returns
        -------
        SolveState
            State with final positions in ``state.pos``.
        """
        config = self.config
        levels = _build_kadraw_hierarchy(problem, config)
        state.hierarchy = levels
        rng = _GlibcRand(problem.seed)

        if not levels:
            adjacency = _weighted_adjacency(
                problem.edge_index,
                problem.num_nodes,
                problem.edge_weights,
            )
            node_weights = [1] * problem.num_nodes
            pos = _coarsest_initial_positions(problem.num_nodes, node_weights, rng)
            src, dst, lengths = _directed_edges_with_distances(adjacency, node_weights)
            state.pos = _run_maxent_exact(pos, src, dst, lengths, config).to(
                dtype=config.fidelity_dtype
            )
            return state

        coarsest = levels[-1]
        assert coarsest.edge_index is not None
        assert coarsest.node_masses is not None
        coarsest_weights = [int(weight) for weight in coarsest.node_masses.tolist()]
        pos = _coarsest_initial_positions(coarsest.num_nodes, coarsest_weights, rng)
        coarsest_adjacency = _weighted_adjacency(
            coarsest.edge_index,
            coarsest.num_nodes,
            coarsest.edge_weights,
        )
        src, dst, lengths = _directed_edges_with_distances(coarsest_adjacency, coarsest_weights)
        pos = _run_maxent_exact(pos, src, dst, lengths, config)

        plus_levels = _faster_drawing_num_levels(problem.num_nodes)

        for level_index in range(len(levels) - 1, -1, -1):
            level = levels[level_index]
            assert level.fine_to_coarse is not None
            assert level.node_masses is not None
            pos = _project_with_polar_jitter(
                pos,
                level.fine_to_coarse,
                level.node_masses,
                rng,
            )

            if level_index == 0:
                fine_edge_index = problem.edge_index
                fine_edge_weights = problem.edge_weights
                fine_nodes = problem.num_nodes
                fine_weights = [1] * problem.num_nodes
            else:
                previous = levels[level_index - 1]
                assert previous.edge_index is not None
                assert previous.node_masses is not None
                fine_edge_index = previous.edge_index
                fine_edge_weights = previous.edge_weights
                fine_nodes = previous.num_nodes
                fine_weights = [int(weight) for weight in previous.node_masses.tolist()]

            assert fine_edge_index is not None
            adjacency = _weighted_adjacency(fine_edge_index, fine_nodes, fine_edge_weights)
            src, dst, lengths = _directed_edges_with_distances(adjacency, fine_weights)
            if src.numel() == 0:
                continue
            cluster_of, num_clusters = _compose_plus_x_mapping(levels, level_index, plus_levels)
            pos = _run_maxent_fast_approx(
                pos,
                src,
                dst,
                lengths,
                torch.tensor(fine_weights, dtype=torch.float64),
                cluster_of,
                num_clusters,
                config,
            )

        state.pos = pos.to(dtype=config.fidelity_dtype)
        return state


def build_mulment_pipeline(config: Optional[MulMentConfig] = None) -> Pipeline:
    """Build the MulMent pipeline.

    Parameters
    ----------
    config : MulMentConfig, optional
        Pipeline configuration.

    Returns
    -------
    Pipeline
        Pipeline containing the multilevel coarsen/refine operation.
    """
    resolved = config or MulMentConfig()
    _validate_config(resolved)
    return Pipeline([MulMentCoarsenAndRefine(resolved)], name="mulment_pipeline")


def layout_mulment_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = _DEFAULT_OUTER_ITERATIONS,
    seed: int = 42,
    inner_iterations: int = _DEFAULT_INNER_ITERATIONS,
    alpha: float = _DEFAULT_ALPHA,
    min_alpha: float = _DEFAULT_MIN_ALPHA,
    q: float = _DEFAULT_Q,
    tol: float = _DEFAULT_TOL,
    max_levels: int = _DEFAULT_MAX_LEVELS,
    edge_weights: Optional[torch.Tensor] = None,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run the MulMent layout pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node sizes with shape ``[N, 2]``.
    steps : int, default=13
        MaxEnt outer iterations (KaDraw fast preset). ``0`` selects the
        preset default so engine dispatch with ``LayoutConfig.steps == 0``
        keeps reference behavior; pass ``MulMentConfig(steps=0)`` to
        ``build_mulment_pipeline`` to skip optimization entirely. Negative
        values raise.
    seed : int, default=42
        Seed for both KaDraw RNG streams (MT19937 tie-breaks and glibc
        ``rand()`` coordinates).
    inner_iterations : int, default=2
        MaxEnt inner iterations per outer iteration.
    alpha : float, default=1.0
        Initial MaxEnt alpha.
    min_alpha : float, default=0.008
        Minimum alpha.
    q : float, default=0.0
        Entropy exponent.
    tol : float, default=1e-4
        Relative convergence tolerance.
    max_levels : int, default=64
        Safety cap on hierarchy depth.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``. Weights steer label
        propagation and quotient cut weights; desired edge lengths always
        derive from node weights, matching KaDraw.
    fidelity_dtype : torch.dtype, optional
        Output dtype.

    Returns
    -------
    torch.Tensor
        Final positions with shape ``[N, 2]``.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    resolved_dtype = torch.float32 if fidelity_dtype is None else fidelity_dtype
    device = layout_device(edge_index=edge_index, node_sizes=node_sizes)
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=resolved_dtype, device=device)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=resolved_dtype, device=device)

    config = MulMentConfig(
        steps=steps if steps != 0 else _DEFAULT_OUTER_ITERATIONS,
        inner_iterations=inner_iterations,
        alpha=alpha,
        min_alpha=min_alpha,
        q=q,
        tol=tol,
        max_levels=max_levels,
        fidelity_dtype=resolved_dtype,
    )
    _validate_config(config)
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    final_state = build_mulment_pipeline(config).apply(
        problem,
        SolveState(),
        RuntimeContext(plan=ExecutionPlan(device="cpu")),
    )
    if final_state.pos is None:
        raise RuntimeError("MulMent pipeline did not produce final positions.")
    return final_state.pos.to(device=device)


__all__ = ["MulMentConfig", "build_mulment_pipeline", "layout_mulment_pipeline"]
