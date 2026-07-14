"""MulMent multilevel MaxEnt-Stress layout pipeline."""

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
_DEFAULT_COARSEST_SIZE = 32
_DEFAULT_MAX_LEVELS = 20
_ZERO_DISTANCE_EPSILON = 1.0e-9
_KADRAW_LABEL_ITERATIONS = 5
_KADRAW_CLUSTER_COARSENING_FACTOR = 20.0
_KADRAW_SIZE_BASE = 2.0
_KADRAW_STOP_NODES = 2
_MT19937_N = 624
_MT19937_M = 397
_MT19937_MATRIX_A = 0x9908B0DF
_MT19937_UPPER_MASK = 0x80000000
_MT19937_LOWER_MASK = 0x7FFFFFFF
_UINT32_MAX = 0xFFFFFFFF
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


@dataclass(frozen=True)
class MulMentConfig:
    """Configuration for KaDraw-style MulMent layout.

    Parameters
    ----------
    steps : int
        Total refinement budget. The value is split across coarsest and
        uncoarsening levels.
    alpha : float
        Initial MaxEnt repulsion weight.
    min_alpha : float
        Lower bound for the outer-loop alpha schedule.
    q : float
        KaDraw MaxEnt exponent. ``0`` gives inverse-square entropy forces.
    tol : float
        Relative coordinate-change tolerance for each refinement stage.
    coarsest_size : int
        Stop coarsening when the graph reaches this size.
    max_levels : int
        Maximum number of coarsening transitions.
    fidelity_dtype : torch.dtype
        Floating-point dtype used by the local optimizer.
    """

    steps: int = 200
    alpha: float = _DEFAULT_ALPHA
    min_alpha: float = _DEFAULT_MIN_ALPHA
    q: float = _DEFAULT_Q
    tol: float = _DEFAULT_TOL
    coarsest_size: int = _DEFAULT_COARSEST_SIZE
    max_levels: int = _DEFAULT_MAX_LEVELS
    fidelity_dtype: torch.dtype = torch.float32


class _KaDrawRng:
    """KaDraw-compatible random stream for coarsening tie-breaks.

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
    if config.alpha < 0.0:
        raise ValueError("alpha must be non-negative.")
    if config.min_alpha < 0.0:
        raise ValueError("min_alpha must be non-negative.")
    if config.tol <= 0.0:
        raise ValueError("tol must be positive.")
    if config.coarsest_size < 1:
        raise ValueError("coarsest_size must be positive.")
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
    KaDraw's ``configuration::standard`` selects ``DEGREE_NODEORDERING``.
    ``std::sort`` is not stable, but libstdc++'s implementation leaves equal
    degrees in ascending ID order for the small deterministic ranges covered
    by the fidelity tests.
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
        Sorted undirected adjacency lists.
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
    original_num_nodes = problem.num_nodes
    cluster_factor = _KADRAW_CLUSTER_COARSENING_FACTOR
    level_index = 0

    while level_index < config.max_levels:
        upper_bound = min(
            _KADRAW_SIZE_BASE ** float(level_index + 1),
            math.ceil(float(original_num_nodes) / cluster_factor),
        )
        upper_bound = min(upper_bound, max(0, original_num_nodes - 1))
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
        if current_num_nodes / float(coarse_num_nodes) < 1.1:
            cluster_factor *= 0.7
        if not contraction_stop:
            break

        current_edge_index = coarse_edges
        current_edge_weights = coarse_weights
        current_num_nodes = coarse_num_nodes
        current_node_weights = coarse_node_weights
        level_index += 1

    return levels


def _undirected_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
    dtype: torch.dtype,
    edge_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unique undirected edges and desired lengths.

    Parameters
    ----------
    edge_index : torch.Tensor
        Input edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    dtype : torch.dtype
        Floating-point dtype for desired lengths.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``. KaDraw's unit-distance graph
        format is modeled by interpreting weights as desired lengths when
        explicitly supplied.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Unique edge tensor with shape ``[2, U]`` and lengths with shape ``[U]``.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")
    if edge_weights is not None and edge_weights.shape[0] != edge_index.shape[1]:
        raise ValueError("edge_weights length must match edge count.")

    edge_index_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    weights_cpu = (
        torch.ones((edge_index_cpu.shape[1],), dtype=dtype)
        if edge_weights is None
        else edge_weights.detach().to(device="cpu", dtype=dtype)
    )
    seen: dict[tuple[int, int], list[float]] = {}
    for edge_id, (source, target) in enumerate(
        zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist())
    ):
        if source < 0 or source >= num_nodes or target < 0 or target >= num_nodes:
            raise ValueError("edge_index contains a node outside [0, num_nodes).")
        if source == target:
            continue
        key = (min(source, target), max(source, target))
        seen.setdefault(key, []).append(float(weights_cpu[edge_id].item()))

    if not seen:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=dtype),
        )

    pairs = sorted(seen)
    lengths = [sum(seen[pair]) / len(seen[pair]) for pair in pairs]
    return (
        torch.tensor(pairs, dtype=torch.long).transpose(0, 1).contiguous(),
        torch.tensor(lengths, dtype=dtype),
    )


def _seeded_initial_positions(num_nodes: int, seed: int, dtype: torch.dtype) -> torch.Tensor:
    """Create deterministic KaDraw-compatible starting coordinates.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    seed : int
        Random seed.
    dtype : torch.dtype
        Output dtype.

    Returns
    -------
    torch.Tensor
        Initial coordinates with shape ``[N, 2]``.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randn((num_nodes, 2), generator=generator, dtype=dtype)


def _run_local_maxent(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    edge_lengths: torch.Tensor,
    config: MulMentConfig,
    outer_iterations: int,
) -> torch.Tensor:
    """Run KaDraw's fixed-point MaxEnt local optimizer.

    Parameters
    ----------
    positions : torch.Tensor
        Initial positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Unique undirected edge tensor with shape ``[2, E]``.
    edge_lengths : torch.Tensor
        Desired edge lengths with shape ``[E]``.
    config : MulMentConfig
        Optimizer parameters.
    outer_iterations : int
        Number of alpha-schedule outer iterations.

    Returns
    -------
    torch.Tensor
        Refined positions with shape ``[N, 2]``.
    """
    num_nodes = int(positions.shape[0])
    if num_nodes <= 1 or edge_index.numel() == 0 or outer_iterations <= 0:
        return positions

    src = edge_index[0].to(dtype=torch.long)
    dst = edge_index[1].to(dtype=torch.long)
    lengths = edge_lengths.to(dtype=positions.dtype).clamp_min(_ZERO_DISTANCE_EPSILON)
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(num_nodes)]
    for source, target, length in zip(src.tolist(), dst.tolist(), lengths.tolist()):
        adjacency[source].append((target, float(length)))
        adjacency[target].append((source, float(length)))

    pos = positions.clone()
    alpha = float(config.alpha)
    inner_iterations = max(1, int(math.ceil(float(config.steps) / max(1, outer_iterations))))
    q = float(config.q)
    sign_q = 0.0 if q == 0.0 else (-1.0 if q < 0.0 else 1.0)
    all_indices = torch.arange(num_nodes, dtype=torch.long)

    for _ in range(outer_iterations):
        for _inner in range(inner_iterations):
            new_pos = pos.clone()
            for node, neighbors in enumerate(adjacency):
                if not neighbors:
                    continue
                neighbor_ids = torch.tensor([item[0] for item in neighbors], dtype=torch.long)
                neighbor_lengths = torch.tensor(
                    [item[1] for item in neighbors],
                    dtype=pos.dtype,
                )
                rho = torch.reciprocal(torch.sum(torch.reciprocal(neighbor_lengths.square())))
                diff = pos[node].unsqueeze(0) - pos[neighbor_ids]
                dist = torch.linalg.norm(diff, dim=1).clamp_min(_ZERO_DISTANCE_EPSILON)
                scaled = neighbor_lengths / dist
                stress_center = (
                    torch.sum(
                        (pos[neighbor_ids] + scaled.unsqueeze(1) * diff)
                        / neighbor_lengths.square().unsqueeze(1),
                        dim=0,
                    )
                    * rho
                )

                other_mask = all_indices != node
                other = all_indices[other_mask]
                other_diff = pos[node].unsqueeze(0) - pos[other]
                other_dist = torch.linalg.norm(other_diff, dim=1).clamp_min(_ZERO_DISTANCE_EPSILON)
                repulsion = torch.sum(other_diff / other_dist.pow(q + 2.0).unsqueeze(1), dim=0)
                edge_repulsion = torch.sum(diff / dist.pow(q + 2.0).unsqueeze(1), dim=0)
                entropy = (repulsion - edge_repulsion) * (alpha * float(rho.item()))
                new_pos[node] = stress_center + sign_q * entropy

            norm_coords = float(torch.sum(pos.square()).item())
            norm_diff = float(torch.sum((pos - new_pos).square()).item())
            pos = new_pos
            if norm_coords > 0.0 and norm_diff / norm_coords < config.tol:
                return pos
        alpha = max(0.3 * alpha, float(config.min_alpha))
    return pos


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
        levels = _build_kadraw_hierarchy(problem, self.config)
        state.hierarchy = levels
        if not levels:
            edge_index, lengths = _undirected_edges(
                problem.edge_index,
                problem.num_nodes,
                self.config.fidelity_dtype,
                problem.edge_weights,
            )
            state.pos = _run_local_maxent(
                _seeded_initial_positions(
                    problem.num_nodes,
                    problem.seed,
                    self.config.fidelity_dtype,
                ),
                edge_index,
                lengths,
                self.config,
                outer_iterations=max(1, min(self.config.steps, 8)),
            )
            return state

        coarsest = levels[-1]
        assert coarsest.edge_index is not None
        assert coarsest.edge_weights is not None
        pos = _seeded_initial_positions(
            coarsest.num_nodes,
            problem.seed,
            self.config.fidelity_dtype,
        )
        edge_index, lengths = _undirected_edges(
            coarsest.edge_index,
            coarsest.num_nodes,
            self.config.fidelity_dtype,
            coarsest.edge_weights,
        )
        pos = _run_local_maxent(pos, edge_index, lengths, self.config, outer_iterations=4)

        for level_index in range(len(levels) - 1, -1, -1):
            level = levels[level_index]
            assert level.fine_to_coarse is not None
            mapping = level.fine_to_coarse.to(dtype=torch.long)
            jitter = torch.zeros((level.num_fine, 2), dtype=self.config.fidelity_dtype)
            pos = pos[mapping] + jitter
            if level_index == 0:
                fine_edge_index = problem.edge_index
                fine_edge_weights = problem.edge_weights
                fine_nodes = problem.num_nodes
            else:
                previous = levels[level_index - 1]
                fine_edge_index = previous.edge_index
                fine_edge_weights = previous.edge_weights
                fine_nodes = previous.num_nodes
            assert fine_edge_index is not None
            edge_index, lengths = _undirected_edges(
                fine_edge_index,
                fine_nodes,
                self.config.fidelity_dtype,
                fine_edge_weights,
            )
            pos = _run_local_maxent(pos, edge_index, lengths, self.config, outer_iterations=2)

        state.pos = pos
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
    steps: int = 200,
    seed: int = 42,
    alpha: float = _DEFAULT_ALPHA,
    min_alpha: float = _DEFAULT_MIN_ALPHA,
    q: float = _DEFAULT_Q,
    tol: float = _DEFAULT_TOL,
    coarsest_size: int = _DEFAULT_COARSEST_SIZE,
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
    steps : int, default=200
        Refinement budget.
    seed : int, default=42
        Seed for coarsening and projected jitter.
    alpha : float, default=1.0
        Initial MaxEnt alpha.
    min_alpha : float, default=0.008
        Minimum alpha.
    q : float, default=0.0
        Entropy exponent.
    tol : float, default=1e-4
        Relative convergence tolerance.
    coarsest_size : int, default=32
        Coarsening target.
    max_levels : int, default=20
        Maximum hierarchy depth.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.
    fidelity_dtype : torch.dtype, optional
        Internal dtype.

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
        steps=steps,
        alpha=alpha,
        min_alpha=min_alpha,
        q=q,
        tol=tol,
        coarsest_size=coarsest_size,
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
