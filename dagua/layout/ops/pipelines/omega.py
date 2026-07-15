"""Omega/RDMDS resistance-distance stress pipeline.

This module ports the core shape of ``likr/egraph-rs`` Omega: an RDMDS
spectral embedding is used to define target distances, then a sparse SGD stress
pass refines edge pairs plus sampled non-edge pairs. The Rust CLI currently
uses ``thread_rng`` and is not seedable, so this port keeps the same stage order
while using Dagua's explicit ``seed`` for deterministic runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple, Union

import numpy as np
import torch

from dagua.layout.ops.base import Op, Pipeline
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_DEFAULT_D = 2
_DEFAULT_K = 30
_DEFAULT_MIN_DIST = 1.0e-3
_DEFAULT_SHIFT = 1.0e-3
_DEFAULT_UNIT_EDGE_LENGTH = 1.0
_DEFAULT_SGD_ITERATIONS = 100
_DEFAULT_SGD_EPS = 0.1
_DEFAULT_EIGENVALUE_TOLERANCE = 1.0e-4
_U32_MASK = 0xFFFFFFFF
_U64_MASK = 0xFFFFFFFFFFFFFFFF
_CHACHA_CONSTANTS = (0x61707865, 0x3320646E, 0x79622D32, 0x6B206574)
_PCG32_MULTIPLIER = 6364136223846793005
_PCG32_INCREMENT = 11634580027462260723


class _OmegaRng(Protocol):
    """Protocol for Omega random streams.

    Implementations provide the subset of sampling operations used by the
    egraph-rs Omega path.
    """

    def gen_range_usize(self, upper: int) -> int:
        """Sample uniformly from ``0..upper``.

        Parameters
        ----------
        upper : int
            Exclusive upper bound.

        Returns
        -------
        int
            Sampled integer.
        """
        ...

    def gen_index(self, upper: int) -> int:
        """Sample a slice index like ``rand::seq::gen_index``.

        Parameters
        ----------
        upper : int
            Exclusive upper bound.

        Returns
        -------
        int
            Sampled index.
        """
        ...

    def shuffle(self, values: np.ndarray) -> None:
        """Shuffle an integer array in place.

        Parameters
        ----------
        values : numpy.ndarray
            One-dimensional integer array.

        Returns
        -------
        None
            ``values`` is shuffled in place.
        """
        ...


@dataclass(frozen=True)
class OmegaConfig:
    """Configuration for the Omega/RDMDS pipeline.

    Parameters
    ----------
    d : int, default=2
        RDMDS embedding rank. The final layout uses the first two dimensions.
    k : int, default=30
        Number of random node-pair attempts per source node.
    min_dist : float, default=1e-3
        Minimum target distance in the embedding-distance matrix.
    shift : float, default=1e-3
        Positive diagonal shift used by the Rust RDMDS inverse iteration.
        The dense port records this value for API parity; direct eigensolve does
        not require the shifted system.
    unit_edge_length : float, default=1.0
        Weight assigned to every graph edge in the standard Laplacian.
    sgd_iterations : int, default=100
        Number of SparseSGD refinement iterations.
    sgd_eps : float, default=0.1
        Final scheduler epsilon used to derive the minimum learning rate.
    seed : int, default=42
        Deterministic sampler and shuffle seed.
    dtype : torch.dtype, default=torch.float32
        Output tensor dtype.
    """

    d: int = _DEFAULT_D
    k: int = _DEFAULT_K
    min_dist: float = _DEFAULT_MIN_DIST
    shift: float = _DEFAULT_SHIFT
    unit_edge_length: float = _DEFAULT_UNIT_EDGE_LENGTH
    sgd_iterations: int = _DEFAULT_SGD_ITERATIONS
    sgd_eps: float = _DEFAULT_SGD_EPS
    seed: int = 42
    dtype: torch.dtype = torch.float32


@dataclass
class _OmegaPair:
    """One sparse stress pair.

    Parameters
    ----------
    i : int
        First node index.
    j : int
        Second node index.
    distance : float
        Target Euclidean distance.
    weight : float
        Stress weight, equal to ``1 / distance**2``.
    """

    i: int
    j: int
    distance: float
    weight: float


def _rotate_left_u32(value: int, amount: int) -> int:
    """Rotate a 32-bit integer left.

    Parameters
    ----------
    value : int
        Input word.
    amount : int
        Rotation amount in bits.

    Returns
    -------
    int
        Rotated 32-bit word.
    """
    return ((value << amount) & _U32_MASK) | (value >> (32 - amount))


def _chacha_quarter_round(state: List[int], a: int, b: int, c: int, d: int) -> None:
    """Apply one ChaCha quarter round.

    Parameters
    ----------
    state : list[int]
        Sixteen-word ChaCha state.
    a : int
        First state index.
    b : int
        Second state index.
    c : int
        Third state index.
    d : int
        Fourth state index.

    Returns
    -------
    None
        ``state`` is updated in place.
    """
    state[a] = (state[a] + state[b]) & _U32_MASK
    state[d] ^= state[a]
    state[d] = _rotate_left_u32(state[d], 16)
    state[c] = (state[c] + state[d]) & _U32_MASK
    state[b] ^= state[c]
    state[b] = _rotate_left_u32(state[b], 12)
    state[a] = (state[a] + state[b]) & _U32_MASK
    state[d] ^= state[a]
    state[d] = _rotate_left_u32(state[d], 8)
    state[c] = (state[c] + state[d]) & _U32_MASK
    state[b] ^= state[c]
    state[b] = _rotate_left_u32(state[b], 7)


def _pcg32_seed_words(seed: int) -> List[int]:
    """Expand a u64 seed like ``rand_core::SeedableRng::seed_from_u64``.

    Parameters
    ----------
    seed : int
        Unsigned 64-bit seed value.

    Returns
    -------
    list[int]
        Eight little-endian u32 words for the ChaCha key.
    """
    state = int(seed) & _U64_MASK
    words: List[int] = []
    for _ in range(8):
        state = (state * _PCG32_MULTIPLIER + _PCG32_INCREMENT) & _U64_MASK
        xorshifted = (((state >> 18) ^ state) >> 27) & _U32_MASK
        rotation = (state >> 59) & 31
        word = (xorshifted >> rotation) | ((xorshifted << ((-rotation) & 31)) & _U32_MASK)
        words.append(word & _U32_MASK)
    return words


class _RustStdRng:
    """Small port of rand 0.8 ``StdRng`` for Omega fidelity.

    The egraph-rs CLI uses ``StdRng`` from ``rand`` 0.8.7, which is ChaCha12
    with PCG32 seed expansion. Only the operations required by Omega are
    implemented here.
    """

    def __init__(self, seed: int) -> None:
        """Initialize the RNG from a u64 seed.

        Parameters
        ----------
        seed : int
            Seed forwarded to ``StdRng::seed_from_u64``.
        """
        self._state = list(_CHACHA_CONSTANTS) + _pcg32_seed_words(seed) + [0, 0, 0, 0]
        self._buffer: List[int] = []

    def _refill4(self) -> None:
        """Generate four buffered ChaCha12 blocks.

        Returns
        -------
        None
            The internal u32 buffer is replaced.
        """
        words: List[int] = []
        for _ in range(4):
            working = self._state.copy()
            for _round_pair in range(6):
                _chacha_quarter_round(working, 0, 4, 8, 12)
                _chacha_quarter_round(working, 1, 5, 9, 13)
                _chacha_quarter_round(working, 2, 6, 10, 14)
                _chacha_quarter_round(working, 3, 7, 11, 15)
                _chacha_quarter_round(working, 0, 5, 10, 15)
                _chacha_quarter_round(working, 1, 6, 11, 12)
                _chacha_quarter_round(working, 2, 7, 8, 13)
                _chacha_quarter_round(working, 3, 4, 9, 14)
            words.extend((working[index] + self._state[index]) & _U32_MASK for index in range(16))
            self._state[12] = (self._state[12] + 1) & _U32_MASK
            if self._state[12] == 0:
                self._state[13] = (self._state[13] + 1) & _U32_MASK
        self._buffer = words

    def next_u32(self) -> int:
        """Return the next u32 from the Rust ``BlockRng`` stream.

        Returns
        -------
        int
            Unsigned 32-bit random word.
        """
        if not self._buffer:
            self._refill4()
        return self._buffer.pop(0)

    def next_u64(self) -> int:
        """Return the next u64 from two little-endian u32 words.

        Returns
        -------
        int
            Unsigned 64-bit random word.
        """
        low = self.next_u32()
        high = self.next_u32()
        return low | (high << 32)

    def skip_f32_ranges(self, count: int) -> None:
        """Skip ``gen_range(f32..f32)`` draws.

        Parameters
        ----------
        count : int
            Number of f32 range samples to consume.

        Returns
        -------
        None
            The RNG stream is advanced.
        """
        for _ in range(max(0, int(count))):
            self.next_u32()

    def gen_range_f32(self, low: float, high: float) -> np.float32:
        """Sample a Rust ``f32`` range.

        Parameters
        ----------
        low : float
            Inclusive lower bound.
        high : float
            Exclusive upper bound.

        Returns
        -------
        numpy.float32
            Sampled value.
        """
        if not low < high:
            raise ValueError("low must be less than high.")
        scale = np.float32(high) - np.float32(low)
        value_0_1 = np.float32(self.next_u32() >> 9) * np.float32(1.0 / (1 << 23))
        return value_0_1 * scale + np.float32(low)

    def gen_range_usize(self, upper: int) -> int:
        """Sample uniformly from ``0usize..upper`` using rand 0.8 arithmetic.

        Parameters
        ----------
        upper : int
            Exclusive upper bound.

        Returns
        -------
        int
            Sampled integer.
        """
        if upper <= 0:
            raise ValueError("upper must be positive.")
        leading_zeros = 64 - int(upper).bit_length()
        zone = (((int(upper) << leading_zeros) & _U64_MASK) - 1) & _U64_MASK
        while True:
            value = self.next_u64()
            product = value * int(upper)
            high = (product >> 64) & _U64_MASK
            low = product & _U64_MASK
            if low <= zone:
                return int(high)

    def gen_range_u32(self, upper: int) -> int:
        """Sample uniformly from ``0u32..upper`` using rand 0.8 arithmetic.

        Parameters
        ----------
        upper : int
            Exclusive upper bound, constrained to ``u32``.

        Returns
        -------
        int
            Sampled integer.
        """
        if upper <= 0 or upper > _U32_MASK:
            raise ValueError("upper must be in 1..=u32::MAX.")
        leading_zeros = 32 - int(upper).bit_length()
        zone = (((int(upper) << leading_zeros) & _U32_MASK) - 1) & _U32_MASK
        while True:
            value = self.next_u32()
            product = value * int(upper)
            high = (product >> 32) & _U32_MASK
            low = product & _U32_MASK
            if low <= zone:
                return int(high)

    def gen_index(self, upper: int) -> int:
        """Sample a slice index like Rust rand's private ``gen_index`` helper.

        Parameters
        ----------
        upper : int
            Exclusive upper bound.

        Returns
        -------
        int
            Sampled index.
        """
        if upper <= _U32_MASK:
            return self.gen_range_u32(upper)
        return self.gen_range_usize(upper)

    def shuffle(self, values: np.ndarray) -> None:
        """Shuffle values like Rust ``SliceRandom::shuffle``.

        Parameters
        ----------
        values : numpy.ndarray
            One-dimensional integer array.

        Returns
        -------
        None
            ``values`` is shuffled in place.
        """
        for index in range(len(values) - 1, 0, -1):
            swap_index = self.gen_index(index + 1)
            values[index], values[swap_index] = values[swap_index], values[index]


class _NumpyOmegaRng:
    """Adapter preserving existing NumPy-based omega unit tests."""

    def __init__(self, rng: np.random.Generator) -> None:
        """Store a NumPy generator.

        Parameters
        ----------
        rng : numpy.random.Generator
            Generator used for compatibility tests.
        """
        self._rng = rng

    def gen_range_usize(self, upper: int) -> int:
        """Sample uniformly from ``0..upper``.

        Parameters
        ----------
        upper : int
            Exclusive upper bound.

        Returns
        -------
        int
            Sampled integer.
        """
        return int(self._rng.integers(0, upper))

    def gen_index(self, upper: int) -> int:
        """Sample a shuffle index.

        Parameters
        ----------
        upper : int
            Exclusive upper bound.

        Returns
        -------
        int
            Sampled index.
        """
        return self.gen_range_usize(upper)

    def shuffle(self, values: np.ndarray) -> None:
        """Shuffle values with NumPy.

        Parameters
        ----------
        values : numpy.ndarray
            One-dimensional integer array.

        Returns
        -------
        None
            ``values`` is shuffled in place.
        """
        self._rng.shuffle(values)


def _undirected_edges(edge_index: torch.Tensor, num_nodes: int) -> List[Tuple[int, int]]:
    """Return valid undirected edges in input order.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    list[tuple[int, int]]
        Unique undirected edges with self-loops and invalid endpoints removed.
    """
    if edge_index.numel() == 0:
        return []
    edges: List[Tuple[int, int]] = []
    seen: set[Tuple[int, int]] = set()
    for raw_u, raw_v in edge_index.detach().cpu().t().tolist():
        u = int(raw_u)
        v = int(raw_v)
        if u == v or u < 0 or v < 0 or u >= num_nodes or v >= num_nodes:
            continue
        key = (u, v) if u < v else (v, u)
        if key in seen:
            continue
        seen.add(key)
        edges.append((u, v))
    return edges


def _laplacian(num_nodes: int, edges: List[Tuple[int, int]], unit_edge_length: float) -> np.ndarray:
    """Build the standard graph Laplacian used by egraph-rs RDMDS.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    edges : list[tuple[int, int]]
        Unique undirected graph edges.
    unit_edge_length : float
        Edge weight added to the Laplacian degree and adjacency terms.

    Returns
    -------
    numpy.ndarray
        Dense Laplacian matrix with shape ``[N, N]``.
    """
    lap = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    weight = float(unit_edge_length)
    for u, v in edges:
        lap[u, u] += weight
        lap[v, v] += weight
        lap[u, v] -= weight
        lap[v, u] -= weight
    return lap


def _rdmds_shifted_laplacian(
    num_nodes: int,
    edges: List[Tuple[int, int]],
    config: OmegaConfig,
) -> np.ndarray:
    """Build the f32 ``L + shift * I`` matrix used by Rust RDMDS.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    edges : list[tuple[int, int]]
        Unique undirected graph edges.
    config : OmegaConfig
        RDMDS configuration.

    Returns
    -------
    numpy.ndarray
        Shifted Laplacian matrix with shape ``[N, N]`` and dtype ``float32``.
    """
    matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    weight = np.float32(config.unit_edge_length)
    for u, v in edges:
        matrix[u, u] = np.float32(matrix[u, u] + weight)
        matrix[v, v] = np.float32(matrix[v, v] + weight)
        matrix[u, v] = np.float32(matrix[u, v] - weight)
        matrix[v, u] = np.float32(matrix[v, u] - weight)
    shift = np.float32(config.shift)
    for index in range(num_nodes):
        matrix[index, index] = np.float32(matrix[index, index] + shift)
    return matrix


def _incomplete_cholesky(
    matrix: np.ndarray,
) -> Tuple[List[List[Tuple[int, np.float32]]], np.ndarray]:
    """Compute the IC(0) preconditioner used by egraph-rs.

    Parameters
    ----------
    matrix : numpy.ndarray
        Symmetric f32 matrix with shape ``[N, N]``.

    Returns
    -------
    tuple[list[list[tuple[int, numpy.float32]]], numpy.ndarray]
        Lower-triangular row entries and diagonal terms.
    """
    num_nodes = int(matrix.shape[0])
    row_entries: List[List[Tuple[int, np.float32]]] = [[] for _ in range(num_nodes)]
    adjacency: List[dict[int, np.float32]] = [dict() for _ in range(num_nodes)]
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            value = np.float32(matrix[i, j])
            if value == np.float32(0.0):
                continue
            adjacency[i][j] = value
            adjacency[j][i] = value
            row_entries[j].append((i, value))
    diagonal = np.zeros(num_nodes, dtype=np.float32)
    for i in range(num_nodes):
        row_entries[i].sort(key=lambda item: item[0])
    for i in range(num_nodes):
        sum_value = np.float32(0.0)
        for _, entry_value in row_entries[i]:
            sum_value = np.float32(sum_value + entry_value * entry_value)
        diagonal[i] = np.sqrt(np.maximum(np.float32(matrix[i, i] - sum_value), np.float32(0.0)))
        if diagonal[i] <= np.float32(0.0):
            diagonal[i] = np.float32(1.0e-6)
        for j in sorted(adjacency[i]):
            if j <= i:
                continue
            entry_pos = next(
                (pos for pos, (col, _) in enumerate(row_entries[j]) if col == i),
                None,
            )
            if entry_pos is None:
                continue
            overlap = np.float32(0.0)
            left = 0
            right = 0
            while left < len(row_entries[i]) and right < len(row_entries[j]):
                left_col, left_value = row_entries[i][left]
                right_col, right_value = row_entries[j][right]
                if left_col == right_col and left_col < i:
                    overlap = np.float32(overlap + left_value * right_value)
                    left += 1
                    right += 1
                elif left_col < right_col:
                    left += 1
                else:
                    right += 1
            row_entries[j][entry_pos] = (
                i,
                np.float32((adjacency[i][j] - overlap) / diagonal[i]),
            )
    return row_entries, diagonal


def _apply_ic_preconditioner(
    row_entries: List[List[Tuple[int, np.float32]]],
    diagonal: np.ndarray,
    residual: np.ndarray,
) -> np.ndarray:
    """Apply the IC(0) preconditioner.

    Parameters
    ----------
    row_entries : list[list[tuple[int, numpy.float32]]]
        Lower-triangular IC entries by row.
    diagonal : numpy.ndarray
        IC diagonal vector with shape ``[N]``.
    residual : numpy.ndarray
        Residual vector with shape ``[N]``.

    Returns
    -------
    numpy.ndarray
        Preconditioned residual with shape ``[N]``.
    """
    num_nodes = int(residual.shape[0])
    y_value = np.zeros(num_nodes, dtype=np.float32)
    for i in range(num_nodes):
        total = np.float32(0.0)
        for j, entry_value in row_entries[i]:
            total = np.float32(total + entry_value * y_value[j])
        y_value[i] = np.float32((residual[i] - total) / diagonal[i])

    z_value = np.zeros(num_nodes, dtype=np.float32)
    col_entries: List[List[Tuple[int, np.float32]]] = [[] for _ in range(num_nodes)]
    for row, entries in enumerate(row_entries):
        for col, entry_value in entries:
            col_entries[col].append((row, entry_value))
    for i in range(num_nodes - 1, -1, -1):
        total = np.float32(0.0)
        for j, entry_value in col_entries[i]:
            total = np.float32(total + entry_value * z_value[j])
        z_value[i] = np.float32((y_value[i] - total) / diagonal[i])
    return z_value


def _solve_with_conjugate_gradient(
    matrix: np.ndarray,
    row_entries: List[List[Tuple[int, np.float32]]],
    diagonal: np.ndarray,
    rhs: np.ndarray,
    initial_solution: np.ndarray,
    cg_max_iterations: int,
    cg_tolerance: float,
) -> np.ndarray:
    """Solve one shifted Laplacian system with preconditioned CG.

    Parameters
    ----------
    matrix : numpy.ndarray
        Shifted Laplacian matrix with shape ``[N, N]``.
    row_entries : list[list[tuple[int, numpy.float32]]]
        IC lower-triangular row entries.
    diagonal : numpy.ndarray
        IC diagonal vector with shape ``[N]``.
    rhs : numpy.ndarray
        Right-hand side vector with shape ``[N]``.
    initial_solution : numpy.ndarray
        Mutable CG initial guess with shape ``[N]``. Rust RDMDS reuses this
        vector across inverse-iteration solves instead of zeroing each system.
    cg_max_iterations : int
        Maximum CG iterations.
    cg_tolerance : float
        Residual tolerance.

    Returns
    -------
    numpy.ndarray
        Approximate solution vector with shape ``[N]``.
    """
    solution = initial_solution.astype(np.float32, copy=True)
    residual = np.float32(rhs - matrix @ solution)
    z_value = _apply_ic_preconditioner(row_entries, diagonal, residual)
    direction = z_value.copy()
    rsold = np.float32(np.dot(residual, z_value))
    tolerance_sq = np.float32(cg_tolerance) * np.float32(cg_tolerance)
    for _ in range(int(cg_max_iterations)):
        q_value = np.float32(matrix @ direction)
        alpha = np.float32(rsold / np.float32(np.dot(direction, q_value)))
        solution = np.float32(solution + alpha * direction)
        residual = np.float32(residual - alpha * q_value)
        z_value = _apply_ic_preconditioner(row_entries, diagonal, residual)
        rsnew = np.float32(np.dot(residual, z_value))
        if rsnew < tolerance_sq:
            break
        beta = np.float32(rsnew / rsold)
        direction = np.float32(beta * direction + z_value)
        rsold = rsnew
    return solution


def _gram_schmidt(vector: np.ndarray, known_vectors: np.ndarray) -> np.ndarray:
    """Orthogonalize a vector against known columns.

    Parameters
    ----------
    vector : numpy.ndarray
        Input vector with shape ``[N]``.
    known_vectors : numpy.ndarray
        Matrix whose columns are known vectors.

    Returns
    -------
    numpy.ndarray
        Orthogonalized f32 vector.
    """
    result = vector.astype(np.float32, copy=True)
    for col_index in range(int(known_vectors.shape[1])):
        column = known_vectors[:, col_index]
        dot_value = np.float32(np.dot(result, column))
        result = np.float32(result - column * dot_value)
    return result


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normalize a vector with f32 arithmetic.

    Parameters
    ----------
    vector : numpy.ndarray
        Input vector with shape ``[N]``.

    Returns
    -------
    numpy.ndarray
        Normalized vector.
    """
    norm = np.sqrt(np.float32(np.dot(vector, vector)))
    if norm > np.float32(0.0):
        return np.float32(vector / norm)
    return vector.astype(np.float32, copy=True)


def _rdmds_embedding_iterative(
    num_nodes: int,
    edges: List[Tuple[int, int]],
    config: OmegaConfig,
    rng: _RustStdRng,
) -> np.ndarray:
    """Compute the Rust-style approximate RDMDS embedding.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    edges : list[tuple[int, int]]
        Unique undirected graph edges.
    config : OmegaConfig
        RDMDS configuration.
    rng : _RustStdRng
        Rust-compatible RNG consumed by random initial vectors.

    Returns
    -------
    numpy.ndarray
        Embedding with shape ``[N, d]``.
    """
    rank = max(1, int(config.d))
    if num_nodes == 0:
        return np.zeros((0, rank), dtype=np.float32)
    if num_nodes == 1 or not edges:
        return np.zeros((num_nodes, rank), dtype=np.float32)

    matrix = _rdmds_shifted_laplacian(num_nodes, edges, config)
    row_entries, diagonal = _incomplete_cholesky(matrix)
    all_vectors = np.zeros((num_nodes, rank + 1), dtype=np.float32)
    all_vectors[:, 0] = np.float32(1.0 / math.sqrt(float(num_nodes)))
    all_values = np.zeros(rank + 1, dtype=np.float32)
    y_value = np.zeros(num_nodes, dtype=np.float32)
    tolerance = np.float32(_DEFAULT_EIGENVALUE_TOLERANCE)
    for eigen_index in range(1, rank + 1):
        x_iter = np.array(
            [rng.gen_range_f32(-1.0, 1.0) for _ in range(num_nodes)],
            dtype=np.float32,
        )
        x_iter = _normalize_vector(_gram_schmidt(x_iter, all_vectors[:, :eigen_index]))
        previous = np.float32(0.0)
        for _ in range(1000):
            y_value = _solve_with_conjugate_gradient(
                matrix,
                row_entries,
                diagonal,
                x_iter,
                y_value,
                cg_max_iterations=100,
                cg_tolerance=1.0e-4,
            )
            x_next = _normalize_vector(_gram_schmidt(y_value, all_vectors[:, :eigen_index]))
            numerator = np.float32(np.dot(x_next, np.float32(matrix @ x_next)))
            denominator = np.float32(np.dot(x_next, x_next))
            estimate = np.float32(numerator / denominator)
            converged = np.abs(np.float32(estimate - previous)) < tolerance
            x_iter = x_next
            previous = estimate
            if bool(converged):
                break
        all_values[eigen_index] = previous
        all_vectors[:, eigen_index] = x_iter

    coords = np.zeros((num_nodes, rank), dtype=np.float32)
    shift = np.float32(config.shift)
    for dim in range(rank):
        eigenvalue = np.maximum(np.float32(all_values[dim + 1] - shift), np.float32(0.0))
        if eigenvalue > np.float32(0.0):
            coords[:, dim] = np.float32(all_vectors[:, dim + 1] / np.sqrt(eigenvalue))
    return coords


def _rdmds_embedding(
    num_nodes: int,
    edges: List[Tuple[int, int]],
    config: OmegaConfig,
    rng: Optional[_RustStdRng] = None,
) -> np.ndarray:
    """Compute the resistance-distance MDS spectral embedding.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    edges : list[tuple[int, int]]
        Unique undirected graph edges.
    config : OmegaConfig
        RDMDS configuration.
    rng : _RustStdRng, optional
        Rust-compatible RNG. When provided, the egraph-rs inverse-iteration
        path is used and consumes the RNG stream.

    Returns
    -------
    numpy.ndarray
        Embedding array with shape ``[N, d]``.
    """
    if rng is not None:
        return _rdmds_embedding_iterative(num_nodes, edges, config, rng)
    rank = max(1, int(config.d))
    if num_nodes == 0:
        return np.zeros((0, rank), dtype=np.float64)
    if num_nodes == 1 or not edges:
        return np.zeros((num_nodes, rank), dtype=np.float64)

    lap = _laplacian(num_nodes, edges, config.unit_edge_length)
    eigenvalues, eigenvectors = np.linalg.eigh(lap)
    order = np.argsort(eigenvalues, kind="stable")
    coords = np.zeros((num_nodes, rank), dtype=np.float64)
    for out_dim, eigen_idx in enumerate(order[1 : rank + 1]):
        value = max(float(eigenvalues[eigen_idx]), 0.0)
        if value <= 0.0:
            continue
        coords[:, out_dim] = eigenvectors[:, eigen_idx] / math.sqrt(value)
    return coords


def _embedding_distance(embedding: np.ndarray, i: int, j: int, min_dist: float) -> float:
    """Return the clamped Euclidean distance between two embedding rows.

    Parameters
    ----------
    embedding : numpy.ndarray
        RDMDS embedding with shape ``[N, d]``.
    i : int
        First node index.
    j : int
        Second node index.
    min_dist : float
        Minimum returned distance.

    Returns
    -------
    float
        Euclidean distance clamped to ``min_dist``.
    """
    delta = np.float32(embedding[i] - embedding[j])
    distance = np.sqrt(np.float32(np.dot(delta, delta)))
    return float(np.maximum(distance, np.float32(min_dist)))


def _build_pairs(
    edges: List[Tuple[int, int]],
    embedding: np.ndarray,
    config: OmegaConfig,
    rng: Union[np.random.Generator, _OmegaRng],
) -> List[_OmegaPair]:
    """Build edge pairs plus reference-ordered random pairs.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Unique undirected graph edges in input order.
    embedding : numpy.ndarray
        RDMDS embedding with shape ``[N, d]``.
    config : OmegaConfig
        SparseSGD pair configuration.
    rng : numpy.random.Generator or _OmegaRng
        Deterministic random generator.

    Returns
    -------
    list[_OmegaPair]
        Sparse stress pairs.
    """
    num_nodes = int(embedding.shape[0])
    pairs: List[_OmegaPair] = []
    used: set[Tuple[int, int]] = set()
    sampler: _OmegaRng = _NumpyOmegaRng(rng) if isinstance(rng, np.random.Generator) else rng

    for u, v in edges:
        key = (u, v) if u < v else (v, u)
        if key in used:
            continue
        used.add(key)
        distance = np.float32(_embedding_distance(embedding, u, v, config.min_dist))
        weight = np.float32(np.float32(1.0) / np.float32(distance * distance))
        pairs.append(_OmegaPair(u, v, float(distance), float(weight)))

    for i in range(num_nodes):
        for _ in range(max(0, int(config.k))):
            j = sampler.gen_range_usize(num_nodes)
            if i == j:
                continue
            key = (i, j) if i < j else (j, i)
            if key in used:
                continue
            used.add(key)
            distance = np.float32(_embedding_distance(embedding, i, j, config.min_dist))
            weight = np.float32(np.float32(1.0) / np.float32(distance * distance))
            pairs.append(_OmegaPair(i, j, float(distance), float(weight)))
    return pairs


def _scheduler_bounds(pairs: List[_OmegaPair], epsilon: float) -> Tuple[float, float]:
    """Compute egraph-rs exponential scheduler bounds.

    Parameters
    ----------
    pairs : list[_OmegaPair]
        Sparse stress pairs.
    epsilon : float
        Scheduler epsilon.

    Returns
    -------
    tuple[float, float]
        ``(eta_min, eta_max)`` bounds.
    """
    weights = [pair.weight for pair in pairs if pair.weight > 0.0]
    if not weights:
        return 0.0, 0.0
    eta_min = np.float32(np.float32(epsilon) / np.float32(max(weights)))
    eta_max = np.float32(np.float32(1.0) / np.float32(min(weights)))
    return float(eta_min), float(eta_max)


def _run_sparse_sgd(
    embedding: np.ndarray,
    pairs: List[_OmegaPair],
    config: OmegaConfig,
    rng: Union[np.random.Generator, _OmegaRng],
) -> np.ndarray:
    """Run Omega SparseSGD refinement.

    Parameters
    ----------
    embedding : numpy.ndarray
        Initial RDMDS embedding with shape ``[N, d]``.
    pairs : list[_OmegaPair]
        Sparse stress pairs.
    config : OmegaConfig
        SGD configuration.
    rng : numpy.random.Generator or _OmegaRng
        Deterministic random generator used for per-iteration shuffles.

    Returns
    -------
    numpy.ndarray
        Refined positions with shape ``[N, 2]``.
    """
    pos = _initial_egraph_positions(embedding.shape[0])
    if not pairs or config.sgd_iterations <= 0:
        return pos

    eta_min, eta_max = _scheduler_bounds(pairs, config.sgd_eps)
    if eta_max <= 0.0:
        return pos
    decay = np.float32(0.0)
    if config.sgd_iterations > 1 and eta_min > 0.0:
        decay = np.float32(
            np.log(np.float32(eta_max) / np.float32(eta_min))
            / np.float32(config.sgd_iterations - 1)
        )

    sampler: _OmegaRng = _NumpyOmegaRng(rng) if isinstance(rng, np.random.Generator) else rng
    order = np.arange(len(pairs), dtype=np.int64)
    for step in range(config.sgd_iterations):
        eta = np.float32(np.float32(eta_max) * np.exp(np.float32(-decay * np.float32(step))))
        sampler.shuffle(order)
        for pair_idx in order.tolist():
            pair = pairs[pair_idx]
            delta = np.float32(pos[pair.i] - pos[pair.j])
            norm = np.sqrt(np.float32(np.dot(delta, delta)))
            if norm <= np.float32(0.0):
                continue
            mu = np.minimum(np.float32(eta * np.float32(pair.weight)), np.float32(1.0))
            ratio = np.float32(
                np.float32(0.5) * np.float32(norm - np.float32(pair.distance)) / norm
            )
            move = np.float32(delta * ratio * mu)
            pos[pair.i] = np.float32(pos[pair.i] - move)
            pos[pair.j] = np.float32(pos[pair.j] + move)
    return pos


def _initial_egraph_positions(num_nodes: int) -> np.ndarray:
    """Return egraph-rs Euclidean 2D initial placement.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    numpy.ndarray
        Initial position array with shape ``[N, 2]``.
    """
    positions = np.zeros((num_nodes, 2), dtype=np.float32)
    golden_angle = np.float32(np.pi) * (np.float32(3.0) - np.sqrt(np.float32(5.0)))
    for index in range(num_nodes):
        radius = np.float32(10.0) * np.sqrt(np.float32(index))
        theta = np.float32(golden_angle * np.float32(index))
        positions[index, 0] = np.float32(radius * np.cos(theta))
        positions[index, 1] = np.float32(radius * np.sin(theta))
    return positions


@register_op
@dataclass
class ComputeOmegaEmbedding(Op):
    """Compute RDMDS coordinates for the Omega pipeline."""

    config: OmegaConfig
    name: str = "compute_omega_embedding"
    category: OpCategory = OpCategory.INIT
    writes: Tuple[str, ...] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute and cache the RDMDS embedding.

        Parameters
        ----------
        problem : LayoutProblem
            Graph topology and node count.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context; accepted for pipeline compatibility.

        Returns
        -------
        SolveState
            State with ``omega_edges`` and ``omega_embedding`` cached.
        """
        del ctx
        edges = _undirected_edges(problem.edge_index, problem.num_nodes)
        rng = _RustStdRng(self.config.seed)
        embedding = _rdmds_embedding(problem.num_nodes, edges, self.config, rng)
        state.extras["omega_edges"] = edges
        state.extras["omega_embedding"] = embedding
        state.extras["omega_rng"] = rng
        return state


@register_op
@dataclass
class BuildOmegaPairs(Op):
    """Build Omega SparseSGD pairs from the RDMDS embedding."""

    config: OmegaConfig
    name: str = "build_omega_pairs"
    category: OpCategory = OpCategory.PREPROCESS
    reads: Tuple[str, ...] = ("extras",)
    writes: Tuple[str, ...] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Sample edge and random pairs in Omega order.

        Parameters
        ----------
        problem : LayoutProblem
            Graph topology and deterministic seed.
        state : SolveState
            Mutable solve state with cached RDMDS embedding.
        ctx : RuntimeContext
            Runtime context; accepted for pipeline compatibility.

        Returns
        -------
        SolveState
            State with ``omega_pairs`` cached.
        """
        del ctx
        embedding = state.extras.get("omega_embedding")
        edges = state.extras.get("omega_edges")
        if not isinstance(embedding, np.ndarray) or not isinstance(edges, list):
            raise RuntimeError("Omega embedding stage must run before pair construction.")
        rng = state.extras.get("omega_rng")
        if not isinstance(rng, _RustStdRng):
            raise RuntimeError("Omega RDMDS stage did not preserve the Rust RNG stream.")
        state.extras["omega_pairs"] = _build_pairs(edges, embedding, self.config, rng)
        return state


@register_op
@dataclass
class RunOmegaSparseSgd(Op):
    """Run the Omega SparseSGD position refinement."""

    config: OmegaConfig
    name: str = "run_omega_sparse_sgd"
    category: OpCategory = OpCategory.OPTIMIZE
    reads: Tuple[str, ...] = ("extras",)
    writes: Tuple[str, ...] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Refine and store Omega positions.

        Parameters
        ----------
        problem : LayoutProblem
            Graph topology and output device.
        state : SolveState
            Mutable solve state with cached embedding and pairs.
        ctx : RuntimeContext
            Runtime context; accepted for pipeline compatibility.

        Returns
        -------
        SolveState
            State with final ``pos`` tensor populated.
        """
        del ctx
        embedding = state.extras.get("omega_embedding")
        pairs = state.extras.get("omega_pairs")
        rng = state.extras.get("omega_rng")
        if not isinstance(embedding, np.ndarray) or not isinstance(pairs, list):
            raise RuntimeError("Omega pair construction must run before SparseSGD.")
        if not isinstance(rng, _RustStdRng):
            raise RuntimeError("Omega pair construction did not preserve the Rust RNG stream.")
        pos = _run_sparse_sgd(embedding, pairs, self.config, rng)
        state.pos = torch.as_tensor(pos, dtype=self.config.dtype, device=problem.edge_index.device)
        return state


def build_omega_pipeline(config: Optional[OmegaConfig] = None) -> Pipeline:
    """Build the Omega/RDMDS pipeline.

    Parameters
    ----------
    config : OmegaConfig, optional
        Pipeline configuration. ``None`` uses Omega reference defaults.

    Returns
    -------
    Pipeline
        RDMDS embedding, pair construction, and SparseSGD refinement stages.
    """
    resolved = OmegaConfig() if config is None else config
    if resolved.d <= 0:
        raise ValueError("d must be positive.")
    if resolved.k < 0:
        raise ValueError("k must be non-negative.")
    if resolved.min_dist <= 0.0:
        raise ValueError("min_dist must be positive.")
    if resolved.unit_edge_length <= 0.0:
        raise ValueError("unit_edge_length must be positive.")
    return Pipeline(
        [
            ComputeOmegaEmbedding(resolved),
            BuildOmegaPairs(resolved),
            RunOmegaSparseSgd(resolved),
        ],
        name="omega_pipeline",
    )


def layout_omega_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    *,
    d: int = _DEFAULT_D,
    k: int = _DEFAULT_K,
    min_dist: float = _DEFAULT_MIN_DIST,
    shift: float = _DEFAULT_SHIFT,
    unit_edge_length: float = _DEFAULT_UNIT_EDGE_LENGTH,
    sgd_iterations: int = _DEFAULT_SGD_ITERATIONS,
    sgd_eps: float = _DEFAULT_SGD_EPS,
    seed: Optional[int] = 42,
    dtype: Union[torch.dtype, str] = torch.float32,
    **kwargs: object,
) -> torch.Tensor:
    """Lay out a graph with Omega/RDMDS.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor, optional
        Node sizes with shape ``[N, 2]``. Accepted for pipeline API
        compatibility; Omega does not use node boxes.
    d : int, default=2
        RDMDS embedding rank.
    k : int, default=30
        Number of random pair attempts per node.
    min_dist : float, default=1e-3
        Minimum target distance.
    shift : float, default=1e-3
        RDMDS shift parameter retained for API parity.
    unit_edge_length : float, default=1.0
        Standard Laplacian edge weight.
    sgd_iterations : int, default=100
        Number of SparseSGD refinement iterations.
    sgd_eps : float, default=0.1
        Scheduler epsilon.
    seed : int, optional
        Deterministic sampler seed. ``None`` resolves to ``42``.
    dtype : torch.dtype or str, default=torch.float32
        Output dtype.
    **kwargs : object
        Additional dispatch kwargs accepted for compatibility.

    Returns
    -------
    torch.Tensor
        Final positions with shape ``[N, 2]``.
    """
    del node_sizes, kwargs
    resolved_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    config = OmegaConfig(
        d=d,
        k=k,
        min_dist=min_dist,
        shift=shift,
        unit_edge_length=unit_edge_length,
        sgd_iterations=sgd_iterations,
        sgd_eps=sgd_eps,
        seed=42 if seed is None else int(seed),
        dtype=resolved_dtype,
    )
    problem = LayoutProblem(edge_index=edge_index, num_nodes=num_nodes, seed=config.seed)
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device=str(edge_index.device)))
    final_state = build_omega_pipeline(config).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("Omega pipeline did not produce positions.")
    return final_state.pos.to(device=edge_index.device, dtype=resolved_dtype)


__all__ = [
    "BuildOmegaPairs",
    "ComputeOmegaEmbedding",
    "OmegaConfig",
    "RunOmegaSparseSgd",
    "build_omega_pipeline",
    "layout_omega_pipeline",
]
