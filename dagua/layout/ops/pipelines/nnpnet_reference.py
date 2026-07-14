"""Reference-exact NNP-NET stage ports (pure reimplementation, no reference calls).

This module reimplements the upstream NNP-NET C++ pipeline stages with
bit-matching arithmetic order so the dagua pipeline reproduces the reference
binary's positions at fixed seed:

* glibc ``srand``/``rand`` (TYPE_3 additive feedback generator) used by the
  reference PivotMDS power-iteration initialization.
* OGDF-derived PivotMDS (maxmin pivots, double centering, seeded power
  iteration, singular-value projection).
* The tsNET* teacher: Barnes-Hut t-SNE (SPTree) with tsNET compression and
  repulsion phases, graph-BFS KNN input similarities, and the reference's
  phase-switch control flow.

All floating-point accumulations follow the C++ statement order (sequential
scalar sums, elementwise vector updates) so IEEE-754 double results are
identical to the single-threaded reference build. Every constant and control
path was transcribed from the reference sources (NNP-NET/LayoutMethods/
PivotMDS.h, tsNET.h, tsNETTree/sptree.cpp, Graph.h); the reference binary is
never invoked or imported here.
"""

from __future__ import annotations

import math
from typing import Iterator, Optional

import numpy as np

_DBL_MIN = 2.2250738585072014e-308
_FLT_MIN = 1.1754943508222875e-38
_DBL_MAX = 1.7976931348623157e308
_RAND_MAX = 2147483647


def glibc_rand_stream(seed: int) -> Iterator[int]:
    """Reproduce glibc's ``srand``/``rand`` output stream.

    Parameters
    ----------
    seed : int
        Seed passed to ``srand``.

    Yields
    ------
    int
        Successive ``rand()`` values in ``[0, 2**31 - 1]``.
    """
    seed = seed & 0xFFFFFFFF
    if seed == 0:
        seed = 1
    values = [0] * 344
    values[0] = seed
    for index in range(1, 31):
        # values[i] = (16807 * values[i-1]) % 2147483647 via Schrage's method.
        high, low = divmod(values[index - 1], 127773)
        word = 16807 * low - 2836 * high
        if word < 0:
            word += 2147483647
        values[index] = word
    for index in range(31, 34):
        values[index] = values[index - 31]
    for index in range(34, 344):
        values[index] = (values[index - 31] + values[index - 3]) & 0xFFFFFFFF
    index = 344
    while True:
        value = (values[index - 31] + values[index - 3]) & 0xFFFFFFFF
        values.append(value)
        yield value >> 1
        index += 1


def canonical_adjacency(edge_index: "np.ndarray", num_nodes: int) -> list[list[int]]:
    """Build adjacency lists in the reference graph-file order.

    The reference adapter serializes graphs as sorted unique undirected edge
    pairs; the reference loader appends both endpoints per line. This function
    reproduces that adjacency ordering, which drives KNN tie order.

    Parameters
    ----------
    edge_index : numpy.ndarray
        Edge array with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    list[list[int]]
        Neighbor lists in reference insertion order.
    """
    pairs: set[tuple[int, int]] = set()
    for source, target in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        if source == target:
            continue
        pairs.add((min(source, target), max(source, target)))
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    for source, target in sorted(pairs):
        adjacency[source].append(target)
        adjacency[target].append(source)
    return adjacency


def _bfs_distances(adjacency: list[list[int]], source: int) -> np.ndarray:
    """Compute reference BFS hop distances (unreachable nodes stay ``-1``).

    Parameters
    ----------
    adjacency : list[list[int]]
        Neighbor lists.
    source : int
        BFS start node.

    Returns
    -------
    numpy.ndarray
        Distances with shape ``[N]`` as float64.
    """
    count = len(adjacency)
    distances = np.full(count, -1.0, dtype=np.float64)
    distances[source] = 0.0
    current = [source]
    depth = 1.0
    while current:
        upcoming: list[int] = []
        for node in current:
            for other in adjacency[node]:
                if distances[other] != -1.0:
                    continue
                distances[other] = depth
                upcoming.append(other)
        depth += 1.0
        current = upcoming
    return distances


def _seq_dot(left: np.ndarray, right: np.ndarray) -> float:
    """Compute a left-to-right sequential dot product.

    Parameters
    ----------
    left : numpy.ndarray
        First vector.
    right : numpy.ndarray
        Second vector.

    Returns
    -------
    float
        Sequential accumulation matching the C++ loop order.
    """
    total = 0.0
    left_list = left.tolist()
    right_list = right.tolist()
    for index in range(len(left_list)):
        total += left_list[index] * right_list[index]
    return total


def _normalize_vector(vector: np.ndarray) -> float:
    """Normalize a vector in place with the reference norm rule.

    Parameters
    ----------
    vector : numpy.ndarray
        Vector normalized in place.

    Returns
    -------
    float
        The pre-normalization Euclidean norm.
    """
    norm = math.sqrt(_seq_dot(vector, vector))
    if norm != 0.0:
        vector /= norm
    return norm


def reference_pmds(
    adjacency: list[list[int]],
    dims: int,
    pivots: int,
    seed: int,
) -> np.ndarray:
    """Port of the reference PivotMDS layout (PivotMDS.h).

    Parameters
    ----------
    adjacency : list[list[int]]
        Neighbor lists in reference order.
    dims : int
        Output dimension count (power-iteration eigenvector count).
    pivots : int
        Requested maxmin pivot count.
    seed : int
        glibc ``srand`` seed for the power-iteration initialization.

    Returns
    -------
    numpy.ndarray
        Node-major coordinates with shape ``[N, dims]`` (float64).
    """
    count = len(adjacency)
    if count == 0:
        return np.zeros((0, dims), dtype=np.float64)
    if count == 1:
        return np.zeros((1, dims), dtype=np.float64)
    pivot_count = min(pivots, count)

    # Maxmin pivot selection (getPivotDistanceMatrix, inAll == nullptr).
    pivot_rows: list[np.ndarray] = []
    lowest = np.zeros(count, dtype=np.float64)
    pivot = 0
    for row_id in range(pivot_count):
        row = _bfs_distances(adjacency, pivot)
        pivot_rows.append(row)
        if row_id + 1 < pivot_count:
            highest = 0.0
            if row_id == 0:
                lowest = row.copy()
                lowest_list = lowest.tolist()
                for node in range(count):
                    if lowest_list[node] > highest:
                        highest = lowest_list[node]
                        pivot = node
                continue
            row_list = row.tolist()
            lowest_list = lowest.tolist()
            for node in range(count):
                if row_list[node] < lowest_list[node]:
                    lowest_list[node] = row_list[node]
                if lowest_list[node] > highest:
                    highest = lowest_list[node]
                    pivot = node
            lowest = np.asarray(lowest_list, dtype=np.float64)

    # centerPivotmatrix.
    matrix = [row.tolist() for row in pivot_rows]
    normalization_factor = 0.0
    col_normalization = [0.0] * pivot_count
    for row_id in range(pivot_count):
        row_col_normalizer = 0.0
        row = matrix[row_id]
        for node in range(count):
            row_col_normalizer += row[node] * row[node]
        normalization_factor += row_col_normalizer
        col_normalization[row_id] = row_col_normalizer / count
    normalization_factor = normalization_factor / (count * pivot_count)
    for node in range(count):
        row_col_normalizer = 0.0
        for row_id in range(pivot_count):
            square = matrix[row_id][node] * matrix[row_id][node]
            matrix[row_id][node] = square + normalization_factor - col_normalization[row_id]
            row_col_normalizer += square
        row_col_normalizer /= pivot_count
        for row_id in range(pivot_count):
            matrix[row_id][node] = -0.5 * (matrix[row_id][node] - row_col_normalizer)
    centered = np.asarray(matrix, dtype=np.float64)

    # selfProduct: K = C @ C.T with sequential inner sums.
    kernel = np.zeros((pivot_count, pivot_count), dtype=np.float64)
    for row_id in range(pivot_count):
        for other in range(row_id + 1):
            total = 0.0
            left = matrix[row_id]
            right = matrix[other]
            for node in range(count):
                total += left[node] * right[node]
            kernel[row_id][other] = total
            kernel[other][row_id] = total

    # eigenValueDecomposition: seeded power iteration.
    stream = glibc_rand_stream(seed)
    eigenvectors = np.empty((dims, pivot_count), dtype=np.float64)
    for dim in range(dims):
        for column in range(pivot_count):
            eigenvectors[dim][column] = float(next(stream)) / _RAND_MAX
    eigenvalues = [0.0] * dims
    for dim in range(dims):
        eigenvalues[dim] = _normalize_vector(eigenvectors[dim])
    epsilon = 1.0 - 1.0e-5 if dims > 3 else 1.0 - 1.0e-10
    residual = 0.0
    iteration = 0
    while residual < epsilon:
        if iteration >= 10000:
            break
        iteration += 1
        if math.isnan(residual) or math.isinf(residual):
            break
        previous = eigenvectors.copy()
        eigenvectors[:] = 0.0
        for dim in range(dims):
            accumulator = eigenvectors[dim]
            old = previous[dim].tolist()
            for column in range(pivot_count):
                accumulator += kernel[column] * old[column]
        for dim in range(dims):
            for other in range(dim):
                denominator = _seq_dot(eigenvectors[other], eigenvectors[other])
                factor = _seq_dot(eigenvectors[other], eigenvectors[dim]) / denominator
                eigenvectors[dim] -= factor * eigenvectors[other]
        for dim in range(dims):
            eigenvalues[dim] = _normalize_vector(eigenvectors[dim])
        residual = 1.0
        for dim in range(dims):
            alignment = _seq_dot(eigenvectors[dim], previous[dim])
            if alignment < 0.0:
                alignment = -alignment
            if residual > alignment:
                residual = alignment

    # singularValueDecomposition projection (single-thread order).
    coordinates = np.zeros((dims, count), dtype=np.float64)
    for dim in range(dims):
        eigenvalues[dim] = math.sqrt(eigenvalues[dim])
        accumulator = coordinates[dim]
        weights = eigenvectors[dim].tolist()
        for column in range(pivot_count):
            accumulator += centered[column] * weights[column]
    for dim in range(dims):
        _normalize_vector(coordinates[dim])

    # pivotMDSLayout aspect-ratio scaling.
    for dim in range(dims):
        eigenvalues[dim] = math.sqrt(eigenvalues[dim])
        coordinates[dim] *= eigenvalues[dim]
    return np.ascontiguousarray(coordinates.T)


def graph_normalize(positions: np.ndarray) -> np.ndarray:
    """Port of ``Graph::normalize`` (per-dim min, global max range).

    Parameters
    ----------
    positions : numpy.ndarray
        Node-major coordinates; dtype selects float or double semantics.

    Returns
    -------
    numpy.ndarray
        Normalized coordinates with the input dtype.
    """
    dtype = positions.dtype
    minimum = positions.min(axis=0)
    maximum = positions.max(axis=0)
    ranges = maximum - minimum
    max_size = ranges[0]
    for dim in range(1, ranges.shape[0]):
        if ranges[dim] > max_size:
            max_size = ranges[dim]
    return ((positions - minimum) / max_size).astype(dtype, copy=False)


def reference_pivot_embedding(adjacency: list[list[int]], pivots: int) -> np.ndarray:
    """Port of ``createPivotEmbedding`` including its transposed layout.

    The reference writes pivot-major rows but downstream reads the buffer
    node-major; this NaN-fallback path reproduces that reinterpretation.

    Parameters
    ----------
    adjacency : list[list[int]]
        Neighbor lists.
    pivots : int
        Number of pivot dimensions.

    Returns
    -------
    numpy.ndarray
        Buffer with shape ``[N, pivots]`` (float32) matching the reference
        memory reinterpretation.
    """
    count = len(adjacency)
    flat = np.zeros(count * pivots, dtype=np.float32)
    pivot = 0
    lowest = np.zeros(count, dtype=np.float32)
    for row_id in range(pivots):
        row = _bfs_distances(adjacency, pivot).astype(np.float32)
        flat[row_id * count : (row_id + 1) * count] = row
        if row_id + 1 < pivots:
            highest = 0.0
            if row_id == 0:
                lowest = row.copy()
                lowest_list = lowest.tolist()
                for node in range(count):
                    if lowest_list[node] > highest:
                        highest = lowest_list[node]
                        pivot = node
                continue
            row_list = row.tolist()
            lowest_list = lowest.tolist()
            for node in range(count):
                if row_list[node] < lowest_list[node]:
                    lowest_list[node] = row_list[node]
                if lowest_list[node] > highest:
                    highest = lowest_list[node]
                    pivot = node
            lowest = np.asarray(lowest_list, dtype=np.float32)
    biggest = np.float32(0.0)
    for value in flat.tolist():
        if value > biggest:
            biggest = np.float32(value)
    if biggest != 0.0:
        flat = (flat / biggest).astype(np.float32)
    return flat.reshape(count, pivots)


def _reference_knn(
    adjacency: list[list[int]],
    source: int,
    neighbors: int,
) -> tuple[list[int], list[float]]:
    """Port of ``Graph::knn`` (bucketed expansion, LIFO within buckets).

    Parameters
    ----------
    adjacency : list[list[int]]
        Neighbor lists in reference order.
    source : int
        Query node.
    neighbors : int
        Number of neighbors requested.

    Returns
    -------
    tuple[list[int], list[float]]
        Selected node ids and distances in reference emission order.
    """
    count = len(adjacency)
    visited = {source}
    buckets: dict[float, list[int]] = {}
    add_extra = len(adjacency[source]) != count - 1
    for other in adjacency[source]:
        buckets.setdefault(1.0, []).append(other)
    nodes: list[int] = []
    distances: list[float] = []
    points_left = neighbors
    while buckets and points_left > 0:
        key = min(buckets)
        while buckets and not buckets[key]:
            del buckets[key]
            if buckets:
                key = min(buckets)
        if not buckets:
            break
        lowest = buckets[key].pop()
        if lowest in visited:
            continue
        points_left -= 1
        visited.add(lowest)
        distances.append(key)
        nodes.append(lowest)
        if not add_extra:
            continue
        for other in adjacency[lowest]:
            if other in visited:
                continue
            buckets.setdefault(key + 1.0, []).append(other)
    while len(nodes) < neighbors:
        nodes.append(0)
        distances.append(0.0)
    return nodes, distances


def _gaussian_perplexity_sparse(
    adjacency: list[list[int]],
    count: int,
    perplexity: float,
    neighbors: int,
) -> tuple[list[int], list[int], list[float]]:
    """Port of the sparse ``computeGaussianPerplexity`` (KNN beta search).

    Parameters
    ----------
    adjacency : list[list[int]]
        Neighbor lists.
    count : int
        Number of nodes.
    perplexity : float
        Target perplexity (already reference-adjusted).
    neighbors : int
        Neighbor count ``K``.

    Returns
    -------
    tuple[list[int], list[int], list[float]]
        CSR ``row_P``, ``col_P``, ``val_P`` arrays.
    """
    row_p = [row * neighbors for row in range(count + 1)]
    col_p = [0] * (count * neighbors)
    val_p = [0.0] * (count * neighbors)
    log_perplexity = math.log(perplexity) if perplexity > 0.0 else -math.inf
    for node in range(count):
        node_ids, distances = _reference_knn(adjacency, node, neighbors)
        found = False
        beta = 1.0
        min_beta = -_DBL_MAX
        max_beta = _DBL_MAX
        tolerance = 1.0e-5
        iteration = 0
        current = [0.0] * neighbors
        sum_p = _DBL_MIN
        while not found and iteration < 200:
            for column in range(neighbors):
                current[column] = math.exp(-beta * distances[column] * distances[column])
            sum_p = _DBL_MIN
            for column in range(neighbors):
                sum_p += current[column]
            entropy = 0.0
            for column in range(neighbors):
                entropy += beta * (distances[column] * distances[column] * current[column])
            entropy = (entropy / sum_p) + math.log(sum_p)
            difference = entropy - log_perplexity
            if difference < tolerance and -difference < tolerance:
                found = True
            else:
                if difference > 0:
                    min_beta = beta
                    if max_beta == _DBL_MAX or max_beta == -_DBL_MAX:
                        beta *= 2.0
                    else:
                        beta = (beta + max_beta) / 2.0
                else:
                    max_beta = beta
                    if min_beta == -_DBL_MAX or min_beta == _DBL_MAX:
                        beta /= 2.0
                    else:
                        beta = (beta + min_beta) / 2.0
            iteration += 1
        base = row_p[node]
        for column in range(neighbors):
            current[column] /= sum_p
            col_p[base + column] = node_ids[column]
            val_p[base + column] = current[column]
    return row_p, col_p, val_p


def _symmetrize_sparse(
    row_p: list[int],
    col_p: list[int],
    val_p: list[float],
    count: int,
) -> tuple[list[int], list[int], list[float]]:
    """Port of ``symmetrizeMatrix`` including its exact fill order.

    Parameters
    ----------
    row_p : list[int]
        CSR row offsets.
    col_p : list[int]
        CSR column ids.
    val_p : list[float]
        CSR values.
    count : int
        Number of nodes.

    Returns
    -------
    tuple[list[int], list[int], list[float]]
        Symmetrized CSR arrays.
    """
    row_counts = [0] * count
    for node in range(count):
        for entry in range(row_p[node], row_p[node + 1]):
            present = False
            other = col_p[entry]
            for candidate in range(row_p[other], row_p[other + 1]):
                if col_p[candidate] == node:
                    present = True
            if present:
                row_counts[node] += 1
            else:
                row_counts[node] += 1
                row_counts[other] += 1
    total = sum(row_counts)
    sym_row = [0] * (count + 1)
    for node in range(count):
        sym_row[node + 1] = sym_row[node] + row_counts[node]
    sym_col = [0] * total
    sym_val = [0.0] * total
    offset = [0] * count
    for node in range(count):
        for entry in range(row_p[node], row_p[node + 1]):
            other = col_p[entry]
            present = False
            for candidate in range(row_p[other], row_p[other + 1]):
                if col_p[candidate] == node:
                    present = True
                    if node <= other:
                        sym_col[sym_row[node] + offset[node]] = other
                        sym_col[sym_row[other] + offset[other]] = node
                        sym_val[sym_row[node] + offset[node]] = val_p[entry] + val_p[candidate]
                        sym_val[sym_row[other] + offset[other]] = val_p[entry] + val_p[candidate]
            if not present:
                sym_col[sym_row[node] + offset[node]] = other
                sym_col[sym_row[other] + offset[other]] = node
                sym_val[sym_row[node] + offset[node]] = val_p[entry]
                sym_val[sym_row[other] + offset[other]] = val_p[entry]
            if not present or node <= other:
                offset[node] += 1
                if other != node:
                    offset[other] += 1
    for entry in range(total):
        sym_val[entry] /= 2.0
    return sym_row, sym_col, sym_val


class _SPTree:
    """Port of the reference Barnes-Hut space-partitioning tree."""

    __slots__ = (
        "dimension",
        "corner",
        "width",
        "data",
        "is_leaf",
        "size",
        "cum_size",
        "index",
        "children",
        "no_children",
        "center_of_mass",
    )

    def __init__(
        self,
        dimension: int,
        data: list[float],
        corner: list[float],
        width: list[float],
    ) -> None:
        """Initialize one tree cell.

        Parameters
        ----------
        dimension : int
            Embedding dimensionality.
        data : list[float]
            Flat node-major coordinate buffer shared by the whole tree.
        corner : list[float]
            Cell center per dimension.
        width : list[float]
            Cell half-width per dimension.

        Returns
        -------
        None
            The cell is initialized empty.
        """
        self.dimension = dimension
        self.no_children = 2**dimension
        self.data = data
        self.is_leaf = True
        self.size = 0
        self.cum_size = 0
        self.corner = list(corner)
        self.width = list(width)
        self.children: list[Optional["_SPTree"]] = [None] * self.no_children
        self.center_of_mass = [0.0] * dimension
        self.index = [0]

    @classmethod
    def build(cls, dimension: int, data: list[float], count: int) -> "_SPTree":
        """Build a filled tree following the reference constructor.

        Parameters
        ----------
        dimension : int
            Embedding dimensionality.
        data : list[float]
            Flat node-major coordinate buffer.
        count : int
            Number of points.

        Returns
        -------
        _SPTree
            Filled tree.
        """
        mean = [0.0] * dimension
        minimum = [_DBL_MAX] * dimension
        maximum = [-_DBL_MAX] * dimension
        offset = 0
        for _ in range(count):
            for dim in range(dimension):
                value = data[offset + dim]
                mean[dim] += value
                if value < minimum[dim]:
                    minimum[dim] = value
                if value > maximum[dim]:
                    maximum[dim] = value
            offset += dimension
        for dim in range(dimension):
            mean[dim] /= float(count)
        width = [
            max(maximum[dim] - mean[dim], mean[dim] - minimum[dim]) + 1.0e-5
            for dim in range(dimension)
        ]
        tree = cls(dimension, data, mean, width)
        for point in range(count):
            tree.insert(point)
        return tree

    def _contains(self, offset: int) -> bool:
        """Check whether a point lies inside this cell.

        Parameters
        ----------
        offset : int
            Flat offset of the point in the data buffer.

        Returns
        -------
        bool
            True when the point is inside the cell bounds.
        """
        data = self.data
        for dim in range(self.dimension):
            value = data[offset + dim]
            if self.corner[dim] - self.width[dim] > value:
                return False
            if self.corner[dim] + self.width[dim] < value:
                return False
        return True

    def insert(self, new_index: int) -> bool:
        """Insert a point following the reference insertion rules.

        Parameters
        ----------
        new_index : int
            Point index to insert.

        Returns
        -------
        bool
            True when the point was placed in this subtree.
        """
        dimension = self.dimension
        offset = new_index * dimension
        if not self._contains(offset):
            return False
        self.cum_size += 1
        mult1 = float(self.cum_size - 1) / float(self.cum_size)
        mult2 = 1.0 / float(self.cum_size)
        center = self.center_of_mass
        data = self.data
        for dim in range(dimension):
            center[dim] *= mult1
        for dim in range(dimension):
            center[dim] += mult2 * data[offset + dim]
        if self.is_leaf and self.size < 1:
            self.index[0] = new_index
            self.size += 1
            return True
        any_duplicate = False
        for slot in range(self.size):
            duplicate = True
            existing = self.index[slot] * dimension
            for dim in range(dimension):
                if data[offset + dim] != data[existing + dim]:
                    duplicate = False
                    break
            any_duplicate = any_duplicate or duplicate
        if any_duplicate:
            return True
        if self.is_leaf:
            self._subdivide()
        for child_id in range(self.no_children):
            child = self.children[child_id]
            assert child is not None
            if child.insert(new_index):
                return True
        return False

    def _subdivide(self) -> None:
        """Split this cell into children and reinsert stored points.

        Returns
        -------
        None
            Children are created and points moved.
        """
        dimension = self.dimension
        for child_id in range(self.no_children):
            divider = 1
            corner = [0.0] * dimension
            width = [0.0] * dimension
            for dim in range(dimension):
                width[dim] = 0.5 * self.width[dim]
                if (child_id // divider) % 2 == 1:
                    corner[dim] = self.corner[dim] - 0.5 * self.width[dim]
                else:
                    corner[dim] = self.corner[dim] + 0.5 * self.width[dim]
                divider *= 2
            self.children[child_id] = _SPTree(dimension, self.data, corner, width)
        for slot in range(self.size):
            success = False
            for child_id in range(self.no_children):
                child = self.children[child_id]
                assert child is not None
                if not success:
                    success = child.insert(self.index[slot])
        self.size = 0
        self.is_leaf = False

    def compute_non_edge_forces(
        self,
        point_index: int,
        theta: float,
        negative: list[float],
        negative_offset: int,
        sum_q: float,
        w_kl: float,
    ) -> float:
        """Accumulate Barnes-Hut repulsive forces for one point.

        Parameters
        ----------
        point_index : int
            Query point index.
        theta : float
            Barnes-Hut opening threshold.
        negative : list[float]
            Flat repulsive-force buffer.
        negative_offset : int
            Offset of the query point in ``negative``.
        sum_q : float
            Running normalization accumulator.
        w_kl : float
            KL weight applied to the accumulated force.

        Returns
        -------
        float
            Updated normalization accumulator.
        """
        if self.cum_size == 0 or (self.is_leaf and self.size == 1 and self.index[0] == point_index):
            return sum_q
        dimension = self.dimension
        data = self.data
        offset = point_index * dimension
        center = self.center_of_mass
        distance = 0.0
        buffer = [0.0] * dimension
        for dim in range(dimension):
            buffer[dim] = data[offset + dim] - center[dim]
        for dim in range(dimension):
            distance += buffer[dim] * buffer[dim]
        max_width = 0.0
        for dim in range(dimension):
            current_width = self.width[dim]
            if current_width > max_width:
                max_width = current_width
        use_summary = self.is_leaf
        if not use_summary:
            root = math.sqrt(distance)
            if root != 0.0:
                use_summary = max_width / root < theta
            else:
                use_summary = False
        if use_summary:
            distance = 1.0 / (1.0 + distance)
            mult = self.cum_size * distance
            sum_q += mult
            mult *= distance
            for dim in range(dimension):
                negative[negative_offset + dim] += mult * buffer[dim] * w_kl
        else:
            for child_id in range(self.no_children):
                child = self.children[child_id]
                assert child is not None
                sum_q = child.compute_non_edge_forces(
                    point_index,
                    theta,
                    negative,
                    negative_offset,
                    sum_q,
                    w_kl,
                )
        return sum_q


def _compute_edge_forces(
    data: list[float],
    row_p: list[int],
    col_p: list[int],
    val_p: list[float],
    count: int,
    dimension: int,
    positive: list[float],
    w_kl: float,
    w_r: float,
) -> None:
    """Port of ``SPTree::computeEdgeForces`` (single-thread order).

    Parameters
    ----------
    data : list[float]
        Flat node-major coordinates.
    row_p : list[int]
        CSR row offsets.
    col_p : list[int]
        CSR column ids.
    val_p : list[float]
        CSR values.
    count : int
        Number of nodes.
    dimension : int
        Embedding dimensionality.
    positive : list[float]
        Flat attractive-force output buffer.
    w_kl : float
        KL weight.
    w_r : float
        Repulsion weight for the tsNET term.

    Returns
    -------
    None
        Forces accumulate into ``positive``.
    """
    denominator_scale = 2 * count * count
    for node in range(count):
        ind1 = node * dimension
        for entry in range(row_p[node], row_p[node + 1]):
            distance = 1.0
            ind2 = col_p[entry] * dimension
            buffer = [0.0] * dimension
            for dim in range(dimension):
                buffer[dim] = data[ind1 + dim] - data[ind2 + dim]
            for dim in range(dimension):
                distance += buffer[dim] * buffer[dim]
            distance = val_p[entry] / distance
            if w_r == 0.0:
                for dim in range(dimension):
                    positive[ind1 + dim] += distance * buffer[dim] * w_kl
            else:
                for dim in range(dimension):
                    shift = 1.0 / 20 if buffer[dim] > 0 else -1.0 / 20
                    denominator = buffer[dim] + shift
                    if denominator == 0.0:
                        inverse = math.inf
                    else:
                        inverse = 1.0 / denominator
                    positive[ind1 + dim] += (
                        distance * buffer[dim] * w_kl - inverse * w_r / denominator_scale
                    )


def _evaluate_error(
    data: list[float],
    row_p: list[int],
    col_p: list[int],
    val_p: list[float],
    count: int,
    dimension: int,
    theta: float,
    w_kl: float,
    w_c: float,
    w_r: float,
) -> float:
    """Port of the Barnes-Hut ``evaluateError`` used for phase control.

    Parameters
    ----------
    data : list[float]
        Flat node-major coordinates.
    row_p : list[int]
        CSR row offsets.
    col_p : list[int]
        CSR column ids.
    val_p : list[float]
        CSR values.
    count : int
        Number of nodes.
    dimension : int
        Embedding dimensionality.
    theta : float
        Barnes-Hut opening threshold.
    w_kl : float
        KL weight.
    w_c : float
        Compression weight.
    w_r : float
        Repulsion weight.

    Returns
    -------
    float
        tsNET objective estimate.
    """
    tree = _SPTree.build(dimension, data, count)
    scratch = [0.0] * dimension
    sum_q = 0.0
    for node in range(count):
        sum_q = tree.compute_non_edge_forces(node, theta, scratch, 0, sum_q, 1.0)
    error = 0.0
    repulsion = 0.0
    for node in range(count):
        ind1 = node * dimension
        for entry in range(row_p[node], row_p[node + 1]):
            quotient = 0.0
            ind2 = col_p[entry] * dimension
            for dim in range(dimension):
                scratch[dim] = data[ind1 + dim]
            for dim in range(dimension):
                scratch[dim] -= data[ind2 + dim]
            for dim in range(dimension):
                quotient += scratch[dim] * scratch[dim]
            repulsion += math.log(math.sqrt(quotient) + 1.0 / 20.0)
            quotient = (1.0 / (1.0 + quotient)) / sum_q
            error += val_p[entry] * math.log((val_p[entry] + _FLT_MIN) / (quotient + _FLT_MIN))
    repulsion /= 2 * count * count
    compression = 0.0
    offset = 0
    for _ in range(count):
        total = 0.0
        for _ in range(dimension):
            total += data[offset] * data[offset]
            offset += 1
        compression += total
    compression /= 2 * count
    return error * w_kl + compression * w_c + repulsion * w_r


def reference_tsnet_star(
    adjacency: list[list[int]],
    initial: np.ndarray,
    perplexity: float,
    theta: float,
    max_iter: int,
) -> np.ndarray:
    """Port of the reference tsNET* Barnes-Hut optimization loop.

    Parameters
    ----------
    adjacency : list[list[int]]
        Neighbor lists in reference order.
    initial : numpy.ndarray
        PivotMDS initialization with shape ``[N, 2]`` (float64).
    perplexity : float
        Requested perplexity before the reference small-graph adjustment.
    theta : float
        Barnes-Hut opening threshold.
    max_iter : int
        Iterations per phase (reference ``iterations / 2``).

    Returns
    -------
    numpy.ndarray
        Teacher coordinates with shape ``[N, 2]`` (float64).
    """
    count = len(adjacency)
    dimension = int(initial.shape[1])
    if count - 1 < 3 * perplexity:
        perplexity = float((count - 1) // 3 - 1)
    neighbors = int(3 * perplexity)

    w_kl = 1.0
    w_c = 0.1
    w_r = 0.0
    momentum = 0.5
    final_momentum = 0.7
    eta = 200.0

    positions = np.ascontiguousarray(initial, dtype=np.float64).reshape(-1).copy()
    velocity = np.zeros(count * dimension, dtype=np.float64)
    gains = np.ones(count * dimension, dtype=np.float64)

    row_p, col_p, val_p = _gaussian_perplexity_sparse(
        adjacency,
        count,
        perplexity,
        neighbors,
    )
    row_p, col_p, val_p = _symmetrize_sparse(row_p, col_p, val_p, count)
    total = 0.0
    for entry in range(row_p[count]):
        total += val_p[entry]
    for entry in range(row_p[count]):
        val_p[entry] /= total

    stop_lying_iter = -1
    mom_switch_iter = 250
    previous_error = _DBL_MAX
    switched = False

    while True:
        iteration = 0
        while iteration < max_iter:
            data = positions.tolist()
            tree = _SPTree.build(dimension, data, count)
            positive = [0.0] * (count * dimension)
            negative = [0.0] * (count * dimension)
            _compute_edge_forces(
                data,
                row_p,
                col_p,
                val_p,
                count,
                dimension,
                positive,
                w_kl,
                w_r,
            )
            sum_q = 0.0
            for node in range(count):
                sum_q = tree.compute_non_edge_forces(
                    node,
                    theta,
                    negative,
                    node * dimension,
                    sum_q,
                    w_kl,
                )
            gradient = (
                np.asarray(positive, dtype=np.float64)
                - (np.asarray(negative, dtype=np.float64) / sum_q)
                + positions * w_c / count
            )
            gains = np.where(
                np.sign(gradient) != np.sign(velocity),
                gains + 0.2,
                gains * 0.8,
            )
            gains = np.where(gains < 0.01, 0.01, gains)
            velocity = momentum * velocity - eta * gains * gradient
            positions = positions + velocity
            means = [0.0] * dimension
            offset = 0
            position_list = positions.tolist()
            for _ in range(count):
                for dim in range(dimension):
                    means[dim] += position_list[offset + dim]
                offset += dimension
            for dim in range(dimension):
                means[dim] /= float(count)
            positions = positions - np.asarray(
                [means[dim] for dim in range(dimension)] * count,
                dtype=np.float64,
            ).reshape(count, dimension).reshape(-1)
            if iteration == mom_switch_iter:
                momentum = final_momentum
            if iteration % 50 == 0 or iteration == max_iter - 1:
                current_error = _evaluate_error(
                    positions.tolist(),
                    row_p,
                    col_p,
                    val_p,
                    count,
                    dimension,
                    theta,
                    w_kl,
                    w_c,
                    w_r,
                )
                if previous_error - current_error < 0.00001 and stop_lying_iter + 5 < iteration:
                    if switched:
                        return positions.reshape(count, dimension)
                    w_kl = 1.0
                    w_c = 0.01
                    w_r = 0.6
                    iteration = -1
                    switched = True
                    current_error = _DBL_MAX
                    momentum = 0.5
                    stop_lying_iter = -1
                previous_error = current_error
            iteration += 1
        if not switched:
            w_kl = 1.0
            w_c = 0.01
            w_r = 0.6
            switched = True
            momentum = 0.5
            continue
        break
    return positions.reshape(count, dimension)


def is_connected(adjacency: list[list[int]]) -> bool:
    """Check connectivity from node 0 like the reference loader.

    Parameters
    ----------
    adjacency : list[list[int]]
        Neighbor lists.

    Returns
    -------
    bool
        True when all nodes are reachable from node 0.
    """
    if not adjacency:
        return True
    distances = _bfs_distances(adjacency, 0)
    return bool((distances != -1.0).all())


__all__ = [
    "canonical_adjacency",
    "glibc_rand_stream",
    "graph_normalize",
    "is_connected",
    "reference_pivot_embedding",
    "reference_pmds",
    "reference_tsnet_star",
]
