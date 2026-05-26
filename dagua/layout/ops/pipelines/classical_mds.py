"""Classical multidimensional scaling layout pipeline."""

from __future__ import annotations

import ctypes
import math
from typing import Optional

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.distance import ClassicalMDSDistanceMatrix
from dagua.layout.ops.embed import ClassicalMDSComputeEmbedding, ClassicalMDSComputeEmbeddingConfig
from dagua.layout.ops.pipelines import resolve_fidelity_dtype
from dagua.layout.ops.postprocess import (
    ClassicalMDSFinalizePositions,
    ClassicalMDSFinalizePositionsConfig,
)
from dagua.layout.ops.state import (
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)

_OGDF_EDGE_COST = 100.0
_OGDF_POWER_EPSILON = 1.0 - 1e-10
_OGDF_CENTERING_FACTOR = -0.5
_LIBC_RAND_MAX = 2_147_483_647


def build_classical_mds_pipeline(
    *,
    igraph_fidelity: bool = False,
    ogdf_fidelity: bool = False,
) -> Pipeline:
    """Build a classical multidimensional scaling pipeline.

    Reference fidelity
    ------------------
    Targets: igraph 1.0.0 MDS layout / Torgerson (1952) classical metric MDS.
    Fidelity mode: ``igraph_fidelity=True`` uses igraph-compatible raw
        embedding and final scaling semantics.
    Verified at: final 100-seed report, strong equivalent; median RMSD 0.000
        for both default and igraph-fidelity variants.
    Known divergences:
        - Graph distances are prepared in Dagua tensor ops rather than through
          igraph's C path.
        - Disconnected-graph behavior follows the benchmark distance matrix
          contract, not arbitrary user-provided dissimilarities.

    Parameters
    ----------
    igraph_fidelity : bool, default=False
        If ``True``, opt into igraph-compatible raw embedding and final scaling
        semantics for benchmark parity checks.
    ogdf_fidelity : bool, default=False
        Reserved for the public wrapper's OGDF-compatible full-PivotMDS path.

    Returns
    -------
    Pipeline
        Pipeline implementing classical MDS. The pipeline produces final node
        coordinates by computing the all-pairs graph distance matrix, solving
        the double-centered eigendecomposition, and finalizing the embedding
        into a 2D layout.
    """
    if ogdf_fidelity:
        raise ValueError("ogdf_fidelity is only supported by layout_classical_mds_pipeline.")
    return Pipeline(
        [
            ClassicalMDSDistanceMatrix(),
            ClassicalMDSComputeEmbedding(
                config=ClassicalMDSComputeEmbeddingConfig(igraph_fidelity=igraph_fidelity)
            ),
            ClassicalMDSFinalizePositions(
                config=ClassicalMDSFinalizePositionsConfig(igraph_fidelity=igraph_fidelity)
            ),
        ],
        name="classical_mds_pipeline",
    )


def layout_classical_mds_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
    igraph_fidelity: bool = False,
    ogdf_fidelity: bool = False,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run the classical multidimensional scaling pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]`` used only to pick a
        stable output extent.
    seed : int, default=42
        Accepted for interface compatibility. Classical MDS is deterministic
        once graph distances are fixed.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.
    igraph_fidelity : bool, default=False
        If ``True``, ignore edge weights and use igraph-compatible embedding
        and scaling semantics. This is intended for fidelity benchmarking
        against ``igraph.layout("mds")``.
    ogdf_fidelity : bool, default=False
        If ``True``, run OGDF ``PivotMDS`` with all nodes as pivots. This
        matches OGDF's documented classical-MDS mode and uses uniform edge cost
        ``100`` plus OGDF's fixed ``srand(0)`` power-iteration basis.
    fidelity_dtype : torch.dtype, optional
        Internal and returned dtype for fidelity-mode comparisons. ``None``
        defaults to ``torch.float64`` when fidelity mode is selected.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If ``num_nodes`` or ``edge_weights`` are invalid.
    RuntimeError
        If the pipeline does not produce final positions.
    """
    _ = seed, node_sizes

    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if igraph_fidelity and ogdf_fidelity:
        raise ValueError("igraph_fidelity and ogdf_fidelity are mutually exclusive.")
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must have shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge count {edge_index.shape[1]}"
            )
    if ogdf_fidelity:
        return _layout_ogdf_classical_mds(
            edge_index=edge_index,
            num_nodes=num_nodes,
            fidelity_dtype=resolve_fidelity_dtype(True, fidelity_dtype),
        )

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=None if igraph_fidelity else edge_weights,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_classical_mds_pipeline(igraph_fidelity=igraph_fidelity).apply(
        problem,
        state,
        ctx,
    )
    if final_state.pos is None:
        raise RuntimeError("Classical MDS pipeline did not produce final positions.")
    return final_state.pos


def _layout_ogdf_classical_mds(
    edge_index: torch.Tensor,
    num_nodes: int,
    fidelity_dtype: torch.dtype,
) -> torch.Tensor:
    """Run OGDF's all-pivots PivotMDS implementation.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes ``N``.
    fidelity_dtype : torch.dtype
        Output dtype for fidelity comparisons.

    Returns
    -------
    torch.Tensor
        Raw OGDF-compatible coordinates with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If the graph is disconnected, matching OGDF PivotMDS' connected-graph
        precondition.
    RuntimeError
        If OGDF's power iteration diverges numerically.
    """
    if num_nodes == 0:
        return torch.zeros((0, 2), dtype=fidelity_dtype)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=fidelity_dtype)

    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    edges = edge_index.detach().to(device="cpu", dtype=torch.long)
    for edge_pos in range(int(edges.shape[1])):
        source = int(edges[0, edge_pos].item())
        target = int(edges[1, edge_pos].item())
        if source == target:
            continue
        adjacency[source].append(target)
        adjacency[target].append(source)

    simple_neighbors = [list(dict.fromkeys(neighbors)) for neighbors in adjacency]
    endpoints = [node for node, neighbors in enumerate(simple_neighbors) if len(neighbors) == 1]
    if len(endpoints) == 2 and not any(
        len(neighbors) > 2 or len(neighbors) == 0 for neighbors in simple_neighbors
    ):
        positions = torch.zeros((num_nodes, 2), dtype=fidelity_dtype)
        previous = -1
        current = endpoints[0]
        for path_index in range(num_nodes):
            positions[current, 0] = float(path_index) * _OGDF_EDGE_COST
            next_nodes = [
                neighbor for neighbor in simple_neighbors[current] if neighbor != previous
            ]
            previous, current = current, next_nodes[0] if next_nodes else -1
            if current == -1 and path_index == num_nodes - 1:
                return positions

    visited = [False] * num_nodes
    queue = [0]
    visited[0] = True
    head = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        for neighbor in adjacency[node]:
            if visited[neighbor]:
                continue
            visited[neighbor] = True
            queue.append(neighbor)
    if not all(visited):
        raise ValueError("OGDF classical MDS fidelity requires a connected graph.")

    pivot_matrix: list[list[float]] = []
    min_distances = [math.inf] * num_nodes
    pivot_node = 0
    for _ in range(num_nodes):
        distances = [math.inf] * num_nodes
        distances[pivot_node] = 0.0
        queue = [pivot_node]
        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            next_distance = distances[node] + _OGDF_EDGE_COST
            for neighbor in adjacency[node]:
                if math.isfinite(distances[neighbor]):
                    continue
                distances[neighbor] = next_distance
                queue.append(neighbor)
        pivot_matrix.append(distances)
        min_distances[pivot_node] = 0.0
        for node in range(num_nodes):
            min_distances[node] = min(min_distances[node], distances[node])
            if min_distances[node] > min_distances[pivot_node]:
                pivot_node = node

    normalization_factor = 0.0
    col_normalization = [0.0] * num_nodes
    for pivot_idx in range(num_nodes):
        row_col_normalizer = 0.0
        for node_idx in range(num_nodes):
            row_col_normalizer += (
                pivot_matrix[pivot_idx][node_idx] * pivot_matrix[pivot_idx][node_idx]
            )
        normalization_factor += row_col_normalizer
        col_normalization[pivot_idx] = row_col_normalizer / float(num_nodes)
    normalization_factor /= float(num_nodes * num_nodes)
    for node_idx in range(num_nodes):
        row_col_normalizer = 0.0
        for pivot_idx in range(num_nodes):
            square = pivot_matrix[pivot_idx][node_idx] * pivot_matrix[pivot_idx][node_idx]
            pivot_matrix[pivot_idx][node_idx] = (
                square + normalization_factor - col_normalization[pivot_idx]
            )
            row_col_normalizer += square
        row_col_normalizer /= float(num_nodes)
        for pivot_idx in range(num_nodes):
            pivot_matrix[pivot_idx][node_idx] = _OGDF_CENTERING_FACTOR * (
                pivot_matrix[pivot_idx][node_idx] - row_col_normalizer
            )

    product_matrix = [[0.0 for _ in range(num_nodes)] for _ in range(num_nodes)]
    for row_idx in range(num_nodes):
        for col_idx in range(row_idx + 1):
            total = 0.0
            for node_idx in range(num_nodes):
                total += pivot_matrix[row_idx][node_idx] * pivot_matrix[col_idx][node_idx]
            product_matrix[row_idx][col_idx] = total
            product_matrix[col_idx][row_idx] = total

    libc = ctypes.CDLL(None)
    libc.srand(0)
    eigenvectors = [
        [float(libc.rand()) / float(_LIBC_RAND_MAX) for _ in range(num_nodes)] for _ in range(2)
    ]
    eigenvalues = [0.0, 0.0]
    for dim_idx in range(2):
        norm = math.sqrt(sum(value * value for value in eigenvectors[dim_idx]))
        eigenvalues[dim_idx] = norm
        if norm != 0.0:
            eigenvectors[dim_idx] = [value / norm for value in eigenvectors[dim_idx]]

    convergence = 0.0
    while convergence < _OGDF_POWER_EPSILON:
        if math.isnan(convergence) or math.isinf(convergence):
            raise RuntimeError("OGDF classical MDS power iteration diverged.")
        previous = [row.copy() for row in eigenvectors]
        eigenvectors = [[0.0 for _ in range(num_nodes)] for _ in range(2)]
        for dim_idx in range(2):
            for row_idx in range(num_nodes):
                for col_idx in range(num_nodes):
                    eigenvectors[dim_idx][col_idx] += (
                        product_matrix[row_idx][col_idx] * previous[dim_idx][row_idx]
                    )
        denominator = sum(value * value for value in eigenvectors[0])
        factor = (
            sum(eigenvectors[0][index] * eigenvectors[1][index] for index in range(num_nodes))
            / denominator
        )
        for node_idx in range(num_nodes):
            eigenvectors[1][node_idx] -= factor * eigenvectors[0][node_idx]
        for dim_idx in range(2):
            norm = math.sqrt(sum(value * value for value in eigenvectors[dim_idx]))
            eigenvalues[dim_idx] = norm
            if norm != 0.0:
                eigenvectors[dim_idx] = [value / norm for value in eigenvectors[dim_idx]]
        convergence = 1.0
        for dim_idx in range(2):
            product = sum(
                eigenvectors[dim_idx][index] * previous[dim_idx][index]
                for index in range(num_nodes)
            )
            convergence = min(convergence, abs(product))

    coordinate_rows = [[0.0 for _ in range(num_nodes)] for _ in range(2)]
    for dim_idx in range(2):
        eigenvalues[dim_idx] = math.sqrt(eigenvalues[dim_idx])
        for node_idx in range(num_nodes):
            for pivot_idx in range(num_nodes):
                coordinate_rows[dim_idx][node_idx] += (
                    pivot_matrix[pivot_idx][node_idx] * eigenvectors[dim_idx][pivot_idx]
                )
    for dim_idx in range(2):
        norm = math.sqrt(sum(value * value for value in coordinate_rows[dim_idx]))
        if norm != 0.0:
            coordinate_rows[dim_idx] = [value / norm for value in coordinate_rows[dim_idx]]
    for dim_idx in range(2):
        eigenvalues[dim_idx] = math.sqrt(eigenvalues[dim_idx])
        for node_idx in range(num_nodes):
            coordinate_rows[dim_idx][node_idx] *= eigenvalues[dim_idx]

    return torch.tensor(
        [
            [coordinate_rows[0][node_idx], coordinate_rows[1][node_idx]]
            for node_idx in range(num_nodes)
        ],
        dtype=fidelity_dtype,
    )


__all__ = ["build_classical_mds_pipeline", "layout_classical_mds_pipeline"]
