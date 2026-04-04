"""Kamada-Kawai layout translated from NetworkX's reference implementation."""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import torch

from dagua.layout._archive.classic._graph_distances import (
    bfs_distances,
    build_directed_adjacency,
    dijkstra_distances,
)

UNREACHABLE_DISTANCE = 1.0e6
DISTANCE_EPSILON = 1.0e-3
CENTERING_WEIGHT = 1.0e-3


def _layout_device(
    edge_index: torch.Tensor,
    node_sizes: Optional[torch.Tensor],
) -> torch.device:
    """Choose the device used for the returned tensor.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor.

    Returns
    -------
    torch.device
        Output device for the final layout tensor.
    """
    if edge_index.numel() > 0:
        return edge_index.device
    if node_sizes is not None:
        return node_sizes.device
    return torch.device("cpu")


def _shortest_path_distance_matrix(
    adjacency: list[list[tuple[int, float]]],
    weighted: bool,
) -> np.ndarray:
    """Compute directed all-pairs shortest-path distances.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Directed adjacency list.
    weighted : bool
        Whether to compute weighted shortest paths with Dijkstra.

    Returns
    -------
    numpy.ndarray
        Distance matrix with shape ``[N, N]`` and ``1e6`` for unreachable pairs.
    """
    num_nodes = len(adjacency)
    distances = np.full((num_nodes, num_nodes), UNREACHABLE_DISTANCE, dtype=np.float64)
    for source in range(num_nodes):
        if weighted:
            source_distances = dijkstra_distances(adjacency, source)
            source_distances[np.isinf(source_distances)] = UNREACHABLE_DISTANCE
            distances[source] = source_distances
            continue

        source_distances = bfs_distances(adjacency, source).astype(np.float64)
        source_distances[source_distances < 0] = UNREACHABLE_DISTANCE
        distances[source] = source_distances
    return distances


def _rescale_layout(positions: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Center and scale coordinates like ``networkx.rescale_layout``.

    Parameters
    ----------
    positions : numpy.ndarray
        Position array with shape ``[N, 2]``.
    scale : float, default=1.0
        Target half-width after rescaling.

    Returns
    -------
    numpy.ndarray
        Rescaled coordinates.
    """
    positions = positions.copy()
    positions -= positions.mean(axis=0)
    limit = np.abs(positions).max()
    if limit > 0:
        positions *= scale / limit
    return positions


def _circular_layout(num_nodes: int) -> np.ndarray:
    """Create the exact 2D circular initialization used by NetworkX.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.

    Returns
    -------
    numpy.ndarray
        Circular coordinates with shape ``[N, 2]``.
    """
    if num_nodes == 0:
        return np.empty((0, 2), dtype=np.float64)
    if num_nodes == 1:
        return np.zeros((1, 2), dtype=np.float64)

    theta = np.linspace(0, 1, num_nodes + 1)[:-1] * (2.0 * np.pi)
    theta = theta.astype(np.float32)
    positions = np.column_stack((np.cos(theta), np.sin(theta)))
    return _rescale_layout(positions.astype(np.float64, copy=False), scale=1.0)


def _kamada_kawai_costfn(
    pos_vec: np.ndarray,
    np_module: Any,
    invdist: np.ndarray,
    meanweight: float,
    dim: int,
) -> tuple[float, np.ndarray]:
    """Compute the exact NetworkX Kamada-Kawai objective and gradient.

    Parameters
    ----------
    pos_vec : numpy.ndarray
        Flattened position vector with shape ``[N * dim]``.
    np_module : Any
        NumPy module passed through to mirror the NetworkX helper signature.
    invdist : numpy.ndarray
        Inverse preferred-distance matrix with shape ``[N, N]``.
    meanweight : float
        Weight of the origin-centering penalty.
    dim : int
        Layout dimension.

    Returns
    -------
    tuple[float, numpy.ndarray]
        Objective value and flattened analytic gradient.
    """
    num_nodes = invdist.shape[0]
    pos_arr = pos_vec.reshape((num_nodes, dim))

    delta = pos_arr[:, np_module.newaxis, :] - pos_arr[np_module.newaxis, :, :]
    node_separation = np_module.linalg.norm(delta, axis=-1)
    direction = np_module.einsum(
        "ijk,ij->ijk",
        delta,
        1.0 / (node_separation + np_module.eye(num_nodes) * DISTANCE_EPSILON),
    )

    offset = (node_separation * invdist) - 1.0
    offset[np_module.diag_indices(num_nodes)] = 0.0

    cost = 0.5 * float(np_module.sum(offset**2))
    gradient = np_module.einsum(
        "ij,ij,ijk->ik",
        invdist,
        offset,
        direction,
    ) - np_module.einsum(
        "ij,ij,ijk->jk",
        invdist,
        offset,
        direction,
    )

    sum_positions = np_module.sum(pos_arr, axis=0)
    cost += 0.5 * meanweight * float(np_module.sum(sum_positions**2))
    gradient += meanweight * sum_positions
    return cost, gradient.ravel()


def _solve_kamada_kawai(
    distance_matrix: np.ndarray,
    initial_positions: np.ndarray,
    steps: Optional[int],
    trace_every: int,
) -> tuple[np.ndarray, list[torch.Tensor]]:
    """Run the exact SciPy L-BFGS-B solver used by NetworkX KK.

    Parameters
    ----------
    distance_matrix : numpy.ndarray
        Preferred graph distances with shape ``[N, N]``.
    initial_positions : numpy.ndarray
        Initial coordinates with shape ``[N, 2]``.
    steps : int, optional
        Maximum optimization iterations. ``None`` or ``0`` mirrors the
        uncapped NetworkX behavior by leaving ``maxiter`` unset.
    trace_every : int
        Callback snapshot cadence.

    Returns
    -------
    tuple[numpy.ndarray, list[torch.Tensor]]
        Final coordinates and optional trace snapshots.
    """
    try:
        import scipy as sp
    except ImportError as error:
        raise ImportError("layout_kk requires scipy to match NetworkX exactly.") from error

    inverse_distances = 1.0 / (
        distance_matrix + np.eye(distance_matrix.shape[0], dtype=np.float64) * DISTANCE_EPSILON
    )
    traces: list[torch.Tensor] = []
    iteration = 0

    def _callback(pos_vec: np.ndarray) -> None:
        """Collect optimizer traces without affecting the solve."""
        nonlocal iteration

        iteration += 1
        if trace_every > 0 and iteration % trace_every == 0:
            snapshot = pos_vec.reshape((-1, 2)).copy()
            traces.append(torch.from_numpy(snapshot).to(dtype=torch.float32))

    minimize_kwargs: dict[str, Any] = {
        "method": "L-BFGS-B",
        "args": (np, inverse_distances, CENTERING_WEIGHT, 2),
        "jac": True,
        "callback": _callback,
    }
    if steps not in {None, 0}:
        minimize_kwargs["options"] = {"maxiter": steps}

    optresult = sp.optimize.minimize(
        _kamada_kawai_costfn,
        initial_positions.ravel(),
        **minimize_kwargs,
    )
    return optresult.x.reshape((-1, 2)), traces


def layout_kk(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: Optional[int] = None,
    seed: int = 42,
    trace_every: int = 0,
    solver: str = "auto",
    pos: Optional[torch.Tensor] = None,
    edge_weights: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, list[torch.Tensor]]]:
    """Lay out a graph with the NetworkX Kamada-Kawai reference algorithm.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Unused, accepted for interface compatibility.
    steps : int, optional
        Maximum L-BFGS-B iterations. ``None`` or ``0`` leaves SciPy's
        ``maxiter`` unset to match NetworkX's default solve budget.
    seed : int, default=42
        Accepted for interface compatibility. The translated 2D NetworkX path
        uses deterministic circular initialization and does not consume a seed.
    trace_every : int, default=0
        If greater than zero, record optimizer snapshots every
        ``trace_every`` callback invocations.
    solver : {"auto", "newton", "adam"}, default="auto"
        Retained for interface compatibility. All accepted values resolve to
        the exact SciPy L-BFGS-B NetworkX solver path.
    pos : torch.Tensor, optional
        Initial positions with shape ``[N, 2]``. When provided, replaces the
        circular initialization. Must be convertible to ``float64``.
    edge_weights : torch.Tensor, optional
        Optional per-edge weights with shape ``[E]``. When provided, directed
        shortest-path targets are computed with Dijkstra instead of BFS.

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, list[torch.Tensor]]
        Final positions with shape ``[N, 2]``. When tracing is enabled,
        returns ``(positions, traces)``.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``steps``, ``trace_every``, or ``solver`` are invalid.
    ImportError
        If SciPy is unavailable.
    """
    _ = node_sizes
    _ = seed

    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if steps is not None and steps < 0:
        raise ValueError("steps must be non-negative.")
    if trace_every < 0:
        raise ValueError("trace_every must be non-negative.")
    if solver not in {"auto", "newton", "adam"}:
        raise ValueError("solver must be one of 'auto', 'newton', or 'adam'.")

    device = _layout_device(edge_index=edge_index, node_sizes=node_sizes)
    if num_nodes == 0:
        empty = torch.empty((0, 2), dtype=torch.float32, device=device)
        return (empty, []) if trace_every > 0 else empty
    if num_nodes == 1:
        single = torch.zeros((1, 2), dtype=torch.float32, device=device)
        return (single, []) if trace_every > 0 else single

    adjacency = build_directed_adjacency(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weights=edge_weights,
    )
    distance_matrix = _shortest_path_distance_matrix(
        adjacency,
        weighted=edge_weights is not None,
    )
    if pos is not None:
        if pos.shape != (num_nodes, 2):
            raise ValueError(f"pos must have shape ({num_nodes}, 2), got {tuple(pos.shape)}")
        initial_positions = pos.detach().cpu().numpy().astype(np.float64)
    else:
        initial_positions = _circular_layout(num_nodes=num_nodes)
    solved_positions, traces = _solve_kamada_kawai(
        distance_matrix=distance_matrix,
        initial_positions=initial_positions,
        steps=steps,
        trace_every=trace_every,
    )
    final_positions = torch.from_numpy(_rescale_layout(solved_positions, scale=1.0)).to(
        dtype=torch.float32,
        device=device,
    )

    if trace_every > 0:
        device_traces = [trace.to(device=device) for trace in traces]
        return final_positions, device_traces
    return final_positions
