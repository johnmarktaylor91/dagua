"""Kamada-Kawai spring layout based on graph-theoretic distances.

This implementation matches NetworkX's stress objective and default SciPy
solver path while preserving Dagua's existing public signature. The default
solver uses full-graph L-BFGS-B when SciPy is available; the older per-node
Newton-Raphson solver remains as the fallback path.

Reference: Kamada & Kawai, "An Algorithm for Drawing General Graphs" (1989),
Information Processing Letters.
"""

from __future__ import annotations

import importlib.util
from collections import deque
from math import pi
from typing import Optional, Union

import numpy as np
import torch

_PIVOT_THRESHOLD = 5000
_PIVOT_COUNT = 200
_DISTANCE_EPSILON = 1.0e-3
_NEWTON_DISTANCE_EPSILON = 1.0e-6
_NEWTON_GLOBAL_EPSILON = 1.0e-4
_NEWTON_LOCAL_EPSILON = 1.0e-4
_NEWTON_HESSIAN_REGULARIZATION = 1.0e-6
_CENTERING_WEIGHT = 1.0e-3
_UNREACHABLE_DISTANCE_MINIMUM = 1.0


def _scipy_available() -> bool:
    """Report whether SciPy is importable in the current environment.

    Returns
    -------
    bool
        ``True`` when SciPy is installed.
    """
    return importlib.util.find_spec("scipy") is not None


def _build_directed_adjacency(edge_index: torch.Tensor, num_nodes: int) -> list[list[int]]:
    """Build a directed adjacency list from a ``[2, E]`` edge tensor.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    list[list[int]]
        One outgoing-neighbor list per node.

    Raises
    ------
    ValueError
        If ``edge_index`` has an invalid shape or contains an out-of-range node.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")

    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    if edge_index.numel() == 0:
        return adjacency

    edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
    sources = edge_index_cpu[0].tolist()
    targets = edge_index_cpu[1].tolist()

    for source, target in zip(sources, targets):
        if source < 0 or source >= num_nodes or target < 0 or target >= num_nodes:
            raise ValueError("edge_index contains a node index outside [0, num_nodes).")
        adjacency[source].append(target)

    return adjacency


def _bfs_distances(adjacency: list[list[int]], start: int) -> list[int]:
    """Compute directed unweighted shortest-path distances from one source.

    Parameters
    ----------
    adjacency : list[list[int]]
        Directed adjacency list.
    start : int
        Source node index.

    Returns
    -------
    list[int]
        Directed distances from ``start`` with ``-1`` for unreachable nodes.
    """
    num_nodes = len(adjacency)
    distances = [-1] * num_nodes
    distances[start] = 0
    frontier: deque[int] = deque([start])

    while frontier:
        node = frontier.popleft()
        next_distance = distances[node] + 1
        for neighbor in adjacency[node]:
            if distances[neighbor] == -1:
                distances[neighbor] = next_distance
                frontier.append(neighbor)

    return distances


def _replace_unreachable_numpy(distances: np.ndarray) -> np.ndarray:
    """Replace infinite shortest-path entries with a large finite distance.

    Parameters
    ----------
    distances : numpy.ndarray
        Distance matrix with shape ``[N, N]`` and ``np.inf`` for unreachable pairs.

    Returns
    -------
    numpy.ndarray
        Fully finite distance matrix.
    """
    finite_mask = np.isfinite(distances)
    finite_distances = distances[finite_mask]
    max_finite = float(finite_distances.max()) if finite_distances.size > 0 else 0.0
    fill_value = 2.0 * max(max_finite, _UNREACHABLE_DISTANCE_MINIMUM)
    resolved = distances.copy()
    resolved[~finite_mask] = fill_value
    return resolved


def _replace_unreachable_torch(distances: torch.Tensor) -> torch.Tensor:
    """Replace negative BFS sentinels with the Dagua/NX fallback distance.

    Parameters
    ----------
    distances : torch.Tensor
        Distance tensor with shape ``[N, N]`` or ``[N, P]`` and ``-1`` for
        unreachable entries.

    Returns
    -------
    torch.Tensor
        Fully finite distance tensor.
    """
    finite_mask = distances >= 0
    finite_distances = distances[finite_mask]
    max_finite = float(finite_distances.max().item()) if finite_distances.numel() > 0 else 0.0
    fill_value = 2.0 * max(max_finite, _UNREACHABLE_DISTANCE_MINIMUM)
    return torch.where(finite_mask, distances, torch.full_like(distances, fill_value))


def _compute_distance_matrix(
    edge_index: torch.Tensor,
    num_nodes: int,
    adjacency: list[list[int]],
) -> torch.Tensor:
    """Compute the directed all-pairs shortest-path matrix.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    adjacency : list[list[int]]
        Directed adjacency list used by the non-SciPy fallback.

    Returns
    -------
    torch.Tensor
        Distance matrix with shape ``[N, N]`` and dtype ``float32``.
    """
    if _scipy_available():
        from scipy import sparse
        from scipy.sparse.csgraph import shortest_path

        if edge_index.numel() == 0:
            dense = np.full((num_nodes, num_nodes), np.inf, dtype=np.float64)
            np.fill_diagonal(dense, 0.0)
            return torch.from_numpy(_replace_unreachable_numpy(dense)).to(dtype=torch.float32)

        edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
        rows = edge_index_cpu[0].numpy()
        cols = edge_index_cpu[1].numpy()
        data = np.ones(rows.shape[0], dtype=np.float64)
        matrix = sparse.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
        distances = shortest_path(matrix, directed=True, unweighted=True)
        return torch.from_numpy(_replace_unreachable_numpy(distances)).to(dtype=torch.float32)

    rows = [_bfs_distances(adjacency, node) for node in range(num_nodes)]
    distance_matrix = torch.tensor(rows, dtype=torch.float32)
    return _replace_unreachable_torch(distance_matrix)


def _sample_pivot_distances(
    adjacency: list[list[int]],
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate directed all-pairs distances with sampled pivots.

    Parameters
    ----------
    adjacency : list[list[int]]
        Directed adjacency list.
    seed : int
        Random seed for deterministic pivot sampling.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Pivot indices with shape ``[P]`` and pivot distances with shape ``[N, P]``.
    """
    num_nodes = len(adjacency)
    pivot_count = min(_PIVOT_COUNT, num_nodes)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    pivot_indices = torch.randperm(num_nodes, generator=generator)[:pivot_count]

    rows = [_bfs_distances(adjacency, int(pivot.item())) for pivot in pivot_indices]
    pivot_distances = torch.tensor(rows, dtype=torch.float32).transpose(0, 1).contiguous()
    return pivot_indices.to(dtype=torch.long), _replace_unreachable_torch(pivot_distances)


def _target_lengths_and_strengths(distances: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert graph distances into NetworkX-style KK lengths and weights.

    Parameters
    ----------
    distances : torch.Tensor
        Distance tensor with shape ``[N, N]`` or ``[N, P]``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Preferred Euclidean lengths ``L`` and spring strengths ``K``.
    """
    safe_distances = torch.where(distances > 0, distances, torch.ones_like(distances))
    strengths = torch.where(
        distances > 0,
        1.0 / safe_distances.square(),
        torch.zeros_like(distances),
    )
    return distances, strengths


def _initialize_positions(num_nodes: int) -> torch.Tensor:
    """Create the circular initial layout used by NetworkX in 2D.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    torch.Tensor
        Initial positions with shape ``[N, 2]``.
    """
    if num_nodes <= 1:
        return torch.zeros((num_nodes, 2), dtype=torch.float32)

    angles = torch.linspace(0.0, 2.0 * pi, steps=num_nodes + 1, dtype=torch.float32)[:-1]
    return torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)


def _rescale_layout(positions: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Center and scale coordinates like ``networkx.rescale_layout``.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    scale : float, default=1.0
        Target half-width after rescaling.

    Returns
    -------
    torch.Tensor
        Rescaled coordinates with shape ``[N, 2]``.
    """
    centered = positions - positions.mean(dim=0, keepdim=True)
    limit = float(centered.abs().max().item())
    if limit > 0.0:
        centered = centered * (scale / limit)
    return centered


def _resolve_solver(num_nodes: int, solver: str) -> str:
    """Resolve the public solver mode into a concrete implementation name.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    solver : str
        Requested solver mode.

    Returns
    -------
    str
        Concrete solver name.

    Raises
    ------
    ValueError
        If ``solver`` is not one of ``"auto"``, ``"newton"``, or ``"adam"``.
    """
    del num_nodes

    if solver not in {"auto", "newton", "adam"}:
        raise ValueError("solver must be one of 'auto', 'newton', or 'adam'.")
    if solver == "auto":
        return "lbfgsb" if _scipy_available() else "newton"
    return solver


def _node_energy_gradient_and_hessian(
    positions: torch.Tensor,
    target_lengths: torch.Tensor,
    spring_strengths: torch.Tensor,
    node_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the KK gradient and Hessian for one Newton update.

    Parameters
    ----------
    positions : torch.Tensor
        Current node positions with shape ``[N, 2]``.
    target_lengths : torch.Tensor
        Preferred Euclidean lengths with shape ``[N, N]``.
    spring_strengths : torch.Tensor
        Spring strengths with shape ``[N, N]``.
    node_index : int
        Node index whose local system should be evaluated.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Gradient vector with shape ``[2]`` and Hessian with shape ``[2, 2]``.
    """
    delta = positions[node_index] - positions
    distances = torch.linalg.norm(delta, dim=1).clamp(min=_NEWTON_DISTANCE_EPSILON)
    mask = torch.ones(positions.shape[0], dtype=torch.bool, device=positions.device)
    mask[node_index] = False

    dx = delta[mask, 0]
    dy = delta[mask, 1]
    dist = distances[mask]
    lengths = target_lengths[node_index, mask]
    strengths = spring_strengths[node_index, mask]

    factor = 1.0 - lengths / dist
    gradient = torch.stack(
        (
            torch.sum(strengths * dx * factor),
            torch.sum(strengths * dy * factor),
        )
    )

    inv_dist_cubed = dist.pow(-3)
    hessian_xx = torch.sum(strengths * (1.0 - lengths * dy.square() * inv_dist_cubed))
    hessian_yy = torch.sum(strengths * (1.0 - lengths * dx.square() * inv_dist_cubed))
    hessian_xy = torch.sum(strengths * lengths * dx * dy * inv_dist_cubed)
    hessian = torch.stack(
        (
            torch.stack((hessian_xx, hessian_xy)),
            torch.stack((hessian_xy, hessian_yy)),
        )
    )

    # Match NetworkX's centering penalty in the fallback solver too.
    sum_pos = positions.sum(dim=0)
    gradient = gradient + (_CENTERING_WEIGHT * sum_pos)
    hessian = hessian + (
        _CENTERING_WEIGHT * torch.eye(2, dtype=positions.dtype, device=positions.device)
    )
    return gradient, hessian


def _all_node_deltas(
    positions: torch.Tensor,
    target_lengths: torch.Tensor,
    spring_strengths: torch.Tensor,
) -> torch.Tensor:
    """Compute the KK stationarity measure for every node.

    Parameters
    ----------
    positions : torch.Tensor
        Current node positions with shape ``[N, 2]``.
    target_lengths : torch.Tensor
        Preferred Euclidean lengths with shape ``[N, N]``.
    spring_strengths : torch.Tensor
        Spring strengths with shape ``[N, N]``.

    Returns
    -------
    torch.Tensor
        Per-node gradient norms with shape ``[N]``.
    """
    deltas = torch.empty((positions.shape[0],), dtype=positions.dtype, device=positions.device)
    for node_index in range(positions.shape[0]):
        gradient, _ = _node_energy_gradient_and_hessian(
            positions=positions,
            target_lengths=target_lengths,
            spring_strengths=spring_strengths,
            node_index=node_index,
        )
        deltas[node_index] = torch.linalg.norm(gradient)
    return deltas


def _solve_newton_displacement(gradient: torch.Tensor, hessian: torch.Tensor) -> torch.Tensor:
    """Solve the 2x2 Newton system for one KK node update.

    Parameters
    ----------
    gradient : torch.Tensor
        KK gradient vector with shape ``[2]``.
    hessian : torch.Tensor
        KK Hessian matrix with shape ``[2, 2]``.

    Returns
    -------
    torch.Tensor
        Position displacement vector with shape ``[2]``.
    """
    regularized_hessian = hessian + (
        _NEWTON_HESSIAN_REGULARIZATION * torch.eye(2, dtype=hessian.dtype, device=hessian.device)
    )
    rhs = -gradient.unsqueeze(1)
    try:
        return torch.linalg.solve(regularized_hessian, rhs).squeeze(1)
    except RuntimeError:
        return torch.zeros((2,), dtype=gradient.dtype, device=gradient.device)


def _layout_kk_newton(
    target_lengths: torch.Tensor,
    spring_strengths: torch.Tensor,
    num_nodes: int,
    steps: int,
    trace_every: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run the fallback per-node Newton-Raphson solver.

    Parameters
    ----------
    target_lengths : torch.Tensor
        Preferred Euclidean lengths with shape ``[N, N]``.
    spring_strengths : torch.Tensor
        Spring strengths with shape ``[N, N]``.
    num_nodes : int
        Number of graph nodes.
    steps : int
        Maximum number of outer iterations.
    trace_every : int
        Snapshot cadence in outer iterations.

    Returns
    -------
    tuple[torch.Tensor, list[torch.Tensor]]
        Final positions and optional trace snapshots.
    """
    positions = _initialize_positions(num_nodes=num_nodes).to(dtype=torch.float64)
    lengths = target_lengths.to(dtype=torch.float64)
    strengths = spring_strengths.to(dtype=torch.float64)
    traces: list[torch.Tensor] = []
    if steps == 0:
        return positions.to(dtype=torch.float32), traces

    inner_limit = max(8, min(steps, 32))
    for step_index in range(steps):
        deltas = _all_node_deltas(
            positions=positions,
            target_lengths=lengths,
            spring_strengths=strengths,
        )
        max_delta, node_tensor = torch.max(deltas, dim=0)
        if float(max_delta.item()) < _NEWTON_GLOBAL_EPSILON:
            break

        node_index = int(node_tensor.item())
        for _ in range(inner_limit):
            gradient, hessian = _node_energy_gradient_and_hessian(
                positions=positions,
                target_lengths=lengths,
                spring_strengths=strengths,
                node_index=node_index,
            )
            if float(torch.linalg.norm(gradient).item()) < _NEWTON_LOCAL_EPSILON:
                break
            positions[node_index] = positions[node_index] + _solve_newton_displacement(
                gradient=gradient,
                hessian=hessian,
            )

        if trace_every > 0 and (step_index + 1) % trace_every == 0:
            traces.append(positions.to(dtype=torch.float32).clone())

    return positions.to(dtype=torch.float32), traces


def _kamada_kawai_costfn(
    pos_vec: np.ndarray,
    invdist: np.ndarray,
    meanweight: float,
    dim: int,
) -> tuple[float, np.ndarray]:
    """Compute the NetworkX KK stress objective and analytic gradient.

    Parameters
    ----------
    pos_vec : numpy.ndarray
        Flattened position vector with shape ``[N * dim]``.
    invdist : numpy.ndarray
        Inverse distance matrix with shape ``[N, N]``.
    meanweight : float
        Weight of the origin-centering penalty.
    dim : int
        Layout dimension.

    Returns
    -------
    tuple[float, numpy.ndarray]
        Objective value and flattened gradient.
    """
    num_nodes = int(invdist.shape[0])
    pos_arr = pos_vec.reshape((num_nodes, dim))

    delta = pos_arr[:, np.newaxis, :] - pos_arr[np.newaxis, :, :]
    node_separation = np.linalg.norm(delta, axis=-1)
    direction = np.einsum(
        "ijk,ij->ijk",
        delta,
        1.0 / (node_separation + np.eye(num_nodes) * _DISTANCE_EPSILON),
    )

    offset = (node_separation * invdist) - 1.0
    offset[np.diag_indices(num_nodes)] = 0.0

    cost = 0.5 * float(np.sum(offset**2))
    gradient = np.einsum("ij,ij,ijk->ik", invdist, offset, direction) - np.einsum(
        "ij,ij,ijk->jk", invdist, offset, direction
    )

    sum_pos = np.sum(pos_arr, axis=0)
    cost += 0.5 * meanweight * float(np.sum(sum_pos**2))
    gradient += meanweight * sum_pos
    return cost, gradient.ravel()


def _layout_kk_lbfgsb(
    distance_matrix: torch.Tensor,
    num_nodes: int,
    steps: int,
    trace_every: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run the NetworkX-style full-stress L-BFGS-B solver.

    Parameters
    ----------
    distance_matrix : torch.Tensor
        Directed shortest-path distances with shape ``[N, N]``.
    num_nodes : int
        Number of graph nodes.
    steps : int
        Maximum optimizer iterations.
    trace_every : int
        Callback snapshot cadence in optimizer iterations.

    Returns
    -------
    tuple[torch.Tensor, list[torch.Tensor]]
        Final positions and optional trace snapshots.
    """
    import scipy as sp

    initial_positions = _initialize_positions(num_nodes=num_nodes).numpy().astype(np.float64)
    if steps == 0:
        return torch.from_numpy(initial_positions).to(dtype=torch.float32), []

    dist_mtx = distance_matrix.numpy().astype(np.float64, copy=False)
    invdist = 1.0 / (dist_mtx + np.eye(num_nodes, dtype=np.float64) * _DISTANCE_EPSILON)
    traces: list[torch.Tensor] = []
    iteration = 0

    def _callback(pos_vec: np.ndarray) -> None:
        """Collect optional optimizer traces without affecting convergence."""
        nonlocal iteration

        iteration += 1
        if trace_every > 0 and iteration % trace_every == 0:
            traces.append(
                torch.from_numpy(pos_vec.reshape((num_nodes, 2)).copy()).to(dtype=torch.float32)
            )

    optresult = sp.optimize.minimize(
        _kamada_kawai_costfn,
        initial_positions.ravel(),
        method="L-BFGS-B",
        args=(invdist, _CENTERING_WEIGHT, 2),
        jac=True,
        callback=_callback,
        options={"maxiter": steps},
    )
    final_positions = optresult.x.reshape((num_nodes, 2))
    return torch.from_numpy(final_positions).to(dtype=torch.float32), traces


def _layout_kk_adam(
    adjacency: list[list[int]],
    num_nodes: int,
    steps: int,
    seed: int,
    trace_every: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run the legacy Adam-based stress minimizer.

    Parameters
    ----------
    adjacency : list[list[int]]
        Directed adjacency list.
    num_nodes : int
        Number of graph nodes.
    steps : int
        Number of optimizer iterations.
    seed : int
        Random seed for deterministic pivot sampling.
    trace_every : int
        Snapshot cadence in optimizer iterations.

    Returns
    -------
    tuple[torch.Tensor, list[torch.Tensor]]
        Final positions and optional trace snapshots.
    """
    use_pivots = num_nodes > _PIVOT_THRESHOLD
    if use_pivots:
        pivot_indices, pivot_distances = _sample_pivot_distances(adjacency=adjacency, seed=seed)
        target_lengths, spring_strengths = _target_lengths_and_strengths(pivot_distances)
    else:
        pivot_indices = torch.empty(0, dtype=torch.long)
        distance_matrix = _replace_unreachable_torch(
            torch.tensor(
                [_bfs_distances(adjacency, node) for node in range(num_nodes)],
                dtype=torch.float32,
            )
        )
        target_lengths, spring_strengths = _target_lengths_and_strengths(distance_matrix)

    positions = torch.nn.Parameter(_initialize_positions(num_nodes=num_nodes))
    optimizer = torch.optim.Adam([positions], lr=1.0)
    traces: list[torch.Tensor] = []

    for step_index in range(steps):
        optimizer.zero_grad()
        if use_pivots:
            pivot_positions = positions[pivot_indices]
            pairwise_distances = torch.cdist(positions, pivot_positions)
            stress = (spring_strengths * (pairwise_distances - target_lengths).square()).sum()
        else:
            pairwise_distances = torch.cdist(positions, positions)
            weighted_error = spring_strengths * (pairwise_distances - target_lengths).square()
            stress = weighted_error.sum()

        sum_pos = positions.sum(dim=0)
        stress = stress + (0.5 * _CENTERING_WEIGHT * sum_pos.square().sum())
        stress.backward()
        optimizer.step()

        if trace_every > 0 and (step_index + 1) % trace_every == 0:
            traces.append(positions.detach().clone())

    return positions.detach(), traces


def layout_kk(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 500,
    seed: int = 42,
    trace_every: int = 0,
    solver: str = "auto",
) -> Union[torch.Tensor, tuple[torch.Tensor, list[torch.Tensor]]]:
    """Run Kamada-Kawai layout.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Unused, accepted for interface compatibility.
    steps : int, default=500
        Maximum number of optimization iterations.
    seed : int, default=42
        Accepted for interface compatibility. The NetworkX-matching circular
        initialization is deterministic and does not depend on the seed.
    trace_every : int, default=0
        If greater than zero, record snapshots every ``trace_every`` iterations.
    solver : {"auto", "newton", "adam"}, default="auto"
        ``"auto"`` uses SciPy L-BFGS-B when available and otherwise falls back
        to the legacy Newton solver. ``"adam"`` keeps Dagua's previous
        large-graph escape hatch.

    Returns
    -------
    torch.Tensor or tuple
        Final positions with shape ``[N, 2]``, or ``(positions, traces)`` when
        tracing is enabled.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``steps``, ``trace_every``, or ``solver`` are invalid.
    """
    _ = node_sizes
    _ = seed

    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if trace_every < 0:
        raise ValueError("trace_every must be non-negative.")

    if num_nodes == 0:
        empty_positions = torch.empty((0, 2), dtype=torch.float32)
        return (empty_positions, []) if trace_every > 0 else empty_positions
    if num_nodes == 1:
        single = torch.zeros((1, 2), dtype=torch.float32)
        return (single, []) if trace_every > 0 else single

    adjacency = _build_directed_adjacency(edge_index=edge_index, num_nodes=num_nodes)
    resolved_solver = _resolve_solver(num_nodes=num_nodes, solver=solver)

    if resolved_solver == "lbfgsb":
        distance_matrix = _compute_distance_matrix(
            edge_index=edge_index,
            num_nodes=num_nodes,
            adjacency=adjacency,
        )
        final_positions, traces = _layout_kk_lbfgsb(
            distance_matrix=distance_matrix,
            num_nodes=num_nodes,
            steps=steps,
            trace_every=trace_every,
        )
    elif resolved_solver == "newton":
        distance_matrix = _compute_distance_matrix(
            edge_index=edge_index,
            num_nodes=num_nodes,
            adjacency=adjacency,
        )
        target_lengths, spring_strengths = _target_lengths_and_strengths(distance_matrix)
        final_positions, traces = _layout_kk_newton(
            target_lengths=target_lengths,
            spring_strengths=spring_strengths,
            num_nodes=num_nodes,
            steps=steps,
            trace_every=trace_every,
        )
    else:
        final_positions, traces = _layout_kk_adam(
            adjacency=adjacency,
            num_nodes=num_nodes,
            steps=steps,
            seed=seed,
            trace_every=trace_every,
        )

    scaled_positions = _rescale_layout(final_positions)
    if trace_every > 0:
        return scaled_positions, traces
    return scaled_positions
