"""Stress majorization via dense SMACOF updates."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from dagua.layout._archive.classic.classical_mds import (
    _shortest_path_distances,
    layout_classical_mds,
)

_MIN_DISTANCE = 1.0e-9


def _layout_device(
    edge_index: torch.Tensor,
    node_sizes: Optional[torch.Tensor],
) -> torch.device:
    """Choose the device used for the returned layout tensor.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]``.

    Returns
    -------
    torch.device
        Device for the final output tensor.
    """
    if edge_index.numel() > 0:
        return edge_index.device
    if node_sizes is not None:
        return node_sizes.device
    return torch.device("cpu")


def _pairwise_distances(positions: np.ndarray) -> np.ndarray:
    """Compute dense Euclidean distances between all node pairs.

    Parameters
    ----------
    positions : numpy.ndarray
        Position array with shape ``[N, 2]``.

    Returns
    -------
    numpy.ndarray
        Pairwise Euclidean distances with shape ``[N, N]``.
    """
    deltas = positions[:, None, :] - positions[None, :, :]
    return np.sqrt(np.sum(deltas * deltas, axis=2))


def _stress_value(
    positions: np.ndarray,
    target_distances: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Compute the weighted stress objective for one embedding.

    Parameters
    ----------
    positions : numpy.ndarray
        Position array with shape ``[N, 2]``.
    target_distances : numpy.ndarray
        Desired graph distances with shape ``[N, N]``.
    weights : numpy.ndarray
        SMACOF weight matrix with shape ``[N, N]``.

    Returns
    -------
    float
        Weighted stress value.
    """
    current_distances = _pairwise_distances(positions)
    errors = current_distances - target_distances
    return 0.5 * float(np.sum(weights * errors * errors))


def _smacof_update(
    positions: np.ndarray,
    target_distances: np.ndarray,
    weights: np.ndarray,
    laplacian_pinv: np.ndarray,
) -> np.ndarray:
    """Apply one SMACOF majorization step.

    Parameters
    ----------
    positions : numpy.ndarray
        Current positions with shape ``[N, 2]``.
    target_distances : numpy.ndarray
        Desired graph distances with shape ``[N, N]``.
    weights : numpy.ndarray
        SMACOF weight matrix with shape ``[N, N]``.
    laplacian_pinv : numpy.ndarray
        Pseudoinverse of the weighted Laplacian with shape ``[N, N]``.

    Returns
    -------
    numpy.ndarray
        Updated centered positions with shape ``[N, 2]``.
    """
    current_distances = np.maximum(_pairwise_distances(positions), _MIN_DISTANCE)
    ratio = np.zeros_like(target_distances)
    active_mask = weights > 0.0
    ratio[active_mask] = target_distances[active_mask] / current_distances[active_mask]

    b_matrix = -weights * ratio
    np.fill_diagonal(b_matrix, 0.0)
    np.fill_diagonal(b_matrix, -b_matrix.sum(axis=1))

    updated = laplacian_pinv @ (b_matrix @ positions)
    return updated - updated.mean(axis=0, keepdims=True)


def _initial_positions(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor],
    seed: int,
    edge_weights: Optional[torch.Tensor],
) -> np.ndarray:
    """Build a stable stochastic warm start for SMACOF.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]``.
    seed : int
        Random seed for the warm-start jitter.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.

    Returns
    -------
    numpy.ndarray
        Initial positions with shape ``[N, 2]``.
    """
    baseline = layout_classical_mds(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        seed=42,
        edge_weights=edge_weights,
    )
    rng = np.random.default_rng(seed)
    jitter = rng.normal(loc=0.0, scale=0.05, size=(num_nodes, 2))
    initialized = baseline.detach().cpu().numpy().astype(np.float64) + jitter
    return initialized - initialized.mean(axis=0, keepdims=True)


def layout_stress_majorization(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    iterations: int = 200,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
    trace_every: int = 0,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    """Lay out a graph with dense stress majorization.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]``. Accepted for
        interface compatibility and forwarded to the classical-MDS warm start.
    iterations : int, default=200
        Number of SMACOF majorization steps to run.
    seed : int, default=42
        Random seed for the stochastic warm-start jitter.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]`` used when computing
        graph distances.
    trace_every : int, default=0
        If positive, return periodic position snapshots every
        ``trace_every`` iterations.

    Returns
    -------
    torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]
        Final position tensor with shape ``[N, 2]``. When ``trace_every > 0``,
        also returns periodic snapshots in layout order.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``iterations``, or ``trace_every`` is invalid, or if
        ``edge_weights`` has the wrong shape.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if iterations < 0:
        raise ValueError("iterations must be non-negative.")
    if trace_every < 0:
        raise ValueError("trace_every must be non-negative.")
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must have shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge count {edge_index.shape[1]}"
            )

    device = _layout_device(edge_index=edge_index, node_sizes=node_sizes)
    if num_nodes == 0:
        empty = torch.empty((0, 2), dtype=torch.float32, device=device)
        return (empty, []) if trace_every > 0 else empty
    if num_nodes == 1:
        single = torch.zeros((1, 2), dtype=torch.float32, device=device)
        return (single, []) if trace_every > 0 else single

    target_distances = _shortest_path_distances(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weights=edge_weights,
    )
    with np.errstate(divide="ignore"):
        weights = np.where(target_distances > 0.0, 1.0 / np.square(target_distances), 0.0)
    np.fill_diagonal(weights, 0.0)

    laplacian = -weights
    np.fill_diagonal(laplacian, weights.sum(axis=1))
    laplacian_pinv = np.linalg.pinv(laplacian)

    current = _initial_positions(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        seed=seed,
        edge_weights=edge_weights,
    )
    current_stress = _stress_value(current, target_distances=target_distances, weights=weights)

    traces: list[torch.Tensor] = []
    for iteration_idx in range(iterations):
        candidate = _smacof_update(
            positions=current,
            target_distances=target_distances,
            weights=weights,
            laplacian_pinv=laplacian_pinv,
        )
        candidate_stress = _stress_value(
            candidate,
            target_distances=target_distances,
            weights=weights,
        )

        if candidate_stress > current_stress + 1.0e-8:
            # Dense pseudoinverse updates can drift numerically on nearly
            # singular cases; conservative blending preserves monotonic stress.
            blended = candidate
            for _ in range(8):
                blended = 0.5 * (blended + current)
                candidate_stress = _stress_value(
                    blended,
                    target_distances=target_distances,
                    weights=weights,
                )
                if candidate_stress <= current_stress + 1.0e-8:
                    candidate = blended
                    break
            else:
                candidate = current
                candidate_stress = current_stress

        current = candidate
        current_stress = candidate_stress

        if trace_every > 0 and (
            (iteration_idx + 1) % trace_every == 0 or iteration_idx + 1 == iterations
        ):
            traces.append(torch.from_numpy(current).to(dtype=torch.float32, device=device))

    final_positions = torch.from_numpy(current).to(dtype=torch.float32, device=device)
    if trace_every > 0:
        if not traces or not torch.allclose(traces[-1], final_positions):
            traces.append(final_positions.clone())
        return final_positions, traces
    return final_positions
