"""Nonmetric SMACOF layout pipeline with sklearn-compatible disparities."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from scipy import interpolate, optimize

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.graph_utils import shortest_path_distances

_MISSING_DISTANCE = 0.0
_ZERO_DISTANCE_FILL = 1.0e-5


def _euclidean_distances(positions: np.ndarray) -> np.ndarray:
    """Compute dense Euclidean distances without sklearn delegation.

    Parameters
    ----------
    positions : numpy.ndarray
        Coordinate matrix with shape ``[N, D]``.

    Returns
    -------
    numpy.ndarray
        Pairwise Euclidean distances with shape ``[N, N]``.
    """
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _make_unique_weighted(
    x_values: np.ndarray,
    y_values: np.ndarray,
    sample_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse sorted duplicate x-values like sklearn isotonic regression.

    Parameters
    ----------
    x_values : numpy.ndarray
        Sorted one-dimensional predictor values with shape ``[M]``.
    y_values : numpy.ndarray
        Response values with shape ``[M]``.
    sample_weight : numpy.ndarray
        Positive sample weights with shape ``[M]``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Unique x-values, weighted-average y-values, and summed weights.
    """
    if x_values.size == 0:
        return x_values, y_values, sample_weight

    unique_x: list[float] = []
    unique_y: list[float] = []
    unique_weight: list[float] = []
    start = 0
    while start < x_values.size:
        stop = start + 1
        while stop < x_values.size and x_values[stop] == x_values[start]:
            stop += 1
        weights = sample_weight[start:stop]
        unique_x.append(float(x_values[start]))
        unique_y.append(float(np.average(y_values[start:stop], weights=weights)))
        unique_weight.append(float(weights.sum()))
        start = stop

    dtype = x_values.dtype
    return (
        np.asarray(unique_x, dtype=dtype),
        np.asarray(unique_y, dtype=dtype),
        np.asarray(unique_weight, dtype=dtype),
    )


def _sklearn_isotonic_fit_transform(x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
    """Fit-transform increasing isotonic regression with sklearn wrapper semantics.

    Parameters
    ----------
    x_values : numpy.ndarray
        Dissimilarities with shape ``[M]``.
    y_values : numpy.ndarray
        Current embedding distances with shape ``[M]``.

    Returns
    -------
    numpy.ndarray
        Interpolated disparities at ``x_values`` with shape ``[M]``.
    """
    x = np.asarray(x_values, dtype=np.float64).reshape(-1)
    y = np.asarray(y_values, dtype=np.float64).reshape(-1)
    sample_weight = np.ones_like(x, dtype=np.float64)

    order = np.lexsort((y, x))
    x_sorted, y_sorted, weight_sorted = (
        x[order],
        y[order],
        sample_weight[order],
    )
    unique_x, unique_y, unique_weight = _make_unique_weighted(
        x_values=x_sorted,
        y_values=y_sorted,
        sample_weight=weight_sorted,
    )
    if unique_x.size == 0:
        return np.empty_like(x)

    isotonic = optimize.isotonic_regression(
        y=unique_y,
        weights=unique_weight,
        increasing=True,
    )
    thresholds_y = np.asarray(isotonic.x, dtype=np.float64)

    keep_data = np.ones((thresholds_y.shape[0],), dtype=bool)
    if thresholds_y.shape[0] > 2:
        keep_data[1:-1] = np.logical_or(
            np.not_equal(thresholds_y[1:-1], thresholds_y[:-2]),
            np.not_equal(thresholds_y[1:-1], thresholds_y[2:]),
        )
    thresholds_x = unique_x[keep_data]
    thresholds_y = thresholds_y[keep_data]

    clipped_x = np.clip(x, float(thresholds_x.min()), float(thresholds_x.max()))
    if thresholds_y.shape[0] == 1:
        return np.full_like(clipped_x, float(thresholds_y[0]), dtype=np.float64)
    fitted = interpolate.interp1d(
        thresholds_x,
        thresholds_y,
        kind="linear",
        bounds_error=False,
    )(clipped_x)
    return np.asarray(fitted, dtype=np.float64)


def smacof_nonmetric_positions(
    dissimilarities: np.ndarray,
    *,
    seed: int = 42,
    max_iter: int = 300,
    eps: float = 1.0e-6,
    init: Optional[np.ndarray] = None,
    normalized_stress: bool = False,
) -> tuple[np.ndarray, float, int]:
    """Run sklearn-compatible nonmetric SMACOF on a dissimilarity matrix.

    Parameters
    ----------
    dissimilarities : numpy.ndarray
        Symmetric dissimilarity matrix with shape ``[N, N]``.
    seed : int, default=42
        Random seed forwarded to sklearn-compatible random initialization.
    max_iter : int, default=300
        Maximum number of SMACOF iterations.
    eps : float, default=1e-6
        Relative stress convergence tolerance.
    init : numpy.ndarray, optional
        Optional initial coordinates with shape ``[N, 2]``.
    normalized_stress : bool, default=False
        Whether to return sklearn's Stress-1 normalized stress value.

    Returns
    -------
    tuple[numpy.ndarray, float, int]
        Coordinates with shape ``[N, 2]``, final stress, and iteration count.
    """
    if max_iter < 0:
        raise ValueError("max_iter must be non-negative.")
    if eps <= 0.0:
        raise ValueError("eps must be positive.")

    matrix = np.asarray(dissimilarities, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("dissimilarities must be a square matrix.")
    n_samples = int(matrix.shape[0])
    if n_samples == 0:
        return np.empty((0, 2), dtype=np.float64), 0.0, 0

    random_state = np.random.RandomState(seed)
    dissimilarities_flat = ((1 - np.tri(n_samples)) * matrix).ravel()
    observed_mask = dissimilarities_flat != _MISSING_DISTANCE
    dissimilarities_flat_w = dissimilarities_flat[observed_mask]
    if init is None:
        x_positions = random_state.uniform(size=n_samples * 2).reshape((n_samples, 2))
    else:
        x_positions = np.asarray(init, dtype=np.float64)
        if x_positions.shape != (n_samples, 2):
            raise ValueError(f"init matrix should be of shape ({n_samples}, 2)")

    distances = _euclidean_distances(x_positions)
    old_stress: Optional[float] = None
    iterations_run = 0
    for iteration in range(max_iter):
        distances_flat = distances.ravel()
        distances_flat_w = distances_flat[observed_mask]
        if iteration < 1:
            disparities_flat = dissimilarities_flat_w
        else:
            disparities_flat = _sklearn_isotonic_fit_transform(
                x_values=dissimilarities_flat_w,
                y_values=distances_flat_w,
            )

        disparities = np.zeros_like(distances_flat)
        disparities[observed_mask] = disparities_flat
        disparities = disparities.reshape((n_samples, n_samples))
        scale_denominator = float((disparities**2).sum())
        if scale_denominator > 0.0:
            disparities *= np.sqrt((n_samples * (n_samples - 1) / 2) / scale_denominator)
        disparities = disparities + disparities.T

        distances[distances == 0] = _ZERO_DISTANCE_FILL
        ratio = disparities / distances
        b_matrix = -ratio
        diagonal = np.arange(n_samples)
        b_matrix[diagonal, diagonal] += ratio.sum(axis=1)
        x_positions = (1.0 / n_samples) * np.dot(b_matrix, x_positions)

        distances = _euclidean_distances(x_positions)
        stress = float(((distances.ravel() - disparities.ravel()) ** 2).sum() / 2)
        iterations_run = iteration + 1
        if old_stress is not None:
            sum_squared_distances = float((distances.ravel() ** 2).sum())
            if ((old_stress - stress) / (sum_squared_distances / 2)) < eps:
                break
        old_stress = stress

    if normalized_stress:
        sum_squared_distances = float((distances.ravel() ** 2).sum())
        stress = float(np.sqrt(stress / (sum_squared_distances / 2)))

    return x_positions, stress, iterations_run


def build_smacof_nonmetric_pipeline() -> Pipeline:
    """Build the nonmetric SMACOF pipeline placeholder.

    Returns
    -------
    Pipeline
        Empty marker pipeline; the public entrypoint performs the sklearn-
        compatible NumPy loop directly because the loop's isotonic regression
        state is tightly coupled to each SMACOF iteration.
    """
    return Pipeline([], name="smacof_nonmetric_pipeline")


def layout_smacof_nonmetric_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: int = 42,
    max_iter: int = 300,
    eps: float = 1.0e-6,
    normalized_stress: bool = False,
    edge_weights: Optional[torch.Tensor] = None,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run nonmetric SMACOF on graph geodesic distances.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Accepted for engine compatibility; nonmetric SMACOF is size-blind.
    seed : int, default=42
        Random seed for sklearn-compatible initialization.
    max_iter : int, default=300
        Maximum number of nonmetric SMACOF iterations.
    eps : float, default=1e-6
        Relative stress convergence tolerance.
    normalized_stress : bool, default=False
        Whether to compute normalized stress internally.
    edge_weights : torch.Tensor, optional
        Optional edge weights for geodesic distances.
    fidelity_dtype : torch.dtype, optional
        Optional output dtype. Defaults to ``torch.float64`` for fidelity.

    Returns
    -------
    torch.Tensor
        Final coordinates with shape ``[N, 2]``.
    """
    del node_sizes
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if num_nodes == 0:
        dtype = torch.float64 if fidelity_dtype is None else fidelity_dtype
        return torch.empty((0, 2), dtype=dtype, device=edge_index.device)

    distances = shortest_path_distances(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weights=edge_weights,
    )
    positions, _stress, _iterations = smacof_nonmetric_positions(
        distances,
        seed=seed,
        max_iter=max_iter,
        eps=eps,
        normalized_stress=normalized_stress,
    )
    dtype = torch.float64 if fidelity_dtype is None else fidelity_dtype
    return torch.tensor(positions, dtype=dtype, device=edge_index.device)


__all__ = [
    "build_smacof_nonmetric_pipeline",
    "layout_smacof_nonmetric_pipeline",
    "smacof_nonmetric_positions",
]
