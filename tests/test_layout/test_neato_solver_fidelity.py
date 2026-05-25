"""Regression tests for Graphviz neato PCA and CG solver fidelity."""

from __future__ import annotations

import numpy as np
import torch

from dagua.layout.ops.pipelines.stress_majorization import (
    _graphviz_conjugate_gradient_packed,
    _graphviz_normalize_pca_positions,
    _graphviz_packed_index,
    _graphviz_pca_project_distances,
    _graphviz_random_initialize_positions,
    layout_stress_majorization_pipeline,
)


def _path4_distances() -> np.ndarray:
    """Return the all-pairs shortest-path distances for a four-node path.

    Returns
    -------
    numpy.ndarray
        Dense distance matrix with shape ``[4, 4]``.
    """
    return np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 1.0],
            [3.0, 2.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )


def _path4_edge_index() -> torch.Tensor:
    """Return a directed edge tensor for a four-node path.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, 3]``.
    """
    return torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)


def test_graphviz_pca_projection_matches_source_derived_path4() -> None:
    """Graphviz PCA projection should match a source-derived path golden."""
    projected = _graphviz_pca_project_distances(target_distances=_path4_distances())
    normalized = _graphviz_normalize_pca_positions(positions=projected)

    expected = np.array(
        [
            [1.0, 0.70710677],
            [0.41421354, -0.70710677],
            [-0.41421354, -0.70710677],
            [-1.0, 0.70710677],
        ],
        dtype=np.float32,
    )

    assert np.allclose(normalized, expected, rtol=0.0, atol=1.0e-7)


def test_graphviz_packed_cg_solves_centered_complete3_golden() -> None:
    """Packed CG should solve a centered complete-graph Laplacian system."""
    size = 3
    packed = np.zeros(size * (size + 1) // 2, dtype=np.float32)
    for row in range(size):
        packed[_graphviz_packed_index(row, row, size)] = np.float32(-2.0)
        for col in range(row + 1, size):
            packed[_graphviz_packed_index(row, col, size)] = np.float32(1.0)

    x = np.zeros(size, dtype=np.float32)
    b = np.array([1.0, -2.0, 1.0], dtype=np.float32)
    result = _graphviz_conjugate_gradient_packed(
        packed_matrix=packed,
        x=x,
        b=b,
        tolerance=1.0e-6,
        max_iterations=size,
    )

    expected = np.array([-1.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0], dtype=np.float32)
    assert result == 0
    assert np.allclose(x, expected, rtol=0.0, atol=1.0e-6)


def test_graphviz_random_initialization_matches_source_derived_path4() -> None:
    """Graphviz random initialization should match ``srand48``/``drand48``."""
    initialized = _graphviz_random_initialize_positions(num_nodes=4, dimensions=2, seed=1)

    expected = np.array(
        [
            [-0.36575127, 0.00882258],
            [0.4274356, -0.10968383],
            [0.15810779, -0.44390294],
            [-0.2197921, 0.5447642],
        ],
        dtype=np.float32,
    )

    assert np.allclose(initialized, expected, rtol=0.0, atol=1.0e-7)


def test_graphviz_fidelity_zero_iterations_uses_graphviz_default_random_init() -> None:
    """The ``graphviz`` fidelity mode should initialize like default neato."""
    positions = layout_stress_majorization_pipeline(
        edge_index=_path4_edge_index(),
        num_nodes=4,
        iterations=0,
        fidelity_mode="graphviz",
    )

    expected = torch.tensor(
        [
            [0.3856448, -0.18237238],
            [-0.24779494, -0.1027349],
            [-0.27776906, 0.33136684],
            [0.1399192, -0.04625957],
        ],
        dtype=torch.float32,
    )
    assert isinstance(positions, torch.Tensor)
    assert torch.allclose(positions, expected, rtol=0.0, atol=1.0e-7)
