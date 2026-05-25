"""Regression tests for UMAP graph-fidelity behavior."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from scipy import sparse

from dagua.layout.classic.umap_layout import layout_umap
from dagua.layout.ops import umap as umap_ops
from dagua.layout.ops.pipelines.umap_layout import layout_umap_layout_pipeline
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.umap import (
    StoreUMAPHyperparameters,
    _build_undirected_adjacency,
    _connected_spectral_embedding,
    _fit_ab,
    _knn_from_distances,
    _make_tau_state,
    _optimize_embedding,
    _smooth_knn_dist,
    _spectral_initialization,
    _tau_rand_int,
)


def test_build_undirected_adjacency_uses_weights_as_distances() -> None:
    """Verify weighted UMAP graph distances match the reference adapter."""
    edge_index = torch.tensor([[0, 0], [1, 1]], dtype=torch.long)
    edge_weights = torch.tensor([2.0, 3.0], dtype=torch.float32)

    adjacency = _build_undirected_adjacency(
        edge_index=edge_index,
        num_nodes=2,
        edge_weights=edge_weights,
    )

    assert adjacency == [[(1, 5.0)], [(0, 5.0)]]


def test_knn_from_distances_counts_self_neighbor() -> None:
    """Verify precomputed UMAP kNN semantics count the self neighbor."""
    distances = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    indices, knn_distances = _knn_from_distances(distances=distances, n_neighbors=2)

    expected_indices = torch.tensor([[0, 1], [1, 0], [2, 1]], dtype=torch.long)
    expected_distances = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    assert torch.equal(indices, expected_indices)
    assert torch.equal(knn_distances, expected_distances)


def test_knn_from_distances_uses_stable_index_tie_order() -> None:
    """Verify tied graph distances retain ascending node-index order."""
    distances = torch.tensor(
        [
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    indices, _ = _knn_from_distances(distances=distances, n_neighbors=4)

    expected_indices = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 0, 2, 3],
            [2, 0, 1, 3],
            [3, 0, 1, 2],
        ],
        dtype=torch.long,
    )
    assert torch.equal(indices, expected_indices)


def test_store_umap_hyperparameters_caps_neighbors_like_reference_adapter() -> None:
    """Verify small graph UMAP neighborhoods use the adapter's ``N - 1`` cap."""
    problem = LayoutProblem(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=6,
    )
    state = StoreUMAPHyperparameters(n_neighbors=15).apply(
        problem=problem,
        state=SolveState(),
        ctx=RuntimeContext(plan=ExecutionPlan(device="cpu")),
    )

    assert state.extras["umap_n_neighbors"] == 5


def test_connected_spectral_embedding_uses_reference_arpack_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify connected spectral initialization uses umap-learn ARPACK settings."""
    num_nodes = 100
    rows = np.arange(num_nodes - 1)
    cols = rows + 1
    graph = sparse.csr_matrix(
        (
            np.ones((2 * (num_nodes - 1),), dtype=np.float64),
            (np.concatenate([rows, cols]), np.concatenate([cols, rows])),
        ),
        shape=(num_nodes, num_nodes),
    )
    captured: dict[str, Any] = {}

    def fake_eigsh(
        matrix: sparse.csr_matrix,
        k: int,
        which: str,
        ncv: int,
        v0: np.ndarray,
        tol: float,
        maxiter: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Capture ARPACK arguments and return deterministic eigenvectors."""
        captured.update(
            {
                "shape": matrix.shape,
                "k": k,
                "which": which,
                "ncv": ncv,
                "v0": v0.copy(),
                "tol": tol,
                "maxiter": maxiter,
            }
        )
        eigenvalues = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        eigenvectors = np.column_stack(
            [
                np.ones((num_nodes,), dtype=np.float64),
                np.linspace(0.0, 1.0, num_nodes),
                np.linspace(1.0, 0.0, num_nodes),
            ]
        )
        return eigenvalues, eigenvectors

    monkeypatch.setattr(umap_ops.sparse_linalg, "eigsh", fake_eigsh)

    coordinates = _connected_spectral_embedding(graph=graph)

    assert coordinates.shape == (num_nodes, 2)
    assert captured["shape"] == (num_nodes, num_nodes)
    assert captured["k"] == 3
    assert captured["which"] == "SM"
    assert captured["ncv"] == 10
    assert np.array_equal(captured["v0"], np.ones((num_nodes,), dtype=np.float64))
    assert captured["tol"] == pytest.approx(1.0e-4)
    assert captured["maxiter"] == 5 * num_nodes


def test_fit_ab_uses_scipy_default_initial_guess(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify UMAP curve fitting leaves ``p0`` at SciPy's default ``(1, 1)``."""
    captured: dict[str, Any] = {}

    def fake_curve_fit(
        function: object,
        xdata: np.ndarray,
        ydata: np.ndarray,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Capture curve-fit keyword arguments and return positive parameters."""
        captured.update(kwargs)
        assert function is umap_ops._curve_function
        assert xdata.shape == ydata.shape
        return np.array([1.0, 1.0], dtype=np.float64), np.eye(2, dtype=np.float64)

    monkeypatch.setattr(umap_ops.optimize, "curve_fit", fake_curve_fit)

    assert _fit_ab(min_dist=0.5, spread=2.0) == (1.0, 1.0)
    assert "p0" not in captured
    assert captured["maxfev"] == 10_000


def test_tau_rand_int_uses_umap_per_source_state_sequence() -> None:
    """Verify negative-sampling RNG states match umap-learn source rows."""
    embedding = torch.tensor(
        [[0.0, 0.0], [1.5, 0.0], [-2.25, 0.0], [10.0, 0.0]],
        dtype=torch.float32,
    )

    state = _make_tau_state(embedding=embedding, seed=42)
    draws = [[_tau_rand_int(state=state[row]) for _ in range(3)] for row in range(state.shape[0])]

    assert draws == [
        [1512524284, -1869814856, -86469464],
        [-1507374596, 1334627390, 417043368],
        [-1691923972, 2005724090, -447646465],
        [-936949252, -562855019, -769362841],
    ]


def test_optimize_embedding_waits_until_first_sample_interval() -> None:
    """Verify UMAP SGD does not perform reference-forbidden epoch-zero updates."""
    embedding = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)

    optimized = _optimize_embedding(
        embedding=embedding.clone(),
        head=torch.tensor([0], dtype=torch.long),
        tail=torch.tensor([1], dtype=torch.long),
        epochs_per_sample=torch.tensor([1.0], dtype=torch.float32),
        n_epochs=1,
        learning_rate=1.0,
        negative_sample_rate=0,
        gamma=1.0,
        a=1.0,
        b=1.0,
        seed=42,
    )

    assert torch.equal(optimized, embedding)


def test_smooth_knn_dist_uses_reference_global_floor_when_rho_is_zero() -> None:
    """Verify zero-radius rows use UMAP's global smooth-kNN sigma floor."""
    knn_distances = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 2.0, 4.0],
        ],
        dtype=torch.float32,
    )

    sigmas, rhos = _smooth_knn_dist(knn_distances=knn_distances, n_neighbors=3)

    assert rhos[0].item() == 0.0
    assert torch.isclose(sigmas[0], torch.tensor(0.001), atol=1.0e-7)


def test_spectral_initialization_uses_umap_per_axis_unit_square_frame() -> None:
    """Verify spectral init is rescaled per axis into UMAP's ``[0, 10]`` frame."""
    head = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    tail = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    weight = torch.ones((4,), dtype=torch.float32)

    coordinates = _spectral_initialization(
        num_nodes=5,
        head=head,
        tail=tail,
        weight=weight,
        seed=42,
    )

    assert coordinates.shape == (5, 2)
    assert torch.allclose(coordinates.min(dim=0).values, torch.zeros(2), atol=1.0e-5)
    assert torch.allclose(coordinates.max(dim=0).values, torch.full((2,), 10.0), atol=1.0e-5)


def test_spectral_initialization_uses_random_init_for_small_umap_graphs() -> None:
    """Verify ``4 <= N < 10`` follows the reference adapter's random init policy."""
    head = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    tail = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    weight = torch.ones((5,), dtype=torch.float32)

    coordinates = _spectral_initialization(
        num_nodes=6,
        head=head,
        tail=tail,
        weight=weight,
        seed=42,
    )

    assert coordinates.shape == (6, 2)
    assert torch.allclose(coordinates.min(dim=0).values, torch.zeros(2), atol=1.0e-5)
    assert torch.allclose(coordinates.max(dim=0).values, torch.full((2,), 10.0), atol=1.0e-5)


def test_layout_umap_tiny_graph_matches_classic_adapter() -> None:
    """Verify single-node handling remains bit-exact with classic UMAP."""
    edge_index = torch.empty((2, 0), dtype=torch.long)
    expected = layout_umap(
        edge_index=edge_index,
        num_nodes=1,
        n_neighbors=1,
        n_epochs=50,
        seed=123,
    )

    coordinates = layout_umap_layout_pipeline(
        edge_index=edge_index,
        num_nodes=1,
        n_neighbors=1,
        n_epochs=50,
        seed=123,
    )

    assert torch.equal(coordinates, expected)
