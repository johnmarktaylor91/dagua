"""Tests for embedding ops in ``dagua.layout.ops.embed``."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dagua.layout.ops import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.embed import (
    BuildLaplacian,
    BuildLaplacianConfig,
    BuildNormalizedAdjacency,
    CurveFit_ab,
    CurveFitABConfig,
    Eigendecomposition,
    EigendecompositionConfig,
    GCNForward,
    GCNForwardConfig,
    PerplexityMatch,
    PerplexityMatchConfig,
    SymmetrizeAdjacency,
    _perplexity_row,
)


def _path_problem(num_nodes: int) -> LayoutProblem:
    """Build a path-graph layout problem.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    LayoutProblem
        Path-graph problem definition.
    """
    edge_count = max(num_nodes - 1, 0)
    if edge_count == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        source = torch.arange(0, edge_count, dtype=torch.long)
        target = torch.arange(1, num_nodes, dtype=torch.long)
        edge_index = torch.stack([source, target], dim=0)
    return LayoutProblem(edge_index=edge_index, num_nodes=num_nodes)


def _dense_path_adjacency(num_nodes: int) -> torch.Tensor:
    """Build a dense symmetric path-graph adjacency matrix.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    torch.Tensor
        Dense adjacency matrix with shape ``[N, N]``.
    """
    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for node_idx in range(max(num_nodes - 1, 0)):
        adjacency[node_idx, node_idx + 1] = 1.0
        adjacency[node_idx + 1, node_idx] = 1.0
    return adjacency


def test_symmetrize_adjacency_produces_symmetric_output() -> None:
    """SymmetrizeAdjacency should replace the payload with ``A + A^T``."""

    adjacency = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    state = SolveState(adjacency=adjacency)

    result = SymmetrizeAdjacency().apply(_path_problem(3), state, RuntimeContext())

    assert isinstance(result.adjacency, torch.Tensor)
    expected = torch.tensor(
        [
            [0.0, 1.0, 3.0],
            [1.0, 0.0, 2.0],
            [3.0, 2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(result.adjacency, expected)
    assert torch.allclose(result.adjacency, result.adjacency.transpose(0, 1))


def test_build_laplacian_matches_symmetric_normalized_reference_on_ten_node_graph() -> None:
    """BuildLaplacian should match the normalized-Laplacian reference formula."""

    num_nodes = 10
    adjacency = _dense_path_adjacency(num_nodes)
    state = SolveState(adjacency=adjacency)

    result = BuildLaplacian().apply(_path_problem(num_nodes), state, RuntimeContext())

    laplacian = result.extras["laplacian"]
    assert laplacian.shape == (num_nodes, num_nodes)

    adjacency_np = adjacency.numpy().astype(np.float64, copy=False)
    degrees = adjacency_np.sum(axis=1)
    inv_sqrt = np.zeros_like(degrees)
    nonzero_mask = degrees > 0.0
    inv_sqrt[nonzero_mask] = 1.0 / np.sqrt(degrees[nonzero_mask])
    expected = np.eye(num_nodes, dtype=np.float64) - (
        np.diag(inv_sqrt) @ adjacency_np @ np.diag(inv_sqrt)
    )

    assert np.allclose(laplacian.toarray(), expected)


def test_eigendecomposition_returns_requested_number_of_eigenpairs() -> None:
    """Eigendecomposition should return the configured number of smallest pairs."""

    num_nodes = 6
    adjacency = _dense_path_adjacency(num_nodes)
    state = SolveState(adjacency=adjacency)
    BuildLaplacian().apply(_path_problem(num_nodes), state, RuntimeContext())

    result = Eigendecomposition(EigendecompositionConfig(sparse_threshold=100, k=3)).apply(
        _path_problem(num_nodes),
        state,
        RuntimeContext(),
    )

    eigenpairs = result.extras["eigenpairs"]
    eigenvalues = eigenpairs["eigenvalues"]
    eigenvectors = eigenpairs["eigenvectors"]

    assert eigenvalues.shape == (3,)
    assert eigenvectors.shape == (num_nodes, 3)
    assert torch.isfinite(eigenvalues).all()
    assert torch.isfinite(eigenvectors).all()


def test_perplexity_match_produces_finite_symmetric_probability_matrix() -> None:
    """PerplexityMatch should produce a valid symmetric probability matrix."""

    distance_matrix = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 1.0],
            [3.0, 2.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    state = SolveState(distance_matrix=distance_matrix)

    result = PerplexityMatch(PerplexityMatchConfig(perplexity=2.0)).apply(
        _path_problem(4),
        state,
        RuntimeContext(),
    )

    probabilities = result.extras["probabilities"]

    assert probabilities.shape == distance_matrix.shape
    assert torch.isfinite(probabilities).all()
    assert torch.all(probabilities > 0)
    assert torch.allclose(probabilities, probabilities.transpose(0, 1), atol=1.0e-6)
    assert float(probabilities.sum().item()) == pytest.approx(1.0, rel=1.0e-5, abs=1.0e-5)


def test_symmetrize_adjacency_supports_list_payloads() -> None:
    """SymmetrizeAdjacency should preserve list adjacency while adding reverse weights."""

    state = SolveState(adjacency=[[(1, 2.0)], [(2, 3.0)], []])

    result = SymmetrizeAdjacency().apply(_path_problem(3), state, RuntimeContext())

    assert result.adjacency == [[(1, 2.0)], [(0, 2.0), (2, 3.0)], [(1, 3.0)]]


def test_build_laplacian_unnormalized_has_zero_row_sums_and_nonnegative_spectrum() -> None:
    """The unnormalized Laplacian should have zero row sums and PSD eigenvalues."""

    adjacency = _dense_path_adjacency(5)
    state = SolveState(adjacency=adjacency)

    result = BuildLaplacian(BuildLaplacianConfig(normalization="unnormalized")).apply(
        _path_problem(5),
        state,
        RuntimeContext(),
    )

    laplacian = result.extras["laplacian"].toarray()
    eigenvalues = np.linalg.eigvalsh(laplacian)

    assert np.allclose(laplacian.sum(axis=1), np.zeros(5), atol=1.0e-8)
    assert np.all(eigenvalues >= -1.0e-8)


def test_eigendecomposition_caps_requested_count_at_node_count() -> None:
    """Eigendecomposition should cap the returned eigenpair count at ``N``."""

    adjacency = _dense_path_adjacency(4)
    state = SolveState(adjacency=adjacency)
    BuildLaplacian().apply(_path_problem(4), state, RuntimeContext())

    result = Eigendecomposition(EigendecompositionConfig(sparse_threshold=0, k=10)).apply(
        _path_problem(4),
        state,
        RuntimeContext(),
    )

    eigenpairs = result.extras["eigenpairs"]
    assert eigenpairs["eigenvalues"].shape == (4,)
    assert eigenpairs["eigenvectors"].shape == (4, 4)


def test_perplexity_match_rows_sum_to_one_before_symmetrization() -> None:
    """The internal conditional probabilities should form valid per-row distributions."""

    distance_matrix = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 1.5, 2.5],
            [2.0, 1.5, 0.0, 1.0],
            [3.0, 2.5, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    rows = [
        _perplexity_row(
            distance_matrix[index],
            perplexity=2.0,
            tol=1.0e-5,
            max_iter=100,
        )
        for index in range(4)
    ]
    conditional = torch.stack(rows)

    torch.testing.assert_close(conditional.sum(dim=1), torch.ones(4), atol=1.0e-5, rtol=1.0e-5)
    torch.testing.assert_close(torch.diag(conditional), torch.zeros(4), atol=1.0e-8, rtol=0.0)


def test_gcn_forward_outputs_expected_coordinate_shape() -> None:
    """GCNForward should materialize an ``[N, dim]`` position tensor."""

    problem = _path_problem(4)
    state = SolveState(adjacency=_dense_path_adjacency(4))
    BuildNormalizedAdjacency().apply(problem, state, RuntimeContext())

    result = GCNForward(GCNForwardConfig(hidden_sizes=(8, 4), output_dim=3)).apply(
        problem,
        state,
        RuntimeContext(),
    )

    assert result.pos is not None
    assert result.pos.shape == (4, 3)
    assert torch.isfinite(result.pos).all()


def test_curve_fit_ab_stores_positive_float_parameters() -> None:
    """CurveFit_ab should store positive Python floats for ``a`` and ``b``."""

    result = CurveFit_ab(CurveFitABConfig(min_dist=0.2, spread=1.5)).apply(
        _path_problem(3),
        SolveState(),
        RuntimeContext(),
    )

    assert isinstance(result.extras["umap_a"], float)
    assert isinstance(result.extras["umap_b"], float)
    assert result.extras["umap_a"] > 0.0
    assert result.extras["umap_b"] > 0.0
