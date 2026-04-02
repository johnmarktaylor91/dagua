"""Tests for embedding ops in ``dagua.layout.ops.embed``."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dagua.layout.ops import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.embed import (
    SVD,
    BuildLaplacian,
    BuildLaplacianConfig,
    BuildNormalizedAdjacency,
    BuildNormalizedAdjacencyConfig,
    CurveFit_ab,
    CurveFitABConfig,
    Eigendecomposition,
    EigendecompositionConfig,
    FuzzySimplicialSet,
    GCNForward,
    GCNForwardConfig,
    PerplexityMatch,
    PerplexityMatchConfig,
    Pseudoinverse,
    SmoothKNNBandwidth,
    SmoothKNNBandwidthConfig,
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


def _csr_payload_from_dense(adjacency: torch.Tensor) -> dict[str, torch.Tensor]:
    """Convert a dense adjacency tensor into the ops CSR payload format.

    Parameters
    ----------
    adjacency : torch.Tensor
        Dense adjacency matrix with shape ``[N, N]``.

    Returns
    -------
    dict[str, torch.Tensor]
        Payload containing CSR ``indptr``, ``indices``, and ``weights`` tensors.
    """
    row_counts = (adjacency != 0).sum(dim=1, dtype=torch.long)
    indptr = torch.zeros((adjacency.shape[0] + 1,), dtype=torch.long)
    indptr[1:] = torch.cumsum(row_counts, dim=0)
    row_index, col_index = torch.nonzero(adjacency != 0, as_tuple=True)
    weights = adjacency[row_index, col_index].to(dtype=torch.float64)
    return {
        "indptr": indptr,
        "indices": col_index.to(dtype=torch.long),
        "weights": weights,
    }


def _distance_matrix() -> torch.Tensor:
    """Build a small deterministic distance matrix for t-SNE and UMAP tests.

    Returns
    -------
    torch.Tensor
        Symmetric distance matrix with shape ``[5, 5]``.
    """
    return torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [1.0, 0.0, 1.2, 2.2, 3.1],
            [2.0, 1.2, 0.0, 1.1, 2.0],
            [3.0, 2.2, 1.1, 0.0, 0.9],
            [4.0, 3.1, 2.0, 0.9, 0.0],
        ],
        dtype=torch.float32,
    )


def _fuzzy_graph_to_dense(
    fuzzy_graph: dict[str, torch.Tensor],
    num_nodes: int,
) -> torch.Tensor:
    """Convert a sparse fuzzy-graph payload into a dense symmetric matrix.

    Parameters
    ----------
    fuzzy_graph : dict[str, torch.Tensor]
        Sparse UMAP graph payload with ``head``, ``tail``, and ``weight``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    torch.Tensor
        Dense adjacency matrix with shape ``[N, N]``.
    """
    dense = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for head, tail, weight in zip(
        fuzzy_graph["head"].tolist(),
        fuzzy_graph["tail"].tolist(),
        fuzzy_graph["weight"].tolist(),
    ):
        dense[head, tail] = float(weight)
        dense[tail, head] = float(weight)
    return dense


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


def test_symmetrize_adjacency_keeps_empty_dense_graph_empty() -> None:
    """SymmetrizeAdjacency should preserve the shape of an empty dense graph."""

    state = SolveState(adjacency=torch.zeros((0, 0), dtype=torch.float32))

    result = SymmetrizeAdjacency().apply(_path_problem(0), state, RuntimeContext())

    assert isinstance(result.adjacency, torch.Tensor)
    assert result.adjacency.shape == (0, 0)


def test_symmetrize_adjacency_preserves_symmetric_support_for_dense_inputs() -> None:
    """Symmetrizing an already symmetric matrix should keep the same edge support."""

    adjacency = _dense_path_adjacency(4)
    state = SolveState(adjacency=adjacency.clone())

    result = SymmetrizeAdjacency().apply(_path_problem(4), state, RuntimeContext())

    assert isinstance(result.adjacency, torch.Tensor)
    assert torch.equal(result.adjacency != 0, adjacency != 0)
    assert torch.allclose(result.adjacency, result.adjacency.transpose(0, 1))


def test_symmetrize_adjacency_preserves_csr_payload_type() -> None:
    """SymmetrizeAdjacency should round-trip the CSR payload container type."""

    adjacency = torch.tensor(
        [
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    state = SolveState(adjacency=_csr_payload_from_dense(adjacency))

    result = SymmetrizeAdjacency().apply(_path_problem(3), state, RuntimeContext())

    assert isinstance(result.adjacency, dict)
    reconstructed = torch.zeros((3, 3), dtype=torch.float64)
    for row_idx in range(3):
        start = int(result.adjacency["indptr"][row_idx].item())
        stop = int(result.adjacency["indptr"][row_idx + 1].item())
        for offset in range(start, stop):
            column = int(result.adjacency["indices"][offset].item())
            reconstructed[row_idx, column] = float(result.adjacency["weights"][offset].item())
    torch.testing.assert_close(
        reconstructed,
        torch.tensor(
            [
                [0.0, 2.0, 1.0],
                [2.0, 0.0, 3.0],
                [1.0, 3.0, 0.0],
            ],
            dtype=torch.float64,
        ),
    )


def test_build_laplacian_random_walk_matches_reference() -> None:
    """Random-walk Laplacians should match ``I - D^-1 A``."""

    adjacency = _dense_path_adjacency(5)
    state = SolveState(adjacency=adjacency)

    result = BuildLaplacian(BuildLaplacianConfig(normalization="random_walk")).apply(
        _path_problem(5),
        state,
        RuntimeContext(),
    )

    laplacian = result.extras["laplacian"].toarray()
    adjacency_np = adjacency.numpy().astype(np.float64, copy=False)
    degrees = adjacency_np.sum(axis=1)
    inv_degree = np.zeros_like(degrees)
    inv_degree[degrees > 0.0] = 1.0 / degrees[degrees > 0.0]
    expected = np.eye(5, dtype=np.float64) - np.diag(inv_degree) @ adjacency_np

    assert np.allclose(laplacian, expected)


def test_build_laplacian_symmetric_normalization_has_eigenvalues_in_unit_interval() -> None:
    """The symmetric normalized Laplacian spectrum should stay within ``[0, 2]``."""

    adjacency = _dense_path_adjacency(8)
    state = SolveState(adjacency=adjacency)

    result = BuildLaplacian(BuildLaplacianConfig(normalization="symmetric")).apply(
        _path_problem(8),
        state,
        RuntimeContext(),
    )

    eigenvalues = np.linalg.eigvalsh(result.extras["laplacian"].toarray())

    assert np.all(eigenvalues >= -1.0e-8)
    assert np.all(eigenvalues <= 2.0 + 1.0e-8)


def test_build_laplacian_empty_graph_produces_empty_matrix() -> None:
    """BuildLaplacian should handle the zero-node case without special casing in tests."""

    state = SolveState(adjacency=torch.zeros((0, 0), dtype=torch.float32))

    result = BuildLaplacian().apply(_path_problem(0), state, RuntimeContext())

    assert result.extras["laplacian"].shape == (0, 0)


def test_build_normalized_adjacency_adds_self_loops() -> None:
    """BuildNormalizedAdjacency should produce a positive diagonal by default."""

    problem = _path_problem(4)
    state = SolveState(adjacency=_dense_path_adjacency(4))

    result = BuildNormalizedAdjacency().apply(problem, state, RuntimeContext())

    dense = result.extras["normalized_adjacency"].to_dense()
    assert torch.all(torch.diag(dense) > 0.0)


def test_build_normalized_adjacency_output_is_symmetric() -> None:
    """BuildNormalizedAdjacency should return a symmetric normalized matrix."""

    problem = _path_problem(5)
    state = SolveState(adjacency=_dense_path_adjacency(5))

    result = BuildNormalizedAdjacency().apply(problem, state, RuntimeContext())

    dense = result.extras["normalized_adjacency"].to_dense()
    torch.testing.assert_close(dense, dense.transpose(0, 1))


def test_build_normalized_adjacency_shape_matches_input_size() -> None:
    """BuildNormalizedAdjacency should preserve the square graph shape."""

    problem = _path_problem(6)
    state = SolveState(adjacency=_dense_path_adjacency(6))

    result = BuildNormalizedAdjacency().apply(problem, state, RuntimeContext())

    assert result.extras["normalized_adjacency"].shape == (6, 6)


def test_build_normalized_adjacency_can_skip_self_loops() -> None:
    """Disabling self-loops should leave the diagonal at zero for a loop-free graph."""

    problem = _path_problem(4)
    state = SolveState(adjacency=_dense_path_adjacency(4))

    result = BuildNormalizedAdjacency(BuildNormalizedAdjacencyConfig(add_self_loops=False)).apply(
        problem, state, RuntimeContext()
    )

    dense = result.extras["normalized_adjacency"].to_dense()
    torch.testing.assert_close(torch.diag(dense), torch.zeros(4))


def test_eigendecomposition_dense_and_sparse_modes_agree_on_small_graph() -> None:
    """Dense and sparse eigensolvers should match on the same small Laplacian."""

    problem = _path_problem(6)
    state = SolveState(adjacency=_dense_path_adjacency(6))
    BuildLaplacian().apply(problem, state, RuntimeContext())

    dense_state = Eigendecomposition(EigendecompositionConfig(sparse_threshold=100, k=2)).apply(
        problem,
        SolveState(extras=dict(state.extras)),
        RuntimeContext(),
    )
    sparse_state = Eigendecomposition(EigendecompositionConfig(sparse_threshold=0, k=2)).apply(
        problem,
        SolveState(extras=dict(state.extras)),
        RuntimeContext(),
    )

    dense_values = dense_state.extras["eigenpairs"]["eigenvalues"]
    sparse_values = sparse_state.extras["eigenpairs"]["eigenvalues"]
    dense_vectors = dense_state.extras["eigenpairs"]["eigenvectors"].to(dtype=torch.float64)
    sparse_vectors = sparse_state.extras["eigenpairs"]["eigenvectors"].to(dtype=torch.float64)

    torch.testing.assert_close(dense_values, sparse_values, atol=1.0e-5, rtol=1.0e-5)
    dense_projector = dense_vectors @ dense_vectors.transpose(0, 1)
    sparse_projector = sparse_vectors @ sparse_vectors.transpose(0, 1)
    torch.testing.assert_close(dense_projector, sparse_projector, atol=1.0e-4, rtol=1.0e-4)


def test_eigendecomposition_handles_zero_node_graph() -> None:
    """Eigendecomposition should return empty tensors for an empty Laplacian."""

    state = SolveState(extras={"laplacian": np.zeros((0, 0), dtype=np.float64)})

    result = Eigendecomposition(EigendecompositionConfig(k=2)).apply(
        _path_problem(0),
        state,
        RuntimeContext(),
    )

    eigenpairs = result.extras["eigenpairs"]
    assert eigenpairs["eigenvalues"].shape == (0,)
    assert eigenpairs["eigenvectors"].shape == (0, 0)


def test_eigendecomposition_handles_degenerate_zero_laplacian() -> None:
    """Disconnected zero graphs should return zero eigenvalues for the requested pairs."""

    state = SolveState(adjacency=torch.zeros((3, 3), dtype=torch.float32))
    BuildLaplacian(BuildLaplacianConfig(normalization="unnormalized")).apply(
        _path_problem(3),
        state,
        RuntimeContext(),
    )

    result = Eigendecomposition(EigendecompositionConfig(k=2)).apply(
        _path_problem(3),
        state,
        RuntimeContext(),
    )

    expected = torch.zeros(2, dtype=result.extras["eigenpairs"]["eigenvalues"].dtype)
    torch.testing.assert_close(result.extras["eigenpairs"]["eigenvalues"], expected)


def test_svd_returns_expected_factor_shapes() -> None:
    """SVD should return compact factor shapes for rectangular inputs."""

    state = SolveState(
        pivot_distances=torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            dtype=torch.float32,
        )
    )

    result = SVD().apply(_path_problem(3), state, RuntimeContext())

    assert result.extras["svd_result"]["u"].shape == (3, 2)
    assert result.extras["svd_result"]["s"].shape == (2,)
    assert result.extras["svd_result"]["vh"].shape == (2, 2)


def test_svd_singular_values_are_non_negative() -> None:
    """SVD singular values should be non-negative."""

    state = SolveState(
        pivot_distances=torch.tensor(
            [[1.0, -1.0, 0.0], [0.0, 2.0, 2.0]],
            dtype=torch.float32,
        )
    )

    result = SVD().apply(_path_problem(2), state, RuntimeContext())

    assert torch.all(result.extras["svd_result"]["s"] >= 0.0)


def test_pseudoinverse_satisfies_moore_penrose_reconstruction() -> None:
    """The stored pseudoinverse should satisfy ``A @ A^+ @ A ~= A``."""

    state = SolveState(adjacency=_dense_path_adjacency(4))
    BuildLaplacian(BuildLaplacianConfig(normalization="unnormalized")).apply(
        _path_problem(4),
        state,
        RuntimeContext(),
    )

    result = Pseudoinverse().apply(_path_problem(4), state, RuntimeContext())

    laplacian = state.extras["laplacian"].toarray()
    pinv = result.extras["laplacian_pinv"].numpy()
    assert np.allclose(laplacian @ pinv @ laplacian, laplacian, atol=1.0e-6)


@pytest.mark.parametrize(
    ("hidden_sizes", "output_dim"),
    [
        ((4, 2), 2),
        ((8, 3), 4),
        ((6, 1), 1),
    ],
)
def test_gcn_forward_supports_multiple_hidden_size_configurations(
    hidden_sizes: tuple[int, int],
    output_dim: int,
) -> None:
    """GCNForward should support the requested hidden-size and output-dim combinations."""

    problem = _path_problem(5)
    state = SolveState(adjacency=_dense_path_adjacency(5))
    BuildNormalizedAdjacency().apply(problem, state, RuntimeContext())

    result = GCNForward(GCNForwardConfig(hidden_sizes=hidden_sizes, output_dim=output_dim)).apply(
        problem,
        state,
        RuntimeContext(),
    )

    assert result.pos is not None
    assert result.pos.shape == (5, output_dim)


def test_gcn_forward_reuses_cached_model_when_configuration_is_unchanged() -> None:
    """GCNForward should reuse the cached model for the same config and adjacency object."""

    problem = _path_problem(4)
    state = SolveState(adjacency=_dense_path_adjacency(4))
    BuildNormalizedAdjacency().apply(problem, state, RuntimeContext())
    op = GCNForward(GCNForwardConfig(hidden_sizes=(5, 2), output_dim=2))

    first = op.apply(problem, state, RuntimeContext())
    cached_model = first.extras["gcn_model"]
    second = op.apply(problem, first, RuntimeContext())

    assert second.extras["gcn_model"] is cached_model


def test_gcn_forward_rebuilds_model_when_hidden_sizes_change() -> None:
    """GCNForward should invalidate the cached model when the config changes."""

    problem = _path_problem(4)
    state = SolveState(adjacency=_dense_path_adjacency(4))
    BuildNormalizedAdjacency().apply(problem, state, RuntimeContext())

    first = GCNForward(GCNForwardConfig(hidden_sizes=(4, 2), output_dim=2)).apply(
        problem,
        state,
        RuntimeContext(),
    )
    first_model = first.extras["gcn_model"]
    second = GCNForward(GCNForwardConfig(hidden_sizes=(6, 2), output_dim=2)).apply(
        problem,
        first,
        RuntimeContext(),
    )

    assert second.extras["gcn_model"] is not first_model


def test_gcn_forward_allows_gradients_to_flow_through_the_model() -> None:
    """GCNForward outputs should remain connected to the model parameters for backprop."""

    problem = _path_problem(4)
    state = SolveState(adjacency=_dense_path_adjacency(4))
    BuildNormalizedAdjacency().apply(problem, state, RuntimeContext())

    result = GCNForward(GCNForwardConfig(hidden_sizes=(5, 3), output_dim=2)).apply(
        problem,
        state,
        RuntimeContext(),
    )
    loss = result.pos.square().sum()
    loss.backward()
    model = result.extras["gcn_model"]

    assert model.weight1.grad is not None
    assert model.weight2.grad is not None
    assert torch.isfinite(model.weight1.grad).all()
    assert torch.isfinite(model.weight2.grad).all()


def test_perplexity_row_matches_the_requested_target_perplexity() -> None:
    """The binary search in ``_perplexity_row`` should hit the target perplexity."""

    row = torch.tensor([0.0, 1.0, 2.0, 4.0, 8.0], dtype=torch.float32)

    probabilities = _perplexity_row(row, perplexity=2.5, tol=1.0e-5, max_iter=200)
    active = probabilities[probabilities > 0]
    entropy = -(active * active.log()).sum()
    perplexity = float(torch.exp(entropy).item())

    assert perplexity == pytest.approx(2.5, rel=5.0e-3, abs=5.0e-3)


def test_perplexity_row_lower_perplexity_produces_a_peakier_distribution() -> None:
    """Lower perplexity should concentrate more mass on the nearest neighbor."""

    row = torch.tensor([0.0, 1.0, 2.0, 4.0, 8.0], dtype=torch.float32)

    low = _perplexity_row(row, perplexity=1.5, tol=1.0e-5, max_iter=200)
    high = _perplexity_row(row, perplexity=3.5, tol=1.0e-5, max_iter=200)

    assert float(low.max().item()) > float(high.max().item())


def test_perplexity_match_perplexity_configuration_changes_neighbor_mass() -> None:
    """Changing perplexity should change the symmetric probability matrix."""

    distance_matrix = _distance_matrix()
    low_state = SolveState(distance_matrix=distance_matrix)
    high_state = SolveState(distance_matrix=distance_matrix)
    problem = _path_problem(distance_matrix.shape[0])

    low = PerplexityMatch(PerplexityMatchConfig(perplexity=1.5)).apply(
        problem,
        low_state,
        RuntimeContext(),
    )
    high = PerplexityMatch(PerplexityMatchConfig(perplexity=3.0)).apply(
        problem,
        high_state,
        RuntimeContext(),
    )

    assert not torch.allclose(low.extras["probabilities"], high.extras["probabilities"])
    assert float(low.extras["probabilities"][0, 1].item()) > float(
        high.extras["probabilities"][0, 1].item()
    )


def test_perplexity_match_converges_to_a_finite_normalized_probability_matrix() -> None:
    """PerplexityMatch should converge to a finite, normalized matrix on dense inputs."""

    state = SolveState(distance_matrix=_distance_matrix())

    result = PerplexityMatch(PerplexityMatchConfig(perplexity=2.0, max_iter=200)).apply(
        _path_problem(5),
        state,
        RuntimeContext(),
    )

    probabilities = result.extras["probabilities"]
    assert torch.isfinite(probabilities).all()
    assert float(probabilities.sum().item()) == pytest.approx(1.0, rel=1.0e-5, abs=1.0e-5)


def test_smooth_knn_bandwidth_outputs_positive_sigmas_and_nonnegative_rhos() -> None:
    """SmoothKNNBandwidth should store valid positive bandwidth terms."""

    state = SolveState(distance_matrix=_distance_matrix())

    result = SmoothKNNBandwidth(SmoothKNNBandwidthConfig(n_neighbors=3)).apply(
        _path_problem(5),
        state,
        RuntimeContext(),
    )

    assert torch.all(result.extras["sigmas"] > 0.0)
    assert torch.all(result.extras["rhos"] >= 0.0)


def test_smooth_knn_bandwidth_neighbor_count_changes_bandwidths() -> None:
    """Changing ``n_neighbors`` should change the stored bandwidth estimates."""

    distance_matrix = _distance_matrix()
    problem = _path_problem(5)
    small = SmoothKNNBandwidth(SmoothKNNBandwidthConfig(n_neighbors=2)).apply(
        problem,
        SolveState(distance_matrix=distance_matrix),
        RuntimeContext(),
    )
    large = SmoothKNNBandwidth(SmoothKNNBandwidthConfig(n_neighbors=4)).apply(
        problem,
        SolveState(distance_matrix=distance_matrix),
        RuntimeContext(),
    )

    assert small.extras["umap_n_neighbors"] == 2
    assert large.extras["umap_n_neighbors"] == 4
    assert not torch.allclose(small.extras["sigmas"], large.extras["sigmas"])


def test_fuzzy_simplicial_set_is_symmetric_when_reconstructed() -> None:
    """The fuzzy simplicial set should reconstruct to a symmetric dense graph."""

    state = SolveState(distance_matrix=_distance_matrix())
    problem = _path_problem(5)
    SmoothKNNBandwidth(SmoothKNNBandwidthConfig(n_neighbors=3)).apply(
        problem,
        state,
        RuntimeContext(),
    )

    result = FuzzySimplicialSet().apply(problem, state, RuntimeContext())
    dense = _fuzzy_graph_to_dense(result.extras["fuzzy_graph"], num_nodes=5)

    torch.testing.assert_close(dense, dense.transpose(0, 1))


def test_fuzzy_simplicial_set_weights_are_bounded_and_sparse() -> None:
    """UMAP fuzzy graph weights should lie in ``[0, 1]`` and stay sparse."""

    state = SolveState(distance_matrix=_distance_matrix())
    problem = _path_problem(5)
    SmoothKNNBandwidth(SmoothKNNBandwidthConfig(n_neighbors=2)).apply(
        problem,
        state,
        RuntimeContext(),
    )

    result = FuzzySimplicialSet().apply(problem, state, RuntimeContext())
    fuzzy_graph = result.extras["fuzzy_graph"]

    assert torch.all(fuzzy_graph["weight"] >= 0.0)
    assert torch.all(fuzzy_graph["weight"] <= 1.0)
    assert fuzzy_graph["head"].numel() < 25


def test_curve_fit_ab_changes_with_min_dist_configuration() -> None:
    """Changing ``min_dist`` should change the fitted UMAP curve parameters."""

    problem = _path_problem(3)
    close = CurveFit_ab(CurveFitABConfig(min_dist=0.05, spread=1.0)).apply(
        problem,
        SolveState(),
        RuntimeContext(),
    )
    far = CurveFit_ab(CurveFitABConfig(min_dist=0.5, spread=1.0)).apply(
        problem,
        SolveState(),
        RuntimeContext(),
    )

    assert close.extras["umap_a"] != pytest.approx(far.extras["umap_a"])
    assert close.extras["umap_b"] != pytest.approx(far.extras["umap_b"])


def test_curve_fit_ab_accepts_zero_min_dist() -> None:
    """CurveFit_ab should support the boundary case ``min_dist=0``."""

    result = CurveFit_ab(CurveFitABConfig(min_dist=0.0, spread=1.0)).apply(
        _path_problem(3),
        SolveState(),
        RuntimeContext(),
    )

    assert result.extras["umap_a"] > 0.0
    assert result.extras["umap_b"] > 0.0
