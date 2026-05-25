"""Regression tests for opt-in spectral NetworkX fidelity mode."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch
from scipy import sparse

from dagua.eval.competitors.classic_competitor import ClassicSpectral
from dagua.eval.competitors.networkx_competitor import (
    NetworkXSpectral,
    _graph_to_nx,
    _nx_pos_to_tensor,
)
from dagua.eval.variants import get_variant
from dagua.graph import DaguaGraph
from dagua.layout.ops.embed import (
    _dense_spectral_embedding,
    _select_embedding_columns,
    _sparse_spectral_embedding,
)
from dagua.layout.ops.pipelines.spectral import layout_spectral_pipeline
from dagua.layout.ops.preprocess import _build_spectral_adjacency, _spectral_laplacian

embed_ops = importlib.import_module("dagua.layout.ops.embed")
spectral_pipeline = importlib.import_module("dagua.layout.ops.pipelines.spectral")


def _single_edge_index() -> torch.Tensor:
    """Build a two-node edge tensor.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, 1]``.
    """
    return torch.tensor([[0], [1]], dtype=torch.long)


def _path_edge_index(num_nodes: int) -> torch.Tensor:
    """Build a directed path edge tensor.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    return torch.tensor(
        [[index for index in range(num_nodes - 1)], [index + 1 for index in range(num_nodes - 1)]],
        dtype=torch.long,
    )


def _weighted_duplicate_graph() -> DaguaGraph:
    """Build a graph with duplicate weighted edges.

    Returns
    -------
    DaguaGraph
        Three-node graph where edge ``0 -> 1`` appears twice with distinct
        weights.
    """
    graph = DaguaGraph()
    graph.add_edge(0, 1, weight=2.0)
    graph.add_edge(0, 1, weight=3.0)
    graph.add_edge(1, 2, weight=4.0)
    graph.compute_node_sizes()
    return graph


def test_networkx_fidelity_collapses_two_node_graph_to_center() -> None:
    """NetworkX fidelity should match the reference two-node special case."""
    edge_index = _single_edge_index()

    positions = layout_spectral_pipeline(
        edge_index=edge_index,
        num_nodes=2,
        networkx_fidelity=True,
    )
    default_positions = layout_spectral_pipeline(edge_index=edge_index, num_nodes=2)

    assert torch.equal(positions, torch.zeros((2, 2), dtype=torch.float32))
    assert not torch.equal(default_positions, positions)


def test_networkx_fidelity_forces_unnormalized_laplacian() -> None:
    """NetworkX fidelity should override Dagua's symmetric default Laplacian."""
    edge_index = _path_edge_index(5)

    fidelity_positions = layout_spectral_pipeline(
        edge_index=edge_index,
        num_nodes=5,
        normalization="symmetric",
        networkx_fidelity=True,
    )
    unnormalized_positions = layout_spectral_pipeline(
        edge_index=edge_index,
        num_nodes=5,
        normalization="unnormalized",
        networkx_fidelity=True,
    )

    assert torch.equal(fidelity_positions, unnormalized_positions)


def test_networkx_fidelity_selects_sorted_slice_after_first_eigenvector() -> None:
    """NetworkX fidelity should keep extra zero modes after skipping index zero."""
    eigenvalues = np.array([0.0, 0.0, 2.0, 3.0], dtype=np.float64)
    eigenvectors = np.eye(4, dtype=np.float64)

    selected = _select_embedding_columns(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        dim=2,
        skip_first=True,
    )
    robust_selected = _select_embedding_columns(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        dim=2,
    )

    np.testing.assert_array_equal(selected, eigenvectors[:, [1, 2]])
    np.testing.assert_array_equal(robust_selected, eigenvectors[:, [2, 3]])


def test_networkx_fidelity_variant_is_registered_against_nx_spectral() -> None:
    """The opt-in fidelity path should be benchmark-addressable as a variant."""
    variant = get_variant("classic_spectral_nx_fidelity")

    assert variant is not None
    assert variant.reimpl_params == {"networkx_fidelity": True}
    assert variant.original_engine == "nx_spectral"


def test_classic_spectral_direct_adapter_forwards_edge_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The direct spectral competitor path should preserve NetworkX parity inputs."""
    graph = _weighted_duplicate_graph()
    seen: dict[str, torch.Tensor | bool | None] = {}

    def _layout_fake(
        edge_index: torch.Tensor,
        num_nodes: int,
        node_sizes: torch.Tensor | None = None,
        seed: int = 42,
        edge_weights: torch.Tensor | None = None,
        networkx_fidelity: bool = False,
    ) -> torch.Tensor:
        """Capture forwarded parity inputs and return a valid position tensor.

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge tensor with shape ``[2, E]``.
        num_nodes : int
            Number of graph nodes.
        node_sizes : torch.Tensor | None, default=None
            Optional node-size tensor. Unused by this fake.
        seed : int, default=42
            Resolved layout seed. Unused by this fake.
        edge_weights : torch.Tensor | None, default=None
            Optional edge-weight tensor with shape ``[E]``.
        networkx_fidelity : bool, default=False
            Whether the direct adapter requested NetworkX-compatible layout
            semantics.

        Returns
        -------
        torch.Tensor
            Zero position tensor with shape ``[N, 2]``.
        """
        _ = edge_index, node_sizes, seed
        seen["edge_weights"] = edge_weights
        seen["networkx_fidelity"] = networkx_fidelity
        return torch.zeros((num_nodes, 2), dtype=torch.float32)

    monkeypatch.setattr(spectral_pipeline, "layout_spectral_pipeline", _layout_fake)

    result = ClassicSpectral().layout(graph)

    assert result.error is None
    assert result.pos is not None
    assert seen["edge_weights"] is not None
    assert seen["edge_weights"].tolist() == [2.0, 3.0, 4.0]
    assert seen["networkx_fidelity"] is True


def test_networkx_graph_converter_default_sums_parallel_edges() -> None:
    """The shared NetworkX converter should keep its duplicate-summing default."""
    graph = _weighted_duplicate_graph()

    nx_graph = _graph_to_nx(graph)

    assert nx_graph[0][1]["weight"] == 5.0
    assert nx_graph[1][2]["weight"] == 4.0


def test_networkx_spectral_reference_uses_last_duplicate_edge() -> None:
    """The spectral reference should match repeated ``DiGraph.add_edge`` semantics."""
    assert NetworkXSpectral.duplicate_policy == "last"


def test_networkx_fidelity_adjacency_keeps_last_duplicate_edge() -> None:
    """NetworkX fidelity should mirror repeated ``DiGraph.add_edge`` semantics."""
    graph = _weighted_duplicate_graph()

    adjacency = _build_spectral_adjacency(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        edge_weights=graph.edge_weights,
        duplicate_policy="last",
    )
    summed_adjacency = _build_spectral_adjacency(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        edge_weights=graph.edge_weights,
    )

    assert adjacency[0, 1] == 3.0
    assert adjacency[1, 2] == 4.0
    assert summed_adjacency[0, 1] == 5.0


def test_igraph_fidelity_normalized_laplacian_zeros_isolated_diagonal() -> None:
    """igraph fidelity should mirror normalized isolated-node diagonal handling."""
    edge_index = _path_edge_index(3)
    adjacency = _build_spectral_adjacency(edge_index=edge_index, num_nodes=4)
    symmetric_adjacency = adjacency + adjacency.T

    default_laplacian, _ = _spectral_laplacian(
        adjacency=symmetric_adjacency,
        normalization="symmetric",
    )
    igraph_laplacian, _ = _spectral_laplacian(
        adjacency=symmetric_adjacency,
        normalization="symmetric",
        igraph_fidelity=True,
    )

    assert default_laplacian[3, 3] == 1.0
    assert igraph_laplacian[3, 3] == 0.0


def test_igraph_fidelity_mode_keeps_disconnected_zero_mode() -> None:
    """The opt-in igraph path should retain one extra zero mode after the first."""
    edge_index = _path_edge_index(3)

    default_positions = layout_spectral_pipeline(edge_index=edge_index, num_nodes=4)
    igraph_positions = layout_spectral_pipeline(
        edge_index=edge_index,
        num_nodes=4,
        fidelity_mode="igraph",
    )

    assert not torch.equal(default_positions, igraph_positions)
    assert torch.count_nonzero(igraph_positions[3]).item() > 0


def test_networkx_spectral_adapter_uses_raw_algorithm_scale() -> None:
    """The spectral NetworkX adapter should not apply the legacy 500x scale."""
    tensor = _nx_pos_to_tensor(
        {0: (1.0, -2.0)}, num_nodes=1, output_scale=NetworkXSpectral.output_scale
    )

    assert NetworkXSpectral.output_scale == 1.0
    assert torch.equal(tensor, torch.tensor([[1.0, -2.0]]))


def test_networkx_fidelity_dense_branch_uses_generic_eig(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NetworkX fidelity should use ``np.linalg.eig`` on dense Laplacians."""
    calls = {"eig": 0, "eigh": 0}

    def _eig(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Record generic eigensolver use.

        Parameters
        ----------
        matrix : numpy.ndarray
            Dense Laplacian matrix with shape ``[N, N]``.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Eigenvalues and eigenvectors.
        """
        calls["eig"] += 1
        return np.array([0.0, 1.0, 2.0]), np.eye(matrix.shape[0])

    def _eigh(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Record symmetric eigensolver use.

        Parameters
        ----------
        matrix : numpy.ndarray
            Dense Laplacian matrix with shape ``[N, N]``.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Eigenvalues and eigenvectors.
        """
        calls["eigh"] += 1
        return np.array([0.0, 1.0, 2.0]), np.eye(matrix.shape[0])

    monkeypatch.setattr(embed_ops.np.linalg, "eig", _eig)
    monkeypatch.setattr(embed_ops.np.linalg, "eigh", _eigh)

    _dense_spectral_embedding(
        laplacian=sparse.eye(3, format="csr"),
        dim=2,
        symmetric=True,
        networkx_fidelity=True,
    )

    assert calls == {"eig": 1, "eigh": 0}


def test_networkx_fidelity_sparse_branch_matches_reference_k_and_ncv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NetworkX fidelity should request ``dim + 1`` sparse eigenpairs."""
    captured: dict[str, int | str] = {}

    def _eigsh(
        laplacian: sparse.csr_matrix,
        k: int,
        which: str,
        ncv: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Capture sparse eigensolver sizing.

        Parameters
        ----------
        laplacian : scipy.sparse.csr_matrix
            Sparse Laplacian matrix with shape ``[N, N]``.
        k : int
            Requested eigenpair count.
        which : str
            ARPACK eigenvalue selector.
        ncv : int
            Requested Lanczos vector count.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Eigenvalues and eigenvectors.
        """
        captured.update({"k": k, "which": which, "ncv": ncv})
        return np.array([0.0, 1.0, 2.0]), np.eye(laplacian.shape[0], k)

    monkeypatch.setattr(embed_ops.sparse_linalg, "eigsh", _eigsh)

    _sparse_spectral_embedding(
        laplacian=sparse.eye(500, format="csr"),
        dim=2,
        symmetric=True,
        networkx_fidelity=True,
    )

    assert captured == {"k": 3, "which": "SM", "ncv": 22}
