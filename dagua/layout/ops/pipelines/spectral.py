"""Spectral graph layout pipeline."""

from __future__ import annotations

from typing import Optional

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.embed import SpectralEmbed
from dagua.layout.ops.postprocess import SpectralFinalizePositions
from dagua.layout.ops.preprocess import SpectralPrepareState
from dagua.layout.ops.state import (  # noqa: E402
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)

SPARSE_EIGEN_THRESHOLD = 500


def build_spectral_pipeline(
    normalization: str = "symmetric",
    sparse_threshold: int = SPARSE_EIGEN_THRESHOLD,
    networkx_fidelity: bool = False,
) -> Pipeline:
    """Build a spectral graph layout pipeline.

    Parameters
    ----------
    normalization : str, default="symmetric"
        Laplacian normalization mode.
    sparse_threshold : int, default=500
        Node-count threshold for the sparse eigensolver branch.
    networkx_fidelity : bool, default=False
        Whether to mirror NetworkX spectral-layout edge cases and eigenvector
        selection while preserving the public Dagua default when disabled.

    Returns
    -------
    Pipeline
        Pipeline implementing the spectral-layout algorithm. The pipeline
        produces final node coordinates by preparing the requested graph
        Laplacian, solving the leading non-trivial eigenvectors with dense or
        sparse eigendecomposition, and finalizing the embedding.

    Raises
    ------
    ValueError
        If ``sparse_threshold`` is not positive.
    """
    if sparse_threshold <= 0:
        raise ValueError("sparse_threshold must be positive.")

    effective_normalization = "unnormalized" if networkx_fidelity else normalization
    return Pipeline(
        [
            SpectralPrepareState(
                normalization=effective_normalization,
                networkx_fidelity=networkx_fidelity,
            ),
            SpectralEmbed(sparse_threshold=sparse_threshold, networkx_fidelity=networkx_fidelity),
            SpectralFinalizePositions(),
        ],
        name="spectral_pipeline",
    )


def layout_spectral_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
    normalization: str = "symmetric",
    networkx_fidelity: bool = False,
) -> torch.Tensor:
    """Run the spectral graph layout pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]``. Unused and accepted
        for interface compatibility.
    seed : int, default=42
        Accepted for interface compatibility. Spectral layout is deterministic
        once the graph is fixed.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.
    normalization : str, default="symmetric"
        Laplacian normalization mode.
    networkx_fidelity : bool, default=False
        Whether to mirror NetworkX's unnormalized Laplacian, trivial two-node
        output, and eigenvector-selection behavior.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If ``num_nodes`` or ``edge_weights`` are invalid.
    RuntimeError
        If the pipeline fails to populate final positions.
    """
    _ = seed

    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must have shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge count {edge_index.shape[1]}"
            )

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_spectral_pipeline(
        normalization=normalization,
        networkx_fidelity=networkx_fidelity,
    ).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("Spectral pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_spectral_pipeline", "layout_spectral_pipeline"]
