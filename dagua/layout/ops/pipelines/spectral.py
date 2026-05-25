"""Spectral graph layout pipeline."""

from __future__ import annotations

from typing import Optional, Union

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


def _resolve_spectral_fidelity_mode(
    fidelity_mode: Optional[Union[str, bool]],
    networkx_fidelity: bool,
) -> tuple[bool, bool]:
    """Resolve reference-specific spectral compatibility flags.

    Parameters
    ----------
    fidelity_mode : str | bool | None
        Optional fidelity target. ``"networkx"`` preserves the historical
        NetworkX path, ``"igraph"`` enables igraph Laplacian and eigenvector
        selection semantics, and ``True`` maps to ``"igraph"`` for consistency
        with other igraph-targeted pipelines.
    networkx_fidelity : bool
        Legacy NetworkX compatibility flag.

    Returns
    -------
    tuple[bool, bool]
        Resolved ``(networkx_fidelity, igraph_fidelity)`` flags.

    Raises
    ------
    ValueError
        If the requested fidelity mode is unsupported or conflicting.
    """
    if fidelity_mode is None or fidelity_mode is False:
        return bool(networkx_fidelity), False
    if fidelity_mode is True:
        if networkx_fidelity:
            raise ValueError("fidelity_mode=True conflicts with networkx_fidelity=True.")
        return False, True

    normalized_mode = str(fidelity_mode).lower()
    if normalized_mode == "networkx":
        return True, False
    if normalized_mode == "igraph":
        if networkx_fidelity:
            raise ValueError("fidelity_mode='igraph' conflicts with networkx_fidelity=True.")
        return False, True
    raise ValueError("fidelity_mode must be one of None, False, True, 'networkx', or 'igraph'.")


def build_spectral_pipeline(
    normalization: str = "symmetric",
    sparse_threshold: int = SPARSE_EIGEN_THRESHOLD,
    networkx_fidelity: bool = False,
    fidelity_mode: Optional[Union[str, bool]] = None,
    fidelity_dtype: torch.dtype = torch.float32,
) -> Pipeline:
    """Build a spectral graph layout pipeline.

    Reference fidelity
    ------------------
    Targets: NetworkX 3.6.1 ``spectral_layout`` / Hall (1970), "An
        r-Dimensional Quadratic Placement Algorithm".
    Fidelity mode: ``networkx_fidelity=True`` switches to NetworkX edge cases,
        unnormalized Laplacian behavior, and eigenvector selection.
    Verified at: final 100-seed report, strong equivalent; median RMSD 0.000
        for both default and NetworkX-fidelity variants.
    Known divergences:
        - Sparse/dense eigensolver choice is Dagua-controlled by node count.
        - Native tensor finalization keeps Dagua's device and dtype contracts.

    Parameters
    ----------
    normalization : str, default="symmetric"
        Laplacian normalization mode.
    sparse_threshold : int, default=500
        Node-count threshold for the sparse eigensolver branch.
    networkx_fidelity : bool, default=False
        Whether to mirror NetworkX spectral-layout edge cases and eigenvector
        selection while preserving the public Dagua default when disabled.
    fidelity_mode : str | bool | None, default=None
        Optional reference target. ``"igraph"`` mirrors igraph normalized
        Laplacian and eigenvector selection details; ``"networkx"`` is
        equivalent to ``networkx_fidelity=True``.

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

    networkx_mode, igraph_mode = _resolve_spectral_fidelity_mode(
        fidelity_mode=fidelity_mode,
        networkx_fidelity=networkx_fidelity,
    )
    effective_normalization = "unnormalized" if networkx_mode else normalization
    return Pipeline(
        [
            SpectralPrepareState(
                normalization=effective_normalization,
                networkx_fidelity=networkx_mode,
                igraph_fidelity=igraph_mode,
            ),
            SpectralEmbed(
                sparse_threshold=sparse_threshold,
                networkx_fidelity=networkx_mode,
                igraph_fidelity=igraph_mode,
            ),
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
    fidelity_mode: Optional[Union[str, bool]] = None,
    fidelity_dtype: torch.dtype = torch.float32,
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
    fidelity_mode : str | bool | None, default=None
        Optional reference target. ``"igraph"`` mirrors igraph normalized
        Laplacian and eigenvector selection details; ``"networkx"`` is
        equivalent to ``networkx_fidelity=True``.

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
        fidelity_mode=fidelity_mode,
    ).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("Spectral pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_spectral_pipeline", "layout_spectral_pipeline"]
