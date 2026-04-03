"""Spectral layout expressed as a composable ops pipeline."""

from __future__ import annotations

from typing import ClassVar, Optional, Tuple

import torch

from dagua.layout.classic.spectral import (
    SPARSE_EIGEN_THRESHOLD,
    _build_adjacency,
    _dense_spectral_embedding,
    _laplacian_matrix,
    _layout_device,
    _rescale_layout,
    _sparse_spectral_embedding,
    _symmetrized_adjacency,
)
from dagua.layout.ops.base import Op, Pipeline
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory

_SYMMETRIC_FLAG_KEY = "spectral_is_symmetric"


class _PrepareSpectralState(Op):
    """Build the exact Laplacian state required by the classic solver."""

    name: ClassVar[str] = "spectral_prepare_state"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    writes: ClassVar[Tuple[str, ...]] = ("pos", "laplacian", "extras")

    def __init__(self, normalization: str) -> None:
        """Store the requested Laplacian normalization mode.

        Parameters
        ----------
        normalization : str
            Laplacian normalization mode.

        Returns
        -------
        None
            This constructor stores configuration only.
        """
        self.normalization = normalization

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Prepare trivial outputs or cache the spectral Laplacian.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this deterministic op.

        Returns
        -------
        SolveState
            State with trivial positions for degenerate graphs or the cached
            Laplacian for the eigensolver step.
        """
        del ctx

        if problem.num_nodes == 0:
            state.pos = torch.empty((0, 2), dtype=torch.float32)
            return state
        if problem.num_nodes == 1:
            state.pos = torch.zeros((1, 2), dtype=torch.float32)
            return state

        adjacency = _build_adjacency(
            edge_index=problem.edge_index,
            num_nodes=problem.num_nodes,
            edge_weights=problem.edge_weights,
        )
        laplacian, is_symmetric = _laplacian_matrix(
            adjacency=_symmetrized_adjacency(adjacency),
            normalization=self.normalization,
        )
        state.laplacian = laplacian
        state.extras[_SYMMETRIC_FLAG_KEY] = is_symmetric
        return state


class _EmbedSpectral(Op):
    """Recover the raw 2D spectral embedding."""

    name: ClassVar[str] = "spectral_embed"
    category: ClassVar[OpCategory] = OpCategory.EMBED
    reads: ClassVar[Tuple[str, ...]] = ("pos", "laplacian", "extras")
    writes: ClassVar[Tuple[str, ...]] = ("pos",)

    def __init__(self, sparse_threshold: int) -> None:
        """Store the dense-versus-sparse eigensolver threshold.

        Parameters
        ----------
        sparse_threshold : int
            Node-count threshold for the sparse eigensolver branch.

        Returns
        -------
        None
            This constructor stores configuration only.
        """
        self.sparse_threshold = int(sparse_threshold)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run the same dense or sparse eigensolver as the classic layout.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state containing the cached Laplacian.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this deterministic op.

        Returns
        -------
        SolveState
            State with raw spectral coordinates stored in ``state.pos``.

        Raises
        ------
        RuntimeError
            If the Laplacian or symmetry flag is missing.
        """
        del ctx

        if state.pos is not None:
            return state
        if state.laplacian is None:
            raise RuntimeError("_EmbedSpectral requires state.laplacian to be set.")
        is_symmetric = state.extras.get(_SYMMETRIC_FLAG_KEY)
        if not isinstance(is_symmetric, bool):
            raise RuntimeError("_EmbedSpectral requires a cached symmetry flag.")

        if problem.num_nodes < self.sparse_threshold:
            coordinates = _dense_spectral_embedding(
                laplacian=state.laplacian,
                dim=2,
                symmetric=is_symmetric,
            )
        else:
            coordinates = _sparse_spectral_embedding(
                laplacian=state.laplacian,
                dim=2,
                symmetric=is_symmetric,
            )

        state.pos = torch.from_numpy(coordinates)
        return state


class _FinalizeSpectralPositions(Op):
    """Apply the classic spectral center-and-scale normalization."""

    name: ClassVar[str] = "spectral_finalize_positions"
    category: ClassVar[OpCategory] = OpCategory.POSTPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Rescale raw coordinates like classic ``layout_spectral``.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state containing raw coordinates.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with final output positions.

        Raises
        ------
        ValueError
            If ``state.pos`` is missing.
        """
        del ctx

        if state.pos is None:
            raise ValueError("_FinalizeSpectralPositions requires state.pos to be set.")

        output_device = _layout_device(
            edge_index=problem.edge_index,
            node_sizes=problem.node_sizes,
        )
        if problem.num_nodes == 0:
            state.pos = torch.empty((0, 2), dtype=torch.float32, device=output_device)
            return state
        if problem.num_nodes == 1:
            state.pos = torch.zeros((1, 2), dtype=torch.float32, device=output_device)
            return state

        positions = state.pos.detach().to(device="cpu", dtype=torch.float64).numpy()
        scaled = _rescale_layout(positions, scale=1.0)
        state.pos = torch.from_numpy(scaled).to(dtype=torch.float32, device=output_device)
        return state


def build_spectral_pipeline(
    normalization: str = "symmetric",
    sparse_threshold: int = SPARSE_EIGEN_THRESHOLD,
) -> Pipeline:
    """Build a spectral pipeline that matches classic ``layout_spectral``.

    Parameters
    ----------
    normalization : str, default="symmetric"
        Laplacian normalization mode.
    sparse_threshold : int, default=500
        Node-count threshold for the sparse eigensolver branch.

    Returns
    -------
    Pipeline
        Pipeline that reproduces classic spectral layout exactly.

    Raises
    ------
    ValueError
        If ``sparse_threshold`` is not positive.
    """
    if sparse_threshold <= 0:
        raise ValueError("sparse_threshold must be positive.")

    return Pipeline(
        [
            _PrepareSpectralState(normalization=normalization),
            _EmbedSpectral(sparse_threshold=sparse_threshold),
            _FinalizeSpectralPositions(),
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
) -> torch.Tensor:
    """Run the spectral pipeline as a drop-in replacement.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Unused, accepted for interface compatibility.
    seed : int, default=42
        Accepted for interface compatibility. Spectral layout is deterministic
        once the graph is fixed.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.
    normalization : str, default="symmetric"
        Laplacian normalization mode.

    Returns
    -------
    torch.Tensor
        Final position tensor with the same dtype, device, and values as
        classic ``layout_spectral``.

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
    final_state = build_spectral_pipeline(normalization=normalization).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("Spectral pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_spectral_pipeline", "layout_spectral_pipeline"]
