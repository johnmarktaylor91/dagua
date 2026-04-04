"""Classical MDS expressed as a composable ops pipeline."""

from __future__ import annotations

from typing import ClassVar, Optional, Tuple

import numpy as np
import torch

from dagua.layout.ops.graph_utils import (
    layout_device as _layout_device,
)
from dagua.layout.ops.graph_utils import (
    layout_extent as _layout_extent,
)
from dagua.layout.ops.graph_utils import (
    shortest_path_distances as _shortest_path_distances,
)

_MIN_SPAN = 1.0e-6


def _normalize_positions(positions: torch.Tensor, extent: float) -> torch.Tensor:
    """Center and scale coordinates into a stable bounding box.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    extent : float
        Target half-width after normalization.

    Returns
    -------
    torch.Tensor
        Centered and scaled coordinates.
    """
    if positions.shape[0] <= 1:
        return torch.zeros_like(positions)

    centered = positions - positions.mean(dim=0, keepdim=True)
    span = float(centered.abs().max().item())
    if span < _MIN_SPAN:
        centered = centered.clone()
        centered[:, 0] = torch.linspace(
            -1.0,
            1.0,
            steps=positions.shape[0],
            device=positions.device,
            dtype=positions.dtype,
        )
        span = float(centered.abs().max().item())
    return centered * (extent / max(span, _MIN_SPAN))


def _classical_mds_embedding(distances: np.ndarray) -> torch.Tensor:
    """Recover a rank-2 embedding from pairwise graph distances.

    Parameters
    ----------
    distances : numpy.ndarray
        Dense shortest-path distance matrix with shape ``[N, N]``.

    Returns
    -------
    torch.Tensor
        Raw coordinates with shape ``[N, 2]`` on CPU.
    """
    num_nodes = int(distances.shape[0])
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float32)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=torch.float32)

    squared = distances * distances
    centering = np.eye(num_nodes, dtype=np.float64) - (
        np.ones((num_nodes, num_nodes), dtype=np.float64) / float(num_nodes)
    )
    gram = -0.5 * centering @ squared @ centering

    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    positive_indices = [index for index in sorted_indices if eigenvalues[index] > 0.0][:2]

    coordinates = np.zeros((num_nodes, 2), dtype=np.float64)
    if positive_indices:
        selected_values = np.clip(eigenvalues[positive_indices], a_min=0.0, a_max=None)
        selected_vectors = eigenvectors[:, positive_indices]
        coordinates[:, : len(positive_indices)] = selected_vectors * np.sqrt(selected_values)
    else:
        coordinates[:, 0] = np.linspace(-1.0, 1.0, num_nodes, dtype=np.float64)

    return torch.from_numpy(coordinates).to(dtype=torch.float32)


from dagua.layout.ops.base import Op, Pipeline  # noqa: E402
from dagua.layout.ops.state import (  # noqa: E402
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)
from dagua.layout.ops.taxonomy import OpCategory  # noqa: E402

_DISTANCE_MATRIX_KEY = "classical_mds_distance_matrix"


class _PrepareClassicalMDSState(Op):
    """Compute the exact graph-distance matrix used by classical MDS."""

    name: ClassVar[str] = "classical_mds_prepare_state"
    category: ClassVar[OpCategory] = OpCategory.DISTANCE
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Store dense shortest-path distances for the embedding step.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with the classical MDS distance matrix stored in ``extras``.
        """
        del ctx

        distances = _shortest_path_distances(
            edge_index=problem.edge_index,
            num_nodes=problem.num_nodes,
            edge_weights=problem.edge_weights,
        )
        state.extras[_DISTANCE_MATRIX_KEY] = distances
        return state


class _EmbedClassicalMDS(Op):
    """Recover the raw rank-2 embedding from the distance matrix."""

    name: ClassVar[str] = "classical_mds_embed"
    category: ClassVar[OpCategory] = OpCategory.EMBED
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute the raw classical MDS coordinates on CPU.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state containing the prepared distance matrix.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with raw CPU coordinates in ``state.pos``.
        """
        del problem, ctx

        distances = state.extras[_DISTANCE_MATRIX_KEY]
        state.pos = _classical_mds_embedding(distances)
        return state


class _FinalizeClassicalMDSPositions(Op):
    """Apply the classic center-and-scale normalization."""

    name: ClassVar[str] = "classical_mds_finalize_positions"
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
        """Normalize the raw embedding onto the classic output device.

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
            State with final classical MDS positions.

        Raises
        ------
        ValueError
            If ``state.pos`` is missing.
        """
        del ctx

        if state.pos is None:
            raise ValueError("_FinalizeClassicalMDSPositions requires state.pos to be set.")

        output_device = _layout_device(
            edge_index=problem.edge_index,
            node_sizes=problem.node_sizes,
        )
        extent = _layout_extent(num_nodes=problem.num_nodes, node_sizes=problem.node_sizes)
        normalized = _normalize_positions(state.pos.to(device=output_device), extent=extent)
        state.pos = normalized.to(dtype=torch.float32, device=output_device)
        return state


def build_classical_mds_pipeline() -> Pipeline:
    """Build a classical MDS pipeline that matches ``layout_classical_mds``.

    Returns
    -------
    Pipeline
        Pipeline that reproduces classic classical MDS exactly.
    """
    return Pipeline(
        [
            _PrepareClassicalMDSState(),
            _EmbedClassicalMDS(),
            _FinalizeClassicalMDSPositions(),
        ],
        name="classical_mds_pipeline",
    )


def layout_classical_mds_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the classical MDS pipeline as a drop-in replacement.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor used only to pick a stable output extent.
    seed : int, default=42
        Accepted for interface compatibility. Classical MDS is deterministic
        once graph distances are fixed.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]`` used when computing
        shortest-path distances.

    Returns
    -------
    torch.Tensor
        Final position tensor with the same dtype, device, and values as
        classic ``layout_classical_mds``.

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
    final_state = build_classical_mds_pipeline().apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("Classical MDS pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_classical_mds_pipeline", "layout_classical_mds_pipeline"]
