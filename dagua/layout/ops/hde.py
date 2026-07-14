"""Harel-Koren high-dimensional embedding operations."""

from __future__ import annotations

from typing import ClassVar, Tuple

import torch

from dagua.layout.ops.base import Op
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_HDE_METADATA_KEY = "hde"
_OUTPUT_DIMENSIONS = 2


def hde_project_distances(distance_matrix: torch.Tensor) -> torch.Tensor:
    """Project pivot-distance coordinates to two PCA dimensions.

    Parameters
    ----------
    distance_matrix : torch.Tensor
        Pivot-to-node distance matrix with shape ``[P, N]``.

    Returns
    -------
    torch.Tensor
        PCA score matrix with shape ``[N, 2]`` and dtype ``float64``.
    """
    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must have shape [P, N].")

    num_nodes = int(distance_matrix.shape[1])
    if num_nodes == 0:
        return torch.empty((0, _OUTPUT_DIMENSIONS), dtype=torch.float64)
    if num_nodes == 1 or int(distance_matrix.shape[0]) == 0:
        return torch.zeros((num_nodes, _OUTPUT_DIMENSIONS), dtype=torch.float64)

    coordinates = distance_matrix.detach().to(device="cpu", dtype=torch.float64).transpose(0, 1)
    centered = coordinates - coordinates.mean(dim=0, keepdim=True)
    if float(centered.abs().max().item()) == 0.0:
        return torch.zeros((num_nodes, _OUTPUT_DIMENSIONS), dtype=torch.float64)

    u_matrix, singular_values, vh_matrix = torch.linalg.svd(centered, full_matrices=False)
    coord_dims = min(_OUTPUT_DIMENSIONS, int(singular_values.shape[0]))
    scores = u_matrix[:, :coord_dims] * singular_values[:coord_dims].unsqueeze(0)

    # PCA signs are arbitrary. Pin each component by making the loading with
    # largest absolute magnitude positive so fidelity reports do not depend on
    # LAPACK sign choices.
    for component in range(coord_dims):
        loading = vh_matrix[component]
        anchor = int(torch.argmax(loading.abs()).item())
        if float(loading[anchor].item()) < 0.0:
            scores[:, component] = -scores[:, component]

    if coord_dims < _OUTPUT_DIMENSIONS:
        padding = torch.zeros((num_nodes, _OUTPUT_DIMENSIONS - coord_dims), dtype=scores.dtype)
        scores = torch.cat((scores, padding), dim=1)
    return scores.to(dtype=torch.float64)


@register_op
class HDEProjectPivotDistances(Op):
    """Assign HDE coordinates from selected pivot-distance rows."""

    name: ClassVar[str] = "hde_project_pivot_distances"
    category: ClassVar[OpCategory] = OpCategory.INIT
    reads: ClassVar[Tuple[str, ...]] = ("pivot_distances", "pivot_indices")
    writes: ClassVar[Tuple[str, ...]] = ("pos", "extras.hde")
    requires: ClassVar[Tuple[str, ...]] = ("pivot_distances",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Project HDE pivot distances into ``state.pos``.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state carrying ``pivot_distances`` with shape
            ``[P, N]``.
        ctx : RuntimeContext
            Runtime context. Unused by this deterministic CPU projection.

        Returns
        -------
        SolveState
            State with ``pos`` populated as a ``[N, 2]`` tensor.

        Raises
        ------
        ValueError
            If pivot distances are unavailable.
        """
        del ctx
        if state.pivot_distances is None:
            raise ValueError("HDE requires pivot distances. Run PivotDistanceQueries first.")

        positions = hde_project_distances(state.pivot_distances)
        state.pos = positions
        state.extras[_HDE_METADATA_KEY] = {
            "pivots": []
            if state.pivot_indices is None
            else state.pivot_indices.detach().to(device="cpu", dtype=torch.long).tolist(),
            "dimensions": int(state.pivot_distances.shape[0]),
            "reference": "Harel-Koren HDE / graphlayouts PMDS-style pivot PCA",
            "reusable_init_op": self.name,
            "num_nodes": int(problem.num_nodes),
        }
        return state


__all__ = ["HDEProjectPivotDistances", "hde_project_distances"]
