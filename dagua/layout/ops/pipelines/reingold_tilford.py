"""Reingold-Tilford tidy-tree layout pipeline."""

from __future__ import annotations

from typing import Optional

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.coordinate import ReingoldTilfordTree, ReingoldTilfordTreeConfig
from dagua.layout.ops.state import (  # noqa: E402
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)


def build_reingold_tilford_pipeline(horizontal: bool = False) -> Pipeline:
    """Build a Reingold-Tilford tidy-tree pipeline.

    Parameters
    ----------
    horizontal : bool, default=False
        If ``True``, rotate the final layout so depth grows on x.

    Returns
    -------
    Pipeline
        Pipeline implementing the Reingold-Tilford tree-drawing algorithm.
        The pipeline produces final node coordinates by computing the tidy-tree
        contour placement and optionally rotating the result for horizontal
        depth.
    """
    return Pipeline(
        [ReingoldTilfordTree(ReingoldTilfordTreeConfig(horizontal=horizontal))],
        name="reingold_tilford_pipeline",
    )


def layout_reingold_tilford_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: int = 42,
    horizontal: bool = False,
    fidelity_mode: Optional[str] = None,
    traversal_mode: str = "out",
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the Reingold-Tilford tidy-tree pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]``.
    seed : int, default=42
        Accepted for interface compatibility. Reingold-Tilford is deterministic.
    horizontal : bool, default=False
        If ``True``, rotate the final layout so depth grows along x.
    fidelity_mode : str | None, default=None
        Optional compatibility mode. ``"igraph"`` uses unit spacing and
        mode-sensitive traversal for reference-fidelity comparisons.
    traversal_mode : str, default="out"
        Edge traversal mode for ``fidelity_mode="igraph"``.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Final layout coordinates with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If ``num_nodes`` or ``edge_weights`` are invalid.
    RuntimeError
        If the pipeline does not populate final positions.
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
    pipeline = Pipeline(
        [
            ReingoldTilfordTree(
                ReingoldTilfordTreeConfig(
                    horizontal=horizontal,
                    fidelity_mode=fidelity_mode,
                    traversal_mode=traversal_mode,
                )
            )
        ],
        name="reingold_tilford_pipeline",
    )
    final_state = pipeline.apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("Reingold-Tilford pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_reingold_tilford_pipeline", "layout_reingold_tilford_pipeline"]
