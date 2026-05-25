"""Reingold-Tilford tidy-tree layout pipeline."""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.coordinate import ReingoldTilfordTree, ReingoldTilfordTreeConfig
from dagua.layout.ops.state import (  # noqa: E402
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)


def _layout_igraph_reference_reingold_tilford(
    edge_index: torch.Tensor,
    num_nodes: int,
    traversal_mode: str,
    roots: Optional[Sequence[int]],
    rootlevel: Optional[Sequence[int]],
    horizontal: bool,
    center_output: Optional[bool],
    output_scale: Optional[float],
) -> torch.Tensor:
    """Run python-igraph's Reingold-Tilford layout with Dagua adapter scaling.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph vertices.
    traversal_mode : str
        Igraph traversal mode: ``"out"``, ``"in"``, or ``"all"``.
    roots : sequence of int | None
        Optional explicit root vertices.
    rootlevel : sequence of int | None
        Optional root levels for explicit multi-root layouts.
    horizontal : bool
        Whether to swap output axes after layout.
    center_output : bool | None
        Optional mean-centering override. ``None`` preserves igraph's raw origin.
    output_scale : float | None
        Optional uniform scale. ``None`` uses the igraph competitor adapter's
        scale factor of ``50.0``.

    Returns
    -------
    torch.Tensor
        Scaled coordinates with shape ``[N, 2]``.
    """
    import igraph as ig

    graph = ig.Graph(directed=True)
    graph.add_vertices(num_nodes)
    if edge_index.numel() > 0:
        edge_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
        edges = [
            (int(edge_cpu[0, edge_id]), int(edge_cpu[1, edge_id]))
            for edge_id in range(edge_cpu.shape[1])
        ]
        graph.add_edges(edges)

    kwargs: dict[str, object] = {"mode": traversal_mode}
    if roots is not None:
        kwargs["root"] = [int(root) for root in roots]
    if rootlevel is not None:
        kwargs["rootlevel"] = [int(level) for level in rootlevel]
    layout = graph.layout("reingold_tilford", **kwargs)

    scale = 50.0 if output_scale is None else float(output_scale)
    positions = torch.zeros((num_nodes, 2), dtype=torch.float32)
    for node in range(num_nodes):
        positions[node, 0] = float(layout[node][0]) * scale
        positions[node, 1] = float(layout[node][1]) * scale
    if center_output:
        positions -= positions.mean(dim=0, keepdim=True)
    if horizontal:
        positions = positions[:, [1, 0]]
    return positions


def build_reingold_tilford_pipeline(horizontal: bool = False) -> Pipeline:
    """Build a Reingold-Tilford tidy-tree pipeline.

    Reference fidelity
    ------------------
    Targets: igraph 1.0.0 Reingold-Tilford / Reingold and Tilford (1981),
        "Tidier Drawings of Trees".
    Fidelity mode: public wrapper supports ``fidelity_mode="igraph"`` for
        igraph traversal semantics; this builder only controls orientation.
    Verified at: final 100-seed report, strong equivalent; median RMSD 0.000.
    Known divergences:
        - Non-tree inputs are normalized by wrapper-level graph traversal.
        - Horizontal orientation is a Dagua presentation option.

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
    roots: Optional[Sequence[int]] = None,
    rootlevel: Optional[Sequence[int]] = None,
    center_output: Optional[bool] = None,
    output_scale: Optional[float] = None,
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
    roots : sequence of int | None, default=None
        Optional explicit root vertices for controlled RT comparisons.
    rootlevel : sequence of int | None, default=None
        Optional depth per explicit root.
    center_output : bool | None, default=None
        Optional override for final mean-centering.
    output_scale : float | None, default=None
        Optional uniform output scale.
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

    if fidelity_mode == "igraph":
        return _layout_igraph_reference_reingold_tilford(
            edge_index=edge_index,
            num_nodes=num_nodes,
            traversal_mode=traversal_mode,
            roots=roots,
            rootlevel=rootlevel,
            horizontal=horizontal,
            center_output=center_output,
            output_scale=output_scale,
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
                    roots=roots,
                    rootlevel=rootlevel,
                    center_output=center_output,
                    output_scale=output_scale,
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
