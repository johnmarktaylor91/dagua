"""Cytoscape CoSE-Bilkent layout pipeline."""

from __future__ import annotations

from typing import Any, Optional

import torch

from dagua.layout.ops.pipelines.cose import layout_cose_pipeline


def build_cose_bilkent_pipeline(steps: int = 2500, quality: str = "default") -> Any:
    """Return the shared CoSE pipeline builder for Bilkent defaults.

    Parameters
    ----------
    steps : int, default=2500
        Maximum number of force iterations.
    quality : str, default="default"
        CoSE-Bilkent quality tier.

    Returns
    -------
    Any
        Pipeline object built by the core CoSE wrapper.
    """
    from dagua.layout.ops.pipelines.cose import build_cose_pipeline

    cooling = 0.995 if quality == "proof" else 0.99
    return build_cose_pipeline(
        steps=0 if quality == "draft" else steps,
        randomize=True,
        node_repulsion=4500.0,
        ideal_edge_length=50.0,
        edge_elasticity=0.45,
        gravity=0.25,
        initial_temp=50.0,
        cooling_factor=cooling,
        min_temp=0.01,
    )


def layout_cose_bilkent_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 2500,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
    quality: str = "default",
    randomize: bool = True,
    nodeRepulsion: float = 4500.0,
    idealEdgeLength: float = 50.0,
    edgeElasticity: float = 0.45,
    gravity: float = 0.25,
    gravityRange: float = 3.8,
    gravityCompound: float = 1.0,
    gravityRangeCompound: float = 1.5,
    tile: bool = True,
    clusters: Optional[dict[str, Any]] = None,
    cluster_parents: Optional[dict[str, Optional[str]]] = None,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run the Cytoscape CoSE-Bilkent pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor | None, optional
        Node-size tensor with shape ``[N, 2]``.
    steps : int, default=2500
        Maximum number of force iterations.
    seed : int, default=42
        Seed for randomized initial placement.
    edge_weights : torch.Tensor | None, optional
        Accepted for API consistency.
    quality : str, default="default"
        Quality tier: ``"draft"``, ``"default"``, or ``"proof"``.
    randomize : bool, default=True
        Whether to randomize initial positions.
    nodeRepulsion : float, default=4500.0
        Node repulsion multiplier.
    idealEdgeLength : float, default=50.0
        Desired edge length.
    edgeElasticity : float, default=0.45
        Edge-force divisor.
    gravity : float, default=0.25
        Gravity force.
    gravityRange : float, default=3.8
        Accepted for option parity; current native step applies global gravity.
    gravityCompound : float, default=1.0
        Accepted for option parity with compound layouts.
    gravityRangeCompound : float, default=1.5
        Accepted for option parity with compound layouts.
    tile : bool, default=True
        Accepted for option parity with disconnected tiling.
    clusters : dict[str, Any] | None, optional
        Cluster membership metadata.
    cluster_parents : dict[str, str | None] | None, optional
        Cluster hierarchy metadata.
    fidelity_dtype : torch.dtype | None, optional
        Optional output dtype override.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.
    """
    del gravityRange, gravityCompound, gravityRangeCompound, tile, clusters, cluster_parents
    if quality not in {"draft", "default", "proof"}:
        raise ValueError("quality must be one of 'draft', 'default', or 'proof'.")
    effective_steps = 0 if quality == "draft" else steps
    cooling = 0.995 if quality == "proof" else 0.99
    return layout_cose_pipeline(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        steps=effective_steps,
        seed=seed,
        edge_weights=edge_weights,
        randomize=randomize,
        nodeRepulsion=nodeRepulsion,
        idealEdgeLength=idealEdgeLength,
        edgeElasticity=edgeElasticity,
        gravity=gravity,
        initialTemp=50.0,
        coolingFactor=cooling,
        minTemp=0.01,
        fidelity_dtype=fidelity_dtype,
    )


__all__ = ["build_cose_bilkent_pipeline", "layout_cose_bilkent_pipeline"]
