"""Classical multidimensional scaling layout pipeline."""

from __future__ import annotations

from typing import Optional

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.distance import ClassicalMDSDistanceMatrix
from dagua.layout.ops.embed import ClassicalMDSComputeEmbedding, ClassicalMDSComputeEmbeddingConfig
from dagua.layout.ops.postprocess import (
    ClassicalMDSFinalizePositions,
    ClassicalMDSFinalizePositionsConfig,
)
from dagua.layout.ops.state import (
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)


def build_classical_mds_pipeline(*, igraph_fidelity: bool = False) -> Pipeline:
    """Build a classical multidimensional scaling pipeline.

    Reference fidelity
    ------------------
    Targets: igraph 1.0.0 MDS layout / Torgerson (1952) classical metric MDS.
    Fidelity mode: ``igraph_fidelity=True`` uses igraph-compatible raw
        embedding and final scaling semantics.
    Verified at: final 100-seed report, strong equivalent; median RMSD 0.000
        for both default and igraph-fidelity variants.
    Known divergences:
        - Graph distances are prepared in Dagua tensor ops rather than through
          igraph's C path.
        - Disconnected-graph behavior follows the benchmark distance matrix
          contract, not arbitrary user-provided dissimilarities.

    Parameters
    ----------
    igraph_fidelity : bool, default=False
        If ``True``, opt into igraph-compatible raw embedding and final scaling
        semantics for benchmark parity checks.

    Returns
    -------
    Pipeline
        Pipeline implementing classical MDS. The pipeline produces final node
        coordinates by computing the all-pairs graph distance matrix, solving
        the double-centered eigendecomposition, and finalizing the embedding
        into a 2D layout.
    """
    return Pipeline(
        [
            ClassicalMDSDistanceMatrix(),
            ClassicalMDSComputeEmbedding(
                config=ClassicalMDSComputeEmbeddingConfig(igraph_fidelity=igraph_fidelity)
            ),
            ClassicalMDSFinalizePositions(
                config=ClassicalMDSFinalizePositionsConfig(igraph_fidelity=igraph_fidelity)
            ),
        ],
        name="classical_mds_pipeline",
    )


def layout_classical_mds_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
    igraph_fidelity: bool = False,
) -> torch.Tensor:
    """Run the classical multidimensional scaling pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]`` used only to pick a
        stable output extent.
    seed : int, default=42
        Accepted for interface compatibility. Classical MDS is deterministic
        once graph distances are fixed.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.
    igraph_fidelity : bool, default=False
        If ``True``, ignore edge weights and use igraph-compatible embedding
        and scaling semantics. This is intended for fidelity benchmarking
        against ``igraph.layout("mds")``.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If ``num_nodes`` or ``edge_weights`` are invalid.
    RuntimeError
        If the pipeline does not produce final positions.
    """
    _ = seed, node_sizes

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
        edge_weights=None if igraph_fidelity else edge_weights,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_classical_mds_pipeline(igraph_fidelity=igraph_fidelity).apply(
        problem,
        state,
        ctx,
    )
    if final_state.pos is None:
        raise RuntimeError("Classical MDS pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_classical_mds_pipeline", "layout_classical_mds_pipeline"]
