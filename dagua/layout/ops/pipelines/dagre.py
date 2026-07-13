"""Dagre.js-compatible composable layered-layout pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.brandes_koepf import (
    BrandesKoepfConfig,
    BrandesKoepfXAssignment,
)
from dagua.layout.ops.dagre import (
    DagreAssignRanks,
    DagreAssignY,
    DagreFinalizeCoordinates,
    DagreMakeAcyclic,
    DagreNormalizeEdges,
    DagreOrderNodes,
    DagrePrepareGraph,
)
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState

if TYPE_CHECKING:
    from dagua.config import LayoutConfig

_DAGRE_DEFAULT_NODE_SEP = 50.0
_DAGRE_DEFAULT_RANK_SEP = 50.0
_DAGRE_DEFAULT_EDGE_SEP = 20.0


def build_dagre_pipeline(
    ranker: str = "network-simplex",
    rankdir: str = "TB",
    align: Optional[str] = None,
    node_sep: float = _DAGRE_DEFAULT_NODE_SEP,
    rank_sep: float = _DAGRE_DEFAULT_RANK_SEP,
    edge_sep: float = _DAGRE_DEFAULT_EDGE_SEP,
    acyclicer: str = "dfs",
    config: Optional["LayoutConfig"] = None,
) -> Pipeline:
    """Build the dagre.js 0.8.5 node-placement pipeline.

    Reference fidelity
    ------------------
    Targets: ``dagrejs/dagre`` 0.8.5, the layered engine used by Mermaid,
        React Flow integrations, and cytoscape-dagre.
    Fidelity mode: deterministic source port; no runtime delegation.
    Variant controls: ``ranker``, ``rankdir``, ``align``, ``node_sep``,
        ``rank_sep``, ``edge_sep``, and ``acyclicer``.

    Parameters
    ----------
    ranker : str, default="network-simplex"
        Ranker: ``network-simplex``, ``tight-tree``, or ``longest-path``.
    rankdir : str, default="TB"
        Direction: ``TB``, ``BT``, ``LR``, or ``RL``.
    align : str | None, default=None
        Brandes-Koepf alignment: ``UL``, ``UR``, ``DL``, or ``DR``.
    node_sep : float, default=50.0
        Gap between adjacent real-node boxes.
    rank_sep : float, default=50.0
        Gap between adjacent rank boxes.
    edge_sep : float, default=20.0
        Gap contributed by adjacent normalized-edge dummies.
    acyclicer : str, default="dfs"
        Feedback-arc heuristic: ``dfs`` or ``greedy``.
    config : LayoutConfig | None, optional
        Optional config used only for resolved hard pins at finalization.

    Returns
    -------
    Pipeline
        Seven composable stages: prepare, acyclic, rank, normalize, order,
        Brandes-Koepf x assignment, y assignment, and final transform.
    """
    return Pipeline(
        [
            DagrePrepareGraph(
                rank_sep=rank_sep,
                node_sep=node_sep,
                edge_sep=edge_sep,
                rankdir=rankdir,
                ranker=ranker,
                acyclicer=acyclicer,
            ),
            DagreMakeAcyclic(),
            DagreAssignRanks(),
            DagreNormalizeEdges(),
            DagreOrderNodes(),
            BrandesKoepfXAssignment(
                BrandesKoepfConfig(node_sep=node_sep, edge_sep=edge_sep, align=align)
            ),
            DagreAssignY(),
            DagreFinalizeCoordinates(config=config),
        ],
        name="dagre_pipeline",
    )


def layout_dagre_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: Optional[int] = 42,
    edge_weights: Optional[torch.Tensor] = None,
    ranker: str = "network-simplex",
    rankdir: Optional[str] = None,
    align: Optional[str] = None,
    node_sep: float = _DAGRE_DEFAULT_NODE_SEP,
    rank_sep: float = _DAGRE_DEFAULT_RANK_SEP,
    edge_sep: float = _DAGRE_DEFAULT_EDGE_SEP,
    nodesep: Optional[float] = None,
    ranksep: Optional[float] = None,
    edgesep: Optional[float] = None,
    acyclicer: str = "dfs",
    fidelity_dtype: Optional[torch.dtype] = None,
    config: Optional["LayoutConfig"] = None,
) -> torch.Tensor:
    """Run the deterministic Dagre layered-layout pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Directed graph edges with shape ``[2, E]``.
    num_nodes : int
        Number of original nodes ``N``.
    node_sizes : torch.Tensor | None, optional
        Node box widths and heights with shape ``[N, 2]``. Missing sizes use
        Dagre's canonical zero-size node defaults.
    seed : int | None, default=42
        Accepted for pipeline API consistency; Dagre is deterministic.
    edge_weights : torch.Tensor | None, optional
        Per-edge weights with shape ``[E]``.
    ranker : str, default="network-simplex"
        ``network-simplex``, ``tight-tree``, or ``longest-path``.
    rankdir : str | None, default=None
        ``TB``, ``BT``, ``LR``, or ``RL``. ``None`` reads ``config.direction``
        when available and otherwise uses ``TB``.
    align : str | None, default=None
        ``UL``, ``UR``, ``DL``, or ``DR``; ``None`` balances four assignments.
    node_sep : float, default=50.0
        Python-style real-node separation option.
    rank_sep : float, default=50.0
        Python-style rank separation option.
    edge_sep : float, default=20.0
        Python-style dummy-edge separation option.
    nodesep : float | None, optional
        Dagre.js spelling alias overriding ``node_sep``.
    ranksep : float | None, optional
        Dagre.js spelling alias overriding ``rank_sep``.
    edgesep : float | None, optional
        Dagre.js spelling alias overriding ``edge_sep``.
    acyclicer : str, default="dfs"
        ``dfs`` or ``greedy`` feedback-arc heuristic.
    fidelity_dtype : torch.dtype | None, optional
        Accepted for engine compatibility. The source port computes scalar
        coordinates in Python double precision and returns ``float64`` before
        the public engine converts output to ``float32``.
    config : LayoutConfig | None, optional
        Full layout config used for direction fallback and hard pins.

    Returns
    -------
    torch.Tensor
        Node positions with shape ``[N, 2]``.

    Raises
    ------
    RuntimeError
        If the composed stages fail to produce final positions.
    """
    del seed, fidelity_dtype
    resolved_rankdir = rankdir
    if resolved_rankdir is None:
        resolved_rankdir = str(getattr(config, "direction", "TB"))
    resolved_node_sep = node_sep if nodesep is None else float(nodesep)
    resolved_rank_sep = rank_sep if ranksep is None else float(ranksep)
    resolved_edge_sep = edge_sep if edgesep is None else float(edgesep)

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
    )
    state = SolveState()
    context = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    pipeline = build_dagre_pipeline(
        ranker=ranker,
        rankdir=resolved_rankdir,
        align=align,
        node_sep=resolved_node_sep,
        rank_sep=resolved_rank_sep,
        edge_sep=resolved_edge_sep,
        acyclicer=acyclicer,
        config=config,
    )
    final_state = pipeline.apply(problem, state, context)
    if final_state.pos is None:
        raise RuntimeError("Dagre pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_dagre_pipeline", "layout_dagre_pipeline"]
