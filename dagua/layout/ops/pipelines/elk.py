"""ELK Layered-style composable layout pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.elk import (
    ElkAssignLayers,
    ElkBreakCycles,
    ElkMinimizeCrossings,
    ElkPlaceNodes,
    ElkPrepareGraph,
)
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState

if TYPE_CHECKING:
    from dagua.config import LayoutConfig

_ELK_DEFAULT_NODE_NODE_SPACING = 40.0
_ELK_DEFAULT_BETWEEN_LAYERS_SPACING = 60.0


def build_elk_pipeline(
    direction: str = "DOWN",
    node_node_spacing: float = _ELK_DEFAULT_NODE_NODE_SPACING,
    between_layers_spacing: float = _ELK_DEFAULT_BETWEEN_LAYERS_SPACING,
    cycle_breaking_strategy: str = "greedy",
    layering_strategy: str = "network_simplex",
    crossing_minimization_strategy: str = "layer_sweep",
    node_placement_strategy: str = "brandes_koepf",
) -> Pipeline:
    """Build the native ELK Layered-style node-placement pipeline.

    Parameters
    ----------
    direction : str, default="DOWN"
        ELK direction: ``DOWN``, ``UP``, ``RIGHT``, or ``LEFT``. Dagre-style
        aliases ``TB``, ``BT``, ``LR``, and ``RL`` are also accepted.
    node_node_spacing : float, default=40.0
        ELK ``elk.spacing.nodeNode`` value.
    between_layers_spacing : float, default=60.0
        ELK ``elk.layered.spacing.nodeNodeBetweenLayers`` value.
    cycle_breaking_strategy : str, default="greedy"
        Cycle strategy selector. The current native implementation uses the
        same deterministic DFS reversal for every exposed selector.
    layering_strategy : str, default="network_simplex"
        Layering strategy selector. The current native implementation uses
        longest-path layering as the deterministic local approximation.
    crossing_minimization_strategy : str, default="layer_sweep"
        Crossing minimization selector.
    node_placement_strategy : str, default="brandes_koepf"
        Node placement selector. ``brandes_koepf`` uses the shared BK op.

    Returns
    -------
    Pipeline
        Five-stage composable ELK pipeline.
    """
    return Pipeline(
        [
            ElkPrepareGraph(
                direction=direction,
                node_node_spacing=node_node_spacing,
                between_layers_spacing=between_layers_spacing,
                cycle_breaking_strategy=cycle_breaking_strategy,
                layering_strategy=layering_strategy,
                crossing_minimization_strategy=crossing_minimization_strategy,
                node_placement_strategy=node_placement_strategy,
            ),
            ElkBreakCycles(),
            ElkAssignLayers(),
            ElkMinimizeCrossings(),
            ElkPlaceNodes(),
        ],
        name="elk_pipeline",
    )


def layout_elk_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: Optional[int] = 42,
    edge_weights: Optional[torch.Tensor] = None,
    direction: Optional[str] = None,
    node_node_spacing: float = _ELK_DEFAULT_NODE_NODE_SPACING,
    between_layers_spacing: float = _ELK_DEFAULT_BETWEEN_LAYERS_SPACING,
    nodeNode: Optional[float] = None,
    node_node: Optional[float] = None,
    node_node_between_layers: Optional[float] = None,
    cycle_breaking_strategy: str = "greedy",
    layering_strategy: str = "network_simplex",
    crossing_minimization_strategy: str = "layer_sweep",
    node_placement_strategy: str = "brandes_koepf",
    variant: Optional[str] = None,
    fidelity_dtype: Optional[torch.dtype] = None,
    config: Optional["LayoutConfig"] = None,
) -> torch.Tensor:
    """Run the deterministic ELK Layered-style pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Directed graph edges with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N``.
    node_sizes : torch.Tensor | None, optional
        Node box widths and heights with shape ``[N, 2]``.
    seed : int | None, default=42
        Accepted for API consistency; ELK Layered is deterministic.
    edge_weights : torch.Tensor | None, optional
        Accepted for pipeline API consistency; ignored by this source stage.
    direction : str | None, optional
        ELK direction. ``None`` reads ``config.direction`` when present and
        otherwise uses ``DOWN``.
    node_node_spacing : float, default=40.0
        Python spelling for ELK ``elk.spacing.nodeNode``.
    between_layers_spacing : float, default=60.0
        Python spelling for ELK between-layer spacing.
    nodeNode : float | None, optional
        ELK camel-case alias overriding ``node_node_spacing``.
    node_node : float | None, optional
        Alternate snake-case alias overriding ``node_node_spacing``.
    node_node_between_layers : float | None, optional
        Alias overriding ``between_layers_spacing``.
    cycle_breaking_strategy : str, default="greedy"
        Exposed ELK cycle strategy selector.
    layering_strategy : str, default="network_simplex"
        Exposed ELK layer strategy selector.
    crossing_minimization_strategy : str, default="layer_sweep"
        Exposed ELK crossing strategy selector.
    node_placement_strategy : str, default="brandes_koepf"
        Exposed ELK node placement selector.
    variant : str | None, optional
        Named variants: ``elk_layered_ns``, ``elk_layered_bk``, or ``elk_lp``.
    fidelity_dtype : torch.dtype | None, optional
        Accepted for engine compatibility; this deterministic pipeline returns
        ``float64`` coordinates.
    config : LayoutConfig | None, optional
        Optional config for direction fallback.

    Returns
    -------
    torch.Tensor
        Top-left node positions with shape ``[N, 2]``.

    Raises
    ------
    RuntimeError
        If the composed pipeline does not produce positions.
    """
    del seed, edge_weights, fidelity_dtype
    resolved_direction = direction or str(getattr(config, "direction", "DOWN"))
    resolved_node_spacing = node_node_spacing
    if nodeNode is not None:
        resolved_node_spacing = float(nodeNode)
    if node_node is not None:
        resolved_node_spacing = float(node_node)
    resolved_between_layers = between_layers_spacing
    if node_node_between_layers is not None:
        resolved_between_layers = float(node_node_between_layers)

    if variant is not None:
        normalized_variant = variant.lower()
        if normalized_variant == "elk_layered_ns":
            layering_strategy = "network_simplex"
            node_placement_strategy = "brandes_koepf"
        elif normalized_variant == "elk_layered_bk":
            node_placement_strategy = "brandes_koepf"
        elif normalized_variant == "elk_lp":
            layering_strategy = "longest_path"
            node_placement_strategy = "simple"
        else:
            raise ValueError("variant must be elk_layered_ns, elk_layered_bk, elk_lp, or None.")

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
    )
    state = SolveState()
    context = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    pipeline = build_elk_pipeline(
        direction=resolved_direction,
        node_node_spacing=resolved_node_spacing,
        between_layers_spacing=resolved_between_layers,
        cycle_breaking_strategy=cycle_breaking_strategy,
        layering_strategy=layering_strategy,
        crossing_minimization_strategy=crossing_minimization_strategy,
        node_placement_strategy=node_placement_strategy,
    )
    final_state = pipeline.apply(problem, state, context)
    if final_state.pos is None:
        raise RuntimeError("ELK pipeline did not produce final positions.")
    return final_state.pos


def layout_elk_layered_ns_pipeline(*args: object, **kwargs: object) -> torch.Tensor:
    """Run the ELK network-simplex/BK named variant.

    Parameters
    ----------
    *args : object
        Positional arguments forwarded to :func:`layout_elk_pipeline`.
    **kwargs : object
        Keyword arguments forwarded to :func:`layout_elk_pipeline`.

    Returns
    -------
    torch.Tensor
        Node positions with shape ``[N, 2]``.
    """
    kwargs["variant"] = "elk_layered_ns"
    return layout_elk_pipeline(*args, **kwargs)  # type: ignore[arg-type]


def layout_elk_layered_bk_pipeline(*args: object, **kwargs: object) -> torch.Tensor:
    """Run the ELK Brandes-Koepf named variant.

    Parameters
    ----------
    *args : object
        Positional arguments forwarded to :func:`layout_elk_pipeline`.
    **kwargs : object
        Keyword arguments forwarded to :func:`layout_elk_pipeline`.

    Returns
    -------
    torch.Tensor
        Node positions with shape ``[N, 2]``.
    """
    kwargs["variant"] = "elk_layered_bk"
    return layout_elk_pipeline(*args, **kwargs)  # type: ignore[arg-type]


def layout_elk_lp_pipeline(*args: object, **kwargs: object) -> torch.Tensor:
    """Run the ELK longest-path/simple-placement named variant.

    Parameters
    ----------
    *args : object
        Positional arguments forwarded to :func:`layout_elk_pipeline`.
    **kwargs : object
        Keyword arguments forwarded to :func:`layout_elk_pipeline`.

    Returns
    -------
    torch.Tensor
        Node positions with shape ``[N, 2]``.
    """
    kwargs["variant"] = "elk_lp"
    return layout_elk_pipeline(*args, **kwargs)  # type: ignore[arg-type]


__all__ = [
    "build_elk_pipeline",
    "layout_elk_layered_bk_pipeline",
    "layout_elk_layered_ns_pipeline",
    "layout_elk_lp_pipeline",
    "layout_elk_pipeline",
]
