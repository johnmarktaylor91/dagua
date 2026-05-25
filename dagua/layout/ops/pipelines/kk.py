"""Kamada-Kawai spring-embedding layout pipeline."""

from __future__ import annotations

from typing import Optional

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.converge import FixedSteps, FixedStepsConfig
from dagua.layout.ops.distance import KamadaKawaiAllPairsShortestPaths
from dagua.layout.ops.graph_utils import layout_device as _layout_device
from dagua.layout.ops.init import KamadaKawaiInitializePositions
from dagua.layout.ops.optimize import LBFGSStep, LBFGSStepConfig
from dagua.layout.ops.postprocess import (
    KamadaKawaiFinalizePositions,
    KamadaKawaiFinalizePositionsConfig,
)
from dagua.layout.ops.state import (
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)

_DIRECTIONAL_ORIENTATION_MIN_GAIN = 0.05
_DIRECTION_TOP_TO_BOTTOM = "TB"
_DIRECTION_BOTTOM_TO_TOP = "BT"
_DIRECTION_LEFT_TO_RIGHT = "LR"
_DIRECTION_RIGHT_TO_LEFT = "RL"


def _directional_edge_fraction(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    direction: str,
) -> float:
    """Return the fraction of edges aligned with the requested direction.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    direction : str
        Layout direction, one of ``"TB"``, ``"BT"``, ``"LR"``, or ``"RL"``.

    Returns
    -------
    float
        Fraction of edges whose target is not behind the source along the
        requested layout axis. Self-loops count as aligned.
    """
    if edge_index.numel() == 0:
        return 1.0

    device_edge_index = edge_index.to(device=positions.device)
    source = device_edge_index[0]
    target = device_edge_index[1]
    self_loops = source == target
    if direction == _DIRECTION_LEFT_TO_RIGHT:
        aligned = (positions[target, 0] >= positions[source, 0]) | self_loops
    elif direction == _DIRECTION_RIGHT_TO_LEFT:
        aligned = (positions[target, 0] <= positions[source, 0]) | self_loops
    elif direction == _DIRECTION_BOTTOM_TO_TOP:
        aligned = (positions[target, 1] <= positions[source, 1]) | self_loops
    else:
        aligned = (positions[target, 1] >= positions[source, 1]) | self_loops
    return float(aligned.to(dtype=torch.float32).mean().item())


def _orient_positions_to_direction(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    direction: str,
) -> torch.Tensor:
    """Flip a KK embedding when doing so materially improves edge direction.

    Parameters
    ----------
    positions : torch.Tensor
        Solved position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    direction : str
        Layout direction, one of ``"TB"``, ``"BT"``, ``"LR"``, or ``"RL"``.

    Returns
    -------
    torch.Tensor
        Original or axis-flipped positions with shape ``[N, 2]``.
    """
    if positions.ndim != 2 or positions.shape[1] < 2 or edge_index.numel() == 0:
        return positions
    if direction not in {
        _DIRECTION_TOP_TO_BOTTOM,
        _DIRECTION_BOTTOM_TO_TOP,
        _DIRECTION_LEFT_TO_RIGHT,
        _DIRECTION_RIGHT_TO_LEFT,
    }:
        return positions

    base_fraction = _directional_edge_fraction(
        positions=positions,
        edge_index=edge_index,
        direction=direction,
    )
    flipped = positions.clone()
    if direction in {_DIRECTION_LEFT_TO_RIGHT, _DIRECTION_RIGHT_TO_LEFT}:
        flipped[:, 0] = -flipped[:, 0]
    else:
        flipped[:, 1] = -flipped[:, 1]
    flipped_fraction = _directional_edge_fraction(
        positions=flipped,
        edge_index=edge_index,
        direction=direction,
    )
    if flipped_fraction >= base_fraction + _DIRECTIONAL_ORIENTATION_MIN_GAIN:
        return flipped
    return positions


def build_kk_pipeline(
    steps: Optional[int] = None,
    trace_every: int = 0,
    preserve_float64: bool = False,
) -> Pipeline:
    """Build a Kamada-Kawai spring-embedding pipeline.

    Reference fidelity
    ------------------
    Targets: NetworkX 3.6.1 ``kamada_kawai_layout`` / Kamada and Kawai
        (1989), "An Algorithm for Drawing General Undirected Graphs".
    Fidelity mode: no dedicated flag; ``preserve_float64=True`` keeps audit
        output in double precision.
    Verified at: final 100-seed report, strong equivalent; median RMSD 0.000
        across 100, 300, and 1000 iteration variants.
    Known divergences:
        - Wrapper-level post-flip may orient directed graphs top-to-bottom for
          Dagua ergonomics.
        - The pipeline is tensor-native after shortest-path preparation rather
          than a direct NetworkX call.

    Parameters
    ----------
    steps : int, optional
        Maximum L-BFGS-B iterations. ``None`` or ``0`` leaves ``maxiter``
        unset to match classic KK.
    trace_every : int, default=0
        Snapshot cadence for optimizer traces.
    preserve_float64 : bool, default=False
        When ``True``, keep final coordinates in float64 for fidelity audits.

    Returns
    -------
    Pipeline
        Pipeline implementing the classical Kamada-Kawai algorithm. The
        pipeline produces final node coordinates by computing all-pairs
        shortest paths, initializing positions, minimizing the spring-energy
        objective with L-BFGS, and finalizing the layout.

    Raises
    ------
    ValueError
        If ``steps`` or ``trace_every`` are invalid.
    """
    if steps is not None and steps < 0:
        raise ValueError("steps must be non-negative.")
    if trace_every < 0:
        raise ValueError("trace_every must be non-negative.")

    optimize_config = LBFGSStepConfig(
        maxiter=steps,
        trace_every=trace_every,
        trace_key="kk_traces" if trace_every > 0 else None,
    )
    final_dtype = torch.float64 if preserve_float64 else torch.float32
    return Pipeline(
        [
            FixedSteps(FixedStepsConfig(n=0 if steps is None else steps)),
            KamadaKawaiAllPairsShortestPaths(),
            KamadaKawaiInitializePositions(),
            LBFGSStep(config=optimize_config),
            KamadaKawaiFinalizePositions(
                config=KamadaKawaiFinalizePositionsConfig(output_dtype=final_dtype)
            ),
        ],
        name="kk_pipeline",
    )


def layout_kk_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: Optional[int] = None,
    seed: int = 42,
    trace_every: int = 0,
    solver: str = "auto",
    pos: Optional[torch.Tensor] = None,
    edge_weights: Optional[torch.Tensor] = None,
    direction: str = "TB",
    orient_to_direction: bool = False,
    preserve_float64: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    """Run the Kamada-Kawai pipeline as a drop-in replacement.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]``. Unused and accepted
        for interface compatibility.
    steps : int, optional
        Maximum L-BFGS-B iterations. ``None`` or ``0`` leaves ``maxiter``
        unset to match classic KK.
    seed : int, default=42
        Accepted for interface compatibility. The 2D classic KK path uses
        deterministic circular initialization and does not consume a seed.
    trace_every : int, default=0
        Snapshot cadence for optimizer traces.
    solver : {"auto", "newton", "adam"}, default="auto"
        Retained for interface compatibility. All accepted values resolve to
        the classic SciPy L-BFGS-B path.
    pos : torch.Tensor, optional
        Initial positions with shape ``[N, 2]``. When provided, overrides the
        default circular initialization.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.
    direction : {"TB", "BT", "LR", "RL"}, default="TB"
        Requested graph reading direction used only when
        ``orient_to_direction`` is enabled.
    orient_to_direction : bool, default=False
        Whether to choose the axis orientation that materially improves edge
        direction consistency. The default is disabled so direct pipeline calls
        remain bit-exact with the archived NetworkX-style KK port.
    preserve_float64 : bool, default=False
        When ``True``, preserve float64 final coordinates for sub-percent
        fidelity audits. The default keeps historical float32 outputs.

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, list[torch.Tensor]]
        Final positions with shape ``[N, 2]``. When ``trace_every > 0``,
        periodic optimizer snapshots are returned alongside the final layout.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``steps``, ``trace_every``, ``solver``, or ``pos``
        are invalid.
    RuntimeError
        If the pipeline does not produce final positions.
    """
    _ = seed

    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if steps is not None and steps < 0:
        raise ValueError("steps must be non-negative.")
    if trace_every < 0:
        raise ValueError("trace_every must be non-negative.")
    if solver not in {"auto", "newton", "adam"}:
        raise ValueError("solver must be one of 'auto', 'newton', or 'adam'.")
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must have shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge count {edge_index.shape[1]}"
            )
        if bool(torch.any(edge_weights < 0).item()):
            raise ValueError("edge_weights must be nonnegative for KK shortest paths.")
    if pos is not None and pos.shape != (num_nodes, 2):
        raise ValueError(f"pos must have shape ({num_nodes}, 2), got {tuple(pos.shape)}")

    output_device = _layout_device(edge_index=edge_index, node_sizes=node_sizes)
    output_dtype = torch.float64 if preserve_float64 else torch.float32
    if num_nodes == 0:
        empty = torch.empty((0, 2), dtype=output_dtype, device=output_device)
        return (empty, []) if trace_every > 0 else empty
    if num_nodes == 1:
        single = torch.zeros((1, 2), dtype=output_dtype, device=output_device)
        return (single, []) if trace_every > 0 else single

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        direction=direction,
    )
    state = SolveState()
    if pos is not None:
        state.extras["kk_initial_pos"] = pos
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_kk_pipeline(
        steps=steps,
        trace_every=trace_every,
        preserve_float64=preserve_float64,
    ).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("KK pipeline did not produce final positions.")
    if orient_to_direction:
        final_state.pos = _orient_positions_to_direction(
            positions=final_state.pos,
            edge_index=edge_index,
            direction=direction,
        )

    if trace_every > 0:
        traces = final_state.extras.get("kk_traces", [])
        return final_state.pos, traces
    return final_state.pos


__all__ = ["build_kk_pipeline", "layout_kk_pipeline"]
